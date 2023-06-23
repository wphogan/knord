import copy
import re
from collections import OrderedDict
from math import ceil
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from transformers import AutoConfig, AutoModel


def onedim_gather(src: torch.Tensor, dim: int, index: torch.LongTensor) -> torch.Tensor:
    for i in range(len(src.shape)):  # b
        if i != 0 and i != dim:
            # index is missing this dimension
            index = index.unsqueeze(dim=i)
    target_index_size = copy.deepcopy(list(src.shape))
    target_index_size[dim] = 1
    index = index.expand(target_index_size)
    max_idx = src.shape[1] - 1
    index = torch.clamp(index, min=0, max=max_idx)
    output = torch.gather(src, dim, index)
    return output


class MultiviewModel(nn.Module):
    def __init__(self, config, tokenizer) -> None:
        super().__init__()
        self.config = config
        self.layer = self.config.layer
        self.model_config = AutoConfig.from_pretrained(config.foundation_model)
        self.pretrained_model = AutoModel.from_pretrained(config.foundation_model, config=self.model_config)
        self.pretrained_model.resize_token_embeddings(len(tokenizer))
        self.pretrained_model.to(config.device)
        self.views = nn.ModuleList()
        feature_types = ['token', 'mask']
        known_head_types = self.config.n_seen_classes
        if config.constrain_pred_type == 'novel_only' or config.use_confidence or config.constrain_pred_type == 'unconstrained':
            n_classes_unlabeled = config.n_novel_classes
        elif config.constrain_pred_type == 'seen_only':
            n_classes_unlabeled = config.n_seen_classes
        for view_idx, ft in enumerate(feature_types):
            if ft == 'mask':
                view_model = nn.ModuleDict(
                    {
                        'common_space_proj': MLP(self.model_config.hidden_size, self.config.hidden_dim,
                                                 self.config.kmeans_dim,
                                                 norm=True, norm_type='batch', layers_n=2, dropout_p=0.1),
                        'known_type_center_loss': CenterLoss(self.config.kmeans_dim, self.config.n_seen_classes,
                                                             weight_by_prob=False),
                        'unknown_type_center_loss': CenterLoss(self.config.kmeans_dim, n_classes_unlabeled,
                                                               weight_by_prob=False),
                        'known_type_classifier': ClassifierHead(self.config.kmeans_dim,
                                                                known_head_types,
                                                                layers_n=self.config.classifier_layers,
                                                                n_heads=1, dropout_p=0.0,
                                                                hidden_size=self.config.kmeans_dim),
                        'unknown_type_classifier': ClassifierHead(self.config.kmeans_dim,
                                                                  n_classes_unlabeled,
                                                                  layers_n=self.config.classifier_layers,
                                                                  n_heads=1, dropout_p=0.0,
                                                                  hidden_size=self.config.kmeans_dim)
                    }
                )
            else:
                input_size = 2 * self.model_config.hidden_size  # head, tail
                view_model = nn.ModuleDict(
                    {
                        'common_space_proj': MLP(input_size, self.config.hidden_dim, self.config.kmeans_dim,
                                                 norm=True, norm_type='batch', layers_n=2, dropout_p=0.1),
                        'known_type_center_loss': CenterLoss(self.config.kmeans_dim, self.config.n_seen_classes,
                                                             weight_by_prob=False),
                        'unknown_type_center_loss': CenterLoss(self.config.kmeans_dim, n_classes_unlabeled,
                                                               weight_by_prob=False),
                        'known_type_classifier': ClassifierHead(self.config.kmeans_dim,
                                                                known_head_types,
                                                                layers_n=self.config.classifier_layers,
                                                                n_heads=1, dropout_p=0.0,
                                                                hidden_size=self.config.kmeans_dim),
                        'unknown_type_classifier': ClassifierHead(self.config.kmeans_dim,
                                                                  n_classes_unlabeled,
                                                                  layers_n=self.config.classifier_layers,
                                                                  n_heads=1, dropout_p=0.0,
                                                                  hidden_size=self.config.kmeans_dim)
                    }
                )
            self.views.append(view_model)

        # this commonspace means that known classes and unknown classes are projected into the same space
        self.commonspace_cache = nn.ModuleList([
            CommonSpaceCache(feature_size=self.config.kmeans_dim, known_cache_size=512, unknown_cache_size=256,
                             sim_thres=0.8),
            CommonSpaceCache(feature_size=self.config.kmeans_dim, known_cache_size=512, unknown_cache_size=256,
                             sim_thres=0.8)
        ])

        return

        # FIXME: this function is taken from ROCORE, will use layer.7 instead of layer.8 as described in the paper.

    @staticmethod
    def finetune(model, unfreeze_layers):
        params_name_mapping = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5',
                               'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'layer.12']
        for name, param in model.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if params_name_mapping[ele] in name:
                    param.requires_grad = True
                    break
        return model

    def _compute_features(self, batch: Dict, method: str = 'token') -> torch.FloatTensor:
        inputs = {
            'input_ids': batch['token_ids'].to(self.config.device),
            'attention_mask': batch['attn_mask'].to(self.config.device)
        }
        outputs = self.pretrained_model(**inputs, output_hidden_states=True)
        seq_output = outputs[-1][-1]

        if method == 'token':
            head_spans = batch['head_spans'].to(self.config.device)  # (batch, 2)
            tail_spans = batch['tail_spans'].to(self.config.device)  # (batch,2 )

            # taking the first token as the representation for the entity
            head_rep = onedim_gather(seq_output, dim=1, index=head_spans[:, 0].unsqueeze(1))
            tail_rep = onedim_gather(seq_output, dim=1, index=tail_spans[:, 0].unsqueeze(1))
            feat = torch.cat([head_rep, tail_rep], dim=2).squeeze(1)

        elif method == 'mask':
            mask_bpe_idx = batch['mask_bpe_idx'].to(self.config.device)  # (batch)
            feat = onedim_gather(seq_output, dim=1, index=mask_bpe_idx.unsqueeze(1)).squeeze(1)

        else:
            raise NotImplementedError
        return feat

    def compute_logits(self, batch: Dict, method: str = 'token', view_idx: int = 0):
        '''
        :param batch: batchof data
        :param seq_output: (batch, seq_len, hidden_dim)
        :param method: str, one of 'token', 'mask'
        :param view_idx: int
        '''
        feat = self._compute_features(batch, method)

        view_model = self.views[view_idx]
        common_space_feat = view_model['common_space_proj'](feat)
        known_head_logits = view_model['known_type_classifier'](common_space_feat)
        if not self.config.constrain_pred_type == 'seen_only':
            unknown_head_logits = view_model['unknown_type_classifier'](common_space_feat)
            logits = torch.cat([known_head_logits, unknown_head_logits], dim=1)
            return logits
        else:
            return known_head_logits

    def _on_train_batch_start(self):
        # normalize all centroids
        for view_model in self.views:
            view_model['known_type_classifier'].update_centroid()
            view_model['unknown_type_classifier'].update_centroid()
        return

    def update_centers(self, centers: torch.FloatTensor, known: bool = True, view_idx: int = 0):
        if known or self.config.constrain_pred_type == 'seen_only':
            self.views[view_idx]['known_type_center_loss'].centers = centers
        else:
            self.views[view_idx]['unknown_type_center_loss'].centers = centers
        return


class SinkhornKnopp(torch.nn.Module):
    def __init__(self, num_iters=3, epsilon=0.05, queue_len: int = 1024, classes_n: int = 10, delta=0.0):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.delta = delta

        self.classes_n = classes_n
        self.queue_len = queue_len
        self.register_buffer(name='logit_queue', tensor=torch.zeros((queue_len, classes_n)), persistent=False)
        self.cur_len = 0

    def add_to_queue(self, logits: torch.FloatTensor) -> None:
        '''
        :param logits: (N, K)
        '''
        batch_size = logits.size(0)
        classes_n = logits.size(1)
        assert (classes_n == self.classes_n)

        new_queue = torch.concat([logits, self.logit_queue], dim=0)
        self.logit_queue = new_queue[:self.queue_len, :]

        self.cur_len += batch_size

        self.cur_len = min(self.cur_len, self.queue_len)

        return

    def queue_full(self) -> bool:
        return self.cur_len == self.queue_len

    @torch.no_grad()
    def forward(self, logits: torch.FloatTensor):
        '''
        :param logits: (N, K)
        '''
        batch_size = logits.size(0)
        all_logits = self.logit_queue

        initial_Q = torch.softmax(all_logits / self.epsilon, dim=1)
        # Q = torch.exp(logits / self.epsilon).t() # (K, N)
        Q = initial_Q.clone().t()
        N = Q.shape[1]
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        assert (torch.any(torch.isinf(sum_Q)) == False), "sum_Q is too large"
        Q /= sum_Q

        for it in range(self.num_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            sum_of_rows += self.delta  # for numerical stability
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols
            Q /= N

        Q *= N  # the colomns must sum to 1 so that Q is an assignment

        batch_assignments = Q.t()[:batch_size, :]
        return batch_assignments, sum_of_rows.squeeze(), sum_of_cols.squeeze()


class Prototypes(nn.Module):
    def __init__(self, feat_dim, num_prototypes, norm: bool = False):
        super().__init__()

        if norm:
            self.norm = nn.LayerNorm(feat_dim)
        else:
            self.norm = lambda x: x
        self.prototypes = nn.Linear(feat_dim, num_prototypes, bias=False)

    @torch.no_grad()
    def initialize_prototypes(self, centers):
        self.prototypes.weight.copy_(centers)
        return

    @torch.no_grad()
    def normalize_prototypes(self):
        w = self.prototypes.weight.data.clone()
        w = F.normalize(w, dim=1, p=2)
        self.prototypes.weight.copy_(w)

    def freeze_prototypes(self):
        self.prototypes.requires_grad_(False)

    def unfreeze_prototypes(self):
        self.prototypes.requires_grad_(True)

    def forward(self, x):
        x = self.norm(x)
        return self.prototypes(x)


class MLP(nn.Module):
    '''
    Simple n layer MLP with ReLU activation and batch norm.
    The order is Linear, Norm, ReLU
    '''

    def __init__(self, feat_dim: int, hidden_dim: int, latent_dim: int,
                 norm: bool = False, norm_type: str = 'batch', layers_n: int = 1, dropout_p: float = 0.1):
        '''
        :param norm_type: one of layer, batch
        '''
        super().__init__()
        self.feat_dim = feat_dim
        self._hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.input2hidden = nn.Linear(feat_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_p)
        layers = [self.dropout, ]
        for i in range(layers_n):
            if i == 0:
                layers.append(nn.Linear(feat_dim, hidden_dim))
                out_dim = hidden_dim
            elif i == 1:
                layers.append(nn.Linear(hidden_dim, latent_dim))
                out_dim = latent_dim
            else:
                layers.append(nn.Linear(latent_dim, latent_dim))
                out_dim = latent_dim
            if norm:
                if norm_type == 'batch':
                    layers.append(nn.BatchNorm1d(out_dim))
                else:
                    layers.append(nn.LayerNorm(out_dim))
            if i < layers_n - 1:  # last layer has no relu
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, input):
        '''
        :param input: torch.FloatTensor (batch, ..., feat_dim)

        :return output: torch.FloatTensor (batch, ..., hidden_dim)
        '''
        output = self.net(input.reshape(-1, self.feat_dim))

        original_shape = input.shape
        new_shape = tuple(list(input.shape[:-1]) + [self.latent_dim])

        output = output.reshape(new_shape)
        return output


class ReconstructionNet(nn.Module):
    '''
    projection from hidden_size back to feature_size.
    '''

    def __init__(self, feature_size: int, hidden_size: int, latent_size: int) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        assert (feature_size > hidden_size)
        self.latent_size = latent_size
        self.net = nn.Sequential(
            nn.Linear(in_features=self.latent_size, out_features=self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.feature_size)
        )

    def forward(self, inputs: torch.FloatTensor):
        output = self.net(inputs.reshape(-1, self.hidden_size))
        new_shape = tuple(list(inputs.shape[:-1]) + [self.feature_size])

        output = output.reshape(new_shape)
        return output


class CommonSpaceCache(nn.Module):
    '''
    A cache for saving common space embeddings and using it to compute contrastive loss.
    '''

    def __init__(self, feature_size: int, known_cache_size: int, unknown_cache_size: int, metric_type: str = 'cosine',
                 sim_thres: float = 0.8) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.known_cache_size = known_cache_size
        self.unknown_cache_size = unknown_cache_size
        self.known_len = 0
        self.unknown_len = 0

        self.metric_type = metric_type
        self.metric = nn.CosineSimilarity(dim=2, eps=1e-8)
        self.sim_thres = sim_thres

        self.temp = 0.1  # temperature for softmax

        self.register_buffer("known_cache", torch.zeros((known_cache_size, feature_size), dtype=torch.float),
                             persistent=False)
        self.register_buffer("unknown_cache", torch.zeros((unknown_cache_size, feature_size), dtype=torch.float),
                             persistent=False)

        self.register_buffer("known_labels", torch.zeros((known_cache_size,), dtype=torch.long), persistent=False)
        self.register_buffer("unknown_labels", torch.zeros((unknown_cache_size,), dtype=torch.long), persistent=False)

    def cache_full(self) -> bool:
        if (self.known_len == self.known_cache_size) and (self.unknown_len == self.unknown_cache_size):
            return True
        else:
            return False

    @torch.no_grad()
    def update_batch(self, embeddings: torch.FloatTensor, is_labeled_mask: torch.BoolTensor,
                     labels: Optional[torch.LongTensor] = None) -> None:
        '''
        Add embeddings to cache.
        :param embeddings: (batch, feature_size)
        '''
        embeddings_detached = embeddings.detach()

        known_embeddings = embeddings_detached[is_labeled_mask, :]
        known_size = known_embeddings.size(0)
        new_known_cache = torch.concat([known_embeddings, self.known_cache], dim=0)
        self.known_cache = new_known_cache[:self.known_cache_size]
        self.known_len = min(self.known_len + known_size, self.known_cache_size)
        if labels != None:
            known_labels = labels[is_labeled_mask]
            self.known_labels = torch.concat([known_labels, self.known_labels], dim=0)[:self.known_cache_size]
            unknown_labels = labels[~is_labeled_mask]
            self.unknown_labels = torch.concat([unknown_labels, self.unknown_labels], dim=0)[:self.unknown_cache_size]

        unknown_embeddings = embeddings_detached[~is_labeled_mask, :]
        unknown_size = unknown_embeddings.size(0)
        new_unknown_cache = torch.concat([unknown_embeddings, self.unknown_cache], dim=0)
        self.unknown_cache = new_unknown_cache[:self.unknown_cache_size]
        self.unknown_len = min(self.unknown_len + unknown_size, self.unknown_cache_size)
        return

    @torch.no_grad()
    def get_positive_example(self, embedding: torch.FloatTensor, known: bool = False) -> Tuple[
        torch.FloatTensor, torch.BoolTensor]:
        '''
        :param embeddings (N, feature_dim)

        :returns (N, feature_dim)
        '''
        embedding_detached = embedding.detach()
        if known:
            cache = self.known_cache
            label_cache = self.known_labels
        else:
            cache = self.unknown_cache
            label_cache = self.unknown_labels

        if self.metric_type == 'cosine':
            similarity = self.metric(embedding_detached.unsqueeze(dim=1), cache.unsqueeze(dim=0))  # N, cache_size
        else:
            similarity = torch.einsum("ik,jk->ij", embedding_detached, cache)

        max_sim, max_idx = torch.max(similarity, dim=1)  # (N, )
        min_thres = self.sim_thres
        valid_pos_mask = (max_sim > min_thres)  # (N, )
        pos_embeddings = cache[max_idx, :]  # (N, feature_dim)
        pos_labels = label_cache[max_idx]  # (N, )

        return pos_embeddings, valid_pos_mask, pos_labels

    @torch.no_grad()
    def get_negative_example_for_unknown(self, embedding: torch.FloatTensor, k: int = 3) -> Tuple[
        torch.FloatTensor, torch.FloatTensor]:
        '''
        Take half of the negative examples from the unknown cache and half from the known cache.
        :param embeddings (N, feature_dim)
        '''
        embedding_detached = embedding.detach()
        N = embedding_detached.size(0)
        if self.metric_type == 'cosine':
            unknown_similarity = self.metric(embedding_detached.unsqueeze(dim=1),
                                             self.unknown_cache.unsqueeze(dim=0))  # N, cache_size
        else:
            unknown_similarity = torch.einsum('ik,jk->ij', embedding_detached, self.unknown_cache)

        sorted_unk_idx = torch.argsort(unknown_similarity, dim=1)  # N, cache_size
        unk_n = ceil(sorted_unk_idx.size(1) / 2)
        candidate_neg_unk_idx = sorted_unk_idx[:, :unk_n]  # N, cache_size/2
        # this is used for generating indexes
        neg_unk_list = []
        for i in range(N):
            random_idx = torch.randperm(n=unk_n, dtype=torch.long, device=embedding.device)[:k]
            chosen_neg_unk_idx = candidate_neg_unk_idx[i, :][random_idx]
            chosen_neg_unk = self.unknown_cache[chosen_neg_unk_idx, :]  # K, feature_size
            neg_unk_list.append(chosen_neg_unk)

        if self.metric_type == 'cosine':
            known_similarity = self.metric(embedding_detached.unsqueeze(dim=1),
                                           self.known_cache.unsqueeze(dim=0))  # (N, cache_size)
        else:
            known_similarity = torch.einsum("ik,jk->ij", embedding_detached, self.known_cache)

        sorted_known_idx = torch.argsort(known_similarity, dim=1,
                                         descending=True)  # choose hard examples (N, cache_size)
        neg_known_list = []
        chosen_neg_known_idx = sorted_known_idx[:, :k]
        for i in range(N):
            chosen_neg_known = self.known_cache[chosen_neg_known_idx[i], :]
            neg_known_list.append(chosen_neg_known)

        neg_unk = torch.stack(neg_unk_list, dim=0)
        neg_known = torch.stack(neg_known_list, dim=0)  # (N, K, feature_size)

        return neg_unk, neg_known

    def get_contrastive_candidates(self, embeddings: torch.FloatTensor, neg_n: int = 6,
                                   labels: Optional[torch.LongTensor] = None):
        N = embeddings.size(0)
        if labels != None: assert (labels.size(0) == N)

        pos_embeddings, valid_pos_mask, pos_labels = self.get_positive_example(embeddings,
                                                                               known=False)  # (N, hidden_dim)
        assert (pos_embeddings.shape == embeddings.shape)
        # report positive sample accuracy
        pos_acc = self.compute_accuracy(labels[valid_pos_mask], pos_labels[valid_pos_mask])

        neg_unk_embeddings, neg_known_embeddings = self.get_negative_example_for_unknown(embeddings, k=ceil(
            neg_n / 2))  # (N, K, hidden_dim)
        candidates = torch.concat([pos_embeddings.unsqueeze(dim=1), neg_unk_embeddings, neg_known_embeddings],
                                  dim=1)  # (N, 2K+1, hidden_dim)
        # scores = torch.einsum('ik,ijk->ij', embeddings, candidates) # (N, 2K+1 )
        # targets = torch.zeros((N,), dtype=torch.long, device=scores.device)
        # loss = F.cross_entropy(scores/self.temp, targets)
        return candidates, valid_pos_mask, pos_acc

    def compute_accuracy(self, labels, other_labels):
        # consider moving average
        assert (labels.shape == other_labels.shape)
        acc = torch.sum(labels == other_labels) * 1.0 / labels.size(0)
        return acc


class ClassifierHead(nn.Module):
    def __init__(self, feature_size: int,
                 n_classes: int, layers_n: int = 1,
                 n_heads: int = 1, dropout_p: float = 0.2, hidden_size: Optional[int] = None) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.n_classes = n_classes
        self.n_heads = n_heads
        if hidden_size:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = feature_size

        if layers_n == 1:
            self.classifier = nn.Sequential(OrderedDict(
                [('dropout', nn.Dropout(p=dropout_p)),
                 ('centroids', Prototypes(feat_dim=self.hidden_size, num_prototypes=self.n_classes))]
            ))
        elif layers_n > 1:
            self.classifier = nn.Sequential(OrderedDict(
                [('mlp',
                  MLP(feat_dim=self.feature_size, hidden_dim=self.hidden_size, latent_dim=self.hidden_size, norm=True,
                      layers_n=layers_n - 1)),
                 ('dropout', nn.Dropout(p=dropout_p)),
                 ('centroids', Prototypes(feat_dim=self.hidden_size, num_prototypes=self.n_classes))]
            ))

    def initialize_centroid(self, centers):
        for n, module in self.classifier.named_modules():
            if n == 'centroids':
                module.initialize_prototypes(centers)

        return

    def update_centroid(self):
        '''
        The centroids are essentially just the vectors in the final Linear layer. Here we normalize them. they are trained along with the model.
        '''
        for n, module in self.classifier.named_modules():
            if n == 'centroids':
                module.normalize_prototypes()

        return

    def freeze_centroid(self):
        '''
        From Swav paper, freeze the prototypes to help with initial optimization.
        '''
        for n, module in self.classifier.named_modules():
            if n == 'centroids':
                module.freeze_prototypes()

        return

    def unfreeze_centroid(self):
        for n, module in self.classifier.named_modules():
            if n == 'centroids':
                module.unfreeze_prototypes()

        return

    def forward(self, inputs: torch.FloatTensor):
        '''
        :params inputs: (batch, feat_dim)

        :returns logits: (batch, n_classes)
        '''
        outputs = self.classifier(inputs)
        return outputs


class CenterLoss(nn.Module):
    '''
    L2 loss for pushing representations close to their centroid.
    '''

    def __init__(self, dim_hidden: int, num_classes: int, lambda_c: float = 1.0, alpha: float = 1.0,
                 weight_by_prob: bool = False):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.alpha = alpha
        self.weight_by_prob = weight_by_prob
        self.register_buffer('centers', torch.zeros((self.num_classes, self.dim_hidden), dtype=torch.float),
                             persistent=False)

    def _compute_prob(self, distance_centers: torch.FloatTensor, y: torch.LongTensor):
        '''
        compute the probability according to student-t distribution
        Bug in original RoCORE code, added (-1) to power operation.
        '''

        q = 1.0 / (1.0 + distance_centers / self.alpha)  # (batch_size, num_class)
        q = q ** ((-1) * (self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        prob = q.gather(1, y.unsqueeze(1)).squeeze()  # (batch_size)
        return prob

    def forward(self, y: torch.LongTensor, hidden: torch.FloatTensor) -> torch.FloatTensor:
        '''
        :param y: (batch_size, )
        :param hidden: (batch_size, dim_hidden)
        '''

        batch_size = hidden.size(0)
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1,
                                                                            0)  # (num_class, batch_size, hid_dim) => (batch_size, num_class, hid_dim)
        centers = getattr(self, 'centers')
        expanded_centers = centers.expand(batch_size, -1, -1)  # (batch_size, num_class, hid_dim)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(
            dim=-1)  # (batch_size, num_class, hid_dim) => (batch_size, num_class)
        intra_distances = distance_centers.gather(1, y.unsqueeze(
            1)).squeeze()  # (batch_size, num_class) => (batch_size, 1) => (batch_size)

        if self.weight_by_prob:
            prob = self._compute_prob(distance_centers, y)
            loss = 0.5 * self.lambda_c * torch.mean(intra_distances * prob)  # (batch_size) => scalar

        else:
            loss = 0.5 * self.lambda_c * torch.mean(intra_distances)  # (batch_size) => scalar
        return loss


def clean_text(text: List[str]) -> List[str]:
    ret = []
    for word in text:
        normalized_word = re.sub(u"([^\u0020-\u007f])", "", word)
        if normalized_word == '' or normalized_word == ' ' or normalized_word == '    ':
            normalized_word = '[UNK]'
        ret.append(normalized_word)
    return ret


def cluster_acc(y_true: np.array, y_pred: np.array, reassign: bool = False):
    """
    Calculate clustering accuracy with assigment

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)  # N*K
    y_pred = y_pred.astype(np.int64)  # N*K
    assert y_pred.size == y_true.size  # same number of clusters

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)  # cost matrix
    try:
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
    except:
        print('D: ', D)
        print('i: ', i)
        print('y_pred[i]: ', y_pred[i])
        print('y_true[i]: ', y_true[i])
        print('y_pred.size: ', y_pred.size)
        raise NotImplementedError

    if reassign:
        mapping = compute_best_mapping(w)
        return sum([w[i, j] for i, j in mapping]) * 1.0 / y_pred.size
    else:
        acc = sum([w[i, i] for i in range(D)]) * 1.0 / y_pred.size
        return acc


def compute_best_mapping(w):
    return np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
