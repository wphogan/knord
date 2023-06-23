from collections import defaultdict
from os.path import join
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.cluster import KMeans
from torchmetrics import Accuracy
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# Local imports
from utils.utils_common import get_n_classes_unlabeled, test_metrics
from utils.utils_tabs import MultiviewModel


class TypeDiscoveryModel(LightningModule):
    def __init__(self, config, train_len, tokenizer) -> None:
        super().__init__()
        self.config = config
        self.train_len = train_len
        self.model = MultiviewModel(config, tokenizer)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.config.n_seen_classes)

        self.test_preds = np.array([])
        self.test_probs = np.array([])
        self.test_targets = np.array([])
        self.test_uids = np.array([])

    def load_pretrained_model(self, pretrained_dict):
        '''
        load model and common space proj,
        '''
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           ('pretrained_model' in k or 'common_space_proj' in k)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        return

    def _compute_psuedo_labels(self, pred_logits: torch.FloatTensor, pred_logits_other: torch.FloatTensor,
                               is_labeled_mask: torch.BoolTensor, targets: torch.FloatTensor, temp: float = 0.5) \
            -> Union[torch.FloatTensor, None]:
        '''
        Compute the psuedo labels by combine and sharpen.
        :param pred_logits_other: (batch, M+N)
        :param is_labeled_mask: (batch)
        :param targets: (batch, M+N)
        :param temp: float in (0, 1)
        '''
        target_logits = pred_logits_other
        target_logits_d = target_logits.detach()
        n_known = self.config.n_seen_classes

        # TABs (default): assumes all unlabeled data is novel
        if self.config.constrain_pred_type == 'novel_only' or self.config.use_confidence:
            targets[~is_labeled_mask, n_known:] = torch.softmax(target_logits_d[~is_labeled_mask, n_known:] / temp,dim=1)
        # TABs (modified): assumes unlabeled data is either seen or unseen
        elif self.config.constrain_pred_type == 'unconstrained':
            targets[~is_labeled_mask, :] = torch.softmax(target_logits_d[~is_labeled_mask, :] / temp, dim=1)
        else:
            raise ValueError(f"invalid constrain type {self.config.constrain_pred_type}")
        return targets

    def _smooth_targets(self, targets: torch.FloatTensor, all_types: int):
        assert self.config.label_smoothing_alpha > 0.0
        alpha = self.config.label_smoothing_alpha
        alpha = max(alpha, 0.0)
        targets = (1 - alpha) * targets + alpha * torch.full_like(targets, fill_value=1.0 / all_types,
                                                                  dtype=torch.float, device=self.config.device)
        return targets

    def _compute_targets(self, batch_size: int, labels: torch.LongTensor, is_labeled_mask: torch.BoolTensor,
                         predicted_logits: torch.FloatTensor, predicted_logits_other: torch.FloatTensor,
                         hard: bool = False):

        if self.config.constrain_pred_type == 'seen_only':
            all_types = self.config.n_seen_classes
        else:
            all_types = self.config.n_seen_classes + self.config.n_novel_classes

        targets = torch.zeros((batch_size, all_types), dtype=torch.float,
                              device=self.config.device)  # soft targets

        assert labels.max() <= all_types, f'labels.max < all_types --> {labels.max()} !<= {all_types}'
        known_labels = F.one_hot(labels, num_classes=all_types).float()  # (batch, all_types)

        assert (is_labeled_mask.long().max() <= 1 and is_labeled_mask.long().min() >= 0)
        if is_labeled_mask.long().max() > 0:  # has known elements
            targets[is_labeled_mask, :] = known_labels[is_labeled_mask, :]

        if is_labeled_mask.long().min() < 1:  # has unknown elements
            # compute psuedo labels
            targets = self._compute_psuedo_labels(predicted_logits, predicted_logits_other, is_labeled_mask, targets,
                                                  temp=self.config.temp)

            targets_other = self._compute_psuedo_labels(predicted_logits_other, predicted_logits, is_labeled_mask,
                                                        targets,
                                                        temp=self.config.temp)
        else:
            targets_other = targets

        targets = self._smooth_targets(targets, all_types)
        targets_other = self._smooth_targets(targets_other, all_types)

        return targets, targets_other

    def _compute_batch_pairwise_loss(self, predicted_logits: torch.FloatTensor,
                                     labels: Optional[torch.LongTensor] = None,
                                     targets: Optional[torch.FloatTensor] = None, sigmoid: float = 2.0):
        known_types = self.config.n_seen_classes

        if self.config.constrain_pred_type == 'novel_only' or self.config.use_confidence:
            predicted_logits = predicted_logits[:, known_types:]  # Logits for unlabeled data only
        elif self.config.constrain_pred_type == 'unconstrained':
            pass  # Use unconstrained logits
        else:
            raise ValueError(f"invalid constrain type {self.config.constrain_pred_type}")

        def compute_kld(p_logit, q_logit):
            p = F.softmax(p_logit, dim=-1)  # (B, B, n_class)
            q = F.softmax(q_logit, dim=-1)  # (B, B, n_class)
            return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim=-1)  # (B, B)

        if targets != None:
            if self.config.constrain_pred_type == 'novel_only' or self.config.use_confidence:
                targets = targets[:, known_types:]
            elif self.config.constrain_pred_type == 'unconstrained':
                pass  # Use unconstrained logits
            assert (targets.shape == predicted_logits.shape)
            # convert targets into pairwise labels
            targets = targets.detach()
            batch_size = targets.size(0)
            target_val, target_idx = torch.max(targets, dim=1)
            pairwise_label = (target_idx.unsqueeze(0) == target_idx.unsqueeze(1)).float()  # (batch, batch)
        else:
            batch_size = labels.size(0)
            label_mask = (labels != -1)

            pairwise_label = (labels[label_mask].unsqueeze(0) == labels[label_mask].unsqueeze(1)).float()
            predicted_logits = predicted_logits[label_mask]

        # KL loss
        expanded_logits = predicted_logits.expand(batch_size, -1, -1)
        expanded_logits2 = expanded_logits.transpose(0, 1)
        kl1 = compute_kld(expanded_logits.detach(), expanded_logits2)
        kl2 = compute_kld(expanded_logits2.detach(), expanded_logits)  # (batch_size, batch_size)
        pair_loss = torch.mean(pairwise_label * (kl1 + kl2) + (1 - pairwise_label) * (
                torch.relu(sigmoid - kl1) + torch.relu(sigmoid - kl2)))

        return pair_loss

    def _compute_consistency_loss(self, predicted_logits: torch.FloatTensor, predicted_logits_other: torch.FloatTensor,
                                  is_labeled_mask: torch.BoolTensor):
        assert (predicted_logits.shape == predicted_logits_other.shape)

        # KL loss
        def compute_kld(p_logit, q_logit):
            p = F.softmax(p_logit, dim=-1)  # (B, n_class)
            q = F.softmax(q_logit, dim=-1)  # (B, n_class)
            return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim=-1)  # (B,)

        kl1 = compute_kld(predicted_logits.detach(), predicted_logits_other)
        kl2 = compute_kld(predicted_logits_other.detach(), predicted_logits)
        consistency_loss = torch.mean(kl1 + kl2) * 0.5
        return consistency_loss

    def on_train_epoch_start(self) -> None:
        if self.config.supervised_pretrain: return

        if self.current_epoch == 0:
            print('updating cluster centers....')
            train_dl = self.trainer.train_dataloader.loaders
            known_centers = torch.zeros((2, self.config.n_seen_classes, self.config.kmeans_dim),
                                        device=self.config.device)
            num_samples = torch.zeros((2, self.config.n_seen_classes), device=self.config.device)
            with torch.no_grad():
                uid2pl = defaultdict(list)  # pseudo labels
                unknown_uid_list = defaultdict(list)
                unknown_vec_list = defaultdict(list)
                seen_uid = [set(), set()]  # oversampling for the unknown part, so we remove them here
                for batch in tqdm(iter(train_dl)):
                    labels = batch[0]['labels']
                    is_labeled_mask = batch[0]['is_labeled_mask']
                    metadata = batch[0]['meta']
                    batch_size = len(metadata)
                    for view_idx, view in enumerate(batch):
                        # move batch to gpu
                        for key in ['token_ids', 'attn_mask', 'head_spans', 'tail_spans', 'mask_bpe_idx',
                                    'trigger_spans']:
                            if key in view: view[key] = view[key].to(self.config.device)

                        feature_type = view['meta'][0]['feature_type']
                        feats = self.model._compute_features(view, feature_type)
                        view_model = self.model.views[view_idx]
                        commonspace_rep = view_model['common_space_proj'](feats)  # (batch_size, hidden_dim)
                        for i in range(batch_size):
                            if is_labeled_mask[i] == True:
                                l = labels[i]
                                known_centers[view_idx][l] += commonspace_rep[i]
                                num_samples[view_idx][l] += 1
                            else:
                                uid = metadata[i]['uid']
                                if uid not in seen_uid[view_idx]:
                                    seen_uid[view_idx].add(uid)
                                    unknown_uid_list[view_idx].append(uid)
                                    unknown_vec_list[view_idx].append(commonspace_rep[i])

                assert len(unknown_uid_list[0]) == len(unknown_uid_list[1])
                for view_idx in range(2):
                    # cluster unknown classes
                    rep = torch.stack(unknown_vec_list[view_idx], dim=0).cpu().numpy()
                    class_offset = 0
                    if self.config.constrain_pred_type == 'novel_only' or self.config.use_confidence:
                        class_offset = self.config.n_seen_classes
                    n_classes_unlabeled = get_n_classes_unlabeled(self.config)
                    clf = KMeans(n_clusters=n_classes_unlabeled, random_state=0, algorithm='full')
                    label_pred = clf.fit_predict(rep)  # from 0 to self.config.new_class - 1

                    for i in range(len(unknown_vec_list[view_idx])):
                        uid = unknown_uid_list[view_idx][i]
                        pseudo = label_pred[i]
                        uid2pl[uid].append(
                            pseudo + class_offset)  # shift the pseudo label to the unknown part

                    # update center for known types
                    for c in range(self.config.n_seen_classes):
                        known_centers[view_idx][c] /= num_samples[view_idx][c]

                train_dl.dataset.update_pseudo_labels(uid2pl)
                print('updating pseudo labels...')
                pl_acc = train_dl.dataset.check_pl_acc()
                self.log('train/kmeans_acc', pl_acc, on_epoch=True, sync_dist=True)

        return

    def training_step(self, batch, batch_idx):
        self.model._on_train_batch_start()
        view_n = len(batch)
        batch_size = len(batch[0]['meta'])
        labels = batch[0]['labels']
        is_labeled_mask = batch[0]['is_labeled_mask']  # (batch, )

        for i in range(view_n):
            # check feature type
            feature_type = batch[i]['meta'][0]['feature_type']
            if i == 0:
                predicted_logits = self.model.compute_logits(batch[i], method=feature_type, view_idx=i)
            else:
                predicted_logits_other = self.model.compute_logits(batch[i], method=feature_type, view_idx=i)

        targets, targets_other = self._compute_targets(batch_size, labels, is_labeled_mask, predicted_logits,
                                                       predicted_logits_other, hard=False)
        if (targets is None) or (targets_other is None):
            return 0.0  # no loss

        known_loss = F.cross_entropy(predicted_logits[is_labeled_mask, :],
                                     target=targets[is_labeled_mask, :]) + \
                     F.cross_entropy(predicted_logits_other[is_labeled_mask, :],
                                     target=targets_other[is_labeled_mask, :])

        if self.config.supervised_pretrain:
            loss = known_loss
        else:
            pl_loss = self._compute_batch_pairwise_loss(predicted_logits[~is_labeled_mask],
                                                        labels=batch[1]['pseudo_labels'][~is_labeled_mask],
                                                        sigmoid=self.config.sigmoid) \
                      + self._compute_batch_pairwise_loss(predicted_logits_other[~is_labeled_mask],
                                                          labels=batch[0]['pseudo_labels'][~is_labeled_mask],
                                                          sigmoid=self.config.sigmoid)
            self.log('train/unknown_margin_loss', pl_loss, sync_dist=True)
            loss = known_loss + pl_loss

        self.log('train/known_loss', known_loss, sync_dist=True)
        assert self.config.consistency_loss > 0
        consistency_loss = self._compute_consistency_loss(predicted_logits, predicted_logits_other, is_labeled_mask)
        self.log('train/consistency_loss', consistency_loss, sync_dist=True)
        loss += consistency_loss
        self.log('train/loss', loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        view_n = len(batch)
        batch_size = len(batch[0]['meta'])
        labels = batch[0]['labels']
        is_labeled_mask = batch[0]['is_labeled_mask']  # (batch, )
        assert len(is_labeled_mask) == len(labels), f'Val is only labeled data: {len(is_labeled_mask)} != {len(labels)}'
        for i in range(view_n):
            feature_type = batch[i]['meta'][0]['feature_type']
            if i == 0:
                predicted_logits = self.model.compute_logits(batch[i], method=feature_type, view_idx=0)
            else:
                predicted_logits_other = self.model.compute_logits(batch[i], method=feature_type, view_idx=1)

        targets, targets_other = self._compute_targets(batch_size, labels, is_labeled_mask, predicted_logits,
                                                       predicted_logits_other, hard=False)
        if (targets is None) or (targets_other is None):
            val_loss = 0.0  # no loss

        else:
            val_known_loss = F.cross_entropy(predicted_logits[is_labeled_mask, :], target=targets[is_labeled_mask, :]) + \
                             F.cross_entropy(predicted_logits_other[is_labeled_mask, :],
                                             target=targets_other[is_labeled_mask, :])
            val_consistency_loss = self._compute_consistency_loss(predicted_logits, predicted_logits_other,
                                                                  is_labeled_mask)

            val_loss = val_known_loss + val_consistency_loss
            self.log('val/known_loss', val_known_loss, sync_dist=True)
            self.log('val/consistency_loss', val_consistency_loss, sync_dist=True)

        self.log('val/loss', val_loss, sync_dist=True)

        # Predictions
        _, preds = torch.max(predicted_logits, 1)
        self.val_acc(preds, labels)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx) -> None:
        view_n = len(batch)
        labels = batch[0]['labels_unseen']
        for i in range(view_n):
            feature_type = batch[i]['meta'][0]['feature_type']
            if i == 0:
                logits_known = self.model.compute_logits(batch[i], method=feature_type, view_idx=0)
            else:
                logits_unlabeled = self.model.compute_logits(batch[i], method=feature_type, view_idx=1)

        # TABS++ Version 1. Confidence-based seen/novel masking
        if self.config.use_confidence:
            n_preds_qa = 0
            is_seen_conf_based_mask = torch.tensor([x['is_seen_conf_based'] for x in batch[0]['meta']]).to(
                self.config.device)
            is_novel_conf_based_mask = ~is_seen_conf_based_mask

            # Only predict known classes for known instances (conf based)
            if sum(is_seen_conf_based_mask).item() > 0:
                logits_seen = logits_known[is_seen_conf_based_mask]
                prob, pred_cluster = torch.max(logits_seen[:, :self.config.n_seen_classes],
                                               dim=1)  # constrained: seen classes
                self.test_preds = np.append(self.test_preds, pred_cluster.cpu().numpy())
                self.test_probs = np.append(self.test_probs, prob.cpu().numpy())
                n_preds_qa += len(pred_cluster)

            # Only predict novel classes for novel instances (conf based)
            if sum(is_novel_conf_based_mask).item() > 0:
                logits_novel = logits_unlabeled[is_novel_conf_based_mask]
                prob, pred_cluster = torch.max(logits_novel[:, self.config.n_seen_classes:],
                                               dim=1)  # constrained: novel classes
                self.test_preds = np.append(self.test_preds, pred_cluster.cpu().numpy())
                self.test_probs = np.append(self.test_probs, prob.cpu().numpy())
                n_preds_qa += len(pred_cluster)

            # QA Check
            assert len(labels) == n_preds_qa, f'len(targets) {len(labels)} != n_preds_qa {n_preds_qa}'

        # TABS
        else:
            if self.config.constrain_pred_type == 'novel_only':  # TABs (novel only)
                prob, pred_cluster = torch.max(logits_unlabeled[:, self.config.n_seen_classes:], dim=1)
            elif self.config.constrain_pred_type == 'unconstrained':
                prob, pred_cluster = torch.max(logits_unlabeled, dim=1)
            else:
                raise ValueError(
                    f'Unknown tabs_orig_pred_type {self.config.constrain_pred_type}. Must be seen or novel.')
            self.test_preds = np.append(self.test_preds, pred_cluster.cpu().numpy())
            self.test_probs = np.append(self.test_probs, prob.cpu().numpy())
            self.test_uids = np.append(self.test_uids, [x['uid'] for x in batch[0]['meta']])

        # Collect Targets
        self.test_targets = np.append(self.test_targets, labels.cpu().numpy())
        return

    def test_epoch_end(self, outputs) -> None:
        self._type_adjust()
        # Compute metrics
        metrics = test_metrics(self.config, self.test_preds, self.test_targets)
        self.log_dict(metrics, sync_dist=True)
        print(metrics)
        self._clear_data()
        return

    def save_confidence_scores(self, dataloader, split):
        # Treat data as test data just to get confidence scores
        self.model.to(self.config.device)
        self.model.eval()
        self._clear_data()
        with torch.no_grad():
            for batch in dataloader:
                self.test_step(batch, 0)

        # Save confidence scores
        self._type_adjust()
        self._save_predictions(split=split)
        self._clear_data()

    def _save_predictions(self, split='test'):
        df = pd.DataFrame({
            'uid': self.test_uids,
            'pred': self.test_preds,
            'prob': self.test_probs,
            'target': self.test_targets
        })
        fname = join(self.config.output_dir, f'confidences_{split}.csv')
        df.to_csv(fname, index=False)
        print(f'Saved predictions to {fname}')

    def _clear_data(self):
        self.test_preds = np.array([])
        self.test_probs = np.array([])
        self.test_targets = np.array([])
        self.test_uids = np.array([])

    def _type_adjust(self):
        self.test_targets = self.test_targets.astype(int)
        self.test_preds = self.test_preds.astype(int)
        self.test_uids = self.test_uids.astype(int)
        self.test_probs = self.test_probs.astype(float)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate,
                                      eps=self.config.adam_epsilon)

        if self.config.max_steps > 0:
            t_total = self.config.max_steps
            self.config.epochs = self.config.max_steps // self.train_len // self.config.accumulate_grad_batches + 1
        else:
            t_total = self.train_len // self.config.accumulate_grad_batches * self.config.epochs

        print('{} training steps in total.. '.format(t_total))

        # scheduler is called only once per epoch by default
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warmup_steps,
                                                    num_training_steps=t_total)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'linear-schedule',
        }

        return [optimizer, ], [scheduler_dict, ]
