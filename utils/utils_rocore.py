from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

# Local imports
from utils.utils_common import get_n_classes_unlabeled
from utils.utils_tabs import CenterLoss


class ZeroShotModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        # self.IL = args.IL
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config.foundation_model, output_hidden_states=True)
        self.pretrained_model = AutoModel.from_pretrained(config.foundation_model, config=self.model_config)
        self.pretrained_model.resize_token_embeddings(len(tokenizer))
        self.pretrained_model.to(config.device)

        self.initial_dim = self.model_config.hidden_size
        self.layer = self.config.layer
        self.unfreeze_layers = self.layer
        self.pretrained_model = self.finetune(self.pretrained_model, self.unfreeze_layers)  # fix bert weights

        self.similarity_encoder = nn.Sequential(
            nn.Linear(2 * self.initial_dim, self.config.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.config.hidden_dim, self.config.kmeans_dim)
        )
        self.similarity_decoder = nn.Sequential(
            nn.Linear(self.config.kmeans_dim, self.config.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.config.hidden_dim, 2 * self.initial_dim)
        )
        n_classes_unlabeled = get_n_classes_unlabeled(self.config)
        self.ct_loss_u = CenterLoss(dim_hidden=self.config.kmeans_dim, num_classes=n_classes_unlabeled,
                                    alpha=1.0,
                                    weight_by_prob=True)
        self.ct_loss_l = CenterLoss(dim_hidden=self.config.kmeans_dim, num_classes=self.config.n_seen_classes)
        self.labeled_head = nn.Linear(2 * self.initial_dim, self.config.n_seen_classes)
        self.unlabeled_head = nn.Linear(2 * self.initial_dim, n_classes_unlabeled)
        self.bert_params = []
        for name, param in self.pretrained_model.named_parameters():
            if param.requires_grad is True:
                self.bert_params.append(param)

    @staticmethod
    def finetune(model, unfreeze_layers):
        params_name_mapping = ['embeddings', 'layer.0', 'layer.1', 'layer.2', 'layer.3', 'layer.4', 'layer.5',
                               'layer.6', 'layer.7', 'layer.8', 'layer.9', 'layer.10', 'layer.11', 'layer.12']
        for name, param in model.named_parameters():
            param.requires_grad = False
            for ele in [unfreeze_layers]:
                if params_name_mapping[ele] in name:
                    param.requires_grad = True
                    break
        return model

    def get_pretrained_feature(self, input_id, input_mask, head_span, tail_span):
        outputs = self.pretrained_model(input_id,
                                        attention_mask=input_mask)  # (13 * [batch_size, seq_len, bert_embedding_len])
        batch_size = outputs.last_hidden_state.size(0)
        head_entity_rep = torch.stack(
            [torch.max(outputs.last_hidden_state[i, head_span[i][0]:head_span[i][1], :], dim=0)[0] for i in
             range(batch_size)],
            dim=0)
        tail_entity_rep = torch.stack(
            [torch.max(outputs.last_hidden_state[i, tail_span[i][0]:tail_span[i][1], :], dim=0)[0] for i in
             range(batch_size)],
            dim=0)  # (batch_size, bert_embedding)
        pretrained_feat = torch.cat([head_entity_rep, tail_entity_rep], dim=1)  # (batch_size, 2 * bert_embedding)
        return pretrained_feat

    def forward(self, batch: Dict[str, torch.Tensor], mask: Optional[torch.BoolTensor] = None, msg: str = 'similarity'):
        input_ids = batch['token_ids'].to(self.config.device)
        input_mask = batch['attn_mask'].to(self.config.device)
        head_span = batch['head_spans'].to(self.config.device)
        tail_span = batch['tail_spans'].to(self.config.device)

        if mask != None:
            input_ids = input_ids[mask]
            input_mask = input_mask[mask]
            head_span = head_span[mask]
            tail_span = tail_span[mask]

        if msg == 'similarity':  # used for centroid update
            with torch.no_grad():
                pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span,
                                                              tail_span)  # (batch_size, 2 * bert_embedding)
            commonspace_rep = self.similarity_encoder(pretrained_feat)  # (batch_size, keamns_dim)
            return commonspace_rep  # (batch_size, keamns_dim)

        elif msg == 'reconstruct':
            with torch.no_grad():
                pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span,
                                                              tail_span)  # (batch_size, 2 * bert_embedding)
            commonspace_rep = self.similarity_encoder(pretrained_feat)  # (batch_size, kmeans_dim)
            rec_rep = self.similarity_decoder(commonspace_rep)  # (batch_size, 2 * bert_embedding)
            rec_loss = (rec_rep - pretrained_feat).pow(2).mean(-1)
            return commonspace_rep, rec_loss

        elif msg == 'labeled':
            pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span,
                                                          tail_span)  # (batch_size, 2 * bert_embedding)
            logits = self.labeled_head(pretrained_feat)
            return logits  # (batch_size, num_class)

        elif msg == 'unlabeled':
            pretrained_feat = self.get_pretrained_feature(input_ids, input_mask, head_span,
                                                          tail_span)  # (batch_size, 2 * bert_embedding)
            logits = self.unlabeled_head(pretrained_feat)
            return logits  # (batch_size, new_class)

        else:
            raise NotImplementedError('not implemented!')


def L2Reg(net):
    reg_loss = 0
    for name, params in net.named_parameters():
        if name[-4:] != 'bias':
            reg_loss += torch.sum(torch.pow(params, 2))
    return reg_loss


def compute_kld(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)  # (B, B, n_class)
    q = F.softmax(q_logit, dim=-1)  # (B, B, n_class)
    return torch.sum(p * (torch.log(p + 1e-16) - torch.log(q + 1e-16)), dim=-1)  # (B, B)
