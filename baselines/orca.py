import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn import metrics as sk_metrics
import torch.nn.functional as F
from transformers import AutoModel

from utils.utils_orca import MarginLoss, entropy
from utils.utils_common import hungarian_algorithm, test_metrics
from utils.utils_metrics import p_r_f1_f1cls


class ORCA_Model(LightningModule):
    """
    Relation extraction model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dvc = config.device
        scale = 2
        self.test_labels = np.array([])
        self.test_preds = np.array([])
        self.test_confs = np.array([])
        self.mean_uncertainty = 1

        self.rel_fc = nn.Linear(config.hidden_size * scale, config.n_seen_classes + config.n_novel_classes)
        self.model = AutoModel.from_pretrained(config.foundation_model,
                                               hidden_dropout_prob=config.dropout)

        # ORCA Loss
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.margin_loss = None
        self.entropy_loss = entropy

    def batch_split(self, batch):
        index_labeled = (batch['is_labeled'] == 1).nonzero().squeeze()
        index_unlabeled = (batch['is_labeled'] == 0).nonzero().squeeze()
        batch_labeled = {k: torch.index_select(v, 0, index_labeled) for k, v in batch.items()}
        batch_unlabeled = {k: torch.index_select(v, 0, index_unlabeled) for k, v in batch.items()}
        return batch_labeled, batch_unlabeled

    def forward(self, batch):
        # From batch
        input_ids = batch['input_ids'].to(self.dvc)
        attention_mask = batch['attention_mask'].to(self.dvc)
        h_pos = batch['h_pos'].to(self.dvc)
        t_pos = batch['t_pos'].to(self.dvc)
        h_pos_l = batch['h_pos_l'].to(self.dvc)
        t_pos_l = batch['t_pos_l'].to(self.dvc)

        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        h_state, t_state = [], []
        for i in range(input_ids.size()[0]):
            h_state.append(torch.mean(outputs[0][i, h_pos[i]: h_pos_l[i]], dim=0))
            t_state.append(torch.mean(outputs[0][i, t_pos[i]: t_pos_l[i]], dim=0))

        h_state = torch.stack(h_state, dim=0)
        t_state = torch.stack(t_state, dim=0)
        h_t_state = torch.cat((h_state, t_state), 1)  # (batch_size, hidden_size*2)

        # FCL
        logits = self.rel_fc(h_t_state)  # (batch_size, rel_num)
        _, output = torch.max(logits, 1)
        return logits, output, h_t_state  # logits, preds, features

    def on_train_epoch_start(self):
        self.mean_uncertainty = min(self.mean_uncertainty, self.config.max_mean_uncert)
        self.margin_loss = MarginLoss(m=-1 * self.mean_uncertainty)

    def training_step(self, batch, batch_idx):
        # 1/2 weak labels for unlabeled data, 1/2 gold labels for labeled data
        batch_labeled, batch_unlabeled = self.batch_split(batch)
        batch_size_labeled = len(batch_labeled['input_ids'])
        batch_size_unlabeled = len(batch_unlabeled['input_ids'])
        logits_labeled, _, features_labeled = self(batch_labeled)
        logits_unlabeled, _, _ = self(batch_unlabeled)

        # Similarity labels
        features_detached = features_labeled.detach()
        feat_norm = features_detached / torch.norm(features_detached, 2, 1, keepdim=True)
        cosine_dist = torch.mm(feat_norm, feat_norm.t())
        pos_pairs = []
        labels = batch_labeled['labels']
        labels_np = labels.cpu().numpy()

        # Labeled part
        for n in range(batch_size_labeled):  # 0 to n labeled instances
            rel_id = labels_np[n]  # rel id -- mostly zeros
            match_idxs = np.where(labels_np == rel_id)[0]  # indices where labels == rel_id
            if len(match_idxs) == 1:
                pos_pairs.append(match_idxs[0])
            else:
                selec_idx = np.random.choice(match_idxs, 1)
                while selec_idx == n:
                    selec_idx = np.random.choice(match_idxs, 1)
                pos_pairs.append(int(selec_idx))

        # Loss
        loss = self.margin_loss(logits_labeled, labels)

        # Unlabeled part
        if batch_size_labeled > 2 and batch_size_unlabeled > 2:
            unlabel_cosine_dist = cosine_dist[batch_size_labeled:, :]
            try:
                vals, pos_idx = torch.topk(unlabel_cosine_dist, 2, dim=1)
            except RuntimeError:
                print('batch_size_labeled:', batch_size_labeled)
                print('batch_size_unlabeled:', batch_size_unlabeled)
                print('unlabel_cosine_dist:', unlabel_cosine_dist)
                raise RuntimeError
            pos_idx = pos_idx[:, 1].cpu().numpy().flatten().tolist()
            pos_pairs.extend(pos_idx)

            # Clustering and consistency losses
            prob_labeled = F.softmax(logits_labeled, dim=1)
            prob_unlabeled = F.softmax(logits_unlabeled, dim=1)
            max_idx_for_unlabeled_batch = prob_unlabeled.size(0) - 1
            if max(pos_pairs) > max_idx_for_unlabeled_batch:
                pos_pairs = [x for x in pos_pairs if x < max_idx_for_unlabeled_batch]
                pos_prob = prob_unlabeled[pos_pairs, :]  # 49 x 60
                pos_prob_expanded = torch.zeros_like(prob_labeled)
                pos_prob_size_0 = pos_prob.size(0)
                pos_prob_expanded[:pos_prob_size_0] = pos_prob
                pos_prob = pos_prob_expanded
                pos_sim = torch.bmm(prob_labeled.view(batch_size_labeled, 1, -1),
                                    pos_prob.view(batch_size_labeled, -1, 1)).squeeze()
                ones = torch.ones_like(pos_sim)
                ones[pos_prob_size_0:] = 0
            else:
                pos_prob = prob_unlabeled[pos_pairs, :]
                pos_sim = torch.bmm(prob_labeled.view(batch_size_labeled, 1, -1),
                                    pos_prob.view(batch_size_labeled, -1, 1)).squeeze()
                ones = torch.ones_like(pos_sim)

            # BCE loss
            loss += self.bce_loss(pos_sim, ones)

            # Entropy loss
            if "++" in self.config.model_name:  # ORCA++ modification
                loss -= self.entropy_loss(torch.mean(prob_unlabeled, 0))
            else:  # Regular ORCA
                loss -= self.entropy_loss(torch.mean(prob_labeled, 0))

        self.log("train_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        labels = batch['labels'].to(self.dvc)
        logits, outputs, features = self(batch)
        confs, _ = F.softmax(logits, dim=1).max(1)

        # Collect labels, predictions, and confidences across batches
        self.test_labels = np.append(self.test_labels, labels.cpu().numpy())
        self.test_preds = np.append(self.test_preds, outputs.cpu().numpy())
        self.test_confs = np.append(self.test_confs, confs.cpu().numpy())
        return 0

    def on_test_epoch_end(self):
        self.test_labels = self.test_labels.astype(int)
        self.test_preds = self.test_preds.astype(int)
        self.test_confs = self.test_confs.astype(int)

        # Update mean uncertainty
        self.mean_uncertainty = 1 - np.mean(self.test_confs)

        # Compute metrics
        metrics = test_metrics(self.config, self.test_preds, self.test_labels)

        # Log metrics
        metrics["mean_uncertainty"] = self.mean_uncertainty
        self.log_dict(metrics, sync_dist=True)
        print(metrics)
        self.clear_numpy_arrays()

    def clear_numpy_arrays(self):
        self.test_labels = np.array([])
        self.test_preds = np.array([])
        self.test_confs = np.array([])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=float(self.config.lr))
