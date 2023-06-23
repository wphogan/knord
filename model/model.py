import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn import metrics as sk_metrics
from torch import optim
from transformers import AutoModel

# Local imports
from utils.utils_common import hungarian_algorithm
from utils.utils_metrics import p_r_f1_f1cls


class KnordModel(LightningModule):
    """
    Relation extraction model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dvc = config.device
        scale = 2
        self.val_preds = np.array([])
        self.val_weak_targets = np.array([])
        self.test_preds = np.array([])
        self.test_targets = np.array([])

        self.ce_loss = nn.CrossEntropyLoss()
        self.rel_fc = nn.Linear(config.hidden_size * scale, config.n_seen_classes + config.n_novel_classes)
        self.model = AutoModel.from_pretrained(config.foundation_model,
                                               hidden_dropout_prob=config.dropout)
        self.model = self.model.to(self.dvc)

    def forward(self, batch):
        # From batch
        input_ids = batch['input_ids'].to(self.dvc)
        mask = batch['attention_mask'].to(self.dvc)
        h_pos = batch['h_pos'].to(self.dvc)
        t_pos = batch['t_pos'].to(self.dvc)
        h_pos_l = batch['h_pos_l'].to(self.dvc)
        t_pos_l = batch['t_pos_l'].to(self.dvc)

        # Forward pass
        outputs = self.model(input_ids, mask)
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

    def training_step(self, batch, batch_idx):
        # 1/2 weak labels for unlabeled data, 1/2 gold labels for labeled data
        targets = batch['weak_labels'].to(self.dvc)
        logits, outputs, features = self(batch)
        loss = self.ce_loss(logits, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch['weak_labels'].to(self.dvc)
        logits, outputs, features = self(batch)
        loss = self.ce_loss(logits, targets)
        self.log("val_loss", loss)

        # Collect validation predictions and targets across batches
        self.val_weak_targets = np.append(self.val_weak_targets, targets.cpu().numpy())
        self.val_preds = np.append(self.val_preds, outputs.cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        self.type_adjust()
        _, _, f1_all, _ = p_r_f1_f1cls(self.val_preds, self.val_weak_targets, self.config.all_class_ids)
        self.log("val_f1_all_weak", f1_all, sync_dist=True)
        self.clear_data()

    def test_step(self, batch, batch_idx):
        targets = batch['labels'].to(self.dvc)
        logits, outputs, features = self(batch)
        loss = self.ce_loss(logits, targets)

        # Collect validation predictions and targets across batches
        self.test_preds = np.append(self.test_preds, outputs.cpu().numpy())
        self.test_targets = np.append(self.test_targets, targets.cpu().numpy())
        return loss

    def on_test_epoch_end(self):
        self.type_adjust()

        # Compute metrics
        seen_mask = np.isin(self.test_targets, self.config.seen_class_ids)
        novel_mask = ~seen_mask
        preds_mapped = hungarian_algorithm(self.config, self.test_preds, self.test_targets, seen_mask, novel_mask)
        _, _, f1_all, _ = p_r_f1_f1cls(preds_mapped, self.test_targets, self.config.all_class_ids)
        _, _, f1_seen, _ = p_r_f1_f1cls(preds_mapped[seen_mask], self.test_targets[seen_mask],
                                        self.config.seen_class_ids)
        _, _, f1_novel, _ = p_r_f1_f1cls(preds_mapped[novel_mask], self.test_targets[novel_mask],
                                         self.config.novel_class_ids)
        unseen_nmi = sk_metrics.normalized_mutual_info_score(self.test_targets[novel_mask], preds_mapped[novel_mask])

        # Log metrics
        metrics = {
            "f1_all": f1_all,
            "f1_seen": f1_seen,
            "f1_novel": f1_novel,
            "unseen_nmi": unseen_nmi
        }
        self.log_dict(metrics, sync_dist=True)
        print(metrics)
        self.clear_data()

    def type_adjust(self):
        self.val_preds = self.val_preds.astype(int)
        self.val_weak_targets = self.val_weak_targets.astype(int)
        self.test_preds = self.test_preds.astype(int)
        self.test_targets = self.test_targets.astype(int)

    def clear_data(self):
        self.val_preds = np.array([])
        self.val_weak_targets = np.array([])
        self.test_preds = np.array([])
        self.test_targets = np.array([])

    def configure_optimizers(self):
        return getattr(optim, self.config.optimizer)(self.model.parameters(), lr=float(self.config.lr))
