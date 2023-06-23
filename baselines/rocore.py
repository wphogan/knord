from os.path import join
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from sklearn.cluster import KMeans
from torchmetrics import Accuracy
from tqdm import tqdm

# Local imports
from utils.utils_common import get_n_classes_unlabeled, test_metrics
from utils.utils_rocore import ZeroShotModel, L2Reg, compute_kld


class RoCOREModel(LightningModule):
    def __init__(self, config, train_len, tokenizer) -> None:
        super().__init__()
        self.config = config
        self.train_len = train_len  # this is required to set up the optimizer
        self.net = ZeroShotModel(config, tokenizer)
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

    def forward(self, inputs):
        pass

    def on_train_epoch_start(self):
        if self.config.supervised_pretrain: return

        print('updating cluster centers....')
        train_dl = self.trainer.train_dataloader.loaders
        known_centers = torch.zeros(self.config.n_seen_classes, self.config.kmeans_dim, device=self.config.device)
        num_samples = [0] * self.config.n_seen_classes
        with torch.no_grad():
            uid2pl = {}  # pseudo labels
            unknown_uid_list = []
            unknown_vec_list = []
            seen_uid = set()  # oversampling for the unknown part, so we remove them here
            for batch in tqdm(iter(train_dl)):
                labels = batch[0]['labels']
                is_labeled_mask = batch[0]['is_labeled_mask']
                metadata = batch[0]['meta']
                batch_size = len(metadata)

                # move batch to gpu
                for key in ['token_ids', 'attn_mask', 'head_spans', 'tail_spans']:
                    batch[0][key] = batch[0][key].to(self.config.device)

                commonspace_rep = self.net.forward(batch[0], msg='similarity')  # (batch_size, hidden_dim)
                for i in range(batch_size):
                    if is_labeled_mask[i] == True:
                        l = labels[i]
                        known_centers[l] += commonspace_rep[i]
                        num_samples[l] += 1
                    else:
                        uid = metadata[i]['uid']
                        if uid not in seen_uid:
                            seen_uid.add(uid)
                            unknown_uid_list.append(uid)
                            unknown_vec_list.append(commonspace_rep[i].cpu().numpy())

            rep = np.stack(unknown_vec_list, axis=0)
            class_offset = self.config.n_seen_classes

            n_classes_unlabeled = get_n_classes_unlabeled(self.config)
            clf = KMeans(n_clusters=n_classes_unlabeled, random_state=0, algorithm='full')
            label_pred = clf.fit_predict(rep)  # from 0 to args.new_class - 1
            self.net.ct_loss_u.centers = torch.from_numpy(clf.cluster_centers_).to(
                self.config.device)  # (num_class, kmeans_dim)
            for i in range(len(unknown_vec_list)):
                uid = unknown_uid_list[i]
                pseudo = label_pred[i]
                uid2pl[uid] = pseudo + class_offset

            train_dl.dataset.update_pseudo_labels(uid2pl)
            print('updating pseudo labels...')
            pl_acc = train_dl.dataset.check_pl_acc()
            self.log('train/pl_acc', pl_acc, on_epoch=True)

            # update center for known types
            for c in range(self.config.n_seen_classes):
                known_centers[c] /= num_samples[c]
            self.net.ct_loss_l.centers = known_centers
        return

    def _compute_unknown_margin_loss(self, batch: Dict[str, torch.Tensor], pseudo_labels: torch.LongTensor,
                                     is_labeled_mask: torch.BoolTensor) -> torch.FloatTensor:
        # convert 1d pseudo label into 2d pairwise pseudo label
        assert (torch.min(pseudo_labels) >= self.config.n_seen_classes)

        pair_label = (pseudo_labels.unsqueeze(0) == pseudo_labels.unsqueeze(1)).float()
        logits = self.net.forward(batch, mask=~is_labeled_mask, msg='unlabeled')  # (batch_size, new_class)
        # this only predicts over new classes
        unknown_batch_size = pseudo_labels.size(0)
        expanded_logits = logits.expand(unknown_batch_size, -1, -1)
        expanded_logits2 = expanded_logits.transpose(0, 1)
        kl1 = compute_kld(expanded_logits.detach(), expanded_logits2)
        kl2 = compute_kld(expanded_logits2.detach(), expanded_logits)  # (batch_size, batch_size)
        assert kl1.requires_grad
        unknown_class_loss = torch.mean(pair_label * (kl1 + kl2) + (1 - pair_label) * (
                torch.relu(self.config.sigmoid - kl1) + torch.relu(self.config.sigmoid - kl2)))
        return unknown_class_loss

    def _compute_reconstruction_loss(self, batch: Dict[str, torch.Tensor], is_labeled_mask: torch.BoolTensor,
                                     labels: torch.LongTensor, unis_labeled_mask_conf_based=[]) -> torch.FloatTensor:
        # recon loss for known classes
        commonspace_rep_known, rec_loss_known = self.net.forward(batch, mask=is_labeled_mask,
                                                                 msg='reconstruct')  # (batch_size, kmeans_dim)
        # recon loss for unknown classes
        _, rec_loss_unknown = self.net.forward(batch, mask=~is_labeled_mask,
                                               msg='reconstruct')  # (batch_size, kmeans_dim)
        reconstruction_loss = (rec_loss_known.mean() + rec_loss_unknown.mean()) / 2
        # center loss for known classes
        center_loss = self.config.center_loss * self.net.ct_loss_l(labels[is_labeled_mask], commonspace_rep_known)
        l2_reg = 1e-5 * (L2Reg(self.net.similarity_encoder) + L2Reg(self.net.similarity_decoder))
        loss = reconstruction_loss + center_loss + l2_reg
        return loss

    def _compute_ce_loss(self, batch: Dict[str, torch.Tensor], is_labeled_mask: torch.BoolTensor,
                         labels: torch.LongTensor) -> torch.FloatTensor:
        '''
        Cross entropy loss for known classes.
        '''
        known_logits = self.net.forward(batch, mask=is_labeled_mask, msg='labeled')  # single layer labeled head
        _, label_pred = torch.max(known_logits, dim=-1)
        known_label = labels[is_labeled_mask]
        acc = 1.0 * torch.sum(label_pred == known_label) / len(label_pred)
        ce_loss = F.cross_entropy(input=known_logits, target=known_label)
        return ce_loss, acc

    def training_step(self, batch: List[Dict[str, torch.Tensor]], batch_idx: int):
        loss, ce_loss, acc, margin_loss = 0, 0, 0, 0
        labels = batch[0]['labels']
        is_labeled_mask = batch[0]['is_labeled_mask']
        psuedo_labels = batch[0]['pseudo_labels']
        if sum(is_labeled_mask).item() > 0:
            loss = self._compute_reconstruction_loss(batch[0], is_labeled_mask, labels)
            ce_loss, acc = self._compute_ce_loss(batch[0], is_labeled_mask, labels)

        if self.config.supervised_pretrain:
            loss += ce_loss

        elif self.current_epoch >= self.config.num_pretrain_epochs:
            if sum(~is_labeled_mask).item() > 0:
                margin_loss = self._compute_unknown_margin_loss(batch[0], psuedo_labels[~is_labeled_mask],
                                                                is_labeled_mask)
                loss += margin_loss
                loss += ce_loss

            self.log('train/unknown_margin_loss', margin_loss)
            self.log('train/known_ce_loss', ce_loss)

        self.log('train/known_acc', acc)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ce_loss, acc = 0, 0, 0
        labels = batch[0]['labels']
        is_labeled_mask = batch[0]['is_labeled_mask']
        if sum(is_labeled_mask).item() > 0:
            loss = self._compute_reconstruction_loss(batch[0], is_labeled_mask, labels)
            ce_loss, acc = self._compute_ce_loss(batch[0], is_labeled_mask, labels)
        self.log('val_acc', acc, on_step=True, on_epoch=True)
        self.log('val/known_ce_loss', ce_loss)
        self.log('val/loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch[0]['labels_unseen']
        if self.config.use_confidence:
            n_preds_qa = 0
            is_seen_conf_based_mask = torch.tensor([x['is_seen_conf_based'] for x in batch[0]['meta']]).to(
                self.config.device)
            is_novel_conf_based_mask = ~is_seen_conf_based_mask

            # Only predict known classes for known instances (conf based)
            if sum(is_seen_conf_based_mask).item() > 0:
                logits_seen = self.net.forward(batch[0], mask=is_seen_conf_based_mask, msg='labeled')
                prob, pred_cluster = torch.max(logits_seen, dim=1)
                self.test_preds = np.append(self.test_preds, pred_cluster.cpu().numpy())
                self.test_probs = np.append(self.test_probs, prob.cpu().numpy())
                n_preds_qa += len(pred_cluster)

            # Only predict novel classes for novel instances (conf based)
            if sum(is_novel_conf_based_mask).item() > 0:
                logits_novel = self.net.forward(batch[0], mask=is_novel_conf_based_mask, msg='unlabeled')
                prob, pred_cluster = torch.max(logits_novel, dim=1)
                pred_cluster += self.config.n_seen_classes
                self.test_preds = np.append(self.test_preds, pred_cluster.cpu().numpy())
                self.test_probs = np.append(self.test_probs, prob.cpu().numpy())
                n_preds_qa += len(pred_cluster)

            # QA Check
            assert len(labels) == n_preds_qa, f'len(targets) {len(labels)} != n_preds_qa {n_preds_qa}'

        # Original RoCORE
        else:
            if self.config.constrain_pred_type == 'seen_only':
                logits = self.net.forward(batch[0], msg='labeled')
            elif self.config.constrain_pred_type == 'novel_only':
                logits = self.net.forward(batch[0], msg='unlabeled')
            else:
                raise ValueError(f'Invalid constrain_pred_type {self.config.constrain_pred_type}')
            prob, pred_cluster = torch.max(logits, dim=1)
            self.test_preds = np.append(self.test_preds, pred_cluster.cpu().numpy())
            self.test_probs = np.append(self.test_probs, prob.cpu().numpy())
            self.test_uids = np.append(self.test_uids, [x['uid'] for x in batch[0]['meta']])

        # Collect Targets
        self.test_targets = np.append(self.test_targets, labels.cpu().numpy())
        return

    def test_epoch_end(self, outputs: List[List[Dict]]) -> None:
        self._type_adjust()

        # Compute metrics
        metrics = test_metrics(self.config, self.test_preds, self.test_targets)
        self.log_dict(metrics, sync_dist=True)

        print(metrics)
        self._clear_data()
        return

    def _save_predictions(self, split):
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

    def save_confidence_scores(self, train_dataloader, split):
        # Treat train data as validation data just to get confidence scores on train data
        self.net.to(self.config.device)
        self.net.eval()
        self._clear_data()
        with torch.no_grad():
            for batch in train_dataloader:
                self.test_step(batch, 0)

        # Save confidence scores
        self._type_adjust()
        self._save_predictions(split=split)
        self._clear_data()

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
        return [optimizer, ]
