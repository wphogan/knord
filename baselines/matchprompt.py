from os.path import join
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import dump, load
from pytorch_lightning import LightningModule
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, logging

from utils.utils_common import test_metrics
from utils.utils_metrics import kl_div_loss_custom

logging.set_verbosity_error()


class MatchPromptModel(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.foundation_model)
        self.model = MixedLMModel(config)
        self.ce_loss = torch.nn.CrossEntropyLoss()

        self.test_preds = np.array([])
        self.test_uids = np.array([])
        self.test_targets = np.array([])
        self.test_probs = np.array([])

        self.test_reps_novel = []
        self.test_preds_novel = np.array([])
        self.test_uids_novel = np.array([])
        self.test_targets_novel = np.array([])

        self.test_preds_seen = np.array([])
        self.test_uids_seen = np.array([])
        self.test_targets_seen = np.array([])
        self.test_probs_seen = np.array([])

        self.n_clusters = config.n_seen_classes + config.n_novel_classes
        self.km_model = MiniBatchKMeans(init="k-means++",
                                        n_clusters=self.n_clusters,
                                        random_state=0,
                                        batch_size=config.batch_size // 2,
                                        n_init=10,
                                        max_no_improvement=3,
                                        verbose=0
                                        )

    def load_pretrained_model(self, pretrained_dict):
        '''
        load model and common space proj,
        '''
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           ('pretrained_model' in k)}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        return

    def save_kmeans_model(self):
        fname = join(self.config.output_dir, 'best_kmeans_model.joblib')
        dump(self.km_model, fname)
        print("Kmeans model saved")

    def load_kmeans_model(self):
        try:
            checkpoint_dir = self.config.checkpoint_dir
            if self.config.checkpoint_dir == 'None' or self.config.checkpoint_dir is None:
                checkpoint_dir = self.config.output_dir
            fname = join(checkpoint_dir, 'best_kmeans_model.joblib')
            self.km_model = load(fname)
            print("Kmeans model loaded")
        except FileNotFoundError:
            print("No kmeans model found, using random initialization")

    def on_save_checkpoint(self, checkpoint) -> None:
        self.save_kmeans_model()

    def on_load_checkpoint(self, checkpoint) -> None:
        self.load_kmeans_model()

    def get_rel_rep(self, batch, mask=None):
        # Relation template masking
        input_ids = batch['input_ids']
        input_mask = batch['attention_mask']
        masked_pos = batch['masked_pos']
        if mask is not None:
            input_ids = input_ids[mask]
            input_mask = input_mask[mask]
            masked_pos = masked_pos[mask]
        batch_size = input_ids.shape[0]
        _, hidden_states = self.model.get_logits_and_feats(input_ids, input_mask)
        last_hidden_state = hidden_states[-1]
        r_rep = last_hidden_state[torch.arange(batch_size), masked_pos]
        return r_rep

    def loss_s(self, batch, is_labeled_mask):
        # Get relation representation from masked template: [SENT] [H] [MASK] [T]
        batch_size = batch['input_ids'][is_labeled_mask].shape[0]
        labels = batch['label'][is_labeled_mask]
        r_rep = self.get_rel_rep(batch, mask=is_labeled_mask)

        # Pre-training: align CLS token from context with masked LM
        _, hidden_states = self.model.get_logits_and_feats(batch['input_ids_rd'][is_labeled_mask],
                                                           batch['attention_mask'][is_labeled_mask])
        cls_rd_rep = hidden_states[-1][:, 0, :]  # (12th layer)(CLS token)

        # Loss S: Run KL divergence loss between context hidden state and masked rel hidden state
        lossL_s = 0
        zeros = torch.tensor(0.).to(self.config.device)
        for i, relation_embedding_i in enumerate(r_rep):
            max_val = torch.tensor(0.).to(self.config.device)
            for j, relation_description_j in enumerate(cls_rd_rep):
                if i == j:
                    pos = kl_div_loss_custom(v_s=relation_embedding_i, v_d=relation_description_j).to(
                        self.config.device)
                else:
                    if labels[i] != labels[j]:
                        tmp = kl_div_loss_custom(v_s=relation_embedding_i, v_d=relation_description_j).to(
                            self.config.device)
                        if tmp > max_val:
                            max_val = tmp
                        else:
                            continue
            neg = max_val
            instance_loss = torch.max(zeros, pos - neg + self.config.delta_1).to(self.config.device)
            lossL_s += instance_loss

        return lossL_s / batch_size

    def loss_scr(self, batch, r_rep, is_unlabeled_mask):
        batch_size = batch['input_ids'][is_unlabeled_mask].shape[0]
        logits, hidden_states = self.model.get_logits_and_feats(batch['input_ids_m15p'][is_unlabeled_mask],
                                                                batch['attention_mask'][is_unlabeled_mask])
        r_rep_m15p = hidden_states[-1][torch.arange(batch_size), batch['masked_pos'][is_unlabeled_mask]]
        return kl_div_loss_custom(r_rep, r_rep_m15p) / batch_size, logits, hidden_states[-1], r_rep_m15p

    def loss_o(self, batch, r_rep, r_rep_m15p, is_unlabeled_mask):
        lossL_o = 0
        batch_size = batch['input_ids'][is_unlabeled_mask].shape[0]
        zeros = torch.tensor(0.).to(self.config.device)
        if len(r_rep) < self.n_clusters:
            return lossL_o

        kmeans = self.km_model.fit(r_rep.detach().cpu().numpy())
        kmeans_tilda = self.km_model.fit(r_rep_m15p.detach().cpu().numpy())
        for i, (p_i, p_i_tilda) in enumerate(zip(kmeans.labels_, kmeans_tilda.labels_)):
            for j, (p_j, p_j_tilda) in enumerate(zip(kmeans.labels_, kmeans_tilda.labels_)):
                if i == j:
                    continue
                # b_tilda = 1
                if p_i == p_j and p_i_tilda == p_j_tilda:
                    lossL_o += kl_div_loss_custom(r_rep[i], r_rep_m15p[j])
                # b_tilda = 0
                elif p_i != p_j and p_i_tilda != p_j_tilda:
                    lossL_o += torch.max(zeros, self.config.delta_2 - kl_div_loss_custom(r_rep[i],
                                                                                         r_rep_m15p[j]))
        return lossL_o / batch_size

    def loss_mlm(self, batch, logits_m15p, is_unlabeled_mask):
        input_ids_trim = batch['input_ids_m15p'][is_unlabeled_mask].clone().to(self.config.device)
        input_ids_trim[torch.arange(logits_m15p.shape[0]), batch[
            'masked_pos'][is_unlabeled_mask]] = self.tokenizer.pad_token_id  # exclude mask token from template
        insent_mask_token_indices = (input_ids_trim == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
        logits_mlm_insent = logits_m15p[
            insent_mask_token_indices[:, 0], insent_mask_token_indices[:, 1]]  # Logits from insent_mask_tokens
        mlm_insent = batch['input_ids'][is_unlabeled_mask]
        targets_mlm_insent = mlm_insent[insent_mask_token_indices[:, 0], insent_mask_token_indices[:, 1]]
        return self.ce_loss(logits_mlm_insent, targets_mlm_insent) / targets_mlm_insent.shape[0]

    def forward(self, batch):
        lossL_s, lossL_ce, lossL_scr, lossL_o, lossL_mlm = 0, 0, 0, 0, 0
        is_labeled_mask = batch['is_labeled'].to(self.config.device)

        # Pre-training: align CLS token from context with masked LM
        if sum(is_labeled_mask).item() > 0:
            lossL_s = self.loss_s(batch, is_labeled_mask)
            lossL_ce, _, _ = self._compute_ce_loss(batch, is_labeled_mask)

        # Pre-training loss: only run for first few epochs
        is_pretrain = (self.current_epoch < self.config.pretrain_epochs)
        if is_pretrain:
            lossL = lossL_s + lossL_ce
            return lossL

        # Get open class relation representations (last hidden state of masked token)
        if sum(~is_labeled_mask).item() > 0:
            r_rep = self.get_rel_rep(batch, mask=~is_labeled_mask)

            # Loss SCR: in-sentence masks
            lossL_scr, logits_m15p, lhs_m15p, r_rep_m15p = self.loss_scr(batch, r_rep, ~is_labeled_mask)

            # Loss O: Kmeans clustering + KL divergence
            lossL_o = self.loss_o(batch, r_rep, r_rep_m15p, ~is_labeled_mask)

            # Loss MLM: Masked LM
            lossL_mlm = self.loss_mlm(batch, logits_m15p, ~is_labeled_mask)

        # Total loss
        lossL = lossL_ce + lossL_o + (lossL_s * self.config.coef_ls) + (self.config.coef_lsc * (lossL_scr + lossL_mlm))
        return lossL

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        # Eval only on true novel instances
        if self.config.use_confidence:
            n_preds_qa = 0
            is_seen_conf_based_mask = torch.tensor(batch['is_seen_conf_based'].to(self.config.device)) > 0
            is_novel_conf_based_mask = ~is_seen_conf_based_mask
            # Only predict known classes for known instances (conf based)
            if sum(is_seen_conf_based_mask) > 0:
                _, probs, preds = self._compute_ce_loss(batch, is_seen_conf_based_mask, pred_only=True)
                self.test_preds_seen = np.append(self.test_preds_seen, preds.detach().cpu().numpy())
                self.test_probs_seen = np.append(self.test_probs_seen, probs.detach().cpu().numpy())
                self.test_uids_seen = np.append(self.test_uids_seen,
                                                batch['uid'][is_seen_conf_based_mask].detach().cpu().numpy())
                self.test_targets_seen = np.append(self.test_targets_seen, np.array(
                    batch['label'][is_seen_conf_based_mask].detach().cpu().numpy()))
                n_preds_qa += len(preds)

            # Collect representations for novel instances (conf based)
            if sum(is_novel_conf_based_mask) > 0:
                r_rep = self.get_rel_rep(batch, mask=is_novel_conf_based_mask)
                self.test_reps_novel.append(r_rep.detach().cpu().numpy())
                self.test_uids_novel = np.append(self.test_uids_novel,
                                                 batch['uid'][is_novel_conf_based_mask].detach().cpu().numpy())
                self.test_targets_novel = np.append(self.test_targets_novel, np.array(
                    batch['label'][is_novel_conf_based_mask].detach().cpu().numpy()))
                n_preds_qa += len(r_rep)

            # QA Check
            assert len(batch['label']) == n_preds_qa, f'QA Check Failed: {len(batch["label"])} != {n_preds_qa}'

        elif self.config.constrain_pred_type == 'novel_only':
            r_rep = self.get_rel_rep(batch)
            self.test_reps_novel.append(r_rep.detach().cpu().numpy())
            self.test_uids_novel = np.append(self.test_uids_novel, batch['uid'].detach().cpu().numpy())
            self.test_targets_novel = np.append(self.test_targets_novel,
                                                np.array(batch['label'].detach().cpu().numpy()))

        elif self.config.constrain_pred_type == 'seen_only':
            _, probs, preds = self._compute_ce_loss(batch, pred_only=True)
            self.test_preds_seen = np.append(self.test_preds_seen, preds.detach().cpu().numpy())
            self.test_probs_seen = np.append(self.test_probs_seen, probs.detach().cpu().numpy())
            self.test_uids_seen = np.append(self.test_uids_seen, batch['uid'].detach().cpu().numpy())
            self.test_targets_seen = np.append(self.test_targets_seen, np.array(batch['label'].detach().cpu().numpy()))

    def test_epoch_end(self, outputs: List[Dict]) -> None:
        # Fit KMeans model on all test data
        if len(self.test_reps_novel) > 0:
            all_reps = np.concatenate(self.test_reps_novel, axis=0)
            self.km_model.fit(all_reps)
            km_preds = self.km_model.labels_ + self.config.n_seen_classes
            self.test_preds_novel = np.array(km_preds)

        # Combine seen and novel predictions
        self.test_preds = np.append(self.test_preds_seen, self.test_preds_novel)
        self.test_uids = np.append(self.test_uids_seen, self.test_uids_novel)
        self.test_targets = np.append(self.test_targets_seen, self.test_targets_novel)
        self.test_probs = np.append(self.test_probs_seen, np.zeros(len(self.test_preds_novel)))

        # Compute metrics
        self._type_adjust()
        metrics = test_metrics(self.config, self.test_preds, self.test_targets)

        # Log metrics
        self.log_dict(metrics, sync_dist=True)

        # Save predictions on the last epoch
        if self.current_epoch == self.config.epochs - 1 and self.config.supervised_pretrain:
            self._save_predictions()

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
        self.test_uids = np.array([])
        self.test_targets = np.array([])
        self.test_probs = np.array([])

        self.test_reps_novel = []
        self.test_preds_novel = np.array([])
        self.test_uids_novel = np.array([])
        self.test_targets_novel = np.array([])

        self.test_preds_seen = np.array([])
        self.test_uids_seen = np.array([])
        self.test_targets_seen = np.array([])
        self.test_probs_seen = np.array([])

    def _type_adjust(self):
        self.test_targets = self.test_targets.astype(int)
        self.test_preds = self.test_preds.astype(int)
        self.test_uids = self.test_uids.astype(int)
        self.test_probs = self.test_probs.astype(float)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-5)

    def get_masked_prediction(self, inputs, logits):
        # Get predicted token
        mask_token_index = (inputs['input_ids'] == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        masked_word_prediction = self.tokenizer.decode(predicted_token_id)

        # Replace the masked token with predicted token and return text
        predicted_text = self.tokenizer.decode(inputs['input_ids'][0].tolist())
        predicted_text = predicted_text.replace(self.tokenizer.mask_token, masked_word_prediction)
        print(predicted_text)

    def _compute_ce_loss(self, batch: Dict[str, torch.Tensor], is_labeled_mask=None,
                         pred_only=False) -> torch.FloatTensor:
        '''
        Cross entropy loss for known classes.
        '''
        ce_loss = 0
        r_rep = self.get_rel_rep(batch, mask=is_labeled_mask)
        logits = self.model.labeled_head(r_rep)
        prob, pred = torch.max(logits, dim=-1)
        if pred_only:
            return ce_loss, prob, pred
        else:
            ce_loss = F.cross_entropy(input=logits, target=batch['label'][is_labeled_mask])
            return ce_loss, prob, pred


class MixedLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = AutoConfig.from_pretrained(config.foundation_model, output_hidden_states=True)
        self.pretrained_model = AutoModelForMaskedLM.from_pretrained(config.foundation_model, config=self.model_config)
        self.pretrained_model.to(config.device)
        self.initial_dim = self.model_config.hidden_size
        if self.config.constrain_pred_type == 'seen_only':
            n_unlabeled_classes = config.n_seen_classes
        else:
            n_unlabeled_classes = config.n_seen_classes + config.n_novel_classes
        self.labeled_head = nn.Linear(self.initial_dim, n_unlabeled_classes)

    def get_logits_and_feats(self, input_ids, attention_mask):
        inputs = {
            'input_ids': input_ids.to(self.config.device),
            'attention_mask': attention_mask.to(self.config.device),
        }
        outputs = self.pretrained_model(**inputs, output_hidden_states=True)
        return outputs.logits, outputs.hidden_states
