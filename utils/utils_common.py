import json
from os.path import join
from typing import List

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as sk_metrics
from torch.utils import data

from utils.utils_metrics import p_r_f1_f1cls


def print_warning(msg):
    print("\033[93m --> {} <--\033[00m ".format(msg))


def test_metrics(config, preds, targets):
    novel_and_no_rel_ids = [x for x in config.novel_class_ids] + [0]
    seen_mask = np.isin(targets, config.seen_class_ids)  # ground truth seen mask
    novel_mask = np.isin(targets, novel_and_no_rel_ids)

    # Constrainted metrics
    if config.use_hungarian:
        preds_mapped = remap_preds(preds, targets)
    else:
        preds_mapped = preds
    f1_all, f1_seen, f1_novel, unseen_nmi = calc_f1(config, preds_mapped, targets, seen_mask, novel_mask)

    # Log metrics
    return {
        "f1_all": f1_all,
        "f1_seen": f1_seen,
        "f1_novel": f1_novel,
        "unseen_nmi": unseen_nmi,
    }


def calc_f1(config, preds, targets, seen_mask, novel_mask):
    _, _, f1_all, _ = p_r_f1_f1cls(preds, targets, config.all_class_ids)
    _, _, f1_seen, _ = p_r_f1_f1cls(preds[seen_mask], targets[seen_mask],
                                    config.seen_class_ids)
    _, _, f1_novel, _ = p_r_f1_f1cls(preds[novel_mask], targets[novel_mask],
                                     config.novel_class_ids)
    unseen_nmi = sk_metrics.normalized_mutual_info_score(targets[novel_mask], preds[novel_mask])
    return f1_all, f1_seen, f1_novel, unseen_nmi


def hungarian_algorithm(config, y_pred, y_true, seen_mask, novel_mask):
    """
    Hungarian algorithm for matching predicted labels to ground truth labels
    """
    preds_mapped = y_pred.copy()
    if config.constrain_pred_type == 'novel_only':
        preds_mapped[novel_mask] = remap_preds(y_pred[novel_mask], y_true[novel_mask])
    elif config.constrain_pred_type == 'seen_only':
        preds_mapped[seen_mask] = remap_preds(y_pred[seen_mask], y_true[seen_mask])
    elif config.constrain_pred_type == 'unconstrained' or config.use_confidence:
        preds_mapped = remap_preds(y_pred, y_true)
    return preds_mapped


def remap_preds(y_pred, y_true):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    y_pred_remapped = np.zeros(y_pred.size, dtype=np.int64)
    for i in range(y_pred.size):
        mapped_pred = col_ind[y_pred[i]]
        y_pred_remapped[i] = mapped_pred
    return y_pred_remapped


def load_jsonl_data(f_path):
    re_data = []
    print(f"Preprocessing data from: {f_path} ")
    with open(f_path) as f:
        all_lines = f.readlines()
        for line in all_lines:
            ins = json.loads(line)
            re_data.append(ins)
    assert type(re_data[0]) is dict
    return re_data


def split_train_val(config, labeled_exs):
    train_set_size = int(len(labeled_exs) * 0.8)
    valid_set_size = len(labeled_exs) - train_set_size
    if config.is_debug:
        trim = min(valid_set_size, 500)
        train_set_size, valid_set_size = trim, trim
        labeled_exs = labeled_exs[:train_set_size + valid_set_size]

    seed = torch.Generator().manual_seed(config.seed)
    train, val = data.random_split(labeled_exs, [train_set_size, valid_set_size], generator=seed)
    return train, val


def define_callbacks(config, monitor='val_acc', mode='max'):
    if config.save_best_model:
        callback = ModelCheckpoint(
            dirpath=config.output_dir,
            save_top_k=1,
            save_last=False,
            monitor=monitor,
            mode=mode,
            save_weights_only=True,
            filename='pretrained' if config.supervised_pretrain else 'best'
        )
        return [callback]
    else:
        return []


def define_trainer(config, logger, callbacks):
    return Trainer(
        accelerator='auto',
        max_epochs=config.epochs,
        fast_dev_run=config.is_debug,
        default_root_dir=config.output_dir,
        num_sanity_val_steps=1,
        gradient_clip_val=1.0,
        log_every_n_steps=1 if config.is_debug else 20,
        accumulate_grad_batches=1,
        logger=logger,
        strategy='ddp',
        callbacks=callbacks,
        precision=16,
        # profiler='simple',
    )


def print_final_results(config, trainer):
    print('Final performance:')
    f1_all = trainer.callback_metrics["f1_all"].item()
    f1_seen = trainer.callback_metrics["f1_seen"].item()
    f1_novel = trainer.callback_metrics["f1_novel"].item()
    print(f'{f1_all:.3f},{f1_seen:.3f},{f1_novel:.3f}')
    print(f'Outputs saved to: {config.output_dir}.')


def save_confidence(config, model, dm):
    model.load_pretrained_model(load_state_dict(config, load_pretrained=True))
    model.to(config.device)
    model.eval()
    model.save_confidence_scores(dm.test_dataloader(), split='test')
    model.save_confidence_scores(dm.train_dataloader(), split='train')


def load_state_dict(config, load_pretrained=False):
    fname = 'pretrained.ckpt' if load_pretrained else 'best.ckpt'
    if config.checkpoint_dir == 'None' or config.checkpoint_dir is None:
        checkpoint_dir = config.output_dir
    else:
        checkpoint_dir = join('logs', config.checkpoint_dir)
    print(f'Loading pretrained model ({fname}) from {checkpoint_dir}')
    checkpoint = torch.load(join(checkpoint_dir, fname), map_location=config.device)
    return checkpoint['state_dict']


def get_n_classes_unlabeled(config):
    '''Get the number of classes to predict for the unlabeled set'''
    if config.constrain_pred_type == 'novel_only' or config.use_confidence:
        return config.n_novel_classes
    elif config.constrain_pred_type == 'seen_only':
        return config.n_seen_classes
    elif config.constrain_pred_type == 'unconstrained':
        return config.n_seen_classes + config.n_novel_classes
    else:
        raise ValueError('Unknown constrain_pred_type: {}'.format(config.constrain_pred_type))


def batch_var_length(tensors: List[torch.Tensor], max_length: int = 300):
    batch_size = len(tensors)
    pad_len = min(max_length, max([t.size(0) for t in tensors]))
    batch_tensors = torch.zeros((batch_size, pad_len)).type_as(tensors[0])
    for i in range(batch_size):
        actual_len = min(pad_len, tensors[i].size(0))
        batch_tensors[i, :actual_len] = tensors[i][:actual_len]

    return batch_tensors
