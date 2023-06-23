import copy
import csv
import json
import random
from os.path import join
from typing import List, Dict

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from utils.utils_common import load_jsonl_data, split_train_val


class MPDatamodule(LightningDataModule):
    """Data module"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_labeled = []
        self.train_unlabeled = []
        self.val_labeled = []
        self.val_unlabeled = []
        self.test_unlabeled = []
        self.is_novel_conf_based = []
        self.tools = MPTools(self.config)

    def prepare_data(self) -> None:
        return super().prepare_data()

    def print_len_data(self):
        print('train_labeled', len(self.train_labeled))
        print('train_unlabeled', len(self.train_unlabeled))
        print('val_labeled', len(self.val_labeled))
        print('val_unlabeled', len(self.val_unlabeled))
        print('test_unlabeled', len(self.test_unlabeled))
        print('is_novel_conf_based', len(self.is_novel_conf_based))

    def load_data(self):
        labeled_exs = load_jsonl_data(join(self.config.data_dir, self.config.fname_labeled))
        unlabeled_exs = load_jsonl_data(join(self.config.data_dir, self.config.fname_unlabeled))
        return labeled_exs, unlabeled_exs

    def _split(self):
        confidence_dict = {}
        if self.config.use_confidence:
            with open(join('logs', self.config.checkpoint_dir, 'uid2is_seen_conf_based.json')) as f:
                confidence_dict = json.load(f)
                print(f'Loaded confidence dict with {len(confidence_dict)} entries.')
        labeled_exs = self.gather_examples(self.config.fname_labeled, confidence_dict, is_labeled=True)
        unlabeled_exs = self.gather_examples(self.config.fname_unlabeled, confidence_dict, is_labeled=False)
        assert len(labeled_exs) > 0, f'No labeled_exs found'
        assert len(unlabeled_exs) > 0, f'No unlabeled_exs found'
        return labeled_exs, unlabeled_exs

    def gather_examples(self, fname, confidence_dict, is_labeled=False):
        re_data = load_jsonl_data(join(self.config.data_dir, fname))
        assert type(re_data[0]) is dict
        n_conf_seen = 0
        n_conf_novel = 0
        for ex in re_data:
            ex['is_seen_conf_based'] = False
            if self.config.use_confidence and not is_labeled:
                is_seen_conf_based = True if confidence_dict[str(ex['uid'])] == 1 else False
                ex['is_seen_conf_based'] = is_seen_conf_based
                n_conf_seen += 1 if is_seen_conf_based else 0
                n_conf_novel += 1 if not is_seen_conf_based else 0

        if self.config.use_confidence and not is_labeled:
            print(f'Confidence based seen/unseen: {n_conf_seen}/{n_conf_novel}')
        return re_data

    @staticmethod
    def collate_batch_feat(batch: List[List[Dict]]) -> Dict:
        if isinstance(batch[0], tuple):
            # expand mixed batch
            new_batch = []
            for tup in batch:
                new_batch.append(tup[0])
                new_batch.append(tup[1])
            batch = new_batch

        assert isinstance(batch, list)
        return {
            'is_labeled': torch.BoolTensor([x['is_labeled'] for x in batch]),
            'is_seen_conf_based': torch.BoolTensor([x['is_seen_conf_based'] for x in batch]),
            'uid': torch.LongTensor([x['uid'] for x in batch]),
            'label': torch.LongTensor(np.asarray([x['label'] for x in batch])),
            'input_ids': torch.LongTensor(np.asarray([x['input_ids'] for x in batch])),
            'attention_mask': torch.LongTensor(np.asarray([x['attention_mask'] for x in batch])),
            'masked_pos': torch.LongTensor(np.asarray([x['masked_pos'] for x in batch])),
            'input_ids_m15p': torch.LongTensor(np.asarray([x['input_ids_m15p'] for x in batch])),
            'input_ids_rd': torch.LongTensor(np.asarray([x['input_ids_rd'] for x in batch])),
            'attention_mask_rd': torch.LongTensor(np.asarray([x['attention_mask_rd'] for x in batch])),
        }

    def setup(self, stage=None):
        # Load data
        labeled_exs, unlabeled_exs = self._split()
        # Create val splits
        labeled_exs_train, labeled_exs_val = split_train_val(self.config, labeled_exs)
        unlabeled_exs_train, unlabeled_exs_val = split_train_val(self.config, unlabeled_exs)

        # Assign to use in dataloaders
        self.train_labeled = labeled_exs_train
        self.train_unlabeled = unlabeled_exs_train
        self.val_labeled = labeled_exs_val
        self.val_unlabeled = labeled_exs_val
        self.test_unlabeled = unlabeled_exs

        if self.config.use_confidence:
            self.is_novel_conf_based = [ex for ex in unlabeled_exs if not ex['is_seen_conf_based']]
        self.print_len_data()

    def train_dataloader(self):
        if self.config.use_confidence:
            train_dataset = MixedBatchMultiviewDataset(self.config,
                                                       known_exs=self.train_labeled,
                                                       unknown_exs=self.is_novel_conf_based,
                                                       tools=self.tools,
                                                       )
        else:
            train_dataset = MixedBatchMultiviewDataset(self.config,
                                                       known_exs=self.train_labeled,
                                                       unknown_exs=self.train_unlabeled,
                                                       tools=self.tools,
                                                       )
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.config.batch_size // 2,
                                      shuffle=True, num_workers=self.config.num_workers,
                                      pin_memory=True,
                                      drop_last=True,
                                      collate_fn=self.collate_batch_feat)  # set to False for extracting features
        return train_dataloader

    def val_dataloader(self):
        val_dataset = MixedBatchMultiviewDataset(self.config,
                                                 known_exs=self.val_labeled,
                                                 unknown_exs=self.val_unlabeled,
                                                 tools=self.tools,
                                                 is_val_labeled=True,
                                                 )
        val_dataloader = DataLoader(val_dataset, batch_size=4,
                                    shuffle=False, num_workers=self.config.num_workers,
                                    drop_last=True,
                                    pin_memory=True, collate_fn=self.collate_batch_feat)
        return val_dataloader

    def test_dataloader(self):
        data_test_unlabeled = MPDataset(self.config, self.test_unlabeled, self.tools, is_labeled=False)
        test_dataloader = DataLoader(data_test_unlabeled, batch_size=self.config.batch_size,
                                     shuffle=False, num_workers=self.config.num_workers)
        return test_dataloader


class MixedBatchMultiviewDataset(Dataset):
    '''
    A dataset that produces a batch with mixed labeled and unlabeled instances.
    '''

    def __init__(self, config, known_exs, unknown_exs, tools, is_val_labeled=False) -> None:
        super().__init__()
        self.config = config
        self.known_exs = MPDataset(config, known_exs, tools, is_labeled=True)
        self.unknown_exs = MPDataset(config, unknown_exs, tools, is_labeled=is_val_labeled)

    def __len__(self):
        return max([len(self.known_exs), len(self.unknown_exs)])

    def __getitem__(self, index):
        labeled_index = int(index % len(self.known_exs))
        labeled_ins = self.known_exs[labeled_index]
        unlabeled_index = int(index % len(self.unknown_exs))
        unlabeled_ins = self.unknown_exs[unlabeled_index]
        return labeled_ins, unlabeled_ins


class MPTools:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.foundation_model)

        # Load confidence dict
        self.confidence_dict = {}
        if self.config.use_confidence:
            with open(join('logs', self.config.checkpoint_dir, 'uid2is_seen_conf_based.json')) as f:
                self.confidence_dict = json.load(f)
                print(f'Loaded confidence dict with {len(self.confidence_dict)} entries.')

        # Load relation descriptions
        self.rel_id2des = {}
        self.load_rel_descriptions(config.rel2id)

    def load_rel_descriptions(self, rel2id):
        fname = join('data', self.config.dataset, 'relation_description.csv')
        reader = csv.reader(open(fname), delimiter='\t')
        for line in reader:
            rel, description = line
            if line:
                if self.config.dataset == 'fewrel':
                    rel, rel_name = rel.split(':')
                rel_id = rel2id[rel]
                self.rel_id2des[rel_id] = description


class MPDataset(Dataset):
    def __init__(self, config, re_data, tools, is_labeled=False):
        self.config = config
        self.is_labeled_data = is_labeled
        self.re_data = re_data
        self.n_conf_seen = 0

        self.tokenizer = tools.tokenizer
        self.confidence_dict = tools.confidence_dict
        self.rel_id2des = tools.rel_id2des

        # Init and tokenize data
        n_instances = len(self.re_data)
        self.uids = np.zeros(n_instances, dtype=int)
        self.labels = np.zeros(n_instances, dtype=int)
        self.masked_pos = np.zeros(n_instances, dtype=int)
        self.attention_masks = np.zeros((n_instances, self.config.max_length), dtype=int)
        self.attention_masks_r_dec = np.zeros((n_instances, self.config.max_length), dtype=int)
        self.is_labeled = np.ones(n_instances, dtype=int) if is_labeled else np.zeros(n_instances, dtype=int)
        self.is_seen_conf_based = np.zeros(n_instances, dtype=int)

        # Input IDs with PAD token defaults
        if self.tokenizer.pad_token_id == 1:
            self.input_ids = np.ones((n_instances, self.config.max_length), dtype=int)  # 1 = pad
            self.input_ids_m15p = np.ones((n_instances, self.config.max_length), dtype=int)  # 1 = pad
            self.input_ids_r_dec = np.ones((n_instances, self.config.max_length), dtype=int)  # 1 = pad
        elif self.tokenizer.pad_token_id == 0:
            self.input_ids = np.zeros((n_instances, self.config.max_length), dtype=int)  # 0 = pad
            self.input_ids_m15p = np.zeros((n_instances, self.config.max_length), dtype=int)  # 0 = pad
            self.input_ids_r_dec = np.zeros((n_instances, self.config.max_length), dtype=int)  # 0 = pad
        else:
            raise ValueError("Pad token id not found in tokenizer")
        self.tokenize()

    def tokenize(self):
        # Tokenize relation descriptions
        trim = self.config.max_length - 2  # 2 for [CLS] and [SEP]
        for rel_id, des in self.rel_id2des.items():
            tokenized_desc = self.tokenizer.tokenize(' '.join(des))
            tokenized_desc_ids = self.tokenizer.convert_tokens_to_ids(tokenized_desc)
            tokenized_ids = [self.tokenizer.cls_token_id] + tokenized_desc_ids[:trim] + \
                            [self.tokenizer.sep_token_id]

            self.input_ids_r_dec[rel_id][0:len(tokenized_ids)] = tokenized_ids
            self.attention_masks_r_dec[rel_id][0:len(tokenized_ids)] = 1

        # Tokenize data
        for i, ins in enumerate(self.re_data):
            tokens = ins['token']
            h_start, h_end = ins['h']['pos']
            t_start, t_end = ins['t']['pos']
            self.labels[i] = self.config.rel2id[ins['relation']]

            # UIDs
            self.uids[i] = ins['uid']

            # Confidence based seen/unseen
            if self.config.use_confidence and not self.is_labeled_data:
                self.is_seen_conf_based[i] = int(self.confidence_dict[str(ins['uid'])])
                self.n_conf_seen += int(self.confidence_dict[str(ins['uid'])])

            # Tokenize
            tokenized_sent = self.tokenizer.tokenize(' '.join(tokens))
            head = ' '.join(tokens[h_start:h_end])
            tokenized_head = self.tokenizer.tokenize(head)
            tail = ' '.join(tokens[t_start:t_end])
            tokenized_tail = self.tokenizer.tokenize(tail)
            trim = self.config.max_length - (
                    len(tokenized_head) + len(tokenized_tail) + 4)  # 4 = cls + sep + attention_masks + sep
            tokenized_sent = tokenized_sent[:trim]

            # Gen token ids
            tokenized_sent_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
            tokenized_head_ids = self.tokenizer.convert_tokens_to_ids(tokenized_head)
            tokenized_tail_ids = self.tokenizer.convert_tokens_to_ids(tokenized_tail)

            # Template: [CLS] [SENT] [SEP] [HEAD] [MASK] [TAIL] [SEP]
            tokenized_ids = [self.tokenizer.cls_token_id] + tokenized_sent_ids + \
                            [self.tokenizer.sep_token_id] + tokenized_head_ids + \
                            [self.tokenizer.mask_token_id] + tokenized_tail_ids + \
                            [self.tokenizer.sep_token_id]

            assert len(tokenized_tail_ids) == len(tokenized_tail)

            self.input_ids[i][0:len(tokenized_ids)] = tokenized_ids
            self.masked_pos[i] = len(tokenized_ids) - len(tokenized_tail_ids) - 2  # 2 = [SEP] + [MASK]
            assert self.input_ids[i][self.masked_pos[i]] == self.tokenizer.mask_token_id

            # 15p masked inputs
            tokenized_sent_ids_15p_masked = self.mask_15p_ids(tokenized_sent_ids, tokenized_head_ids,
                                                              tokenized_tail_ids)
            tokenized_ids_15p_masked = [self.tokenizer.cls_token_id] + tokenized_sent_ids_15p_masked + \
                                       [self.tokenizer.sep_token_id] + tokenized_head_ids + \
                                       [self.tokenizer.mask_token_id] + tokenized_tail_ids + \
                                       [self.tokenizer.sep_token_id]
            self.input_ids_m15p[i][0:len(tokenized_ids_15p_masked)] = tokenized_ids_15p_masked
            # Quality check
            idx_masked_pos_m15p = len(tokenized_ids_15p_masked) - len(tokenized_tail_ids) - 2  # 2 = [SEP] + [MASK]
            assert self.input_ids_m15p[i][idx_masked_pos_m15p] == self.tokenizer.mask_token_id

            # Attention mask tokens
            length = min(len(tokenized_ids), self.config.max_length)
            self.attention_masks[i][0:length] = 1
            assert len(tokenized_ids_15p_masked) == len(tokenized_ids)

        print("Number of confidence seen instances: {}".format(self.n_conf_seen))

    def enumerate_spans(self, span):
        return [x for x in range(span[0], span[1] + 1)]

    def mask_15p_ids(self, sent_ids, entity1_ids, entity2_ids):
        sent_ids_15p_masked = copy.deepcopy(sent_ids)
        ids_to_exclude = entity1_ids + entity2_ids

        # Identify the number of tokens to mask
        indices_to_mask = [i for i in range(len(sent_ids)) if sent_ids[i] not in ids_to_exclude]
        num_to_mask = int(len(indices_to_mask) * 0.15)

        # Randomly select the tokens to mask
        masked_indices = random.sample(indices_to_mask, num_to_mask)

        # Mask the selected tokens with the mask token
        for index in masked_indices:
            sent_ids_15p_masked[index] = self.tokenizer.mask_token_id

        return sent_ids_15p_masked

    def __len__(self):
        return len(self.re_data)

    def __getitem__(self, index):
        label = self.labels[index]
        return {
            # Src / open class
            'is_labeled': self.is_labeled[index],
            'is_seen_conf_based': self.is_seen_conf_based[index],
            'uid': self.uids[index],
            # Regular sentence
            'label': label,
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_masks[index],
            'masked_pos': self.masked_pos[index],
            # 15% masked
            'input_ids_m15p': self.input_ids_m15p[index],
            # Relation description
            'input_ids_rd': self.input_ids_r_dec[label],
            'attention_mask_rd': self.attention_masks_r_dec[label],
        }


def batch_var_length(tensors: List[torch.Tensor], max_length: int = 300):
    batch_size = len(tensors)
    pad_len = min(max_length, max([t.size(0) for t in tensors]))
    batch_tensors = torch.zeros((batch_size, pad_len)).type_as(tensors[0])
    for i in range(batch_size):
        actual_len = min(pad_len, tensors[i].size(0))
        batch_tensors[i, :actual_len] = tensors[i][:actual_len]

    return batch_tensors
