import re

import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class REDataset(Dataset):
    def __init__(self, config, re_data, em, is_labeled=False, is_weak=False):
        self.config = config
        n_instances = len(re_data)
        self.mask = np.zeros((n_instances, config.max_length), dtype=int)
        self.h_pos = np.zeros(n_instances, dtype=int)
        self.t_pos = np.zeros(n_instances, dtype=int)
        self.h_pos_l = np.zeros(n_instances, dtype=int)
        self.t_pos_l = np.zeros(n_instances, dtype=int)
        self.label = np.zeros(n_instances, dtype=int)
        self.weak_labels = np.zeros(n_instances, dtype=int)
        self.uids = np.zeros(n_instances, dtype=int)

        # Is labeled/unlabeled
        self.is_labeled = np.ones(n_instances, dtype=int) if is_labeled else np.zeros(n_instances, dtype=int)

        # Input IDs with appropriate padding id (0, 1), depending on the foundation model used
        if em.tokenizer.pad_token_id == 1:
            self.input_ids = np.ones((n_instances, config.max_length), dtype=int)  # pad = 1
        elif em.tokenizer.pad_token_id == 0:
            self.input_ids = np.zeros((n_instances, config.max_length), dtype=int)  # pad = 0
        else:
            raise ValueError('pad_token_id is neither 0 nor 1')

        # Collect data
        for i, ins in enumerate(re_data):
            if is_weak:  # custom data format
                self.label[i] = self.config.rel2id[ins["relation"]]
                self.weak_labels[i] = ins['cluster_pred_m']
                token = re_data[i]["raw_data"]["token"]
                h_pos = re_data[i]["raw_data"]['h']['pos']
                t_pos = re_data[i]["raw_data"]['t']['pos']
                h_type = re_data[i]["raw_data"]['h']['type'].lower()
                t_type = re_data[i]["raw_data"]['t']['type'].lower()

            else:
                self.weak_labels[i] = self.config.rel2id[ins["relation"]]
                self.label[i] = self.config.rel2id[ins["relation"]]
                token = re_data[i]["token"]
                h_pos = re_data[i]['h']['pos']
                t_pos = re_data[i]['t']['pos']
                h_type = re_data[i]['h']['type'].lower()
                t_type = re_data[i]['t']['type'].lower()

            # UIDs
            self.uids[i] = re_data[i]['uid']

            # Insert entity types
            token.insert(0, em.tokenizer.eos_token)
            token.insert(0, t_type)
            token.insert(0, em.tokenizer.sep_token)
            token.insert(0, em.tokenizer.eos_token)
            token.insert(0, h_type)
            token.insert(0, em.tokenizer.sep_token)

            h_pos[0] += 6  # 6 is the number of inserted tokens
            h_pos[1] += 6
            t_pos[0] += 6
            t_pos[1] += 6

            # Tokenize
            ids, ph, pt, ph_l, pt_l = em.tokenize(token, h_pos, t_pos)
            length = min(len(ids), config.max_length)
            self.input_ids[i][0:length] = ids[0:length]
            self.mask[i][0:length] = 1
            self.h_pos[i] = min(ph, config.max_length - 1)
            self.t_pos[i] = min(pt, config.max_length - 1)
            self.h_pos_l[i] = min(ph_l, config.max_length)
            self.t_pos_l[i] = min(pt_l, config.max_length)
        if em.err > 0:
            print(f"Ratio tokenizer can't find head/tail entity is {em.err}/{em.n_total}")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            'input_ids': self.input_ids[index],
            'attention_mask': self.mask[index],
            'h_pos': self.h_pos[index],
            't_pos': self.t_pos[index],
            'h_pos_l': self.h_pos_l[index],
            't_pos_l': self.t_pos_l[index],
            'labels': self.label[index],
            'weak_labels': self.weak_labels[index],
            'is_labeled': self.is_labeled[index],
        }


class EntityMarker:
    """Converts raw text to BERT-style input ids and finds entity position."""

    def __init__(self, config=None):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.foundation_model)
        self.h_pattern = re.compile("\* h \*")
        self.t_pattern = re.compile("\^ t \^")
        self.err = 0
        self.n_total = 0.0001

    def tokenize(self, raw_text, h_pos_li, t_pos_li):
        tokens = []
        for i, token in enumerate(raw_text):
            if h_pos_li[0] <= i < h_pos_li[-1]:
                if i == h_pos_li[0]:
                    tokens += [self.tokenizer.sep_token] + raw_text[h_pos_li[0]: h_pos_li[-1]] + [
                        self.tokenizer.eos_token]
                continue
            if t_pos_li[0] <= i < t_pos_li[-1]:
                if i == t_pos_li[0]:
                    tokens += [self.tokenizer.sep_token] + raw_text[t_pos_li[0]: t_pos_li[-1]] + [
                        self.tokenizer.eos_token]
                continue
            tokens.append(token)

        tokenized_text = [self.tokenizer.cls_token] + self.tokenizer.tokenize(' '.join(tokens)) + [
            self.tokenizer.sep_token]

        i = [i for i, n in enumerate(tokenized_text) if n == '[SEP]'][3] + 1  # offset for added entity type tokens

        pos, pos_types = [], []
        unused_start, unused_end, tail_before_head = self.set_special_tokens(h_pos_li, t_pos_li)
        n_found_sep_token = 0

        while i <= len(tokenized_text) - 2:
            if tokenized_text[i] == self.tokenizer.sep_token:
                n_found_sep_token += 1
                tokenized_text[i] = f'[unsused{unused_start}]'
                xx = i + 1
                while tokenized_text[i] != self.tokenizer.eos_token:
                    i += 1
                tokenized_text[i] = f'[unsused{unused_end}]'
                yy = i
                i += 1

                # First two ents with <s>...</s> tokens are ent types
                pos.append((xx, yy))

                # Swap unused tokens to handle next head or tail ent
                if tail_before_head:
                    unused_start = '301'
                    unused_end = '302'
                else:
                    unused_start = '303'
                    unused_end = '304'

            else:
                i += 1

        assert len(
            pos) == 2, f'Must have 2 positions. Found: {len(pos)}. Tokens: {tokens}, n_fnd_sep_token: {n_found_sep_token}'

        if h_pos_li[0] < t_pos_li[0]:
            h_pos = pos[0][0]
            h_pos_l = pos[0][1]
            t_pos = pos[1][0]
            t_pos_l = pos[1][1]
        else:
            t_pos = pos[0][0]
            t_pos_l = pos[0][1]
            h_pos = pos[1][0]
            h_pos_l = pos[1][1]

        tokenized_input = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return tokenized_input, h_pos, t_pos, h_pos_l, t_pos_l

    @staticmethod
    def set_special_tokens(h_pos_li, t_pos_li):
        unused_start = '301'
        unused_end = '302'
        tail_before_head = False
        if h_pos_li[0] > t_pos_li[0]:
            tail_before_head = True
            unused_start = '303'
            unused_end = '304'

        return unused_start, unused_end, tail_before_head
