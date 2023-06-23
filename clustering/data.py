import json
import os

import torch
from torch.utils import data


class REDataset(data.Dataset):

    def __init__(self, data, tokenizer, args, is_eval=False):

        self.args = args
        self.tokenizer = tokenizer
        self.is_eval = is_eval
        self.data = data

        self.data = self.preprocess(self.data)

    def process_relation_name(self, relation):

        if relation == 'no_relation':
            relation = relation.replace('_', ' ')
        elif ':' in relation:
            relation = relation.split(':')[1].replace('_', ' ')
            relation = relation.replace('stateorprovince', 'state or province')
        else:
            pass
        return relation

    def preprocess(self, data):

        processed_data = []

        for line in data:
            tokens = line["token"]
            relation = self.process_relation_name(line["relation"])
            subj = line["h"]
            obj = line["t"]
            tokenized_text = [self.tokenizer.cls_token] + self.tokenizer.tokenize(' '.join(tokens)) + [
                self.tokenizer.sep_token]
            subj_name = tokens[subj["pos"][0]:subj["pos"][1]]
            obj_name = tokens[obj["pos"][0]:obj["pos"][1]]
            tokenized_text += self.tokenizer.tokenize(' '.join(subj_name))

            tokenized_relation = self.tokenizer.tokenize(relation)

            relation_pos = [len(tokenized_text), len(tokenized_text) + len(tokenized_relation)]

            tokenized_text += tokenized_relation + self.tokenizer.tokenize(' '.join(obj_name)) + [
                self.tokenizer.sep_token]

            assert len(tokenized_text) <= self.args.max_length

            text = self.tokenizer.convert_tokens_to_string(tokenized_text)
            processed_data.append({'text': text, 'relation_pos': relation_pos})

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, example):

        texts = []
        relation_pos = []

        for line in example:
            texts.append(line['text'])
            relation_pos.append(line['relation_pos'])

        features = self.tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")

        mlm_input_ids, mlm_labels = self.mask_tokens(features['input_ids'])

        if self.is_eval:
            mlm_input_ids = features['input_ids'].clone()
            mlm_labels[:] = -100

        for i in range(len(example)):
            start, end = relation_pos[i]

            for j in range(start, end):
                mlm_labels[i][j] = features['input_ids'][i][j]
                mlm_input_ids[i][j] = self.tokenizer.mask_token_id

        features['input_ids'] = mlm_input_ids
        features['labels'] = mlm_labels

        return features

    def mask_tokens(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.args.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class RETestDataset(data.Dataset):

    def __init__(self, fname, tokenizer, args, is_eval=False):

        self.args = args
        self.tokenizer = tokenizer
        self.is_eval = is_eval

        self.data = []
        with open(os.path.join(args.dataset_dir, fname)) as f:
            for line in f:
                self.data.append(json.loads(line))

        self.data = self.preprocess(self.data)

    def process_relation_name(self, relation):

        relation = relation.split(':')[1].replace('_', ' ')
        relation = relation.replace('stateorprovince', 'state or province')
        return relation

    def preprocess(self, data):

        processed_data = []

        for line in data:
            tokens = line["token"]
            relation = line["relation"]
            subj = line["h"]
            obj = line["t"]

            tokenized_text = [self.tokenizer.cls_token] + self.tokenizer.tokenize(' '.join(tokens)) + [
                self.tokenizer.sep_token]

            subj_name = tokens[subj["pos"][0]:subj["pos"][1]]
            obj_name = tokens[obj["pos"][0]:obj["pos"][1]]
            tokenized_text += self.tokenizer.tokenize(' '.join(subj_name))

            relation_pos = len(tokenized_text)

            tokenized_text += [self.tokenizer.mask_token] + self.tokenizer.tokenize(' '.join(obj_name)) + [
                self.tokenizer.sep_token]

            assert len(tokenized_text) <= self.args.max_length

            text = self.tokenizer.convert_tokens_to_string(tokenized_text)
            processed_data.append({'text': text, 'relation': relation, 'relation_pos': relation_pos, 'raw_data': line})

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, example):

        texts = []
        relation_pos = []

        for line in example:
            texts.append(line['text'])
            relation_pos.append(line['relation_pos'])

        features = self.tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt")

        return features, relation_pos, example
