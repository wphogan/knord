import json
from copy import deepcopy
from os.path import join
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from utils.utils_common import load_jsonl_data, split_train_val, batch_var_length
from utils.utils_tabs import clean_text, cluster_acc


class InputExample(object):
    def __init__(self, unique_id: Union[int, str], text: List[str],
                 head_span: List[int], tail_span: List[int],
                 label: int, relation_name: str, is_labeled: bool = True, is_seen_conf_based: bool = False):
        self.uid = unique_id
        self.tokens = text  # type: List[str]
        self.head_span = head_span
        self.tail_span = tail_span
        self.label = label if is_labeled else 0
        self.is_labeled = is_labeled
        self.is_seen_conf_based = is_seen_conf_based
        self.relation_name = relation_name
        self.pseudo_label = -1
        self.label_unseen = label if not is_labeled else -1  # ENSURE NO DATA LEAKAGE


class OpenTypeDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.foundation_model)
        self.train = []
        self.val_labeled = []
        self.val_unlabeled = []
        self.test = []
        self.is_novel_conf_based = []
        self.add_special_tokens()  # add new tokens to the tokenizer vocabulary

    def add_special_tokens(self):
        # new tokens
        new_tokens = ['<h>', '</h>', '<t>', '</t>', '<tgr>', '</tgr>']

        # check if the tokens are already in the vocabulary
        new_tokens = set(new_tokens) - set(self.tokenizer.vocab.keys())

        # add the tokens to the tokenizer vocabulary
        self.tokenizer.add_tokens(list(new_tokens))

    def prepare_data(self) -> None:
        return super().prepare_data()

    def gather_examples(self, fname, rel2id, confidence_dict, is_labeled=False):
        examples = []
        is_seen_conf_based = True
        re_data = load_jsonl_data(join(self.config.data_dir, fname))
        assert type(re_data[0]) is dict
        n_conf_seen = 0
        n_conf_novel = 0
        for ex in re_data:
            ex = map_to_tabs_format(ex)
            if self.config.use_confidence and not is_labeled:
                is_seen_conf_based = True if confidence_dict[str(ex['uid'])] == 1 else False
                n_conf_seen += 1 if is_seen_conf_based else 0
                n_conf_novel += 1 if not is_seen_conf_based else 0
            input_example = _create_example(ex, rel2id, is_labeled=is_labeled,
                                            is_seen_conf_based=is_seen_conf_based)
            examples.append(input_example)
        if self.config.use_confidence and not is_labeled:
            print(f'Confidence based seen/unseen: {n_conf_seen}/{n_conf_novel}')
        return examples

    def _split(self) -> Tuple[List[InputExample], List[InputExample]]:
        rel2id = json.load(open(join(self.config.data_dir, 'rel2id.json')))
        confidence_dict = {}
        if self.config.use_confidence:
            with open(join('logs', self.config.checkpoint_dir, 'uid2is_seen_conf_based.json')) as f:
                confidence_dict = json.load(f)
                print(f'Loaded confidence dict with {len(confidence_dict)} entries.')
        labeled_exs = self.gather_examples(self.config.fname_labeled, rel2id, confidence_dict, is_labeled=True)

        unlabeled_exs = self.gather_examples(self.config.fname_unlabeled, rel2id, confidence_dict, is_labeled=False)
        assert len(labeled_exs) > 0, f'No labeled_exs found'
        assert len(unlabeled_exs) > 0, f'No unlabeled_exs found'
        return labeled_exs, unlabeled_exs

    @staticmethod
    def collate_batch_feat(batch: List[List[Dict]]) -> List[Dict]:
        if isinstance(batch[0], tuple):
            # expand mixed batch
            new_batch = []
            for tup in batch:
                new_batch.append(tup[0])
                new_batch.append(tup[1])
            batch = new_batch

        assert isinstance(batch[0], list)

        views_n = len(batch[0])
        output = []
        for i in range(views_n):
            output_i = {
                'task': 'rel',
                'meta': [x[i]['meta'] for x in batch],
                'token_ids': batch_var_length([x[i]['token_ids'] for x in batch]),
                'attn_mask': batch_var_length([x[i]['attn_mask'] for x in batch]),
                'labels': torch.LongTensor([x[i]['label'] for x in batch]),
                'pseudo_labels': torch.LongTensor([x[i]['pseudo_label'] for x in batch]),
                'is_labeled_mask': torch.BoolTensor([x[i]['is_labeled'] for x in batch]),
                'labels_unseen': torch.LongTensor([x[i]['label_unseen'] for x in batch]),
            }

            if 'head_span' in batch[0][i]:
                output_i['head_spans'] = torch.LongTensor([x[i]['head_span'] for x in batch])
                output_i['tail_spans'] = torch.LongTensor([x[i]['tail_span'] for x in batch])

            if 'mask_bpe_idx' in batch[0][i]:
                output_i['mask_bpe_idx'] = torch.LongTensor([x[i]['mask_bpe_idx'] for x in batch])

            output.append(output_i)

        return output

    def setup(self, stage: Optional[str] = ''):
        '''
        Load labeled / unlabeled data
        '''
        labeled_exs, unlabeled_exs = self._split()
        labeled_exs_train, labeled_exs_val = split_train_val(self.config, labeled_exs)
        _, unlabeled_exs_val = split_train_val(self.config, unlabeled_exs)

        self.train = labeled_exs_train
        self.val_labeled = labeled_exs_val
        self.val_unlabeled = unlabeled_exs_val
        self.test = unlabeled_exs[:300] if self.config.is_debug else unlabeled_exs
        if self.config.use_confidence:
            self.is_novel_conf_based = [ex for ex in unlabeled_exs if not ex.is_seen_conf_based]
            assert len(self.is_novel_conf_based) > 0, f'No unknown instances found'
        print(f'dm.setup()--> train: {len(self.train)}, '
              f'val_labeled:, {len(self.val_labeled)}, '
              f'val_unlabeled: {len(self.val_unlabeled)}, '
              f'test: {len(self.test)}')
        print(f'dm.setup()--> novel train (conf based): {len(self.is_novel_conf_based)}')

    def train_dataloader(self):
        if self.config.use_confidence:
            train_dataset = MixedBatchMultiviewDataset(self.config, self.tokenizer,
                                                       known_exs=self.train,
                                                       unknown_exs=self.is_novel_conf_based,
                                                       feature=self.config.feature
                                                       )
        else:
            train_dataset = MixedBatchMultiviewDataset(self.config, self.tokenizer,
                                                       known_exs=self.train,
                                                       unknown_exs=self.test,
                                                       feature=self.config.feature
                                                       )

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.config.batch_size,
                                      shuffle=True, num_workers=self.config.num_workers,
                                      pin_memory=True,
                                      collate_fn=self.collate_batch_feat)  # set to False for extracting features

        return train_dataloader

    def val_dataloader(self):
        val_dataset = MixedBatchMultiviewDataset(self.config, self.tokenizer,
                                                 known_exs=self.val_labeled,
                                                 unknown_exs=self.val_unlabeled,
                                                 feature=self.config.feature
                                                 )
        # val_dataset = MultiviewDataset(self.config, self.tokenizer, exs=self.val,
        #                                feature=self.config.feature)
        val_dataloader = DataLoader(val_dataset, batch_size=4,
                                    shuffle=False, num_workers=self.config.num_workers,
                                    pin_memory=True, collate_fn=self.collate_batch_feat)

        return val_dataloader

    def test_dataloader(self):
        test_dataset = MultiviewDataset(self.config, self.tokenizer, exs=self.test,
                                        feature=self.config.feature)
        test_dataloader = DataLoader(test_dataset, batch_size=4,
                                     shuffle=False, num_workers=self.config.num_workers,
                                     pin_memory=True, collate_fn=self.collate_batch_feat)

        return test_dataloader


class MultiviewDataset(Dataset):
    def __init__(self, config, tokenizer: PreTrainedTokenizer,
                 exs: List[InputExample], feature: str = 'token') -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.feature_func = lambda x, t: [_convert_example_to_tok_feature(x, t),
                                          _convert_example_to_mask_feature(x, t)]
        self.feats = [self.feature_func(ex, self.tokenizer) for ex in exs]

    def update_pseudo_labels(self, uid2pl: Dict):
        for ex in self.feats:
            for view in ex:
                uid = view['meta']['uid']
                if uid in uid2pl:
                    view['pseudo_label'] = uid2pl[uid]
        return

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, index):
        return self.feats[index]


class MixedBatchMultiviewDataset(Dataset):
    '''
    A dataset that produces a batch with mixed labeled and unlabeled instances.
    '''

    def __init__(self, config, tokenizer: PreTrainedTokenizer,
                 known_exs: List[InputExample], unknown_exs: List[InputExample], feature: str = 'token') -> None:
        '''
        Following the UNO paper, we will sample 1 batch from the known classes and 1 batch from the unknown classes.
        '''
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.feature_func = lambda x, t: [_convert_example_to_tok_feature(x, t),
                                          _convert_example_to_mask_feature(x, t)]

        print('tokenizing features....')
        self.known_feats = [self.feature_func(ex, self.tokenizer) for ex in known_exs]
        self.unknown_feats = [self.feature_func(ex, self.tokenizer) for ex in unknown_exs]

    def update_pseudo_labels(self, uid2pl: Dict):
        for ex in self.unknown_feats:
            for view_idx, view in enumerate(ex):
                uid = view['meta']['uid']
                if uid in uid2pl:
                    if isinstance(uid2pl[uid], list):
                        view['pseudo_label'] = uid2pl[uid][view_idx]
                    else:
                        view['pseudo_label'] = uid2pl[uid]
        return

    def check_pl_acc(self):
        '''
        report pseudo label accuracy
        '''
        labels = []
        pls = []
        for feat in self.unknown_feats:
            labels.append(feat[0]['label'])
            pls.append(feat[0]['pseudo_label'])

        acc = cluster_acc(np.array(labels), np.array(pls), reassign=True)
        return acc

    def __len__(self):
        return max([len(self.known_feats), len(self.unknown_feats)])

    def __getitem__(self, index):
        labeled_index = int(index % len(self.known_feats))
        labeled_ins = self.known_feats[labeled_index]
        unlabeled_index = int(index % len(self.unknown_feats))
        unlabeled_ins = self.unknown_feats[unlabeled_index]
        return labeled_ins, unlabeled_ins


def _create_example(sample: Dict, rel2id: Dict, max_word_length: int = 80, is_labeled: bool = True,
                    is_seen_conf_based: bool = False) -> InputExample:
    '''
    Convert instance into InputExample class.
    The 'subj_end' and 'obj_end' indexes are contained in the entity span.
    Part of this code is from RoCORE.
    '''
    text = clean_text(sample['tokens'])  # remove non-ascii
    head_span = [sample['subj_start'], sample['subj_end']]
    tail_span = [sample['obj_start'], sample['obj_end']]
    if len(text) >= max_word_length:
        if head_span[1] < tail_span[0]:
            text = text[head_span[0]:tail_span[1] + 1]
            num_remove = head_span[0]
        else:
            text = text[tail_span[0]:head_span[1] + 1]
            num_remove = tail_span[0]
        head_span = [head_span[0] - num_remove, head_span[1] - num_remove]
        tail_span = [tail_span[0] - num_remove, tail_span[1] - num_remove]

    relation_id = rel2id[sample['relation']]
    relation_name = sample['relation']
    uid = sample['uid']
    input_example = InputExample(uid, text, head_span, tail_span, relation_id, relation_name, is_labeled,
                                 is_seen_conf_based)
    return input_example


def _convert_example_to_tok_feature(ex: InputExample, tokenizer: PreTrainedTokenizer) -> Dict:
    tokens = ex.tokens
    subj_tokens = tokens[ex.head_span[0]: ex.head_span[1] + 1]
    obj_tokens = tokens[ex.tail_span[0]: ex.tail_span[1] + 1]

    meta = {
        'uid': ex.uid,
        'tokens': ex.tokens,
        'is_labeled': ex.is_labeled,
        'is_seen_conf_based': ex.is_seen_conf_based,
        'label': ex.relation_name,
        'feature_type': 'token',
        'subj': ' '.join(subj_tokens),
        'obj': " ".join(obj_tokens)
    }

    # insert entity markers
    input_tokens = deepcopy(tokens)
    if ex.head_span[0] < ex.tail_span[0]:
        input_tokens.insert(ex.head_span[0], '<h>')
        input_tokens.insert(ex.head_span[1] + 2, '</h>')

        input_tokens.insert(ex.tail_span[0] + 2, '<t>')
        input_tokens.insert(ex.tail_span[1] + 4, '</t>')
        # get the head span and tail span in bpe offset
        substart = tokenizer.encode(' '.join(input_tokens[:ex.head_span[0] + 1]))
        subend = tokenizer.encode(' '.join(input_tokens[:ex.head_span[1] + 2]))
        head_span = (len(substart) - 1, len(subend) - 1)
        objstart = tokenizer.encode(' '.join(input_tokens[:ex.tail_span[0] + 3]))
        objend = tokenizer.encode(' '.join(input_tokens[:ex.tail_span[1] + 4]))
        tail_span = (len(objstart) - 1, len(objend) - 1)
    else:
        input_tokens.insert(ex.tail_span[0], '<t>')
        input_tokens.insert(ex.tail_span[1] + 2, '</t>')
        input_tokens.insert(ex.head_span[0] + 2, '<h>')
        input_tokens.insert(ex.head_span[1] + 4, '</h>')
        start1 = tokenizer.encode(' '.join(input_tokens[:ex.tail_span[0] + 1]))
        end1 = tokenizer.encode(' '.join(input_tokens[:ex.tail_span[1] + 2]))
        tail_span = (len(start1) - 1, len(end1) - 1)  # account for the ending [sep] token
        start2 = tokenizer.encode(' '.join(input_tokens[:ex.head_span[0] + 3]))
        end2 = tokenizer.encode(' '.join(input_tokens[:ex.head_span[1] + 4]))
        head_span = (len(start2) - 1, len(end2) - 1)

    sentence = ' '.join(input_tokens)

    token_ids = tokenizer.encode(sentence, return_tensors='pt').squeeze(0)  # (seq_len)
    seq_len = token_ids.size(0)
    attn_mask = torch.ones((seq_len))

    # assert (tokenizer.decode(token_ids[head_span[0]:head_span[1]]) == ' '.join(subj_tokens))
    return {
        'meta': meta,
        'token_ids': token_ids,
        'attn_mask': attn_mask,
        'head_span': head_span,
        'tail_span': tail_span,
        'label': ex.label,
        'is_labeled': ex.is_labeled,
        'pseudo_label': ex.pseudo_label,
        'is_seen_conf_based': ex.is_seen_conf_based,
        'label_unseen': ex.label_unseen,
    }


def _convert_example_to_mask_feature(ex: InputExample, tokenizer: PreTrainedTokenizer, prompt_idx: int = 0,
                                     max_len: int = 300) -> Dict:
    tokens = ex.tokens
    subj_tokens = tokens[ex.head_span[0]: ex.head_span[1] + 1]
    obj_tokens = tokens[ex.tail_span[0]: ex.tail_span[1] + 1]
    meta = {
        'uid': ex.uid,
        'tokens': ex.tokens,
        'is_labeled': ex.is_labeled,
        'label': ex.relation_name,
        'feature_type': 'mask',
        'subj': ' '.join(subj_tokens),
        'obj': " ".join(obj_tokens)
    }

    input_tokens = deepcopy(tokens)
    if ex.head_span[0] < ex.tail_span[0]:
        input_tokens.insert(ex.head_span[0], '<h>')
        input_tokens.insert(ex.head_span[1] + 2, '</h>')

        input_tokens.insert(ex.tail_span[0] + 2, '<t>')
        input_tokens.insert(ex.tail_span[1] + 4, '</t>')
    else:
        input_tokens.insert(ex.tail_span[0], '<t>')
        input_tokens.insert(ex.tail_span[1] + 2, '</t>')
        input_tokens.insert(ex.head_span[0] + 2, '<h>')
        input_tokens.insert(ex.head_span[1] + 4, '</h>')

    if prompt_idx == 0:
        # obj is the <mask> of subj
        prompt = obj_tokens + ['is', 'the', tokenizer.mask_token, 'of'] + subj_tokens
        mask_word_prefix = input_tokens + obj_tokens + ['is', 'the']
    elif prompt_idx == 1:
        # the relation between <subj> and <obj> is <mask>
        prompt = ['the', 'relation', 'between'] + subj_tokens + ['and'] + obj_tokens + ['is', tokenizer.mask_token]
        mask_word_prefix = input_tokens + prompt[:1]

    prefix_bpe = tokenizer.encode(' '.join(mask_word_prefix))
    mask_bpe_idx = len(prefix_bpe) - 1

    token_ids = tokenizer.encode(' '.join(input_tokens + prompt), return_tensors='pt').squeeze(0)
    seq_len = token_ids.size(0)

    attn_mask = torch.ones(seq_len, dtype=torch.bool)

    return {
        'meta': meta,
        'token_ids': token_ids,
        'attn_mask': attn_mask,
        'mask_bpe_idx': mask_bpe_idx,
        'label': ex.label,
        'is_labeled': ex.is_labeled,
        'pseudo_label': ex.pseudo_label,
        'is_seen_conf_based': ex.is_seen_conf_based,
        'label_unseen': ex.label_unseen,
    }


def map_to_tabs_format(instance):
    '''
    map the instance to the format of the original tabs dataset
    '''
    return {
        'tokens': instance['token'],
        'relation': instance['relation'],
        'subj_start': instance['h']['pos'][0],
        'subj_end': instance['h']['pos'][1],
        'obj_start': instance['t']['pos'][0],
        'obj_end': instance['t']['pos'][1],
        'subj_type': instance['h']['type'],
        'obj_type': instance['t']['type'],
        'uid': instance['uid']
    }
