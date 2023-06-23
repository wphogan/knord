from copy import deepcopy as dc
from os.path import join
from typing import List, Dict

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from utils.utils_common import load_jsonl_data
from utils.utils_dataset import REDataset, EntityMarker


class ORCADatamodule(LightningDataModule):
    """Data module"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.entityMarker = EntityMarker(config)
        self.train = []
        self.val = []
        self.test = []

    def prepare_data(self) -> None:
        return super().prepare_data()

    def print_len_data(self):
        print(f'Length of train data: {len(self.train)}')
        print(f'Length of test data: {len(self.test)}')

    def gather_examples(self):
        re_data_lab = load_jsonl_data(join(self.config.data_dir, self.config.fname_labeled))
        re_data_unl = load_jsonl_data(join(self.config.data_dir, self.config.fname_unlabeled))
        assert type(re_data_lab[0]) is dict
        assert type(re_data_unl[0]) is dict
        return re_data_lab, re_data_unl

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
            'input_ids': torch.LongTensor(np.asarray([x['input_ids'] for x in batch])),
            'attention_mask': torch.LongTensor(np.asarray([x['attention_mask'] for x in batch])),
            'h_pos': torch.LongTensor(np.asarray([x['h_pos'] for x in batch])),
            't_pos': torch.LongTensor(np.asarray([x['t_pos'] for x in batch])),
            'h_pos_l': torch.LongTensor(np.asarray([x['h_pos_l'] for x in batch])),
            't_pos_l': torch.LongTensor(np.asarray([x['t_pos_l'] for x in batch])),
            'labels': torch.LongTensor(np.asarray([x['labels'] for x in batch])),
            'weak_labels': torch.LongTensor(np.asarray([x['weak_labels'] for x in batch])),
            'is_labeled': torch.BoolTensor([x['is_labeled'] for x in batch]),
        }

    def setup(self, stage=None):
        # Load data
        labeled_exs, unlabeled_exs = self.gather_examples()

        # Assign to use in dataloaders
        self.train = labeled_exs
        self.test = unlabeled_exs
        self.print_len_data()

    def train_dataloader(self):

        train_dataset = MixedBatchMultiviewDataset(self.config,
                                                   known_exs=dc(self.train),
                                                   unknown_exs=dc(self.test),
                                                   em=self.entityMarker,
                                                   )
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.config.batch_size // 2,
                                      shuffle=True, num_workers=self.config.num_workers,
                                      pin_memory=True,
                                      collate_fn=self.collate_batch_feat)  # set to False for extracting features
        return train_dataloader

    def test_dataloader(self):
        data_test_unlabeled = REDataset(self.config, dc(self.test), em=self.entityMarker, is_labeled=True)
        test_dataloader = DataLoader(data_test_unlabeled, batch_size=self.config.batch_size,
                                     shuffle=False, num_workers=self.config.num_workers)
        return test_dataloader


class MixedBatchMultiviewDataset(Dataset):
    '''
    A dataset that produces a batch with mixed labeled and unlabeled instances.
    '''

    def __init__(self, config, known_exs, unknown_exs, em) -> None:
        super().__init__()
        self.config = config
        self.known_exs = REDataset(config, known_exs, em, is_labeled=True)
        self.unknown_exs = REDataset(config, unknown_exs, em, is_labeled=False)

    def __len__(self):
        return max([len(self.known_exs), len(self.unknown_exs)])

    def __getitem__(self, index):
        labeled_index = int(index % len(self.known_exs))
        labeled_ins = self.known_exs[labeled_index]
        unlabeled_index = int(index % len(self.unknown_exs))
        unlabeled_ins = self.unknown_exs[unlabeled_index]
        return labeled_ins, unlabeled_ins
