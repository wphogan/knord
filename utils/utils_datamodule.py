import json
from os.path import join

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, ConcatDataset

from utils.utils_common import load_jsonl_data, split_train_val
from utils.utils_dataset import EntityMarker, REDataset


class KnordDatamodule(LightningDataModule):
    """Data module"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.entityMarker = EntityMarker(config)
        self.train_labeled = []
        self.val_labeled = []
        self.train_unlabeled_weak = []
        self.val_unlabeled_weak = []
        self.test_unlabeled = []

    def prepare_data(self) -> None:
        return super().prepare_data()

    def load_data(self):
        labeled_exs = load_jsonl_data(join(self.config.data_dir, self.config.fname_labeled))
        unlabeled_exs = load_jsonl_data(join(self.config.data_dir, self.config.fname_unlabeled))

        # Weak labels
        hq_weak_labeles_ex = []  # high quality pseudo labels
        with open(join(self.config.data_dir, self.config.fname_unlabeled_clusters)) as f:
            all_lines = f.readlines()
            for line in all_lines:
                if len(line.strip()) == 0:
                    continue
                ins = json.loads(line)
                if ins['is_lower_40']:
                    if ins['high_q'] <= self.config.high_q_data_threshold:
                        hq_weak_labeles_ex.append(ins)

        assert type(labeled_exs[0]) is dict
        assert type(hq_weak_labeles_ex[0]) is dict

        return labeled_exs, unlabeled_exs, hq_weak_labeles_ex

    def setup(self, stage=None):
        # Load data
        labeled_exs, unlabeled_exs, hq_unlabeled_weak_exs = self.load_data()

        # Create val splits
        labeled_exs_train, labeled_exs_val = split_train_val(self.config, labeled_exs)
        hq_unlabeled_weak_exs_train, hq_unlabeled_weak_exs_val = split_train_val(self.config, hq_unlabeled_weak_exs)

        # Assign to use in dataloaders
        self.train_labeled = labeled_exs_train
        self.train_unlabeled_weak = hq_unlabeled_weak_exs_train

        self.val_labeled = labeled_exs_val
        self.val_unlabeled_weak = hq_unlabeled_weak_exs_val

        self.test_unlabeled = unlabeled_exs

    def train_dataloader(self):
        data_train_labeled = REDataset(self.config, self.train_labeled, em=self.entityMarker, is_labeled=True)
        data_train_unlabeled = REDataset(self.config, self.train_unlabeled_weak, em=self.entityMarker,
                                         is_labeled=False, is_weak=True)

        train_concat_set = ConcatDataset([data_train_labeled, data_train_unlabeled])
        train_dataloader = DataLoader(train_concat_set, batch_size=self.config.batch_size, shuffle=True,
                                      num_workers=self.config.num_workers)
        return train_dataloader

    def val_dataloader(self):
        data_val_labeled = REDataset(self.config, self.val_labeled, em=self.entityMarker, is_labeled=True)
        data_val_unlabeled = REDataset(self.config, self.val_unlabeled_weak, em=self.entityMarker,
                                       is_labeled=False, is_weak=True)
        val_concat_set = ConcatDataset([data_val_labeled, data_val_unlabeled])
        val_dataloader = DataLoader(val_concat_set, batch_size=self.config.batch_size, shuffle=False,
                                    num_workers=self.config.num_workers)
        return val_dataloader

    def test_dataloader(self):
        data_test_unlabeled = REDataset(self.config, self.test_unlabeled, em=self.entityMarker, is_labeled=False)
        test_dataloader = DataLoader(data_test_unlabeled, batch_size=self.config.batch_size, shuffle=False,
                                     num_workers=self.config.num_workers)
        return test_dataloader
