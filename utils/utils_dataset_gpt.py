from torch.utils.data import Dataset

from utils.utils_gpt import GPTExamples


class OpenAIDataset(Dataset):
    def __init__(self, re_data, dataset_name):
        self.instances = ['' for x in range(len(re_data))]
        self.labels = ['' for x in range(len(re_data))]
        self.uids = ['' for x in range(len(re_data))]
        self.gpt_examples = GPTExamples(dataset_name)

        # Collect data
        for i, ins in enumerate(re_data):
            example, label = self.gpt_examples.raw_ins2ex_and_label(ins)
            self.instances[i] = example
            self.labels[i] = label
            self.uids[i] = ins['uid']

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return {
            'instance': self.instances[index],
            'label': self.labels[index],
            'uid': self.uids[index]
        }
