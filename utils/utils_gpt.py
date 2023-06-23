import json
import os
from collections import defaultdict


class GPTExamples:
    def __init__(self, dataset):
        self.rel_code2name = None
        self.dataset = dataset
        if dataset == 'fewrel':
            self.get_rel_code2name()

        # Relation ids
        self.rel2id = json.load(open(f'data/{self.dataset}/rel2id.json'))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        # Relation ids, code
        n_seen_classes, n_novel_classes = 0, 0
        if dataset == 'fewrel':
            n_seen_classes, n_novel_classes = 41, 79
        elif dataset == 'tacred':
            n_seen_classes, n_novel_classes = 21, 21
        elif dataset == 'retacred':
            n_seen_classes, n_novel_classes = 20, 20

        self.seen_class_ids = list(range(n_seen_classes))
        self.novel_class_ids = list(range(n_seen_classes, n_seen_classes + n_novel_classes))
        self.all_class_ids = self.seen_class_ids + self.novel_class_ids

        self.known_rel_codes = [self.id2rel[i] for i in self.seen_class_ids]
        self.novel_rel_codes = [self.id2rel[i] for i in self.novel_class_ids if i in self.id2rel]

        # Class names
        if dataset == 'fewrel':
            self.known_rel_names = [self.rel_code2name[code] for code in self.known_rel_codes]
            self.novel_rel_names = [self.rel_code2name[code] for code in self.novel_rel_codes]
        else:
            self.known_rel_names = [self.id2rel[code] for code in self.seen_class_ids]
            self.novel_rel_names = [self.id2rel[code] for code in self.novel_class_ids]
            self.tacred_natural_class_names()

        self.all_rel_names = self.known_rel_names + self.novel_rel_names
        assert len(self.all_rel_names) == len(self.id2rel)

    def tacred_natural_class_names(self):
        self.known_rel_names = [self.tr_rplc(x) for x in self.known_rel_names]
        self.novel_rel_names = [self.tr_rplc(x) for x in self.novel_rel_names]

    def tr_rplc(self, input_class):
        return input_class.replace('_', ' ').replace('per:', 'person ').replace('org:', 'organization ')

    def get_rel_code2name(self):
        rel_code2name = {}
        with open(f'data/{self.dataset}/relation_description.csv') as f:
            all_lines = f.readlines()
            for line in all_lines:
                code_name, desc = line.split('\t')
                code, name = code_name.split(':')
                rel_code2name[code] = name.replace('_', ' ')
        self.rel_code2name = rel_code2name

    def raw_ins2ex_and_label(self, instance):
        sentence = ' '.join(instance['token'])
        head, tail = self.recover_original_capitalization(instance)
        relation = instance['relation']
        return f'{sentence} ({head}) ({tail}) \n => ?', relation

    def recover_original_capitalization(self, instance):
        head = self.resolve_spans(instance['token'], instance['h']['pos'])
        tail = self.resolve_spans(instance['token'], instance['t']['pos'])
        assert instance['h']['name'].lower() == head.lower()
        assert instance['t']['name'].lower() == tail.lower()
        return head, tail

    def resolve_spans(self, tokens, span):
        # Resolve spans
        span2text = ' '.join(tokens[span[0]: span[1]])
        return span2text


# Load jsonl file
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


def get_processed_uids(fname):
    if os.path.exists(fname):
        # If file is empty, just continue
        if os.stat(fname).st_size == 0:
            return []
        else:
            prev_data = load_jsonl_data(fname)
            return [x['uid'] for x in prev_data]
    else:
        return []


# {rel_code: [ins1, ins2, ...]}
def rel_code2instances(data):
    rel_code2instances = defaultdict(list)
    for ins in data:
        rel_code2instances[ins['relation']].append(ins)
    return rel_code2instances


def save_jsonl_data(data, f_path):
    with open(f_path, 'w') as f:
        for ins in data:
            f.write(f'{json.dumps(ins)}\n')
    print(f'Saved data to: {f_path}')
