import json
import os
from collections import defaultdict
from os.path import join

import numpy as np


def load_json_dict(fname):
    with open(fname) as f:
        return json.load(f)


def load_instances_from_retacred(fname):
    with open(join('retacred', fname)) as f:
        all_lines = f.readlines()
    return all_lines


def test_jsonl_load(fname):
    with open(join('processed', fname)) as f:
        all_lines = f.readlines()
    for line in all_lines:
        ins = json.loads(line)


def collect_no_rel_instances(data_type, tacred2fewrel_types):
    fname = f're-tacred_{data_type}_0.15_longtail_combo.jsonl'
    data = load_instances_from_retacred(fname)
    types = set()
    no_rels = []
    for instance in data:
        ins = json.loads(instance)
        ins['h']['type'] = tacred2fewrel_types[ins['h']['type']]
        ins['t']['type'] = tacred2fewrel_types[ins['t']['type']]
        types.add(ins['h']['type'])
        types.add(ins['t']['type'])
        if ins['relation'] == 'no_relation':
            ins['relation'] = 'P0'
            no_rels.append(ins)
    print(types)
    return no_rels


def load_instances_from_jsonl(fname):
    collected_dict = {}
    if not os.path.exists(fname):
        with open(fname, 'w') as f:
            f.write('')

    with open(fname) as f:
        all_lines = f.readlines()
    for line in all_lines:
        dict_ins = json.loads(line)

        for k, v in dict_ins.items():
            collected_dict[k] = v
    return collected_dict


def span_mod(instance, h_or_t):
    # if len(instance[h_or_t][2]) != 1:
    # print('Map to first mention in multi-mention sentence. (TACRED only supports single mentions)')
    # assert len(instance[h_or_t][2]) == 1, 'uh oh, multi pos found, not supported with tacred'

    if len(instance[h_or_t][2][0]) == 1:
        start = instance[h_or_t][2][0][0]
        end = start + 1
    elif len(instance[h_or_t][2][0]) >= 2:
        start = instance[h_or_t][2][0][0]
        end = instance[h_or_t][2][0][-1] + 1
    else:
        raise NotImplementedError
    return start, end


def ent_mod(instance, h_or_t, e2t):
    # h: [name, qid, [[locations]]] -> h: {pos: [], type: "", name: "", qid = ""}
    start, end = span_mod(instance, h_or_t)
    name = instance[h_or_t][0]
    qid = instance[h_or_t][1]
    return {
        "name": name,
        "qid": qid,
        "pos": [start, end],
        "type": e2t.get(qid, 'unknown')
    }


def adapt_to_tacred_format(instance, rel_class, e2t, qid2name):
    '''
    {'tokens': ['Merpati', 'flight', '106', 'departed', 'Jakarta', '(', 'CGK', ')', 'on', 'a', 'domestic', 'flight',
                'to', 'Tanjung', 'Pandan', '(', 'TJQ', ')', '.'], 'h': ['tjq', 'Q1331049', [[16]]],
     't': ['tanjung pandan', 'Q3056359', [[13, 14]]]}
    --to-->
    {"token": ["In", "1983", ",", "a", "year", "after", "the", "rally", ",", "Forsberg", "received", "the", "so-called", 
    "``", "genius", "award", "''", "from", "the", "John", "D.", "and", "Catherine", "T.", "MacArthur", "Foundation", "."], 
    "relation": "no_relation", "h": {"pos": [9, 10], "type": "PERSON", "name": "Forsberg"}, 
    "t": {"pos": [19, 21], "type": "PERSON", "name": "John D."}}
    '''

    # -- Map --
    instance_mapped = {}
    instance_mapped['token'] = instance['tokens']
    instance_mapped['h'] = ent_mod(instance, 'h', e2t)
    instance_mapped['t'] = ent_mod(instance, 't', e2t)

    # -- Type --
    h_qid = instance['h'][1]
    t_qid = instance['t'][1]
    instance_mapped['h']['type'] = qid2name.get(e2t[h_qid], 'entity')
    instance_mapped['t']['type'] = qid2name.get(e2t[t_qid], 'entity')

    # -- Add --
    instance_mapped['relation'] = rel_class

    return instance_mapped


def stats(data_dict, name):
    total = 0
    for rel, instances in data_dict.items():
        total += len(instances)
    print(f'Total instances in {name}: {total}, n classes: {len(data_dict)}')


def save_list_to_file(f_path, list_to_save):
    with open(f_path, "w") as f:
        for s in list_to_save:
            f.write(str(s) + "\n")


def write_jsonl_file(data, fname):
    stats(data, fname)
    with open(join('processed', fname), 'w') as fout:
        for rel_class, instances in data.items():
            for instance in instances:
                json.dump(instance, fout)
                fout.write('\n')

    print(f'Wrote file: {fname}')


FEW_REL_WIKICOUNTS = {
    'P710': '632695', 'P137': '566748', 'P674': '136266', 'P466': '38111', 'P136': '1671000', 'P306': '42452',
    'P127': '493832', 'P400': '115873', 'P974': '72731', 'P1346': '260645', 'P460': '296124', 'P86': '235116',
    'P118': '193933', 'P264': '353608', 'P750': '175290', 'P58': '198600', 'P3450': '85545', 'P105': '3628487',
    'P276': '2833412', 'P101': '819431', 'P407': '16537473', 'P1001': '973537', 'P800': '111525', 'P131': '11819538',
    'P177': '29646', 'P364': '374018', 'P2094': '244475', 'P361': '4577559', 'P641': '2010730', 'P59': '7374358',
    'P413': '412121', 'P206': '84675', 'P412': '35322', 'P155': '1271900', 'P26': '754937', 'P410': '166688',
    'P25': '637855', 'P463': '483291', 'P40': '1644343', 'P921': '25737346', 'P931': '12414', 'P4552': '64753',
    'P140': '475680', 'P1923': '190586', 'P150': '1133315', 'P6': '28409', 'P27': '4777589', 'P449': '101480',
    'P1435': '2239892', 'P175': '524960', 'P1344': '761549', 'P39': '1305223', 'P527': '2119425', 'P740': '46942',
    'P706': '86080', 'P84': '83502', 'P495': '1534946', 'P123': '502250', 'P57': '355086', 'P22': '1008703',
    'P178': '52809', 'P241': '133603', 'P403': '115301', 'P1411': '54201', 'P135': '62887', 'P991': '27048',
    'P156': '1263534', 'P176': '151282', 'P31': '105465153', 'P1877': '13084', 'P102': '512905', 'P1408': '16735',
    'P159': '469128', 'P3373': '434872', 'P1303': '214286', 'P17': '15704053', 'P106': '10391419', 'P551': '313958',
    'P937': '400746', 'P355': '99501'
}


def sort_by_values(dict_in):
    dict_as_int = {}
    for k, v in dict_in.items():
        dict_as_int[k] = int(v)

    return dict(sorted(dict_as_int.items(), key=lambda x: x[1], reverse=True))


def main():
    # Seed
    np.random.seed(0)

    # Load files
    data_train_wiki = json.load(open('train_wiki.json'))
    data_val_wiki = json.load(open('val_wiki.json'))
    few_rel_wikicounts = sort_by_values(FEW_REL_WIKICOUNTS)
    tacred2fewrel_types = json.load(open('tacred_types2few_rel_types.json'))

    seen_classes = list(few_rel_wikicounts.keys())[:40]
    novel_classes = list(few_rel_wikicounts.keys())[40:]

    # Save seen / novel splits
    save_list_to_file('processed/few_rel_seen_class_ids.txt', seen_classes)
    save_list_to_file('processed/few_rel_novel_class_ids.txt', novel_classes)

    # Get type information
    qid2tid = json.load(open('type_stats/threshold_1000/id2tid_7_n24.json'))
    tid2name = json.load(open('clean_id2type_name.json'))

    # Run settings
    is_nonorel = True
    ratio = 0.15
    is_fully_sup = True

    if is_fully_sup:
        print('=' * 30)
        print('is_fully_sup:', is_fully_sup)
        print('=' * 30)

    # F-out names
    dataset_version = '_nonorel' if is_nonorel else ''
    prefix = 'full_sup_' if is_fully_sup else ''
    fname_out_label = f'{prefix}fewrel_label_{ratio}{dataset_version}.jsonl'
    fname_out_unlabel = f'{prefix}fewrel_unlabel_{ratio}{dataset_version}.jsonl'

    # Collect
    labeled_data = defaultdict(list)
    unlabeled_data = defaultdict(list)

    # Get no rel instances from retacred
    if not is_nonorel:
        labeled_data['no_relation'].extend(collect_no_rel_instances('label', tacred2fewrel_types))
        unlabeled_data['no_relation'].extend(collect_no_rel_instances('unlabel', tacred2fewrel_types))

    # Go through FewRel
    data_splits = [data_train_wiki, data_val_wiki]
    for data_split in data_splits:
        for rel_class, instances in data_split.items():
            if rel_class in seen_classes or is_fully_sup:
                for instance in instances:
                    instance_mapped = adapt_to_tacred_format(instance, rel_class, qid2tid, tid2name)
                    if np.random.rand() > ratio:
                        labeled_data[rel_class].append(instance_mapped)
                    else:
                        unlabeled_data[rel_class].append(instance_mapped)
            else:
                assert not is_fully_sup
                for instance in instances:
                    instance_mapped = adapt_to_tacred_format(instance, rel_class, qid2tid, tid2name)
                    unlabeled_data[rel_class].append(instance_mapped)

    write_jsonl_file(labeled_data, fname_out_label)
    write_jsonl_file(unlabeled_data, fname_out_unlabel)

    # Make sure new files load properly
    test_jsonl_load(fname_out_label)
    test_jsonl_load(fname_out_unlabel)
    print('Done. New data files saved and tested.')


if __name__ == '__main__':
    main()
