import json
from collections import Counter
from collections import defaultdict


def parent_id_loop_fixes(in_dict):
    out_dict = {}
    n_looping = 0
    fixes = {
        'Q443153': 'Q968159',
        'Q935666': 'Q27943370',
        'Q651120': 'Q4830453',
        'Q43229': 'Q106559804',
        'Q155076': 'Q3778211',
        'Q610877': 'Q968159',
        'Q1799072': 'Q104637332',
        'Q8171': 'Q115372263',
        'Q11336353': 'Q108289644',
        'Q2695280': 'Q1799072',
        'Q2944660': 'Q11953984',
        'Q27318': 'Q1379672',
        'Q1355298': 'Q2585724',
        'Q190122': 'Q28038717',
    }
    for k, v in in_dict.items():
        if k in fixes:
            out_dict[k] = fixes[k]
            n_looping += 1
        else:
            out_dict[k] = v
    print(f'Fixed {n_looping} looping parents.')
    return out_dict


# Load data and gather ent ids
def load_qids_from_fewrel():
    data_train_wiki = json.load(open('train_wiki.json'))
    data_val_wiki = json.load(open('val_wiki.json'))
    data_splits = [data_train_wiki, data_val_wiki]
    all_ent_ids = set()
    for data_split in data_splits:
        for rel_class, instances in data_split.items():
            for instance in instances:
                all_ent_ids.add(instance['h'][1])
                all_ent_ids.add(instance['t'][1])

    all_ent_ids = list(all_ent_ids)
    return all_ent_ids


def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]


def clean_tid2tname(fname):
    clean_tid2tn = {}
    name_tracker = defaultdict(list)

    with open(fname) as f:
        all_lines = f.readlines()
        for line in all_lines:
            dict_ins = json.loads(line)
            for t_id, t_name in dict_ins.items():
                if t_name:
                    name_tracker[t_id].append(t_name)

    distinct_names = {}
    for t_id, name_list in name_tracker.items():
        mcn = most_common(name_list)
        clean_tid2tn[t_id] = mcn
        distinct_names[mcn] = t_id

    return clean_tid2tn
