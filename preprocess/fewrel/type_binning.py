import json
import os
import os.path
import shutil
from collections import defaultdict

from utils import load_qids_from_fewrel, clean_tid2tname, parent_id_loop_fixes


def sort_by_values(dict_in):
    return dict(sorted(dict_in.items(), key=lambda x: x[1], reverse=True))


def load_instances_from_jsonl(fname):
    collected_dict = {}
    with open(fname) as f:
        all_lines = f.readlines()
    for line in all_lines:
        dict_ins = json.loads(line)

        for k, v in dict_ins.items():
            if k not in collected_dict:
                collected_dict[k] = v
            else:
                assert v != collected_dict[k]
    return collected_dict


def get_type_counts(qid2tid, id2name):
    type_name_0_counts = defaultdict(int)
    type_id_0_counts = defaultdict(int)
    for ent_id, type_0_id in qid2tid.items():
        type_0_name = id2name.get(type_0_id, 'unknown name')
        type_name_0_counts[type_0_name] += 1
        type_id_0_counts[type_0_id] += 1
    type_id_0_counts = sort_by_values(type_id_0_counts)
    type_name_0_counts = sort_by_values(type_name_0_counts)
    return type_id_0_counts, type_name_0_counts


def convert_ent_id2name_to_ent_id2typeid(ent_id2type_name, type_name2type_id):
    # Get type ids for each entity
    ent_id2type_id = {}
    for ent_id, ent_type_name_0 in ent_id2type_name.items():
        if 'Q26895936' == ent_type_name_0:
            type_id_0 = ent_type_name_0
        else:
            type_id_0 = type_name2type_id[ent_type_name_0]
        if ent_id not in ent_id2type_id:
            ent_id2type_id[ent_id] = type_id_0
    return ent_id2type_id


def create_qid2tid_dict(all_qids, id2p):
    qid2parent_id = {}
    for qid in all_qids:
        qid2parent_id[qid] = id2p.get(qid, 'Q35120')
    return qid2parent_id


def main():
    # Load data and gather type ids
    fname_id2parent = "raw_id2parent.jsonl"
    fname_id2name = "raw_id2name.jsonl"
    all_qids = load_qids_from_fewrel()
    id2parent = load_instances_from_jsonl(fname_id2parent)
    id2parent = parent_id_loop_fixes(id2parent)
    id2name = clean_tid2tname(fname_id2name)
    qid2tid = create_qid2tid_dict(all_qids, id2parent)
    stats = defaultdict(list)

    # Main loop: different min bins size for entity classes
    thresholds = [50, 100, 200, 300, 400, 500, 1000]
    for threshold in thresholds:
        type_id_0_counts, type_name_0_counts = get_type_counts(qid2tid, id2name)
        i = 0
        type_id_counts = []
        rounds_name_counts = {}
        rounds_name_counts[0] = type_name_0_counts
        rounds_id_counts = [type_id_0_counts]
        tree_tacker = [qid2tid]

        #  While bins are too small, consolidate
        while len(type_id_counts) != len(rounds_id_counts[i - 1]):
            # If count is less than 1000, climb up one level
            new_qid2t_id = {}
            low_count_t_id2counts = {k: v for k, v in rounds_id_counts[i].items() if v < threshold}
            low_count_t_ids = low_count_t_id2counts.keys()
            for qid, t_id in tree_tacker[i].items():
                if t_id in low_count_t_ids:
                    parent_t_id = id2parent.get(t_id, 'deadend')
                    new_qid2t_id[qid] = parent_t_id
                else:
                    new_qid2t_id[qid] = t_id
            i += 1

            # Get new type counts
            type_id_counts, _ = get_type_counts(new_qid2t_id, id2name)
            rounds_id_counts.append(type_id_counts)
            tree_tacker.append(new_qid2t_id)

        # Last item in tree tacker is a duplicate
        tree_tacker.pop()
        rounds_id_counts.pop()

        # Resolve ids to names
        rounds_name_counts = []
        unks = []
        for id2count_dict in rounds_id_counts:
            name2count_dict = {}
            for t_id, count in id2count_dict.items():
                t_name = id2name.get(t_id, 'cannot find name')
                if t_name == 'cannot find name':
                    unks.append(t_id)
                name2count_dict[t_name] = count
            rounds_name_counts.append(sort_by_values(name2count_dict))

        print('unk names: ', unks)

        # Save stats for this threshold
        if not os.path.exists(f'type_stats/threshold_{threshold}'):
            os.mkdir(f'type_stats/threshold_{threshold}')
        else:
            shutil.rmtree(f'type_stats/threshold_{threshold}')
            os.mkdir(f'type_stats/threshold_{threshold}')

        for round, (name2count_dict, qid2tid) in enumerate(zip(rounds_name_counts, tree_tacker)):
            n = len(name2count_dict)
            stats[threshold].append(n)

            with open(f'type_stats/threshold_{threshold}/id2tid_{round}_n{n}.json', 'w') as f_out:
                json.dump(qid2tid, f_out)

            with open(f'type_stats/threshold_{threshold}/round_{round}_n{n}.txt', 'w') as f_out:
                assert 'human' in name2count_dict
                for name, count in name2count_dict.items():
                    f_out.write(f'{name},{count}\n')
        print(f'Wrote files with thrshold: {threshold}')

    # Print stats
    for threshold, stat_list in stats.items():
        print(f'threshold: {threshold}, {",".join([str(s) for s in stat_list])}')


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    main()
    print('\nCompelted file: ', os.path.basename(__file__))
