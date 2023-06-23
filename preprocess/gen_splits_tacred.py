import json
import os
import shutil
from os.path import join

import numpy as np


def save_list_to_file(f_path, list_to_save):
    with open(f_path, "w") as f:
        for s in list_to_save:
            f.write(str(s) + "\n")


def custom_dir(cust_dir):
    if not os.path.exists(cust_dir):
        os.makedirs(cust_dir)
    return cust_dir


def map_retacred_to_tacred(instance):
    '''
    TACRED:
        "h": {"pos": [19, 20], "type": "ORGANIZATION", "name": "UNC"},
        "t": {"pos": [29, 33], "type": "ORGANIZATION", "name": "National Alliance for Reconstruction"},
    RETACRED:
        "subj_start": 19,
        "subj_end": 19, -->(-1)
        "obj_start": 29,
        "obj_end": 32, -->(-1)
        "subj_type": "ORGANIZATION",
        "obj_type": "ORGANIZATION"
    '''
    mapped_instance = {}
    mapped_instance['token'] = instance['token']
    mapped_instance['relation'] = instance['relation']
    h_pos = [instance['subj_start'], instance['subj_end'] + 1]
    t_pos = [instance['obj_start'], instance['obj_end'] + 1]
    mapped_instance['h'] = {
        "pos": h_pos,
        "type": instance['subj_type'],
        "name": ' '.join(instance['token'][h_pos[0]:h_pos[1]])
    }
    mapped_instance['t'] = {
        "pos": t_pos,
        "type": instance['obj_type'],
        "name": ' '.join(instance['token'][t_pos[0]:t_pos[1]])
    }
    return mapped_instance


def get_filter_ids(fnames):
    ids_to_filter = set()
    for fname in fnames:
        with open(fname) as f_in:
            data = f_in.readlines()
            for line in data:
                ids_to_filter.add(line.strip())
    return ids_to_filter


def gen_tacred_list(config):
    # Static class splits: seen classes
    updated_counts_retacred_longtail = ['no_relation', 'per:identity', 'per:title', 'per:employee_of',
                                        'org:top_members/employees', 'org:alternate_names',
                                        'org:country_of_branch', 'org:city_of_branch', 'per:age', 'org:members',
                                        'per:origin', 'per:countries_of_residence',
                                        'per:spouse', 'org:member_of', 'org:stateorprovince_of_branch',
                                        'per:date_of_death',
                                        'per:children',
                                        'per:cities_of_residence', 'per:stateorprovinces_of_residence', 'per:parents']

    updated_counts_retacred_longtail_nonorel = ['per:identity', 'per:title', 'per:employee_of',
                                                'org:top_members/employees',
                                                'org:alternate_names',
                                                'org:country_of_branch', 'org:city_of_branch', 'per:age', 'org:members',
                                                'per:origin', 'per:countries_of_residence',
                                                'per:spouse', 'org:member_of', 'org:stateorprovince_of_branch',
                                                'per:date_of_death', 'per:children',
                                                'per:cities_of_residence', 'per:stateorprovinces_of_residence',
                                                'per:parents']

    tacred_longtail = ['no_relation', 'per:title', 'org:top_members/employees', 'per:employee_of',
                       'org:alternate_names', 'org:country_of_headquarters', 'per:countries_of_residence', 'per:age',
                       'org:city_of_headquarters', 'per:cities_of_residence', 'per:stateorprovinces_of_residence',
                       'per:origin', 'org:subsidiaries', 'org:parents', 'per:spouse',
                       'org:stateorprovince_of_headquarters', 'per:children', 'per:other_family', 'org:members',
                       'per:siblings', 'per:parents']
    tacred_longtail_nonorel = ['per:title', 'org:top_members/employees', 'per:employee_of',
                               'org:alternate_names', 'org:country_of_headquarters', 'per:countries_of_residence',
                               'per:age',
                               'org:city_of_headquarters', 'per:cities_of_residence',
                               'per:stateorprovinces_of_residence',
                               'per:origin', 'org:subsidiaries', 'org:parents', 'per:spouse',
                               'org:stateorprovince_of_headquarters', 'per:children', 'per:other_family', 'org:members',
                               'per:siblings', 'per:parents']
    np.random.seed(0)
    # 0: longtail_nonorel_combo 'longtail_combo'| longtail | nonorel | combo
    # 1: updated_counts_retacred_longtail_nonorel, updated_counts_retacred_longtail
    # 2: ratio
    runs = [
        ('tacred_longtail_nonorel', tacred_longtail, 0.15)
    ]
    controller(config, runs)


def controller(config, runs):
    ### RUN SETTINGS
    for run in runs:
        dataset_version, seen_rel_class_names, ratio = run
        no_norel = ('nonorel' in dataset_version)
        is_fully_sup = False

        # Load rel name 2 id dict
        destination_dir = custom_dir(join(config.data_dir, 'processed', config.name))
        with open(os.path.join(config.dataset_dir, 'rel2id.json')) as f:
            rel2id = json.loads(f.read())
        shutil.copy(os.path.join(config.dataset_dir, 'rel2id.json'), os.path.join(destination_dir, 'rel2id.json'))

        seen_rel_class_ids = [rel2id[x] for x in seen_rel_class_names]
        print('--> n seen classes: ', len(seen_rel_class_names))
        prefix = 'full_sup_' if is_fully_sup else ''

        fname_out_label = f'/{prefix}{config.name}_label_%.1f_{dataset_version}.jsonl'
        fname_out_unlabel = f'/{prefix}{config.name}_unlabel_%.1f_{dataset_version}.jsonl'
        fout_train_label = open(destination_dir + fname_out_label % (ratio), 'w')
        fout_train_unlabel = open(destination_dir + fname_out_unlabel % (ratio), 'w')

        # Write seen class ids to file
        save_list_to_file(destination_dir + f'/{config.name}_seen_class_ids_{dataset_version}.txt', seen_rel_class_ids)

        # Load dataset, save to
        files = ['train', 'dev', 'test']

        for file in files:
            with open(os.path.join(config.dataset_dir, f'{file}.json')) as f:
                if config.name == 'tacred':  # parse tacred
                    all_instances = []
                    all_lines = f.readlines()
                    for line in all_lines:
                        ins = json.loads(line)
                        all_instances.append(ins)
                elif config.name == 're-tacred':  # parse re-tacred
                    data = f.readlines()
                    all_instances = json.loads(data[0])
                else:
                    raise NotImplementedError

                # Make sure data is parsed properly
                assert type(all_instances[0]) is dict

                # Iter through all instances
                for instance in all_instances:
                    if no_norel:
                        if instance['relation'] == 'no_relation': continue
                    if config.name == 're-tacred':
                        instance = map_retacred_to_tacred(instance)

                    if instance['relation'] in seen_rel_class_names or is_fully_sup:
                        if np.random.rand() > ratio:
                            json.dump(instance, fout_train_label)
                            fout_train_label.write('\n')
                        else:
                            json.dump(instance, fout_train_unlabel)
                            fout_train_unlabel.write('\n')
                    else:
                        json.dump(instance, fout_train_unlabel)
                        fout_train_unlabel.write('\n')

        fout_train_label.close()
        fout_train_unlabel.close()
        print(f'Preprocess files saved to {destination_dir}.')
        print(f'Wrote: {fname_out_label}')
        print(f'Wrote: {fname_out_unlabel}')
