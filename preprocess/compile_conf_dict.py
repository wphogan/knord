import argparse
import json
import os
from os.path import join

import numpy as np
import pandas as pd


def full_path(checkpoint_path, split):
    return join('logs', checkpoint_path, f'confidences_{split}.csv')


def get_bools_above_mean(checkpoint_path):
    full_path_train = full_path(checkpoint_path, 'train')
    full_path_test = full_path(checkpoint_path, 'test')

    # Load csv files using dataframe: cols: uid, pred, prob, target
    df_train = pd.read_csv(full_path_train)
    df_test = pd.read_csv(full_path_test)
    data_train = np.array(df_train['prob'])
    data_test = np.array(df_test['prob'])

    # Get mean prob from train and get bools for test
    data_train_mean = np.mean(data_train, axis=0)
    data_test_above_mean = data_test >= data_train_mean
    print('data_train_mean:', data_train_mean)
    print('n test instances below mean:', np.sum(data_test_above_mean))
    return data_test_above_mean


def main():
    parser = argparse.ArgumentParser()  # Add an argument
    parser.add_argument('--checkpoint_name', type=str, required=True)  # Parse the argument
    args = parser.parse_args()

    checkpoints = [
        args.checkpoint_name
    ]

    for checkpoint in checkpoints:
        # Get unlabeled data with low confidence
        data_test_above_mean = get_bools_above_mean(checkpoint)

        # Get dataset and run name
        checkpoint_split = checkpoint.split('_')
        dataset = checkpoint_split[2]

        # Load data
        is_nonorel = '_nonorel' if 'nonorel' in checkpoint else ''
        data_unlabeled_path = join('data', dataset, f'{dataset}_unlabel{is_nonorel}.jsonl')
        with open(data_unlabeled_path, 'r') as f:
            orig_data_unlabeled = [json.loads(line) for line in f]
        assert len(orig_data_unlabeled) == len(
            data_test_above_mean), f'{len(orig_data_unlabeled)} != {len(data_test_above_mean)}'

        # Create uid2conf
        uid2conf = {}
        for i, d in enumerate(orig_data_unlabeled):
            uid2conf[d['uid']] = int(data_test_above_mean[i])

        # Save data
        output_path = join('logs', checkpoint, f'uid2is_seen_conf_based.json')
        with open(output_path, 'w') as json_file:
            json.dump(uid2conf, json_file)
        print('Saved', output_path)


if __name__ == '__main__':
    print('Starting file: ', os.path.basename(__file__))
    main()
    print('\nCompelted file: ', os.path.basename(__file__))
