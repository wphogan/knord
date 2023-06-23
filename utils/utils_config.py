import argparse
import json
import os
import sys
from datetime import datetime
from os.path import join
from pprint import pprint

import torch
import yaml
from attrdict import AttrDict
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from yaml.loader import SafeLoader

from utils.utils_common import print_warning


def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def custom_vals_from_command_line():
    parser = argparse.ArgumentParser()
    assert 'config=' in ' '.join(sys.argv), 'Must specify config file'
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith('--'):
                arg = arg[2:]
                if '=' in arg:
                    k, v = arg.split('=')
                    parser.add_argument(f'--{k}', type=type(v), default=v)
                else:
                    parser.add_argument(f'--{arg}', type=type(arg), default=True)
    args = parser.parse_args()
    return args


def add_defaults(config_dict):
    defaults = {
        'load_pretrained': False,
        'supervised_pretrain': False,
        'eval_only': False,
        'checkpoint_dir': '',
        'save_best_model': True,
        'use_confidence': False,
        'constrain_pred_type': 'unconstrained',
        'num_workers': 4,
        'use_hungarian': True,
        'high_q_data_threshold': 0.15,
        'use_cluster_data': False,
    }
    for k, v in defaults.items():
        if k not in config_dict:
            config_dict[k] = v
    return config_dict


def load_config(hp_optimize_params=None):
    # Parse args and load main config file
    args = custom_vals_from_command_line()
    config_dict = yaml.load(open(join('configs', args.config)), Loader=SafeLoader)
    config_dict = add_defaults(config_dict)

    # Merge command line args to config file settings, prioritize command line args
    for arg in vars(args):
        val = getattr(args, arg)
        if val in ['True', 'False']:
            config_dict[arg] = eval(val)
        # if string is a number, convert to int or float
        elif is_float(val):
            if '.' in val:
                config_dict[arg] = float(val)
            else:
                config_dict[arg] = int(val)
        else:
            config_dict[arg] = val

        # Merge hyperparameter optimization settings
    if hp_optimize_params is not None:
        for k, v in hp_optimize_params.items():
            config_dict[k] = v

    # Set float types
    for k, v in config_dict.items():
        if 'e-' in str(v):
            config_dict[k] = float(v)

    # Print config
    pprint(config_dict)

    # QA on config
    if config_dict['use_confidence']:
        assert not config_dict['supervised_pretrain'], 'Cannot use confidence and supervised pretrain at the same time'
    if config_dict['supervised_pretrain']:
        assert not config_dict['use_confidence'], 'Cannot use supervised pretrain and confidence at the same time'
        # assert not config_dict['load_pretrained'], 'Cannot use supervised pretrain and load pretrained at the same time'

    # Generate file names and data dir, load rel2id
    config_dict['data_dir'] = join('data', config_dict["dataset"])
    combo = '_and_label' if config_dict['use_cluster_data'] else ''
    clusters = '_clustered' if config_dict['use_cluster_data'] else ''
    nonorel = '_nonorel' if config_dict['is_nonorel'] else ''
    representations = f'_{config_dict["representations"]}' if 'representations' in config_dict else ''
    config_dict['fname_labeled'] = f'{config_dict["dataset"]}_label{nonorel}.jsonl'
    config_dict['fname_unlabeled'] = f'{config_dict["dataset"]}_unlabel{nonorel}.jsonl'
    config_dict[
        'fname_unlabeled_clusters'] = f'{config_dict["dataset"]}_unlabel{combo}{nonorel}{clusters}{representations}.jsonl'
    config_dict['rel2id'] = json.load(open(join(config_dict['data_dir'], 'rel2id.json')))

    # Run name and working dir
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_time = 'debug' if config_dict['is_debug'] else current_time
    pretrain = '_pretrain' if config_dict['supervised_pretrain'] else ''
    exp_desc = f"_{config_dict['exp_description']}" if 'exp_description' in config_dict else ''
    run_name = f'{current_time}_{config_dict["dataset"]}_{config_dict["model_name"]}{nonorel}{exp_desc}{pretrain}'
    work_dir = join('logs', run_name)
    os.makedirs(work_dir, exist_ok=True)
    config_dict['run_name'] = run_name
    config_dict['output_dir'] = work_dir
    print(f'Working/output dir: {work_dir}')

    # Device settings
    try:  # jLab 5 server has newer version of PyTorch
        torch.set_float32_matmul_precision('medium')
    except AttributeError:
        pass
    config_dict['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Seen / novel classes
    if config_dict['dataset'] == 'fewrel':
        config_dict['n_seen_classes'], config_dict['n_novel_classes'] = 41, 79
    elif config_dict['dataset'] == 'tacred':
        config_dict['n_seen_classes'], config_dict['n_novel_classes'] = 21, 39
    elif config_dict['dataset'] == 'retacred':
        config_dict['n_seen_classes'], config_dict['n_novel_classes'] = 20, 40
    config_dict['seen_class_ids'] = list(range(config_dict['n_seen_classes']))
    config_dict['novel_class_ids'] = list(range(config_dict['n_seen_classes'],
                                                config_dict['n_seen_classes'] +
                                                config_dict['n_novel_classes']))
    config_dict['all_class_ids'] = \
        config_dict['seen_class_ids'] + config_dict['novel_class_ids']

    # Save dict to yaml file
    yaml.dump(config_dict, open(join(work_dir, 'run_settings.yaml'), 'w'), default_flow_style=False)

    # Warnings to user
    if config_dict['is_debug']: print_warning('Running in debug mode!')
    print_warning(f'Device {config_dict["device"]}')

    # Set seed
    seed_everything(config_dict['seed'])

    # Create attrdict
    config = AttrDict(config_dict)

    # Logger
    if config_dict['is_debug']:
        logger = False
    else:
        logger = WandbLogger(
            log_model=False,  # log_model="all",
            # project="x-open-re-eval", # EVAL RUN
            # name=f'x_{config_dict["orig_ckpt"]}',  # EVAL RUN
            project="open-re",
            name=run_name,
            save_dir=work_dir)
        logger.log_hyperparams(config_dict)

    return config, logger
