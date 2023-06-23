from transformers import logging

from baselines.matchprompt import MatchPromptModel
from baselines.rocore import RoCOREModel
from baselines.tabs import TypeDiscoveryModel
from utils.utils_common import save_confidence, load_state_dict
from utils.utils_config import load_config
from utils.utils_matchprompt_datamodule import MPDatamodule
from utils.utils_tabs_datamodule import OpenTypeDataModule

logging.set_verbosity_error()


def main():
    # Load config
    config, logger = load_config()

    # DataModule
    if 'matchprompt' in config.model_name:
        dm = MPDatamodule(config)
        dm.setup()
        model = MatchPromptModel(config)
    elif 'tabs' in config.model_name:
        dm = OpenTypeDataModule(config)
        dm.setup()
        model = TypeDiscoveryModel(config, len(dm.train_dataloader()), tokenizer=dm.tokenizer)
    elif 'rocore' in config.model_name:
        dm = OpenTypeDataModule(config)
        dm.setup()
        model = RoCOREModel(config, len(dm.train_dataloader()), tokenizer=dm.tokenizer)
    else:
        raise ValueError('Unknown model name')

    # Load pretrained model
    model.load_pretrained_model(load_state_dict(config, load_pretrained=True))

    # Save confidence scores from pre-trained model
    save_confidence(config, model, dm)
    print(f'Run  finished --> {config.run_name}')


if __name__ == '__main__':
    print('Starting...')
    main()
    print('Done')
