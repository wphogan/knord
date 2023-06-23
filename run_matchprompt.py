from transformers import logging

from baselines.matchprompt import MatchPromptModel
from utils.utils_common import define_callbacks, define_trainer, save_confidence, load_state_dict
from utils.utils_config import load_config
from utils.utils_matchprompt_datamodule import MPDatamodule

logging.set_verbosity_error()


def main():
    # Load config
    config, logger = load_config()
    assert 'matchprompt' in config.model_name

    # QA: required for MatchPrompt predictions
    assert config.use_confidence \
           or config.constrain_pred_type == 'seen_only' \
           or config.constrain_pred_type == 'novel_only'

    # DataModule
    dm = MPDatamodule(config)
    dm.setup()
    model = MatchPromptModel(config)

    # Load pretrained model
    if config.load_pretrained:
        model.load_pretrained_model(load_state_dict(config, load_pretrained=True))

    # Trainer & callbacks
    callbacks = define_callbacks(config, monitor='val_loss', mode='min')
    trainer = define_trainer(config, logger, callbacks)
    if not config.eval_only:
        trainer.fit(model, datamodule=dm)

    # Test
    trainer.test(
        ckpt_path=trainer.checkpoint_callback.best_model_path,
        datamodule=dm
    )

    # Save confidence scores from pre-trained model
    if config.supervised_pretrain:
        save_confidence(config, model, dm)
    print(f'Run  finished --> {config.run_name}')


if __name__ == '__main__':
    print('Starting...')
    main()
    print('Done')
