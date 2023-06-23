from baselines.tabs import TypeDiscoveryModel
from utils.utils_common import define_callbacks, define_trainer, load_state_dict, save_confidence
from utils.utils_config import load_config
from utils.utils_tabs_datamodule import OpenTypeDataModule


def main():
    # Load config
    config, logger = load_config()
    assert 'tabs' in config.model_name

    # DataModule
    dm = OpenTypeDataModule(config)
    dm.setup()
    model = TypeDiscoveryModel(config, len(dm.train_dataloader()), tokenizer=dm.tokenizer)

    # Load pretrained model
    if config.load_pretrained:
        model.load_pretrained_model(load_state_dict(config, load_pretrained=True))

    # Trainer & callbacks
    callbacks = define_callbacks(config, monitor='val/loss', mode='min')
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
