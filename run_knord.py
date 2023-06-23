from model.model import KnordModel
from utils.utils_common import define_callbacks, define_trainer
from utils.utils_config import load_config
from utils.utils_datamodule import KnordDatamodule


def main():
    # Load config
    config, logger = load_config()
    assert 'knord' in config.model_name

    # DataModule
    dm = KnordDatamodule(config)
    dm.setup()
    model = KnordModel(config)

    # Trainer & callbacks
    callbacks = define_callbacks(config, monitor='val_f1_all_weak', mode='max')
    trainer = define_trainer(config, logger, callbacks)
    if not config.eval_only:
        trainer.fit(model, datamodule=dm)

    # Test
    trainer.test(
        ckpt_path=trainer.checkpoint_callback.best_model_path,
        datamodule=dm
    )
    print(f'Run  finished --> {config.run_name}')
    print(f'Best model path: {trainer.checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    print('Starting...')
    main()
    print('Done')
