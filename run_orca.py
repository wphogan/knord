from baselines.orca import ORCA_Model
from utils.utils_common import define_callbacks, define_trainer
from utils.utils_config import load_config
from utils.utils_orca_datamodule import ORCADatamodule


def main():
    # Load config
    config, logger = load_config()
    assert 'orca' in config.model_name

    # DataModule
    dm = ORCADatamodule(config)
    dm.setup()
    model = ORCA_Model(config)

    # Trainer & callbacks
    callbacks = define_callbacks(config, monitor='train_loss', mode='min')
    trainer = define_trainer(config, logger, callbacks)
    if not config.eval_only:
        trainer.fit(model, dm.train_dataloader(), dm.test_dataloader())

    # Test
    trainer.test(
        ckpt_path=trainer.checkpoint_callback.best_model_path,
        dataloaders=dm.test_dataloader()
    )
    print(f'Run  finished --> {config.run_name}')


if __name__ == '__main__':
    print('Starting...')
    main()
    print('Done')
