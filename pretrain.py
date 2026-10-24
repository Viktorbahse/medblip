from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import DictConfig, OmegaConf
from model import BLIPLightningModule
import pytorch_lightning as pl
import torch
from data_utils import create_dataset, create_loader
def main(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)
    torch.backends.cudnn.benchmark = True

    print("Creating dataset")
    dataset = create_dataset(config.dataset, min_scale=0.2)
    print("number of training samples: %d" % len(dataset))


    data_loader = create_loader(
        [dataset],
        [None],
        batch_size=[config.batch_size],
        num_workers=[4],
        is_trains=[True],
        collate_fns=[None],
    )[0]

    print("Creating model")
    module = BLIPLightningModule(config.model)

    trainer = pl.Trainer(
        fast_dev_run=config.fast_dev_run,
        max_epochs=config.max_epoch,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy="ddp" if (isinstance(config.devices, int) and config.devices > 1) or config.devices == -1 else "auto",
        precision=config.precision,
        use_distributed_sampler=True,
        log_every_n_steps=50,
        default_root_dir=config.output_dir,
        callbacks=[
            ModelCheckpoint(
                dirpath=config.output_dir,
                filename="blip-{epoch:02d}",
                save_top_k=-1,
                every_n_epochs=1,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.loggers.TensorBoardLogger(save_dir=config.output_dir, name=""),
    )

    trainer.fit(module, train_dataloaders=data_loader)


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    config = config.pretrain
    main(config)