import os

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data_utils import create_dataset, create_loader
from model import BLIPLightningModule


def main(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)
    torch.backends.cudnn.benchmark = True

    print("Creating dataset")
    dataset = create_dataset(config.dataset, min_scale=0.2)
    print(f"number of training samples: {len(dataset)}")

    data_loader = create_loader(
        [dataset],
        [None],
        batch_size=[config.batch_size],
        num_workers=[config.num_workers],
        is_trains=[True],
        collate_fns=[None],
    )[0]

    print("Creating model")
    module = BLIPLightningModule(config.model)

    checkpoint_cb = ModelCheckpoint(
        dirpath=config.output_dir,
        filename="blip-{epoch:02d}",
        save_top_k=1,
        every_n_epochs=1,
        save_last=True,
        save_on_train_epoch_end=True,
        monitor=None,
        verbose=True,
    )

    trainer = pl.Trainer(
        fast_dev_run=config.fast_dev_run,
        max_epochs=config.max_epoch,
        accelerator=config.accelerator,
        devices=config.devices,
        strategy="ddp_find_unused_parameters_true"
        if (isinstance(config.devices, int) and config.devices > 1) or config.devices == -1
        else "auto",
        precision=config.precision,
        use_distributed_sampler=True,
        log_every_n_steps=50,
        default_root_dir=config.output_dir,
        callbacks=[
            checkpoint_cb,
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=pl.loggers.TensorBoardLogger(save_dir=config.output_dir, name="", version=0),
    )

    last_ckpt = os.path.join(config.output_dir, "last.ckpt")
    ckpt_path = last_ckpt if os.path.exists(last_ckpt) else None
    if ckpt_path:
        print(f"[resume] found {ckpt_path}, resuming training")
    else:
        print("[resume] no checkpoint found, starting from scratch")

    trainer.fit(
        module,
        train_dataloaders=data_loader,
        ckpt_path=ckpt_path,
        weights_only=False,
    )


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    config = config.pretrain
    main(config)
