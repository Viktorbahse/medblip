import gc
import json
import os

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from data_utils import create_dataset, create_loader
from model import BLIPCaption, BLIPRetrieval


def main(config: DictConfig):
    pl.seed_everything(config.seed, workers=True)
    torch.backends.cudnn.benchmark = True

    print("Creating test dataset")
    test_dataset = create_dataset(config.dataset, type="retrieval")
    print(f"number of test samples: {len(test_dataset)}")
    print(f"number of test texts  : {len(test_dataset.text)}")

    test_loader = create_loader(
        [test_dataset],
        [None],
        batch_size=[config.batch_size],
        num_workers=[config.num_workers],
        is_trains=[False],  # shuffle=False, drop_last=False
        collate_fns=[None],
    )[0]

    metrics = {}

    if "retrieval" in config.tests:
        print(f"Loading model from {config.ckpt_path}")
        module = BLIPRetrieval.load_from_checkpoint(
            config.ckpt_path,
            map_location="cpu",
            config=config.model,
            weights_only=False,
        )

        trainer = pl.Trainer(
            accelerator=config.accelerator,
            devices=config.devices,
            precision=config.precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )

        trainer.test(module, dataloaders=[test_loader], verbose=True)
        metrics.update(
            {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("test/")}
        )
        del module, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "caption" in config.tests:
        module = BLIPCaption.load_from_checkpoint(
            config.ckpt_path,
            map_location="cpu",
            config=config.model,
            test_file=config.dataset.test_file,
            weights_only=False,
        )

        trainer = pl.Trainer(
            accelerator=config.accelerator,
            devices=config.devices,
            precision=config.precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        trainer.test(module, dataloaders=[test_loader], verbose=True)
        metrics.update(
            {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith("test/")}
        )
        del module, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out = os.path.join(config.ckpt_path.rsplit("/", 1)[0], "test_metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved: {out}")


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    config = config.test
    main(config)
