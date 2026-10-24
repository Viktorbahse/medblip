import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule


from Original_BLIP.blip_pretrain import blip_pretrain


class BLIPLightningModule(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters({"config": config})
        self.config = config

        self.model = blip_pretrain(
            med_config = self.config.med_config,
            image_size=self.config.img_size,
            vit=self.config.vit,
            vit_grad_ckpt=self.config.vit_grad_ckpt,
            vit_ckpt_layer=self.config.vit_ckpt_layer,
            queue_size=self.config.queue_size,
        )

    def forward(self, image, caption, alpha):
        return self.model(image, caption, alpha=alpha)

    def training_step(self, batch, batch_idx):
        image, caption = batch

        # alpha = alpha_max * min(1, (epoch + i/len) / 2)
        n_batches = self.trainer.num_training_batches
        alpha = alpha = self.config.alpha * min(
            1.0, (self.current_epoch + batch_idx / n_batches) / 2.0
        )

        loss_ita, loss_itm, loss_lm = self(image, caption, alpha)
        loss = loss_ita + loss_itm + loss_lm

        bs = image.size(0)
        self.log("train/loss",     loss,     on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train/loss_ita", loss_ita, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/loss_itm", loss_itm, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/loss_lm",  loss_lm,  on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/lr", float(self.trainer.optimizers[0].param_groups[0]["lr"]), on_step=True, on_epoch=False)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.config.init_lr,
            weight_decay=self.config.weight_decay,
        )

        init_lr      = self.config.init_lr
        warmup_lr    = self.config.warmup_lr
        warmup_steps = self.config.warmup_steps
        min_lr       = self.config.min_lr
        decay_rate   = self.config.lr_decay_rate

        def lr_lambda(step: int) -> float:
            epoch = self.trainer.current_epoch
            if epoch == 0:
                i = max(step - 1, 0)
                lr = warmup_lr + (init_lr - warmup_lr) * (i / warmup_steps)
            else:
                lr = max(min_lr, init_lr * (decay_rate ** epoch))
            return lr / init_lr

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }




