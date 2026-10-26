import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from tqdm import tqdm

from data_utils import build_caption_refs
from Original_BLIP.blip_pretrain import blip_pretrain


class BLIPLightningModule(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters({"config": config})
        self.config = config

        self.model = blip_pretrain(
            med_config=self.config.med_config,
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
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train/loss_ita", loss_ita, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/loss_itm", loss_itm, on_step=False, on_epoch=True, batch_size=bs)
        self.log("train/loss_lm", loss_lm, on_step=False, on_epoch=True, batch_size=bs)
        self.log(
            "train/lr",
            float(self.trainer.optimizers[0].param_groups[0]["lr"]),
            on_step=True,
            on_epoch=False,
        )

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.config.init_lr,
            weight_decay=self.config.weight_decay,
        )

        init_lr = self.config.init_lr
        warmup_lr = self.config.warmup_lr
        warmup_steps = self.config.warmup_steps
        min_lr = self.config.min_lr
        decay_rate = self.config.lr_decay_rate

        def lr_lambda(step: int) -> float:
            epoch = self.trainer.current_epoch
            if epoch == 0:
                i = max(step - 1, 0)
                lr = warmup_lr + (init_lr - warmup_lr) * (i / warmup_steps)
            else:
                lr = max(min_lr, init_lr * (decay_rate**epoch))
            return lr / init_lr

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


class BLIPRetrieval(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.save_hyperparameters({"config": config})
        self.config = config

        self.model = blip_pretrain(
            med_config=self.config.med_config,
            image_size=self.config.img_size,
            vit=self.config.vit,
            vit_grad_ckpt=self.config.vit_grad_ckpt,
            vit_ckpt_layer=self.config.vit_ckpt_layer,
            queue_size=self.config.queue_size,
        )

    def on_test_epoch_start(self):
        self._image_feats = []
        self._image_embeds = []
        self._image_ids = []

    def test_step(self, batch, batch_idx):
        image, img_id = batch
        image_feat = self.model.visual_encoder(image)
        image_embed = F.normalize(self.model.vision_proj(image_feat[:, 0, :]), dim=-1)

        self._image_feats.append(image_feat.cpu())
        self._image_embeds.append(image_embed.cpu())
        self._image_ids.append(img_id.cpu())

    @torch.no_grad()
    def on_test_epoch_end(self):
        device = self.device
        dataset = self.trainer.test_dataloaders[0].dataset

        image_feats = torch.cat(self._image_feats).to(device).float()
        image_embeds = torch.cat(self._image_embeds).to(device).float()
        image_ids = torch.cat(self._image_ids).to(device)

        order = image_ids.argsort()
        image_feats = image_feats[order]
        image_embeds = image_embeds[order]

        texts = dataset.text
        text_embeds, text_ids, text_atts = [], [], []
        for i in tqdm(range(0, len(texts), 256)):
            chunk = texts[i : i + 256]
            ti = self.model.tokenizer(
                chunk,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(device)
            to = self.model.text_encoder(
                ti.input_ids,
                attention_mask=ti.attention_mask,
                mode="text",
            )
            te = F.normalize(self.model.text_proj(to.last_hidden_state[:, 0, :]), dim=-1)
            text_embeds.append(te)
            text_ids.append(ti.input_ids)
            text_atts.append(ti.attention_mask)
        text_embeds = torch.cat(text_embeds)
        text_ids = torch.cat(text_ids)
        text_atts = torch.cat(text_atts)
        text_ids[:, 0] = self.model.tokenizer.enc_token_id  # [CLS] → [ENC]

        sims_matrix = image_embeds @ text_embeds.t()  # [N_img, N_txt]
        k = self.config.k_test

        score_i2t = torch.full((len(image_embeds), len(texts)), -100.0, device=device)
        for i, sims in enumerate(tqdm(sims_matrix)):
            topk_sim, topk_idx = sims.topk(k=k, dim=0)
            enc_out = image_feats[i : i + 1].repeat(k, 1, 1)
            enc_att = torch.ones(enc_out.size()[:-1], dtype=torch.long, device=device)
            out = self.model.text_encoder(
                text_ids[topk_idx],
                attention_mask=text_atts[topk_idx],
                encoder_hidden_states=enc_out,
                encoder_attention_mask=enc_att,
                return_dict=True,
            )
            score = self.model.itm_head(out.last_hidden_state[:, 0, :])[:, 1]
            score_i2t[i, topk_idx] = score + topk_sim

        score_t2i = torch.full((len(texts), len(image_embeds)), -100.0, device=device)
        sims_t = sims_matrix.t()
        for i, sims in enumerate(tqdm(sims_t)):
            topk_sim, topk_idx = sims.topk(k=k, dim=0)
            enc_out = image_feats[topk_idx]
            enc_att = torch.ones(enc_out.size()[:-1], dtype=torch.long, device=device)
            out = self.model.text_encoder(
                text_ids[i : i + 1].repeat(k, 1),
                attention_mask=text_atts[i : i + 1].repeat(k, 1),
                encoder_hidden_states=enc_out,
                encoder_attention_mask=enc_att,
                return_dict=True,
            )
            score = self.model.itm_head(out.last_hidden_state[:, 0, :])[:, 1]
            score_t2i[i, topk_idx] = score + topk_sim

        result = self._itm_eval(
            score_i2t.cpu().numpy(),
            score_t2i.cpu().numpy(),
            dataset.txt2img,
            dataset.img2txt,
        )
        boot = self._bootstrap_retrieval(
            score_i2t.cpu().numpy(),
            score_t2i.cpu().numpy(),
            dataset.txt2img,
            dataset.img2txt,
            n_boot=self.config.get("n_boot", 1000),
        )
        self.log_dict({f"test/{k_}": v for k_, v in {**result, **boot}.items()})

    @staticmethod
    def _itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
        tr1 = 100.0 * np.mean(ranks < 1)
        tr5 = 100.0 * np.mean(ranks < 5)
        tr10 = 100.0 * np.mean(ranks < 10)

        ranks = np.zeros(scores_t2i.shape[0])
        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]
        ir1 = 100.0 * np.mean(ranks < 1)
        ir5 = 100.0 * np.mean(ranks < 5)
        ir10 = 100.0 * np.mean(ranks < 10)

        return {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
        }

    @staticmethod
    def _bootstrap_retrieval(scores_i2t, scores_t2i, txt2img, img2txt, n_boot=1000, seed=42):
        """Bootstrap CI для r@1/5/10 (i2t и t2i)."""
        rng = np.random.default_rng(seed)
        n_i = scores_i2t.shape[0]
        n_t = scores_t2i.shape[0]

        # заранее считаем rank каждого запроса
        rank_i2t = np.empty(n_i)
        for idx, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            r = 1e20
            for i in img2txt[idx]:
                tmp = np.where(inds == i)[0][0]
                if tmp < r:
                    r = tmp
            rank_i2t[idx] = r

        rank_t2i = np.empty(n_t)
        for idx, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            rank_t2i[idx] = np.where(inds == txt2img[idx])[0][0]

        stats = {k: [] for k in ["txt_r1", "txt_r5", "txt_r10", "img_r1", "img_r5", "img_r10"]}
        for _ in range(n_boot):
            bi = rng.integers(0, n_i, n_i)
            bt = rng.integers(0, n_t, n_t)
            ri = rank_i2t[bi]
            rt = rank_t2i[bt]
            stats["txt_r1"].append(100.0 * np.mean(ri < 1))
            stats["txt_r5"].append(100.0 * np.mean(ri < 5))
            stats["txt_r10"].append(100.0 * np.mean(ri < 10))
            stats["img_r1"].append(100.0 * np.mean(rt < 1))
            stats["img_r5"].append(100.0 * np.mean(rt < 5))
            stats["img_r10"].append(100.0 * np.mean(rt < 10))

        out = {}
        for k, v in stats.items():
            v = np.asarray(v)
            out[f"{k}_ci_lo"] = float(np.percentile(v, 2.5))
            out[f"{k}_ci_hi"] = float(np.percentile(v, 97.5))
            out[f"{k}_std"] = float(v.std())
        return out


class BLIPCaption(LightningModule):
    def __init__(self, config, test_file=None):
        super().__init__()
        self.save_hyperparameters({"config": config})
        self.config = config
        self.model = blip_pretrain(
            med_config=self.config.med_config,
            image_size=self.config.img_size,
            vit=self.config.vit,
            vit_grad_ckpt=self.config.vit_grad_ckpt,
            vit_ckpt_layer=self.config.vit_ckpt_layer,
            queue_size=self.config.queue_size,
        )

        self._enable_generate(self.model.text_decoder)  # ← ЧАСТЬ 1: вызов
        self._refs = build_caption_refs(test_file) if test_file else None

    @staticmethod
    def _enable_generate(td):  # ← ЧАСТЬ 2: сам хак
        """Подмешивает GenerationMixin в BertLMHeadModel (transformers>=4.50)."""
        if hasattr(td, "generate"):
            return  # старая версия transformers — ничего не делаем

        from transformers import GenerationConfig
        from transformers.generation import GenerationMixin

        new_cls = type(
            td.__class__.__name__ + "WithGenerate",
            (td.__class__, GenerationMixin),
            {"__module__": td.__class__.__module__},
        )
        td.__class__ = new_cls

        if getattr(td, "generation_config", None) is None:
            td.generation_config = GenerationConfig.from_model_config(td.config)

    def on_test_epoch_start(self):
        self._preds = {}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        image, img_id = batch
        image_embeds = self.model.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image.device)

        input_ids = (
            torch.tensor([[self.model.tokenizer.bos_token_id, self.model.tokenizer.enc_token_id]])
            .repeat(image.size(0), 1)
            .to(image.device)
        )

        outputs = self.model.text_decoder.generate(
            input_ids=input_ids,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            num_beams=self.config.num_beams,
            max_length=self.config.max_length,
            min_length=self.config.min_length,
            eos_token_id=self.model.tokenizer.sep_token_id,
            pad_token_id=self.model.tokenizer.pad_token_id,
            repetition_penalty=1.0,
            length_penalty=1.0,
            early_stopping=True,
            return_dict_in_generate=False,
        )
        captions = self.model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for i, c in zip(img_id.tolist(), captions, strict=False):
            self._preds[str(i)] = [c.strip()]

    def on_test_epoch_end(self):
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider

        assert self._refs is not None, "refs not loaded: pass test_file to __init__"
        preds = self._preds
        refs = {k: v for k, v in self._refs.items() if k in preds}

        bleu_score, _ = Bleu(4).compute_score(refs, preds)
        cider_score, _ = Cider().compute_score(refs, preds)

        self.log_dict(
            {
                "test/Bleu_1": float(bleu_score[0]) * 100,
                "test/Bleu_2": float(bleu_score[1]) * 100,
                "test/Bleu_3": float(bleu_score[2]) * 100,
                "test/Bleu_4": float(bleu_score[3]) * 100,
                "test/CIDEr": float(cider_score) * 100,
            }
        )
