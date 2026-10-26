import gc
import json
import os

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from data_utils import pre_caption
from model import BLIPCaption  # содержит _enable_generate для text_decoder

# ---------------------------------------------------------------------------
# Dataset / transform
# ---------------------------------------------------------------------------


class CapFiltDataset(Dataset):
    def __init__(self, ann, transform, image_root=None):
        self.ann = ann
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        ann = self.ann[idx]
        path = ann["image"]
        if self.image_root is not None:
            path = os.path.join(self.image_root, path)
        img = Image.open(path).convert("RGB")
        return self.transform(img), idx


def build_eval_transform(img_size):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ]
    )


# ---------------------------------------------------------------------------
# Captioner
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_captions(model, images, num_generated, num_beams, max_length, min_length):
    """Генерирует num_generated описаний на изображение. Возвращает list[list[str]]."""
    blip = model.model
    device = images.device

    image_embeds = blip.visual_encoder(images)  # [B, 197, D]
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)

    input_ids = (
        torch.tensor([[blip.tokenizer.bos_token_id, blip.tokenizer.enc_token_id]])
        .repeat(images.size(0), 1)
        .to(device)
    )

    beams = max(num_beams, num_generated)  # требование HF: num_beams >= num_return_sequences
    outputs = blip.text_decoder.generate(
        input_ids=input_ids,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        num_beams=beams,
        num_return_sequences=num_generated,
        max_length=max_length,
        min_length=min_length,
        eos_token_id=blip.tokenizer.sep_token_id,
        pad_token_id=blip.tokenizer.pad_token_id,
        repetition_penalty=1.0,
        length_penalty=1.0,
        early_stopping=True,
        return_dict_in_generate=False,
    )
    captions = blip.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    B = images.size(0)
    return [captions[i * num_generated : (i + 1) * num_generated] for i in range(B)]


# ---------------------------------------------------------------------------
# Filter (ITM)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_itm_scores(model, images, captions_per_image, max_length=35):
    """
    Считает ITM-скор для каждой пары (image, caption).

    captions_per_image : list[list[str]] длины B
    Возвращает list[list[float]] той же структуры.
    """
    blip = model.model
    device = images.device
    B = images.size(0)

    image_feats = blip.visual_encoder(images)  # [B, 197, D]

    # Плоский список caption'ов + длины групп
    caps_flat, lens = [], []
    for i in range(B):
        caps = [pre_caption(c, 30) for c in captions_per_image[i]]
        caps_flat.extend(caps)
        lens.append(len(caps))

    text = blip.tokenizer(
        caps_flat,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    # В BLIP при ITM первый токен должен быть [ENC], а не [CLS]
    text.input_ids[:, 0] = blip.tokenizer.enc_token_id

    # Разворачиваем image_feats так, чтобы каждому caption соответствовал свой image
    enc_out = torch.repeat_interleave(image_feats, torch.tensor(lens, device=device), dim=0)
    enc_att = torch.ones(enc_out.size()[:-1], dtype=torch.long, device=device)

    out = blip.text_encoder(
        text.input_ids,
        attention_mask=text.attention_mask,
        encoder_hidden_states=enc_out,
        encoder_attention_mask=enc_att,
        return_dict=True,
    )
    scores = blip.itm_head(out.last_hidden_state[:, 0, :])[:, 1]  # P(match)

    # Разбиваем обратно по изображениям
    result, offset = [], 0
    for i in range(B):
        result.append(scores[offset : offset + lens[i]].cpu().tolist())
        offset += lens[i]
    return result


# ---------------------------------------------------------------------------
# Пайплайн для одного файла
# ---------------------------------------------------------------------------


def process_file(model, file_path, image_root, transform, config):
    with open(file_path) as f:
        data = json.load(f)

    data = data[:10]

    dataset = CapFiltDataset(data, transform, image_root=image_root)
    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 8),
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    max_captions = config.get("max_captions", 4)
    num_generated = config.get("num_generated", 2)
    num_beams = config.get("num_beams", config.model.get("num_beams", 3))
    max_length = config.get("max_length", config.model.get("max_length", 20))
    min_length = config.get("min_length", config.model.get("min_length", 5))

    device = next(model.parameters()).device
    new_data = [None] * len(data)
    total_gen = 0
    kept_gen = 0

    pbar = tqdm(loader, desc=os.path.basename(file_path))
    for images, indices in pbar:
        images = images.to(device, non_blocking=True)
        idx_list = indices.tolist()

        originals = [data[i]["captions"] for i in idx_list]

        # 1) Captioner
        generated = generate_captions(
            model, images, num_generated, num_beams, max_length, min_length
        )

        # 2) Комбинируем и считаем ITM
        combined = [originals[j] + generated[j] for j in range(len(idx_list))]
        scores = compute_itm_scores(model, images, combined)

        # 3) Фильтр: top-max_captions по ITM
        for j, idx in enumerate(idx_list):
            all_caps = combined[j]
            all_scores = scores[j]
            n_orig = len(originals[j])

            order = sorted(range(len(all_caps)), key=lambda k: all_scores[k], reverse=True)
            kept_idx = order[:max_captions]
            kept_caps = [all_caps[k] for k in kept_idx]

            kept_gen += sum(1 for k in kept_idx if k >= n_orig)
            total_gen += num_generated

            new_data[idx] = {"image": data[idx]["image"], "captions": kept_caps}

        pbar.set_postfix({"kept_gen": f"{kept_gen}/{total_gen}"})

    # 4) Сохранение
    out_path = file_path.rsplit(".", 1)[0] + "_capfilt.json"
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    ratio = kept_gen / max(total_gen, 1)
    print(
        f"[{file_path}] kept {kept_gen}/{total_gen} generated captions "
        f"({100 * ratio:.2f}%) → {out_path}"
    )
    return ratio


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(config: DictConfig):
    pl.seed_everything(config.get("seed", 42), workers=True)
    torch.backends.cudnn.benchmark = True

    ckpt_path = config.get("ckpt_path", None)
    if ckpt_path is None:
        raise ValueError("capfilt.ckpt_path is required in config.yaml")

    print(f"Loading model from {ckpt_path}")
    module = BLIPCaption.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        config=config.model,
        weights_only=False,
    )
    module.eval()

    if config.get("device", None):
        device = torch.device(config.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    module = module.to(device)

    transform = build_eval_transform(config.model.img_size)
    image_root = config.get("image_root", None)

    results = {}
    for fp in config.files:
        results[fp] = process_file(module, fp, image_root, transform, config)

    print("\nSummary (fraction of generated captions kept):")
    for fp, ratio in results.items():
        print(f"  {os.path.basename(fp):<24s} {100 * ratio:6.2f}%")

    del module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    config = config.capfilt
    main(config)
