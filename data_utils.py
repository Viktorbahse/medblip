import json
import os
import random
import re

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from Original_BLIP.randaugment import RandomAugment


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        " ",
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        " ",
        caption,
    )
    caption = caption.rstrip("\n")
    caption = caption.strip(" ")

    caption_words = caption.split(" ")
    if len(caption_words) > max_words:
        caption = " ".join(caption_words[:max_words])

    return caption


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root=None):
        self.ann_pretrain = []
        for f in ann_file:
            print("loading " + f)
            with open(f) as fh:
                ann = json.load(fh)
            self.ann_pretrain += ann

        self.annotation = self.ann_pretrain
        self.transform = transform
        self.image_root = image_root

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = ann["image"]
        if self.image_root is not None:
            image_path = os.path.join(self.image_root, image_path)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = pre_caption(random.choice(ann["captions"]), 30)

        return image, caption


def create_dataset(config, min_scale=0.5):
    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config.img_size, scale=(min_scale, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2,
                5,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Brightness",
                    "Sharpness",
                    "Equalize",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    """
    transform_test = transforms.Compose(
        [
            transforms.Resize(
                (config.img_size, config.img_size), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    """
    dataset = pretrain_dataset(config.train_file, transform_train, image_root=config.img_root)
    return dataset


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns, strict=False
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
