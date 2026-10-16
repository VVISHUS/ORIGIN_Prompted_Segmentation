"""Prompt sampling, dataset and loaders.

Every sample is (image, prompt, target mask):
  train: the prompt is drawn from the class's prompt list. With probability `negative_prob` the prompt
         belongs to the OTHER class and the target is empty, which forces the model to read the prompt.
  eval:  the class's primary prompt. Every image is scored with its own prompt (positive) and, if
         enabled, with each other class's prompt (negative, empty target) to measure prompt leakage.
"""
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augment import Augment
from .utils import seed_worker


class PromptBank:
    def __init__(self, classes, primary_prob=0.5):
        self.classes = list(classes)
        self.primary = {c: v["primary"] for c, v in classes.items()}
        self.all = {c: [v["primary"], *v.get("variants", [])] for c, v in classes.items()}
        self.primary_prob = primary_prob
        self._class_of = {p.lower(): c for c, ps in self.all.items() for p in ps}

    @classmethod
    def from_config(cls, cfg_prompts):
        classes = {c: dict(v) for c, v in cfg_prompts.classes.items()}
        return cls(classes, cfg_prompts.get("primary_prob", 0.5))

    def sample(self, cls):
        return self.primary[cls] if random.random() < self.primary_prob else random.choice(self.all[cls])

    def other(self, cls):
        return random.choice([c for c in self.classes if c != cls])

    def class_of(self, prompt):
        return self._class_of.get(prompt.lower())


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(img, size, mean, std):
    """RGB uint8 HxWx3 → normalised float tensor 3xSxS."""
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
    x = cv2.resize(img, (size, size), interpolation=interp).astype(np.float32) / 255.0
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class SegDataset(Dataset):
    def __init__(self, df, bank, size, mean, std, train, augment=None, negative_prob=0.0, eval_negatives=False):
        self.rows = df[["file", "mask", "dataset", "image_id"]].to_dict("records")
        self.bank, self.size, self.mean, self.std = bank, size, mean, std
        self.train, self.augment, self.negative_prob = train, augment, negative_prob
        if train:
            self.items = [(i, None, None) for i in range(len(self.rows))]
        else:
            self.items = [(i, r["dataset"], False) for i, r in enumerate(self.rows)]
            if eval_negatives:
                self.items += [(i, c, True) for i, r in enumerate(self.rows) for c in bank.classes if c != r["dataset"]]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, prompt_class, negative = self.items[k]
        row = self.rows[i]
        img = read_rgb(row["file"])
        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        if self.train:
            if self.augment:
                img, mask = self.augment(img, mask)
            negative = random.random() < self.negative_prob
            prompt_class = self.bank.other(row["dataset"]) if negative else row["dataset"]
            prompt = self.bank.sample(prompt_class)
        else:
            prompt = self.bank.primary[prompt_class]
        if negative:
            mask = np.zeros_like(mask)
        # INTER_AREA keeps thin cracks as soft (fractional) targets instead of dropping them
        target = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        item = dict(
            pixel_values=preprocess(img, self.size, self.mean, self.std),
            target=torch.from_numpy(target)[None],
            prompt=prompt, prompt_class=prompt_class, image_class=row["dataset"],
            is_negative=negative, image_id=row["image_id"],
        )
        if not self.train:
            item["full_target"] = torch.from_numpy(mask > 127)[None]   # original resolution, for metrics
        return item


def collate(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        same = torch.is_tensor(vals[0]) and all(v.shape == vals[0].shape for v in vals)
        out[k] = torch.stack(vals) if same else vals
    return out


def make_loader(cfg, df, split, model, bank, train):
    sub = df[df.split == split]
    n = cfg.eval.get("max_images")
    if not train and n:
        sub = pd.concat([g.sample(min(len(g), int(n)), random_state=cfg.seed) for _, g in sub.groupby("dataset")])
    sub = sub.reset_index(drop=True)
    ds = SegDataset(
        sub, bank, int(cfg.data.get("image_size") or model.input_size), model.image_mean, model.image_std,
        train=train,
        augment=Augment(cfg.data.augment) if train and cfg.data.get("augment") else None,
        negative_prob=cfg.data.get("negative_prob", 0.0) if train else 0.0,
        eval_negatives=not train and cfg.eval.get("negatives", True),
    )
    gen = torch.Generator().manual_seed(int(cfg.seed))
    sampler = None
    if train and cfg.data.get("balance_classes", True):
        weights = 1.0 / sub.dataset.map(sub.dataset.value_counts())     # each class gets equal total weight
        sampler = WeightedRandomSampler(torch.tensor(weights.to_numpy(), dtype=torch.double),
                                        num_samples=int(cfg.data.get("samples_per_epoch") or len(sub)),
                                        replacement=True, generator=gen)
    workers = int(cfg.data.get("num_workers", 0))
    return DataLoader(
        ds, batch_size=int(cfg.train.batch_size if train else cfg.eval.batch_size),
        shuffle=train and sampler is None, sampler=sampler, drop_last=train,
        num_workers=workers, persistent_workers=workers > 0, worker_init_fn=seed_worker, generator=gen,
        pin_memory=torch.cuda.is_available(), collate_fn=collate,
    )
