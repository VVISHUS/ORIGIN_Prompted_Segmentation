"""Paired image/mask augmentations (numpy + OpenCV, no extra dependency).

Each config key is the probability of applying that transform. Geometric transforms change the
image AND the mask; photometric ones change the image only.
"""
import random

import cv2
import numpy as np


class Augment:
    OPS = ("hflip", "vflip", "rot90", "scale_crop", "brightness_contrast", "hue_saturation", "blur", "noise")

    def __init__(self, probs):
        unknown = set(probs) - set(self.OPS)
        if unknown:
            raise ValueError(f"unknown augmentations {sorted(unknown)}; available: {self.OPS}")
        self.p = dict(probs)

    def _on(self, name):
        return random.random() < self.p.get(name, 0.0)

    def __call__(self, img, mask):
        if self._on("hflip"):
            img, mask = img[:, ::-1], mask[:, ::-1]
        if self._on("vflip"):
            img, mask = img[::-1], mask[::-1]
        if self._on("rot90"):
            k = random.randint(1, 3)
            img, mask = np.rot90(img, k), np.rot90(mask, k)
        img, mask = np.ascontiguousarray(img), np.ascontiguousarray(mask)
        if self._on("scale_crop"):
            img, mask = _scale_crop(img, mask)
        if self._on("brightness_contrast"):
            img = cv2.convertScaleAbs(img, alpha=random.uniform(0.75, 1.25), beta=random.uniform(-25, 25))
        if self._on("hue_saturation"):
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-8, 8)) % 180
            hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        if self._on("blur"):
            k = random.choice((3, 5))
            img = cv2.GaussianBlur(img, (k, k), 0)
        if self._on("noise"):
            noise = np.random.normal(0, random.uniform(3, 10), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img, mask


def _scale_crop(img, mask, min_scale=0.6):
    """Random crop of 60–100% of the side, resized back to the original size."""
    h, w = mask.shape
    s = random.uniform(min_scale, 1.0)
    ch, cw = int(h * s), int(w * s)
    y, x = random.randint(0, h - ch), random.randint(0, w - cw)
    img = cv2.resize(img[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_NEAREST)
    return img, mask
