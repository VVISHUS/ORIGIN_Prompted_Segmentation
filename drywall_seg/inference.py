import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_config
from .data import preprocess
from .models import build_model


def load_model(cfg, ckpt=None, device="cpu"):
    """Build from config (pretrained weights) and, if given, load fine-tuned weights + tuned thresholds."""
    model = build_model(cfg.model)
    thresholds = None
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        _, unexpected = model.load_state_dict(state["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"checkpoint does not match the model: unexpected keys {unexpected[:5]}")
        thresholds = state.get("thresholds")
    return model.to(device).eval(), thresholds


def load_run(run_dir, device="cpu", overrides=()):
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml", overrides)
    model, thresholds = load_model(cfg, run_dir / "best.pt", device)
    return cfg, model, thresholds


@torch.no_grad()
def predict_probs(model, img, prompts, device, amp=False):
    """One RGB image, several prompts → probabilities [len(prompts), H, W] at the image's size."""
    x = preprocess(img, model.input_size, model.image_mean, model.image_std)[None].to(device)
    x = x.repeat(len(prompts), 1, 1, 1)
    with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
        logits = model(x, list(prompts)).float()
    probs = F.interpolate(logits, size=img.shape[:2], mode="bilinear", align_corners=False).sigmoid()
    return probs[:, 0].cpu().numpy()


def time_inference(model, images, prompts, device, amp=False, warmup=5):
    """Mean / std ms per image for preprocess → forward → upsample → threshold (disk I/O excluded)."""
    def run(img, prompt):
        (predict_probs(model, img, [prompt], device, amp) > 0.5).astype(np.uint8)
        if device.type == "cuda":
            torch.cuda.synchronize()

    for img, prompt in list(zip(images, prompts))[:warmup]:
        run(img, prompt)
    times = []
    for img, prompt in zip(images, prompts):
        t0 = time.perf_counter()
        run(img, prompt)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))
