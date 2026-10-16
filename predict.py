"""Write prediction masks: single-channel PNG, same size as the source image, values {0, 255},
named <image_id>__<prompt_slug>.png (e.g. 123__segment_crack.png).

    # deliverable: every test image with its own class prompt
    python predict.py --run runs/clipseg_ft-20261016-120000 --split test
    # every test image with every class prompt
    python predict.py --run runs/... --split test --all-prompts
    # arbitrary images / prompts
    python predict.py --run runs/... --images some_dir/ --prompt "segment crack" --prompt "segment drywall seam"
"""
import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from drywall_seg.config import load_config
from drywall_seg.data import PromptBank, read_rgb
from drywall_seg.inference import load_model, predict_probs
from drywall_seg.utils import get_device

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def slug(prompt):
    return re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="training run directory")
    src.add_argument("--config", help="config file (zero-shot unless --ckpt)")
    p.add_argument("--ckpt")
    what = p.add_mutually_exclusive_group()
    what.add_argument("--split", default="test", help="split from index.csv (default: test)")
    what.add_argument("--images", help="image file or directory")
    p.add_argument("--prompt", action="append", help="prompt(s); default for --split: each image's class prompt")
    p.add_argument("--all-prompts", action="store_true", help="with --split: every class's primary prompt")
    p.add_argument("--threshold", type=float, help="override the tuned per-class threshold")
    p.add_argument("--out", help="output directory")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args()

    if args.run:
        cfg, ckpt = load_config(Path(args.run) / "config.yaml", args.set), Path(args.run) / "best.pt"
    else:
        cfg, ckpt = load_config(args.config, args.set), args.ckpt
    device = get_device(cfg.get("device", "auto"))
    amp = bool(cfg.train.get("amp", False))
    model, thr = load_model(cfg, ckpt, device)
    thr = thr or {}
    bank = PromptBank.from_config(cfg.prompts)

    if args.images:
        if not args.prompt:
            p.error("--images needs at least one --prompt")
        path = Path(args.images)
        files = sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS) if path.is_dir() else [path]
        jobs = [(f.stem, f, args.prompt) for f in files]
        out = Path(args.out or "predictions")
    else:
        df = pd.read_csv(cfg.data.index_csv)
        df = df[df.split == args.split]
        jobs = [(r.image_id, Path(r.file),
                 args.prompt or ([bank.primary[c] for c in bank.classes] if args.all_prompts else [bank.primary[r.dataset]]))
                for r in df.itertuples()]
        out = Path(args.out or (Path(args.run) if args.run else Path(cfg.output_dir)) / f"predictions_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    def threshold_for(prompt):
        if args.threshold is not None:
            return args.threshold
        return thr.get(bank.class_of(prompt), 0.5)     # unknown prompt or zero-shot → 0.5

    t_model, n = 0.0, 0
    for image_id, file, prompts in tqdm(jobs, desc="predict"):
        img = read_rgb(file)
        t0 = time.perf_counter()
        probs = predict_probs(model, img, prompts, device, amp)
        t_model += time.perf_counter() - t0
        n += len(prompts)
        for prompt, prob in zip(prompts, probs):
            mask = (prob > threshold_for(prompt)).astype(np.uint8) * 255
            cv2.imwrite(str(out / f"{image_id}__{slug(prompt)}.png"), mask)
    print(f"wrote {n} masks to {out} | {1000 * t_model / max(n, 1):.1f} ms per (image, prompt)")


if __name__ == "__main__":
    main()
