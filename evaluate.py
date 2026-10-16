"""Evaluate a trained run (or the pretrained model zero-shot) on a split.

    python evaluate.py --run runs/clipseg_ft-20261016-120000            # fine-tuned, test split
    python evaluate.py --config configs/clipseg_ft.yaml                 # zero-shot baseline

Thresholds come from the run's checkpoint (tuned on val during training); for zero-shot, or with
--retune, they are tuned on val here. Never tuned on the split being reported.

Writes to <out>/: results.md, summary.json, per_image.csv, buckets.csv, examples_<class>.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from drywall_seg.config import load_config
from drywall_seg.data import PromptBank, make_loader, read_rgb
from drywall_seg.evaluation import apply_thresholds, bucket_table, choose_thresholds, evaluate, summarize
from drywall_seg.inference import load_model, predict_probs, time_inference
from drywall_seg.utils import count_params, get_device, model_size_mb, save_json, seed_everything
from drywall_seg.viz import save_panels


def md_table(df, floats=3):
    fmt = lambda v: f"{v:.{floats}f}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save_examples(model, res, thr, device, amp, out):
    """Best / median / worst test image per class: image | GT | prediction | error map."""
    for cls, g in res[~res.is_negative].groupby("prompt_class"):
        g = g.sort_values("iou").reset_index(drop=True)
        picks = {"worst": g.iloc[0], "median": g.iloc[len(g) // 2], "best": g.iloc[-1]}
        items = []
        for tag, r in picks.items():       # r is a Series: use [] (r.mask would be Series.mask)
            img = read_rgb(r["file"])
            gt = cv2.imread(r["mask"], cv2.IMREAD_GRAYSCALE) > 127
            pred = predict_probs(model, img, [r["prompt"]], device, amp)[0] > thr[cls]
            items.append(dict(img=img, gt=gt, pred=pred, title=f"{tag}\nIoU {r['iou']:.2f}\n{r['image_id'][:28]}"))
        save_panels(items, out / f"examples_{cls}.png", title=f'prompt: "{g.prompt.iloc[0]}"')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="training run directory (config.yaml + best.pt)")
    src.add_argument("--config", help="config file; evaluates the pretrained model unless --ckpt is given")
    p.add_argument("--ckpt", help="checkpoint to load with --config")
    p.add_argument("--split", default="test")
    p.add_argument("--retune", action="store_true", help="re-tune thresholds on val")
    p.add_argument("--out", help="output directory")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = p.parse_args()

    if args.run:
        run = Path(args.run)
        cfg, ckpt = load_config(run / "config.yaml", args.set), run / "best.pt"
        out = Path(args.out or run / f"eval_{args.split}")
    else:
        cfg, ckpt = load_config(args.config, args.set), args.ckpt
        tag = "ckpt" if ckpt else "zeroshot"
        out = Path(args.out or Path(cfg.output_dir) / f"{cfg.experiment}-{tag}" / f"eval_{args.split}")
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg.seed))
    device = get_device(cfg.get("device", "auto"))
    amp = bool(cfg.train.get("amp", False))
    model, thr = load_model(cfg, ckpt, device)
    df = pd.read_csv(cfg.data.index_csv)
    bank = PromptBank.from_config(cfg.prompts)
    thresholds = list(cfg.eval.thresholds)

    if thr is None or args.retune:
        print("tuning thresholds on val ...")
        val = evaluate(model, make_loader(cfg, df, "val", model, bank, train=False), device, thresholds, amp)
        thr = choose_thresholds(val, thresholds)
    print("thresholds:", thr)

    raw = evaluate(model, make_loader(cfg, df, args.split, model, bank, train=False), device, thresholds, amp)
    res = apply_thresholds(raw, thr)
    meta = df.rename(columns={"dataset": "image_class"})[
        ["image_class", "image_id", "file", "mask", "fg_frac", "brightness_bucket", "contrast_bucket", "sharpness_bucket"]]
    res = res.merge(meta, on=["image_class", "image_id"], how="left")
    summary = summarize(res)
    buckets = bucket_table(res)

    # runtime & footprint
    pos = res[~res.is_negative].sample(frac=1.0, random_state=int(cfg.seed)).head(int(cfg.eval.get("timing_images", 50)))
    ms, ms_std = time_inference(model, [read_rgb(f) for f in pos.file], list(pos.prompt), device, amp)
    total_p, train_p = count_params(model)
    footprint = dict(device=str(device), ms_per_image=ms, ms_per_image_std=ms_std, params_total=total_p,
                     params_finetuned=train_p if ckpt else 0, model_size_mb_fp32=model_size_mb(model),
                     checkpoint_mb=Path(ckpt).stat().st_size / 2**20 if ckpt else 0.0)

    res.to_csv(out / "per_image.csv", index=False)
    buckets.to_csv(out / "buckets.csv", index=False)
    save_json(dict(split=args.split, thresholds=thr, **summary, footprint=footprint), out / "summary.json")
    save_examples(model, res, thr, device, amp, out)

    main_tbl = pd.DataFrame([
        {"prompt": bank.primary[c], "n": v["n"], "IoU": v["iou"], "IoU std": v["iou_std"], "IoU p10": v["iou_p10"],
         "Dice": v["dice"], "Dice std": v["dice_std"], "threshold": v["threshold"],
         "leak (other prompt fg)": v["neg_leak"]}
        for c, v in summary["per_class"].items()])
    level = {"dark": "low", "blurry": "low", "bright": "high", "sharp": "high"}
    buckets["level"] = buckets.bucket.map(lambda b: level.get(b, b))
    bucket_pivot = buckets.pivot_table(index=["class", "factor"], columns="level", values="iou")
    bucket_pivot = bucket_pivot.reindex(columns=["low", "mid", "high"])
    bucket_pivot["spread (max-min)"] = bucket_pivot.max(axis=1) - bucket_pivot.min(axis=1)
    bucket_pivot = bucket_pivot.reset_index()
    report = "\n\n".join([
        f"## Results on `{args.split}`",
        md_table(main_tbl),
        f"**mIoU {summary['mIoU']:.3f} | mDice {summary['mDice']:.3f}**",
        "### IoU by scene bucket (consistency)",
        md_table(bucket_pivot),
        "### Runtime & footprint",
        md_table(pd.DataFrame([footprint])),
    ])
    (out / "results.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()
