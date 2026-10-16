"""Root-cause analysis of an evaluation: where and why IoU is lost.

    python analyze_errors.py --eval-dir runs/<run>/eval_test

Reads <eval-dir>/per_image.csv (written by evaluate.py) and the GT masks; no model or GPU needed.
Writes <eval-dir>/rca.md and rca_per_image.csv.

What it measures, per positive (image, own-prompt) pair:
  width_px      mean GT stroke width ≈ 2·area / perimeter (thin cracks vs coarse polygons)
  pred_vs_gt    predicted area / GT area   (>1 over-segmentation, <1 under-segmentation)
  error_type    miss | under | over | shape   (what kind of mistake dominates)
  oracle_iou    IoU of the GT itself after a round trip through the model's output resolution
                (resize to 352 and back): the best IoU any 352-px model can reach on that image
  oracle_grid   same round trip through the 22×22 ViT token grid CLIPSeg's features live on:
                a rough ceiling for detail the frozen backbone can represent
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _round_trip_iou(gt, res):
    small = cv2.resize(gt.astype(np.float32), (res, res), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, gt.shape[::-1], interpolation=cv2.INTER_LINEAR) > 0.5
    return float((back & gt).sum() / max((back | gt).sum(), 1))


def gt_stats(mask_path, model_res, grid_res):
    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127
    area = int(gt.sum())
    if area == 0:
        return dict(width_px=0.0, n_components=0, oracle_iou=1.0, oracle_grid=1.0)
    cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in cnts)
    return dict(width_px=2 * area / max(perim, 1), n_components=len(cnts),
                oracle_iou=_round_trip_iou(gt, model_res), oracle_grid=_round_trip_iou(gt, grid_res))


def error_type(r):
    if r.iou < 0.1 and r.pred_vs_gt < 0.3:
        return "miss"
    if r.pred_vs_gt > 1.5:
        return "over"
    if r.pred_vs_gt < 0.67:
        return "under"
    return "shape"          # area roughly right, placement/outline wrong


def md(df, floats=3):
    df = df.reset_index()
    fmt = lambda v: f"{v:.{floats}f}" if isinstance(v, (float, np.floating)) else str(v)
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def table(pos, by, bins=None, labels=None):
    key = pd.cut(pos[by], bins=bins, labels=labels) if bins is not None else pos[by]
    return (pos.groupby(["prompt_class", key], observed=True)
               .agg(n=("iou", "size"), iou=("iou", "mean"), oracle_iou=("oracle_iou", "mean"),
                    oracle_grid=("oracle_grid", "mean"), pred_vs_gt=("pred_vs_gt", "median")))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--model-res", type=int, default=352, help="model output resolution (CLIPSeg: 352)")
    p.add_argument("--grid-res", type=int, default=22, help="feature grid (CLIPSeg ViT-B/16 at 352: 22)")
    args = p.parse_args()
    out = Path(args.eval_dir)

    res = pd.read_csv(out / "per_image.csv")
    pos = res[~res.is_negative].copy()
    stats = [gt_stats(m, args.model_res, args.grid_res) for m in tqdm(pos["mask"], desc="GT stats")]
    pos = pos.join(pd.DataFrame(stats, index=pos.index))
    pos["pred_vs_gt"] = pos.fg / pos.fg_frac.clip(lower=1e-6)
    pos["error_type"] = pos.apply(error_type, axis=1)
    pos.to_csv(out / "rca_per_image.csv", index=False)

    overall = pos.groupby("prompt_class").agg(
        n=("iou", "size"), iou=("iou", "mean"), oracle_iou=("oracle_iou", "mean"),
        oracle_grid=("oracle_grid", "mean"), pred_vs_gt=("pred_vs_gt", "median"), width_px=("width_px", "median"))
    errors = pd.crosstab(pos.prompt_class, pos.error_type, normalize="index")
    errors_iou = pos.pivot_table(index="prompt_class", columns="error_type", values="iou", aggfunc="mean")
    width = table(pos, "width_px", [0, 5, 10, 20, 40, 1e9], ["<5", "5-10", "10-20", "20-40", ">40"])
    size = table(pos, "fg_frac", [0, 0.01, 0.03, 0.1, 0.3, 1.0], ["<1%", "1-3%", "3-10%", "10-30%", ">30%"])
    comps = table(pos.assign(n_obj=pos.n_components.clip(upper=4)), "n_obj")
    worst = pos.nsmallest(15, "iou")[["prompt_class", "image_id", "iou", "oracle_iou", "width_px",
                                      "pred_vs_gt", "error_type", "fg_frac"]]

    report = "\n\n".join([
        "# Error analysis",
        "## Overall (oracle_iou / oracle_grid = ceilings at 352 px / 22×22 grid; pred_vs_gt > 1 = over-segmenting)",
        md(overall),
        "## Error type share (rows sum to 1)", md(errors),
        "## Mean IoU per error type", md(errors_iou),
        "## IoU by GT stroke width (px at 640)", md(width),
        "## IoU by GT size (share of image)", md(size),
        "## IoU by number of GT objects (4 = 4+)", md(comps),
        "## 15 worst images (inspect these in the panels / with predict.py)", md(worst),
    ])
    (out / "rca.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
