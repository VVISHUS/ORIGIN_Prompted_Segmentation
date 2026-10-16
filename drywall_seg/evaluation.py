"""Per-image IoU / Dice at several thresholds, threshold tuning and summaries.

Metrics are computed at the ORIGINAL image resolution (logits are upsampled), per image, then
averaged per prompt class. An image where both GT and prediction are empty scores 1.0.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _col(kind, t):
    return f"{kind}@{t:.2f}"


@torch.no_grad()
def evaluate(model, loader, device, thresholds, amp=False):
    """One row per (image, prompt) with iou/dice/fg columns for every threshold."""
    model.eval()
    T = torch.tensor(thresholds, device=device).view(-1, 1, 1)
    rows = []
    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["pixel_values"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x, batch["prompt"]).float()
        for j in range(len(x)):
            gt = batch["full_target"][j][0].to(device)
            prob = F.interpolate(logits[j:j + 1], size=gt.shape, mode="bilinear", align_corners=False).sigmoid()[0, 0]
            pred = prob[None] > T
            tp = (pred & gt).sum((1, 2)).float()
            fp = (pred & ~gt).sum((1, 2)).float()
            fn = (~pred & gt).sum((1, 2)).float()
            union = tp + fp + fn
            ones = torch.ones_like(tp)
            iou = torch.where(union > 0, tp / union.clamp(min=1), ones)
            dice = torch.where(union > 0, 2 * tp / (2 * tp + fp + fn).clamp(min=1), ones)
            fg = pred.float().mean((1, 2))
            row = {k: batch[k][j] for k in ("image_id", "image_class", "prompt_class", "prompt")}
            row["is_negative"] = bool(batch["is_negative"][j])
            for t, a, b, c in zip(thresholds, iou.tolist(), dice.tolist(), fg.tolist()):
                row[_col("iou", t)], row[_col("dice", t)], row[_col("fg", t)] = a, b, c
            rows.append(row)
    return pd.DataFrame(rows)


def choose_thresholds(df, thresholds):
    """Per class, the threshold with the best mean IoU on positive pairs (tune on val only)."""
    pos = df[~df.is_negative]
    return {cls: float(max(thresholds, key=lambda t: g[_col("iou", t)].mean()))
            for cls, g in pos.groupby("prompt_class")}


def apply_thresholds(df, thr):
    """Collapse the per-threshold columns to iou / dice / fg at each row's class threshold."""
    out = df[[c for c in df.columns if "@" not in c]].copy()
    for kind in ("iou", "dice", "fg"):
        out[kind] = [df.at[i, _col(kind, thr[c])] for i, c in zip(df.index, df.prompt_class)]
    out["threshold"] = out.prompt_class.map(thr)
    return out


def summarize(res):
    """res = apply_thresholds(...) output.

    leak  = mean predicted foreground when the prompt names the OTHER class (should be ~0)
    clean = share of those negative pairs with < 1% predicted foreground
    """
    pos, neg = res[~res.is_negative], res[res.is_negative]
    per = {}
    for cls, g in pos.groupby("prompt_class"):
        n = neg[neg.prompt_class == cls]
        stats = dict(
            threshold=g.threshold.iloc[0],
            iou=g.iou.mean(), iou_std=g.iou.std(ddof=0), iou_p10=g.iou.quantile(0.10),
            dice=g.dice.mean(), dice_std=g.dice.std(ddof=0),
            neg_leak=n.fg.mean() if len(n) else np.nan,
            neg_clean=(n.fg < 0.01).mean() if len(n) else np.nan,
        )
        # plain Python floats: numpy scalars would break torch.load(weights_only=True) on the checkpoint
        per[cls] = dict(n=len(g), **{k: float(v) for k, v in stats.items()})
    return dict(per_class=per,
                mIoU=float(np.mean([v["iou"] for v in per.values()])),
                mDice=float(np.mean([v["dice"] for v in per.values()])))


def bucket_table(res):
    """Mean IoU per scene bucket (brightness / contrast / sharpness terciles) for the consistency check."""
    pos = res[~res.is_negative]
    tables = []
    for col in [c for c in pos.columns if c.endswith("_bucket")]:
        t = pos.groupby(["prompt_class", col]).iou.agg(["mean", "count"]).reset_index()
        t.columns = ["class", "bucket", "iou", "n"]
        t.insert(1, "factor", col.removesuffix("_bucket"))
        tables.append(t)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
