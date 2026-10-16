"""Roboflow COCO exports → binary mask PNGs + index.csv with an augmentation-aware split.

Roboflow names every augmented copy `<source>.rf.<hash>.jpg`. All copies of one source photo are
assigned to the same split, so no photo is seen in training and then scored again in val/test.
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def anns_to_mask(anns, h, w, min_box_size=2):
    """Polygons are filled; box-only annotations become filled rectangles. Returns (mask, n_dropped)."""
    mask = np.zeros((h, w), np.uint8)
    dropped = 0
    for a in anns:
        seg = a.get("segmentation")
        polys = [np.asarray(p, np.float32).reshape(-1, 2) for p in seg if len(p) >= 6] if isinstance(seg, list) else []
        if polys:
            cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polys], 255)
            continue
        x, y, bw, bh = a["bbox"]
        if bw < min_box_size or bh < min_box_size:      # degenerate box
            dropped += 1
            continue
        mask[int(round(y)):int(round(y + bh)), int(round(x)):int(round(x + bw))] = 255
    return mask, dropped


def assign_splits(df, fractions, seed):
    """Split per class by source photo; group boundaries use each group's midpoint for balance."""
    names = list(fractions)
    bounds = np.cumsum([fractions[n] for n in names])
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["split"] = ""
    for _, sub in df.groupby("dataset"):
        sources = np.array(sorted(sub.source.unique()), dtype=object)
        rng.shuffle(sources)
        sizes = sub.source.value_counts()[sources]
        mid = (sizes.cumsum() - sizes / 2) / len(sub)
        idx = np.minimum(np.searchsorted(bounds, mid.to_numpy(), side="right"), len(names) - 1)
        df.loc[sub.index, "split"] = sub.source.map(dict(zip(sources, np.array(names)[idx])))
    return df


def prepare(cfg):
    out = Path(cfg.data.prepared_dir)
    rows, dropped = [], 0
    for cls, root in cfg.data.datasets.items():
        jsons = sorted(Path(root).glob("*/_annotations.coco.json"))
        if not jsons:
            raise FileNotFoundError(f"no */_annotations.coco.json under {root}")
        (out / "masks" / cls).mkdir(parents=True, exist_ok=True)
        for js in jsons:
            coco = json.loads(js.read_text(encoding="utf-8"))
            by_img = defaultdict(list)
            for a in coco["annotations"]:
                by_img[a["image_id"]].append(a)
            for im in tqdm(coco["images"], desc=f"{cls}/{js.parent.name}"):
                path = js.parent / im["file_name"]
                img = cv2.imread(str(path))
                if img is None:
                    print(f"WARNING unreadable image skipped: {path}")
                    continue
                h, w = img.shape[:2]
                mask, d = anns_to_mask(by_img[im["id"]], h, w, cfg.data.get("min_box_size", 2))
                dropped += d
                stem = Path(im["file_name"]).stem
                mask_path = out / "masks" / cls / f"{stem}.png"
                cv2.imwrite(str(mask_path), mask)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rows.append(dict(
                    image_id=stem, dataset=cls, source=im["file_name"].split(".rf.")[0],
                    rf_split=js.parent.name, file=path.as_posix(), mask=mask_path.as_posix(),
                    width=w, height=h, n_objects=len(by_img[im["id"]]), fg_frac=float((mask > 0).mean()),
                    brightness=float(gray.mean()), contrast=float(gray.std()),
                    sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                ))

    df = pd.DataFrame(rows)
    if df.duplicated(["dataset", "image_id"]).any():
        raise ValueError("duplicate image ids within a dataset")

    # scene-variety buckets (per-class terciles) for the consistency analysis
    for col, labels in [("brightness", ["dark", "mid", "bright"]),
                        ("contrast", ["low", "mid", "high"]),
                        ("sharpness", ["blurry", "mid", "sharp"])]:
        df[f"{col}_bucket"] = df.groupby("dataset")[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 3, labels=labels)).astype(str)

    df = assign_splits(df, cfg.data.split, cfg.seed)
    leaked = df.groupby(["dataset", "source"]).split.nunique().gt(1).sum()
    assert leaked == 0, f"{leaked} source photos span several splits"

    df.to_csv(cfg.data.index_csv, index=False)
    counts = df.groupby(["dataset", "split"]).size().unstack(fill_value=0)[list(cfg.data.split)]
    counts["total"] = counts.sum(axis=1)
    counts["sources"] = df.groupby("dataset").source.nunique()
    counts.to_csv(out / "split_counts.csv")
    print(f"\ndropped degenerate boxes: {dropped}\n\nimages per split (augmentation-aware, seed {cfg.seed}):\n{counts}")
    return df
