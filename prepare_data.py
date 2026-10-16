"""COCO exports → masks + index.csv with an augmentation-aware train/val/test split.

    python prepare_data.py --config configs/clipseg_ft.yaml
"""
import argparse
from pathlib import Path

from drywall_seg.config import load_config
from drywall_seg.prepare import prepare


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    Path(cfg.data.prepared_dir).mkdir(parents=True, exist_ok=True)
    prepare(cfg)


if __name__ == "__main__":
    main()
