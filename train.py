"""Train a text-prompted segmentation model.

    python train.py --config configs/clipseg_ft.yaml
    python train.py --config configs/clipseg_ft.yaml --set train.epochs=30 optim.lr=5e-4
"""
import argparse

from drywall_seg.config import load_config
from drywall_seg.engine import train


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    args = p.parse_args()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
