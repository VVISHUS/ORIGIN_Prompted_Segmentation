# Prompted Segmentation for Drywall QA

One text-conditioned model that returns a binary mask for **"segment crack"** and **"segment taping area"**.

## Layout

```
configs/                    experiment configs (YAML; `_base_` inheritance, CLI overrides)
drywall_seg/                library
  prepare.py                COCO → masks + index.csv, augmentation-aware split
  data.py                   prompts, dataset (positive/negative pairs), loaders
  augment.py                paired image/mask augmentations
  models/                   model registry + CLIPSeg wrapper
  losses.py                 loss registry (bce, dice, focal, tversky) + weighted sum
  evaluation.py             IoU/Dice per threshold, threshold tuning, summaries
  engine.py                 training loop, optimizer/scheduler factories
  inference.py, viz.py      prediction helpers, report panels
prepare_data.py / train.py / evaluate.py / predict.py     entry points
```

## Run (Colab / Kaggle GPU)

```bash
pip install -r requirements.txt
python prepare_data.py --config configs/clipseg_ft.yaml                 # masks + split (seed 42)
python evaluate.py     --config configs/clipseg_ft.yaml                 # zero-shot baseline
python train.py        --config configs/clipseg_ft.yaml                 # fine-tune
python evaluate.py     --run runs/<run-dir>                             # test metrics, tables, panels
python predict.py      --run runs/<run-dir> --split test                # <id>__segment_crack.png masks
```

Override any config value: `--set train.epochs=30 optim.lr=5e-4 model.unfreeze_last_n_vision_layers=3`.
Smoke test (CPU, ~1 min): `--set train.epochs=1 train.max_steps_per_epoch=2 train.batch_size=4 eval.max_images=6 data.num_workers=0`.

## Extending

- **Model**: subclass `PromptSegModel` (`forward(pixel_values, prompts) -> logits [B,1,h,w]`), decorate with
  `@register("name")`, import it in `drywall_seg/models/__init__.py`, set `model.name: name`.
- **Loss**: add a class with `@register("name")` in `losses.py`, list it under `loss.terms`.

## Reproducibility

Seed `42` (config `seed`) drives the split, sampling, augmentation and initialisation. Set
`deterministic: true` for bit-reproducible GPU runs (slower). Each run saves its resolved
`config.yaml`, `history.jsonl`, `train_info.json` (train time, params) and `best.pt`
(fine-tuned weights + val-tuned thresholds).
