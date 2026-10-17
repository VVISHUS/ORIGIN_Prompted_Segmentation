# Prompted Segmentation for Drywall QA

One text-conditioned model (fine-tuned **CLIPSeg**) that returns a binary mask for
**"segment crack"** and **"segment taping area"**.

**Test results:** mIoU **0.612**, mDice **0.738** (crack IoU 0.492, taping IoU 0.732).
Zero-shot CLIPSeg gets mIoU 0.244 on the same test set.

📄 **Full report (approach, results, error analysis, failed experiments): [REPORT.md](REPORT.md)**

## Layout

```
REPORT.md                   the report
results/                    final metrics, error analysis, training log and config of the submitted model
report_assets/              figures used in the report
configs/
  clipseg_ft.yaml           final model (all settings in one file)
  clipseg_ft_640.yaml       tried: 640 px input (not used, see report)
drywall_seg/                code
  prepare.py                COCO → masks + index.csv, split by original photo
  data.py                   prompts, dataset (positive / negative pairs), loaders
  augment.py                image + mask augmentations
  models/                   model registry + CLIPSeg wrapper
  losses.py                 BCE, Dice, Focal, Tversky + weighted sum
  evaluation.py             IoU / Dice, threshold tuning, summaries
  engine.py                 training loop
  inference.py, viz.py      prediction helpers, example images
prepare_data.py / train.py / evaluate.py / predict.py / analyze_errors.py     scripts
01_data_analysis.ipynb      data exploration
colab_train.ipynb           the Colab notebook I trained with (with outputs)
```

## Use the trained model (no training needed)

The submission zip contains the trained model in `model/` (`best.pt` = fine-tuned part + tuned thresholds,
`config.yaml`). From the unzipped folder:

```bash
pip install -r requirements.txt
python predict.py --run model --images <image file or folder> --prompt "segment crack" --prompt "segment taping area" --out my_predictions
```

One mask per (image, prompt) is written to `my_predictions/` as `<image name>__<prompt>.png`
(white = predicted area). Any image size works; the mask has the same size as the image.
The first run downloads the base CLIPSeg (~600 MB) from Hugging Face. Works on CPU too (~0.7 s per image).
From the repo instead of the zip, use a training run folder: `--run runs/<run-folder>`.

## Data

Download both datasets from Roboflow in **COCO** format and put them in `dataset/`:

| folder | Roboflow dataset |
|---|---|
| `dataset/cracks.v1i.coco` | [fyp-ny1jt/cracks-3ii36](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36) (COCO segmentation) |
| `dataset/Drywall-Join-Detect.v2i.coco` | [objectdetect-pu6rn/drywall-join-detect](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect), version 2 (COCO) |

## Run (Colab T4 or any GPU)

```bash
pip install -r requirements.txt
python prepare_data.py --config configs/clipseg_ft.yaml                     # masks + split (seed 42)
python evaluate.py     --config configs/clipseg_ft.yaml                     # zero-shot baseline
python train.py        --config configs/clipseg_ft.yaml --set train.epochs=40
python evaluate.py     --run runs/<run-folder>                              # test metrics, tables, examples
python analyze_errors.py --eval-dir runs/<run-folder>/eval_test             # error analysis
python predict.py      --run runs/<run-folder> --split test                 # <id>__segment_crack.png masks
```

Change any setting from the command line, e.g. `--set train.epochs=30 optim.lr=5e-4`.
Quick CPU check (~1 min): `--set train.epochs=1 train.max_steps_per_epoch=2 train.batch_size=4 eval.max_images=6 data.num_workers=0`.

## Seeds and reproducibility

- Seed **42** (config `seed`) controls the split, sampling, augmentation and initialisation.
- `deterministic: false` for speed, so GPU runs differ slightly: two runs gave best val mIoU 0.591 and 0.593.
- Every run folder saves its full `config.yaml`, `history.jsonl` (every epoch), `train_info.json`
  (train time, sizes) and `best.pt` (trained decoder weights + tuned thresholds, 4.3 MB).

## Prediction masks

`predict.py` writes one PNG per (image, prompt): single channel, same size as the image, values {0, 255},
named `<image_id>__segment_crack.png` / `<image_id>__segment_taping_area.png`.
