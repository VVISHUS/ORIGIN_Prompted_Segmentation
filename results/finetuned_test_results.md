## Results on `test`

| prompt | n | IoU | IoU std | IoU p10 | Dice | Dice std | threshold | leak (other prompt fg) |
|---|---|---|---|---|---|---|---|---|
| segment crack | 805 | 0.492 | 0.180 | 0.252 | 0.638 | 0.180 | 0.400 | 0.000 |
| segment taping area | 157 | 0.732 | 0.122 | 0.588 | 0.838 | 0.094 | 0.450 | 0.001 |

**mIoU 0.612 | mDice 0.738**

### IoU by scene bucket (consistency)

| class | factor | low | mid | high | spread (max-min) |
|---|---|---|---|---|---|
| crack | brightness | 0.484 | 0.503 | 0.491 | 0.020 |
| crack | contrast | 0.499 | 0.496 | 0.482 | 0.017 |
| crack | sharpness | 0.474 | 0.472 | 0.526 | 0.053 |
| taping | brightness | 0.737 | 0.767 | 0.688 | 0.079 |
| taping | contrast | 0.717 | 0.773 | 0.707 | 0.066 |
| taping | sharpness | 0.732 | 0.724 | 0.739 | 0.015 |

### Runtime & footprint

| device | ms_per_image | ms_per_image_std | params_total | params_finetuned | model_size_mb_fp32 | checkpoint_mb |
|---|---|---|---|---|---|---|
| cuda | 23.280 | 3.929 | 150747746 | 1127009 | 575.059 | 4.315 |
