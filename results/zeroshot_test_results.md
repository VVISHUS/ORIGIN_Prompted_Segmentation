## Results on `test`

| prompt | n | IoU | IoU std | IoU p10 | Dice | Dice std | threshold | leak (other prompt fg) |
|---|---|---|---|---|---|---|---|---|
| segment crack | 805 | 0.313 | 0.209 | 0.054 | 0.439 | 0.242 | 0.300 | 0.153 |
| segment taping area | 157 | 0.175 | 0.202 | 0.000 | 0.256 | 0.250 | 0.200 | 0.298 |

**mIoU 0.244 | mDice 0.347**

### IoU by scene bucket (consistency)

| class | factor | low | mid | high | spread (max-min) |
|---|---|---|---|---|---|
| crack | brightness | 0.302 | 0.290 | 0.343 | 0.053 |
| crack | contrast | 0.355 | 0.302 | 0.273 | 0.082 |
| crack | sharpness | 0.368 | 0.307 | 0.269 | 0.099 |
| taping | brightness | 0.226 | 0.195 | 0.076 | 0.150 |
| taping | contrast | 0.248 | 0.145 | 0.108 | 0.140 |
| taping | sharpness | 0.193 | 0.166 | 0.167 | 0.027 |

### Runtime & footprint

| device | ms_per_image | ms_per_image_std | params_total | params_finetuned | model_size_mb_fp32 | checkpoint_mb |
|---|---|---|---|---|---|---|
| cuda | 22.553 | 4.154 | 150747746 | 0 | 575.059 | 0.000 |
