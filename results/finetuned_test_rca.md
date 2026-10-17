# Error analysis

## Overall (oracle_iou / oracle_grid = ceilings at 352 px / 22×22 grid; pred_vs_gt > 1 = over-segmenting)

| prompt_class | n | iou | oracle_iou | oracle_grid | pred_vs_gt | width_px |
|---|---|---|---|---|---|---|
| crack | 805 | 0.492 | 0.997 | 0.438 | 1.105 | 18.300 |
| taping | 157 | 0.732 | 1.000 | 0.923 | 1.026 | 89.488 |

## Error type share (rows sum to 1)

| prompt_class | miss | over | shape | under |
|---|---|---|---|---|
| crack | 0.015 | 0.302 | 0.470 | 0.214 |
| taping | 0.000 | 0.045 | 0.930 | 0.025 |

## Mean IoU per error type

| prompt_class | miss | over | shape | under |
|---|---|---|---|---|
| crack | 0.011 | 0.364 | 0.608 | 0.451 |
| taping | nan | 0.446 | 0.753 | 0.451 |

## IoU by GT stroke width (px at 640)

| prompt_class | width_px | n | iou | oracle_iou | oracle_grid | pred_vs_gt |
|---|---|---|---|---|---|---|
| crack | <5 | 68 | 0.227 | 0.982 | 0.000 | 3.542 |
| crack | 5-10 | 150 | 0.441 | 0.995 | 0.013 | 1.639 |
| crack | 10-20 | 210 | 0.526 | 0.998 | 0.223 | 1.035 |
| crack | 20-40 | 269 | 0.532 | 0.999 | 0.755 | 0.778 |
| crack | >40 | 108 | 0.565 | 1.000 | 0.930 | 0.700 |
| taping | 10-20 | 2 | 0.505 | 1.000 | 0.401 | 1.473 |
| taping | 20-40 | 28 | 0.671 | 1.000 | 0.815 | 1.081 |
| taping | >40 | 127 | 0.749 | 1.000 | 0.956 | 1.020 |

## IoU by GT size (share of image)

| prompt_class | fg_frac | n | iou | oracle_iou | oracle_grid | pred_vs_gt |
|---|---|---|---|---|---|---|
| crack | <1% | 104 | 0.289 | 0.989 | 0.000 | 3.196 |
| crack | 1-3% | 283 | 0.487 | 0.996 | 0.173 | 1.342 |
| crack | 3-10% | 355 | 0.544 | 0.999 | 0.690 | 0.848 |
| crack | 10-30% | 62 | 0.557 | 1.000 | 0.927 | 0.646 |
| crack | >30% | 1 | 0.690 | 1.000 | 0.955 | 0.700 |
| taping | 1-3% | 15 | 0.572 | 1.000 | 0.720 | 1.210 |
| taping | 3-10% | 47 | 0.737 | 1.000 | 0.898 | 1.013 |
| taping | 10-30% | 71 | 0.768 | 1.000 | 0.962 | 1.029 |
| taping | >30% | 24 | 0.715 | 1.000 | 0.985 | 1.026 |

## IoU by number of GT objects (4 = 4+)

| prompt_class | n_obj | n | iou | oracle_iou | oracle_grid | pred_vs_gt |
|---|---|---|---|---|---|---|
| crack | 1 | 669 | 0.493 | 0.997 | 0.451 | 1.092 |
| crack | 2 | 82 | 0.518 | 0.998 | 0.416 | 1.114 |
| crack | 3 | 33 | 0.447 | 0.990 | 0.316 | 1.603 |
| crack | 4 | 21 | 0.446 | 0.995 | 0.281 | 1.345 |
| taping | 1 | 137 | 0.744 | 1.000 | 0.919 | 1.013 |
| taping | 2 | 20 | 0.646 | 1.000 | 0.955 | 1.180 |

## 15 worst images (inspect these in the panels / with predict.py)

| index | prompt_class | image_id | iou | oracle_iou | width_px | pred_vs_gt | error_type | fg_frac |
|---|---|---|---|---|---|---|---|---|
| 208 | crack | crack289_jpg.rf.938b438a43f63bcdf85d2e7f3c11c1f4 | 0.000 | 1.000 | 21.662 | 0.000 | miss | 0.035 |
| 219 | crack | crack289_jpg.rf.21cbbedc3446b662d185696bbfa5b6b7 | 0.000 | 1.000 | 15.019 | 0.000 | miss | 0.024 |
| 221 | crack | crack289_jpg.rf.42f7f1b1c32502e9ada5e856420d4bff | 0.000 | 1.000 | 15.020 | 0.000 | miss | 0.024 |
| 224 | crack | crack289_jpg.rf.d21b5a61a7ee77f3574d89a8ba6adec2 | 0.000 | 1.000 | 21.662 | 0.000 | miss | 0.035 |
| 233 | crack | crack289_jpg.rf.51a5e015d1ff3b3f3ea91a3ba764e7b1 | 0.000 | 1.000 | 15.019 | 0.000 | miss | 0.024 |
| 313 | crack | cracking411_jpg.rf.d283cfd9d40064030bcc73b0e281f143 | 0.000 | 1.000 | 20.024 | 0.000 | miss | 0.033 |
| 331 | crack | cracking411_jpg.rf.f28552b126e8e9e4febed70383e178d9 | 0.000 | 1.000 | 20.021 | 0.282 | miss | 0.033 |
| 348 | crack | cracking411_jpg.rf.a3b152eeb17e157edf82c3e7f66510be | 0.000 | 1.000 | 20.021 | 0.000 | miss | 0.033 |
| 316 | crack | cracking411_jpg.rf.a499bc8b1a362616e19d32ca1d1d588c | 0.020 | 1.000 | 20.305 | 0.020 | miss | 0.033 |
| 296 | crack | cracking411_jpg.rf.c2dee9051679c88fb8c24594340373a3 | 0.030 | 1.000 | 20.305 | 0.260 | miss | 0.033 |
| 229 | crack | crack289_jpg.rf.52320bc41d03c99fa3c30a70b5395ad4 | 0.040 | 1.000 | 14.683 | 0.040 | miss | 0.023 |
| 315 | crack | cracking411_jpg.rf.ac84070e4a527a018042004baf72b63a | 0.042 | 1.000 | 20.305 | 0.042 | miss | 0.033 |
| 417 | crack | cracking0102109_jpg.rf.0dee1e9ee4a7ecb72f48f44580099288 | 0.085 | 0.960 | 4.059 | 11.762 | over | 0.002 |
| 82 | crack | cracking0157_jpg.rf.0b72c804fc46b7b790b6e01547709604 | 0.094 | 0.962 | 2.296 | 8.966 | over | 0.003 |
| 483 | crack | cracking462_jpg.rf.097aab32a0c3a688fc6c9f72fe20cf2b | 0.095 | 0.994 | 3.743 | 10.549 | over | 0.004 |
