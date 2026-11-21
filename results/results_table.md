| Method | Slot | Top-1 acc. (%) [95% CI] | Macro F1 | Runtime (ms/img, CPU) | Params (M) | Train time (min) |
|---|---|---|---|---|---|---|
| HOG + linear SVM | A. Classical | 52.9 [47.6-58.0] | 0.519 | 3.6 | 0.00 | 0.3 |
| 4-block CNN (scratch) | B. From scratch | 71.3 [66.5-75.8] | 0.694 | 16.8 | 1.13 | 100.8 |
| MobileNetV3-Large (pretrained) | C. Pretrained | 92.8 [90.2-95.2] | 0.926 | 29.1 | 4.21 | 44.8 |
| HOG + colour hist. + SVM (variant) | A. (variant) | 59.6 [54.8-64.6] | 0.587 | 13.6 | 0.00 | 0.5 |
