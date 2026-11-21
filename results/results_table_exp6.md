| Method | TrashNet test acc. (%) only-TrashNet -> mixed | Real-world acc. (%) only-TrashNet -> mixed [95% CI] | Real-world acc., clean subset (%) | Real-world macro-F1 | 'Not sure' threshold | Answered / accuracy of answers | Runtime (ms/img, CPU) | Params (M) | Train time (min, CPU) |
|---|---|---|---|---|---|---|---|---|---|
| HOG + linear SVM | 52.9 -> 47.9 | 9.3 -> 47.8 [44.0-51.5] | 43.7 | 0.465 | 0.45 | 4% / 88.0% | 2.6 | 0.00 | 4.4 |
| 4-block CNN (scratch) | 71.3 -> 69.7 | 36.2 -> 78.8 [75.5-82.2] | 78.2 | 0.782 | 0.95 | 21% / 98.4% | 17.6 | 1.13 | 186.4 |
| MobileNetV3-Large (pretrained) | 92.8 -> 94.1 | 62.8 -> 97.7 [96.3-98.8] | 97.7 | 0.977 | 0.65 | 94% / 99.5% | 37.6 | 4.21 | 101.8 |
