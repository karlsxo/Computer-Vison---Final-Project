# Datasets used in this project

Everything the code needs is in this folder, at the paths referenced by `splits/*.json`, so the
project runs without downloading anything.

| Folder | What | Images | Used for |
|---|---|---|---|
| `raw/dataset-resized/<class>/` | **TrashNet** (complete) | 2,527 | train / validation / test split (`splits/split.json`) |
| `raw/garbage12/garbage_classification/<source class>/` | the images **we used** from the 12-class **Garbage Classification** web dataset | 2,495 | 600 real-world test photos (`splits/realworld.json`) + 1,895 extra training/validation photos for Experiment 6 (`splits/mixed.json`) |
| `own_photos/<class>/` | optional: your own phone photos | 0 | extra real-world test set, evaluated automatically |

Web images by source class (mapping to our 6 classes: the three glass classes → glass; battery,
biological, clothes and shoes are not used):

| Source class | Images |
|---|---|
| `brown-glass` | 142 |
| `cardboard` | 440 |
| `green-glass` | 145 |
| `metal` | 355 |
| `paper` | 440 |
| `plastic` | 380 |
| `trash` | 440 |
| `white-glass` | 153 |

## Sources and credit

* **TrashNet** — G. Thung and M. Yang, *Classification of Trash for Recyclability Status*, Stanford
  CS229, 2016. <https://github.com/garythung/trashnet> (file `data/dataset-resized.zip`).
* **Garbage Classification (12 classes)** — M. Mohamed, Kaggle, 2021
  (<https://www.kaggle.com/datasets/mostafaabla/garbage-classification>), downloaded from the mirror
  <https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes>. Only the
  2,495 images used by the project are included here; the full set has 15,515 images.

The images belong to their original authors and are redistributed here only so that the course
project is reproducible; see the original pages for their terms of use and cite them as above.

## How the files were chosen and cleaned
See `docs/DOCUMENTATION.md` §3: exact and near-duplicate removal, the group-aware 70/15/15 split,
the removal of 490 web images that are copies of TrashNet photos, and the leakage rules for the
Experiment 6 training photos (§3.5).

## Getting the full original datasets
Delete `data/raw/` and run the notebook's download cell (or download the two links above). The
splits in `splits/` are fixed and are only rebuilt if you delete them.
