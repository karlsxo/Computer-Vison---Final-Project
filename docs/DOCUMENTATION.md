# Project Documentation — Trash Sorting by Material

*Computer Vision Group Project, West Visayas State University.*
This document records **everything that was done**, in order: what was built, how, why each
decision was taken, what went wrong and how it was fixed, and what the results are. It is the
technical companion to the PDF report (`report/`) and the executed notebook
(`notebooks/CV_Project.ipynb`).

---

## 0. Summary

| | |
|---|---|
| **Task** | Image classification — material of one trash item: cardboard, glass, metal, paper, plastic, trash (residual) |
| **A. Classical** | HOG + linear SVM, both implemented from scratch in NumPy |
| **B. From scratch** | Our 4-block CNN in PyTorch (1.13 M parameters, random init) |
| **C. Pretrained** | MobileNetV3-Large (ImageNet, torchvision), fine-tuned |
| **Main data** | TrashNet (2,527 → 2,519 images after cleaning), one saved 70/15/15 split |
| **Domain-shift data** | 600 unseen web photos (100/class) from the 12-class Garbage Classification dataset |
| **Research question** | Which method best sorts trash by material, and does it still work on messy real-world photos after training on clean ones? |

Final numbers: see [§9 Results](#9-results) and `results/results_table.md`.

---

## 1. Requirements analysis (from the course brief)

The brief (*Computer Vision Comparison Project*) requires, and this project delivers:

| Requirement | Where it is satisfied |
|---|---|
| One task, three methods (classical / from scratch / pretrained ≥ 2015) | §4–§6; MobileNetV3 is from 2019 |
| Classical core algorithm written by the group in NumPy, no library call for the main step | `src/hog_svm.py` — HOG **and** SVM written by us; scikit-image/scikit-learn only appear in the sanity check |
| Sanity check of the classical method | §4.4, figure `results/figures/A_sanity_checks.png` |
| Own architecture, training + validation curves, overfitting addressed | §5, `exp1_training_curves.png` |
| Pretrained: correct preprocessing and output format, same splits | §6 |
| Same train/val/test split for all methods | `splits/split.json`, loaded by every method through `common.load_split()` |
| Metrics: top-1 accuracy, macro-F1, confusion matrix + runtime (ms, CPU/GPU), params (M), train time (min) | §9, `results/results_table.md` |
| 5 experiments | §8 (the brief does not list them; we defined five — see §8) |
| System: upload image → results drawn on input → recommended method with explanation | §10, `app/app.py`, `src/system.py` |
| Deliverables: executed notebook + one .py per method, data link, PDF (Intro/Methods/Experiments/Results), slides, contribution statement | §12 |

---

## 2. Environment and setup

| Item | Value |
|---|---|
| Machine | Windows 11 laptop, Intel Core i5-1135G7 (4 cores / 8 threads), 8 GB RAM, NVIDIA GeForce MX330 (2 GB) |
| Python | 3.14.6 in a project virtual environment `.venv` |
| Main libraries | PyTorch 2.14.1 (+cu126), torchvision 0.29.1, NumPy 2.5.4, SciPy 1.17, Pillow 12, matplotlib 3.11, Gradio 6.32; scikit-image 0.26 / scikit-learn 1.9 (reference checks only) |

Steps:
1. `python -m venv .venv`, then `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
   and `pip install -r requirements.txt`.
2. **GPU problem.** `torch.cuda.is_available()` returned `True`, but every CUDA call failed with
   *"CUDA-capable device(s) is/are busy or unavailable"*. Cause: the installed NVIDIA driver
   (528.96) supports CUDA 12.0, while the only PyTorch wheels for Python 3.14 are built for CUDA 12.6.
   Fix in code: `src/torch_common.py` probes CUDA in a child process before importing torch and,
   if the probe fails, hides the GPU (`CUDA_VISIBLE_DEVICES=-1`) so PyTorch falls back to the CPU
   cleanly. On Colab (working GPU) the same code uses the GPU automatically.
   *Optional for the group:* updating the NVIDIA driver to ≥ 560 would enable the local GPU.
3. Consequence: all local training ran on the **CPU**. Training times in the results are CPU times
   and are reported as such. This also made us measure the cost of every design choice (§5, §6).
4. Memory: only ~0.5–0.6 GB RAM was free during the work (other applications open), so all images
   are cached once as resized `uint8` arrays (`cache/`) and all numerics use `float32`.

---

## 3. Data

### 3.1 Sources
| Dataset | Use | Link |
|---|---|---|
| TrashNet (Thung & Yang, 2016) — 2,527 photos, 512×384, white/grey studio background | train / val / test | https://github.com/garythung/trashnet (`data/dataset-resized.zip`) |
| Garbage Classification, 12 classes, ~15.5 k web-scraped images (Kaggle; HF mirror) | **only** as the unseen real-world test set | https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes |
| Own phone photos (optional) | extra real-world test set | put in `data/own_photos/<class>/` — evaluated automatically |

TACO (from the original plan) was not used: cropping its COCO annotations and mapping its
60 categories onto 6 materials is slow and error-prone; the 12-class dataset gives a cleaner,
larger domain-shift set with a simple class mapping.

Class counts in TrashNet: cardboard 403, glass 501, metal 410, paper 594, plastic 482, **trash 137** (imbalanced).

### 3.2 Cleaning (`src/prepare_data.py`)
1. **Exact duplicates (MD5).** Three files are byte-identical but stored under two classes:
   `glass115 = metal91`, `glass176 = plastic152`, `glass389 = plastic332`. The correct label is
   unknowable, so **all six copies are dropped**.
2. **Near-duplicates.** TrashNet often contains the same object photographed twice. If one shot
   lands in train and the other in test, accuracy is inflated (leakage).
   * First attempt: 64-bit dHash, Hamming distance ≤ 4. **Rejected** — on plain backgrounds it
     linked 1,660 pairs, 1,053 of them across *different classes*, and would have discarded 333 images.
   * Second attempt: global thumbnail correlation. Rejected after visual inspection — pairs with
     correlation 0.97–0.99 were different objects; the shared studio lighting gradient dominates.
   * **Final method — "detail signature":** 64×64 grayscale minus a Gaussian blur (σ = 6) — this
     removes the smooth background illumination — then zero-mean, unit-norm. Two images are
     near-duplicates if their signatures correlate > 0.80. We inspected contact sheets of pairs in
     the bands > 0.94, 0.8–0.94, 0.7–0.8, 0.6–0.7: above 0.80 they were the same item, below 0.80
     they were different items. Result: 30 linked pairs → 29 groups of ≥ 2 images.
3. **Group-aware stratified split.** Linked images form a group; a group is assigned to one split
   as a whole. Groups are assigned per class (largest first) to whichever split is furthest below
   its 70/15/15 target. One group mixed two classes and was dropped (2 images). Seed = 42.
4. **Leakage self-check:** the maximum detail-signature similarity between any val/test image and
   any train image is 0.755/0.756 (< 0.80) — asserted in code.

Final split (`splits/split.json`, paths relative to the project root):

| class | train | val | test |
|---|---|---|---|
| cardboard | 282 | 61 | 60 |
| glass | 348 | 75 | 74 |
| metal | 286 | 61 | 61 |
| paper | 416 | 89 | 89 |
| plastic | 336 | 72 | 72 |
| trash | 96 | 21 | 20 |
| **total** | **1,764** | **379** | **376** |

### 3.3 Real-world (domain-shift) test set
* Class mapping: cardboard→cardboard, paper→paper, metal→metal, plastic→plastic, trash→trash,
  brown-glass/green-glass/white-glass→glass; battery, biological, clothes, shoes excluded.
* **Discovery:** 490 sampled images of this dataset were exact copies of TrashNet photos
  (similarity 1.0 — the Kaggle set incorporates TrashNet). They were skipped, otherwise the
  "unseen" test would contain training images.
* 100 images per class were sampled (seeded) → 600 images (`splits/realworld.json`). They are
  product shots, magazine covers, outdoor scenes and some illustrations — very different from
  TrashNet's single items on a white table (`results/figures/data_domain_examples.png`).

### 3.4 Common pipeline
* The same source images feed every method; images are squashed to a square (128×128 for A and
  B, 224×224 for C — the resolution MobileNetV3 was pretrained at), bilinear resampling.
* Same labels, same split file, same metric function (`common.classification_metrics`), same
  timing function (`common.time_per_image`: batch size 1, 10 warm-up passes, mean over 200 test
  images, CPU) for every method.
* Seeds fixed (`common.set_seed(42)`); library versions printed in the notebook.

---

## 4. Method A — HOG + linear SVM (classical)

### 4.1 HOG, our implementation (`hog_svm.py`)
1. RGB → grayscale (BT.601 weights), values in [0, 1].
2. Gradients with the centred `[-1, 0, 1]` kernel (borders 0); magnitude `hypot(gx, gy)`;
   unsigned orientation `atan2(gy, gx) mod 180°`.
3. Cells of 8×8 px → 16×16 cells on a 128×128 image; 9 orientation bins of 20°, votes weighted by
   magnitude. **Soft binning**: each vote is split linearly between the two nearest bin centres
   (Dalal & Triggs) — implemented with `np.bincount` on flattened (cell, bin) indices.
4. Blocks of 2×2 cells, stride one cell (15×15 blocks) built with `sliding_window_view`;
   **L2-Hys** normalisation (L2, clip at 0.2, L2 again).
5. Concatenation → 15·15·36 = **8,100-D** descriptor. Extraction time ≈ 4 ms/image.

### 4.2 Linear SVM, our implementation
* One-vs-rest, all six classifiers trained jointly as one matrix `W (8100×6)`, bias `b (6)`.
* Objective: `λ/2·‖W‖² + (1/n)·Σᵢ Σₖ cᵢ·max(0, 1 − tᵢₖ(xᵢ·wₖ + bₖ))²` with `tᵢₖ = ±1`
  — the squared-hinge ("L2-loss") SVM also used by LIBLINEAR; it is differentiable, so plain
  gradient methods converge well.
* `cᵢ` = class-balanced weight `n / (K·n_class)` (the trash class is 5 % of the data).
* Optimiser: our own full-batch **Adam** (β = 0.9/0.999) with cosine learning-rate decay,
  lr 0.01, 300 iterations. Features are standardised with training-set mean/std.
* **Performance bug found and fixed:** the first version ran at 0.23 s/iteration. Profiling
  showed that a `float32` data matrix multiplied with `float64` weights makes NumPy up-cast the
  whole 1,764×8,100 matrix on every product (12× slower). Keeping everything in `float32` brought a
  full fit down to ~15–30 s.
* Prediction confidence for the app = softmax of the six SVM scores (a score, not a calibrated
  probability — stated in the app).

### 4.3 Model selection (validation split only)
Grid: cell size {8, 16} × binning {soft, hard} × λ {1, 3, 10, 30, 100}. The first grid
{0.3 … 10} had its best value at the edge (λ = 10), so it was extended upward; the optimum is now
interior (λ = 30). Selected: **cell 8, soft binning, λ = 30** (val accuracy 61.5 %, val macro-F1 0.590).
Full log: `results/method_A.json → tuning_log`.

### 4.4 Sanity checks (required by the rubric) — all passed
| Check | Result |
|---|---|
| (a) synthetic vertical bar → expected 0°, horizontal bar → 90°, diagonal edge → 45° | dominant bin = expected bin in all three |
| (b) our HOG vs scikit-image `hog()` (hard binning, L2-Hys) on 8 real images | max abs. difference 1.0·10⁻⁷, correlation 1.000000 |
| (c) our SVM on a 3-class 2-D Gaussian toy set | 100 % training accuracy, decision regions plotted |

### 4.5 Variant (extension, reported separately)
HOG + a 12×4×4 HSV colour histogram (our own RGB→HSV conversion, L1-normalised, square-rooted)
→ 8,292-D. Motivation: materials differ in colour/shine, which HOG discards.

---

## 5. Method B — 4-block CNN trained from scratch (`scratch_cnn.py`)

### 5.1 Architecture (ours)
```
input 3×128×128
Block 1  Conv3×3(3→32)    BN ReLU                         MaxPool2 → 32×64×64
Block 2  Conv3×3(32→64)   BN ReLU                         MaxPool2 → 64×32×32
Block 3  Conv3×3(64→128)  BN ReLU  Conv3×3(128→128) BN ReLU MaxPool2 → 128×16×16
Block 4  Conv3×3(128→256) BN ReLU  Conv3×3(256→256) BN ReLU MaxPool2 → 256×8×8
Global average pool → Dropout(0.4) → Linear(256→6)      1.13 M parameters, He initialisation
```
Design decision (measured on the CPU, seconds per epoch over 1,764 images):

| Variant | Params | s/epoch |
|---|---|---|
| single conv in every block | 0.39 M | 78 |
| **double conv in blocks 3–4 (chosen)** | 1.13 M | 124 |
| double conv in every block | 1.17 M | 209 |

The second conv is most useful in the deep blocks (larger receptive field, cheap on small maps);
doubling blocks 1–2 nearly doubles the compute for few extra parameters.

### 5.2 Training
* **Augmentation (our own, `torch_common.gpu_augment`)**, a fresh random draw per image, scaled
  by a `strength` factor s: one affine warp combining random zoom-crop (keeps (1−0.5s)–100 % of the
  area, aspect 3/4–4/3), rotation ±25s° and horizontal + vertical flips (trash photographed from
  above has no canonical orientation), reflection padding; then brightness/contrast/saturation
  jitter ±30s %, a ±5s % colour tint and random erasing (p = 0.25s). Shown in `data_augmentation.png`.
  It runs on whichever device trains the model (GPU on Colab), avoiding Windows DataLoader workers.
* **Attempt 1 (rejected) — strength 1.0**, label smoothing 0.1, AdamW lr 2·10⁻³, weight decay 0.05.
  After 15 epochs (~28 min) training accuracy was only 47 % and validation accuracy 53 %, both
  climbing very slowly. (Training accuracy is measured on augmented images with dropout active, so
  it is naturally a little below validation accuracy; the warning sign is that *both* were low.)
  The network was **under-fitting** — the regularisation was so strong that a small from-scratch
  network could not fit even the training data within a CPU-affordable number of epochs. Log kept in
  `results/log_method_B_attempt1_strong_aug.txt`. Run stopped.
* **Attempt 2 (final) — strength 0.6** (zoom 70–100 %, rotation ±15°, jitter ±18 %, erasing
  p = 0.15), label smoothing 0.05, AdamW lr 3·10⁻³, weight decay 0.02, batch 32, 2 warm-up epochs
  then cosine decay, max 40 epochs. Loss: cross-entropy with class-balanced weights.
  For fairness, Method C uses the same augmentation strength and label smoothing.
* **Overfitting controls:** augmentation, dropout 0.4, weight decay, label smoothing, early stopping
  (patience 12 epochs) on validation macro-F1 with the best checkpoint restored.
* Lesson: "overfitting addressed" means balancing — we checked both directions using the
  train-vs-validation curves rather than adding regularisation blindly.

---

## 6. Method C — MobileNetV3-Large, pretrained (`pretrained_mobilenet.py`)

* Howard et al., *Searching for MobileNetV3*, ICCV 2019 (satisfies "2015 or newer").
  Weights: torchvision `MobileNet_V3_Large_Weights.IMAGENET1K_V2` (ImageNet top-1 75.3 %).
* **Output format:** the final `Linear(1280→1000)` is replaced by `Linear(1280→6)`.
* **Preprocessing:** 224×224, ImageNet mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225).
* **Why Large instead of Small (change from the proposal):** the project goal is the best
  accuracy; Large is ~3 points better on ImageNet and still a mobile model (4.2 M params).
* **Training plan:** stage 1 — 3 epochs training only the classifier head (backbone frozen,
  lr 10⁻³) so the random head does not wreck the pretrained features; stage 2 — fine-tune at
  lr 5·10⁻⁴ (AdamW, wd 0.01, 1 warm-up epoch + cosine), up to 12 epochs, early stopping patience 6.
* **Partial freezing:** the stem and first three inverted-residual blocks (`features[0:4]`) stay
  frozen, with their BatchNorm layers kept in eval mode. They contain generic edge/colour filters
  and, operating at high resolution, dominate the cost: CPU time per epoch 320 s (all layers) vs
  215 s (first 4 frozen).
* Same augmentation, loss, label smoothing and early-stopping rule as Method B.

---

## 7. Shared training / evaluation code (`torch_common.py`, `common.py`)
* `train_model()` — one loop for B and C: AdamW + warm-up/cosine schedule, optional head-only
  phase, per-epoch validation (or every *k* epochs for the small-data runs), early stopping,
  best-checkpoint restore, full history saved for the curves.
* `evaluate()` / `predict_logits()` — deterministic resize → normalise → forward.
* `runtime_ms()` — CPU (and GPU if available) latency at batch size 1 including normalisation,
  excluding JPEG decoding (identical for all methods).
* `classification_metrics()` — our own confusion matrix, accuracy, per-class P/R/F1, macro-F1.

Smoke test before the long runs: 2 epochs of B and C on 5 images/class — both pipelines ran end
to end, frozen layers verified frozen, augmentation output range verified in [0, 1].

---

## 8. Experiments (`experiments.py`)
The brief's rubric mentions "all 5 experiments" without listing them; we defined:

1. **Main comparison** — accuracy, macro-F1, confusion matrices, per-class precision/recall/F1 on
   the 376-image test split; 95 % bootstrap confidence intervals (2,000 resamples) and exact
   McNemar tests between methods.
2. **Domain shift** — the models trained on TrashNet are evaluated, unchanged, on the 600 real-world
   web photos (and on `data/own_photos/` if the group adds photos).
3. **Small data** — every method retrained on 10, 25 and 50 images per class (seeded subsamples of
   the training split), same validation and test splits. A re-tunes λ ∈ {1, 10, 100}; B uses
   ~500 optimisation steps, C 3 head epochs + ~150 steps.
4. **Efficiency** — runtime per image (CPU, batch 1), parameters, model size, training time;
   accuracy-vs-cost plots.
5. **Error analysis** — top confused class pairs, failure examples, agreement between methods,
   majority-vote ensemble, Grad-CAM (Selvaraju et al., 2017) for B and C and gradient-magnitude maps
   for A.

Run order and wall-clock times on the laptop CPU: Method A 5 min (tuning incl.), Method B attempt 1
stopped after 28 min, attempt 2 101 min, Method C 45 min, experiments 1/2/4/5 under 1 min,
experiment 3 86 min.

---

## 9. Results

### 9.1 Experiment 1 — main comparison (TrashNet test split, 376 images)

| Method | Slot | Top-1 acc. (%) [95 % CI] | Macro F1 | Runtime (ms/img, CPU) | Params (M) | Train time (min, CPU) |
|---|---|---|---|---|---|---|
| HOG + linear SVM | A. Classical | 52.9 [47.6–58.0] | 0.519 | 3.6 | 0 | 0.3 |
| 4-block CNN (scratch) | B. From scratch | 71.3 [66.5–75.8] | 0.694 | 16.8 | 1.13 | 100.8 |
| **MobileNetV3-Large** | C. Pretrained | **92.8 [90.2–95.2]** | **0.926** | 29.1 | 4.21 | 44.8 |
| HOG + colour histogram + SVM | A. variant | 59.6 [54.8–64.6] | 0.587 | 13.6 | 0 | 0.5 |

* Exact McNemar tests: C vs B p = 1.7·10⁻¹⁷, C vs A p = 8.1·10⁻³⁹, B vs A p = 8.6·10⁻⁹,
  A+colour vs A p = 4.6·10⁻³ — every difference is significant.
* Best checkpoints: B epoch 34 / 40 (validation 76.5 %), C epoch 9 / 15 (validation 97.4 %;
  early-stopped at 15). C's head alone (backbone frozen) already reached 88.7 % validation.
* Figures: `exp1_training_curves.png`, `exp1_confusion_matrices.png`, `exp1_per_class_f1.png`.

### 9.2 Experiment 2 — domain shift (600 real-world web photos)

| Method | TrashNet test | Real-world | Drop | Accuracy retained |
|---|---|---|---|---|
| HOG + SVM | 52.9 | 9.3 | −43.6 | 18 % |
| HOG + colour | 59.6 | 16.5 | −43.1 | 28 % |
| Scratch CNN | 71.3 | 36.2 | −35.1 | 51 % |
| **MobileNetV3-Large** | 92.8 | **62.8** | −30.0 | 68 % |

* HOG + SVM falls **below chance** (16.7 %): it labels most web photos *paper* and real paper
  *cardboard* — its rules encode TrashNet's edge statistics, not material.
* **No method recognises the web "trash" class** (F1 ≤ 0.05): TrashNet "trash" = wrappers and
  packaging; web "trash" = diapers, toothbrushes, … — a *label-definition* shift.
  Excluding that class, C reaches 74.8 % on the real-world photos.

### 9.3 Experiment 3 — small data (test accuracy %)

| Method | 10/class | 25/class | 50/class | all (1,764) |
|---|---|---|---|---|
| HOG + SVM | 36.7 | 35.4 | 41.0 | 52.9 |
| Scratch CNN | 43.4 | 58.8 | 61.4 | 71.3 |
| MobileNetV3-Large | **67.6** | **75.5** | **83.8** | **92.8** |

MobileNetV3 with 10 images per class (67.6 %) is close to the scratch CNN with all data (71.3 %).

### 9.4 Experiment 4 — efficiency
HOG + SVM is the cheapest (3.6 ms, 0.26 MB file, 20 s training) but least accurate;
the scratch CNN needs 16.8 ms and 4.5 MB; MobileNetV3 needs 29.1 ms (≈ 34 images/s on a laptop CPU)
and 16.8 MB. MobileNetV3 trained in **less** time than the scratch CNN (45 vs 101 min) because it
starts from good features. Figure `exp4_efficiency.png`.

### 9.5 Experiment 5 — error analysis
* Top confusions — A: glass→metal (15), plastic→cardboard (13), plastic→paper (10);
  B: plastic→glass (10), paper→trash (9), plastic→paper (8); C: cardboard→paper (5), plastic→glass (4), glass→metal (4).
* Agreement on the 376 test images: all three correct 159, all wrong 13, only C correct 57,
  C wrong but A or B right 14. Majority vote = 87.8 % (< C alone, 92.8 %).
* Grad-CAM (`exp5_gradcam.png`): C attends to the object; B often attends to background and
  shadows (studio-specific cues → explains its larger domain-shift drop); HOG responds to every edge.

### 9.6 Answer to the research question
**MobileNetV3-Large (pretrained) is the best method** on every axis that matters for accuracy —
clean data, small data and real-world photos — and is fast enough for a phone. It **only partly**
survives the move to real-world photos (62.8 %, ≈ 75 % excluding the ill-defined trash class):
before deployment it should be fine-tuned on local photos, with "trash" defined as the local MRF defines it.

### 9.7 Hypotheses from the proposal — verdict
| Hypothesis | Verdict |
|---|---|
| Pretrained features win on small data | **Supported** (67.6 % vs 43.4 % vs 36.7 % at 10/class) |
| HOG does reasonably on shape but confuses colour/texture/transparency materials | **Supported** (glass→metal, plastic→paper/cardboard; colour histogram adds +6.7 points) |
| All methods lose accuracy under domain shift; scratch CNN may lose most | Partly — all lose; HOG loses the most, then the scratch CNN (Grad-CAM shows background reliance) |
| Paper/cardboard and glass/plastic are confusion pairs | **Supported** for C (cardboard→paper, plastic↔glass) and B (plastic→glass) |

---

## 10. The system (`src/system.py`, `app/app.py`)
* **Input:** the user uploads a photo (or uses the webcam / clipboard) in a Gradio web app.
* **Processing:** the image is resized for each method (128 px for A and B, 224 px for C) and all
  three run; each prediction's label, confidence and measured runtime are kept.
* **Output drawn on the input:** the recommended method's label in large text on the photo and
  three coloured bars underneath (one per method: label, confidence, runtime, ★ for the
  recommended one), plus a table with all class probabilities.
* **Recommendation:** computed automatically from `results/exp1_main.json` and
  `exp2_domain_shift.json` — the method with the highest test macro-F1 is the default; the
  explanation quotes its accuracy, its real-world accuracy and its CPU runtime. The app also says
  whether the three methods agree, warns when confidence < 50 %, and gives a disposal tip for the
  predicted material (Philippine context: RA 9003 segregation into biodegradable, recyclable and
  residual waste).
* **Deployment:** `python app/app.py` (local, http://127.0.0.1:7860) or, in Colab,
  `LAUNCH_APP = True` in the notebook, which prints a public `*.gradio.live` link.

---

## 11. Phone app (`web/`)

A Progressive Web App that runs **all three methods on the phone**, designed after the group's mock-up
(green header, rounded cards, bottom bar with a centre scan button, scanner frame, stats donut, profile).

* **URL:** https://karlsxo.github.io/Computer-Vison---Final-Project/ — open it in Chrome (Android) or Safari (iPhone), then *Add to Home Screen*.
* **Input:** live camera with a scanner frame (the photo is cropped to the frame, so the item fills the
  image like the training photos), the phone's camera app, or the gallery. EXIF orientation is applied.
* **Output:** material + recyclable/residual, the three methods' labels, confidences and on-phone runtimes,
  the recommended method (highest test macro-F1, read from `web/models/meta.json`) with its measured
  reason, a disposal tip, and a "Not right? choose the material" correction.
* **Extras:** history with thumbnails, points (+10 recyclable, +5 residual) and levels, stats donut,
  badges, editable name, Learn guides/articles (wish-cycling, RA 9003, how the AI decides, photo tips).
  Everything is stored only on the phone (localStorage); no photo is uploaded.
* **How the models run:** `tools/export_mobile.py` exports B and C to ONNX (opset 17) with
  normalisation and softmax inside the graph (max difference to PyTorch 3·10⁻⁶, 100 % same predictions)
  and A's SVM weights + standardisation to a 259 kB binary; HOG is re-implemented in `web/js/hog.js`.
  onnxruntime-web 1.20.1 (WebAssembly) is bundled in `web/vendor/ort/`.
* **Exact preprocessing:** `web/js/imaging.js` re-implements Pillow's BILINEAR resize (anti-aliased
  triangle filter, 22-bit fixed-point, horizontal then vertical pass), because a plain canvas resize
  gives slightly different pixels.
* **Verification:** `tools/verify_web.py` runs the phone pipeline in a headless browser on all 376 test
  images: agreement with Python = **100 % for A, B and C** (accuracy 52.9 / 71.3 / 92.8 %).
* **Offline:** a service worker (`web/sw.js`) caches the app and models (~32 MB) after the first visit.
* **Tested** in an emulated phone (390×844) with a fake camera: every screen and flow, no JS errors.



| Deliverable | File |
|---|---|
| Executed notebook (outputs visible, Colab-ready) | `notebooks/CV_Project.ipynb` |
| One .py module per method | `src/hog_svm.py`, `src/scratch_cnn.py`, `src/pretrained_mobilenet.py` |
| Data links + local data | README §Datasets; `splits/*.json`; `data/own_photos/` (optional) |
| PDF report (Introduction, Methods, Experiments, Results) | `report/Trash_Sorting_Report.pdf` |
| Slides (10 min) | `report/Trash_Sorting_Slides.pptx` |
| Contribution statement incl. AI disclosure | `report/Contribution_Statement.md` |
| Application | `app/app.py` |
| Results table | `results/results_table.md` / `.csv` |

## 13. AI-assistance disclosure (required by the brief)
An AI coding assistant (Claude) was used to help write and debug the code, run the experiments
and draft this documentation, the report and the slides. The group must review every file and be
able to explain every line in the Q&A; this disclosure must also appear in the contribution
statement.

## 14. References
1. R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed., Springer, 2022.
2. R. Krishna et al., *Computer Vision: Foundations and Applications* (Stanford CS131 notes), 2017.
3. PyTorch / torchvision documentation, https://pytorch.org/vision/stable/models.html
4. N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," CVPR 2005.
5. A. Howard et al., "Searching for MobileNetV3," ICCV 2019.
6. G. Thung and M. Yang, "Classification of Trash for Recyclability Status," CS229 project report, Stanford, 2016 (TrashNet).
7. D. P. Kingma and J. Ba, "Adam: A Method for Stochastic Optimization," ICLR 2015.
8. R.-E. Fan et al., "LIBLINEAR: A Library for Large Linear Classification," JMLR 9, 2008.
9. R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017.
10. I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," ICLR 2019 (AdamW).
11. Garbage Classification (12 classes), Kaggle dataset by M. Mohamed, 2021; mirror on Hugging Face.
12. Republic Act No. 9003, Ecological Solid Waste Management Act of 2000, Philippines.

## 15. What the group still has to do before submitting
1. Fill in the group number and member names: report header (`tools/build_report.py`, line with
   "Group __"), slide 1 (`tools/build_slides.js`), `report/Contribution_Statement.md`; then rebuild with
   `python tools/build_report.py` and `node tools/build_slides.js` (from `tools/`).
2. Rename the zip to your real group number (`GroupNo_CVProject.zip` → e.g. `Group3_CVProject.zip`).
3. Record the 3-minute demo video: run `python app/app.py`, classify the same new photos with all
   three methods (the `app/examples/` images or your own).
4. Optional but recommended: take 20–30 phone photos per class in Iloilo, save them in
   `data/own_photos/<class>/`, and re-run experiment 2 (`python src/experiments.py 2`) — they are
   evaluated automatically as a second real-world test set.
5. Every member: read the code of all three methods and this document; be ready to explain every line.
6. Optional: update the NVIDIA driver (≥ 560) to train on the local GPU, or run the notebook on Colab.
