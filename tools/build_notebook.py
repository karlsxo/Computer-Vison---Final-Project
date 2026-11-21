"""Generate notebooks/CV_Project.ipynb (the single executed notebook deliverable).

The notebook is the top-to-bottom driver of the whole project. In a fresh Colab
session it downloads the data and trains every model; when trained models and
result files already exist (e.g. produced by the src/*.py scripts on the same
split) it re-uses them unless RETRAIN = True.

Run:  python tools/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(r"""
# Trash Sorting by Material: Hand-Designed Features vs. a Small CNN vs. a Pretrained Mobile Model
**Computer Vision Group Project - West Visayas State University**

**Task:** image classification. Given a photo of one piece of trash, predict its material:
`cardboard, glass, metal, paper, plastic, trash` (residual).

| Slot | Method | Implementation |
|---|---|---|
| A. Classical | HOG + linear SVM | HOG **and** the SVM written from scratch in NumPy (`src/hog_svm.py`) |
| B. Trained from scratch | Our 4-block CNN | Designed and written in PyTorch, random init (`src/scratch_cnn.py`) |
| C. Pretrained, modern | MobileNetV3-Large (2019) | torchvision ImageNet weights, fine-tuned (`src/pretrained_mobilenet.py`) |

**Research question:** *Which method best sorts trash by material, and does it still work on messy
real-world photos after training on clean studio ones?*

All three methods use **the same saved split** (`splits/split.json`), the **same metric code**
(`src/common.py`) and the **same timing protocol** (CPU, batch size 1, warm-up, mean over 200 images).

**How to run in Google Colab:** open this notebook from GitHub (the first cell clones the repository; or upload `GroupNo_CVProject.zip` to *My Drive* and set `GIT_URL = ""`),
choose *Runtime -> Change runtime type -> GPU*, then *Runtime -> Run all*.
A full run from scratch takes roughly 30-40 minutes on a Colab T4 GPU.
""")

code(r"""
# ---- 0. Setup: locate the project (local or Colab) ----------------------------------------
import os, sys, subprocess, time, json
from pathlib import Path

RETRAIN = False          # True = retrain every model even if saved models exist
GIT_URL = "https://github.com/karlsxo/Computer-Vison---Final-Project.git"   # project repository (Colab clones it); set "" to use the zip on Drive
ZIP_ON_DRIVE = "/content/drive/MyDrive/GroupNo_CVProject.zip"

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    PROJECT = Path("/content/project")
    if not (PROJECT / "src").exists():
        if GIT_URL:
            subprocess.run(["git", "clone", GIT_URL, str(PROJECT)], check=True)
        else:
            from google.colab import drive
            drive.mount("/content/drive")
            subprocess.run(["unzip", "-q", "-o", ZIP_ON_DRIVE, "-d", "/content/unzipped"], check=True)
            src = next(Path("/content/unzipped").rglob("src")).parent
            subprocess.run(["cp", "-r", str(src), str(PROJECT)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT / "requirements.txt")])
else:
    PROJECT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))
print("Project folder:", PROJECT, "| Colab:", IN_COLAB)
""")

code(r"""
# ---- 0b. Download the datasets if they are not present ------------------------------------
import urllib.request, zipfile
RAW = PROJECT / "data" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
SOURCES = {
    # TrashNet (Thung & Yang, 2016) - 2,527 images, 6 classes, 512x384
    "dataset-resized": "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip",
    # 12-class web-scraped Garbage Classification dataset (Kaggle, mirrored on Hugging Face)
    "garbage12": "https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes/resolve/main/garbage_classification.zip",
}
for name, url in SOURCES.items():
    target = RAW / name
    if target.exists():
        print(f"{name}: present"); continue
    zpath = RAW / f"{name}.zip"
    print(f"Downloading {name} ..."); urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW if name == "dataset-resized" else target)
    print(f"{name}: extracted")
""")

code(r"""
# ---- 0c. Imports, versions, seeds ---------------------------------------------------------
import torch_common as tc          # probes CUDA first (falls back to CPU if the GPU is unusable)
import numpy as np, torch, torchvision, platform, matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image as ShowImage, Markdown, display
from common import *
set_seed(SEED)
print(f"python {platform.python_version()} | numpy {np.__version__} | torch {torch.__version__} | "
      f"torchvision {torchvision.__version__} | device: {tc.DEVICE}")
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
def show(name, width=1000): display(ShowImage(filename=str(FIG_DIR / name), width=width))
def need(path): return RETRAIN or not Path(path).exists()
""")

md(r"""
## 1. Data: cleaning and a leakage-safe split

* **Exact duplicates:** identical files (MD5). Three files appear under *two different classes*
  (e.g. `glass176.jpg` = `plastic152.jpg`) - the label is ambiguous, so every copy is dropped.
* **Near-duplicates:** TrashNet often contains the same item photographed twice. We compute a
  *detail signature* (64x64 grayscale minus a heavy blur, which removes the shared studio lighting,
  then zero-mean/unit-norm) and link pairs whose correlation exceeds 0.80. Linked images form a
  *group* and a group always goes to one split, so the same item never appears in train and test.
  (A 64-bit dHash was tried first and rejected: on plain backgrounds it linked 1,053 pairs of
  *different-class* images.)
* **Split:** seeded, stratified, group-aware 70/15/15, saved to `splits/split.json` and loaded by all methods.
* **Real-world test set (domain shift):** 100 images/class from the web-scraped 12-class Garbage
  Classification dataset (the three glass classes -> glass; battery, biological, clothes, shoes excluded).
  490 of its images turned out to be copies of TrashNet images and were excluded.
""")

code(r"""
import prepare_data
if need(SPLIT_FILE) or need(REALWORLD_FILE):
    sigs = prepare_data.build_trashnet_split()
    prepare_data.build_realworld(sigs)
js = load_json(SPLIT_FILE)
print(json.dumps({k: v for k, v in js["stats"].items() if k != "counts"}, indent=1))
import pandas as pd
display(pd.DataFrame(js["stats"]["counts"]).assign(total=lambda d: d.sum(1)))
print("Real-world set:", len(load_realworld()), "images; own photos:", len(own_photo_items()))
""")

code(r"""
import experiments as E
E.dataset_figures()
show("data_class_counts.png", 700); show("data_domain_examples.png"); show("data_augmentation.png", 900)
""")

md(r"""
## 2. Method A - HOG + linear SVM (classical, written from scratch)

**HOG (Dalal & Triggs, 2005)** - our NumPy implementation (`src/hog_svm.py`):
1. grayscale (BT.601), gradients with the `[-1, 0, 1]` kernel, magnitude and unsigned orientation (0-180 deg);
2. 8x8-pixel cells, 9-bin orientation histograms weighted by magnitude (linear interpolation between bins);
3. overlapping 2x2-cell blocks, L2-Hys normalisation (clip at 0.2); concatenation -> 8,100-D vector (128x128 input).

**Linear SVM** - one-vs-rest, L2-regularised squared hinge loss (the L2-loss SVM of LIBLINEAR),
class-balanced weights, optimised by our own full-batch Adam with cosine learning-rate decay.
Features are standardised with training-set statistics. Hyper-parameters (cell size, soft/hard
binning, regularisation lambda) are chosen on the **validation** split.

**Sanity checks:** (a) synthetic bars/edges must peak in the expected orientation bin;
(b) our HOG must match scikit-image's reference HOG; (c) our SVM must separate a toy dataset.
""")

code(r"""
import hog_svm as A
sc = A.sanity_checks()
for name, t in sc["synthetic"].items():
    print(f"(a) {name:<15} expected {t['expected_deg']:>4.0f} deg -> peak bin {t['peak_range_deg']}  pass={t['pass']}")
print("(b) vs scikit-image:", sc["skimage_reference"])
print("(c) toy SVM:", sc["toy_svm"])
show("A_sanity_checks.png")
""")

code(r"""
if need(RESULTS_DIR / "method_A.json"):
    A.main()
rA = load_json(RESULTS_DIR / "method_A.json")
display(pd.DataFrame([{**r["cfg"], "lambda": r["lam"], "val_acc": round(r["val_acc"], 1),
                       "val_macro_F1": round(r["val_macro_f1"], 3)} for r in rA["tuning_log"]]))
print("Selected:", rA["selected"]["cfg"], "lambda =", rA["selected"]["lam"])
print(f"TEST accuracy {rA['test']['accuracy']:.2f}%  macro-F1 {rA['test']['macro_f1']:.3f}  "
      f"runtime {rA['runtime_ms_cpu']:.2f} ms/img (CPU)  train {rA['train_time_min']:.2f} min")
v = rA["variant_color"]
print(f"Variant HOG + HSV colour histogram: TEST accuracy {v['test']['accuracy']:.2f}%  macro-F1 {v['test']['macro_f1']:.3f}")
""")

md(r"""
## 3. Method B - our 4-block CNN, trained from scratch

```
input 3x128x128
Block 1: Conv3x3(3->32)   BN ReLU                      MaxPool -> 32x64x64
Block 2: Conv3x3(32->64)  BN ReLU                      MaxPool -> 64x32x32
Block 3: Conv3x3(64->128) BN ReLU Conv3x3(128->128) BN ReLU MaxPool -> 128x16x16
Block 4: Conv3x3(128->256)BN ReLU Conv3x3(256->256) BN ReLU MaxPool -> 256x8x8
GlobalAvgPool -> Dropout(0.4) -> Linear(256->6)          (1.13 M parameters, He init)
```
**Training:** class-weighted cross-entropy with label smoothing 0.1, AdamW (lr 2e-3, weight decay 0.05),
3-epoch warm-up + cosine decay, batch 32, up to 45 epochs.
**Overfitting controls:** our per-image augmentation (random zoom-crop 50-100 %, rotation +-25 deg,
horizontal/vertical flips, colour jitter, random erasing), dropout, weight decay, label smoothing,
early stopping on validation macro-F1 (best checkpoint restored).
""")

code(r"""
import scratch_cnn as B
print(B.build_model()); print(f"parameters: {tc.count_params(B.build_model()):.3f} M")
if need(RESULTS_DIR / "method_B.json"):
    B.main()
rB = load_json(RESULTS_DIR / "method_B.json")
print(f"best epoch {rB['train_info']['best_epoch']} of {rB['train_info']['epochs_run']} | "
      f"TEST accuracy {rB['test']['accuracy']:.2f}%  macro-F1 {rB['test']['macro_f1']:.3f}  "
      f"runtime {rB['runtime_ms_cpu']:.2f} ms/img (CPU)  train {rB['train_time_min']:.1f} min on {rB['train_device']}")
""")

md(r"""
## 4. Method C - MobileNetV3-Large, ImageNet-pretrained and fine-tuned

* Weights `IMAGENET1K_V2` from torchvision (Howard et al., *Searching for MobileNetV3*, ICCV 2019).
* The 1000-class output layer is replaced by a 6-class layer (same output format as A and B).
* Correct preprocessing: 224x224 input normalised with the ImageNet mean/std.
* Stage 1: train only the classifier head for 3 epochs (backbone frozen).
  Stage 2: fine-tune at lr 5e-4 (AdamW, cosine) for up to 12 epochs with early stopping.
  The stem and first three blocks stay frozen with BatchNorm in eval mode (generic low-level filters).
* Same split, augmentation function, loss and early-stopping rule as Method B.
""")

code(r"""
import pretrained_mobilenet as C
if need(RESULTS_DIR / "method_C.json"):
    C.main()
rC = load_json(RESULTS_DIR / "method_C.json")
print(f"params {rC['params_M']:.2f} M (trainable {rC['trainable_params_M']:.2f} M) | best epoch "
      f"{rC['train_info']['best_epoch']} | TEST accuracy {rC['test']['accuracy']:.2f}%  macro-F1 "
      f"{rC['test']['macro_f1']:.3f}  runtime {rC['runtime_ms_cpu']:.2f} ms/img (CPU)  "
      f"train {rC['train_time_min']:.1f} min on {rC['train_device']}")
""")

md("## 5. Experiment 1 - main comparison on the held-out test split")
code(r"""
e1 = E.exp1_main()
display(Markdown((RESULTS_DIR / "results_table.md").read_text()))
show("exp1_training_curves.png"); show("exp1_confusion_matrices.png"); show("exp1_per_class_f1.png", 800)
display(pd.DataFrame(e1["mcnemar"]).T)
""")

md("## 6. Experiment 2 - domain shift: studio photos -> real-world web photos")
code(r"""
models = E.load_models()
e2 = E.exp2_domain_shift(models)
show("exp2_domain_shift.png"); show("exp2_confusion_realworld.png")
""")

md(r"""
## 7. Experiment 3 - small data (10, 25, 50 images per class)
Each method is retrained on a seeded per-class subsample of the training split, with the same
validation split for model selection and the same test split for scoring. The classical method
re-tunes lambda; the networks use a fixed number of optimisation steps (B ~500, C ~150 + head).
""")
code(r"""
e3 = E.exp3_small_data()
display(pd.DataFrame(e3["runs"]).T[["method", "n_per_class", "n_train", "accuracy", "macro_f1", "train_time_min"]])
show("exp3_small_data.png")
""")

md("## 8. Experiment 4 - efficiency (same CPU, batch size 1)")
code(r"""
e4 = E.exp4_efficiency()
display(pd.DataFrame(e4).T)
show("exp4_efficiency.png")
""")

md("## 9. Experiment 5 - error analysis")
code(r"""
e5 = E.exp5_error_analysis(models)
print(json.dumps(e5, indent=1))
show("exp5_failures.png"); show("exp5_gradcam.png")
""")

md(r"""
## 10. The system: all three methods on new inputs + recommendation
`src/system.py` runs the three methods, draws each label/confidence/runtime on the image and names the
recommended method (chosen automatically from the measured macro-F1, real-world accuracy and runtime).
The web app is `app/app.py` (Gradio). Below, the system is run on new images it has never seen
(real-world web photos and, if present, the group's own photos in `data/own_photos/`).
""")
code(r"""
from system import TrashSorter, METHOD_NAMES
sorter = TrashSorter()
display(Markdown(sorter.reason))
demo_items = (own_photo_items()[:3] or []) + load_realworld()[::100][:6 - min(3, len(own_photo_items()))]
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
for ax, (p, y) in zip(axes.ravel(), demo_items):
    img = load_rgb(p); preds = sorter.predict(img)
    ax.imshow(sorter.annotate(img, preds)); ax.set_title(f"true: {CLASSES[y]}"); ax.axis("off")
plt.tight_layout(); fig.savefig(FIG_DIR / "system_demo.png", dpi=90); plt.close(fig)
show("system_demo.png")
""")
code(r"""
LAUNCH_APP = False   # set True to start the web app (in Colab a public *.gradio.live link is printed)
if LAUNCH_APP:
    sys.path.insert(0, str(PROJECT / "app"))
    import app as webapp
    webapp.build_ui().launch(share=IN_COLAB)
""")

md(r"""
## 11. Answer to the research question
The discussion below is filled from the measured numbers (see `docs/` and the report for the full analysis).
""")
code(r"""
s = {k: e1["methods"][k] for k in ["A", "B", "C"]}
rw = e2["results"]["realworld"]
best = max(s, key=lambda k: s[k]["macro_f1"])
lines = [f"* **Best method:** {E.NAMES[best]} - test accuracy {s[best]['accuracy']:.1f}% "
         f"(95% CI {s[best]['acc_ci95'][0]:.1f}-{s[best]['acc_ci95'][1]:.1f}), macro-F1 {s[best]['macro_f1']:.3f}."]
for k in ["A", "B", "C"]:
    lines.append(f"* {E.NAMES[k]}: {s[k]['accuracy']:.1f}% on TrashNet test -> {rw[k]['accuracy']:.1f}% "
                 f"on real-world photos (drop {rw[k]['acc_drop_pts']:.1f} points, keeps "
                 f"{rw[k]['relative_retention'] * 100:.0f}% of its accuracy).")
small = e3["runs"]
lines.append("* With only 10 images per class: " + ", ".join(
    f"{E.NAMES[k]} {small[f'{k}_10']['accuracy']:.1f}%" for k in ["A", "B", "C"]) + ".")
display(Markdown("\n".join(lines)))
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}, "accelerator": "GPU",
                  "colab": {"provenance": [], "gpuType": "T4"}}
out = ROOT / "notebooks" / "CV_Project.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
