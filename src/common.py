"""Shared utilities used by all three methods.

Everything that must be identical across methods for a fair comparison lives here:
class names, the saved split, image loading, metric code and the timing protocol.
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRASHNET_DIR = RAW_DIR / "dataset-resized"
SPLIT_FILE = ROOT / "splits" / "split.json"
REALWORLD_FILE = ROOT / "splits" / "realworld.json"
OWN_PHOTOS_DIR = DATA_DIR / "own_photos"          # optional: <class>/<img>.jpg
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
CACHE_DIR = ROOT / "cache"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for _d in (MODELS_DIR, RESULTS_DIR, FIG_DIR, CACHE_DIR, SPLIT_FILE.parent):
    _d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    """Fix every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------------
# Split handling
# ----------------------------------------------------------------------------
def load_split(path: Path = SPLIT_FILE) -> dict:
    """Return {'train': [(relpath, label), ...], 'val': [...], 'test': [...]}.

    Paths are stored relative to the project root so the split works on any
    machine (local or Colab).
    """
    with open(path) as f:
        js = json.load(f)
    return {k: [(p, int(y)) for p, y in v] for k, v in js["splits"].items()}


def load_realworld(path: Path = REALWORLD_FILE) -> list[tuple[str, int]]:
    with open(path) as f:
        js = json.load(f)
    return [(p, int(y)) for p, y in js["items"]]


def own_photo_items() -> list[tuple[str, int]]:
    """Optional group-collected photos in data/own_photos/<class>/*.jpg."""
    items = []
    if OWN_PHOTOS_DIR.exists():
        for c in CLASSES:
            d = OWN_PHOTOS_DIR / c
            if d.exists():
                for p in sorted(d.iterdir()):
                    if p.suffix.lower() in IMG_EXTS:
                        items.append((str(p.relative_to(ROOT)).replace("\\", "/"), CLASS_TO_IDX[c]))
    return items


def subsample_per_class(items, n_per_class: int, seed: int = SEED):
    """Pick n images per class (used by the small-data experiment)."""
    rng = np.random.default_rng(seed)
    out = []
    for c in range(NUM_CLASSES):
        cls_items = [it for it in items if it[1] == c]
        idx = rng.permutation(len(cls_items))[:n_per_class]
        out.extend(cls_items[i] for i in sorted(idx))
    return out


# ----------------------------------------------------------------------------
# Image loading (same source images for all methods)
# ----------------------------------------------------------------------------
def load_rgb(path: str | Path) -> Image.Image:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    img = Image.open(p)
    img = img.convert("RGB")
    return img


def resize_square(img: Image.Image, size: int) -> np.ndarray:
    """Resize to size x size (bilinear) and return an HxWx3 uint8 array."""
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)


def load_array_cache(items, size: int, tag: str) -> np.ndarray:
    """Load all images of a list at a fixed size, caching to disk as uint8.

    Returns N x size x size x 3 uint8. The cache key includes the file list so a
    changed split never silently reuses stale data.
    """
    import hashlib
    key = hashlib.md5(("|".join(p for p, _ in items) + str(size)).encode()).hexdigest()[:10]
    f = CACHE_DIR / f"{tag}_{size}_{key}.npy"
    if f.exists():
        return np.load(f)
    arr = np.stack([resize_square(load_rgb(p), size) for p, _ in items])
    np.save(f, arr)
    return arr


def labels_of(items) -> np.ndarray:
    return np.array([y for _, y in items], dtype=np.int64)


# ----------------------------------------------------------------------------
# Metrics (one implementation used for every method)
# ----------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, n: int = NUM_CLASSES) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm  # rows = true class, columns = predicted class


def classification_metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(0).astype(float)
    true_pos = cm.sum(1).astype(float)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = true_pos > 0  # macro average over classes present in y_true
    return {
        "accuracy": float(tp.sum() / cm.sum() * 100.0),
        "macro_f1": float(f1[present].mean()),
        "per_class": {
            c: {"precision": float(precision[i]), "recall": float(recall[i]),
                "f1": float(f1[i]), "support": int(true_pos[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": cm.tolist(),
    }


# ----------------------------------------------------------------------------
# Timing protocol (shared by all methods)
# ----------------------------------------------------------------------------
def time_per_image(predict_one, images, warmup: int = 10, n: int = 200) -> float:
    """Average wall-clock ms per image, batch size 1, after warm-up passes.

    `predict_one` takes one preprocessed image (whatever that method's input
    is) and returns a prediction. Preprocessing that is part of the method
    (e.g. HOG extraction) must happen inside `predict_one`.
    """
    n = min(n, len(images))
    for i in range(min(warmup, len(images))):
        predict_one(images[i])
    t0 = time.perf_counter()
    for i in range(n):
        predict_one(images[i])
    return (time.perf_counter() - t0) / n * 1000.0


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
