"""Method A (classical): HOG features + linear SVM, both written from scratch in NumPy.

Core algorithm written by the group (no library call for the main step):
  * HOG (Dalal & Triggs, 2005): gradients with [-1, 0, 1] kernels, magnitude and
    unsigned orientation, per-cell orientation histograms (magnitude-weighted,
    optional linear interpolation between neighbouring bins), overlapping 2x2
    blocks with L2-Hys normalisation, concatenation.
  * Linear SVM: one-vs-rest, L2-regularised squared hinge loss
    (the "L2-loss SVM" also used by LIBLINEAR), optimised with our own
    full-batch Adam gradient descent.
  * Optional extension (reported as a separate variant, not the main method):
    an HSV colour histogram appended to the HOG vector, with our own RGB->HSV.

scikit-image / scikit-learn are only imported inside the sanity checks to verify
that our implementation agrees with a reference.

Run:  python src/hog_svm.py            (sanity checks + tuning + test evaluation)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, FIG_DIR, MODELS_DIR, NUM_CLASSES, RESULTS_DIR, SEED,
                    classification_metrics, labels_of, load_array_cache, load_split,
                    save_json, set_seed, softmax, time_per_image)

IMG_SIZE = 128


# =============================================================================
# 1. HOG feature extraction
# =============================================================================
def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float HxW in [0, 1] using ITU-R BT.601 luma weights."""
    img = img.astype(np.float64) / 255.0
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


def image_gradients(gray: np.ndarray):
    """Centred differences with the [-1, 0, 1] kernel (border gradients = 0)."""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.hypot(gx, gy)
    orientation = np.rad2deg(np.arctan2(gy, gx)) % 180.0   # unsigned, [0, 180)
    return magnitude, orientation


def cell_histograms(magnitude, orientation, cell=8, nbins=9, soft=True):
    """Orientation histogram for every cell -> (n_cells_y, n_cells_x, nbins)."""
    ncy, ncx = magnitude.shape[0] // cell, magnitude.shape[1] // cell
    mag = magnitude[:ncy * cell, :ncx * cell]
    ang = orientation[:ncy * cell, :ncx * cell]
    # index of the cell each pixel belongs to
    cy = np.arange(ncy * cell) // cell
    cx = np.arange(ncx * cell) // cell
    cell_idx = (cy[:, None] * ncx + cx[None, :]).ravel()
    bin_width = 180.0 / nbins
    m = mag.ravel()
    if soft:
        # linear interpolation between the two nearest bin centres
        pos = ang.ravel() / bin_width - 0.5
        lo = np.floor(pos)
        w_hi = pos - lo
        b_lo = lo.astype(int) % nbins
        b_hi = (b_lo + 1) % nbins
        hist = np.bincount(cell_idx * nbins + b_lo, weights=m * (1 - w_hi),
                           minlength=ncy * ncx * nbins)
        hist += np.bincount(cell_idx * nbins + b_hi, weights=m * w_hi,
                            minlength=ncy * ncx * nbins)
    else:
        # hard assignment: bin i covers [i*w, (i+1)*w)
        b = np.minimum((ang.ravel() // bin_width).astype(int), nbins - 1)
        hist = np.bincount(cell_idx * nbins + b, weights=m, minlength=ncy * ncx * nbins)
    hist = hist.reshape(ncy, ncx, nbins) / (cell * cell)
    return hist


def normalize_blocks(hist, block=2, eps=1e-5, method="L2-Hys"):
    """Group cells into overlapping block x block windows (stride 1 cell) and
    normalise each block vector."""
    win = sliding_window_view(hist, (block, block), axis=(0, 1))   # (nby, nbx, nbins, b, b)
    win = win.transpose(0, 1, 3, 4, 2)                              # (nby, nbx, b, b, nbins)
    v = win.reshape(win.shape[0], win.shape[1], -1)
    if method == "L2-Hys":
        v = v / np.sqrt((v ** 2).sum(-1, keepdims=True) + eps ** 2)
        v = np.minimum(v, 0.2)
        v = v / np.sqrt((v ** 2).sum(-1, keepdims=True) + eps ** 2)
    elif method == "L2":
        v = v / np.sqrt((v ** 2).sum(-1, keepdims=True) + eps ** 2)
    else:
        raise ValueError(method)
    return v


def hog(gray: np.ndarray, cell=8, block=2, nbins=9, soft=True, block_norm="L2-Hys"):
    """Full HOG descriptor of a 2-D float image."""
    magnitude, orientation = image_gradients(gray)
    hist = cell_histograms(magnitude, orientation, cell, nbins, soft)
    return normalize_blocks(hist, block, method=block_norm).ravel()


# =============================================================================
# 2. Colour histogram (optional extension)
# =============================================================================
def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float HSV with H in [0, 1), S, V in [0, 1]."""
    rgb = img.astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    delta = mx - mn
    h = np.zeros_like(mx)
    nz = delta > 1e-12
    rm = nz & (mx == r)
    gm = nz & (mx == g) & ~rm
    bm = nz & ~rm & ~gm
    h[rm] = ((g - b)[rm] / delta[rm]) % 6
    h[gm] = (b - r)[gm] / delta[gm] + 2
    h[bm] = (r - g)[bm] / delta[bm] + 4
    h = h / 6.0
    s = np.where(mx > 1e-12, delta / np.maximum(mx, 1e-12), 0.0)
    return np.stack([h, s, mx], -1)


def color_histogram(img: np.ndarray, bins=(12, 4, 4)) -> np.ndarray:
    """Joint HSV histogram, L1-normalised then square-rooted (Hellinger)."""
    hsv = rgb_to_hsv(img)
    idx = [np.minimum((hsv[..., k] * bins[k]).astype(int), bins[k] - 1) for k in range(3)]
    flat = (idx[0] * bins[1] + idx[1]) * bins[2] + idx[2]
    hist = np.bincount(flat.ravel(), minlength=int(np.prod(bins))).astype(np.float64)
    hist /= hist.sum()
    return np.sqrt(hist)


# =============================================================================
# 3. Feature pipeline
# =============================================================================
DEFAULT_CFG = {"cell": 8, "block": 2, "nbins": 9, "soft": True, "color": False}


def extract_one(img: np.ndarray, cfg: dict) -> np.ndarray:
    f = hog(rgb_to_gray(img), cfg["cell"], cfg["block"], cfg["nbins"], cfg["soft"])
    if cfg.get("color"):
        f = np.concatenate([f, color_histogram(img)])
    return f.astype(np.float32)


def extract_features(images: np.ndarray, cfg: dict) -> np.ndarray:
    return np.stack([extract_one(im, cfg) for im in images])


# =============================================================================
# 4. Linear SVM (one-vs-rest, squared hinge, Adam) - written from scratch
# =============================================================================
class LinearSVM:
    """Multi-class linear SVM, one-vs-rest.

    Objective (for every class k, solved jointly):
        lam/2 * ||W||^2 + mean_i  c_i * max(0, 1 - t_ik * (x_i . w_k + b_k))^2
    where t_ik = +1 if y_i == k else -1, and c_i is an optional class weight.
    """

    def __init__(self, lam=1.0, lr=0.01, iters=300, class_weight=True, seed=SEED):
        self.lam, self.lr, self.iters = lam, lr, iters
        self.class_weight = class_weight
        self.seed = seed
        self.loss_history = []

    def _loss_grad(self, X, T, c):
        scores = X @ self.W + self.b                     # (n, K)
        margin = np.maximum(0.0, 1.0 - T * scores)      # (n, K)
        n = X.shape[0]
        loss = (c[:, None] * margin ** 2).sum() / n + 0.5 * self.lam * (self.W ** 2).sum()
        dscores = (-2.0 / n) * c[:, None] * margin * T   # d loss / d scores (float32)
        gW = X.T @ dscores + self.lam * self.W
        gb = dscores.sum(0)
        return loss, gW, gb

    def fit(self, X, y):
        # everything in float32: mixing float32 data with float64 weights makes
        # NumPy up-cast the whole data matrix on every product (~12x slower)
        X = np.ascontiguousarray(X, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        K = NUM_CLASSES
        self.W = rng.normal(0, 1e-3, size=(d, K)).astype(np.float32)
        self.b = np.zeros(K, np.float32)
        T = -np.ones((n, K), np.float32)
        T[np.arange(n), y] = 1.0
        if self.class_weight:   # "balanced": n / (K * count_k)
            counts = np.bincount(y, minlength=K).astype(float)
            cw = n / (K * np.maximum(counts, 1))
            c = cw[y].astype(np.float32)
        else:
            c = np.ones(n, np.float32)
        # Adam optimiser (Kingma & Ba, 2015), written out explicitly
        b1, b2, eps = 0.9, 0.999, 1e-8
        mW, vW = np.zeros_like(self.W), np.zeros_like(self.W)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.loss_history = []
        for t in range(1, self.iters + 1):
            lr = np.float32(self.lr * 0.5 * (1 + np.cos(np.pi * (t - 1) / self.iters)))  # cosine decay
            loss, gW, gb = self._loss_grad(X, T, c)
            self.loss_history.append(float(loss))
            mW = b1 * mW + (1 - b1) * gW
            vW = b2 * vW + (1 - b2) * gW ** 2
            mb = b1 * mb + (1 - b1) * gb
            vb = b2 * vb + (1 - b2) * gb ** 2
            self.W -= lr * (mW / (1 - b1 ** t)) / (np.sqrt(vW / (1 - b2 ** t)) + eps)
            self.b -= lr * (mb / (1 - b1 ** t)) / (np.sqrt(vb / (1 - b2 ** t)) + eps)
        return self

    def decision_function(self, X):
        return X @ self.W + self.b

    def predict(self, X):
        return self.decision_function(X).argmax(1)


# =============================================================================
# 5. Full classifier (features + standardisation + SVM)
# =============================================================================
class HogSvmClassifier:
    def __init__(self, cfg=None, lam=1.0, iters=300):
        self.cfg = dict(DEFAULT_CFG if cfg is None else cfg)
        self.svm = LinearSVM(lam=lam, iters=iters)

    def fit_features(self, F, y):
        self.mean = F.mean(0).astype(np.float32)
        self.std = (F.std(0) + 1e-6).astype(np.float32)
        self.svm.fit((F - self.mean) / self.std, y)
        return self

    def scores_from_features(self, F):
        return self.svm.decision_function((F - self.mean) / self.std)

    def predict_proba(self, img: np.ndarray) -> np.ndarray:
        """img: uint8 HxWx3 at IMG_SIZE. Returns softmax of SVM scores
        (a confidence *score*, not a calibrated probability)."""
        f = extract_one(img, self.cfg)[None]
        return softmax(self.scores_from_features(f))[0]

    def save(self, path: Path):
        np.savez(path, W=self.svm.W, b=self.svm.b, mean=self.mean, std=self.std,
                 cfg=np.array([str(self.cfg)]), lam=self.svm.lam)

    @classmethod
    def load(cls, path: Path):
        import ast
        z = np.load(path, allow_pickle=False)
        m = cls(cfg=ast.literal_eval(str(z["cfg"][0])), lam=float(z["lam"]))
        m.svm.W, m.svm.b, m.mean, m.std = z["W"], z["b"], z["mean"], z["std"]
        return m


# =============================================================================
# 6. Sanity checks (required by the rubric)
# =============================================================================
def sanity_checks(save_fig=True) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}

    # (a) synthetic images: the dominant orientation bin must match the edge direction
    size = 64
    tests = {}
    vbar = np.zeros((size, size)); vbar[:, 28:36] = 1.0      # vertical edges -> gradient along x -> 0 deg
    hbar = np.zeros((size, size)); hbar[28:36, :] = 1.0      # horizontal edges -> gradient along y -> 90 deg
    yy, xx = np.mgrid[:size, :size]
    diag = (xx + yy > size).astype(float)                    # 45-degree edge -> gradient at 45 deg
    for name, im, expected_deg in [("vertical bar", vbar, 0.0), ("horizontal bar", hbar, 90.0),
                                   ("diagonal edge", diag, 45.0)]:
        mag, ori = image_gradients(im)
        h = cell_histograms(mag, ori, cell=8, nbins=9, soft=False).sum((0, 1))
        peak_bin = int(h.argmax())
        peak_range = (peak_bin * 20.0, peak_bin * 20.0 + 20.0)
        ok = peak_range[0] <= expected_deg < peak_range[1]
        tests[name] = {"expected_deg": expected_deg, "peak_bin": peak_bin,
                       "peak_range_deg": peak_range, "pass": bool(ok), "hist": h.tolist()}
    out["synthetic"] = tests

    # (b) agreement with the scikit-image reference implementation (hard binning, L2-Hys)
    try:
        from skimage.feature import hog as sk_hog
        split = load_split()
        imgs = load_array_cache(split["test"][:8], IMG_SIZE, "test")
        diffs, corrs = [], []
        for im in imgs:
            g = rgb_to_gray(im)
            ours = hog(g, 8, 2, 9, soft=False)
            ref = sk_hog(g, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                         block_norm="L2-Hys", feature_vector=True)
            diffs.append(float(np.abs(ours - ref).max()))
            corrs.append(float(np.corrcoef(ours, ref)[0, 1]))
        out["skimage_reference"] = {"max_abs_diff": max(diffs), "min_correlation": min(corrs),
                                    "n_images": len(imgs), "pass": bool(min(corrs) > 0.99)}
    except ImportError:
        out["skimage_reference"] = {"skipped": "scikit-image not installed"}

    # (c) the SVM must separate a toy dataset (3 Gaussian blobs in 2-D)
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [4, 0], [2, 4]])
    Xt = np.concatenate([rng.normal(c, 0.7, size=(100, 2)) for c in centers])
    yt = np.repeat(np.arange(3), 100)
    # LinearSVM is written for NUM_CLASSES outputs; extra classes simply have no positives
    svm = LinearSVM(lam=1e-3, lr=0.05, iters=500, class_weight=False).fit(Xt, yt)
    acc = float((svm.predict(Xt) == yt).mean() * 100)
    out["toy_svm"] = {"train_accuracy": acc, "pass": acc > 97}

    if save_fig:
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
        for k, (name, im) in enumerate([("vertical bar", vbar), ("horizontal bar", hbar),
                                        ("diagonal edge", diag)]):
            h = np.array(tests[name]["hist"])
            ax[k].bar(np.arange(9) * 20 + 10, h / h.max(), width=18)
            ax[k].set_title(f"{name}: peak {tests[name]['peak_range_deg'][0]:.0f}-"
                            f"{tests[name]['peak_range_deg'][1]:.0f} deg")
            ax[k].set_xlabel("gradient orientation (degrees)")
            ax[k].set_ylabel("normalised magnitude")
            ins = ax[k].inset_axes([0.62, 0.55, 0.35, 0.4]); ins.imshow(im, cmap="gray")
            ins.set_xticks([]); ins.set_yticks([])
        g1, g2 = np.meshgrid(np.linspace(-2, 6, 300), np.linspace(-2, 6.5, 300))
        Z = svm.predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
        ax[3].contourf(g1, g2, Z, alpha=0.25, levels=[-0.5, 0.5, 1.5, 2.5], cmap="viridis")
        ax[3].scatter(Xt[:, 0], Xt[:, 1], c=yt, s=8, cmap="viridis")
        ax[3].set_title(f"our linear SVM on toy data: {acc:.1f}% train acc.")
        ax[3].set_xlabel("feature 1 (a.u.)"); ax[3].set_ylabel("feature 2 (a.u.)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "A_sanity_checks.png", dpi=130)
        plt.close(fig)
    return out


# =============================================================================
# 7. Training / tuning / evaluation
# =============================================================================
def tune_and_train(train_items, val_items, cfg_grid=None, lam_grid=None, iters=300,
                   verbose=True):
    """Pick feature config and regularisation by validation macro F1, then return
    the selected model trained on the training split only."""
    cfg_grid = cfg_grid or [DEFAULT_CFG]
    lam_grid = lam_grid or [1.0, 3.0, 10.0, 30.0, 100.0]
    Xtr_img = load_array_cache(train_items, IMG_SIZE, "trainA")
    Xva_img = load_array_cache(val_items, IMG_SIZE, "valA")
    ytr, yva = labels_of(train_items), labels_of(val_items)
    log, best = [], None
    for cfg in cfg_grid:
        t0 = time.perf_counter()
        Ftr = extract_features(Xtr_img, cfg)
        Fva = extract_features(Xva_img, cfg)
        feat_time = time.perf_counter() - t0
        for lam in lam_grid:
            t1 = time.perf_counter()
            m = HogSvmClassifier(cfg, lam=lam, iters=iters).fit_features(Ftr, ytr)
            fit_time = time.perf_counter() - t1
            met = classification_metrics(yva, m.scores_from_features(Fva).argmax(1))
            row = {"cfg": cfg, "lam": lam, "val_acc": met["accuracy"],
                   "val_macro_f1": met["macro_f1"], "dim": int(Ftr.shape[1]),
                   "train_time_s": feat_time + fit_time}
            log.append(row)
            if verbose:
                print(f"  cfg={cfg} lam={lam:g} dim={Ftr.shape[1]} "
                      f"val_acc={met['accuracy']:.1f} val_F1={met['macro_f1']:.3f}")
            if best is None or row["val_macro_f1"] > best[0]["val_macro_f1"]:
                best = (row, m)
    return best[1], best[0], log


def evaluate(model: HogSvmClassifier, items, tag="test"):
    imgs = load_array_cache(items, IMG_SIZE, tag)
    F = extract_features(imgs, model.cfg)
    scores = model.scores_from_features(F)
    pred = scores.argmax(1)
    return classification_metrics(labels_of(items), pred), pred, softmax(scores)


def main():
    set_seed()
    print("=== Method A: HOG + linear SVM (NumPy) ===")
    print("Sanity checks ...")
    sc = sanity_checks()
    for k, v in sc.items():
        print(f"  {k}: {v if k != 'synthetic' else {n: t['pass'] for n, t in v.items()}}")

    split = load_split()
    cell_opts = [8, 16]
    main_grid = [dict(DEFAULT_CFG, cell=c, soft=s) for c in cell_opts for s in (True, False)]
    print("Tuning HOG + SVM on the validation split ...")
    model, best, log = tune_and_train(split["train"], split["val"], main_grid)
    print(f"Selected: {best}")

    # Final training-time measurement for the selected configuration (features + SVM)
    t0 = time.perf_counter()
    Xtr = load_array_cache(split["train"], IMG_SIZE, "trainA")
    Ftr = extract_features(Xtr, model.cfg)
    model = HogSvmClassifier(model.cfg, lam=best["lam"]).fit_features(Ftr, labels_of(split["train"]))
    train_min = (time.perf_counter() - t0) / 60
    model.save(MODELS_DIR / "hog_svm.npz")

    test_met, pred, proba = evaluate(model, split["test"], "test")
    test_imgs = load_array_cache(split["test"], IMG_SIZE, "test")
    rt = time_per_image(lambda im: model.predict_proba(im), test_imgs)
    print(f"TEST accuracy={test_met['accuracy']:.2f}%  macro-F1={test_met['macro_f1']:.4f}  "
          f"runtime={rt:.2f} ms/img (CPU)  train={train_min:.2f} min")

    # Extension variant: HOG + colour histogram (reported separately)
    print("Variant: HOG + HSV colour histogram ...")
    var_grid = [dict(model.cfg, color=True)]
    vmodel, vbest, vlog = tune_and_train(split["train"], split["val"], var_grid)
    t0 = time.perf_counter()
    vmodel = HogSvmClassifier(vmodel.cfg, lam=vbest["lam"]).fit_features(
        extract_features(Xtr, vmodel.cfg), labels_of(split["train"]))
    vtrain_min = (time.perf_counter() - t0) / 60
    vmodel.save(MODELS_DIR / "hog_color_svm.npz")
    vtest, vpred, _ = evaluate(vmodel, split["test"], "test")
    vrt = time_per_image(lambda im: vmodel.predict_proba(im), test_imgs)
    print(f"VARIANT TEST accuracy={vtest['accuracy']:.2f}%  macro-F1={vtest['macro_f1']:.4f} "
          f"runtime={vrt:.2f} ms")

    save_json({
        "method": "HOG + linear SVM", "slot": "A. Classical",
        "sanity_checks": sc, "tuning_log": log, "selected": best,
        "test": test_met, "test_pred": pred.tolist(),
        "runtime_ms_cpu": rt, "params_M": 0.0, "train_time_min": train_min,
        "feature_dim": int(Ftr.shape[1]), "loss_history": model.svm.loss_history,
        "variant_color": {"selected": vbest, "test": vtest, "test_pred": vpred.tolist(),
                          "runtime_ms_cpu": vrt, "train_time_min": vtrain_min,
                          "tuning_log": vlog},
    }, RESULTS_DIR / "method_A.json")


if __name__ == "__main__":
    main()
