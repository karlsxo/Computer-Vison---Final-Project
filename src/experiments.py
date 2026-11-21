"""The five experiments, run on the saved split with the trained models.

  1. Main comparison   - accuracy, macro-F1, confusion matrices, per-class P/R,
                          95 % bootstrap CIs and McNemar significance tests.
  2. Domain shift      - models trained on TrashNet (studio photos) evaluated on an
                          unseen real-world web-photo set (+ own photos if present).
  3. Small data        - every method retrained with 10, 25 and 50 images per class.
  4. Efficiency        - runtime/image, parameters, training time; accuracy-vs-cost.
  5. Error analysis    - most-confused pairs, failure examples, method agreement,
                          Grad-CAM for the CNNs and gradient maps for HOG.

Run:  python src/experiments.py [1 2 3 4 5 figs]      (default: all)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_common as tc   # noqa: E402  (CUDA probe before torch import)
import torch                # noqa: E402
import torch.nn.functional as F   # noqa: E402
import numpy as np          # noqa: E402
import matplotlib           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import hog_svm as A                 # noqa: E402
import scratch_cnn as B             # noqa: E402
import pretrained_mobilenet as C    # noqa: E402
from common import (CLASSES, FIG_DIR, MODELS_DIR, NUM_CLASSES, RESULTS_DIR, ROOT, SEED,  # noqa: E402
                    classification_metrics, labels_of, load_array_cache, load_json,
                    load_realworld, load_rgb, load_split, own_photo_items, resize_square,
                    save_json, set_seed, subsample_per_class)

METHODS = ["A", "B", "C"]
NAMES = {"A": "HOG + linear SVM", "A+": "HOG + colour hist. + SVM (variant)",
         "B": "4-block CNN (scratch)", "C": "MobileNetV3-Large (pretrained)"}
COLORS = {"A": "#d1495b", "A+": "#edae49", "B": "#00798c", "C": "#30638e"}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 140, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 10})


# =============================================================================
# helpers
# =============================================================================
def load_models():
    mA = A.HogSvmClassifier.load(MODELS_DIR / "hog_svm.npz")
    mAv = A.HogSvmClassifier.load(MODELS_DIR / "hog_color_svm.npz")
    mB = B.load_trained()
    mC = C.load_trained()
    return {"A": mA, "A+": mAv, "B": mB, "C": mC}


def predict_items(models, items, tag):
    """Probabilities of every method for a list of (path, label) items."""
    out = {}
    x128 = load_array_cache(items, 128, f"{tag}_128")
    for k in ("A", "A+"):
        F_ = A.extract_features(x128, models[k].cfg)
        s = models[k].scores_from_features(F_)
        out[k] = torch.softmax(torch.from_numpy(s), 1).numpy()
    xb = torch.from_numpy(x128).permute(0, 3, 1, 2)
    out["B"] = torch.softmax(tc.predict_logits(models["B"], xb, B.IMG_SIZE, B.MEAN, B.STD), 1).numpy()
    x224 = torch.from_numpy(load_array_cache(items, 224, f"{tag}_224")).permute(0, 3, 1, 2)
    out["C"] = torch.softmax(tc.predict_logits(models["C"], x224, C.IMG_SIZE, C.MEAN, C.STD), 1).numpy()
    return out


def bootstrap_ci(y, pred, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        m = classification_metrics(y[idx], pred[idx])
        accs.append(m["accuracy"]); f1s.append(m["macro_f1"])
    return {"acc_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
            "f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]}


def mcnemar(y, p1, p2):
    """Exact McNemar test on the discordant pairs."""
    from scipy.stats import binomtest
    c1, c2 = p1 == y, p2 == y
    b = int((c1 & ~c2).sum()); c = int((~c1 & c2).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"only_first_correct": b, "only_second_correct": c, "p_value": float(p)}


def plot_confusions(cms, titles, path, suptitle):
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.3))
    axes = np.atleast_1d(axes)
    for ax, cm, t in zip(axes, cms, titles):
        cm = np.asarray(cm)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(NUM_CLASSES), CLASSES)
        ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
        ax.set_title(t, fontsize=10); ax.grid(False)
    fig.suptitle(suptitle + "  (cell = image count, colour = row-normalised recall)")
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


# =============================================================================
# Experiment 1 - main comparison
# =============================================================================
def exp1_main():
    print("\n=== Experiment 1: main comparison on the TrashNet test split ===")
    split = load_split()
    y = labels_of(split["test"])
    res = {k: load_json(RESULTS_DIR / f"method_{k}.json") for k in METHODS}
    var = res["A"]["variant_color"]
    preds = {k: np.array(res[k]["test_pred"]) for k in METHODS}
    preds["A+"] = np.array(var["test_pred"])
    out = {"n_test": int(len(y)), "methods": {}}
    for k in ["A", "B", "C", "A+"]:
        met = classification_metrics(y, preds[k])
        ci = bootstrap_ci(y, preds[k])
        src = res[k] if k != "A+" else var
        out["methods"][k] = {"name": NAMES[k], "accuracy": met["accuracy"],
                             "macro_f1": met["macro_f1"], **ci, "per_class": met["per_class"],
                             "confusion_matrix": met["confusion_matrix"],
                             "runtime_ms_cpu": src["runtime_ms_cpu"],
                             "runtime_ms_gpu": src.get("runtime_ms_gpu"),
                             "params_M": res[k]["params_M"] if k != "A+" else 0.0,
                             "train_time_min": src["train_time_min"]}
        print(f"  {NAMES[k]:<38} acc {met['accuracy']:6.2f}% "
              f"[{ci['acc_ci95'][0]:.1f}, {ci['acc_ci95'][1]:.1f}]  macro-F1 {met['macro_f1']:.4f}")
    out["mcnemar"] = {f"{a}_vs_{b}": mcnemar(y, preds[a], preds[b])
                      for a, b in [("C", "B"), ("C", "A"), ("B", "A"), ("A+", "A")]}
    for k, v in out["mcnemar"].items():
        print(f"  McNemar {k}: p = {v['p_value']:.2e}")
    save_json(out, RESULTS_DIR / "exp1_main.json")

    # results table (markdown + csv)
    rows = ["| Method | Slot | Top-1 acc. (%) [95% CI] | Macro F1 | Runtime (ms/img, CPU) | Params (M) | Train time (min) |",
            "|---|---|---|---|---|---|---|"]
    slots = {"A": "A. Classical", "B": "B. From scratch", "C": "C. Pretrained", "A+": "A. (variant)"}
    csv = ["method,slot,accuracy,acc_lo,acc_hi,macro_f1,runtime_ms_cpu,params_M,train_time_min"]
    for k in ["A", "B", "C", "A+"]:
        m = out["methods"][k]
        rows.append(f"| {m['name']} | {slots[k]} | {m['accuracy']:.1f} [{m['acc_ci95'][0]:.1f}-"
                    f"{m['acc_ci95'][1]:.1f}] | {m['macro_f1']:.3f} | {m['runtime_ms_cpu']:.1f} | "
                    f"{m['params_M']:.2f} | {m['train_time_min']:.1f} |")
        csv.append(f"{m['name']},{slots[k]},{m['accuracy']:.2f},{m['acc_ci95'][0]:.2f},"
                   f"{m['acc_ci95'][1]:.2f},{m['macro_f1']:.4f},{m['runtime_ms_cpu']:.2f},"
                   f"{m['params_M']:.3f},{m['train_time_min']:.2f}")
    (RESULTS_DIR / "results_table.md").write_text("\n".join(rows) + "\n")
    (RESULTS_DIR / "results_table.csv").write_text("\n".join(csv) + "\n")

    plot_confusions([out["methods"][k]["confusion_matrix"] for k in METHODS],
                    [f"{NAMES[k]}\nacc {out['methods'][k]['accuracy']:.1f}%, F1 "
                     f"{out['methods'][k]['macro_f1']:.3f}" for k in METHODS],
                    FIG_DIR / "exp1_confusion_matrices.png", "TrashNet test split")

    # per-class F1 bars
    fig, ax = plt.subplots(figsize=(9, 3.8))
    w = 0.2
    for i, k in enumerate(["A", "A+", "B", "C"]):
        f1 = [out["methods"][k]["per_class"][c]["f1"] for c in CLASSES]
        ax.bar(np.arange(NUM_CLASSES) + (i - 1.5) * w, f1, w, label=NAMES[k], color=COLORS[k])
    ax.set_xticks(range(NUM_CLASSES), CLASSES)
    ax.set_ylabel("F1 score (0-1)"); ax.set_ylim(0, 1.05); ax.set_xlabel("class")
    ax.set_title("Per-class F1 on the TrashNet test split")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp1_per_class_f1.png"); plt.close(fig)

    # training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    for k, ls in (("B", "-"), ("C", "--")):
        h = res[k]["history"]
        axes[0].plot(h["epoch"], h["train_loss"], ls, color=COLORS[k], alpha=0.6, label=f"{k} train")
        axes[0].plot(h["epoch"], h["val_loss"], ls, color=COLORS[k], label=f"{k} validation", lw=2)
        axes[1].plot(h["epoch"], h["train_acc"], ls, color=COLORS[k], alpha=0.6, label=f"{k} train")
        axes[1].plot(h["epoch"], h["val_acc"], ls, color=COLORS[k], lw=2, label=f"{k} validation")
        be = res[k]["train_info"]["best_epoch"]
        axes[1].axvline(be, color=COLORS[k], ls=":", lw=1)
    axes[0].set_title("Loss (train loss includes augmentation + label smoothing)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("cross-entropy loss (nats)"); axes[0].legend(fontsize=8)
    axes[1].set_title("Accuracy (dotted = selected checkpoint)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy (%)"); axes[1].legend(fontsize=8)
    lh = res["A"]["loss_history"]
    axes[2].plot(np.arange(1, len(lh) + 1), lh, color=COLORS["A"])
    axes[2].set_yscale("log"); axes[2].set_title("Method A: our SVM objective (Adam)")
    axes[2].set_xlabel("gradient-descent iteration"); axes[2].set_ylabel("objective (log scale)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp1_training_curves.png"); plt.close(fig)
    return out


# =============================================================================
# Experiment 2 - domain shift
# =============================================================================
def exp2_domain_shift(models=None):
    print("\n=== Experiment 2: domain shift (studio -> real-world web photos) ===")
    models = models or load_models()
    sets = {"realworld": load_realworld()}
    own = own_photo_items()
    if own:
        sets["own_photos"] = own
    e1 = load_json(RESULTS_DIR / "exp1_main.json")
    out = {"sets": {k: len(v) for k, v in sets.items()}, "results": {}}
    for name, items in sets.items():
        y = labels_of(items)
        probs = predict_items(models, items, name)
        out["results"][name] = {}
        for k in ["A", "A+", "B", "C"]:
            pred = probs[k].argmax(1)
            met = classification_metrics(y, pred)
            ci = bootstrap_ci(y, pred, n_boot=1000)
            base = e1["methods"][k]
            out["results"][name][k] = {
                "accuracy": met["accuracy"], "macro_f1": met["macro_f1"], **ci,
                "per_class": met["per_class"], "confusion_matrix": met["confusion_matrix"],
                "acc_drop_pts": base["accuracy"] - met["accuracy"],
                "relative_retention": met["accuracy"] / base["accuracy"],
                "pred": pred.tolist(), "mean_confidence": float(probs[k].max(1).mean())}
            print(f"  [{name}] {NAMES[k]:<38} acc {met['accuracy']:6.2f}%  F1 {met['macro_f1']:.3f}  "
                  f"(drop {base['accuracy'] - met['accuracy']:+.1f} pts)")
    save_json(out, RESULTS_DIR / "exp2_domain_shift.json")

    r = out["results"]["realworld"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ks = ["A", "A+", "B", "C"]
    xs = np.arange(len(ks))
    ax[0].bar(xs - 0.2, [e1["methods"][k]["accuracy"] for k in ks], 0.4, label="TrashNet test (in-domain)", color="#8fb8de")
    ax[0].bar(xs + 0.2, [r[k]["accuracy"] for k in ks], 0.4, label="real-world web photos (shifted)", color="#d1495b")
    for i, k in enumerate(ks):
        ax[0].text(i + 0.2, r[k]["accuracy"] + 1, f"{r[k]['accuracy']:.0f}", ha="center", fontsize=8)
        ax[0].text(i - 0.2, e1["methods"][k]["accuracy"] + 1, f"{e1['methods'][k]['accuracy']:.0f}", ha="center", fontsize=8)
    ax[0].axhline(100 / 6, color="gray", ls=":", label="chance (16.7 %)")
    short = {"A": "HOG +\nlinear SVM", "A+": "HOG + colour\n(variant)", "B": "4-block CNN\n(scratch)",
             "C": "MobileNetV3-L\n(pretrained)"}
    ax[0].set_xticks(xs, [short[k] for k in ks], fontsize=9)
    ax[0].set_ylabel("top-1 accuracy (%)"); ax[0].set_ylim(0, 105); ax[0].legend(fontsize=8)
    ax[0].set_title("Accuracy before and after the domain shift")
    for k in ks:
        f1 = [r[k]["per_class"][c]["f1"] for c in CLASSES]
        ax[1].plot(CLASSES, f1, "o-", color=COLORS[k], label=NAMES[k])
    ax[1].set_ylabel("F1 score (0-1)"); ax[1].set_xlabel("class"); ax[1].set_ylim(0, 1)
    ax[1].set_title("Per-class F1 on the real-world set")
    ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp2_domain_shift.png", bbox_inches="tight"); plt.close(fig)
    plot_confusions([r[k]["confusion_matrix"] for k in METHODS],
                    [f"{NAMES[k]}\nacc {r[k]['accuracy']:.1f}%" for k in METHODS],
                    FIG_DIR / "exp2_confusion_realworld.png", "Real-world web photos")
    return out


# =============================================================================
# Experiment 3 - small data
# =============================================================================
SMALL_SIZES = (10, 25, 50)


def small_data_schedule(n_train, batch_size=32, target_steps_B=500, target_steps_C=150):
    steps = max(1, math.ceil(n_train / batch_size))
    eB = int(min(250, max(45, math.ceil(target_steps_B / steps))))
    eC = int(min(80, max(12, math.ceil(target_steps_C / steps))))
    return eB, eC


def exp3_small_data(sizes=SMALL_SIZES):
    print("\n=== Experiment 3: small-data regime ===")
    split = load_split()
    path = RESULTS_DIR / "exp3_small_data.json"
    out = load_json(path) if path.exists() else {"runs": {}}
    resA = load_json(RESULTS_DIR / "method_A.json")
    cfgA = resA["selected"]["cfg"]
    ytest = labels_of(split["test"])
    for n in sizes:
        tr = subsample_per_class(split["train"], n)
        eB, eC = small_data_schedule(len(tr))
        for k in METHODS:
            key = f"{k}_{n}"
            if key in out["runs"]:
                continue   # resume support
            set_seed()
            t0 = time.perf_counter()
            if k == "A":
                model, best, _ = A.tune_and_train(tr, split["val"], [cfgA], [1.0, 10.0, 100.0], verbose=False)
                met, pred, _ = A.evaluate(model, split["test"], "test")
                info = {"lam": best["lam"]}
            elif k == "B":
                model, hist, info = B.train(tr, split["val"], tag=f"cnn_n{n}", epochs=eB, patience=10 ** 6,
                                            eval_every=max(1, eB // 25), log_every=10 ** 6, verbose=False)
                met, pred, _ = tc.evaluate(model, split["test"], B.IMG_SIZE, B.MEAN, B.STD, "test_128")
            else:
                model, hist, info = C.train(tr, split["val"], tag=f"mnv3_n{n}", epochs=eC, patience=10 ** 6,
                                            eval_every=max(1, eC // 12), log_every=10 ** 6, verbose=False)
                met, pred, _ = tc.evaluate(model, split["test"], C.IMG_SIZE, C.MEAN, C.STD, "test_224")
            dt = (time.perf_counter() - t0) / 60
            out["runs"][key] = {"method": k, "n_per_class": n, "n_train": len(tr),
                                "accuracy": met["accuracy"], "macro_f1": met["macro_f1"],
                                "train_time_min": dt, "info": {kk: v for kk, v in info.items()
                                                                if not isinstance(v, (list, dict))}}
            print(f"  n={n:>3}/class  {NAMES[k]:<34} acc {met['accuracy']:6.2f}%  "
                  f"F1 {met['macro_f1']:.3f}  ({dt:.1f} min)")
            save_json(out, path)
            del model
    # full-data points from experiment 1
    e1 = load_json(RESULTS_DIR / "exp1_main.json")
    full_n = len(split["train"])
    out["full"] = {k: {"n_train": full_n, "accuracy": e1["methods"][k]["accuracy"],
                       "macro_f1": e1["methods"][k]["macro_f1"]} for k in METHODS}
    save_json(out, path)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k in METHODS:
        xs = [out["runs"][f"{k}_{n}"]["n_train"] for n in sizes] + [full_n]
        for j, metric in enumerate(["accuracy", "macro_f1"]):
            ys = [out["runs"][f"{k}_{n}"][metric] for n in sizes] + [out["full"][k][metric]]
            ax[j].plot(xs, ys, "o-", color=COLORS[k], label=NAMES[k], lw=2)
    for j, (lab, ch) in enumerate([("top-1 accuracy (%)", 100 / 6), ("macro F1 (0-1)", None)]):
        ax[j].set_xscale("log")
        ax[j].set_xticks([60, 150, 300, full_n], ["10/cls\n(60)", "25/cls\n(150)", "50/cls\n(300)", f"all\n({full_n})"])
        ax[j].set_xlabel("training images (log scale)"); ax[j].set_ylabel(lab)
        if ch:
            ax[j].axhline(ch, color="gray", ls=":", label="chance")
        ax[j].legend(fontsize=8)
    ax[0].set_title("Learning curve: accuracy vs training-set size")
    ax[1].set_title("Learning curve: macro F1 vs training-set size")
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp3_small_data.png"); plt.close(fig)
    return out


# =============================================================================
# Experiment 4 - efficiency
# =============================================================================
def exp4_efficiency():
    print("\n=== Experiment 4: efficiency ===")
    e1 = load_json(RESULTS_DIR / "exp1_main.json")
    res = {k: load_json(RESULTS_DIR / f"method_{k}.json") for k in METHODS}
    out = {}
    for k in ["A", "A+", "B", "C"]:
        m = e1["methods"][k]
        out[k] = {"name": NAMES[k], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                  "runtime_ms_cpu": m["runtime_ms_cpu"], "params_M": m["params_M"],
                  "train_time_min": m["train_time_min"],
                  "train_device": res[k[0]].get("train_device", "cpu (NumPy)"),
                  "acc_per_ms": m["accuracy"] / m["runtime_ms_cpu"],
                  "model_size_MB": (m["params_M"] * 4 if k in ("B", "C") else
                                    (MODELS_DIR / ("hog_svm.npz" if k == "A" else "hog_color_svm.npz")).stat().st_size / 1e6)}
        print(f"  {NAMES[k]:<38} {m['runtime_ms_cpu']:7.2f} ms  {m['params_M']:6.3f} M params  "
              f"{m['train_time_min']:6.1f} min train  size {out[k]['model_size_MB']:.2f} MB")
    save_json(out, RESULTS_DIR / "exp4_efficiency.json")
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.1))
    for k, v in out.items():
        s = 60 + 120 * max(v["params_M"], 0.02)
        ax[0].scatter(v["runtime_ms_cpu"], v["accuracy"], s=s, color=COLORS[k], edgecolor="k", alpha=0.85)
        ax[0].annotate(f"{v['params_M']:.2f} M params", (v["runtime_ms_cpu"], v["accuracy"]),
                       textcoords="offset points", xytext=(0, 13), fontsize=8, ha="center")
        ax[1].scatter(v["train_time_min"], v["accuracy"], s=s, color=COLORS[k], edgecolor="k", alpha=0.85)
    ax[0].set_xscale("log"); ax[0].set_xlabel("runtime per image on CPU (ms, log scale, batch 1)")
    ax[0].set_ylabel("top-1 accuracy (%)"); ax[0].set_title("Accuracy vs inference cost (marker area ~ parameters)")
    ax[0].set_xlim(2, 60)
    ax[1].set_xscale("log"); ax[1].set_xlabel("training time (min, log scale; CPU)")
    ax[1].set_ylabel("top-1 accuracy (%)"); ax[1].set_title("Accuracy vs training cost")
    ax[1].set_xlim(0.1, 300)
    for a in ax:
        a.set_ylim(45, 100)
        a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        a.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: ""))
    handles = [Line2D([], [], marker="o", ls="", color=COLORS[k], markeredgecolor="k", markersize=8,
                      label=NAMES[k]) for k in out]
    ax[0].legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp4_efficiency.png"); plt.close(fig)
    return out


# =============================================================================
# Experiment 5 - error analysis
# =============================================================================
def gradcam(model, layer, x, cls=None):
    """Grad-CAM (Selvaraju et al., 2017): weight the feature maps of `layer` by
    the spatially averaged gradient of the class score."""
    feats, grads = {}, {}
    def fwd_hook(m, i, o):
        feats["v"] = o
        o.register_hook(lambda g: grads.__setitem__("v", g))   # gradient w.r.t. the feature map
    h1 = layer.register_forward_hook(fwd_hook)
    model.eval()
    x = x.clone().requires_grad_(True)
    logits = model(x)
    cls = int(logits.argmax(1)) if cls is None else cls
    model.zero_grad()
    logits[0, cls].backward()
    h1.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats["v"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().numpy(), cls


def exp5_error_analysis(models=None):
    print("\n=== Experiment 5: error analysis ===")
    models = models or load_models()
    split = load_split()
    items = split["test"]
    y = labels_of(items)
    e1 = load_json(RESULTS_DIR / "exp1_main.json")
    res = {k: load_json(RESULTS_DIR / f"method_{k}.json") for k in METHODS}
    preds = {k: np.array(res[k]["test_pred"]) for k in METHODS}
    out = {"top_confusions": {}, "agreement": {}}
    for k in METHODS:
        cm = np.array(e1["methods"][k]["confusion_matrix"])
        off = [(int(cm[i, j]), CLASSES[i], CLASSES[j]) for i in range(NUM_CLASSES)
               for j in range(NUM_CLASSES) if i != j]
        off.sort(reverse=True)
        out["top_confusions"][k] = [{"true": t, "pred": p, "count": c} for c, t, p in off[:5]]
        print(f"  {NAMES[k]} top confusions: " +
              ", ".join(f"{t}->{p} ({c})" for c, t, p in off[:4]))
    ok = {k: preds[k] == y for k in METHODS}
    out["agreement"] = {
        "all_correct": int((ok["A"] & ok["B"] & ok["C"]).sum()),
        "all_wrong": int((~ok["A"] & ~ok["B"] & ~ok["C"]).sum()),
        "only_C_correct": int((~ok["A"] & ~ok["B"] & ok["C"]).sum()),
        "C_wrong_but_B_or_A_right": int((~ok["C"] & (ok["A"] | ok["B"])).sum()),
        "A_right_C_wrong": int((ok["A"] & ~ok["C"]).sum()),
        "n": int(len(y))}
    print(f"  agreement: {out['agreement']}")
    # majority-vote ensemble (for discussion only)
    vote = np.stack([preds[k] for k in METHODS], 1)
    maj = np.array([np.bincount(v, minlength=NUM_CLASSES).argmax() if len(set(v)) < 3 else v[2]
                    for v in vote])
    out["majority_vote_accuracy"] = classification_metrics(y, maj)["accuracy"]
    save_json(out, RESULTS_DIR / "exp5_error_analysis.json")

    # failure examples: 6 misclassified test images per method
    imgs = load_array_cache(items, 224, "test_224")
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 6, figsize=(15, 8))
    for r, k in enumerate(METHODS):
        wrong = np.where(~ok[k])[0]
        pick = rng.choice(wrong, size=min(6, len(wrong)), replace=False) if len(wrong) else []
        for c in range(6):
            ax = axes[r, c]; ax.axis("off")
            if c < len(pick):
                i = pick[c]
                ax.imshow(imgs[i])
                ax.set_title(f"true: {CLASSES[y[i]]}\npred: {CLASSES[preds[k][i]]}", fontsize=9,
                             color="#b00020")
        axes[r, 0].text(-0.15, 0.5, NAMES[k], transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=10, weight="bold")
    fig.suptitle("Failure examples on the TrashNet test split (one row per method)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp5_failures.png"); plt.close(fig)

    # Grad-CAM (B, C) and HOG gradient magnitude (A) on the same images
    sel = []
    for c in range(NUM_CLASSES):
        idx = np.where(y == c)[0]
        sel.append(int(idx[0]))
    x128 = load_array_cache(items, 128, "test_128")
    fig, axes = plt.subplots(4, NUM_CLASSES, figsize=(2.6 * NUM_CLASSES, 10.6))
    mB, mC = models["B"], models["C"]
    for j, i in enumerate(sel):
        axes[0, j].imshow(imgs[i]); axes[0, j].set_title(f"{CLASSES[y[i]]}", fontsize=10)
        mag, _ = A.image_gradients(A.rgb_to_gray(x128[i]))
        axes[1, j].imshow(mag, cmap="magma")
        axes[1, j].set_title(f"A: {CLASSES[preds['A'][i]]}", fontsize=9,
                             color="green" if preds["A"][i] == y[i] else "#b00020")
        xb = tc.to_float(torch.from_numpy(x128[i]).permute(2, 0, 1)[None], B.MEAN, B.STD)
        camB, pb = gradcam(mB, mB.features[3], xb)
        axes[2, j].imshow(x128[i]); axes[2, j].imshow(camB, cmap="jet", alpha=0.45)
        axes[2, j].set_title(f"B: {CLASSES[pb]}", fontsize=9, color="green" if pb == y[i] else "#b00020")
        xc = tc.to_float(torch.from_numpy(imgs[i]).permute(2, 0, 1)[None], C.MEAN, C.STD)
        camC, pc = gradcam(mC, mC.net.features[-1], xc)
        axes[3, j].imshow(imgs[i]); axes[3, j].imshow(camC, cmap="jet", alpha=0.45)
        axes[3, j].set_title(f"C: {CLASSES[pc]}", fontsize=9, color="green" if pc == y[i] else "#b00020")
    for r, lab in enumerate(["input", "A: HOG gradient magnitude", "B: Grad-CAM", "C: Grad-CAM"]):
        axes[r, 0].set_ylabel(lab, fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("What each method looks at (green title = correct prediction, red = wrong)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp5_gradcam.png"); plt.close(fig)
    return out


# =============================================================================
# Dataset figures (for the report)
# =============================================================================
def dataset_figures():
    split = load_split()
    js = load_json(ROOT / "splits" / "split.json")
    counts = js["stats"]["counts"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bottom = np.zeros(NUM_CLASSES)
    for k, col in zip(["train", "val", "test"], ["#30638e", "#edae49", "#d1495b"]):
        v = np.array([counts[k][c] for c in CLASSES])
        ax.bar(CLASSES, v, bottom=bottom, label=f"{k} ({v.sum()})", color=col)
        bottom += v
    ax.set_ylabel("number of images"); ax.set_xlabel("class")
    ax.set_title("TrashNet after cleaning: class counts per split (70/15/15)")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "data_class_counts.png"); plt.close(fig)

    rw = load_realworld()
    fig, axes = plt.subplots(2, NUM_CLASSES, figsize=(2.5 * NUM_CLASSES, 5.4))
    rng = np.random.default_rng(1)
    for c in range(NUM_CLASSES):
        tn = [p for p, yy in split["train"] if yy == c]
        rr = [p for p, yy in rw if yy == c]
        for r, (lst, src) in enumerate([(tn, "TrashNet"), (rr, "real-world")]):
            p = lst[rng.integers(len(lst))]
            axes[r, c].imshow(resize_square(load_rgb(p), 224))
            axes[r, c].set_title(f"{CLASSES[c]} ({src})", fontsize=9)
            axes[r, c].axis("off")
    fig.suptitle("Domain shift: TrashNet studio photos (top) vs web photos (bottom)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "data_domain_examples.png"); plt.close(fig)

    x = torch.from_numpy(load_array_cache(split["train"][:6], 128, "aug_demo")).permute(0, 3, 1, 2).float() / 255
    torch.manual_seed(0)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.3))
    for c in range(6):
        axes[0, c].imshow(x[c].permute(1, 2, 0)); axes[0, c].set_title("original", fontsize=9)
        for r in (1, 2):
            a = tc.gpu_augment(x[c:c + 1], 128)[0]
            axes[r, c].imshow(a.permute(1, 2, 0)); axes[r, c].set_title("augmented", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Our training-time augmentation (zoom crop, rotation, flips, colour jitter, erasing)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "data_augmentation.png"); plt.close(fig)


def main(which):
    t0 = time.perf_counter()
    models = None
    if "figs" in which:
        dataset_figures()
    if "1" in which:
        exp1_main()
    if any(k in which for k in ("2", "5")):
        models = load_models()
    if "2" in which:
        exp2_domain_shift(models)
    if "4" in which:
        exp4_efficiency()
    if "5" in which:
        exp5_error_analysis(models)
    if "3" in which:
        exp3_small_data()
    print(f"\nDone in {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1:] or ["figs", "1", "2", "4", "5", "3"])
