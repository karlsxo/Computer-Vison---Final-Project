"""Experiment 6: mixed-domain training - does adding real-world photos to training
fix the domain-shift problem found in Experiment 2?

All three methods are retrained, with exactly the same architectures and
hyper-parameters as before, on TrashNet train + web photos (splits/mixed.json,
built by prepare_mixed.py), and evaluated on the UNCHANGED test sets:
  * TrashNet test split (376 studio photos)  - same as Experiment 1
  * real-world test set (600 web photos)     - same as Experiment 2
The trained models are saved in models/mixed/ and are the ones the phone app uses.

A confidence threshold is also calibrated on the validation split, so the app can
answer "not sure - retake the photo" instead of guessing.

Run:  python src/mixed_domain.py            (resumes: finished methods are skipped)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_common as tc   # noqa: E402  (CUDA probe before torch import)
import numpy as np          # noqa: E402
import torch                # noqa: E402
import matplotlib           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import hog_svm as A                 # noqa: E402
import scratch_cnn as B             # noqa: E402
import pretrained_mobilenet as C    # noqa: E402
from common import (CLASSES, FIG_DIR, MODELS_DIR, RESULTS_DIR, classification_metrics,  # noqa: E402
                    labels_of, load_array_cache, load_json, load_realworld, load_split,
                    save_json, set_seed, softmax, time_per_image)
from prepare_mixed import load_mixed   # noqa: E402

MIXED_DIR = MODELS_DIR / "mixed"
MIXED_DIR.mkdir(exist_ok=True)
OUT = RESULTS_DIR / "exp6_mixed_domain.json"
NAMES = {"A": "HOG + linear SVM", "B": "4-block CNN (scratch)", "C": "MobileNetV3-Large (pretrained)"}
COLORS = {"A": "#d1495b", "B": "#00798c", "C": "#30638e"}


def bootstrap(y, pred, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    accs = [float((y[i] == pred[i]).mean() * 100) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def score(y, probs):
    pred = probs.argmax(1)
    m = classification_metrics(y, pred)
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "acc_ci95": bootstrap(y, pred),
            "per_class": m["per_class"], "confusion_matrix": m["confusion_matrix"],
            "pred": pred.tolist(), "confidence": probs.max(1).tolist()}


def probs_A(model, items, tag):
    F = A.extract_features(load_array_cache(items, 128, tag), model.cfg)
    return softmax(model.scores_from_features(F))


def probs_net(model, items, size, mean, std, tag):
    x = tc.load_uint8(items, size, tag)
    return torch.softmax(tc.predict_logits(model, x, size, mean, std), 1).numpy()


def train_method(k, mixed):
    """Train method k on the mixed data; returns (predict_fn(items, tag) -> probs, info)."""
    t0 = time.perf_counter()
    if k == "A":
        cfg = load_json(RESULTS_DIR / "method_A.json")["selected"]["cfg"]
        model, best, log = A.tune_and_train(mixed["train"], mixed["val"], [cfg], [3.0, 10.0, 30.0, 100.0])
        model.save(MIXED_DIR / "hog_svm.npz")
        info = {"lambda": best["lam"], "val_acc": best["val_acc"], "tuning_log": log}
        fn = lambda items, tag: probs_A(model, items, f"{tag}_128")
        params = 0.0
        rt = time_per_image(lambda im: model.predict_proba(im), load_array_cache(load_split()["test"], 128, "test_128"))
    elif k == "B":
        model, hist, info = B.train(mixed["train"], mixed["val"], tag="cnn_mixed", epochs=30, patience=10)
        torch.save(model.state_dict(), MIXED_DIR / "scratch_cnn.pt")
        info["history"] = hist
        fn = lambda items, tag: probs_net(model, items, B.IMG_SIZE, B.MEAN, B.STD, f"{tag}_128")
        params = tc.count_params(model)
        rt = tc.runtime_ms(model, load_split()["test"], B.IMG_SIZE, B.MEAN, B.STD, "cpu", tag="test_128")
    else:
        model, hist, info = C.train(mixed["train"], mixed["val"], tag="mnv3_mixed", epochs=10, patience=4)
        torch.save(model.state_dict(), MIXED_DIR / "mobilenetv3_large.pt")
        info["history"] = hist
        fn = lambda items, tag: probs_net(model, items, C.IMG_SIZE, C.MEAN, C.STD, f"{tag}_224")
        params = tc.count_params(model)
        rt = tc.runtime_ms(model, load_split()["test"], C.IMG_SIZE, C.MEAN, C.STD, "cpu", tag="test_224")
    info["train_time_min"] = (time.perf_counter() - t0) / 60
    info["params_M"] = params
    info["runtime_ms_cpu"] = rt
    return fn, info


TARGET = 0.985   # required accuracy of the answers BIN gives (validation); below the threshold -> "not sure"


def calibrate_threshold(y, probs, target=TARGET):
    """Smallest confidence threshold whose accepted predictions are >= target accurate."""
    conf, ok = probs.max(1), probs.argmax(1) == y
    table = []
    for t in np.round(np.arange(0.30, 0.96, 0.05), 2):
        keep = conf >= t
        table.append({"threshold": float(t), "coverage": float(keep.mean()),
                      "selective_accuracy": float(ok[keep].mean()) if keep.any() else None})
    chosen = next((r for r in table if r["selective_accuracy"] and r["selective_accuracy"] >= target), table[-1])
    return chosen, table


def main(order=("A", "C", "B")):
    set_seed()
    mixed = load_mixed()
    test, rw = load_split()["test"], load_realworld()
    y_test, y_rw, y_val = labels_of(test), labels_of(rw), labels_of(mixed["val"])
    out = load_json(OUT) if OUT.exists() else {"methods": {}}
    out.update({"n_train": len(mixed["train"]), "n_val": len(mixed["val"]),
                "n_trashnet_test": len(test), "n_realworld_test": len(rw)})
    e1 = load_json(RESULTS_DIR / "exp1_main.json")["methods"]
    e2 = load_json(RESULTS_DIR / "exp2_domain_shift.json")["results"]["realworld"]
    for k in order:
        if k in out["methods"]:
            print(f"{NAMES[k]}: already done, skipping")
            continue
        print(f"\n=== Experiment 6: training {NAMES[k]} on mixed-domain data (device {tc.DEVICE}) ===")
        fn, info = train_method(k, mixed)
        p_test, p_rw, p_val = fn(test, "test"), fn(rw, "realworld"), fn(mixed["val"], "mixedval")
        thr, table = calibrate_threshold(y_val, p_val)
        res = {"name": NAMES[k], "train": info, "trashnet_test": score(y_test, p_test),
               "realworld": score(y_rw, p_rw), "threshold": thr, "threshold_table": table,
               "baseline_trashnet_only": {"trashnet_test_acc": e1[k]["accuracy"], "realworld_acc": e2[k]["accuracy"]}}
        keep = np.array(res["realworld"]["confidence"]) >= thr["threshold"]
        res["realworld_at_threshold"] = {"coverage": float(keep.mean()),
                                         "accuracy": float((np.array(res["realworld"]["pred"])[keep] == y_rw[keep]).mean() * 100)}
        out["methods"][k] = res
        save_json(out, OUT)
        print(f"  TrashNet test {res['trashnet_test']['accuracy']:.1f}% (was {e1[k]['accuracy']:.1f}%)  |  "
              f"real-world {res['realworld']['accuracy']:.1f}% (was {e2[k]['accuracy']:.1f}%)  |  "
              f"threshold {thr['threshold']:.2f}: answers {keep.mean() * 100:.0f}% of real-world photos at "
              f"{res['realworld_at_threshold']['accuracy']:.1f}% accuracy  ({info['train_time_min']:.1f} min)")
    recalibrate(out, y_rw)
    save_json(out, OUT)
    if all(k in out["methods"] for k in "ABC"):
        figures(out)


def recalibrate(out, y_rw, target=TARGET):
    """(Re)choose every method's threshold from its stored validation table, so all
    methods use the same target even if they were trained with an older setting."""
    y_rw = np.asarray(y_rw)
    for m in out["methods"].values():
        table = m["threshold_table"]
        m["threshold"] = next((r for r in table if r["selective_accuracy"] and r["selective_accuracy"] >= target), table[-1])
        m["threshold_target"] = target
        keep = np.array(m["realworld"]["confidence"]) >= m["threshold"]["threshold"]
        pred = np.array(m["realworld"]["pred"])
        m["realworld_at_threshold"] = {"coverage": float(keep.mean()),
                                       "accuracy": float((pred[keep] == y_rw[keep]).mean() * 100) if keep.any() else None}


def write_table(out):
    """results/results_table_exp6.md / .csv: TrashNet-only vs mixed-domain models."""
    M = out["methods"]
    lk = RESULTS_DIR / "exp6_leakage_check.json"
    clean = load_json(lk)["accuracy_all_vs_clean"] if lk.exists() else {}
    rows = ["| Method | TrashNet test acc. (%) only-TrashNet -> mixed | Real-world acc. (%) only-TrashNet -> mixed [95% CI] "
            "| Real-world acc., clean subset (%) | Real-world macro-F1 | 'Not sure' threshold | Answered / accuracy of answers "
            "| Runtime (ms/img, CPU) | Params (M) | Train time (min, CPU) |", "|---|---|---|---|---|---|---|---|---|---|"]
    csv = ["method,trashnet_acc_before,trashnet_acc_after,realworld_acc_before,realworld_acc_after,realworld_clean_acc,"
           "realworld_macro_f1,threshold,answered_pct,answered_acc,runtime_ms_cpu,params_M,train_time_min"]
    for k in "ABC":
        m, b = M[k], M[k]["baseline_trashnet_only"]
        at = m["realworld_at_threshold"]
        acc_ans = at["accuracy"] if at["accuracy"] is not None else float("nan")
        cl = clean.get(k, {}).get("clean_subset", float("nan"))
        rows.append(f"| {NAMES[k]} | {b['trashnet_test_acc']:.1f} -> {m['trashnet_test']['accuracy']:.1f} | "
                    f"{b['realworld_acc']:.1f} -> {m['realworld']['accuracy']:.1f} [{m['realworld']['acc_ci95'][0]:.1f}-{m['realworld']['acc_ci95'][1]:.1f}] | "
                    f"{cl:.1f} | {m['realworld']['macro_f1']:.3f} | {m['threshold']['threshold']:.2f} | "
                    f"{at['coverage'] * 100:.0f}% / {acc_ans:.1f}% | {m['train']['runtime_ms_cpu']:.1f} | "
                    f"{m['train']['params_M']:.2f} | {m['train']['train_time_min']:.1f} |")
        csv.append(f"{NAMES[k]},{b['trashnet_test_acc']:.2f},{m['trashnet_test']['accuracy']:.2f},{b['realworld_acc']:.2f},"
                   f"{m['realworld']['accuracy']:.2f},{cl:.2f},{m['realworld']['macro_f1']:.4f},{m['threshold']['threshold']:.2f},"
                   f"{at['coverage'] * 100:.1f},{acc_ans:.2f},{m['train']['runtime_ms_cpu']:.2f},{m['train']['params_M']:.3f},"
                   f"{m['train']['train_time_min']:.2f}")
    nl = chr(10)
    (RESULTS_DIR / "results_table_exp6.md").write_text(nl.join(rows) + nl)
    (RESULTS_DIR / "results_table_exp6.csv").write_text(nl.join(csv) + nl)


def figures(out):
    write_table(out)
    M = out["methods"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    xs = np.arange(3)
    for j, (key, base_key, title) in enumerate([("trashnet_test", "trashnet_test_acc", "TrashNet test (studio photos)"),
                                                ("realworld", "realworld_acc", "Real-world test (web photos)")]):
        before = [M[k]["baseline_trashnet_only"][base_key] for k in "ABC"]
        after = [M[k][key]["accuracy"] for k in "ABC"]
        ax[j].bar(xs - 0.2, before, 0.4, color="#b8c4bd", label="trained on TrashNet only")
        ax[j].bar(xs + 0.2, after, 0.4, color=[COLORS[k] for k in "ABC"], label="trained on TrashNet + web photos")
        for i in range(3):
            ax[j].text(i - 0.2, before[i] + 1, f"{before[i]:.0f}", ha="center", fontsize=9)
            ax[j].text(i + 0.2, after[i] + 1, f"{after[i]:.0f}", ha="center", fontsize=9, weight="bold")
        ax[j].set_xticks(xs, ["HOG +\nlinear SVM", "4-block CNN\n(scratch)", "MobileNetV3-L\n(pretrained)"])
        ax[j].set_ylim(0, 105); ax[j].set_ylabel("top-1 accuracy (%)"); ax[j].set_title(title)
        ax[j].axhline(100 / 6, color="gray", ls=":", lw=1)
        ax[j].legend(fontsize=8, loc="upper left")
    fig.suptitle("Experiment 6: adding real-world photos to training (same test sets)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp6_mixed_domain.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, k in zip(axes, "ABC"):
        cm = np.array(M[k]["realworld"]["confusion_matrix"])
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap="Greens", vmin=0, vmax=1)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(6), CLASSES, rotation=45, ha="right"); ax.set_yticks(range(6), CLASSES)
        ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
        ax.set_title(f"{NAMES[k]}\nreal-world acc {M[k]['realworld']['accuracy']:.1f}%", fontsize=10)
    fig.suptitle("Experiment 6: confusion matrices on the 600 real-world photos (mixed-domain models)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp6_confusion_realworld.png"); plt.close(fig)

    t = M["C"]["threshold_table"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot([r["threshold"] for r in t], [r["coverage"] * 100 for r in t], "o-", color="#00798c", label="photos answered (%)")
    ax.plot([r["threshold"] for r in t], [(r["selective_accuracy"] or 0) * 100 for r in t], "o-", color="#30638e",
            label="accuracy of answered photos (%)")
    ax.axvline(M["C"]["threshold"]["threshold"], color="#d1495b", ls="--", label="chosen threshold")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("percent"); ax.set_ylim(0, 102)
    ax.set_title("MobileNetV3 (mixed): when to say 'not sure' (validation)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "exp6_confidence_threshold.png"); plt.close(fig)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("A", "C", "B"))
