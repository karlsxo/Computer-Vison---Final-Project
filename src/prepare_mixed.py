"""Experiment 6 data: a mixed-domain training set (TrashNet + real-world web photos).

Experiment 2 showed that models trained only on TrashNet's studio photos lose a
lot of accuracy on real-world photos. The phone app sees real-world photos, so
we build a second training set that adds web photos from the 12-class Garbage
Classification dataset to the TrashNet training split.

Leakage rules (all enforced in code):
  * The 600 real-world TEST photos (splits/realworld.json) are never used, and
    neither is any web image that is a near-duplicate of one of them.
  * No web image that is a near-duplicate of ANY TrashNet image is used (the web
    set contains TrashNet copies, some of which could be TrashNet test images).
  * Near-duplicates inside the web set are grouped, and a group goes entirely to
    train or entirely to validation.
  * The TrashNet test split and the real-world test set are unchanged, so the
    results are directly comparable with Experiments 1 and 2.

Output: splits/mixed.json with
  train = TrashNet train + web train (up to WEB_TRAIN per class)
  val   = TrashNet val   + web val   (up to WEB_VAL per class)

Run:  python src/prepare_mixed.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, CLASS_TO_IDX, IMG_EXTS, ROOT, SEED, load_realworld, load_split,  # noqa: E402
                    save_json)
from prepare_data import (GARBAGE12_MAP, NEAR_DUP_CORR, UnionFind, detail_signature,  # noqa: E402
                          find_garbage12_root, rel, scan_trashnet)

MIXED_FILE = ROOT / "splits" / "mixed.json"
WEB_TRAIN = 300   # web images per class added to training
WEB_VAL = 40      # web images per class added to validation


def safe_signature(p):
    try:
        return detail_signature(p)
    except Exception:
        return None   # unreadable / truncated file


def main():
    rng = np.random.default_rng(SEED)
    root = find_garbage12_root()
    assert root is not None, "data/raw/garbage12 missing - run the notebook download cell first"
    realworld = load_realworld()
    rw_paths = {p for p, _ in realworld}

    print("Signatures of TrashNet and the real-world test set ...")
    tn_sig = np.stack([detail_signature(p) for p, _ in scan_trashnet()])
    rw_sig = np.stack([detail_signature(ROOT / p) for p, _ in realworld])

    stats = {"per_class": {}}
    web_train, web_val = [], []
    for cls in CLASSES:
        srcs = [s for s, d in GARBAGE12_MAP.items() if d == cls]
        cand = [p for s in srcs for p in sorted((root / s).iterdir())
                if p.suffix.lower() in IMG_EXTS and rel(p) not in rw_paths]
        sigs, keep = [], []
        for p in cand:
            s = safe_signature(p)
            if s is not None:
                sigs.append(s); keep.append(p)
        sigs = np.stack(sigs)
        near_tn = (sigs @ tn_sig.T).max(1) > NEAR_DUP_CORR
        near_rw = (sigs @ rw_sig.T).max(1) > NEAR_DUP_CORR
        ok = ~(near_tn | near_rw)
        keep = [p for p, k in zip(keep, ok) if k]
        sigs = sigs[ok]
        # group near-duplicates inside the web set
        S = sigs @ sigs.T
        np.fill_diagonal(S, -1)
        uf = UnionFind(len(keep))
        for i, j in np.argwhere(np.triu(S > NEAR_DUP_CORR)):
            uf.union(int(i), int(j))
        groups = defaultdict(list)
        for i in range(len(keep)):
            groups[uf.find(i)].append(i)
        groups = list(groups.values())
        order = rng.permutation(len(groups))
        val, train = [], []
        for g in (groups[i] for i in order):
            target = val if len(val) < WEB_VAL else train if len(train) < WEB_TRAIN else None
            if target is None:
                break
            target.extend(g)
        y = CLASS_TO_IDX[cls]
        web_val += [(rel(keep[i]), y) for i in val]
        web_train += [(rel(keep[i]), y) for i in train]
        stats["per_class"][cls] = {"candidates": len(cand), "unreadable": len(cand) - len(ok),
                                   "near_trashnet": int(near_tn.sum()), "near_realworld_test": int(near_rw.sum()),
                                   "usable": len(keep), "web_train": len(train), "web_val": len(val)}
        print(f"  {cls:<10} candidates {len(cand):5d}  near-TrashNet {int(near_tn.sum()):4d}  "
              f"near-test {int(near_rw.sum()):3d}  usable {len(keep):5d} -> train {len(train)}, val {len(val)}")

    split = load_split()
    out = {
        "description": "Mixed-domain training data (Experiment 6): TrashNet + web photos. "
                       "Test sets unchanged (TrashNet test split, splits/realworld.json).",
        "seed": SEED, "web_train_per_class": WEB_TRAIN, "web_val_per_class": WEB_VAL,
        "near_dup_corr": NEAR_DUP_CORR, "stats": stats,
        "web_train": [[p, y] for p, y in web_train], "web_val": [[p, y] for p, y in web_val],
        "splits": {"train": [[p, y] for p, y in split["train"] + web_train],
                   "val": [[p, y] for p, y in split["val"] + web_val]},
    }
    # self-checks: no overlap with either test set
    used = {p for p, _ in web_train + web_val}
    assert not used & rw_paths
    assert not used & {p for p, _ in split["test"]}
    save_json(out, MIXED_FILE)
    print(f"Mixed train {len(out['splits']['train'])} (web {len(web_train)}), "
          f"val {len(out['splits']['val'])} (web {len(web_val)}). Saved {MIXED_FILE}")


def load_mixed():
    import json
    js = json.loads(MIXED_FILE.read_text())
    return {k: [(p, int(y)) for p, y in v] for k, v in js["splits"].items()}


if __name__ == "__main__":
    main()
