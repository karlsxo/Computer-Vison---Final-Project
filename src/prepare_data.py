"""Data preparation: duplicate removal, leakage-safe split, real-world test set.

Steps
1. Scan TrashNet (6 classes).
2. Remove exact duplicates (identical MD5 of file bytes).
   If the identical file appears under two different classes the label is
   ambiguous and every copy is dropped.
3. Find near-duplicates with a "detail signature" written here in NumPy:
   64x64 grayscale minus a heavy Gaussian blur (this removes the smooth studio
   lighting that every TrashNet photo shares), then zero-mean, unit-norm.
   Pairs whose signatures correlate above NEAR_DUP_CORR are linked with
   union-find into "groups". A whole group always goes to the same split, so
   near-identical shots of one item can never be in both train and test.
   (A plain 64-bit dHash was tried first and rejected: on plain backgrounds it
   linked 1,053 pairs of *different-class* images.)
4. Seeded, stratified, group-aware 70/15/15 split saved to splits/split.json.
5. Build the real-world (domain-shift) test set from the web-scraped 12-class
   "Garbage Classification" dataset, mapping its classes to our 6 and dropping
   any image that is a near-duplicate of a TrashNet image.

Run:  python src/prepare_data.py
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLASSES, CLASS_TO_IDX, IMG_EXTS, RAW_DIR, REALWORLD_FILE, ROOT,
                    SEED, SPLIT_FILE, TRASHNET_DIR, save_json)

NEAR_DUP_CORR = 0.80       # detail-signature correlation threshold (chosen by visual inspection)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
REALWORLD_PER_CLASS = 100  # images per class in the domain-shift test set

# Web-scraped 12-class dataset -> our 6 material classes (classes not listed are excluded)
GARBAGE12_MAP = {
    "cardboard": "cardboard",
    "paper": "paper",
    "metal": "metal",
    "plastic": "plastic",
    "trash": "trash",
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
    # excluded: battery, biological, clothes, shoes (not one of our 6 materials)
}


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def md5_of(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def detail_signature(p: Path, size: int = 64, blur: float = 6.0) -> np.ndarray:
    """High-pass grayscale thumbnail, zero-mean and unit-norm (float32 vector).

    Subtracting a heavy blur removes the smooth background illumination, so two
    signatures only correlate strongly when the *object detail* matches."""
    from PIL import ImageFilter
    g = Image.open(p).convert("L").resize((size, size), Image.BILINEAR)
    a = np.asarray(g, np.float64) - np.asarray(g.filter(ImageFilter.GaussianBlur(blur)), np.float64)
    a = a.ravel() - a.mean()
    return (a / (np.linalg.norm(a) + 1e-9)).astype(np.float32)


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def scan_trashnet():
    items = []
    for c in CLASSES:
        for p in sorted((TRASHNET_DIR / c).iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, CLASS_TO_IDX[c]))
    return items


def group_stratified_split(groups_by_class, rng):
    """Assign whole groups to train/val/test so each class hits ~70/15/15."""
    out = {"train": [], "val": [], "test": []}
    names = ["train", "val", "test"]
    for c, groups in groups_by_class.items():
        groups = [groups[i] for i in rng.permutation(len(groups))]
        # place big groups first so the remaining small groups can balance
        groups.sort(key=len, reverse=True)
        total = sum(len(g) for g in groups)
        target = np.array(SPLIT_FRACTIONS) * total
        filled = np.zeros(3)
        for g in groups:
            # put the group where it is most "under-filled" relative to target
            k = int(np.argmax(target - filled))
            out[names[k]].extend(g)
            filled[k] += len(g)
    for k in out:
        out[k].sort()
    return out


def build_trashnet_split():
    items = scan_trashnet()
    print(f"TrashNet images found: {len(items)}")

    # 1) exact duplicates (same bytes). Conflicting labels -> drop every copy.
    by_md5 = defaultdict(list)
    for p, y in items:
        by_md5[md5_of(p)].append((p, y))
    kept, exact_dups, label_conflicts = [], [], []
    for copies in by_md5.values():
        if len({y for _, y in copies}) > 1:
            label_conflicts.extend(rel(p) for p, _ in copies)
            continue
        kept.append(copies[0])
        exact_dups.extend((rel(p), rel(copies[0][0])) for p, _ in copies[1:])
    kept.sort(key=lambda t: (t[1], str(t[0])))
    print(f"Exact duplicates removed: {len(exact_dups)}; "
          f"identical files with conflicting labels dropped: {label_conflicts}")

    # 2) near duplicates -> groups
    sigs = np.stack([detail_signature(p) for p, _ in kept])
    S = sigs @ sigs.T
    np.fill_diagonal(S, -1)
    uf = UnionFind(len(kept))
    pairs = np.argwhere(np.triu(S > NEAR_DUP_CORR))
    cross_class = 0
    for i, j in pairs:
        if kept[i][1] != kept[j][1]:
            cross_class += 1
        uf.union(int(i), int(j))
    print(f"Near-duplicate pairs (detail corr > {NEAR_DUP_CORR}): {len(pairs)} "
          f"({cross_class} across different classes)")

    # distribution of each image's highest similarity to any other image (for the report)
    nn = S.max(1)
    nn_hist = {f">{t}": int((nn > t).sum()) for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)}
    near_dup_pairs = [[rel(kept[i][0]), rel(kept[j][0]), round(float(S[i, j]), 3)] for i, j in pairs]

    # groups (a group containing several classes is dropped - ambiguous label)
    comp = defaultdict(list)
    for i in range(len(kept)):
        comp[uf.find(i)].append(i)
    groups_by_class = defaultdict(list)
    dropped_ambiguous = []
    for members in comp.values():
        labels = {kept[i][1] for i in members}
        if len(labels) > 1:
            dropped_ambiguous.extend(rel(kept[i][0]) for i in members)
            continue
        groups_by_class[labels.pop()].append([(rel(kept[i][0]), kept[i][1]) for i in members])
    n_groups = sum(len(g) for g in groups_by_class.values())
    multi = sum(1 for gs in groups_by_class.values() for g in gs if len(g) > 1)
    print(f"Groups: {n_groups} ({multi} with >1 image); "
          f"dropped {len(dropped_ambiguous)} images in mixed-label groups")

    rng = np.random.default_rng(SEED)
    splits = group_stratified_split(groups_by_class, rng)

    counts = {k: {c: 0 for c in CLASSES} for k in splits}
    for k, v in splits.items():
        for _, y in v:
            counts[k][CLASSES[y]] += 1
    print("\nClass counts per split:")
    print(f"{'class':<10}" + "".join(f"{k:>8}" for k in splits))
    for c in CLASSES:
        print(f"{c:<10}" + "".join(f"{counts[k][c]:>8}" for k in splits))
    print(f"{'total':<10}" + "".join(f"{len(v):>8}" for v in splits.values()))

    # leakage self-check: no test image within threshold of a train image
    smap = {rel(p): s_ for (p, _), s_ in zip(kept, sigs)}
    tr = np.stack([smap[p] for p, _ in splits["train"]])
    for k in ("val", "test"):
        other = np.stack([smap[p] for p, _ in splits[k]])
        smax = float((other @ tr.T).max())
        print(f"Max similarity {k} -> train: {smax:.3f} (must be <= {NEAR_DUP_CORR})")
        assert smax <= NEAR_DUP_CORR

    save_json({
        "description": "TrashNet split: exact dups removed, near-dup groups kept in one split",
        "seed": SEED, "fractions": SPLIT_FRACTIONS, "near_dup_corr": NEAR_DUP_CORR,
        "classes": CLASSES,
        "stats": {"found": len(items), "exact_duplicates_removed": len(exact_dups),
                  "label_conflicts_dropped": len(label_conflicts),
                  "near_duplicate_pairs": int(len(pairs)),
                  "cross_class_pairs": int(cross_class),
                  "groups": n_groups, "multi_image_groups": multi,
                  "dropped_mixed_label": len(dropped_ambiguous),
                  "max_similarity_hist": nn_hist,
                  "counts": counts},
        "exact_duplicates": exact_dups,
        "label_conflicts": label_conflicts,
        "near_duplicate_pairs": near_dup_pairs,
        "dropped_mixed_label": dropped_ambiguous,
        "splits": {k: [[p, y] for p, y in v] for k, v in splits.items()},
    }, SPLIT_FILE)
    print(f"Saved {SPLIT_FILE}")
    return sigs


def find_garbage12_root() -> Path | None:
    base = RAW_DIR / "garbage12"
    if not base.exists():
        return None
    for d in [base, *base.rglob("*")]:
        if d.is_dir() and (d / "brown-glass").is_dir():
            return d
    return None


def build_realworld(trashnet_sigs: np.ndarray):
    root = find_garbage12_root()
    if root is None:
        print("12-class garbage dataset not found; skipping real-world set.")
        return
    rng = np.random.default_rng(SEED)
    per_class = defaultdict(list)
    for src, dst in GARBAGE12_MAP.items():
        for p in sorted((root / src).iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                per_class[dst].append(p)
    available = {c: len(v) for c, v in per_class.items()}
    print(f"\nReal-world candidates per class: {available}")

    items, removed_overlap = [], 0
    for c in CLASSES:
        cand = per_class[c]
        order = rng.permutation(len(cand))
        chosen = []
        for i in order:
            p = cand[i]
            try:
                sig = detail_signature(p)
            except Exception:
                continue  # unreadable file
            if float((trashnet_sigs @ sig).max()) > NEAR_DUP_CORR:
                removed_overlap += 1   # this image also exists in TrashNet -> not "unseen"
                continue
            chosen.append((rel(p), CLASS_TO_IDX[c]))
            if len(chosen) == REALWORLD_PER_CLASS:
                break
        items.extend(chosen)
    print(f"Real-world test set: {len(items)} images "
          f"({REALWORLD_PER_CLASS}/class), {removed_overlap} skipped as TrashNet overlaps")
    save_json({
        "description": "Domain-shift test set from the web-scraped 12-class Garbage "
                       "Classification dataset (never used for training)",
        "class_mapping": GARBAGE12_MAP, "available_per_class": available,
        "per_class": REALWORLD_PER_CLASS, "skipped_trashnet_overlap": removed_overlap,
        "items": [[p, y] for p, y in items],
    }, REALWORLD_FILE)
    print(f"Saved {REALWORLD_FILE}")


if __name__ == "__main__":
    h = build_trashnet_split()
    build_realworld(h)
