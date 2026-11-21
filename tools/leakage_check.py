"""Second leakage check for Experiment 6: semantic near-duplicates.

The pixel "detail signature" used to clean the data catches resized/re-encoded copies,
but not crops, mirror images or watermarked versions of the same web picture. Here every
real-world TEST photo is compared with every web TRAINING photo using features of an
ImageNet-pretrained MobileNetV3-Large (not our fine-tuned model, so the check is neutral).
Pairs with cosine similarity above a threshold are shown for visual inspection and the
test accuracy is reported again on the subset of test photos without any such neighbour.

Run:  python tools/leakage_check.py [threshold]     (default 0.85, chosen by inspecting pairs)
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import torchvision                      # noqa: E402
from PIL import Image                   # noqa: E402

from common import FIG_DIR, RESULTS_DIR, load_array_cache, load_realworld, save_json  # noqa: E402


def embed(items, tag):
    net = torchvision.models.mobilenet_v3_large(weights="IMAGENET1K_V2").eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    arr = load_array_cache(items, 224, tag, mmap=True)
    out = []
    with torch.no_grad():
        for i in range(0, len(arr), 64):
            x = torch.from_numpy(np.array(arr[i:i + 64])).permute(0, 3, 1, 2).float() / 255
            f = net.avgpool(net.features((x - mean) / std)).flatten(1)
            out.append(torch.nn.functional.normalize(f, dim=1).numpy())
    return np.concatenate(out)


def main(thr=0.85):
    mixed = json.loads((ROOT / "splits" / "mixed.json").read_text())
    web = [(p, int(y)) for p, y in mixed["web_train"] + mixed["web_val"]]
    test = load_realworld()
    ft, fw = embed(test, "realworld_224"), embed(web, "webtrainval_224")
    S = ft @ fw.T
    nn_sim, nn_idx = S.max(1), S.argmax(1)
    flagged = np.where(nn_sim > thr)[0]
    hist = {f">{t}": int((nn_sim > t).sum()) for t in (0.8, 0.85, 0.9, 0.95, 0.98)}
    print("nearest web-training neighbour of each test photo, cosine similarity:", hist)
    e6 = json.loads((RESULTS_DIR / "exp6_mixed_domain.json").read_text())["methods"]
    y = np.array([t[1] for t in test])
    clean = np.setdiff1d(np.arange(len(test)), flagged)
    res = {"threshold": thr, "nn_similarity_hist": hist, "n_flagged": int(len(flagged)),
           "flagged": [[test[i][0], web[nn_idx[i]][0], round(float(nn_sim[i]), 3)] for i in flagged],
           "accuracy_all_vs_clean": {}}
    for k, m in e6.items():
        pred = np.array(m["realworld"]["pred"])
        res["accuracy_all_vs_clean"][k] = {"all": float((pred == y).mean() * 100),
                                           "clean_subset": float((pred[clean] == y[clean]).mean() * 100),
                                           "n_clean": int(len(clean))}
        print(f"  {k}: real-world accuracy all {res['accuracy_all_vs_clean'][k]['all']:.1f}%  "
              f"| without flagged photos ({len(clean)}) {res['accuracy_all_vs_clean'][k]['clean_subset']:.1f}%")
    save_json(res, RESULTS_DIR / "exp6_leakage_check.json")
    # contact sheet of the most similar pairs (test | nearest web-training photo)
    order = np.argsort(-nn_sim)[:24]
    sheet = Image.new("RGB", (4 * 330, 6 * 165), "white")
    for r, i in enumerate(order):
        a = Image.open(ROOT / test[i][0]).convert("RGB").resize((160, 160))
        b = Image.open(ROOT / web[nn_idx[i]][0]).convert("RGB").resize((160, 160))
        x, yy = (r % 4) * 330, (r // 4) * 165
        sheet.paste(a, (x, yy)); sheet.paste(b, (x + 162, yy))
    sheet.save(FIG_DIR / "exp6_leakage_pairs.png")
    print("most similar pairs ->", FIG_DIR / "exp6_leakage_pairs.png", "| sims:", np.round(nn_sim[order][:12], 3).tolist())


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.85)
