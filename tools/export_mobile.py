"""Export the three trained methods for the phone app (web/).

* Method B and C -> ONNX (run on the phone with onnxruntime-web). The wrapper
  takes raw RGB pixel values (0-255, NCHW float32) and does the normalisation
  and softmax inside the graph, so the app cannot get the preprocessing wrong.
* Method A (HOG + SVM) -> a small binary file with the SVM weights and the
  feature standardisation; the HOG itself is re-implemented in web/js/hog.js.
* summary.json -> the measured results the app uses for its recommendation.

Each ONNX model is checked against PyTorch on the test images.

Run:  python tools/export_mobile.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np            # noqa: E402
import torch                  # noqa: E402
import torch.nn as nn         # noqa: E402

import hog_svm as A                 # noqa: E402
import scratch_cnn as B             # noqa: E402
import pretrained_mobilenet as C    # noqa: E402
from common import (CLASSES, MODELS_DIR, RESULTS_DIR, load_array_cache, load_json,  # noqa: E402
                    load_split)

OUT = ROOT / "web" / "models"
OUT.mkdir(parents=True, exist_ok=True)


class Deployable(nn.Module):
    """raw pixels (0-255) -> normalise -> network -> softmax probabilities."""

    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net.eval()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255.0)

    def forward(self, image):
        return torch.softmax(self.net((image - self.mean) / self.std), dim=1)


def export_onnx(net, mean, std, size, name):
    model = Deployable(net, mean, std).eval()
    dummy = torch.rand(1, 3, size, size) * 255
    path = OUT / f"{name}.onnx"
    torch.onnx.export(model, (dummy,), str(path), input_names=["image"], output_names=["probs"],
                      opset_version=17, dynamo=False)
    # verify against PyTorch on real test images
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    imgs = load_array_cache(load_split()["test"], size, f"test_{size}")[:64]
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float()
    with torch.no_grad():
        ref = model(x).numpy()
    got = np.concatenate([sess.run(None, {"image": x[i:i + 1].numpy()})[0] for i in range(len(x))])
    diff = float(np.abs(ref - got).max())
    agree = float((ref.argmax(1) == got.argmax(1)).mean() * 100)
    print(f"  {name}.onnx  {path.stat().st_size / 1e6:.1f} MB  max|torch-onnx| = {diff:.2e}  "
          f"argmax agreement {agree:.0f}%")
    assert agree == 100.0 and diff < 1e-3
    return {"file": f"models/{name}.onnx", "size": size, "max_abs_diff": diff}


def export_hog_svm():
    m = A.HogSvmClassifier.load(MODELS_DIR / "hog_svm.npz")
    W = m.svm.W.astype("<f4")          # (8100, 6)
    blob = np.concatenate([W.ravel(), m.svm.b.astype("<f4"), m.mean.astype("<f4"),
                           m.std.astype("<f4")]).astype("<f4")
    (OUT / "hog_svm.bin").write_bytes(blob.tobytes())
    # a few reference feature vectors so the JS port can be unit-tested
    imgs = load_array_cache(load_split()["test"], 128, "test_128")[:3]
    ref = [{"index": i, "first10": A.extract_one(im, m.cfg)[:10].tolist(),
            "scores": m.scores_from_features(A.extract_one(im, m.cfg)[None])[0].tolist()}
           for i, im in enumerate(imgs)]
    meta = {"file": "models/hog_svm.bin", "dim": int(W.shape[0]), "classes": len(CLASSES),
            "layout": "W[dim*classes] row-major (dim, classes), b[classes], mean[dim], std[dim]; float32 LE",
            "cfg": m.cfg, "size": 128, "reference": ref}
    print(f"  hog_svm.bin  {(OUT / 'hog_svm.bin').stat().st_size / 1e3:.0f} kB  (dim {W.shape[0]})")
    return meta


def main():
    print("Exporting models for the phone app ...")
    meta = {"classes": CLASSES,
            "A": export_hog_svm(),
            "B": export_onnx(B.load_trained(), B.MEAN, B.STD, B.IMG_SIZE, "scratch_cnn"),
            "C": export_onnx(C.load_trained(), C.MEAN, C.STD, C.IMG_SIZE, "mobilenetv3_large")}
    e1 = load_json(RESULTS_DIR / "exp1_main.json")["methods"]
    rw = load_json(RESULTS_DIR / "exp2_domain_shift.json")["results"]["realworld"]
    meta["results"] = {k: {"accuracy": round(e1[k]["accuracy"], 1), "macro_f1": round(e1[k]["macro_f1"], 3),
                           "realworld_accuracy": round(rw[k]["accuracy"], 1),
                           "runtime_ms_cpu": round(e1[k]["runtime_ms_cpu"], 1),
                           "params_M": round(e1[k]["params_M"], 2)} for k in ("A", "B", "C")}
    (ROOT / "web" / "models" / "meta.json").write_text(json.dumps(meta, indent=1))
    print("  wrote web/models/meta.json")


if __name__ == "__main__":
    main()
