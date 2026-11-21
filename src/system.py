"""System build: run all three methods on a new image, draw the results on it and
recommend a default method based on our measured results.

Used by the Gradio app (app/app.py) and by the notebook demo.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_common as tc   # noqa: E402,F401
import numpy as np          # noqa: E402
from PIL import Image, ImageDraw, ImageFont   # noqa: E402

import hog_svm as A                 # noqa: E402
import scratch_cnn as B             # noqa: E402
import pretrained_mobilenet as C    # noqa: E402
from common import CLASSES, MODELS_DIR, RESULTS_DIR, load_json, resize_square  # noqa: E402

METHOD_NAMES = {"A": "HOG + linear SVM", "B": "4-block CNN (scratch)", "C": "MobileNetV3-Large"}
SLOTS = {"A": "classical", "B": "trained from scratch", "C": "pretrained"}

# Short disposal tips (Philippine context: RA 9003 requires segregation at source
# into biodegradable, recyclable and residual waste).
TIPS = {
    "cardboard": "Recyclable. Flatten boxes and keep them dry; wet or greasy cardboard (e.g. pizza boxes) goes to residual.",
    "glass": "Recyclable. Rinse bottles and jars; wrap broken glass before disposal so collectors are not injured.",
    "metal": "Recyclable. Rinse cans and crush them to save space; junk shops buy aluminium and steel.",
    "paper": "Recyclable. Keep paper clean and dry; tissue and food-soiled paper are not recyclable.",
    "plastic": "Often recyclable. Rinse PET bottles and rigid containers; sachets and soft wrappers are usually residual.",
    "trash": "Residual waste. Not recyclable - place it in the residual bin for proper disposal.",
}


class TrashSorter:
    def __init__(self):
        self.models = {
            "A": A.HogSvmClassifier.load(MODELS_DIR / "hog_svm.npz"),
            "B": B.load_trained(),
            "C": C.load_trained(),
        }
        self.recommended, self.reason, self.summary = self._recommendation()
        # warm-up: the first call of each model pays one-off initialisation costs,
        # which would make the runtime shown to the user misleading
        for _ in range(3):
            self.predict(np.full((224, 224, 3), 128, np.uint8))

    # ------------------------------------------------------------------
    def _recommendation(self):
        """Pick the default method from the measured results: highest macro-F1 on
        the held-out test split, with real-world accuracy and CPU runtime as
        supporting evidence."""
        e1 = load_json(RESULTS_DIR / "exp1_main.json")["methods"]
        e2p = RESULTS_DIR / "exp2_domain_shift.json"
        rw = load_json(e2p)["results"]["realworld"] if e2p.exists() else {}
        summary = {}
        for k in ("A", "B", "C"):
            summary[k] = {"acc": e1[k]["accuracy"], "f1": e1[k]["macro_f1"],
                          "ms": e1[k]["runtime_ms_cpu"], "params": e1[k]["params_M"],
                          "rw_acc": rw.get(k, {}).get("accuracy")}
        best = max(summary, key=lambda k: summary[k]["f1"])
        s = summary[best]
        others = [k for k in summary if k != best]
        reason = (f"**{METHOD_NAMES[best]}** is recommended by default: it had the highest macro-F1 "
                  f"on our held-out test set ({s['f1']:.3f}, accuracy {s['acc']:.1f}% vs "
                  + ", ".join(f"{summary[k]['acc']:.1f}% for {METHOD_NAMES[k]}" for k in others) + ")")
        if s["rw_acc"] is not None:
            best_rw = max(summary, key=lambda k: summary[k]["rw_acc"] or 0)
            if best_rw == best:
                reason += (f", and it also stayed the most accurate on unseen real-world photos "
                           f"({s['rw_acc']:.1f}% vs " + ", ".join(
                               f"{summary[k]['rw_acc']:.1f}%" for k in others) + ")")
            else:
                reason += f", and it reached {s['rw_acc']:.1f}% on unseen real-world photos"
        reason += (f". It runs in about {s['ms']:.0f} ms per image on a laptop CPU with "
                   f"{s['params']:.1f} M parameters, fast enough for a phone or an MRF kiosk.")
        return best, reason, summary

    # ------------------------------------------------------------------
    def predict(self, img) -> dict:
        """img: PIL image or HxWx3 uint8 array. Returns per-method predictions."""
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img).astype(np.uint8))
        img = img.convert("RGB")
        x128 = resize_square(img, 128)
        x224 = resize_square(img, 224)
        out = {}
        for k, fn, x in (("A", lambda x: self.models["A"].predict_proba(x), x128),
                         ("B", lambda x: B.predict_proba(self.models["B"], x), x128),
                         ("C", lambda x: C.predict_proba(self.models["C"], x), x224)):
            t0 = time.perf_counter()
            p = fn(x)
            ms = (time.perf_counter() - t0) * 1000
            i = int(np.argmax(p))
            out[k] = {"label": CLASSES[i], "confidence": float(p[i]), "ms": ms,
                      "probs": {c: float(v) for c, v in zip(CLASSES, p)}}
        return out

    # ------------------------------------------------------------------
    def annotate(self, img, preds) -> Image.Image:
        """Draw the recommended answer above the photo and every method's label,
        confidence and runtime below it."""
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img).astype(np.uint8))
        img = img.convert("RGB")
        w = 640
        img = img.resize((w, max(1, int(img.height * w / img.width))))
        top_h, bar_h = 64, 34
        canvas = Image.new("RGB", (w, top_h + img.height + bar_h * 3), "white")
        canvas.paste(img, (0, top_h))
        d = ImageDraw.Draw(canvas)

        def font(name, size):
            for n in (name, name.replace("bd", ""), "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
                try:
                    return ImageFont.truetype(n, size)
                except OSError:
                    continue
            return ImageFont.load_default()
        f_small, f_bold, f_big = font("arial.ttf", 18), font("arialbd.ttf", 18), font("arialbd.ttf", 30)
        rec = preds[self.recommended]
        d.rectangle([0, 0, w, top_h], fill=(18, 59, 47))
        d.text((14, 14), f"{rec['label'].upper()}  {rec['confidence'] * 100:.0f}%", fill="white", font=f_big)
        d.text((w - 14, 22), f"by {METHOD_NAMES[self.recommended]} (recommended)", fill=(200, 230, 210),
               font=f_small, anchor="ra")
        colors = {"A": (209, 73, 91), "B": (0, 121, 140), "C": (48, 99, 142)}
        for r, k in enumerate(("A", "B", "C")):
            p = preds[k]
            y0 = top_h + img.height + r * bar_h
            d.rectangle([0, y0, w, y0 + bar_h - 2], fill=colors[k])
            star = "  * RECOMMENDED" if k == self.recommended else ""
            text = (f"{k}. {METHOD_NAMES[k]}: {p['label'].upper()}  "
                    f"{p['confidence'] * 100:.0f}%  ({p['ms']:.0f} ms){star}")
            d.text((10, y0 + 7), text, fill="white", font=f_bold if star else f_small)
        return canvas

    def explain(self, preds) -> str:
        rec = preds[self.recommended]
        labels = {preds[k]["label"] for k in preds}
        lines = [f"### Result: **{rec['label']}** "
                 f"({rec['confidence'] * 100:.0f}% confidence from {METHOD_NAMES[self.recommended]})",
                 "", self.reason, ""]
        if len(labels) == 1:
            lines.append("All three methods agree, so this prediction is very likely correct.")
        else:
            lines.append("The methods disagree. Trust the recommended method; disagreement usually means "
                         "the photo is unusual (cluttered background, several items, unusual lighting).")
        if rec["confidence"] < 0.5:
            lines.append("Low confidence: retake the photo with a single item on a plain background.")
        lines += ["", f"**Disposal tip:** {TIPS[rec['label']]}",
                  "", "_Runtimes are the wall-clock time of this request on this computer; the table above lists "
                  "the benchmark (CPU, batch 1, mean of 200 images). Confidence for the SVM is a softmax of its "
                  "margins (a score, not a calibrated probability)._"]
        return "\n".join(lines)
