"""BIN system: run all three methods on a new photo, draw the results on it and
answer with the recommended method (or "not sure").

Used by the Gradio app (app/app.py) and the notebook demo. It mirrors the phone
app (web/): same models, same recommendation rule, same "not sure" threshold.

* Models: the mixed-domain models of Experiment 6 (models/mixed/, trained on
  TrashNet + real-world web photos) when they exist, otherwise the TrashNet-only
  models of Experiments 1-5.
* Recommended method: best macro-F1 on the real-world test photos, because a
  deployed system sees real-world photos.
* Not sure: when the recommended method's confidence is below the threshold
  calibrated on the validation split (Experiment 6).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_common as tc   # noqa: E402,F401
import numpy as np          # noqa: E402
import torch                # noqa: E402
from PIL import Image, ImageDraw, ImageFont   # noqa: E402

import hog_svm as A                 # noqa: E402
import scratch_cnn as B             # noqa: E402
import pretrained_mobilenet as C    # noqa: E402
from common import CLASSES, MODELS_DIR, RESULTS_DIR, load_json, resize_square  # noqa: E402

METHOD_NAMES = {"A": "HOG + linear SVM", "B": "4-block CNN (scratch)", "C": "MobileNetV3-Large"}
SLOTS = {"A": "classical", "B": "trained from scratch", "C": "pretrained"}
BIN_OF = {"cardboard": "Recyclable", "glass": "Recyclable", "metal": "Recyclable", "paper": "Recyclable",
          "plastic": "Recyclable", "trash": "Residual"}

# What to do with each material (Philippine context: RA 9003 segregation at source).
TIPS = {
    "cardboard": "Recyclable. Flatten it and keep it dry; wet or greasy cardboard (e.g. pizza boxes) goes to residual.",
    "glass": "Recyclable. Rinse bottles and jars; wrap broken glass before disposal so collectors are not injured.",
    "metal": "Recyclable. Rinse cans and crush them to save space; junk shops buy aluminium and steel.",
    "paper": "Recyclable. Keep paper clean and dry; tissue and food-soiled paper are not recyclable.",
    "plastic": "Recyclable. Rinse PET bottles and rigid containers; sachets and soft wrappers are usually residual.",
    "trash": "Residual waste: it cannot be recycled. Put it in the residual bin.",
}


def _mixed_available():
    d = MODELS_DIR / "mixed"
    e6 = RESULTS_DIR / "exp6_mixed_domain.json"
    return (all((d / f).exists() for f in ("hog_svm.npz", "scratch_cnn.pt", "mobilenetv3_large.pt"))
            and e6.exists() and all(k in load_json(e6)["methods"] for k in "ABC"))


class BinSystem:
    def __init__(self, source: str | None = None):
        self.source = source or ("mixed" if _mixed_available() else "trashnet")
        d = MODELS_DIR / "mixed" if self.source == "mixed" else MODELS_DIR
        self.models = {"A": A.HogSvmClassifier.load(d / "hog_svm.npz"),
                       "B": B.load_trained(d / "scratch_cnn.pt"),
                       "C": C.load_trained(d / "mobilenetv3_large.pt")}
        self.summary, self.thresholds = self._results()
        self.recommended = max("ABC", key=lambda k: self.summary[k]["rw_f1"])
        self.threshold = self.thresholds[self.recommended]
        self.reason = self._reason()
        # warm-up: the first call of each model pays one-off initialisation costs,
        # which would make the runtime shown to the user misleading
        for _ in range(3):
            self.predict(np.full((224, 224, 3), 128, np.uint8))

    # ------------------------------------------------------------------
    def _results(self):
        if self.source == "mixed":
            M = load_json(RESULTS_DIR / "exp6_mixed_domain.json")["methods"]
            s = {k: {"acc": M[k]["trashnet_test"]["accuracy"], "f1": M[k]["trashnet_test"]["macro_f1"],
                     "rw_acc": M[k]["realworld"]["accuracy"], "rw_f1": M[k]["realworld"]["macro_f1"],
                     "ms": M[k]["train"]["runtime_ms_cpu"], "params": M[k]["train"]["params_M"]} for k in "ABC"}
            return s, {k: M[k]["threshold"]["threshold"] for k in "ABC"}
        e1 = load_json(RESULTS_DIR / "exp1_main.json")["methods"]
        rw = load_json(RESULTS_DIR / "exp2_domain_shift.json")["results"]["realworld"]
        s = {k: {"acc": e1[k]["accuracy"], "f1": e1[k]["macro_f1"], "rw_acc": rw[k]["accuracy"],
                 "rw_f1": rw[k]["macro_f1"], "ms": e1[k]["runtime_ms_cpu"], "params": e1[k]["params_M"]} for k in "ABC"}
        return s, {k: 0.5 for k in "ABC"}

    def _reason(self):
        b, s = self.recommended, self.summary
        others = [k for k in "ABC" if k != b]
        trained = ("TrashNet + real-world web photos" if self.source == "mixed" else "TrashNet only")
        return (f"**{METHOD_NAMES[b]}** gives BIN's answer because it is the most accurate on unseen real-world "
                f"photos ({s[b]['rw_acc']:.1f}% vs " + ", ".join(f"{s[k]['rw_acc']:.1f}% for {METHOD_NAMES[k]}" for k in others)
                + f") and reaches {s[b]['acc']:.1f}% on the TrashNet test photos. It runs in about {s[b]['ms']:.0f} ms "
                f"per image on a laptop CPU with {s[b]['params']:.1f} M parameters. Models trained on: {trained}.")

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

    def is_sure(self, preds) -> bool:
        return preds[self.recommended]["confidence"] >= self.threshold

    # ------------------------------------------------------------------
    def annotate(self, img, preds) -> Image.Image:
        """Draw BIN's answer above the photo and every method's label, confidence
        and runtime below it."""
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
        head = (f"{rec['label'].upper()}  {rec['confidence'] * 100:.0f}%" if self.is_sure(preds)
                else f"NOT SURE ({rec['label']}?)")
        d.text((14, 14), head, fill="white", font=f_big)
        d.text((w - 14, 22), f"BIN · {METHOD_NAMES[self.recommended]}", fill=(200, 230, 210), font=f_small, anchor="ra")
        colors = {"A": (209, 73, 91), "B": (0, 121, 140), "C": (48, 99, 142)}
        for r, k in enumerate(("A", "B", "C")):
            p = preds[k]
            y0 = top_h + img.height + r * bar_h
            d.rectangle([0, y0, w, y0 + bar_h - 2], fill=colors[k])
            star = "  * ANSWER" if k == self.recommended else ""
            text = (f"{k}. {METHOD_NAMES[k]}: {p['label'].upper()}  "
                    f"{p['confidence'] * 100:.0f}%  ({p['ms']:.0f} ms){star}")
            d.text((10, y0 + 7), text, fill="white", font=f_bold if star else f_small)
        return canvas

    def explain(self, preds) -> str:
        rec = preds[self.recommended]
        labels = {preds[k]["label"] for k in preds}
        if self.is_sure(preds):
            lines = [f"### {rec['label'].capitalize()} — {BIN_OF[rec['label']]}",
                     f"{rec['confidence'] * 100:.0f}% sure ({METHOD_NAMES[self.recommended]}).", "",
                     f"**What to do:** {TIPS[rec['label']]}", ""]
        else:
            lines = [f"### Not sure",
                     f"Best guess: **{rec['label']}** ({rec['confidence'] * 100:.0f}%), below BIN's confidence "
                     f"threshold of {self.threshold * 100:.0f}%. Retake the photo: one item filling the frame, a plain "
                     f"background and good light.", ""]
        lines.append("All three methods agree." if len(labels) == 1 else
                     "The methods disagree; BIN uses the recommended method.")
        lines += ["", self.reason, "",
                  "_Runtimes are the wall-clock time of this request on this computer. Confidence for the SVM is a "
                  "softmax of its margins (a score, not a calibrated probability)._"]
        return "\n".join(lines)


TrashSorter = BinSystem   # backwards-compatible name used by earlier notebook versions
