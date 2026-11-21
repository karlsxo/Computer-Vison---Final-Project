"""Gradio application: upload a photo of one piece of trash, see what each of the
three methods predicts (label, confidence, runtime) drawn on the image, and get
the recommended method's answer with an explanation and a disposal tip.

Run locally:   python app/app.py          -> http://127.0.0.1:7860
In Colab:      launch(share=True) gives a public link.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr   # noqa: E402

from common import CLASSES       # noqa: E402
from system import METHOD_NAMES, SLOTS, TrashSorter   # noqa: E402

sorter = TrashSorter()


def run(image):
    if image is None:
        return None, "Upload or capture a photo first.", None
    preds = sorter.predict(image)
    annotated = sorter.annotate(image, preds)
    table = [[f"{k}. {METHOD_NAMES[k]}", SLOTS[k], p["label"], f"{p['confidence'] * 100:.1f}%",
              f"{p['ms']:.1f}"] + [f"{p['probs'][c] * 100:.0f}" for c in CLASSES]
             for k, p in preds.items()]
    return annotated, sorter.explain(preds), table


def build_ui():
    examples_dir = ROOT / "app" / "examples"
    examples = sorted(str(p) for p in examples_dir.glob("*.jpg")) if examples_dir.exists() else []
    s = sorter.summary
    with gr.Blocks(title="Trash Sorter") as demo:
        gr.Markdown(
            "# Trash Sorter: what material is this?\n"
            "Upload a photo of **one** item. Three computer-vision methods classify it as "
            "cardboard, glass, metal, paper, plastic or trash (residual).\n\n"
            "| Method | Slot | Test accuracy | Macro F1 | CPU ms/img |\n|---|---|---|---|---|\n" +
            "\n".join(f"| {METHOD_NAMES[k]} | {SLOTS[k]} | {v['acc']:.1f}% | {v['f1']:.3f} | {v['ms']:.1f} |"
                      for k, v in s.items()))
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Photo of one trash item",
                               sources=["upload", "webcam", "clipboard"])
                btn = gr.Button("Classify", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=inp)
            with gr.Column():
                out_img = gr.Image(label="Predictions drawn on the input", show_label=False)
                out_md = gr.Markdown()
        out_tab = gr.Dataframe(headers=["method", "slot", "prediction", "confidence", "runtime (ms)"]
                               + [f"{c} %" for c in CLASSES], label="All methods")
        btn.click(run, inp, [out_img, out_md, out_tab])
        inp.upload(run, inp, [out_img, out_md, out_tab])
        demo_img = os.environ.get("APP_DEMO_IMAGE")   # optional: classify an image on page load
        if demo_img:
            from PIL import Image
            demo.load(lambda: (Image.open(demo_img), *run(Image.open(demo_img))), None,
                      [inp, out_img, out_md, out_tab])
    return demo


if __name__ == "__main__":
    share = "--share" in sys.argv
    build_ui().launch(share=share)
