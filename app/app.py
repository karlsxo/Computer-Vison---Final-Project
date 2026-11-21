"""BIN - desktop / Colab version (Gradio).

Upload or capture a photo of one piece of trash: all three methods predict its
material (label, confidence and runtime are drawn on the image) and BIN answers
with the recommended method, or says "not sure". The phone version of BIN is in
web/ and uses the same models and rules.

Run locally:   python app/app.py          -> http://127.0.0.1:7860
In Colab:      python app/app.py --share  -> public *.gradio.live link
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr   # noqa: E402

from common import CLASSES       # noqa: E402
from system import METHOD_NAMES, SLOTS, BinSystem   # noqa: E402

sorter = BinSystem()


def run(image):
    if image is None:
        return None, "Upload or capture a photo first.", None
    preds = sorter.predict(image)
    annotated = sorter.annotate(image, preds)
    table = [[f"{'★ ' if k == sorter.recommended else ''}{k}. {METHOD_NAMES[k]}", SLOTS[k], p["label"],
              f"{p['confidence'] * 100:.1f}%", f"{p['ms']:.1f}"] + [f"{p['probs'][c] * 100:.0f}" for c in CLASSES]
             for k, p in preds.items()]
    return annotated, sorter.explain(preds), table


def build_ui():
    examples_dir = ROOT / "app" / "examples"
    examples = sorted(str(p) for p in examples_dir.glob("*.jpg")) if examples_dir.exists() else []
    s = sorter.summary
    rows = "\n".join(f"| {'★ ' if k == sorter.recommended else ''}{METHOD_NAMES[k]} | {SLOTS[k]} | {v['acc']:.1f}% | "
                     f"{v['rw_acc']:.1f}% | {v['ms']:.1f} |" for k, v in s.items())
    with gr.Blocks(title="BIN") as demo:
        gr.Markdown(
            "# BIN: what material is this trash?\n"
            "Upload or capture a photo of **one** item. Three computer-vision methods identify it as "
            "cardboard, glass, metal, paper, plastic or trash (residual); ★ = the method BIN answers with.\n\n"
            "| Method | Slot | TrashNet test acc. | Real-world acc. | CPU ms/img |\n|---|---|---|---|---|\n" + rows)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Photo of one trash item",
                               sources=["upload", "webcam", "clipboard"])
                btn = gr.Button("Identify", variant="primary")
                if examples:
                    gr.Examples(examples, inputs=inp)
            with gr.Column():
                out_img = gr.Image(label="Predictions drawn on the input", show_label=False)
                out_md = gr.Markdown()
        out_tab = gr.Dataframe(headers=["method", "slot", "prediction", "confidence", "runtime (ms)"]
                               + [f"{c} %" for c in CLASSES], label="All methods")
        btn.click(run, inp, [out_img, out_md, out_tab])
        inp.upload(run, inp, [out_img, out_md, out_tab])
        demo_img = os.environ.get("APP_DEMO_IMAGE")   # optional: classify an image on page load (screenshots)
        if demo_img:
            from PIL import Image
            demo.load(lambda: (Image.open(demo_img), *run(Image.open(demo_img))), None,
                      [inp, out_img, out_md, out_tab])
    return demo


if __name__ == "__main__":
    build_ui().launch(share="--share" in sys.argv)
