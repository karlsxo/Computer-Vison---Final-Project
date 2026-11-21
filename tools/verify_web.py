"""Check that the phone app (web/) predicts like the Python models.

Serves the project folder locally, opens tools/verify_web.html in a headless
browser, classifies every test image with the JavaScript pipeline (our PIL-exact
resize, JS HOG + SVM, ONNX CNNs) and compares with the Python predictions saved
in results/method_*.json.

Run:  python tools/verify_web.py      (needs: pip install playwright; Edge or Chrome)
"""
import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def main():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = functools.partial(Quiet, directory=str(ROOT))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8765), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    split = json.loads((ROOT / "splits" / "split.json").read_text())["splits"]["test"]
    py = {k: json.loads((ROOT / "results" / f"method_{k}.json").read_text())["test_pred"] for k in "ABC"}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page()
        pg.goto("http://127.0.0.1:8765/tools/verify_web.html")
        pg.wait_for_function("!!(window.trashSorter && window.decodeImage)")
        pg.evaluate("trashSorter.loadModels()")
        preds = pg.evaluate("""async (paths) => {
            const out = []; let ms = {A: 0, B: 0, C: 0};
            for (const p of paths) {
                const blob = await (await fetch('/' + p)).blob();
                const r = await trashSorter.classify(await decodeImage(blob));
                out.push([r.A.index, r.B.index, r.C.index]);
                for (const k of ['A','B','C']) ms[k] += r[k].ms;
            }
            for (const k of ['A','B','C']) ms[k] /= paths.length;
            return {out, ms};
        }""", [path for path, _ in split])
        b.close()
    srv.shutdown()
    y = [label for _, label in split]
    report = {}
    for i, k in enumerate("ABC"):
        js = [row[i] for row in preds["out"]]
        agree = sum(a == b for a, b in zip(js, py[k])) / len(y) * 100
        acc_js = sum(a == t for a, t in zip(js, y)) / len(y) * 100
        acc_py = sum(a == t for a, t in zip(py[k], y)) / len(y) * 100
        report[k] = {"agreement_with_python_%": round(agree, 1), "phone_accuracy_%": round(acc_js, 1),
                     "python_accuracy_%": round(acc_py, 1), "phone_ms_per_image_desktop_browser": round(preds["ms"][k], 1)}
        print(f"  {k}: agreement {agree:5.1f}%  accuracy phone {acc_js:5.1f}% vs python {acc_py:5.1f}%  "
              f"({preds['ms'][k]:.1f} ms/img in browser)")
    (ROOT / "results" / "web_verification.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
