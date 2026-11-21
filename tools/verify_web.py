"""Check that the BIN phone app (web/) predicts exactly like the Python models.

Serves the project folder locally, opens tools/verify_web.html in a headless
browser, classifies every image of the TrashNet test split (376) and of the
real-world test set (600) with the JavaScript pipeline (PIL-exact resize,
JS HOG + SVM, ONNX CNNs) and compares with the Python predictions of the same
models: results/exp6_mixed_domain.json for the mixed-domain models (default
export), or results/method_*.json + exp2 for the TrashNet-only models.

Run:  python tools/verify_web.py      (needs: pip install playwright; Edge or Chrome)
"""
import functools
import http.server
import json
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def python_predictions(source):
    if source == "mixed":
        M = json.loads((ROOT / "results" / "exp6_mixed_domain.json").read_text())["methods"]
        return {"trashnet_test": {k: M[k]["trashnet_test"]["pred"] for k in "ABC"},
                "realworld": {k: M[k]["realworld"]["pred"] for k in "ABC"}}
    e2 = json.loads((ROOT / "results" / "exp2_domain_shift.json").read_text())["results"]["realworld"]
    return {"trashnet_test": {k: json.loads((ROOT / "results" / f"method_{k}.json").read_text())["test_pred"] for k in "ABC"},
            "realworld": {k: e2[k]["pred"] for k in "ABC"}}


def main():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8765), functools.partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    meta = json.loads((ROOT / "web" / "models" / "meta.json").read_text())
    sets = {"trashnet_test": json.loads((ROOT / "splits" / "split.json").read_text())["splits"]["test"],
            "realworld": json.loads((ROOT / "splits" / "realworld.json").read_text())["items"]}
    py = python_predictions(meta["source"])
    report = {"source": meta["source"], "sets": {}}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page()
        pg.goto("http://127.0.0.1:8765/tools/verify_web.html")
        pg.wait_for_function("!!(window.binApp && window.decodeImage)")
        pg.evaluate("binApp.loadModels()")
        for name, items in sets.items():
            preds = pg.evaluate("""async (paths) => {
                const out = []; let ms = {A: 0, B: 0, C: 0};
                for (const p of paths) {
                    const blob = await (await fetch('/' + p)).blob();
                    const r = await binApp.classify(await decodeImage(blob));
                    out.push([r.A.index, r.B.index, r.C.index]);
                    for (const k of ['A','B','C']) ms[k] += r[k].ms;
                }
                for (const k of ['A','B','C']) ms[k] /= paths.length;
                return {out, ms};
            }""", [path for path, _ in items])
            y = [label for _, label in items]
            report["sets"][name] = {}
            print(f"{name} ({len(y)} images, {meta['source']} models):")
            for i, k in enumerate("ABC"):
                js = [row[i] for row in preds["out"]]
                agree = sum(a == b for a, b in zip(js, py[name][k])) / len(y) * 100
                acc_js = sum(a == t for a, t in zip(js, y)) / len(y) * 100
                acc_py = sum(a == t for a, t in zip(py[name][k], y)) / len(y) * 100
                report["sets"][name][k] = {"agreement_with_python_%": round(agree, 2), "phone_accuracy_%": round(acc_js, 2),
                                           "python_accuracy_%": round(acc_py, 2),
                                           "ms_per_image_desktop_browser": round(preds["ms"][k], 1)}
                print(f"  {k}: agreement {agree:6.2f}%  accuracy phone {acc_js:5.1f}% vs python {acc_py:5.1f}%  "
                      f"({preds['ms'][k]:.1f} ms/img in browser)")
        b.close()
    srv.shutdown()
    (ROOT / "results" / "web_verification.json").write_text(json.dumps(report, indent=1))
    worst = min(v["agreement_with_python_%"] for s in report["sets"].values() for v in s.values())
    if worst < 99.0:
        sys.exit(f"Agreement too low ({worst}%)")


if __name__ == "__main__":
    main()
