"""End-to-end test of the BIN phone app in an emulated phone (390x844) with a fake camera.

Serves web/ locally, opens the app with no route (it must start on the scanner with the
camera running), identifies a fake-camera frame and an uploaded photo, and visits history, about and the help pages. Fails on any JavaScript error.
Screenshots are written to the folder given as the first argument.

Run:  python tools/test_phone_app.py <screenshot-folder>
"""
import functools
import http.server
import sys
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "phone_screens")
OUT.mkdir(parents=True, exist_ok=True)


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def main():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8766), functools.partial(Quiet, directory=str(ROOT / "web")))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    errors, base = [], "http://127.0.0.1:8766/"
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True,
                              args=["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"])
        ctx = b.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True,
                            has_touch=True, permissions=["camera"])
        pg = ctx.new_page()
        pg.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        pg.goto(base)                                        # no route: must open the scanner
        pg.wait_for_selector("#video")
        pg.wait_for_function("document.querySelector('#video').videoWidth > 0", timeout=20000)
        pg.wait_for_selector("#status.ready", timeout=180000)
        assert "#/" not in pg.url or pg.url.endswith("#/scan") or pg.url == base
        pg.screenshot(path=str(OUT / "1_scan.png"))
        pg.click("#shutter"); pg.wait_for_url("**#/result", timeout=60000); pg.wait_for_timeout(500)
        pg.screenshot(path=str(OUT / "2_result_camera.png"), full_page=True)
        pg.goto(base + "#/scan"); pg.wait_for_selector("#galleryInput", state="attached")
        pg.set_input_files("#galleryInput", str(ROOT / "app" / "examples" / "metal.jpg"))
        pg.wait_for_url("**#/result", timeout=60000); pg.wait_for_timeout(500)
        pg.screenshot(path=str(OUT / "3_result_upload.png"), full_page=True)
        assert pg.locator("[data-pick]").count() == 0       # BIN decides the material; no manual picking
        # comparison table in the course format (7 columns, 3 methods) + explanation of the three slots
        assert pg.locator("table.cmp thead th").count() == 7
        assert pg.locator("table.cmp tbody tr").count() == 3
        assert pg.locator(".slot-card").count() == 3
        for page, name in [("#/history", "5_history"), ("#/about", "6_about"), ("#/about/a/ai", "7_how_bin_works"),
                           ("#/about/m/glass", "8_material")]:
            pg.goto(base + page); pg.wait_for_timeout(500)
            pg.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        assert pg.evaluate("JSON.parse(localStorage.getItem('bin.history.v1')).length") == 2   # both scans saved
        sw = pg.evaluate("navigator.serviceWorker.getRegistration().then(r => !!(r && r.active))")
        b.close()
    srv.shutdown()
    # five-screen figure for the report: camera, result, not sure, history, how BIN works
    from PIL import Image
    names = ["1_scan", "3_result_upload", "2_result_camera", "5_history", "7_how_bin_works"]
    ims = [Image.open(OUT / f"{n}.png").convert("RGB") for n in names]
    ims = [im.crop((0, 0, im.width, min(im.height, im.width * 844 // 390))) for im in ims]
    ims = [im.resize((390, im.height * 390 // im.width)) for im in ims]
    h = max(im.height for im in ims)
    sheet = Image.new("RGB", (len(ims) * 400 - 10, h), "white")
    for i, im in enumerate(ims):
        sheet.paste(im, (i * 400, 0))
    sheet.save(ROOT / "report" / "bin_phone_screens.png")
    print("service worker active:", sw)
    print("javascript errors:", errors or "none")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
