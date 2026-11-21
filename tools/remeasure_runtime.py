"""Re-measure the CPU runtime of the Experiment 6 (mixed-domain) models on an idle CPU,
with the shared timing protocol (batch 1, 10 warm-up passes, mean over 200 test images),
and store it in results/exp6_mixed_domain.json. The first measurement was taken while
other jobs were running on the same laptop.

Run:  python tools/remeasure_runtime.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch_common as tc          # noqa: E402
import hog_svm as A                # noqa: E402
import scratch_cnn as B            # noqa: E402
import pretrained_mobilenet as C   # noqa: E402
from common import MODELS_DIR, RESULTS_DIR, load_array_cache, load_json, load_split, save_json, time_per_image  # noqa: E402

d, test = MODELS_DIR / "mixed", load_split()["test"]
out = load_json(RESULTS_DIR / "exp6_mixed_domain.json")
a = A.HogSvmClassifier.load(d / "hog_svm.npz")
ms = {"A": time_per_image(lambda im: a.predict_proba(im), load_array_cache(test, 128, "test_128")),
      "B": tc.runtime_ms(B.load_trained(d / "scratch_cnn.pt"), test, B.IMG_SIZE, B.MEAN, B.STD, "cpu", tag="test_128"),
      "C": tc.runtime_ms(C.load_trained(d / "mobilenetv3_large.pt"), test, C.IMG_SIZE, C.MEAN, C.STD, "cpu", tag="test_224")}
for k, v in ms.items():
    out["methods"][k]["train"]["runtime_ms_cpu_first_measurement"] = out["methods"][k]["train"].get(
        "runtime_ms_cpu_first_measurement", out["methods"][k]["train"]["runtime_ms_cpu"])
    out["methods"][k]["train"]["runtime_ms_cpu"] = v
    print(f"{k}: {v:.1f} ms/img (CPU, idle)  [first measurement under load: "
          f"{out['methods'][k]['train']['runtime_ms_cpu_first_measurement']:.1f}]")
save_json(out, RESULTS_DIR / "exp6_mixed_domain.json")
