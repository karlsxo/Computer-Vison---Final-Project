"""Build the PDF report (Introduction, Methods, Experiments, Results, ...) from the
result files and figures. HTML -> PDF with headless Microsoft Edge (or Chrome).

Run:  python tools/build_report.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from report_text import TEXT   # noqa: E402  (the written analysis, kept separate from layout)

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NAMES = {"A": "HOG + linear SVM", "A+": "HOG + colour histogram + SVM (variant)",
         "B": "4-block CNN (from scratch)", "C": "MobileNetV3-Large (pretrained)"}


def J(name):
    return json.loads((RES / name).read_text())


e1, e2, e3, e4, e5 = (J("exp1_main.json"), J("exp2_domain_shift.json"), J("exp3_small_data.json"),
                      J("exp4_efficiency.json"), J("exp5_error_analysis.json"))
rA, rB, rC = J("method_A.json"), J("method_B.json"), J("method_C.json")
e6, WV, LK = J("exp6_mixed_domain.json"), J("web_verification.json"), J("exp6_leakage_check.json")
mixed = json.loads((ROOT / "splits" / "mixed.json").read_text())
split = json.loads((ROOT / "splits" / "split.json").read_text())
M = e1["methods"]
RW = e2["results"]["realworld"]
M6 = e6["methods"]
REC6 = max("ABC", key=lambda k: M6[k]["realworld"]["macro_f1"])   # the method BIN answers with


def fig(name, caption, width="100%"):
    return (f'<figure><img src="../results/figures/{name}" style="width:{width}">'
            f"<figcaption>{caption}</figcaption></figure>")


def table(headers, rows, cls=""):
    h = "".join(f"<th>{x}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


# ---------------------------------------------------------------- tables
main_rows = []
for k, slot in [("A", "A. Classical"), ("B", "B. From scratch"), ("C", "C. Pretrained")]:
    m = M[k]
    main_rows.append([f"<b>{NAMES[k]}</b>", slot, f"{m['accuracy']:.1f} <span class=ci>[{m['acc_ci95'][0]:.1f}, {m['acc_ci95'][1]:.1f}]</span>",
                      f"{m['macro_f1']:.3f}", f"{m['runtime_ms_cpu']:.1f}", f"{m['params_M']:.2f}",
                      f"{m['train_time_min']:.1f}"])
m = M["A+"]
main_rows.append([f"<i>{NAMES['A+']}</i>", "A. variant", f"{m['accuracy']:.1f} <span class=ci>[{m['acc_ci95'][0]:.1f}, {m['acc_ci95'][1]:.1f}]</span>",
                  f"{m['macro_f1']:.3f}", f"{m['runtime_ms_cpu']:.1f}", "0", f"{m['train_time_min']:.1f}"])
main_table = table(["Method", "Slot", "Top-1 acc. (%) [95% CI]", "Macro F1", "Runtime (ms/img, CPU)",
                    "Params (M)", "Train time (min, CPU)"], main_rows, "results")

counts = split["stats"]["counts"]
split_table = table(["class", "train", "val", "test"],
                    [[c, counts["train"][c], counts["val"][c], counts["test"][c]] for c in CLASSES] +
                    [["<b>total</b>"] + [f"<b>{sum(counts[s].values())}</b>" for s in ("train", "val", "test")]])

per_class_rows = []
for c in CLASSES:
    row = [c]
    for k in ("A", "B", "C"):
        pc = M[k]["per_class"][c]
        row.append(f"{pc['precision']:.2f} / {pc['recall']:.2f} / {pc['f1']:.2f}")
    per_class_rows.append(row)
per_class_table = table(["class"] + [f"{NAMES[k]}<br><small>P / R / F1</small>" for k in ("A", "B", "C")],
                        per_class_rows)

mc_rows = [[k.replace("_vs_", " vs "), v["only_first_correct"], v["only_second_correct"],
            f"{v['p_value']:.1e}"] for k, v in e1["mcnemar"].items()]
mc_table = table(["comparison", "only 1st correct", "only 2nd correct", "exact McNemar p"], mc_rows)

dom_rows = []
for k in ("A", "A+", "B", "C"):
    r = RW[k]
    dom_rows.append([NAMES[k], f"{M[k]['accuracy']:.1f}", f"{r['accuracy']:.1f} <span class=ci>[{r['acc_ci95'][0]:.1f}, {r['acc_ci95'][1]:.1f}]</span>",
                     f"{r['macro_f1']:.3f}", f"{r['acc_drop_pts']:.1f}", f"{r['relative_retention'] * 100:.0f}%"])
dom_table = table(["Method", "TrashNet test acc. (%)", "Real-world acc. (%) [95% CI]", "Real-world macro F1",
                   "Drop (points)", "Accuracy retained"], dom_rows)

sd_rows = []
for k in ("A", "B", "C"):
    row = [NAMES[k]]
    for n in (10, 25, 50):
        r = e3["runs"][f"{k}_{n}"]
        row.append(f"{r['accuracy']:.1f} / {r['macro_f1']:.2f}")
    row.append(f"{e3['full'][k]['accuracy']:.1f} / {e3['full'][k]['macro_f1']:.2f}")
    sd_rows.append(row)
sd_table = table(["Method", "10 / class (60)", "25 / class (150)", "50 / class (300)",
                  f"all ({e3['full']['A']['n_train']})"], sd_rows)

eff_rows = [[v["name"], f"{v['accuracy']:.1f}", f"{v['runtime_ms_cpu']:.1f}", f"{v['params_M']:.2f}",
             f"{v['model_size_MB']:.1f}", f"{v['train_time_min']:.1f}"] for v in e4.values()]
eff_table = table(["Method", "Acc. (%)", "Runtime (ms/img, CPU)", "Params (M)", "Model file (MB)",
                   "Train time (min, CPU)"], eff_rows)

conf_rows = []
for k in ("A", "B", "C"):
    conf_rows.append([NAMES[k], ", ".join(f"{d['true']}&rarr;{d['pred']} ({d['count']})"
                                          for d in e5["top_confusions"][k][:3])])
conf_table = table(["Method", "Most frequent confusions on the test split (true &rarr; predicted, count)"], conf_rows)

sc = rA["sanity_checks"]
sanity_table = table(["Check", "Result"], [
    ["(a) synthetic vertical bar / horizontal bar / 45&deg; edge",
     " / ".join(f"peak {int(t['peak_range_deg'][0])}&ndash;{int(t['peak_range_deg'][1])}&deg; "
                f"({'pass' if t['pass'] else 'FAIL'})" for t in sc["synthetic"].values())],
    ["(b) our HOG vs scikit-image hog() on 8 test images",
     f"max |diff| = {sc['skimage_reference']['max_abs_diff']:.1e}, correlation = {sc['skimage_reference']['min_correlation']:.6f}"],
    ["(c) our SVM on a 3-class toy dataset", f"{sc['toy_svm']['train_accuracy']:.0f}% training accuracy"]])

mix_rows = [[c, s["candidates"], s["near_trashnet"], s["near_realworld_test"], s["usable"], s["web_train"], s["web_val"]]
            for c, s in mixed["stats"]["per_class"].items()]
mix_table = table(["class", "web candidates", "near-copies of TrashNet (removed)", "near-copies of test photos (removed)",
                   "usable", "added to train", "added to val"], mix_rows)
e6_rows = []
for k in "ABC":
    m, b = M6[k], M6[k]["baseline_trashnet_only"]
    e6_rows.append([("&#9733; " if k == REC6 else "") + NAMES[k],
                    f"{b['trashnet_test_acc']:.1f} &rarr; <b>{m['trashnet_test']['accuracy']:.1f}</b>",
                    f"{b['realworld_acc']:.1f} &rarr; <b>{m['realworld']['accuracy']:.1f}</b> <span class=ci>[{m['realworld']['acc_ci95'][0]:.1f}, {m['realworld']['acc_ci95'][1]:.1f}]</span>",
                    f"{m['realworld']['macro_f1']:.3f}", f"{m['threshold']['threshold']:.2f}",
                    (f"{m['realworld_at_threshold']['coverage'] * 100:.0f}% / {m['realworld_at_threshold']['accuracy']:.1f}%"
                     if m['realworld_at_threshold']['accuracy'] is not None else "0% / n.a."),
                    f"{m['train']['train_time_min']:.0f}"])
e6_table = table(["Method", "TrashNet test acc. (%)<br><small>TrashNet-only &rarr; mixed</small>",
                  "Real-world acc. (%) [95% CI]<br><small>TrashNet-only &rarr; mixed</small>", "Real-world macro-F1",
                  "'Not sure' threshold", "Real-world answered / accuracy of answers", "Train time (min, CPU)"], e6_rows, "results")
wv_rows = [[s.replace("trashnet_test", "TrashNet test (376)").replace("realworld", "Real-world (600)"), NAMES[k],
            f"{v['agreement_with_python_%']:.1f}%", f"{v['phone_accuracy_%']:.1f}%", f"{v['python_accuracy_%']:.1f}%"]
           for s, d in WV["sets"].items() for k, v in d.items()]
wv_table = table(["Test set", "Method", "Same prediction as Python", "Accuracy in the app", "Accuracy in Python"], wv_rows)

T = {k: v.format(**{
    "A_acc": M["A"]["accuracy"], "B_acc": M["B"]["accuracy"], "C_acc": M["C"]["accuracy"],
    "Av_acc": M["A+"]["accuracy"], "A_f1": M["A"]["macro_f1"], "B_f1": M["B"]["macro_f1"],
    "C_f1": M["C"]["macro_f1"], "A_rw": RW["A"]["accuracy"], "B_rw": RW["B"]["accuracy"],
    "C_rw": RW["C"]["accuracy"], "Av_rw": RW["A+"]["accuracy"],
    "A_ms": M["A"]["runtime_ms_cpu"], "B_ms": M["B"]["runtime_ms_cpu"], "C_ms": M["C"]["runtime_ms_cpu"],
    "B_par": M["B"]["params_M"], "C_par": M["C"]["params_M"],
    "A_10": e3["runs"]["A_10"]["accuracy"], "B_10": e3["runs"]["B_10"]["accuracy"], "C_10": e3["runs"]["C_10"]["accuracy"],
    "A_50": e3["runs"]["A_50"]["accuracy"], "B_50": e3["runs"]["B_50"]["accuracy"], "C_50": e3["runs"]["C_50"]["accuracy"],
    "vote": e5["majority_vote_accuracy"], **e5["agreement"],
    "B_ep": rB["train_info"]["best_epoch"], "B_eps": rB["train_info"]["epochs_run"],
    "C_ep": rC["train_info"]["best_epoch"], "C_eps": rC["train_info"]["epochs_run"],
    "C_train": rC["trainable_params_M"],
    **{f"{k}6_tn": M6[k]["trashnet_test"]["accuracy"] for k in "ABC"},
    **{f"{k}6_rw": M6[k]["realworld"]["accuracy"] for k in "ABC"},
    **{f"{k}6_rwf1": M6[k]["realworld"]["macro_f1"] for k in "ABC"},
    "rec6": NAMES[REC6], "thr6": M6[REC6]["threshold"]["threshold"] * 100,
    "cov6": M6[REC6]["realworld_at_threshold"]["coverage"] * 100,
    "sel6": M6[REC6]["realworld_at_threshold"]["accuracy"],
    "valsel6": (M6[REC6]["threshold"]["selective_accuracy"] or 0) * 100,
    "n_mixed": e6["n_train"], "n_web": len(mixed["web_train"]), "n_webval": len(mixed["web_val"]),
    "wv_min": min(v["agreement_with_python_%"] for d in WV["sets"].values() for v in d.values()),
    **{f"{k}6_cl": LK["accuracy_all_vs_clean"][k]["clean_subset"] for k in "ABC"},
    "n_clean": LK["accuracy_all_vs_clean"]["C"]["n_clean"],
}) for k, v in TEXT.items()}

CSS = """
@page { size: A4; margin: 18mm 17mm 18mm 17mm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: 10.3pt; line-height: 1.45; color: #1d2329; }
h1 { font-size: 21pt; margin: 0 0 4px; color: #123b2f; line-height: 1.2; }
h2 { font-size: 14pt; color: #123b2f; margin: 20px 0 6px; border-bottom: 1px solid #c9d6cf; padding-bottom: 3px; }
h3 { font-size: 11.5pt; color: #1f5641; margin: 14px 0 4px; }
.sub { color: #55606a; margin-bottom: 10px; }
.abstract { background: #eef5f1; padding: 10px 14px; border-radius: 6px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 10px; font-size: 9.2pt; }
th { background: #1f5641; color: white; text-align: left; padding: 4px 6px; }
td { padding: 3px 6px; border-bottom: 1px solid #dfe5e1; vertical-align: top; }
table.results td:nth-child(n+3) { text-align: right; }
.ci { color: #6b7780; font-size: 8.4pt; }
figure { margin: 8px 0 12px; page-break-inside: avoid; text-align: center; }
figcaption { font-size: 8.8pt; color: #4a555e; margin-top: 3px; text-align: left; }
code, pre { font-family: Consolas, monospace; font-size: 8.6pt; }
pre { background: #f4f6f5; padding: 6px 9px; border-radius: 4px; white-space: pre; }
.pb { page-break-before: always; }
h2, h3 { break-after: avoid; page-break-after: avoid; }
table, figure, pre, .key { break-inside: avoid; page-break-inside: avoid; }
p { orphans: 3; widows: 3; }
.key { background: #fff6e0; padding: 8px 12px; border-radius: 6px; }
ol.refs li { margin-bottom: 3px; font-size: 9.3pt; }
"""

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Trash Sorting Report</title>
<style>{CSS}</style></head><body>
<h1>Trash Sorting by Material: Hand-Designed Features vs. a Small CNN vs. a Pretrained Mobile Model</h1>
<div class="sub">Computer Vision Group Project &middot; West Visayas State University &middot; with <b>BIN</b>, a trash identification app &middot; Group __ &middot; Members: ____________________</div>
<div class="abstract"><b>Abstract.</b> {T['abstract']}</div>

<h2>1. Introduction</h2>
{T['intro']}
<p><b>Research question.</b> <i>Which method best sorts trash by material, and does it still work on messy
real-world photos after training on clean ones?</i> Sub-questions: (1) how much does each method gain or lose
with only 10, 25 or 50 training images per class; (2) which materials does each method confuse, and why;
(3) what does each method cost in runtime, parameters and training time; (4) which method should the
application recommend by default?</p>

<h2>2. Methods</h2>
<h3>2.1 Data and a leakage-safe split</h3>
{T['data']}
{split_table}
{fig('data_class_counts.png', 'Figure 1. Class counts per split after cleaning. "trash" is the minority class (5 %).', '70%')}
{fig('data_domain_examples.png', 'Figure 2. The domain shift: TrashNet studio photos (top) vs. real-world web photos used only for testing (bottom).')}

<h3>2.2 Method A &mdash; HOG + linear SVM (classical, implemented from scratch)</h3>
{T['methodA']}
{sanity_table}
{fig('A_sanity_checks.png', 'Figure 3. Sanity checks of Method A: orientation histograms of synthetic images peak at the expected angle; our SVM separates a toy dataset.')}

<h3>2.3 Method B &mdash; our 4-block CNN trained from scratch</h3>
<pre>input 3x128x128
Block 1  Conv3x3(3->32)    BN ReLU                            MaxPool -> 32x64x64
Block 2  Conv3x3(32->64)   BN ReLU                            MaxPool -> 64x32x32
Block 3  Conv3x3(64->128)  BN ReLU  Conv3x3(128->128) BN ReLU MaxPool -> 128x16x16
Block 4  Conv3x3(128->256) BN ReLU  Conv3x3(256->256) BN ReLU MaxPool -> 256x8x8
GlobalAvgPool -> Dropout(0.4) -> Linear(256->6)        1.13 M parameters</pre>
{T['methodB']}
{fig('data_augmentation.png', 'Figure 4. Our training-time augmentation (shown at full strength; the final runs use strength 0.6).', '80%')}

<h3>2.4 Method C &mdash; MobileNetV3-Large, ImageNet-pretrained and fine-tuned</h3>
{T['methodC']}

<h3>2.5 Evaluation protocol</h3>
{T['protocol']}

<h2>3. Experiments</h2>
{T['experiments']}

<h2 class="pb">4. Results</h2>
<h3>4.1 Experiment 1 &mdash; main comparison</h3>
{main_table}
<p style="font-size:8.8pt;color:#4a555e">Runtime: batch size 1 on an Intel i5-1135G7 CPU, mean of 200 test images after 10 warm-up passes,
including normalisation (and HOG extraction for A). Training times are CPU wall-clock times on the same laptop
(the GPU could not be used, see the documentation).</p>
{fig('exp1_training_curves.png', f"Figure 5. Training and validation curves of B (solid) and C (dashed); dotted lines mark the restored checkpoints (B: epoch {rB['train_info']['best_epoch']}, C: epoch {rC['train_info']['best_epoch']}). Right: convergence of our SVM objective.")}
{fig('exp1_confusion_matrices.png', 'Figure 6. Confusion matrices on the 376-image test split (rows = true class).')}
{per_class_table}
{fig('exp1_per_class_f1.png', 'Figure 7. Per-class F1 on the test split.', '80%')}
{mc_table}
{T['res1']}

<h3>4.2 Experiment 2 &mdash; domain shift</h3>
{dom_table}
{fig('exp2_domain_shift.png', 'Figure 8. Accuracy on TrashNet test vs. unseen real-world web photos, and per-class F1 on the real-world set.')}
{fig('exp2_confusion_realworld.png', 'Figure 9. Confusion matrices on the 600 real-world photos.')}
{T['res2']}

<h3>4.3 Experiment 3 &mdash; small data</h3>
{sd_table}
<p style="font-size:8.8pt;color:#4a555e">Cells: test accuracy (%) / macro F1.</p>
{fig('exp3_small_data.png', 'Figure 10. Learning curves: test accuracy and macro F1 versus number of training images (log scale).')}
{T['res3']}

<h3>4.4 Experiment 4 &mdash; efficiency</h3>
{eff_table}
{fig('exp4_efficiency.png', 'Figure 11. Accuracy versus inference cost (left) and training cost (right). Marker area grows with parameter count.')}
{T['res4']}

<h3>4.5 Experiment 5 &mdash; error analysis</h3>
{conf_table}
{fig('exp5_failures.png', 'Figure 12. Random failure cases of each method (true and predicted labels).')}
{fig('exp5_gradcam.png', 'Figure 13. What the methods look at: gradient magnitude used by HOG (A) and Grad-CAM heat maps of B and C on one test image per class.')}
{T['res5']}

<h3>4.6 Experiment 6 &mdash; mixed-domain training</h3>
{T['data6']}
{mix_table}
{e6_table}
<p style="font-size:8.8pt;color:#4a555e">Same test sets as Experiments 1 and 2. Arrows: model trained on TrashNet only &rarr; same method retrained
on TrashNet + web photos with identical architecture and hyper-parameters. &#9733; = method BIN answers with (best real-world macro-F1).</p>
{fig('exp6_mixed_domain.png', 'Figure 14. Experiment 6: accuracy before (grey) and after (colour) adding real-world photos to training, on both unchanged test sets.')}
{fig('exp6_confusion_realworld.png', 'Figure 15. Confusion matrices of the mixed-domain models on the 600 real-world test photos.')}
{fig('exp6_confidence_threshold.png', 'Figure 16. Calibrating the "not sure" threshold on the validation split: share of photos answered and accuracy of those answers.', '62%')}
{T['res6']}

<h3>4.7 Answer to the research question</h3>
<div class="key">{T['answer']}</div>

<h2 class="pb">5. System: BIN</h2>
{T['system']}
<figure><img src="bin_phone_screens.png" style="width:100%"><figcaption>Figure 17. The BIN phone app: it opens on the camera with a scanner frame; a result with what to do; a "not sure" answer; the scan history; the about page with the measured accuracy.</figcaption></figure>
{wv_table}
<p style="font-size:8.8pt;color:#4a555e">Phone-app check (tools/verify_web.py): every test photo classified by the app's own code (PIL-exact resize, JavaScript HOG + SVM, ONNX networks) in a headless browser and compared with the Python models.</p>
<figure><img src="app_screenshot.png" style="width:72%"><figcaption>Figure 18. The desktop/Colab version of BIN (Gradio): BIN's answer is drawn above the photo, every method's label, confidence and runtime underneath, with the reason and what to do.</figcaption></figure>
{fig('system_demo.png', 'Figure 19. BIN on six real-world web photos it has never seen (one per class); the green band shows its answer or NOT SURE.', '88%')}

<h2>6. Conclusion and limitations</h2>
{T['conclusion']}

<h2>References</h2>
<ol class="refs">
<li>R. Szeliski, <i>Computer Vision: Algorithms and Applications</i>, 2nd ed., Springer, 2022.</li>
<li>R. Krishna et al., <i>Computer Vision: Foundations and Applications</i> (Stanford CS131 notes), 2017.</li>
<li>PyTorch and torchvision documentation, https://pytorch.org/vision/stable/models.html.</li>
<li>N. Dalal and B. Triggs, &ldquo;Histograms of Oriented Gradients for Human Detection,&rdquo; CVPR, 2005.</li>
<li>A. Howard et al., &ldquo;Searching for MobileNetV3,&rdquo; ICCV, 2019.</li>
<li>G. Thung and M. Yang, &ldquo;Classification of Trash for Recyclability Status,&rdquo; CS229 project report, Stanford, 2016 (TrashNet, https://github.com/garythung/trashnet).</li>
<li>M. Mohamed, &ldquo;Garbage Classification (12 classes),&rdquo; Kaggle dataset, 2021; mirror: https://huggingface.co/datasets/UdaraChamidu/Garbage-Classification-with-12-classes.</li>
<li>R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang and C.-J. Lin, &ldquo;LIBLINEAR: A Library for Large Linear Classification,&rdquo; JMLR 9, 2008.</li>
<li>D. P. Kingma and J. Ba, &ldquo;Adam: A Method for Stochastic Optimization,&rdquo; ICLR, 2015.</li>
<li>I. Loshchilov and F. Hutter, &ldquo;Decoupled Weight Decay Regularization,&rdquo; ICLR, 2019.</li>
<li>R. R. Selvaraju et al., &ldquo;Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,&rdquo; ICCV, 2017.</li>
<li>K. He, X. Zhang, S. Ren and J. Sun, &ldquo;Delving Deep into Rectifiers,&rdquo; ICCV, 2015 (He initialisation).</li>
<li>S. Ioffe and C. Szegedy, &ldquo;Batch Normalization,&rdquo; ICML, 2015.</li>
<li>Republic Act No. 9003, Ecological Solid Waste Management Act of 2000, Philippines.</li>
</ol>

<h2>Contribution statement</h2>
{T['contrib']}
</body></html>"""

html_path = OUT / "report.html"
html_path.write_text(html, encoding="utf-8")
pdf_path = OUT / "Trash_Sorting_Report.pdf"
browsers = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            shutil.which("chromium") or "", shutil.which("google-chrome") or ""]
exe = next((b for b in browsers if b and Path(b).exists()), None)
if exe:
    subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()],
                   check=True, capture_output=True, timeout=180)
    print("wrote", pdf_path)
else:
    print("No Edge/Chrome found: open report/report.html and print to PDF.")
