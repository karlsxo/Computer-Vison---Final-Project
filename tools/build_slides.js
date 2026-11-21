// Build the 10-minute presentation from the result files.
// Run:  node tools/build_slides.js     (from the project root or tools/)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const RES = path.join(ROOT, "results");
const FIG = path.join(RES, "figures");
const J = (f) => JSON.parse(fs.readFileSync(path.join(RES, f), "utf8"));
const e1 = J("exp1_main.json"), e2 = J("exp2_domain_shift.json"), e3 = J("exp3_small_data.json");
const e4 = J("exp4_efficiency.json"), e5 = J("exp5_error_analysis.json");
const rA = J("method_A.json"), rB = J("method_B.json"), rC = J("method_C.json");
const split = JSON.parse(fs.readFileSync(path.join(ROOT, "splits", "split.json"), "utf8"));
const M = e1.methods, RW = e2.results.realworld;
const f1 = (x) => x.toFixed(1), f3 = (x) => x.toFixed(3);

const THEME = {
  name: "Recycle Green", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B2A24", lt1: "FFFFFF", dk2: "123B2F", lt2: "EEF5F1",
            accent1: "2C7A4B", accent2: "97BC62", accent3: "D1495B", accent4: "00798C",
            accent5: "30638E", accent6: "EDAE49", hlink: "00798C", folHlink: "30638E" },
};
const HEX = THEME.colors;
const COL = { A: "D1495B", B: "00798C", C: "30638E", Av: "EDAE49" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";          // 13.33 x 7.5 in
pres.title = "Trash Sorting by Material";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "DARK", background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.2, w: 11.7, h: 1.8, fontSize: 40,
      bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.15, w: 11.7, h: 1.6, fontSize: 18,
      color: C.background2, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: HEX.lt1 },
  margin: [0.5, 0.6, 0.5, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85, fontSize: 30,
      bold: true, color: C.text2, fontFace: THEME.headFontFace, valign: "middle" }, text: "" } },
    { text: { text: "Trash Sorting by Material | WVSU Computer Vision", options: { x: 0.6, y: 7.05, w: 7, h: 0.3,
      fontSize: 10, color: "6B7780", isTextBox: true } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: "6B7780" },
});

// Fit a PNG inside the box (x, y, w, h) without distortion: read its real size from the
// IHDR header, scale to fit, centre horizontally, align to the top.
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fitImage(s, file, x, y, w, h, alt) {
  const d = pngSize(file);
  const r = Math.min(w / d.w, h / d.h);
  const iw = d.w * r, ih = d.h * r;
  s.addImage({ path: file, x: x + (w - iw) / 2, y, w: iw, h: ih, altText: alt });
}
const img = (s, name, x, y, w, h, alt) => fitImage(s, path.join(FIG, name), x, y, w, h, alt);
const card = (s, x, y, w, h, fill) => s.addShape(pres.shapes.ROUNDED_RECTANGLE,
  { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.12 });
const txt = (s, text, o) => s.addText(text, Object.assign({ isTextBox: true, margin: 0, color: C.text1, fontSize: 15 }, o));
const bullets = (s, items, o) => s.addText(items.map((t, i) => ({ text: t, options: { bullet: true,
  breakLine: i < items.length - 1 } })), Object.assign({ isTextBox: true, fontSize: 15, color: C.text1,
  paraSpaceAfter: 8, valign: "top" }, o));

// ---------------------------------------------------------------- 1 title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
s.addText("Trash Sorting by Material", { placeholder: "title" });
s.addText([{ text: "Hand-designed features vs. a small CNN vs. a pretrained mobile model", options: { breakLine: true } },
  { text: "with BIN, a trash identification app  ·  Computer Vision Group Project | West Visayas State University | Group __", options: { fontSize: 14 } }],
  { placeholder: "body" });
s.addNotes("Introduce the group. One sentence: we compared three families of computer-vision methods on the job of telling what material a piece of trash is made of, and built an app around them.");

// ---------------------------------------------------------------- 2 problem
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Why sort trash with a camera?", { placeholder: "title" });
bullets(s, ["RA 9003 requires segregation at source, but sorting in barangay MRFs and schools is manual and inconsistent",
  "Task: one photo of one item -> cardboard, glass, metal, paper, plastic or trash (residual)",
  "Hard because materials look alike (paper vs cardboard, glass vs clear plastic) and studio photos differ from real ones"],
  { x: 0.6, y: 1.45, w: 6.6, h: 3.6 });
card(s, 7.6, 1.45, 5.1, 3.9, HEX.lt2);
txt(s, "Research question", { x: 7.9, y: 1.7, w: 4.5, h: 0.4, bold: true, fontSize: 16, color: C.accent1 });
txt(s, "Which method best sorts trash by material, and does it still work on messy real-world photos after training on clean ones?",
  { x: 7.9, y: 2.2, w: 4.5, h: 1.6, fontSize: 18, italic: true, fontFace: THEME.headFontFace, color: C.text2 });
txt(s, "Also: small data (10/25/50 per class), confusions, cost, and which method the app should recommend.",
  { x: 7.9, y: 4.0, w: 4.5, h: 1.0, fontSize: 13, color: "4A555E" });
s.addNotes("Explain the local motivation, then read the research question. Mention the four sub-questions briefly.");

// ---------------------------------------------------------------- 3 data
pres.addSection({ title: "Methods" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Methods" });
s.addText("Data: one leakage-safe split for all three methods", { placeholder: "title" });
const st = split.stats;
const stats = [["2,519", "TrashNet images kept\n(1,764 / 379 / 376)"], ["6", "identical files with\nconflicting labels dropped"],
  ["29", "near-duplicate groups\nkept in one split"], ["600", "unseen real-world\nweb photos (100/class)"]];
stats.forEach(([n, l], i) => {
  const x = 0.6 + i * 3.1;
  card(s, x, 1.4, 2.85, 1.75, HEX.lt2);
  txt(s, n, { x: x + 0.2, y: 1.5, w: 2.45, h: 0.8, fontSize: 36, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
  txt(s, l, { x: x + 0.2, y: 2.3, w: 2.45, h: 0.75, fontSize: 12, color: "4A555E" });
});
img(s, "data_domain_examples.png", 0.6, 3.4, 8.0, 3.5, "TrashNet studio photos vs real-world web photos");
bullets(s, ["Near-duplicates found with a background-removed 'detail signature' (dHash failed on white backgrounds)",
  "Seeded, stratified, group-aware 70/15/15 split saved to splits/split.json",
  "490 web images were TrashNet copies and were excluded"], { x: 8.9, y: 3.5, w: 3.85, h: 3.3, fontSize: 13 });
s.addNotes("Fairness starts with data. Mention the three identical files under two classes, the near-duplicate grouping, and that the real-world set is never used for training.");

// ---------------------------------------------------------------- 4 three methods
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Methods" });
s.addText("Three methods, three philosophies", { placeholder: "title" });
const meth = [["A", "Classical", "HOG + linear SVM", ["HOG and SVM written by us in NumPy", "8,100-D edge-orientation descriptor", "squared-hinge SVM, our own Adam optimiser", "0 learned features"]],
  ["B", "Trained from scratch", "Our 4-block CNN", ["Conv-BN-ReLU-Pool x4 (double conv in 3-4)", `${f3(M.B.params_M)} M parameters, random init`, "augmentation, dropout, weight decay, early stop", "128 x 128 input"]],
  ["C", "Pretrained, modern", "MobileNetV3-Large", ["ImageNet weights (torchvision), 2019", "new 6-class head; head first, then fine-tune", `${f1(M.C.params_M)} M parameters, 224 x 224`, "same split, loss, augmentation as B"]]];
meth.forEach(([k, slot, name, items], i) => {
  const x = 0.6 + i * 4.1;
  card(s, x, 1.45, 3.85, 5.25, HEX.lt2);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 1.7, w: 0.75, h: 0.75, fill: { color: COL[k] }, line: { color: COL[k] } });
  txt(s, k, { x: x + 0.25, y: 1.7, w: 0.75, h: 0.75, align: "center", valign: "middle", fontSize: 24, bold: true, color: C.background1 });
  txt(s, slot.toUpperCase(), { x: x + 1.15, y: 1.72, w: 2.5, h: 0.3, fontSize: 11, bold: true, color: "6B7780" });
  txt(s, name, { x: x + 1.15, y: 2.02, w: 2.6, h: 0.45, fontSize: 18, bold: true, color: C.text2, fontFace: THEME.headFontFace });
  bullets(s, items, { x: x + 0.25, y: 2.8, w: 3.4, h: 3.6, fontSize: 14 });
});
s.addNotes("Contrast: hand-designed vs learned, small vs large, no prior knowledge vs ImageNet knowledge. All three output the same thing: one of six labels.");

// ---------------------------------------------------------------- 5 method A sanity
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Methods" });
s.addText("Method A is ours, and verified", { placeholder: "title" });
img(s, "A_sanity_checks.png", 0.6, 1.4, 12.1, 2.85, "Sanity checks of HOG and SVM");
const sc = rA.sanity_checks;
const scs = [["Synthetic edges", "peak bins 0, 90 and 45 degrees: all pass"],
  ["vs scikit-image HOG", `max difference ${sc.skimage_reference.max_abs_diff.toExponential(1)}, correlation 1.000`],
  ["Toy SVM", `${sc.toy_svm.train_accuracy.toFixed(0)}% on 3 Gaussian blobs`],
  ["Tuned on validation", `cell 8 px, soft binning, lambda = ${rA.selected.lam}`]];
scs.forEach(([h, d], i) => {
  const x = 0.6 + i * 3.1;
  card(s, x, 4.55, 2.85, 2.2, HEX.lt2);
  txt(s, h, { x: x + 0.2, y: 4.7, w: 2.45, h: 0.4, bold: true, fontSize: 15, color: C.accent1 });
  txt(s, d, { x: x + 0.2, y: 5.15, w: 2.45, h: 1.45, fontSize: 13 });
});
s.addNotes("The rubric asks for the core algorithm written by us plus a sanity check. Explain gradient -> cell histograms -> L2-Hys blocks, then the three checks.");

// ---------------------------------------------------------------- 6 training curves
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Methods" });
s.addText("Training B and C: curves and over/under-fitting", { placeholder: "title" });
img(s, "exp1_training_curves.png", 0.6, 1.35, 12.1, 3.15, "Training and validation curves");
bullets(s, [`B attempt 1 under-fitted: only 47% train / 53% validation accuracy after 15 epochs, so we reduced augmentation`,
  `B final: best epoch ${rB.train_info.best_epoch} of ${rB.train_info.epochs_run}; C: head-only warm-up, best epoch ${rC.train_info.best_epoch}`,
  "Early stopping on validation macro-F1 restores the best checkpoint; same augmentation strength for B and C"],
  { x: 0.6, y: 4.75, w: 12.1, h: 2.1, fontSize: 15 });
s.addNotes("Point to the gap between train and validation curves. Explain why our first CNN run was under-fitting and what we changed.");

// ---------------------------------------------------------------- 7 results
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Experiment 1: pretrained features win clearly", { placeholder: "title" });
[["A", "HOG + SVM"], ["B", "Scratch CNN"], ["C", "MobileNetV3"]].forEach(([k, n], i) => {
  const x = 0.6 + i * 2.75;
  card(s, x, 1.4, 2.55, 2.05, HEX.lt2);
  txt(s, `${f1(M[k].accuracy)}%`, { x: x + 0.15, y: 1.5, w: 2.25, h: 0.95, fontSize: 40, bold: true, color: COL[k], align: "center", fontFace: THEME.headFontFace });
  txt(s, `${n}\nmacro-F1 ${f3(M[k].macro_f1)}`, { x: x + 0.15, y: 2.5, w: 2.25, h: 0.8, fontSize: 13, align: "center", color: "4A555E" });
});
const rows = [["Method", "Acc. %", "F1", "ms/img", "Params M", "Train min"].map((h) => ({ text: h, options: { bold: true, color: "FFFFFF", fill: { color: HEX.accent1 } } }))];
[["A", "HOG + SVM"], ["A+", "HOG + colour (variant)"], ["B", "Scratch CNN"], ["C", "MobileNetV3-L"]].forEach(([k, n]) => {
  const m = M[k];
  rows.push([n, f1(m.accuracy), f3(m.macro_f1), f1(m.runtime_ms_cpu), m.params_M.toFixed(2), f1(m.train_time_min)]);
});
s.addTable(rows, { x: 0.6, y: 3.7, w: 8.05, colW: [2.45, 1.0, 1.0, 1.15, 1.2, 1.25], fontSize: 13, color: HEX.dk1,
  border: { type: "solid", pt: 0.5, color: "D5DED9" }, rowH: 0.42 });
card(s, 8.9, 1.4, 3.85, 5.3, HEX.lt2);
txt(s, "Why C wins", { x: 9.15, y: 1.6, w: 3.4, h: 0.4, bold: true, fontSize: 16, color: C.accent1 });
bullets(s, ["Material = colour, shine, transparency, texture",
  "HOG keeps only edge directions: adding colour lifts A to " + f1(M["A+"].accuracy) + "%",
  "1,764 images are too few to learn good filters from scratch",
  "ImageNet features already separate glossy, transparent, printed"],
  { x: 9.15, y: 2.15, w: 3.4, h: 4.4, fontSize: 14 });
txt(s, `95% CI for C: ${f1(M.C.acc_ci95[0])}-${f1(M.C.acc_ci95[1])}%. McNemar C vs B p = ${e1.mcnemar.C_vs_B.p_value.toExponential(1)}. Runtime: CPU, batch 1.`,
  { x: 0.6, y: 6.0, w: 8.05, h: 0.6, fontSize: 12, color: "4A555E" });
s.addNotes("Give the three headline numbers, then the cost columns. Mention the confidence interval and that the differences are statistically significant.");

// ---------------------------------------------------------------- 8 domain shift (native chart)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Experiment 2: studio to real-world photos", { placeholder: "title" });
const ks = ["A", "A+", "B", "C"], labs = ["HOG + SVM", "HOG + colour", "Scratch CNN", "MobileNetV3"];
s.addChart(pres.charts.BAR, [
  { name: "TrashNet test", labels: labs, values: ks.map((k) => +M[k].accuracy.toFixed(1)) },
  { name: "Real-world photos", labels: labs, values: ks.map((k) => +RW[k].accuracy.toFixed(1)) }],
  { x: 0.6, y: 1.35, w: 7.4, h: 5.4, barDir: "col", barGapWidthPct: 60, chartColors: ["8FB8DE", "D1495B"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
    valAxisMaxVal: 100, valAxisMinVal: 0, valAxisTitle: "top-1 accuracy (%)", showValAxisTitle: true,
    valAxisTitleFontSize: 12, valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt", valAxisLabelColor: "4A555E", catAxisLabelColor: "4A555E",
    valGridLine: { color: "E1E7E3", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", showTitle: false });
const drops = ["A", "B", "C"].map((k) => [k, RW[k]]);
drops.forEach(([k, r], i) => {
  const y = 1.45 + i * 1.75;
  card(s, 8.4, y, 4.3, 1.55, HEX.lt2);
  txt(s, `-${f1(r.acc_drop_pts)} pts`, { x: 8.6, y: y + 0.12, w: 2.0, h: 0.7, fontSize: 28, bold: true, color: COL[k], fontFace: THEME.headFontFace });
  txt(s, `${["HOG + SVM", "Scratch CNN", "MobileNetV3"][i]}\nkeeps ${(r.relative_retention * 100).toFixed(0)}% of its accuracy`,
    { x: 10.6, y: y + 0.18, w: 2.0, h: 1.2, fontSize: 13, color: "4A555E" });
});
s.addNotes("This answers the second half of the research question. Every method drops; describe which drops most and why (background and context cues learned from studio photos).");

// ---------------------------------------------------------------- 9 small data + efficiency (native line chart)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Experiments 3-4: small data and cost", { placeholder: "title" });
const sizes = ["10/class", "25/class", "50/class", "all"];
s.addChart(pres.charts.LINE, ["A", "B", "C"].map((k, i) => ({ name: ["HOG + SVM", "Scratch CNN", "MobileNetV3"][i], labels: sizes,
  values: [10, 25, 50].map((n) => +e3.runs[`${k}_${n}`].accuracy.toFixed(1)).concat([+e3.full[k].accuracy.toFixed(1)]) })),
  { x: 0.6, y: 1.35, w: 6.6, h: 5.4, chartColors: [COL.A, COL.B, COL.C], lineSize: 3, lineDataSymbolSize: 9,
    showValue: true, dataLabelPosition: "t", dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt",
    valAxisMinVal: 0, valAxisMaxVal: 100, valAxisTitle: "test accuracy (%)", showValAxisTitle: true, valAxisTitleFontSize: 12,
    valAxisTitleFontFace: "+mn-lt", catAxisTitle: "training images per class", showCatAxisTitle: true, catAxisTitleFontSize: 12,
    catAxisTitleFontFace: "+mn-lt", valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt",
    catAxisLabelFontFace: "+mn-lt", valAxisLabelColor: "4A555E", catAxisLabelColor: "4A555E",
    valGridLine: { color: "E1E7E3", size: 0.5 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b",
    legendFontSize: 12, legendFontFace: "+mn-lt" });
img(s, "exp4_efficiency.png", 7.45, 1.4, 5.3, 2.4, "Accuracy versus cost");
bullets(s, [`With 10 images per class: C ${f1(e3.runs.C_10.accuracy)}%, B ${f1(e3.runs.B_10.accuracy)}%, A ${f1(e3.runs.A_10.accuracy)}%`,
  `HOG + SVM is fastest (${f1(M.A.runtime_ms_cpu)} ms) but least accurate`,
  `MobileNetV3: ${f1(M.C.runtime_ms_cpu)} ms on a laptop CPU, phone-ready`],
  { x: 7.45, y: 4.05, w: 5.3, h: 2.7, fontSize: 14 });
s.addNotes("Pretrained features matter most when data is scarce. Then cost: the classical method is the cheapest, the pretrained one is the best accuracy per millisecond that still runs in real time.");

// ---------------------------------------------------------------- 10 error analysis
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Experiment 5: what each method looks at", { placeholder: "title" });
img(s, "exp5_gradcam.png", 0.6, 1.3, 6.9, 5.55, "Gradient maps and Grad-CAM per class");
const tc = (k) => e5.top_confusions[k].slice(0, 2).map((d) => `${d.true} -> ${d.pred} (${d.count})`).join(", ");
bullets(s, [`A confuses: ${tc("A")} - HOG sees edges only, no colour or shine`,
  `B confuses: ${tc("B")}`, `C confuses: ${tc("C")}`,
  `Grad-CAM: C focuses on the object; B often looks at background and shadows, which explains its larger real-world drop`,
  `Majority vote of A+B+C: ${f1(e5.majority_vote_accuracy)}% (no gain over C alone)`],
  { x: 7.8, y: 1.4, w: 4.95, h: 5.4, fontSize: 14 });
s.addNotes("Explain the confusion pairs with theory: HOG discards colour, so glass vs plastic and paper vs cardboard are hard; texture and colour learned by CNNs fix much of this.");

// ---------------------------------------------------------------- 11 experiment 6 (native chart)
const e6 = J("exp6_mixed_domain.json"), M6 = e6.methods;
const REC6 = ["A", "B", "C"].reduce((a, b) => (M6[b].realworld.macro_f1 > M6[a].realworld.macro_f1 ? b : a));
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Experiment 6: fixing the domain gap", { placeholder: "title" });
const labs3 = ["HOG + SVM", "Scratch CNN", "MobileNetV3"];
s.addChart(pres.charts.BAR, [
  { name: "Trained on TrashNet only", labels: labs3, values: ["A", "B", "C"].map((k) => +M6[k].baseline_trashnet_only.realworld_acc.toFixed(1)) },
  { name: "Trained on TrashNet + web photos", labels: labs3, values: ["A", "B", "C"].map((k) => +M6[k].realworld.accuracy.toFixed(1)) }],
  { x: 0.6, y: 1.35, w: 7.4, h: 5.4, barDir: "col", barGapWidthPct: 60, chartColors: ["B8C4BD", "1F8B4C"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt",
    valAxisMaxVal: 100, valAxisMinVal: 0, valAxisTitle: "real-world test accuracy (%)", showValAxisTitle: true,
    valAxisTitleFontSize: 12, valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontFace: "+mn-lt", valAxisTitleFontFace: "+mn-lt", valAxisLabelColor: "4A555E", catAxisLabelColor: "4A555E",
    valGridLine: { color: "E1E7E3", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", showTitle: false });
const r6 = M6[REC6];
[[`${f1(r6.realworld.accuracy)}%`, `${labs3["ABC".indexOf(REC6)]} on unseen real-world photos (was ${f1(r6.baseline_trashnet_only.realworld_acc)}%)`],
 [`${f1(r6.trashnet_test.accuracy)}%`, "same model on the TrashNet test photos"],
 [`${f1(r6.realworld_at_threshold.accuracy)}%`, `correct when BIN answers (${Math.round(r6.realworld_at_threshold.coverage * 100)}% answered, else "not sure")`]]
  .forEach(([n, l], i) => {
    const y = 1.45 + i * 1.75;
    card(s, 8.4, y, 4.3, 1.55, HEX.lt2);
    txt(s, n, { x: 8.6, y: y + 0.12, w: 3.9, h: 0.7, fontSize: 30, bold: true, color: C.accent1, fontFace: THEME.headFontFace });
    txt(s, l, { x: 8.6, y: y + 0.85, w: 3.9, h: 0.6, fontSize: 13, color: "4A555E" });
  });
s.addNotes("Same architectures and settings, same test sets: only the training data changed (TrashNet + web photos, with every near-copy of a test photo removed). This is what made BIN accurate on real photos. Below the calibrated confidence threshold BIN says 'not sure' instead of guessing.");

// ---------------------------------------------------------------- 12 BIN app
pres.addSection({ title: "System" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "System" });
s.addText("BIN: the trash identification app", { placeholder: "title" });
const screens = path.join(ROOT, "report", "bin_phone_screens.png");
if (fs.existsSync(screens)) fitImage(s, screens, 0.6, 1.3, 12.1, 4.15, "BIN phone app screens");
bullets(s, ["Opens straight to the camera: frame the item, tap Identify (or pick from the gallery)",
  `All three methods run on the phone; BIN answers with ${labs3["ABC".indexOf(REC6)]} (best on real photos) or says "not sure"`,
  "Shows recyclable / residual and what to do; photos never leave the phone; works offline",
  "karlsxo.github.io/Computer-Vison---Final-Project  ·  desktop version: python app/app.py"],
  { x: 0.6, y: 5.6, w: 12.1, h: 1.4, fontSize: 14, paraSpaceAfter: 4 });
s.addNotes("Live demo on a phone: open the link, point the camera at a bottle and a paper, show the result, the three methods and the not-sure behaviour.");

// ---------------------------------------------------------------- 13 conclusion
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
s.addText("Answer: pretrained MobileNetV3, trained on real photos", { placeholder: "title" });
s.addText([
  { text: `Best on clean photos: ${f1(M.C.accuracy)}% (vs ${f1(M.B.accuracy)}% scratch CNN, ${f1(M.A.accuracy)}% HOG + SVM)`, options: { bullet: true, breakLine: true } },
  { text: `Trained on studio photos only, every method drops on real photos (MobileNetV3: ${f1(RW.C.accuracy)}%)`, options: { bullet: true, breakLine: true } },
  { text: `Adding real-world photos to training fixes most of it: ${f1(M6.C.realworld.accuracy)}% (MobileNetV3), now in the BIN app`, options: { bullet: true } }],
  { placeholder: "body" });
s.addNotes("Close with the answer to the research question: pretraining wins, and the training data must look like the deployment photos. Then invite questions.");

(async () => {
  const out = path.join(ROOT, "report", "Trash_Sorting_Slides.pptx");
  await pres.writeFile({ fileName: out });
  const { applyTheme } = require("./apply_theme.js");
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
