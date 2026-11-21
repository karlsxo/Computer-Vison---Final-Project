// BIN: loads the three methods and classifies one photo entirely on the phone.
// A = HOG + linear SVM (pure JavaScript), B = our CNN, C = MobileNetV3-Large
// (both ONNX, run with onnxruntime-web / WebAssembly).
import { resizeLikePIL, toNCHW } from "./imaging.js";
import { hogDescriptor, parseSvm, svmScores, softmax } from "./hog.js";

const ort = window.ort;
ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
ort.env.wasm.numThreads = 1;          // GitHub Pages cannot enable multi-threading

let meta = null, svm = null, sessB = null, sessC = null;

async function fetchWithProgress(url, onBytes) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${url} (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) {
    const buf = await res.arrayBuffer();
    onBytes(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onBytes(got, total);
  }
  const out = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out.buffer;
}

/** onProgress(fraction 0..1, text) */
export async function loadModels(onProgress = () => {}) {
  const base = new URL("../models/", import.meta.url);
  meta = await (await fetch(new URL("meta.json", base))).json();
  const files = [["hog_svm.bin", 0.26], ["scratch_cnn.onnx", 4.5], ["mobilenetv3_large.onnx", 16.8]];
  const totalMB = files.reduce((a, f) => a + f[1], 0);
  let doneMB = 0;
  const bufs = {};
  for (const [name, mb] of files) {
    bufs[name] = await fetchWithProgress(new URL(name, base), (got, total) => {
      onProgress((doneMB + (mb * got) / (total || got)) / totalMB, `Downloading AI models (${Math.round(totalMB)} MB, only once)`);
    });
    doneMB += mb;
  }
  onProgress(1, "Starting the AI engine");
  svm = parseSvm(bufs["hog_svm.bin"], meta.A.dim, meta.A.classes);
  const opts = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  sessB = await ort.InferenceSession.create(bufs["scratch_cnn.onnx"], opts);
  sessC = await ort.InferenceSession.create(bufs["mobilenetv3_large.onnx"], opts);
  return meta;
}

export const getMeta = () => meta;

async function runOnnx(sess, rgb, size) {
  const t = new ort.Tensor("float32", toNCHW(rgb, size), [1, 3, size, size]);
  const out = await sess.run({ image: t });
  return Array.from(out.probs.data);
}

function pack(probs, ms) {
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { probs, label: meta.classes[best], index: best, confidence: probs[best], ms };
}

/** imageData: RGBA ImageData of the photo. Returns {A, B, C, recommended}. */
export async function classify(imageData) {
  const rgb128 = resizeLikePIL(imageData, 128, 128);
  const rgb224 = resizeLikePIL(imageData, 224, 224);
  const out = {};
  let t = performance.now();
  out.A = pack(softmax(svmScores(svm, hogDescriptor(rgb128, 128))), performance.now() - t);
  t = performance.now();
  out.B = pack(await runOnnx(sessB, rgb128, 128), performance.now() - t);
  t = performance.now();
  out.C = pack(await runOnnx(sessC, rgb224, 224), performance.now() - t);
  out.recommended = recommendedMethod();
  return out;
}

/** The method whose answer BIN shows: chosen at export time as the one with the best
 *  macro-F1 on real-world photos (tools/export_mobile.py). */
export function recommendedMethod() {
  return meta.recommended;
}

// exposed for automated verification (tools/verify_web.py)
window.binApp = { loadModels, classify, getMeta };
