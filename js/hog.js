// Method A on the phone: our HOG descriptor + linear SVM, ported line by line from
// src/hog_svm.py (same gradients, soft orientation binning, 8x8 cells, 2x2 blocks
// with L2-Hys, same standardisation and SVM weights exported by
// tools/export_mobile.py).

const CELL = 8, BLOCK = 2, NBINS = 9, EPS = 1e-5;

/** rgb: Uint8Array HWC of size x size. Returns Float32Array descriptor. */
export function hogDescriptor(rgb, size) {
  const n = size * size;
  const gray = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    gray[i] = (rgb[i * 3] / 255) * 0.299 + (rgb[i * 3 + 1] / 255) * 0.587 + (rgb[i * 3 + 2] / 255) * 0.114;
  }
  const ncy = Math.floor(size / CELL), ncx = Math.floor(size / CELL);
  const hist = new Float64Array(ncy * ncx * NBINS);
  const binWidth = 180 / NBINS;
  for (let y = 0; y < ncy * CELL; y++) {
    for (let x = 0; x < ncx * CELL; x++) {
      const gx = x > 0 && x < size - 1 ? gray[y * size + x + 1] - gray[y * size + x - 1] : 0;
      const gy = y > 0 && y < size - 1 ? gray[(y + 1) * size + x] - gray[(y - 1) * size + x] : 0;
      const mag = Math.hypot(gx, gy);
      let ang = (Math.atan2(gy, gx) * 180) / Math.PI;
      ang = ((ang % 180) + 180) % 180;
      const pos = ang / binWidth - 0.5;
      const lo = Math.floor(pos);
      const wHi = pos - lo;
      const bLo = ((lo % NBINS) + NBINS) % NBINS;
      const bHi = (bLo + 1) % NBINS;
      const c = (Math.floor(y / CELL) * ncx + Math.floor(x / CELL)) * NBINS;
      hist[c + bLo] += mag * (1 - wHi);
      hist[c + bHi] += mag * wHi;
    }
  }
  for (let i = 0; i < hist.length; i++) hist[i] /= CELL * CELL;
  const nby = ncy - BLOCK + 1, nbx = ncx - BLOCK + 1, bl = BLOCK * BLOCK * NBINS;
  const out = new Float32Array(nby * nbx * bl);
  const v = new Float64Array(bl);
  for (let by = 0; by < nby; by++) {
    for (let bx = 0; bx < nbx; bx++) {
      let k = 0;
      for (let dy = 0; dy < BLOCK; dy++)
        for (let dx = 0; dx < BLOCK; dx++)
          for (let b = 0; b < NBINS; b++) v[k++] = hist[((by + dy) * ncx + (bx + dx)) * NBINS + b];
      // L2-Hys: L2-normalise, clip at 0.2, L2-normalise again
      let s = 0;
      for (let i = 0; i < bl; i++) s += v[i] * v[i];
      let d = Math.sqrt(s + EPS * EPS);
      s = 0;
      for (let i = 0; i < bl; i++) { v[i] = Math.min(v[i] / d, 0.2); s += v[i] * v[i]; }
      d = Math.sqrt(s + EPS * EPS);
      const o = (by * nbx + bx) * bl;
      for (let i = 0; i < bl; i++) out[o + i] = v[i] / d;
    }
  }
  return out;
}

/** Parse the exported SVM: W (dim x K), b (K), mean (dim), std (dim). */
export function parseSvm(buffer, dim, K) {
  const f = new Float32Array(buffer);
  let o = 0;
  const W = f.subarray(o, (o += dim * K));
  const b = f.subarray(o, (o += K));
  const mean = f.subarray(o, (o += dim));
  const std = f.subarray(o, (o += dim));
  return { W, b, mean, std, dim, K };
}

export function svmScores(svm, feat) {
  const { W, b, mean, std, dim, K } = svm;
  const scores = Float64Array.from(b);
  for (let i = 0; i < dim; i++) {
    const z = (feat[i] - mean[i]) / std[i];
    const row = i * K;
    for (let k = 0; k < K; k++) scores[k] += z * W[row + k];
  }
  return scores;
}

export function softmax(z) {
  const m = Math.max(...z);
  const e = Array.from(z, (v) => Math.exp(v - m));
  const s = e.reduce((a, c) => a + c, 0);
  return e.map((v) => v / s);
}
