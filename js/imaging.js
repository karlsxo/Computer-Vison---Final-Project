// Image decoding and resizing for the phone app.
//
// The models were trained on images resized with Pillow's BILINEAR filter, which
// (unlike a plain canvas resize) widens its triangle kernel when shrinking, i.e. it
// anti-aliases. resizeLikePIL() re-implements Pillow's algorithm (Resample.c:
// separable horizontal-then-vertical pass, 22-bit fixed-point coefficients, 8-bit
// intermediate image) so the phone feeds the networks the same pixels as Python.

const PRECISION_BITS = 22;            // 32 - 8 - 2, as in Pillow
const ONE = 1 << PRECISION_BITS;

/** Decode a File/Blob into RGBA pixels, honouring the photo's EXIF orientation.
 *  Very large phone photos are first reduced to at most maxSide pixels. */
export async function decodeImage(blob, maxSide = 2048) {
  const bmp = await createImageBitmap(blob, { imageOrientation: "from-image" });
  return bitmapToImageData(bmp, maxSide);
}

export function bitmapToImageData(source, maxSide = 2048) {
  const sw = source.videoWidth || source.width;
  const sh = source.videoHeight || source.height;
  const s = Math.min(1, maxSide / Math.max(sw, sh));
  const w = Math.max(1, Math.round(sw * s));
  const h = Math.max(1, Math.round(sh * s));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 1.0 * filterscale;                 // bilinear support = 1
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const ss = 1 / filterscale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const k = new Float64Array(xmax);
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const t = Math.abs((x + xmin - center + 0.5) * ss);
      const w = t < 1 ? 1 - t : 0;
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      kk[xx * ksize + x] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

const clip8 = (v) => (v <= 0 ? 0 : v >= 255 ? 255 : v);

/** Resize RGBA ImageData to outW x outH, returns RGB Uint8Array (HWC). */
export function resizeLikePIL(img, outW, outH) {
  const { width: w, height: h, data } = img;
  // RGBA -> RGB
  let src = new Uint8Array(w * h * 3);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    src[i * 3] = data[j]; src[i * 3 + 1] = data[j + 1]; src[i * 3 + 2] = data[j + 2];
  }
  let curW = w;
  // horizontal pass
  if (outW !== w) {
    const { ksize, bounds, kk } = coefficients(w, outW);
    const out = new Uint8Array(outW * h * 3);
    for (let y = 0; y < h; y++) {
      const row = y * w * 3;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1], kb = xx * ksize;
        let r = 1 << (PRECISION_BITS - 1), g = r, b = r;
        for (let x = 0; x < xmax; x++) {
          const k = kk[kb + x], p = row + (x + xmin) * 3;
          r += src[p] * k; g += src[p + 1] * k; b += src[p + 2] * k;
        }
        const o = (y * outW + xx) * 3;
        out[o] = clip8(r >> PRECISION_BITS); out[o + 1] = clip8(g >> PRECISION_BITS); out[o + 2] = clip8(b >> PRECISION_BITS);
      }
    }
    src = out;
    curW = outW;
  }
  // vertical pass
  if (outH !== h) {
    const { ksize, bounds, kk } = coefficients(h, outH);
    const out = new Uint8Array(curW * outH * 3);
    for (let yy = 0; yy < outH; yy++) {
      const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1], kb = yy * ksize;
      for (let x = 0; x < curW; x++) {
        let r = 1 << (PRECISION_BITS - 1), g = r, b = r;
        for (let y = 0; y < ymax; y++) {
          const k = kk[kb + y], p = ((y + ymin) * curW + x) * 3;
          r += src[p] * k; g += src[p + 1] * k; b += src[p + 2] * k;
        }
        const o = (yy * curW + x) * 3;
        out[o] = clip8(r >> PRECISION_BITS); out[o + 1] = clip8(g >> PRECISION_BITS); out[o + 2] = clip8(b >> PRECISION_BITS);
      }
    }
    src = out;
  }
  return src;
}

/** RGB HWC uint8 -> Float32Array NCHW with raw 0-255 values (the ONNX models
 *  normalise internally). */
export function toNCHW(rgb, size) {
  const n = size * size;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = rgb[i * 3]; out[n + i] = rgb[i * 3 + 1]; out[2 * n + i] = rgb[i * 3 + 2];
  }
  return out;
}

/** Small JPEG thumbnail (data URL) for the history list. */
export function thumbnail(img, side = 160) {
  const c = document.createElement("canvas");
  const s = side / Math.max(img.width, img.height);
  c.width = Math.round(img.width * s);
  c.height = Math.round(img.height * s);
  const tmp = document.createElement("canvas");
  tmp.width = img.width; tmp.height = img.height;
  tmp.getContext("2d").putImageData(img, 0, 0);
  const ctx = c.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(tmp, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.8);
}
