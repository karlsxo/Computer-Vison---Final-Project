// Service worker: caches the app and the AI models so Trash Sorter opens instantly
// and works offline after the first visit. Bump VERSION when files change.
const VERSION = "trash-sorter-v1";
const SHELL = [
  "./", "index.html", "manifest.webmanifest", "css/app.css",
  "js/app.js", "js/models.js", "js/imaging.js", "js/hog.js", "js/content.js",
  "vendor/ort/ort.wasm.min.js", "vendor/ort/ort-wasm-simd-threaded.mjs", "vendor/ort/ort-wasm-simd-threaded.wasm",
  "models/meta.json", "models/hog_svm.bin", "models/scratch_cnn.onnx", "models/mobilenetv3_large.onnx",
  "icons/icon-192.png", "icons/icon-512.png",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

// cache first (fast, offline), fall back to the network and remember the answer
self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request).then((res) => {
    if (res.ok && (res.type === "basic" || res.type === "cors")) {
      const copy = res.clone();
      caches.open(VERSION).then((c) => c.put(e.request, copy));
    }
    return res;
  })));
});
