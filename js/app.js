// BIN - identifies the material of a piece of trash from a photo.
// Opens straight to the camera. Everything runs on the phone: photos never leave the device.
import { loadModels, classify, getMeta } from "./models.js";
import { decodeImage, bitmapToImageData, thumbnail } from "./imaging.js";
import { MATERIALS, BIN_LABEL, ARTICLES } from "./content.js";

const REPO = "https://github.com/karlsxo/Computer-Vison---Final-Project";
const METHOD = {
  A: { name: "HOG + linear SVM", slot: "Classical", color: "#d1495b" },
  B: { name: "4-block CNN", slot: "Trained from scratch", color: "#00798c" },
  C: { name: "MobileNetV3-Large", slot: "Pretrained", color: "#30638e" },
};

// ------------------------------------------------------------------ icons
const P = {
  back: '<path d="M15 5l-7 7 7 7"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-9 9"/>',
  flip: '<path d="M4 8h12l-3-3M20 16H8l3 3"/>',
  camera: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
  history: '<path d="M4 12a8 8 0 1 0 2.4-5.7L4 8.5"/><path d="M4 4v4.5h4.5"/><path d="M12 8v4l3 2"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
  del: '<path d="M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
  link: '<path d="M10 14l4-4"/><path d="M8.5 11.5l-2 2a3.2 3.2 0 0 0 4.5 4.5l2-2"/><path d="M15.5 12.5l2-2a3.2 3.2 0 0 0-4.5-4.5l-2 2"/>',
};
const icon = (n) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${P[n]}</svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const pct = (x) => `${Math.round(x * 100)}%`;
const LOGO = '<img src="icons/logo.svg" alt="" class="logo">';

// ------------------------------------------------------------------ scan history (saved on the phone only)
const KEY = "bin.history.v1";
function loadHistory() {
  try {
    const h = JSON.parse(localStorage.getItem(KEY));
    if (Array.isArray(h)) return h;
  } catch (e) { /* storage unavailable or corrupt: start empty */ }
  return [];
}
let history = loadHistory();
function saveHistory() {
  try { localStorage.setItem(KEY, JSON.stringify(history.slice(-100))); }
  catch (e) { /* storage full or blocked: history just is not kept */ }
}

// ------------------------------------------------------------------ models
const models = { ready: false, progress: 0, text: "Preparing the AI", error: null, promise: null };
function startModels() {
  models.promise = loadModels((f, t) => {
    models.progress = f; models.text = t;
    updateStatus();
  }).then(() => { models.ready = true; updateStatus(); })
    .catch((e) => { models.error = e.message || String(e); updateStatus(); });
}
function updateStatus() {
  const el = document.querySelector("#status");
  if (!el) return;
  if (models.error) {
    el.className = "status error";
    el.textContent = "AI could not load - check your connection and reopen BIN";
  } else if (!models.ready) {
    el.className = "status";
    el.textContent = `${models.text} · ${Math.round(models.progress * 100)}%`;
  } else {
    el.className = "status ready";
    el.textContent = "Point the camera at one piece of trash";
  }
}

// ------------------------------------------------------------------ routing (scan is the start screen)
const view = () => document.querySelector("#view");
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  return { name: parts[0] || "scan", arg: parts[1], arg2: parts[2] };
}
const go = (h) => { location.hash = h; };
window.addEventListener("hashchange", render);

let current = null;   // the latest result shown on the result screen
function render() {
  const r = route();
  if (r.name !== "scan") stopCamera();
  const screens = { scan, result, history: historyPage, about };
  (screens[r.name] || scan)(r);
  window.scrollTo(0, 0);
}

function header(title, backTo = "#/scan") {
  return `<header class="header"><div class="header-row">
    <button class="icon-btn" onclick="location.hash='${backTo}'" aria-label="Back">${icon("back")}</button>
    <h1>${title}</h1>${LOGO}</div></header>`;
}

// ------------------------------------------------------------------ scanner (start screen)
let stream = null, facing = "environment";
function stopCamera() {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
}
function scan() {
  view().innerHTML = `
  <div class="scan">
    <video id="video" playsinline muted autoplay></video>
    <div class="no-camera hidden" id="nocam"><div><div class="big">📷</div>
      <p>Tap the green button to take a photo, or choose one from your gallery.</p>
      <p class="small" id="nocam-why"></p></div></div>
    <div class="top">
      <div class="brand">${LOGO}<b>BIN</b></div>
      <div class="top-actions">
        <button class="icon-btn" onclick="location.hash='#/history'" aria-label="Scan history">${icon("history")}</button>
        <button class="icon-btn" onclick="location.hash='#/about'" aria-label="About BIN">${icon("info")}</button>
      </div>
    </div>
    <div class="status" id="status"></div>
    <div class="frame" id="frame"><i></i><i></i><i></i><i></i></div>
    <div class="sheet">
      <div class="controls">
        <div><button class="round" id="gallery" aria-label="Choose from gallery">${icon("image")}</button><div class="ctrl-label">Gallery</div></div>
        <div><button class="shutter" id="shutter" aria-label="Identify"></button><div class="ctrl-label">Identify</div></div>
        <div><button class="round" id="flip" aria-label="Switch camera">${icon("flip")}</button><div class="ctrl-label">Flip</div></div>
      </div>
      <p class="muted small" style="text-align:center;margin:10px 0 0">One item inside the frame · plain background · good light</p>
    </div>
    <input type="file" accept="image/*" id="galleryInput" hidden>
    <input type="file" accept="image/*" capture="environment" id="cameraInput" hidden>
  </div>`;
  updateStatus();
  const video = document.querySelector("#video");
  document.querySelector("#gallery").onclick = () => document.querySelector("#galleryInput").click();
  document.querySelector("#galleryInput").onchange = (e) => e.target.files[0] && analyseFile(e.target.files[0]);
  document.querySelector("#cameraInput").onchange = (e) => e.target.files[0] && analyseFile(e.target.files[0]);
  document.querySelector("#flip").onclick = () => { facing = facing === "environment" ? "user" : "environment"; startCamera(video); };
  document.querySelector("#shutter").onclick = () => {
    if (stream && video.videoWidth) analyseImageData(cropToFrame(video));
    else document.querySelector("#cameraInput").click();   // the phone's camera app
  };
  startCamera(video);
}

async function startCamera(video) {
  stopCamera();
  const nocam = document.querySelector("#nocam");
  const why = document.querySelector("#nocam-why");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    nocam.classList.remove("hidden");
    why.textContent = window.isSecureContext ? "" : "(The live camera needs the https:// address of BIN.)";
    return;
  }
  try {
    const s = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    if (route().name !== "scan") { s.getTracks().forEach((t) => t.stop()); return; }
    stream = s;
    video.srcObject = stream;
    await video.play();
    nocam.classList.add("hidden");
  } catch (e) {
    nocam.classList.remove("hidden");
    why.textContent = "Camera permission was not given. Allow it in your browser settings, or tap the green button to use your camera app.";
  }
}

/** Crop the video frame to the square scanner frame shown on screen, so the
 *  item fills the picture like in the training photos. */
function cropToFrame(video) {
  const vr = video.getBoundingClientRect(), fr = document.querySelector("#frame").getBoundingClientRect();
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(vr.width / vw, vr.height / vh);            // object-fit: cover
  const ox = (vr.width - vw * s) / 2, oy = (vr.height - vh * s) / 2;
  let x = (fr.left - vr.left - ox) / s, y = (fr.top - vr.top - oy) / s;
  const side = Math.min(fr.width / s, vw, vh);
  x = Math.max(0, Math.min(vw - side, x)); y = Math.max(0, Math.min(vh - side, y));
  const c = document.createElement("canvas");
  c.width = c.height = Math.round(Math.min(side, 1024));
  c.getContext("2d").drawImage(video, x, y, side, side, 0, 0, c.width, c.height);
  return bitmapToImageData(c);
}

function busy(text) {
  let el = document.querySelector(".busy");
  if (!el) {
    el = document.createElement("div");
    el.className = "busy";
    document.body.appendChild(el);
  }
  el.innerHTML = `<div><div class="spinner"></div><div id="busy-text">${esc(text)}</div></div>`;
  return () => el.remove();
}

async function analyseFile(file) {
  const done = busy("Reading photo");
  try {
    const img = await decodeImage(file);
    done();
    await analyseImageData(img);
  } catch (e) {
    done();
    toast("Could not open that image. Try a JPG or PNG photo.");
  }
}

async function analyseImageData(img) {
  const done = busy(models.ready ? "Identifying" : "Preparing the AI");
  try {
    if (!models.ready) {
      const iv = setInterval(() => {
        const t = document.querySelector("#busy-text");
        if (t) t.textContent = `Preparing the AI · ${Math.round(models.progress * 100)}%`;
      }, 300);
      await models.promise;
      clearInterval(iv);
      if (models.error) throw new Error(models.error);
    }
    const res = await classify(img);
    const meta = getMeta(), rec = res.recommended, top = res[rec];
    const sure = top.confidence >= (meta.thresholds?.[rec] ?? 0.5);
    const entry = { id: String(Date.now()), ts: Date.now(), label: top.label, predicted: top.label,
                    confidence: top.confidence, sure, method: rec, thumb: thumbnail(img, 160) };
    history.push(entry);
    saveHistory();
    current = { res, photo: thumbnail(img, 720), entry };
    done();
    go("#/result");
  } catch (e) {
    done();
    toast(`Something went wrong: ${e.message || e}`);
  }
}

// ------------------------------------------------------------------ result
function reasonText(rec) {
  const r = getMeta().results, others = Object.keys(r).filter((k) => k !== rec);
  return `<b>${METHOD[rec].name}</b> gives BIN's answer because it is the most accurate on real-world photos in our tests:
    ${r[rec].realworld_accuracy}% (vs ${others.map((k) => `${r[k].realworld_accuracy}% for ${METHOD[k].name}`).join(", ")}),
    and ${r[rec].trashnet_accuracy}% on clean test photos.`;
}

function result() {
  if (!current) { go("#/scan"); return; }
  const { res, entry } = current, rec = res.recommended, top = res[rec];
  const chosen = entry.label, m = MATERIALS[chosen], corrected = chosen !== entry.predicted;
  const agree = new Set(["A", "B", "C"].map((k) => res[k].label)).size === 1;
  const verdict = entry.sure || corrected
    ? `<div class="verdict"><div class="emoji">${m.emoji}</div><div>
         <h2>${chosen}</h2><span class="chip ${m.bin}">${BIN_LABEL[m.bin]}</span>
         <span class="muted small"> ${corrected ? "chosen by you" : `${pct(top.confidence)} sure`}</span></div></div>`
    : `<div class="verdict unsure"><div class="emoji">🤔</div><div>
         <h2>Not sure</h2><span class="muted small">Best guess: <b>${top.label}</b> (${pct(top.confidence)})</span></div></div>
       <p class="small" style="margin:10px 0 0">BIN is not confident enough to answer. Retake the photo: one item filling the
       frame, a plain background and good light. Or pick the material below.</p>`;
  view().innerHTML = `
  ${header("Result")}
  <main class="content with-cta">
    <div class="photo"><img src="${current.photo}" alt="Your photo">
      <div class="band">${entry.sure || corrected ? chosen.toUpperCase() : "NOT SURE"}</div></div>
    <div class="card" style="margin-top:12px">${verdict}
      ${entry.sure || corrected ? `<p class="small" style="margin:12px 0 0"><b>What to do:</b> ${m.tip}</p>
        <ul class="small guide">${m.guide.map((g) => `<li>${g}</li>`).join("")}</ul>` : ""}
    </div>
    <div class="section-title">Not right? Pick the material</div>
    <div class="card"><div class="picks">${getMeta().classes.map((c) =>
      `<button class="chip pick ${corrected && c === chosen ? "on" : ""}" data-pick="${c}">${MATERIALS[c].emoji} ${cap(c)}</button>`).join("")}</div></div>
    <div class="section-title">All three methods</div>
    <div class="card">${["A", "B", "C"].map((k) => methodRow(k, res[k], k === rec)).join("")}
      <p class="muted small" style="margin:10px 0 0">${agree ? "All three methods agree." : "The methods disagree; BIN uses the starred method."}
      ${reasonText(rec)} Times measured on this phone.</p></div>
  </main>
  <div class="cta"><button class="btn" onclick="location.hash='#/scan'">${icon("camera")}Scan next item</button></div>`;
  document.querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", () => {
    entry.label = b.dataset.pick;
    const h = history.find((x) => x.id === entry.id);
    if (h) { h.label = entry.label; saveHistory(); }
    result();
  }));
}

function methodRow(k, r, isRec) {
  const M = METHOD[k];
  return `<div class="method"><div class="head"><div><div class="name">${isRec ? '<span class="star">★</span> ' : ""}${M.name}</div>
      <div class="slot">${M.slot}${isRec ? " · used for the answer" : ""}</div></div>
      <div class="pred">${MATERIALS[r.label].emoji} ${r.label} · ${pct(r.confidence)}</div></div>
    <div class="bar"><div style="width:${Math.max(3, r.confidence * 100)}%;background:${M.color}"></div></div>
    <div class="muted small" style="margin-top:4px">${r.ms.toFixed(0)} ms${k === "A" ? " · score, not a calibrated probability" : ""}</div></div>`;
}

// ------------------------------------------------------------------ history
function historyPage() {
  const items = history.slice().reverse();
  view().innerHTML = `
  ${header("Scan history")}
  <main class="content with-cta">
    <div class="card">${items.length ? items.map((h) => {
      const m = MATERIALS[h.label], d = new Date(h.ts);
      return `<div class="item-row"><img class="thumb" src="${h.thumb}" alt="">
        <div class="meta"><b>${m.emoji} ${h.sure || h.label !== h.predicted ? h.label : `${h.label}?`}</b>
          <span class="chip ${m.bin}">${BIN_LABEL[m.bin]}</span>
          <div class="muted small">${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          · ${h.label !== h.predicted ? "corrected by you" : h.sure ? `${pct(h.confidence)} sure` : "not sure"}</div></div>
        <button class="del" data-del="${h.id}" aria-label="Delete">${icon("del")}</button></div>`;
    }).join("") : `<div class="empty"><div class="big">📷</div>No scans yet.</div>`}</div>
    ${items.length ? `<button class="btn ghost" id="clear" style="margin-top:14px">Clear history</button>` : ""}
  </main>
  <div class="cta"><button class="btn" onclick="location.hash='#/scan'">${icon("camera")}Scan an item</button></div>`;
  document.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", () => {
    history = history.filter((h) => h.id !== b.dataset.del); saveHistory(); historyPage();
  }));
  const clear = document.querySelector("#clear");
  if (clear) clear.onclick = () => { if (confirm("Delete all scans from this phone?")) { history = []; saveHistory(); historyPage(); } };
}

// ------------------------------------------------------------------ about / help
function aiArticle() {
  const meta = getMeta();
  if (!meta) return "<p>The AI is still loading. Open this page again in a moment.</p>";
  const r = meta.results;
  const rows = ["A", "B", "C"].map((k) => `<tr><td>${k === meta.recommended ? "★ " : ""}${METHOD[k].name}<br><span class="muted small">${METHOD[k].slot}</span></td>
    <td>${r[k].trashnet_accuracy}%</td><td>${r[k].realworld_accuracy}%</td></tr>`).join("");
  return `<p>Every photo is identified by <b>three computer-vision methods</b>, all running on your phone:</p>
  <ul><li><b>HOG + linear SVM</b> (classical): describes the directions of the edges in the photo and separates
  the materials with a straight-line rule. Written from scratch by our group.</li>
  <li><b>4-block CNN</b> (trained from scratch): a small neural network we designed and trained on trash photos.</li>
  <li><b>MobileNetV3-Large</b> (pretrained): a network that first learned from 1.28 million everyday photos and was then
  fine-tuned on trash photos.</li></ul>
  <p>Trained on: ${esc(meta.training)}. Accuracy on photos the models never saw:</p>
  <table><tr><th>Method</th><th>Clean test photos</th><th>Real-world photos</th></tr>${rows}</table>
  <p>BIN answers with the starred method, the most accurate on real-world photos. If its confidence is below
  ${pct(meta.thresholds[meta.recommended])} it says "Not sure" instead of guessing.${(() => {
    const t = meta.threshold_stats?.[meta.recommended];
    return t ? ` On real-world test photos it answered ${pct(t.realworld_coverage)} of them, and those answers were
    ${pct(t.realworld_accuracy)} correct.` : "";
  })()}</p>
  <p><a href="${REPO}" target="_blank" rel="noopener">Read the full project, code and report on GitHub</a>.</p>`;
}

function about(r) {
  if (r.arg === "m" && MATERIALS[r.arg2]) return materialPage(r.arg2);
  if (r.arg === "a") return articlePage(r.arg2);
  view().innerHTML = `
  ${header("About BIN")}
  <main class="content with-cta">
    <div class="card about-top"><img src="icons/logo.svg" alt="BIN logo" class="logo-big">
      <h2>BIN</h2><p class="muted small">Take a photo of one piece of trash. BIN tells you if it is cardboard,
      glass, metal, paper, plastic or residual trash, and what to do with it.</p></div>
    <div class="section-title">Help</div>
    <div class="card">${ARTICLES.map((a) => `<button class="article-row" onclick="location.hash='#/about/a/${a.id}'">
      <span class="ico">${a.emoji}</span><span><b>${a.title}</b><div class="muted small">${a.summary}</div></span></button>`).join("")}</div>
    <div class="section-title">What BIN recognises</div>
    <div class="grid2">${Object.entries(MATERIALS).map(([k, m]) => `<button class="tile" style="background:${m.color}"
      onclick="location.hash='#/about/m/${k}'"><span class="emoji">${m.emoji}</span><div><b>${k}</b>
      <div class="small" style="opacity:.9">${BIN_LABEL[m.bin]}</div></div></button>`).join("")}</div>
    <div class="card" style="margin-top:14px">
      <div class="menu-row"><span class="ico">${icon("shield")}</span><span>Photos are identified on this phone and never uploaded.</span></div>
      <a class="menu-row" href="${REPO}" target="_blank" rel="noopener" style="color:inherit;text-decoration:none">
        <span class="ico">${icon("link")}</span><span>Project on GitHub</span><span class="chev">›</span></a>
    </div>
    <p class="muted small" style="text-align:center;margin-top:16px">BIN · Computer Vision Group Project · West Visayas State University</p>
  </main>
  <div class="cta"><button class="btn" onclick="location.hash='#/scan'">${icon("camera")}Scan an item</button></div>`;
}

function pageShell(title, body) {
  view().innerHTML = `${header(title, "#/about")}
  <main class="content with-cta"><div class="card article">${body}</div></main>
  <div class="cta"><button class="btn" onclick="location.hash='#/scan'">${icon("camera")}Scan an item</button></div>`;
}
function materialPage(k) {
  const m = MATERIALS[k];
  pageShell(cap(k), `<div class="verdict"><div class="emoji">${m.emoji}</div><div><h2>${k}</h2>
    <span class="chip ${m.bin}">${BIN_LABEL[m.bin]}</span></div></div>
    <ul>${m.guide.map((g) => `<li>${g}</li>`).join("")}</ul>`);
}
function articlePage(id) {
  const a = ARTICLES.find((x) => x.id === id) || ARTICLES[0];
  pageShell(a.title, `<h2>${a.emoji} ${a.title}</h2>${a.id === "ai" ? aiArticle() : a.body}`);
}

// ------------------------------------------------------------------ misc
let toastTimer = null;
function toast(msg) {
  document.querySelector(".toast")?.remove();
  const t = document.createElement("div");
  t.className = "toast";
  t.textContent = msg;
  document.body.appendChild(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), 3500);
}

if ("serviceWorker" in navigator && location.protocol !== "file:") {
  const hadController = !!navigator.serviceWorker.controller;
  navigator.serviceWorker.register("sw.js").catch(() => {});
  // when a new version of BIN is installed, reload once so the user gets it
  navigator.serviceWorker.addEventListener("controllerchange", () => { if (hadController) location.reload(); });
}
startModels();
render();
