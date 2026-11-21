// Trash Sorter - phone app (screens, camera, history, stats, profile).
// Everything runs on the phone: photos never leave the device.
import { loadModels, classify, getMeta } from "./models.js";
import { decodeImage, bitmapToImageData, thumbnail } from "./imaging.js";
import { MATERIALS, BIN_LABEL, TIPS, ARTICLES, BADGES } from "./content.js";

const REPO = "https://github.com/karlsxo/Computer-Vison---Final-Project";
const METHOD = {
  A: { name: "HOG + linear SVM", slot: "Classical", color: "#d1495b" },
  B: { name: "4-block CNN", slot: "Trained from scratch", color: "#00798c" },
  C: { name: "MobileNetV3-Large", slot: "Pretrained", color: "#30638e" },
};

// ------------------------------------------------------------------ icons
const P = {
  home: '<path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z"/>',
  learn: '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v15H6.5A2.5 2.5 0 0 0 4 20.5z"/><path d="M4 20.5A2.5 2.5 0 0 0 6.5 23H20v-5"/>',
  scan: '<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><path d="M4 12h16"/>',
  stats: '<path d="M12 3a9 9 0 1 0 9 9h-9z"/><path d="M15 2.6A9 9 0 0 1 21.4 9H15z"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5"/>',
  back: '<path d="M15 5l-7 7 7 7"/>',
  image: '<rect x="3" y="4" width="18" height="16" rx="3"/><circle cx="9" cy="10" r="2"/><path d="M21 16l-5-5-9 9"/>',
  flip: '<path d="M4 8h12l-3-3M20 16H8l3 3"/>',
  camera: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
  del: '<path d="M5 7h14M9 7V4h6v3M7 7l1 13h8l1-13"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
  reset: '<path d="M4 12a8 8 0 1 0 2.4-5.7L4 8.5"/><path d="M4 4v4.5h4.5"/>',
  shield: '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/>',
  link: '<path d="M10 14l4-4"/><path d="M8.5 11.5l-2 2a3.2 3.2 0 0 0 4.5 4.5l2-2"/><path d="M15.5 12.5l2-2a3.2 3.2 0 0 0-4.5-4.5l-2 2"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
};
const icon = (n) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${P[n]}</svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const pct = (x) => `${Math.round(x * 100)}%`;

// ------------------------------------------------------------------ state (saved on the phone only)
const KEY = "trashSorter.v1";
function loadState() {
  try {
    const s = JSON.parse(localStorage.getItem(KEY));
    if (s && Array.isArray(s.history)) return s;
  } catch (e) { /* storage unavailable or corrupt: start fresh */ }
  return { name: "Recycler", school: "", history: [] };
}
const state = loadState();
function save() {
  try { localStorage.setItem(KEY, JSON.stringify(state)); }
  catch (e) { toast("Could not save on this device (storage full or blocked)."); }
}
function summary() {
  const points = state.history.reduce((a, h) => a + h.points, 0);
  const counts = {};
  for (const h of state.history) counts[h.label] = (counts[h.label] || 0) + 1;
  return { points, items: state.history.length, counts, kinds: Object.keys(counts).length,
           level: Math.floor(points / 100) + 1, toNext: 100 - (points % 100), frac: (points % 100) / 100 };
}

// ------------------------------------------------------------------ models
const models = { ready: false, progress: 0, text: "Preparing", error: null, promise: null };
function startModels() {
  models.promise = loadModels((f, t) => {
    models.progress = f; models.text = t;
    const bar = document.querySelector("#model-progress");
    if (bar) bar.style.width = `${Math.round(f * 100)}%`;
    const lab = document.querySelector("#model-text");
    if (lab) lab.textContent = `${t} · ${Math.round(f * 100)}%`;
  }).then(() => { models.ready = true; if (route().name === "home") render(); })
    .catch((e) => { models.error = e.message || String(e); if (route().name === "home") render(); });
}

// ------------------------------------------------------------------ routing
const view = () => document.querySelector("#view");
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  return { name: parts[0] || "home", arg: parts[1], arg2: parts[2] };
}
const go = (h) => { location.hash = h; };
window.addEventListener("hashchange", render);

function navBar(active) {
  const item = (h, n, ico, label) => `<a href="#/${h}" class="${active === n ? "on" : ""}">${icon(ico)}${label}</a>`;
  return `<nav class="nav">${item("home", "home", "home", "Home")}${item("learn", "learn", "learn", "Learn")}
    <a href="#/scan" aria-label="Scan an item"><span class="fab">${icon("scan")}</span></a>
    ${item("stats", "stats", "stats", "My stats")}${item("profile", "profile", "user", "Profile")}</nav>`;
}

let current = null;   // latest classification result
function render() {
  stopCamera();
  const r = route();
  const screens = { home, scan, result, stats, learn, profile };
  (screens[r.name] || home)(r);
  window.scrollTo(0, 0);
}

// ------------------------------------------------------------------ home
function home() {
  const s = summary();
  const C = 2 * Math.PI * 50;
  const recent = state.history.slice(-5).reverse();
  const tip = TIPS[new Date().getDate() % TIPS.length];
  view().innerHTML = `
  <header class="header"><div class="header-row">
    <div class="avatar">${esc(state.name.trim().charAt(0).toUpperCase() || "R")}</div>
    <div style="flex:1"><div class="sub">Hello,</div><h1>${esc(state.name)}</h1></div>
    <span class="coin"><i>★</i>${s.points}</span></div></header>
  <main class="content">
    ${modelBanner()}
    <div class="card level-card" style="margin-top:12px">
      <div class="ring"><svg viewBox="0 0 120 120"><circle cx="60" cy="60" r="50" stroke="#e6f4ea" stroke-width="12" fill="none"/>
        <circle cx="60" cy="60" r="50" stroke="#1f8b4c" stroke-width="12" fill="none" stroke-linecap="round"
          stroke-dasharray="${C}" stroke-dashoffset="${C * (1 - s.frac)}"/></svg>
        <div class="center"><div><span style="font-size:26px">♻️</span><b>Lv ${s.level}</b></div></div></div>
      <div><h3>Level ${s.level}</h3>
        <div class="muted small">${s.items} item${s.items === 1 ? "" : "s"} sorted · ${s.toNext} points to level ${s.level + 1}</div>
        <button class="btn" style="margin-top:12px;min-height:44px" onclick="location.hash='#/scan'">${icon("camera")}Scan an item</button></div>
    </div>
    <div class="section-title">Recently sorted ${state.history.length ? '<a href="#/stats">See all</a>' : ""}</div>
    <div class="card">${recent.length ? recent.map(itemRow).join("") :
      `<div class="empty"><div class="big">📷</div>Nothing yet. Tap the green scan button to sort your first item.</div>`}</div>
    <div class="section-title">Tip of the day</div>
    <div class="card tip"><span class="emoji">${tip[0]}</span><div>${tip[1]}</div></div>
  </main>${navBar("home")}`;
  bindDeletes();
}

function modelBanner() {
  if (models.ready) return "";
  if (models.error) {
    return `<div class="card"><b>The AI models could not load.</b><div class="muted small">${esc(models.error)}.
      Check your internet connection for the first start, then reopen the app.</div></div>`;
  }
  return `<div class="card model-banner"><div style="font-size:28px">🤖</div><div style="flex:1">
    <b class="small">Getting the AI ready</b><div class="muted small" id="model-text">${esc(models.text)} · ${Math.round(models.progress * 100)}%</div>
    <div class="progress"><div id="model-progress" style="width:${Math.round(models.progress * 100)}%"></div></div></div></div>`;
}

function itemRow(h) {
  const m = MATERIALS[h.label];
  const d = new Date(h.ts);
  return `<div class="item-row"><img class="thumb" src="${h.thumb}" alt="">
    <div class="meta"><b>${m.emoji} ${h.label}</b>
      <span class="chip ${m.bin}">${BIN_LABEL[m.bin]}</span>
      <span class="muted small"> · ${d.toLocaleDateString()} ${d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span></div>
    <button class="del" data-del="${h.id}" aria-label="Delete">${icon("del")}</button></div>`;
}
function bindDeletes() {
  document.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", () => {
    state.history = state.history.filter((h) => h.id !== b.dataset.del);
    save(); render();
  }));
}

// ------------------------------------------------------------------ scanner
let stream = null, facing = "environment";
function stopCamera() {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
}
function scan() {
  view().innerHTML = `
  <div class="scan">
    <video id="video" playsinline muted autoplay></video>
    <div class="no-camera hidden" id="nocam"><div><div class="big">📷</div>
      <p>Use the buttons below to take a photo with your camera or choose one from your gallery.</p>
      <p class="small" id="nocam-why"></p></div></div>
    <div class="top"><button class="icon-btn" onclick="history.length > 1 ? history.back() : location.hash='#/home'" aria-label="Back">${icon("back")}</button>
      <span class="hint">Put your item in the scanner frame</span></div>
    <div class="frame" id="frame"><i></i><i></i><i></i><i></i></div>
    <div class="sheet">
      <div class="controls">
        <div><button class="round" id="gallery" aria-label="Upload from gallery">${icon("image")}</button><div class="ctrl-label">Gallery</div></div>
        <div><button class="shutter" id="shutter" aria-label="Take photo"></button><div class="ctrl-label">Scan</div></div>
        <div><button class="round" id="flip" aria-label="Switch camera">${icon("flip")}</button><div class="ctrl-label">Flip</div></div>
      </div>
      <p class="muted small" style="text-align:center;margin:12px 0 0">One item, plain background, good light. Photos stay on your phone.</p>
    </div>
    <input type="file" accept="image/*" id="galleryInput" hidden>
    <input type="file" accept="image/*" capture="environment" id="cameraInput" hidden>
  </div>`;
  const video = document.querySelector("#video");
  document.querySelector("#gallery").onclick = () => document.querySelector("#galleryInput").click();
  document.querySelector("#galleryInput").onchange = (e) => e.target.files[0] && analyseFile(e.target.files[0]);
  document.querySelector("#cameraInput").onchange = (e) => e.target.files[0] && analyseFile(e.target.files[0]);
  document.querySelector("#flip").onclick = () => { facing = facing === "environment" ? "user" : "environment"; startCamera(video); };
  document.querySelector("#shutter").onclick = () => {
    if (stream && video.videoWidth) analyseImageData(cropToFrame(video));
    else document.querySelector("#cameraInput").click();   // native camera app
  };
  startCamera(video);
}

async function startCamera(video) {
  stopCamera();
  const nocam = document.querySelector("#nocam");
  const why = document.querySelector("#nocam-why");
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    nocam.classList.remove("hidden");
    why.textContent = window.isSecureContext ? "" : "(The live viewfinder needs a secure https:// address; the Scan button opens your camera app instead.)";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    if (route().name !== "scan") { stopCamera(); return; }
    video.srcObject = stream;
    await video.play();
    nocam.classList.add("hidden");
  } catch (e) {
    nocam.classList.remove("hidden");
    why.textContent = "Camera permission was not given; the Scan button opens your camera app instead.";
  }
}

/** Crop the video frame to the square scanner frame shown on screen. */
function cropToFrame(video) {
  const vr = video.getBoundingClientRect(), fr = document.querySelector("#frame").getBoundingClientRect();
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(vr.width / vw, vr.height / vh);            // object-fit: cover
  const ox = (vr.width - vw * s) / 2, oy = (vr.height - vh * s) / 2;
  let x = (fr.left - vr.left - ox) / s, y = (fr.top - vr.top - oy) / s;
  let side = Math.min(fr.width / s, vw, vh);
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
  const done = busy(models.ready ? "Analysing" : "Getting the AI ready");
  try {
    if (!models.ready) {
      const iv = setInterval(() => {
        const t = document.querySelector("#busy-text");
        if (t) t.textContent = `Getting the AI ready · ${Math.round(models.progress * 100)}%`;
      }, 300);
      await models.promise;
      clearInterval(iv);
      if (models.error) throw new Error(models.error);
    }
    const res = await classify(img);
    current = { res, photo: thumbnail(img, 720), thumb: thumbnail(img, 160), chosen: res[res.recommended].label };
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
  return `<b>${METHOD[rec].name}</b> is the default because it was the most accurate in our tests:
    ${r[rec].accuracy}% on held-out test photos (vs ${others.map((k) => `${r[k].accuracy}% for ${METHOD[k].name}`).join(", ")}),
    and ${r[rec].realworld_accuracy}% on unseen real-world photos.`;
}

function result() {
  if (!current) { go("#/scan"); return; }
  const { res } = current, rec = res.recommended, top = res[rec];
  const chosen = current.chosen, m = MATERIALS[chosen];
  const agree = new Set(["A", "B", "C"].map((k) => res[k].label)).size === 1;
  view().innerHTML = `
  <header class="header"><div class="header-row">
    <button class="icon-btn" onclick="location.hash='#/scan'" aria-label="Back">${icon("back")}</button>
    <h1>Scan result</h1></div></header>
  <main class="content">
    <div class="photo"><img src="${current.photo}" alt="Your photo"><div class="band">${top.label.toUpperCase()} ${pct(top.confidence)}</div></div>
    <div class="card" style="margin-top:12px">
      <div class="verdict"><div class="emoji">${m.emoji}</div><div>
        <h2>${chosen}</h2>
        <span class="chip ${m.bin}">${BIN_LABEL[m.bin]}</span>
        <span class="muted small"> ${chosen === top.label ? `${pct(top.confidence)} confident` : "corrected by you"}</span></div></div>
      <p class="small" style="margin:12px 0 0"><b>What to do:</b> ${m.tip}</p>
      ${top.confidence < 0.5 && chosen === top.label ? `<p class="small" style="color:#a12f2f;margin:8px 0 0">Low confidence: try again with one item on a plain background.</p>` : ""}
    </div>
    <div class="section-title">All three methods</div>
    <div class="card">${["A", "B", "C"].map((k) => methodRow(k, res[k], k === rec)).join("")}
      <p class="muted small" style="margin:10px 0 0">${agree ? "All three methods agree." : "The methods disagree; the recommended method (★) is the most reliable."}
      ${reasonText(rec)} Runtimes measured on this phone.</p></div>
    <div class="section-title">Not right? Choose the material</div>
    <div class="card"><div class="picks">${getMeta().classes.map((c) =>
      `<button class="chip pick ${c === chosen ? "on" : ""}" data-pick="${c}">${MATERIALS[c].emoji} ${cap(c)}</button>`).join("")}</div></div>
    <div class="btn-row"><button class="btn secondary" onclick="location.hash='#/scan'">${icon("camera")}Scan again</button>
      <button class="btn" id="add">${icon("plus")}Add +${m.bin === "recyclable" ? 10 : 5} pts</button></div>
  </main>${navBar("")}`;
  document.querySelectorAll("[data-pick]").forEach((b) => b.addEventListener("click", () => { current.chosen = b.dataset.pick; result(); }));
  document.querySelector("#add").onclick = () => {
    const pts = MATERIALS[current.chosen].bin === "recyclable" ? 10 : 5;
    state.history.push({ id: String(Date.now()), ts: Date.now(), label: current.chosen, predicted: top.label,
                         confidence: top.confidence, method: rec, thumb: current.thumb, points: pts });
    save();
    current = null;
    toast(`+${pts} points! Thanks for sorting right.`);
    go("#/home");
  };
}

function methodRow(k, r, isRec) {
  const M = METHOD[k];
  return `<div class="method"><div class="head"><div><div class="name">${isRec ? '<span class="star">★</span> ' : ""}${M.name}</div>
      <div class="slot">${M.slot}${isRec ? " · recommended" : ""}</div></div>
      <div class="pred">${MATERIALS[r.label].emoji} ${r.label} · ${pct(r.confidence)}</div></div>
    <div class="bar"><div style="width:${Math.max(3, r.confidence * 100)}%;background:${M.color}"></div></div>
    <div class="muted small" style="margin-top:4px">${r.ms.toFixed(0)} ms${k === "A" ? " · score, not a calibrated probability" : ""}</div></div>`;
}

// ------------------------------------------------------------------ stats
function stats() {
  const s = summary(), classes = Object.keys(MATERIALS);
  const total = s.items;
  let arcs = "", off = 0;
  const R = 80, C = 2 * Math.PI * R;
  for (const c of classes) {
    const n = s.counts[c] || 0;
    if (!n) continue;
    const len = (n / total) * C;
    arcs += `<circle cx="100" cy="100" r="${R}" fill="none" stroke="${MATERIALS[c].color}" stroke-width="26"
      stroke-dasharray="${Math.max(0, len - 2)} ${C}" stroke-dashoffset="${-off}"/>`;
    off += len;
  }
  view().innerHTML = `
  <header class="header"><div class="header-row"><h1>My stats</h1><span class="coin"><i>★</i>${s.points}</span></div></header>
  <main class="content">
    <div class="card"><b>Sorted materials</b>
      <div class="donut"><svg viewBox="0 0 200 200" style="transform:rotate(-90deg)">
        <circle cx="100" cy="100" r="${R}" fill="none" stroke="#e6f4ea" stroke-width="26"/>${arcs}</svg>
        <div class="center"><div><span class="muted small">Sorted</span><b>${total}</b><span class="muted small">items</span></div></div></div>
      ${classes.map((c) => `<div class="legend-row"><span class="dot" style="background:${MATERIALS[c].color}"></span>
        <span style="flex:1;text-transform:capitalize">${MATERIALS[c].emoji} ${c}</span>
        <b>${s.counts[c] || 0}</b><span class="muted small" style="width:44px;text-align:right">${total ? Math.round(((s.counts[c] || 0) / total) * 100) : 0}%</span></div>`).join("")}
    </div>
    <div class="card stat-pair"><div class="stat"><b>${s.points}</b><span class="muted small">points</span></div>
      <div class="stat"><b>Lv ${s.level}</b><span class="muted small">${s.toNext} pts to next</span></div></div>
    <div class="section-title">History</div>
    <div class="card">${state.history.length ? state.history.slice().reverse().map(itemRow).join("") :
      `<div class="empty"><div class="big">📊</div>Your sorted items will appear here.</div>`}</div>
  </main>${navBar("stats")}`;
  bindDeletes();
}

// ------------------------------------------------------------------ learn
function aiArticle() {
  const r = getMeta()?.results;
  if (!r) return "<p>The AI models are still loading. Open this page again in a moment.</p>";
  const rows = ["A", "B", "C"].map((k) => `<tr><td>${METHOD[k].name}<br><span class="muted small">${METHOD[k].slot}</span></td>
    <td>${r[k].accuracy}%</td><td>${r[k].realworld_accuracy}%</td><td>${k === "A" ? "0 (no learned layers)" : `${r[k].params_M} M`}</td></tr>`).join("");
  return `<p>Every photo is checked by <b>three different computer-vision methods</b>, all running on your phone:</p>
  <ul><li><b>HOG + linear SVM</b> (classical): describes the directions of the edges in the photo and separates
  materials with a straight-line rule. Written from scratch by our group.</li>
  <li><b>4-block CNN</b> (trained from scratch): a small neural network we designed and trained only on trash photos.</li>
  <li><b>MobileNetV3-Large</b> (pretrained): a network that first learned from 1.28 million everyday photos and was then
  fine-tuned on trash. It knows what shiny, transparent and printed surfaces look like.</li></ul>
  <table><tr><th>Method</th><th>Test photos</th><th>Real-world photos</th><th>Parameters</th></tr>${rows}</table>
  <p>The app recommends the method with the best results. All three were trained and tested on the same images
  (TrashNet), with near-duplicate photos kept out of the test set. Real-world photos are harder for every method,
  which is why you can correct the answer.</p>
  <p><a href="${REPO}" target="_blank" rel="noopener">Read the full project, code and report on GitHub</a>.</p>`;
}

function learn(r) {
  if (r.arg === "m" && MATERIALS[r.arg2]) return materialPage(r.arg2);
  if (r.arg === "a") return articlePage(r.arg2);
  view().innerHTML = `
  <header class="header"><div class="header-row"><h1>Learn</h1></div>
    <div class="sub">Sort right, recycle more.</div></header>
  <main class="content">
    <div class="section-title">Material guides</div>
    <div class="grid2">${Object.entries(MATERIALS).map(([k, m]) => `<button class="tile" style="background:${m.color}"
      onclick="location.hash='#/learn/m/${k}'"><span class="emoji">${m.emoji}</span><div><b>${k}</b>
      <div class="small" style="opacity:.9">${BIN_LABEL[m.bin]}</div></div></button>`).join("")}</div>
    <div class="section-title">Read more</div>
    <div class="card">${ARTICLES.map((a) => `<button class="article-row" onclick="location.hash='#/learn/a/${a.id}'">
      <span class="ico">${a.emoji}</span><span><b>${a.title}</b><div class="muted small">${a.summary}</div></span></button>`).join("")}</div>
  </main>${navBar("learn")}`;
}

function pageShell(title, body, active = "learn") {
  view().innerHTML = `
  <header class="header"><div class="header-row">
    <button class="icon-btn" onclick="history.length > 1 ? history.back() : location.hash='#/learn'" aria-label="Back">${icon("back")}</button>
    <h1>${title}</h1></div></header>
  <main class="content"><div class="card article">${body}</div></main>${navBar(active)}`;
}
function materialPage(k) {
  const m = MATERIALS[k];
  pageShell(cap(k), `<div class="verdict"><div class="emoji">${m.emoji}</div><div><h2>${k}</h2>
    <span class="chip ${m.bin}">${BIN_LABEL[m.bin]}</span></div></div>
    <ul>${m.guide.map((g) => `<li>${g}</li>`).join("")}</ul>
    <button class="btn" onclick="location.hash='#/scan'">${icon("camera")}Scan an item</button>`);
}
function articlePage(id) {
  const a = ARTICLES.find((x) => x.id === id) || ARTICLES[0];
  pageShell(a.title, `<h2>${a.emoji} ${a.title}</h2>${a.id === "ai" ? aiArticle() : a.body}`);
}

// ------------------------------------------------------------------ profile
function profile() {
  const s = summary();
  view().innerHTML = `
  <header class="header"><div class="header-row"><h1>Profile</h1></div></header>
  <main class="content">
    <div class="card profile-top"><div class="avatar">${esc(state.name.trim().charAt(0).toUpperCase() || "R")}</div>
      <b style="font-size:18px">${esc(state.name)}</b><div class="muted small">${esc(state.school || "Add your school or barangay")}</div>
      <label class="field" style="text-align:left"><span>Your name</span><input id="name" maxlength="40" value="${esc(state.name)}"></label>
      <label class="field" style="text-align:left"><span>School / barangay (optional)</span><input id="school" maxlength="60" value="${esc(state.school)}"></label>
    </div>
    <div class="card stat-pair"><div class="stat"><b>⭐ ${s.points}</b><span class="muted small">points</span></div>
      <div class="stat"><b>♻️ ${s.items}</b><span class="muted small">items sorted</span></div></div>
    <div class="section-title">Badges</div>
    <div class="badges">${BADGES.map((b) => `<div class="badge ${b.test(s) ? "" : "locked"}"><span class="emoji">${b.emoji}</span>${b.name}</div>`).join("")}</div>
    <div class="section-title">More</div>
    <div class="card">
      <button class="menu-row" onclick="location.hash='#/learn/a/ai'"><span class="ico">${icon("info")}</span><span>How the AI decides</span><span class="chev">›</span></button>
      <button class="menu-row" id="privacy"><span class="ico">${icon("shield")}</span><span>Privacy</span><span class="chev">›</span></button>
      <a class="menu-row" href="${REPO}" target="_blank" rel="noopener" style="color:inherit;text-decoration:none"><span class="ico">${icon("link")}</span><span>Project on GitHub</span><span class="chev">›</span></a>
      <button class="menu-row" id="reset"><span class="ico" style="color:#a12f2f;background:#fde4e4">${icon("reset")}</span><span>Reset my data</span><span class="chev">›</span></button>
    </div>
    <p class="muted small" style="text-align:center;margin-top:18px">Trash Sorter · Computer Vision Group Project · West Visayas State University</p>
  </main>${navBar("profile")}`;
  const bind = (id, field) => document.querySelector(id).addEventListener("change", (e) => {
    state[field] = e.target.value.trim() || (field === "name" ? "Recycler" : ""); save(); profile();
  });
  bind("#name", "name");
  bind("#school", "school");
  document.querySelector("#privacy").onclick = () => toast("Your photos are analysed on this phone and never uploaded. History is stored only on this device.");
  document.querySelector("#reset").onclick = () => {
    if (confirm("Delete your history, points and name from this phone?")) {
      state.history = []; state.name = "Recycler"; state.school = ""; save(); profile();
    }
  };
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
  toastTimer = setTimeout(() => t.remove(), 3200);
}

if ("serviceWorker" in navigator && location.protocol !== "file:") {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
startModels();
render();
