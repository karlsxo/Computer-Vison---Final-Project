<img src="web/icons/logo.svg" alt="BIN logo" width="88" align="right">

# How to open and use BIN

**BIN** identifies the material of one piece of trash from a photo — **cardboard, glass, metal,
paper, plastic or residual trash** — and tells you which bin it belongs in and what to do with it.
It runs entirely on your phone: photos are never uploaded.

**App link: <https://karlsxo.github.io/Computer-Vison---Final-Project/>**

---

## 1. Open BIN on your phone

### Android (Chrome)
1. Open **Chrome** and go to the app link above.
2. When asked *"Allow karlsxo.github.io to use your camera?"*, tap **Allow**.
3. The first time, BIN downloads its AI models (about 32 MB, once). Wait until the message at the
   top says **"Point the camera at one piece of trash"**.

### iPhone (Safari)
1. Open **Safari** and go to the app link above.
2. Tap **Allow** when Safari asks for the camera.
3. Wait until the message at the top says **"Point the camera at one piece of trash"**.

> Use Wi-Fi or mobile data for the first visit. After that BIN also works **offline**.

## 2. Install BIN like a normal app (recommended)

| Phone | Steps |
|---|---|
| **Android** | In Chrome tap the **⋮** menu (top right) → **Add to home screen** (or **Install app**) → **Install**. |
| **iPhone** | In Safari tap the **Share** button (square with arrow) → **Add to Home Screen** → **Add**. |

The green **BIN** icon (a trash can with a B on its cover) appears on your home screen. It opens
full screen, without the browser bar, straight to the camera.

## 3. Identify a piece of trash

BIN opens directly on the camera.

1. Put **one** piece of trash **inside the bright square** so that it fills most of it. Only the bright
   square is scanned; the darkened area around it is ignored.
2. Use a **plain background** (floor, table, a sheet of paper) and **good light**.
3. Tap the big green **Identify** button.

Other buttons on the camera screen:

| Button | What it does |
|---|---|
| 🖼️ **Gallery** (left) | Choose a photo that is already on your phone. |
| ⇄ **Flip** (right) | Switch between the back and the front camera. |
| 🕘 **History** (top right) | Your previous scans. |
| ⓘ **About** (top right) | Help, material guides and how BIN works. |

## 4. Read the result

* **Material and bin** — e.g. *Plastic · Recyclable · 93 % sure*. The answer is also drawn on your photo.
* **What to do** — how to prepare the item for its bin (rinse, flatten, keep dry, …).
* **"Not sure"** — BIN only answers when it is confident enough (65 % or more). Otherwise it shows
  its best guess and asks you to retake the photo: closer, from another side, with better light or
  a plainer background.
* **All three methods** — the predictions of the three computer-vision methods of the project
  (HOG + SVM, our CNN, MobileNetV3) with their confidence and the time they took on your phone.
  BIN answers with the ★ method, MobileNetV3, the most accurate on real-world photos.
* **Comparison of the three methods** — a table in the same format as the course instructions:
  *Method · Slot · Metric 1 (top-1 accuracy %) · Metric 2 (macro-F1) · Runtime (ms, measured on your
  phone's CPU for this photo) · Params (millions) · Train time (min)*. The accuracy numbers were measured on
  600 real-world photos the models never saw.
* **What the three slots mean** — a plain-language explanation of *A. Classical* (HOG + linear SVM),
  *B. Trained from scratch* (our 4-block CNN) and *C. Pretrained* (MobileNetV3): how each works, its
  strength and its weakness.
* Tap **Scan next item** to return to the camera.

## 5. What BIN can recognise

| Material | Bin | Examples |
|---|---|---|
| 📦 Cardboard | Recyclable | boxes, cartons, corrugated sheets |
| 🍾 Glass | Recyclable | bottles (clear, green, brown), jars, glass cups |
| 🥫 Metal | Recyclable | soda and beer cans, food tins, paint cans, aluminium trays |
| 📄 Paper | Recyclable | newspapers, magazines, flyers, office paper, envelopes |
| 🧴 Plastic | Recyclable | PET bottles, shampoo/detergent bottles, food tubs, containers |
| 🗑️ Trash | Residual | wrappers, chip bags, styrofoam, face masks, toothbrushes, diapers |

Measured accuracy of BIN's model: **97.7 %** on 600 real-world photos it never saw and **94.1 %**
on the TrashNet test photos.

**Not supported:** food scraps and other biodegradable waste, batteries and electronics (special
waste), several items in one photo, and mixed-material items such as Tetra Pak cartons. For these
BIN may guess or say "Not sure".

## 6. Troubleshooting

| Problem | Solution |
|---|---|
| Black screen / no camera | Allow the camera: Chrome **⋮ → Settings → Site settings → Camera**, or iPhone **Settings → Safari → Camera**. You can also tap **Identify** to use your normal camera app, or **Gallery**. |
| "Preparing the AI" stays at a low percent | The first visit downloads ~32 MB. Check your internet connection and wait, or reload the page. |
| "AI could not load" | Reload with an internet connection; after one successful load BIN works offline. |
| Often "Not sure" | One item only, fill the frame, plain background, good light, hold the phone still. |
| Old version of the app | Close BIN completely and open it again; it updates itself. |

## 7. Open BIN on a phone connected to a laptop (USB, Android)

With **USB debugging** enabled on the phone (*Settings → About phone → Software information → tap
Build number 7 times*, then *Developer options → USB debugging*) and the Android platform tools
installed, this command opens BIN on the phone:

```bash
adb shell am start -a android.intent.action.VIEW -d "https://karlsxo.github.io/Computer-Vison---Final-Project/"
```

Accept the *"Allow USB debugging?"* prompt on the phone the first time.

## 8. Desktop version (laptop / Colab)

The same models and rules are available as a desktop web page made with Gradio:

```bash
python -m venv .venv
.venv\Scripts\activate              # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app/app.py                   # then open http://127.0.0.1:7860
```

Upload or capture a photo and click **Identify**. In Google Colab, run
`python app/app.py --share` to get a temporary public link.

## 9. Run the phone app from your own computer (developers)

```bash
python -m http.server 8000 --directory web    # then open http://127.0.0.1:8000
```

The live camera needs `https://` or `localhost`; on a plain `http://` address in your local network
the **Identify** button opens the phone's camera app instead. The app is published with GitHub
Pages from the `gh-pages` branch (the contents of `web/`).

---

*BIN is part of the Computer Vision Group Project, West Visayas State University. Full technical
documentation: [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md).*
