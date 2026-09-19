# MONGDEE MINI KIOSK

AI Booth OS | 360° Product Scanner — a local web app that watches a webcam
pointed at the booth's display area, recognizes whichever product is put in
front of it (set on the turntable, or just picked up and held toward the
camera), and shows that product's story on screen. When a product is
recognized the kiosk pops up a full detail card — photo or autoplaying
video, price, origin, materials, story — that the customer can dismiss to
go back to the live view. Also logs every scan so a dashboard can report
which products get the most attention.

## Why "local web app"

The backend (Python/FastAPI) runs entirely on the booth PC and talks to
the webcam directly — no internet needed at the venue. It happens to serve
its UI as a normal website on `localhost`, which means:

- Today: open a fullscreen browser window pointed at `http://127.0.0.1:8000/`
  on the kiosk PC. Fully offline.
- Later: change `HOST` in `backend/app/config.py` from `127.0.0.1` to
  `0.0.0.0` and the same server is reachable from other devices on the
  booth's Wi-Fi (e.g. the shop owner checking `/dashboard` from their
  phone). No rewrite needed.

## Why the turntable isn't "controlled"

The turntable spins continuously on its own and can't be triggered or
synced to. So instead of "start rotation → shoot N synced frames", the
camera watches the feed constantly and:

1. **Presence detection** (`backend/app/camera.py`) — every live frame's ROI
   is diffed against a single fixed "empty platform" reference snapshot
   (captured explicitly for the selected camera via the
   "บันทึกภาพพื้นเปล่า" button on `/calibrate`). This intentionally does
   *not* adapt over time: an adaptive background model would slowly
   relearn a motionless object as "background" the longer it sat there,
   silently breaking detection — a fixed reference only changes when
   someone deliberately recaptures it. Enough changed pixels in the ROI
   flips the system into a "present" state after a few consecutive frames
   and a minimum hold time (debounced against flicker/shadows). Without a
   reference, the kiosk asks for calibration; it never assumes a first
   camera frame is empty. Exposure is compensated using pixels outside ROI.
2. **Passive burst capture** — once "present", the system just samples
   frames for ~7 seconds (`BURST_DURATION_S` in config). Because the
   turntable never stops, that window alone carries several different
   angles of the object past the camera — no coordination with the motor
   required.

## Why embeddings instead of a trained classifier

Products get added one at a time through the enrollment flow, not as a
big labeled dataset — there's nothing to retrain a classifier on after
each addition. Instead, every captured frame is converted into a feature
vector using a pretrained MobileNetV3-Small (ImageNet weights, classifier
head removed). A new product is matchable immediately after enrollment —
no training step — and cosine similarity between live and stored vectors
doubles as a similarity score, not a calibrated probability (low similarity
→ "no confident match" + manual pick, rather than a false-confident guess).

Before a frame is embedded, the booth background (turntable, table, nearby
clutter) is masked out against the fixed empty-platform reference — the same
snapshot presence detection uses — so the vector describes the product, not
the booth. Enrollment and recognition run this masking identically, so an
empty-platform reference must be captured on `/calibrate` before enrolling.
Embeddings are versioned (`config.EMBED_VERSION`); products enrolled under an
older pipeline show a "re-scan" flag and are skipped by matching until
re-captured (their metadata is untouched). See [STABILITY.md](STABILITY.md).

## Recognition outcomes

Every scan lands in one of three states (`backend/app/recognition_loop.py`):

- **matched** — a clear best candidate above `MATCH_CONFIDENCE_THRESHOLD`,
  with at least five usable frames and multi-view consensus →
  shown immediately, no human step.
- **ambiguous** — candidates are within `AMBIGUOUS_MARGIN` of each other,
  or live frames disagree → too close to call automatically,
  so the kiosk asks a person to pick which one it actually is.
- **unknown** — nothing clears the threshold → the kiosk says so and offers
  "เพิ่มสินค้านี้เข้าระบบ". The frames already captured during that failed
  scan are held in `enroll_sessions` and handed to `/enroll` via a
  `?session=` id, so accepting the offer goes straight to frame selection
  instead of re-capturing from scratch.

## Enrollment: review before saving

See [STABILITY.md](STABILITY.md) for setup after the stability update,
regression tests, and validation limits. Operational states `unstable`,
`camera_unavailable` and `needs_reference` do not count as product scans.
Unusable captures retry at most twice. Three fresh views and two failed
verifications are required before rescanning a displayed match. A manual
selection is kept until removal.

`/enroll` no longer commits a single auto-picked frame. A capture burst
returns every frame it grabbed (pre-selected by default); staff click to
deselect blurry/badly-framed ones, or hit "ถ่ายเพิ่ม" to capture another
burst if too few survive review. At least `ENROLL_MIN_IMAGES` (default 3)
must stay selected before the metadata form unlocks — only the selected
frames' embeddings are saved for that product. Metadata now also includes
an optional `video_url` (a link or presentation video), shown on the kiosk
display as a clickable pill when a product is matched.

## Setup (Windows)

1. Install Python 3.10+ if not already installed.
2. Connect the USB webcam.
3. **One-time, needs internet:** run `start.bat` once before going offline.
   It creates a venv, installs dependencies, and torchvision downloads the
   MobileNetV3 pretrained weights (~10MB, cached under
   `%USERPROFILE%\.cache\torch`). After this, the app runs fully offline.
4. From then on, just double-click `start.bat` and open
   `http://127.0.0.1:8000/` in a browser (fullscreen with F11 for the kiosk
   display).

Pages:
- `/` — kiosk display (fullscreen on the booth screen)
- `/products` — customer-facing showcase: browsable grid of every product;
  tap a card for the same photo/video/detail popup the kiosk shows on a scan
- `/enroll` — add a new product (staff/admin use)
- `/manage` — edit or delete existing products
- `/dashboard` — scan reports (most-viewed products, busiest hours)
- `/calibrate` — live ROI/sensitivity tuning + "capture empty platform" reference

## Developing without a camera

Double-click `start-fake.bat` (or set `KIOSK_FAKE_CAMERA=1`; `auto` cycles
empty/product scenes by itself). A simulated camera (`backend/app/fake_camera.py`)
replaces the webcam: it shows an empty turntable and can place a "rotating"
product on it, so calibration, enrollment and recognition all run end to end.
Switch scenes at `http://127.0.0.1:8000/dev/fake-camera`. Products come from
`backend/data/fake_camera/products/*.jpg|png` (falls back to
`data/captures/covers`), the empty platform from `data/fake_camera/empty.png`
(falls back to `data/reference_camera_0.png`). It uses camera index 99, so it
never touches a real camera's saved reference. Similarity scores from it say
nothing about real-world accuracy — thresholds still need tuning on site.

## Admin PIN (deleting data)

Deleting a product, a product video or a gallery photo asks for an admin PIN
(4–12 digits). The server enforces it (`X-Admin-Pin` header), so it can't be
bypassed from the browser. The first delete asks you to choose the PIN;
`/manage` has a button to change it. It is stored salted and hashed, and five
wrong guesses lock the check for a minute. Forgot it? On the kiosk PC run
`backend\venv\Scripts\python.exe backend\scripts\reset_admin_pin.py`, then
choose a new one on the next delete.

## On-site calibration (do this before the demo, with the real hardware)

Do this at `/calibrate` (live sliders, no restart needed) rather than
editing `config.py` by hand:

0. **Capture the empty-platform reference first, before anything else.**
   Make sure the platform is genuinely empty, then click "บันทึกภาพพื้นเปล่า"
   at the top of the page. Recapture it any time the camera moves, the
   lighting changes, or the platform is left occupied for a long stretch
   (see `Presence detection` above for why this step exists — everything
   else is measured relative to this one snapshot).
- `ROI_FRACTION` (the orange box) — crop this to just the turntable surface
  in-frame. The tighter this box, the more reliable presence detection is.
- `PRESENCE_ON_RATIO` / `PRESENCE_OFF_RATIO` — watch the live number with
  the platform empty vs. with a product on it, and set thresholds between
  the two observed values. Realistic values are usually 0.05–0.3, not
  close to 1 — the sliders are capped accordingly for a reason.
- `BURST_DURATION_S` — should cover at least one slow rotation of the
  turntable so multiple sides get captured.
- `MATCH_CONFIDENCE_THRESHOLD` — after enrolling a few real products, place
  one back on the platform and check the score it gets vs. the score
  other, wrong products get; set the threshold between them.
- `AMBIGUOUS_MARGIN` — with several similar-looking real products enrolled,
  check how close their scores land against each other and set this margin
  wide enough to catch genuine ties without flagging every scan as ambiguous.

## Architecture

```
backend/app/
  camera.py            presence detection + frame capture (background thread)
  vision.py            MobileNetV3 embeddings + cosine-similarity matching
  recognition_loop.py  background task: presence change -> capture -> match(3-way) -> log
  enroll_sessions.py   in-memory captured-frames store shared by enrollment + "unknown" handoff
  db.py                SQLite (products, product_embeddings, scan_events)
  routers/
    enrollment.py       multi-batch capture -> frame selection -> confirm-with-metadata
    recognition.py      current live result + manual correction
    products.py         CRUD
    analytics.py        summary + hourly aggregates for the dashboard
  main.py               FastAPI app, MJPEG stream, static frontend serving

frontend/                plain HTML/CSS/JS (no build step, no CDN — offline-safe)
  index.html / kiosk.js       live display
  enroll.html / enroll.js     add-product flow
  dashboard.html / dashboard.js  reports
```

## Known limitations (MVP)

- A product is only scanned once per "placed" session — remove and replace
  it (or use the manual pick UI, or the ambiguous/unknown flows) to retry.
- Manual corrections (and "unknown" enrollments) are logged for analytics
  but the embedding index isn't automatically re-weighted from corrections —
  a natural next step per the pitch script's Q5/Q8.
- Recognition accuracy at scale (hundreds of products) is untested; the
  in-memory cosine-similarity index rebuilds from all embeddings on every
  enrollment/deletion, which is fine for a small catalog but would need a
  proper vector index (e.g. FAISS) to scale further, per Q4.
- Enrollment sessions (both manual and the "unknown → add" handoff) live
  in memory with a 5-minute TTL — restarting the server or waiting too
  long between capture and confirm loses that in-progress session (saved
  products in the database are unaffected).
