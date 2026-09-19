# Camera / Detection / Face fix — report

Branch: `fix/camera-ai-pack`. Everything in the first sections was built and tested in a Linux sandbox
(real pytest, real OpenCV/numpy/scipy, **stubbed torch**). Nothing here has been run on
real USB cameras, and the 613-test Windows baseline has not been re-run. Those gates are
marked BLOCKED, not passed.

## What changed (integrated in the app)

| Problem | Cause found | Change | Evidence |
|---|---|---|---|
| Deleted product still detected | Gallery `.npy` on disk was independent of the catalog (orphan `product-4621553d`, 30 samples); COCO-named product keys were frozen at worker start; import threads could recreate a gallery after delete | `ProductRecognizer.set_active_provider` + `prune_inactive` (moves orphans to `gallery/_quarantine/<ts>/`, never deletes), `add_sample` refuses deleted keys, detections/interest/analytics filtered by catalog, COCO products live-checked | Run on a copy of the real gallery: old code returned `('product-4621553d', 1.0)`, new returns `(None, 0.0)`, orphan quarantined. Mutation-proven tests in `tests/core/test_product_lifecycle.py` |
| Man labelled woman | On-screen label was a single-frame FairFace guess on a tiny person crop | Label now comes from the track's aggregated category; category needs minimum evidence (1.2) and a lead over the runner-up (0.6) else `unknown` | Simulation at 82% per-frame accuracy: wrong label after 1 frame 17.5% → 0%, after 6 frames 3.3% → ~0.5% |
| Tracking loses people on fast moves | Greedy IoU only, no motion model | `core/tracker_v2.py` (constant-velocity Kalman + Hungarian); `PersonTracker(use_motion=True)`, `use_motion=False` = exact legacy path | Real-code A/B, 12 seeds: ID switches walking 18→5 (2 Hz), 6→2 (3 Hz), 7→3 (5 Hz); fast scenes 351→147, 205→119. Synthetic scenes only |
| Broken images on fast camera motion | Torn / truncated frames reached the AI | `core/frame_integrity.py` (temporal tear-seam, truncated tail, exact freeze, colour cast); failed frames dropped as `frame_corrupt:<reason>` | Tear recall 3/4 on fixtures, 0 false positives on live-like sequences |
| Camera hangs / cannot reconnect | `cap.read()` blocks forever on DSHOW | Stall timer + `recover_from_hung_read`, BoothManager watchdog thread (2 s); startup probing limited to enumerated devices; HRESULT decoder `core/hresult.py` | Unit tests with fake hung captures |
| No face detection | — | `core/face/` (YuNet full-frame, quality score, best-shot store, face→person binding), overlay + `faces` count in camera snapshot; `--no-face-detect` flag | Tests against the real YuNet model on fixture images |

## Built but NOT wired into production
`core/capture_process.py` (process-isolated capture with shared memory and watchdog; 6 tests pass),
`core/face/attribute_policy.py` (`AttributeAggregator`). They are ready for the A/B in the hardware steps below.

## Test results (sandbox, torch stubbed)
core1 125, core2 135, core3 67, web 59, misc 198 (2 deselected) = **584 passed, 0 failed**.
Real-torch tests (`test_attributes`, `test_device`, GPU fallback) and the FastAPI server tests were not runnable there.

## Gates that remain BLOCKED — run on the Windows machine
1. Full suite: `.venv\Scripts\python -m pytest` (baseline was 613 passed, 7 deselected).
2. Real cameras: `python tools/camera_experiments.py --run all --indices 1,2` (E1–E7), then `python tools/camera_soak.py`.
3. Gender accuracy on real data: `python tools/eval_attributes.py`.
4. Ghost gallery: it is quarantined automatically on next app start, or run `python tools/audit_gallery.py --apply`.
5. Whether simultaneous-camera instability is a USB-controller bandwidth limit is unverified; E1–E7 are designed to answer that.

## Licence note
No third-party code was copied. `ultralytics` (AGPL-3.0) is already a dependency — review before distributing the product commercially.

## Update 2 — real Windows results and follow-up changes

**Measured on the real machine (Windows venv):** full suite 710 passed, 7 deselected.

**Camera evidence (reports/camera-ai-fix/):** three cameras are attached: `USB Camera` (idx 0, behind a hub),
`HD WebCam` (idx 1) and a second `USB Camera` with the *same* VID/PID 4C4A:4A55 (idx 2), all on one xHCI
controller. In the 10-minute soak, camera 1 ran at ~30 fps after a 15 s warm-up (5 kills at start); camera 2 never
delivered a frame (0 fps, 56 kills). The E2–E7 results are **not valid**: the soak was still running while they
ran, so every open failed ("opened": false). E1 also held the devices open through its own pygrabber graphs (fixed).
`camera_experiments.py` now refuses to start while another process is using the cameras, and has an `E0` solo test.

**Only male / female / product now.** Child detection was removed end to end (on-screen label and colour,
fused-attribute override, live counts, dashboard age panels, CLI text). Age is never analysed; stored history is untouched.

**Gender accuracy changes (all unit-tested; real accuracy must be measured with `tools/eval_fairface_gender.py`):**
- crops now carry FairFace's 25% margin (the tight box did not match how the model was trained);
- mirror-image test-time augmentation in the FairFace forward pass;
- per-frame floor 0.6 -> 0.70; per-track gate 1.2/0.6 -> 1.6/0.8;
- the per-person smoother is now log-odds evidence (clipped, decaying, hysteresis), needs >= 2 frames and evidence >= 2.0,
  so one confident wrong frame can no longer label or flip a person;
- simulation at 82% per-frame accuracy: wrong label after 1 frame 0%, after 6 frames ~1.3%, after 10 frames ~0.5%.

**Product accuracy:** embedding-matched products must be seen in two passes at an overlapping place before they are
drawn or counted (`core/product_confirm.py`), which removes one-frame false matches on background.

**Still to run on the Windows machine:** `python tools/eval_fairface_gender.py` (gate: person-level error <= 2%),
then `python tools/camera_experiments.py --run E0 --indices 0,1,2` with the app and the soak CLOSED.

## Update 3 - long-range gender from the whole body, and Re-ID / Unique People stability

**Flickering boxes -> new IDs (fixed in `core/tracker.py`, `core/vision.py`).** YOLO now runs at a low confidence
(0.25) for persons; low-score boxes may only *extend* an existing track (ByteTrack-style), never start one. New tracks
are tentative until seen in 2 frames (or one frame with score >= 0.7). A track that misses a frame keeps a predicted
"coasting" box for 0.6 s instead of vanishing, and the track lives 2.5 s, so a blinking detector no longer churns IDs.

**Same person counted twice (fixed in `core/reid.py`, `web/booth_manager.py`).** The visitor registry now:
- stitches a new track to a person who just disappeared on the same camera (gap <= 4 s, similar position and size,
  low appearance floor) - this is what covers the flicker case;
- vetoes matches between people that are on screen in the same pass (two people at once are two people);
- counts a visitor only after >= 2 observations over >= 0.6 s (a one-frame false detection is never a visitor);
- merges duplicate identities later when their appearance is very similar (>= 0.72 confirmed, >= 0.50 provisional) and
  they never coexisted; merges are propagated to the database (`merge_global_person`, merged rows are hidden from
  counts), the booth sessions (`rekey`) and the gender evidence;
- appearance embedding = MobileNet features blended with a clothing-colour descriptor (`core/body_cues.py`).
Counters are in `/api/state` -> `reid` (`stitched`, `merged`, `vetoed`, ...): watch them on site to tune thresholds.

**Gender from face + body + hair + clothing (`core/body_cues.py`, `core/body_gender.py`, `core/attributes.py`).**
When the face is too small to read, hand-crafted whole-body cues (hair length/darkness, exposed skin, clothing colour
bands, silhouette proportions) feed an online logistic model. It is trained on THIS booth: whenever the face model
decides a person confidently, that person's earlier far-away frames are retro-labelled and used for training. The
model's own accuracy is measured prequentially (on people it had not yet seen) and its evidence is weighted by that
measured accuracy, then fused with face evidence as log-odds in the per-person smoother (body-only labels need more
evidence). The dashboard `source` field says whether a label came from face, body or both.

**Honest limits.** The body model starts EMPTY: it needs a number of close-up people before it helps, and hand-crafted
cues are individually weak (long-haired men, unisex clothing). Its real accuracy is unmeasured - it is visible in
`/api/state` -> `body_gender` once it has data. If it is not competent it contributes nothing and the person stays
UNKNOWN rather than guessed. Re-ID thresholds are judgment calls; two very similarly dressed people can be merged
(`GlobalIdentityRegistry(merging=False)` disables merging).

**Verified in the sandbox (torch stubbed):** core1 140, core2 150, core3 75, web 64, misc 198 passed; test_attributes/
test_race_isolation 46 passed + 7 real-torch tests that only run on Windows. New tests: tracker/vision flicker,
`test_reid_identity.py`, `test_body_gender.py`, `test_booth_manager_reid_stability.py`, DB/session merge tests.

**To run on Windows:** `.venv\Scripts\python -m pytest`, `python tools/eval_fairface_gender.py`, then run the app and
watch `/api/state` (`reid`, `body_gender`) for a while with real people.

## Update 4 - unplug / replug and "picture breaks after a fast camera move"

**What was wrong (read from the code, then reproduced with fake cameras).**
- An unplugged camera could keep showing its last picture for 8+ s: a frozen buffer only counted as frozen after
  8 s, a hung `cap.read()` was only noticed after 6 s (+2 s watchdog tick), 20 bad reads were needed to go offline.
- `cap.release()` itself can block forever on a wedged DirectShow graph. It was called from the capture thread AND from
  the shared watchdog thread, so one wedged camera could stop recovery for every camera and never reconnect.
- Every failed open of an absent camera tried all 5 formats (isOpened() is False before any format is chosen, so
  they cannot differ) while holding the process-wide DirectShow lock: 10-15 s of lock per attempt, starving the other
  cameras, and the retry delay then grew to 30 s, so a replugged camera could wait that long.
- The hot-plug scan learned about a replug only by *opening* the device (more open/close churn) on a 5-60 s timer.

**What changed.**
- `core/device_presence.py` (new): polls Windows' camera list every 0.5 s WITHOUT opening any camera. An appearance
  is believed at once, a disappearance after 2 consecutive polls, "could not look" is never treated as "gone".
- `CameraWorker.check_stream_health` (called every 0.5 s by the watchdog): no good frame for 1.0 s (0.4 s for 3 s after
  the device list lost a camera) or a bit-identical picture for 1.5 s (0.5 s) -> the camera is reported offline
  immediately and the tile shows the reconnecting card instead of a frozen/garbled picture. Coming back online
  needs 3 good frames in a row (no flapping alerts). If Windows no longer lists the device the blocked capture is
  released at once.
- Corrupt-frame gate: 12 consecutive bad frames (was 20) reconnect; a frozen picture is dropped after 1.5 s (was 8 s);
  hung-read window 3 s (was 6 s); watchdog tick 0.5 s (was 2 s).
- `_release_capture` runs `cap.release()` on a helper thread and never blocks the caller; a reopen waits for it
  (max 2 s) because the same device cannot be opened twice.
- An unplugged camera is not reopened at all until it re-appears (no futile opens, no lock hogging); the moment it
  re-appears every disconnected worker is told to reconnect NOW and the hot-plug scan wakes up. A camera that came back
  at a different index is followed by its physical identity, and `add_camera` refuses to create a second camera_id for it.
- A camera Windows still lists but that will not open retries every <= 6 s (was up to 30 s).
- Two stream faults within 2 minutes -> that camera reopens with the small low-bandwidth profiles for 5 minutes (fast
  motion enlarges MJPEG frames and a marginal USB cable/hub then tears them).
- `_open_capture`: an index that does not open is tried once per backend, not once per format.
- Hot-plug scan no longer open-probes a known camera when the OS list is available.

**Verified (sandbox, fake cameras through the real `run()` loop, torch stubbed):** unplug -> offline < 1.5 s and no
frame and no open attempt while gone; replug -> online and frames < 2.5 s; frozen picture -> offline -> reconnected;
hung read with a wedged `release()` returns in < 1.5 s; core1 157, core2 151, core3 75, web 71, misc 198 passed.

**NOT verified on real hardware - please test on Windows** (app running, watch the tile):
1. Pull a USB camera: tile should show the reconnecting card in about 1 s. Plug it in: picture back in ~1-3 s.
2. Repeat with the *other* cameras still streaming (they must not blink).
3. Shake / move the camera fast: torn frames are dropped; if it keeps failing, the log says
   "reopening with the low-bandwidth profiles". Look for `[PRESENCE]` and `no good frame` lines in the log.
4. If pygrabber is not installed the presence monitor is inert (workers fall back to the timers above, still faster
   than before). If a camera still cannot reopen after a hard glitch, the cause is below software (driver/USB
   controller: use a powered hub, a shorter cable, a different port; see the E-series experiments).

## Update 5 - far-range gender, real-time boxes, product names

**Why far-range people stayed UNKNOWN (found, not guessed):**
1. Faces smaller than 64 px were thrown away outright, and a person 5-8 m from a 320x240 stream has a face of 15-40 px.
2. Re-ID / attribute work was gated by box height and ran every 0.75 s, so a far person often never got a sample.
3. The whole-body gender model had almost no data: the real `body_gender` npz on this machine held 80 examples and
   its own tests scored 2/6, so it never became "ready" and never contributed.

**Changes**
- Size-aware faces (`core/attributes.py`): faces of 14-64 px are analysed (head-region search on an up-scaled person
  crop with YuNet, cubic up-scale to 224 for FairFace, relaxed blur floor). A small face is *weaker evidence*: it
  needs a higher per-frame confidence (0.80) and its log-odds are divided by a size temperature (1.0 at >=64 px ->
  3.0 at 14 px), so it takes >=3 agreeing frames instead of one. The smoother's evidence gate is unchanged, so the
  "man labelled woman" protection is kept. Unknown people are sampled every 0.35 s, independent of the Re-ID clock.
- `CameraWorker._classify_person` (per-frame box colour) uses the same size policy (`predict_gender_detail`,
  `face_min_confidence`, `attenuate_confidence`) so a tiny unreliable face cannot vote a track's category.
- Body model gets more training data (more people/frames feed it) and an optional CLIP prior
  (`core/clip_gender.py`): set `MONGDEE_CLIP_GENDER=1` (needs `open_clip`/`transformers` weights downloadable on the
  Windows machine). It is only used once its own running accuracy (Wilson lower bound >= 0.55) proves it.
- Real-time boxes (`core/box_motion.py`): boxes are published right after YOLO + tracker (before the slow
  Re-ID/attribute work), and a Lucas-Kanade `BoxFollower` shifts each box every captured frame and catches late AI
  boxes up through a 24-frame ring buffer (expires after 1.5 s without a fresh detection). Sandbox measurement:
  error < 6-8 px vs > 60 px for the stale box. Browser stream is capped at 960 px width.
- Product boxes show the catalog product name (`product_name_resolver`), never the internal id; Thai names are drawn
  with Pillow (`core/text_render.py`, font search + `MONGDEE_LABEL_FONT=<path to .ttf>`), falling back to ASCII.
- Higher resolution helps far range most: `MONGDEE_CAPTURE_SIZE=1280x720` (opt-in; costs USB bandwidth/CPU).

**Honest limits**
- Far-range accuracy is bounded by pixels. Below ~14 px of face, the answer stays UNKNOWN on purpose.
- Verified only in the sandbox (torch stubbed, no real cameras/weights): core1 189, core2 181, core3 75, web 75,
  misc 198 passed; the 7 `tests/core/test_attributes.py` failures are the torch stub, not code.
- NOT verified: real far-range accuracy, Thai font on your PC, CLIP with real weights, on-hardware behaviour.

**Check on Windows:** `.venv\Scripts\python -m pytest`; `python tools/eval_fairface_gender.py --far` (accuracy at
32/24 px faces); watch `/api/state` -> `body_gender` to see the body model become ready.

## Update 6 - one person seen by two cameras = one ID; more reliable "female"

**Why one person became two IDs across cameras (found in `core/reid.py` / `core/body_cues.py`)**
1. Clothing colour was measured in the camera's own colours. Two cameras have different white balance / exposure, so the same
   shirt landed in different hue / brightness bins (synthetic test: same person, two cameras, colour similarity only 0.60).
2. A cross-camera comparison is systematically lower than a same-camera one, and a stranger already known on THIS camera
   out-scored the true owner known only on the OTHER camera; the match floor (0.55) + margin (0.08) then created a new ID.
3. The duplicate-merge floor (0.72) was far above what one person scores against themselves across two cameras.

**Changes**
- `body_cues.color_descriptor` (Re-ID only; the gender body model is untouched): clamped gray-world white balance + exposure
  normalisation + soft hue/brightness binning. Same person across two synthetic cameras: similarity 0.60 -> 0.90, rank-1 0.45 -> 0.92.
- `GlobalIdentityRegistry` is camera-aware: every stored embedding remembers its camera; cross-camera scores re-weight
  towards the (normalised) colour cue; a person seen on the OTHER camera within 1.5 s needs only floor 0.45 / margin 0.06;
  identities only ever seen by different cameras merge at 0.58, or at 0.50 when each is the other's single best match and beats
  every other identity by 0.25 - and any cross-camera merge must hold on 2 consecutive checks. People seen together on one
  camera are still never merged. `stats()` now reports `cross_camera_matches` / `cross_camera_merges`.
  Simulation (10 people x 2 overlapping cameras x 24 runs, real colour code + simulated CNN): duplicated people 20 -> 3 of 60;
  wrongly merged different people 6 -> 8 (the simulated palette has look-alike outfits; the old code merged them too). Honest
  trade-off: people in near-identical outfits can still be merged across cameras. `GlobalIdentityRegistry(cross_camera=False)` restores the old logic.
- Vectorised the similarity maths (a registry with many identities is no slower than before).

**Female / male reliability - what is and is not proven**
The stored `fairface_gender_eval.json` on this machine was produced by a stub model (n=40, identical rows), so it says nothing about
the real model. FairFace validation accuracy is ~95% on clean faces, but a webcam face is 15-60 px, soft, dim and often partly
hidden, where a network typically collapses towards one class. Fixing that needs the real weights, so it is a tool to RUN:
- `core/face_degrade.py` - simulates webcam faces (size 14-160 px, blur, JPEG, noise, exposure, colour cast, hats / masks / shadows,
  crop jitter). `python tools/finetune_fairface_gender.py --dry-run` writes `reports/camera-ai-fix/degrade_preview.png` (no torch needed).
- `python tools/finetune_fairface_gender.py` (GTX 1050: about 25-40 min for the defaults) measures the current model on degraded validation faces
  (accuracy, FEMALE recall, MALE recall, per face size), fits a decision bias (`core/gender_calibration.py`), fine-tunes the gender output on
  degraded `dataset/train` faces, and PROMOTES only a measurable gain (>= +1 pt balanced accuracy, clean accuracy not down > 1.5 pts):
  `models/fairface/webcam_model_state_dict.pt` (auto-preferred by app.py / web_server.py; `MONGDEE_FAIRFACE_ORIGINAL=1` forces the original back)
  and `models/fairface/gender_calibration.json` (bias per face-size bucket, ignored for any other checkpoint). If nothing is better, nothing changes.
  Result: `reports/camera-ai-fix/finetune_gender_report.json`. Age / race outputs of the fine-tuned file are not meaningful (MongDee does not use them).
- Not verified here: the training itself (no real torch/weights in the sandbox); the degradations, calibration maths, checkpoint/calibration loading
  and the metrics code are unit-tested (`tests/core/test_face_degrade.py`, `test_gender_calibration.py`, `test_far_face_attributes.py`).

**Check on Windows:** `.venv\Scripts\python -m pytest`; `.venv\Scripts\python tools\finetune_fairface_gender.py`; watch `/api/state` -> `reid`
(`cross_camera_matches`) with the same person in front of two cameras.

## Update 7 - a woman on CAM-1 and a man on CAM-2 shared one ID (field screenshot) - false merges fixed

**What happened:** Update 6 made cross-camera matching more permissive to stop one person becoming two IDs. In the field it went
the other way: a woman (white top) on CAM-1 and a man (dark jacket) on CAM-2 both showed `P000001` and the man inherited "FEMALE 88%".
The CNN half of the embedding is dominated by the shared office background, so it says "similar" for everybody, and nothing else
vetoed the match. A relaxed floor without a veto was the wrong trade; this update makes a merge require positive evidence and
makes wrong merges self-correcting.

**Changes (`core/reid.py`, `web/booth_manager.py`)**
- Clothing veto: an observation whose (camera-normalised) clothing colours agree < 0.30 with EVERY stored look of an identity can
  never match it, and two identities that disagree that much can never be merged (synthetic data: kills ~85% of different-person
  pairs, wrongly vetoes ~1% of same-person pairs).
- Any match across cameras also needs the clothes to agree >= 0.50 (>= 0.65 for the relaxed "seen on the other camera just now"
  floor, >= 0.60 for a cross-camera merge).
- Gender veto: `resolve(..., gender=)` gets what THIS track's own face says (>= 2 readings of >= 0.85 confidence from a face
  >= 24 px, 3x more than the other gender); an identity already decided as the other gender is never matched or merged.
  The booth manager tells the registry each identity's decided gender (`set_person_gender`).
- Self-correction of an existing wrong mapping: a track whose clothes contradict its identity for 3 samples in a row, or whose
  face says the other gender for 2, is split off, banned from that identity, its crops are removed from it, and the identity's gender
  evidence is reset (`pop_splits`) so it is re-learned from clean faces. `stats()` shows `vetoed` and `split_wrong_identity`.
- Tests reproduce the field bug (embeddings identical apart from clothes / genders): old logic merges, new logic keeps two IDs
  (`tests/core/test_reid_cross_camera.py`, `tests/web/test_booth_manager_two_cameras_two_people.py`).
- Remaining limit: two people in near-identical outfits and of the same gender can still be merged across cameras - nothing in
  the current cues separates them. Faces do, but far/side faces are too small to compare reliably.

**Asian / Thai focus and the booth's own data (gender)**
- `tools/finetune_fairface_gender.py` now draws East/Southeast Asian faces (Thai faces are in this group) 2x as often, reports
  Asian-only metrics, and decides on the mean of overall and Asian-only balanced accuracy.
- Best data is your own cameras: `set MONGDEE_SAVE_FACES=1` before starting MongDee -> every ~2 s a face crop is saved to
  `data/face_labels/unlabeled/` as `<guess>_<conf>_<px>px_<time>.jpg` (opt-in, local disk, capped at 4000 files). Move the crops
  into `data/face_labels/male/` and `data/face_labels/female/` (correct the wrong guesses), then run the fine-tune tool: 80% train (x3 per
  epoch), 20% are held out and reported as "THIS BOOTH's own faces", and a model that gets worse on them is not promoted.
- Not possible from here (no internet from the sandbox): downloading other datasets, cloning other repositories, or running any
  training. FairFace (in `dataset/`) already contains ~27% East/Southeast Asian faces.

**Note:** "AI ประมวลผลไม่ทัน" in the screenshot is the adaptive controller (CPU or AI latency too high with two cameras): it lowers the
AI rate, which also makes boxes lag more. Re-ID / face work runs inside the AI pass. Check the device in the start-up log (CUDA vs
CPU) and `/api/state` performance numbers; a hardware limit, not something this update changes.

## Update 8 - "the man is FEMALE, same ID as the woman, and the booth page says 2 people for one ID" (second field screenshot)

**Root causes found (from the screenshot and the code):**
1. The man on CAM-2 had no readable face, so his own category was UNKNOWN. Re-ID still glued him to the woman's `P000001`
   (measured on the real crops: clothing-colour agreement 0.57, CNN similarity dominated by the shared booth background),
   and `CameraWorker._append_global_ids_to_labels` then *replaced* his label with the identity's fused gender - FEMALE.
   The percentage in the label is the YOLO person confidence, not a gender confidence.
2. The booth page (`people_now`, per-camera counts) summed **tracks**, not people: one ID visible on two cameras = "2 people".

**Fixes**
- `core/face_identity.py` (new): face **identity** with OpenCV `FaceRecognizerSF` (SFace ONNX) + the YuNet landmarks already used.
  Same face (cosine >= 0.40) links tracks/identities across cameras even when clothes differ; a clearly different face
  (< 0.22) is a hard veto, splits a wrongly attached track, and blocks merges. `tools/get_face_models.py` downloads the ONNX
  (needs internet; not bundled). Without the file everything falls back to body cues - `/api/state` shows `face_identity: true/false`.
- `core/reid.py`: body-only cross-camera matching is much stricter (colour minimum 0.72/0.80/0.78, cross-camera colour veto 0.62,
  floors 0.62/0.72, 3 merge checks). New `body_only_cams` / `is_link_trusted()`: an identity that reached a camera **only** by
  a look-alike match is "unproven"; a face proof clears it.
- `core/vision.py`: an unproven link no longer lends its gender to the track (it keeps its own reading or UNKNOWN);
  `web/booth_manager.py` does the same for tripwire/interest snapshots.
- `web/booth_manager.py`: `people_now` and the per-camera snapshot count **distinct Global IDs** (one ID on two cameras = 1).
- Trade-off (intended): without faces, a real person may occasionally get two IDs on two cameras rather than two people
  sharing one. Faces remove most of those duplicates.

**Tests:** `tests/core/test_reid_cross_camera.py` (face join/veto, the 0.57 woman/man case, trust flag), `test_face_identity.py`,
`tests/web/test_booth_manager_distinct_people.py`, overlay trust tests. Sandbox only (torch stubbed, no cameras).

**Run on the Windows machine:** `python tools\get_face_models.py`, restart, then watch `/api/state` -> `reid.face_matches`,
`reid.vetoed`, `reid.split_wrong_identity`. To improve gender further: `MONGDEE_SAVE_FACES=1`, sort the saved faces into
`male/`/`female/`, then `python tools\finetune_fairface_gender.py --extra-faces ...` (East/Southeast Asian faces are up-weighted).

## Update 9 - UNKNOWN was still being counted as a visitor, and could still show an ID on screen

**Context.** This update re-audited the whole identity/gender pipeline from scratch against a fresh, very detailed
requirements pass (same class of requirement as every earlier update: same person across cameras = same Global ID,
different people never share one, UNKNOWN never gets a permanent ID or reuses someone else's, UNKNOWN never counted,
gender never flips on noise). Updates 1-8 already covered most of this correctly (multi-signal cross-camera matching,
gender/clothing/face veto on merge, hard temporal-window reject, camera-scoped local track IDs never crossing cameras,
log-odds gender smoother with hysteresis). Two concrete gaps were still open, found by reading the actual code path
from a detected person to (a) a Global ID being written to the database and (b) a Global ID reaching the on-screen box
label - not by re-deriving the architecture from scratch.

**Gap 1 - UNKNOWN counted as a customer.** `web/booth_manager.py`'s `_update_reid` wrote the `global_persons` DB row
(the row `db.query_unique_people_count` / the Dashboard's "Unique People" total sums) the moment
`core.reid.GlobalIdentityRegistry.is_counted()` said the *identity* was confirmed (2 observations, >= 0.6 s span) -
entirely independent of whether that person's *gender* had been decided yet. A person who stayed UNKNOWN the whole
time they were in frame (turned away, too far, bad light) was still counted as a visitor and fed into the Unique
People total.

**Gap 2 - UNKNOWN could show a confirmed customer's ID.** `BoothManager._make_worker()` wired the on-screen box
label's ID lookup to `GlobalIdentityRegistry.get_global_id_for`, which returns a Global ID the instant Re-ID matches a
track even once - before `is_counted()` confirms anything. A one-blink or still-provisional detection could show
`P000042` on screen exactly as if it were an already-confirmed person.

**Changes**
- `web/booth_manager.py`: new `_maybe_count_customer(global_id, now)`, called from both `_update_reid` (the normal
  Re-ID-sampled path) and `_analyze_known_person` (the faster "still UNKNOWN, analyse urgently" path) right after
  `_update_attributes` runs. It writes the `global_persons` row only when `is_counted()` is true **and** - only when
  this booth actually runs gender classification (`self.attribute_backend is not None`) - the person's
  `GlobalPersonAttributeSmoother` status is `"ok"`. A booth with no gender backend configured has nothing to wait for,
  so it keeps counting by identity confirmation alone, exactly as before this gate existed (kept true by a dedicated
  test). The transition from UNKNOWN to counted is atomic and keeps the same `global_id` - no re-identification, no
  new row, no double count, matching the earlier "distinct Global IDs" counting fix from Update 8.
- `core/reid.py`: new `GlobalIdentityRegistry.get_confirmed_global_id_for(camera_id, local_track_id)` - like
  `get_global_id_for`, but returns `None` until that identity's own `is_counted()` is true. `get_global_id_for` itself
  is unchanged and still used internally by the matcher/merge bookkeeping (`resolve()`, `is_link_trusted()`, the
  interest-event snapshot in `_attribute_snapshot_for`) - only the *externally visible* accessor (the on-screen label)
  was switched to the confirmed-only one, in `BoothManager._make_worker()`'s `global_id_resolver=` wiring.
- `core/vision.py`: doc-comment update only, describing the new confirmed-only wiring.

**Tests:** `tests/web/test_booth_manager_customer_counting.py` (new - UNKNOWN with a gender backend stays uncounted
even once identity-confirmed; becomes counted the moment gender reaches `"ok"`, under the same `global_id`; a booth
with no gender backend counts by identity confirmation alone, unchanged), `tests/web/test_booth_manager_global_id_wiring.py`
(updated - asserts the worker is wired to `get_confirmed_global_id_for`, and that a mapped-but-not-yet-confirmed track
resolves to `None` on screen until `person.confirmed` flips), `tests/web/test_booth_manager_presence.py` (fake registry
stub renamed to match).

**Full suite, clean run on this machine (Windows, real torch/opencv, no sandbox stub):** 910 passed, 1 failed, 7
deselected. The 1 failure (`tests/core/test_camera_open.py::test_multiple_cameras_configured_before_first_read`) passes
in isolation - real DirectShow / OBS Virtual Camera enumeration state on this dev machine leaking across the full-suite
run, not caused by this update (same failure present in the untouched baseline before any of this update's changes).

**Honest limits / not done here**
- No real USB camera was exercised this session. Nobody watched a physical Camera 1 / Camera 2 pair, so cross-camera
  same-person-one-ID, different-people-different-IDs, and gender-stability-over-time are unverified on real hardware -
  only unit/integration-tested against synthetic tracks and fake backends, same caveat as every earlier update's
  "sandbox only" note.
- `vision/` + `backend/` (the separate "Remote Camera AI Server" subsystem - not wired into `app.py`, `web_server.py`,
  or the `.spec` build, so not the shipped Booth OS) has its own, weaker `GlobalPerson._merge_attributes`: it replaces
  an attribute whenever the new reading has higher confidence, with no temporal hysteresis - a real gender-flip risk if
  that subsystem is ever wired up as a product. Left untouched: out of scope for the deployed app, and rewriting a
  separately-tested (206 tests) module was not asked for.
- Child/age classification was deliberately removed in an earlier update ("only male / female / product now" - a
  recorded product decision, not an oversight). It was not brought back here even though a fresh requirements pass
  mentioned preventing Male/Female/Child flipping - re-adding a whole category is a product decision for a human to
  make, not something to silently re-introduce or silently ignore.
- The pre-existing "two near-identically dressed people of the same gender, no readable face on either" limitation
  from Update 7/8 still stands: nothing in the current cues can separate them across cameras.

**To check on Windows:** `.venv\Scripts\python -m pytest`; then run the app with real cameras and watch `/api/state` /
the Dashboard's Unique People total while someone stands with their back to a camera the whole visit (should never
appear in the total) versus someone whose face is read normally (should appear once, promptly, after ~2 confident
frames).
**Not done here:** no model could be trained or downloaded in the sandbox (GitHub/HuggingFace blocked, no GPU/torch).

## Update 10 - two different people who walk in together could share one ID (this was likely THE "different person, same ID" bug)

**Reported:** "why different person have same ID" - no screenshot this time, so this update was found by reading the
actual matching code path end to end, not by reproducing a specific field report.

**Root cause, confirmed by a standalone repro script before touching any code.** `GlobalIdentityRegistry`'s "two
people on screen together are never the same identity" guard (`observe_frame()`, checked via `_on_screen_now()` inside
`_resolve_locked`'s matching loop) only knows about a local track once it already has a `global_id` mapping.
`web/booth_manager.py`'s `_on_person_tracks` calls `observe_frame()` exactly ONCE per AI pass, right at the top,
BEFORE `_update_reid` resolves any of that pass's tracks. So for two people who are BOTH brand new - first time ever
seen, e.g. two people who walked in together, or two people who both happen to clear `ReIDSampler`'s throttle in the
same pass - `observe_frame()` sees neither of them (neither has a mapping yet), and the on-screen set it records is
missing both. When the first track resolves and mints a new Global ID, and the second track resolves moments later
in the exact same pass, the matching loop's on-screen exclusion still doesn't know the first one exists (that
snapshot predates it) - so if the second person's appearance embedding happens to clear the match floor/margin
against the first person's brand-new identity (plausible without looking alike at all - same-colour shirts, similar
build, is enough with the CNN half of the embedding), they get resolved onto the SAME Global ID immediately. Confirmed
with a synthetic repro (`core/reid.py`, cosine 0.9 between two different people's embeddings, same pass, same
camera): before the fix, `a == b`; not an edge case, no unusual clothing/lighting condition needed.

**Fix (`core/reid.py`).** New `GlobalIdentityRegistry._mark_on_screen(camera_id, global_id, ts)`, called from
`resolve()` itself right after each track's final `global_id` is known - so by the time the SECOND track in a pass
is matched, the first one (even though it was only just minted moments earlier, in this same pass) is already
excluded, and their `coexisted` sets are updated too (so a later merge pass, which separately checks `coexisted`,
also can't fold them back into one person). This needed its own staleness window rather than reusing `observe_frame`'s
`VISIBLE_HOLD_SEC` (1.0 s): `resolve()` is also the path used to RE-match the same physical person onto a brand new
local track after their old one was evicted (see the existing
`test_forgotten_local_track_can_rematch_same_global_person`) - reusing the 1.0 s window there caused a real
regression (a re-match exactly 1.0 s after the original sighting was wrongly blocked, because `_mark_on_screen`'s own
mark from the earlier sighting hadn't expired yet). Fixed by keeping `_mark_on_screen`'s marks in a separate store
(`_pass_marks`, new) with its own much shorter `SAME_PASS_WINDOW_SEC` (0.25 s - comfortably longer than the real,
sub-millisecond gap between `observe_frame()`'s own `time.time()` call and `_update_reid`'s separate one for the
same pass, since all of one pass's `resolve()` calls share literally the same `now` float; comfortably shorter than
any genuine re-identification gap). `_on_screen_now()` now unions `_visible` (observe_frame's slower-expiring
snapshot) and `_pass_marks` (the new same-pass marks), each read with its own window, so neither job's timescale
leaks into the other's.

**Tests:** `tests/core/test_reid_identity.py::test_two_people_who_walk_in_together_never_share_an_id` (new -
reproduces the exact bug: two brand-new tracks resolved in one pass, asserts different IDs and that `coexisted` was
recorded both ways). Full suite re-run clean after the fix (including the regression this surfaced and then fixed
along the way, `test_forgotten_local_track_can_rematch_same_global_person`): see the run this update's session
reported at the time.

**Honest limits.** Only reproduced and fixed in a synthetic unit test - not confirmed this was actually the specific
cause of whatever the live report was describing, since no screenshot/log was given this time. If the same class of
report recurs, the next most useful thing to capture is `/api/state -> reid` (`vetoed`, `split_wrong_identity`,
`stitched`) at the moment it's observed, and ideally which two local track IDs / cameras were involved, the same way
Update 7 and 8's field screenshots made those bugs reproducible instead of theoretical.

## Update 11 - face boxes blinked; "same person" now needs every cue; one gender per person, never UNKNOWN

**Read first:** `docs/IDENTITY_GENDER_CHECKPOINT.md`. Reported: the FACE box blinks / is not continuous; identity must be decided
from face + clothes + colour + hairstyle across all cameras (same ID only when confirmed, never when it cannot be confirmed);
the same person must never be drawn as two genders on two cameras, and no box may say UNKNOWN.

**Why the FACE box blinked (two causes, both in `core/vision.py` / `core/face/service.py`)**
1. Every AI pass first publishes the person boxes ("early publish", before the slow Re-ID / face part) and only at the END of
   the pass publishes them again with the face boxes. During the slow part the screen therefore had NO face boxes.
2. The detector's boxes were drawn raw: one pass where YuNet missed the face (turned head, blur, small face) made it vanish.

**Fixes**
- `FaceStabilizer` (`core/face/service.py`): a detection continues the box of the same person track (or the overlapping box),
  blends with the old box (no jitter), and a miss keeps the box for `FACE_HOLD_SEC` (1.6 s) while it moves/scales with the
  person's own box. A missed face also gets a second, lower-threshold search in its neighbourhood
  (`YuNetFaceDetector.detect_in_region`, max 3 per pass). The steady boxes are added to the early publish too.
- Identity ("is this the same person?", `core/reid.py`): without a face proof, a link between two cameras now needs torso colour
  AND trousers colour AND hairstyle (`core.body_cues.hair_descriptor`: head darkness/width, hair beside/below the neck, hair
  colour) to agree (`CROSS_PART_MIN` 0.72, `CROSS_HAIR_MIN` 0.55); clearly different torso/trousers (< 0.55) or hairstyle
  (< 0.35) is a veto; merges use the same rule. A face embedding (SFace) still overrides everything (same face = same person,
  different face = different people). If the cues cannot confirm, the person gets a separate ID.
- One gender per person (`BoothManager._resolved_gender_for_track` -> `CameraWorker.gender_resolver`): the box label, the
  coasting box and the booth-page counts all use the person's Global-ID gender (decided label, else the sticky best guess from
  `GlobalPersonAttributeSmoother.guess`, else the track's own face reading). Two tracks with the same ID always show the same
  gender. The labels are re-applied after Re-ID each pass, so they are not one pass stale.
- No "UNKNOWN" box: a person with ANY evidence gets a best guess; a person with none at all yet (first ~second, or turned away
  with no body model) is drawn as `PERSON` (neutral) - not a gender, and not counted as one.
- Removed the Update-8 "unproven link" flag (`body_only_cams` / `is_link_trusted`): it let one ID show two genders, which is what
  this update forbids; it is replaced by the stricter link rule above.

**Tests:** `tests/core/test_face_stability.py` (8), `test_gender_guess.py` (4), new cases in `test_reid_cross_camera.py`,
`test_camera_worker_global_id_overlay.py`, `tests/web/test_booth_manager_distinct_people.py`. Sandbox only (torch stubbed).

**Honest limits**
- A wrong link that passes all cues (two look-alikes, no faces) still shows one gender for both until a face reading contradicts
  it (then the track is split off after 2 samples). A best guess from weak evidence can be wrong; it flips only after real
  opposite evidence. Hairstyle from a 320x240 crop is a coarse cue - it is used as a veto/consistency check, never as proof.
- Nothing was run against real cameras here. The hair/colour thresholds are judgment calls: watch `/api/state -> reid`
  (`vetoed`, `face_matches`, `split_wrong_identity`); if the same person keeps getting two IDs, lower `CROSS_PART_MIN`.
