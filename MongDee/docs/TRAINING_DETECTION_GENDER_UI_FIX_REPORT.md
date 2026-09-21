# Training, Long-Range Detection, Gender Signal, Box Stability & Camera-Mixup Fix

**Branch:** `main` · **Base commit:** `e374369` · **Status:** all changes are in the working tree, uncommitted (per your default workflow — nothing was committed or pushed).

## 1. Product training interface — capture until comprehensive

**Before:** the web trainer already supported still-image upload, video upload, and a guided 360° "record from camera" flow — but that flow ran for a **fixed 15-second timer** (30 ticks), then batch-uploaded everything at once. Whether the result was actually good training data (enough distinct angles, not 30 near-duplicates of the same three seconds) was only known *after* the fact.

**What changed:**
- New `core.training.LiveTrainingSession` — feeds one captured frame at a time through the exact same quality/duplicate gate `import_images`/`import_video` already used (blur, exposure, near-duplicate embedding check), adding each accepted frame straight to the product's gallery and reporting a running **distinct-views** count back immediately.
- New endpoints: `POST /api/products/{key}/training_session/{start,frame,finish}`.
- `web/static/trainer.js`'s recording loop no longer runs for a fixed time. It now keeps capturing — showing a live "N/15 distinct views collected" indicator — **until it actually reaches the recommended coverage target** (`RECOMMENDED_SAMPLES`, currently 15), with a 45-second safety cap for a product that genuinely never rotates into new views. The progress ring now reflects real coverage, not elapsed time.
- Video/image upload paths are unchanged (they were already solid).

**Tests:** 6 new tests for `LiveTrainingSession` in `tests/core/test_training.py` (running count, duplicate rejection, quality-reject reasons, the "nothing usable" error, incremental gallery writes).

## 2. Long-range detection — no more flat-confidence guessing

**Before:** both product-detection paths (YOLO COCO classes, and the custom embedding-matched gallery) used one **flat confidence floor** regardless of how small/far the detected box was — a tiny, low-detail box at range cleared the same bar as a close, sharp one.

**What changed:** mirrored the exact pattern already used for far-away faces (`core/attributes.py`'s `face_min_confidence`/`attenuate_confidence`) onto products, in `core/product_confirm.py`:
- `product_min_confidence(box_px, base_floor)` — raises the required confidence as the box shrinks (full trust ≥80px, up to a hard 0.75 floor at 24px).
- `product_confirm_hits_for(box_px, base_hits)` — a small/far box needs more agreeing frames (up to +2) before being reported, not just a higher score.
- Wired into `CameraWorker._run_yolo` (COCO product classes) and `_run_custom_recognition` (embedding matches, via `recognizer.identify(crop, floor=...)`).

**Tests:** 3 new unit tests for the scaling functions, 2 new `CameraWorker`-level tests proving a small box needs both a stricter floor and more confirmation passes.

## 3. Gender classification — hairstyle as a *learned* signal, not a rule

You explicitly asked that hairstyle **never** directly decide MALE/FEMALE, never override accumulated evidence, and that the system fall back to UNKNOWN when evidence is thin — and that nothing be hardcoded from appearance stereotypes.

**What I found:** MongDee's gender pipeline (`core/attributes.py` + `core/body_gender.py`) already does exactly this by design — it's a self-calibrating online model that only trusts a body-appearance signal once it has *measured*, on this booth's own labelled people, that it beats chance (`MIN_ACCURACY=0.62`); it already fuses face + body evidence in log-odds space with decay and hysteresis, requires multiple samples before deciding, and reports UNKNOWN until evidence clears a threshold. Hair length was already a weak, unlabelled input to that model, but only through a coarse two-value proxy.

**What changed:** `core/body_cues.py`'s `body_features()` now also folds in the richer, illumination-normalised `hair_descriptor()` (6 values: head darkness, head/hair width, hair beside/below the neck, hair colour saturation/brightness) as additional inputs to the *same* learned, competence-gated model — nothing about the decision logic, evidence thresholds, temporal smoothing, or UNKNOWN fallback in `core/attributes.py` was touched. Hair still can't decide anything by itself; it's six more numbers the model learns a weight for, exactly like clothing colour or build already were.

**Tests:** 3 new tests confirming (a) the hair descriptor is actually present in the feature vector, (b) long vs. short hair produce measurably different features, (c) an extreme hair reading alone still cannot make the model answer before it has met its own accuracy bar.

## 4. Box tracking stability — no flicker on a momentary miss

Person boxes already had a "coasting" mechanism (`core.tracker.PersonTracker.coasting_tracks`) that keeps showing a track's predicted box through a brief detector miss. **Product boxes had no equivalent** — one AI pass where an embedding match dipped a hair below the floor (motion blur, a hand briefly crossing it, a lighting flicker) made the box vanish and reappear a moment later.

**What changed:** `core/product_confirm.py`'s `ProductConfirmer` now remembers each product's last confirmed box *and score*; `_run_custom_recognition` shows that last box (with its real score, not a placeholder) for up to 0.6s (matching the person tracker's own coast window) if the product isn't re-matched that pass. Static, not motion-predicted (products don't have a tracker) — a genuinely-moved product will lag briefly before the next real match corrects it, which is preferable to blinking off.

**Tests:** 4 new `ProductConfirmer` unit tests + 1 `CameraWorker`-level integration test proving a confirmed product survives a pass where the proposer finds nothing at all, with its real last confidence carried through (not 0%, which would otherwise have corrupted `core/aggregator.py`'s per-camera confidence tracking).

## 5. Camera-feed mixup ("CAM-1 window shows CAM-2")

I traced the entire pipeline (device→worker mapping, per-camera JPEG encode/storage, the `/stream/{camera_id}` endpoint, and all three frontend pages that display camera video) and found every `camera_id` association structurally correct — no closure bugs, no shared mutable state, no dict-key crossover.

**What I did find:** `web/static/booth.js`'s multi-camera viewer grid already carries a code comment describing a *real, previously-observed* browser-level race — several `<img src="/stream/...">` connections opened in the exact same JS tick can have their first frames corrupted or cross-delivered by the browser (documented there as "a broken-image icon on a random panel, not always the same camera") — and was already mitigated there with a staggered connection start. **That same mitigation was missing from two other pages that show camera video:**
- `web/static/product_view.js` — rebuilt *all* camera panels via one `innerHTML` write with `src` set directly in the HTML string, opening every stream connection simultaneously. Now uses the same staggered, cache-busted connect as `booth.js`.
- `web/templates/camera_view.html` (the single-camera popout window) — had a static `src="/stream/{{camera_id}}"`. Opening several popouts in quick succession could race the same way. Now connects from JS with a unique cache-busting suffix, closing off a second, related risk: two requests for the literal same `/stream/{id}` string (no unique suffix) being coalesced by a browser/proxy into one shared connection. `booth.js`'s own initial connection was missing that same cache-busting suffix (present on every *reconnect* already) — added for consistency.

**Honesty note:** I could not reproduce this bug live (this dev machine has only one confirmed working physical camera), so I can't certify the exact race is now impossible — only that I found and closed the one concrete, well-precedented mechanism this codebase's own comments already document as a real cause of cross-camera display bugs, consistently across all three pages that previously had it in only one.

## Tests

Full suite: **1015 passed**, 1 failed (`test_multiple_cameras_configured_before_first_read`) — confirmed via `git stash` against the unmodified base commit to be a pre-existing environment flake (an OBS Virtual Camera occupying a device index on this dev machine, unrelated to any change here).

## What wasn't (and couldn't be) verified

- The training/detection/gender changes were verified with unit and integration tests using synthetic data, not a real multi-hour training session against physical product hardware.
- The camera-mixup fix addresses the one concrete mechanism found in code; a live multi-camera browser session would be needed to fully confirm the original symptom is gone.
- No frontend automated tests exist in this repo for `booth.js`/`product_view.js`/`camera_view.js`; the JS changes were verified by rendering the pages via a live server (200 OK, correct injected constants) and manual code review, not a browser test suite.

## Files changed

`core/body_cues.py`, `core/product_confirm.py`, `core/training.py`, `core/vision.py`, `web/booth_manager.py`, `web/server.py`, `web/static/{booth,camera_view,product_view,trainer}.js`, `web/templates/camera_view.html`, plus new/updated tests in `tests/core/{test_training,test_product_confirm,test_vision,test_body_gender}.py`.
