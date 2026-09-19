# Identity / Gender pipeline — checkpoint (read this first, before re-auditing anything)

Purpose of this file: orient a future session on the person-identity / gender-classification
pipeline in under a minute, without re-reading `core/reid.py`, `core/attributes.py`,
`core/vision.py`, `web/booth_manager.py` and `core/database.py` from scratch. It states what is
already correct (trust it, don't re-derive it), what is still open, and exactly where to look for
more detail. The full chronological history (10 rounds of fixes, each with root cause + evidence)
is `docs/CAMERA_AI_FIX_REPORT.md` — this file is the current-state summary, that file is the log.

Last updated: branch `fix/camera-ai-pack`, after `docs/CAMERA_AI_FIX_REPORT.md` Update 11.

## Which system is live

Two person-tracking implementations exist in this repo. **Only one is shipped:**

- **`core/` + `web/booth_manager.py` + `app.py` / `web_server.py`** — the deployed MongDee "AI
  Booth OS" (desktop PySide6 app and/or browser UI). This is what the `.spec` build packages and
  what `GUIDE.md` documents. **This is the one to fix when a person/gender bug is reported.**
- **`vision/` + `backend/`** — a separate "Remote Camera AI Server" system (`docs/architecture.md`
  describes it), not wired into `app.py`, `web_server.py`, or the `.spec` build. It has its own,
  simpler `vision/identity/` module with a weaker gender-merge policy (see "Known gaps" below).
  Don't spend time here unless the task specifically says this subsystem, or it gets wired into a
  real build.

## File map (core/ system)

| File | Owns |
|---|---|
| `core/reid.py` | `GlobalIdentityRegistry` — local track → Global Person ID matching, merging, collision/coexistence protection. The core of "same person = same ID, different people ≠ same ID." |
| `core/attributes.py` | `GlobalPersonAttributeSmoother` — per-Global-Person gender evidence accumulation (log-odds, decay, hysteresis), `AttributeSampler` (throttle), FairFace/YuNet backends. The core of "gender doesn't flip." |
| `core/face_identity.py` | Face embedding (SFace) — strongest single signal for cross-camera identity linking/veto. |
| `core/body_cues.py`, `core/body_gender.py`, `core/clip_gender.py` | Whole-body appearance cues (hair/clothing/build) as gender evidence when no readable face — feeds the same smoother in `core/attributes.py`, weighted by measured accuracy. |
| `core/tracker.py`, `core/tracker_v2.py` | Local, camera-scoped, ephemeral track IDs. Never cross a camera boundary; `core/reid.py` is the only thing that bridges them into a Global ID. |
| `core/camera_identity.py` | **Not person identity** — physical *camera device* identity (stable name vs. USB re-enumeration index). Unrelated module, don't confuse it with `core/reid.py`. |
| `web/booth_manager.py` | Wires everything together per booth: owns the one shared `reid_registry` / `attribute_smoother`, decides when a person is written to the DB as a counted customer (`_maybe_count_customer`), decides what a camera's on-screen box label shows. |
| `core/database.py` | `global_persons` table (`upsert_global_person`, `query_unique_people_count` = the Dashboard "Unique People" total), `person_attributes` table (gender/age summary per Global Person), `merge_global_person` (folds a duplicate ID into its survivor, hides it from counts). |

## Invariants already correct — trust these, don't re-derive

1. **Local track ID ≠ Global Person ID**, always. `core/tracker.py` IDs are camera-scoped and
   ephemeral; `core/reid.py`'s `(camera_id, local_track_id) -> global_id` map is the only bridge.
2. **Matching uses multiple signals, never one alone**: CNN appearance + camera-normalised
   clothing colour + face embedding (when available) + body aspect ratio + a hard
   temporal-transition-window reject (impossible walk time between cameras = automatic no-match,
   regardless of appearance) + "two people visible in the same AI pass are never the same
   identity" (see Update 10 for the one real gap found here, now fixed).
3. **Gender is a veto, never a positive identity key.** Two different people of the same gender
   are not treated as more likely to be the same person. An identity already decided as one gender
   can never match/merge with an observation confidently read as the other.
4. **Gender is stable once decided.** `GlobalPersonAttributeSmoother` accumulates signed log-odds
   per Global Person (decay 0.95/sample), requires ≥2 face samples and evidence ≥2.0 before showing
   MALE/FEMALE at all, and an established label needs the opposite side to build real evidence past
   a keep-fraction dead zone before it can change — a single noisy/wrong frame cannot flip it.
5. **UNKNOWN is not a counted customer.** `web/booth_manager.py::_maybe_count_customer` only writes
   the `global_persons` row (the thing `query_unique_people_count` / the Dashboard total sums) once
   the identity itself is confirmed (≥2 observations, ≥0.6 s span — not a one-blink ghost) **and**,
   when this booth runs gender classification at all, gender has reached the smoother's `"ok"`
   status. A booth with no gender backend configured falls back to identity-confirmation-only
   counting (nothing to gate on) — this fallback is locked in by test.
6. **UNKNOWN never shows a confirmed customer's ID on screen.** The box-label resolver
   (`BoothManager._make_worker`'s `global_id_resolver=`) uses
   `GlobalIdentityRegistry.get_confirmed_global_id_for`, which returns `None` until that identity's
   own confirmation gate passes — separate from `get_global_id_for`, which the matcher still uses
   internally for its own bookkeeping (a track can be tracked under a provisional ID without ever
   displaying one).
7. **Collision protection on merges**: two identities are only folded into one when they clearly
   look alike over sustained evidence (not a single lucky frame), were never seen on screen
   together (`coexisted`, see Update 10 for the one gap found in how this was populated), and don't
   disagree on decided gender / face / clothing. Ambiguous cases are left as two separate,
   unconfirmed identities rather than force-merged — "prefer unconfirmed over wrongly merged" is a
   real, enforced property, not just a stated goal.
8. **Race/protected characteristics are never used.** FairFace's race output is sliced off and
   discarded before it reaches `core/attributes.py` at all (see that module's own docstring, and
   `tests/core/test_race_isolation.py`).

9. **One gender per person, never UNKNOWN on a box** (Update 11). `BoothManager._resolved_gender_for_track` decides the gender
   drawn for a track from its Global Person (decided label, else `GlobalPersonAttributeSmoother.guess`, else the track's own face);
   the same Global ID shows the same gender on every camera. Nobody-with-evidence-yet is `PERSON`, not UNKNOWN.
10. **A cross-camera link with no face proof needs torso colour + trousers colour + hairstyle to all agree**
    (`CROSS_PART_MIN`, `CROSS_HAIR_MIN` in `core/reid.py`); otherwise the person gets a separate ID. A face embedding overrides.
11. **The FACE box is steady**: `FaceStabilizer` holds a face 1.6 s through detector misses and the steady boxes are part of the
    early publish (`CameraWorker._draw_stable_faces`).

## Known gaps / honest limits (check here first before assuming something new is broken)

- Two people in **near-identical outfits, same gender, no readable face on either**, can still be
  merged across cameras — nothing in the current cues separates them (documented since Update 7/8).
- The separate `vision/`+`backend/` subsystem's `GlobalPerson._merge_attributes` has no temporal
  hysteresis (replace-on-higher-confidence) — a real gender-flip risk *if that subsystem is ever
  wired up as a shipped product*. Not fixed, because it isn't live (see "Which system is live").
- Child/age classification was deliberately removed in an earlier round ("only male / female /
  product now" — a recorded product decision). Don't silently re-add it; that's a product call.
- No real USB camera has been exercised by any of these sessions (sandbox/CI only, real torch on
  this Windows machine but no physical camera loop). Cross-camera same-person-one-ID and
  gender-stability-over-time claims are unit/integration-tested, not hardware-verified.

## How to verify quickly, without a full re-audit

```
.venv\Scripts\python -m pytest -q tests/core/test_reid.py tests/core/test_reid_identity.py \
    tests/core/test_reid_cross_camera.py tests/core/test_attributes.py tests/web/
```
This is the fast, targeted subset covering everything in "Invariants already correct" above.
Full suite (`pytest -q`, ~5-6 min on this machine) should be **910 passed, 0 real failures** —
`tests/core/test_camera_open.py` has two tests that are order-dependent on this dev machine's real
DirectShow/OBS-virtual-camera enumeration state (pass clean in isolation); that's environment
flakiness, not a code bug, and has been present since before any of these fix rounds.

If a NEW "different person / same ID" or "gender flipped" report comes in: the single most useful
thing to capture at the moment it happens is `/api/state -> reid` (`vetoed`, `split_wrong_identity`,
`stitched`, `cross_camera_matches`) plus which local track IDs / cameras were involved — that's
what made Update 7, 8 and 10's bugs reproducible instead of theoretical. Write the finding as the
next `## Update N` section in `docs/CAMERA_AI_FIX_REPORT.md` (root cause, exact code change,
tests, honest limits — follow the existing updates' shape), then come back and revise the relevant
line in *this* file if an invariant above turned out to be wrong.
