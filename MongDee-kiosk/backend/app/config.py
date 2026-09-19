"""Central configuration for MONGDEE MINI KIOSK backend.

Every threshold here is a starting point, not a calibrated value — the
camera angle, lighting, and turntable speed are unique to the physical
booth setup and were not available while building this, so these MUST be
re-tuned on-site with the real hardware before the demo. See README.md
"On-site calibration" for the procedure.
"""
import os
from pathlib import Path

# --- Paths -------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
DB_PATH = DATA_DIR / "mongdee.db"
CAPTURES_DIR = DATA_DIR / "captures"
REFERENCE_FRAME_PATH = DATA_DIR / "reference_frame.jpg"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# --- Camera --------------------------------------------------------------
# KIOSK_FAKE_CAMERA=1 swaps the webcam for a simulated one (app/fake_camera.py);
# "auto" also cycles empty/product scenes by itself. Uses its own camera index
# so it never reads or overwrites a real camera's empty-platform reference.
_FAKE = os.environ.get("KIOSK_FAKE_CAMERA", "").strip().lower()
FAKE_CAMERA = _FAKE in ("1", "true", "yes", "auto")
FAKE_CAMERA_AUTO = _FAKE == "auto"
FAKE_CAMERA_INDEX = 99
FAKE_CAMERA_DIR = DATA_DIR / "fake_camera"
CAMERA_INDEX = FAKE_CAMERA_INDEX if FAKE_CAMERA else 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
JPEG_QUALITY = 80

# Region of interest (as fractions of frame width/height: x, y, w, h) where
# the turntable sits in the frame. Default assumes the platform occupies the
# middle ~60% of the frame. Narrowing this to just the platform area makes
# presence detection far more reliable — tune with the real camera angle.
ROI_FRACTION = (0.2, 0.15, 0.6, 0.75)

# --- Presence detection ---------------------------------------------------
# Every live frame's ROI is compared against ONE fixed "empty platform"
# reference snapshot (see camera.py) — not an adaptive background model.
# An adaptive model would slowly relearn a motionless object as background
# the longer it sits there, silently breaking detection; a fixed reference
# only ever changes when explicitly recaptured via /calibrate.
DIFF_PIXEL_THRESHOLD = 25  # per-pixel grayscale difference (0-255) to count as "changed"
# Live frames are averaged over this many recent reads before diffing, to
# cancel per-frame sensor noise (significant in low light) that would
# otherwise swing the ratio even with nothing moving. Higher = steadier
# reading but slower to react to a real change.
NOISE_SMOOTHING_FRAMES = 6

# Fraction of ROI pixels flagged as changed to call a product "present".
PRESENCE_ON_RATIO = 0.12
PRESENCE_OFF_RATIO = 0.04
# Consecutive frames required at ~10fps sampling before flipping state,
# to debounce flicker/shadows from the rotating platform. Kept deliberately
# high so a hand reaching in, a passing shadow, or one noisy frame can't
# flip presence (and thus reset recognition) on its own.
PRESENCE_ON_FRAMES = 10
PRESENCE_OFF_FRAMES = 12

# --- Capture burst (enrollment + recognition) -----------------------------
# The turntable can't be triggered, so instead of "spin then shoot" we just
# passively sample frames for a fixed window once a product is detected —
# long enough that its continuous slow rotation carries multiple angles
# past the camera.
BURST_DURATION_S = 9.0  # covers more than one full turntable rotation -> more distinct angles
BURST_SAMPLE_INTERVAL_S = 0.5  # ~18 frames per burst

# --- Enrollment ------------------------------------------------------------
# Staff review every captured frame and pick which ones are usable (in
# focus, well-framed) before a product is saved — a burst captures more
# frames than this so there's something to choose from.
ENROLL_MIN_IMAGES = 3
ENROLL_TARGET_IMAGES = 5

# --- Background masking -----------------------------------------------------
# Before embedding a frame, the turntable/booth background is removed using
# the same fixed "empty platform" reference that presence detection uses, so
# the embedding describes the product rather than the booth. Applied
# identically at enrollment and at recognition time. Bumping EMBED_VERSION
# invalidates older embeddings (they are excluded from matching and their
# products are flagged for a one-time re-scan) — never reuse an old number.
EMBED_VERSION = 2
# Per-pixel grayscale difference (live vs reference, after exposure
# compensation) above which a pixel is considered "product, not background".
MASK_DIFF_THRESHOLD = 30
# If the isolated product covers less than this fraction of the ROI the mask
# is treated as unreliable and the plain (unmasked) crop is used instead —
# the same deterministic fallback at enrollment and recognition.
MASK_MIN_COVERAGE = 0.02

# --- Feature weighting ------------------------------------------------------
# Each embedding concatenates CNN shape/texture features with an HSV color
# histogram (see vision.py) before normalizing, so relative weight here
# controls how much color vs. shape drives the final similarity score.
# Raise EMBED_COLOR_WEIGHT if same-shaped products differing mainly by an
# accent color (flavor variants, etc.) keep getting confused for each other.
EMBED_SHAPE_WEIGHT = 0.6
EMBED_COLOR_WEIGHT = 0.4

# --- Recognition -----------------------------------------------------------
# Cosine similarity (0..1) above which a match is shown with confidence;
# below this, the kiosk treats the product as unrecognized rather than
# guessing. CNN embeddings of the same object from a different angle/
# lighting rarely approach 1.0 — expect real matches to land ~0.55-0.85.
# MUST be tuned against real enrolled products.
MATCH_CONFIDENCE_THRESHOLD = 0.62
TOP_K_CANDIDATES = 3
# If the top two candidates are within this margin, the match is too close
# to call, including when the runner-up is just below the threshold.
# automatically — a person picks instead of the system guessing.
AMBIGUOUS_MARGIN = 0.03

# Three fresh views per verification; several failed verifications are needed
# before a new full burst. Human-confirmed choices last until removal.
REVERIFY_INTERVAL_S = 4.0

# Stability guards. Similarity is a score, not a calibrated probability.
FRAME_MAX_AGE_S = 2.0  # tolerate a brief camera hiccup without blanking a shown result
CAMERA_RECONNECT_INTERVAL_S = 2.0
CAMERA_WARMUP_S = 1.5
PRESENCE_PROCESS_WIDTH = 320
MAX_EXPOSURE_OFFSET = 40.0
MIN_COMPONENT_RATIO = 0.001
PRESENCE_ON_HOLD_S = 0.6
PRESENCE_OFF_HOLD_S = 0.6
CAPTURE_SETTLE_S = 0.7
MIN_SHARPNESS = 8.0
MAX_CAPTURE_MOTION = 0.30
RECOGNITION_MIN_FRAMES = 5
MATCH_MIN_AGREEMENT = 0.70
MATCH_MIN_SUPPORT = 0.65
REVERIFY_FAILURES = 3
SCAN_RETRY_INTERVAL_S = 2.0
SCAN_MAX_RETRIES = 3
# Consecutive independent scans that must agree before the kiosk *changes*
# what it is already showing (switches to a different product, or drops a
# good match to unknown/ambiguous). First acquisition from idle is immediate;
# this only rate-limits changes, so a single shaky burst can't flip the screen.
SWITCH_CONFIRM_SCANS = 2

# --- Server ---------------------------------------------------------------
HOST = "127.0.0.1"  # change to "0.0.0.0" to expose on the LAN later
PORT = 8000
