"""Model-predicted age/gender attribute analysis, keyed by Global Person ID
(core/reid.py) — spec: MongDee_Master_Prompt_FairFace_Age_Gender.md.

Deliberately layered *alongside*, not instead of, the existing per-frame
gender/age classifier already wired into core/vision.py's
CameraWorker._classify_person() (vision.attributes.gender_age_backend's
OpenCVDnnGenderAgeBackend, fed the full person crop, smoothed only within
one local track's lifetime via core.tracker.PersonTracker's category vote).
That existing path still drives the live on-screen box color/label and
presence_sessions.category exactly as before — nothing here touches it.

This module adds a second, independent analysis keyed by *Global Person ID*
instead of local track ID, so a person's attribute profile survives track ID
churn and is shared correctly across every camera that re-identifies them
(spec sections 7/8/18) — something a local-track-scoped vote structurally
cannot do. It is also throttled per Global Person (spec section 12), not run
on every AI pass, to stay affordable on a GTX 1050.

FairFaceBackend also satisfies vision.attributes.extractor.GenderAgeBackend's
(and core/vision.py's duck-typed) interface — predict_gender(crop),
predict_age_group(crop) — so it is a drop-in replacement for
OpenCVDnnGenderAgeBackend if an operator wants FairFace driving the
existing on-screen classification too. Its predict_gender/predict_age_group
methods run face detection internally on whatever crop they're given, so
either caller (the old per-frame path, or the new AttributeAnalyzer below)
gets a face-only prediction, never a full-body one (spec section 4).

No FairFace checkpoint ships with this repo, matching
vision/attributes/gender_age_backend.py's own documented reasoning: this
project never fakes a prediction it can't actually produce, and a
third-party model file's hosting isn't something this repo can vouch for as
stable. Bring your own res34_fair_align_multi_7_20190809.pt (or equivalent)
via --fairface-checkpoint; see the official repo for the file:
https://github.com/joojs/fairface
"""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from core.gender_calibration import GenderCalibration

# ---------------------------------------------------------------- constants

# FairFace's 9 age groups, in the exact order the official model's age head
# outputs them (github.com/joojs/fairface predict.py) — index 0..8.
AGE_GROUP_LABELS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]

# FairFace's gender head order is [Male, Female] (opposite of this repo's
# existing Levi&Hassner-based OpenCVDnnGenderAgeBackend, which is
# [female, male] — see vision/attributes/gender_age_backend.py). Kept
# lowercase here to match this project's existing 'female'/'male'/'unknown'
# convention used throughout core/vision.py and presence_sessions.category.
GENDER_LABELS = ["male", "female"]

# The official 7-race FairFace model's fc layer outputs 18 logits:
# race(7) + gender(2) + age(9). Only gender/age are used here (spec: race is
# out of scope) — race logits are sliced off and discarded, never surfaced.
NUM_RACE_CLASSES = 7
FC_OUT_FEATURES = NUM_RACE_CLASSES + len(GENDER_LABELS) + len(AGE_GROUP_LABELS)

# Spec section 6: thresholds live in one place, as config, not scattered
# hardcoded literals. A prediction below its floor is UNKNOWN, never forced.
ATTRIBUTE_GENDER_MIN_CONFIDENCE = 0.70
# Age is no longer analysed or shown (product decision: only male / female / product).
# The constant stays so old imports keep working; nothing reads it any more.
ATTRIBUTE_AGE_MIN_CONFIDENCE = 0.5

# A wrong gender label is worse than none, so a Global Person is only labelled once the
# accumulated evidence is strong AND comes from several frames. Each frame contributes the
# log-odds of its prediction (clipped, because FairFace is over-confident on small faces);
# the running sum decays a little per sample so old frames fade, and the label only flips
# when the opposite side has built up real evidence of its own.
ATTRIBUTE_GENDER_EVIDENCE_MIN = 2.0     # summed log-odds needed to show MALE/FEMALE (~2 agreeing 0.8 frames)
ATTRIBUTE_GENDER_MIN_SAMPLES = 2        # never label from a single frame
ATTRIBUTE_GENDER_LOGIT_CLIP = 2.5       # one over-confident frame can contribute at most this much
ATTRIBUTE_GENDER_DECAY = 0.95           # evidence multiplier applied before each new sample
ATTRIBUTE_GENDER_KEEP_FRACTION = 0.5
# Whole-person (body) evidence, used when the face is too small/absent - see core/body_gender.py.
ATTRIBUTE_BODY_LOGIT_CLIP = 1.5           # a body sample is weaker evidence than a face sample
ATTRIBUTE_BODY_ONLY_EVIDENCE_MIN = 2.5    # stricter when there is (almost) no readable face
ATTRIBUTE_BODY_ONLY_MIN_SAMPLES = 4
ATTRIBUTE_FACE_DECISIVE_EVIDENCE = 3.0    # face-only evidence needed before the person may teach the body model
ATTRIBUTE_FACE_DECISIVE_MIN_SAMPLES = 3    # a shown label is kept until evidence drops below this x MIN

# Spec section 12: throttle — re-analyze the same Global Person at most this
# often, never every AI pass.
ATTRIBUTE_ANALYSIS_INTERVAL = 0.75
ATTRIBUTE_URGENT_ANALYSIS_INTERVAL = 0.35   # a person still UNKNOWN is analysed more often: evidence is what they lack

# Found via a live camera test: a person who turned their back to the
# camera kept showing their earlier, correctly-observed gender/age
# indefinitely, because GlobalPersonAttributeSmoother's confidence-weighted
# vote sum never decayed -- once a label won, it stayed the reported
# result forever, even with zero fresh face evidence since. If a Global
# Person hasn't produced a real add_sample() call in this many seconds,
# get() now expires that person's accumulated votes and reports UNKNOWN
# again instead of continuing to show a stale result as if it were still
# current. Chosen to comfortably outlast a normal brief head-turn or
# look-away (avoiding flicker) while still expiring within a few AI passes
# of someone genuinely no longer facing any camera -- an evidence-informed
# judgment call, not a rigorously swept optimum, same as
# ATTRIBUTE_MIN_FACE_SIZE_PX and SINGLE_PRODUCT_MATCH_FLOOR.
ATTRIBUTE_STALE_GRACE_SEC = 8.0

# Spec section 10 (track quality gate): a track has to be old/large enough,
# and its face crop sharp/bright enough, before spending a forward pass on
# it. Mirrors core.tripwire/core.reid's own *_MIN_AGE_SEC / *_MIN_*_NORM gates.
ATTRIBUTE_MIN_TRACK_AGE_SEC = 0.5
ATTRIBUTE_MIN_BBOX_HEIGHT_NORM = 0.10  # small enough that a far person still yields whole-body evidence (the face model itself rejects unreadable faces)
# Raised from 32 -> 64 after a live two-camera test caught a real, confident
# misclassification: a genuine 58x68px face crop (CAM-2, 320x240 feed,
# non-frontal angle) cleared the old 32px floor and produced a wrong,
# over-threshold "FEMALE 69%" -- the model itself wasn't malfunctioning
# (blur variance 1180 vs the 40.0 floor, brightness 80 -- comfortably
# within quality bounds), the *input* was just too small: FairFace resizes
# every crop to 224x224 regardless of source size, so a ~58px source gets
# ~4x upsampled, which is interpolation, not real facial detail -- a
# textbook cause of confident-but-wrong attribute predictions. 64px is an
# evidence-informed judgment call (large enough to reject that specific
# observed failure), not a rigorously swept optimum -- revisit once more
# real hard-case examples are collected (see the upgrade brief's "hard case
# dataset" step). Trades recall (more distant/small faces become UNKNOWN)
# for precision (fewer confident wrong answers), matching this module's
# and _classify_person's own stated preference: a wrong category is worse
# than none.
ATTRIBUTE_MIN_FACE_SIZE_PX = 64          # a face at least this large (source pixels) gets full trust
# ---- far-range faces -------------------------------------------------------------------------------------
# The 64 px floor above was a hard cut-off: on a 640x480 stream a face is only 64 px wide when the person
# fills the frame, so anybody at normal booth distance (or further) was never analysed at all and stayed
# UNKNOWN. Small faces are now analysed too, but they are treated as WEAKER evidence instead of being thrown
# away: detection runs on an up-scaled head region, the FairFace input is up-scaled with a cubic filter, a
# small face must be more confident to count, and its log-odds are divided by a temperature that grows as the
# face shrinks - so several agreeing frames are needed before a far person is labelled. The person-level
# error is still bounded by the smoother's evidence gate (>= 2 samples, evidence >= 2.0).
ATTRIBUTE_SMALL_FACE_MIN_PX = 14         # smallest face (source pixels) that is analysed at all
ATTRIBUTE_SMALL_FACE_MIN_CONFIDENCE = 0.80   # a face below full-trust size must be at least this confident
ATTRIBUTE_SMALL_FACE_MAX_TEMPERATURE = 3.0   # log-odds divisor at ATTRIBUTE_SMALL_FACE_MIN_PX
ATTRIBUTE_HEAD_REGION_FRACTION = 0.5     # faces are searched in the top half of a person box
ATTRIBUTE_DETECT_TARGET_HEAD_PX = 240    # the head region is up-scaled to about this height for the detector
ATTRIBUTE_MAX_UPSCALE = 4.0
ATTRIBUTE_FAR_MIN_BLUR_VARIANCE = 12.0   # blur floor at the smallest face size (the full floor applies at 64 px)
ATTRIBUTE_MIN_BLUR_VARIANCE = 40.0     # Laplacian variance floor — below this, too blurry to trust
ATTRIBUTE_MIN_BRIGHTNESS = 25.0        # mean pixel value floor (0-255) — below this, too dark
ATTRIBUTE_MAX_BRIGHTNESS = 235.0       # ceiling — blown-out/overexposed is equally unreadable

def to_age_category(age_group: str) -> str:
    """Age is no longer analysed (only male / female / product are reported), so there is no
    CHILD/ADULT bucket any more. Kept as a function so old callers and stored rows still work."""
    return "UNKNOWN"


# --------------------------------------------------------------- face crop

def face_temperature(face_px: "float | None") -> float:
    """Divisor applied to a face prediction's log-odds: 1.0 for a face of full-trust size or unknown size,
    growing (log-linearly in size) to ATTRIBUTE_SMALL_FACE_MAX_TEMPERATURE at the smallest analysed face."""
    if face_px is None or face_px >= ATTRIBUTE_MIN_FACE_SIZE_PX:
        return 1.0
    px = max(float(face_px), float(ATTRIBUTE_SMALL_FACE_MIN_PX))
    span = float(np.log(ATTRIBUTE_MIN_FACE_SIZE_PX / ATTRIBUTE_SMALL_FACE_MIN_PX))
    frac = float(np.log(ATTRIBUTE_MIN_FACE_SIZE_PX / px)) / span
    return 1.0 + (ATTRIBUTE_SMALL_FACE_MAX_TEMPERATURE - 1.0) * frac


def attenuate_confidence(confidence: float, face_px: "float | None") -> float:
    """The confidence a face prediction is worth once its size is taken into account: the log-odds divided by
    face_temperature(face_px), turned back into a probability. Unchanged for a face of full-trust size."""
    t = face_temperature(face_px)
    if t == 1.0:
        return float(confidence)
    p = min(max(float(confidence), 1e-6), 1 - 1e-6)
    return float(1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / t)))


WEBCAM_CHECKPOINT_NAME = "webcam_model_state_dict.pt"
FACE_DUMP_MIN_INTERVAL_SEC = 2.0
FACE_DUMP_MAX_FILES = 4000


class FaceDumper:
    """Saves face crops the booth really sees (opt-in, local disk only) for later hand-labelling and training.
    File name: <guess>_<conf%>_<face px>px_<time>.jpg inside data/face_labels/unlabeled/. Rate limited and capped."""

    def __init__(self, directory: "str | Path", min_interval_sec: float = FACE_DUMP_MIN_INTERVAL_SEC,
                 max_files: int = FACE_DUMP_MAX_FILES):
        self.directory = Path(directory)
        self.min_interval_sec = min_interval_sec
        self.max_files = max_files
        self._last = 0.0
        self._count = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "FaceDumper | None":
        import os
        if os.environ.get("MONGDEE_SAVE_FACES", "").strip().lower() not in ("1", "true", "yes"):
            return None
        default = Path(__file__).resolve().parent.parent / "data" / "face_labels" / "unlabeled"
        return cls(os.environ.get("MONGDEE_SAVE_FACES_DIR") or default)

    def maybe_save(self, face_bgr: np.ndarray, gender: str, confidence: float, face_px: "float | None",
                   now: "float | None" = None) -> "Path | None":
        now = time.time() if now is None else now
        with self._lock:
            if now - self._last < self.min_interval_sec or face_bgr is None or face_bgr.size == 0:
                return None
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                if self._count is None:
                    self._count = sum(1 for _ in self.directory.glob("*.jpg"))
                if self._count >= self.max_files:
                    return None
                px = "na" if face_px is None else f"{int(round(face_px))}"
                path = self.directory / f"{gender}_{int(round(confidence * 100))}_{px}px_{int(now * 1000)}.jpg"
                if not cv2.imwrite(str(path), face_bgr):
                    return None
            except Exception:
                return None
            self._last = now
            self._count += 1
            return path


def resolve_fairface_checkpoint(default_path: "str | Path") -> Path:
    """The FairFace weights to load: the webcam-adapted checkpoint written by tools/finetune_fairface_gender.py when it
    sits next to `default_path` (that tool only writes it after it beat the original on webcam-like faces), else
    `default_path`. Set MONGDEE_FAIRFACE_ORIGINAL=1 to force the original."""
    import os
    default = Path(default_path)
    webcam = default.parent / WEBCAM_CHECKPOINT_NAME
    if webcam.is_file() and os.environ.get("MONGDEE_FAIRFACE_ORIGINAL", "").strip().lower() not in ("1", "true", "yes"):
        return webcam
    return default


def face_min_confidence(face_px: "float | None") -> float:
    """Per-frame confidence a face prediction needs to count at all."""
    if face_px is None or face_px >= ATTRIBUTE_MIN_FACE_SIZE_PX:
        return ATTRIBUTE_GENDER_MIN_CONFIDENCE
    return max(ATTRIBUTE_GENDER_MIN_CONFIDENCE, ATTRIBUTE_SMALL_FACE_MIN_CONFIDENCE)


def blur_floor_for(face_px: "float | None") -> float:
    """Laplacian-variance floor, relaxed for small faces (an honest small face is soft by nature)."""
    if face_px is None or face_px >= ATTRIBUTE_MIN_FACE_SIZE_PX:
        return ATTRIBUTE_MIN_BLUR_VARIANCE
    frac = (float(face_px) - ATTRIBUTE_SMALL_FACE_MIN_PX) / (ATTRIBUTE_MIN_FACE_SIZE_PX - ATTRIBUTE_SMALL_FACE_MIN_PX)
    frac = min(max(frac, 0.0), 1.0)
    return ATTRIBUTE_FAR_MIN_BLUR_VARIANCE + (ATTRIBUTE_MIN_BLUR_VARIANCE - ATTRIBUTE_FAR_MIN_BLUR_VARIANCE) * frac


class YuNetFaceDetector:
    """Lightweight DNN face detector — OpenCV's built-in YuNet
    (cv2.FaceDetectorYN), per spec section 4's "lightweight face detector
    ที่เหมาะกับ GTX 1050": a small (~230KB) ONNX model, fast enough to run
    on CPU alone, with a real calibrated confidence score per detection
    (unlike a Haar cascade). This build of opencv-python has no
    cv2.CascadeClassifier at all (verified — it's simply absent from this
    wheel's API surface), so YuNet is used instead of the originally-planned
    Haar cascade; it is also the more accurate option either way.

    Bring your own ONNX weights (same "this repo never bundles/
    auto-downloads third-party model files" policy as
    vision/attributes/gender_age_backend.py and this module's
    FairFaceBackend) — get face_detection_yunet_2023mar.onnx from the
    official OpenCV Zoo: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
    """

    def __init__(self, onnx_model_path: str | Path, score_threshold: float = 0.6,
                 min_face_size_px: int = ATTRIBUTE_SMALL_FACE_MIN_PX):
        self._min_face_size_px = min_face_size_px
        self._detector = cv2.FaceDetectorYN_create(
            str(onnx_model_path), "", (320, 320), score_threshold, 0.3, 5000)

    def detect(self, person_crop_bgr: np.ndarray) -> tuple[list[float], float] | None:
        """Returns (face_bbox_in_crop_coords, confidence) for the
        highest-confidence detected face, or None if no face clears
        min_face_size_px."""
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return None
        h, w = person_crop_bgr.shape[:2]
        if h == 0 or w == 0:
            return None
        self._detector.setInputSize((w, h))
        _retval, faces = self._detector.detect(person_crop_bgr)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[14])  # column 14 = detection score
        x, y, fw, fh, score = best[0], best[1], best[2], best[3], best[14]
        if fw < self._min_face_size_px or fh < self._min_face_size_px:
            return None
        return [float(x), float(y), float(x + fw), float(y + fh)], float(score)


    def detect_in_person(self, person_crop_bgr: np.ndarray) -> "tuple[list[float], float] | None":
        """Face search designed for people who are far away: only the head region (top part of the person box)
        is searched, after being up-scaled so a 15-20 px face becomes something the detector was trained on.
        Returns (face_bbox in the ORIGINAL person-crop pixels, score), or None. Faces below
        `min_face_size_px` (measured in original pixels) are ignored; among several candidates the one nearest
        the head's horizontal centre wins (a poster or a neighbour's head at the edge should not)."""
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return None
        ch, cw = person_crop_bgr.shape[:2]
        head_h = max(int(ch * ATTRIBUTE_HEAD_REGION_FRACTION), 1)
        head = person_crop_bgr[:head_h]
        if head.shape[0] < 6 or head.shape[1] < 6:
            return None
        scale = float(min(max(ATTRIBUTE_DETECT_TARGET_HEAD_PX / head.shape[0], 1.0), ATTRIBUTE_MAX_UPSCALE))
        img = cv2.resize(head, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1.0 else head
        ih, iw = img.shape[:2]
        self._detector.setInputSize((iw, ih))
        _retval, faces = self._detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        best, best_key = None, -1.0
        for f in faces:
            x, y, fw, fh, score = (float(v) for v in (f[0], f[1], f[2], f[3], f[14]))
            if fw / scale < self._min_face_size_px or fh / scale < self._min_face_size_px:
                continue
            centre_offset = abs((x + fw / 2.0) / iw - 0.5) * 2.0            # 0 centred .. 1 at the edge
            key = score * (1.0 - 0.5 * centre_offset)
            if key > best_key:
                best, best_key = (x / scale, y / scale, (x + fw) / scale, (y + fh) / scale, score), key
        if best is None:
            return None
        return [best[0], best[1], best[2], best[3]], float(best[4])


ATTRIBUTE_FACE_CROP_MARGIN = 0.25   # FairFace training images include ~25% context around the face


def crop_face(person_crop_bgr: np.ndarray, face_bbox: list[float],
              margin: float = ATTRIBUTE_FACE_CROP_MARGIN) -> np.ndarray:
    """Face crop for the gender model. The FairFace model was trained on faces padded by ~25%
    (forehead, hair, ears, chin), so a tight detector box is a distribution mismatch that costs
    accuracy: expand the box by `margin` of its width/height on every side, clipped to the image."""
    h, w = person_crop_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in face_bbox)
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - margin * bw, x2 + margin * bw
    y1, y2 = y1 - margin * bh, y2 + margin * bh
    x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return person_crop_bgr[y1:y2, x1:x2]


def _blur_variance(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _brightness(image_bgr: np.ndarray) -> float:
    return float(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).mean())


# ------------------------------------------------------------- FairFace CNN

class FairFaceBackend:
    """ResNet34 with an 18-wide fc head (race7 + gender2 + age9), the exact
    architecture github.com/joojs/fairface's res34_fair_align_multi_7_*.pt
    checkpoints use. Bring your own checkpoint (see module docstring) — this
    class never downloads one.

    Satisfies the same predict_gender/predict_age_group(crop) -> (label,
    confidence) interface as vision.attributes.extractor.GenderAgeBackend
    (and core/vision.py's duck-typed equivalent), except the crop it expects
    may be a full person crop *or* a face crop — face detection runs
    internally either way (spec section 4), so both existing and new
    callers always get a face-only prediction under the hood.

    Gender is predicted from the original face AND its horizontal mirror (logits averaged, one
    batch of two). That test-time augmentation cancels much of the pose/lighting asymmetry that
    makes single-view predictions flip on a webcam, at the cost of one extra image per pass.
    """

    def __init__(self, checkpoint_path: str | Path, face_detector: "YuNetFaceDetector",
                 device: str = "cpu"):
        import torch
        from torchvision.models import resnet34

        self.device = torch.device(device) if not isinstance(device, int) else torch.device(f"cuda:{device}")
        self._face_detector = face_detector

        model = resnet34()
        model.fc = torch.nn.Linear(model.fc.in_features, FC_OUT_FEATURES)
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        try:
            model.to(self.device)
        except Exception:
            # Same GTX-1050/Optimus-safe fallback as core/recognizer.py and
            # core/reid.py — a GPU that enumerates but is actually busy
            # never crashes the booth process over an attribute model.
            self.device = torch.device("cpu")
            model.to(self.device)
        self._model = model
        self._lock = threading.Lock()
        # Decision bias fitted on webcam-like faces for THIS checkpoint (tools/finetune_fairface_gender.py); None
        # when there is none or it belongs to another checkpoint. See core/gender_calibration.py.
        self.calibration = GenderCalibration.load_for(checkpoint_path)
        # Opt-in (MONGDEE_SAVE_FACES=1): keep the faces this booth really sees, with the model's guess in the file name,
        # so a person can sort them into data/face_labels/male|female and tools/finetune_fairface_gender.py can train on them.
        self.face_dump = FaceDumper.from_env()

        from torchvision import transforms
        self._preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _forward(self, face_bgr: np.ndarray) -> np.ndarray:
        import torch

        if min(face_bgr.shape[:2]) < 112:
            # A far-away face is a few dozen pixels: enlarge it with a cubic filter ourselves rather than the
            # default bilinear resize inside the preprocessing pipeline (smoother edges, closer to how the
            # network's training faces look).
            face_bgr = cv2.resize(face_bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._preprocess(face_rgb)
        batch = torch.stack([tensor, torch.flip(tensor, dims=[2])]).to(self.device)  # original + mirror
        with self._lock, torch.no_grad():
            logits = self._model(batch).mean(dim=0).cpu().numpy()
        return logits

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        exp = np.exp(x)
        return exp / exp.sum()

    def _predict_face(self, face_bgr: np.ndarray, face_px: "float | None" = None) -> tuple[str, float, str, float]:
        """Returns (gender_label, gender_conf, age_group_label, age_conf)
        for an already-cropped face — the one forward pass both
        predict_gender/predict_age_group need, split apart below only
        because that's the interface both callers expect."""
        logits = self._forward(face_bgr)
        gender_logits = logits[NUM_RACE_CLASSES:NUM_RACE_CLASSES + len(GENDER_LABELS)]
        age_logits = logits[NUM_RACE_CLASSES + len(GENDER_LABELS):]
        calibration = getattr(self, "calibration", None)
        if calibration is not None:
            gender_logits = calibration.apply(gender_logits, face_px)
        gender_probs = self._softmax(gender_logits)
        age_probs = self._softmax(age_logits)
        g_idx, a_idx = int(np.argmax(gender_probs)), int(np.argmax(age_probs))
        return (GENDER_LABELS[g_idx], float(gender_probs[g_idx]),
                AGE_GROUP_LABELS[a_idx], float(age_probs[a_idx]))

    def _face_crop_or_none(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Detected + cropped face, or None if no face was found *or* the
        crop fails face_quality_ok's blur/brightness gate (spec section 12:
        "ห้ามส่งface crop ที่ไม่มีคุณภาพเข้า gender/age model แบบไร้เงื่อนไข").
        Both predict_gender/predict_age_group funnel through this one
        choke point, so a too-blurry/too-dark/blown-out face never reaches
        the model at all — treated exactly like "no face detected" (returns
        ("unknown", 0.0) to the caller), which
        GlobalPersonAttributeSmoother.add_sample already drops rather than
        casting a vote with, so a single bad-quality frame can't flip an
        already-stabilized gender/age state (spec section 12's "ให้ใช้
        temporal state เดิม")."""
        located = self._locate_face(crop_bgr)
        return None if located is None else located[0]

    def _locate_face(self, crop_bgr: np.ndarray) -> "tuple[np.ndarray, float] | None":
        """(padded face crop, face size in source pixels) or None. Uses the far-range head search when the
        detector offers it (attributes.YuNetFaceDetector.detect_in_person), else the plain detector."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        search = getattr(self._face_detector, "detect_in_person", None) or self._face_detector.detect
        face = search(crop_bgr)
        if face is None:
            return None
        face_bbox, _conf = face
        face_px = float(min(face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1]))
        face_crop = crop_face(crop_bgr, face_bbox)
        if not face_crop.size or not face_quality_ok(face_crop, face_px):
            return None
        return face_crop, face_px

    def predict_gender_detail(self, crop_bgr: np.ndarray) -> "dict | None":
        """Like predict_gender, but also reports how large the face was (source pixels) so the smoother can
        weigh a far-away face as weaker evidence. None when no usable face was found."""
        located = self._locate_face(crop_bgr)
        if located is None:
            return None
        face, face_px = located
        gender, gender_conf, _age, _age_conf = self._predict_face(face, face_px)
        dump = getattr(self, "face_dump", None)
        if dump is not None:
            dump.maybe_save(face, gender, gender_conf, face_px)
        return {"gender": gender, "confidence": gender_conf, "face_px": face_px}

    def predict_gender(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        detail = self.predict_gender_detail(crop_bgr)
        if detail is None:
            return "unknown", 0.0
        return detail["gender"], detail["confidence"]

    def predict_age_group(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        face = self._face_crop_or_none(crop_bgr)
        if face is None:
            return "unknown", 0.0
        _gender, _gender_conf, age, age_conf = self._predict_face(face)
        return age, age_conf


# ------------------------------------------------------------- result type

@dataclasses.dataclass
class AttributeResult:
    gender: str = "UNKNOWN"
    gender_confidence: float = 0.0
    age_group: str = "UNKNOWN"
    age_category: str = "UNKNOWN"
    age_confidence: float = 0.0
    sample_count: int = 0
    last_updated: float = 0.0
    status: str = "unknown"  # "ok" once the gender evidence is strong enough
    source: str = "none"     # "face", "face+body" or "body": which evidence produced the label

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# --------------------------------------------------------- temporal smoothing

class GlobalPersonAttributeSmoother:
    """Per-Global-Person-ID temporal fusion of GENDER evidence (age is no longer analysed).

    Two kinds of evidence are added to one running score (signed log-odds, male positive):
      * FACE samples  (add_sample)      - the FairFace face model, when a face is readable;
      * BODY samples  (add_body_sample) - the whole-person model in core/body_gender.py (hair, clothing, build,
                                          appearance), weighted by that model's own measured accuracy - this is
                                          what still works when the person is too far away for a face.
    The score decays slightly before every new sample. MALE/FEMALE is only reported once the evidence is strong
    enough from enough samples - with a readable face history that is >= 2 samples and evidence >= 2.0; with
    (almost) only body evidence it is stricter (>= 4 samples, evidence >= 2.5). A label that is already showing is
    held until its evidence has mostly drained away and only replaced when the other side builds real evidence.
    In-memory only per global_id; core.database persists a summary snapshot, never every sample (spec section 9)."""

    _MAX_KEPT_CONFS = 30

    def __init__(self, stale_grace_sec: float = ATTRIBUTE_STALE_GRACE_SEC):
        # global_id -> {"e": total signed evidence, "ef": face-only signed evidence, "nf"/"nb": face/body samples,
        #               "label": "male"/"female"/None, "conf": {"male": [...], "female": [...]}}
        self._votes: dict[str, dict] = {}
        self._last_sample_ts: dict[str, float] = {}
        self._stale_grace_sec = stale_grace_sec
        self._lock = threading.Lock()

    @staticmethod
    def _logit(p: float) -> float:
        p = min(max(p, 1e-6), 1 - 1e-6)
        return float(np.log(p / (1 - p)))

    @staticmethod
    def _new_bucket() -> dict:
        return {"e": 0.0, "ef": 0.0, "nf": 0, "nb": 0, "label": None, "guess": None, "conf": {"male": [], "female": []}}

    def add_sample(self, global_id: str, gender: str, gender_conf: float,
                    age_group: str = "unknown", age_conf: float = 0.0,
                    now: float = 0.0, face_px: "float | None" = None) -> AttributeResult:
        """`face_px` (optional): size of the face this prediction came from, in source pixels. A face below
        full-trust size must be more confident to count and contributes proportionally less evidence."""
        with self._lock:
            bucket = self._votes.setdefault(global_id, self._new_bucket())
            g = (gender or "").lower()
            if g in ("male", "female") and gender_conf >= face_min_confidence(face_px):
                push = min(self._logit(gender_conf), ATTRIBUTE_GENDER_LOGIT_CLIP) / face_temperature(face_px)
                signed = push if g == "male" else -push
                bucket["e"] = bucket["e"] * ATTRIBUTE_GENDER_DECAY + signed
                bucket["ef"] = bucket["ef"] * ATTRIBUTE_GENDER_DECAY + signed
                bucket["nf"] += 1
                confs = bucket["conf"][g]
                confs.append(float(gender_conf))
                del confs[:-self._MAX_KEPT_CONFS]
            self._last_sample_ts[global_id] = now
            return self._aggregate(bucket, now)

    def add_body_sample(self, global_id: str, p_male: float, weight: float, now: float = 0.0) -> AttributeResult:
        """Whole-person evidence. `weight` (0..1) is the body model's measured competence."""
        with self._lock:
            bucket = self._votes.setdefault(global_id, self._new_bucket())
            if weight > 0:
                push = float(np.clip(self._logit(p_male), -ATTRIBUTE_BODY_LOGIT_CLIP, ATTRIBUTE_BODY_LOGIT_CLIP)) * weight
                bucket["e"] = bucket["e"] * ATTRIBUTE_GENDER_DECAY + push
                bucket["nb"] += 1
            self._last_sample_ts[global_id] = now
            return self._aggregate(bucket, now)

    GUESS_FLIP_EVIDENCE = 0.75   # a shown best-guess only flips once the OTHER side has built this much evidence

    def guess(self, global_id: str, now: "float | None" = None) -> "tuple[str, float] | None":
        """Best available gender for someone whose evidence is still below the 'decided' bar: the sign of the
        accumulated evidence ('male'/'female', confidence 0.5-0.99), or None when nothing at all was observed. Used so a
        box is never drawn UNKNOWN once ANY evidence exists; sticky (it does not flicker on tiny changes)."""
        now = time.time() if now is None else now
        with self._lock:
            bucket = self._votes.get(global_id)
            if bucket is None or bucket["nf"] + bucket["nb"] == 0:
                return None
            if now - self._last_sample_ts.get(global_id, 0.0) > self._stale_grace_sec:
                return None
            e, prev = bucket["e"], bucket.get("guess")
            if prev is None:
                if e == 0.0:
                    return None
                current = "male" if e > 0 else "female"
            elif prev == "male" and e <= -self.GUESS_FLIP_EVIDENCE:
                current = "female"
            elif prev == "female" and e >= self.GUESS_FLIP_EVIDENCE:
                current = "male"
            else:
                current = prev
            bucket["guess"] = current
            conf = float(1.0 / (1.0 + np.exp(-abs(e))))
            return current, min(max(conf, 0.5), 0.99)

    def face_label(self, global_id: str) -> "str | None":
        """'male'/'female' only when the FACE evidence alone is decisive (used to teach the body model)."""
        with self._lock:
            bucket = self._votes.get(global_id)
            if bucket is None or bucket["nf"] < ATTRIBUTE_FACE_DECISIVE_MIN_SAMPLES:
                return None
            if bucket["ef"] >= ATTRIBUTE_FACE_DECISIVE_EVIDENCE:
                return "male"
            if bucket["ef"] <= -ATTRIBUTE_FACE_DECISIVE_EVIDENCE:
                return "female"
            return None

    @staticmethod
    def _decide(bucket: dict) -> "str | None":
        e, prev = bucket["e"], bucket["label"]
        face_backed = bucket["nf"] >= ATTRIBUTE_GENDER_MIN_SAMPLES
        need_e = ATTRIBUTE_GENDER_EVIDENCE_MIN if face_backed else ATTRIBUTE_BODY_ONLY_EVIDENCE_MIN
        need_n = ATTRIBUTE_GENDER_MIN_SAMPLES if face_backed else ATTRIBUTE_BODY_ONLY_MIN_SAMPLES
        if bucket["nf"] + bucket["nb"] < need_n:
            return None
        if e >= need_e:
            return "male"
        if e <= -need_e:
            return "female"
        keep = need_e * ATTRIBUTE_GENDER_KEEP_FRACTION
        if prev == "male" and e >= keep:
            return "male"
        if prev == "female" and e <= -keep:
            return "female"
        return None

    def _aggregate(self, bucket: dict, now: float) -> AttributeResult:
        label = self._decide(bucket)
        bucket["label"] = label
        n = bucket["nf"] + bucket["nb"]
        if label is None:
            return AttributeResult(sample_count=n, last_updated=now)
        confs = bucket["conf"][label]
        if confs:
            confidence = sum(confs) / len(confs)
        else:                                       # decided by the body model alone: report the evidence strength
            confidence = float(1.0 / (1.0 + np.exp(-abs(bucket["e"]))))
        face_backed = bucket["nf"] >= ATTRIBUTE_GENDER_MIN_SAMPLES
        source = ("face+body" if bucket["nb"] else "face") if face_backed else "body"
        return AttributeResult(gender=label.upper(), gender_confidence=confidence, sample_count=n,
                               last_updated=now, status="ok", source=source)

    def merge(self, loser: str, winner: str) -> None:
        """Two identities turned out to be one person: pool their evidence under the winner."""
        with self._lock:
            lb = self._votes.pop(loser, None)
            lt = self._last_sample_ts.pop(loser, None)
            if lb is None:
                return
            wb = self._votes.setdefault(winner, self._new_bucket())
            for k in ("e", "ef", "nf", "nb"):
                wb[k] += lb[k]
            for g in ("male", "female"):
                wb["conf"][g] = (wb["conf"][g] + lb["conf"][g])[-self._MAX_KEPT_CONFS:]
            wb["label"] = wb["label"] or lb["label"]
            self._last_sample_ts[winner] = max(self._last_sample_ts.get(winner, 0.0), lt or 0.0)

    def get(self, global_id: str, now: "float | None" = None) -> AttributeResult:
        """Returns the Global Person's current fused attribute result, or a
        fresh/empty (UNKNOWN, status="unknown") result if nothing has been
        observed yet -- or if what was observed has gone stale (no real
        add_sample() call within ATTRIBUTE_STALE_GRACE_SEC), in which case
        the expired votes are also discarded so a person returning to view
        after a long absence starts from a clean slate rather than an old
        result instantly winning again. now defaults to the real wall clock
        so callers that don't pass it (the common case) still get correct
        staleness behavior."""
        if now is None:
            now = time.time()
        with self._lock:
            bucket = self._votes.get(global_id)
            if bucket is None:
                return AttributeResult()
            last_sample = self._last_sample_ts.get(global_id, 0.0)
            if now - last_sample > self._stale_grace_sec:
                self._votes.pop(global_id, None)
                self._last_sample_ts.pop(global_id, None)
                return AttributeResult()
            return self._aggregate(bucket, now)

    def forget(self, global_id: str) -> None:
        with self._lock:
            self._votes.pop(global_id, None)
            self._last_sample_ts.pop(global_id, None)


class AttributeSampler:
    """Per-Global-Person-ID throttle + track quality gate (spec sections 10
    and 12) — the analogue of core.reid.ReIDSampler, deliberately keyed by
    global_id rather than (camera_id, local_track_id): the whole point is
    not to re-run FairFace once per camera per track for what might be the
    same physical person (spec section 8)."""

    def __init__(self, interval_sec: float = ATTRIBUTE_ANALYSIS_INTERVAL,
                 min_track_age_sec: float = ATTRIBUTE_MIN_TRACK_AGE_SEC,
                 min_bbox_height_norm: float = ATTRIBUTE_MIN_BBOX_HEIGHT_NORM):
        self._interval_sec = interval_sec
        self._min_track_age_sec = min_track_age_sec
        self._min_bbox_height_norm = min_bbox_height_norm
        self._last_analyzed_ts: dict[str, float] = {}

    def should_analyze(self, global_id: str, track: dict, frame_height: int, now: float,
                        urgent: bool = False) -> bool:
        """`urgent`: this person still has no gender label, so evidence is what they lack - sample faster."""
        x1, y1, x2, y2 = track["bbox"]
        bbox_height_norm = (y2 - y1) / frame_height if frame_height > 0 else 0.0
        if bbox_height_norm < self._min_bbox_height_norm:
            return False
        track_age = now - track.get("first_seen", now)
        if track_age < self._min_track_age_sec:
            return False
        last = self._last_analyzed_ts.get(global_id, 0.0)
        interval = min(self._interval_sec, ATTRIBUTE_URGENT_ANALYSIS_INTERVAL) if urgent else self._interval_sec
        if now - last < interval:
            return False
        self._last_analyzed_ts[global_id] = now
        return True

    def forget(self, global_id: str) -> None:
        self._last_analyzed_ts.pop(global_id, None)


def face_quality_ok(face_crop_bgr: np.ndarray, face_px: "float | None" = None) -> bool:
    """Spec section 10's blur/brightness checks — run once a face has
    actually been detected/cropped, before spending a FairFace forward pass
    on it. A too-blurry or too-dark/blown-out face crop can't be read
    reliably even by a good classifier, so it's better to skip and try
    again next sample than to feed noise in and report a confident-looking
    but meaningless prediction."""
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return False
    if _blur_variance(face_crop_bgr) < blur_floor_for(face_px):
        return False
    brightness = _brightness(face_crop_bgr)
    return ATTRIBUTE_MIN_BRIGHTNESS <= brightness <= ATTRIBUTE_MAX_BRIGHTNESS
