"""Multi-camera Person Re-ID — maps each camera's own local, ephemeral
track ID (core.tracker.PersonTracker's) to a shared GlobalPerson identity,
so "how many *different* people visited" doesn't double-count someone who
walked past two cameras, or whose local track ID churned (lost/regained) on
the same camera.

Deliberately layered *on top of*, not instead of, core.tracker.PersonTracker:
local track IDs stay camera-scoped and ephemeral exactly as before — nothing
here changes tracker.py, and "CAM-01 Track 17" / "CAM-02 Track 17" are never
assumed to be the same person just because the numbers match (spec section
6). This module only adds a second, independent mapping built from
appearance embeddings plus simple temporal/spatial context.

Matching is in-memory only (spec section 26: "DB เป็น persistence layer
ไม่ใช่ real-time matcher ทุก frame") — GlobalIdentityRegistry never touches
the database itself; callers (web/booth_manager.py) decide when to write a
GlobalPerson summary row for dashboard/analytics use.

Embedding approach mirrors core/recognizer.py's ProductRecognizer (frozen
MobileNetV3-Small backbone, mean-pooled feature, L2-normalized, then
mean-centered against a fixed synthetic calibration set — see that module's
docstring for *why* centering is required before comparing raw embeddings).
Kept as a separate model instance/gallery rather than reusing
ProductRecognizer directly: person crops and product crops are a different
visual domain, and conflating their calibration/threshold would make both
worse. This does mean a second small CNN is loaded when Re-ID is enabled —
acceptable on the CPU/GTX1050-class hardware this project targets (spec
section 32) because embedding only ever runs on a throttled sample of
tracks (see ReIDSampler), never every frame.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from core.body_cues import color_descriptor, hair_similarity
from core.face_identity import FACE_DIFFERENT, FACE_SAME

# ------------------------------------------------------------- matching config

REID_COLOR_WEIGHT = 0.30      # share of the embedding similarity that comes from clothing colours
MATCH_FLOOR = 0.55            # minimum centered cosine similarity to even consider a match
MATCH_MARGIN = 0.08           # winner must beat the runner-up by this much (avoid ambiguous merges)
MAX_EMBEDDINGS_PER_PERSON = 8  # memory cap (spec section 22) — oldest dropped first

# An unconfigured camera pair (spec section 25) still needs *some* transition
# window, or Re-ID could never match across cameras at all out of the box.
# Wide and permissive by design — a real booth should configure tighter,
# topology-accurate windows via GlobalIdentityRegistry.set_camera_transition().
DEFAULT_MIN_TRANSITION_SEC = 0.0
DEFAULT_MAX_TRANSITION_SEC = 30.0

# ---------------------------------------------- camera-layout-derived topology
#
# set_camera_transition() existed but nothing at startup ever called it with real numbers, so
# every deployment silently ran on the topology-blind defaults above (any camera to any camera,
# 0-30s, "plausible"). Human walking speed bounds turn a physical camera-to-camera distance (from
# a booth layout file -- same schema as configs/booth_layout.example.json) into a real transition
# window instead: "if camera positions/configuration are available, use them" (spec section 12).
# Deliberately wide (brisk walk to a slow browse-while-walking pace), plus a pause allowance on the
# slow end because a booth visitor typically stops to look at something between cameras rather
# than walking straight through.
MIN_WALK_SPEED_MPS = 0.35
MAX_WALK_SPEED_MPS = 2.0
TRANSITION_PAUSE_ALLOWANCE_SEC = 5.0
_LAYOUT_UNIT_TO_METERS = {"m": 1.0, "meter": 1.0, "meters": 1.0, "ft": 0.3048, "feet": 0.3048}


def transition_window_from_distance(distance_m: float, min_speed_mps: float = MIN_WALK_SPEED_MPS,
                                     max_speed_mps: float = MAX_WALK_SPEED_MPS,
                                     pause_allowance_sec: float = TRANSITION_PAUSE_ALLOWANCE_SEC
                                     ) -> "tuple[float, float]":
    """(min_sec, max_sec) a real person could plausibly take to walk `distance_m` between two
    cameras. Pure function (no registry/IO) so it's directly unit-testable against known
    distances/speeds without needing a layout file."""
    if distance_m <= 0:
        return (0.0, pause_allowance_sec)
    return (distance_m / max_speed_mps, distance_m / min_speed_mps + pause_allowance_sec)


def camera_transitions_from_booth_layout(layout: dict) -> "dict[tuple[str, str], tuple[float, float]]":
    """Extracts every camera's (x, y) position from a booth-layout dict (the schema
    configs/booth_layout.example.json uses: {"unit": "m"|"ft", "objects": [{"object_type":
    "camera", "id" or "metadata": {"camera_id": ...}, "x": ..., "y": ...}, ...]}) and returns a
    {(camera_a, camera_b): (min_sec, max_sec)} map for every camera pair, ready to feed into
    GlobalIdentityRegistry.set_camera_transition(). Returns {} (never raises) for a layout with
    fewer than two identifiable cameras -- an incomplete/malformed layout should fall back to the
    existing topology-blind default, not break startup."""
    unit_scale = _LAYOUT_UNIT_TO_METERS.get(str(layout.get("unit", "m")).lower(), 1.0)
    cameras: dict[str, tuple[float, float]] = {}
    for obj in layout.get("objects", []):
        if obj.get("object_type") != "camera":
            continue
        camera_id = (obj.get("metadata") or {}).get("camera_id") or obj.get("id")
        if not camera_id or "x" not in obj or "y" not in obj:
            continue
        cameras[camera_id] = (float(obj["x"]) * unit_scale, float(obj["y"]) * unit_scale)

    windows: "dict[tuple[str, str], tuple[float, float]]" = {}
    ids = sorted(cameras)
    for i, cam_a in enumerate(ids):
        for cam_b in ids[i + 1:]:
            ax, ay = cameras[cam_a]
            bx, by = cameras[cam_b]
            distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            windows[(cam_a, cam_b)] = transition_window_from_distance(distance)
    return windows


def apply_camera_layout(registry: "GlobalIdentityRegistry", layout: dict) -> int:
    """Configures `registry` with every camera-pair transition window derivable from `layout`
    (see camera_transitions_from_booth_layout). Returns how many pairs were configured, so a
    caller can log/warn when a supplied layout file yielded zero usable camera positions (e.g. a
    typo'd camera_id that never matches this run's actual camera_ids) -- harmless to the registry
    either way, but worth an operator noticing."""
    windows = camera_transitions_from_booth_layout(layout)
    for (cam_a, cam_b), (min_sec, max_sec) in windows.items():
        registry.set_camera_transition(cam_a, cam_b, min_sec, max_sec)
    return len(windows)

# ------------------------------------------------------- sampling/quality gate

# Spec section 21: don't embed every frame. Re-embed the same track at most
# this often.
REID_SAMPLE_INTERVAL_SEC = 0.75

# Spec sections 12/21: a track has to be old/large enough to trust before
# spending a forward pass and a matching decision on it. Mirrors
# core.tripwire's TRACK_MIN_AGE_SEC / TRACK_MIN_BBOX_HEIGHT_NORM — same
# reasoning (new/tiny tracks are disproportionately detector noise).
REID_MIN_TRACK_AGE_SEC = 0.5
REID_MIN_BBOX_HEIGHT_NORM = 0.12

_CALIBRATION_SEED = 7
_embed_lock = threading.Lock()  # serialize forward passes across camera threads


def _build_calibration_images() -> list[np.ndarray]:
    rng = np.random.default_rng(_CALIBRATION_SEED)
    images = []
    for hue in range(0, 180, 12):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :] = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        images.append(img)
    for _ in range(15):
        noise = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        images.append(cv2.GaussianBlur(noise, (21, 21), 0))
    return images


class PersonReIDEmbedder:
    """Turns a person crop (BGR ndarray) into a fixed-length appearance
    embedding. One instance shared across every camera in a booth."""

    def __init__(self, device: str = "cpu", cache_dir: Path | None = None):
        self.device = torch.device(device)
        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)
        backbone.classifier = torch.nn.Identity()
        backbone.eval()
        try:
            backbone.to(self.device)
        except Exception:
            # Same defense-in-depth as ProductRecognizer.__init__ — a GPU
            # that enumerates but is actually busy/unavailable must degrade
            # to CPU, never crash the booth process.
            self.device = torch.device("cpu")
            backbone.to(self.device)
        self._model = backbone
        self._preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self._calibration_mean = self._build_calibration_mean(cache_dir)

    def _build_calibration_mean(self, cache_dir: Path | None) -> np.ndarray:
        cache_path = (cache_dir / "_reid_calibration_mean.npy") if cache_dir else None
        if cache_path is not None and cache_path.exists():
            return np.load(cache_path)
        embeddings = np.array([self._embed_raw(img) for img in _build_calibration_images()])
        mean = embeddings.mean(axis=0)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, mean)
        return mean

    def _embed_raw(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._preprocess(image_rgb).unsqueeze(0).to(self.device)
        with _embed_lock, torch.no_grad():
            feature = self._model(tensor).squeeze(0).cpu().numpy()
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else feature

    def embed(self, image_bgr: np.ndarray) -> np.ndarray:
        """Mean-centered, L2-normalized embedding ready for cosine-similarity
        comparison (see module docstring on why centering is required)."""
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("empty crop")
        raw = self._embed_raw(image_bgr)
        centered = raw - self._calibration_mean
        norm = np.linalg.norm(centered)
        cnn = centered / norm if norm > 0 else centered
        # Blend in the clothing-colour layout (torso + legs): the cosine of two embeddings becomes
        # (1-W)*cos(appearance CNN) + W*cos(clothing colours). An ImageNet CNN alone is a weak person
        # descriptor; what people are wearing is the strongest cue a booth camera reliably sees.
        colour = color_descriptor(image_bgr)
        cnorm = np.linalg.norm(colour)
        if cnorm > 0:
            colour = colour / cnorm
        combined = np.concatenate([np.sqrt(1.0 - REID_COLOR_WEIGHT) * cnn, np.sqrt(REID_COLOR_WEIGHT) * colour])
        n = np.linalg.norm(combined)
        return combined / n if n > 0 else combined


class ReIDSampler:
    """Per-(camera_id, track_id) throttle + quality gate deciding whether a
    given AI pass should spend a forward pass re-embedding this track.
    Framework-agnostic (no camera/model access) — just timestamps and bbox
    geometry, same shape as core.tripwire's gate."""

    def __init__(self, sample_interval_sec: float = REID_SAMPLE_INTERVAL_SEC,
                 min_track_age_sec: float = REID_MIN_TRACK_AGE_SEC,
                 min_bbox_height_norm: float = REID_MIN_BBOX_HEIGHT_NORM):
        self._sample_interval_sec = sample_interval_sec
        self._min_track_age_sec = min_track_age_sec
        self._min_bbox_height_norm = min_bbox_height_norm
        self._last_embed_ts: dict[tuple[str, int], float] = {}

    def should_embed(self, camera_id: str, track: dict, frame_height: int, now: float) -> bool:
        track_id = track["track_id"]
        x1, y1, x2, y2 = track["bbox"]
        bbox_height_norm = (y2 - y1) / frame_height if frame_height > 0 else 0.0
        if bbox_height_norm < self._min_bbox_height_norm:
            return False
        track_age = now - track.get("first_seen", now)
        if track_age < self._min_track_age_sec:
            return False
        key = (camera_id, track_id)
        last = self._last_embed_ts.get(key, 0.0)
        if now - last < self._sample_interval_sec:
            return False
        self._last_embed_ts[key] = now
        return True

    def forget(self, camera_id: str, track_id: int) -> None:
        self._last_embed_ts.pop((camera_id, track_id), None)


# ------------------------------------------------------------ global identity

# --- anti-flicker / anti-duplicate identity logic (all thresholds are conservative judgment calls; the
# registry exposes stats() so a real booth can watch how often each rule fires) ------------------------
STITCH_MAX_GAP_SEC = 4.0          # a "new" track this soon after a person vanished nearby is the same person
STITCH_FLOOR = 0.30               # appearance is only a sanity check when place and time already agree
STITCH_MAX_CENTER_DIST = 1.2      # in person heights: how far the person may have moved during the gap
CONFIRM_MIN_OBS = 2               # an identity is only COUNTED once it has this many observations ...
CONFIRM_MIN_SPAN_SEC = 0.6        # ... spread over at least this long (a one-blink ghost never counts)
MERGE_FLOOR = 0.72                # two confirmed identities this alike (and never seen together) are one person
MERGE_PROVISIONAL_FLOOR = 0.50    # an unconfirmed identity only needs to resemble a confirmed one this much
MERGE_INTERVAL_SEC = 2.0
RELINK_LOW_SIM = 0.10             # a track whose look keeps disagreeing with its identity ...
RELINK_AFTER = 3                  # ... this many samples in a row is re-matched from scratch (tracker ID swap)
VISIBLE_HOLD_SEC = 1.0            # identities seen in the latest pass of a camera count as "on screen now"
# web/booth_manager.py's _update_reid computes `now` ONCE per AI pass and passes that same value to every
# track's resolve() call that pass; observe_frame() (called once, moments earlier in the same pass, from
# _on_person_tracks) uses its own separate time.time() call. SAME_PASS_WINDOW_SEC is how close two resolve()/
# observe_frame() timestamps have to be to be treated as "the same pass" by _mark_on_screen — generous next to
# the sub-millisecond real gap between those two time.time() calls, but far short of VISIBLE_HOLD_SEC (1.0 s),
# so a resolve() call a genuine second-plus later (e.g. the same person on a re-minted local track after their
# old one was evicted) is never mistaken for "still on screen" just because _mark_on_screen marked them a
# second ago — that would wrongly block re-matching them (see test_forgotten_local_track_can_rematch_same_global_person).
SAME_PASS_WINDOW_SEC = 0.25

# --- one person seen by several cameras -----------------------------------------------------------------
# Two cameras see the same person from different angles, with different white balance / exposure / lens. An
# appearance embedding compared ACROSS cameras is therefore systematically lower than one compared within a
# camera, and (worse) a stranger already known on this camera scores higher than the true owner known only on
# the other one. Three counter-measures, all switchable through the constructor:
CROSS_CAMERA_COLOUR_WEIGHT = 0.35  # across cameras, clothing colour (camera-normalised) counts this much, the CNN less
CROSS_CAMERA_BONUS = 0.0           # added to a cross-camera similarity so it competes fairly with same-camera ones
CROSS_LIVE_SEC = 1.5               # "seen on the other camera just now": the same person is probably in both views
# Body appearance alone (no readable face) is a WEAK cue between two different cameras: a real booth showed a woman
# on CAM-1 and a man on CAM-2 (colour agreement 0.57) being glued into one ID. So without face evidence a link
# across cameras must be clearly strong, and a face embedding (core.face_identity) decides whenever both have one.
CROSS_LIVE_FLOOR = 0.62            # ... so a slightly lower match floor is enough (still needs a clear winner)
CROSS_LIVE_MARGIN = 0.10
MERGE_CROSS_FLOOR = 0.72           # two confirmed identities from different cameras this alike are one person ...
MERGE_CROSS_CHECKS = 3             # ... if they stay that alike in this many merge checks in a row
MERGE_MARGIN_MIN_CROSS = 0.64      # "clear winner" merge: this alike across cameras (0.60 on the same camera) ...
MERGE_MARGIN_MIN = 0.60
COLOUR_VETO = 0.30                 # clothing colour this different from EVERY stored look of a person: not the same person
CROSS_COLOUR_VETO = 0.62           # ... and across cameras (the person was never seen by THIS camera) the bar is higher
CROSS_COLOUR_MIN = 0.72            # a match ACROSS cameras also needs the clothes to agree at least this much
CROSS_LIVE_COLOUR_MIN = 0.80       # ... and the relaxed "seen on the other camera just now" floor needs them to agree clearly
MERGE_CROSS_COLOUR_MIN = 0.78      # ... as does merging two identities that were seen by different cameras
# --- "same person?" needs ALL the cues to agree when there is no face proof (torso colour, trousers colour, hairstyle)
CROSS_PART_VETO = 0.55             # across cameras: torso OR legs colour agreeing less than this = different people
CROSS_PART_MIN = 0.72              # ... and a body-only link across cameras needs BOTH to agree at least this much
CROSS_HAIR_VETO = 0.35             # across cameras: hairstyle this different (length / width / colour) = different people
CROSS_HAIR_MIN = 0.55              # ... and a body-only link across cameras needs the hairstyle to agree this much
RELINK_FACE_AFTER = 2              # a track whose FACE is clearly not its identity's face, this many samples in a row
FACE_MARGIN = 0.06                 # the best face match must beat the runner-up by this much
MAX_FACES_PER_PERSON = 6
RELINK_CONFLICT_AFTER = 3          # a track whose clothes contradict its identity this many samples in a row is split off
RELINK_GENDER_AFTER = 2            # ... and a track whose FACE says the other gender than its identity, this many
MERGE_MARGIN = 0.25                # ... AND each other's single best match, ahead of every other identity by this much


@dataclasses.dataclass
class GlobalPerson:
    global_id: str
    first_seen: float
    last_seen: float
    cameras: set = dataclasses.field(default_factory=set)
    embeddings: list = dataclasses.field(default_factory=list)  # list[np.ndarray], most-recent last
    last_camera: str | None = None
    last_bbox: list | None = None
    n_obs: int = 0
    confirmed: bool = False
    coexisted: set = dataclasses.field(default_factory=set)      # ids seen on screen at the same moment
    emb_cams: list = dataclasses.field(default_factory=list)     # camera each stored embedding came from
    emb_keys: list = dataclasses.field(default_factory=list)     # (camera, local track) each embedding came from
    gender: "str | None" = None                                   # 'male' / 'female' once decided (set by the caller)
    faces: list = dataclasses.field(default_factory=list)         # unit face embeddings (core.face_identity), newest last
    hairs: list = dataclasses.field(default_factory=list)         # hairstyle descriptors (core.body_cues), newest last

    def add_hair(self, hair) -> None:
        if hair is None:
            return
        self.hairs.append(np.asarray(hair, dtype=np.float32))
        if len(self.hairs) > MAX_FACES_PER_PERSON:
            self.hairs.pop(0)

    def hair_similarity(self, hair) -> "float | None":
        """Best hairstyle agreement between `hair` and any stored look of this person (None when unknown)."""
        if hair is None or not self.hairs:
            return None
        return max(hair_similarity(hair, h) for h in self.hairs)

    def add_face(self, face: np.ndarray) -> None:
        f = np.asarray(face, dtype=np.float32)
        n = float(np.linalg.norm(f))
        if n == 0.0:
            return
        self.faces.append(f / n)
        if len(self.faces) > MAX_FACES_PER_PERSON:
            self.faces.pop(0)

    def face_similarity(self, face: "np.ndarray | None") -> "float | None":
        """Best cosine between `face` and any stored face of this person; None when either side has no face."""
        if face is None or not self.faces:
            return None
        f = np.asarray(face, dtype=np.float32)
        n = float(np.linalg.norm(f))
        if n == 0.0:
            return None
        return float(max(float(np.dot(f / n, g)) for g in self.faces))

    def add_embedding(self, embedding: np.ndarray, camera: "str | None" = None, key: "tuple | None" = None) -> None:
        self.embeddings.append(embedding)
        self.emb_cams.append(camera)
        self.emb_keys.append(key)
        if len(self.embeddings) > MAX_EMBEDDINGS_PER_PERSON:
            self.embeddings.pop(0)
            self.emb_cams.pop(0)
            self.emb_keys.pop(0)

    def keys_of_embeddings(self) -> list:
        keys = list(self.emb_keys)[-len(self.embeddings):] if self.embeddings else []
        return [None] * (len(self.embeddings) - len(keys)) + keys

    def drop_embeddings_of(self, key: tuple) -> int:
        """Forget every embedding that came from this (camera, local track) - used when that track turns out to be someone else."""
        keys = self.keys_of_embeddings()
        keep = [i for i, k in enumerate(keys) if k != key]
        dropped = len(self.embeddings) - len(keep)
        if dropped:
            cams = self.cameras_of_embeddings()
            self.embeddings = [self.embeddings[i] for i in keep]
            self.emb_cams = [cams[i] for i in keep]
            self.emb_keys = [keys[i] for i in keep]
        return dropped

    def cameras_of_embeddings(self) -> list:
        """Camera per stored embedding, padded with None if embeddings were assigned directly."""
        cams = list(self.emb_cams)[-len(self.embeddings):] if self.embeddings else []
        return [None] * (len(self.embeddings) - len(cams)) + cams

    def prototype(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        mean = np.mean(np.stack(self.embeddings), axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _unit_mean(vectors: list) -> np.ndarray:
    mean = np.mean(np.stack(vectors), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def _cos_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """All pairwise cosines between the rows of A and B (0 where a row is all zeros)."""
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    na[na == 0] = np.inf
    nb[nb == 0] = np.inf
    return (A / na) @ (B / nb).T


class _Scorer:
    """Appearance similarity that knows which camera each embedding came from.

    Within one camera it is the plain cosine. Across cameras the embedding (CNN part + clothing-colour part, see
    PersonReIDEmbedder.embed) is re-weighted towards the colour part - which was normalised for white balance and
    exposure - because a CNN feature moves a lot with viewpoint, and a small bonus is added so that a cross-camera
    score is comparable to a same-camera one. `colour_dim=0` (or a too-short vector) disables the re-weighting."""

    def __init__(self, colour_dim: int = 0, colour_weight: "float | None" = None,
                 cross_bonus: "float | None" = None):
        self.colour_dim = int(colour_dim)
        self.colour_weight = CROSS_CAMERA_COLOUR_WEIGHT if colour_weight is None else colour_weight
        self.cross_bonus = CROSS_CAMERA_BONUS if cross_bonus is None else cross_bonus

    def matrix(self, A: np.ndarray, cams_a: list, B: np.ndarray, cams_b: list) -> np.ndarray:
        """Similarity of every row of A to every row of B, camera-aware."""
        plain = _cos_matrix(A, B)
        cross = np.array([[(p is not None and q is not None and p != q) for q in cams_b] for p in cams_a], dtype=bool)
        if not cross.any():
            return plain
        d = self.colour_dim
        if 0 < d < A.shape[1]:
            cnn = _cos_matrix(A[:, :-d], B[:, :-d])
            col = _cos_matrix(A[:, -d:], B[:, -d:])
            has_a = np.linalg.norm(A[:, -d:], axis=1) > 0
            has_b = np.linalg.norm(B[:, -d:], axis=1) > 0
            usable = np.outer(has_a, has_b)
            wc = self.colour_weight
            weighted = np.where(usable, (1.0 - wc) * cnn + wc * col, plain)
        else:
            weighted = plain
        return np.where(cross, weighted + self.cross_bonus, plain)

    def colour_agreement(self, embedding: np.ndarray, person: GlobalPerson) -> "float | None":
        """How well the clothing colours of `embedding` agree with the best-agreeing stored look of `person`
        (cosine, -1..1); None when no colour information is available on either side."""
        d = self.colour_dim
        if d <= 0 or not person.embeddings or len(embedding) <= d:
            return None
        x = np.asarray(embedding, dtype=np.float32)[None, -d:]
        E = np.stack([np.asarray(e, dtype=np.float32)[-d:] for e in person.embeddings if len(e) > d])
        if E.size == 0 or float(np.linalg.norm(x)) == 0.0:
            return None
        usable = np.linalg.norm(E, axis=1) > 0
        if not usable.any():
            return None
        return float(_cos_matrix(x, E[usable]).max())

    def part_agreements(self, embedding: np.ndarray, person: GlobalPerson) -> "tuple[float, float] | None":
        """(torso colour, trousers colour) agreement of `embedding` with the best-agreeing stored look of `person`;
        None when colour information is missing. The colour block is [torso | legs] (core.body_cues.color_descriptor)."""
        d = self.colour_dim
        if d <= 0 or d % 2 or not person.embeddings or len(embedding) <= d:
            return None
        half = d // 2
        x = np.asarray(embedding, dtype=np.float32)[-d:]
        rows = [np.asarray(e, dtype=np.float32)[-d:] for e in person.embeddings if len(e) > d]
        if not rows:
            return None
        out = []
        for sl in (slice(0, half), slice(half, d)):
            xs = x[None, sl]
            E = np.stack([r[sl] for r in rows])
            ok = np.linalg.norm(E, axis=1) > 0
            if float(np.linalg.norm(xs)) == 0.0 or not ok.any():
                return None
            out.append(float(_cos_matrix(xs, E[ok]).max()))
        return out[0], out[1]

    def part_agreements_persons(self, a: GlobalPerson, b: GlobalPerson) -> "tuple[float, float] | None":
        best = None
        for e in a.embeddings:
            probe = GlobalPerson(global_id="_", first_seen=0.0, last_seen=0.0, embeddings=list(b.embeddings))
            r = self.part_agreements(np.asarray(e, dtype=np.float32), probe)
            if r is None:
                continue
            best = r if best is None else (max(best[0], r[0]), max(best[1], r[1]))
        return best

    def colour_agreement_persons(self, a: GlobalPerson, b: GlobalPerson) -> "float | None":
        d = self.colour_dim
        if d <= 0 or not a.embeddings or not b.embeddings:
            return None
        A = np.stack([np.asarray(e, dtype=np.float32)[-d:] for e in a.embeddings if len(e) > d])
        B = np.stack([np.asarray(e, dtype=np.float32)[-d:] for e in b.embeddings if len(e) > d])
        if A.size == 0 or B.size == 0:
            return None
        A, B = A[np.linalg.norm(A, axis=1) > 0], B[np.linalg.norm(B, axis=1) > 0]
        if A.size == 0 or B.size == 0:
            return None
        return float(_cos_matrix(A, B).max())

    @staticmethod
    def _groups(person: GlobalPerson) -> "tuple[list, np.ndarray]":
        """Per camera, the person's mean look (as unit vectors), and the cameras in the same order."""
        cams = person.cameras_of_embeddings()
        groups: dict = {}
        for e, c in zip(person.embeddings, cams):
            groups.setdefault(c, []).append(e)
        keys = list(groups)
        return keys, np.stack([_unit_mean(groups[k]) for k in keys])

    def to_person(self, embedding: np.ndarray, camera: "str | None", person: GlobalPerson) -> float:
        """Best single match blended with the match against the person's average look (per camera it was seen
        by), so one odd stored crop can neither create nor block a match."""
        if not person.embeddings:
            return -1.0
        x = np.asarray(embedding, dtype=np.float32)[None, :]
        cams = person.cameras_of_embeddings()
        best = float(self.matrix(x, [camera], np.stack(person.embeddings), cams).max())
        if len(person.embeddings) < 2:
            return best
        keys, protos = self._groups(person)
        proto = float(self.matrix(x, [camera], protos, keys).max())
        return 0.5 * best + 0.5 * proto

    def between(self, a: GlobalPerson, b: GlobalPerson) -> float:
        if not a.embeddings or not b.embeddings:
            return -1.0
        best = float(self.matrix(np.stack(a.embeddings), a.cameras_of_embeddings(),
                                 np.stack(b.embeddings), b.cameras_of_embeddings()).max())
        ka, pa = self._groups(a)
        kb, pb = self._groups(b)
        proto = float(self.matrix(pa, ka, pb, kb).max())
        return 0.5 * best + 0.5 * proto


_DEFAULT_SCORER = _Scorer()


def _similarity(embedding: np.ndarray, person: GlobalPerson) -> float:
    return _DEFAULT_SCORER.to_person(embedding, None, person)


def _person_similarity(a: GlobalPerson, b: GlobalPerson) -> float:
    return _DEFAULT_SCORER.between(a, b)


def _center_and_height(bbox) -> tuple[float, float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(y2 - y1, 1.0)


class GlobalIdentityRegistry:
    """One instance shared across every camera in a booth. Thread-safe —
    fed from whichever camera's AI pass most recently ran on AIWorker's
    shared thread (see core/vision.py's AIWorker), which may not be the
    same camera twice in a row.

    Three things keep one physical person from becoming several IDs when boxes flicker:
      * STITCHING - a new local track that appears close (in time and place) to where a person of
        similar look just vanished on the same camera continues that identity;
      * CONFIRMATION - with require_confirmation=True an identity is only counted (unique_people_count)
        once it has been observed repeatedly, so a one-blink false detection is never a "person";
      * MERGING - two identities that look alike and were never on screen together are merged
        (pop_merges() tells the caller so it can fix its own rows).
    Two people visible in the same pass on one camera are never merged and never matched to each other."""

    def __init__(self, match_floor: float = MATCH_FLOOR, match_margin: float = MATCH_MARGIN,
                 require_confirmation: bool = False, stitching: bool = True, merging: bool = True,
                 colour_dim: int = 0, cross_camera: bool = True):
        self._scorer = _Scorer(colour_dim=colour_dim if cross_camera else 0,
                               cross_bonus=None if cross_camera else 0.0)
        self._cross_camera = cross_camera
        self._merge_streak: dict[tuple[str, str], int] = {}
        self._split_log: list[str] = []
        self._banned: dict[tuple, set] = {}                # (camera, track) -> identities it was split from
        self._conflict_streak: dict[tuple, int] = {}
        self._match_floor = match_floor
        self._match_margin = match_margin
        self._require_confirmation = require_confirmation
        self._stitching = stitching
        self._merging = merging
        self._people: dict[str, GlobalPerson] = {}
        self._local_to_global: dict[tuple[str, int], str] = {}
        self._topology: dict[frozenset, tuple[float, float]] = {}
        self._visible: dict[str, tuple[float, set]] = {}     # camera -> (ts, {global ids on screen})
        # camera -> (ts, {global ids resolve() has already produced THIS pass}) - see _mark_on_screen. Kept
        # separate from _visible (observe_frame()'s own, slower-expiring snapshot) so the two can each use a
        # staleness window suited to their own timescale instead of fighting over one shared one.
        self._pass_marks: dict[str, tuple[float, set]] = {}
        self._low_streak: dict[tuple[str, int], int] = {}
        self._merge_log: list[tuple[str, str]] = []
        self._last_merge_check = 0.0
        self._stats = {"new_ids": 0, "stitched": 0, "merged": 0, "relinked": 0,
                       "cross_camera_matches": 0, "cross_camera_merges": 0, "vetoed": 0, "split_wrong_identity": 0,
                       "face_matches": 0, "face_merges": 0}
        self._next_id = 1
        self._lock = threading.Lock()

    def set_camera_transition(self, camera_a: str, camera_b: str,
                               min_transition_sec: float, max_transition_sec: float) -> None:
        """Spec section 25 (camera topology): configure how long a real walk
        between two cameras plausibly takes. A candidate match across these
        two cameras outside [min, max] is rejected regardless of appearance
        similarity — this is what stops "dressed similarly" false merges
        from an impossible/implausible transition (spec section 24)."""
        self._topology[frozenset((camera_a, camera_b))] = (min_transition_sec, max_transition_sec)

    def _transition_window(self, camera_a: str, camera_b: str) -> tuple[float, float]:
        if camera_a == camera_b:
            return (0.0, float("inf"))
        return self._topology.get(frozenset((camera_a, camera_b)),
                                   (DEFAULT_MIN_TRANSITION_SEC, DEFAULT_MAX_TRANSITION_SEC))

    # ------------------------------------------------------------------ helpers
    def _on_screen_now(self, camera_id: str, ts: float) -> set:
        """Union of observe_frame()'s own (slower-expiring, VISIBLE_HOLD_SEC) snapshot and _mark_on_screen's
        (fast-expiring, SAME_PASS_WINDOW_SEC) same-pass marks — see _mark_on_screen for why these need two
        different staleness windows rather than one shared one."""
        result = set()
        entry = self._visible.get(camera_id)
        if entry is not None and ts - entry[0] <= VISIBLE_HOLD_SEC:
            result |= entry[1]
        marks = self._pass_marks.get(camera_id)
        if marks is not None and abs(ts - marks[0]) <= SAME_PASS_WINDOW_SEC:
            result |= marks[1]
        return result

    def _touch(self, person: GlobalPerson, camera_id: str, ts: float, bbox) -> None:
        person.last_seen = ts
        person.last_camera = camera_id
        person.last_bbox = bbox
        person.cameras.add(camera_id)
        person.n_obs += 1
        if (not person.confirmed and person.n_obs >= CONFIRM_MIN_OBS
                and person.last_seen - person.first_seen >= CONFIRM_MIN_SPAN_SEC):
            person.confirmed = True

    def observe_frame(self, camera_id: str, visible_local_ids, ts: float) -> None:
        """Call once per AI pass of a camera with the local track ids on screen. Two identities on screen
        in the same pass are different people: remember that, so they are never merged or cross-matched.

        Only covers tracks that ALREADY have a global_id from a previous pass — web/booth_manager.py calls
        this once, at the very top of a pass, before resolve() has run for any of this pass's tracks, so a
        track appearing for the first time this pass (the very common case of two people walking in
        together) has no mapping yet and is invisible here. _mark_on_screen (called from resolve() itself,
        below) is what actually closes that gap for brand-new/just-resolved tracks."""
        with self._lock:
            gids = {self._local_to_global[(camera_id, i)] for i in visible_local_ids
                    if (camera_id, i) in self._local_to_global}
            self._visible[camera_id] = (ts, gids)
            for g in gids:
                person = self._people.get(g)
                if person is not None:
                    person.coexisted |= (gids - {g})

    def _mark_on_screen(self, camera_id: str, global_id: str, ts: float) -> None:
        """Widens THIS pass's on-screen set the instant a track resolves, rather than waiting for the next
        pass's observe_frame() call. Without this, two different real people who are BOTH resolved for the
        first time in the same AI pass (two people who walked in together — an ordinary, common case, not
        an edge case) are invisible to each other: observe_frame()'s snapshot at the top of the pass predates
        either of their mappings, so the on-screen exclusion in _resolve_locked's matching loop doesn't
        exclude the first one's brand-new identity when the second one is matched moments later in the same
        pass — and the same staleness also means they'd never be recorded as coexisted, so a later merge
        pass wouldn't be stopped either. Called once per resolve(), after the final global_id for this track
        is known, so every track in a pass (not just ones already known from a previous pass) is visible to
        every track resolved after it in the same pass.

        Writes to _pass_marks, never _visible: _visible is observe_frame()'s own store, read with the much
        longer VISIBLE_HOLD_SEC. Sharing one store between the two would force one staleness window on both
        jobs — either resolve() calls a genuine ~1s+ apart (e.g. the same person re-matched on a freshly
        minted local track after their old one was evicted) get wrongly treated as still "on screen" and
        blocked from matching, or observe_frame()'s own snapshot expires too fast. _on_screen_now unions both
        stores, each read with its own window, so neither job's timescale leaks into the other's."""
        others = self._on_screen_now(camera_id, ts) - {global_id}
        marks = self._pass_marks.get(camera_id)
        if marks is None or abs(ts - marks[0]) > SAME_PASS_WINDOW_SEC:
            gids = {global_id}
        else:
            gids = marks[1]
            gids.add(global_id)
        self._pass_marks[camera_id] = (ts, gids)
        if not others:
            return
        person = self._people.get(global_id)
        if person is not None:
            person.coexisted |= others
        for other_id in others:
            other = self._people.get(other_id)
            if other is not None:
                other.coexisted.add(global_id)

    def set_person_gender(self, global_id: str, gender: "str | None") -> None:
        """Tell the registry which gender the caller has decided for an identity ('male' / 'female' / None). Two
        identities (or a track and an identity) with different decided genders are never the same person."""
        with self._lock:
            person = self._people.get(global_id)
            if person is not None:
                person.gender = gender if gender in ("male", "female") else None

    def _contradicts(self, embedding, camera_id: str, person: GlobalPerson, track_gender, key, face=None,
                     hair=None) -> str | None:
        """Why `person` cannot be the owner of this new observation ('face' / 'gender' / 'colour' / 'banned'), or None.
        A face embedding is the strongest evidence there is: clearly a different face is a veto, clearly the same face
        overrides every weaker (clothes / gender-reading) veto."""
        if key is not None and person.global_id in self._banned.get(key, ()):
            return "banned"
        face_sim = person.face_similarity(face)
        if face_sim is not None:
            if face_sim < FACE_DIFFERENT:
                return "face"
            if face_sim >= FACE_SAME:
                return None
        if track_gender and person.gender and track_gender != person.gender:
            return "gender"
        if self._cross_camera:
            agreement = self._scorer.colour_agreement(embedding, person)
            if agreement is not None:
                veto = COLOUR_VETO if camera_id in person.cameras else CROSS_COLOUR_VETO
                if agreement < veto:
                    return "colour"
            if camera_id not in person.cameras:
                parts = self._scorer.part_agreements(embedding, person)
                if parts is not None and min(parts) < CROSS_PART_VETO:
                    return "colour"            # torso OR trousers clearly different (e.g. red top vs white top)
                hs = person.hair_similarity(hair)
                if hs is not None and hs < CROSS_HAIR_VETO:
                    return "hair"
        return None

    def _stitch_candidate(self, camera_id, embedding, ts, bbox, on_screen, track_gender=None, key=None,
                          face=None) -> tuple[str | None, float]:
        if bbox is None:
            return None, -1.0
        cx, cy, h = _center_and_height(bbox)
        best_id, best_score = None, -1.0
        for gid, person in self._people.items():
            if gid in on_screen or person.last_camera != camera_id or person.last_bbox is None:
                continue
            gap = ts - person.last_seen
            if gap < 0 or gap > STITCH_MAX_GAP_SEC:
                continue
            pcx, pcy, ph = _center_and_height(person.last_bbox)
            tall = max(h, ph)
            if ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5 > STITCH_MAX_CENTER_DIST * tall:
                continue
            if not (0.5 <= h / ph <= 2.0):
                continue
            if self._contradicts(embedding, camera_id, person, track_gender, key, face):
                continue
            score = self._scorer.to_person(embedding, camera_id, person)
            if score >= STITCH_FLOOR and score > best_score:
                best_id, best_score = gid, score
        return best_id, best_score

    # ------------------------------------------------------------------ face evidence
    def _learn_face(self, person: GlobalPerson, camera_id: str, face, hair=None) -> None:
        """Remember this track's face (and hairstyle) as one of the person's (never when it is clearly another face)."""
        if hair is not None:
            person.add_hair(hair)
        if face is None:
            return
        sim = person.face_similarity(face)
        if sim is not None and sim < FACE_DIFFERENT:
            return
        person.add_face(face)

    def _match_by_face(self, camera_id: str, face, on_screen: set, ts: float, track_gender, key) -> "str | None":
        """The identity whose stored face is unmistakably this face (and clearly better than any other), or None."""
        if face is None:
            return None
        ranked = []
        for gid, person in self._people.items():
            if gid in on_screen or (key is not None and gid in self._banned.get(key, ())):
                continue
            sim = person.face_similarity(face)
            if sim is None or sim < FACE_SAME:
                continue
            min_t, max_t = self._transition_window(person.last_camera or camera_id, camera_id)
            if not (min_t <= ts - person.last_seen <= max_t):
                continue
            if track_gender and person.gender and track_gender != person.gender:
                continue
            ranked.append((sim, gid))
        ranked.sort(reverse=True)
        if ranked and (len(ranked) == 1 or ranked[0][0] - ranked[1][0] >= FACE_MARGIN):
            return ranked[0][1]
        return None

    # ------------------------------------------------------------------ resolve
    def resolve(self, camera_id: str, local_track_id: int, embedding: np.ndarray,
                ts: float, bbox: list | None = None, gender: "str | None" = None,
                face: "np.ndarray | None" = None, hair: "np.ndarray | None" = None) -> tuple[str, bool]:
        """Map one local track's embedding to a GlobalPerson. Returns
        (global_id, is_new_person).

        `gender` - what THIS track's own face says ('male' / 'female' / None): an identity already decided as the
        other gender is never matched to it, and a track already mapped to such an identity is split off.

        `face` - the face embedding (core.face_identity) of this track, when a usable face was visible: the same face
        links tracks / identities across cameras with certainty, a clearly different face keeps two people apart.

        Already-mapped (camera_id, local_track_id) pairs refine (add an embedding to) their existing global identity
        rather than re-matching from scratch - re-matching every call could let a borderline-similar bystander steal
        an already-confirmed mapping. Exceptions, each re-matching the track from scratch (and never again to the same
        identity): its look keeps disagreeing with the identity (RELINK_AFTER samples), its clothes contradict every look
        the identity has (RELINK_CONFLICT_AFTER), or its own face says the other gender (RELINK_GENDER_AFTER).
        """
        with self._lock:
            gid, is_new = self._resolve_locked(camera_id, local_track_id, embedding, ts, bbox, gender, face, hair)
            final = self._local_to_global.get((camera_id, local_track_id), gid)
            self._mark_on_screen(camera_id, final, ts)
            self._maybe_merge(ts)
            return final, is_new and final == gid

    def _resolve_locked(self, camera_id: str, local_track_id: int, embedding: np.ndarray,
                        ts: float, bbox: list | None = None, track_gender: "str | None" = None,
                        face: "np.ndarray | None" = None, hair: "np.ndarray | None" = None) -> tuple[str, bool]:
        key = (camera_id, local_track_id)
        existing_global_id = self._local_to_global.get(key)
        if existing_global_id is not None and existing_global_id in self._people:
            person = self._people[existing_global_id]
            low = person.n_obs >= 3 and self._scorer.to_person(embedding, camera_id, person) < RELINK_LOW_SIM
            self._low_streak[key] = self._low_streak.get(key, 0) + 1 if low else 0
            contradiction = self._contradicts(embedding, camera_id, person, track_gender, None, face, hair)
            streak = self._conflict_streak.get((key, contradiction), 0) + 1 if contradiction else 0
            self._conflict_streak = {k: v for k, v in self._conflict_streak.items() if k[0] != key}
            if contradiction:
                self._conflict_streak[(key, contradiction)] = streak
            must_split = ((contradiction == "gender" and streak >= RELINK_GENDER_AFTER)
                          or (contradiction == "face" and streak >= RELINK_FACE_AFTER)
                          or (contradiction == "colour" and streak >= RELINK_CONFLICT_AFTER))
            if self._low_streak.get(key, 0) < RELINK_AFTER and not must_split:
                if self._low_streak.get(key, 0) == 0 and not contradiction:
                    person.add_embedding(embedding, camera_id, key)   # a disagreeing crop must not pollute the identity
                    self._learn_face(person, camera_id, face, hair)
                self._touch(person, camera_id, ts, bbox)
                return existing_global_id, False
            self._low_streak.pop(key, None)
            self._local_to_global.pop(key, None)       # fall through: match this look afresh
            self._stats["relinked"] += 1
            if must_split or self._cross_camera:
                # this track is somebody else: never map it back to that identity, and take its crops out of it
                self._banned.setdefault(key, set()).add(existing_global_id)
                person.drop_embeddings_of(key)
                if must_split:
                    self._stats["split_wrong_identity"] += 1
                    self._split_log.append(existing_global_id)

        on_screen = self._on_screen_now(camera_id, ts)
        if self._stitching:
            stitched, _score = self._stitch_candidate(camera_id, embedding, ts, bbox, on_screen, track_gender, key, face)
            if stitched is not None:
                person = self._people[stitched]
                person.add_embedding(embedding, camera_id, key)
                self._learn_face(person, camera_id, face, hair)
                self._touch(person, camera_id, ts, bbox)
                self._local_to_global[key] = stitched
                self._stats["stitched"] += 1
                return stitched, False

        by_face = self._match_by_face(camera_id, face, on_screen, ts, track_gender, key)
        if by_face is not None:
            person = self._people[by_face]
            if camera_id not in person.cameras:
                self._stats["cross_camera_matches"] += 1
            self._stats["face_matches"] += 1
            person.add_embedding(embedding, camera_id, key)
            self._learn_face(person, camera_id, face, hair)
            self._touch(person, camera_id, ts, bbox)
            self._local_to_global[key] = by_face
            return by_face, False

        best_id, best_score, second_score = None, -1.0, -1.0
        best_live = False
        best_agreement = None
        for global_id, person in self._people.items():
            if not person.embeddings or global_id in on_screen:
                continue  # someone on screen right now under another track is a different person
            min_t, max_t = self._transition_window(person.last_camera or camera_id, camera_id)
            elapsed = ts - person.last_seen
            if elapsed < min_t or elapsed > max_t:
                continue  # implausible/impossible transition — appearance alone never overrides this
            if self._contradicts(embedding, camera_id, person, track_gender, key, face, hair):
                self._stats["vetoed"] += 1
                continue  # different clothes / different gender / already split from this track
            score = self._scorer.to_person(embedding, camera_id, person)
            if score > best_score:
                second_score = best_score
                best_id, best_score = global_id, score
                best_live = (self._cross_camera and person.last_camera not in (None, camera_id)
                             and elapsed <= CROSS_LIVE_SEC)
                best_agreement = self._scorer.colour_agreement(embedding, person) if self._cross_camera else None
            elif score > second_score:
                second_score = score

        # A person who was seen on ANOTHER camera a moment ago is very likely the one who just appeared here
        # (overlapping views): a slightly lower floor / margin is enough for them - but only if their CLOTHES clearly
        # agree. Any match across cameras needs the clothes to agree at least CROSS_COLOUR_MIN.
        if best_id is not None and self._cross_camera and best_agreement is not None:
            person = self._people[best_id]
            crosses = camera_id not in person.cameras or person.last_camera != camera_id
            if best_live and best_agreement < CROSS_LIVE_COLOUR_MIN:
                best_live = False
            if crosses and best_agreement < CROSS_COLOUR_MIN:
                best_id = None
            if crosses and best_id is not None:
                # No face proof: only link two cameras when EVERY cue agrees (torso colour, trousers colour, hairstyle)
                sim_f = person.face_similarity(face)
                if sim_f is None or sim_f < FACE_SAME:
                    parts = self._scorer.part_agreements(embedding, person)
                    hs = person.hair_similarity(hair)
                    if (parts is not None and min(parts) < CROSS_PART_MIN) or (hs is not None and hs < CROSS_HAIR_MIN):
                        best_id = None
        floor = min(self._match_floor, CROSS_LIVE_FLOOR) if best_live else self._match_floor
        margin = min(self._match_margin, CROSS_LIVE_MARGIN) if best_live else self._match_margin
        if best_id is not None and best_score >= floor and (best_score - max(second_score, 0.0)) >= margin:
            person = self._people[best_id]
            if camera_id not in person.cameras:
                self._stats["cross_camera_matches"] += 1
            person.add_embedding(embedding, camera_id, key)
            self._learn_face(person, camera_id, face, hair)
            self._touch(person, camera_id, ts, bbox)
            self._local_to_global[key] = best_id
            return best_id, False

        # UNKNOWN — no confident match. Never force-match (spec section
        # 23): mint a new global identity instead.
        global_id = f"P{self._next_id:06d}"
        self._next_id += 1
        person = GlobalPerson(global_id=global_id, first_seen=ts, last_seen=ts,
                               cameras={camera_id}, last_camera=camera_id, last_bbox=bbox)
        person.add_embedding(embedding, camera_id, key)
        if face is not None:
            person.add_face(face)
        person.add_hair(hair)
        if track_gender:
            person.gender = track_gender
        self._touch(person, camera_id, ts, bbox)
        self._people[global_id] = person
        self._local_to_global[key] = global_id
        self._stats["new_ids"] += 1
        return global_id, True

    # ------------------------------------------------------------------ merging
    def _maybe_merge(self, ts: float) -> None:
        if not self._merging or ts - self._last_merge_check < MERGE_INTERVAL_SEC:
            return
        self._last_merge_check = ts
        hits: set = set()
        while True:
            pair = self._merge_candidate(hits)
            if pair is None:
                break
            self._merge(winner=self._people[pair[0]], loser=self._people[pair[1]], by_face=pair[2])
        for key in list(self._merge_streak):
            if key not in hits:
                del self._merge_streak[key]

    def _merge_candidate(self, hits: set) -> "tuple[str, str, bool] | None":
        """The next pair of identities that is one person, or None. Two ways to qualify:
          * plain floor - alike enough (MERGE_FLOOR for two confirmed identities, MERGE_PROVISIONAL_FLOOR when one is
            still a blip). Identities only ever seen by DIFFERENT cameras use the lower MERGE_CROSS_FLOOR;
          * clear winner - moderately alike (MERGE_MARGIN_MIN) but each is the OTHER's single best match and beats
            everybody else by MERGE_MARGIN. Two cameras with different colour/viewpoint rarely score a person
            above 0.6 against themselves, yet score every stranger far lower - that gap is the real evidence.
        Cross-camera and clear-winner merges must hold on MERGE_CROSS_CHECKS consecutive checks, so one lucky
        similar-clothes stranger cannot merge two people. Identities seen together are never merged."""
        ids = sorted(self._people, key=lambda g: self._people[g].first_seen)
        sims: dict = {}
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                a, b = self._people[a_id], self._people[b_id]
                if a.embeddings and b.embeddings:
                    sims[(a_id, b_id)] = sims[(b_id, a_id)] = self._scorer.between(a, b)
        if not sims:
            return None
        best: dict = {}
        for (x, y), v in sims.items():
            best.setdefault(x, []).append((v, y))
        for x in best:
            best[x].sort(reverse=True)

        def runner_up(x: str, other: str) -> float:
            for v, y in best.get(x, []):
                if y != other:
                    return v
            return -1.0

        for i, a_id in enumerate(ids):
            a = self._people[a_id]
            for b_id in ids[i + 1:]:
                b = self._people[b_id]
                if (a_id, b_id) not in sims or b_id in a.coexisted or a_id in b.coexisted:
                    continue
                if not (a.confirmed or b.confirmed):
                    continue        # two unconfirmed blips: leave them, they never get counted anyway
                fs = self._face_between(a, b)
                if fs is not None and fs < FACE_DIFFERENT:
                    continue        # clearly different faces: two people, whatever else looks alike
                by_face = fs is not None and fs >= FACE_SAME
                if a.gender and b.gender and a.gender != b.gender and not by_face:
                    continue        # decided as different genders: two people, whatever they look like
                cross = self._cross_camera and a.cameras.isdisjoint(b.cameras)
                if self._cross_camera and not by_face:
                    agreement = self._scorer.colour_agreement_persons(a, b)
                    if agreement is not None and (agreement < COLOUR_VETO or (cross and agreement < MERGE_CROSS_COLOUR_MIN)):
                        continue    # different clothes: never one person
                    if cross:
                        parts = self._scorer.part_agreements_persons(a, b)
                        if parts is not None and min(parts) < CROSS_PART_MIN:
                            continue    # torso or trousers do not clearly agree: cannot confirm it is one person
                        if a.hairs and b.hairs:
                            hs = max(hair_similarity(x, y) for x in a.hairs for y in b.hairs)
                            if hs < CROSS_HAIR_MIN:
                                continue    # hairstyle differs
                sim = sims[(a_id, b_id)]
                both = a.confirmed and b.confirmed
                if cross:
                    floor = MERGE_CROSS_FLOOR if both else MERGE_PROVISIONAL_FLOOR
                else:
                    floor = MERGE_FLOOR if both else MERGE_PROVISIONAL_FLOOR
                by_floor = sim >= floor
                by_margin = False
                if self._cross_camera and sim >= (MERGE_MARGIN_MIN_CROSS if cross else MERGE_MARGIN_MIN):
                    top_a = best[a_id][0][1] if best.get(a_id) else None
                    top_b = best[b_id][0][1] if best.get(b_id) else None
                    by_margin = (top_a == b_id and top_b == a_id
                                 and sim - max(runner_up(a_id, b_id), runner_up(b_id, a_id)) >= MERGE_MARGIN)
                if not (by_floor or by_margin or by_face):
                    continue
                if cross or (by_margin and not by_floor):
                    pair = (a_id, b_id)
                    hits.add(pair)
                    self._merge_streak[pair] = self._merge_streak.get(pair, 0) + 1
                    if self._merge_streak[pair] < (2 if by_face else MERGE_CROSS_CHECKS):
                        continue
                    if cross:
                        self._stats["cross_camera_merges"] += 1
                if by_face:
                    self._stats["face_merges"] += 1
                return a_id, b_id, by_face
        return None

    @staticmethod
    def _face_between(a: GlobalPerson, b: GlobalPerson) -> "float | None":
        if not a.faces or not b.faces:
            return None
        return float(max(float(np.dot(x, y)) for x in a.faces for y in b.faces))

    def _merge(self, winner: GlobalPerson, loser: GlobalPerson, by_face: bool = False) -> None:
        winner.faces = (winner.faces + loser.faces)[-MAX_FACES_PER_PERSON:]
        winner.hairs = (winner.hairs + loser.hairs)[-MAX_FACES_PER_PERSON:]
        winner.emb_cams = (winner.cameras_of_embeddings() + loser.cameras_of_embeddings())[-MAX_EMBEDDINGS_PER_PERSON:]
        winner.emb_keys = (winner.keys_of_embeddings() + loser.keys_of_embeddings())[-MAX_EMBEDDINGS_PER_PERSON:]
        winner.embeddings = (winner.embeddings + loser.embeddings)[-MAX_EMBEDDINGS_PER_PERSON:]
        winner.cameras |= loser.cameras
        winner.first_seen = min(winner.first_seen, loser.first_seen)
        if loser.last_seen > winner.last_seen:
            winner.last_seen, winner.last_camera, winner.last_bbox = loser.last_seen, loser.last_camera, loser.last_bbox
        winner.n_obs += loser.n_obs
        winner.confirmed = winner.confirmed or loser.confirmed
        winner.gender = winner.gender or loser.gender
        winner.coexisted |= loser.coexisted
        winner.coexisted.discard(winner.global_id); winner.coexisted.discard(loser.global_id)
        for k, g in list(self._local_to_global.items()):
            if g == loser.global_id:
                self._local_to_global[k] = winner.global_id
        for cam, (ts, gids) in list(self._visible.items()):
            if loser.global_id in gids:
                self._visible[cam] = (ts, (gids - {loser.global_id}) | {winner.global_id})
        for cam, (ts, gids) in list(self._pass_marks.items()):
            if loser.global_id in gids:
                self._pass_marks[cam] = (ts, (gids - {loser.global_id}) | {winner.global_id})
        for p in self._people.values():
            if loser.global_id in p.coexisted:
                p.coexisted.discard(loser.global_id); p.coexisted.add(winner.global_id)
        del self._people[loser.global_id]
        for pair in [p for p in self._merge_streak if loser.global_id in p]:
            del self._merge_streak[pair]
        self._merge_log.append((loser.global_id, winner.global_id))
        self._stats["merged"] += 1

    def pop_splits(self) -> list[str]:
        """Identities that had a wrongly attached track split off since the last call: whatever the caller learned
        about them (gender evidence...) may include the other person's data and should be re-learned."""
        with self._lock:
            out, self._split_log = self._split_log, []
            return out

    def pop_merges(self) -> list[tuple[str, str]]:
        """(loser_id, winner_id) merges since the last call, so the caller can update its own rows/state."""
        with self._lock:
            out, self._merge_log = self._merge_log, []
            return out

    def stats(self) -> dict:
        with self._lock:
            confirmed = sum(1 for p in self._people.values() if p.confirmed)
            return dict(self._stats, identities=len(self._people), confirmed=confirmed,
                        provisional=len(self._people) - confirmed)

    def unique_people_count(self) -> int:
        with self._lock:
            if self._require_confirmation:
                return sum(1 for p in self._people.values() if p.confirmed)
            return len(self._people)

    def is_counted(self, global_id: str) -> bool:
        """Whether this identity counts as a visitor (always, unless require_confirmation and not yet confirmed)."""
        with self._lock:
            person = self._people.get(global_id)
            return person is not None and (person.confirmed or not self._require_confirmation)

    def forget_local_track(self, camera_id: str, local_track_id: int) -> None:
        """Call when core.tracker.PersonTracker evicts a local track. Only
        forgets the local->global *mapping* — the GlobalPerson (and its
        embeddings) survives, so the same physical person can be re-matched
        later (same or a different camera) instead of minted as a brand new
        global identity every time a local track ID merely churns (spec
        section 13)."""
        with self._lock:
            self._local_to_global.pop((camera_id, local_track_id), None)
            self._low_streak.pop((camera_id, local_track_id), None)
            self._banned.pop((camera_id, local_track_id), None)
            self._conflict_streak = {k: v for k, v in self._conflict_streak.items()
                                     if k[0] != (camera_id, local_track_id)}

    def get_person(self, global_id: str) -> GlobalPerson | None:
        with self._lock:
            return self._people.get(global_id)

    def get_global_id_for(self, camera_id: str, local_track_id: int) -> str | None:
        with self._lock:
            return self._local_to_global.get((camera_id, local_track_id))

    def get_confirmed_global_id_for(self, camera_id: str, local_track_id: int) -> str | None:
        """Like get_global_id_for, but None until the identity is actually
        counted (see is_counted) — a track that Re-ID has only just matched
        (one observation, require_confirmation still pending) is a real
        person to the matcher internally, but must not be surfaced to
        anything a viewer/downstream consumer would read as "this is a
        confirmed customer's permanent ID" (spec: an UNKNOWN/not-yet-
        confirmed observation must never display or carry a confirmed
        person's identity). Use this instead of get_global_id_for wherever
        the ID is exposed outward — the on-screen box label — rather than
        used for the matcher's own internal bookkeeping."""
        with self._lock:
            gid = self._local_to_global.get((camera_id, local_track_id))
            if gid is None:
                return None
            person = self._people.get(gid)
            return gid if person is not None and (person.confirmed or not self._require_confirmation) else None
