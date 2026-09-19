"""Optional CLIP-based whole-body gender evidence for people too far away for a readable face.

Why: the hand-crafted body cues in core/body_cues.py start with no knowledge and need dozens of close-up people
before they help. A CLIP image encoder already "knows" what men and women, their hair, clothes and build look
like, so from the very first person it gives a usable zero-shot opinion ("a photo of a man" vs "a photo of a
woman"), and its 512-number embedding is a far better feature for the online learner than colour histograms.

How it plugs in (nothing else changes): `ClipGenderBackend.features(person_crop)` returns one vector
[zero-shot log-odds, embedding...]; `ClipBodyGenderModel` turns it into P(male) as
    logit = a * zero_shot + w . embedding + b
where a, w, b are learned online from people whose FACE has already decided (core/body_gender.py's
BodyGenderLearner does the labelling, exactly as for the hand-crafted model). Its accuracy is measured on people
it has not learned yet (prequential test) and its evidence is scaled by that measured accuracy - if CLIP turns
out useless on this booth's cameras the measured accuracy says so and it contributes nothing.

Opt-in (the project never downloads third-party model files by itself): `pip install open_clip_torch`, then set
MONGDEE_CLIP_GENDER=1 (optionally MONGDEE_CLIP_MODEL=ViT-B-32 and MONGDEE_CLIP_PRETRAINED=openai). The first run
downloads the CLIP weights (~350 MB) once. Any failure to load simply leaves the hand-crafted model in charge.
"""
from __future__ import annotations

import collections
import logging
import os
import threading
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger("mongdee.core.clip_gender")

PROMPTS_MALE = ("a photo of a man", "a photo of a male person", "a full body photo of a man",
                "a surveillance camera image of a man walking")
PROMPTS_FEMALE = ("a photo of a woman", "a photo of a female person", "a full body photo of a woman",
                  "a surveillance camera image of a woman walking")
ZERO_SHOT_SCALE = 100.0        # CLIP's own logit scale
MIN_CROP_SIDE_PX = 24
PAD_COLOR = (124, 116, 104)    # ~ImageNet mean, BGR

MIN_PER_CLASS = 4              # labelled examples of each gender before the model may be used
MIN_PREQUENTIAL = 12           # honest test predictions before the measured accuracy is believed
MIN_ACCURACY_LOWER_BOUND = 0.55    # Wilson lower bound of the measured accuracy must beat chance by this much
PREQUENTIAL_WINDOW = 200
BUFFER_SIZE = 800
L2 = 5e-2
Z_CLIP = 4.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _wilson_lower(successes: float, n: int, z: float = 1.28) -> float:
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre - margin) / denom)


def pad_to_square(crop_bgr: np.ndarray) -> np.ndarray:
    """A person crop is tall and narrow; CLIP's own pre-processing centre-crops to a square and would cut off the
    head or the legs. Pad it to a square first so the whole person is seen."""
    h, w = crop_bgr.shape[:2]
    side = max(h, w)
    out = np.empty((side, side, 3), dtype=np.uint8)
    out[:] = PAD_COLOR
    y0, x0 = (side - h) // 2, (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = crop_bgr
    return out


class ClipGenderBackend:
    """Turns a person crop into [zero-shot log-odds, unit-norm embedding]. The two encoders are injected so the
    numpy logic is testable without CLIP; `from_open_clip` builds the real ones."""

    def __init__(self, encode_image: Callable[[np.ndarray], np.ndarray],
                 encode_texts: Callable[[list], np.ndarray]):
        self._encode_image = encode_image
        male = self._unit(np.mean([self._unit(v) for v in encode_texts(list(PROMPTS_MALE))], axis=0))
        female = self._unit(np.mean([self._unit(v) for v in encode_texts(list(PROMPTS_FEMALE))], axis=0))
        self._direction = male - female                      # image . direction > 0  ->  looks male
        self.embedding_dim = int(male.shape[0])
        self.dim = 1 + self.embedding_dim

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def features(self, crop_bgr: "np.ndarray | None") -> "np.ndarray | None":
        if crop_bgr is None or crop_bgr.ndim != 3 or min(crop_bgr.shape[:2]) < MIN_CROP_SIDE_PX:
            return None
        emb = self._unit(self._encode_image(pad_to_square(crop_bgr)))
        zero_shot = ZERO_SHOT_SCALE * float(emb @ self._direction)
        return np.concatenate([[zero_shot], emb]).astype(np.float32)

    @classmethod
    def from_open_clip(cls, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        """The real thing. Raises if open_clip / the weights are unavailable (the caller falls back)."""
        import open_clip
        import torch
        from PIL import Image

        model, _train_tf, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model.eval()
        try:
            model.to(device)
        except Exception:
            device = "cpu"
            model.to(device)
        tokenizer = open_clip.get_tokenizer(model_name)

        def encode_image(square_bgr: np.ndarray) -> np.ndarray:
            pil = Image.fromarray(cv2.cvtColor(square_bgr, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                feats = model.encode_image(preprocess(pil).unsqueeze(0).to(device))
            return feats[0].float().cpu().numpy()

        def encode_texts(prompts: list) -> np.ndarray:
            with torch.no_grad():
                feats = model.encode_text(tokenizer(prompts).to(device))
            return feats.float().cpu().numpy()

        logger.info("CLIP gender backend ready: %s/%s on %s", model_name, pretrained, device)
        return cls(encode_image, encode_texts)


def clip_gender_requested() -> bool:
    return os.environ.get("MONGDEE_CLIP_GENDER", "").strip().lower() in ("1", "true", "yes", "on")


def try_load_backend(device: str = "cpu") -> "ClipGenderBackend | None":
    """The CLIP backend when it was requested (MONGDEE_CLIP_GENDER) and can be loaded, else None."""
    if not clip_gender_requested():
        return None
    try:
        return ClipGenderBackend.from_open_clip(
            os.environ.get("MONGDEE_CLIP_MODEL", "ViT-B-32"), os.environ.get("MONGDEE_CLIP_PRETRAINED", "openai"), device)
    except Exception as exc:
        logger.warning("MONGDEE_CLIP_GENDER is set but CLIP could not be loaded (%s: %s) - using the hand-crafted "
                       "body model instead. Install with: pip install open_clip_torch", type(exc).__name__, exc)
        return None


class ClipBodyGenderModel:
    """P(male) = sigmoid(a * zero_shot + w . standardised_embedding + b), trained online. Same interface as
    core.body_gender.BodyGenderModel (predict / partial_fit / ready / stats / save / load), so BodyGenderLearner
    drives it unchanged. Unlike that model it is useful from the first examples because of the zero-shot term;
    it is only *used* once its measured accuracy on unseen people is credibly better than chance."""

    def __init__(self, embedding_dim: int, seed: int = 0):
        self.embedding_dim = embedding_dim
        self.dim = 1 + embedding_dim
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self.a, self.b = 1.0, 0.0
        self.w = np.zeros(embedding_dim)
        self.mu, self.m2, self.n = np.zeros(embedding_dim), np.zeros(embedding_dim), 0
        self.counts = {"male": 0, "female": 0}
        self.updates = 0
        self._buf_x: collections.deque = collections.deque(maxlen=BUFFER_SIZE)
        self._buf_y: collections.deque = collections.deque(maxlen=BUFFER_SIZE)
        self._preq: collections.deque = collections.deque(maxlen=PREQUENTIAL_WINDOW)

    # ------------------------------------------------------------------ maths
    def _std(self):
        return np.sqrt(np.maximum(self.m2 / max(self.n - 1, 1), 1e-8))

    def _z(self, emb):
        return np.clip((emb - self.mu) / self._std(), -Z_CLIP, Z_CLIP)

    def _logit(self, x):
        x = np.asarray(x, dtype=float)
        return float(self.a * x[0] + self.w @ self._z(x[1:]) + self.b)

    # ------------------------------------------------------------------ status
    def accuracy(self) -> "float | None":
        with self._lock:
            return float(np.mean(self._preq)) if len(self._preq) >= MIN_PREQUENTIAL else None

    def ready(self) -> bool:
        with self._lock:
            n = len(self._preq)
            return (self.counts["male"] >= MIN_PER_CLASS and self.counts["female"] >= MIN_PER_CLASS
                    and n >= MIN_PREQUENTIAL and _wilson_lower(float(sum(self._preq)), n) >= MIN_ACCURACY_LOWER_BOUND)

    def competence(self) -> float:
        acc = self.accuracy()
        return 0.0 if acc is None else float(np.clip((acc - 0.5) * 2.0, 0.0, 1.0))

    def stats(self) -> dict:
        with self._lock:
            acc = self.accuracy()
            return {"kind": "clip", "ready": self.ready(), "male_examples": self.counts["male"],
                    "female_examples": self.counts["female"],
                    "prequential_accuracy": None if acc is None else round(acc, 3),
                    "test_predictions": len(self._preq), "updates": self.updates, "zero_shot_weight": round(self.a, 3)}

    def predict(self, x) -> "tuple[float, float] | None":
        with self._lock:
            if not self.ready():
                return None
            return float(_sigmoid(self._logit(x))), self.competence()

    # ------------------------------------------------------------------ learning
    def partial_fit(self, x, label: str, steps: int = 6, lr: float = 0.05, test: bool = True) -> None:
        if label not in ("male", "female"):
            return
        y = 1.0 if label == "male" else 0.0
        x = np.asarray(x, dtype=float)
        with self._lock:
            if test:                                                   # honest test BEFORE learning it
                self._preq.append(float((self._logit(x) >= 0.0) == (y == 1.0)))
            emb = x[1:]
            self.n += 1
            delta = emb - self.mu
            self.mu = self.mu + delta / self.n
            self.m2 = self.m2 + delta * (emb - self.mu)
            self.counts[label] += 1
            self._buf_x.append(x)
            self._buf_y.append(y)
            Y = np.asarray(self._buf_y)
            if len(Y) < 6:
                return
            X = np.stack(self._buf_x)
            n_pos, n_neg = max(Y.sum(), 1.0), max((1 - Y).sum(), 1.0)
            sw = np.where(Y == 1.0, 0.5 / n_pos, 0.5 / n_neg) * len(Y)
            Z = np.clip((X[:, 1:] - self.mu) / self._std(), -Z_CLIP, Z_CLIP)
            for _ in range(steps):
                idx = self._rng.integers(0, len(Y), size=min(64, len(Y)))
                p = _sigmoid(self.a * X[idx, 0] + Z[idx] @ self.w + self.b)
                err = (p - Y[idx]) * sw[idx]
                self.w -= lr * (Z[idx].T @ err / len(idx) + L2 * self.w)
                self.b -= lr * float(err.mean())
                self.a = float(np.clip(self.a - lr * 0.3 * float((err * X[idx, 0]).mean()), 0.2, 3.0))
            self.updates += 1

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        with self._lock:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, kind="clip", dim=self.dim, a=self.a, b=self.b, w=self.w, mu=self.mu, m2=self.m2, n=self.n,
                     counts=np.asarray([self.counts["male"], self.counts["female"]]), updates=self.updates,
                     bx=np.stack(self._buf_x) if self._buf_x else np.zeros((0, self.dim)),
                     by=np.asarray(self._buf_y, dtype=float), preq=np.asarray(self._preq, dtype=float))
            tmp.replace(path)

    @classmethod
    def load(cls, path: Path, embedding_dim: int) -> "ClipBodyGenderModel":
        model = cls(embedding_dim)
        path = Path(path)
        if not path.is_file():
            return model
        try:
            with np.load(path, allow_pickle=False) as d:
                if int(d["dim"]) != model.dim:
                    return model          # a different CLIP model: start over
                model.a, model.b, model.w = float(d["a"]), float(d["b"]), d["w"]
                model.mu, model.m2, model.n = d["mu"], d["m2"], int(d["n"])
                model.counts = {"male": int(d["counts"][0]), "female": int(d["counts"][1])}
                model.updates = int(d["updates"])
                model._buf_x.extend(list(d["bx"]))
                model._buf_y.extend(list(d["by"]))
                model._preq.extend(list(d["preq"]))
        except Exception:
            return cls(embedding_dim)
        return model
