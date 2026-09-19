"""Whole-person gender evidence that keeps working when the face is too small to read.

At a distance a 320x240 stream gives a face of 15-30 px, which no face model can classify. What remains is the
whole person: hair length, clothing, exposed skin, build. There is no labelled whole-body dataset shipped with
MongDee, so this module LEARNS ON THE BOOTH ITSELF:

  * whenever a person is close enough for the face model to be decisive (evidence from the face alone), the
    person's whole-body feature vectors - including the earlier, far-away frames of the same person - become
    labelled training examples;
  * a small online logistic model (numpy only) learns how hair/clothing/build/CNN-appearance features relate to
    that label; it is saved to disk and keeps improving for as long as the booth runs;
  * its own competence is MEASURED, not assumed: before each new labelled example is learned the model first
    predicts it (prequential accuracy). The body evidence handed to the gender fusion is scaled by that measured
    accuracy, and is switched off until it beats chance by a margin.

So on day one it contributes nothing (and the face model decides alone); after enough close-up people it starts
helping at range, and `stats()` shows how good it currently is.
"""
from __future__ import annotations

import collections
import threading
from pathlib import Path

import numpy as np

MIN_PER_CLASS = 20            # labelled examples of EACH gender before the model may be used at all
MIN_PREQUENTIAL = 40          # honest test predictions needed before trusting the measured accuracy
MIN_ACCURACY = 0.62           # measured accuracy needed to switch on
PREQUENTIAL_WINDOW = 200
BUFFER_SIZE = 600
MAX_PER_PERSON = 40           # one person must not dominate the training set
L2 = 2e-2
Z_CLIP = 4.0


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class BodyGenderModel:
    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._lock = threading.RLock()
        self.mu = np.zeros(dim)
        self.m2 = np.zeros(dim)
        self.n = 0
        self.w = np.zeros(dim)
        self.b = 0.0
        self.counts = {"male": 0, "female": 0}
        self.updates = 0
        self._buf_x: collections.deque = collections.deque(maxlen=BUFFER_SIZE)
        self._buf_y: collections.deque = collections.deque(maxlen=BUFFER_SIZE)
        self._preq: collections.deque = collections.deque(maxlen=PREQUENTIAL_WINDOW)

    # ------------------------------------------------------------------ maths
    def _std(self):
        var = self.m2 / max(self.n - 1, 1)
        return np.sqrt(np.maximum(var, 1e-6))

    def _z(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mu) / self._std(), -Z_CLIP, Z_CLIP)

    def _p_male_locked(self, x: np.ndarray) -> float:
        return float(_sigmoid(float(self.w @ self._z(x)) + self.b))

    # ------------------------------------------------------------------ status
    def accuracy(self) -> float | None:
        with self._lock:
            if len(self._preq) < MIN_PREQUENTIAL:
                return None
            return float(np.mean(self._preq))

    def ready(self) -> bool:
        with self._lock:
            acc = self.accuracy()
            return (self.counts["male"] >= MIN_PER_CLASS and self.counts["female"] >= MIN_PER_CLASS
                    and acc is not None and acc >= MIN_ACCURACY)

    def competence(self) -> float:
        """0..1 weight for the fusion: 0 at chance level, 1 at 100% measured accuracy."""
        acc = self.accuracy()
        return 0.0 if acc is None else float(np.clip((acc - 0.5) * 2.0, 0.0, 1.0))

    def stats(self) -> dict:
        with self._lock:
            acc = self.accuracy()
            return {"ready": self.ready(), "male_examples": self.counts["male"], "female_examples": self.counts["female"],
                    "prequential_accuracy": None if acc is None else round(acc, 3),
                    "test_predictions": len(self._preq), "updates": self.updates}

    # ------------------------------------------------------------------ inference
    def predict(self, x: np.ndarray) -> tuple[float, float] | None:
        """(p_male, competence weight) or None while the model is not trustworthy yet."""
        with self._lock:
            if not self.ready():
                return None
            return self._p_male_locked(np.asarray(x, dtype=float)), self.competence()

    # ------------------------------------------------------------------ learning
    def partial_fit(self, x: np.ndarray, label: str, steps: int = 6, lr: float = 0.08, test: bool = True) -> None:
        y = 1.0 if label == "male" else 0.0
        if label not in ("male", "female"):
            return
        x = np.asarray(x, dtype=float)
        with self._lock:
            if test and self.n >= 10:                                  # honest test BEFORE learning it
                self._preq.append(float((self._p_male_locked(x) >= 0.5) == (y == 1.0)))
            self.n += 1
            delta = x - self.mu
            self.mu = self.mu + delta / self.n
            self.m2 = self.m2 + delta * (x - self.mu)
            self.counts[label] += 1
            self._buf_x.append(x)
            self._buf_y.append(y)
            Y = np.asarray(self._buf_y)
            if len(Y) < 8:
                return
            X = np.stack(self._buf_x)
            n_pos, n_neg = max(Y.sum(), 1.0), max((1 - Y).sum(), 1.0)
            sw = np.where(Y == 1.0, 0.5 / n_pos, 0.5 / n_neg) * len(Y)      # class-balanced
            Z = np.clip((X - self.mu) / self._std(), -Z_CLIP, Z_CLIP)
            for _ in range(steps):
                idx = self._rng.integers(0, len(Y), size=min(64, len(Y)))
                p = _sigmoid(Z[idx] @ self.w + self.b)
                err = (p - Y[idx]) * sw[idx]
                grad_w = Z[idx].T @ err / len(idx) + L2 * self.w
                self.w -= lr * grad_w
                self.b -= lr * float(err.mean())
            self.updates += 1

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        with self._lock:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, dim=self.dim, mu=self.mu, m2=self.m2, n=self.n, w=self.w, b=self.b,
                     counts=np.asarray([self.counts["male"], self.counts["female"]]), updates=self.updates,
                     bx=np.stack(self._buf_x) if self._buf_x else np.zeros((0, self.dim)),
                     by=np.asarray(self._buf_y, dtype=float), preq=np.asarray(self._preq, dtype=float))
            tmp.replace(path)

    @classmethod
    def load(cls, path: Path, dim: int) -> "BodyGenderModel":
        model = cls(dim)
        path = Path(path)
        if not path.is_file():
            return model
        try:
            with np.load(path) as d:
                if int(d["dim"]) != dim:
                    return model          # feature layout changed: start over rather than mis-apply old weights
                model.mu, model.m2, model.n = d["mu"], d["m2"], int(d["n"])
                model.w, model.b, model.updates = d["w"], float(d["b"]), int(d["updates"])
                model.counts = {"male": int(d["counts"][0]), "female": int(d["counts"][1])}
                model._buf_x.extend(list(d["bx"])); model._buf_y.extend(list(d["by"]))
                model._preq.extend(list(d["preq"]))
        except Exception:
            return cls(dim)
        return model


class BodyGenderLearner:
    """Glue between per-person sightings and the model: keeps each person's recent feature vectors so that when
    the face finally decides who they are (they walked closer), their earlier far-away frames become labelled
    examples too."""

    PENDING_PER_PERSON = 24

    def __init__(self, model: BodyGenderModel, save_path: Path | None = None, autosave_every: int = 40):
        self.model = model
        self._save_path = Path(save_path) if save_path else None
        self._autosave_every = autosave_every
        self._pending: dict[str, collections.deque] = {}
        self._label: dict[str, str] = {}
        self._used: dict[str, int] = {}
        self._since_save = 0
        self._lock = threading.Lock()

    def observe(self, person_id: str, features: np.ndarray | None, face_label: str | None) -> None:
        """features: body feature vector of this sighting (or None). face_label: 'male'/'female' ONLY when the face
        alone is decisive for this person, else None."""
        if features is None:
            return
        with self._lock:
            pend = self._pending.setdefault(person_id, collections.deque(maxlen=self.PENDING_PER_PERSON))
            pend.append(np.asarray(features, dtype=float))
            if face_label in ("male", "female"):
                if self._label.get(person_id) not in (None, face_label):
                    self._used[person_id] = 0            # the label changed: what we learned may be wrong, stop here
                self._label[person_id] = face_label
            label = self._label.get(person_id)
            if label is None:
                return
            while pend and self._used.get(person_id, 0) < MAX_PER_PERSON:
                # Only the first sightings of a person count as accuracy tests: later ones come from someone the
                # model has already seen, so predicting them would only measure memorised clothes.
                self.model.partial_fit(pend.popleft(), label, test=self._used.get(person_id, 0) < 3)
                self._used[person_id] = self._used.get(person_id, 0) + 1
                self._since_save += 1
            pend.clear()
            if self._save_path is not None and self._since_save >= self._autosave_every:
                self._since_save = 0
                try:
                    self.model.save(self._save_path)
                except OSError:
                    pass

    def merge(self, loser: str, winner: str) -> None:
        with self._lock:
            pend = self._pending.pop(loser, None)
            if pend:
                self._pending.setdefault(winner, collections.deque(maxlen=self.PENDING_PER_PERSON)).extend(pend)
            lab = self._label.pop(loser, None)
            if lab and winner not in self._label:
                self._label[winner] = lab
            self._used.pop(loser, None)

    def forget(self, person_id: str) -> None:
        with self._lock:
            for d in (self._pending, self._label, self._used):
                d.pop(person_id, None)

    def flush(self) -> None:
        if self._save_path is not None:
            try:
                self.model.save(self._save_path)
            except OSError:
                pass
