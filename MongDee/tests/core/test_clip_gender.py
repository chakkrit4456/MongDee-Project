"""core/clip_gender.py: zero-shot prior + online probe, measured before it is trusted (fake encoders, no CLIP)."""
from __future__ import annotations

import numpy as np
import pytest

from core import clip_gender as cg
from core.body_gender import BodyGenderLearner

D = 32


def _unit(v):
    return v / np.linalg.norm(v)


RNG = np.random.default_rng(3)
MALE_DIR = _unit(RNG.normal(size=D))
FEMALE_DIR = _unit(MALE_DIR * -0.4 + _unit(RNG.normal(size=D)) * 0.9)
HIDDEN = _unit(np.random.default_rng(9).normal(size=D))         # a direction that also separates the genders


def _text_encoder(prompts):
    out = []
    for i, p in enumerate(prompts):
        base = MALE_DIR if ("man" in p or " male" in p) and "woman" not in p else FEMALE_DIR
        out.append(base + 0.01 * np.random.default_rng(i).normal(size=D))
    return np.stack(out)


class _ImageEncoder:
    """The 'image' encoder reads the crop's first pixel as an id and returns the embedding registered for it."""

    def __init__(self):
        self.table = {}

    def __call__(self, square_bgr):
        return self.table[int(square_bgr[0, 0, 0])] if int(square_bgr[0, 0, 0]) in self.table else self.table["default"]


def _backend(img=None):
    return cg.ClipGenderBackend(img or _ImageEncoder(), _text_encoder)


# ------------------------------------------------------------------------------------------ backend
def test_pad_to_square_keeps_the_whole_person():
    crop = np.full((100, 40, 3), 7, np.uint8)
    sq = cg.pad_to_square(crop)
    assert sq.shape == (100, 100, 3)
    assert (sq[:, 30:70] == 7).all() and tuple(sq[0, 0]) == cg.PAD_COLOR


def test_zero_shot_points_the_right_way_and_features_have_the_documented_layout():
    enc = _ImageEncoder()
    enc.table["default"] = MALE_DIR
    b = _backend(enc)
    f = b.features(np.full((80, 40, 3), 1, np.uint8))
    assert f.shape == (1 + D,) and b.dim == 1 + D and b.embedding_dim == D
    assert f[0] > 5
    assert np.linalg.norm(f[1:]) == pytest.approx(1.0, abs=1e-5)
    enc.table["default"] = FEMALE_DIR
    assert b.features(np.full((80, 40, 3), 1, np.uint8))[0] < -5


def test_a_tiny_or_missing_crop_gives_no_features():
    b = _backend()
    assert b.features(None) is None
    assert b.features(np.zeros((10, 10, 3), np.uint8)) is None


# --------------------------------------------------------------------------------------- the model
def _person(rng, truth, zs_acc=0.8, signal=1.0):
    """One person's features: a zero-shot log-odds that is right with probability zs_acc, and an embedding with the
    HIDDEN direction carrying `signal` of the gender."""
    sign = 1.0 if truth == "male" else -1.0
    zs = sign * abs(rng.normal(3.0, 1.0)) * (1.0 if rng.random() < zs_acc else -1.0)
    emb = _unit(rng.normal(size=D) * 0.5 + sign * signal * HIDDEN)
    return np.concatenate([[zs], emb])


def _feed(model, n_people, rng, **kw):
    learner = BodyGenderLearner(model)
    for i in range(n_people):
        truth = "male" if i % 2 == 0 else "female"
        for _ in range(3):
            learner.observe(f"p{i}", _person(rng, truth, **kw), truth)
    return learner


def test_a_cold_start_model_is_not_trusted_yet():
    m = cg.ClipBodyGenderModel(D)
    assert m.ready() is False and m.predict(np.zeros(1 + D)) is None


def test_it_becomes_ready_from_a_good_prior_after_a_handful_of_people_and_is_right():
    rng = np.random.default_rng(1)
    m = cg.ClipBodyGenderModel(D)
    _feed(m, 24, rng)
    assert m.ready() and m.accuracy() >= 0.7
    right = 0
    for i in range(200):
        truth = "male" if i % 2 == 0 else "female"
        p, weight = m.predict(_person(rng, truth))
        right += (p >= 0.5) == (truth == "male")
        assert 0.0 < weight <= 1.0
    assert right / 200 >= 0.8


def test_the_probe_learns_beyond_a_useless_prior():
    rng = np.random.default_rng(2)
    m = cg.ClipBodyGenderModel(D)
    _feed(m, 120, rng, zs_acc=0.5, signal=1.6)               # zero-shot is a coin flip; the embedding carries it
    assert m.ready()
    right = 0
    for i in range(300):
        truth = "male" if i % 2 == 0 else "female"
        right += (m.predict(_person(rng, truth, zs_acc=0.5, signal=1.6))[0] >= 0.5) == (truth == "male")
    assert right / 300 >= 0.85


def test_a_model_that_is_no_better_than_chance_never_switches_on():
    rng = np.random.default_rng(4)
    m = cg.ClipBodyGenderModel(D)
    _feed(m, 100, rng, zs_acc=0.5, signal=0.0)               # no information anywhere
    assert m.ready() is False and m.predict(_person(rng, "male")) is None


def test_persistence_roundtrip_and_a_different_clip_model_resets(tmp_path):
    rng = np.random.default_rng(5)
    m = cg.ClipBodyGenderModel(D)
    _feed(m, 24, rng)
    path = tmp_path / "clip.npz"
    m.save(path)
    m2 = cg.ClipBodyGenderModel.load(path, D)
    x = _person(rng, "male")
    assert m2.ready() == m.ready() and m2.counts == m.counts
    assert m2.predict(x)[0] == pytest.approx(m.predict(x)[0])
    other = cg.ClipBodyGenderModel.load(path, D + 8)         # e.g. ViT-L/14 instead of ViT-B/32
    assert other.counts == {"male": 0, "female": 0}


def test_stats_are_reported_for_the_dashboard():
    s = cg.ClipBodyGenderModel(D).stats()
    assert s["kind"] == "clip" and s["ready"] is False and s["male_examples"] == 0


# ------------------------------------------------------------------------------------ opt-in loading
def test_clip_is_never_loaded_unless_requested(monkeypatch):
    monkeypatch.delenv("MONGDEE_CLIP_GENDER", raising=False)
    assert cg.try_load_backend() is None


def test_a_missing_open_clip_falls_back_quietly(monkeypatch, caplog):
    monkeypatch.setenv("MONGDEE_CLIP_GENDER", "1")
    monkeypatch.setitem(__import__("sys").modules, "open_clip", None)          # import raises ImportError
    assert cg.try_load_backend() is None
    assert "open_clip_torch" in caplog.text
