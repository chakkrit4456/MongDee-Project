"""Whole-person gender evidence: body cues, the online model that learns them on the booth, and the fusion."""
import numpy as np
import cv2
import pytest

from core.attributes import GlobalPersonAttributeSmoother
from core.body_cues import BODY_FEATURE_DIM, COLOR_DESCRIPTOR_DIM, body_features, color_descriptor
from core.body_gender import BodyGenderLearner, BodyGenderModel, MIN_PER_CLASS


def synth_person(gender, rng, h=180, w=70):
    """A crude synthetic person with the kind of cue correlations a real booth has (80% typical, 20% not).
    This only exercises the learning MECHANISM; real-world accuracy is measured by the model's own stats."""
    img = np.full((h, w, 3), rng.integers(90, 170), dtype=np.uint8)
    img += rng.integers(0, 12, img.shape, dtype=np.uint8)
    skin = (int(rng.integers(120, 170)), int(rng.integers(140, 185)), int(rng.integers(190, 235)))   # BGR skin-ish
    hair = (25, 22, 20) if rng.random() < 0.85 else (60, 90, 130)
    typical = rng.random() < 0.8
    long_hair = (gender == "female") == typical
    skirt = (gender == "female") == (rng.random() < 0.8)
    cv2.ellipse(img, (w // 2, int(0.09 * h)), (int(0.16 * w), int(0.08 * h)), 0, 0, 360, skin, -1)
    cv2.ellipse(img, (w // 2, int(0.06 * h)), (int(0.19 * w), int(0.06 * h)), 0, 180, 360, hair, -1)
    if long_hair:
        cv2.rectangle(img, (int(0.2 * w), int(0.07 * h)), (int(0.3 * w), int(0.40 * h)), hair, -1)
        cv2.rectangle(img, (int(0.7 * w), int(0.07 * h)), (int(0.8 * w), int(0.40 * h)), hair, -1)
    torso = tuple(int(v) for v in rng.integers(30, 230, 3))
    cv2.rectangle(img, (int(0.15 * w), int(0.22 * h)), (int(0.85 * w), int(0.55 * h)), torso, -1)
    legs = skin if skirt else (int(rng.integers(20, 70)), int(rng.integers(20, 60)), int(rng.integers(30, 110)))
    cv2.rectangle(img, (int(0.25 * w), int(0.55 * h)), (int(0.75 * w), int(0.95 * h)), legs, -1)
    return img


def test_body_features_have_a_fixed_length_and_reject_unusable_crops():
    rng = np.random.default_rng(0)
    img = synth_person("male", rng)
    f = body_features(img)
    assert f.shape == (BODY_FEATURE_DIM,) and np.isfinite(f).all()
    assert body_features(cv2.resize(img, (30, 60))).shape == (BODY_FEATURE_DIM,)      # tiny far-away crop still works
    assert body_features(np.zeros((5, 5, 3), np.uint8)) is None
    assert body_features(None) is None
    assert color_descriptor(img).shape == (COLOR_DESCRIPTOR_DIM,)


def test_colour_descriptor_tells_clothing_colours_apart():
    def dressed(bgr_torso, bgr_legs):
        img = np.full((160, 60, 3), 128, np.uint8)
        img[int(0.22 * 160):int(0.55 * 160), 8:52] = bgr_torso
        img[int(0.55 * 160):int(0.95 * 160), 12:48] = bgr_legs
        return img
    red_blue = color_descriptor(dressed((30, 30, 220), (200, 60, 30)))
    red_blue_2 = color_descriptor(dressed((40, 35, 205), (190, 70, 40)))
    green_black = color_descriptor(dressed((40, 200, 40), (20, 20, 20)))
    cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos(red_blue, red_blue_2) > 0.9
    assert cos(red_blue, green_black) < 0.3


def _train(model_seed=0, n_people=80, frames=6, rng_seed=1):
    rng = np.random.default_rng(rng_seed)
    model = BodyGenderModel(BODY_FEATURE_DIM, seed=model_seed)
    learner = BodyGenderLearner(model)
    for i in range(n_people):
        g = "male" if i % 2 == 0 else "female"
        pid = f"P{i}"
        for _ in range(frames):
            learner.observe(pid, body_features(synth_person(g, rng)), None)      # far away: no face yet
        learner.observe(pid, body_features(synth_person(g, rng)), g)              # walks closer: face decides
    return model, learner


def test_model_is_off_until_it_has_learned_from_enough_people():
    model = BodyGenderModel(BODY_FEATURE_DIM)
    assert model.predict(np.zeros(BODY_FEATURE_DIM)) is None and not model.ready()
    rng = np.random.default_rng(0)
    learner = BodyGenderLearner(model)
    for i in range(6):
        learner.observe(f"p{i}", body_features(synth_person("male", rng)), "male")
    assert not model.ready()                       # no female examples at all


def test_far_frames_become_training_examples_once_the_face_decides():
    model = BodyGenderModel(BODY_FEATURE_DIM)
    learner = BodyGenderLearner(model)
    rng = np.random.default_rng(0)
    for _ in range(9):
        learner.observe("P1", body_features(synth_person("female", rng)), None)
    assert model.counts["female"] == 0
    learner.observe("P1", body_features(synth_person("female", rng)), "female")
    assert model.counts["female"] == 10             # the 9 earlier far frames + this one


def test_one_person_cannot_dominate_the_training_set():
    model = BodyGenderModel(BODY_FEATURE_DIM)
    learner = BodyGenderLearner(model)
    rng = np.random.default_rng(0)
    for _ in range(200):
        learner.observe("P1", body_features(synth_person("male", rng)), "male")
    assert model.counts["male"] <= 40


def test_the_model_learns_cues_on_unseen_people_and_reports_a_measured_accuracy():
    model, _ = _train()
    stats = model.stats()
    assert stats["ready"] and stats["prequential_accuracy"] >= 0.7
    rng = np.random.default_rng(99)                 # people the model has never seen
    hits = 0
    for i in range(100):
        g = "male" if i % 2 == 0 else "female"
        p, weight = model.predict(body_features(synth_person(g, rng)))
        hits += (p >= 0.5) == (g == "male")
        assert 0 < weight <= 1
    assert hits / 100 >= 0.7


def test_model_persists_and_a_changed_feature_layout_starts_over(tmp_path):
    model, _ = _train(n_people=50)
    path = tmp_path / "body.npz"
    model.save(path)
    again = BodyGenderModel.load(path, BODY_FEATURE_DIM)
    x = body_features(synth_person("male", np.random.default_rng(5)))
    assert again.counts == model.counts and again.predict(x)[0] == pytest.approx(model.predict(x)[0])
    assert BodyGenderModel.load(path, BODY_FEATURE_DIM + 3).n == 0
    assert BodyGenderModel.load(tmp_path / "missing.npz", BODY_FEATURE_DIM).n == 0


# ----------------------------------------------------------------------- fusion in the smoother
def _fed(n_face=0, face=("male", 0.9), n_body=0, body=(0.85, 0.8)):
    sm = GlobalPersonAttributeSmoother()
    t, result = 0.0, None
    for _ in range(n_face):
        result = sm.add_sample("P1", face[0], face[1], now=t); t += 1
    for _ in range(n_body):
        result = sm.add_body_sample("P1", body[0], body[1], now=t); t += 1
    return sm, result


def test_body_evidence_alone_labels_a_far_person_only_after_several_consistent_samples():
    _, r = _fed(n_body=3)
    assert r.gender == "UNKNOWN"
    _, r = _fed(n_body=6, body=(0.9, 0.9))
    assert r.gender == "MALE" and r.source == "body" and r.status == "ok"
    _, r = _fed(n_body=6, body=(0.1, 0.9))
    assert r.gender == "FEMALE"


def test_a_body_model_with_no_measured_skill_adds_nothing():
    _, r = _fed(n_body=20, body=(0.99, 0.0))
    assert r.gender == "UNKNOWN"


def test_face_and_body_evidence_add_up_and_report_both_sources():
    _, r = _fed(n_face=2, face=("female", 0.75), n_body=3, body=(0.2, 0.8))
    assert r.gender == "FEMALE" and r.source == "face+body"


def test_body_cannot_outvote_a_clear_face_history():
    sm, r = _fed(n_face=5, face=("male", 0.95), n_body=2, body=(0.05, 0.5))
    assert r.gender == "MALE"


def test_face_label_is_only_given_when_the_face_alone_is_decisive():
    sm, _ = _fed(n_face=2, face=("male", 0.9))
    assert sm.face_label("P1") is None
    sm, _ = _fed(n_face=4, face=("male", 0.9))
    assert sm.face_label("P1") == "male"
    sm, _ = _fed(n_body=30, body=(0.99, 1.0))
    assert sm.face_label("P1") is None              # body evidence never teaches the body model (no circularity)


def test_merging_identities_pools_their_evidence():
    sm = GlobalPersonAttributeSmoother()
    sm.add_sample("A", "female", 0.9, now=0.0)
    sm.add_sample("B", "female", 0.9, now=1.0)
    sm.add_sample("B", "female", 0.9, now=2.0)
    sm.merge("B", "A")
    r = sm.get("A", now=2.5)
    assert r.gender == "FEMALE" and r.sample_count == 3
    assert sm.get("B", now=2.5).status == "unknown"
