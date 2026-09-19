import cv2, numpy as np
from tests.core.camera_ai_fixtures import load
from core.face import FaceService, FaceBox, AttributeAggregator, face_quality
from core.face.service import BestShotStore


def _astro():
    return load("astronaut")   # 512x512, one clear face


def _yunet_path():
    import os
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "models", "face_detector", "face_detection_yunet_2023mar.onnx")


def _yunet(**kw):
    """YuNet (the repo's own model) is the reference detector. OpenCV 5 dropped
    cv2.CascadeClassifier, so Haar is NOT usable in this project's venv."""
    import os
    import pytest
    from core.face import YuNetFaceDetector
    if not os.path.exists(_yunet_path()) or not (hasattr(cv2, "FaceDetectorYN_create") or hasattr(cv2, "FaceDetectorYN")):
        pytest.skip("YuNet model / cv2.FaceDetectorYN not available")
    return YuNetFaceDetector(_yunet_path(), **kw)


def test_detects_real_face_in_full_frame():
    faces = _yunet().detect(_astro())
    assert len(faces) == 1 and faces[0].landmarks.shape == (5, 2)
    f = faces[0]
    assert 150 < f.center[0] < 300 and 20 < f.center[1] < 200      # astronaut face, upper-middle


def test_no_face_on_non_face_images():
    det = _yunet()
    for name in ("coffee", "rocket", "chelsea"):
        assert det.detect(load(name)) == [], name


def test_quality_prefers_sharp_large_over_blurred_small():
    frame = _astro(); f = _yunet().detect(frame)[0]
    q_good = face_quality(frame, f)
    q_blur = face_quality(cv2.GaussianBlur(frame, (21, 21), 8), f)
    small = FaceBox(f.x1, f.y1, f.x1 + 18, f.y1 + 18, 0.9)
    assert q_good > q_blur and q_good > face_quality(frame, small)
    assert 0.0 <= q_good <= 1.0


def test_face_bound_to_right_person_and_best_shot_kept():
    frame = _astro(); det = _yunet()
    f = det.detect(frame)[0]
    cx, cy = f.center
    tracks = [
        {"track_id": 1, "bbox": (cx - 120, cy - 60, cx + 120, cy + 400)},    # the astronaut
        {"track_id": 2, "bbox": (400, 300, 500, 500)},                         # someone else
        {"track_id": 3, "bbox": (cx - 300, cy - 300, cx + 300, cy + 600)},   # huge box also containing the face
    ]
    svc = FaceService(det)
    out = svc.process(frame, tracks)
    assert out and out[0].track_id == 1                  # smallest containing box wins
    assert svc.store.best(1) is not None and svc.store.best(2) is None
    q1 = svc.store.best_quality(1)
    svc.process(cv2.GaussianBlur(frame, (25, 25), 10), tracks)   # a later blurred frame must not replace the best shot
    assert svc.store.best_quality(1) >= q1


def test_face_in_lower_body_region_not_bound():
    svc = FaceService(_yunet())
    fb = FaceBox(10, 180, 40, 210, 0.9)
    assert svc._bind(fb, [{"track_id": 9, "bbox": (0, 0, 60, 220)}]) is None


def test_best_shot_store_keeps_top_k():
    s = BestShotStore(keep=2)
    for q in (0.2, 0.9, 0.5, 0.1, 0.7):
        s.offer(1, q, np.full((4, 4, 3), int(q * 100), np.uint8))
    assert s.best_quality(1) == 0.9
    assert len(s._shots[1]) == 2


def test_yunet_small_face_on_320x240_stream():
    """The camera runs at 320x240: report whether a ~30px face is found at upscale 1 vs 2.
    Requirement: upscale 2 must find it (this is why the integration guide upscales)."""
    big = _astro()
    f = _yunet().detect(big)[0]
    s = 30.0 / f.w
    small = cv2.resize(big, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    canvas = np.full((240, 320, 3), 90, np.uint8); canvas[:small.shape[0], :small.shape[1]] = small[:240, :320]
    n1 = len(_yunet(upscale=1.0).detect(canvas)); n2 = len(_yunet(upscale=2.0).detect(canvas))
    print("30px face: found at upscale1 =", n1, " upscale2 =", n2)
    assert n2 >= 1


# ---------------- attribute policy: simulation of the "man shown as woman" bug ---------------
def _sim(rng, truth_male, n, quality, acc):
    """classifier that is right with prob `acc` at this quality; confident-ish either way."""
    out = []
    for _ in range(n):
        correct = rng.random() < acc
        p_true = rng.uniform(0.6, 0.9)
        p_male = p_true if (truth_male == correct) else 1 - p_true
        out.append((quality, p_male))
    return out


def test_aggregation_beats_single_frame_and_abstains_on_bad_faces():
    rng = np.random.default_rng(0)
    n_tracks = 400; wrong_agg = wrong_single = unknown_good = 0
    for i in range(n_tracks):
        male = bool(i % 2)
        votes = _sim(rng, male, 12, quality=0.6, acc=0.82)      # small-face 320x240 regime
        agg = AttributeAggregator(); a = None
        for q, pm in votes: a = agg.add(i, q, p_male=pm)
        single = "male" if votes[0][1] > 0.5 else "female"
        wrong_single += single != ("male" if male else "female")
        if a.gender == "unknown": unknown_good += 1
        else: wrong_agg += a.gender != ("male" if male else "female")
    print("single-frame error", wrong_single / n_tracks, "aggregated error", wrong_agg / n_tracks, "unknown", unknown_good / n_tracks)
    assert wrong_single / n_tracks > 0.10                        # baseline behaves like the reported bug
    assert wrong_agg / n_tracks < 0.03
    assert unknown_good / n_tracks < 0.10


def test_low_quality_only_never_labels():
    agg = AttributeAggregator()
    for _ in range(50):
        a = agg.add(1, 0.1, p_male=0.2)          # confident-female but from garbage face
    assert a.gender == "unknown" and a.n_gender_votes == 0


def test_needs_min_votes_and_hysteresis_prevents_flicker():
    agg = AttributeAggregator()
    a = agg.add(1, 0.8, p_male=0.9); assert a.gender == "unknown"     # 1 vote
    for _ in range(4): a = agg.add(1, 0.8, p_male=0.9)
    assert a.gender == "male"
    for _ in range(2): a = agg.add(1, 0.8, p_male=0.1)               # short contrary burst
    assert a.gender == "male"
    for _ in range(40): a = agg.add(1, 0.8, p_male=0.05)             # sustained contrary evidence
    assert a.gender == "female"
