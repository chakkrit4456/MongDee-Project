"""Unit tests for core/vision.py's gender/age-based person box coloring.

CameraWorker itself does real camera I/O and runs on a background thread,
so it isn't unit-testable end to end — but _person_label_and_color is pure
enough to call directly (construction alone never opens a camera or starts
the thread) with a fake gender_age_backend implementing the same duck-typed
interface vision.attributes.extractor.GenderAgeBackend defines.
"""

from __future__ import annotations

import numpy as np

from core.vision import (
    CUSTOM_RECOGNITION_BASE_FLOOR,
    MIN_GENDER_CONFIDENCE,
    PERSON_FEMALE_COLOR,
    PERSON_MALE_COLOR,
    PERSON_UNKNOWN_COLOR,
    CameraWorker,
)


class _FakeModel:
    names = {0: "person"}


class _FakeGenderBackend:
    def __init__(self, label, confidence):
        self.label = label
        self.confidence = confidence

    def predict_gender(self, crop):
        return self.label, self.confidence


class _BoomGenderBackend:
    def predict_gender(self, crop):
        raise RuntimeError("model exploded")


class _FakeGenderAgeBackend:
    """Independently-controllable gender + age_group predictions, for
    exercising the CHILD-overrides-gender priority in
    _person_label_and_color."""

    def __init__(self, gender_label="unknown", gender_conf=0.0,
                 age_group="unknown", age_conf=0.0):
        self.gender_label = gender_label
        self.gender_conf = gender_conf
        self.age_group = age_group
        self.age_conf = age_conf

    def predict_gender(self, crop):
        return self.gender_label, self.gender_conf

    def predict_age_group(self, crop):
        return self.age_group, self.age_conf


def _make_worker(gender_age_backend=None):
    # Constructing a CameraWorker never opens a camera or starts its thread
    # (that only happens on .start()), so this is safe/cheap in a unit test.
    return CameraWorker(
        camera_id="TEST", device=0, model=_FakeModel(), allowed_classes=[],
        gender_age_backend=gender_age_backend,
    )


def _frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_no_backend_keeps_neutral_person_color():
    worker = _make_worker(gender_age_backend=None)
    label, color = worker._person_label_and_color(_frame(), [10, 10, 50, 90], 0.9)
    assert color == PERSON_UNKNOWN_COLOR
    assert label.startswith("UNKNOWN")


def test_confident_female_prediction_colors_box_red():
    worker = _make_worker(gender_age_backend=_FakeGenderBackend("female", 0.9))
    label, color = worker._person_label_and_color(_frame(), [10, 10, 50, 90], 0.9)
    assert color == PERSON_FEMALE_COLOR
    assert "FEMALE" in label


def test_confident_male_prediction_colors_box_blue():
    worker = _make_worker(gender_age_backend=_FakeGenderBackend("male", 0.9))
    label, color = worker._person_label_and_color(_frame(), [10, 10, 50, 90], 0.9)
    assert color == PERSON_MALE_COLOR
    assert "MALE" in label


def test_low_confidence_prediction_stays_neutral():
    worker = _make_worker(gender_age_backend=_FakeGenderBackend("female", MIN_GENDER_CONFIDENCE - 0.01))
    label, color = worker._person_label_and_color(_frame(), [10, 10, 50, 90], 0.9)
    assert color == PERSON_UNKNOWN_COLOR


def test_unrecognized_label_stays_neutral():
    worker = _make_worker(gender_age_backend=_FakeGenderBackend("unknown", 0.99))
    label, color = worker._person_label_and_color(_frame(), [10, 10, 50, 90], 0.9)
    assert color == PERSON_UNKNOWN_COLOR


def test_backend_exception_falls_back_to_neutral_instead_of_crashing():
    worker = _make_worker(gender_age_backend=_BoomGenderBackend())
    label, color = worker._person_label_and_color(_frame(), [10, 10, 50, 90], 0.9)
    assert color == PERSON_UNKNOWN_COLOR


def test_child_detection_was_removed_a_child_age_group_never_makes_a_child_box():
    # Product decision: only male / female / unknown are reported. Even a backend that is very
    # sure the person is a child (for either bucket naming scheme) must not produce a CHILD label
    # or colour; the gender guess is used as-is.
    for age_group in ("4-6", "0-2", "3-9", "10-19"):
        backend = _FakeGenderAgeBackend(gender_label="male", gender_conf=0.95, age_group=age_group, age_conf=0.99)
        backend.CHILD_AGE_GROUPS = {"0-2", "3-9", "10-19", "4-6"}   # legacy backends still expose this
        worker = _make_worker(gender_age_backend=backend)
        label, color = worker._person_label_and_color(_frame(), [10, 10, 50, 90], 0.9)
        assert color == PERSON_MALE_COLOR
        assert "CHILD" not in label


def test_age_model_is_never_even_called():
    class _Explodes(_FakeGenderAgeBackend):
        def predict_age_group(self, crop):
            raise AssertionError("age must not be analysed any more")

    worker = _make_worker(gender_age_backend=_Explodes(
        gender_label="female", gender_conf=0.9, age_group="4-6", age_conf=0.9))
    label, color = worker._person_label_and_color(_frame(), [10, 10, 50, 90], 0.9)
    assert color == PERSON_FEMALE_COLOR


def test_only_male_female_unknown_and_product_have_box_colours():
    import core.vision as vision
    assert set(vision.BOX_COLORS) == {"male", "female", "unknown", "product"}
    assert not hasattr(vision, "PERSON_CHILD_COLOR")
    assert CameraWorker._category_label_and_color("child", 0.9)[0].startswith("UNKNOWN")


def test_gender_confidence_floor_is_strict_enough_to_reject_a_coin_flip():
    assert MIN_GENDER_CONFIDENCE >= 0.7


def test_frame_sequence_starts_below_zero_and_tracks_captures():
    worker = _make_worker()
    assert worker.get_frame_sequence() == -1   # nothing captured yet
    worker._latest_capture_seq = 42
    assert worker.get_frame_sequence() == 42


# --------------------------------------------------------------- deleted COCO-named product
class _Scalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _Row:
    def __init__(self, v):
        self._v = v

    def tolist(self):
        return list(self._v)


class _Box:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = _Scalar(cls), _Scalar(conf), [_Row(xyxy)]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class _CocoModel:
    names = {0: "person", 1: "bottle"}

    def predict(self, **kwargs):
        return [_Result([_Box(1, 0.9, [10.0, 10.0, 50.0, 90.0]), _Box(0, 0.9, [60.0, 10.0, 90.0, 90.0])])]


def _coco_worker(is_product_live):
    return CameraWorker(camera_id="T", device=0, model=_CocoModel(), allowed_classes=["bottle"],
                        is_product_live=is_product_live)


def test_coco_named_product_is_reported_while_it_is_in_the_catalog():
    draw = []
    _p, _c, _f, legacy, claimed = _coco_worker(lambda k: True)._run_yolo(_frame(), draw)
    assert [d["class_name"] for d in legacy] == ["bottle"] and len(claimed) == 2


def test_coco_named_product_deleted_after_worker_creation_is_not_detected_or_drawn():
    draw = []
    persons, _c, _f, legacy, claimed = _coco_worker(lambda k: False)._run_yolo(_frame(), draw)
    assert legacy == []                                        # not reported
    assert all("BOTTLE" not in label for _b, label, _col in draw)   # not drawn
    assert len(persons) == 1 and len(claimed) == 1             # people are unaffected


def test_without_liveness_callback_behaviour_is_unchanged():
    draw = []
    _p, _c, _f, legacy, _cl = _coco_worker(None)._run_yolo(_frame(), draw)
    assert [d["class_name"] for d in legacy] == ["bottle"]


# --------------------------------------- on-screen label follows the track's smoothed category
def test_on_screen_label_uses_track_category_not_the_single_frame_guess():
    worker = _make_worker()
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw = [(bbox, "FEMALE 92%", PERSON_FEMALE_COLOR)]          # this frame's (wrong) guess
    worker._apply_track_categories_to_labels(draw, [{"track_id": 1, "bbox": bbox, "category": "male"}])
    assert draw[0][1] == "MALE 92%" and draw[0][2] == PERSON_MALE_COLOR


def test_on_screen_label_is_unknown_when_track_has_no_stable_category():
    worker = _make_worker()
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw = [(bbox, "FEMALE 71%", PERSON_FEMALE_COLOR)]
    worker._apply_track_categories_to_labels(draw, [{"track_id": 1, "bbox": bbox, "category": "unknown"}])
    assert draw[0][1] == "UNKNOWN 71%" and draw[0][2] == PERSON_UNKNOWN_COLOR


def test_worker_tracker_needs_two_agreeing_frames_before_labelling_a_person():
    tracker = _make_worker()._person_tracker
    v, _ = tracker.update([[10, 10, 50, 90]], ["female"], [0.69], now=0.0)   # one confident, wrong frame
    assert v[0]["category"] == "unknown"
    for k in range(1, 5):                                                    # then the man keeps reading male
        v, _ = tracker.update([[10, 10, 50, 90]], ["male"], [0.85], now=k * 0.4)
    assert v[0]["category"] == "male"


# ---------------------------------------------------------------- full-frame face detection
class _FakeFaceDetector:
    def __init__(self, faces=None, boom=False):
        from core.face import FaceBox
        self.faces = faces if faces is not None else [FaceBox(20.0, 15.0, 44.0, 42.0, 0.93)]
        self.boom = boom
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.boom:
            raise RuntimeError("onnx exploded")
        return list(self.faces)


def _face_worker(detector):
    return CameraWorker(camera_id="T", device=0, model=_FakeModel(), allowed_classes=[], face_detector=detector)


def _textured_frame():
    rng = np.random.default_rng(0)
    return rng.integers(60, 200, (100, 100, 3), dtype=np.uint8)


def test_faces_are_detected_drawn_bound_to_the_person_and_counted():
    det = _FakeFaceDetector()
    worker = _face_worker(det)
    draw = []
    tracks = [{"track_id": 5, "bbox": [10.0, 5.0, 60.0, 95.0]}]
    worker._detect_faces(_textured_frame(), tracks, [], draw)
    assert worker.get_face_count() == 1
    assert draw and draw[0][1].startswith("FACE 93%")
    face = worker.get_latest_faces()[0]
    assert face["track_id"] == 5 and face["bbox"] == [20.0, 15.0, 44.0, 42.0]


def test_no_face_detection_work_when_nobody_is_on_screen():
    det = _FakeFaceDetector()
    worker = _face_worker(det)
    worker._latest_faces = [{"stale": True}]
    worker._detect_faces(_textured_frame(), [], [], [])
    assert det.calls == 0 and worker.get_face_count() == 0


def test_face_detector_failure_never_breaks_the_ai_pass():
    worker = _face_worker(_FakeFaceDetector(boom=True))
    draw = []
    worker._detect_faces(_textured_frame(), [{"track_id": 1, "bbox": [0.0, 0.0, 90.0, 95.0]}], [], draw)
    assert draw == [] and worker.get_face_count() == 0


def test_feature_is_off_without_a_detector():
    worker = _make_worker()
    draw = []
    worker._detect_faces(_textured_frame(), [{"track_id": 1, "bbox": [0.0, 0.0, 90.0, 95.0]}], [], draw)
    assert draw == [] and worker.get_face_count() == 0


def test_best_shot_is_dropped_when_the_track_is_evicted():
    worker = _face_worker(_FakeFaceDetector())
    worker._face_service.min_q = 0.0                       # keep a shot even from a synthetic frame
    tracks = [{"track_id": 5, "bbox": [10.0, 5.0, 60.0, 95.0]}]
    worker._detect_faces(_textured_frame(), tracks, [], [])
    assert worker._face_service.store.best(5) is not None
    worker._detect_faces(_textured_frame(), [], [{"track_id": 5}], [])   # person left, track evicted
    assert worker._face_service.store.best(5) is None


# ------------------------------------------------------- hung cap.read() watchdog / frame gate
class _BlockableCap:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return not self.released

    def release(self):
        self.released = True


def test_watchdog_recovers_a_capture_whose_read_is_blocked():
    import time
    worker = _make_worker()
    statuses = []
    worker.on_status = lambda cid, st, msg: statuses.append(st)
    cap = _BlockableCap()
    worker._cap = cap
    worker._read_started = time.monotonic() - 10.0            # read() has been stuck for 10 s
    assert worker.read_stall_seconds() >= 10.0
    assert worker.recover_from_hung_read(stall_sec=6.0) is True
    assert cap.released and worker._cap is None
    assert statuses[-1] == "offline" and worker.hang_recoveries == 1
    assert worker.recover_from_hung_read(stall_sec=6.0) is False   # window restarted: no repeat storm


def test_watchdog_leaves_healthy_or_idle_capture_alone():
    import time
    worker = _make_worker()
    cap = _BlockableCap(); worker._cap = cap
    assert worker.recover_from_hung_read() is False               # not reading at all
    worker._read_started = time.monotonic() - 1.0                  # a normal, short read
    assert worker.recover_from_hung_read(stall_sec=6.0) is False
    assert not cap.released and worker.hang_recoveries == 0


def test_integrity_stats_start_empty_and_are_a_copy():
    worker = _make_worker()
    assert worker.get_integrity_stats() == {}
    worker.get_integrity_stats()["x"] = 1
    assert worker.get_integrity_stats() == {}



# ---------------------------------------------------------------- flicker: same person, same track, box stays drawn
class _ScriptedPersonModel:
    names = {0: "person"}

    def __init__(self, script):
        self.script = list(script)          # per pass: list of (conf, xyxy) or None for "detector saw nothing"
        self.confs = []

    def predict(self, **kwargs):
        self.confs.append(kwargs.get("conf"))
        step = self.script.pop(0)
        boxes = [] if step is None else [_Box(0, c, xy) for c, xy in step]
        return [_Result(boxes)]


def _flicker_worker(script, gender_age_backend=None):
    seen = []
    worker = CameraWorker(camera_id="T", device=0, model=_ScriptedPersonModel(script), allowed_classes=["person"],
                          conf_threshold=0.45, gender_age_backend=gender_age_backend,
                          on_person_tracks=lambda cam, vis, ev, w, h, frame: seen.append(([t["track_id"] for t in vis],
                                                                                        [t["track_id"] for t in ev])))
    worker._latest_capture_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    return worker, seen


def test_detector_runs_at_the_low_threshold_so_dips_can_be_rescued():
    worker, _seen = _flicker_worker([[(0.9, [10.0, 10.0, 50.0, 100.0])]])
    worker.run_ai_pass()
    assert worker.model.confs == [0.25]


def test_confidence_dips_do_not_change_the_track_id_or_cause_a_rebirth():
    """Tracking/Re-ID continuity must survive a confidence dip. This worker
    has no gender_age_backend, so the person's category stays "unknown" --
    but an unclassified person is still drawn (as a pending "PERSON"/"UNKNOWN"
    box, never hidden: a person must be shown before gender is known). See
    test_confidence_dips_do_not_blink_a_classified_persons_box below for the
    same scenario once the person is actually classified."""
    box = [10.0, 10.0, 50.0, 100.0]
    script = [[(0.9, box)], [(0.32, box)], [(0.30, box)], None, [(0.9, box)]]
    worker, seen = _flicker_worker(script)
    drawn = []
    for _ in script:
        worker.run_ai_pass()
        drawn.append(len(worker._last_boxes))
    ids = {i for vis, _ev in seen for i in vis}
    assert ids == {1}                                   # one person, one track, all along
    assert all(ev == [] for _vis, ev in seen)           # no eviction / rebirth
    assert all(n >= 1 for n in drawn[1:])               # unclassified but still shown, on every reported pass


def test_confidence_dips_do_not_blink_a_classified_persons_box():
    """Same scenario as above, but with a confident gender backend so the
    person actually clears the render bar -- this is what the original
    "box stays drawn" guarantee this file used to check now looks like.
    Two confident frames up front (see test_worker_tracker_needs_two_
    agreeing_frames_before_labelling_a_person) so the track's category is
    already confirmed "female" before the dip -- otherwise the dip
    assertion below would really be testing classification-under-
    uncertainty, not "does an already-classified box blink"."""
    box = [10.0, 10.0, 50.0, 100.0]
    script = [[(0.9, box)], [(0.9, box)], [(0.32, box)], [(0.30, box)], None, [(0.9, box)]]
    worker, seen = _flicker_worker(script, gender_age_backend=_FakeGenderBackend("female", 0.95))
    drawn = []
    for _ in script:
        worker.run_ai_pass()
        drawn.append(len(worker._last_boxes))
    ids = {i for vis, _ev in seen for i in vis}
    assert ids == {1}
    assert all(ev == [] for _vis, ev in seen)
    assert all(n >= 1 for n in drawn[1:])   # once classified (pass 2 on), the box is drawn on every remaining pass


def test_a_lone_low_confidence_detection_is_neither_reported_nor_drawn():
    worker, seen = _flicker_worker([[(0.31, [10.0, 10.0, 50.0, 100.0])]])
    worker.run_ai_pass()
    assert seen == [([], [])] and worker._last_boxes == []


# --------------------------------------------- _run_custom_recognition
class _FakeProposer:
    def __init__(self, boxes):
        self._boxes = boxes

    def propose(self, frame, exclude_boxes=None, max_regions=2):
        return list(self._boxes)


class _FakeCustomRecognizer:
    def __init__(self, match=None, has_gallery=True):
        self._match = match  # (product_key, score) or None
        self._has_gallery = has_gallery

    def has_any_gallery(self):
        return self._has_gallery

    def identify(self, crop, floor=None, margin=None):
        return self._match if self._match is not None else (None, 0.0)


class _FakeSegmenter:
    def __init__(self, mask):
        self._mask = mask

    def person_mask(self, frame, conf=0.35):
        return self._mask


def test_hand_only_candidate_never_becomes_a_product():
    """A proposal fully covered by the current-frame human mask must never
    reach recognition at all, no matter how well it would otherwise match
    -- hand/arm exclusion runs before the embedding lookup, not after."""
    worker = _make_worker()
    worker.recognizer = _FakeCustomRecognizer(match=("prod-1", 0.9))
    worker._proposer = _FakeProposer([[10.0, 10.0, 60.0, 60.0]])
    worker._person_segmenter = _FakeSegmenter(np.ones((100, 100), dtype=bool))
    draw = []
    detections = worker._run_custom_recognition(_frame(), claimed_boxes=[], draw_boxes=draw)
    assert detections == []
    assert draw == []


def test_clean_candidate_needs_two_passes_before_it_is_reported():
    # A full-trust-size box (>= PRODUCT_FULL_TRUST_PX, core/product_confirm.py): this test is
    # about the base two-pass confirmation rule, not the extra-hits-for-a-small-box scaling
    # covered by test_a_small_far_candidate_needs_more_passes_than_a_close_one below.
    worker = _make_worker()
    worker.recognizer = _FakeCustomRecognizer(match=("prod-1", 0.9))
    worker._proposer = _FakeProposer([[10.0, 10.0, 100.0, 100.0]])
    worker._person_segmenter = _FakeSegmenter(np.zeros((100, 100), dtype=bool))

    first = worker._run_custom_recognition(_frame(), claimed_boxes=[], draw_boxes=[])
    assert first == []  # one-off match: could be background clutter

    second = worker._run_custom_recognition(_frame(), claimed_boxes=[], draw_boxes=[])
    assert [d["class_name"] for d in second] == ["prod-1"]


def test_a_small_far_candidate_needs_more_passes_than_a_close_one():
    """A box well under PRODUCT_FULL_TRUST_PX (core/product_confirm.py) must need MORE confirming
    passes before being reported than the full-trust-size case above -- see
    product_confirm_hits_for. Uses the same 50x50 box size that, before this size-aware scaling
    existed, needed only the base two passes."""
    worker = _make_worker()
    worker.recognizer = _FakeCustomRecognizer(match=("prod-1", 0.9))
    worker._proposer = _FakeProposer([[10.0, 10.0, 60.0, 60.0]])   # 50x50: below PRODUCT_FULL_TRUST_PX
    worker._person_segmenter = _FakeSegmenter(np.zeros((100, 100), dtype=bool))

    results = [worker._run_custom_recognition(_frame(), claimed_boxes=[], draw_boxes=[]) for _ in range(3)]
    assert results[0] == [] and results[1] == []          # two passes is no longer enough at this size
    assert [d["class_name"] for d in results[2]] == ["prod-1"]


def test_a_small_far_candidate_requires_a_higher_identify_floor():
    """identify() must be asked for a stricter floor when the candidate box is small; a close,
    full-trust-size box must not have its floor touched at all (recognizer's own default floor
    applies unmodified) -- no behaviour change for the common case."""
    class _RecordingRecognizer:
        def __init__(self):
            self.calls = []

        def has_any_gallery(self):
            return True

        def identify(self, crop, floor=None, margin=None):
            self.calls.append(floor)
            return None, 0.0

    worker = _make_worker()
    rec = _RecordingRecognizer()
    worker.recognizer = rec
    worker._proposer = _FakeProposer([[10.0, 10.0, 60.0, 60.0], [10.0, 10.0, 100.0, 100.0]])
    worker._person_segmenter = _FakeSegmenter(np.zeros((100, 100), dtype=bool))
    worker._run_custom_recognition(_frame(), claimed_boxes=[], draw_boxes=[])
    small_floor, full_trust_floor = rec.calls
    assert full_trust_floor is None                # unscaled -- recognizer's own default applies
    assert small_floor is not None and small_floor > CUSTOM_RECOGNITION_BASE_FLOOR


def test_no_gallery_skips_proposer_and_segmenter_entirely():
    worker = _make_worker()
    worker.recognizer = _FakeCustomRecognizer(has_gallery=False)

    class _BoomProposer:
        def propose(self, *a, **k):
            raise AssertionError("propose() must not be called with no trained gallery")

    worker._proposer = _BoomProposer()
    detections = worker._run_custom_recognition(_frame(), claimed_boxes=[], draw_boxes=[])
    assert detections == []


def test_unmatched_candidate_is_never_drawn_as_unknown():
    """STRICT UNKNOWN DETECTION rule: a candidate that doesn't clear
    identify()'s bar must be dropped outright, never drawn/reported as
    UNKNOWN (there is deliberately no show_unknown flag any more)."""
    worker = _make_worker()
    worker.recognizer = _FakeCustomRecognizer(match=None)
    worker._proposer = _FakeProposer([[10.0, 10.0, 60.0, 60.0]])
    worker._person_segmenter = _FakeSegmenter(np.zeros((100, 100), dtype=bool))
    draw = []
    detections = worker._run_custom_recognition(_frame(), claimed_boxes=[], draw_boxes=draw)
    assert detections == []
    assert draw == []
    assert not hasattr(worker, "show_unknown")


# ------------------------------------- real ProductRecognizer end-to-end regression
def test_registered_product_plus_unrelated_object_in_the_same_frame():
    """Permanent regression test for the exact reported failure: Product A
    is registered, and later a completely different, never-registered
    object is shown alongside it in the same frame. Product A must be
    detected; the unrelated object must be ignored, not labelled Product A.
    Uses the real ProductRecognizer/MobileNetV3 pipeline (not a fake), and
    a real two-candidate frame -- core.localizer.ForegroundProposer and
    core.person_segmenter are still faked, since this test is about
    recognition identity, not foreground proposal or human exclusion
    (both already have their own dedicated tests)."""
    import cv2

    from core.recognizer import ProductRecognizer

    def make_obj(color, seed):
        rng = np.random.default_rng(seed)
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:, :] = (40, 40, 40)
        cv2.rectangle(img, (40, 40), (160, 160), color, -1)
        noise = rng.integers(-15, 15, (200, 200, 3))
        return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        recognizer = ProductRecognizer(gallery_dir=Path(d) / "gallery", device="cpu")
        for i in range(6):
            recognizer.add_sample("product-a", make_obj((30, 30, 220), 100 + i))  # red = Product A

        frame = np.zeros((200, 400, 3), dtype=np.uint8)
        frame[:, :200] = make_obj((30, 30, 220), 900)          # left half: Product A (registered)
        frame[:, 200:] = make_obj((40, 200, 40), 901)  # right half: unrelated green object

        worker = _make_worker()
        worker.recognizer = recognizer
        worker._proposer = _FakeProposer([[0.0, 0.0, 200.0, 200.0], [200.0, 0.0, 400.0, 200.0]])
        worker._person_segmenter = _FakeSegmenter(np.zeros((200, 400), dtype=bool))

        # Needs two agreeing passes to confirm (ProductConfirmer, see core/product_confirm.py).
        worker._run_custom_recognition(frame, claimed_boxes=[], draw_boxes=[])
        detections = worker._run_custom_recognition(frame, claimed_boxes=[], draw_boxes=[])

        classes = [d["class_name"] for d in detections]
        assert classes == ["product-a"]  # Product A detected exactly once, unrelated object ignored
