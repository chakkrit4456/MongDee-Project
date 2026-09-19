"""Unit tests for CameraWorker._append_global_ids_to_labels (spec: MongDee
AI Vision master prompt section 19 -- the live video box must show the
Global Person ID once Re-ID has resolved it, e.g. "MALE 90% | P000001", not
just the category).

Constructing a CameraWorker never opens a camera or starts its thread, so
this private method is safe to call directly, same pattern as
test_vision.py's _person_label_and_color tests.
"""

from __future__ import annotations

from core.attributes import AttributeResult
from core.vision import CameraWorker


class _FakeModel:
    names = {0: "person"}


def _make_worker(global_id_resolver=None, global_attribute_resolver=None):
    return CameraWorker(
        camera_id="CAM-1", device=0, model=_FakeModel(), allowed_classes=[],
        global_id_resolver=global_id_resolver,
        global_attribute_resolver=global_attribute_resolver,
    )


def test_appends_global_id_when_resolver_confirms_a_match():
    worker = _make_worker(global_id_resolver=lambda camera_id, track_id: "P000001")
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw_boxes = [(bbox, "MALE 90%", (255, 0, 0))]
    visible_tracks = [{"track_id": 7, "bbox": bbox}]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)

    assert draw_boxes[0][1] == "MALE 90% | P000001"
    assert draw_boxes[0][2] == (255, 0, 0)  # color untouched
    assert draw_boxes[0][0] is bbox  # bbox untouched


def test_label_unchanged_when_no_resolver_configured():
    worker = _make_worker(global_id_resolver=None)
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw_boxes = [(bbox, "MALE 90%", (255, 0, 0))]
    visible_tracks = [{"track_id": 7, "bbox": bbox}]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)

    assert draw_boxes[0][1] == "MALE 90%"  # unchanged -- exactly pre-existing behavior


def test_label_unchanged_when_reid_has_not_matched_this_track_yet():
    # Re-ID is throttled (core.reid.ReIDSampler) -- a brand new/unsampled
    # track has no global_id yet. Must never fabricate one.
    worker = _make_worker(global_id_resolver=lambda camera_id, track_id: None)
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw_boxes = [(bbox, "FEMALE 75%", (0, 0, 255))]
    visible_tracks = [{"track_id": 3, "bbox": bbox}]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)

    assert draw_boxes[0][1] == "FEMALE 75%"


def test_only_matching_person_box_is_relabeled_not_product_boxes():
    # draw_boxes mixes person and product entries in detection order --
    # only the person entry whose bbox matches a visible track may change.
    person_bbox = [10.0, 10.0, 50.0, 90.0]
    product_bbox = [100.0, 100.0, 150.0, 150.0]
    worker = _make_worker(global_id_resolver=lambda camera_id, track_id: "P000042")
    draw_boxes = [
        (product_bbox, "BOTTLE 80%", (0, 255, 0)),
        (person_bbox, "MALE 90%", (255, 0, 0)),
    ]
    visible_tracks = [{"track_id": 1, "bbox": person_bbox}]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)

    assert draw_boxes[0][1] == "BOTTLE 80%"  # product untouched
    assert draw_boxes[1][1] == "MALE 90% | P000042"


def test_multiple_people_each_get_their_own_global_id():
    bbox_a = [10.0, 10.0, 50.0, 90.0]
    bbox_b = [200.0, 10.0, 250.0, 90.0]
    ids = {1: "P000001", 2: "P000002"}
    worker = _make_worker(global_id_resolver=lambda camera_id, track_id: ids.get(track_id))
    draw_boxes = [(bbox_a, "MALE 90%", (255, 0, 0)), (bbox_b, "FEMALE 85%", (0, 0, 255))]
    visible_tracks = [
        {"track_id": 1, "bbox": bbox_a},
        {"track_id": 2, "bbox": bbox_b},
    ]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)

    assert draw_boxes[0][1] == "MALE 90% | P000001"
    assert draw_boxes[1][1] == "FEMALE 85% | P000002"


def test_empty_visible_tracks_is_a_noop():
    worker = _make_worker(global_id_resolver=lambda camera_id, track_id: "P000001")
    draw_boxes = [([10.0, 10.0, 50.0, 90.0], "MALE 90%", (255, 0, 0))]
    worker._append_global_ids_to_labels(draw_boxes, [])
    assert draw_boxes[0][1] == "MALE 90%"


def test_resolver_exception_never_breaks_the_label():
    def boom(camera_id, track_id):
        raise RuntimeError("registry exploded")

    worker = _make_worker(global_id_resolver=boom)
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw_boxes = [(bbox, "MALE 90%", (255, 0, 0))]
    visible_tracks = [{"track_id": 1, "bbox": bbox}]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)  # must not raise

    assert draw_boxes[0][1] == "MALE 90%"


# ---- global_attribute_resolver: same physical person must not show a
# different category on two cameras just because each camera's own local
# classification saw different evidence (found via a live two-camera test:
# one camera reported UNKNOWN, the other reported FEMALE, for the same
# global_id at the same moment) --------------------------------------------

def test_fused_gender_overrides_the_local_per_camera_guess():
    fused = AttributeResult(gender="FEMALE", gender_confidence=0.93, status="ok")
    worker = _make_worker(
        global_id_resolver=lambda camera_id, track_id: "P000001",
        global_attribute_resolver=lambda global_id: fused,
    )
    bbox = [10.0, 10.0, 50.0, 90.0]
    # This camera's own local guess was "unknown" (e.g. a poor-angle frame) --
    # the Global Person's fused result (seen confidently on another camera)
    # should still win, so both cameras end up showing the same category.
    draw_boxes = [(bbox, "UNKNOWN 91%", (128, 128, 128))]
    visible_tracks = [{"track_id": 7, "bbox": bbox}]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)

    assert draw_boxes[0][1] == "FEMALE 93% | P000001"


def test_fused_child_age_is_ignored_and_gender_is_used():
    # Child detection was removed: a stored/legacy CHILD age bucket must not change the label.
    fused = AttributeResult(gender="MALE", gender_confidence=0.8,
                             age_category="CHILD", age_confidence=0.77, status="ok")
    worker = _make_worker(
        global_id_resolver=lambda camera_id, track_id: "P000001",
        global_attribute_resolver=lambda global_id: fused,
    )
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw_boxes = [(bbox, "FEMALE 90%", (255, 0, 0))]
    visible_tracks = [{"track_id": 7, "bbox": bbox}]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)

    assert draw_boxes[0][1] == "MALE 80% | P000001"
    assert "CHILD" not in draw_boxes[0][1]


def test_local_guess_kept_when_fused_result_not_confident_yet():
    # status == "unknown" -- not enough Global-Person evidence yet (e.g. just
    # merged via Re-ID, no attribute samples yet). Must not overwrite with
    # nothing, and must not crash.
    fused = AttributeResult(status="unknown")
    worker = _make_worker(
        global_id_resolver=lambda camera_id, track_id: "P000001",
        global_attribute_resolver=lambda global_id: fused,
    )
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw_boxes = [(bbox, "MALE 90%", (255, 0, 0))]
    visible_tracks = [{"track_id": 7, "bbox": bbox}]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)

    assert draw_boxes[0][1] == "MALE 90% | P000001"


def test_local_guess_kept_when_no_attribute_resolver_configured():
    # Backward compatibility: a caller that never wires global_attribute_resolver
    # (e.g. no FairFace configured) gets exactly the pre-existing behavior.
    worker = _make_worker(global_id_resolver=lambda camera_id, track_id: "P000001",
                           global_attribute_resolver=None)
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw_boxes = [(bbox, "FEMALE 75%", (0, 0, 255))]
    visible_tracks = [{"track_id": 7, "bbox": bbox}]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)

    assert draw_boxes[0][1] == "FEMALE 75% | P000001"


def test_attribute_resolver_exception_never_breaks_the_label():
    def boom(global_id):
        raise RuntimeError("smoother exploded")

    worker = _make_worker(global_id_resolver=lambda camera_id, track_id: "P000001",
                           global_attribute_resolver=boom)
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw_boxes = [(bbox, "MALE 90%", (255, 0, 0))]
    visible_tracks = [{"track_id": 7, "bbox": bbox}]

    worker._append_global_ids_to_labels(draw_boxes, visible_tracks)  # must not raise

    assert draw_boxes[0][1] == "MALE 90% | P000001"


# ---- one gender per person, on every camera; never "UNKNOWN" (2026-09-19) ----------------------------------------
def _resolver_worker(resolver, camera_id="CAM-1"):
    worker = _make_worker(global_id_resolver=lambda camera_id, track_id: "P000001")
    worker.camera_id = camera_id
    worker.gender_resolver = resolver
    return worker


def _label_of(worker, category="unknown"):
    bbox = [10.0, 10.0, 50.0, 90.0]
    draw_boxes = [(bbox, "UNKNOWN 91%", (128, 128, 128))]
    tracks = [{"track_id": 7, "bbox": bbox, "category": category}]
    worker._apply_track_categories_to_labels(draw_boxes, tracks)
    return draw_boxes[0][1]


def test_the_same_person_shows_the_same_gender_on_both_cameras():
    resolver = lambda camera_id, track_id: ("female", 0.9)          # one decision per Global Person
    assert _label_of(_resolver_worker(resolver, "CAM-1"), "male").startswith("FEMALE")
    assert _label_of(_resolver_worker(resolver, "CAM-2"), "unknown").startswith("FEMALE")


def test_a_person_with_no_evidence_yet_is_pending_not_unknown():
    label = _label_of(_resolver_worker(lambda camera_id, track_id: None))
    assert label.startswith("PERSON") and "UNKNOWN" not in label


def test_without_a_resolver_the_old_labels_are_unchanged():
    worker = _make_worker()
    assert _label_of(worker, "unknown").startswith("UNKNOWN")
    assert _label_of(worker, "male").startswith("MALE")


def test_a_failing_resolver_falls_back_to_the_tracks_own_category():
    def boom(camera_id, track_id):
        raise RuntimeError("x")
    assert _label_of(_resolver_worker(boom), "male").startswith("MALE")
