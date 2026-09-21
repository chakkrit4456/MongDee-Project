"""Unit tests for core/training.py: product-registration frame localization
(auto_crop), human-region exclusion wiring, and the quality/duplicate
filtering + validation added to import_images/import_video.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from core import training


# ------------------------------------------------------------------- fakes
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


class _NoBoxModel:
    names = {0: "person"}

    def predict(self, **kwargs):
        return [_Result([])]


class _PersonOnlyModel:
    names = {0: "person"}

    def predict(self, **kwargs):
        return [_Result([_Box(0, 0.9, [0.0, 0.0, 50.0, 50.0])])]


class _ProductBoxModel:
    names = {0: "person", 1: "product"}
    box = [10.0, 10.0, 150.0, 110.0]

    def predict(self, **kwargs):
        return [_Result([_Box(1, 0.8, self.box)])]


class _FakePersonSegmenter:
    def __init__(self, mask):
        self._mask = mask

    def person_mask(self, frame_bgr, conf=0.35):
        return self._mask


class _FakeRecognizer:
    """centered_embed is the crop's mean BGR color, L2-normalized -- two
    crops with the same average color look identical (cosine == 1.0,
    flagged a duplicate); visibly different crops don't."""

    def __init__(self):
        self.added: list[tuple[str, np.ndarray]] = []

    def centered_embed(self, crop):
        v = crop.reshape(-1, 3).mean(axis=0).astype(np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def add_sample(self, product_key, crop):
        self.added.append((product_key, crop))
        return len(self.added)


def _checkerboard(size=(120, 160), tile=8, c1=(20, 20, 20), c2=(235, 235, 235)):
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            color = c1 if ((x // tile) + (y // tile)) % 2 == 0 else c2
            img[y:y + tile, x:x + tile] = color
    return img


def _solid(size=(120, 160), color=(128, 128, 128)):
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def _bright_with_specks(size=(120, 160), step=6):
    # Sparse dark specks on a near-white field: high Laplacian variance
    # (sharp point edges) but mean brightness still well above
    # EXPOSURE_MAX_MEAN -- unlike a checkerboard, contrast doesn't have to
    # trade off against staying near the 255 ceiling.
    img = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    img[::step, ::step] = (0, 0, 0)
    return img


def _dark_with_specks(size=(120, 160), step=6):
    img = np.full((size[0], size[1], 3), 0, dtype=np.uint8)
    img[::step, ::step] = (255, 255, 255)
    return img


# ---------------------------------------------------------------- auto_crop
def test_auto_crop_returns_none_when_nothing_detected():
    assert training.auto_crop(_checkerboard(), _NoBoxModel(), "cpu") is None


def test_auto_crop_returns_none_when_only_a_person_box_found():
    assert training.auto_crop(_checkerboard(), _PersonOnlyModel(), "cpu") is None


def test_auto_crop_never_falls_back_to_the_whole_frame():
    # Old behaviour returned frame_bgr unchanged here; the whole point of
    # this fix is that a frame with no product-shaped box is rejected, not
    # silently trained on.
    frame = _checkerboard()
    result = training.auto_crop(frame, _NoBoxModel(), "cpu")
    assert result is None


def test_auto_crop_returns_the_localized_region_for_a_product_box():
    frame = _checkerboard()
    crop = training.auto_crop(frame, _ProductBoxModel(), "cpu")
    assert crop is not None
    assert crop.shape[0] < frame.shape[0] or crop.shape[1] < frame.shape[1]


def test_auto_crop_rejects_when_candidate_is_mostly_human():
    frame = _checkerboard()
    mask = np.ones(frame.shape[:2], dtype=bool)
    segmenter = _FakePersonSegmenter(mask)
    assert training.auto_crop(frame, _ProductBoxModel(), "cpu", person_segmenter=segmenter) is None


def test_auto_crop_tightens_box_away_from_partial_human_overlap():
    frame = _checkerboard()
    mask = np.zeros(frame.shape[:2], dtype=bool)
    x1, y1, x2, y2 = (int(v) for v in _ProductBoxModel.box)
    mid = x1 + (x2 - x1) // 2
    mask[y1:y2, x1:mid] = True  # left half of the candidate box is "human"
    segmenter = _FakePersonSegmenter(mask)
    full_crop = training.auto_crop(frame, _ProductBoxModel(), "cpu")
    trimmed_crop = training.auto_crop(frame, _ProductBoxModel(), "cpu", person_segmenter=segmenter)
    assert trimmed_crop is not None
    assert trimmed_crop.shape[1] < full_crop.shape[1]


# ------------------------------------------------------------- import_images
def test_import_images_raises_when_nothing_readable(tmp_path: Path):
    recognizer = _FakeRecognizer()
    with pytest.raises(RuntimeError):
        training.import_images([str(tmp_path / "missing.jpg")], "prod-1", recognizer,
                                _ProductBoxModel(), "cpu")


def test_import_images_adds_distinct_sharp_frames(tmp_path: Path):
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    cv2.imwrite(str(p1), _checkerboard(c1=(20, 20, 20), c2=(235, 235, 235)))
    cv2.imwrite(str(p2), _checkerboard(c1=(0, 120, 0), c2=(0, 40, 0)))
    recognizer = _FakeRecognizer()
    added = training.import_images([str(p1), str(p2)], "prod-1", recognizer,
                                    _ProductBoxModel(), "cpu")
    assert added == 2
    assert len(recognizer.added) == 2


def test_import_images_rejects_blurry_frame(tmp_path: Path):
    p1 = tmp_path / "flat.jpg"
    cv2.imwrite(str(p1), _solid())
    recognizer = _FakeRecognizer()
    with pytest.raises(RuntimeError, match="blurry"):
        training.import_images([str(p1)], "prod-1", recognizer, _ProductBoxModel(), "cpu")
    assert recognizer.added == []


def test_import_images_rejects_overexposed_frame(tmp_path: Path):
    p1 = tmp_path / "blown.jpg"
    cv2.imwrite(str(p1), _bright_with_specks())
    recognizer = _FakeRecognizer()
    with pytest.raises(RuntimeError, match="overexposed"):
        training.import_images([str(p1)], "prod-1", recognizer, _ProductBoxModel(), "cpu")
    assert recognizer.added == []


def test_import_images_rejects_underexposed_frame(tmp_path: Path):
    p1 = tmp_path / "dark.jpg"
    cv2.imwrite(str(p1), _dark_with_specks())
    recognizer = _FakeRecognizer()
    with pytest.raises(RuntimeError, match="underexposed"):
        training.import_images([str(p1)], "prod-1", recognizer, _ProductBoxModel(), "cpu")
    assert recognizer.added == []


def test_import_images_rejects_near_duplicate_frames(tmp_path: Path):
    p1 = tmp_path / "a.jpg"
    p2 = tmp_path / "b.jpg"
    same = _checkerboard()
    cv2.imwrite(str(p1), same)
    cv2.imwrite(str(p2), same)  # identical content -> identical embedding
    recognizer = _FakeRecognizer()
    added = training.import_images([str(p1), str(p2)], "prod-1", recognizer,
                                    _ProductBoxModel(), "cpu")
    assert added == 1
    assert len(recognizer.added) == 1


def test_import_images_rejects_when_candidate_is_all_human(tmp_path: Path):
    p1 = tmp_path / "hand.jpg"
    cv2.imwrite(str(p1), _checkerboard())
    recognizer = _FakeRecognizer()
    mask = np.ones((120, 160), dtype=bool)
    with pytest.raises(RuntimeError):
        training.import_images([str(p1)], "prod-1", recognizer, _ProductBoxModel(), "cpu",
                                person_segmenter=_FakePersonSegmenter(mask))
    assert recognizer.added == []


# -------------------------------------------------------------- import_video
def _write_video(path: Path, frames: list[np.ndarray], fps: float = 10.0):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def test_import_video_raises_when_it_cannot_open():
    recognizer = _FakeRecognizer()
    with pytest.raises(RuntimeError):
        training.import_video("__no_such_file__.avi", "prod-1", recognizer,
                               _ProductBoxModel(), "cpu")


def test_import_video_samples_and_adds_distinct_frames(tmp_path: Path):
    # fps=10, VIDEO_SAMPLE_INTERVAL_SEC=0.4 -> step=4: frames 0, 4, 8 sampled
    # from 12 total. Each sampled frame gets a visibly different color so
    # none are rejected as near-duplicates of each other.
    palette = [(0, 0, 0), (0, 150, 0), (150, 0, 0)]
    frames = []
    for i in range(12):
        color_a = palette[(i // 4) % len(palette)]
        frames.append(_checkerboard(c1=color_a, c2=(235, 235, 235)))
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, frames)

    recognizer = _FakeRecognizer()
    added = training.import_video(str(video_path), "prod-1", recognizer,
                                   _ProductBoxModel(), "cpu")
    assert added == 3
    assert len(recognizer.added) == 3


# ------------------------------------------------------------- LiveTrainingSession
def _session(recognizer=None, model=None, person_segmenter=None):
    return training.LiveTrainingSession(
        "prod-1", recognizer or _FakeRecognizer(), model or _ProductBoxModel(), "cpu",
        person_segmenter=person_segmenter,
    )


def test_feed_reports_a_running_distinct_view_count():
    session = _session()
    r1 = session.feed(_checkerboard(c1=(20, 20, 20), c2=(235, 235, 235)))
    assert r1 == {"accepted": True, "distinct_views": 1, "attempted": 1, "reason": None}
    r2 = session.feed(_checkerboard(c1=(0, 120, 0), c2=(0, 40, 0)))
    assert r2 == {"accepted": True, "distinct_views": 2, "attempted": 2, "reason": None}


def test_feed_does_not_count_a_near_duplicate_as_a_new_distinct_view():
    session = _session()
    same = _checkerboard()
    first = session.feed(same)
    second = session.feed(same)
    assert first["accepted"] is True and first["distinct_views"] == 1
    assert second["accepted"] is False and second["distinct_views"] == 1 and second["reason"] == "duplicate"
    assert second["attempted"] == 2


def test_feed_reports_the_rejection_reason_for_a_bad_frame():
    session = _session()
    result = session.feed(_solid())        # flat/low-texture -> blurry
    assert result["accepted"] is False and result["reason"] == "blurry" and result["distinct_views"] == 0


def test_feed_reports_no_product_region_when_nothing_is_detected():
    session = _session(model=_NoBoxModel())
    result = session.feed(_checkerboard())
    assert result["accepted"] is False and result["reason"] == "no_product_region"


def test_finish_raises_when_nothing_was_ever_accepted():
    session = _session(model=_NoBoxModel())
    session.feed(_checkerboard())
    with pytest.raises(RuntimeError):
        session.finish()


def test_finish_reports_totals_and_forgets_nothing_it_already_added():
    recognizer = _FakeRecognizer()
    session = _session(recognizer=recognizer)
    session.feed(_checkerboard(c1=(20, 20, 20), c2=(235, 235, 235)))
    session.feed(_checkerboard(c1=(0, 120, 0), c2=(0, 40, 0)))
    session.feed(_solid())     # rejected -- must not count toward `added`
    result = session.finish()
    assert result == {"added": 2, "attempted": 3, "counts": {"blurry": 1}}
    assert len(recognizer.added) == 2      # samples already went to the recognizer as they arrived
