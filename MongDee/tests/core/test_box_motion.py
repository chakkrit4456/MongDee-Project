"""Boxes must follow the moving picture at the capture frame rate and be corrected for the AI's latency."""
from __future__ import annotations

import cv2
import numpy as np

from core.box_motion import BoxFollower, flow_shift

W, H = 640, 480


def _texture(seed, w, h):
    """A calm gradient covered with shapes of assorted size and brightness: corners for the tracker to grab,
    yet nothing the sensor-noise detector would mistake for static."""
    rng = np.random.default_rng(seed)
    t = np.tile(np.linspace(70, 170, w, dtype=np.float32), (h, 1)).astype(np.uint8)
    for _ in range(max(12, w * h // 1800)):
        x, y = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
        sw, sh = int(rng.integers(18, 60)), int(rng.integers(18, 60))
        cv2.rectangle(t, (x, y), (min(x + sw, w - 1), min(y + sh, h - 1)), int(rng.integers(30, 225)), -1)
    return cv2.GaussianBlur(t, (0, 0), 1.0)


BACKGROUND = _texture(1, W, H)
SPRITE = _texture(2, 120, 260)


def scene(frame_no, vx=6.0, vy=1.0, x0=100, y0=80):
    """A textured 'person' sprite moving over a static textured background; returns (frame, true bbox)."""
    x, y = int(round(x0 + vx * frame_no)), int(round(y0 + vy * frame_no))
    g = BACKGROUND.copy()
    g[y:y + 260, x:x + 120] = SPRITE
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), [float(x), float(y), float(x + 120), float(y + 260)]


def _err(a, b):
    return max(abs(p - q) for p, q in zip(a, b))


def test_flow_shift_measures_the_motion_of_the_object_inside_the_box():
    (f0, b0), (f1, b1) = scene(0), scene(1)
    g0, g1 = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (f0, f1))
    dx, dy = flow_shift(g0, g1, b0)
    assert abs(dx - 6) < 1.0 and abs(dy - 1) < 1.0


def test_a_flat_region_gives_no_estimate_instead_of_a_wrong_one():
    flat = np.full((H, W), 90, np.uint8)
    assert flow_shift(flat, flat, [100, 100, 220, 360]) is None


def test_the_box_follows_the_object_every_frame_between_ai_passes():
    fo = BoxFollower(work_width=W)
    f, box = scene(0)
    fo.push_frame(0, f)
    fo.submit(0, [(box, "MALE 90%", (255, 0, 0))])
    for i in range(1, 13):                                   # 12 frames with no new AI result
        f, truth = scene(i)
        fo.push_frame(i, f)
    (bbox, label, color), = fo.boxes()
    assert label == "MALE 90%" and color == (255, 0, 0)
    assert _err(bbox, truth) < 6.0                           # a frozen box would be 72 px behind
    assert _err(box, truth) > 60


def test_a_late_ai_result_is_carried_forward_to_the_current_frame():
    fo = BoxFollower(work_width=W)
    truths = {}
    for i in range(0, 10):
        f, truths[i] = scene(i)
        fo.push_frame(i, f)
    fo.submit(3, [(truths[3], "FEMALE", (0, 0, 255))])       # the AI started on frame 3, finished at frame 9
    (bbox, _l, _c), = fo.boxes()
    assert _err(bbox, truths[9]) < 6.0
    assert _err(truths[3], truths[9]) > 30                   # without catch-up it would trail by ~36 px


def test_a_fresh_detection_corrects_drift_without_a_jump():
    fo = BoxFollower(work_width=W)
    f, box = scene(0)
    fo.push_frame(0, f)
    drifted = [v + 12 for v in box]                          # the box on screen is 12 px off
    fo.submit(0, [(drifted, "M", (1, 2, 3))])
    fo.submit(0, [(box, "M", (1, 2, 3))])                    # the AI's next answer is exact
    (bbox, _l, _c), = fo.boxes()
    assert 0 < _err(bbox, box) < _err(drifted, box)          # moved toward the truth, not a hard snap


def test_boxes_expire_when_the_ai_goes_quiet():
    now = [100.0]
    fo = BoxFollower(work_width=W, max_age_sec=1.0, clock=lambda: now[0])
    f, box = scene(0)
    fo.push_frame(0, f)
    fo.submit(0, [(box, "M", (1, 2, 3))])
    assert fo.boxes()
    now[0] += 1.2
    assert fo.boxes() == []


def test_without_the_frame_in_the_ring_the_raw_box_is_used():
    fo = BoxFollower(work_width=W, ring_frames=3)
    for i in range(6):
        fo.push_frame(i, scene(i)[0])
    _f, box = scene(0)
    fo.submit(0, [(box, "M", (1, 2, 3))])                    # frame 0 fell out of the ring
    (bbox, _l, _c), = fo.boxes()
    assert _err(bbox, box) < 1e-6


def test_a_featureless_scene_leaves_boxes_where_they_are():
    fo = BoxFollower(work_width=W)
    flat = np.full((H, W, 3), 80, np.uint8)
    fo.push_frame(0, flat)
    fo.submit(0, [([100, 100, 220, 360], "M", (1, 2, 3))])
    for i in range(1, 5):
        fo.push_frame(i, flat)
    (bbox, _l, _c), = fo.boxes()
    assert bbox == [100.0, 100.0, 220.0, 360.0]


def test_reset_forgets_everything():
    fo = BoxFollower(work_width=W)
    fo.push_frame(0, scene(0)[0])
    fo.submit(0, [(scene(0)[1], "M", (1, 2, 3))])
    fo.reset()
    assert fo.boxes() == []


def test_the_capture_step_is_cheap():
    import time
    fo = BoxFollower(work_width=W)
    f, box = scene(0)
    fo.push_frame(0, f)
    fo.submit(0, [([box[0] + 200 * k, box[1], box[2] + 200 * k, box[3]], "M", (1, 2, 3)) for k in range(0, 3)][:1] * 4)
    t0 = time.perf_counter()
    for i in range(1, 31):
        fo.push_frame(i, scene(i)[0])
    per_frame = (time.perf_counter() - t0) / 30
    assert per_frame < 0.02, per_frame                         # 4 boxes tracked per frame in well under 20 ms


# ----------------------------------------------------------------- through the real capture loop
def test_the_worker_draws_boxes_that_move_with_the_picture(monkeypatch):
    import threading
    import time

    from core import vision

    truths = {}

    class Cap:
        n = 0

        def isOpened(self):
            return True

        def read(self):
            time.sleep(0.01)
            Cap.n += 1
            frame, truth = scene(Cap.n % 60)
            truths[Cap.n] = truth
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr(vision, "_open_capture", lambda device, label=None, low_bandwidth_only=False: Cap())
    model = type("Model", (), {"names": {0: "person"}})()
    drawn = []
    holder = {}

    def on_frame(cam, frame):
        w = holder["w"]
        red = np.all(frame == (0, 0, 255), axis=2)
        ys, xs = np.nonzero(red)
        if xs.size:
            drawn.append((w._frame_seq, [xs.min(), ys.min(), xs.max(), ys.max()]))

    w = vision.CameraWorker("CAM-1", 0, model, [], on_frame=on_frame, ai_worker=vision.AIWorker())
    holder["w"] = w
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + 5
        while w._frame_seq < 6 and time.monotonic() < deadline:
            time.sleep(0.01)
        seq = w._latest_capture_seq                                  # the AI "sees" this frame ...
        w._publish_boxes(seq, [(list(truths[seq]), "MALE 90%", (0, 0, 255))])
        n_before = len(drawn)
        while w._frame_seq < seq + 15 and time.monotonic() < deadline:
            time.sleep(0.01)                                          # ... and 15 more frames go by with no new AI pass
    finally:
        w._running = False
        w._stop_event.set()
        t.join(timeout=3)
    late = [(s, b) for s, b in drawn[n_before:] if s >= seq + 10]
    assert late, "no frames were drawn after the AI result"
    seq_late, box = late[0]
    truth = truths[seq_late]
    err = max(abs(box[i] - truth[i]) for i in (0, 2, 3))              # (the label patch sits above the top edge)
    assert err < 8, (err, box, truth)                                 # a stale box would be ~60 px behind
