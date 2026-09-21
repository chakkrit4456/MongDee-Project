"""discover_cameras probes only the DirectShow indices that exist (no blind 0..15 sweep)."""
import numpy as np

from core import camera_identity, vision


class _Cap:
    def __init__(self, index, log):
        self.index = index
        log.append(index)

    def isOpened(self):
        return True

    def read(self):
        return True, np.full((240, 320, 3), 100, np.uint8)

    def release(self):
        pass


def _run(monkeypatch, devices, **kw):
    opened = []
    monkeypatch.setattr(vision, "_open_capture", lambda i, label=None, low_bandwidth_only=False: _Cap(i, opened))
    monkeypatch.setattr(vision, "_read_with_warmup", lambda cap, attempts=5: True)
    monkeypatch.setattr(camera_identity, "list_directshow_devices", lambda: devices)
    found = vision.discover_cameras(**kw)
    return found, opened


def test_only_enumerated_indices_are_opened(monkeypatch):
    """Not a literally-exact match to `known` any more (see core.vision.DISCOVERY_INDEX_SAFETY_
    MARGIN's own docstring: a real DirectShow enumeration can undercount by a device or two if
    called while a driver is still initializing -- the missing-cameras investigation's confirmed
    root cause), but still nowhere near a blind sweep: enumerated max=3 probes up to 5, not 15."""
    found, opened = _run(monkeypatch, {0: "HD WebCam", 1: "USB Camera", 2: "USB Camera", 3: "OBS Virtual Camera"})
    assert opened == [0, 1, 2, 3, 4, 5] and found == [0, 1, 2, 3, 4, 5]


def test_enumeration_unavailable_keeps_the_blind_sweep(monkeypatch):
    found, opened = _run(monkeypatch, {}, max_index=6)
    assert opened == [0, 1, 2, 3, 4, 5]


def test_explicit_max_index_still_caps_and_skip_is_respected(monkeypatch):
    found, opened = _run(monkeypatch, {i: f"cam{i}" for i in range(10)}, max_index=3, skip=frozenset({1}))
    assert opened == [0, 2]
