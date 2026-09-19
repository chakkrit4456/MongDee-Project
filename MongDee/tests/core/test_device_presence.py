"""core/device_presence.py: unplug / replug noticed from the OS device list, without opening any camera."""
from __future__ import annotations

from core.device_presence import DevicePresenceMonitor


class _Enum:
    def __init__(self, first):
        self.value = first

    def __call__(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


def _monitor(devices, confirmations=2):
    e = _Enum(devices)
    m = DevicePresenceMonitor(enumerate_fn=e, remove_confirmations=confirmations)
    events = []
    m.add_listener(events.append)
    return m, e, events


def test_first_look_is_a_baseline_not_a_change():
    m, e, events = _monitor({0: "USB Camera", 1: "HD WebCam"})
    assert m.poll_once() is None
    assert events == []
    assert m.is_present(0) is True and m.is_present(1) is True and m.is_present(2) is False


def test_an_appearing_camera_is_reported_at_once():
    m, e, events = _monitor({0: "USB Camera"})
    m.poll_once()
    e.value = {0: "USB Camera", 1: "HD WebCam"}
    ev = m.poll_once()
    assert ev is not None and ev.added == ("HD WebCam",) and ev.removed == ()
    assert m.is_present(1) is True


def test_a_disappearing_camera_needs_two_consecutive_polls():
    m, e, events = _monitor({0: "USB Camera", 1: "HD WebCam"})
    m.poll_once()
    e.value = {0: "USB Camera"}
    assert m.poll_once() is None                      # one glitchy enumeration is not an unplug
    assert m.is_present(1) is True
    ev = m.poll_once()
    assert ev is not None and ev.removed == ("HD WebCam",)
    assert m.is_present(1) is False and m.is_present(0) is True


def test_a_one_poll_dropout_that_recovers_is_never_reported():
    m, e, events = _monitor({0: "USB Camera", 1: "HD WebCam"})
    m.poll_once()
    e.value = {0: "USB Camera"}
    m.poll_once()
    e.value = {0: "USB Camera", 1: "HD WebCam"}
    m.poll_once()
    e.value = {0: "USB Camera"}
    m.poll_once()                                     # counting restarts, still only one sighting
    assert events == [] and m.is_present(1) is True


def test_could_not_look_is_not_the_same_as_unplugged():
    m, e, events = _monitor({0: "USB Camera"})
    m.poll_once()
    for value in (None, RuntimeError("com error"), None):
        e.value = value
        assert m.poll_once() is None
    assert events == [] and m.is_present(0) is True


def test_no_enumerator_means_unknown_never_absent():
    m, e, events = _monitor(None)
    m.poll_once()
    assert m.snapshot() is None
    assert m.is_present(0) is None and m.is_present("rtsp://x") is None


def test_two_identical_cameras_one_unplugged():
    m, e, events = _monitor({0: "USB Camera", 1: "USB Camera"})
    m.poll_once()
    e.value = {0: "USB Camera"}
    m.poll_once()
    ev = m.poll_once()
    assert ev.removed == ("USB Camera",) and m.is_present(1) is False


def test_windows_renumbering_the_same_cameras_is_not_an_event():
    m, e, events = _monitor({0: "A", 1: "B"})
    m.poll_once()
    e.value = {0: "B", 1: "A"}
    assert m.poll_once() is None and events == []
    assert m.name_at(0) == "B"


def test_a_failing_listener_does_not_stop_the_others():
    m, e, events = _monitor({0: "A"})
    m._listeners.insert(0, lambda ev: 1 / 0)
    m.poll_once()
    e.value = {0: "A", 1: "B"}
    m.poll_once()
    assert len(events) == 1


def test_thread_runs_and_stops():
    m, e, events = _monitor({0: "A"})
    m.interval_sec = 0.01
    m.start()
    import time
    time.sleep(0.1)
    e.value = {0: "A", 1: "B"}
    time.sleep(0.15)
    m.stop()
    assert m.polls >= 3 and len(events) == 1
