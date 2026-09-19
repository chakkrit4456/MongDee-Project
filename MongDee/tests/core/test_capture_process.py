import os, time
import numpy as np
from core.capture_process import CaptureProcess


def _wait(pred, timeout=15.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.05)
    return False


def _read_ok(cap):
    ok, f, seq, ts = cap.read()
    return ok


def test_healthy_capture_delivers_fresh_frames_without_blocking():
    cap = CaptureProcess("core.capture_sim:make_fake", {"mode": "ok", "marker": 7}, name="a")
    cap.start()
    try:
        assert _wait(lambda: _read_ok(cap), 20)
        ok, f, seq, ts = cap.read()
        assert f.shape == (240, 320, 3) and f[0, 0, 0] == 7
        t0 = time.time()
        for _ in range(200):
            cap.read()
        assert time.time() - t0 < 2.0        # read() never blocks on the device
        s1 = cap.read()[2]; time.sleep(0.3); s2 = cap.read()[2]
        assert s2 > s1
    finally:
        cap.stop()


def test_hung_camera_is_killed_and_respawned_and_recovers(tmp_path):
    flag = str(tmp_path / "healthy.flag")
    cap = CaptureProcess("core.capture_sim:make_fake",
                         {"mode": "hang_after", "after": 5, "flag_path": flag}, name="h",
                         stale_sec=1.0, startup_grace_sec=8.0, backoff=(0.2,))
    cap.start()
    try:
        assert _wait(lambda: _read_ok(cap), 20)
        assert _wait(lambda: not _read_ok(cap), 10)        # stalls (frames go stale)
        open(flag, "w").close()                            # next spawn is healthy
        assert _wait(lambda: cap.kills >= 1 and cap.respawns >= 1, 15)
        assert _wait(lambda: _read_ok(cap), 20)
    finally:
        cap.stop()


def test_one_hung_camera_does_not_stall_the_other():
    good = CaptureProcess("core.capture_sim:make_fake", {"mode": "ok", "marker": 1}, name="g")
    bad = CaptureProcess("core.capture_sim:make_fake", {"mode": "hang_after", "after": 3, "marker": 2},
                         name="b", stale_sec=1.0, backoff=(0.2,))
    good.start(); bad.start()
    try:
        assert _wait(lambda: _read_ok(good), 20)
        assert _wait(lambda: bad.kills >= 1, 20)
        seqs = []
        t0 = time.time()
        while time.time() - t0 < 1.5:
            ok, f, seq, ts = good.read()
            assert ok, "good camera starved while bad camera hung"
            seqs.append(seq); time.sleep(0.05)
        assert seqs[-1] > seqs[0]
    finally:
        good.stop(); bad.stop()


def test_child_crash_is_respawned():
    cap = CaptureProcess("core.capture_sim:make_fake", {"mode": "crash_after", "after": 4},
                         name="c", stale_sec=1.0, backoff=(0.2,))
    cap.start()
    try:
        assert _wait(lambda: cap.respawns >= 2, 25)
    finally:
        cap.stop()


def test_fail_forever_gives_no_frames_but_never_raises_or_hangs():
    cap = CaptureProcess("core.capture_sim:make_fake", {"mode": "fail_forever"}, name="f",
                         stale_sec=1.0, backoff=(0.1,), fail_limit=5)
    cap.start()
    try:
        time.sleep(3.0)
        assert not _read_ok(cap)
        assert cap.respawns >= 1
    finally:
        cap.stop()


def test_stop_releases_shared_memory_and_process():
    cap = CaptureProcess("core.capture_sim:make_fake", {"mode": "ok"}, name="s")
    cap.start(); assert _wait(lambda: _read_ok(cap), 20)
    proc = cap._proc
    cap.stop()
    assert not proc.is_alive()
    assert cap.read()[0] is False
