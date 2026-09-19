"""Regression tests for the physical-camera-identity performance bug found
while re-running the full suite: core.camera_identity.
get_physical_camera_identities() shells out to PowerShell and was measured
taking several seconds on real hardware (see this session's audit). Calling
it synchronously inside CameraWorker._open() (as the original physical-
identity integration did) delayed every camera's first "online" status by
that same multi-second amount, and delayed every reconnect too — this is
what core/vision.py's _resolve_physical_id (now async, fire-and-forget) and
_cached_physical_camera_identities (used by _relocate_device_if_moved) fix.
"""

from __future__ import annotations

import time

import core.vision as vision


def _reset_physical_identity_cache():
    vision._physical_identity_cache = None


def test_resolve_physical_id_does_not_block_the_caller(monkeypatch):
    """The core bug: _open() must return promptly even if the underlying
    PnP lookup is slow. _resolve_physical_id() is fire-and-forget, so
    calling it directly must return near-instantly regardless of how long
    the (mocked, slow) identity lookup takes on its background thread."""
    _reset_physical_identity_cache()

    def slow_get_physical_camera_identities():
        time.sleep(1.0)  # stands in for a real multi-second PowerShell call
        return {0: type("Identity", (), {"physical_id": "physid-0"})()}

    monkeypatch.setattr("core.camera_identity.get_physical_camera_identities",
                         slow_get_physical_camera_identities)

    model = type("Model", (), {"names": {0: "person"}})()
    worker = vision.CameraWorker("CAM-1", 0, model, [])

    start = time.monotonic()
    worker._resolve_physical_id()
    elapsed = time.monotonic() - start
    # Threshold is generous (not e.g. 0.2s) on purpose: this only needs to
    # rule out waiting anywhere near the mocked lookup's full 1.0s sleep --
    # tight thresholds around plain thread-start overhead flake under a
    # loaded machine (seen in practice: 0.22s while running the full suite)
    # without indicating any real blocking regression.
    assert elapsed < 0.6, f"_resolve_physical_id blocked the caller for {elapsed:.2f}s"

    # ...but it does eventually finish and set the identity, on its own thread.
    deadline = time.monotonic() + 3.0
    while worker._physical_id is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert worker._physical_id == "physid-0"


def test_cached_physical_camera_identities_collapses_repeated_calls(monkeypatch):
    """_relocate_device_if_moved runs synchronously (it needs the answer
    before the caller's next _open_capture), so *it* relies on this cache
    to avoid re-paying the slow lookup on every single reconnect attempt in
    a burst."""
    _reset_physical_identity_cache()
    call_count = {"n": 0}

    def counting_get_physical_camera_identities():
        call_count["n"] += 1
        return {0: "fake-identity"}

    monkeypatch.setattr("core.camera_identity.get_physical_camera_identities",
                         counting_get_physical_camera_identities)

    first = vision._cached_physical_camera_identities()
    second = vision._cached_physical_camera_identities()
    third = vision._cached_physical_camera_identities()

    assert call_count["n"] == 1  # only the first call actually hit the (mocked) PowerShell path
    assert first == second == third == {0: "fake-identity"}


def test_cached_physical_camera_identities_refreshes_after_ttl(monkeypatch):
    _reset_physical_identity_cache()
    call_count = {"n": 0}
    monkeypatch.setattr("core.camera_identity.get_physical_camera_identities",
                         lambda: call_count.update(n=call_count["n"] + 1) or {})
    monkeypatch.setattr(vision, "_PHYSICAL_IDENTITY_CACHE_TTL_SEC", 0.05)

    vision._cached_physical_camera_identities()
    time.sleep(0.1)
    vision._cached_physical_camera_identities()

    assert call_count["n"] == 2
