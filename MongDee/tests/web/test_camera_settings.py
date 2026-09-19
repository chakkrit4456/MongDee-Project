import json

from web.booth_manager import compute_camera_skip_indices, load_camera_settings


def _fake_devices(devices):
    return lambda: dict(enumerate(devices))


def test_load_camera_settings_defaults_to_builtin_disabled_and_empty(tmp_path):
    settings = load_camera_settings(tmp_path / "missing.json")
    assert settings == {
        "enable_builtin_camera": False,
        "builtin_camera_names": [],
        "builtin_camera_indices": [],
    }


def test_load_camera_settings_reads_configured_values(tmp_path):
    path = tmp_path / "booth_settings.json"
    path.write_text(json.dumps({
        "active_booth_id": "BOOTH-1",
        "enable_builtin_camera": False,
        "builtin_camera_names": ["HD WebCam"],
        "builtin_camera_indices": [5],
    }), encoding="utf-8")
    settings = load_camera_settings(path)
    assert settings["builtin_camera_names"] == ["HD WebCam"]
    assert settings["builtin_camera_indices"] == [5]


def test_compute_skip_excludes_builtin_by_current_name_not_stale_index(monkeypatch):
    """The exact regression this module exists to fix: the built-in camera's
    index moved (0 -> 1) after USB cameras were unplugged/replugged, but the
    configured *name* still resolves to whichever index it's actually at
    right now."""
    import core.camera_identity as camera_identity
    monkeypatch.setattr(
        camera_identity, "list_directshow_devices",
        _fake_devices(["USB Camera", "HD WebCam", "USB Camera", "OBS Virtual Camera"]),
    )
    settings = {
        "enable_builtin_camera": False,
        "builtin_camera_names": ["HD WebCam"],
        "builtin_camera_indices": [0],  # stale — the real built-in is now at index 1
    }
    skip = compute_camera_skip_indices(settings)
    assert 1 in skip  # excluded by current name, even though configured index (0) is wrong
    assert 3 in skip  # virtual camera always excluded
    assert 0 in skip  # legacy static index still honored too (as configured, if wrong that's on the operator)
    assert 2 not in skip  # a real USB camera must never be excluded


def test_compute_skip_allows_builtin_when_enabled(monkeypatch):
    import core.camera_identity as camera_identity
    monkeypatch.setattr(
        camera_identity, "list_directshow_devices",
        _fake_devices(["USB Camera", "HD WebCam", "OBS Virtual Camera"]),
    )
    settings = {
        "enable_builtin_camera": True,
        "builtin_camera_names": ["HD WebCam"],
        "builtin_camera_indices": [],
    }
    skip = compute_camera_skip_indices(settings)
    assert 1 not in skip  # built-in allowed
    assert 2 in skip      # virtual camera excluded regardless of enable_builtin_camera


def test_compute_skip_with_no_identity_available_falls_back_to_configured_index(monkeypatch):
    import core.camera_identity as camera_identity
    monkeypatch.setattr(camera_identity, "list_directshow_devices", _fake_devices([]))
    settings = {
        "enable_builtin_camera": False,
        "builtin_camera_names": ["HD WebCam"],
        "builtin_camera_indices": [0],
    }
    skip = compute_camera_skip_indices(settings)
    assert skip == frozenset({0})
