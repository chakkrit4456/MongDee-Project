from __future__ import annotations

import json
from pathlib import Path

import pytest

from camera_agent.config import AgentConfig


def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "agent.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_load_minimal_config(tmp_path):
    path = _write(tmp_path, {"cameras": [{"id": "CAM01", "protocol": "usb", "device_index": 0}]})
    cfg = AgentConfig.load(path)
    assert cfg.server_url == "http://127.0.0.1:8100"  # default
    assert len(cfg.cameras) == 1
    assert cfg.cameras[0].id == "CAM01"


def test_load_full_config(tmp_path):
    path = _write(tmp_path, {
        "server_url": "http://ai.example.com:8100",
        "api_key": "secret",
        "register_retry_sec": 2.5,
        "push_timeout_sec": 3.0,
        "jpeg_quality": 60,
        "cameras": [{"id": "CAM01", "protocol": "usb"}, {"id": "CAM02", "protocol": "usb", "device_index": 1}],
    })
    cfg = AgentConfig.load(path)
    assert cfg.server_url == "http://ai.example.com:8100"
    assert cfg.api_key == "secret"
    assert cfg.register_retry_sec == 2.5
    assert cfg.jpeg_quality == 60
    assert [c.id for c in cfg.cameras] == ["CAM01", "CAM02"]


def test_requires_at_least_one_camera(tmp_path):
    path = _write(tmp_path, {"cameras": []})
    with pytest.raises(ValueError, match="at least one camera"):
        AgentConfig.load(path)


def test_rejects_duplicate_camera_ids(tmp_path):
    path = _write(tmp_path, {"cameras": [{"id": "CAM01", "protocol": "usb"}, {"id": "CAM01", "protocol": "usb"}]})
    with pytest.raises(ValueError, match="duplicate camera id"):
        AgentConfig.load(path)


def test_rejects_unknown_fields(tmp_path):
    path = _write(tmp_path, {"cameras": [{"id": "CAM01", "protocol": "usb"}], "not_a_real_field": 1})
    with pytest.raises(ValueError, match="unknown field"):
        AgentConfig.load(path)


def test_env_var_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_AGENT_KEY", "from-env")
    path = _write(tmp_path, {"api_key": "${MY_AGENT_KEY}", "cameras": [{"id": "CAM01", "protocol": "usb"}]})
    cfg = AgentConfig.load(path)
    assert cfg.api_key == "from-env"


def test_missing_file_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        AgentConfig.load(tmp_path / "does_not_exist.json")


def test_shipped_example_config_loads(monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("MONGDEE_AGENT_API_KEY", "")
    example = Path(__file__).resolve().parent.parent.parent / "configs" / "camera_agent.example.json"
    cfg = AgentConfig.load(example)
    assert len(cfg.cameras) == 2
    assert cfg.cameras[0].id == "REMOTE-CAM01"
