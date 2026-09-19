import json
import os

import pytest

from backend.config import MongDeeConfig
from vision.reid.config import ReIDConfig


def _write(tmp_path, obj):
    p = tmp_path / "mongdee.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_defaults_when_sections_missing(tmp_path):
    cfg = MongDeeConfig.load(_write(tmp_path, {"cameras": []}))
    assert cfg.cameras == []
    assert cfg.detection.model_path == "yolo11n.pt"
    assert cfg.identity.match_threshold == 0.80
    assert cfg.api.port == 8100


def test_loads_the_checked_in_example():
    os.environ.setdefault("CAM02_PASSWORD", "x")
    os.environ.setdefault("CAM03_PASSWORD", "y")
    cfg = MongDeeConfig.load("configs/mongdee.example.json")
    assert [c.id for c in cfg.cameras] == ["CAM01", "CAM02", "CAM03"]
    assert cfg.topology.transitions[("CAM01", "CAM02")] == (3.0, 60.0)
    assert cfg.topology.spatial_feasibility("CAM01", "CAM02") == 1.0


def test_env_var_substitution(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_CAM_PW", "s3cr3t")
    monkeypatch.setenv("MONGDEE_API_KEY", "topkey")
    cfg = MongDeeConfig.load(_write(tmp_path, {
        "cameras": [{"id": "CAM01", "protocol": "rtsp", "url": "rtsp://u:${MY_CAM_PW}@host/1"}],
        "api": {"api_key": "${MONGDEE_API_KEY}"},
    }))
    assert cfg.cameras[0].url == "rtsp://u:s3cr3t@host/1"
    assert cfg.api.api_key == "topkey"


def test_unknown_field_in_subconfig_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown field"):
        MongDeeConfig.load(_write(tmp_path, {"cameras": [], "reid": {"backend": "color", "bogus": 1}}))


def test_duplicate_camera_id_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate camera id"):
        MongDeeConfig.load(_write(tmp_path, {"cameras": [
            {"id": "CAM01", "protocol": "usb"}, {"id": "CAM01", "protocol": "usb"},
        ]}))
