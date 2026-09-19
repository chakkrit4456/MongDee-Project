"""Camera Gateway package — Phase 1 of the MongDee multi-camera person
pipeline (see MongDee_Master_Prompt.md). Public surface:

    from camera import CameraGateway, CameraConfig, load_camera_configs

Adapters (camera/usb.py, rtsp.py, http.py, hls.py, onvif.py) are
implementation detail — code outside this package should not import them
directly; go through CameraGateway / create_camera_source instead, so a
new protocol never requires touching pipeline code (Master Prompt section 3).
"""

from camera.base import (
    CameraConfig,
    CameraConnectionError,
    CameraProtocol,
    CameraSource,
    CameraStatus,
    Frame,
)
from camera.config import load_camera_configs
from camera.gateway import CameraGateway, create_camera_source

__all__ = [
    "CameraConfig",
    "CameraConnectionError",
    "CameraProtocol",
    "CameraSource",
    "CameraStatus",
    "Frame",
    "CameraGateway",
    "create_camera_source",
    "load_camera_configs",
]
