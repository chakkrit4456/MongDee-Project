"""Exercises the ONVIF adapter against a real local SOAP server (not
mocks). The server independently recomputes the WS-Security PasswordDigest
from the received nonce/created/password and rejects the request if it
does not match — this proves camera.onvif's digest implementation is
spec-correct (ONVIF core spec 5.12.2), not just "doesn't crash".
"""

from __future__ import annotations

import base64
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from xml.etree import ElementTree as ET

import pytest

import camera.rtsp as rtsp_mod
from camera.base import CameraConfig, CameraConnectionError
from camera.onvif import _NS_D, ONVIFCamera, _parse_probe_match, resolve_stream_url
from tests.camera.conftest import FakeVideoCapture

_USERNAME = "admin"
_PASSWORD = "secret"

_WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
_WSU = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
_QUERY_NS = {"wsse": _WSSE, "wsu": _WSU}


def _digest_ok(body: bytes) -> bool:
    root = ET.fromstring(body)
    username_el = root.find(".//wsse:Username", _QUERY_NS)
    password_el = root.find(".//wsse:Password", _QUERY_NS)
    nonce_el = root.find(".//wsse:Nonce", _QUERY_NS)
    created_el = root.find(".//wsu:Created", _QUERY_NS)
    if None in (username_el, password_el, nonce_el, created_el):
        return False
    if username_el.text != _USERNAME:
        return False
    nonce = base64.b64decode(nonce_el.text)
    expected = base64.b64encode(
        hashlib.sha1(nonce + created_el.text.encode("utf-8") + _PASSWORD.encode("utf-8")).digest()
    ).decode("ascii")
    return password_el.text == expected


class _ONVIFHandler(BaseHTTPRequestHandler):
    port_holder: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if not _digest_ok(body):
            self.send_response(401)
            self.end_headers()
            return
        port = self.port_holder["port"]
        if b"GetCapabilities" in body:
            xml = (
                '<?xml version="1.0"?>'
                '<soapenv:Envelope xmlns:soapenv="http://www.w3.org/2003/05/soap-envelope" '
                'xmlns:tds="http://www.onvif.org/ver10/device/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">'
                "<soapenv:Body><tds:GetCapabilitiesResponse><tds:Capabilities>"
                f'<tt:Media><tt:XAddr>http://127.0.0.1:{port}/onvif/media_service</tt:XAddr></tt:Media>'
                "</tds:Capabilities></tds:GetCapabilitiesResponse></soapenv:Body></soapenv:Envelope>"
            )
        elif b"GetProfiles" in body:
            xml = (
                '<?xml version="1.0"?>'
                '<soapenv:Envelope xmlns:soapenv="http://www.w3.org/2003/05/soap-envelope" '
                'xmlns:trt="http://www.onvif.org/ver10/media/wsdl">'
                '<soapenv:Body><trt:GetProfilesResponse><trt:Profiles token="Profile_1"/>'
                "</trt:GetProfilesResponse></soapenv:Body></soapenv:Envelope>"
            )
        elif b"GetStreamUri" in body:
            xml = (
                '<?xml version="1.0"?>'
                '<soapenv:Envelope xmlns:soapenv="http://www.w3.org/2003/05/soap-envelope" '
                'xmlns:trt="http://www.onvif.org/ver10/media/wsdl" xmlns:tt="http://www.onvif.org/ver10/schema">'
                "<soapenv:Body><trt:GetStreamUriResponse><trt:MediaUri>"
                f"<tt:Uri>rtsp://127.0.0.1:{port}/stream1</tt:Uri>"
                "</trt:MediaUri></trt:GetStreamUriResponse></soapenv:Body></soapenv:Envelope>"
            )
        else:
            self.send_response(500)
            self.end_headers()
            return
        payload = xml.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/soap+xml")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def onvif_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ONVIFHandler)
    _ONVIFHandler.port_holder["port"] = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def _config(port, **overrides):
    base = {
        "id": "CAM03",
        "protocol": "onvif",
        "host": "127.0.0.1",
        "port": port,
        "username": _USERNAME,
        "password": _PASSWORD,
    }
    base.update(overrides)
    return CameraConfig.from_dict(base)


def test_resolve_stream_url_end_to_end(onvif_server):
    port = onvif_server.server_address[1]
    url = resolve_stream_url(_config(port))
    assert url == f"rtsp://{_USERNAME}:{_PASSWORD}@127.0.0.1:{port}/stream1"


def test_resolve_stream_url_wrong_password_rejected(onvif_server):
    port = onvif_server.server_address[1]
    cfg = _config(port, password="wrong-password")
    with pytest.raises(CameraConnectionError, match="HTTP 401"):
        resolve_stream_url(cfg)


def test_resolve_stream_url_requires_host():
    cfg = CameraConfig.from_dict({"id": "CAM03", "protocol": "onvif"})
    with pytest.raises(ValueError, match="config.host is required"):
        resolve_stream_url(cfg)


def test_onvifcamera_end_to_end_via_gateway_adapter(onvif_server, monkeypatch):
    port = onvif_server.server_address[1]
    monkeypatch.setattr(rtsp_mod.cv2, "VideoCapture", lambda url, backend: FakeVideoCapture(url, backend))
    cam = ONVIFCamera(_config(port))
    cam.open()
    try:
        frame = cam.read()
        assert frame is not None
    finally:
        cam.release()


def test_onvifcamera_skips_resolution_when_url_given(monkeypatch):
    monkeypatch.setattr(rtsp_mod.cv2, "VideoCapture", lambda url, backend: FakeVideoCapture(url, backend))
    cfg = CameraConfig.from_dict({"id": "CAM03", "protocol": "onvif", "url": "rtsp://direct-override/stream"})
    cam = ONVIFCamera(cfg)
    cam.open()  # must not attempt any HTTP/SOAP call at all
    frame = cam.read()
    assert frame is not None
    cam.release()


def test_parse_probe_match():
    xml = (
        '<?xml version="1.0"?>'
        f'<soapenv:Envelope xmlns:soapenv="http://www.w3.org/2003/05/soap-envelope" xmlns:d="{_NS_D}">'
        "<soapenv:Body><d:ProbeMatches><d:ProbeMatch>"
        "<d:XAddrs>http://192.168.1.65/onvif/device_service</d:XAddrs>"
        "<d:Scopes>onvif://www.onvif.org/type/NetworkVideoTransmitter</d:Scopes>"
        "</d:ProbeMatch></d:ProbeMatches></soapenv:Body></soapenv:Envelope>"
    ).encode("utf-8")
    parsed = _parse_probe_match(xml)
    assert parsed["xaddrs"] == ["http://192.168.1.65/onvif/device_service"]
    assert parsed["scopes"] == ["onvif://www.onvif.org/type/NetworkVideoTransmitter"]


def test_parse_probe_match_invalid_xml_returns_none():
    assert _parse_probe_match(b"not xml") is None


def test_parse_probe_match_missing_xaddrs_returns_none():
    xml = (
        '<?xml version="1.0"?>'
        f'<soapenv:Envelope xmlns:soapenv="http://www.w3.org/2003/05/soap-envelope" xmlns:d="{_NS_D}">'
        "<soapenv:Body><d:ProbeMatches><d:ProbeMatch/></d:ProbeMatches></soapenv:Body></soapenv:Envelope>"
    ).encode("utf-8")
    assert _parse_probe_match(xml) is None
