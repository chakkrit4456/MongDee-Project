"""ONVIF adapter.

Scope: enough of the ONVIF spec to go from "camera IP + credentials" to a
working RTSP stream, which is what Master Prompt section 1 (#5) and
section 3 ask for ("รองรับ ONVIF สำหรับค้นหาและจัดการกล้องที่รองรับ").
Implemented as raw SOAP over HTTP using the stdlib's xml.etree — no
zeep/WSDL dependency, since only three operations are ever called
(GetCapabilities, GetProfiles, GetStreamUri) rather than the whole ONVIF
surface.

Not implemented: PTZ control, event subscriptions, video analytics
configuration — out of scope for "get frames into the pipeline" (Phase 1).

WS-Discovery (`discover()`) is a best-effort LAN broadcast; it depends on
UDP multicast reaching the cameras (many managed/VLAN'd networks block
it), so config-driven host/port (skip discovery, go straight to
GetCapabilities) is the primary, reliable path — which is also how ONVIF
cameras are normally set up in a fixed booth deployment anyway.
"""

from __future__ import annotations

import base64
import hashlib
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

import requests

from camera.base import CameraConfig, CameraConnectionError, CameraSource, redact_url
from camera.rtsp import RTSPCamera

_NS_SOAP = "http://www.w3.org/2003/05/soap-envelope"
_NS_WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
_NS_WSU = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-utility-1.0.xsd"
_NS_TDS = "http://www.onvif.org/ver10/device/wsdl"
_NS_TRT = "http://www.onvif.org/ver10/media/wsdl"
_NS_TT = "http://www.onvif.org/ver10/schema"
_NS_D = "http://schemas.xmlsoap.org/ws/2005/04/discovery"
_NS_WSA = "http://schemas.xmlsoap.org/ws/2004/08/addressing"

_DEFAULT_DEVICE_PATH = "/onvif/device_service"
_DISCOVERY_MULTICAST_ADDR = "239.255.255.250"
_DISCOVERY_PORT = 3702


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _security_header(username: str, password: str) -> str:
    """WS-Security UsernameToken, PasswordDigest profile (ONVIF core spec
    5.12.2): Digest = Base64(SHA1(nonce + created + password))."""
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    nonce = os.urandom(16)
    digest = base64.b64encode(
        hashlib.sha1(nonce + created.encode("utf-8") + password.encode("utf-8")).digest()
    ).decode("ascii")
    nonce_b64 = base64.b64encode(nonce).decode("ascii")
    return (
        f'<wsse:Security soapenv:mustUnderstand="1" xmlns:wsse="{_NS_WSSE}" xmlns:wsu="{_NS_WSU}">'
        f"<wsse:UsernameToken>"
        f"<wsse:Username>{_xml_escape(username)}</wsse:Username>"
        f'<wsse:Password Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordDigest">'
        f"{digest}</wsse:Password>"
        f'<wsse:Nonce EncodingType="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-soap-message-security-1.0#Base64Binary">'
        f"{nonce_b64}</wsse:Nonce>"
        f"<wsu:Created>{created}</wsu:Created>"
        f"</wsse:UsernameToken>"
        f"</wsse:Security>"
    )


def _envelope(body: str, username: str = "", password: str = "") -> str:
    header = f"<soapenv:Header>{_security_header(username, password)}</soapenv:Header>" if username else "<soapenv:Header/>"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<soapenv:Envelope xmlns:soapenv="{_NS_SOAP}" xmlns:tds="{_NS_TDS}" xmlns:trt="{_NS_TRT}" xmlns:tt="{_NS_TT}">'
        f"{header}<soapenv:Body>{body}</soapenv:Body></soapenv:Envelope>"
    )


def _soap_post(url: str, envelope: str, timeout: float) -> ET.Element:
    try:
        resp = requests.post(
            url,
            data=envelope.encode("utf-8"),
            headers={"Content-Type": "application/soap+xml; charset=utf-8"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise CameraConnectionError(f"ONVIF request to {redact_url(url)} failed: {exc}") from exc
    if resp.status_code >= 400:
        raise CameraConnectionError(
            f"ONVIF request to {redact_url(url)} returned HTTP {resp.status_code}: {resp.text[:300]}"
        )
    try:
        return ET.fromstring(resp.content)
    except ET.ParseError as exc:
        raise CameraConnectionError(f"ONVIF response from {redact_url(url)} was not valid XML: {exc}") from exc


def get_media_xaddr(device_url: str, username: str, password: str, timeout: float) -> str:
    """GetCapabilities on the device service -> the media service's XAddr (its own service URL)."""
    body = "<tds:GetCapabilities><tds:Category>Media</tds:Category></tds:GetCapabilities>"
    root = _soap_post(device_url, _envelope(body, username, password), timeout)
    xaddr = root.find(f".//{{{_NS_TT}}}Media/{{{_NS_TT}}}XAddr")
    if xaddr is None:
        # some devices put Media capabilities directly under the media-wsdl namespace instead of tt
        xaddr = root.find(f".//{{{_NS_TRT}}}XAddr")
    if xaddr is None or not xaddr.text:
        raise CameraConnectionError(f"ONVIF GetCapabilities at {redact_url(device_url)}: no Media service XAddr in response")
    return xaddr.text.strip()


def get_first_profile_token(media_url: str, username: str, password: str, timeout: float) -> str:
    root = _soap_post(media_url, _envelope("<trt:GetProfiles/>", username, password), timeout)
    profile = root.find(f".//{{{_NS_TRT}}}Profiles")
    if profile is None or "token" not in profile.attrib:
        raise CameraConnectionError(f"ONVIF GetProfiles at {redact_url(media_url)}: no profile token in response")
    return profile.attrib["token"]


def get_stream_uri(media_url: str, profile_token: str, username: str, password: str, timeout: float) -> str:
    body = (
        "<trt:GetStreamUri><trt:StreamSetup>"
        f'<tt:Stream xmlns:tt="{_NS_TT}">RTP-Unicast</tt:Stream>'
        f'<tt:Transport xmlns:tt="{_NS_TT}"><tt:Protocol>RTSP</tt:Protocol></tt:Transport>'
        "</trt:StreamSetup>"
        f"<trt:ProfileToken>{_xml_escape(profile_token)}</trt:ProfileToken>"
        "</trt:GetStreamUri>"
    )
    root = _soap_post(media_url, _envelope(body, username, password), timeout)
    uri = root.find(f".//{{{_NS_TT}}}Uri")
    if uri is None:
        uri = root.find(f".//{{{_NS_TRT}}}Uri")
    if uri is None or not uri.text:
        raise CameraConnectionError(f"ONVIF GetStreamUri at {redact_url(media_url)}: no stream URI in response")
    return uri.text.strip()


def _inject_credentials(rtsp_url: str, username: str, password: str) -> str:
    if not username:
        return rtsp_url
    _, _, rest = rtsp_url.partition("://")
    if "@" in rest:
        return rtsp_url  # camera already returned credentials embedded
    scheme = rtsp_url.split("://", 1)[0]
    return f"{scheme}://{username}:{password}@{rest}"


def resolve_stream_url(config: CameraConfig) -> str:
    """host/port/credentials -> a ready-to-use rtsp:// URL, credentials embedded."""
    if not config.host:
        raise ValueError(f"ONVIF camera {config.id!r}: config.host is required (or set config.url directly)")
    device_url = f"http://{config.host}:{config.port}{config.onvif_path or _DEFAULT_DEVICE_PATH}"
    timeout = config.connect_timeout_sec
    media_xaddr = get_media_xaddr(device_url, config.username, config.password, timeout)
    profile_token = config.profile_token or get_first_profile_token(media_xaddr, config.username, config.password, timeout)
    stream_uri = get_stream_uri(media_xaddr, profile_token, config.username, config.password, timeout)
    return _inject_credentials(stream_uri, config.username, config.password)


class ONVIFCamera(CameraSource):
    """Resolves an RTSP stream URI via ONVIF (unless config.url is already
    set, which skips resolution and is treated as a direct RTSP override),
    then delegates all frame reading to an internal RTSPCamera."""

    def __init__(self, config: CameraConfig):
        super().__init__(config)
        self._rtsp: RTSPCamera | None = None

    def open(self) -> None:
        stream_url = self.config.url or resolve_stream_url(self.config)
        rtsp = RTSPCamera(self.config, url=stream_url)
        rtsp.open()
        self._rtsp = rtsp

    def read(self):
        if self._rtsp is None:
            raise CameraConnectionError(f"ONVIF camera {self.config.id!r}: read() called before open()")
        return self._rtsp.read()

    def release(self) -> None:
        if self._rtsp is not None:
            self._rtsp.release()
            self._rtsp = None

    def native_resolution(self):
        return self._rtsp.native_resolution() if self._rtsp else None


# --- WS-Discovery (best-effort LAN probe) -----------------------------------


def _parse_probe_match(data: bytes) -> dict | None:
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return None
    xaddrs_el = root.find(f".//{{{_NS_D}}}XAddrs")
    scopes_el = root.find(f".//{{{_NS_D}}}Scopes")
    if xaddrs_el is None or not xaddrs_el.text:
        return None
    return {
        "xaddrs": xaddrs_el.text.split(),
        "scopes": scopes_el.text.split() if scopes_el is not None and scopes_el.text else [],
    }


def discover(timeout: float = 3.0) -> list[dict]:
    """Best-effort WS-Discovery probe for ONVIF NetworkVideoTransmitters on
    the local network. Returns [{"xaddrs": [...], "scopes": [...]}, ...]
    for every device that answered within `timeout` seconds. An empty
    result is not necessarily an error — many networks block UDP
    multicast — so the primary, reliable way to add a camera is still
    config-driven host/port (see resolve_stream_url)."""
    message_id = f"uuid:{uuid.uuid4()}"
    probe = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<soapenv:Envelope xmlns:soapenv="{_NS_SOAP}" xmlns:wsa="{_NS_WSA}" xmlns:d="{_NS_D}" '
        'xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
        f"<soapenv:Header><wsa:MessageID>{message_id}</wsa:MessageID>"
        "<wsa:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</wsa:To>"
        "<wsa:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</wsa:Action>"
        "</soapenv:Header>"
        "<soapenv:Body><d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></soapenv:Body>"
        "</soapenv:Envelope>"
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)
    results: list[dict] = []
    try:
        sock.sendto(probe.encode("utf-8"), (_DISCOVERY_MULTICAST_ADDR, _DISCOVERY_PORT))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                break
            parsed = _parse_probe_match(data)
            if parsed:
                results.append(parsed)
    except OSError:
        return []  # multicast unavailable on this interface/network — not fatal, see docstring
    finally:
        sock.close()
    return results
