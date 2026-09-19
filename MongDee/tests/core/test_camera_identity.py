import core.camera_identity as camera_identity


def test_list_directshow_devices_empty_on_non_windows(monkeypatch):
    monkeypatch.setattr(camera_identity.sys, "platform", "linux")
    assert camera_identity.list_directshow_devices() == {}


def test_list_directshow_devices_empty_when_pygrabber_missing(monkeypatch):
    monkeypatch.setattr(camera_identity.sys, "platform", "win32")
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pygrabber.dshow_graph" or name.startswith("pygrabber"):
            raise ImportError("no pygrabber")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert camera_identity.list_directshow_devices() == {}


def test_list_directshow_devices_returns_enumeration_order(monkeypatch):
    monkeypatch.setattr(camera_identity.sys, "platform", "win32")

    class FakeFilterGraph:
        def get_input_devices(self):
            return ["USB Camera", "HD WebCam", "USB Camera", "OBS Virtual Camera"]

    fake_module = type(camera_identity)("pygrabber.dshow_graph")
    fake_module.FilterGraph = FakeFilterGraph
    import sys
    monkeypatch.setitem(sys.modules, "pygrabber.dshow_graph", fake_module)
    monkeypatch.setitem(sys.modules, "pygrabber", type(camera_identity)("pygrabber"))

    devices = camera_identity.list_directshow_devices()
    assert devices == {0: "USB Camera", 1: "HD WebCam", 2: "USB Camera", 3: "OBS Virtual Camera"}


def test_is_virtual_camera_name():
    assert camera_identity.is_virtual_camera_name("OBS Virtual Camera")
    assert camera_identity.is_virtual_camera_name("obs virtual camera output")
    assert camera_identity.is_virtual_camera_name("ManyCam Virtual Webcam")
    assert not camera_identity.is_virtual_camera_name("USB Camera")
    assert not camera_identity.is_virtual_camera_name("HD WebCam")


# ------------------------------------------------- normalize_device_key
# Spec: MongDee multi-USB-camera root-cause repair — "DEVICE INDEX
# COLLISION". device values arrive as either int (auto-discovery) or str
# (the /settings "add camera" form, the --cameras CLI flag) for what may be
# the exact same physical camera; this is what BoothManager.add_camera and
# dedupe_camera_devices use to catch that.

def test_normalize_device_key_int_and_str_index_are_equal():
    assert camera_identity.normalize_device_key(0) == camera_identity.normalize_device_key("0")
    assert camera_identity.normalize_device_key(2) == camera_identity.normalize_device_key("2")
    assert camera_identity.normalize_device_key(" 2 ") == camera_identity.normalize_device_key(2)


def test_normalize_device_key_distinguishes_different_indices():
    assert camera_identity.normalize_device_key(0) != camera_identity.normalize_device_key(1)


def test_normalize_device_key_non_numeric_device_is_case_insensitive_string():
    assert camera_identity.normalize_device_key("/dev/video0") == camera_identity.normalize_device_key("/dev/video0")
    assert camera_identity.normalize_device_key("RTSP://Cam/Stream") == \
        camera_identity.normalize_device_key("rtsp://cam/stream")
    assert camera_identity.normalize_device_key("/dev/video0") != camera_identity.normalize_device_key("/dev/video1")


def test_normalize_device_key_never_conflates_index_with_path():
    # "0" (a numeric index) and a device path must never collide just
    # because both are strings.
    assert camera_identity.normalize_device_key(0) != camera_identity.normalize_device_key("/dev/video0")


# ------------------------------------------------- list_pnp_camera_devices

def test_list_pnp_camera_devices_empty_on_non_windows(monkeypatch):
    monkeypatch.setattr(camera_identity.sys, "platform", "linux")
    assert camera_identity.list_pnp_camera_devices() == []


def test_list_pnp_camera_devices_empty_when_powershell_fails(monkeypatch):
    monkeypatch.setattr(camera_identity.sys, "platform", "win32")

    class FakeResult:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(camera_identity.subprocess, "run", lambda *a, **k: FakeResult())
    assert camera_identity.list_pnp_camera_devices() == []


def test_list_pnp_camera_devices_empty_on_subprocess_exception(monkeypatch):
    monkeypatch.setattr(camera_identity.sys, "platform", "win32")

    def boom(*a, **k):
        raise OSError("powershell not found")

    monkeypatch.setattr(camera_identity.subprocess, "run", boom)
    assert camera_identity.list_pnp_camera_devices() == []


def test_list_pnp_camera_devices_parses_single_object_json(monkeypatch):
    # ConvertTo-Json on a single-element array (real PowerShell behavior
    # this code must handle) yields a bare JSON object, not a one-item list.
    monkeypatch.setattr(camera_identity.sys, "platform", "win32")

    class FakeResult:
        returncode = 0
        stdout = (
            '{"FriendlyName":"USB Camera",'
            '"InstanceId":"USB\\\\VID_4C4A&PID_4A55&MI_00\\\\7&7199EFD&0&0000",'
            '"LocationInfo":"0000.0014.0000.004.001.000.000.000.000",'
            '"ContainerId":"{92F4336D-AE71-11F1-B165-DC71966ECC5F}"}'
        )

    monkeypatch.setattr(camera_identity.subprocess, "run", lambda *a, **k: FakeResult())
    devices = camera_identity.list_pnp_camera_devices()
    assert devices == [{
        "friendly_name": "USB Camera",
        "instance_id": "USB\\VID_4C4A&PID_4A55&MI_00\\7&7199EFD&0&0000",
        "location_info": "0000.0014.0000.004.001.000.000.000.000",
        "container_id": "{92F4336D-AE71-11F1-B165-DC71966ECC5F}",
    }]


def test_list_pnp_camera_devices_parses_array_json(monkeypatch):
    monkeypatch.setattr(camera_identity.sys, "platform", "win32")

    class FakeResult:
        returncode = 0
        stdout = (
            '[{"FriendlyName":"USB Camera","InstanceId":"A","LocationInfo":"L1","ContainerId":"C1"},'
            '{"FriendlyName":"HD WebCam","InstanceId":"B","LocationInfo":"L2","ContainerId":"C2"}]'
        )

    monkeypatch.setattr(camera_identity.subprocess, "run", lambda *a, **k: FakeResult())
    devices = camera_identity.list_pnp_camera_devices()
    assert len(devices) == 2
    assert devices[0]["friendly_name"] == "USB Camera"
    assert devices[1]["friendly_name"] == "HD WebCam"


def test_list_pnp_camera_devices_empty_on_malformed_json(monkeypatch):
    monkeypatch.setattr(camera_identity.sys, "platform", "win32")

    class FakeResult:
        returncode = 0
        stdout = "not json {{{"

    monkeypatch.setattr(camera_identity.subprocess, "run", lambda *a, **k: FakeResult())
    assert camera_identity.list_pnp_camera_devices() == []


# ------------------------------------------------------ _parse_instance_id

def test_parse_instance_id_extracts_vid_pid():
    vid, pid, serial = camera_identity._parse_instance_id(
        "USB\\VID_4C4A&PID_4A55&MI_00\\7&7199EFD&0&0000")
    assert vid == "4C4A"
    assert pid == "4A55"


def test_parse_instance_id_synthesized_suffix_is_not_a_serial():
    # "7&7199EFD&0&0000" is a Windows-synthesized per-port instance suffix
    # (verified against this project's own real hardware registry) -- it
    # must never be reported as a genuine hardware serial number.
    _vid, _pid, serial = camera_identity._parse_instance_id(
        "USB\\VID_4C4A&PID_4A55&MI_00\\7&7199EFD&0&0000")
    assert serial is None


def test_parse_instance_id_real_serial_has_no_ampersand():
    _vid, _pid, serial = camera_identity._parse_instance_id(
        "USB\\VID_1234&PID_5678&MI_00\\ABC123SERIAL")
    assert serial == "ABC123SERIAL"


def test_parse_instance_id_none_input():
    assert camera_identity._parse_instance_id(None) == (None, None, None)


# ------------------------------------------------ get_physical_camera_identities

def test_identity_survives_index_reorder_same_port():
    # The exact scenario Phase 5 of the physical-identity spec asks for:
    # the same physical camera (same PnP present entry, same USB port) must
    # resolve to the same physical_id even though its DirectShow index
    # changed out from under it (Windows re-enumeration).
    pnp = [{"friendly_name": "USB Camera", "instance_id": "USB\\VID_4C4A&PID_4A55&MI_00\\7&AAA&0&0000",
            "location_info": "PORT-1", "container_id": "{X}"}]
    before = camera_identity.get_physical_camera_identities(
        directshow_devices={1: "USB Camera"}, pnp_devices=pnp)
    after = camera_identity.get_physical_camera_identities(
        directshow_devices={5: "USB Camera"}, pnp_devices=pnp)
    assert before[1].physical_id == after[5].physical_id
    assert before[1].identity_source == "vid_pid_location"


def test_same_name_different_physical_cameras_are_distinguished():
    # Phase 4/7: identical friendly names + identical VID/PID (same camera
    # model) must NOT be deduplicated into one physical camera when their
    # USB port locations differ.
    pnp = [
        {"friendly_name": "USB Camera", "instance_id": "USB\\VID_4C4A&PID_4A55&MI_00\\7&AAA&0&0000",
         "location_info": "PORT-A", "container_id": "{A}"},
        {"friendly_name": "USB Camera", "instance_id": "USB\\VID_4C4A&PID_4A55&MI_00\\7&BBB&0&0000",
         "location_info": "PORT-B", "container_id": "{B}"},
    ]
    identities = camera_identity.get_physical_camera_identities(
        directshow_devices={1: "USB Camera", 2: "USB Camera"}, pnp_devices=pnp)
    assert identities[1].physical_id != identities[2].physical_id
    assert identities[1].location_info == "PORT-A"
    assert identities[2].location_info == "PORT-B"


def test_real_serial_wins_over_vid_pid_location():
    pnp = [{"friendly_name": "Pro Camera", "instance_id": "USB\\VID_1234&PID_5678&MI_00\\REALSERIAL42",
            "location_info": "PORT-1", "container_id": "{X}"}]
    identities = camera_identity.get_physical_camera_identities(
        directshow_devices={0: "Pro Camera"}, pnp_devices=pnp)
    assert identities[0].identity_source == "serial"
    assert identities[0].serial == "REALSERIAL42"
    assert identities[0].physical_id == "serial:1234:5678:REALSERIAL42"


def test_no_matching_pnp_entry_falls_back_to_name_index():
    # A virtual camera (OBS Virtual Camera, etc.) never registers in the PnP
    # Camera class -- must degrade to the documented fallback, not crash or
    # silently omit the index.
    identities = camera_identity.get_physical_camera_identities(
        directshow_devices={2: "OBS Virtual Camera"}, pnp_devices=[])
    assert identities[2].identity_source == "name_index_fallback"
    assert identities[2].physical_id == "name_index:obs virtual camera:2"


def test_more_directshow_indices_than_present_pnp_entries_fall_back_individually():
    # 3 DirectShow entries named "USB Camera" but only 1 present PnP match
    # (e.g. two were just unplugged) -- the matched one gets a real
    # physical_id, the other two degrade to fallback instead of guessing.
    pnp = [{"friendly_name": "USB Camera", "instance_id": "USB\\VID_4C4A&PID_4A55&MI_00\\7&AAA&0&0000",
            "location_info": "PORT-1", "container_id": "{X}"}]
    identities = camera_identity.get_physical_camera_identities(
        directshow_devices={1: "USB Camera", 2: "USB Camera", 3: "USB Camera"}, pnp_devices=pnp)
    assert identities[1].identity_source == "vid_pid_location"
    assert identities[2].identity_source == "name_index_fallback"
    assert identities[3].identity_source == "name_index_fallback"


# ---------------------------------------------- resolve_current_index_for_physical_id

def test_resolve_current_index_finds_relocated_camera(monkeypatch):
    pnp = [{"friendly_name": "USB Camera", "instance_id": "USB\\VID_4C4A&PID_4A55&MI_00\\7&AAA&0&0000",
            "location_info": "PORT-1", "container_id": "{X}"}]
    monkeypatch.setattr(camera_identity, "list_directshow_devices", lambda: {5: "USB Camera"})
    monkeypatch.setattr(camera_identity, "list_pnp_camera_devices", lambda: pnp)
    physical_id = "vid_pid_location:4C4A:4A55:PORT-1"
    assert camera_identity.resolve_current_index_for_physical_id(physical_id) == 5


def test_resolve_current_index_returns_none_when_not_present(monkeypatch):
    monkeypatch.setattr(camera_identity, "list_directshow_devices", lambda: {0: "HD WebCam"})
    monkeypatch.setattr(camera_identity, "list_pnp_camera_devices", lambda: [])
    assert camera_identity.resolve_current_index_for_physical_id("serial:AAAA:BBBB:XYZ") is None
