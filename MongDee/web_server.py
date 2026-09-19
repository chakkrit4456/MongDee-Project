"""MONGDEE AI Booth OS — Web entrypoint.

Runs the exact same AI Vision / Product Assistant / Readiness Check /
Health Monitoring / Dashboard / Trainer feature set as app.py, but served
over HTTP so any browser — on this machine or another device on the same
network (a tablet at the booth counter, for example) — can use it without
installing PySide6.

Usage:
    python web_server.py --booth-name "MONGDEE Demo Booth" --event-id "1-Day-at-IMPACT"
    python web_server.py --host 0.0.0.0 --port 8000   # reachable from other devices on the LAN
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
import uuid
import webbrowser
from pathlib import Path

# This module's own console messages are Thai — stdout/stderr default to the
# OS codepage (e.g. cp1252/cp874 on Windows) whenever they aren't attached to
# a UTF-8-aware terminal (piped/redirected output, a log file, some legacy
# consoles), which raises UnicodeEncodeError and kills the process before it
# can print the very port-conflict/shutdown messages this file exists to
# show. Forcing UTF-8 here (errors="replace" so a still-unmappable byte
# degrades to "?" instead of crashing) makes those messages unconditionally
# reliable instead of only working in whichever terminal a developer tested.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import uvicorn

from core import database as db
from core.device import device_label, resolve_device
from core.performance import AdaptiveConfig, format_benchmark_report, load_adaptive_config
from core.products import ProductCatalog
from core.recognizer import ProductRecognizer
from core.vision import discover_cameras
from web.booth_manager import (
    BoothManager, booth_settings_path_for, compute_camera_skip_indices, load_active_booth_id,
    load_camera_settings,
)
from web.server import create_app

ROOT = Path(__file__).resolve().parent

# Auto-detected default locations for a locally-trained FairFace model (see
# train.py) — same "use it automatically if present, no flag required"
# pattern detect.py/test_model.py already use for --yunet-model. Only ever
# used as an argparse *default*; an explicit --fairface-checkpoint/--yunet-model
# still overrides these.
_DEFAULT_FAIRFACE_CHECKPOINT = ROOT / "models" / "fairface" / "best_model_state_dict.pt"
try:  # prefer the webcam-adapted weights when tools/finetune_fairface_gender.py produced them
    from core.attributes import resolve_fairface_checkpoint as _resolve_fairface_checkpoint
    _DEFAULT_FAIRFACE_CHECKPOINT = _resolve_fairface_checkpoint(_DEFAULT_FAIRFACE_CHECKPOINT)
except Exception:  # pragma: no cover - never let a helper import stop start-up
    pass
_DEFAULT_YUNET_MODEL = ROOT / "models" / "face_detector" / "face_detection_yunet_2023mar.onnx"


def _ensure_port_available(host: str, port: int) -> None:
    """Single-instance protection: refuse to start if `port` is already
    bound, with a clear, actionable message up front — instead of spending
    a minute loading the YOLO/recognizer models and opening cameras only to
    crash on WinError 10048 right at the very end.

    Deliberately does NOT try to detect/kill whatever already holds the
    port: a bare process list has no reliable way to tell "another MongDee
    instance" apart from an unrelated Python process also named
    python.exe/uvicorn, and guessing wrong risks killing the operator's own
    unrelated work. A clear error asking them to stop the other instance
    themselves (Ctrl+C in its own terminal) is the safe default; this is
    exactly the "check the port before starting" option this project's own
    lifecycle requirements call out as acceptable.
    """
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind((probe_host, port))
        except OSError:
            print(f"[WEB] พอร์ต {port} มีการใช้งานอยู่แล้ว — MongDee เวอร์ชันก่อนหน้าน่าจะยังรันอยู่ "
                  f"(หรือโปรแกรมอื่นใช้พอร์ตนี้อยู่)")
            print(f"[WEB] ปิด server ตัวเดิมก่อน (กด Ctrl+C ที่หน้าต่างเทอร์มินัลที่รันอยู่) แล้วรันคำสั่งนี้ใหม่")
            print(f"[WEB] หรือถ้าไม่แน่ใจว่าตัวเดิมยังอยู่ไหม ให้ตรวจสอบด้วย: netstat -ano | findstr :{port}")
            raise SystemExit(1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--booth-id", default=None,
                         help="Omit to reuse the identity last saved via the /settings page, "
                              "or a fresh random ID if none was ever saved.")
    parser.add_argument("--booth-name", default=None)
    parser.add_argument("--event-id", default=None)
    parser.add_argument("--cameras", default=None,
                         help="Comma-separated device paths, e.g. /dev/video0,/dev/video2. "
                              "Omit to auto-discover.")
    parser.add_argument("--model", default=str(ROOT / "yolo11n.pt"))
    parser.add_argument("--gender-model", default=None,
                         help="Path to a Caffe .caffemodel or ONNX gender classifier "
                              "(female/male). When set, person detection boxes on the "
                              "live stream turn red for female / blue for male instead "
                              "of the default neutral color. Omitted by default — no "
                              "gender model ships with this repo (bring your own).")
    parser.add_argument("--gender-prototxt", default=None,
                         help="Matching Caffe .prototxt for --gender-model (Caffe only; "
                              "omit for an ONNX model).")
    parser.add_argument("--age-model", default=None,
                         help="Path to a Caffe .caffemodel or ONNX age-group classifier. "
                              "Requires --gender-model to also be set (they share the same "
                              "backend). DEPRECATED: age / child detection was removed "
                              "(only male / female / product are reported), so this "
                              "option no longer changes any box.")
    parser.add_argument("--age-prototxt", default=None,
                         help="Matching Caffe .prototxt for --age-model (Caffe only; omit "
                              "for an ONNX model).")
    parser.add_argument("--fairface-checkpoint",
                         default=str(_DEFAULT_FAIRFACE_CHECKPOINT) if _DEFAULT_FAIRFACE_CHECKPOINT.is_file() else None,
                         help="Path to a FairFace ResNet34 checkpoint (e.g. "
                              "res34_fair_align_multi_7_20190809.pt from "
                              "https://github.com/joojs/fairface, or one trained locally via "
                              "train.py). Auto-detected under models/fairface/ if present "
                              "(train.py's own output location). When set together with "
                              "--yunet-model, enables Global-Person-scoped (Re-ID-aware) "
                              "age/gender attribute analysis shown on the Dashboard, AND "
                              "(unless --gender-model is also given) drives the live "
                              "male/female on-screen box coloring by default — FairFace "
                              "is a drop-in replacement for the legacy --gender-model/"
                              "--age-model backend (see core/attributes.py's FairFaceBackend). "
                              "Pass --gender-model to opt back into the legacy Caffe/ONNX path "
                              "instead. Omit both --fairface-checkpoint and --yunet-model "
                              "explicitly (pointing at a nonexistent path) to disable "
                              "attribute analysis entirely.")
    parser.add_argument("--yunet-model",
                         default=str(_DEFAULT_YUNET_MODEL) if _DEFAULT_YUNET_MODEL.is_file() else None,
                         help="Path to OpenCV's YuNet face detector ONNX model "
                              "(face_detection_yunet_2023mar.onnx from "
                              "https://github.com/opencv/opencv_zoo). Auto-detected under "
                              "models/face_detector/ if present. Required together with "
                              "--fairface-checkpoint — FairFace is only ever run on a "
                              "detected face crop, never a full-body crop.")
    parser.add_argument("--no-face-detect", action="store_true",
                         help="turn off the full-frame face detection overlay (needs the YuNet model)")
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH))
    parser.add_argument("--products", default=str(ROOT / "products.json"))
    parser.add_argument("--host", default="127.0.0.1",
                         help="0.0.0.0 to allow other devices on the LAN to connect")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="don't auto-open a browser tab")
    parser.add_argument("--performance-config", default=None,
                         help="Path to a performance.json (see configs/performance.example.json) "
                              "tuning the Adaptive Controller's thresholds. Omit to use built-in "
                              "defaults.")
    parser.add_argument("--benchmark", type=float, default=None, metavar="SECONDS",
                         help="Run cameras + AI headlessly for SECONDS (no HTTP server, no browser), "
                              "then print a Camera Benchmark report (see "
                              "MongDee_Multi_Webcam_Real_Time_Performance_Prompt.md section 45) and exit.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Fail fast, before spending a minute loading models/opening cameras,
    # if another instance is already listening here — the benchmark path
    # below never binds a port at all, so it's exempt.
    if args.benchmark is None:
        _ensure_port_available(args.host, args.port)
    # Every camera thread's non-fatal problems (a camera that won't open, a
    # frame that fails to encode, ...) go through Python logging rather than
    # print(), so they'd otherwise vanish silently — nothing here previously
    # configured a handler for them. This makes them show up in the same
    # terminal window this was launched from, which is the main tool for
    # answering "why isn't camera X showing anything".
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db_path = Path(args.db)
    db.init_db(db_path)

    # Booth ID / Event ID are real registry rows (core.database's `booths`/
    # `events` tables), not free text — resolve which registry Booth this
    # process should report as. An explicit --booth-id always wins (and
    # auto-registers itself + its event if new, so README's "simulate a
    # second booth" CLI pattern keeps working); otherwise reuse whichever
    # booth was last activated via /settings, so a plain restart doesn't
    # revert to a random identity. A brand-new database bootstraps one booth
    # (+ event) from the CLI defaults so zero-config startup is unchanged.
    if args.booth_id:
        if args.event_id:
            db.ensure_event(db_path, args.event_id)
        if not db.get_booth(db_path, args.booth_id):
            db.create_booth(db_path, args.booth_id, args.booth_name or args.booth_id, args.event_id)
        active_booth_id = args.booth_id
    elif db.count_booths(db_path) == 0:
        boot_event_id = args.event_id or "1-Day-at-IMPACT"
        db.ensure_event(db_path, boot_event_id)
        active_booth_id = f"BOOTH-{uuid.uuid4().hex[:6].upper()}"
        db.create_booth(db_path, active_booth_id, args.booth_name or "MONGDEE Demo Booth", boot_event_id)
    else:
        active_booth_id = load_active_booth_id(booth_settings_path_for(db_path))
        if not active_booth_id or not db.get_booth(db_path, active_booth_id):
            active_booth_id = db.list_booths(db_path)[0]["id"]

    booth_row = db.get_booth(db_path, active_booth_id)
    booth_id = booth_row["id"]
    booth_name = booth_row["name"]
    event_id = booth_row["event_id"] or ""

    if args.cameras:
        devices = [d.strip() for d in args.cameras.split(",") if d.strip()]
    else:
        camera_settings = load_camera_settings(booth_settings_path_for(db_path))
        builtin_skip = compute_camera_skip_indices(camera_settings)
        if builtin_skip:
            print(f"[BUILTIN] Camera/virtual device disabled by configuration (index {sorted(builtin_skip)})")
        print("[WEB] ไม่ได้ระบุกล้อง กำลังค้นหากล้องอัตโนมัติ...")
        devices = discover_cameras(skip=builtin_skip)
        print(f"[WEB] พบกล้อง: {devices}")

    if not devices:
        print("[WEB] ไม่พบเว็บแคมที่ใช้งานได้ — เปิดหน้าเว็บต่อได้ตามปกติ "
              "เชื่อมต่อกล้องแล้วรันคำสั่งนี้ใหม่เพื่อให้บูธเห็นกล้อง")

    camera_devices = [(f"CAM-{i+1}", dev) for i, dev in enumerate(devices)]

    print("[WEB] กำลังโหลดโมเดล AI Vision (YOLO11)...")
    from ultralytics import YOLO

    model = YOLO(args.model)
    model_device = resolve_device()
    print(f"[WEB] ใช้งานอุปกรณ์ประมวลผล: {device_label(model_device)}")

    catalog = ProductCatalog(Path(args.products))

    print("[WEB] กำลังโหลดโมเดล AI จดจำสินค้า (Product Recognizer)...")
    recognizer = ProductRecognizer(device=model_device)

    gender_age_backend = None
    if args.gender_model:
        from vision.attributes.gender_age_backend import OpenCVDnnGenderAgeBackend

        print("[WEB] กำลังโหลดโมเดลจำแนกเพศ (สำหรับสีกรอบ detect ผู้หญิง/ผู้ชาย)..."
              if not args.age_model else
              "[WEB] กำลังโหลดโมเดลจำแนกเพศ/เด็ก (สำหรับสีกรอบ detect ผู้หญิง/ผู้ชาย/เด็ก)...")
        gender_age_backend = OpenCVDnnGenderAgeBackend(
            gender_model=args.gender_model, gender_prototxt=args.gender_prototxt,
            age_model=args.age_model, age_prototxt=args.age_prototxt,
        )
    elif args.age_model:
        print("[WEB] ข้าม --age-model: ต้องระบุ --gender-model ด้วย (ใช้ backend เดียวกัน)")

    attribute_backend = None
    if args.fairface_checkpoint and args.yunet_model:
        from core.attributes import FairFaceBackend, YuNetFaceDetector

        print("[WEB] กำลังโหลดโมเดล FairFace (วิเคราะห์เพศ/อายุต่อ Global Person ID)...")
        try:
            attribute_backend = FairFaceBackend(
                checkpoint_path=args.fairface_checkpoint,
                face_detector=YuNetFaceDetector(onnx_model_path=args.yunet_model),
                device=model_device,
            )
            if gender_age_backend is None:
                # No explicit --gender-model given -- FairFaceBackend already
                # satisfies the same predict_gender/predict_age_group duck-typed
                # interface (see its own docstring in core/attributes.py) and is
                # a drop-in for the legacy per-frame classifier, so the live
                # on-screen box also gets real male/female colors by
                # default instead of staying stuck on the generic "UNKNOWN"
                # bucket. The same loaded model instance is reused for both --
                # its forward pass is already lock-protected for concurrent use.
                gender_age_backend = attribute_backend
                print("[WEB] ใช้โมเดล FairFace เป็นตัวจำแนกเพศ/อายุสำหรับกรอบ detect บนหน้าจอด้วย "
                      "(male/female) — ระบุ --gender-model เพื่อกลับไปใช้โมเดล Caffe/ONNX เดิมแทน")
        except Exception as exc:
            print(f"[WEB] โหลด FairFace ไม่สำเร็จ ({exc}) — ปิดใช้งานการวิเคราะห์เพศ/อายุแบบ Global Person")
    elif args.fairface_checkpoint or args.yunet_model:
        print("[WEB] ข้าม FairFace: ต้องระบุทั้ง --fairface-checkpoint และ --yunet-model คู่กัน")

    # Full-frame face detection (YuNet, the same ONNX already used for FairFace): draws FACE boxes,
    # binds each face to a person track and keeps the best shot. Independent of FairFace, so it works
    # with only the YuNet model present. Disable with --no-face-detect.
    face_detector = None
    if args.yunet_model and not args.no_face_detect:
        try:
            from core.face import YuNetFaceDetector as FullFrameFaceDetector

            # upscale 2.0: the cameras stream 320x240, where a face is often only 20-40 px wide
            face_detector = FullFrameFaceDetector(str(args.yunet_model), upscale=2.0)
            print("[WEB] เปิดระบบตรวจจับใบหน้า (YuNet เต็มเฟรม) — ปิดได้ด้วย --no-face-detect")
        except Exception as exc:
            print(f"[WEB] เปิดระบบตรวจจับใบหน้าไม่สำเร็จ ({exc}) — ทำงานต่อโดยไม่มี face detection")

    adaptive_config = AdaptiveConfig()
    if args.performance_config:
        adaptive_config = load_adaptive_config(args.performance_config)

    booth = BoothManager(
        booth_id=booth_id,
        booth_name=booth_name,
        event_id=event_id,
        camera_devices=camera_devices,
        model=model,
        model_device=model_device,
        catalog=catalog,
        recognizer=recognizer,
        db_path=db_path,
        gender_age_backend=gender_age_backend,
        attribute_backend=attribute_backend,
        adaptive_config=adaptive_config,
        face_detector=face_detector,
    )
    booth.activate_booth(booth_id)  # persist active_booth_id (covers a freshly bootstrapped/new booth)

    if args.benchmark is not None:
        run_benchmark(booth, args.benchmark)
        return

    app = create_app(booth)

    url = f"http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}/"
    print(f"[WEB] เปิดใช้งานได้ที่: {url}")
    if args.host == "0.0.0.0":
        print("[WEB] เครื่องอื่นในวง LAN เดียวกันเข้าถึงได้ที่ http://<IP ของเครื่องนี้>:%d/" % args.port)
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    except KeyboardInterrupt:
        # uvicorn already catches SIGINT itself and runs the ASGI shutdown
        # lifespan (which calls booth.stop_scheduled() — see
        # web/server.py's create_app) before returning normally; this only
        # catches the rarer case where Ctrl+C's KeyboardInterrupt slips
        # past that (a known Windows asyncio timing gap) and would
        # otherwise skip straight past cleanup.
        pass
    finally:
        # Defense in depth, not redundant busywork: booth.stop() is a no-op
        # if already stopped, so calling it again here costs nothing on the
        # normal path, but guarantees every camera thread is signaled to
        # stop and every cv2 capture handle is released even if the ASGI
        # shutdown event above never ran at all.
        print("[WEB] กำลังหยุด server และปิดกล้องทั้งหมด...")
        booth.stop_scheduled()
        print("[WEB] หยุดเรียบร้อยแล้ว — คืนพอร์ตและกล้องทั้งหมดแล้ว")


def run_benchmark(booth: BoothManager, seconds: float) -> None:
    """--benchmark SECONDS: runs the exact same camera+AI pipeline the web
    app uses (no HTTP server, no browser needed) for a fixed duration, then
    prints the report format from
    MongDee_Multi_Webcam_Real_Time_Performance_Prompt.md section 45. Useful
    for measuring what N cameras on this specific machine can sustain
    before deploying, or for comparing before/after a hardware or config
    change."""
    print(f"[BENCHMARK] เริ่มทดสอบ {len(booth.camera_ids)} กล้อง เป็นเวลา {seconds:.0f} วินาที...")
    start = time.time()
    booth.start()
    try:
        while time.time() - start < seconds:
            time.sleep(1.0)
    finally:
        booth.stop()
    elapsed = time.time() - start
    print()
    print(format_benchmark_report(booth.hardware_profile, booth.performance_monitor, elapsed))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Last-resort safety net: main()'s own try/finally around
        # uvicorn.run() already handles the normal Ctrl+C path (including
        # camera cleanup) — this only stops a raw traceback from printing
        # if a second Ctrl+C or an interrupt during startup/shutdown itself
        # lands here instead.
        print("\n[WEB] Shutdown requested — stopped.")
