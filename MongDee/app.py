"""MONGDEE AI Booth OS — entrypoint.

Launches the booth window: N webcams running AI Vision concurrently, the AI
Product Assistant, Booth Readiness Check, Booth Health Monitoring, and a
link into the AI Dashboard & Analytics window. See README.md for the full
feature-to-spec mapping.

Usage:
    python app.py --booth-id BOOTH-01 --booth-name "MONGDEE Demo Booth" \
        --event-id "1-Day-at-IMPACT" --cameras /dev/video0,/dev/video2
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from core import database as db
from core.assistant import AIAssistant
from core.device import device_label, resolve_device
from core.performance import AdaptiveConfig, load_adaptive_config
from core.products import ProductCatalog
from core.recognizer import ProductRecognizer
from core.vision import discover_cameras
from ui.dashboard_window import DashboardWindow
from ui.main_window import MainWindow
from ui.trainer_window import TrainerWindow
from web.booth_manager import (
    dedupe_camera_devices, booth_settings_path_for, compute_camera_skip_indices, load_camera_settings,
)

ROOT = Path(__file__).resolve().parent

# See web_server.py's identical constants — auto-detected default locations
# for a locally-trained FairFace model (train.py), used automatically if
# present without needing an explicit flag.
_DEFAULT_FAIRFACE_CHECKPOINT = ROOT / "models" / "fairface" / "best_model_state_dict.pt"
try:  # prefer the webcam-adapted weights when tools/finetune_fairface_gender.py produced them
    from core.attributes import resolve_fairface_checkpoint as _resolve_fairface_checkpoint
    _DEFAULT_FAIRFACE_CHECKPOINT = _resolve_fairface_checkpoint(_DEFAULT_FAIRFACE_CHECKPOINT)
except Exception:  # pragma: no cover - never let a helper import stop start-up
    pass
_DEFAULT_YUNET_MODEL = ROOT / "models" / "face_detector" / "face_detection_yunet_2023mar.onnx"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--booth-id", default=f"BOOTH-{uuid.uuid4().hex[:6].upper()}")
    parser.add_argument("--booth-name", default="MONGDEE Demo Booth")
    parser.add_argument("--event-id", default="1-Day-at-IMPACT")
    parser.add_argument("--cameras", default=None,
                         help="Comma-separated device paths, e.g. /dev/video0,/dev/video2. "
                              "Omit to auto-discover.")
    parser.add_argument("--model", default=str(ROOT / "yolo11n.pt"))
    parser.add_argument("--gender-model", default=None,
                         help="Path to a Caffe .caffemodel or ONNX gender classifier "
                              "(female/male). When set, person boxes turn red for "
                              "female / blue for male. Omitted by default (bring your own).")
    parser.add_argument("--gender-prototxt", default=None,
                         help="Matching Caffe .prototxt for --gender-model (Caffe only).")
    parser.add_argument("--age-model", default=None,
                         help="Path to a Caffe .caffemodel or ONNX age-group classifier. "
                              "DEPRECATED: age / child detection was removed (only "
                              "male / female / product are reported); this option no "
                              "longer changes any box.")
    parser.add_argument("--age-prototxt", default=None,
                         help="Matching Caffe .prototxt for --age-model (Caffe only).")
    parser.add_argument("--fairface-checkpoint",
                         default=str(_DEFAULT_FAIRFACE_CHECKPOINT) if _DEFAULT_FAIRFACE_CHECKPOINT.is_file() else None,
                         help="Path to a FairFace ResNet34 checkpoint (see web_server.py's "
                              "--fairface-checkpoint for details). Auto-detected under "
                              "models/fairface/ if present. Unless --gender-model is also "
                              "given, this drives the live male/female on-screen box "
                              "coloring by default (drop-in for the legacy Caffe/ONNX "
                              "backend — see core/attributes.py's FairFaceBackend).")
    parser.add_argument("--yunet-model",
                         default=str(_DEFAULT_YUNET_MODEL) if _DEFAULT_YUNET_MODEL.is_file() else None,
                         help="Path to OpenCV's YuNet face detector ONNX model. Auto-detected "
                              "under models/face_detector/ if present. Required together with "
                              "--fairface-checkpoint.")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--db", default=str(db.DEFAULT_DB_PATH))
    parser.add_argument("--products", default=str(ROOT / "products.json"))
    parser.add_argument("--performance-config", default=None,
                         help="Path to a performance.json (see configs/performance.example.json) "
                              "tuning the Adaptive Controller's thresholds. Omit to use built-in "
                              "defaults.")
    return parser.parse_args()


def load_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    db_path = Path(args.db)
    db.init_db(db_path)

    if args.cameras:
        devices = [d.strip() for d in args.cameras.split(",") if d.strip()]
    else:
        camera_settings = load_camera_settings(booth_settings_path_for(db_path))
        builtin_skip = compute_camera_skip_indices(camera_settings)
        if builtin_skip:
            print(f"[BOOTH] Camera/virtual device disabled by configuration (index {sorted(builtin_skip)})")
        print("[BOOTH] ไม่ได้ระบุกล้อง กำลังค้นหากล้องอัตโนมัติ...")
        devices = discover_cameras(skip=builtin_skip)
        print(f"[BOOTH] พบกล้อง: {devices}")

    app = QApplication(sys.argv)

    if not devices:
        QMessageBox.critical(None, "ไม่พบกล้อง", "ไม่พบเว็บแคมที่ใช้งานได้ กรุณาเชื่อมต่อกล้องแล้วลองใหม่")
        sys.exit(1)

    # Same DEVICE INDEX COLLISION guard as web/booth_manager.py's
    # BoothManager.__init__ (see dedupe_camera_devices's own docstring) —
    # ui/main_window.py's MainWindow._create_workers has no duplicate check
    # of its own, so a bad --cameras value here would otherwise silently
    # give two camera panels the exact same physical camera's video.
    camera_devices = dedupe_camera_devices([(f"CAM-{i+1}", dev) for i, dev in enumerate(devices)])

    print("[BOOTH] กำลังโหลดโมเดล AI Vision (YOLO11)...")
    model = load_model(args.model)
    model_device = resolve_device()
    print(f"[BOOTH] ใช้งานอุปกรณ์ประมวลผล: {device_label(model_device)}")

    catalog = ProductCatalog(Path(args.products))
    assistant = AIAssistant(catalog)
    if not assistant.tts_available:
        print(f"[BOOTH] เสียงพูด (TTS) ไม่พร้อมใช้งาน: {assistant.tts_error} "
              "— ระบบจะยังทำงานได้แบบข้อความ")

    print("[BOOTH] กำลังโหลดโมเดล AI จดจำสินค้า (Product Recognizer)...")
    recognizer = ProductRecognizer(device=model_device)

    gender_age_backend = None
    if args.gender_model:
        from vision.attributes.gender_age_backend import OpenCVDnnGenderAgeBackend

        print("[BOOTH] กำลังโหลดโมเดลจำแนกเพศ (สำหรับสีกรอบ detect ผู้หญิง/ผู้ชาย)..."
              if not args.age_model else
              "[BOOTH] กำลังโหลดโมเดลจำแนกเพศ/เด็ก (สำหรับสีกรอบ detect ผู้หญิง/ผู้ชาย/เด็ก)...")
        gender_age_backend = OpenCVDnnGenderAgeBackend(
            gender_model=args.gender_model, gender_prototxt=args.gender_prototxt,
            age_model=args.age_model, age_prototxt=args.age_prototxt,
        )
    elif args.age_model:
        print("[BOOTH] ข้าม --age-model: ต้องระบุ --gender-model ด้วย (ใช้ backend เดียวกัน)")

    if gender_age_backend is None and args.fairface_checkpoint and args.yunet_model:
        from core.attributes import FairFaceBackend, YuNetFaceDetector

        print("[BOOTH] กำลังโหลดโมเดล FairFace (สำหรับสีกรอบ detect ผู้หญิง/ผู้ชาย/เด็ก)...")
        try:
            gender_age_backend = FairFaceBackend(
                checkpoint_path=args.fairface_checkpoint,
                face_detector=YuNetFaceDetector(onnx_model_path=args.yunet_model),
                device=model_device,
            )
        except Exception as exc:
            print(f"[BOOTH] โหลด FairFace ไม่สำเร็จ ({exc}) — กรอบ detect จะไม่มีสี male/female")
    elif gender_age_backend is None and (args.fairface_checkpoint or args.yunet_model):
        print("[BOOTH] ข้าม FairFace: ต้องระบุทั้ง --fairface-checkpoint และ --yunet-model คู่กัน")

    dashboard_ref = {}
    trainer_ref = {}

    def open_dashboard():
        if "window" not in dashboard_ref or not dashboard_ref["window"].isVisible():
            dashboard_ref["window"] = DashboardWindow(db_path, default_event_id=args.event_id)
        dashboard_ref["window"].show()
        dashboard_ref["window"].raise_()
        dashboard_ref["window"].activateWindow()

    def open_trainer():
        if "window" not in trainer_ref or not trainer_ref["window"].isVisible():
            trainer_ref["window"] = TrainerWindow(catalog, recognizer, model, model_device)
        trainer_ref["window"].show()
        trainer_ref["window"].raise_()
        trainer_ref["window"].activateWindow()

    adaptive_config = load_adaptive_config(args.performance_config) if args.performance_config else AdaptiveConfig()

    window = MainWindow(
        booth_id=args.booth_id,
        booth_name=args.booth_name,
        event_id=args.event_id,
        camera_devices=camera_devices,
        model=model,
        model_device=model_device,
        catalog=catalog,
        assistant=assistant,
        db_path=db_path,
        recognizer=recognizer,
        dashboard_launcher=open_dashboard,
        trainer_launcher=open_trainer,
        gender_age_backend=gender_age_backend,
        adaptive_config=adaptive_config,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
