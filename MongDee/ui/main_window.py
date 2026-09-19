"""Main booth window: live multi-camera AI Vision + AI Product Assistant."""

from __future__ import annotations

import json
import math
import time

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core import database as db
from core.aggregator import DetectionAggregator
from core.performance import AdaptiveController, PerformanceMonitor, detect_hardware
from core.readiness import readiness_to_json, run_readiness_check
from ui.qt_camera_bridge import QtCameraWorker as CameraWorker

HEARTBEAT_INTERVAL_MS = 30_000

STATUS_COLORS = {
    "online": "#2ecc71",
    "offline": "#e74c3c",
    "error": "#e67e22",
    "disabled": "#4a5568",
    "connecting": "#3498db",
    "reconnecting": "#f39c12",
    "degraded": "#f1c40f",
    "unknown": "#7f8c8d",
}


def frame_to_pixmap(frame_bgr) -> QPixmap:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


class CameraPanel(QFrame):
    def __init__(self, camera_id: str, device: str):
        super().__init__()
        self.camera_id = camera_id
        self.device = device
        self.enabled = True

        self.video = QLabel("กำลังเชื่อมต่อ...")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setMinimumSize(400, 300)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video.setStyleSheet(
            "QLabel { background: #111318; color: #aeb4c0; border-radius: 10px; font-size: 15px; }"
        )

        self.dot = QLabel("●")
        self.dot.setStyleSheet(f"color: {STATUS_COLORS['unknown']}; font-size: 16px;")

        title = QLabel(f"{camera_id}  ({device})")
        title.setFont(QFont("Sans", 11, QFont.Bold))
        title.setStyleSheet("color: white;")

        self.toggle_btn = QPushButton("⏻ ปิดกล้อง")
        self.toggle_btn.setFixedWidth(112)
        self.toggle_btn.setToolTip("ปิด/เปิดกล้องนี้ (สำหรับกล้องที่ไม่ได้ใช้งาน)")

        header = QHBoxLayout()
        header.addWidget(self.dot)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.toggle_btn)

        self.status_label = QLabel("กำลังเชื่อมต่อ...")
        self.status_label.setStyleSheet("color: #b7bdc9; font-size: 12px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)
        layout.addLayout(header)
        layout.addWidget(self.video, 1)
        layout.addWidget(self.status_label)

        self.setStyleSheet(
            "QFrame { background: #181a20; border: 1px solid #30333d; border-radius: 12px; }"
        )

    def set_frame(self, frame_bgr):
        pixmap = frame_to_pixmap(frame_bgr)
        self.video.setPixmap(
            pixmap.scaled(self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def set_status(self, status: str, message: str):
        color = STATUS_COLORS.get(status, STATUS_COLORS["unknown"])
        self.dot.setStyleSheet(f"color: {color}; font-size: 16px;")
        self.status_label.setText(message)
        # "degraded" means the AI side is throttled, not that frames stopped
        # arriving (see _on_camera_degraded) — video keeps flowing via
        # set_frame(), so this must not blank it out with an error overlay.
        if status not in ("online", "degraded"):
            self.video.setText(f"❌ {message}")

    def set_enabled_ui(self, enabled: bool):
        """Reflect a set_enabled() toggle in the panel itself — the actual
        "disabled" status text/color arrives a moment later via the worker's
        status_changed signal, but the button should flip instantly so it
        doesn't look unresponsive."""
        self.enabled = enabled
        self.toggle_btn.setText("⏻ เปิดกล้อง" if not enabled else "⏻ ปิดกล้อง")
        if not enabled:
            self.video.setText("⏻ ปิดใช้งานกล้องนี้")


class ReadinessDialog(QDialog):
    def __init__(self, report: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ผลการตรวจสอบความพร้อมของบูธ (Booth Readiness Check)")
        self.resize(680, 480)
        self.setMinimumSize(560, 400)

        layout = QVBoxLayout(self)

        headline = "✅ บูธพร้อมใช้งาน (READY)" if report["overall_ok"] else "❌ บูธยังไม่พร้อม (NOT READY)"
        headline_label = QLabel(headline)
        headline_label.setFont(QFont("Sans", 14, QFont.Bold))
        headline_label.setStyleSheet(
            "color: %s;" % ("#2ecc71" if report["overall_ok"] else "#e74c3c")
        )
        layout.addWidget(headline_label)

        listw = QListWidget()
        for c in report["components"]:
            icon = "✅" if c["ok"] else ("❌" if c["critical"] else "⚠️")
            item = QListWidgetItem(f"{icon}  {c['component']} — {c['detail']}")
            listw.addItem(item)
        layout.addWidget(listw)

        close_btn = QPushButton("ปิด")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setStyleSheet("QDialog { background: #101218; color: white; }"
                            "QListWidget { background: #181a20; color: white; border: none; }")


class MainWindow(QWidget):
    def __init__(self, booth_id, booth_name, event_id, camera_devices, model,
                 model_device, catalog, assistant, db_path, recognizer=None,
                 dashboard_launcher=None, trainer_launcher=None, gender_age_backend=None,
                 adaptive_config=None):
        super().__init__()
        self.booth_id = booth_id
        self.booth_name = booth_name
        self.event_id = event_id
        self.model = model
        self.catalog = catalog
        self.assistant = assistant
        self.db_path = db_path
        self.recognizer = recognizer
        self.dashboard_launcher = dashboard_launcher
        self.trainer_launcher = trainer_launcher
        # Optional — see core/vision.py's CameraWorker: when set, person
        # boxes are colored red/blue by predicted gender.
        self.gender_age_backend = gender_age_backend

        self.aggregator = DetectionAggregator()
        self.camera_panels: dict[str, CameraPanel] = {}
        self.camera_status: dict[str, str] = {}
        self.workers: dict[str, CameraWorker] = {}
        self.current_product_key: str | None = None

        # Performance Monitor + Adaptive Controller — same mechanism as the
        # web app's BoothManager (see web/booth_manager.py and
        # core/performance.py): each CameraWorker reports capture/drop/
        # display/AI timing, and the controller backs off a camera's own AI
        # workload (inference rate, then AI input resolution) when this
        # process's own measurements say it's overloaded, independently per
        # camera, so N cameras stay smooth without needing identical
        # hardware to "just work."
        self.hardware_profile = detect_hardware()
        self.performance_monitor = PerformanceMonitor()
        self.adaptive_controller = AdaptiveController(
            self.performance_monitor, self.workers, adaptive_config,
            on_degraded=self._on_camera_degraded,
        )

        self.setWindowTitle(f"MONGDEE AI Booth OS — {booth_name}")
        self.resize(1600, 960)
        self.setMinimumSize(1180, 720)
        self._build_ui(camera_devices)
        self._create_workers(camera_devices, model_device)
        self.adaptive_controller.start()

        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self._send_heartbeat)
        self.heartbeat_timer.start(HEARTBEAT_INTERVAL_MS)

    # ------------------------------------------------------------------ UI
    def _build_ui(self, camera_devices):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        top_bar = QHBoxLayout()
        title = QLabel(f"MONGDEE AI Booth OS  •  Booth: {self.booth_name} ({self.booth_id})  •  Event: {self.event_id}")
        title.setFont(QFont("Sans", 16, QFont.Bold))
        title.setStyleSheet("color: white;")
        top_bar.addWidget(title)
        top_bar.addStretch(1)

        self.readiness_pill = QLabel("ยังไม่ตรวจสอบ")
        self.readiness_pill.setStyleSheet(
            "background: #444; color: white; padding: 4px 12px; border-radius: 10px;"
        )
        top_bar.addWidget(self.readiness_pill)

        readiness_btn = QPushButton("🔍 ตรวจสอบความพร้อมบูธ")
        readiness_btn.clicked.connect(self._on_run_readiness)
        top_bar.addWidget(readiness_btn)

        trainer_btn = QPushButton("🎓 เทรน AI จดจำสินค้า")
        trainer_btn.clicked.connect(self._on_open_trainer)
        top_bar.addWidget(trainer_btn)

        dashboard_btn = QPushButton("📊 เปิด Dashboard")
        dashboard_btn.clicked.connect(self._on_open_dashboard)
        top_bar.addWidget(dashboard_btn)

        root.addLayout(top_bar)

        legend = QLabel('<span style="color:#2ecc71;">■</span> สินค้า (Product)&nbsp;&nbsp;&nbsp;'
                         '<span style="color:#ff3c3c;">■</span> คน (Person)&nbsp;&nbsp;&nbsp;'
                         '<span style="color:#969696;">■</span> พบวัตถุแต่ยังไม่รู้จัก (ควรเทรน)')
        legend.setStyleSheet("padding: 2px 0 6px 2px;")
        root.addWidget(legend)

        body = QHBoxLayout()
        body.setSpacing(14)

        cam_grid_widget = QWidget()
        self.cam_grid = QGridLayout(cam_grid_widget)
        self.cam_grid.setContentsMargins(0, 0, 0, 0)
        self.cam_grid.setHorizontalSpacing(12)
        self.cam_grid.setVerticalSpacing(12)
        cols = max(1, math.ceil(math.sqrt(len(camera_devices)))) if camera_devices else 1
        for i, (camera_id, device) in enumerate(camera_devices):
            panel = CameraPanel(camera_id, device)
            self.camera_panels[camera_id] = panel
            self.camera_status[camera_id] = "unknown"
            self.cam_grid.addWidget(panel, i // cols, i % cols)
        for column in range(cols):
            self.cam_grid.setColumnStretch(column, 1)
        rows = max(1, (len(camera_devices) + cols - 1) // cols)
        for row in range(rows):
            self.cam_grid.setRowStretch(row, 1)
        body.addWidget(cam_grid_widget, 3)

        assistant_panel = self._build_assistant_panel()
        assistant_panel.setMinimumWidth(360)
        body.addWidget(assistant_panel, 2)

        root.addLayout(body, 1)

        alerts_label = QLabel("🔔 การแจ้งเตือนล่าสุด (Booth Health Monitoring)")
        alerts_label.setStyleSheet("color: white; font-weight: bold; padding-top: 6px;")
        root.addWidget(alerts_label)

        self.alerts_list = QListWidget()
        self.alerts_list.setMinimumHeight(120)
        root.addWidget(self.alerts_list, 0)

        self.setStyleSheet("""
            QWidget { background: #101218; color: white; font-size: 14px; }
            QPushButton { background: #20232c; border: 1px solid #383c48; border-radius: 8px;
                          padding: 10px 16px; color: white; min-height: 20px; }
            QPushButton:hover { background: #2b2f3a; }
            QLineEdit, QTextEdit { background: #181a20; border: 1px solid #30333d;
                                    border-radius: 6px; color: white; padding: 9px; }
            QListWidget { background: #181a20; border: 1px solid #30333d; border-radius: 6px; color: white; }
            QListWidget::item { padding: 6px; }
        """)

    def _build_assistant_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet("QFrame { background: #181a20; border: 1px solid #30333d; border-radius: 12px; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        header = QLabel("🤖 AI Product Assistant")
        header.setFont(QFont("Sans", 15, QFont.Bold))
        layout.addWidget(header)

        self.product_name_label = QLabel("ยังไม่พบสินค้า — วางสินค้าในกรอบกล้องเพื่อเริ่มต้น")
        self.product_name_label.setWordWrap(True)
        self.product_name_label.setFont(QFont("Sans", 18, QFont.Bold))
        layout.addWidget(self.product_name_label)

        self.product_tagline_label = QLabel("")
        self.product_tagline_label.setStyleSheet("color: #7fd3ff;")
        self.product_tagline_label.setWordWrap(True)
        layout.addWidget(self.product_tagline_label)

        self.product_desc_label = QLabel("")
        self.product_desc_label.setWordWrap(True)
        self.product_desc_label.setStyleSheet("color: #cccccc;")
        layout.addWidget(self.product_desc_label)

        self.product_source_label = QLabel("")
        self.product_source_label.setWordWrap(True)
        self.product_source_label.setStyleSheet("color: #a2a8b3; font-size: 12px;")
        layout.addWidget(self.product_source_label)

        layout.addSpacing(8)
        ask_label = QLabel("💬 พิมพ์คำถามเกี่ยวกับสินค้านี้")
        layout.addWidget(ask_label)

        ask_row = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("เช่น ราคาเท่าไหร่ / มีสีอะไรบ้าง")
        self.question_input.returnPressed.connect(self._on_ask)
        ask_row.addWidget(self.question_input)
        ask_btn = QPushButton("ถาม")
        ask_btn.clicked.connect(self._on_ask)
        ask_row.addWidget(ask_btn)
        layout.addLayout(ask_row)

        self.answer_label = QLabel("")
        self.answer_label.setWordWrap(True)
        self.answer_label.setStyleSheet("color: #ffe08a; padding-top: 4px;")
        layout.addWidget(self.answer_label)

        layout.addSpacing(8)
        history_label = QLabel("🕘 ประวัติการโต้ตอบ")
        layout.addWidget(history_label)
        self.history_list = QListWidget()
        layout.addWidget(self.history_list, 1)

        return panel

    # -------------------------------------------------------------- workers
    def _create_workers(self, camera_devices, model_device):
        for camera_id, device in camera_devices:
            worker = CameraWorker(
                camera_id=camera_id,
                device=device,
                model=self.model,
                allowed_classes=self.catalog.product_keys(),
                recognizer=self.recognizer,
                device_target=model_device,
                gender_age_backend=self.gender_age_backend,
                performance_monitor=self.performance_monitor,
            )
            worker.frame_ready.connect(self._on_frame_ready)
            worker.detections_ready.connect(self._on_detections_ready)
            worker.status_changed.connect(self._on_status_changed)
            self.workers[camera_id] = worker
            panel = self.camera_panels.get(camera_id)
            if panel:
                panel.toggle_btn.clicked.connect(
                    lambda checked=False, cid=camera_id: self._on_toggle_camera(cid)
                )
            worker.start()

    def _on_toggle_camera(self, camera_id: str):
        worker = self.workers.get(camera_id)
        panel = self.camera_panels.get(camera_id)
        if worker is None or panel is None:
            return
        new_enabled = not panel.enabled
        worker.set_enabled(new_enabled)
        panel.set_enabled_ui(new_enabled)

    # --------------------------------------------------------------- slots
    def _on_frame_ready(self, camera_id, frame):
        panel = self.camera_panels.get(camera_id)
        if panel:
            panel.set_frame(frame)

    def _on_detections_ready(self, camera_id, detections):
        event = self.aggregator.update(camera_id, detections)
        if event:
            self._handle_product_recognized(event)

    def _on_status_changed(self, camera_id, status, message):
        previous = self.camera_status.get(camera_id)
        self.camera_status[camera_id] = status
        panel = self.camera_panels.get(camera_id)
        if panel:
            panel.set_status(status, message)

        if status == previous:
            return

        severity = {"online": "ok", "degraded": "warning"}.get(
            status, "warning" if status in ("connecting", "reconnecting", "disabled") else "error")
        db.log_health_event(self.db_path, self.booth_id, self.event_id, camera_id,
                             "camera", severity, message)
        # See web/booth_manager.py's _on_status for why "connecting" isn't
        # alert-worthy (expected once per camera at startup) and
        # "reconnecting" is worded differently from a settled "offline".
        if status == "online":
            if previous not in (None, "unknown", "connecting"):
                self._push_alert(f"✅ {camera_id}: กลับมาออนไลน์แล้ว")
            else:
                self._push_alert(f"🔌 {camera_id}: เชื่อมต่อสำเร็จ")
        elif status == "connecting":
            pass
        elif status == "reconnecting":
            self._push_alert(f"🔁 {camera_id}: กำลังพยายามเชื่อมต่อใหม่")
        else:
            self._push_alert(f"⚠️ {camera_id}: {message}")

    def _on_camera_degraded(self, camera_id: str, degraded: bool):
        """AdaptiveController callback — see web/booth_manager.py's
        _on_camera_degraded for the full rationale; same idea here, just
        updating the Qt panel directly instead of a polled JSON dict."""
        panel = self.camera_panels.get(camera_id)
        current = self.camera_status.get(camera_id)
        if degraded and current == "online":
            self.camera_status[camera_id] = "degraded"
            if panel:
                panel.set_status("degraded", "AI ประมวลผลไม่ทัน กำลังลดภาระอัตโนมัติ")
            self._push_alert(f"📉 {camera_id}: ประสิทธิภาพลดลง ระบบกำลังปรับตัวอัตโนมัติ")
        elif not degraded and current == "degraded":
            self.camera_status[camera_id] = "online"
            if panel:
                panel.set_status("online", "ปกติ")
            self._push_alert(f"✅ {camera_id}: กลับมาทำงานปกติแล้ว")

    def _push_alert(self, text: str):
        item = QListWidgetItem(f"[{time.strftime('%H:%M:%S')}] {text}")
        self.alerts_list.insertItem(0, item)
        while self.alerts_list.count() > 50:
            self.alerts_list.takeItem(self.alerts_list.count() - 1)

    def _handle_product_recognized(self, event: dict):
        class_name = event["class_name"]
        self.current_product_key = class_name
        product = self.assistant.describe(class_name, speak=True)
        if not product:
            return

        self.product_name_label.setText(product["name"])
        self.product_tagline_label.setText(product["tagline"])
        self.product_desc_label.setText(product["description"])
        source = ", ".join(event["cameras"])
        reason = "กล้องหลายตัวยืนยันพร้อมกัน" if event["reason"] == "multi_camera_agreement" else "ตรวจพบต่อเนื่อง"
        self.product_source_label.setText(
            f"ตรวจพบโดยกล้อง: {source}  •  {reason}  •  ความมั่นใจ {event['confidence']:.0%}"
        )
        self.answer_label.setText("")

        db.log_interaction(
            self.db_path, self.booth_id, self.event_id, source, class_name,
            product["name"], event["confidence"],
        )
        self._add_history(f"🟢 พบสินค้า: {product['name']} ({source})")

    def _on_ask(self):
        question = self.question_input.text().strip()
        if not question:
            return
        if not self.current_product_key:
            self.answer_label.setText("กรุณานำสินค้าไปวางในกรอบกล้องก่อน แล้วจึงถามคำถามค่ะ")
            return

        answer = self.assistant.answer(self.current_product_key, question, speak=True)
        self.answer_label.setText(answer)
        product = self.catalog.get(self.current_product_key)
        db.log_interaction(
            self.db_path, self.booth_id, self.event_id, "assistant", self.current_product_key,
            product["name"] if product else self.current_product_key, None, question, answer,
        )
        self._add_history(f"❓ {question}  →  {answer}")
        self.question_input.clear()

    def _add_history(self, text: str):
        self.history_list.insertItem(0, QListWidgetItem(f"[{time.strftime('%H:%M:%S')}] {text}"))
        while self.history_list.count() > 100:
            self.history_list.takeItem(self.history_list.count() - 1)

    def _on_run_readiness(self):
        report = run_readiness_check(
            camera_statuses=self.camera_status,
            model_loaded=self.model is not None,
            tts_available=self.assistant.tts_available,
            db_path=self.db_path,
        )
        db.log_readiness_check(
            self.db_path, self.booth_id, self.event_id,
            "ready" if report["overall_ok"] else "not_ready",
            readiness_to_json(report),
        )
        if report["overall_ok"]:
            self.readiness_pill.setText("✅ READY")
            self.readiness_pill.setStyleSheet(
                "background: #1e6b3a; color: white; padding: 4px 12px; border-radius: 10px;"
            )
        else:
            self.readiness_pill.setText("❌ NOT READY")
            self.readiness_pill.setStyleSheet(
                "background: #7a1f1f; color: white; padding: 4px 12px; border-radius: 10px;"
            )
        ReadinessDialog(report, self).exec()

    def _on_open_dashboard(self):
        if self.dashboard_launcher:
            self.dashboard_launcher()

    def _on_open_trainer(self):
        if self.trainer_launcher:
            self.trainer_launcher()

    def _send_heartbeat(self):
        active = sum(1 for s in self.camera_status.values() if s == "online")
        overall = "ok" if active > 0 else "degraded"
        db.log_heartbeat(self.db_path, self.booth_id, self.event_id, overall, active)

    # -------------------------------------------------------------- close
    def closeEvent(self, event):
        self.adaptive_controller.stop()
        for worker in self.workers.values():
            worker.stop()
        db.log_heartbeat(self.db_path, self.booth_id, self.event_id, "stopped", 0)
        event.accept()
