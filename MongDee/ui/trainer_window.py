"""AI Trainer — upload product photos/video so the AI learns to recognize
real products, with no bounding-box labeling and no training step.

Every uploaded image (or video frame) is turned into an embedding and added
to that product's gallery immediately (core/recognizer.py) — accuracy keeps
improving as more samples come in, which is exactly the "keep uploading
until it's accurate enough" workflow the brief asks for.
"""

from __future__ import annotations

import uuid

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.recognizer import RECOMMENDED_SAMPLES
from core.training import import_images, import_video, slugify
from core.vision import discover_cameras
from ui.qt_camera_bridge import QtCameraWorker as CameraWorker
from ui.main_window import frame_to_pixmap


class ImportWorker(QThread):
    progress = Signal(int, int)   # done, total
    finished_ok = Signal(int)     # samples added
    failed = Signal(str)

    def __init__(self, kind, paths, product_key, recognizer, model, model_device, parent=None):
        super().__init__(parent)
        self.kind = kind  # "images" or "video"
        self.paths = paths
        self.product_key = product_key
        self.recognizer = recognizer
        self.model = model
        self.model_device = model_device

    def run(self):
        try:
            if self.kind == "images":
                added = import_images(self.paths, self.product_key, self.recognizer,
                                       self.model, self.model_device, progress_cb=self.progress.emit)
            else:
                added = import_video(self.paths[0], self.product_key, self.recognizer,
                                      self.model, self.model_device, progress_cb=self.progress.emit)
            self.finished_ok.emit(added)
        except Exception as exc:
            self.failed.emit(str(exc))


class LiveTestDialog(QDialog):
    """Opens one camera and runs the exact same detection pipeline used on the
    booth floor, so uploads can be validated immediately without leaving the trainer."""

    def __init__(self, model, model_device, catalog, recognizer, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ทดสอบการจดจำสินค้าด้วยกล้องสด")
        self.resize(960, 760)
        self.setMinimumSize(760, 600)

        layout = QVBoxLayout(self)
        self.video_label = QLabel("กำลังเปิดกล้อง...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(700, 520)
        self.video_label.setStyleSheet("background: #111318; color: #888; border-radius: 8px;")
        layout.addWidget(self.video_label, 1)

        close_btn = QPushButton("ปิด")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self.setStyleSheet("QDialog { background: #101218; color: white; }")

        devices = discover_cameras(max_found=1)
        self.worker = None
        if devices:
            self.worker = CameraWorker(
                camera_id="TEST",
                device=devices[0],
                model=model,
                allowed_classes=catalog.product_keys(),
                recognizer=recognizer,
                device_target=model_device,
            )
            self.worker.frame_ready.connect(self._on_frame)
            self.worker.start()
        else:
            self.video_label.setText("❌ ไม่พบกล้องที่ใช้งานได้")

    def _on_frame(self, _camera_id, frame):
        pixmap = frame_to_pixmap(frame)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()


class AddProductDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("เพิ่มสินค้าใหม่")
        self.setMinimumWidth(420)
        layout = QFormLayout(self)

        self.name_input = QLineEdit()
        self.tagline_input = QLineEdit()
        self.description_input = QTextEdit()
        self.description_input.setMaximumHeight(80)

        layout.addRow("ชื่อสินค้า *", self.name_input)
        layout.addRow("แท็กไลน์", self.tagline_input)
        layout.addRow("รายละเอียด", self.description_input)

        buttons = QHBoxLayout()
        ok = QPushButton("เพิ่มสินค้า")
        cancel = QPushButton("ยกเลิก")
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        buttons.addWidget(ok)
        buttons.addWidget(cancel)
        layout.addRow(buttons)

        self.setStyleSheet("QDialog { background: #101218; color: white; }"
                            "QLineEdit, QTextEdit { background: #181a20; border: 1px solid #30333d; "
                            "border-radius: 6px; color: white; padding: 4px; }")

    def values(self):
        return {
            "name": self.name_input.text().strip(),
            "tagline": self.tagline_input.text().strip(),
            "description": self.description_input.toPlainText().strip(),
        }


class TrainerWindow(QWidget):
    def __init__(self, catalog, recognizer, model, model_device):
        super().__init__()
        self.catalog = catalog
        self.recognizer = recognizer
        self.model = model
        self.model_device = model_device
        self.selected_key: str | None = None
        self.import_worker: ImportWorker | None = None

        self.setWindowTitle("MONGDEE AI Trainer — เทรน AI ให้รู้จักสินค้าจริง")
        self.resize(1280, 820)
        self.setMinimumSize(960, 650)
        self._build_ui()
        self._refresh_product_list()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        header = QLabel("🎓 เทรน AI ให้รู้จักสินค้าจริง")
        header.setFont(QFont("Sans", 19, QFont.Bold))
        root.addWidget(header)

        hint = QLabel(
            "อัปโหลดรูปภาพหรือวิดีโอของสินค้าแต่ละชิ้น ระบบจะเรียนรู้ทันทีโดยไม่ต้องเทรนโมเดลใหม่ — "
            f"ยิ่งอัปโหลดมาก ความแม่นยำยิ่งสูงขึ้น แนะนำอย่างน้อย {RECOMMENDED_SAMPLES} ภาพต่อสินค้า "
            "จากหลายมุมและสภาพแสงที่ต่างกัน"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #999; padding-bottom: 6px;")
        root.addWidget(hint)

        body = QHBoxLayout()
        body.setSpacing(14)

        left = QVBoxLayout()
        left.addWidget(QLabel("รายการสินค้า"))
        self.product_list = QListWidget()
        self.product_list.currentItemChanged.connect(self._on_select_product)
        left.addWidget(self.product_list, 1)
        add_btn = QPushButton("➕ เพิ่มสินค้าใหม่")
        add_btn.clicked.connect(self._on_add_product)
        left.addWidget(add_btn)
        left_widget = QWidget()
        left_widget.setLayout(left)
        left_widget.setMinimumWidth(280)
        left_widget.setMaximumWidth(380)
        body.addWidget(left_widget, 1)

        body.addWidget(self._build_detail_panel(), 3)
        root.addLayout(body, 1)

        self.setStyleSheet("""
            QWidget { background: #101218; color: white; font-size: 14px; }
            QPushButton { background: #20232c; border: 1px solid #383c48; border-radius: 8px;
                          padding: 10px 16px; color: white; min-height: 20px; }
            QPushButton:hover { background: #2b2f3a; }
            QPushButton:disabled { color: #666; }
            QListWidget { background: #181a20; border: 1px solid #30333d; border-radius: 6px; color: white; }
            QListWidget::item { padding: 8px; }
            QProgressBar { background: #181a20; border: 1px solid #30333d; border-radius: 6px;
                           text-align: center; color: white; }
            QProgressBar::chunk { background: #2ecc71; border-radius: 6px; }
        """)

    def _build_detail_panel(self) -> QWidget:
        panel = QFrame()
        panel.setStyleSheet("QFrame { background: #181a20; border: 1px solid #30333d; border-radius: 12px; }")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.detail_title = QLabel("เลือกสินค้าทางซ้าย เพื่อเริ่มอัปโหลดข้อมูลเทรน")
        self.detail_title.setFont(QFont("Sans", 17, QFont.Bold))
        self.detail_title.setWordWrap(True)
        layout.addWidget(self.detail_title)

        self.sample_status = QLabel("")
        self.sample_status.setStyleSheet("color: #ccc;")
        layout.addWidget(self.sample_status)

        self.sample_progress = QProgressBar()
        self.sample_progress.setRange(0, 100)
        layout.addWidget(self.sample_progress)

        btn_row = QHBoxLayout()
        self.upload_images_btn = QPushButton("📁 อัปโหลดรูปภาพ")
        self.upload_images_btn.clicked.connect(self._on_upload_images)
        btn_row.addWidget(self.upload_images_btn)

        self.upload_video_btn = QPushButton("🎞️ อัปโหลดวิดีโอ")
        self.upload_video_btn.clicked.connect(self._on_upload_video)
        btn_row.addWidget(self.upload_video_btn)
        layout.addLayout(btn_row)

        self.import_progress = QProgressBar()
        self.import_progress.setVisible(False)
        layout.addWidget(self.import_progress)

        self.import_log = QLabel("")
        self.import_log.setStyleSheet("color: #7fd3ff;")
        self.import_log.setWordWrap(True)
        layout.addWidget(self.import_log)

        layout.addSpacing(10)
        test_btn = QPushButton("▶️ ทดสอบด้วยกล้องสด")
        test_btn.clicked.connect(self._on_test_live)
        layout.addWidget(test_btn)

        clear_btn = QPushButton("🗑️ ลบข้อมูลเทรนทั้งหมดของสินค้านี้")
        clear_btn.clicked.connect(self._on_clear_samples)
        layout.addWidget(clear_btn)

        layout.addStretch(1)
        self._set_detail_enabled(False)
        return panel

    def _set_detail_enabled(self, enabled: bool):
        for widget in (self.upload_images_btn, self.upload_video_btn):
            widget.setEnabled(enabled)

    # ------------------------------------------------------------- product list
    def _refresh_product_list(self):
        current_key = self.selected_key
        self.product_list.blockSignals(True)
        self.product_list.clear()
        counts = self.recognizer.sample_counts()
        for key in self.catalog.product_keys():
            product = self.catalog.get(key)
            n = counts.get(key, 0)
            icon = "🟢" if n >= RECOMMENDED_SAMPLES else ("🟡" if n > 0 else "⚪")
            item = QListWidgetItem(f"{icon} {product['name']}  ({n} ภาพ)")
            item.setData(Qt.UserRole, key)
            self.product_list.addItem(item)
            if key == current_key:
                self.product_list.setCurrentItem(item)
        self.product_list.blockSignals(False)
        if self.product_list.currentItem() is None and self.product_list.count() > 0:
            self.product_list.setCurrentRow(0)

    def _on_select_product(self, item, _previous=None):
        if item is None:
            self.selected_key = None
            self._set_detail_enabled(False)
            return
        self.selected_key = item.data(Qt.UserRole)
        self._refresh_detail()
        self._set_detail_enabled(True)

    def _refresh_detail(self):
        if not self.selected_key:
            return
        product = self.catalog.get(self.selected_key)
        n = self.recognizer.sample_count(self.selected_key)
        self.detail_title.setText(f"{product['name']}  ·  key: {self.selected_key}")

        pct = min(100, int(100 * n / RECOMMENDED_SAMPLES))
        self.sample_progress.setValue(pct)
        if n == 0:
            status = "⚪ ยังไม่มีข้อมูลเทรน — อัปโหลดรูปภาพหรือวิดีโอเพื่อเริ่มต้น"
        elif n < RECOMMENDED_SAMPLES:
            status = f"🟡 มีข้อมูล {n} ภาพ — แนะนำให้อัปโหลดเพิ่มอีกอย่างน้อย {RECOMMENDED_SAMPLES - n} ภาพ เพื่อความแม่นยำ"
        else:
            status = f"🟢 มีข้อมูล {n} ภาพ — พร้อมใช้งานแล้ว (อัปโหลดเพิ่มได้เสมอเพื่อความแม่นยำที่สูงขึ้น)"
        self.sample_status.setText(status)

    # ------------------------------------------------------------------ add
    def _on_add_product(self):
        dialog = AddProductDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        values = dialog.values()
        if not values["name"]:
            QMessageBox.warning(self, "ข้อมูลไม่ครบ", "กรุณากรอกชื่อสินค้า")
            return
        key = slugify(values["name"])
        while self.catalog.get(key):
            key = f"{key}-{uuid.uuid4().hex[:4]}"
        self.catalog.add_product(key, values["name"], values["tagline"], values["description"])
        self.selected_key = key
        self._refresh_product_list()

    # --------------------------------------------------------------- upload
    def _on_upload_images(self):
        if not self.selected_key:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "เลือกรูปภาพสินค้า (เลือกได้หลายไฟล์)", "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if not paths:
            return
        self._start_import("images", paths)

    def _on_upload_video(self):
        if not self.selected_key:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "เลือกวิดีโอสินค้า", "", "Video (*.mp4 *.mov *.avi *.mkv *.webm)",
        )
        if not path:
            return
        self._start_import("video", [path])

    def _start_import(self, kind, paths):
        self._set_detail_enabled(False)
        self.import_progress.setVisible(True)
        self.import_progress.setValue(0)
        self.import_log.setText("กำลังประมวลผล...")

        self.import_worker = ImportWorker(
            kind, paths, self.selected_key, self.recognizer, self.model, self.model_device,
        )
        self.import_worker.progress.connect(self._on_import_progress)
        self.import_worker.finished_ok.connect(self._on_import_finished)
        self.import_worker.failed.connect(self._on_import_failed)
        self.import_worker.start()

    def _on_import_progress(self, done, total):
        if total > 0:
            self.import_progress.setValue(int(100 * done / total))
        self.import_log.setText(f"ประมวลผลแล้ว {done}/{total}")

    def _on_import_finished(self, added):
        self.import_progress.setVisible(False)
        self.import_log.setText(f"✅ เพิ่มข้อมูลเทรนสำเร็จ {added} ภาพ")
        self._set_detail_enabled(True)
        self._refresh_product_list()
        self._refresh_detail()

    def _on_import_failed(self, message):
        self.import_progress.setVisible(False)
        self.import_log.setText(f"❌ ล้มเหลว: {message}")
        self._set_detail_enabled(True)

    # ------------------------------------------------------------------ misc
    def _on_clear_samples(self):
        if not self.selected_key:
            return
        confirm = QMessageBox.question(
            self, "ยืนยันการลบ",
            f"ลบข้อมูลเทรนทั้งหมดของ '{self.catalog.get(self.selected_key)['name']}' หรือไม่?",
        )
        if confirm == QMessageBox.Yes:
            self.recognizer.clear_product(self.selected_key)
            self._refresh_product_list()
            self._refresh_detail()

    def _on_test_live(self):
        LiveTestDialog(self.model, self.model_device, self.catalog, self.recognizer, self).exec()
