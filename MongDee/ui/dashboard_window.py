"""AI Dashboard & Analytics — central view across one or many booths.

A single machine only has one local SQLite DB, so multi-booth aggregation is
demonstrated via file import: each booth can run `scripts/export_booth_data.py`
to dump its DB to JSON, and this dashboard can load any number of those files
alongside its own local data (see the "นำเข้าข้อมูลบูธอื่น" button). That mirrors
the spec's "Dashboard สำหรับรวบรวมข้อมูลการใช้งานของแต่ละบูธ" without requiring a
live network server.
"""

from __future__ import annotations

import collections
import json
import time

from PySide6.QtCore import QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import database as db

REFRESH_INTERVAL_MS = 5000


def _fmt_ts(ts):
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""


class StatCard(QFrame):
    def __init__(self, title):
        super().__init__()
        self.setStyleSheet("QFrame { background: #181a20; border: 1px solid #30333d; border-radius: 10px; }")
        self.setMinimumHeight(112)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        self.value_label = QLabel("0")
        self.value_label.setFont(QFont("Sans", 28, QFont.Bold))
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #999;")
        layout.addWidget(self.value_label)
        layout.addWidget(title_label)

    def set_value(self, value):
        self.value_label.setText(str(value))


class DashboardWindow(QWidget):
    def __init__(self, db_path, default_event_id=None):
        super().__init__()
        self.db_path = db_path
        self.external_sources: list[dict] = []
        self.setWindowTitle("MONGDEE AI Dashboard & Analytics")
        self.resize(1440, 900)
        self.setMinimumSize(1050, 700)
        self._build_ui()
        self._default_event_id = default_event_id
        self.refresh()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(REFRESH_INTERVAL_MS)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        header = QLabel("📊 MONGDEE AI Dashboard & Analytics")
        header.setFont(QFont("Sans", 19, QFont.Bold))
        root.addWidget(header)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Event ID:"))
        self.event_filter = QComboBox()
        self.event_filter.currentIndexChanged.connect(self.refresh)
        filter_row.addWidget(self.event_filter)

        filter_row.addWidget(QLabel("Booth ID:"))
        self.booth_filter = QComboBox()
        self.booth_filter.currentIndexChanged.connect(self.refresh)
        filter_row.addWidget(self.booth_filter)

        filter_row.addStretch(1)

        import_btn = QPushButton("📂 นำเข้าข้อมูลบูธอื่น")
        import_btn.clicked.connect(self._on_import)
        filter_row.addWidget(import_btn)

        refresh_btn = QPushButton("🔄 รีเฟรช")
        refresh_btn.clicked.connect(self.refresh)
        filter_row.addWidget(refresh_btn)

        root.addLayout(filter_row)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self.card_total = StatCard("Total Interactions")
        self.card_products = StatCard("Unique Products Shown")
        self.card_booths = StatCard("Active Booths")
        self.card_alerts = StatCard("Open Alerts")
        for card in (self.card_total, self.card_products, self.card_booths, self.card_alerts):
            cards_row.addWidget(card)
        root.addLayout(cards_row)

        root.addWidget(self._section_label("🏆 สินค้ายอดนิยม (Top Products)"))
        self.top_products_table = self._make_table(["สินค้า", "จำนวนครั้งที่แสดง"])
        self.top_products_table.setMinimumHeight(170)
        root.addWidget(self.top_products_table, 2)

        root.addWidget(self._section_label("🕘 ปฏิสัมพันธ์ล่าสุด (Recent Interactions)"))
        self.interactions_table = self._make_table(
            ["เวลา", "บูธ", "กล้อง", "สินค้า", "ความมั่นใจ", "คำถาม", "คำตอบ"]
        )
        root.addWidget(self.interactions_table, 4)

        root.addWidget(self._section_label("🩺 สถานะอุปกรณ์ / การแจ้งเตือน (Health & Alerts)"))
        self.health_table = self._make_table(["เวลา", "บูธ", "อุปกรณ์", "สถานะ", "ข้อความ"])
        self.health_table.setMinimumHeight(170)
        root.addWidget(self.health_table, 2)

        self.setStyleSheet("""
            QWidget { background: #101218; color: white; font-size: 14px; }
            QPushButton { background: #20232c; border: 1px solid #383c48; border-radius: 8px;
                          padding: 10px 16px; color: white; min-height: 20px; }
            QPushButton:hover { background: #2b2f3a; }
            QComboBox { background: #181a20; border: 1px solid #30333d; border-radius: 6px;
                        padding: 8px 10px; color: white; min-height: 20px; }
            QTableWidget { background: #181a20; border: 1px solid #30333d; color: white;
                           gridline-color: #30333d; alternate-background-color: #15171d; }
            QTableWidget::item { padding: 7px; }
            QHeaderView::section { background: #20232c; color: white; border: none; padding: 8px; }
        """)

    def _section_label(self, text):
        label = QLabel(text)
        label.setFont(QFont("Sans", 13, QFont.Bold))
        label.setStyleSheet("padding-top: 8px;")
        return label

    def _make_table(self, headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setDefaultSectionSize(34)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        return table

    # ------------------------------------------------------------- import
    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "นำเข้าข้อมูลบูธอื่น (JSON)", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.external_sources.append(data)
            QMessageBox.information(self, "นำเข้าสำเร็จ", f"นำเข้าข้อมูลจาก {path} สำเร็จ")
            self.refresh()
        except Exception as exc:
            QMessageBox.critical(self, "นำเข้าล้มเหลว", str(exc))

    # ------------------------------------------------------------- refresh
    def _merged_interactions(self):
        rows = db.query_interactions(self.db_path, limit=100000)
        for source in self.external_sources:
            rows.extend(source.get("interactions", []))
        return rows

    def _merged_health(self):
        rows = db.query_health_events(self.db_path, limit=100000)
        for source in self.external_sources:
            rows.extend(source.get("health_events", []))
        return rows

    def refresh(self):
        interactions = self._merged_interactions()
        health = self._merged_health()

        event_ids = sorted({r["event_id"] for r in interactions + health if r.get("event_id")})
        booth_ids = sorted({r["booth_id"] for r in interactions + health if r.get("booth_id")})
        self._sync_combo(self.event_filter, event_ids)
        self._sync_combo(self.booth_filter, booth_ids)

        event_id = self.event_filter.currentText() or None
        booth_id = self.booth_filter.currentText() or None
        if event_id == "ทั้งหมด":
            event_id = None
        if booth_id == "ทั้งหมด":
            booth_id = None

        filtered_interactions = [
            r for r in interactions
            if (not event_id or r.get("event_id") == event_id)
            and (not booth_id or r.get("booth_id") == booth_id)
        ]
        filtered_health = [
            r for r in health
            if (not event_id or r.get("event_id") == event_id)
            and (not booth_id or r.get("booth_id") == booth_id)
        ]

        self._update_cards(filtered_interactions, filtered_health)
        self._update_top_products(filtered_interactions)
        self._update_interactions_table(filtered_interactions)
        self._update_health_table(filtered_health)

    def _sync_combo(self, combo: QComboBox, values: list[str]):
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("ทั้งหมด")
        combo.addItems(values)
        idx = combo.findText(current)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)

    def _update_cards(self, interactions, health):
        booths = {r["booth_id"] for r in interactions} | {r["booth_id"] for r in health}
        open_alerts = [h for h in health if h.get("status") == "error"]
        self.card_total.set_value(len(interactions))
        self.card_products.set_value(len({r["product_name"] for r in interactions}))
        self.card_booths.set_value(len(booths))
        self.card_alerts.set_value(len(open_alerts))

    def _update_top_products(self, interactions):
        counts = collections.Counter(r["product_name"] for r in interactions)
        top = counts.most_common(10)
        self.top_products_table.setRowCount(len(top))
        for row, (name, count) in enumerate(top):
            self.top_products_table.setItem(row, 0, QTableWidgetItem(name))
            self.top_products_table.setItem(row, 1, QTableWidgetItem(str(count)))

    def _update_interactions_table(self, interactions):
        rows = sorted(interactions, key=lambda r: r.get("ts", 0), reverse=True)[:200]
        self.interactions_table.setRowCount(len(rows))
        for row, r in enumerate(rows):
            conf = r.get("confidence")
            values = [
                _fmt_ts(r.get("ts")),
                r.get("booth_id", ""),
                r.get("camera_id", "") or "",
                r.get("product_name", ""),
                f"{conf:.0%}" if isinstance(conf, (int, float)) else "",
                r.get("question", "") or "",
                r.get("answer", "") or "",
            ]
            for col, value in enumerate(values):
                self.interactions_table.setItem(row, col, QTableWidgetItem(value))

    def _update_health_table(self, health):
        rows = sorted(health, key=lambda r: r.get("ts", 0), reverse=True)[:100]
        self.health_table.setRowCount(len(rows))
        for row, r in enumerate(rows):
            component = r.get("component") or ""
            camera_id = r.get("camera_id") or ""
            device_label = f"{camera_id} ({component})" if component else camera_id
            values = [
                _fmt_ts(r.get("ts")),
                r.get("booth_id", ""),
                device_label,
                r.get("status", ""),
                r.get("message", "") or "",
            ]
            for col, value in enumerate(values):
                self.health_table.setItem(row, col, QTableWidgetItem(value))
