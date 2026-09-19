"""MONGDEE AI Booth OS — Launcher.

A single window with buttons instead of terminal commands: open the booth,
the dashboard, or the AI trainer with one click. This is also the thing a
desktop shortcut / the built .exe (see BUILD.md) points at, and the primary
way this app is meant to be used day-to-day.

On a machine where the project's Python environment hasn't been set up yet,
this window itself still opens (its own dependency is just PySide6 — see
BUILD.md) and offers a "First-time Setup" button that runs install.sh /
install.bat for you, streaming the output, before enabling the launch
buttons.

Every launch button (Booth/Dashboard/AI Trainer/browser) spawns its target
script as an independent background process — independent on purpose, so
closing this launcher window never closes whatever it opened. To still give
real feedback instead of silently doing nothing when that target script
fails immediately (missing camera, crashed import, ...), a short grace
period after each launch watches for an early exit and surfaces whatever
the process printed — see _launch()/_LaunchWatcher below.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QProcess, QSize, Signal
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# How long after spawning a booth/dashboard/trainer/web process to keep
# watching for an early crash before assuming it opened fine and moving on.
# Camera discovery alone can take several seconds on some machines, so this
# needs real headroom — false "it's fine" is the safe direction to err in
# (worst case: a slow-but-real crash goes unreported), never the reverse.
LAUNCH_GRACE_SEC = 30


def project_root() -> Path:
    """The folder that holds app.py/dashboard.py/trainer.py — computed relative
    to this script's own location so it works whether run as `python
    launcher.py` from source or as a frozen .exe placed at the project root
    (see build_windows.bat / build_linux.sh)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = project_root()
IS_WINDOWS = sys.platform.startswith("win")
VENV_PYTHON = ROOT / (".venv/Scripts/python.exe" if IS_WINDOWS else ".venv/bin/python")
INSTALL_SCRIPT = ROOT / ("install.bat" if IS_WINDOWS else "install.sh")
ICONS_DIR = ROOT / "assets" / "icons"


def _icon(name: str) -> QIcon:
    """Loads assets/icons/{name}.svg — every icon in this window comes from
    that folder instead of emoji, which render inconsistently across
    Windows font/emoji-font configurations and don't reliably ship on a
    bare Windows install the way a bundled SVG does."""
    path = ICONS_DIR / f"{name}.svg"
    return QIcon(str(path)) if path.exists() else QIcon()


class _LaunchWatcher(QObject):
    """Bridges the background thread that watches a spawned process back
    onto the Qt/GUI thread — Qt signals are safe to emit cross-thread (Qt
    auto-queues the connected slot call onto the receiver's own thread),
    which is the only part of this that isn't safe to just call directly
    from the watcher thread."""
    failed = Signal(str, int, str)  # label, exit_code, output_tail


class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MONGDEE AI Booth OS — Launcher")
        self.resize(560, 660)
        icon_path = ROOT / "assets" / "icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.install_process: QProcess | None = None
        self._launch_watcher = _LaunchWatcher()
        self._launch_watcher.failed.connect(self._on_launch_failed)
        self._build_ui()
        self._refresh_ready_state()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)

        title = QLabel("MONGDEE AI Booth OS")
        title.setFont(QFont("Sans", 18, QFont.Bold))
        root.addWidget(title)

        subtitle = QLabel("ระบบปฏิบัติการสำหรับบูธอัจฉริยะ — เลือกสิ่งที่ต้องการเปิด")
        subtitle.setStyleSheet("color: #999;")
        root.addWidget(subtitle)

        status_row = QHBoxLayout()
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(18, 18)
        self.status_icon.setScaledContents(True)
        status_row.addWidget(self.status_icon)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        root.addLayout(status_row)

        self.setup_btn = QPushButton(" ติดตั้งระบบครั้งแรก (First-time Setup)")
        self.setup_btn.setIcon(_icon("wrench"))
        self.setup_btn.setIconSize(QSize(18, 18))
        self.setup_btn.clicked.connect(self._on_run_setup)
        root.addWidget(self.setup_btn)

        self.gpu_checkbox = QCheckBox(
            " ใช้ GPU เร่งความเร็ว (ตรวจจับ NVIDIA/AMD/Intel อัตโนมัติ, ถ้ามี)"
        )
        self.gpu_checkbox.setIcon(_icon("chip"))
        self.gpu_checkbox.setIconSize(QSize(16, 16))
        self.gpu_checkbox.setToolTip(
            "ติ๊กไว้ (ค่าเริ่มต้น) แล้วกด \"ติดตั้งระบบครั้งแรก\": ระบบจะตรวจสอบการ์ดจอในเครื่องเองว่าเป็น "
            "NVIDIA, AMD, หรือ Intel แล้วติดตั้งตัวเร่งความเร็วที่ตรงกับการ์ดจอนั้นให้อัตโนมัติ "
            "(NVIDIA ใช้ CUDA, ยี่ห้ออื่นที่รองรับ DirectX12 ใช้ DirectML) — เอาติ๊กออกถ้าต้องการ "
            "รุ่น CPU เท่านั้น (ใช้ได้ทุกเครื่องแต่ประมวลผลช้ากว่า)"
        )
        # Checked by default — GPU acceleration should be what a fresh
        # install gets out of the box when the hardware supports it.
        # Auto-detection (see install.bat) still falls back to CPU-only
        # cleanly if this machine turns out to have no supported GPU, so
        # leaving it ticked is always safe. (/settings on the web side has
        # no equivalent control anymore — it only shows read-only GPU
        # status now; this checkbox here is first-run installer
        # configuration, a separate concern from booth runtime settings.)
        self.gpu_checkbox.setChecked(True)
        root.addWidget(self.gpu_checkbox)

        booth_box = QFrame()
        booth_box.setStyleSheet("QFrame { background: #181a20; border: 1px solid #30333d; border-radius: 10px; }")
        booth_layout = QVBoxLayout(booth_box)
        booth_layout.addWidget(QLabel("ตั้งค่าบูธ (แก้ไขได้ หรือปล่อยค่าเริ่มต้น)"))

        form_row1 = QHBoxLayout()
        self.booth_name_input = QLineEdit("MONGDEE Demo Booth")
        self.booth_name_input.setPlaceholderText("ชื่อบูธ")
        form_row1.addWidget(QLabel("ชื่อบูธ:"))
        form_row1.addWidget(self.booth_name_input)
        booth_layout.addLayout(form_row1)

        form_row2 = QHBoxLayout()
        self.event_id_input = QLineEdit("1-Day-at-IMPACT")
        self.event_id_input.setPlaceholderText("Event ID")
        form_row2.addWidget(QLabel("Event ID:"))
        form_row2.addWidget(self.event_id_input)
        booth_layout.addLayout(form_row2)

        form_row3 = QHBoxLayout()
        self.cameras_input = QLineEdit()
        self.cameras_input.setPlaceholderText("ปล่อยว่าง = ค้นหากล้องอัตโนมัติ (หรือระบุ เช่น /dev/video0,/dev/video2)")
        form_row3.addWidget(QLabel("กล้อง:"))
        form_row3.addWidget(self.cameras_input)
        booth_layout.addLayout(form_row3)

        root.addWidget(booth_box)

        self.booth_btn = QPushButton(" เปิดบูธ (Start Booth)")
        self.booth_btn.setIcon(_icon("store"))
        self.booth_btn.setIconSize(QSize(18, 18))
        self.booth_btn.clicked.connect(self._on_open_booth)
        root.addWidget(self.booth_btn)

        self.dashboard_btn = QPushButton(" เปิด Dashboard")
        self.dashboard_btn.setIcon(_icon("bar-chart"))
        self.dashboard_btn.setIconSize(QSize(18, 18))
        self.dashboard_btn.clicked.connect(self._on_open_dashboard)
        root.addWidget(self.dashboard_btn)

        self.trainer_btn = QPushButton(" เปิด AI Trainer")
        self.trainer_btn.setIcon(_icon("graduation-cap"))
        self.trainer_btn.setIconSize(QSize(18, 18))
        self.trainer_btn.clicked.connect(self._on_open_trainer)
        root.addWidget(self.trainer_btn)

        self.web_btn = QPushButton(" เปิดผ่านเบราว์เซอร์ (ทุกฟีเจอร์ในหน้าเว็บเดียว)")
        self.web_btn.setIcon(_icon("globe"))
        self.web_btn.setIconSize(QSize(18, 18))
        self.web_btn.clicked.connect(self._on_open_web)
        root.addWidget(self.web_btn)

        root.addWidget(QLabel("บันทึกการติดตั้ง / การเปิดใช้งาน:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        root.addWidget(self.log)

        self.setStyleSheet("""
            QWidget { background: #101218; color: white; }
            QPushButton { background: #20232c; border: 1px solid #383c48; border-radius: 10px;
                          padding: 12px; color: white; font-size: 14px; text-align: left; }
            QPushButton:hover { background: #2b2f3a; }
            QPushButton:disabled { color: #555; }
            QCheckBox { padding: 6px 4px; }
            QLineEdit, QTextEdit { background: #12141a; border: 1px solid #30333d; border-radius: 6px;
                                    color: white; padding: 4px 6px; }
        """)

    # -------------------------------------------------------------- state
    def _is_ready(self) -> bool:
        return VENV_PYTHON.exists()

    def _refresh_ready_state(self):
        ready = self._is_ready()
        for btn in (self.booth_btn, self.dashboard_btn, self.trainer_btn, self.web_btn):
            btn.setEnabled(ready)
        if ready:
            self.status_icon.setPixmap(_icon("check").pixmap(18, 18))
            self.status_label.setText("ระบบพร้อมใช้งาน")
            self.status_label.setStyleSheet("color: #2ecc71; padding: 6px 0;")
        else:
            self.status_icon.setPixmap(_icon("alert-triangle").pixmap(18, 18))
            self.status_label.setText(
                "ยังไม่ได้ติดตั้งระบบ — กดปุ่ม \"ติดตั้งระบบครั้งแรก\" ด้านล่างก่อน "
                "(ใช้เวลาสักครู่ ต้องต่ออินเทอร์เน็ต)"
            )
            self.status_label.setStyleSheet("color: #f1c40f; padding: 6px 0;")

    # ----------------------------------------------------------- launching
    def _launch(self, script_name: str, extra_args: list[str] | None = None, label: str | None = None):
        """Spawns script_name as an independent background process (keeps
        running even after this launcher window closes) and watches it for
        a short grace period — see LAUNCH_GRACE_SEC. If it exits with a
        non-zero code inside that window, whatever it printed is surfaced
        here instead of the click just silently doing nothing; if it's
        still running once the grace period elapses, this stops watching
        and assumes it opened correctly (a real GUI window doesn't need a
        babysitter for the rest of its life)."""
        if not self._is_ready():
            QMessageBox.warning(self, "ยังไม่พร้อม", "กรุณาติดตั้งระบบก่อน (ปุ่ม \"ติดตั้งระบบครั้งแรก\")")
            return
        label = label or script_name
        script_path = ROOT / script_name
        args = [str(VENV_PYTHON), str(script_path)] + (extra_args or [])
        self.log.append(f"กำลังเปิด {label} ...")
        try:
            # stdout/stderr are explicitly captured (not inherited) rather
            # than left to default — when this launcher itself is the
            # compiled --windowed .exe (see build_windows.bat), it has no
            # console of its own at all, and a child process inheriting an
            # invalid/closed stdout handle from a console-less parent can
            # crash or hang the instant it tries to print() anything.
            # Explicit PIPE redirection sidesteps that entirely, on top of
            # giving us the grace-period diagnostics below.
            proc = subprocess.Popen(
                args, cwd=str(ROOT),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except Exception as exc:
            QMessageBox.critical(self, "เปิดไม่สำเร็จ", str(exc))
            self.log.append(f"[ล้มเหลว] เปิด {label} ไม่สำเร็จ: {exc}")
            return

        def watch():
            try:
                _, stderr_bytes = proc.communicate(timeout=LAUNCH_GRACE_SEC)
            except subprocess.TimeoutExpired:
                return  # still running after the grace period — assume it opened fine
            if proc.returncode not in (0, None):
                text = (stderr_bytes or b"").decode("utf-8", "ignore").strip()
                self._launch_watcher.failed.emit(label, proc.returncode, text[-4000:])

        threading.Thread(target=watch, daemon=True).start()

    def _on_launch_failed(self, label: str, exit_code: int, output_tail: str):
        self.log.append(f"[ล้มเหลว] {label} ปิดตัวเองทันที (exit code {exit_code})")
        detail = output_tail or "(ไม่มีข้อความ error เพิ่มเติม — ดู log ด้านบน)"
        QMessageBox.critical(self, f"เปิด {label} ไม่สำเร็จ", detail[-1000:])

    def _on_open_booth(self):
        args = [
            "--booth-name", self.booth_name_input.text().strip() or "MONGDEE Demo Booth",
            "--event-id", self.event_id_input.text().strip() or "1-Day-at-IMPACT",
        ]
        cameras = self.cameras_input.text().strip()
        if cameras:
            args += ["--cameras", cameras]
        self._launch("app.py", args, label="บูธ (Booth)")

    def _on_open_dashboard(self):
        self._launch("dashboard.py", label="Dashboard")

    def _on_open_trainer(self):
        self._launch("trainer.py", label="AI Trainer")

    def _on_open_web(self):
        args = [
            "--booth-name", self.booth_name_input.text().strip() or "MONGDEE Demo Booth",
            "--event-id", self.event_id_input.text().strip() or "1-Day-at-IMPACT",
        ]
        cameras = self.cameras_input.text().strip()
        if cameras:
            args += ["--cameras", cameras]
        self._launch("web_server.py", args, label="เว็บเบราว์เซอร์")

    # -------------------------------------------------------------- setup
    def _on_run_setup(self):
        if not INSTALL_SCRIPT.exists():
            QMessageBox.critical(self, "ไม่พบตัวติดตั้ง", f"ไม่พบไฟล์ {INSTALL_SCRIPT.name}")
            return
        self.setup_btn.setEnabled(False)
        self.log.clear()
        self.log.append(f"กำลังรัน {INSTALL_SCRIPT.name} ... (อาจใช้เวลาหลายนาที)")

        self.install_process = QProcess(self)
        self.install_process.setWorkingDirectory(str(ROOT))
        install_args = ["--gpu"] if self.gpu_checkbox.isChecked() else []
        if IS_WINDOWS:
            self.install_process.setProgram("cmd.exe")
            self.install_process.setArguments(["/c", str(INSTALL_SCRIPT)] + install_args)
        else:
            self.install_process.setProgram("bash")
            self.install_process.setArguments([str(INSTALL_SCRIPT)] + install_args)
        self.install_process.readyReadStandardOutput.connect(self._on_install_output)
        self.install_process.readyReadStandardError.connect(self._on_install_output)
        self.install_process.finished.connect(self._on_install_finished)
        self.install_process.start()

    def _on_install_output(self):
        if not self.install_process:
            return
        data = bytes(self.install_process.readAllStandardOutput()).decode("utf-8", "ignore")
        data += bytes(self.install_process.readAllStandardError()).decode("utf-8", "ignore")
        if data:
            self.log.append(data.rstrip())

    def _on_install_finished(self, exit_code, _status):
        self.setup_btn.setEnabled(True)
        if exit_code == 0:
            self.log.append("\n[เสร็จสมบูรณ์] ติดตั้งเสร็จสมบูรณ์")
        else:
            self.log.append(f"\n[ล้มเหลว] ติดตั้งล้มเหลว (exit code {exit_code}) — ดู log ด้านบน")
        self._refresh_ready_state()


def main():
    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
