"""Training monitor tab – real-time YOLO training tracking with system metrics."""

from __future__ import annotations

import os
from datetime import datetime

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QSplitter,
    QFrame,
    QFileDialog,
    QScrollArea,
    QSizePolicy,
)

from theme import Colors
from utils import YoloLogParser, strip_ansi
from workers import SystemMonitorWorker, TrainingWorker
from widgets import CircularGauge, MetricCard, BestMetricCard, ChartWidget


class TrainingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parser = YoloLogParser()
        self._sys_worker: SystemMonitorWorker | None = None
        self._train_worker: TrainingWorker | None = None
        self._start_time: datetime | None = None
        self._chart_cycle = 0

        self._build_ui()
        self._start_system_monitor()
        self._start_timer()

    # ── UI Build ────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        # ── Top controls ──
        top = QHBoxLayout()
        top.setSpacing(8)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Eğitim scriptini seçin (.py)")
        self._path_edit.setMinimumWidth(300)

        browse_btn = QPushButton("Gözat")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_script)

        self._start_btn = QPushButton("Başlat")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.setFixedWidth(90)
        self._start_btn.clicked.connect(self._start_training)

        self._stop_btn = QPushButton("Durdur")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setFixedWidth(90)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_training)

        self._status_label = QLabel("Bekliyor")
        self._status_label.setObjectName("statusWaiting")
        self._status_label.setFixedWidth(160)

        self._timer_label = QLabel("00:00:00")
        self._timer_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 14px;"
        )

        top.addWidget(QLabel("Script:"))
        top.addWidget(self._path_edit, 1)
        top.addWidget(browse_btn)
        top.addWidget(self._start_btn)
        top.addWidget(self._stop_btn)
        top.addSpacing(12)
        top.addWidget(self._status_label)
        top.addWidget(self._timer_label)
        root.addLayout(top)

        # ── Progress bar ──
        self._progress = QProgressBar()
        self._progress.setTextVisible(True)
        self._progress.setFormat("Epoch: %v / %m")
        self._progress.setMaximum(1)
        self._progress.setValue(0)
        root.addWidget(self._progress)

        # ── Main content ──
        content = QHBoxLayout()
        content.setSpacing(8)

        # Left sidebar
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(210)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        sidebar_widget = QWidget()
        sidebar = QVBoxLayout(sidebar_widget)
        sidebar.setContentsMargins(6, 6, 6, 6)
        sidebar.setSpacing(6)

        # GPU gauges
        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(4)
        self._temp_gauge = CircularGauge("GPU Sıcaklık", "\u00b0C", 0, 100, 70, 85)
        self._util_gauge = CircularGauge("GPU Kullanım", "%", 0, 100, 80, 95)
        gauge_row.addWidget(self._temp_gauge)
        gauge_row.addWidget(self._util_gauge)
        sidebar.addLayout(gauge_row)

        # Metric cards
        self._vram_card = MetricCard("VRAM", "", Colors.ACCENT)
        self._power_card = MetricCard("Güç", " W", Colors.ORANGE)
        self._speed_card = MetricCard("Eğitim Hızı", "", Colors.CYAN)
        self._cpu_card = MetricCard("CPU", "%", Colors.GREEN)
        self._ram_card = MetricCard("RAM", "", Colors.PURPLE)
        self._gpu_name_label = QLabel("")
        self._gpu_name_label.setObjectName("dimLabel")
        self._gpu_name_label.setAlignment(Qt.AlignCenter)
        self._gpu_name_label.setWordWrap(True)

        sidebar.addWidget(self._gpu_name_label)
        sidebar.addWidget(self._vram_card)
        sidebar.addWidget(self._power_card)
        sidebar.addWidget(self._speed_card)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet(f"color: {Colors.BORDER};")
        sidebar.addWidget(sep1)
        sidebar.addWidget(self._cpu_card)
        sidebar.addWidget(self._ram_card)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: {Colors.BORDER};")
        sidebar.addWidget(sep2)

        # Best metrics
        best_title = QLabel("En İyi Metrikler")
        best_title.setObjectName("accentLabel")
        best_title.setAlignment(Qt.AlignCenter)
        sidebar.addWidget(best_title)

        self._best_map50 = BestMetricCard("mAP50", Colors.PURPLE)
        self._best_map50_95 = BestMetricCard("mAP50-95", Colors.ACCENT)
        self._best_precision = BestMetricCard("Precision", Colors.GREEN)
        self._best_recall = BestMetricCard("Recall", Colors.CYAN)
        sidebar.addWidget(self._best_map50)
        sidebar.addWidget(self._best_map50_95)
        sidebar.addWidget(self._best_precision)
        sidebar.addWidget(self._best_recall)

        sidebar.addStretch()
        sidebar_scroll.setWidget(sidebar_widget)
        content.addWidget(sidebar_scroll)

        # Right area – splitter: log | charts
        right_splitter = QSplitter(Qt.Horizontal)
        right_splitter.setChildrenCollapsible(False)

        # Terminal log
        log_frame = QWidget()
        log_layout = QVBoxLayout(log_frame)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(4)

        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Terminal Log"))
        clear_btn = QPushButton("Temizle")
        clear_btn.setFixedWidth(70)
        clear_btn.clicked.connect(lambda: self._log_text.clear())
        log_header.addStretch()
        log_header.addWidget(clear_btn)
        log_layout.addLayout(log_header)

        self._log_text = QPlainTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumBlockCount(5000)
        log_layout.addWidget(self._log_text)

        # Charts
        self._chart = ChartWidget()

        right_splitter.addWidget(log_frame)
        right_splitter.addWidget(self._chart)
        right_splitter.setSizes([400, 500])

        content.addWidget(right_splitter, 1)
        root.addLayout(content, 1)

    # ── System Monitor ──────────────────────────────────────────

    def _start_system_monitor(self):
        self._sys_worker = SystemMonitorWorker(2000)
        self._sys_worker.stats_ready.connect(self._on_sys_stats)
        self._sys_worker.start()

    def _on_sys_stats(self, data: dict):
        gpu = data.get("gpu")
        if gpu:
            self._temp_gauge.set_value(gpu["temperature"])
            self._util_gauge.set_value(gpu["utilization"])
            self._vram_card.set_value(
                f"{gpu['memory_used']}/{gpu['memory_total']}", " MB"
            )
            pw_draw = gpu["power_draw"]
            pw_limit = gpu["power_limit"]
            if pw_limit > 0:
                self._power_card.set_value(f"{pw_draw:.0f}/{pw_limit:.0f}")
            else:
                self._power_card.set_value(f"{pw_draw:.0f}")
            self._gpu_name_label.setText(gpu["name"])
            self._chart.update_temperature(gpu["temperature"])

        self._cpu_card.set_value(f"{data.get('cpu_percent', 0):.0f}")
        self._ram_card.set_value(
            f"{data.get('ram_used_gb', 0)}/{data.get('ram_total_gb', 0)}", " GB"
        )

        # Redraw charts periodically
        self._chart_cycle += 1
        if self._chart_cycle % 3 == 0:
            p = self._parser
            if p.epochs_list:
                self._chart.update_losses(
                    p.epochs_list, p.box_losses, p.cls_losses, p.dfl_losses
                )
            if p.val_epochs:
                self._chart.update_maps(p.val_epochs, p.map50s, p.map50_95s)
            self._chart.redraw()

    # ── Timer ───────────────────────────────────────────────────

    def _start_timer(self):
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._update_timer)
        self._tick_timer.start(1000)

    def _update_timer(self):
        if self._start_time:
            elapsed = datetime.now() - self._start_time
            h, rem = divmod(int(elapsed.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            self._timer_label.setText(f"{h:02d}:{m:02d}:{s:02d}")

    # ── Training Control ────────────────────────────────────────

    def _browse_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Eğitim Scripti Seç",
            os.path.dirname(self._path_edit.text()) or "D:/DirekHasarTespiti",
            "Python Files (*.py)",
        )
        if path:
            self._path_edit.setText(path)

    def _start_training(self):
        script = self._path_edit.text().strip()
        if not script or not os.path.isfile(script):
            self._log_text.appendPlainText("[HATA] Geçerli bir script seçin.")
            return

        self._parser.reset()
        self._start_time = datetime.now()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_label.setText("Eğitim Çalışıyor")
        self._status_label.setObjectName("statusRunning")
        self._status_label.setStyleSheet(f"color: {Colors.GREEN}; font-weight: bold;")

        self._train_worker = TrainingWorker(script)
        self._train_worker.log_line.connect(self._on_log_line)
        self._train_worker.log_replace.connect(self._on_log_replace)
        self._train_worker.finished_signal.connect(self._on_training_finished)
        self._train_worker.start()

    def _stop_training(self):
        if self._train_worker:
            self._train_worker.stop()

    def _on_log_line(self, line: str):
        """Append a new line to the terminal log."""
        line = strip_ansi(line)
        self._log_text.appendPlainText(line)
        self._parse_and_update(line)

    def _on_log_replace(self, line: str):
        """Replace the last line in the terminal log (\\r from tqdm).

        Does NOT parse metrics – batch-level updates should not be added
        to the charts.  Only epoch-final lines (\\n) feed the parser.
        """
        line = strip_ansi(line)
        cursor = self._log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertText(line)
        self._log_text.setTextCursor(cursor)
        self._log_text.ensureCursorVisible()

        # Update epoch/total on progress bar (lightweight, no chart data)
        import re
        m = re.search(r"(\d+)/(\d+)\s+", line)
        if m:
            self._progress.setMaximum(int(m.group(2)))
            self._progress.setValue(int(m.group(1)))

        # Parse training speed from tqdm (e.g. "3.51it/s" or "4.9s/it")
        m = re.search(r"(\d+\.?\d*)(it/s|s/it)", line)
        if m:
            val, unit = m.group(1), m.group(2)
            self._speed_card.set_value(f"{val} {unit}")

    def _parse_and_update(self, line: str):
        """Parse a log line for YOLO metrics and update UI."""
        parsed = self._parser.parse_line(line)
        if parsed:
            if parsed["type"] == "epoch":
                self._progress.setMaximum(parsed["total"])
                self._progress.setValue(parsed["epoch"])
            elif parsed["type"] == "validation":
                p = self._parser
                self._best_map50.set_value(p.best_map50, p.best_map50_epoch)
                self._best_map50_95.set_value(p.best_map50_95, p.best_map50_95_epoch)
                self._best_precision.set_value(p.best_precision, p.best_precision_epoch)
                self._best_recall.set_value(p.best_recall, p.best_recall_epoch)

    def _on_training_finished(self, rc: int):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

        if rc == 0:
            self._status_label.setText("Tamamlandı")
            self._status_label.setStyleSheet(f"color: {Colors.GREEN}; font-weight: bold;")
        else:
            self._status_label.setText("Durduruldu")
            self._status_label.setStyleSheet(f"color: {Colors.RED}; font-weight: bold;")

        elapsed = ""
        if self._start_time:
            dt = datetime.now() - self._start_time
            h, rem = divmod(int(dt.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            elapsed = f" ({h:02d}:{m:02d}:{s:02d})"
        self._log_text.appendPlainText(f"\n{'='*50}")
        self._log_text.appendPlainText(f"Eğitim sonlandı (kod: {rc}){elapsed}")

        # Force final chart update
        p = self._parser
        if p.epochs_list:
            self._chart.update_losses(
                p.epochs_list, p.box_losses, p.cls_losses, p.dfl_losses
            )
        if p.val_epochs:
            self._chart.update_maps(p.val_epochs, p.map50s, p.map50_95s)
        self._chart.redraw()

        self._train_worker = None

    # ── Cleanup ─────────────────────────────────────────────────

    def cleanup(self):
        if self._sys_worker:
            self._sys_worker.stop()
        if self._train_worker:
            self._train_worker.stop()
