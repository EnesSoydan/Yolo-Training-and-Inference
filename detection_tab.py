"""Detection / Test tab – model selection, inference (Normal / SAHI), image browsing."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QKeyEvent
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QFrame,
    QFileDialog,
    QScrollArea,
    QSplitter,
    QCheckBox,
    QMessageBox,
)

from theme import Colors, CLASS_COLORS
from workers import InferenceWorker, ThumbnailWorker
from widgets import ZoomableImageView, ThumbnailCard


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class DetectionTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._model_path = ""
        self._folder_path = ""
        self._image_paths: list[str] = []
        self._current_index = 0
        self._class_names: list[str] = []
        self._class_checks: list[QCheckBox] = []
        self._thumbnail_cards: list[ThumbnailCard] = []
        self._inf_worker: InferenceWorker | None = None
        self._thumb_worker: ThumbnailWorker | None = None

        self._build_ui()

    # ── UI Build ────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        # ── Model row ──
        model_row = QHBoxLayout()
        model_row.setSpacing(6)
        model_row.addWidget(QLabel("Model:"))

        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("best.pt model dosyası")
        model_row.addWidget(self._model_edit, 1)

        model_browse = QPushButton("Gözat")
        model_browse.setFixedWidth(65)
        model_browse.clicked.connect(self._browse_model)

        auto_btn = QPushButton("Otomatik Bul")
        auto_btn.setFixedWidth(100)
        auto_btn.clicked.connect(self._auto_find_model)

        self._model_status = QLabel("")
        self._model_status.setFixedWidth(180)

        model_row.addWidget(model_browse)
        model_row.addWidget(auto_btn)
        model_row.addWidget(self._model_status)
        root.addLayout(model_row)

        # ── Settings row ──
        settings_row = QHBoxLayout()
        settings_row.setSpacing(6)

        settings_row.addWidget(QLabel("Mod:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Normal Predict", "normal")
        self._mode_combo.addItem("SAHI (Dilimli)", "sahi")
        self._mode_combo.setFixedWidth(140)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        settings_row.addWidget(self._mode_combo)

        settings_row.addSpacing(10)
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.01, 1.0)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.25)
        self._conf_spin.setPrefix("Güven: ")
        self._conf_spin.setFixedWidth(120)
        settings_row.addWidget(self._conf_spin)

        self._slice_label = QLabel("Dilim:")
        self._slice_spin = QSpinBox()
        self._slice_spin.setRange(128, 2048)
        self._slice_spin.setSingleStep(64)
        self._slice_spin.setValue(640)
        self._slice_spin.setFixedWidth(100)
        settings_row.addWidget(self._slice_label)
        settings_row.addWidget(self._slice_spin)

        # Initially hide slice settings (Normal mode default)
        self._slice_label.setVisible(False)
        self._slice_spin.setVisible(False)

        settings_row.addStretch()
        root.addLayout(settings_row)

        # ── Folder row ──
        folder_row = QHBoxLayout()
        folder_row.setSpacing(6)
        folder_row.addWidget(QLabel("Klasör:"))

        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Görüntülerin bulunduğu klasör")
        folder_row.addWidget(self._folder_edit, 1)

        folder_browse = QPushButton("Gözat")
        folder_browse.setFixedWidth(65)
        folder_browse.clicked.connect(self._browse_folder)

        load_btn = QPushButton("Görüntüle")
        load_btn.setFixedWidth(85)
        load_btn.clicked.connect(self._load_images)

        self._run_btn = QPushButton("Başlat")
        self._run_btn.setObjectName("startBtn")
        self._run_btn.setFixedWidth(100)
        self._run_btn.clicked.connect(self._start_inference)

        self._stop_btn = QPushButton("Durdur")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setFixedWidth(70)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_inference)

        self._folder_status = QLabel("")
        self._folder_status.setFixedWidth(160)

        folder_row.addWidget(folder_browse)
        folder_row.addWidget(load_btn)
        folder_row.addWidget(self._run_btn)
        folder_row.addWidget(self._stop_btn)
        folder_row.addWidget(self._folder_status)
        root.addLayout(folder_row)

        # ── Main content: thumbnails | image | info ──
        content = QSplitter(Qt.Horizontal)
        content.setChildrenCollapsible(False)

        # Left – thumbnail gallery
        thumb_frame = QWidget()
        thumb_frame.setFixedWidth(170)
        thumb_layout = QVBoxLayout(thumb_frame)
        thumb_layout.setContentsMargins(0, 0, 0, 0)
        thumb_layout.setSpacing(0)

        thumb_header = QLabel("Önizlemeler")
        thumb_header.setObjectName("accentLabel")
        thumb_header.setAlignment(Qt.AlignCenter)
        thumb_header.setFixedHeight(24)
        thumb_layout.addWidget(thumb_header)

        self._thumb_scroll = QScrollArea()
        self._thumb_scroll.setWidgetResizable(True)
        self._thumb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._thumb_container = QWidget()
        self._thumb_list_layout = QVBoxLayout(self._thumb_container)
        self._thumb_list_layout.setContentsMargins(4, 4, 4, 4)
        self._thumb_list_layout.setSpacing(6)
        self._thumb_list_layout.setAlignment(Qt.AlignTop)
        self._thumb_scroll.setWidget(self._thumb_container)
        thumb_layout.addWidget(self._thumb_scroll)

        content.addWidget(thumb_frame)

        # Center – image viewer + navigation
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)

        self._image_viewer = ZoomableImageView()
        self._image_viewer.zoom_changed.connect(self._on_zoom_changed)
        center_layout.addWidget(self._image_viewer, 1)

        # Nav bar
        nav = QHBoxLayout()
        nav.setSpacing(8)

        self._prev_btn = QPushButton("Önceki")
        self._prev_btn.setFixedWidth(70)
        self._prev_btn.clicked.connect(self._prev_image)

        self._page_label = QLabel("0 / 0")
        self._page_label.setAlignment(Qt.AlignCenter)
        self._page_label.setFixedWidth(80)

        self._next_btn = QPushButton("Sonraki")
        self._next_btn.setFixedWidth(70)
        self._next_btn.clicked.connect(self._next_image)

        self._filename_label = QLabel("")
        self._filename_label.setObjectName("dimLabel")

        self._zoom_label = QLabel("100%")
        self._zoom_label.setFixedWidth(50)
        self._zoom_label.setAlignment(Qt.AlignCenter)
        self._zoom_label.setObjectName("dimLabel")

        fit_btn = QPushButton("Sığdır")
        fit_btn.setFixedWidth(55)
        fit_btn.clicked.connect(self._image_viewer.fit_to_view)

        full_btn = QPushButton("100%")
        full_btn.setFixedWidth(45)
        full_btn.clicked.connect(lambda: self._image_viewer.set_zoom(1.0))

        zoom_out = QPushButton("-")
        zoom_out.setFixedWidth(30)
        zoom_out.clicked.connect(
            lambda: self._image_viewer.set_zoom(
                max(0.1, self._image_viewer._zoom / 1.15)
            )
        )

        zoom_in = QPushButton("+")
        zoom_in.setFixedWidth(30)
        zoom_in.clicked.connect(
            lambda: self._image_viewer.set_zoom(
                min(20.0, self._image_viewer._zoom * 1.15)
            )
        )

        nav.addWidget(self._prev_btn)
        nav.addWidget(self._page_label)
        nav.addWidget(self._next_btn)
        nav.addSpacing(12)
        nav.addWidget(self._filename_label, 1)
        nav.addWidget(zoom_out)
        nav.addWidget(self._zoom_label)
        nav.addWidget(zoom_in)
        nav.addWidget(fit_btn)
        nav.addWidget(full_btn)
        center_layout.addLayout(nav)

        content.addWidget(center)

        # Right – class filter + info
        right_frame = QWidget()
        right_frame.setFixedWidth(200)
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(6, 0, 6, 6)
        right_layout.setSpacing(6)

        cls_title = QLabel("Sınıf Filtresi")
        cls_title.setObjectName("accentLabel")
        cls_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(cls_title)

        cls_btns = QHBoxLayout()
        sel_all = QPushButton("Tümünü Seç")
        sel_all.setFixedHeight(26)
        sel_all.clicked.connect(lambda: self._set_all_classes(True))
        desel_all = QPushButton("Kaldır")
        desel_all.setFixedHeight(26)
        desel_all.clicked.connect(lambda: self._set_all_classes(False))
        cls_btns.addWidget(sel_all)
        cls_btns.addWidget(desel_all)
        right_layout.addLayout(cls_btns)

        self._class_scroll = QScrollArea()
        self._class_scroll.setWidgetResizable(True)
        self._class_container = QWidget()
        self._class_layout = QVBoxLayout(self._class_container)
        self._class_layout.setContentsMargins(4, 4, 4, 4)
        self._class_layout.setSpacing(3)
        self._class_layout.setAlignment(Qt.AlignTop)
        self._class_scroll.setWidget(self._class_container)
        right_layout.addWidget(self._class_scroll, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {Colors.BORDER};")
        right_layout.addWidget(sep)

        info_title = QLabel("Görüntü Bilgisi")
        info_title.setObjectName("accentLabel")
        info_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(info_title)

        self._info_file = QLabel("Dosya: -")
        self._info_file.setObjectName("dimLabel")
        self._info_file.setWordWrap(True)
        self._info_dim = QLabel("Boyut: -")
        self._info_dim.setObjectName("dimLabel")
        self._info_size = QLabel("Dosya boyutu: -")
        self._info_size.setObjectName("dimLabel")
        right_layout.addWidget(self._info_file)
        right_layout.addWidget(self._info_dim)
        right_layout.addWidget(self._info_size)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"color: {Colors.BORDER};")
        right_layout.addWidget(sep2)

        shortcuts_title = QLabel("Kısayollar")
        shortcuts_title.setObjectName("dimLabel")
        shortcuts_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(shortcuts_title)

        shortcuts = [
            "\u2190 \u2192  Önceki / Sonraki",
            "+ -   Yakınlaştır / Uzaklaştır",
            "0     Sığdır",
            "1     %100",
            "Home  İlk görüntü",
            "End   Son görüntü",
            "Scroll  Zoom",
            "Çift tık  Sığdır",
        ]
        for s in shortcuts:
            lbl = QLabel(s)
            lbl.setStyleSheet(
                f"color: {Colors.TEXT_DIM}; font-size: 10px; "
                f"font-family: 'Cascadia Code', monospace;"
            )
            right_layout.addWidget(lbl)

        right_layout.addStretch()
        content.addWidget(right_frame)

        content.setSizes([170, 600, 200])
        root.addWidget(content, 1)

    # ── Mode Switch ─────────────────────────────────────────────

    def _on_mode_changed(self, index: int):
        is_sahi = self._mode_combo.currentData() == "sahi"
        self._slice_label.setVisible(is_sahi)
        self._slice_spin.setVisible(is_sahi)

    # ── Model ───────────────────────────────────────────────────

    def _browse_model(self):
        start = os.path.dirname(self._model_edit.text()) or "D:/DirekHasarTespiti/runs"
        path, _ = QFileDialog.getOpenFileName(
            self, "Model Seç", start, "YOLO Model (*.pt)"
        )
        if path:
            self._model_edit.setText(path)
            self._load_model_classes(path)

    def _auto_find_model(self):
        runs_dir = Path("D:/DirekHasarTespiti/runs")
        if not runs_dir.exists():
            self._model_status.setText("runs/ bulunamadı")
            self._model_status.setStyleSheet(f"color: {Colors.RED};")
            return

        best_files = sorted(
            runs_dir.rglob("best.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if best_files:
            path = str(best_files[0])
            self._model_edit.setText(path)
            self._load_model_classes(path)
        else:
            self._model_status.setText("best.pt bulunamadı")
            self._model_status.setStyleSheet(f"color: {Colors.RED};")

    def _load_model_classes(self, path: str):
        self._model_path = path
        try:
            from ultralytics import YOLO

            model = YOLO(path)
            names = model.names
            self._class_names = [names[i] for i in sorted(names.keys())]
            self._model_status.setText(
                f"Model yüklendi ({len(self._class_names)} sınıf)"
            )
            self._model_status.setStyleSheet(f"color: {Colors.GREEN};")
            self._build_class_checkboxes()
        except Exception as e:
            self._model_status.setText(f"Hata: {e}")
            self._model_status.setStyleSheet(f"color: {Colors.RED};")

    def _build_class_checkboxes(self):
        for cb in self._class_checks:
            cb.setParent(None)
        self._class_checks.clear()

        for i, name in enumerate(self._class_names):
            cb = QCheckBox(name)
            cb.setChecked(True)
            color = CLASS_COLORS[i % len(CLASS_COLORS)]
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: bold; }}")
            self._class_layout.addWidget(cb)
            self._class_checks.append(cb)

    def _set_all_classes(self, state: bool):
        for cb in self._class_checks:
            cb.setChecked(state)

    def _get_selected_classes(self) -> list:
        return [cb.text() for cb in self._class_checks if cb.isChecked()]

    # ── Folder / Images ────────────────────────────────────────

    def _browse_folder(self):
        start = self._folder_edit.text() or "D:/DirekHasarTespiti"
        path = QFileDialog.getExistingDirectory(self, "Görüntü Klasörü Seç", start)
        if path:
            self._folder_edit.setText(path)

    def _load_images(self):
        folder = self._folder_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            self._folder_status.setText("Geçersiz klasör")
            self._folder_status.setStyleSheet(f"color: {Colors.RED};")
            return

        self._folder_path = folder
        self._image_paths = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ]
        )

        if not self._image_paths:
            self._folder_status.setText("Görüntü bulunamadı")
            self._folder_status.setStyleSheet(f"color: {Colors.RED};")
            return

        self._folder_status.setText(f"{len(self._image_paths)} görüntü")
        self._folder_status.setStyleSheet(f"color: {Colors.GREEN};")

        self._current_index = 0
        self._build_thumbnails()
        self._show_image(0)

    def _build_thumbnails(self):
        for card in self._thumbnail_cards:
            card.setParent(None)
        self._thumbnail_cards.clear()

        for i, path in enumerate(self._image_paths):
            fname = os.path.basename(path)
            card = ThumbnailCard(i, fname)
            card.clicked.connect(self._on_thumbnail_clicked)
            self._thumb_list_layout.addWidget(card)
            self._thumbnail_cards.append(card)

        if self._thumb_worker:
            self._thumb_worker.stop()
        self._thumb_worker = ThumbnailWorker(self._image_paths)
        self._thumb_worker.thumbnail_ready.connect(self._on_thumbnail_loaded)
        self._thumb_worker.start()

    def _on_thumbnail_loaded(self, index: int, pixmap):
        if 0 <= index < len(self._thumbnail_cards):
            self._thumbnail_cards[index].set_pixmap(pixmap)

    def _on_thumbnail_clicked(self, index: int):
        self._show_image(index)

    # ── Image Display ───────────────────────────────────────────

    def _show_image(self, index: int):
        if not self._image_paths or index < 0 or index >= len(self._image_paths):
            return

        if 0 <= self._current_index < len(self._thumbnail_cards):
            self._thumbnail_cards[self._current_index].set_selected(False)

        self._current_index = index
        path = self._image_paths[index]

        self._image_viewer.set_image(path)

        if index < len(self._thumbnail_cards):
            self._thumbnail_cards[index].set_selected(True)
            card = self._thumbnail_cards[index]
            self._thumb_scroll.ensureWidgetVisible(card)

        fname = os.path.basename(path)
        self._filename_label.setText(fname)
        self._page_label.setText(f"{index + 1} / {len(self._image_paths)}")

        self._info_file.setText(f"Dosya: {fname}")
        w, h = self._image_viewer.get_image_size()
        self._info_dim.setText(f"Boyut: {w} x {h}")
        try:
            size_bytes = os.path.getsize(path)
            if size_bytes > 1024 * 1024:
                self._info_size.setText(
                    f"Dosya boyutu: {size_bytes / (1024*1024):.1f} MB"
                )
            else:
                self._info_size.setText(f"Dosya boyutu: {size_bytes / 1024:.0f} KB")
        except OSError:
            self._info_size.setText("Dosya boyutu: -")

    def _prev_image(self):
        if self._image_paths and self._current_index > 0:
            self._show_image(self._current_index - 1)

    def _next_image(self):
        if self._image_paths and self._current_index < len(self._image_paths) - 1:
            self._show_image(self._current_index + 1)

    # ── Keyboard Shortcuts ──────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        if key == Qt.Key_Left:
            self._prev_image()
        elif key == Qt.Key_Right:
            self._next_image()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self._image_viewer.set_zoom(min(20.0, self._image_viewer._zoom * 1.15))
        elif key == Qt.Key_Minus:
            self._image_viewer.set_zoom(max(0.1, self._image_viewer._zoom / 1.15))
        elif key == Qt.Key_0:
            self._image_viewer.fit_to_view()
        elif key == Qt.Key_1:
            self._image_viewer.set_zoom(1.0)
        elif key == Qt.Key_Home:
            if self._image_paths:
                self._show_image(0)
        elif key == Qt.Key_End:
            if self._image_paths:
                self._show_image(len(self._image_paths) - 1)
        else:
            super().keyPressEvent(event)

    def _on_zoom_changed(self, zoom: float):
        self._zoom_label.setText(f"{zoom * 100:.0f}%")

    # ── Inference ───────────────────────────────────────────────

    def _start_inference(self):
        model = self._model_edit.text().strip()
        folder = self._folder_edit.text().strip()

        if not model or not os.path.isfile(model):
            QMessageBox.warning(self, "Hata", "Geçerli bir model dosyası seçin.")
            return
        if not self._image_paths:
            QMessageBox.warning(self, "Hata", "Önce görüntüleri yükleyin.")
            return

        mode = self._mode_combo.currentData()
        confidence = self._conf_spin.value()
        slice_size = self._slice_spin.value()
        selected = self._get_selected_classes()

        out_subdir = "sahi_results" if mode == "sahi" else "predict_results"
        output_dir = os.path.join(folder, out_subdir)

        self._run_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._folder_status.setText("İşlem başlıyor...")
        self._folder_status.setStyleSheet(f"color: {Colors.YELLOW};")

        for card in self._thumbnail_cards:
            card.set_status("")

        self._inf_worker = InferenceWorker(
            model_path=model,
            image_paths=self._image_paths,
            output_dir=output_dir,
            confidence=confidence,
            slice_size=slice_size,
            selected_classes=selected if selected else None,
            mode=mode,
        )
        self._inf_worker.progress.connect(self._on_inf_progress)
        self._inf_worker.result_ready.connect(self._on_inf_result)
        self._inf_worker.error.connect(self._on_inf_error)
        self._inf_worker.all_done.connect(self._on_inf_done)
        self._inf_worker.start()

    def _stop_inference(self):
        if self._inf_worker:
            self._inf_worker.stop()

    def _on_inf_progress(self, idx: int, total: int, filename: str):
        if 0 <= idx < len(self._thumbnail_cards):
            self._thumbnail_cards[idx].set_status("...", Colors.YELLOW)
        self._folder_status.setText(f"{idx + 1}/{total}")

    def _on_inf_result(self, idx: int, out_path: str, det_count: int):
        if 0 <= idx < len(self._thumbnail_cards):
            status = str(det_count) if det_count > 0 else "0"
            color = Colors.GREEN if det_count > 0 else Colors.TEXT_DIM
            self._thumbnail_cards[idx].set_status(status, color)

            pm = QPixmap(out_path)
            if not pm.isNull():
                self._thumbnail_cards[idx].set_pixmap(pm)

        if 0 <= idx < len(self._image_paths):
            self._image_paths[idx] = out_path

        if idx == self._current_index:
            self._image_viewer.set_image(out_path)

    def _on_inf_error(self, idx: int, msg: str):
        if 0 <= idx < len(self._thumbnail_cards):
            self._thumbnail_cards[idx].set_status("HATA", Colors.RED)

    def _on_inf_done(self, total_dets: int):
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._folder_status.setText(f"Tamamlandı: {total_dets} tespit")
        self._folder_status.setStyleSheet(f"color: {Colors.GREEN};")
        self._inf_worker = None

    # ── Cleanup ─────────────────────────────────────────────────

    def cleanup(self):
        if self._inf_worker:
            self._inf_worker.stop()
        if self._thumb_worker:
            self._thumb_worker.stop()
