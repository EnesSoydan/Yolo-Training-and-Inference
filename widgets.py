"""Custom widgets: CircularGauge, MetricCard, ChartWidget, ZoomableImageView, ThumbnailCard."""

from __future__ import annotations

import os
from collections import deque

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import (
    QPainter,
    QPen,
    QColor,
    QFont,
    QPixmap,
    QWheelEvent,
    QMouseEvent,
)
from PySide6.QtWidgets import (
    QWidget,
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QSizePolicy,
)

from theme import Colors


# ── Circular Gauge ──────────────────────────────────────────────


class CircularGauge(QWidget):
    """Arc-style gauge for temperature / utilization."""

    def __init__(
        self,
        title: str = "",
        unit: str = "",
        min_val: float = 0,
        max_val: float = 100,
        warn_val: float = 70,
        crit_val: float = 85,
        parent=None,
    ):
        super().__init__(parent)
        self._title = title
        self._unit = unit
        self._min = min_val
        self._max = max_val
        self._warn = warn_val
        self._crit = crit_val
        self._value = 0.0
        self.setMinimumSize(100, 100)
        self.setMaximumSize(140, 140)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def set_value(self, value: float):
        self._value = max(self._min, min(self._max, value))
        self.update()

    def _color(self) -> QColor:
        if self._value >= self._crit:
            return QColor(Colors.RED)
        elif self._value >= self._warn:
            return QColor(Colors.ORANGE)
        else:
            return QColor(Colors.GREEN)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        side = min(w, h)
        pad = 14
        arc_w = 8
        rect = QRectF(pad, pad, side - 2 * pad, side - 2 * pad)

        start_angle = 225 * 16  # 5 o'clock position
        span_angle = -270 * 16  # sweep 270 degrees clockwise

        # Background arc
        pen = QPen(QColor(Colors.BORDER), arc_w, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, start_angle, span_angle)

        # Value arc
        ratio = (self._value - self._min) / max(self._max - self._min, 1)
        pen.setColor(self._color())
        painter.setPen(pen)
        painter.drawArc(rect, start_angle, int(span_angle * ratio))

        # Value text
        painter.setPen(QColor(Colors.TEXT))
        font = QFont("Segoe UI", 16, QFont.Bold)
        painter.setFont(font)
        val_text = f"{int(self._value)}"
        painter.drawText(rect, Qt.AlignCenter, val_text)

        # Unit (below value)
        font.setPointSize(8)
        font.setWeight(QFont.Normal)
        painter.setFont(font)
        painter.setPen(QColor(Colors.TEXT_DIM))
        unit_rect = QRectF(rect.x(), rect.y() + 20, rect.width(), rect.height())
        painter.drawText(unit_rect, Qt.AlignCenter, self._unit)

        # Title (above gauge)
        font.setPointSize(9)
        painter.setFont(font)
        painter.setPen(QColor(Colors.TEXT_DIM))
        painter.drawText(0, 10, w, 14, Qt.AlignCenter, self._title)


# ── Metric Card ─────────────────────────────────────────────────


class MetricCard(QFrame):
    """Small card showing a single metric value."""

    def __init__(
        self,
        title: str = "",
        unit: str = "",
        accent_color: str = Colors.ACCENT,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("card")
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(0)

        self._title_label = QLabel(title)
        self._title_label.setObjectName("dimLabel")
        self._title_label.setStyleSheet(f"font-size: 10px; color: {Colors.TEXT_DIM};")

        self._value_label = QLabel("--")
        self._value_label.setStyleSheet(
            f"font-size: 16px; font-weight: bold; color: {accent_color};"
        )

        self._unit = unit
        layout.addWidget(self._title_label)
        layout.addWidget(self._value_label)

    def set_value(self, value, suffix: str = ""):
        text = f"{value}{self._unit}" if not suffix else f"{value}{suffix}"
        self._value_label.setText(text)


# ── Best Metric Card ────────────────────────────────────────────


class BestMetricCard(QFrame):
    """Card showing best metric value with its epoch."""

    def __init__(self, title: str, accent_color: str = Colors.PURPLE, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setFixedHeight(50)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(0)

        self._title_label = QLabel(title)
        self._title_label.setStyleSheet(f"font-size: 10px; color: {Colors.TEXT_DIM};")

        hl = QHBoxLayout()
        hl.setSpacing(6)
        self._value_label = QLabel("--")
        self._value_label.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {accent_color};"
        )
        self._epoch_label = QLabel("")
        self._epoch_label.setStyleSheet(f"font-size: 10px; color: {Colors.TEXT_DIM};")
        hl.addWidget(self._value_label)
        hl.addWidget(self._epoch_label)
        hl.addStretch()

        layout.addWidget(self._title_label)
        layout.addLayout(hl)

    def set_value(self, value: float, epoch: int):
        self._value_label.setText(f"{value:.4f}")
        self._epoch_label.setText(f"@ epoch {epoch}" if epoch > 0 else "")


# ── Chart Widget (Matplotlib) ───────────────────────────────────


class ChartWidget(QWidget):
    """Three-panel matplotlib chart: GPU temp, Loss, mAP."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.fig = Figure(figsize=(8, 6), dpi=90)
        self.fig.patch.set_facecolor(Colors.BG_CARD)
        self.fig.subplots_adjust(hspace=0.45, left=0.10, right=0.96, top=0.95, bottom=0.06)

        self.ax_temp = self.fig.add_subplot(3, 1, 1)
        self.ax_loss = self.fig.add_subplot(3, 1, 2)
        self.ax_map = self.fig.add_subplot(3, 1, 3)

        for ax in (self.ax_temp, self.ax_loss, self.ax_map):
            ax.set_facecolor(Colors.BG_CARD2)
            ax.tick_params(colors=Colors.TEXT_DIM, labelsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(Colors.BORDER)
            ax.spines["left"].set_color(Colors.BORDER)

        self.canvas = FigureCanvasQTAgg(self.fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self._temp_data = deque(maxlen=200)

    def update_temperature(self, temp: float):
        self._temp_data.append(temp)
        ax = self.ax_temp
        ax.clear()
        ax.set_facecolor(Colors.BG_CARD2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(Colors.BORDER)
        ax.spines["left"].set_color(Colors.BORDER)
        ax.tick_params(colors=Colors.TEXT_DIM, labelsize=8)

        xs = list(range(len(self._temp_data)))
        ys = list(self._temp_data)
        ax.fill_between(xs, ys, alpha=0.15, color=Colors.ORANGE)
        ax.plot(xs, ys, color=Colors.ORANGE, linewidth=1.5)
        ax.axhline(y=85, color=Colors.RED, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylabel("GPU C", fontsize=8, color=Colors.TEXT_DIM)
        if ys:
            ymin = max(0, min(ys) - 5)
            ymax = max(ys) + 5
            ax.set_ylim(ymin, ymax)

    def update_losses(self, epochs, box, cls, dfl):
        ax = self.ax_loss
        ax.clear()
        ax.set_facecolor(Colors.BG_CARD2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(Colors.BORDER)
        ax.spines["left"].set_color(Colors.BORDER)
        ax.tick_params(colors=Colors.TEXT_DIM, labelsize=8)

        ep = list(epochs)
        if ep:
            ax.plot(ep, list(box), color=Colors.ACCENT, linewidth=1.3, label="Box")
            ax.plot(ep, list(cls), color=Colors.GREEN, linewidth=1.3, label="Cls")
            ax.plot(ep, list(dfl), color=Colors.YELLOW, linewidth=1.3, label="DFL")
            ax.legend(fontsize=7, loc="upper right", facecolor=Colors.BG_CARD2,
                      edgecolor=Colors.BORDER, labelcolor=Colors.TEXT_DIM)
        ax.set_ylabel("Loss", fontsize=8, color=Colors.TEXT_DIM)

    def update_maps(self, val_epochs, map50, map50_95):
        ax = self.ax_map
        ax.clear()
        ax.set_facecolor(Colors.BG_CARD2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(Colors.BORDER)
        ax.spines["left"].set_color(Colors.BORDER)
        ax.tick_params(colors=Colors.TEXT_DIM, labelsize=8)

        ep = list(val_epochs)
        if ep:
            ax.plot(ep, list(map50), color=Colors.PURPLE, linewidth=1.3,
                    marker="o", markersize=3, label="mAP50")
            ax.plot(ep, list(map50_95), color=Colors.ACCENT, linewidth=1.3,
                    marker="s", markersize=3, label="mAP50-95")
            ax.legend(fontsize=7, loc="lower right", facecolor=Colors.BG_CARD2,
                      edgecolor=Colors.BORDER, labelcolor=Colors.TEXT_DIM)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mAP", fontsize=8, color=Colors.TEXT_DIM)
        ax.set_xlabel("Epoch", fontsize=8, color=Colors.TEXT_DIM)

    def redraw(self):
        self.canvas.draw_idle()


# ── Zoomable Image View ─────────────────────────────────────────


class ZoomableImageView(QGraphicsView):
    """QGraphicsView with mouse wheel zoom and drag-to-pan."""

    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._zoom = 1.0
        self._fit_scale = 1.0

        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setStyleSheet(
            f"background-color: {Colors.BG_CARD}; border: 1px solid {Colors.BORDER}; border-radius: 6px;"
        )
        self.setMinimumSize(300, 200)

        self._placeholder = QLabel("Görüntü seçilmedi", self)
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet(f"color: {Colors.TEXT_DIM}; font-size: 14px; background: transparent; border: none;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._placeholder.setGeometry(0, 0, self.width(), self.height())

    def set_image(self, path: str):
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        self._placeholder.hide()
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_to_view()

    def set_pixmap(self, pixmap: QPixmap):
        if pixmap.isNull():
            return
        self._placeholder.hide()
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_to_view()

    def fit_to_view(self):
        if self._pixmap_item:
            self.resetTransform()
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._fit_scale = self.transform().m11()
            self._zoom = 1.0
            self.zoom_changed.emit(self._zoom)

    def set_zoom(self, factor: float):
        if self._pixmap_item:
            self.resetTransform()
            scale = self._fit_scale * factor
            self.scale(scale, scale)
            self._zoom = factor
            self.zoom_changed.emit(self._zoom)

    def wheelEvent(self, event: QWheelEvent):
        if self._pixmap_item:
            delta = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            new_zoom = self._zoom * delta
            new_zoom = max(0.1, min(new_zoom, 20.0))
            self.set_zoom(new_zoom)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.fit_to_view()

    def get_image_size(self) -> tuple[int, int]:
        if self._pixmap_item:
            pm = self._pixmap_item.pixmap()
            return pm.width(), pm.height()
        return 0, 0


# ── Thumbnail Card ──────────────────────────────────────────────


class ThumbnailCard(QFrame):
    """Clickable thumbnail with status badge."""

    clicked = Signal(int)

    def __init__(self, index: int, filename: str, parent=None):
        super().__init__(parent)
        self._index = index
        self._selected = False
        self.setFixedSize(150, 170)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(2)

        self._image_label = QLabel()
        self._image_label.setFixedSize(138, 130)
        self._image_label.setAlignment(Qt.AlignCenter)
        self._image_label.setStyleSheet(
            f"background: {Colors.BG_CARD2}; border-radius: 4px; border: none;"
        )

        self._status_label = QLabel("")
        self._status_label.setAlignment(Qt.AlignRight)
        self._status_label.setStyleSheet("font-size: 10px; border: none;")
        self._status_label.setFixedHeight(14)

        disp = filename[:18] + "..." if len(filename) > 18 else filename
        self._name_label = QLabel(disp)
        self._name_label.setAlignment(Qt.AlignCenter)
        self._name_label.setStyleSheet(
            f"font-size: 10px; color: {Colors.TEXT_DIM}; border: none;"
        )
        self._name_label.setToolTip(filename)

        layout.addWidget(self._status_label)
        layout.addWidget(self._image_label)
        layout.addWidget(self._name_label)

    def set_pixmap(self, pixmap: QPixmap):
        scaled = pixmap.scaled(
            138, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._image_label.setPixmap(scaled)

    def set_status(self, text: str, color: str = Colors.TEXT_DIM):
        self._status_label.setText(text)
        self._status_label.setStyleSheet(f"font-size: 10px; color: {color}; border: none;")

    def set_selected(self, selected: bool):
        self._selected = selected
        if selected:
            self.setStyleSheet(
                f"QFrame#card {{ border: 2px solid {Colors.ACCENT}; background-color: {Colors.BG_CARD}; border-radius: 8px; }}"
            )
        else:
            self.setStyleSheet(
                f"QFrame#card {{ border: 1px solid {Colors.BORDER}; background-color: {Colors.BG_CARD}; border-radius: 8px; }}"
            )

    def mousePressEvent(self, event):
        self.clicked.emit(self._index)
