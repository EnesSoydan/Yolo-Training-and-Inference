"""Background QThread workers for system monitoring, training, and inference."""

from __future__ import annotations

import os
import subprocess
import sys

import psutil
from PySide6.QtCore import QThread, Signal

from utils import YoloLogParser, get_gpu_stats


class SystemMonitorWorker(QThread):
    """Periodically emits GPU and CPU/RAM statistics."""

    stats_ready = Signal(dict)

    def __init__(self, interval_ms: int = 2000):
        super().__init__()
        self._interval = interval_ms
        self._running = True

    def run(self):
        while self._running:
            data = {}
            gpu = get_gpu_stats()
            if gpu:
                data["gpu"] = gpu
            data["cpu_percent"] = psutil.cpu_percent(interval=0.3)
            mem = psutil.virtual_memory()
            data["ram_percent"] = mem.percent
            data["ram_used_gb"] = round(mem.used / (1024**3), 1)
            data["ram_total_gb"] = round(mem.total / (1024**3), 1)
            self.stats_ready.emit(data)
            self.msleep(self._interval)

    def stop(self):
        self._running = False
        self.wait(3000)


class TrainingWorker(QThread):
    """Runs a YOLO training script as a subprocess.

    Handles \\r (carriage return) from tqdm progress bars so each batch
    update replaces the previous line instead of appending a new one.
    """

    log_line = Signal(str)       # append new line (\n terminated)
    log_replace = Signal(str)    # replace last line (\r terminated – tqdm)
    finished_signal = Signal(int)

    def __init__(self, script_path: str):
        super().__init__()
        self._script = script_path
        self._process = None
        self._running = True

    def run(self):
        work_dir = os.path.dirname(os.path.abspath(self._script))
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000

        try:
            self._process = subprocess.Popen(
                [sys.executable, "-u", self._script],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=work_dir,
                **kwargs,
            )

            buf = bytearray()
            pending_cr = False  # True when we saw \r but haven't seen next byte yet

            while self._running:
                byte = self._process.stdout.read(1)
                if not byte:
                    # EOF – flush pending
                    if pending_cr:
                        line = buf.decode("utf-8", errors="replace")
                        if line.strip():
                            self.log_replace.emit(line)
                        buf.clear()
                    break

                b = byte[0]

                if pending_cr:
                    pending_cr = False
                    if b == 0x0A:
                        # \r\n  →  true newline (Windows line ending)
                        line = buf.decode("utf-8", errors="replace")
                        if line.strip():
                            self.log_line.emit(line)
                        buf.clear()
                        continue
                    else:
                        # \r alone  →  tqdm progress update (replace)
                        line = buf.decode("utf-8", errors="replace")
                        if line.strip():
                            self.log_replace.emit(line)
                        buf.clear()
                        # fall through to process current byte

                if b == 0x0A:
                    # \n alone  →  newline
                    line = buf.decode("utf-8", errors="replace")
                    if line.strip():
                        self.log_line.emit(line)
                    buf.clear()
                elif b == 0x0D:
                    # \r  →  wait to see if \n follows
                    pending_cr = True
                else:
                    buf.append(b)

            # flush remaining
            if buf:
                line = buf.decode("utf-8", errors="replace")
                if line.strip():
                    self.log_line.emit(line)

            self._process.stdout.close()
            rc = self._process.wait()
        except Exception as e:
            self.log_line.emit(f"[HATA] {e}")
            rc = -1
        self.finished_signal.emit(rc)

    def stop(self):
        self._running = False
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()


class InferenceWorker(QThread):
    """Runs detection on a folder of images.

    Supports two modes:
      - "normal"  → ultralytics model.predict()
      - "sahi"    → SAHI sliced prediction (better for small objects)
    """

    progress = Signal(int, int, str)       # index, total, filename
    result_ready = Signal(int, str, int)   # index, output_path, detection_count
    error = Signal(int, str)               # index, error message
    all_done = Signal(int)                 # total detections

    def __init__(
        self,
        model_path: str,
        image_paths: list[str],
        output_dir: str,
        confidence: float = 0.25,
        slice_size: int = 640,
        selected_classes: list[str] | None = None,
        mode: str = "normal",
    ):
        super().__init__()
        self._model_path = model_path
        self._image_paths = image_paths
        self._output_dir = output_dir
        self._confidence = confidence
        self._slice_size = slice_size
        self._selected_classes = selected_classes
        self._mode = mode
        self._running = True

    def run(self):
        os.makedirs(self._output_dir, exist_ok=True)
        if self._mode == "sahi":
            self._run_sahi()
        else:
            self._run_normal()

    # ── Normal Predict ──────────────────────────────────────────

    def _run_normal(self):
        import cv2

        total = len(self._image_paths)
        total_detections = 0

        try:
            from ultralytics import YOLO
            model = YOLO(self._model_path)
        except Exception as e:
            for i in range(total):
                self.error.emit(i, str(e))
            self.all_done.emit(0)
            return

        from theme import CLASS_COLORS

        for idx, img_path in enumerate(self._image_paths):
            if not self._running:
                break
            filename = os.path.basename(img_path)
            self.progress.emit(idx, total, filename)

            try:
                results = model.predict(
                    source=img_path,
                    conf=self._confidence,
                    device="cuda:0",
                    verbose=False,
                )
                image = cv2.imread(img_path)
                det_count = 0

                for result in results:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        if (
                            self._selected_classes is not None
                            and cls_name not in self._selected_classes
                        ):
                            continue

                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        color_hex = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                        color_bgr = tuple(
                            int(color_hex.lstrip("#")[i : i + 2], 16)
                            for i in (4, 2, 0)
                        )

                        cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 2)
                        label = f"{cls_name} {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        cv2.rectangle(
                            image,
                            (x1, y1 - th - 8),
                            (x1 + tw + 4, y1),
                            color_bgr,
                            -1,
                        )
                        cv2.putText(
                            image,
                            label,
                            (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        det_count += 1

                name, ext = os.path.splitext(filename)
                out_path = os.path.join(self._output_dir, f"{name}_detected{ext}")
                cv2.imwrite(out_path, image)
                total_detections += det_count
                self.result_ready.emit(idx, out_path, det_count)

            except Exception as e:
                self.error.emit(idx, str(e))

        self.all_done.emit(total_detections)

    # ── SAHI Sliced Predict ─────────────────────────────────────

    def _run_sahi(self):
        import cv2
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction

        total = len(self._image_paths)
        total_detections = 0

        try:
            detection_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=self._model_path,
                confidence_threshold=self._confidence,
                device="cuda:0",
            )
        except Exception as e:
            for i in range(total):
                self.error.emit(i, str(e))
            self.all_done.emit(0)
            return

        from theme import CLASS_COLORS

        for idx, img_path in enumerate(self._image_paths):
            if not self._running:
                break
            filename = os.path.basename(img_path)
            self.progress.emit(idx, total, filename)

            try:
                result = get_sliced_prediction(
                    img_path,
                    detection_model,
                    slice_height=self._slice_size,
                    slice_width=self._slice_size,
                    overlap_height_ratio=0.3,
                    overlap_width_ratio=0.3,
                    postprocess_type="NMS",
                    postprocess_match_threshold=0.7,
                    postprocess_class_agnostic=True,
                )

                image = cv2.imread(img_path)
                det_count = 0

                for pred in result.object_prediction_list:
                    cls_name = pred.category.name
                    if (
                        self._selected_classes is not None
                        and cls_name not in self._selected_classes
                    ):
                        continue

                    bbox = pred.bbox
                    x1, y1 = int(bbox.minx), int(bbox.miny)
                    x2, y2 = int(bbox.maxx), int(bbox.maxy)
                    conf = pred.score.value
                    cls_id = pred.category.id
                    color_hex = CLASS_COLORS[cls_id % len(CLASS_COLORS)]
                    color_bgr = tuple(
                        int(color_hex.lstrip("#")[i : i + 2], 16)
                        for i in (4, 2, 0)
                    )

                    cv2.rectangle(image, (x1, y1), (x2, y2), color_bgr, 2)
                    label = f"{cls_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        image,
                        (x1, y1 - th - 8),
                        (x1 + tw + 4, y1),
                        color_bgr,
                        -1,
                    )
                    cv2.putText(
                        image,
                        label,
                        (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    det_count += 1

                name, ext = os.path.splitext(filename)
                out_path = os.path.join(self._output_dir, f"{name}_detected{ext}")
                cv2.imwrite(out_path, image)
                total_detections += det_count
                self.result_ready.emit(idx, out_path, det_count)

            except Exception as e:
                self.error.emit(idx, str(e))

        self.all_done.emit(total_detections)

    def stop(self):
        self._running = False


class ThumbnailWorker(QThread):
    """Loads image thumbnails in the background."""

    thumbnail_ready = Signal(int, object)  # index, QPixmap
    all_done = Signal()

    def __init__(self, image_paths: list[str], size: int = 130):
        super().__init__()
        self._paths = image_paths
        self._size = size
        self._running = True

    def run(self):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QImage, QPixmap

        for i, path in enumerate(self._paths):
            if not self._running:
                break
            try:
                img = QImage(path)
                if not img.isNull():
                    scaled = img.scaled(
                        self._size,
                        self._size,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    pixmap = QPixmap.fromImage(scaled)
                    self.thumbnail_ready.emit(i, pixmap)
            except Exception:
                pass
            if i % 20 == 19:
                self.msleep(10)
        self.all_done.emit()

    def stop(self):
        self._running = False
