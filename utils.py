"""GPU statistics and YOLO log parsing utilities."""

from __future__ import annotations

import re
import subprocess
import sys
from collections import deque

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (e.g. \\x1b[K from tqdm)."""
    return _ANSI_RE.sub("", text)


def get_gpu_stats() -> dict | None:
    """Query nvidia-smi for GPU statistics. Returns dict or None on failure."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=temperature.gpu,utilization.gpu,memory.used,"
            "memory.total,power.draw,power.limit,name,fan.speed",
            "--format=csv,noheader,nounits",
        ]
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5, **kwargs
        )
        if result.returncode == 0:
            values = [v.strip() for v in result.stdout.strip().split(",")]
            if len(values) >= 8:
                na = ("[N/A]", "[Not Supported]", "N/A", "")

                def safe_int(v, default=0):
                    return int(v) if v not in na else default

                def safe_float(v, default=0.0):
                    return float(v) if v not in na else default

                return {
                    "temperature": safe_int(values[0]),
                    "utilization": safe_int(values[1]),
                    "memory_used": safe_int(values[2]),
                    "memory_total": safe_int(values[3]),
                    "power_draw": safe_float(values[4]),
                    "power_limit": safe_float(values[5]),
                    "name": values[6],
                    "fan_speed": safe_int(values[7]),
                }
    except Exception:
        pass
    return None


class YoloLogParser:
    """Parse YOLO training output and track metrics."""

    EPOCH_PATTERN = re.compile(
        r"\s*(\d+)/(\d+)\s+"  # epoch / total
        r".*?(\d+\.?\d*)\s*[gGmM]\s+"  # GPU mem
        r"(\d+\.?\d*)\s+"  # box_loss
        r"(\d+\.?\d*)\s+"  # cls_loss
        r"(\d+\.?\d*)\s+"  # dfl_loss
        r"(\d+)\s+"  # instances
        r"(\d+)"  # img_size
    )

    VAL_PATTERN = re.compile(
        r".*all\s+\d+\s+\d+\s+"
        r"(\d+\.?\d*)\s+"  # precision
        r"(\d+\.?\d*)\s+"  # recall
        r"(\d+\.?\d*)\s+"  # mAP50
        r"(\d+\.?\d*)"  # mAP50-95
    )

    def __init__(self, max_points: int = 300):
        self.max_points = max_points
        self.reset()

    def reset(self):
        mk = lambda: deque(maxlen=self.max_points)
        self.epochs_list = mk()
        self.val_epochs = mk()
        self.box_losses = mk()
        self.cls_losses = mk()
        self.dfl_losses = mk()
        self.precisions = mk()
        self.recalls = mk()
        self.map50s = mk()
        self.map50_95s = mk()

        self.current_epoch = 0
        self.total_epochs = 0

        # Best metrics
        self.best_map50 = 0.0
        self.best_map50_epoch = 0
        self.best_map50_95 = 0.0
        self.best_map50_95_epoch = 0
        self.best_precision = 0.0
        self.best_precision_epoch = 0
        self.best_recall = 0.0
        self.best_recall_epoch = 0

    def parse_line(self, line: str) -> dict | None:
        """Parse a YOLO output line. Returns info dict or None."""
        m = self.EPOCH_PATTERN.search(line)
        if m:
            epoch, total = int(m.group(1)), int(m.group(2))
            box_loss = float(m.group(4))
            cls_loss = float(m.group(5))
            dfl_loss = float(m.group(6))

            self.current_epoch = epoch
            self.total_epochs = total
            self.epochs_list.append(epoch)
            self.box_losses.append(box_loss)
            self.cls_losses.append(cls_loss)
            self.dfl_losses.append(dfl_loss)

            return {
                "type": "epoch",
                "epoch": epoch,
                "total": total,
                "box_loss": box_loss,
                "cls_loss": cls_loss,
                "dfl_loss": dfl_loss,
            }

        m = self.VAL_PATTERN.search(line)
        if m:
            precision = float(m.group(1))
            recall = float(m.group(2))
            map50 = float(m.group(3))
            map50_95 = float(m.group(4))

            self.val_epochs.append(self.current_epoch)
            self.precisions.append(precision)
            self.recalls.append(recall)
            self.map50s.append(map50)
            self.map50_95s.append(map50_95)

            if map50 > self.best_map50:
                self.best_map50 = map50
                self.best_map50_epoch = self.current_epoch
            if map50_95 > self.best_map50_95:
                self.best_map50_95 = map50_95
                self.best_map50_95_epoch = self.current_epoch
            if precision > self.best_precision:
                self.best_precision = precision
                self.best_precision_epoch = self.current_epoch
            if recall > self.best_recall:
                self.best_recall = recall
                self.best_recall_epoch = self.current_epoch

            return {
                "type": "validation",
                "precision": precision,
                "recall": recall,
                "map50": map50,
                "map50_95": map50_95,
            }

        return None
