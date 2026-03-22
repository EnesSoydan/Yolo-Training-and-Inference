"""YOLO Training Monitor & Detection Tester – PySide6 desktop application."""

import sys
import os

# Ensure the script directory is in the path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

from theme import get_stylesheet
from training_tab import TrainingTab
from detection_tab import DetectionTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Monitor v3.0")
        self.setMinimumSize(1200, 750)
        self.resize(1440, 900)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)

        self._training_tab = TrainingTab()
        self._detection_tab = DetectionTab()

        self._tabs.addTab(self._training_tab, "  Eğitim  ")
        self._tabs.addTab(self._detection_tab, "  Tespit  ")

        self.setCentralWidget(self._tabs)

        # Focus policy for keyboard shortcuts in detection tab
        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int):
        if index == 1:
            self._detection_tab.setFocus()

    def closeEvent(self, event):
        self._training_tab.cleanup()
        self._detection_tab.cleanup()
        event.accept()


def main():
    # High DPI support
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(get_stylesheet())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
