"""Dark theme colors and QSS stylesheet for the YOLO Monitor application."""


class Colors:
    BG_DARK = "#0d1117"
    BG_CARD = "#161b22"
    BG_CARD2 = "#1c2333"
    BG_INPUT = "#0d1117"
    BORDER = "#30363d"
    TEXT = "#e6edf3"
    TEXT_DIM = "#8b949e"
    ACCENT = "#58a6ff"
    GREEN = "#3fb950"
    YELLOW = "#d29922"
    ORANGE = "#f0883e"
    RED = "#f85149"
    PURPLE = "#bc8cff"
    CYAN = "#39d2c0"


CLASS_COLORS = [
    "#3fb950", "#58a6ff", "#bc8cff", "#f0883e", "#39d2c0",
    "#d29922", "#f85149", "#ff7eb6", "#7ee8fa", "#b5e853",
]


def get_stylesheet() -> str:
    C = Colors
    return f"""
    /* ── Global ── */
    QMainWindow, QWidget {{
        background-color: {C.BG_DARK};
        color: {C.TEXT};
        font-family: "Segoe UI", "Arial", sans-serif;
        font-size: 13px;
    }}

    /* ── Tab Widget ── */
    QTabWidget::pane {{
        border: 1px solid {C.BORDER};
        border-radius: 6px;
        background: {C.BG_DARK};
    }}
    QTabBar::tab {{
        background: {C.BG_CARD};
        color: {C.TEXT_DIM};
        border: 1px solid {C.BORDER};
        border-bottom: none;
        padding: 8px 20px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
        font-weight: bold;
    }}
    QTabBar::tab:selected {{
        background: {C.BG_DARK};
        color: {C.ACCENT};
        border-bottom: 2px solid {C.ACCENT};
    }}
    QTabBar::tab:hover:!selected {{
        background: {C.BG_CARD2};
        color: {C.TEXT};
    }}

    /* ── Buttons ── */
    QPushButton {{
        background-color: {C.BG_CARD2};
        color: {C.TEXT};
        border: 1px solid {C.BORDER};
        border-radius: 6px;
        padding: 6px 14px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background-color: {C.BORDER};
        border-color: {C.TEXT_DIM};
    }}
    QPushButton:pressed {{
        background-color: {C.BG_CARD};
    }}
    QPushButton:disabled {{
        color: {C.TEXT_DIM};
        background-color: {C.BG_CARD};
        border-color: {C.BORDER};
    }}
    QPushButton#startBtn {{
        background-color: #238636;
        border-color: #2ea043;
        color: white;
    }}
    QPushButton#startBtn:hover {{
        background-color: #2ea043;
    }}
    QPushButton#stopBtn {{
        background-color: #b62324;
        border-color: #f85149;
        color: white;
    }}
    QPushButton#stopBtn:hover {{
        background-color: #da3633;
    }}

    /* ── Inputs ── */
    QLineEdit {{
        background-color: {C.BG_INPUT};
        color: {C.TEXT};
        border: 1px solid {C.BORDER};
        border-radius: 6px;
        padding: 6px 10px;
        selection-background-color: {C.ACCENT};
    }}
    QLineEdit:focus {{
        border-color: {C.ACCENT};
    }}
    QSpinBox, QDoubleSpinBox {{
        background-color: {C.BG_INPUT};
        color: {C.TEXT};
        border: 1px solid {C.BORDER};
        border-radius: 6px;
        padding: 4px 8px;
    }}

    /* ── Progress Bar ── */
    QProgressBar {{
        background-color: {C.BG_CARD};
        border: 1px solid {C.BORDER};
        border-radius: 8px;
        text-align: center;
        color: {C.TEXT};
        font-weight: bold;
        height: 22px;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 {C.ACCENT}, stop:1 {C.PURPLE});
        border-radius: 7px;
    }}

    /* ── Scroll Area ── */
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: {C.BG_CARD};
        width: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical {{
        background: {C.BORDER};
        border-radius: 4px;
        min-height: 30px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {C.TEXT_DIM};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: {C.BG_CARD};
        height: 8px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal {{
        background: {C.BORDER};
        border-radius: 4px;
        min-width: 30px;
    }}

    /* ── Plain Text (Terminal Log) ── */
    QPlainTextEdit {{
        background-color: {C.BG_CARD};
        color: {C.TEXT};
        border: 1px solid {C.BORDER};
        border-radius: 6px;
        font-family: "Cascadia Code", "Consolas", monospace;
        font-size: 12px;
        padding: 8px;
        selection-background-color: {C.ACCENT};
    }}

    /* ── Frames / Cards ── */
    QFrame#card {{
        background-color: {C.BG_CARD};
        border: 1px solid {C.BORDER};
        border-radius: 8px;
    }}
    QFrame#sidebar {{
        background-color: {C.BG_CARD};
        border: 1px solid {C.BORDER};
        border-radius: 8px;
    }}

    /* ── Labels ── */
    QLabel {{
        color: {C.TEXT};
        background: transparent;
    }}
    QLabel#dimLabel {{
        color: {C.TEXT_DIM};
        font-size: 11px;
    }}
    QLabel#accentLabel {{
        color: {C.ACCENT};
        font-weight: bold;
    }}
    QLabel#headerLabel {{
        color: {C.TEXT};
        font-size: 15px;
        font-weight: bold;
    }}
    QLabel#statusRunning {{
        color: {C.GREEN};
        font-weight: bold;
    }}
    QLabel#statusStopped {{
        color: {C.RED};
        font-weight: bold;
    }}
    QLabel#statusWaiting {{
        color: {C.YELLOW};
        font-weight: bold;
    }}

    /* ── Splitter ── */
    QSplitter::handle {{
        background: {C.BORDER};
    }}
    QSplitter::handle:horizontal {{
        width: 2px;
    }}
    QSplitter::handle:vertical {{
        height: 2px;
    }}

    /* ── CheckBox ── */
    QCheckBox {{
        color: {C.TEXT};
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {C.BORDER};
        border-radius: 3px;
        background: {C.BG_INPUT};
    }}
    QCheckBox::indicator:checked {{
        background: {C.ACCENT};
        border-color: {C.ACCENT};
    }}

    /* ── Tooltips ── */
    QToolTip {{
        background-color: {C.BG_CARD2};
        color: {C.TEXT};
        border: 1px solid {C.BORDER};
        border-radius: 4px;
        padding: 4px 8px;
    }}

    /* ── Group Box ── */
    QGroupBox {{
        border: 1px solid {C.BORDER};
        border-radius: 6px;
        margin-top: 8px;
        padding-top: 16px;
        font-weight: bold;
        color: {C.TEXT_DIM};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
    }}
    """
