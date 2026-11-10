"""
Modern dark theme stylesheet and UI design tokens for VisionCut AI Cloud.
"""

DARK_THEME_QSS = """
QMainWindow, QWidget#centralWidget {
    background-color: #12141A;
    color: #E2E4EB;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

QFrame#sidebar {
    background-color: #181A22;
    border-right: 1px solid #232733;
}

QPushButton#navBtn {
    background-color: transparent;
    color: #9AA0B4;
    border: none;
    border-radius: 8px;
    padding: 12px 18px;
    text-align: left;
    font-size: 14px;
    font-weight: 500;
}

QPushButton#navBtn:hover {
    background-color: #202430;
    color: #FFFFFF;
}

QPushButton#navBtn:checked {
    background-color: #242A38;
    color: #00E5FF;
    font-weight: 600;
    border-left: 3px solid #00E5FF;
}

QFrame.card {
    background-color: #1A1D26;
    border: 1px solid #282C3A;
    border-radius: 10px;
    padding: 16px;
}

QLabel#titleLabel {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
}

QLabel#subtitleLabel {
    font-size: 13px;
    color: #8E94A8;
}

QLabel#statValue {
    font-size: 24px;
    font-weight: 700;
    color: #00E5FF;
}

QLabel#statTitle {
    font-size: 12px;
    color: #8E94A8;
    text-transform: uppercase;
    font-weight: 600;
}

QPushButton.primaryBtn {
    background-color: #00B0FF;
    color: #000000;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 700;
    font-size: 13px;
}

QPushButton.primaryBtn:hover {
    background-color: #40C4FF;
}

QPushButton.primaryBtn:disabled {
    background-color: #2A3040;
    color: #606880;
}

QPushButton.secondaryBtn {
    background-color: #232734;
    color: #E2E4EB;
    border: 1px solid #32384A;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
}

QPushButton.secondaryBtn:hover {
    background-color: #2E3345;
    border-color: #424A60;
}

QPushButton.successBtn {
    background-color: #00E676;
    color: #000000;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 700;
}

QPushButton.successBtn:hover {
    background-color: #69F0AE;
}

QTableWidget {
    background-color: #161820;
    border: 1px solid #262A38;
    border-radius: 8px;
    gridline-color: #222634;
    color: #E2E4EB;
    selection-background-color: #252D3F;
}

QHeaderView::section {
    background-color: #1E212D;
    color: #8E94A8;
    padding: 8px;
    border: none;
    font-weight: 600;
    font-size: 12px;
}

QProgressBar {
    background-color: #1F222E;
    border: 1px solid #2D3344;
    border-radius: 6px;
    text-align: center;
    color: #FFFFFF;
    font-weight: 600;
}

QProgressBar::chunk {
    background-color: #00E5FF;
    border-radius: 5px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #262B3A;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #00E5FF;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #FFFFFF;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QComboBox, QLineEdit, QSpinBox {
    background-color: #1C1F2A;
    border: 1px solid #2C3243;
    border-radius: 6px;
    padding: 8px 12px;
    color: #FFFFFF;
}

QComboBox:focus, QLineEdit:focus {
    border-color: #00E5FF;
}
"""

