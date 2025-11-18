"""
Advanced Qt theme for Alita
Provides modern, sci-fi styling
"""

from PyQt6.QtGui import QPalette, QColor, QFont
from PyQt6.QtCore import Qt

MAIN_THEME = """
/* Global Styles */
QMainWindow, QWidget {
    background-color: #0B0F1A;
    color: #E6EDF3;
    font-family: 'Segoe UI', Arial, sans-serif;
}

/* Futuristic Cards */
#featureCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
        stop:0 rgba(25, 30, 45, 0.95), 
        stop:1 rgba(35, 40, 60, 0.95));
    border: 1px solid rgba(100, 130, 255, 0.3);
    border-radius: 12px;
    padding: 20px;
}

#featureCard:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 rgba(35, 40, 60, 0.95),
        stop:1 rgba(45, 50, 75, 0.95));
    border: 1px solid rgba(100, 130, 255, 0.6);
}

/* Chat Interface */
QTextEdit {
    background-color: rgba(20, 25, 40, 0.7);
    border: 1px solid rgba(100, 130, 255, 0.2);
    border-radius: 8px;
    padding: 10px;
    selection-background-color: rgba(100, 130, 255, 0.3);
}

/* Buttons */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #2B3245,
        stop:1 #1B2235);
    border: 1px solid rgba(100, 130, 255, 0.3);
    border-radius: 6px;
    padding: 8px 16px;
    color: #E6EDF3;
    font-weight: bold;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #3B4255,
        stop:1 #2B3245);
    border: 1px solid rgba(100, 130, 255, 0.6);
}

QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #1B2235,
        stop:1 #2B3245);
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: rgba(20, 25, 40, 0.7);
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: rgba(100, 130, 255, 0.3);
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(100, 130, 255, 0.6);
}

/* Status Indicators */
#statusIndicator {
    border: 1px solid rgba(100, 130, 255, 0.3);
    border-radius: 4px;
    padding: 4px 8px;
    background: rgba(20, 25, 40, 0.7);
}

/* Graph Widgets */
#graphWidget {
    background: rgba(20, 25, 40, 0.7);
    border: 1px solid rgba(100, 130, 255, 0.2);
    border-radius: 8px;
}

/* Vision Feed */
#visionFeed {
    background: rgba(20, 25, 40, 0.7);
    border: 1px solid rgba(100, 130, 255, 0.2);
    border-radius: 12px;
    padding: 2px;
}

/* Tooltips */
QToolTip {
    background-color: #1B2235;
    color: #E6EDF3;
    border: 1px solid rgba(100, 130, 255, 0.3);
    border-radius: 4px;
    padding: 4px;
}

/* Progress Bars */
QProgressBar {
    border: 1px solid rgba(100, 130, 255, 0.2);
    border-radius: 4px;
    text-align: center;
    background: rgba(20, 25, 40, 0.7);
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(100, 130, 255, 0.6),
        stop:1 rgba(130, 160, 255, 0.6));
    border-radius: 3px;
}

/* Menu and Toolbar */
QMenuBar {
    background-color: #0B0F1A;
    border-bottom: 1px solid rgba(100, 130, 255, 0.2);
}

QMenuBar::item:selected {
    background: rgba(100, 130, 255, 0.2);
}

QMenu {
    background-color: #1B2235;
    border: 1px solid rgba(100, 130, 255, 0.3);
    border-radius: 4px;
}

QMenu::item:selected {
    background: rgba(100, 130, 255, 0.2);
}
"""

def apply_theme(app):
    """Apply the futuristic theme to the application"""
    # Set global stylesheet
    app.setStyleSheet(MAIN_THEME)
    
    # Set dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(11, 15, 26))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(230, 237, 243))
    palette.setColor(QPalette.ColorRole.Base, QColor(27, 34, 53))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(35, 40, 60))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(27, 34, 53))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(230, 237, 243))
    palette.setColor(QPalette.ColorRole.Text, QColor(230, 237, 243))
    palette.setColor(QPalette.ColorRole.Button, QColor(43, 50, 69))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(230, 237, 243))
    palette.setColor(QPalette.ColorRole.Link, QColor(100, 130, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(100, 130, 255, 100))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(230, 237, 243))
    
    app.setPalette(palette)
    
    # Set default font
    font = QFont("Segoe UI", 9)
    app.setFont(font)