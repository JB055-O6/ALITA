"""
Custom Qt widgets for Alita's holographic interface
Provides advanced, animated UI components with futuristic styling
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QPushButton, QProgressBar, QTextEdit,
    QGraphicsOpacityEffect, QVBoxLayout, QHBoxLayout
)
from PyQt6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, pyqtProperty,
    QParallelAnimationGroup, QSequentialAnimationGroup,
    QTimer, QPoint, QSize, QRect
)
from PyQt6.QtGui import (
    QPainter, QColor, QLinearGradient, QPainterPath,
    QPen, QFont, QFontMetrics, QBrush, QPalette
)


class GlowEffect(QGraphicsOpacityEffect):
    """Creates a pulsing glow effect"""
    
    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self.color = color
        self._setup_animation()
    
    def _setup_animation(self):
        self.animation = QPropertyAnimation(self, b"opacity")
        self.animation.setDuration(1500)
        self.animation.setLoopCount(-1)
        self.animation.setStartValue(0.5)
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.animation.start()


class PulseButton(QPushButton):
    """Button with pulsing animation effect"""
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self._setup_effects()
    
    def _setup_effects(self):
        # Add glow effect
        glow = GlowEffect(QColor(100, 130, 255), self)
        self.setGraphicsEffect(glow)
        
        # Size animation
        self.size_anim = QPropertyAnimation(self, b"size")
        self.size_anim.setDuration(200)
        self.size_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
    
    def enterEvent(self, event):
        """Handle mouse enter"""
        size = self.size()
        self.size_anim.setStartValue(size)
        self.size_anim.setEndValue(QSize(int(size.width() * 1.05), int(size.height() * 1.05)))
        self.size_anim.start()
    
    def leaveEvent(self, event):
        """Handle mouse leave"""
        size = self.size()
        self.size_anim.setStartValue(size)
        self.size_anim.setEndValue(self.sizeHint())
        self.size_anim.start()


class CircularProgressBar(QWidget):
    """Circular progress indicator with glow"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(100, 100)
        self._progress = 0
        self._rotation = 0
        self._setup_animations()
    
    def _setup_animations(self):
        self.rotation_anim = QPropertyAnimation(self, b"rotation")
        self.rotation_anim.setDuration(2000)
        self.rotation_anim.setLoopCount(-1)
        self.rotation_anim.setStartValue(0)
        self.rotation_anim.setEndValue(360)
        self.rotation_anim.start()
    
    @pyqtProperty(float)
    def progress(self):
        return self._progress
    
    @progress.setter
    def progress(self, value):
        self._progress = value
        self.update()
    
    @pyqtProperty(float)
    def rotation(self):
        return self._rotation
    
    @rotation.setter
    def rotation(self, value):
        self._rotation = value
        self.update()
    
    def paintEvent(self, event):
        size = min(self.width(), self.height())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background
        pen = QPen()
        pen.setWidth(8)
        pen.setColor(QColor(40, 44, 52))
        painter.setPen(pen)
        painter.drawArc(4, 4, size-8, size-8, 0, 360*16)
        
        # Draw progress
        pen.setColor(QColor(0, 255, 255))
        painter.setPen(pen)
        painter.drawArc(4, 4, size-8, size-8, int(self._rotation * 16), int(-self.progress * 360 * 16))


class FloatingCard(QFrame):
    """Card widget with floating animation"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("floatingCard")
        self._float_offset = 0
        self._setup_animations()
    
    def _setup_animations(self):
        self.float_timer = QTimer()
        self.float_timer.timeout.connect(self._update_float)
        self.float_timer.start(50)
        self.float_phase = 0
    
    def _update_float(self):
        self.float_phase += 0.1
        self._float_offset = int(5 * np.sin(self.float_phase))
        self.update()


class WaveformVisualizer(QWidget):
    """Real-time waveform visualization with holographic effects"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 100)
        self.setMaximumHeight(120)
        self.audio_data = np.zeros(1024)
        self.animation_phase = 0
        
        # Holographic colors
        self.colors = [
            QColor(0, 255, 255, 200),    # Cyan
            QColor(255, 0, 255, 200),    # Magenta
            QColor(0, 255, 0, 200),      # Green
            QColor(255, 255, 0, 200),    # Yellow
            QColor(255, 100, 0, 200),    # Orange
        ]
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._animate)
        self.timer.start(50)
    
    def update_data(self, data):
        """Update waveform data"""
        if len(data) > 0:
            self.audio_data = data[:1024]
        self.update()
    
    def _animate(self):
        """Animate waveform"""
        self.animation_phase += 1
        # Simulate audio activity
        if np.random.random() > 0.7:
            self.audio_data = np.random.randn(1024) * 0.2
        else:
            self.audio_data *= 0.9
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw holographic grid background
        self._draw_grid(painter)
        
        # Draw waveform
        if len(self.audio_data) > 0:
            self._draw_waveform(painter)
    
    def _draw_grid(self, painter):
        """Draw animated holographic grid"""
        painter.save()
        
        grid_size = 20
        alpha = int(30 + 20 * np.sin(self.animation_phase * 0.1))
        pen = QPen(QColor(0, 255, 255, alpha), 1)
        painter.setPen(pen)
        
        # Vertical lines
        for x in range(0, self.width(), grid_size):
            offset = int(3 * np.sin(self.animation_phase * 0.05 + x * 0.01))
            painter.drawLine(x, 0, x, self.height() + offset)
        
        # Horizontal lines
        for y in range(0, self.height(), grid_size):
            offset = int(3 * np.cos(self.animation_phase * 0.05 + y * 0.01))
            painter.drawLine(0, y, self.width() + offset, y)
        
        painter.restore()
    
    def _draw_waveform(self, painter):
        """Draw animated waveform"""
        painter.save()
        
        # Create gradient
        gradient = QLinearGradient(0, 0, self.width(), 0)
        for i, color in enumerate(self.colors):
            gradient.setColorAt(i / len(self.colors), color)
        
        # Draw waveform path
        path = QPainterPath()
        dx = self.width() / len(self.audio_data)
        center_y = self.height() / 2
        
        path.moveTo(0, center_y)
        
        for i, value in enumerate(self.audio_data):
            x = i * dx
            y = center_y + value * self.height() * 0.4
            path.lineTo(x, y)
        
        # Draw filled waveform
        pen = QPen(gradient, 2)
        painter.setPen(pen)
        painter.drawPath(path)
        
        # Draw glow effect
        pen.setWidth(4)
        color = QColor(0, 255, 255, 100)
        pen.setColor(color)
        painter.setPen(pen)
        painter.drawPath(path)
        
        painter.restore()


class EnergyRing(QWidget):
    """Circular energy indicator with particle effects"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self._energy = 0.5
        self._rotation = 0
        self._setup_animations()
    
    def _setup_animations(self):
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_animation)
        self.update_timer.start(16)  # ~60 FPS
    
    @pyqtProperty(float)
    def energy(self):
        return self._energy
    
    @energy.setter
    def energy(self, value):
        self._energy = max(0.0, min(1.0, value))
        self.update()
    
    def _update_animation(self):
        self._rotation += 2
        if self._rotation >= 360:
            self._rotation = 0
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center = self.rect().center()
        radius = min(self.width(), self.height()) / 2 - 15
        
        # Draw outer glow
        for i in range(3):
            alpha = int(50 - i * 15)
            color = QColor(0, 255, 255, alpha)
            pen = QPen(color, 2 + i * 2)
            painter.setPen(pen)
            painter.drawEllipse(center, int(radius + i * 5), int(radius + i * 5))
        
        # Draw main ring
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(0, 255, 255))
        gradient.setColorAt(0.5, QColor(255, 0, 255))
        gradient.setColorAt(1, QColor(0, 255, 255))
        
        pen = QPen(gradient, 8)
        painter.setPen(pen)
        
        # Draw energy arc
        start_angle = self._rotation * 16
        span_angle = int(self._energy * 360 * 16)
        painter.drawArc(
            int(center.x() - radius), int(center.y() - radius),
            int(radius * 2), int(radius * 2),
            start_angle, span_angle
        )


class HolographicButton(QPushButton):
    """Button with holographic styling and effects"""
    
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(40)
        self._glow_intensity = 0.5
        
        # Enhanced holographic styling
        self.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 255, 255, 100),
                    stop:0.5 rgba(255, 0, 255, 120),
                    stop:1 rgba(0, 255, 255, 100));
                border: 2px solid rgba(0, 255, 255, 200);
                border-radius: 8px;
                color: white;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 255, 255, 150),
                    stop:0.5 rgba(255, 0, 255, 170),
                    stop:1 rgba(0, 255, 255, 150));
                border: 2px solid rgba(255, 255, 255, 255);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 0, 255, 200),
                    stop:0.5 rgba(0, 255, 255, 220),
                    stop:1 rgba(255, 0, 255, 200));
            }
        """)
        
        # Setup glow animation
        self.glow_timer = QTimer()
        self.glow_timer.timeout.connect(self._update_glow)
        self.glow_timer.start(50)
        self.glow_phase = 0
    
    def _update_glow(self):
        """Update glow effect"""
        self.glow_phase += 0.1
        self._glow_intensity = 0.5 + 0.3 * np.sin(self.glow_phase)
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Add glow effect
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Outer glow
        glow_color = QColor(0, 255, 255, int(80 * self._glow_intensity))
        painter.setBrush(QBrush(glow_color))
        painter.setPen(Qt.PenStyle.NoPen)
        
        rect = self.rect().adjusted(-3, -3, 3, 3)
        painter.drawRoundedRect(rect, 10, 10)


class HolographicProgressBar(QProgressBar):
    """Progress bar with holographic styling and animations"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(25)
        self.setRange(0, 100)
        self.setValue(0)
        
        # Animation for progress bar
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._animate)
        self.animation_timer.start(100)
        self.animation_phase = 0
        
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid rgba(0, 255, 255, 150);
                border-radius: 8px;
                background: rgba(0, 0, 0, 100);
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 255, 255, 200),
                    stop:0.5 rgba(255, 0, 255, 220),
                    stop:1 rgba(0, 255, 0, 200));
                border-radius: 6px;
            }
        """)
    
    def _animate(self):
        """Animate progress bar effects"""
        self.animation_phase += 1
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Add animated glow effect
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Pulsing glow
        glow_alpha = int(50 + 30 * np.sin(self.animation_phase * 0.2))
        glow_color = QColor(0, 255, 255, glow_alpha)
        
        painter.setBrush(QBrush(glow_color))
        painter.setPen(Qt.PenStyle.NoPen)
        
        rect = self.rect().adjusted(-2, -2, 2, 2)
        painter.drawRoundedRect(rect, 10, 10)


class HolographicPanel(QFrame):
    """Panel with holographic styling and effects"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._animate)
        self.animation_timer.start(50)
        self.animation_phase = 0
        
        self.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(10, 20, 40, 200),
                    stop:0.5 rgba(20, 10, 50, 220),
                    stop:1 rgba(10, 30, 60, 200));
                border: 2px solid rgba(0, 255, 255, 180);
                border-radius: 12px;
            }
        """)
    
    def _animate(self):
        """Animate panel effects"""
        self.animation_phase += 1
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw animated border glow
        glow_alpha = int(100 + 50 * np.sin(self.animation_phase * 0.1))
        glow_color = QColor(0, 255, 255, glow_alpha)
        
        pen = QPen(glow_color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 10, 10)
        
        # Draw corner accents
        accent_color = QColor(255, 0, 255, glow_alpha)
        painter.setBrush(QBrush(accent_color))
        painter.setPen(Qt.PenStyle.NoPen)
        
        # Corner markers
        corner_size = 8
        corners = [
            (2, 2),  # Top-left
            (self.width() - corner_size - 2, 2),  # Top-right
            (2, self.height() - corner_size - 2),  # Bottom-left
            (self.width() - corner_size - 2, self.height() - corner_size - 2)  # Bottom-right
        ]
        
        for x, y in corners:
            painter.drawRect(x, y, corner_size, corner_size)


class HolographicTextEdit(QTextEdit):
    """Text editor with holographic styling"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 20, 40, 200);
                border: 2px solid rgba(0, 255, 255, 150);
                border-radius: 8px;
                color: rgba(255, 255, 255, 230);
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                selection-background-color: rgba(0, 255, 255, 100);
                padding: 10px;
            }
            QTextEdit:focus {
                border: 2px solid rgba(255, 255, 255, 200);
                background: rgba(0, 30, 60, 220);
            }
        """)


class StatusIndicator(QWidget):
    """Animated status indicator with holographic effects"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(20, 20)
        
        self.status = "idle"  # idle, active, error, success, warning
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update)
        self.animation_timer.start(100)
        self.animation_phase = 0
        
        self.status_colors = {
            "idle": QColor(100, 100, 100, 200),
            "active": QColor(0, 255, 255, 255),
            "error": QColor(255, 50, 50, 255),
            "success": QColor(0, 255, 0, 255),
            "warning": QColor(255, 255, 0, 255)
        }
    
    def set_status(self, status):
        """Set status indicator state"""
        self.status = status
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        self.animation_phase += 1
        
        # Get status color
        base_color = self.status_colors.get(self.status, self.status_colors["idle"])
        
        # Animate for active status
        if self.status == "active":
            pulse = 0.7 + 0.3 * np.sin(self.animation_phase * 0.3)
            color = QColor(base_color)
            color.setAlphaF(pulse)
        else:
            color = base_color
        
        # Draw indicator
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        
        center = self.rect().center()
        radius = 7
        painter.drawEllipse(center, radius, radius)
        
        # Draw glow effect
        if self.status in ["active", "error", "success"]:
            glow_color = QColor(color)
            glow_color.setAlpha(80)
            painter.setBrush(QBrush(glow_color))
            painter.drawEllipse(center, radius + 3, radius + 3)
