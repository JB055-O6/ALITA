"""
Alita GUI Interface - Advanced Holographic-Style Interface
Modern Qt-based interface with futuristic animations and effects
"""

import os
import sys
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, 
        QHBoxLayout, QPushButton, QLabel, QTextEdit,
        QStackedWidget, QFrame, QScrollArea, QGraphicsDropShadowEffect
    )
    from PyQt6.QtCore import (
        Qt, QTimer, pyqtSignal, QSize, QPropertyAnimation,
        QEasingCurve, QParallelAnimationGroup, QPoint, pyqtProperty
    )
    from PyQt6.QtGui import (
        QIcon, QFont, QPixmap, QColor, QPainter,
        QLinearGradient, QPainterPath, QBrush, QPen
    )
    from qasync import QEventLoop
except ImportError:
    print("Error: Required PyQt6 packages not found.")
    print("Install with: pip install PyQt6 PyQt6-WebEngine qasync")
    sys.exit(1)

# Custom widgets and theme
from .theme import apply_theme
from .widgets import (
    PulseButton, CircularProgressBar, FloatingCard,
    WaveformVisualizer, EnergyRing, HolographicButton,
    HolographicProgressBar, HolographicPanel, HolographicTextEdit,
    StatusIndicator
)

# Core functionality
try:
    from alita.core.voice import VoiceInterface
    from alita.core.vision import VisionSystem
    from alita.core.system_controller import SystemController
except ImportError:
    VoiceInterface = None
    VisionSystem = None
    SystemController = None


class FeatureCard(FloatingCard):
    """Interactive card for a major feature with holographic styling"""
    
    clicked = pyqtSignal()
    
    def __init__(self, title: str, description: str, icon_path: str):
        super().__init__()
        self.setObjectName("featureCard")
        self.setMinimumHeight(180)
        self.setMaximumWidth(250)
        
        # Enhanced holographic styling
        self.setStyleSheet("""
            #featureCard {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(15, 25, 45, 0.95),
                    stop: 0.5 rgba(25, 15, 55, 0.95),
                    stop: 1 rgba(15, 35, 65, 0.95));
                border-radius: 15px;
                border: 2px solid rgba(0, 255, 255, 0.3);
            }
            #featureCard:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(20, 30, 50, 0.98),
                    stop: 0.5 rgba(30, 20, 60, 0.98),
                    stop: 1 rgba(20, 40, 70, 0.98));
                border: 2px solid rgba(0, 255, 255, 0.6);
            }
        """)
        
        # Create animated glow effect
        glow = QGraphicsDropShadowEffect(self)
        glow.setBlurRadius(25)
        glow.setXOffset(0)
        glow.setYOffset(0)
        glow.setColor(QColor(0, 255, 255, 180))
        self.setGraphicsEffect(glow)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Icon with holographic effect
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if Path(icon_path).exists():
            icon = QPixmap(icon_path).scaled(
                64, 64, 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            icon_label.setPixmap(icon)
        else:
            icon_label.setText("🔷")
            icon_label.setFont(QFont("Segoe UI", 48))
        layout.addWidget(icon_label)
        
        # Animated title
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_label.setStyleSheet("""
            color: rgba(255, 255, 255, 0.95);
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.8);
        """)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_label.setFont(QFont("Segoe UI", 10))
        desc_label.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        layout.addWidget(desc_label)
        
        self.setLayout(layout)
        
        # Setup hover animations
        self.setupAnimations()
    
    def setupAnimations(self):
        """Setup hover and glow animations"""
        self.hover_anim = QPropertyAnimation(self.graphicsEffect(), b"color")
        self.hover_anim.setDuration(300)
        self.hover_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
    
    def enterEvent(self, event):
        """Handle mouse enter with animation"""
        self.hover_anim.setStartValue(QColor(0, 255, 255, 180))
        self.hover_anim.setEndValue(QColor(0, 255, 255, 255))
        self.hover_anim.start()
    
    def leaveEvent(self, event):
        """Handle mouse leave with animation"""
        self.hover_anim.setStartValue(QColor(0, 255, 255, 255))
        self.hover_anim.setEndValue(QColor(0, 255, 255, 180))
        self.hover_anim.start()
    
    def mousePressEvent(self, event):
        """Handle click event"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class ChatWidget(QWidget):
    """Modern chat interface with holographic styling"""
    
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)
        
        # Chat history panel
        chat_panel = HolographicPanel()
        chat_layout = QVBoxLayout(chat_panel)
        chat_layout.setContentsMargins(20, 20, 20, 20)
        
        self.chat_area = HolographicTextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setMinimumHeight(400)
        chat_layout.addWidget(self.chat_area)
        
        layout.addWidget(chat_panel)
        
        # Waveform visualizer
        self.waveform = WaveformVisualizer()
        layout.addWidget(self.waveform)
        
        # Input panel
        input_panel = HolographicPanel()
        input_layout = QHBoxLayout(input_panel)
        input_layout.setContentsMargins(20, 20, 20, 20)
        input_layout.setSpacing(15)
        
        self.input_field = HolographicTextEdit()
        self.input_field.setMaximumHeight(100)
        self.input_field.setPlaceholderText("Type your message here...")
        input_layout.addWidget(self.input_field)
        
        self.send_button = HolographicButton("Send")
        self.send_button.setMinimumWidth(120)
        input_layout.addWidget(self.send_button)
        
        layout.addWidget(input_panel)
        self.setLayout(layout)
        
        # Connect signals
        self.send_button.clicked.connect(self.send_message)
        self.input_field.textChanged.connect(self.on_text_changed)
        
        # Animation timer
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_animations)
        self.anim_timer.start(50)
    
    def on_text_changed(self):
        """Update waveform based on input"""
        text = self.input_field.toPlainText()
        # Simulate audio data from text
        if text:
            data = np.random.randn(1024) * 0.3
            self.waveform.update_data(data)
    
    def update_animations(self):
        """Update all animations"""
        self.waveform.update()
    
    def send_message(self):
        """Send message to Alita"""
        text = self.input_field.toPlainText().strip()
        if text:
            # Add user message with holographic styling
            self.chat_area.append(
                '<div style="margin: 10px; padding: 15px; '
                'background: rgba(0, 255, 255, 0.1); '
                'border-left: 3px solid rgba(0, 255, 255, 0.8); '
                'border-radius: 8px;">'
                f'<span style="color: rgba(0, 255, 255, 1.0); font-weight: bold;">You:</span> '
                f'<span style="color: rgba(255, 255, 255, 0.9);">{text}</span></div>'
            )
            self.input_field.clear()
            
            # Process message through brain directly
            try:
                # Import brain here to avoid circular imports
                from alita.core.brain import Brain
                from alita.config import AIConfig
                
                # Get or create brain instance
                if not hasattr(self, '_brain'):
                    try:
                        config = AIConfig()
                        self._brain = Brain(config=config)
                    except:
                        self._brain = Brain(config=None)
                
                # Process input and get response
                response = self._brain.process_input(text)
                
                # Display response
                self.receive_response(response)
                
            except Exception as e:
                # Error handling
                error_msg = f"I encountered an error: {str(e)}"
                self.receive_response(error_msg)
    
    def receive_response(self, text: str):
        """Receive response from Alita"""
        self.chat_area.append(
            '<div style="margin: 10px; padding: 15px; '
            'background: rgba(255, 0, 255, 0.1); '
            'border-left: 3px solid rgba(255, 0, 255, 0.8); '
            'border-radius: 8px;">'
            f'<span style="color: rgba(255, 0, 255, 1.0); font-weight: bold;">Alita:</span> '
            f'<span style="color: rgba(255, 255, 255, 0.9);">{text}</span></div>'
        )
        # Scroll to bottom
        self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        )


class VisionWidget(QWidget):
    """Vision interface with holographic styling"""
    
    def __init__(self, vision_system=None):
        super().__init__()
        self.vision_system = vision_system
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Camera feed panel
        feed_panel = HolographicPanel()
        feed_layout = QVBoxLayout(feed_panel)
        feed_layout.setContentsMargins(20, 20, 20, 20)
        
        self.feed_label = QLabel("Camera Feed")
        self.feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed_label.setMinimumSize(640, 480)
        self.feed_label.setStyleSheet("""
            QLabel {
                background: rgba(0, 0, 0, 0.5);
                border: 2px dashed rgba(0, 255, 255, 0.3);
                border-radius: 8px;
                color: rgba(255, 255, 255, 0.5);
                font-size: 18px;
            }
        """)
        feed_layout.addWidget(self.feed_label)
        
        layout.addWidget(feed_panel)
        
        # Controls panel
        controls_panel = HolographicPanel()
        controls_layout = QHBoxLayout(controls_panel)
        controls_layout.setContentsMargins(20, 20, 20, 20)
        controls_layout.setSpacing(15)
        
        self.start_button = HolographicButton("Start Camera")
        self.stop_button = HolographicButton("Stop Camera")
        self.capture_button = HolographicButton("Capture Frame")
        
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.capture_button)
        
        layout.addWidget(controls_panel)
        self.setLayout(layout)
        
        # Connect signals
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.capture_button.clicked.connect(self.capture_frame)
    
    def start_camera(self):
        """Start camera feed"""
        if self.vision_system:
            self.vision_system.start()
        self.feed_label.setText("Camera Active")
    
    def stop_camera(self):
        """Stop camera feed"""
        if self.vision_system:
            self.vision_system.stop()
        self.feed_label.setText("Camera Stopped")
    
    def capture_frame(self):
        """Capture current frame"""
        if self.vision_system:
            frame = self.vision_system.capture_frame()
            self.feed_label.setText("Frame Captured!")


class SystemStatusWidget(QWidget):
    """System status with holographic indicators"""
    
    def __init__(self, system_controller=None):
        super().__init__()
        self.system_controller = system_controller
        layout = QVBoxLayout()
        layout.setSpacing(15)
        
        # Status panel
        status_panel = HolographicPanel()
        status_layout = QVBoxLayout(status_panel)
        status_layout.setContentsMargins(20, 20, 20, 20)
        status_layout.setSpacing(15)
        
        # Title
        title = QLabel("System Status")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: rgba(0, 255, 255, 1.0);")
        status_layout.addWidget(title)
        
        # Status indicators
        self.cpu_indicator = self._create_status_row("CPU", "idle")
        self.memory_indicator = self._create_status_row("Memory", "idle")
        self.gpu_indicator = self._create_status_row("GPU", "idle")
        
        status_layout.addWidget(self.cpu_indicator)
        status_layout.addWidget(self.memory_indicator)
        status_layout.addWidget(self.gpu_indicator)
        
        # Progress bars
        self.cpu_progress = HolographicProgressBar()
        self.memory_progress = HolographicProgressBar()
        self.gpu_progress = HolographicProgressBar()
        
        status_layout.addWidget(self.cpu_progress)
        status_layout.addWidget(self.memory_progress)
        status_layout.addWidget(self.gpu_progress)
        
        layout.addWidget(status_panel)
        self.setLayout(layout)
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(1000)
    
    def _create_status_row(self, label: str, status: str):
        """Create a status indicator row"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        label_widget = QLabel(label)
        label_widget.setFont(QFont("Segoe UI", 12))
        label_widget.setStyleSheet("color: rgba(255, 255, 255, 0.9);")
        
        indicator = StatusIndicator()
        indicator.set_status(status)
        
        layout.addWidget(label_widget)
        layout.addStretch()
        layout.addWidget(indicator)
        
        return widget
    
    def update_status(self):
        """Update system status"""
        # Simulate status updates
        import random
        self.cpu_progress.setValue(random.randint(20, 80))
        self.memory_progress.setValue(random.randint(30, 70))
        self.gpu_progress.setValue(random.randint(10, 60))


class AlitaGUI(QMainWindow):
    """Main window for Alita with holographic interface"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ALITA - Advanced AI Assistant")
        self.setMinimumSize(1200, 800)
        
        # Apply enhanced holographic theme
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(5, 10, 20, 255),
                    stop:0.3 rgba(10, 5, 30, 255),
                    stop:0.7 rgba(5, 20, 40, 255),
                    stop:1 rgba(8, 12, 25, 255));
            }
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Sidebar with feature cards
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)
        
        # Main content area
        self.content_stack = QStackedWidget()
        main_layout.addWidget(self.content_stack, stretch=1)
        
        # Create content widgets
        self.chat_widget = ChatWidget()
        self.vision_widget = VisionWidget()
        self.status_widget = SystemStatusWidget()
        
        self.content_stack.addWidget(self.chat_widget)
        self.content_stack.addWidget(self.vision_widget)
        self.content_stack.addWidget(self.status_widget)
        
        # Show chat by default
        self.content_stack.setCurrentWidget(self.chat_widget)
    
    def _create_sidebar(self):
        """Create sidebar with feature cards"""
        sidebar_widget = QWidget()
        sidebar_widget.setMaximumWidth(280)
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setSpacing(15)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Logo/Title
        title_panel = HolographicPanel()
        title_layout = QVBoxLayout(title_panel)
        title_layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("ALITA")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        title.setStyleSheet("""
            color: rgba(0, 255, 255, 1.0);
            text-shadow: 0 0 20px rgba(0, 255, 255, 1.0);
        """)
        title_layout.addWidget(title)
        
        subtitle = QLabel("Advanced AI Assistant")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.7);")
        title_layout.addWidget(subtitle)
        
        sidebar_layout.addWidget(title_panel)
        
        # Feature cards
        cards_data = [
            ("Chat", "Converse with AI", "src/alita/interface/assets/chat.png", 0),
            ("Vision", "Visual Processing", "src/alita/interface/assets/vision.png", 1),
            ("System", "Status Monitor", "src/alita/interface/assets/system.png", 2),
        ]
        
        for title, desc, icon, index in cards_data:
            card = FeatureCard(title, desc, icon)
            card.clicked.connect(lambda idx=index: self.content_stack.setCurrentIndex(idx))
            sidebar_layout.addWidget(card)
        
        sidebar_layout.addStretch()
        
        return sidebar_widget


async def launch_gui():
    """Launch the Alita GUI with async support"""
    app = QApplication(sys.argv)
    
    # Apply theme
    apply_theme(app)
    
    # Set up async event loop
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # Create and show main window
    window = AlitaGUI()
    window.show()
    
    # Run event loop
    with loop:
        loop.run_forever()


def main():
    """Main entry point for the Alita GUI"""
    asyncio.run(launch_gui())


if __name__ == "__main__":
    main()
