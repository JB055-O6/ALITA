"""
ALITA Core Module
=================

Clean, consolidated core functionality for ALITA AI Assistant.

Core Components:
- brain: Main AI reasoning and planning with RAG memory
- voice: Voice interface with Whisper STT and Piper TTS
- vision: Computer vision with YOLOv8, MediaPipe, and OCR
- automation: System control and UI automation
- enhanced_resource_manager: Predictive resource management
- system_controller: Performance optimization and monitoring
- quantum_neural_engine: Advanced neural processing
- advanced_cognition: Meta-learning and cognitive architecture
- learning: Experience replay and adaptation

Utilities:
- configuration_manager: Configuration management
- logging_service: Comprehensive logging
- service_manager: Service orchestration
- system_monitor: System health monitoring
- utils: Common utilities and helpers
- async_file: Async file operations
"""

from .brain import Brain, Memory, Thought
from .voice import VoiceInterface
from .vision import VisionSystem
from .automation import SystemControl
from .enhanced_resource_manager import EnhancedResourceManager
from .system_controller import SystemController
from .quantum_neural_engine import QuantumNeuralNetwork
from .advanced_cognition import AdvancedCognition
from .learning import AdvancedLearning
from .configuration_manager import ConfigurationManager
from .logging_service import LoggingService
from .service_manager import ServiceManager
from .system_monitor import SystemMonitor
from .query_engine import QueryEngine, DatabaseType, QueryMode
from .fusion import FusionEngine, ModalityType, GestureType, FusionConfidence
from .personality import PersonalityEngine
from .emotion import EmotionDetector
from .proactive_agent import ProactiveAgent
from .utils import (
    np, pd, torch, psutil,
    ensure_dir, format_timestamp,
    setup_basic_logging, safe_cuda_percent
)

__all__ = [
    # Core AI
    "Brain",
    "Memory",
    "Thought",
    
    # Database & Query
    "QueryEngine",
    "DatabaseType",
    "QueryMode",
    
    # Multi-Modal Fusion
    "FusionEngine",
    "ModalityType",
    "GestureType",
    "FusionConfidence",
    "VoiceInterface",
    "VisionSystem",
    
    # Personality & Emotion
    "PersonalityEngine",
    "EmotionDetector",
    "ProactiveAgent",
    
    # System Control
    "SystemControl",
    "EnhancedResourceManager",
    "SystemController",
    
    # Advanced AI
    "QuantumNeuralNetwork",
    "AdvancedCognition",
    "AdvancedLearning",
    
    # Infrastructure
    "ConfigurationManager",
    "LoggingService",
    "ServiceManager",
    "SystemMonitor",
    
    # Utilities
    "np", "pd", "torch", "psutil",
    "ensure_dir", "format_timestamp",
    "setup_basic_logging", "safe_cuda_percent"
]

__version__ = "2.0.0"
__author__ = "ALITA Development Team"
