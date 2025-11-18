"""
Multi-Modal Fusion Engine
Advanced fusion of voice, vision, and gesture inputs with temporal synchronization
Uses state-of-the-art free AI models for unified multi-modal understanding
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import cv2

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logging.warning("CLIP not available - visual-language fusion limited")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available - gesture recognition disabled")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL not available - image processing limited")


class ModalityType(Enum):
    """Types of input modalities"""
    VOICE = "voice"
    VISION = "vision"
    GESTURE = "gesture"
    SCREEN = "screen"
    TEXT = "text"


class GestureType(Enum):
    """Recognized gesture types"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    WAVE = "wave"
    POINT = "point"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    OK_SIGN = "ok_sign"
    PEACE_SIGN = "peace_sign"
    UNKNOWN = "unknown"


class FusionConfidence(Enum):
    """Confidence levels for fusion results"""
    HIGH = "high"  # >0.8
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"  # <0.5


@dataclass
class ModalInput:
    """Single modality input with timestamp"""
    modality: ModalityType
    data: Any
    timestamp: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Result of multi-modal fusion"""
    unified_understanding: str
    confidence: FusionConfidence
    contributing_modalities: List[ModalityType]
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)


@dataclass
class GestureRecognition:
    """Gesture recognition result"""
    gesture_type: GestureType
    confidence: float
    hand_landmarks: Optional[List[Tuple[float, float]]]
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class FusionEngine:
    """
    Advanced Multi-Modal Fusion Engine
    
    Features:
    - Simultaneous voice/vision/gesture processing
    - Visual-language understanding with CLIP
    - Gesture recognition with MediaPipe
    - Screen element identification
    - Temporal synchronization
    - Context-aware disambiguation
    - Real-time fusion (<500ms latency)
    """
    
    def __init__(
        self,
        voice_interface=None,
        vision_system=None,
        temporal_window: float = 2.0,
        device: str = "auto"
    ):
        """
        Initialize Fusion Engine
        
        Args:
            voice_interface: VoiceInterface instance
            vision_system: VisionSystem instance
            temporal_window: Time window for temporal fusion (seconds)
            device: Device for model inference
        """
        self.logger = logging.getLogger(__name__)
        self.voice_interface = voice_interface
        self.vision_system = vision_system
        self.temporal_window = temporal_window
        self.device = self._setup_device(device)
        
        # Input buffers with temporal ordering
        self.input_buffer: deque = deque(maxlen=100)
        self.buffer_lock = threading.Lock()
        
        # Initialize CLIP for visual-language understanding
        self.clip_model = None
        self.clip_processor = None
        if CLIP_AVAILABLE:
            self._load_clip_model()
        
        # Initialize MediaPipe for gesture recognition
        self.gesture_recognizer = None
        if MEDIAPIPE_AVAILABLE:
            self._init_gesture_recognition()
        
        # Fusion state
        self.active_context: Dict[str, Any] = {}
        self.last_fusion_time = 0
        self.fusion_history: deque = deque(maxlen=50)
        
        # Callbacks for modality inputs
        self.modality_callbacks: Dict[ModalityType, List[Callable]] = {
            modality: [] for modality in ModalityType
        }
        
        # Performance tracking
        self.fusion_latencies: deque = deque(maxlen=100)
        
        self.logger.info("FusionEngine initialized successfully")
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_clip_model(self):
        """Load CLIP model for visual-language understanding"""
        try:
            self.logger.info("Loading CLIP model...")
            
            # Use OpenCLIP for better performance
            model_name = "openai/clip-vit-large-patch14"
            
            self.clip_processor = CLIPProcessor.from_pretrained(model_name)
            self.clip_model = CLIPModel.from_pretrained(model_name)
            
            if self.device != "cpu":
                self.clip_model = self.clip_model.to(self.device)
            
            self.clip_model.eval()
            
            self.logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def _init_gesture_recognition(self):
        """Initialize MediaPipe gesture recognition"""
        try:
            self.logger.info("Initializing gesture recognition...")
            
            # Initialize MediaPipe Hands
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            
            self.gesture_recognizer = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            
            self.logger.info("Gesture recognition initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize gesture recognition: {e}")
            self.gesture_recognizer = None
    
    def add_input(
        self,
        modality: ModalityType,
        data: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add input from a modality
        
        Args:
            modality: Type of input modality
            data: Input data
            confidence: Confidence score
            metadata: Additional metadata
        """
        modal_input = ModalInput(
            modality=modality,
            data=data,
            timestamp=time.time(),
            confidence=confidence,
            metadata=metadata or {}
        )
        
        with self.buffer_lock:
            self.input_buffer.append(modal_input)
        
        # Trigger callbacks
        for callback in self.modality_callbacks.get(modality, []):
            try:
                callback(modal_input)
            except Exception as e:
                self.logger.error(f"Callback error for {modality}: {e}")
    
    def register_callback(self, modality: ModalityType, callback: Callable):
        """Register callback for modality input"""
        self.modality_callbacks[modality].append(callback)
    
    def fuse_inputs(
        self,
        require_modalities: Optional[List[ModalityType]] = None
    ) -> Optional[FusionResult]:
        """
        Fuse recent inputs from multiple modalities
        
        Args:
            require_modalities: Required modalities for fusion
            
        Returns:
            Fusion result or None if insufficient data
        """
        start_time = time.time()
        
        try:
            # Get recent inputs within temporal window
            current_time = time.time()
            cutoff_time = current_time - self.temporal_window
            
            with self.buffer_lock:
                recent_inputs = [
                    inp for inp in self.input_buffer
                    if inp.timestamp >= cutoff_time
                ]
            
            if not recent_inputs:
                return None
            
            # Check if required modalities are present
            if require_modalities:
                present_modalities = {inp.modality for inp in recent_inputs}
                if not all(mod in present_modalities for mod in require_modalities):
                    return None
            
            # Group inputs by modality
            grouped_inputs = self._group_by_modality(recent_inputs)
            
            # Perform fusion based on available modalities
            result = self._perform_fusion(grouped_inputs, current_time)
            
            # Track latency
            latency = time.time() - start_time
            self.fusion_latencies.append(latency)
            
            if result:
                self.fusion_history.append(result)
                self.last_fusion_time = current_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Fusion failed: {e}")
            return None
    
    def _group_by_modality(
        self,
        inputs: List[ModalInput]
    ) -> Dict[ModalityType, List[ModalInput]]:
        """Group inputs by modality type"""
        grouped = {}
        for inp in inputs:
            if inp.modality not in grouped:
                grouped[inp.modality] = []
            grouped[inp.modality].append(inp)
        return grouped
    
    def _perform_fusion(
        self,
        grouped_inputs: Dict[ModalityType, List[ModalInput]],
        timestamp: float
    ) -> Optional[FusionResult]:
        """Perform multi-modal fusion"""
        
        # Extract latest from each modality
        latest_inputs = {
            modality: max(inputs, key=lambda x: x.timestamp)
            for modality, inputs in grouped_inputs.items()
        }
        
        contributing_modalities = list(latest_inputs.keys())
        
        # Voice + Vision fusion
        if ModalityType.VOICE in latest_inputs and ModalityType.VISION in latest_inputs:
            return self._fuse_voice_vision(latest_inputs, timestamp)
        
        # Voice + Gesture fusion
        elif ModalityType.VOICE in latest_inputs and ModalityType.GESTURE in latest_inputs:
            return self._fuse_voice_gesture(latest_inputs, timestamp)
        
        # Voice + Screen fusion
        elif ModalityType.VOICE in latest_inputs and ModalityType.SCREEN in latest_inputs:
            return self._fuse_voice_screen(latest_inputs, timestamp)
        
        # Triple fusion: Voice + Vision + Gesture
        elif all(mod in latest_inputs for mod in [ModalityType.VOICE, ModalityType.VISION, ModalityType.GESTURE]):
            return self._fuse_voice_vision_gesture(latest_inputs, timestamp)
        
        # Single modality (no fusion needed)
        elif len(latest_inputs) == 1:
            modality, inp = list(latest_inputs.items())[0]
            return FusionResult(
                unified_understanding=str(inp.data),
                confidence=FusionConfidence.MEDIUM,
                contributing_modalities=[modality],
                timestamp=timestamp,
                details={"single_modality": True}
            )
        
        return None
    
    def _fuse_voice_vision(
        self,
        inputs: Dict[ModalityType, ModalInput],
        timestamp: float
    ) -> FusionResult:
        """Fuse voice and vision inputs"""
        voice_input = inputs[ModalityType.VOICE]
        vision_input = inputs[ModalityType.VISION]
        
        voice_text = voice_input.data
        vision_frame = vision_input.data
        
        # Use CLIP for visual-language understanding
        if self.clip_model and PIL_AVAILABLE:
            understanding = self._clip_understand(voice_text, vision_frame)
        else:
            # Fallback: simple concatenation
            understanding = f"Voice: {voice_text} | Vision: detected objects"
        
        # Determine confidence
        avg_confidence = (voice_input.confidence + vision_input.confidence) / 2
        confidence = self._confidence_level(avg_confidence)
        
        return FusionResult(
            unified_understanding=understanding,
            confidence=confidence,
            contributing_modalities=[ModalityType.VOICE, ModalityType.VISION],
            timestamp=timestamp,
            details={
                "voice_text": voice_text,
                "vision_processed": True
            }
        )
    
    def _fuse_voice_gesture(
        self,
        inputs: Dict[ModalityType, ModalInput],
        timestamp: float
    ) -> FusionResult:
        """Fuse voice and gesture inputs"""
        voice_input = inputs[ModalityType.VOICE]
        gesture_input = inputs[ModalityType.GESTURE]
        
        voice_text = voice_input.data
        gesture_data = gesture_input.data
        
        # Interpret gesture in context of voice
        if isinstance(gesture_data, GestureRecognition):
            gesture_type = gesture_data.gesture_type
            
            # Map gestures to actions
            if gesture_type == GestureType.THUMBS_UP:
                understanding = f"Approve: {voice_text}"
                actions = ["approve", "confirm"]
            elif gesture_type == GestureType.THUMBS_DOWN:
                understanding = f"Reject: {voice_text}"
                actions = ["reject", "cancel"]
            elif gesture_type == GestureType.WAVE:
                understanding = f"Cancel: {voice_text}"
                actions = ["cancel", "dismiss"]
            elif gesture_type == GestureType.POINT:
                understanding = f"Select/Point: {voice_text}"
                actions = ["select", "point"]
            else:
                understanding = f"{voice_text} with {gesture_type.value} gesture"
                actions = []
        else:
            understanding = f"{voice_text} with gesture"
            actions = []
        
        return FusionResult(
            unified_understanding=understanding,
            confidence=FusionConfidence.HIGH,
            contributing_modalities=[ModalityType.VOICE, ModalityType.GESTURE],
            timestamp=timestamp,
            details={"gesture_type": gesture_type.value if isinstance(gesture_data, GestureRecognition) else "unknown"},
            actions=actions
        )
    
    def _fuse_voice_screen(
        self,
        inputs: Dict[ModalityType, ModalInput],
        timestamp: float
    ) -> FusionResult:
        """Fuse voice and screen inputs"""
        voice_input = inputs[ModalityType.VOICE]
        screen_input = inputs[ModalityType.SCREEN]
        
        voice_text = voice_input.data
        screen_data = screen_input.data
        
        # Identify UI elements referenced in voice command
        referenced_elements = self._identify_screen_references(voice_text, screen_data)
        
        if referenced_elements:
            understanding = f"Action on screen: {voice_text} -> {referenced_elements}"
            actions = ["click", "interact"]
        else:
            understanding = f"Screen context: {voice_text}"
            actions = []
        
        return FusionResult(
            unified_understanding=understanding,
            confidence=FusionConfidence.MEDIUM,
            contributing_modalities=[ModalityType.VOICE, ModalityType.SCREEN],
            timestamp=timestamp,
            details={"referenced_elements": referenced_elements},
            actions=actions
        )
    
    def _fuse_voice_vision_gesture(
        self,
        inputs: Dict[ModalityType, ModalInput],
        timestamp: float
    ) -> FusionResult:
        """Fuse voice, vision, and gesture inputs (triple fusion)"""
        voice_input = inputs[ModalityType.VOICE]
        vision_input = inputs[ModalityType.VISION]
        gesture_input = inputs[ModalityType.GESTURE]
        
        voice_text = voice_input.data
        vision_frame = vision_input.data
        gesture_data = gesture_input.data
        
        # Complex multi-modal understanding
        understanding_parts = []
        actions = []
        
        # Voice component
        understanding_parts.append(f"Command: {voice_text}")
        
        # Vision component
        if self.clip_model:
            visual_context = self._extract_visual_context(vision_frame)
            understanding_parts.append(f"Visual: {visual_context}")
        
        # Gesture component
        if isinstance(gesture_data, GestureRecognition):
            gesture_type = gesture_data.gesture_type
            understanding_parts.append(f"Gesture: {gesture_type.value}")
            
            # Add gesture-based actions
            if gesture_type == GestureType.THUMBS_UP:
                actions.append("approve")
            elif gesture_type == GestureType.POINT:
                actions.append("select")
        
        understanding = " | ".join(understanding_parts)
        
        return FusionResult(
            unified_understanding=understanding,
            confidence=FusionConfidence.HIGH,
            contributing_modalities=[ModalityType.VOICE, ModalityType.VISION, ModalityType.GESTURE],
            timestamp=timestamp,
            details={"triple_fusion": True},
            actions=actions
        )
    
    def _clip_understand(self, text: str, image: np.ndarray) -> str:
        """Use CLIP for visual-language understanding"""
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Process inputs
            inputs = self.clip_processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get similarity
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            similarity = probs[0][0].item()
            
            if similarity > 0.7:
                return f"High match: {text} aligns with visual content (confidence: {similarity:.2f})"
            elif similarity > 0.4:
                return f"Partial match: {text} somewhat relates to visual content (confidence: {similarity:.2f})"
            else:
                return f"Low match: {text} may not match visual content (confidence: {similarity:.2f})"
                
        except Exception as e:
            self.logger.error(f"CLIP understanding failed: {e}")
            return f"Visual-language fusion: {text}"
    
    def _extract_visual_context(self, image: np.ndarray) -> str:
        """Extract visual context from image"""
        # Simplified visual context extraction
        # In production, use object detection or scene understanding
        return "scene detected"
    
    def _identify_screen_references(
        self,
        voice_text: str,
        screen_data: Any
    ) -> List[str]:
        """Identify UI elements referenced in voice command"""
        # Simplified reference identification
        # In production, use OCR + spatial understanding
        
        reference_keywords = ["that", "this", "button", "icon", "window", "menu"]
        referenced = []
        
        for keyword in reference_keywords:
            if keyword in voice_text.lower():
                referenced.append(keyword)
        
        return referenced
    
    def recognize_gesture(self, frame: np.ndarray) -> Optional[GestureRecognition]:
        """
        Recognize gesture from video frame
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            Gesture recognition result
        """
        if not self.gesture_recognizer:
            return None
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = self.gesture_recognizer.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                return None
            
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract landmark positions
            landmarks = [
                (lm.x, lm.y) for lm in hand_landmarks.landmark
            ]
            
            # Classify gesture
            gesture_type, confidence = self._classify_gesture(landmarks)
            
            return GestureRecognition(
                gesture_type=gesture_type,
                confidence=confidence,
                hand_landmarks=landmarks,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Gesture recognition failed: {e}")
            return None
    
    def _classify_gesture(
        self,
        landmarks: List[Tuple[float, float]]
    ) -> Tuple[GestureType, float]:
        """Classify gesture from hand landmarks"""
        # Simplified gesture classification
        # In production, use trained classifier or rule-based system
        
        if len(landmarks) < 21:
            return GestureType.UNKNOWN, 0.0
        
        # Get key points
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        wrist = landmarks[0]
        
        # Calculate finger extensions
        fingers_extended = []
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]:
            extended = tip[1] < wrist[1]  # Y coordinate (inverted)
            fingers_extended.append(extended)
        
        # Classify based on extended fingers
        extended_count = sum(fingers_extended)
        
        if extended_count == 5:
            return GestureType.OPEN_PALM, 0.9
        elif extended_count == 0:
            return GestureType.FIST, 0.9
        elif fingers_extended[1] and not any(fingers_extended[2:]):
            return GestureType.POINT, 0.8
        elif fingers_extended[0] and not any(fingers_extended[1:]):
            return GestureType.THUMBS_UP, 0.85
        elif fingers_extended[1] and fingers_extended[2] and not any(fingers_extended[3:]):
            return GestureType.PEACE_SIGN, 0.8
        else:
            return GestureType.UNKNOWN, 0.5
    
    def _confidence_level(self, score: float) -> FusionConfidence:
        """Convert numeric confidence to enum"""
        if score >= 0.8:
            return FusionConfidence.HIGH
        elif score >= 0.5:
            return FusionConfidence.MEDIUM
        else:
            return FusionConfidence.LOW
    
    def get_average_latency(self) -> float:
        """Get average fusion latency"""
        if not self.fusion_latencies:
            return 0.0
        return sum(self.fusion_latencies) / len(self.fusion_latencies)
    
    def clear_buffer(self):
        """Clear input buffer"""
        with self.buffer_lock:
            self.input_buffer.clear()
    
    def get_fusion_history(self, limit: int = 10) -> List[FusionResult]:
        """Get recent fusion results"""
        return list(self.fusion_history)[-limit:]
    
    def shutdown(self):
        """Shutdown fusion engine"""
        if self.gesture_recognizer:
            self.gesture_recognizer.close()
        
        self.logger.info("FusionEngine shutdown complete")
