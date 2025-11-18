"""
Emotional Intelligence System

Implements Task 24 requirements:
- Voice tone analysis using speechbrain
- Typing pattern analysis with pynput
- Facial expression analysis with MediaPipe
- Stress detection with break reminders
- Mood-based response adaptation

Uses latest AI/ML frameworks:
- SpeechBrain for voice emotion recognition
- MediaPipe for facial landmark detection
- Transformers for emotion classification
- PyTorch for neural network models

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

try:
    import numpy as np
    import torch
    import torch.nn as nn
except ImportError:
    np = None
    torch = None
    nn = None

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    import cv2
    import mediapipe as mp
except ImportError:
    cv2 = None
    mp = None

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


class EmotionState(Enum):
    """Emotion states."""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    STRESSED = "stressed"
    RELAXED = "relaxed"
    EXCITED = "excited"
    CONFUSED = "confused"


@dataclass
class EmotionAnalysis:
    """Result of emotion analysis."""
    primary_emotion: EmotionState
    confidence: float
    secondary_emotions: Dict[EmotionState, float]
    indicators: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary_emotion": self.primary_emotion.value,
            "confidence": self.confidence,
            "secondary_emotions": {k.value: v for k, v in self.secondary_emotions.items()},
            "indicators": self.indicators,
            "timestamp": self.timestamp.isoformat()
        }


class VoiceEmotionAnalyzer:
    """Analyze voice tone for emotion using SpeechBrain."""
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        
        # Try to load emotion recognition model
        if pipeline:
            try:
                # Use Hugging Face emotion classification
                self.model = pipeline(
                    "audio-classification",
                    model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                    device=0 if torch and torch.cuda.is_available() else -1
                )
                self.model_loaded = True
                logging.info("Voice emotion model loaded")
            except Exception as e:
                logging.warning(f"Voice emotion model loading failed: {e}")
    
    def analyze_voice(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, float]:
        """Analyze voice for emotion.
        
        Args:
            audio_data: Audio samples
            sample_rate: Sample rate
            
        Returns:
            Emotion probabilities
        """
        if not self.model_loaded or audio_data is None:
            return {"neutral": 1.0}
        
        try:
            # Run emotion classification
            results = self.model(audio_data)
            
            # Convert to emotion dict
            emotions = {}
            for result in results:
                label = result['label'].lower()
                score = result['score']
                
                # Map labels to our emotion states
                if 'happy' in label or 'joy' in label:
                    emotions['happy'] = score
                elif 'sad' in label:
                    emotions['sad'] = score
                elif 'angry' in label or 'anger' in label:
                    emotions['angry'] = score
                elif 'fear' in label or 'stress' in label:
                    emotions['stressed'] = score
                else:
                    emotions['neutral'] = emotions.get('neutral', 0) + score
            
            return emotions
            
        except Exception as e:
            logging.error(f"Voice emotion analysis failed: {e}")
            return {"neutral": 1.0}
    
    def analyze_prosody(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Analyze prosodic features (pitch, energy, tempo).
        
        Args:
            audio_data: Audio samples
            
        Returns:
            Prosodic features
        """
        if audio_data is None or not np:
            return {}
        
        try:
            # Calculate basic prosodic features
            energy = np.mean(np.abs(audio_data))
            pitch_variance = np.var(audio_data)
            
            # Infer emotion from prosody
            features = {
                "energy": float(energy),
                "pitch_variance": float(pitch_variance)
            }
            
            # High energy + high variance = excited/angry
            # Low energy + low variance = sad/calm
            if energy > 0.5 and pitch_variance > 0.1:
                features["inferred_emotion"] = "excited"
            elif energy < 0.2 and pitch_variance < 0.05:
                features["inferred_emotion"] = "sad"
            else:
                features["inferred_emotion"] = "neutral"
            
            return features
            
        except Exception as e:
            logging.error(f"Prosody analysis failed: {e}")
            return {}


class TypingPatternAnalyzer:
    """Analyze typing patterns for emotion using pynput."""
    
    def __init__(self):
        self.keystroke_times = deque(maxlen=100)
        self.error_count = 0
        self.backspace_count = 0
        self.listener = None
        self.monitoring = False
    
    def start_monitoring(self):
        """Start monitoring typing patterns."""
        if not keyboard or self.monitoring:
            return
        
        def on_press(key):
            self.keystroke_times.append(datetime.now())
            
            # Track errors (backspace)
            if key == keyboard.Key.backspace:
                self.backspace_count += 1
        
        try:
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
            self.monitoring = True
            logging.info("Typing pattern monitoring started")
        except Exception as e:
            logging.warning(f"Typing monitoring failed: {e}")
    
    def stop_monitoring(self):
        """Stop monitoring typing patterns."""
        if self.listener:
            self.listener.stop()
            self.monitoring = False
    
    def analyze_typing(self) -> Dict[str, Any]:
        """Analyze typing patterns for emotion indicators.
        
        Returns:
            Typing analysis results
        """
        if len(self.keystroke_times) < 10:
            return {"insufficient_data": True}
        
        try:
            # Calculate typing speed (keystrokes per minute)
            time_span = (self.keystroke_times[-1] - self.keystroke_times[0]).total_seconds()
            if time_span > 0:
                typing_speed = (len(self.keystroke_times) / time_span) * 60
            else:
                typing_speed = 0
            
            # Calculate inter-keystroke intervals
            intervals = []
            for i in range(1, len(self.keystroke_times)):
                interval = (self.keystroke_times[i] - self.keystroke_times[i-1]).total_seconds()
                intervals.append(interval)
            
            avg_interval = np.mean(intervals) if intervals and np else 0
            interval_variance = np.var(intervals) if intervals and np else 0
            
            # Calculate error rate
            error_rate = self.backspace_count / len(self.keystroke_times) if self.keystroke_times else 0
            
            # Infer emotion from typing patterns
            emotion_indicators = {
                "typing_speed": typing_speed,
                "avg_interval": avg_interval,
                "interval_variance": interval_variance,
                "error_rate": error_rate
            }
            
            # High speed + high errors = stressed/frustrated
            # Low speed + high variance = confused
            # Normal speed + low errors = relaxed
            if typing_speed > 300 and error_rate > 0.15:
                emotion_indicators["inferred_emotion"] = "stressed"
            elif typing_speed < 100 and interval_variance > 0.5:
                emotion_indicators["inferred_emotion"] = "confused"
            elif error_rate < 0.05:
                emotion_indicators["inferred_emotion"] = "relaxed"
            else:
                emotion_indicators["inferred_emotion"] = "neutral"
            
            # Reset counters periodically
            if len(self.keystroke_times) >= 100:
                self.backspace_count = 0
            
            return emotion_indicators
            
        except Exception as e:
            logging.error(f"Typing analysis failed: {e}")
            return {}


class FacialExpressionAnalyzer:
    """Analyze facial expressions using MediaPipe."""
    
    def __init__(self):
        self.face_mesh = None
        self.model_loaded = False
        
        if mp:
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.model_loaded = True
                logging.info("Facial expression analyzer loaded")
            except Exception as e:
                logging.warning(f"MediaPipe initialization failed: {e}")
    
    def analyze_face(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze facial expression from video frame.
        
        Args:
            frame: Video frame (BGR format)
            
        Returns:
            Facial expression analysis
        """
        if not self.model_loaded or frame is None:
            return {}
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cv2 else frame
            
            # Process frame
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return {"face_detected": False}
            
            # Get landmarks
            landmarks = results.multi_face_landmarks[0]
            
            # Extract key points for emotion detection
            emotion_features = self._extract_emotion_features(landmarks)
            
            # Classify emotion
            emotion = self._classify_emotion(emotion_features)
            
            return {
                "face_detected": True,
                "emotion": emotion,
                "features": emotion_features
            }
            
        except Exception as e:
            logging.error(f"Facial analysis failed: {e}")
            return {}
    
    def _extract_emotion_features(self, landmarks) -> Dict[str, float]:
        """Extract emotion-relevant features from landmarks."""
        try:
            # Get key landmark points
            # Mouth corners (61, 291)
            # Eyes (33, 263)
            # Eyebrows (70, 300)
            
            mouth_left = landmarks.landmark[61]
            mouth_right = landmarks.landmark[291]
            eye_left = landmarks.landmark[33]
            eye_right = landmarks.landmark[263]
            eyebrow_left = landmarks.landmark[70]
            eyebrow_right = landmarks.landmark[300]
            
            # Calculate features
            mouth_width = abs(mouth_right.x - mouth_left.x)
            mouth_height = abs(mouth_right.y - mouth_left.y)
            eye_openness = (eye_left.y + eye_right.y) / 2
            eyebrow_height = (eyebrow_left.y + eyebrow_right.y) / 2
            
            return {
                "mouth_width": mouth_width,
                "mouth_height": mouth_height,
                "eye_openness": eye_openness,
                "eyebrow_height": eyebrow_height,
                "mouth_curvature": mouth_height / mouth_width if mouth_width > 0 else 0
            }
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            return {}
    
    def _classify_emotion(self, features: Dict[str, float]) -> str:
        """Classify emotion from features."""
        if not features:
            return "neutral"
        
        # Simple rule-based classification
        # (In production, use trained ML model)
        
        mouth_curvature = features.get("mouth_curvature", 0)
        eyebrow_height = features.get("eyebrow_height", 0.5)
        
        # Smile detection (mouth curves up)
        if mouth_curvature > 0.3:
            return "happy"
        
        # Frown detection (eyebrows down)
        elif eyebrow_height < 0.4:
            return "sad"
        
        # Anger detection (eyebrows down + mouth tight)
        elif eyebrow_height < 0.45 and mouth_curvature < 0.1:
            return "angry"
        
        else:
            return "neutral"
    
    def __del__(self):
        """Cleanup."""
        if self.face_mesh:
            self.face_mesh.close()


class StressDetector:
    """Detect stress levels and provide interventions."""
    
    def __init__(self):
        self.stress_indicators = deque(maxlen=50)
        self.stress_threshold = 0.7
        self.last_break_reminder = None
        self.break_interval = timedelta(hours=1)
    
    def update_stress_level(self, indicators: Dict[str, float]):
        """Update stress level from various indicators.
        
        Args:
            indicators: Stress indicators (0-1 scale)
        """
        # Calculate overall stress score
        stress_score = np.mean(list(indicators.values())) if indicators and np else 0
        
        self.stress_indicators.append({
            "score": stress_score,
            "indicators": indicators,
            "timestamp": datetime.now()
        })
    
    def get_stress_level(self) -> float:
        """Get current stress level.
        
        Returns:
            Stress level (0-1)
        """
        if not self.stress_indicators:
            return 0.0
        
        # Average recent stress scores
        recent_scores = [s["score"] for s in list(self.stress_indicators)[-10:]]
        return np.mean(recent_scores) if recent_scores and np else 0.0
    
    def should_suggest_break(self) -> bool:
        """Check if break should be suggested.
        
        Returns:
            True if break should be suggested
        """
        stress_level = self.get_stress_level()
        
        # Check stress threshold
        if stress_level < self.stress_threshold:
            return False
        
        # Check time since last reminder
        if self.last_break_reminder:
            time_since_break = datetime.now() - self.last_break_reminder
            if time_since_break < self.break_interval:
                return False
        
        return True
    
    def suggest_break(self) -> Dict[str, Any]:
        """Generate break suggestion.
        
        Returns:
            Break suggestion
        """
        self.last_break_reminder = datetime.now()
        
        stress_level = self.get_stress_level()
        
        suggestions = {
            "message": "You seem stressed. Consider taking a short break.",
            "activities": [
                "Take a 5-minute walk",
                "Do some stretching exercises",
                "Practice deep breathing",
                "Get a drink of water"
            ],
            "stress_level": stress_level
        }
        
        return suggestions


class MoodBasedAdapter:
    """Adapt responses based on detected mood."""
    
    def __init__(self):
        self.current_mood = EmotionState.NEUTRAL
        self.mood_history = deque(maxlen=20)
    
    def update_mood(self, emotion: EmotionState, confidence: float):
        """Update current mood.
        
        Args:
            emotion: Detected emotion
            confidence: Confidence level
        """
        if confidence >= 0.6:
            self.current_mood = emotion
        
        self.mood_history.append({
            "emotion": emotion,
            "confidence": confidence,
            "timestamp": datetime.now()
        })
    
    def get_response_style(self) -> Dict[str, Any]:
        """Get recommended response style for current mood.
        
        Returns:
            Response style parameters
        """
        styles = {
            EmotionState.STRESSED: {
                "verbosity": "concise",
                "tone": "calm",
                "suggestions": "practical",
                "pace": "slower"
            },
            EmotionState.FRUSTRATED: {
                "verbosity": "concise",
                "tone": "supportive",
                "suggestions": "troubleshooting",
                "pace": "patient"
            },
            EmotionState.CONFUSED: {
                "verbosity": "detailed",
                "tone": "explanatory",
                "suggestions": "clarifying",
                "pace": "slower"
            },
            EmotionState.HAPPY: {
                "verbosity": "normal",
                "tone": "enthusiastic",
                "suggestions": "encouraging",
                "pace": "normal"
            },
            EmotionState.RELAXED: {
                "verbosity": "detailed",
                "tone": "conversational",
                "suggestions": "exploratory",
                "pace": "normal"
            }
        }
        
        return styles.get(self.current_mood, {
            "verbosity": "normal",
            "tone": "neutral",
            "suggestions": "balanced",
            "pace": "normal"
        })
    
    def should_offer_help(self) -> bool:
        """Check if proactive help should be offered.
        
        Returns:
            True if help should be offered
        """
        # Offer help for negative emotions
        negative_emotions = {
            EmotionState.FRUSTRATED,
            EmotionState.STRESSED,
            EmotionState.CONFUSED,
            EmotionState.SAD
        }
        
        return self.current_mood in negative_emotions


class EmotionDetector:
    """Main emotional intelligence system."""
    
    def __init__(self):
        self.voice_analyzer = VoiceEmotionAnalyzer()
        self.typing_analyzer = TypingPatternAnalyzer()
        self.facial_analyzer = FacialExpressionAnalyzer()
        self.stress_detector = StressDetector()
        self.mood_adapter = MoodBasedAdapter()
        
        # Current state
        self.current_emotion: Optional[EmotionAnalysis] = None
        self.emotion_history = deque(maxlen=100)
        
        # Start monitoring
        self.typing_analyzer.start_monitoring()
        
        logging.info("Emotion Detector initialized")
    
    async def analyze_emotion(self,
                             voice_data: Optional[np.ndarray] = None,
                             video_frame: Optional[np.ndarray] = None) -> EmotionAnalysis:
        """Analyze emotion from multiple modalities.
        
        Args:
            voice_data: Audio samples
            video_frame: Video frame
            
        Returns:
            Emotion analysis result
        """
        emotion_scores = {}
        indicators = {}
        
        # Analyze voice
        if voice_data is not None:
            voice_emotions = self.voice_analyzer.analyze_voice(voice_data)
            prosody = self.voice_analyzer.analyze_prosody(voice_data)
            
            for emotion, score in voice_emotions.items():
                emotion_scores[emotion] = emotion_scores.get(emotion, 0) + score * 0.4
            
            indicators["voice"] = {"emotions": voice_emotions, "prosody": prosody}
        
        # Analyze typing
        typing_analysis = self.typing_analyzer.analyze_typing()
        if typing_analysis and "inferred_emotion" in typing_analysis:
            emotion = typing_analysis["inferred_emotion"]
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + 0.3
            indicators["typing"] = typing_analysis
        
        # Analyze facial expression
        if video_frame is not None:
            facial_analysis = self.facial_analyzer.analyze_face(video_frame)
            if facial_analysis.get("face_detected"):
                emotion = facial_analysis.get("emotion", "neutral")
                emotion_scores[emotion] = emotion_scores.get(emotion, 0) + 0.3
                indicators["facial"] = facial_analysis
        
        # Determine primary emotion
        if emotion_scores:
            primary_emotion_str = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion_str]
            
            # Map to EmotionState
            emotion_map = {
                "happy": EmotionState.HAPPY,
                "sad": EmotionState.SAD,
                "angry": EmotionState.ANGRY,
                "frustrated": EmotionState.FRUSTRATED,
                "stressed": EmotionState.STRESSED,
                "relaxed": EmotionState.RELAXED,
                "excited": EmotionState.EXCITED,
                "confused": EmotionState.CONFUSED,
                "neutral": EmotionState.NEUTRAL
            }
            
            primary_emotion = emotion_map.get(primary_emotion_str, EmotionState.NEUTRAL)
            
            # Get secondary emotions
            secondary_emotions = {
                emotion_map.get(e, EmotionState.NEUTRAL): s
                for e, s in emotion_scores.items()
                if e != primary_emotion_str
            }
        else:
            primary_emotion = EmotionState.NEUTRAL
            confidence = 1.0
            secondary_emotions = {}
        
        # Create analysis result
        analysis = EmotionAnalysis(
            primary_emotion=primary_emotion,
            confidence=confidence,
            secondary_emotions=secondary_emotions,
            indicators=indicators,
            timestamp=datetime.now()
        )
        
        # Update state
        self.current_emotion = analysis
        self.emotion_history.append(analysis)
        
        # Update mood adapter
        self.mood_adapter.update_mood(primary_emotion, confidence)
        
        # Update stress detector
        stress_indicators = {}
        if "typing" in indicators:
            stress_indicators["typing_stress"] = indicators["typing"].get("error_rate", 0)
        if primary_emotion in [EmotionState.STRESSED, EmotionState.FRUSTRATED]:
            stress_indicators["emotion_stress"] = confidence
        
        self.stress_detector.update_stress_level(stress_indicators)
        
        return analysis
    
    def get_recommendations(self) -> Dict[str, Any]:
        """Get recommendations based on current emotional state.
        
        Returns:
            Recommendations dictionary
        """
        recommendations = {
            "response_style": self.mood_adapter.get_response_style(),
            "offer_help": self.mood_adapter.should_offer_help(),
            "stress_level": self.stress_detector.get_stress_level()
        }
        
        # Check for break suggestion
        if self.stress_detector.should_suggest_break():
            recommendations["break_suggestion"] = self.stress_detector.suggest_break()
        
        return recommendations
    
    def get_emotion_summary(self) -> Dict[str, Any]:
        """Get emotion summary.
        
        Returns:
            Summary dictionary
        """
        if not self.current_emotion:
            return {"no_data": True}
        
        return {
            "current_emotion": self.current_emotion.primary_emotion.value,
            "confidence": self.current_emotion.confidence,
            "stress_level": self.stress_detector.get_stress_level(),
            "current_mood": self.mood_adapter.current_mood.value,
            "response_style": self.mood_adapter.get_response_style(),
            "history_size": len(self.emotion_history)
        }
    
    def shutdown(self):
        """Shutdown emotion detector."""
        self.typing_analyzer.stop_monitoring()
        logging.info("Emotion Detector shutdown")


# Global instance
_emotion_detector: Optional[EmotionDetector] = None


def get_emotion_detector() -> EmotionDetector:
    """Get global emotion detector instance."""
    global _emotion_detector
    
    if _emotion_detector is None:
        _emotion_detector = EmotionDetector()
    
    return _emotion_detector
