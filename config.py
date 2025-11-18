from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import yaml

@dataclass
class APIConfig:
    """Free API service configuration."""
    # HuggingFace - unlimited inference API with community account
    hf_token: Optional[str] = None
    
    # Stability AI - free tier for image generation
    stability_key: Optional[str] = None
    
    # CockroachDB - free serverless tier for cloud storage
    database_url: Optional[str] = None
    
    # Render.com - free tier for model hosting
    render_api_key: Optional[str] = None

@dataclass
class AIConfig:
    """Core AI configuration."""
    # Local models
    model_path: Path
    context_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Free model endpoints
    use_hf_inference: bool = True  # Use HuggingFace's free inference API
    use_pipeline: bool = True      # Use optimized local pipeline
    enable_8bit: bool = True       # Enable 8-bit quantization for memory savings
    
    # Model selection
    primary_model: str = "microsoft/phi-2"  # Free research model
    fallback_model: str = "HuggingFaceH4/zephyr-7b-beta"  # Free backup
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight

@dataclass
class VoiceConfig:
    """Voice interface configuration."""
    wake_word: str = "alita"
    wake_word_threshold: float = 0.5
    
    # Speech recognition stack
    primary_stt: str = "openai/whisper-large-v3"  # Free model
    fallback_stt: str = "vosk-model-small-en-us"  # Offline backup
    vad_enabled: bool = True
    
    # Text to speech stack
    primary_tts: str = "piper/en_US-lessac-medium"  # Free voice
    fallback_tts: str = "espeak"  # Offline backup
    
    # Advanced features
    enable_voice_clone: bool = True  # Using free Coqui training
    speaker_recognition: bool = True  # Using SpeechBrain

@dataclass
class VisionConfig:
    """Computer vision configuration."""
    # Object detection
    yolo_model: str = "yolov8x.pt"  # Free YOLO
    enable_tracking: bool = True     # Using ByteTrack (free)
    
    # Face and body analysis
    use_mediapipe: bool = True          # Google's free ML kit
    face_detection: bool = True          # Using FaceNet (free)
    face_recognition: bool = True        # Using DeepFace (free)
    pose_estimation: bool = True         # Using BlazePose
    
    # OCR and document analysis
    ocr_enabled: bool = True             # Using Tesseract
    document_analysis: bool = True        # Using Layout Parser
    
    # Special features
    enable_depth: bool = True            # Using MiDaS (free)
    enable_segmentation: bool = True      # Using Segment Anything (free)

@dataclass
class SystemConfig:
    """System control configuration."""
    safe_mode: bool = True
    debug: bool = False
    log_path: Path = Path("logs/audit.log")
    max_concurrent_tasks: int = 5

@dataclass
class Config:
    """Main configuration container."""
    ai: AIConfig
    voice: VoiceConfig
    vision: VisionConfig
    system: SystemConfig
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> 'Config':
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = Path("config/config.yaml")
            
        if not config_path.exists():
            return cls.default()
            
        with open(config_path) as f:
            data = yaml.safe_load(f)
            
        return cls(
            ai=AIConfig(**data.get("ai", {})),
            voice=VoiceConfig(**data.get("voice", {})),
            vision=VisionConfig(**data.get("vision", {})),
            system=SystemConfig(**data.get("system", {}))
        )
    
    @classmethod
    def default(cls) -> 'Config':
        """Create default configuration."""
        return cls(
            ai=AIConfig(model_path=Path("models/llama-2-70b-chat.gguf")),
            voice=VoiceConfig(),
            vision=VisionConfig(),
            system=SystemConfig()
        )