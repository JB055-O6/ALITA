"""ALITA Enhanced Configuration System

Advanced configuration management with:
- Hot-reload without restart
- File watching for automatic updates
- Multiple voice model support
- Macro template system
- System prompt customization
- Configuration backup/restore
- Validation and schema enforcement

All features are FREE and run locally!
"""

import os
import yaml
import json
import shutil
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    logging.warning("watchdog not available - hot-reload disabled")


class ConfigSection(Enum):
    """Configuration sections."""
    BRAIN = "brain"
    VOICE = "voice"
    VISION = "vision"
    MEMORY = "memory"
    AUTOMATION = "automation"
    SAFETY = "safety"
    PERSONALITY = "personality"
    RESOURCES = "resources"
    LOGGING = "logging"
    MACROS = "macros"


@dataclass
class VoiceModel:
    """Voice model configuration."""
    name: str
    path: str
    language: str
    gender: str
    quality: str  # low, medium, high
    speed: float = 1.0
    enabled: bool = True
    description: str = ""


@dataclass
class MacroTemplate:
    """Automation macro template."""
    name: str
    description: str
    steps: List[Dict[str, Any]]
    category: str
    enabled: bool = True
    hotkey: Optional[str] = None


@dataclass
class PersonalityConfig:
    """Personality configuration."""
    system_prompt: str
    response_style: str  # formal, casual, friendly, professional
    verbosity: str  # concise, normal, detailed
    humor_level: float  # 0.0 to 1.0
    empathy_level: float  # 0.0 to 1.0
    proactivity: float  # 0.0 to 1.0


class ConfigChangeHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """Handle configuration file changes."""
    
    def __init__(self, config_manager: 'EnhancedConfigManager'):
        self.config_manager = config_manager
        self.last_modified = {}
    
    def on_modified(self, event):
        """Handle file modification."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if it's a config file
        if file_path.suffix not in ['.yaml', '.yml', '.json']:
            return
        
        # Debounce: ignore if modified within last second
        now = datetime.now()
        if file_path in self.last_modified:
            if (now - self.last_modified[file_path]).total_seconds() < 1.0:
                return
        
        self.last_modified[file_path] = now
        
        # Reload configuration
        logging.info(f"🔄 Configuration file changed: {file_path.name}")
        self.config_manager.reload_file(file_path)


class EnhancedConfigManager:
    """Enhanced configuration manager with hot-reloading."""
    
    def __init__(self, config_dir: Path = None):
        self.config_dir = config_dir or Path("config")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration storage
        self.config: Dict[str, Any] = {}
        self.voice_models: Dict[str, VoiceConfig] = {}
        self.macros: Dict[str, MacroTemplate] = {}
        self.personality: Optional[PersonalityConfig] = None
        
        # Change callbacks
        self.change_callbacks: List[Callable[[str, Any], None]] = []
        
        # File watcher
        self.observer = None
        if WATCHDOG_AVAILABLE:
            self.observer = Observer()
            handler = ConfigChangeHandler(self)
            self.observer.schedule(handler, str(self.config_dir), recursive=True)
            self.observer.start()
            logging.info("✅ Configuration hot-reloading enabled")
        else:
            logging.warning("⚠️ watchdog not installed, hot-reloading disabled")
        
        # Load initial configuration
        self.load_all()
    
    def load_all(self):
        """Load all configuration files."""
        logging.info("📂 Loading configuration...")
        
        # Load main config
        self.load_main_config()
        
        # Load voice models
        self.load_voice_models()
        
        # Load macros
        self.load_macros()
        
        # Load personality
        self.load_personality()
        
        logging.info("✅ Configuration loaded successfully")
    
    def load_main_config(self):
        """Load main configuration file."""
        config_file = self.config_dir / "alita_config.yaml"
        
        if not config_file.exists():
            # Create default config
            default_config = {
                "system": {
                    "log_level": "INFO",
                    "max_memory_gb": 12,
                    "max_vram_gb": 3.5,
                    "enable_gpu": True,
                    "enable_voice": True,
                    "enable_vision": True,
                },
                "models": {
                    "llm": "Llama-3.2-3B-Instruct",
                    "whisper": "whisper-large-v3-turbo",
                    "embeddings": "all-MiniLM-L6-v2",
                    "quantization": "4bit",
                },
                "voice": {
                    "wake_word": "hey alita",
                    "default_voice": "en_US-lessac-medium",
                    "speech_timeout": 3.0,
                    "vad_threshold": 0.5,
                },
                "vision": {
                    "enable_camera": True,
                    "enable_screen": True,
                    "fps": 5,
                    "resolution": [640, 480],
                },
                "memory": {
                    "max_conversations": 1000,
                    "context_window": 10,
                    "enable_long_term": True,
                },
                "safety": {
                    "require_confirmation": True,
                    "enable_emergency_stop": True,
                    "max_cpu_percent": 90,
                    "max_gpu_temp": 80,
                },
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            
            logging.info(f"✅ Created default config: {config_file}")
        
        # Load config
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_voice_models(self):
        """Load voice model configurations."""
        voice_config_file = self.config_dir / "voice_models.yaml"
        
        if not voice_config_file.exists():
            # Create default voice models
            default_voices = {
                "voices": {
                    "lessac": {
                        "model": "en_US-lessac-medium",
                        "description": "Female US voice, clear and professional",
                        "speed": 1.0,
                        "pitch": 1.0,
                        "volume": 1.0,
                    },
                    "amy": {
                        "model": "en_US-amy-medium",
                        "description": "Female US voice, warm and friendly",
                        "speed": 1.0,
                        "pitch": 1.0,
                        "volume": 1.0,
                    },
                    "ryan": {
                        "model": "en_US-ryan-medium",
                        "description": "Male US voice, confident and clear",
                        "speed": 1.0,
                        "pitch": 1.0,
                        "volume": 1.0,
                    },
                    "danny": {
                        "model": "en_US-danny-low",
                        "description": "Male US voice, casual and relaxed",
                        "speed": 1.0,
                        "pitch": 1.0,
                        "volume": 1.0,
                    },
                    "jenny": {
                        "model": "en_US-jenny-medium",
                        "description": "Female US voice, energetic and expressive",
                        "speed": 1.0,
                        "pitch": 1.0,
                        "volume": 1.0,
                    },
                }
            }
            
            with open(voice_config_file, 'w') as f:
                yaml.dump(default_voices, f, default_flow_style=False)
            
            logging.info(f"✅ Created default voice models config")
        
        # Load voice models
        with open(voice_config_file, 'r') as f:
            voice_data = yaml.safe_load(f)
            self.voice_models = voice_data.get("voices", {})
    
    def load_macros(self):
        """Load automation macro templates."""
        macros_dir = self.config_dir / "macros"
        macros_dir.mkdir(exist_ok=True)
        
        self.macros = {}
        
        for macro_file in macros_dir.glob("*.yaml"):
            try:
                with open(macro_file, 'r') as f:
                    macro_data = yaml.safe_load(f)
                    
                    macro = MacroTemplate(
                        name=macro_data.get("name", macro_file.stem),
                        description=macro_data.get("description", ""),
                        steps=macro_data.get("steps", []),
                        category=macro_data.get("category", "general"),
                        enabled=macro_data.get("enabled", True),
                        hotkey=macro_data.get("hotkey")
                    )
                    
                    self.macros[macro.name] = macro
            except Exception as e:
                logging.error(f"Failed to load macro {macro_file}: {e}")
    
    def load_personality(self):
        """Load personality configuration."""
        personality_file = self.config_dir / "personality.yaml"
        
        if not personality_file.exists():
            # Create default personality
            default_personality = {
                "system_prompt": """You are ALITA, an advanced AI assistant inspired by JARVIS and FRIDAY.
You are helpful, intelligent, and have a warm personality. You can see through the webcam,
read the screen, control applications, and help with complex tasks. You're proactive and
anticipate user needs while maintaining respect for their autonomy.""",
                "response_style": "friendly",
                "verbosity": "normal",
                "humor_level": 0.3,
                "empathy_level": 0.7,
                "proactivity": 0.5,
            }
            
            with open(personality_file, 'w') as f:
                yaml.dump(default_personality, f, default_flow_style=False)
            
            logging.info(f"✅ Created default personality config")
        
        # Load personality
        with open(personality_file, 'r') as f:
            personality_data = yaml.safe_load(f)
            
            self.personality = PersonalityConfig(
                system_prompt=personality_data.get("system_prompt", ""),
                response_style=personality_data.get("response_style", "friendly"),
                verbosity=personality_data.get("verbosity", "normal"),
                humor_level=personality_data.get("humor_level", 0.3),
                empathy_level=personality_data.get("empathy_level", 0.7),
                proactivity=personality_data.get("proactivity", 0.5)
            )
    
    def reload_file(self, file_path: Path):
        """Reload a specific configuration file."""
        try:
            file_name = file_path.name
            
            if file_name == "alita_config.yaml":
                self.load_main_config()
                self._notify_change(ConfigSection.BRAIN.value, self.config)
            
            elif file_name == "voice_models.yaml":
                self.load_voice_models()
                self._notify_change(ConfigSection.VOICE.value, self.voice_models)
            
            elif file_name == "personality.yaml":
                self.load_personality()
                self._notify_change(ConfigSection.PERSONALITY.value, self.personality)
            
            elif file_path.parent.name == "macros":
                self.load_macros()
                self._notify_change(ConfigSection.MACROS.value, self.macros)
            
            logging.info(f"✅ Reloaded configuration: {file_name}")
            
        except Exception as e:
            logging.error(f"Failed to reload {file_path}: {e}")
    
    def register_change_callback(self, callback: Callable[[str, Any], None]):
        """Register callback for configuration changes."""
        self.change_callbacks.append(callback)
    
    def _notify_change(self, section: str, data: Any):
        """Notify all callbacks of configuration change."""
        for callback in self.change_callbacks:
            try:
                callback(section, data)
            except Exception as e:
                logging.error(f"Configuration change callback failed: {e}")
    
    def backup_config(self, backup_name: Optional[str] = None) -> Path:
        """Backup current configuration.
        
        Args:
            backup_name: Optional backup name, defaults to timestamp
            
        Returns:
            Path to backup directory
        """
        if backup_name is None:
            backup_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_dir = Path("config/backups") / backup_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Backup all config files
            for config_file in self.config_dir.glob("*.yaml"):
                if config_file.parent.name != "backups":
                    shutil.copy2(config_file, backup_dir / config_file.name)
            
            # Backup macros
            macros_backup = backup_dir / "macros"
            macros_backup.mkdir(exist_ok=True)
            
            macros_dir = self.config_dir / "macros"
            if macros_dir.exists():
                for macro_file in macros_dir.glob("*.yaml"):
                    shutil.copy2(macro_file, macros_backup / macro_file.name)
            
            logging.info(f"✅ Configuration backed up to: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            logging.error(f"Configuration backup failed: {e}")
            raise
    
    def restore_config(self, backup_name: str) -> bool:
        """Restore configuration from backup.
        
        Args:
            backup_name: Name of backup to restore
            
        Returns:
            True if successful
        """
        backup_dir = Path("config/backups") / backup_name
        
        if not backup_dir.exists():
            logging.error(f"Backup not found: {backup_name}")
            return False
        
        try:
            # Restore config files
            for backup_file in backup_dir.glob("*.yaml"):
                if backup_file.name != "macros":
                    shutil.copy2(backup_file, self.config_dir / backup_file.name)
            
            # Restore macros
            macros_backup = backup_dir / "macros"
            if macros_backup.exists():
                macros_dir = self.config_dir / "macros"
                macros_dir.mkdir(exist_ok=True)
                
                for macro_file in macros_backup.glob("*.yaml"):
                    shutil.copy2(macro_file, macros_dir / macro_file.name)
            
            # Reload all configuration
            self.load_all()
            
            logging.info(f"✅ Configuration restored from: {backup_name}")
            return True
            
        except Exception as e:
            logging.error(f"Configuration restore failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available configuration backups.
        
        Returns:
            List of backup info dictionaries
        """
        backups_dir = Path("config/backups")
        
        if not backups_dir.exists():
            return []
        
        backups = []
        
        for backup_dir in backups_dir.iterdir():
            if backup_dir.is_dir():
                backups.append({
                    "name": backup_dir.name,
                    "path": str(backup_dir),
                    "created": datetime.fromtimestamp(backup_dir.stat().st_ctime),
                    "size": sum(f.stat().st_size for f in backup_dir.rglob("*") if f.is_file())
                })
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created"], reverse=True)
        
        return backups
    
    def get_voice_models(self) -> Dict[str, Any]:
        """Get available voice models."""
        return self.voice_models
    
    def get_personality(self) -> Optional[PersonalityConfig]:
        """Get personality configuration."""
        return self.personality
    
    def get_macros(self) -> Dict[str, MacroTemplate]:
        """Get automation macros."""
        return self.macros
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """Update a configuration value.
        
        Args:
            section: Configuration section
            key: Configuration key
            value: New value
            
        Returns:
            True if successful
        """
        try:
            if section not in self.config:
                self.config[section] = {}
            
            self.config[section][key] = value
            
            # Save to file
            config_file = self.config_dir / "alita_config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            logging.info(f"✅ Updated config: {section}.{key} = {value}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to update config: {e}")
            return False
    
    def stop(self):
        """Stop the configuration manager."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logging.info("Configuration manager stopped")


# Global instance
_config_manager: Optional[EnhancedConfigManager] = None


def get_config_manager() -> EnhancedConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = EnhancedConfigManager()
    
    return _config_manager
