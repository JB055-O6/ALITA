"""
Advanced Configuration Manager with Dynamic Updates
Features:
- Dynamic configuration management
- Hot reloading
- Environment management
- Secret management
- Configuration validation
"""

import json
import yaml
import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import base64
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _HAS_CRYPTO = True
except Exception:  # pragma: no cover - optional dependency
    Fernet = None
    hashes = None
    PBKDF2HMAC = None
    _HAS_CRYPTO = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    _HAS_WATCHDOG = True
except Exception:  # pragma: no cover - optional
    Observer = None
    FileSystemEventHandler = object
    _HAS_WATCHDOG = False

@dataclass
class ConfigValue:
    """Configuration value with metadata."""
    value: Any
    source: str
    last_updated: datetime
    is_secret: bool = False
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)

class ConfigValidator:
    """Configuration validation."""
    
    def validate_type(self, value: Any, expected_type: type) -> bool:
        """Validate value type."""
        return isinstance(value, expected_type)
    
    def validate_range(self,
                      value: Union[int, float],
                      min_value: Optional[Union[int, float]] = None,
                      max_value: Optional[Union[int, float]] = None) -> bool:
        """Validate numeric range."""
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True
    
    def validate_pattern(self, value: str, pattern: str) -> bool:
        """Validate string pattern."""
        import re
        return bool(re.match(pattern, value))
    
    def validate_enum(self, value: Any, allowed_values: List[Any]) -> bool:
        """Validate enum values."""
        return value in allowed_values
    
    def validate_custom(self,
                       value: Any,
                       validation_func: callable) -> bool:
        """Validate with custom function."""
        try:
            return validation_func(value)
        except Exception:
            return False

class SecretManager:
    """Secret management with encryption."""
    
    def __init__(self, key: Optional[bytes] = None):
        # If cryptography is available, use Fernet. Otherwise provide a safe-ish fallback
        if _HAS_CRYPTO and Fernet is not None:
            self.key = key or self._generate_key()
            self.fernet = Fernet(self.key)
            self._use_crypto = True
        else:
            self.key = None
            self.fernet = None
            self._use_crypto = False
    
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        if not _HAS_CRYPTO or PBKDF2HMAC is None:
            # Fallback: generate a random base64 key (not PBKDF2-derived)
            return base64.urlsafe_b64encode(os.urandom(32))

        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = base64.urlsafe_b64encode(kdf.derive(os.urandom(32)))
        return key
    
    def encrypt(self, value: str) -> str:
        """Encrypt secret value."""
        if self._use_crypto and self.fernet is not None:
            return self.fernet.encrypt(value.encode()).decode()
        # Fallback: base64 encode (note: not secure encryption)
        return base64.urlsafe_b64encode(value.encode()).decode()
    
    def decrypt(self, encrypted: str) -> str:
        """Decrypt secret value."""
        if self._use_crypto and self.fernet is not None:
            return self.fernet.decrypt(encrypted.encode()).decode()
        # Fallback: base64 decode
        try:
            return base64.urlsafe_b64decode(encrypted.encode()).decode()
        except Exception:
            return ""

class ConfigWatcher(FileSystemEventHandler):
    """Configuration file watcher."""
    
    def __init__(self, config_manager: 'ConfigurationManager'):
        self.config_manager = config_manager
        # If watchdog not available, observer is a no-op placeholder
        if _HAS_WATCHDOG and Observer is not None:
            self.observer = Observer()
        else:
            self.observer = None
    
    def start(self, path: Path):
        """Start watching config directory."""
        if self.observer is None:
            return
        self.observer.schedule(self, str(path), recursive=True)
        self.observer.start()
    
    def stop(self):
        """Stop watching config directory."""
        if self.observer is None:
            return
        self.observer.stop()
        self.observer.join()
    
    def on_modified(self, event):
        """Handle file modification."""
        if event.src_path.endswith(('.json', '.yaml', '.yml')):
            asyncio.create_task(
                self.config_manager.reload_config(Path(event.src_path))
            )

class ConfigurationManager:
    """Advanced configuration management."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.config: Dict[str, ConfigValue] = {}
        self.secret_manager = SecretManager()
        self.validator = ConfigValidator()
        self.watcher = ConfigWatcher(self)
        
        # Create config directory if not exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Start file watcher
        self.watcher.start(self.config_dir)
    
    async def initialize(self):
        """Initialize configuration."""
        try:
            # Load all config files
            config_files = list(self.config_dir.glob("**/*.json"))
            config_files.extend(self.config_dir.glob("**/*.yaml"))
            config_files.extend(self.config_dir.glob("**/*.yml"))
            
            for config_file in config_files:
                await self.load_config(config_file)
            
            # Load environment variables
            await self.load_environment()
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration initialization failed: {str(e)}")
            return False
    
    async def load_config(self, config_file: Path) -> bool:
        """Load configuration from file."""
        try:
            with open(config_file) as f:
                if config_file.suffix == '.json':
                    config_data = json.load(f)
                else:
                    config_data = yaml.safe_load(f)
            
            # Process configuration
            for key, value in config_data.items():
                await self.set_config(
                    key,
                    value.get('value'),
                    source=str(config_file),
                    is_secret=value.get('is_secret', False),
                    validation_rules=value.get('validation', [])
                )
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration load failed: {str(e)}")
            return False
    
    async def load_environment(self) -> bool:
        """Load configuration from environment."""
        try:
            for key, value in os.environ.items():
                if key.startswith('CONFIG_'):
                    config_key = key[7:].lower()
                    await self.set_config(
                        config_key,
                        value,
                        source='environment',
                        is_secret=key.endswith('_SECRET')
                    )
            
            return True
            
        except Exception as e:
            logging.error(f"Environment load failed: {str(e)}")
            return False
    
    async def set_config(self,
                        key: str,
                        value: Any,
                        source: str = 'manual',
                        is_secret: bool = False,
                        validation_rules: List[Dict[str, Any]] = None) -> bool:
        """Set configuration value."""
        try:
            # Validate value
            if validation_rules:
                if not await self._validate_value(value, validation_rules):
                    raise ValueError(f"Validation failed for {key}")
            
            # Encrypt if secret
            if is_secret:
                value = self.secret_manager.encrypt(str(value))
            
            # Store configuration
            self.config[key] = ConfigValue(
                value=value,
                source=source,
                last_updated=datetime.now(),
                is_secret=is_secret,
                validation_rules=validation_rules or []
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration set failed: {str(e)}")
            return False
    
    async def get_config(self,
                        key: str,
                        default: Any = None) -> Any:
        """Get configuration value."""
        try:
            if key not in self.config:
                return default
            
            config = self.config[key]
            value = config.value
            
            # Decrypt if secret
            if config.is_secret:
                value = self.secret_manager.decrypt(value)
            
            return value
            
        except Exception as e:
            logging.error(f"Configuration get failed: {str(e)}")
            return default
    
    async def delete_config(self, key: str) -> bool:
        """Delete configuration value."""
        try:
            if key in self.config:
                del self.config[key]
                return True
            return False
        except Exception as e:
            logging.error(f"Configuration delete failed: {str(e)}")
            return False
    
    async def reload_config(self, config_file: Path) -> bool:
        """Reload configuration from file."""
        try:
            # Remove existing configs from this file
            for key, config in list(self.config.items()):
                if config.source == str(config_file):
                    await self.delete_config(key)
            
            # Load new config
            return await self.load_config(config_file)
            
        except Exception as e:
            logging.error(f"Configuration reload failed: {str(e)}")
            return False
    
    async def save_config(self) -> bool:
        """Save configuration to files."""
        try:
            # Group configs by source
            configs_by_source = {}
            for key, config in self.config.items():
                if config.source not in ('environment', 'manual'):
                    if config.source not in configs_by_source:
                        configs_by_source[config.source] = {}
                    
                    configs_by_source[config.source][key] = {
                        'value': config.value,
                        'is_secret': config.is_secret,
                        'validation': config.validation_rules
                    }
            
            # Save each file
            for source, configs in configs_by_source.items():
                path = Path(source)
                if path.suffix == '.json':
                    with open(path, 'w') as f:
                        json.dump(configs, f, indent=2)
                else:
                    with open(path, 'w') as f:
                        yaml.dump(configs, f)
            
            return True
            
        except Exception as e:
            logging.error(f"Configuration save failed: {str(e)}")
            return False
    
    async def _validate_value(self,
                            value: Any,
                            rules: List[Dict[str, Any]]) -> bool:
        """Validate value against rules."""
        try:
            for rule in rules:
                rule_type = rule.get('type')
                
                if rule_type == 'type':
                    type_name = rule.get('expected_type')
                    if not self.validator.validate_type(
                        value, eval(type_name)
                    ):
                        return False
                
                elif rule_type == 'range':
                    if not self.validator.validate_range(
                        value,
                        rule.get('min'),
                        rule.get('max')
                    ):
                        return False
                
                elif rule_type == 'pattern':
                    if not self.validator.validate_pattern(
                        value,
                        rule['pattern']
                    ):
                        return False
                
                elif rule_type == 'enum':
                    if not self.validator.validate_enum(
                        value,
                        rule['allowed_values']
                    ):
                        return False
                
                elif rule_type == 'custom':
                    if not self.validator.validate_custom(
                        value,
                        eval(rule['function'])
                    ):
                        return False
            
            return True
            
        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            return False
    
    async def get_config_info(self) -> Dict[str, Any]:
        """Get configuration information."""
        try:
            info = {
                'total_configs': len(self.config),
                'sources': set(),
                'secret_count': 0,
                'last_updated': None
            }
            
            for config in self.config.values():
                info['sources'].add(config.source)
                if config.is_secret:
                    info['secret_count'] += 1
                if not info['last_updated'] or \
                   config.last_updated > info['last_updated']:
                    info['last_updated'] = config.last_updated
            
            info['sources'] = list(info['sources'])
            return info
            
        except Exception as e:
            logging.error(f"Configuration info failed: {str(e)}")
            return {}
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.watcher.stop()
        except:
            pass