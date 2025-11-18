"""alita package convenience exports.

This module exposes a small, safe set of commonly used symbols so
callers can `from alita import Brain, VoiceInterface, VisionSystem`.

Imports are defensive to avoid breaking import-time for users who
don't have optional native dependencies installed.
"""

__version__ = "0.1.0"

# Defensive, best-effort imports of commonly used classes
try:
    from .core.brain import Brain
except Exception:
    Brain = None

try:
    from .core.voice import VoiceInterface
except Exception:
    VoiceInterface = None

try:
    from .core.vision import VisionSystem
except Exception:
    VisionSystem = None

try:
    from .core.automation import SystemControl
except Exception:
    SystemControl = None

try:
    from .core.service_manager import ServiceManager
except Exception:
    ServiceManager = None

__all__ = [
    "__version__",
    "Brain",
    "VoiceInterface",
    "VisionSystem",
    "SystemControl",
    "ServiceManager",
]


def create_default_system(config=None):
    """Convenience helper: create main subsystems using provided config.

    Returns a dict with keys: brain, voice, vision, system, services.
    Each value may be None if the underlying class couldn't be imported.
    """
    system = {}
    try:
        system["brain"] = Brain(config.ai) if Brain is not None and config is not None else None
    except Exception:
        system["brain"] = None
    try:
        system["voice"] = VoiceInterface(config.voice) if VoiceInterface is not None and config is not None else None
    except Exception:
        system["voice"] = None
    try:
        system["vision"] = VisionSystem(config.vision) if VisionSystem is not None and config is not None else None
    except Exception:
        system["vision"] = None
    try:
        system["system"] = SystemControl(config.system) if SystemControl is not None and config is not None else None
    except Exception:
        system["system"] = None
    try:
        system["services"] = ServiceManager(config.services) if ServiceManager is not None and config is not None else None
    except Exception:
        system["services"] = None
    return system
