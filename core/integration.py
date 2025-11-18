"""
ALITA Integration Layer

Connects all components seamlessly with:
- Unified API for all features
- Event bus for inter-component communication
- Dependency injection
- Error handling and graceful degradation
- Performance monitoring

This is the central orchestrator that makes everything work together.
"""

import logging
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import asyncio
from pathlib import Path

# Import all core components
from .brain import Brain
from .voice import VoiceInterface
from .vision import VisionSystem
from .automation import SystemControl, AutomationExecutor
from .safety import get_safety_manager
from .enhanced_config import get_config_manager
from .enhanced_logging import get_logging_service
from .personality import get_personality_engine
from .proactive_agent import get_proactive_agent
from .image_generation import get_image_generator
from .file_search import get_file_search_engine
from .content_generation import get_content_generator
from .learning_system import get_learning_system


class EventBus:
    """Event bus for inter-component communication."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
    
    def publish(self, event_type: str, data: Any = None):
        """Publish event to subscribers."""
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logging.error(f"Event callback failed for {event_type}: {e}")
    
    async def publish_async(self, event_type: str, data: Any = None):
        """Publish event asynchronously."""
        if event_type in self.subscribers:
            tasks = []
            for callback in self.subscribers[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(data))
                else:
                    try:
                        callback(data)
                    except Exception as e:
                        logging.error(f"Event callback failed for {event_type}: {e}")
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)


class ALITAIntegration:
    """Main integration layer connecting all ALITA components."""
    
    def __init__(self):
        logging.info("🚀 Initializing ALITA Integration Layer...")
        
        # Event bus for communication
        self.event_bus = EventBus()
        
        # Core systems
        self.safety_manager = get_safety_manager()
        self.config_manager = get_config_manager()
        self.logging_service = get_logging_service()
        
        # AI components
        self.brain = None  # Lazy load
        self.personality = get_personality_engine()
        
        # I/O systems
        self.voice = None  # Lazy load
        self.vision = None  # Lazy load
        
        # Action systems
        self.automation = None  # Lazy load
        self.system_control = None  # Lazy load
        
        # Advanced features
        self.proactive_agent = get_proactive_agent()
        self.image_generator = get_image_generator()
        self.file_search = get_file_search_engine()
        self.content_generator = get_content_generator()
        self.learning_system = get_learning_system()
        
        # State
        self.initialized = False
        self.running = False
        
        # Setup event subscriptions
        self._setup_event_subscriptions()
        
        logging.info("✅ ALITA Integration Layer initialized")
    
    def _setup_event_subscriptions(self):
        """Setup event subscriptions between components."""
        
        # Voice command events
        self.event_bus.subscribe("voice_command", self._handle_voice_command)
        
        # Vision events
        self.event_bus.subscribe("face_detected", self._handle_face_detected)
        self.event_bus.subscribe("gesture_detected", self._handle_gesture_detected)
        
        # Proactive events
        self.event_bus.subscribe("greeting", self._handle_greeting)
        self.event_bus.subscribe("reminder", self._handle_reminder)
        self.event_bus.subscribe("suggestion", self._handle_suggestion)
        
        # Learning events
        self.event_bus.subscribe("user_feedback", self._handle_user_feedback)
        
        # Safety events
        self.safety_manager.emergency_stop.register_callback(self._handle_emergency_stop)
    
    async def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return
        
        try:
            logging.info("Initializing ALITA components...")
            
            # Initialize Brain (heavy operation)
            from .config import AIConfig
            config = AIConfig()
            self.brain = Brain(config)
            
            # Initialize Voice
            from .config import VoiceConfig
            voice_config = VoiceConfig()
            self.voice = VoiceInterface(voice_config)
            
            # Initialize Vision
            self.vision = VisionSystem()
            
            # Initialize Automation
            self.system_control = SystemControl()
            self.automation = AutomationExecutor(self.system_control)
            
            # Start proactive agent
            self.proactive_agent.start()
            
            self.initialized = True
            logging.info("✅ All ALITA components initialized")
            
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise
    
    async def start(self):
        """Start ALITA system."""
        if not self.initialized:
            await self.initialize()
        
        if self.running:
            return
        
        try:
            logging.info("🚀 Starting ALITA...")
            
            # Start voice listening
            if self.voice:
                self.voice.start_listening()
            
            # Start vision capture
            if self.vision:
                self.vision.start()
            
            self.running = True
            
            # Publish startup event
            self.event_bus.publish("system_started", {
                "timestamp": datetime.now().isoformat()
            })
            
            logging.info("✅ ALITA is now running!")
            
        except Exception as e:
            logging.error(f"Start failed: {e}")
            raise
    
    async def stop(self):
        """Stop ALITA system."""
        if not self.running:
            return
        
        try:
            logging.info("Stopping ALITA...")
            
            # Stop voice
            if self.voice:
                self.voice.stop()
            
            # Stop vision
            if self.vision:
                self.vision.stop()
            
            # Stop proactive agent
            self.proactive_agent.stop()
            
            self.running = False
            
            # Publish shutdown event
            self.event_bus.publish("system_stopped", {
                "timestamp": datetime.now().isoformat()
            })
            
            logging.info("✅ ALITA stopped")
            
        except Exception as e:
            logging.error(f"Stop failed: {e}")
    
    # Event handlers
    
    def _handle_voice_command(self, data: Dict[str, Any]):
        """Handle voice command event."""
        command = data.get("text", "")
        
        # Track in learning system
        self.learning_system.track_voice_command(command, "voice_input")
        
        # Process through personality
        # Then send to brain for execution
        logging.info(f"Voice command: {command}")
    
    def _handle_face_detected(self, data: Dict[str, Any]):
        """Handle face detection event."""
        expression = data.get("expression", "neutral")
        
        # Log facial expression for proactive agent
        self.proactive_agent.log_facial_expression(expression)
    
    def _handle_gesture_detected(self, data: Dict[str, Any]):
        """Handle gesture detection event."""
        gesture = data.get("gesture", "unknown")
        logging.info(f"Gesture detected: {gesture}")
    
    def _handle_greeting(self, message: str):
        """Handle greeting from proactive agent."""
        # Speak greeting through voice interface
        if self.voice:
            self.voice.speak(message)
    
    def _handle_reminder(self, message: str, event: Dict[str, Any]):
        """Handle reminder from proactive agent."""
        # Speak reminder
        if self.voice:
            self.voice.speak(message)
    
    def _handle_suggestion(self, message: str, data: Dict[str, Any]):
        """Handle suggestion from proactive agent."""
        # Present suggestion to user
        logging.info(f"Suggestion: {message}")
    
    def _handle_user_feedback(self, data: Dict[str, Any]):
        """Handle user feedback event."""
        feedback_type = data.get("type")
        feedback_value = data.get("value")
        
        # Route to appropriate system
        if feedback_type == "verbosity":
            response_length = data.get("response_length", 0)
            self.learning_system.adapt_verbosity(response_length, feedback_value)
    
    def _handle_emergency_stop(self):
        """Handle emergency stop event."""
        logging.critical("🚨 Emergency stop triggered - halting all operations")
        
        # Stop all active operations
        asyncio.create_task(self.stop())
    
    # Unified API methods
    
    async def process_command(self, command: str, source: str = "user") -> str:
        """Process command through complete pipeline.
        
        Args:
            command: User command
            source: Command source (voice, text, gui)
            
        Returns:
            Response string
        """
        try:
            # Log activity
            self.proactive_agent.log_activity("command", {"source": source})
            
            # Process through personality for context
            processed_command = self.personality.process_response(
                command,
                user_input=command,
                context={"source": source}
            )
            
            # Send to brain for reasoning and execution
            if self.brain:
                response = await self.brain.process_async(processed_command)
            else:
                response = "Brain not initialized"
            
            # Process response through personality
            final_response = self.personality.process_response(
                response,
                user_input=command,
                context={"source": source}
            )
            
            # Log performance
            self.logging_service.log_performance(0.5, 50, "command_processing")
            
            return final_response
            
        except Exception as e:
            logging.error(f"Command processing failed: {e}")
            return f"I encountered an error: {str(e)}"
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status."""
        return {
            "initialized": self.initialized,
            "running": self.running,
            "components": {
                "brain": self.brain is not None,
                "voice": self.voice is not None and self.voice._running,
                "vision": self.vision is not None and self.vision._running,
                "proactive_agent": self.proactive_agent.running,
            },
            "safety": {
                "emergency_stop": self.safety_manager.emergency_stop.is_stopped()
            },
            "learning": self.learning_system.get_learning_summary()
        }


# Global instance
_alita_integration: Optional[ALITAIntegration] = None


def get_alita_integration() -> ALITAIntegration:
    """Get global ALITA integration instance."""
    global _alita_integration
    
    if _alita_integration is None:
        _alita_integration = ALITAIntegration()
    
    return _alita_integration


# Convenience function for easy access
async def start_alita():
    """Start ALITA system."""
    integration = get_alita_integration()
    await integration.start()
    return integration


async def stop_alita():
    """Stop ALITA system."""
    integration = get_alita_integration()
    await integration.stop()
