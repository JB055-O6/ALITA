"""
ALITA Integration Orchestrator
Seamlessly connects all backend systems with the frontend GUI
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

try:
    from PyQt6.QtCore import QObject, pyqtSignal, QTimer
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    QObject = object
    def pyqtSignal(*args, **kwargs):
        return None


class IntegrationEvent(Enum):
    """Events that can be triggered in the system"""
    CHAT_MESSAGE = "chat_message"
    VOICE_INPUT = "voice_input"
    VISION_DETECTED = "vision_detected"
    GESTURE_DETECTED = "gesture_detected"
    ACTION_REQUESTED = "action_requested"
    ACTION_COMPLETED = "action_completed"
    ACTION_FAILED = "action_failed"
    SYSTEM_STATUS_CHANGED = "system_status_changed"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class IntegrationMessage:
    """Message passed between systems"""
    event_type: IntegrationEvent
    source: str
    data: Any
    timestamp: float
    metadata: Dict[str, Any]


class IntegrationOrchestrator(QObject if PYQT_AVAILABLE else object):
    """
    Central orchestrator connecting backend AI systems with frontend GUI
    
    Responsibilities:
    - Route messages between systems
    - Coordinate multi-modal inputs
    - Manage action approval workflow
    - Synchronize state across components
    - Handle errors and recovery
    """
    
    # Signals for Qt integration
    if PYQT_AVAILABLE:
        chat_response_ready = pyqtSignal(str)
        vision_result_ready = pyqtSignal(dict)
        action_requires_approval = pyqtSignal(str, str, dict)
        system_status_updated = pyqtSignal(dict)
        error_occurred = pyqtSignal(str, str)
    
    def __init__(self):
        if PYQT_AVAILABLE:
            super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Integration Orchestrator...")
        
        # Backend systems (will be injected)
        self.brain = None
        self.voice = None
        self.vision = None
        self.fusion_engine = None
        self.query_engine = None
        self.personality = None
        self.emotion_detector = None
        self.proactive_agent = None
        self.system_controller = None
        
        # Frontend systems (will be injected)
        self.gui = None
        self.control_dashboard = None
        
        # State management
        self.active_actions: Dict[str, Any] = {}
        self.pending_approvals: Dict[str, Dict[str, Any]] = {}
        self.event_handlers: Dict[IntegrationEvent, List[Callable]] = {}
        
        # Processing queue
        self.message_queue = asyncio.Queue()
        self.processing_task = None
        
        self.logger.info("✓ Integration Orchestrator initialized")
    
    def inject_backend_systems(
        self,
        brain=None,
        voice=None,
        vision=None,
        fusion_engine=None,
        query_engine=None,
        personality=None,
        emotion_detector=None,
        proactive_agent=None,
        system_controller=None
    ):
        """Inject backend systems"""
        self.brain = brain
        self.voice = voice
        self.vision = vision
        self.fusion_engine = fusion_engine
        self.query_engine = query_engine
        self.personality = personality
        self.emotion_detector = emotion_detector
        self.proactive_agent = proactive_agent
        self.system_controller = system_controller
        
        self.logger.info("✓ Backend systems injected")
        self._setup_backend_callbacks()
    
    def inject_frontend_systems(self, gui=None, control_dashboard=None):
        """Inject frontend systems"""
        self.gui = gui
        self.control_dashboard = control_dashboard
        
        self.logger.info("✓ Frontend systems injected")
        self._setup_frontend_callbacks()
    
    def _setup_backend_callbacks(self):
        """Setup callbacks for backend systems"""
        # Voice callbacks
        if self.voice:
            try:
                self.voice.on_speech_detected = self._on_voice_input
            except:
                pass
        
        # Vision callbacks
        if self.vision:
            try:
                self.vision.on_object_detected = self._on_vision_detection
            except:
                pass
        
        # Fusion engine callbacks
        if self.fusion_engine:
            try:
                from alita.core.fusion import ModalityType
                self.fusion_engine.register_callback(
                    ModalityType.VOICE,
                    self._on_fusion_voice
                )
                self.fusion_engine.register_callback(
                    ModalityType.VISION,
                    self._on_fusion_vision
                )
                self.fusion_engine.register_callback(
                    ModalityType.GESTURE,
                    self._on_fusion_gesture
                )
            except Exception as e:
                self.logger.warning(f"Could not setup fusion callbacks: {e}")
    
    def _setup_frontend_callbacks(self):
        """Setup callbacks for frontend systems"""
        if not PYQT_AVAILABLE:
            return
        
        # GUI callbacks
        if self.gui and hasattr(self.gui, 'chat_widget'):
            try:
                self.gui.chat_widget.send_button.clicked.connect(
                    self._on_gui_chat_message
                )
            except:
                pass
        
        # Control dashboard callbacks
        if self.control_dashboard:
            try:
                self.control_dashboard.action_approved.connect(
                    self._on_action_approved
                )
                self.control_dashboard.action_rejected.connect(
                    self._on_action_rejected
                )
                self.control_dashboard.action_undone.connect(
                    self._on_action_undone
                )
            except:
                pass
    
    async def start(self):
        """Start the orchestrator"""
        self.logger.info("Starting Integration Orchestrator...")
        
        # Start message processing
        self.processing_task = asyncio.create_task(self._process_messages())
        
        self.logger.info("✓ Integration Orchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        self.logger.info("Stopping Integration Orchestrator...")
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("✓ Integration Orchestrator stopped")
    
    async def _process_messages(self):
        """Process messages from the queue"""
        while True:
            try:
                message = await self.message_queue.get()
                await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    async def _handle_message(self, message: IntegrationMessage):
        """Handle a single message"""
        self.logger.debug(f"Handling message: {message.event_type}")
        
        # Route to appropriate handler
        if message.event_type == IntegrationEvent.CHAT_MESSAGE:
            await self._handle_chat_message(message)
        elif message.event_type == IntegrationEvent.VOICE_INPUT:
            await self._handle_voice_input(message)
        elif message.event_type == IntegrationEvent.VISION_DETECTED:
            await self._handle_vision_detection(message)
        elif message.event_type == IntegrationEvent.ACTION_REQUESTED:
            await self._handle_action_request(message)
        
        # Call registered handlers
        if message.event_type in self.event_handlers:
            for handler in self.event_handlers[message.event_type]:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"Handler error: {e}")
    
    async def _handle_chat_message(self, message: IntegrationMessage):
        """Handle chat message from user"""
        text = message.data
        self.logger.info(f"Processing chat: {text}")
        
        try:
            # Process with brain if available
            if self.brain and hasattr(self.brain, 'process_input'):
                response = await asyncio.to_thread(
                    self.brain.process_input,
                    text
                )
            elif self.brain and hasattr(self.brain, 'think'):
                # Try alternative method
                response = await asyncio.to_thread(
                    self.brain.think,
                    text
                )
            else:
                # Fallback response
                response = f"I received your message: '{text}'. I'm currently in demo mode with limited AI capabilities. Full brain initialization requires additional configuration."
            
            # Add personality if available
            if self.personality and hasattr(self.personality, 'apply_personality'):
                try:
                    response = await asyncio.to_thread(
                        self.personality.apply_personality,
                        response
                    )
                except:
                    pass  # Keep original response if personality fails
            
            # Send to GUI
            if PYQT_AVAILABLE and self.chat_response_ready:
                self.chat_response_ready.emit(response)
            
            if self.gui and hasattr(self.gui, 'chat_widget'):
                self.gui.chat_widget.receive_response(response)
        
        except Exception as e:
            self.logger.error(f"Chat processing error: {e}")
            error_msg = "I encountered an error processing your request. I'm still learning and improving!"
            if self.gui and hasattr(self.gui, 'chat_widget'):
                self.gui.chat_widget.receive_response(error_msg)
    
    async def _handle_voice_input(self, message: IntegrationMessage):
        """Handle voice input"""
        text = message.data
        self.logger.info(f"Voice input: {text}")
        
        # Convert to chat message
        chat_message = IntegrationMessage(
            event_type=IntegrationEvent.CHAT_MESSAGE,
            source="voice",
            data=text,
            timestamp=message.timestamp,
            metadata=message.metadata
        )
        await self.message_queue.put(chat_message)
    
    async def _handle_vision_detection(self, message: IntegrationMessage):
        """Handle vision detection"""
        detection = message.data
        self.logger.info(f"Vision detection: {detection}")
        
        # Send to GUI
        if PYQT_AVAILABLE and self.vision_result_ready:
            self.vision_result_ready.emit(detection)
    
    async def _handle_action_request(self, message: IntegrationMessage):
        """Handle action request"""
        action_data = message.data
        action_id = str(uuid.uuid4())
        
        self.logger.info(f"Action requested: {action_data.get('name')}")
        
        # Check if approval required
        requires_approval = action_data.get('requires_approval', False)
        
        if requires_approval and self.control_dashboard:
            # Add to approval queue
            from alita.interface.control_dashboard import ActionPriority
            
            priority_map = {
                'low': ActionPriority.LOW,
                'medium': ActionPriority.MEDIUM,
                'high': ActionPriority.HIGH,
                'critical': ActionPriority.CRITICAL
            }
            
            priority = priority_map.get(
                action_data.get('priority', 'medium'),
                ActionPriority.MEDIUM
            )
            
            self.control_dashboard.add_action(
                action_id=action_id,
                name=action_data.get('name', 'Unknown Action'),
                description=action_data.get('description', ''),
                priority=priority,
                requires_approval=True,
                reversible=action_data.get('reversible', True),
                metadata=action_data.get('metadata', {})
            )
            
            self.pending_approvals[action_id] = action_data
        else:
            # Execute immediately
            await self._execute_action(action_id, action_data)
    
    async def _execute_action(self, action_id: str, action_data: Dict[str, Any]):
        """Execute an action"""
        self.logger.info(f"Executing action: {action_id}")
        
        try:
            # Execute based on action type
            action_type = action_data.get('type')
            result = None
            
            if action_type == 'query':
                if self.query_engine:
                    result = await asyncio.to_thread(
                        self.query_engine.query,
                        action_data.get('query', '')
                    )
            elif action_type == 'system_command':
                if self.system_controller:
                    result = await asyncio.to_thread(
                        self.system_controller.execute_command,
                        action_data.get('command', '')
                    )
            
            # Mark as completed
            if self.control_dashboard:
                self.control_dashboard.mark_action_completed(action_id, result)
            
            self.logger.info(f"✓ Action completed: {action_id}")
        
        except Exception as e:
            self.logger.error(f"Action execution error: {e}")
            if self.control_dashboard:
                self.control_dashboard.mark_action_failed(action_id, str(e))
    
    def _on_gui_chat_message(self):
        """Handle chat message from GUI"""
        if not self.gui or not hasattr(self.gui, 'chat_widget'):
            return
        
        text = self.gui.chat_widget.input_field.toPlainText().strip()
        if not text:
            return
        
        # Clear input
        self.gui.chat_widget.input_field.clear()
        
        # Add to chat display
        self.gui.chat_widget.chat_area.append(
            '<div style="margin: 10px; padding: 15px; '
            'background: rgba(0, 255, 255, 0.1); '
            'border-left: 3px solid rgba(0, 255, 255, 0.8); '
            'border-radius: 8px;">'
            f'<span style="color: rgba(0, 255, 255, 1.0); font-weight: bold;">You:</span> '
            f'<span style="color: rgba(255, 255, 255, 0.9);">{text}</span></div>'
        )
        
        # Create message
        import time
        message = IntegrationMessage(
            event_type=IntegrationEvent.CHAT_MESSAGE,
            source="gui",
            data=text,
            timestamp=time.time(),
            metadata={}
        )
        
        # Add to queue
        asyncio.create_task(self.message_queue.put(message))
    
    def _on_voice_input(self, text: str):
        """Handle voice input from voice system"""
        import time
        message = IntegrationMessage(
            event_type=IntegrationEvent.VOICE_INPUT,
            source="voice",
            data=text,
            timestamp=time.time(),
            metadata={}
        )
        asyncio.create_task(self.message_queue.put(message))
    
    def _on_vision_detection(self, detection: Dict[str, Any]):
        """Handle vision detection"""
        import time
        message = IntegrationMessage(
            event_type=IntegrationEvent.VISION_DETECTED,
            source="vision",
            data=detection,
            timestamp=time.time(),
            metadata={}
        )
        asyncio.create_task(self.message_queue.put(message))
    
    def _on_fusion_voice(self, modal_input):
        """Handle fusion voice input"""
        self._on_voice_input(str(modal_input.data))
    
    def _on_fusion_vision(self, modal_input):
        """Handle fusion vision input"""
        self._on_vision_detection({"data": modal_input.data})
    
    def _on_fusion_gesture(self, modal_input):
        """Handle fusion gesture input"""
        self.logger.info(f"Gesture detected: {modal_input.data}")
    
    def _on_action_approved(self, action_id: str):
        """Handle action approval"""
        self.logger.info(f"Action approved: {action_id}")
        
        if action_id in self.pending_approvals:
            action_data = self.pending_approvals.pop(action_id)
            asyncio.create_task(self._execute_action(action_id, action_data))
    
    def _on_action_rejected(self, action_id: str):
        """Handle action rejection"""
        self.logger.info(f"Action rejected: {action_id}")
        
        if action_id in self.pending_approvals:
            self.pending_approvals.pop(action_id)
    
    def _on_action_undone(self, action_id: str):
        """Handle action undo"""
        self.logger.info(f"Action undone: {action_id}")
    
    def register_event_handler(
        self,
        event_type: IntegrationEvent,
        handler: Callable
    ):
        """Register an event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def send_message(self, message: IntegrationMessage):
        """Send a message to the orchestrator"""
        asyncio.create_task(self.message_queue.put(message))
