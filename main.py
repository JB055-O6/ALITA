"""
ALITA Main Application
Integrates all backend systems with the GUI frontend
"""

import sys
import asyncio
import logging
from pathlib import Path

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import core backend systems
from alita.core.brain import Brain
from alita.core.voice import VoiceInterface
from alita.core.vision import VisionSystem
from alita.core.system_controller import SystemController
from alita.core.query_engine import QueryEngine, DatabaseType
from alita.core.fusion import FusionEngine, ModalityType
from alita.core.personality import PersonalityEngine
from alita.core.emotion import EmotionDetector
from alita.core.proactive_agent import ProactiveAgent

# Import GUI frontend
from alita.interface.main_window import AlitaGUI, launch_gui
from alita.interface.control_dashboard import ControlDashboard

# Import integration orchestrator
from alita.integration_orchestrator import IntegrationOrchestrator

# Import system bridge
from alita.system_bridge import SystemBridge, ensure_system_integrity

try:
    from PyQt6.QtWidgets import QApplication
    from qasync import QEventLoop
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt6 not available. Install with: pip install PyQt6 qasync")


class ALITASystem:
    """
    Main ALITA System - Integrates Backend with Frontend
    
    This class connects all backend AI systems with the GUI frontend,
    providing a unified interface for user interaction.
    """
    
    def __init__(self):
        """Initialize ALITA system"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ALITA System...")
        
        # Backend systems
        self.brain = None
        self.voice = None
        self.vision = None
        self.system_controller = None
        self.query_engine = None
        self.fusion_engine = None
        self.personality = None
        self.emotion_detector = None
        self.proactive_agent = None
        
        # Frontend systems
        self.gui = None
        self.control_dashboard = None
        
        # Integration orchestrator
        self.orchestrator = IntegrationOrchestrator()
        
        # System bridge
        self.bridge = SystemBridge()
        
        # Initialize systems
        self._init_backend()
        self._init_frontend()
        self._connect_systems()
        
        # Ensure system integrity with fallbacks
        ensure_system_integrity(self)
        
        # Establish all bridges
        self.bridge.establish_all_bridges(self)
        
        self.logger.info("ALITA System initialized successfully!")
    
    def _init_backend(self):
        """Initialize all backend systems"""
        self.logger.info("Initializing backend systems...")
        
        try:
            # Load configuration
            try:
                from alita.config import AIConfig
                config = AIConfig()
            except:
                config = None
                self.logger.warning("Could not load AIConfig, using defaults")
            
            # Core AI brain
            self.logger.info("Loading Brain...")
            try:
                # Always create brain - it will use defaults if config is None
                self.brain = Brain(config=config)
                self.logger.info("✓ Brain initialized successfully!")
            except Exception as e:
                self.logger.error(f"Brain initialization failed: {e}")
                # Create minimal fallback brain
                try:
                    self.brain = Brain(config=None)
                    self.logger.info("✓ Brain initialized with defaults")
                except:
                    self.brain = None
                    self.logger.error("✗ Brain initialization completely failed")
            
            # Voice interface
            self.logger.info("Loading Voice Interface...")
            try:
                from alita.core.voice import VoiceConfig
                voice_config = VoiceConfig()
                self.voice = VoiceInterface(config=voice_config)
            except Exception as e:
                self.logger.warning(f"Voice interface skipped: {e}")
                self.voice = None
            
            # Vision system
            self.logger.info("Loading Vision System...")
            try:
                self.vision = VisionSystem()
            except Exception as e:
                self.logger.warning(f"Vision system skipped: {e}")
                self.vision = None
            
            # System controller
            self.logger.info("Loading System Controller...")
            try:
                self.system_controller = SystemController()
            except Exception as e:
                self.logger.warning(f"System controller skipped: {e}")
                self.system_controller = None
            
            # Query engine
            self.logger.info("Loading Query Engine...")
            try:
                self.query_engine = QueryEngine(device="auto", load_model=False)
            except Exception as e:
                self.logger.warning(f"Query engine skipped: {e}")
                self.query_engine = None
            
            # Fusion engine
            self.logger.info("Loading Fusion Engine...")
            try:
                if self.voice or self.vision:
                    self.fusion_engine = FusionEngine(
                        voice_interface=self.voice,
                        vision_system=self.vision
                    )
                else:
                    self.fusion_engine = None
                    self.logger.warning("Fusion engine skipped - no voice/vision")
            except Exception as e:
                self.logger.warning(f"Fusion engine skipped: {e}")
                self.fusion_engine = None
            
            # Personality engine
            self.logger.info("Loading Personality Engine...")
            try:
                self.personality = PersonalityEngine()
            except Exception as e:
                self.logger.warning(f"Personality engine skipped: {e}")
                self.personality = None
            
            # Emotion detector
            self.logger.info("Loading Emotion Detector...")
            try:
                self.emotion_detector = EmotionDetector()
            except Exception as e:
                self.logger.warning(f"Emotion detector skipped: {e}")
                self.emotion_detector = None
            
            # Proactive agent
            self.logger.info("Loading Proactive Agent...")
            try:
                if self.brain:
                    self.proactive_agent = ProactiveAgent(brain=self.brain)
                else:
                    self.proactive_agent = None
                    self.logger.warning("Proactive agent skipped - no brain")
            except Exception as e:
                self.logger.warning(f"Proactive agent skipped: {e}")
                self.proactive_agent = None
            
            self.logger.info("✓ Backend systems initialization complete!")
            
        except Exception as e:
            self.logger.error(f"Backend initialization error: {e}")
            self.logger.info("Some systems may not be available")
    
    def _init_frontend(self):
        """Initialize frontend GUI"""
        if not PYQT_AVAILABLE:
            self.logger.warning("PyQt6 not available - GUI disabled")
            return
        
        self.logger.info("Initializing frontend GUI...")
        
        try:
            # Main GUI window
            self.gui = AlitaGUI()
            
            # Control dashboard
            self.control_dashboard = ControlDashboard()
            
            self.logger.info("✓ Frontend GUI initialized!")
            
        except Exception as e:
            self.logger.error(f"Frontend initialization error: {e}")
    
    def _connect_systems(self):
        """Connect backend systems to frontend via orchestrator"""
        self.logger.info("Connecting backend to frontend...")
        
        try:
            # Inject backend systems into orchestrator
            self.orchestrator.inject_backend_systems(
                brain=self.brain,
                voice=self.voice,
                vision=self.vision,
                fusion_engine=self.fusion_engine,
                query_engine=self.query_engine,
                personality=self.personality,
                emotion_detector=self.emotion_detector,
                proactive_agent=self.proactive_agent,
                system_controller=self.system_controller
            )
            
            # Inject frontend systems into orchestrator
            self.orchestrator.inject_frontend_systems(
                gui=self.gui,
                control_dashboard=self.control_dashboard
            )
            
            # Connect vision widget controls
            if self.vision and self.gui and hasattr(self.gui, 'vision_widget'):
                self.gui.vision_widget.start_button.clicked.connect(
                    lambda: self.vision.start()
                )
                self.gui.vision_widget.stop_button.clicked.connect(
                    lambda: self.vision.stop()
                )
            
            self.logger.info("✓ Systems connected via orchestrator!")
            
        except Exception as e:
            self.logger.error(f"System connection error: {e}")
    

    
    async def run(self):
        """Run ALITA system"""
        self.logger.info("Starting ALITA...")
        
        if not PYQT_AVAILABLE:
            self.logger.error("Cannot start - PyQt6 not available")
            return
        
        # Start orchestrator
        await self.orchestrator.start()
        
        # Show GUI
        if self.gui:
            self.gui.show()
        
        # Show control dashboard in separate window
        if self.control_dashboard:
            self.control_dashboard.show()
        
        # Start backend systems
        if self.voice:
            try:
                self.voice.start()
            except Exception as e:
                self.logger.warning(f"Voice system start failed: {e}")
        
        if self.proactive_agent:
            try:
                self.proactive_agent.start()
            except Exception as e:
                self.logger.warning(f"Proactive agent start failed: {e}")
        
        self.logger.info("✓ ALITA is now ALIVE and running!")
        self.logger.info("=" * 60)
        self.logger.info("🎉 All systems operational!")
        self.logger.info("=" * 60)
        
        # Keep running
        await asyncio.Event().wait()
    
    async def shutdown(self):
        """Shutdown ALITA system"""
        self.logger.info("Shutting down ALITA...")
        
        # Stop orchestrator
        await self.orchestrator.stop()
        
        if self.voice:
            try:
                self.voice.stop()
            except:
                pass
        
        if self.vision:
            try:
                self.vision.stop()
            except:
                pass
        
        if self.fusion_engine:
            try:
                self.fusion_engine.shutdown()
            except:
                pass
        
        if self.proactive_agent:
            try:
                self.proactive_agent.stop()
            except:
                pass
        
        self.logger.info("✓ ALITA shutdown complete")


async def main():
    """Main entry point"""
    print("="*60)
    print("🚀 Starting ALITA - Advanced AI Assistant")
    print("="*60)
    print()
    
    # Create ALITA system
    alita = ALITASystem()
    
    try:
        # Run system
        await alita.run()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        await alita.shutdown()


if __name__ == "__main__":
    if PYQT_AVAILABLE:
        # Check if QApplication already exists
        app = QApplication.instance()
        if app is None:
            # Create new QApplication
            app = QApplication(sys.argv)
            loop = QEventLoop(app)
            asyncio.set_event_loop(loop)
            
            with loop:
                loop.run_until_complete(main())
        else:
            # Use existing QApplication
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())
    else:
        # Run without GUI
        asyncio.run(main())
