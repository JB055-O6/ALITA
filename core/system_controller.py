"""
System Controller with Error Recovery

Implements Task 20 requirements:
- Component restart logic integration
- Graceful degradation strategies
- Health monitoring and control
- Performance optimization with error recovery

Integrates with existing:
- SystemMonitor for metrics and anomaly detection
- ServiceManager for service orchestration
- Utils for common functionality

All features are FREE and run locally!
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import asyncio

from .error_recovery import get_error_recovery
from .system_monitor import SystemMonitor
from .service_manager import ServiceManager
from .utils import psutil, torch, ensure_dir

try:
    import pynvml
except ImportError:
    pynvml = None


class SystemController:
    """System controller with integrated error recovery and health management.
    
    Integrates:
    - ErrorRecoverySystem for component restart and reliability
    - SystemMonitor for metrics, anomaly detection, and performance analysis
    - ServiceManager for service orchestration and scaling
    
    Features:
    - Component lifecycle management
    - Automatic error recovery
    - Resource monitoring and optimization
    - Graceful degradation
    - Performance throttling
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        # Initialize error recovery
        self.error_recovery = get_error_recovery()
        
        # Initialize system monitor (integrates existing monitoring)
        # Wrap in try-except to handle async event loop issues
        try:
            log_dir = log_dir or Path("logs")
            ensure_dir(log_dir)
            self.system_monitor = SystemMonitor(log_dir)
        except RuntimeError as e:
            if "no running event loop" in str(e):
                logging.warning("SystemMonitor disabled due to event loop issue")
                self.system_monitor = None
            else:
                raise
        
        # Initialize service manager (integrates existing service orchestration)
        self.service_manager = ServiceManager()
        
        # Component registry
        self.components = {}
        self.degraded_mode = False
        self.gpu_initialized = False
        
        # Initialize GPU monitoring if available
        self._init_gpu_monitoring()
        
        # Register system controller for restart
        self.error_recovery.register_component("system_controller", self._reinitialize)
        
        logging.info("SystemController initialized with error recovery, monitoring, and service management")
    
    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring with pynvml."""
        if pynvml is None or not torch or not torch.cuda.is_available():
            logging.info("GPU monitoring not available")
            return
        
        try:
            pynvml.nvmlInit()
            self.gpu_initialized = True
            logging.info("GPU monitoring initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize GPU monitoring: {str(e)}")
    
    def register_component(self, name: str, component: Any, initializer: callable):
        """Register a component for management and error recovery.
        
        Args:
            name: Component name
            component: Component instance
            initializer: Function to reinitialize component
        """
        self.components[name] = {
            "instance": component,
            "initializer": initializer,
            "status": "active",
            "last_health_check": datetime.now(),
            "failure_count": 0
        }
        
        # Register with error recovery system
        self.error_recovery.register_component(name, initializer)
        
        logging.info(f"Component registered: {name}")
    
    def check_component_health(self, name: str) -> Dict[str, Any]:
        """Check health of a specific component.
        
        Args:
            name: Component name
            
        Returns:
            Health status dictionary
        """
        if name not in self.components:
            return {"error": "Component not found"}
        
        component_info = self.components[name]
        component = component_info["instance"]
        
        health = {
            "name": name,
            "status": component_info["status"],
            "failure_count": component_info["failure_count"],
            "last_check": component_info["last_health_check"].isoformat()
        }
        
        # Try to get component-specific health if available
        if hasattr(component, "get_health_status"):
            try:
                health["component_health"] = component.get_health_status()
            except Exception as e:
                health["health_check_error"] = str(e)
        
        component_info["last_health_check"] = datetime.now()
        return health
    
    def handle_component_failure(self, name: str, error: Exception) -> bool:
        """Handle component failure with automatic recovery.
        
        Args:
            name: Component name
            error: Exception that caused failure
            
        Returns:
            True if recovery successful
        """
        if name not in self.components:
            logging.error(f"Unknown component failed: {name}")
            return False
        
        component_info = self.components[name]
        component_info["failure_count"] += 1
        component_info["status"] = "failed"
        
        logging.error(f"Component failure: {name} - {str(error)}")
        
        # Attempt recovery through error recovery system
        success = self.error_recovery.handle_component_failure(name, error)
        
        if success:
            component_info["status"] = "active"
            component_info["failure_count"] = 0
            logging.info(f"✅ Component recovered: {name}")
        else:
            # Enter degraded mode if critical component fails
            if self._is_critical_component(name):
                self._enter_degraded_mode(f"{name} failure")
        
        return success
    
    def _is_critical_component(self, name: str) -> bool:
        """Check if component is critical for system operation."""
        critical_components = ["brain", "voice_interface", "vision_system"]
        return name in critical_components
    
    def _enter_degraded_mode(self, reason: str):
        """Enter graceful degradation mode.
        
        Args:
            reason: Reason for entering degraded mode
        """
        if self.degraded_mode:
            return
        
        self.degraded_mode = True
        logging.warning(f"⚠️ Entering degraded mode: {reason}")
        
        # Disable non-essential components
        self._disable_non_essential_components()
        
        # Reduce resource usage
        self._reduce_resource_usage()
    
    def _exit_degraded_mode(self):
        """Exit degraded mode and restore full functionality."""
        if not self.degraded_mode:
            return
        
        self.degraded_mode = False
        logging.info("✅ Exiting degraded mode - restoring full functionality")
        
        # Re-enable components
        self._enable_all_components()
    
    def _disable_non_essential_components(self):
        """Disable non-essential components to conserve resources."""
        non_essential = ["proactive_agent", "image_generator", "predictive_engine"]
        
        for name in non_essential:
            if name in self.components:
                component_info = self.components[name]
                if component_info["status"] == "active":
                    component_info["status"] = "suspended"
                    logging.info(f"Suspended non-essential component: {name}")
    
    def _enable_all_components(self):
        """Re-enable all suspended components."""
        for name, component_info in self.components.items():
            if component_info["status"] == "suspended":
                component_info["status"] = "active"
                logging.info(f"Re-enabled component: {name}")
    
    def _reduce_resource_usage(self):
        """Reduce resource usage in degraded mode."""
        # Unload models if possible
        for name, component_info in self.components.items():
            component = component_info["instance"]
            if hasattr(component, "unload_models"):
                try:
                    component.unload_models()
                    logging.info(f"Unloaded models for: {name}")
                except Exception as e:
                    logging.warning(f"Failed to unload models for {name}: {str(e)}")
        
        # Clear GPU cache
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get complete system health status.
        
        Combines data from:
        - ErrorRecoverySystem (health checks, memory stats, uptime)
        - SystemMonitor (metrics, anomalies, performance analysis)
        - ServiceManager (service status)
        
        Returns:
            Comprehensive system health dictionary
        """
        # Get health from error recovery system
        health = self.error_recovery.run_health_check()
        
        # Get system monitoring data
        monitor_status = await self.system_monitor.get_system_status()
        health["monitoring"] = monitor_status
        
        # Get service status
        services = await self.service_manager.list_services()
        health["services"] = services
        
        # Add GPU temperature if available
        if self.gpu_initialized:
            health["gpu_temperature"] = self.get_gpu_temperature()
        
        # Add component statuses
        health["components"] = {}
        for name, component_info in self.components.items():
            health["components"][name] = {
                "status": component_info["status"],
                "failure_count": component_info["failure_count"]
            }
        
        # Add degraded mode status
        health["degraded_mode"] = self.degraded_mode
        
        # Add error recovery stats
        health["error_recovery"] = {
            "memory_stats": self.error_recovery.get_memory_stats(),
            "uptime_stats": self.error_recovery.get_uptime_stats()
        }
        
        return health
    
    def get_gpu_temperature(self) -> Optional[float]:
        """Get GPU temperature in Celsius.
        
        Returns:
            Temperature in Celsius or None if unavailable
        """
        if not self.gpu_initialized:
            return None
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return float(temp)
        except Exception as e:
            logging.warning(f"Failed to get GPU temperature: {str(e)}")
            return None
    
    def check_and_throttle(self) -> bool:
        """Check system resources and throttle if necessary.
        
        Returns:
            True if throttling was applied
        """
        throttled = False
        
        # Check GPU temperature
        if self.gpu_initialized:
            temp = self.get_gpu_temperature()
            if temp and temp > 80:
                logging.warning(f"⚠️ GPU temperature high: {temp}°C - throttling")
                self._throttle_gpu_operations()
                throttled = True
        
        # Check CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 90:
            logging.warning(f"⚠️ CPU usage high: {cpu_percent}% - throttling")
            self._throttle_cpu_operations()
            throttled = True
        
        # Check memory usage
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logging.warning(f"⚠️ Memory usage high: {memory.percent}% - freeing memory")
            self._free_memory()
            throttled = True
        
        return throttled
    
    def _throttle_gpu_operations(self):
        """Throttle GPU operations to reduce temperature."""
        # Reduce batch sizes for components
        for name, component_info in self.components.items():
            component = component_info["instance"]
            if hasattr(component, "reduce_batch_size"):
                try:
                    component.reduce_batch_size()
                except Exception as e:
                    logging.warning(f"Failed to throttle {name}: {str(e)}")
        
        # Add delay between operations
        time.sleep(2)
    
    def _throttle_cpu_operations(self):
        """Throttle CPU operations to reduce load."""
        # Reduce thread count or processing rate
        time.sleep(1)
    
    def _free_memory(self):
        """Attempt to free memory."""
        import gc
        gc.collect()
        
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logging.info("Memory cleanup performed")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run health checks on all components.
        
        Uses SystemMonitor for comprehensive metrics and anomaly detection.
        
        Returns:
            Health check results for all components
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "system": await self.get_system_health(),
            "components": {}
        }
        
        for name in self.components:
            results["components"][name] = self.check_component_health(name)
        
        # Check if we should exit degraded mode
        if self.degraded_mode:
            if self._can_exit_degraded_mode(results):
                self._exit_degraded_mode()
        
        return results
    
    def _can_exit_degraded_mode(self, health_results: Dict[str, Any]) -> bool:
        """Check if system can exit degraded mode.
        
        Args:
            health_results: Current health check results
            
        Returns:
            True if safe to exit degraded mode
        """
        system_health = health_results["system"]
        
        # Check overall system health
        if system_health.get("overall_status") != "healthy":
            return False
        
        # Check critical components are active
        for name in ["brain", "voice_interface", "vision_system"]:
            if name in self.components:
                if self.components[name]["status"] != "active":
                    return False
        
        return True
    
    def _reinitialize(self) -> bool:
        """Reinitialize system controller after failure.
        
        Returns:
            True if reinitialization successful
        """
        try:
            logging.info("Reinitializing SystemController...")
            
            # Reset degraded mode
            self.degraded_mode = False
            
            # Reinitialize GPU monitoring
            if self.gpu_initialized:
                try:
                    pynvml.nvmlShutdown()
                except:
                    pass
                self._init_gpu_monitoring()
            
            logging.info("✅ SystemController reinitialized")
            return True
            
        except Exception as e:
            logging.error(f"SystemController reinitialization failed: {str(e)}")
            return False
    
    async def register_service(self, service_id: str) -> bool:
        """Register a service with the service manager.
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if registration successful
        """
        try:
            service = await self.service_manager.create_service(service_id)
            if service:
                logging.info(f"Service registered: {service_id}")
                return True
            return False
        except Exception as e:
            logging.error(f"Service registration failed: {str(e)}")
            return False
    
    async def unregister_service(self, service_id: str) -> bool:
        """Unregister a service from the service manager.
        
        Args:
            service_id: Service identifier
            
        Returns:
            True if unregistration successful
        """
        try:
            success = await self.service_manager.delete_service(service_id)
            if success:
                logging.info(f"Service unregistered: {service_id}")
            return success
        except Exception as e:
            logging.error(f"Service unregistration failed: {str(e)}")
            return False
    
    async def get_performance_analysis(self) -> Dict[str, Any]:
        """Get performance analysis from system monitor.
        
        Returns:
            Performance analysis including trends and anomalies
        """
        try:
            status = await self.system_monitor.get_system_status()
            return {
                "performance": status.get("performance_analysis", {}),
                "anomalies": status.get("detected_anomalies", []),
                "current_metrics": status.get("current_metrics", {})
            }
        except Exception as e:
            logging.error(f"Performance analysis failed: {str(e)}")
            return {}
    
    async def log_system_event(self, event_type: str, details: Dict[str, Any]):
        """Log a system event through the system monitor.
        
        Args:
            event_type: Type of event
            details: Event details
        """
        try:
            await self.system_monitor.event_logger.log_event(event_type, details)
        except Exception as e:
            logging.error(f"Event logging failed: {str(e)}")
    
    async def shutdown(self):
        """Shutdown system controller and cleanup resources."""
        logging.info("Shutting down SystemController...")
        
        # Log shutdown event
        await self.log_system_event("system_shutdown", {
            "degraded_mode": self.degraded_mode,
            "component_count": len(self.components)
        })
        
        # Shutdown all services
        services = await self.service_manager.list_services()
        for service_info in services:
            await self.service_manager.delete_service(service_info["service_id"])
        
        # Shutdown GPU monitoring
        if self.gpu_initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logging.warning(f"GPU monitoring shutdown error: {str(e)}")
        
        # Shutdown error recovery
        self.error_recovery.shutdown()
        
        logging.info("SystemController shutdown complete")


# Global instance
_system_controller: Optional[SystemController] = None


def get_system_controller() -> SystemController:
    """Get global system controller instance."""
    global _system_controller
    
    if _system_controller is None:
        _system_controller = SystemController()
    
    return _system_controller
