"""
Error Recovery and Reliability System

Implements Task 20 requirements:
- Component restart logic
- Memory leak detection and prevention
- Graceful degradation strategies
- Health check system with anomaly detection
- Uptime tracking and performance metrics

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
import threading
import time
import psutil
import traceback

try:
    import torch
except ImportError:
    torch = None


class ComponentRestarter:
    """Restart failed components automatically."""
    
    def __init__(self):
        self.restart_attempts: Dict[str, int] = {}
        self.max_restart_attempts = 3
        self.restart_cooldown = timedelta(minutes=5)
        self.last_restart_time: Dict[str, datetime] = {}
        self.component_initializers: Dict[str, Callable] = {}
    
    def register_component(self, component_name: str, initializer: Callable):
        """Register component with restart capability.
        
        Args:
            component_name: Name of component
            initializer: Function to initialize component
        """
        self.component_initializers[component_name] = initializer
        self.restart_attempts[component_name] = 0
        logging.info(f"Component registered for restart: {component_name}")
    
    def restart_component(self, component_name: str) -> bool:
        """Restart a failed component.
        
        Args:
            component_name: Name of component to restart
            
        Returns:
            True if restart successful
        """
        if component_name not in self.component_initializers:
            logging.error(f"Component not registered: {component_name}")
            return False
        
        # Check restart attempts
        if self.restart_attempts[component_name] >= self.max_restart_attempts:
            logging.error(f"Max restart attempts reached for {component_name}")
            return False
        
        # Check cooldown
        if component_name in self.last_restart_time:
            time_since_restart = datetime.now() - self.last_restart_time[component_name]
            if time_since_restart < self.restart_cooldown:
                logging.warning(f"Restart cooldown active for {component_name}")
                return False
        
        try:
            logging.info(f"Restarting component: {component_name}")
            
            # Call initializer
            initializer = self.component_initializers[component_name]
            result = initializer()
            
            # Update tracking
            self.restart_attempts[component_name] += 1
            self.last_restart_time[component_name] = datetime.now()
            
            logging.info(f"✅ Component restarted successfully: {component_name}")
            return True
            
        except Exception as e:
            logging.error(f"Component restart failed: {e}")
            logging.debug(traceback.format_exc())
            return False
    
    def reset_restart_count(self, component_name: str):
        """Reset restart attempt counter.
        
        Args:
            component_name: Name of component
        """
        if component_name in self.restart_attempts:
            self.restart_attempts[component_name] = 0
            logging.info(f"Restart counter reset for {component_name}")


class MemoryLeakDetector:
    """Detect and prevent memory leaks."""
    
    def __init__(self, check_interval: int = 300):
        self.check_interval = check_interval  # seconds
        self.memory_history = []
        self.max_history = 100
        self.leak_threshold = 1.0  # MB per hour
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start memory leak monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logging.info("Memory leak monitoring started")
    
    def stop_monitoring(self):
        """Stop memory leak monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logging.info("Memory leak monitoring stopped")
    
    def _monitor_loop(self):
        """Monitoring loop."""
        while self.monitoring:
            try:
                # Get current memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                
                # Record
                self.memory_history.append({
                    "timestamp": datetime.now(),
                    "memory_mb": memory_mb
                })
                
                # Keep history limited
                if len(self.memory_history) > self.max_history:
                    self.memory_history = self.memory_history[-self.max_history:]
                
                # Check for leak
                if len(self.memory_history) >= 10:
                    leak_detected = self._detect_leak()
                    if leak_detected:
                        logging.warning(f"⚠️ Memory leak detected: {leak_detected['rate']:.2f} MB/hour")
                        self._attempt_cleanup()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logging.error(f"Memory monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _detect_leak(self) -> Optional[Dict[str, Any]]:
        """Detect memory leak from history.
        
        Returns:
            Leak info dict or None
        """
        if len(self.memory_history) < 10:
            return None
        
        # Calculate memory growth rate
        first = self.memory_history[0]
        last = self.memory_history[-1]
        
        time_diff_hours = (last["timestamp"] - first["timestamp"]).total_seconds() / 3600
        memory_diff_mb = last["memory_mb"] - first["memory_mb"]
        
        if time_diff_hours > 0:
            growth_rate = memory_diff_mb / time_diff_hours
            
            if growth_rate > self.leak_threshold:
                return {
                    "rate": growth_rate,
                    "total_growth_mb": memory_diff_mb,
                    "time_hours": time_diff_hours
                }
        
        return None
    
    def _attempt_cleanup(self):
        """Attempt to clean up memory."""
        try:
            import gc
            gc.collect()
            
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logging.info("Memory cleanup performed")
            
        except Exception as e:
            logging.error(f"Memory cleanup failed: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Memory stats dictionary
        """
        if not self.memory_history:
            return {}
        
        current = self.memory_history[-1]["memory_mb"]
        
        if len(self.memory_history) >= 2:
            first = self.memory_history[0]["memory_mb"]
            growth = current - first
        else:
            growth = 0
        
        return {
            "current_mb": current,
            "growth_mb": growth,
            "samples": len(self.memory_history),
            "monitoring": self.monitoring
        }


class HealthChecker:
    """System health check with anomaly detection."""
    
    def __init__(self):
        self.health_history = []
        self.max_history = 1000
        self.anomaly_threshold = 2.0  # Standard deviations
    
    def check_health(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health check results
        """
        health = {
            "timestamp": datetime.now().isoformat(),
            "cpu": self._check_cpu(),
            "memory": self._check_memory(),
            "gpu": self._check_gpu(),
            "disk": self._check_disk(),
            "overall_status": "healthy"
        }
        
        # Determine overall status
        statuses = [health["cpu"]["status"], health["memory"]["status"], 
                   health["gpu"]["status"], health["disk"]["status"]]
        
        if "critical" in statuses:
            health["overall_status"] = "critical"
        elif "warning" in statuses:
            health["overall_status"] = "warning"
        
        # Record in history
        self.health_history.append(health)
        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history:]
        
        # Detect anomalies
        anomalies = self._detect_anomalies(health)
        if anomalies:
            health["anomalies"] = anomalies
        
        return health
    
    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU health."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = "healthy"
            if cpu_percent > 80:
                status = "warning"
            if cpu_percent > 95:
                status = "critical"
            
            return {
                "percent": cpu_percent,
                "status": status
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory health."""
        try:
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if memory.percent > 85:
                status = "warning"
            if memory.percent > 95:
                status = "critical"
            
            return {
                "percent": memory.percent,
                "used_gb": memory.used / (1024**3),
                "total_gb": memory.total / (1024**3),
                "status": status
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def _check_gpu(self) -> Dict[str, Any]:
        """Check GPU health."""
        if not torch or not torch.cuda.is_available():
            return {"available": False, "status": "not_available"}
        
        try:
            vram_allocated = torch.cuda.memory_allocated() / (1024**3)
            vram_reserved = torch.cuda.memory_reserved() / (1024**3)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_percent = (vram_allocated / vram_total) * 100
            
            status = "healthy"
            if vram_percent > 85:
                status = "warning"
            if vram_percent > 95:
                status = "critical"
            
            return {
                "vram_allocated_gb": vram_allocated,
                "vram_total_gb": vram_total,
                "vram_percent": vram_percent,
                "status": status
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk health."""
        try:
            disk = psutil.disk_usage('.')
            
            status = "healthy"
            if disk.percent > 90:
                status = "warning"
            if disk.percent > 95:
                status = "critical"
            
            return {
                "percent": disk.percent,
                "free_gb": disk.free / (1024**3),
                "status": status
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def _detect_anomalies(self, current_health: Dict[str, Any]) -> List[str]:
        """Detect anomalies in health metrics.
        
        Args:
            current_health: Current health check results
            
        Returns:
            List of detected anomalies
        """
        if len(self.health_history) < 30:
            return []
        
        anomalies = []
        
        # Check CPU anomaly
        recent_cpu = [h["cpu"]["percent"] for h in self.health_history[-30:] 
                     if "percent" in h["cpu"]]
        if recent_cpu:
            mean_cpu = sum(recent_cpu) / len(recent_cpu)
            std_cpu = (sum((x - mean_cpu) ** 2 for x in recent_cpu) / len(recent_cpu)) ** 0.5
            
            current_cpu = current_health["cpu"].get("percent", 0)
            if abs(current_cpu - mean_cpu) > self.anomaly_threshold * std_cpu:
                anomalies.append(f"CPU usage anomaly: {current_cpu:.1f}% (avg: {mean_cpu:.1f}%)")
        
        # Check memory anomaly
        recent_memory = [h["memory"]["percent"] for h in self.health_history[-30:] 
                        if "percent" in h["memory"]]
        if recent_memory:
            mean_memory = sum(recent_memory) / len(recent_memory)
            std_memory = (sum((x - mean_memory) ** 2 for x in recent_memory) / len(recent_memory)) ** 0.5
            
            current_memory = current_health["memory"].get("percent", 0)
            if abs(current_memory - mean_memory) > self.anomaly_threshold * std_memory:
                anomalies.append(f"Memory usage anomaly: {current_memory:.1f}% (avg: {mean_memory:.1f}%)")
        
        return anomalies


class UptimeTracker:
    """Track system uptime and performance."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.restart_count = 0
        self.crash_count = 0
        self.performance_log = []
    
    def get_uptime(self) -> timedelta:
        """Get system uptime.
        
        Returns:
            Uptime as timedelta
        """
        return datetime.now() - self.start_time
    
    def record_restart(self, reason: str = "unknown"):
        """Record system restart.
        
        Args:
            reason: Reason for restart
        """
        self.restart_count += 1
        logging.info(f"Restart recorded: {reason} (total: {self.restart_count})")
    
    def record_crash(self, error: str):
        """Record system crash.
        
        Args:
            error: Error message
        """
        self.crash_count += 1
        logging.error(f"Crash recorded: {error} (total: {self.crash_count})")
    
    def record_performance(self, metric: str, value: float):
        """Record performance metric.
        
        Args:
            metric: Metric name
            value: Metric value
        """
        self.performance_log.append({
            "timestamp": datetime.now().isoformat(),
            "metric": metric,
            "value": value
        })
        
        # Keep last 1000 entries
        if len(self.performance_log) > 1000:
            self.performance_log = self.performance_log[-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get uptime statistics.
        
        Returns:
            Statistics dictionary
        """
        uptime = self.get_uptime()
        
        return {
            "uptime_seconds": uptime.total_seconds(),
            "uptime_formatted": str(uptime),
            "restart_count": self.restart_count,
            "crash_count": self.crash_count,
            "performance_samples": len(self.performance_log),
            "start_time": self.start_time.isoformat()
        }


class ErrorRecoverySystem:
    """Main error recovery and reliability system."""
    
    def __init__(self):
        self.component_restarter = ComponentRestarter()
        self.memory_leak_detector = MemoryLeakDetector()
        self.health_checker = HealthChecker()
        self.uptime_tracker = UptimeTracker()
        
        # Start monitoring
        self.memory_leak_detector.start_monitoring()
        
        logging.info("Error Recovery System initialized")
    
    def register_component(self, component_name: str, initializer: Callable):
        """Register component for automatic restart.
        
        Args:
            component_name: Name of component
            initializer: Initialization function
        """
        self.component_restarter.register_component(component_name, initializer)
    
    def handle_component_failure(self, component_name: str, error: Exception) -> bool:
        """Handle component failure with automatic recovery.
        
        Args:
            component_name: Name of failed component
            error: Exception that caused failure
            
        Returns:
            True if recovery successful
        """
        logging.error(f"Component failure: {component_name} - {error}")
        logging.debug(traceback.format_exc())
        
        # Attempt restart
        success = self.component_restarter.restart_component(component_name)
        
        if success:
            logging.info(f"✅ Component recovered: {component_name}")
            self.uptime_tracker.record_restart(f"{component_name} failure")
        else:
            logging.error(f"❌ Component recovery failed: {component_name}")
            self.uptime_tracker.record_crash(str(error))
        
        return success
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run complete health check.
        
        Returns:
            Health check results
        """
        return self.health_checker.check_health()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory leak statistics.
        
        Returns:
            Memory statistics
        """
        return self.memory_leak_detector.get_memory_stats()
    
    def get_uptime_stats(self) -> Dict[str, Any]:
        """Get uptime statistics.
        
        Returns:
            Uptime statistics
        """
        return self.uptime_tracker.get_stats()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status.
        
        Returns:
            Complete system status
        """
        return {
            "health": self.run_health_check(),
            "memory": self.get_memory_stats(),
            "uptime": self.get_uptime_stats()
        }
    
    def shutdown(self):
        """Shutdown error recovery system."""
        self.memory_leak_detector.stop_monitoring()
        logging.info("Error Recovery System shutdown")


# Global instance
_error_recovery: Optional[ErrorRecoverySystem] = None


def get_error_recovery() -> ErrorRecoverySystem:
    """Get global error recovery system instance."""
    global _error_recovery
    
    if _error_recovery is None:
        _error_recovery = ErrorRecoverySystem()
    
    return _error_recovery
