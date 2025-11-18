"""
Enhanced Logging System with Diagnostics

Implements Task 11 requirements:
- Rotating file handler (100MB limit)
- Performance metrics logging
- Diagnostic command for system health check
- Log analysis tool for pattern detection
- Graceful degradation handlers

All features are FREE and run locally!
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json
import time
import psutil
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import traceback

try:
    import torch
except ImportError:
    torch = None

try:
    import pynvml
except ImportError:
    pynvml = None


@dataclass
class PerformanceMetrics:
    """Performance metrics data."""
    timestamp: datetime
    inference_time_ms: float
    vram_used_mb: float
    vram_total_mb: float
    cpu_percent: float
    memory_percent: float
    gpu_temp_celsius: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    model_name: Optional[str] = None


class PerformanceLogger:
    """Log performance metrics."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.log_dir / "performance_metrics.jsonl"
        self.metrics_buffer = deque(maxlen=1000)
        
        # Initialize GPU monitoring
        self.gpu_available = False
        if pynvml and torch and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
            except Exception:
                pass
    
    def log_inference(self,
                     inference_time: float,
                     tokens_generated: int = 0,
                     model_name: str = "unknown"):
        """Log inference performance metrics."""
        try:
            # Calculate throughput
            throughput = None
            if tokens_generated > 0 and inference_time > 0:
                throughput = tokens_generated / inference_time
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Get GPU metrics
            vram_used = 0.0
            vram_total = 0.0
            gpu_temp = None
            
            if self.gpu_available and torch.cuda.is_available():
                try:
                    vram_used = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                    
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    pass
            
            # Create metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                inference_time_ms=inference_time * 1000,
                vram_used_mb=vram_used,
                vram_total_mb=vram_total,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_temp_celsius=gpu_temp,
                throughput_tokens_per_sec=throughput,
                model_name=model_name
            )
            
            # Add to buffer
            self.metrics_buffer.append(metrics)
            
            # Write to file
            self._write_metrics(metrics)
            
        except Exception as e:
            logging.error(f"Performance logging failed: {e}")
    
    def _write_metrics(self, metrics: PerformanceMetrics):
        """Write metrics to file."""
        try:
            metrics_dict = asdict(metrics)
            metrics_dict['timestamp'] = metrics.timestamp.isoformat()
            
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')
        except Exception as e:
            logging.error(f"Metrics write failed: {e}")
    
    def get_recent_metrics(self, count: int = 100) -> List[PerformanceMetrics]:
        """Get recent performance metrics."""
        return list(self.metrics_buffer)[-count:]
    
    def get_average_metrics(self, minutes: int = 5) -> Dict[str, float]:
        """Get average metrics for time period."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent = [m for m in self.metrics_buffer if m.timestamp >= cutoff_time]
        
        if not recent:
            return {}
        
        return {
            "avg_inference_time_ms": sum(m.inference_time_ms for m in recent) / len(recent),
            "avg_vram_used_mb": sum(m.vram_used_mb for m in recent) / len(recent),
            "avg_cpu_percent": sum(m.cpu_percent for m in recent) / len(recent),
            "avg_memory_percent": sum(m.memory_percent for m in recent) / len(recent),
            "avg_throughput": sum(m.throughput_tokens_per_sec or 0 for m in recent) / len(recent),
            "sample_count": len(recent)
        }


class LogAnalyzer:
    """Analyze logs for patterns and issues."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.error_patterns = defaultdict(int)
        self.warning_patterns = defaultdict(int)
    
    def analyze_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze recent logs."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Find log files
            log_files = sorted(self.log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            
            stats = {
                "total_lines": 0,
                "errors": 0,
                "warnings": 0,
                "info": 0,
                "debug": 0,
                "error_patterns": {},
                "warning_patterns": {},
                "recent_errors": [],
                "time_range": {
                    "start": cutoff_time.isoformat(),
                    "end": datetime.now().isoformat()
                }
            }
            
            for log_file in log_files[:5]:  # Check last 5 log files
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            stats["total_lines"] += 1
                            
                            # Count by level
                            if "ERROR" in line:
                                stats["errors"] += 1
                                self._extract_error_pattern(line, stats)
                            elif "WARNING" in line:
                                stats["warnings"] += 1
                                self._extract_warning_pattern(line, stats)
                            elif "INFO" in line:
                                stats["info"] += 1
                            elif "DEBUG" in line:
                                stats["debug"] += 1
                except Exception as e:
                    logging.error(f"Failed to analyze {log_file}: {e}")
            
            # Add pattern summaries
            stats["error_patterns"] = dict(sorted(
                self.error_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
            
            stats["warning_patterns"] = dict(sorted(
                self.warning_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
            
            return stats
            
        except Exception as e:
            logging.error(f"Log analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_error_pattern(self, line: str, stats: Dict):
        """Extract error pattern from log line."""
        try:
            # Simple pattern extraction
            if ":" in line:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    error_msg = parts[2].strip()[:100]  # First 100 chars
                    self.error_patterns[error_msg] += 1
                    
                    # Add to recent errors
                    if len(stats["recent_errors"]) < 10:
                        stats["recent_errors"].append({
                            "message": error_msg,
                            "line": line.strip()[:200]
                        })
        except Exception:
            pass
    
    def _extract_warning_pattern(self, line: str, stats: Dict):
        """Extract warning pattern from log line."""
        try:
            if ":" in line:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    warning_msg = parts[2].strip()[:100]
                    self.warning_patterns[warning_msg] += 1
        except Exception:
            pass


class SystemDiagnostics:
    """System health check and diagnostics."""
    
    def __init__(self):
        self.gpu_available = False
        if pynvml and torch and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_available = True
            except Exception:
                pass
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run complete system diagnostics."""
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "system": self._check_system(),
            "gpu": self._check_gpu(),
            "disk": self._check_disk(),
            "models": self._check_models(),
            "health_score": 0.0,
            "issues": [],
            "recommendations": []
        }
        
        # Calculate health score
        diagnostics["health_score"] = self._calculate_health_score(diagnostics)
        
        # Generate recommendations
        diagnostics["recommendations"] = self._generate_recommendations(diagnostics)
        
        return diagnostics
    
    def _check_system(self) -> Dict[str, Any]:
        """Check system resources."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            return {
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": memory.total / (1024 ** 3),
                "memory_used_gb": memory.used / (1024 ** 3),
                "memory_percent": memory.percent,
                "status": "healthy" if cpu_percent < 80 and memory.percent < 85 else "warning"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def _check_gpu(self) -> Dict[str, Any]:
        """Check GPU status."""
        if not self.gpu_available:
            return {"available": False, "status": "not_available"}
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get GPU info
            name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            vram_used_gb = memory_info.used / (1024 ** 3)
            vram_total_gb = memory_info.total / (1024 ** 3)
            vram_percent = (memory_info.used / memory_info.total) * 100
            
            status = "healthy"
            if temp > 80 or vram_percent > 90:
                status = "warning"
            if temp > 85 or vram_percent > 95:
                status = "critical"
            
            return {
                "available": True,
                "name": name,
                "vram_used_gb": vram_used_gb,
                "vram_total_gb": vram_total_gb,
                "vram_percent": vram_percent,
                "temperature_celsius": temp,
                "utilization_percent": utilization.gpu,
                "status": status
            }
        except Exception as e:
            return {"available": True, "error": str(e), "status": "error"}
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk space."""
        try:
            disk = psutil.disk_usage('.')
            
            return {
                "total_gb": disk.total / (1024 ** 3),
                "used_gb": disk.used / (1024 ** 3),
                "free_gb": disk.free / (1024 ** 3),
                "percent": disk.percent,
                "status": "healthy" if disk.percent < 90 else "warning"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def _check_models(self) -> Dict[str, Any]:
        """Check model availability."""
        models_dir = Path("models")
        
        if not models_dir.exists():
            return {"status": "not_found", "models": []}
        
        try:
            models = []
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir():
                    size = sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file())
                    models.append({
                        "name": model_dir.name,
                        "size_gb": size / (1024 ** 3),
                        "path": str(model_dir)
                    })
            
            return {
                "count": len(models),
                "models": models,
                "total_size_gb": sum(m["size_gb"] for m in models),
                "status": "healthy" if models else "warning"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def _calculate_health_score(self, diagnostics: Dict) -> float:
        """Calculate overall health score (0-100)."""
        score = 100.0
        
        # System health
        if diagnostics["system"].get("status") == "warning":
            score -= 15
        elif diagnostics["system"].get("status") == "error":
            score -= 30
        
        # GPU health
        if diagnostics["gpu"].get("status") == "warning":
            score -= 10
        elif diagnostics["gpu"].get("status") == "critical":
            score -= 25
        
        # Disk health
        if diagnostics["disk"].get("status") == "warning":
            score -= 10
        
        # Models health
        if diagnostics["models"].get("status") == "warning":
            score -= 5
        
        return max(0.0, score)
    
    def _generate_recommendations(self, diagnostics: Dict) -> List[str]:
        """Generate recommendations based on diagnostics."""
        recommendations = []
        
        # System recommendations
        system = diagnostics["system"]
        if system.get("memory_percent", 0) > 85:
            recommendations.append("High memory usage detected. Consider closing unused applications.")
        if system.get("cpu_percent", 0) > 80:
            recommendations.append("High CPU usage detected. System may be under heavy load.")
        
        # GPU recommendations
        gpu = diagnostics["gpu"]
        if gpu.get("temperature_celsius", 0) > 80:
            recommendations.append("GPU temperature is high. Ensure adequate cooling.")
        if gpu.get("vram_percent", 0) > 90:
            recommendations.append("VRAM usage is high. Consider unloading unused models.")
        
        # Disk recommendations
        disk = diagnostics["disk"]
        if disk.get("percent", 0) > 90:
            recommendations.append("Disk space is low. Consider cleaning up old files.")
        
        # Models recommendations
        models = diagnostics["models"]
        if models.get("count", 0) == 0:
            recommendations.append("No models found. Run 'python scripts/download_models.py --required'")
        
        return recommendations


class GracefulDegradationHandler:
    """Handle component failures gracefully."""
    
    def __init__(self):
        self.failed_components = set()
        self.fallback_strategies = {}
    
    def register_fallback(self, component: str, fallback_fn):
        """Register fallback strategy for component."""
        self.fallback_strategies[component] = fallback_fn
    
    def handle_failure(self, component: str, error: Exception) -> Any:
        """Handle component failure with fallback."""
        logging.error(f"Component '{component}' failed: {error}")
        logging.debug(traceback.format_exc())
        
        self.failed_components.add(component)
        
        # Try fallback
        if component in self.fallback_strategies:
            try:
                logging.info(f"Attempting fallback for '{component}'")
                return self.fallback_strategies[component]()
            except Exception as fallback_error:
                logging.error(f"Fallback for '{component}' also failed: {fallback_error}")
        
        return None
    
    def is_component_healthy(self, component: str) -> bool:
        """Check if component is healthy."""
        return component not in self.failed_components
    
    def reset_component(self, component: str):
        """Reset component status."""
        self.failed_components.discard(component)


class EnhancedLoggingService:
    """Enhanced logging service with all Task 11 features."""
    
    def __init__(self, log_dir: Path = None):
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup rotating file handler (100MB limit, 5 backups)
        self.logger = logging.getLogger("ALITA")
        self.logger.setLevel(logging.DEBUG)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "alita.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Initialize components
        self.performance_logger = PerformanceLogger(self.log_dir)
        self.log_analyzer = LogAnalyzer(self.log_dir)
        self.diagnostics = SystemDiagnostics()
        self.degradation_handler = GracefulDegradationHandler()
        
        logging.info("✅ Enhanced logging system initialized")
    
    def log_performance(self, inference_time: float, tokens: int = 0, model: str = "unknown"):
        """Log performance metrics."""
        self.performance_logger.log_inference(inference_time, tokens, model)
    
    def analyze_logs(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze recent logs."""
        return self.log_analyzer.analyze_logs(hours)
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run system diagnostics."""
        return self.diagnostics.run_diagnostics()
    
    def get_performance_summary(self, minutes: int = 5) -> Dict[str, Any]:
        """Get performance summary."""
        return self.performance_logger.get_average_metrics(minutes)


# Global instance
_logging_service: Optional[EnhancedLoggingService] = None


def get_logging_service() -> EnhancedLoggingService:
    """Get global logging service instance."""
    global _logging_service
    
    if _logging_service is None:
        _logging_service = EnhancedLoggingService()
    
    return _logging_service
