"""
Advanced System Monitor with Real-time Analytics
Features:
- Real-time system monitoring
- Performance analytics
- Resource tracking
- Event logging
- Anomaly detection
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from collections import deque

from alita.core.utils import psutil, pd, np, torch, ensure_dir
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

@dataclass
class SystemMetrics:
    """System-wide metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_usage: float
    gpu_usage: Optional[float]
    process_count: int
    error_count: int
    timestamp: datetime

class AnomalyDetector:
    """Real-time anomaly detection."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.history = []
        self.model = self._build_model()
        self.scaler = StandardScaler()
    
    def _build_model(self) -> IsolationForest:
        """Build anomaly detection model."""
        return IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
    
    def add_metrics(self, metrics: SystemMetrics):
        """Add metrics to history."""
        self.history.append({
            "cpu": metrics.cpu_usage,
            "memory": metrics.memory_usage,
            "disk": metrics.disk_usage,
            "network": metrics.network_usage,
            "gpu": metrics.gpu_usage or 0.0,
            "processes": metrics.process_count,
            "errors": metrics.error_count,
            "hour": metrics.timestamp.hour,
            "minute": metrics.timestamp.minute
        })
        
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics."""
        if len(self.history) < 100:
            return []
        
        # Prepare data
        df = pd.DataFrame(self.history)
        features = ["cpu", "memory", "disk", "network", 
                   "gpu", "processes", "errors"]
        
        X = df[features].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model and predict
        self.model.fit(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        # Find anomalies
        anomalies = []
        for i, pred in enumerate(predictions):
            if pred == -1:  # Anomaly
                metrics = df.iloc[i].to_dict()
                anomalies.append({
                    "timestamp": 
                        datetime.now().replace(
                            hour=int(metrics["hour"]),
                            minute=int(metrics["minute"])
                        ).isoformat(),
                    "metrics": {
                        k: metrics[k] 
                        for k in features
                    }
                })
        
        return anomalies

class PerformanceAnalyzer:
    """System performance analysis."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.baseline = None
    
    def add_metrics(self, metrics: SystemMetrics):
        """Add metrics to history."""
        self.metrics_history.append(metrics)
        
        # Update baseline if needed
        if len(self.metrics_history) == self.window_size:
            self._update_baseline()
    
    def _update_baseline(self):
        """Update performance baseline."""
        if not self.metrics_history:
            return
        
        # Calculate average metrics
        cpu_avg = np.mean([m.cpu_usage for m in self.metrics_history])
        memory_avg = np.mean([m.memory_usage for m in self.metrics_history])
        disk_avg = np.mean([m.disk_usage for m in self.metrics_history])
        network_avg = np.mean([m.network_usage for m in self.metrics_history])
        
        gpu_values = [m.gpu_usage for m in self.metrics_history 
                     if m.gpu_usage is not None]
        gpu_avg = np.mean(gpu_values) if gpu_values else None
        
        self.baseline = {
            "cpu": cpu_avg,
            "memory": memory_avg,
            "disk": disk_avg,
            "network": network_avg,
            "gpu": gpu_avg,
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance."""
        if not self.metrics_history or not self.baseline:
            return {}
        
        current = self.metrics_history[-1]
        
        # Calculate deviations
        deviations = {
            "cpu": current.cpu_usage - self.baseline["cpu"],
            "memory": current.memory_usage - self.baseline["memory"],
            "disk": current.disk_usage - self.baseline["disk"],
            "network": current.network_usage - self.baseline["network"]
        }
        
        if current.gpu_usage is not None and self.baseline["gpu"] is not None:
            deviations["gpu"] = current.gpu_usage - self.baseline["gpu"]
        
        # Calculate trends
        window = list(self.metrics_history)
        trends = {
            "cpu": self._calculate_trend([m.cpu_usage for m in window]),
            "memory": self._calculate_trend([m.memory_usage for m in window]),
            "disk": self._calculate_trend([m.disk_usage for m in window]),
            "network": self._calculate_trend([m.network_usage for m in window])
        }
        
        if all(m.gpu_usage is not None for m in window):
            trends["gpu"] = self._calculate_trend(
                [m.gpu_usage for m in window]
            )
        
        return {
            "current": {
                "cpu": current.cpu_usage,
                "memory": current.memory_usage,
                "disk": current.disk_usage,
                "network": current.network_usage,
                "gpu": current.gpu_usage,
                "timestamp": current.timestamp.isoformat()
            },
            "baseline": self.baseline,
            "deviations": deviations,
            "trends": trends
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate metric trend."""
        if len(values) < 2:
            return "stable"
        
        # Calculate slope
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        # Determine trend
        if abs(slope) < 0.1:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

class EventLogger:
    """System event logging."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = self.log_dir / "system.log"
        self.event_buffer = []
    
    async def log_event(self,
                       event_type: str,
                       details: Dict[str, Any]):
        """Log system event."""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "type": event_type,
                "details": details
            }
            
            # Add to buffer
            self.event_buffer.append(event)
            
            # Write to file if buffer is full
            if len(self.event_buffer) >= 100:
                # Use async_file helper to append lines
                from alita.core.async_file import async_write_lines
                lines = [json.dumps(e) for e in self.event_buffer]
                await async_write_lines(self.current_log, lines)
                self.event_buffer = []
            
        except Exception as e:
            logging.error(f"Event logging failed: {str(e)}")
    
    async def _flush_buffer(self):
        """Flush event buffer to file."""
        if not self.event_buffer:
            return

        try:
            from alita.core.async_file import async_write_lines
            lines = [json.dumps(e) for e in self.event_buffer]
            await async_write_lines(self.current_log, lines)

            self.event_buffer = []

            # Rotate log if too large
            try:
                if self.current_log.stat().st_size > 10 * 1024 * 1024:  # 10MB
                    await self._rotate_log()
            except Exception:
                # If stat fails (race), ignore rotation for now
                pass

        except Exception as e:
            logging.error(f"Buffer flush failed: {str(e)}")
    
    async def _rotate_log(self):
        """Rotate log file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = self.log_dir / f"system_{timestamp}.log"
            self.current_log.rename(new_name)
            
            # Remove old logs
            old_logs = sorted(
                self.log_dir.glob("system_*.log"),
                key=lambda p: p.stat().st_mtime
            )[:-10]  # Keep last 10 logs
            
            for log in old_logs:
                log.unlink()
                
        except Exception as e:
            logging.error(f"Log rotation failed: {str(e)}")
    
    async def get_recent_events(self,
                              event_type: Optional[str] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events."""
        try:
            events = []
            
            # First get buffered events
            for event in reversed(self.event_buffer):
                if event_type is None or event["type"] == event_type:
                    events.append(event)
                    if len(events) >= limit:
                        return events
            
            # Then read from file using async_file helper
            from alita.core.async_file import async_read_lines
            try:
                lines = await async_read_lines(self.current_log)
            except Exception as e:
                logging.error(f"Failed to read events file: {e}")
                lines = []

            for line in reversed(lines):
                try:
                    event = json.loads(line)
                    if event_type is None or event["type"] == event_type:
                        events.append(event)
                        if len(events) >= limit:
                            break
                except Exception as e:
                    logging.error(f"Event parse failed: {e}")
            
            return events
            
        except Exception as e:
            logging.error(f"Event retrieval failed: {str(e)}")
            return []

class SystemMonitor:
    """Advanced system monitoring."""
    
    def __init__(self, log_dir: Path):
        self.anomaly_detector = AnomalyDetector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.event_logger = EventLogger(log_dir)
        self.current_metrics = None
        
        # Start monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start system monitoring."""
        async def monitor_loop():
            while True:
                try:
                    # Collect metrics
                    metrics = await self._collect_metrics()
                    self.current_metrics = metrics
                    
                    # Update analyzers
                    self.anomaly_detector.add_metrics(metrics)
                    self.performance_analyzer.add_metrics(metrics)
                    
                    # Check for anomalies
                    anomalies = self.anomaly_detector.detect_anomalies()
                    if anomalies:
                        await self.event_logger.log_event(
                            "anomaly_detected",
                            {"anomalies": anomalies}
                        )
                    
                    # Analyze performance
                    analysis = self.performance_analyzer.analyze_performance()
                    if analysis:
                        for metric, trend in analysis["trends"].items():
                            if trend != "stable":
                                await self.event_logger.log_event(
                                    "performance_trend",
                                    {
                                        "metric": metric,
                                        "trend": trend,
                                        "deviation": 
                                            analysis["deviations"][metric]
                                    }
                                )
                    
                    await asyncio.sleep(60)  # Monitor every minute
                    
                except Exception as e:
                    logging.error(f"Monitoring error: {str(e)}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(monitor_loop())
    
    async def _collect_metrics(self) -> SystemMetrics:
        """Collect system metrics."""
        try:
            # CPU usage
            cpu = psutil.cpu_percent()
            
            # Memory usage
            memory = psutil.virtual_memory().percent
            
            # Disk usage
            disk = psutil.disk_usage('/').percent
            
            # Network usage
            net = psutil.net_io_counters()
            network = (net.bytes_sent + net.bytes_recv) / 1024 / 1024  # MB
            
            # GPU usage
            gpu = None
            if torch.cuda.is_available():
                gpu = float(torch.cuda.memory_allocated()) / \
                      float(torch.cuda.max_memory_allocated()) * 100
            
            # Process count
            process_count = len(list(psutil.process_iter()))
            
            # Error count (from logs)
            error_events = await self.event_logger.get_recent_events(
                event_type="error",
                limit=1000
            )
            error_count = len(error_events)
            
            return SystemMetrics(
                cpu_usage=cpu,
                memory_usage=memory,
                disk_usage=disk,
                network_usage=network,
                gpu_usage=gpu,
                process_count=process_count,
                error_count=error_count,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Metrics collection failed: {str(e)}")
            raise
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            if not self.current_metrics:
                return {"status": "no_metrics"}
            
            # Get current analysis
            analysis = self.performance_analyzer.analyze_performance()
            
            # Get recent events
            events = await self.event_logger.get_recent_events(limit=10)
            
            # Get recent anomalies
            anomalies = self.anomaly_detector.detect_anomalies()
            
            return {
                "current_metrics": {
                    "cpu": self.current_metrics.cpu_usage,
                    "memory": self.current_metrics.memory_usage,
                    "disk": self.current_metrics.disk_usage,
                    "network": self.current_metrics.network_usage,
                    "gpu": self.current_metrics.gpu_usage,
                    "processes": self.current_metrics.process_count,
                    "errors": self.current_metrics.error_count,
                    "timestamp": self.current_metrics.timestamp.isoformat()
                },
                "performance_analysis": analysis,
                "recent_events": events,
                "detected_anomalies": anomalies
            }
            
        except Exception as e:
            logging.error(f"Status retrieval failed: {str(e)}")
            return {"status": "error", "message": str(e)}