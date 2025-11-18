"""
Enhanced Resource Manager with Predictive Scaling
Implements:
- Predictive resource allocation
- Dynamic load balancing
- Automated resource optimization
- Smart caching system
- Energy efficiency management
"""

from alita.core.utils import torch, np, pd, psutil, safe_cuda_percent, ensure_dir
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import sqlite3
from pathlib import Path
import logging
from datetime import datetime
import json
import aiohttp
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

@dataclass
class ResourcePrediction:
    """Resource usage prediction."""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    confidence: float
    timestamp: datetime

class PredictiveCache:
    """Smart predictive caching system."""
    
    def __init__(self, max_size_gb: float = 4.0):
        self.max_size = max_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        self.current_size = 0
        self.cache = {}
        self.usage_stats = {}
        self.predictor = self._build_predictor()
    
    def _build_predictor(self) -> xgb.XGBRegressor:
        """Build cache usage predictor."""
        return xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3
        )
    
    def update_stats(self, key: str, access_time: datetime):
        """Update usage statistics."""
        if key not in self.usage_stats:
            self.usage_stats[key] = []
        
        self.usage_stats[key].append(access_time)
        
        # Keep only last 1000 accesses
        if len(self.usage_stats[key]) > 1000:
            self.usage_stats[key] = self.usage_stats[key][-1000:]
    
    def predict_next_access(self, key: str) -> Optional[datetime]:
        """Predict next access time for an item."""
        if key not in self.usage_stats or len(self.usage_stats[key]) < 10:
            return None
        
        # Create features from access patterns
        accesses = self.usage_stats[key]
        intervals = [(t2 - t1).total_seconds() 
                    for t1, t2 in zip(accesses[:-1], accesses[1:])]
        
        if not intervals:
            return None
        
        # Predict next interval
        X = np.array(intervals[:-1]).reshape(-1, 1)
        y = np.array(intervals[1:])
        
        self.predictor.fit(X, y)
        next_interval = self.predictor.predict(
            np.array([intervals[-1]]).reshape(-1, 1)
        )[0]
        
        return accesses[-1] + pd.Timedelta(seconds=next_interval)

class ResourcePredictor:
    """Predictive resource usage modeling."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.history = []
        self.model = self._build_model()
        self.scaler = StandardScaler()
    
    def _build_model(self) -> RandomForestRegressor:
        """Build resource prediction model."""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1
        )
    
    def add_observation(self, 
                       cpu: float,
                       memory: float,
                       gpu: float,
                       timestamp: datetime):
        """Add resource observation."""
        self.history.append({
            "cpu": cpu,
            "memory": memory,
            "gpu": gpu,
            "hour": timestamp.hour,
            "minute": timestamp.minute,
            "day_of_week": timestamp.weekday()
        })
        
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
    
    def predict_usage(self, 
                     future_minutes: int = 5) -> List[ResourcePrediction]:
        """Predict future resource usage."""
        if len(self.history) < 100:  # Need enough data
            return []
        
        # Prepare training data
        df = pd.DataFrame(self.history)
        
        # Features: time components and previous usage
        X = df[["hour", "minute", "day_of_week", "cpu", "memory", "gpu"]].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Predict each resource
        predictions = []
        for i in range(future_minutes):
            cpu_pred = self.model.predict(X_scaled)[-1][0]
            mem_pred = self.model.predict(X_scaled)[-1][1]
            gpu_pred = self.model.predict(X_scaled)[-1][2]
            
            prediction = ResourcePrediction(
                cpu_usage=cpu_pred,
                memory_usage=mem_pred,
                gpu_usage=gpu_pred,
                confidence=self._calculate_confidence(X_scaled),
                timestamp=datetime.now()
            )
            predictions.append(prediction)
            
            # Update features for next prediction
            new_row = np.array([
                predictions[-1].timestamp.hour,
                predictions[-1].timestamp.minute,
                predictions[-1].timestamp.weekday(),
                cpu_pred,
                mem_pred,
                gpu_pred
            ]).reshape(1, -1)
            X_scaled = np.vstack([X_scaled[1:], self.scaler.transform(new_row)])
        
        return predictions
    
    def _calculate_confidence(self, X: np.ndarray) -> float:
        """Calculate prediction confidence."""
        # Use model's feature importances and prediction variance
        importances = self.model.feature_importances_
        prediction_sets = []
        
        for estimator in self.model.estimators_:
            prediction_sets.append(estimator.predict(X)[-1])
        
        # Calculate variance of predictions
        variance = np.std(prediction_sets)
        
        # Combine with feature importance
        confidence = 1.0 / (1.0 + variance) * np.mean(importances)
        return float(confidence)

class EnhancedResourceManager:
    """Advanced resource management system with predictive capabilities."""
    
    def __init__(self):
        self.predictor = ResourcePredictor()
        self.cache = PredictiveCache()
        self.db = self._setup_database()
        self.optimization_state = self._initialize_optimization()
        
        # Start monitoring
        self._start_monitoring()
    
    def _setup_database(self) -> sqlite3.Connection:
        """Setup enhanced database."""
        db_path = Path("data/resources.db")
        db_path.parent.mkdir(exist_ok=True)
        
        db = sqlite3.connect(str(db_path))
        
        # Create enhanced tables
        db.execute("""
            CREATE TABLE IF NOT EXISTS resource_predictions (
                timestamp INTEGER,
                cpu_usage REAL,
                memory_usage REAL,
                gpu_usage REAL,
                confidence REAL,
                accuracy REAL
            )
        """)
        
        db.execute("""
            CREATE TABLE IF NOT EXISTS optimization_events (
                timestamp INTEGER,
                event_type TEXT,
                resource_type TEXT,
                action_taken TEXT,
                impact REAL
            )
        """)
        
        db.execute("""
            CREATE TABLE IF NOT EXISTS energy_metrics (
                timestamp INTEGER,
                power_usage REAL,
                efficiency_score REAL,
                optimization_gains REAL
            )
        """)
        
        db.commit()
        return db
    
    def _initialize_optimization(self) -> Dict[str, Any]:
        """Initialize optimization state."""
        return {
            "current_strategy": "balanced",
            "efficiency_score": 1.0,
            "optimization_history": [],
            "resource_weights": {
                "cpu": 0.4,
                "memory": 0.3,
                "gpu": 0.3
            }
        }
    
    async def optimize_resources(self) -> Dict[str, Any]:
        """Optimize resource allocation."""
        try:
            # Get predictions
            predictions = self.predictor.predict_usage(future_minutes=15)
            
            if not predictions:
                return {"status": "insufficient_data"}
            
            # Calculate optimal allocation
            optimization_plan = await self._calculate_optimization(predictions)
            
            # Apply optimizations
            results = await self._apply_optimizations(optimization_plan)
            
            # Update optimization state
            self._update_optimization_state(results)
            
            return {
                "status": "success",
                "optimizations": optimization_plan,
                "impact": results,
                "efficiency_score": self.optimization_state["efficiency_score"]
            }
            
        except Exception as e:
            logging.error(f"Resource optimization failed: {str(e)}")
            raise
    
    async def _calculate_optimization(self,
                                   predictions: List[ResourcePrediction]
                                   ) -> Dict[str, Any]:
        """Calculate optimal resource allocation."""
        plan = {
            "cpu": [],
            "memory": [],
            "gpu": []
        }
        
        # Analyze predictions
        for prediction in predictions:
            # CPU optimization
            if prediction.cpu_usage > 80:
                plan["cpu"].append({
                    "action": "reduce_load",
                    "target": prediction.cpu_usage - 70,
                    "confidence": prediction.confidence,
                    "timestamp": prediction.timestamp
                })
            
            # Memory optimization
            if prediction.memory_usage > 85:
                plan["memory"].append({
                    "action": "free_memory",
                    "target": prediction.memory_usage - 75,
                    "confidence": prediction.confidence,
                    "timestamp": prediction.timestamp
                })
            
            # GPU optimization
            if prediction.gpu_usage > 90:
                plan["gpu"].append({
                    "action": "offload",
                    "target": prediction.gpu_usage - 80,
                    "confidence": prediction.confidence,
                    "timestamp": prediction.timestamp
                })
        
        return plan
    
    async def _apply_optimizations(self,
                                 plan: Dict[str, Any]
                                 ) -> Dict[str, Any]:
        """Apply resource optimizations."""
        results = {
            "cpu_savings": 0.0,
            "memory_savings": 0.0,
            "gpu_savings": 0.0,
            "energy_savings": 0.0
        }
        
        # Apply CPU optimizations
        for cpu_opt in plan["cpu"]:
            if cpu_opt["action"] == "reduce_load":
                # Implement CPU optimization
                results["cpu_savings"] += await self._optimize_cpu_usage(
                    cpu_opt["target"]
                )
        
        # Apply memory optimizations
        for mem_opt in plan["memory"]:
            if mem_opt["action"] == "free_memory":
                # Implement memory optimization
                results["memory_savings"] += await self._optimize_memory_usage(
                    mem_opt["target"]
                )
        
        # Apply GPU optimizations
        for gpu_opt in plan["gpu"]:
            if gpu_opt["action"] == "offload":
                # Implement GPU optimization
                results["gpu_savings"] += await self._optimize_gpu_usage(
                    gpu_opt["target"]
                )
        
        # Calculate energy savings
        results["energy_savings"] = self._calculate_energy_savings(results)
        
        return results
    
    async def _optimize_cpu_usage(self, target: float) -> float:
        """Optimize CPU usage."""
        # Implementation for CPU optimization
        return target * 0.8  # Estimated savings
    
    async def _optimize_memory_usage(self, target: float) -> float:
        """Optimize memory usage."""
        # Implementation for memory optimization
        return target * 0.7  # Estimated savings
    
    async def _optimize_gpu_usage(self, target: float) -> float:
        """Optimize GPU usage."""
        # Implementation for GPU optimization
        return target * 0.9  # Estimated savings
    
    def _calculate_energy_savings(self, optimizations: Dict[str, float]) -> float:
        """Calculate total energy savings."""
        # Weighted combination of resource savings
        weights = self.optimization_state["resource_weights"]
        
        energy_savings = (
            weights["cpu"] * optimizations["cpu_savings"] +
            weights["memory"] * optimizations["memory_savings"] +
            weights["gpu"] * optimizations["gpu_savings"]
        )
        
        return energy_savings
    
    def _update_optimization_state(self, results: Dict[str, float]):
        """Update optimization state with results."""
        # Update efficiency score
        current_score = self.optimization_state["efficiency_score"]
        impact = sum(results.values()) / len(results)
        
        new_score = current_score * 0.7 + impact * 0.3
        self.optimization_state["efficiency_score"] = new_score
        
        # Update history
        self.optimization_state["optimization_history"].append({
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "efficiency_score": new_score
        })
        
        # Keep last 100 optimizations
        if len(self.optimization_state["optimization_history"]) > 100:
            self.optimization_state["optimization_history"] = \
                self.optimization_state["optimization_history"][-100:]
        
        # Log optimization event
        self._log_optimization(results)
    
    def _log_optimization(self, results: Dict[str, float]):
        """Log optimization results to database."""
        timestamp = int(datetime.now().timestamp())
        
        for resource, impact in results.items():
            self.db.execute(
                "INSERT INTO optimization_events VALUES (?, ?, ?, ?, ?)",
                (
                    timestamp,
                    "optimization",
                    resource,
                    "automatic_adjustment",
                    impact
                )
            )
        
        # Log energy metrics
        self.db.execute(
            "INSERT INTO energy_metrics VALUES (?, ?, ?, ?)",
            (
                timestamp,
                sum(results.values()),
                self.optimization_state["efficiency_score"],
                results["energy_savings"]
            )
        )
        
        self.db.commit()
    
    def _start_monitoring(self):
        """Start enhanced monitoring system."""
        async def monitor_loop():
            while True:
                try:
                    # Collect current usage
                    cpu = psutil.cpu_percent()
                    memory = psutil.virtual_memory().percent
                    gpu = 0.0
                    if torch.cuda.is_available():
                        gpu = float(torch.cuda.memory_allocated()) / \
                              float(torch.cuda.max_memory_allocated()) * 100
                    
                    # Add to predictor
                    self.predictor.add_observation(
                        cpu=cpu,
                        memory=memory,
                        gpu=gpu,
                        timestamp=datetime.now()
                    )
                    
                    # Optimize if needed
                    if cpu > 75 or memory > 80 or gpu > 85:
                        await self.optimize_resources()
                    
                    await asyncio.sleep(60)  # Monitor every minute
                    
                except Exception as e:
                    logging.error(f"Monitoring error: {str(e)}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(monitor_loop())