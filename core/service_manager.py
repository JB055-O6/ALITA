"""
Advanced Service Manager with Orchestration
Features:
- Service lifecycle management
- Dynamic service scaling
- Health monitoring
- Load balancing
- Fault tolerance
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
import aiohttp
from pathlib import Path
from alita.core.utils import np, pd, ensure_dir
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from contextlib import asynccontextmanager

@dataclass
class ServiceState:
    """Service state information."""
    service_id: str
    status: str
    health_score: float
    load: float
    error_rate: float
    response_time: float
    timestamp: datetime

class LoadPredictor:
    """Predictive load modeling."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.history = []
        self.model = self._build_model()
        self.scaler = StandardScaler()
    
    def _build_model(self) -> xgb.XGBRegressor:
        """Build load prediction model."""
        return xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            n_jobs=-1
        )
    
    def add_observation(self,
                       load: float,
                       time_features: Dict[str, int]):
        """Add load observation."""
        self.history.append({
            "load": load,
            **time_features
        })
        
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]
    
    def predict_load(self, 
                    future_minutes: int = 5
                    ) -> List[float]:
        """Predict future load."""
        if len(self.history) < 100:
            return []
        
        # Prepare data
        df = pd.DataFrame(self.history)
        features = [col for col in df.columns if col != "load"]
        
        X = df[features].values
        y = df["load"].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled[:-1], y[1:])
        
        # Generate predictions
        predictions = []
        current_features = X_scaled[-1:]
        
        for _ in range(future_minutes):
            pred = self.model.predict(current_features)[0]
            predictions.append(pred)
            
            # Update features for next prediction
            new_row = current_features[0].copy()
            # Update time features
            new_row[0] = (new_row[0] + 1) % 24  # hour
            new_row[1] = (new_row[1] + 1) % 60  # minute
            
            current_features = np.array([new_row])
        
        return predictions

class ServiceScaler:
    """Dynamic service scaling."""
    
    def __init__(self):
        self.scale_history = []
        self.min_instances = 1
        self.max_instances = 10
        self.scale_threshold = 0.7
    
    async def scale_service(self,
                          service: 'Service',
                          load_predictions: List[float]
                          ) -> bool:
        """Scale service based on predictions."""
        try:
            current_load = service.state.load if service.state else 0
            max_predicted_load = max(load_predictions) if load_predictions else current_load
            
            # Calculate target instances
            current_instances = len(service.instances)
            target_instances = self._calculate_target_instances(
                current_instances,
                max_predicted_load
            )
            
            if target_instances != current_instances:
                # Scale up
                if target_instances > current_instances:
                    for _ in range(target_instances - current_instances):
                        await service.add_instance()
                # Scale down
                else:
                    for _ in range(current_instances - target_instances):
                        await service.remove_instance()
                
                # Record scaling event
                self.scale_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "service_id": service.service_id,
                    "old_instances": current_instances,
                    "new_instances": target_instances,
                    "predicted_load": max_predicted_load
                })
                
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Scaling failed: {str(e)}")
            return False
    
    def _calculate_target_instances(self,
                                 current: int,
                                 predicted_load: float) -> int:
        """Calculate target number of instances."""
        # Base calculation
        target = int(predicted_load / self.scale_threshold)
        
        # Apply limits
        target = max(self.min_instances, min(self.max_instances, target))
        
        # Prevent rapid oscillation
        if abs(target - current) <= 1:
            return current
        
        return target

class LoadBalancer:
    """Advanced load balancing."""
    
    def __init__(self):
        self.algorithm = "weighted_round_robin"
        self.weights = {}
        self.current_index = 0
    
    async def select_instance(self,
                            service: 'Service'
                            ) -> Optional['ServiceInstance']:
        """Select instance for request."""
        if not service.instances:
            return None
        
        if self.algorithm == "weighted_round_robin":
            return await self._weighted_round_robin(service)
        elif self.algorithm == "least_loaded":
            return await self._least_loaded(service)
        else:
            return await self._random(service)
    
    async def _weighted_round_robin(self,
                                  service: 'Service'
                                  ) -> Optional['ServiceInstance']:
        """Weighted round-robin selection."""
        instances = service.instances
        if not instances:
            return None
        
        # Update weights
        for instance in instances:
            if instance.instance_id not in self.weights:
                self.weights[instance.instance_id] = 1.0
            
            # Adjust weight based on health
            health = await instance.get_health()
            self.weights[instance.instance_id] = 1.0 / (1.0 + health.error_rate)
        
        # Select instance
        total_weight = sum(self.weights.values())
        target = (self.current_index / total_weight) % len(instances)
        
        # Find instance
        current_sum = 0
        for instance in instances:
            current_sum += self.weights[instance.instance_id]
            if current_sum > target:
                self.current_index += 1
                return instance
        
        # Fallback
        self.current_index += 1
        return instances[0]
    
    async def _least_loaded(self,
                           service: 'Service'
                           ) -> Optional['ServiceInstance']:
        """Least loaded instance selection."""
        if not service.instances:
            return None
        
        # Get load for each instance
        loads = []
        for instance in service.instances:
            health = await instance.get_health()
            loads.append((instance, health.load))
        
        # Select least loaded
        return min(loads, key=lambda x: x[1])[0]
    
    async def _random(self,
                     service: 'Service'
                     ) -> Optional['ServiceInstance']:
        """Random instance selection."""
        if not service.instances:
            return None
        
        return np.random.choice(service.instances)

class HealthMonitor:
    """Service health monitoring."""
    
    def __init__(self):
        self.health_history = {}
        self.alert_threshold = 0.7
    
    async def check_health(self,
                         instance: 'ServiceInstance'
                         ) -> Dict[str, Any]:
        """Check instance health."""
        try:
            # Collect metrics
            metrics = await instance.get_metrics()
            
            # Calculate health score
            health_score = self._calculate_health_score(metrics)
            
            # Store in history
            if instance.instance_id not in self.health_history:
                self.health_history[instance.instance_id] = []
            
            self.health_history[instance.instance_id].append({
                "timestamp": datetime.now().isoformat(),
                "health_score": health_score,
                "metrics": metrics
            })
            
            # Trim history
            if len(self.health_history[instance.instance_id]) > 1000:
                self.health_history[instance.instance_id] = \
                    self.health_history[instance.instance_id][-1000:]
            
            # Check for alerts
            if health_score < self.alert_threshold:
                await self._handle_poor_health(instance, health_score, metrics)
            
            return {
                "health_score": health_score,
                "metrics": metrics,
                "status": "healthy" if health_score >= self.alert_threshold else "unhealthy"
            }
            
        except Exception as e:
            logging.error(f"Health check failed: {str(e)}")
            return {
                "health_score": 0.0,
                "metrics": {},
                "status": "error"
            }
    
    def _calculate_health_score(self,
                              metrics: Dict[str, Any]) -> float:
        """Calculate health score from metrics."""
        weights = {
            "error_rate": 0.4,
            "response_time": 0.3,
            "load": 0.3
        }
        
        score = 0.0
        
        # Error rate score (lower is better)
        error_score = 1.0 - min(1.0, metrics["error_rate"])
        score += weights["error_rate"] * error_score
        
        # Response time score (lower is better)
        time_score = 1.0 - min(1.0, metrics["response_time"] / 1000)  # Normalize to seconds
        score += weights["response_time"] * time_score
        
        # Load score (lower is better)
        load_score = 1.0 - min(1.0, metrics["load"])
        score += weights["load"] * load_score
        
        return score
    
    async def _handle_poor_health(self,
                                instance: 'ServiceInstance',
                                health_score: float,
                                metrics: Dict[str, Any]):
        """Handle poor health status."""
        # Log alert
        logging.warning(
            f"Poor health detected for instance {instance.instance_id}: "
            f"score={health_score:.2f}"
        )
        
        # Attempt recovery
        if health_score < 0.3:  # Critical
            await instance.restart()
        elif health_score < 0.5:  # Severe
            await instance.refresh()
        else:  # Moderate
            await instance.optimize()

class ServiceInstance:
    """Service instance management."""
    
    def __init__(self, service_id: str, instance_id: str):
        self.service_id = service_id
        self.instance_id = instance_id
        self.state = None
        self.start_time = datetime.now()
        self.metrics = {
            "requests": 0,
            "errors": 0,
            "total_time": 0
        }
    
    async def start(self) -> bool:
        """Start service instance."""
        try:
            # Implementation for starting instance
            self.state = ServiceState(
                service_id=self.service_id,
                status="starting",
                health_score=1.0,
                load=0.0,
                error_rate=0.0,
                response_time=0.0,
                timestamp=datetime.now()
            )
            
            # Simulate startup
            await asyncio.sleep(2)
            
            self.state.status = "running"
            return True
            
        except Exception as e:
            logging.error(f"Instance start failed: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """Stop service instance."""
        try:
            if self.state:
                self.state.status = "stopping"
                # Simulate shutdown
                await asyncio.sleep(1)
                self.state.status = "stopped"
            return True
        except Exception as e:
            logging.error(f"Instance stop failed: {str(e)}")
            return False
    
    async def restart(self) -> bool:
        """Restart service instance."""
        try:
            await self.stop()
            return await self.start()
        except Exception as e:
            logging.error(f"Instance restart failed: {str(e)}")
            return False
    
    async def refresh(self) -> bool:
        """Refresh service instance."""
        try:
            self.metrics = {
                "requests": 0,
                "errors": 0,
                "total_time": 0
            }
            return True
        except Exception as e:
            logging.error(f"Instance refresh failed: {str(e)}")
            return False
    
    async def optimize(self) -> bool:
        """Optimize service instance."""
        try:
            # Implement optimization logic
            return True
        except Exception as e:
            logging.error(f"Instance optimization failed: {str(e)}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get instance metrics."""
        if not self.metrics["requests"]:
            return {
                "error_rate": 0.0,
                "response_time": 0.0,
                "load": 0.0
            }
        
        return {
            "error_rate": self.metrics["errors"] / self.metrics["requests"],
            "response_time": self.metrics["total_time"] / self.metrics["requests"],
            "load": min(1.0, self.metrics["requests"] / 1000)  # Normalize to 0-1
        }
    
    async def get_health(self) -> ServiceState:
        """Get instance health state."""
        if not self.state:
            return ServiceState(
                service_id=self.service_id,
                status="unknown",
                health_score=0.0,
                load=0.0,
                error_rate=0.0,
                response_time=0.0,
                timestamp=datetime.now()
            )
        
        # Update metrics
        metrics = await self.get_metrics()
        self.state.health_score = 1.0 - metrics["error_rate"]
        self.state.load = metrics["load"]
        self.state.error_rate = metrics["error_rate"]
        self.state.response_time = metrics["response_time"]
        self.state.timestamp = datetime.now()
        
        return self.state

class Service:
    """Service management."""
    
    def __init__(self, service_id: str):
        self.service_id = service_id
        self.instances: List[ServiceInstance] = []
        self.state = None
        self.load_predictor = LoadPredictor()
        self.scaler = ServiceScaler()
        self.load_balancer = LoadBalancer()
        self.health_monitor = HealthMonitor()
    
    async def start(self) -> bool:
        """Start service."""
        try:
            # Start with one instance
            await self.add_instance()
            
            # Start monitoring
            asyncio.create_task(self._monitor())
            
            return True
        except Exception as e:
            logging.error(f"Service start failed: {str(e)}")
            return False
    
    async def stop(self) -> bool:
        """Stop service."""
        try:
            for instance in self.instances:
                await instance.stop()
            self.instances = []
            return True
        except Exception as e:
            logging.error(f"Service stop failed: {str(e)}")
            return False
    
    async def add_instance(self) -> Optional[ServiceInstance]:
        """Add new service instance."""
        try:
            instance = ServiceInstance(
                self.service_id,
                f"{self.service_id}-{len(self.instances)}"
            )
            
            if await instance.start():
                self.instances.append(instance)
                return instance
            
            return None
            
        except Exception as e:
            logging.error(f"Add instance failed: {str(e)}")
            return None
    
    async def remove_instance(self) -> bool:
        """Remove service instance."""
        try:
            if self.instances:
                instance = self.instances.pop()
                return await instance.stop()
            return False
        except Exception as e:
            logging.error(f"Remove instance failed: {str(e)}")
            return False
    
    async def get_instance(self) -> Optional[ServiceInstance]:
        """Get instance for request."""
        return await self.load_balancer.select_instance(self)
    
    async def _monitor(self):
        """Monitor service health and scaling."""
        while True:
            try:
                # Check instance health
                for instance in self.instances:
                    health = await self.health_monitor.check_health(instance)
                    
                    # Update load predictor
                    self.load_predictor.add_observation(
                        health["metrics"]["load"],
                        {
                            "hour": datetime.now().hour,
                            "minute": datetime.now().minute,
                            "day": datetime.now().weekday()
                        }
                    )
                
                # Predict load and scale
                predictions = self.load_predictor.predict_load(future_minutes=15)
                if predictions:
                    await self.scaler.scale_service(self, predictions)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Service monitoring failed: {str(e)}")
                await asyncio.sleep(60)

class ServiceManager:
    """Advanced service manager with orchestration."""
    
    def __init__(self):
        self.services: Dict[str, Service] = {}
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load service manager configuration."""
        try:
            config_path = Path("config/services.json")
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Config load failed: {str(e)}")
        
        # Default config
        return {
            "default_min_instances": 1,
            "default_max_instances": 10,
            "health_check_interval": 60,
            "scale_check_interval": 300
        }
    
    async def create_service(self, service_id: str) -> Optional[Service]:
        """Create new service."""
        try:
            if service_id not in self.services:
                service = Service(service_id)
                if await service.start():
                    self.services[service_id] = service
                    return service
            return None
        except Exception as e:
            logging.error(f"Service creation failed: {str(e)}")
            return None
    
    async def delete_service(self, service_id: str) -> bool:
        """Delete service."""
        try:
            if service_id in self.services:
                service = self.services[service_id]
                if await service.stop():
                    del self.services[service_id]
                    return True
            return False
        except Exception as e:
            logging.error(f"Service deletion failed: {str(e)}")
            return False
    
    async def get_service(self, service_id: str) -> Optional[Service]:
        """Get service by ID."""
        return self.services.get(service_id)
    
    async def list_services(self) -> List[Dict[str, Any]]:
        """List all services."""
        try:
            services = []
            for service_id, service in self.services.items():
                instances = []
                for instance in service.instances:
                    health = await instance.get_health()
                    instances.append({
                        "instance_id": instance.instance_id,
                        "status": health.status,
                        "health_score": health.health_score,
                        "load": health.load
                    })
                
                services.append({
                    "service_id": service_id,
                    "instances": instances,
                    "total_instances": len(service.instances)
                })
            
            return services
            
        except Exception as e:
            logging.error(f"Service listing failed: {str(e)}")
            return []
    
    @asynccontextmanager
    async def managed_service(self, service_id: str):
        """Context manager for temporary service."""
        service = None
        try:
            service = await self.create_service(service_id)
            yield service
        finally:
            if service:
                await self.delete_service(service_id)