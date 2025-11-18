from typing import Optional, Dict, Any
import os
from pathlib import Path
import requests
import json
import time
from functools import lru_cache
import sqlite3
import threading
from huggingface_hub import HfApi, InferenceClient
from ..config import APIConfig

class ServiceManager:
    """Manages free API services with automatic fallbacks and caching."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup local cache database
        self._setup_cache_db()
        
        # Initialize API clients
        self.hf_api = self._init_huggingface()
        self.stability_client = self._init_stability()
        self.db_client = self._init_database()
        
        # Service health tracking
        self._health_checks = {}
        self._service_locks = {}
        self._start_health_monitor()
    
    def _setup_cache_db(self):
        """Setup SQLite cache database."""
        self.db = sqlite3.connect(
            self.cache_dir / "cache.db",
            check_same_thread=False
        )
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS api_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                timestamp INTEGER
            )
        """)
        self.db.commit()
    
    def _init_huggingface(self) -> Optional[HfApi]:
        """Initialize HuggingFace's free inference API."""
        try:
            if self.config.hf_token:
                return HfApi(token=self.config.hf_token)
            return HfApi()  # Anonymous access
        except Exception as e:
            print(f"HuggingFace API init failed: {e}")
            return None
            
    def _init_stability(self):
        """Initialize Stability AI's free tier."""
        if self.config.stability_key:
            # Setup client with free API key
            return None  # TODO: Implement
        return None
        
    def _init_database(self):
        """Initialize free CockroachDB instance."""
        if self.config.database_url:
            # Setup connection pool
            return None  # TODO: Implement
        return None
    
    @lru_cache(maxsize=1000)
    def get_cached_response(self, key: str) -> Optional[Dict]:
        """Get cached API response."""
        cursor = self.db.execute(
            "SELECT value, timestamp FROM api_cache WHERE key = ?",
            (key,)
        )
        result = cursor.fetchone()
        if result:
            value, timestamp = result
            # Check if cache is fresh (24 hours)
            if time.time() - timestamp < 86400:
                return json.loads(value)
        return None
    
    def cache_response(self, key: str, value: Dict):
        """Cache API response."""
        self.db.execute(
            "INSERT OR REPLACE INTO api_cache (key, value, timestamp) VALUES (?, ?, ?)",
            (key, json.dumps(value), int(time.time()))
        )
        self.db.commit()
    
    def run_inference(self, model: str, inputs: Any, task: str = "text-generation") -> Dict:
        """Run inference using free APIs with fallbacks."""
        cache_key = f"{model}:{task}:{hash(str(inputs))}"
        
        # Check cache first
        cached = self.get_cached_response(cache_key)
        if cached:
            return cached
            
        try:
            # Try HuggingFace inference API first (free)
            if self.hf_api:
                inference = InferenceClient(model=model, token=self.config.hf_token)
                result = inference.text_generation(inputs) if task == "text-generation" else inference(inputs)
                self.cache_response(cache_key, result)
                return result
        except Exception as e:
            print(f"HuggingFace inference failed: {e}")
            
        # Fallback to local model if available
        try:
            from ..core.brain import Brain
            brain = Brain(None)  # Use default config
            result = brain.generate(inputs)
            self.cache_response(cache_key, result)
            return result
        except Exception as e:
            print(f"Local inference failed: {e}")
            
        raise RuntimeError("All inference methods failed")
    
    def _start_health_monitor(self):
        """Start background health monitoring."""
        def monitor():
            while True:
                # Check each service
                for service in ["huggingface", "stability", "database"]:
                    try:
                        self._check_service_health(service)
                    except Exception as e:
                        print(f"Health check failed for {service}: {e}")
                time.sleep(300)  # Check every 5 minutes
                
        threading.Thread(target=monitor, daemon=True).start()
    
    def _check_service_health(self, service: str):
        """Check if a service is healthy and responding."""
        if service == "huggingface":
            if self.hf_api:
                try:
                    self.hf_api.list_models(limit=1)
                    self._health_checks[service] = True
                    return
                except:
                    pass
        self._health_checks[service] = False
        
    def get_service_status(self) -> Dict[str, bool]:
        """Get current status of all services."""
        return self._health_checks.copy()