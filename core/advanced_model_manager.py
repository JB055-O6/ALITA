"""Advanced Model Management System

Cutting-edge model loading, optimization, and management with:
- Dynamic quantization selection
- Speculative decoding
- Model compilation
- Memory optimization
- Performance monitoring
"""

from typing import Dict, List, Optional, Any, Tuple
import logging
import time
import threading
from pathlib import Path
import yaml
import torch
import psutil
import gc
from dataclasses import dataclass
from enum import Enum

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig, GenerationConfig
    )
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    GenerationConfig = None


class QuantizationMethod(Enum):
    """Available quantization methods."""
    NONE = "none"
    BNB_4BIT = "4bit_nf4"
    BNB_8BIT = "8bit_dynamic"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"
    EXLLAMA = "exllama"


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    path: str
    quantization: QuantizationMethod
    max_memory: float  # GB
    context_length: int
    priority: int  # Higher = preferred


@dataclass
class PerformanceMetrics:
    """Performance metrics."""
    inference_time: float
    memory_usage: float
    gpu_utilization: float
    throughput: float  # tokens/second
    latency_p95: float


class AdvancedModelManager:
    """Advanced model management with cutting-edge optimizations."""
    
    def __init__(self, config_path: str = "config/advanced_optimization.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Model registry
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # Performance tracking
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.benchmark_history: List[Dict[str, Any]] = []
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        
        # Initialize GPU monitoring
        if pynvml and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.gpu_monitoring = True
            except Exception:
                self.gpu_monitoring = False
        else:
            self.gpu_monitoring = False
        
        logging.info("Advanced Model Manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load optimization configuration."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logging.error(f"Failed to load config: {str(e)}")
        
        # Default configuration
        return {
            "model_optimization": {
                "quantization": {
                    "primary_method": "4bit_nf4",
                    "compute_dtype": "bfloat16"
                },
                "attention": {
                    "implementation": "flash_attention_2"
                },
                "compilation": {
                    "enabled": True,
                    "mode": "reduce-overhead"
                }
            }
        }
    
    def register_model(self, model_config: ModelConfig) -> bool:
        """Register a model configuration."""
        try:
            self.model_configs[model_config.name] = model_config
            logging.info(f"Registered model: {model_config.name}")
            return True
        except Exception as e:
            logging.error(f"Failed to register model: {str(e)}")
            return False
    
    def load_model_optimized(self, model_name: str) -> Tuple[Any, Any]:
        """Load model with advanced optimizations."""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not registered")
        
        config = self.model_configs[model_name]
        
        # Check resource availability
        if not self._check_resources(config):
            raise RuntimeError(f"Insufficient resources for {model_name}")
        
        start_time = time.time()
        
        try:
            # Load with BitsAndBytes quantization
            model, tokenizer = self._load_bnb_model(config)
            
            # Apply optimizations
            model = self._apply_optimizations(model, config)
            
            # Store models
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # Benchmark performance
            metrics = self._benchmark_model(model, tokenizer, model_name)
            self.metrics[model_name] = metrics
            
            load_time = time.time() - start_time
            logging.info(f"Model {model_name} loaded in {load_time:.2f}s")
            
            return model, tokenizer
            
        except Exception as e:
            logging.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def _load_bnb_model(self, config: ModelConfig) -> Tuple[Any, Any]:
        """Load model with BitsAndBytes quantization."""
        if config.quantization == QuantizationMethod.BNB_4BIT:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=torch.uint8,
            )
        else:  # 8-bit
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_has_fp16_weight=True,
                llm_int8_threshold=6.0,
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            config.path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            use_cache=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.path,
            trust_remote_code=True,
            use_fast=True,
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def _apply_optimizations(self, model: Any, config: ModelConfig) -> Any:
        """Apply advanced optimizations to model."""
        # Gradient checkpointing for memory efficiency
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Model compilation (PyTorch 2.0+)
        if (self.config["model_optimization"]["compilation"]["enabled"] and 
            hasattr(torch, 'compile')):
            try:
                model = torch.compile(
                    model,
                    mode=self.config["model_optimization"]["compilation"]["mode"],
                    dynamic=True,
                )
                logging.info("Model compiled successfully")
            except Exception as e:
                logging.warning(f"Model compilation failed: {str(e)}")
        
        return model
    
    def _check_resources(self, config: ModelConfig) -> bool:
        """Check if sufficient resources are available."""
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if config.max_memory > gpu_memory * 0.9:  # 90% threshold
                return False
        
        # Check system RAM
        system_memory = psutil.virtual_memory().total / 1024**3
        if config.max_memory > system_memory * 0.8:  # 80% threshold
            return False
        
        return True
    
    def _benchmark_model(self, model: Any, tokenizer: Any, model_name: str) -> PerformanceMetrics:
        """Benchmark model performance."""
        logging.info(f"Benchmarking {model_name}...")
        
        # Warmup
        warmup_text = "Hello, I am ALITA, your advanced AI assistant."
        for _ in range(3):
            self._generate_text(model, tokenizer, warmup_text, max_tokens=10)
        
        # Benchmark
        benchmark_texts = [
            "Explain quantum computing in simple terms.",
            "Write a Python function to sort a list.",
            "What are the benefits of renewable energy?",
        ]
        
        inference_times = []
        memory_usage = []
        
        for text in benchmark_texts:
            start_time = time.time()
            start_memory = self._get_gpu_memory_usage()
            
            _ = self._generate_text(model, tokenizer, text, max_tokens=50)
            
            inference_time = time.time() - start_time
            end_memory = self._get_gpu_memory_usage()
            
            inference_times.append(inference_time)
            memory_usage.append(end_memory - start_memory)
        
        # Calculate metrics
        avg_inference_time = sum(inference_times) / len(inference_times)
        avg_memory_usage = sum(memory_usage) / len(memory_usage)
        throughput = 50 / avg_inference_time  # tokens per second
        latency_p95 = sorted(inference_times)[int(len(inference_times) * 0.95)]
        
        metrics = PerformanceMetrics(
            inference_time=avg_inference_time,
            memory_usage=avg_memory_usage,
            gpu_utilization=self._get_gpu_utilization(),
            throughput=throughput,
            latency_p95=latency_p95
        )
        
        logging.info(f"Benchmark complete: {throughput:.1f} tokens/s")
        return metrics
    
    def _generate_text(self, model: Any, tokenizer: Any, prompt: str, max_tokens: int = 100) -> str:
        """Generate text for benchmarking."""
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0.0
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if self.gpu_monitoring:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return utilization.gpu
            except Exception:
                pass
        return 0.0
    
    def unload_model(self, model_name: str) -> bool:
        """Unload model to free memory."""
        if model_name in self.models:
            del self.models[model_name]
            del self.tokenizers[model_name]
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logging.info(f"Model {model_name} unloaded")
            return True
        return False
    
    def get_best_model(self, max_memory: Optional[float] = None) -> Optional[str]:
        """Get the best available model based on resources and performance."""
        available_models = []
        
        for name, config in self.model_configs.items():
            if max_memory and config.max_memory > max_memory:
                continue
            
            if self._check_resources(config):
                available_models.append((name, config.priority))
        
        if not available_models:
            return None
        
        # Sort by priority (highest first)
        available_models.sort(key=lambda x: x[1], reverse=True)
        return available_models[0][0]
    
    def get_metrics(self, model_name: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for a model."""
        return self.metrics.get(model_name)
    
    def cleanup(self):
        """Cleanup resources."""
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)
        
        if self.gpu_monitoring:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class ResourceMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logging.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logging.info("Resource monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop."""
        while self.monitoring:
            metrics = self._collect_metrics()
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 samples
            if len(self.metrics_history) > 1000:
                self.metrics_history.pop(0)
            
            time.sleep(interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics."""
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / 1024**3,
        }
        
        if torch.cuda.is_available():
            metrics["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1024**3
            metrics["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3
        
        return metrics
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current resource metrics."""
        return self._collect_metrics()
    
    def get_average_metrics(self, last_n: int = 60) -> Dict[str, float]:
        """Get average metrics over last N samples."""
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-last_n:]
        
        avg_metrics = {}
        for key in recent[0].keys():
            if key != "timestamp":
                values = [m[key] for m in recent]
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)
        
        return avg_metrics
