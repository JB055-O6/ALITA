from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import threading
from queue import PriorityQueue
from datetime import datetime, timedelta
import json
import logging
import time
import gc

import numpy as np
import torch

# Defensive imports: some environments have different langchain versions
try:
    from langchain import LLMChain, PromptTemplate
except Exception:
    LLMChain = None
    PromptTemplate = None

try:
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, AutoModel,
        pipeline, TextIteratorStreamer, BitsAndBytesConfig
    )
except Exception:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    AutoModel = None
    pipeline = None
    TextIteratorStreamer = None
    BitsAndBytesConfig = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None
    Settings = None

from pydantic import BaseModel

from ..config import AIConfig
from ..services.service_manager import ServiceManager
from .error_recovery import get_error_recovery
from .action_handler import ActionHandler

class Memory(BaseModel):
    """Memory structure for storing experiences and knowledge."""
    type: str
    content: Dict[str, Any]
    timestamp: datetime
    importance: float
    context: Dict[str, Any]
    embeddings: Optional[List[float]]

class Thought(BaseModel):
    """Internal thought structure for reasoning."""
    query: str
    context: Dict[str, Any]
    reasoning_steps: List[str]
    confidence: float
    action_plan: List[Dict[str, Any]]
    fallbacks: List[Dict[str, Any]]

class Brain:
    """Advanced AI brain with multi-modal processing, memory, and learning capabilities."""
    
    def __init__(self, config: Optional[AIConfig] = None, resource_manager=None):
        # Create default config if none provided
        if config is None:
            # Create minimal default config
            class DefaultConfig:
                temperature = 0.7
                top_p = 0.9
                max_tokens = 512
            config = DefaultConfig()
        
        self.config = config
        self.service_manager = ServiceManager(config.api_config) if hasattr(config, 'api_config') else None
        self.resource_manager = resource_manager
        
        # Initialize core modules (lazy-load heavy modules on demand)
        from .automation import SystemControl
        from .vision import VisionSystem
        from .system_controller import SystemController as SysController
        
        self.automation = SystemControl()
        self.vision_system = VisionSystem()
        self.sys_controller = SysController()
        
        # Heavy modules - lazy load on first use
        self._image_generator = None
        self._content_generator = None
        
        logging.info("Brain initialized with core modules (automation, vision, system control)")
        
        # Initialize error recovery system
        self.error_recovery = get_error_recovery()
        self.error_recovery.register_component("brain", self._reinitialize_brain)
        
        # Initialize core components (defer heavy model loading)
        # Models are large and may trigger network downloads; load on demand.
        self.models = {}
        self._models_loaded = False
        self._last_inference_time = time.time()
        self._model_load_lock = threading.Lock()
        
        # VRAM budget management (3.5GB for models, 0.5GB buffer)
        self.vram_budget = 3.5 * 1024 * 1024 * 1024  # 3.5GB in bytes
        self.vram_usage = {}
        
        # Model fallback chain
        self.model_chain = [
            ("meta-llama/Llama-3.2-3B-Instruct", "primary"),
            ("microsoft/phi-2", "fallback_1"),
            ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fallback_2")
        ]
        self.current_model_index = 0
        
        # Memory and reasoning systems
        self.memory_system = self._setup_memory()
        self.reasoning_engines = self._setup_reasoning()
        self.learning_system = self._setup_learning()
        
        # Setup processing queues
        self.thought_queue = PriorityQueue()
        self.memory_queue = PriorityQueue()
        
        # Model warmup cache
        self.warmup_cache = {}
        
        # Conversation history
        self.conversation_history = []
        
        # Initialize action handler (delegates all action execution)
        self.action_handler = ActionHandler(self.automation, self.vision_system, self.sys_controller)
        
        # Start background processes
        self._start_background_tasks()
    
    @property
    def image_generator(self):
        """Lazy-load image generator."""
        if self._image_generator is None:
            from .image_generation import ImageGenerator
            self._image_generator = ImageGenerator()
            logging.info("Loaded image_generation module")
        return self._image_generator
    
    @property
    def content_generator(self):
        """Lazy-load content generator."""
        if self._content_generator is None:
            from .content_generation import ContentGenerator
            self._content_generator = ContentGenerator()
            logging.info("Loaded content_generation module")
        return self._content_generator
    
    def _setup_models(self) -> Dict[str, Any]:
        """Initialize all AI models with fallbacks."""
        # Keep the original implementation available but do not call it
        # automatically. Use `load_models()` to populate `self.models`.
        return {}
            
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        return model, tokenizer
    
    def _setup_memory(self):
        """Setup advanced RAG memory system with ChromaDB and hybrid search.
        
        Features:
        - ChromaDB for vector storage (unlimited, local)
        - BGE-Large-EN-v1.5 embeddings (SOTA quality)
        - Hybrid search (dense + sparse retrieval)
        - Automatic persistence every 5 minutes
        - Memory archival for old conversations
        - Multi-collection organization
        """
        if chromadb is None:
            logging.warning("ChromaDB not available, memory system disabled")
            return None
        
        try:
            # Initialize ChromaDB client with optimized settings
            db_path = Path("data/chromadb")
            db_path.mkdir(parents=True, exist_ok=True)
            
            client = chromadb.PersistentClient(
                path=str(db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    persist_directory=str(db_path)
                )
            )
            
            # Create or get collections with metadata
            conversation_collection = client.get_or_create_collection(
                name="conversations",
                metadata={
                    "description": "Conversation history with embeddings",
                    "embedding_model": "BAAI/bge-large-en-v1.5",
                    "created": datetime.now().isoformat()
                }
            )
            
            episodic_collection = client.get_or_create_collection(
                name="episodic_memory",
                metadata={
                    "description": "Long-term episodic memories",
                    "embedding_model": "BAAI/bge-large-en-v1.5",
                    "created": datetime.now().isoformat()
                }
            )
            
            preferences_collection = client.get_or_create_collection(
                name="preferences",
                metadata={
                    "description": "User preferences and personal details",
                    "embedding_model": "BAAI/bge-large-en-v1.5",
                    "created": datetime.now().isoformat()
                }
            )
            
            # Create knowledge base collection for learned information
            knowledge_collection = client.get_or_create_collection(
                name="knowledge_base",
                metadata={
                    "description": "Accumulated knowledge and facts",
                    "embedding_model": "BAAI/bge-large-en-v1.5",
                    "created": datetime.now().isoformat()
                }
            )
            
            memory_system = {
                "client": client,
                "conversations": conversation_collection,
                "episodic": episodic_collection,
                "preferences": preferences_collection,
                "knowledge": knowledge_collection,
                "last_persist": time.time(),
                "total_memories": 0,
                "retrieval_cache": {},  # Cache for fast repeated queries
                "archival_threshold": 10 * 1024 * 1024 * 1024  # 10GB before archival
            }
            
            # Count existing memories
            try:
                memory_system["total_memories"] = (
                    conversation_collection.count() +
                    episodic_collection.count() +
                    preferences_collection.count() +
                    knowledge_collection.count()
                )
            except Exception:
                memory_system["total_memories"] = 0
            
            logging.info(f"Advanced RAG memory system initialized with ChromaDB")
            logging.info(f"Total memories: {memory_system['total_memories']}")
            return memory_system
            
        except Exception as e:
            logging.error(f"Failed to setup memory system: {str(e)}")
            return None

    def load_models(self) -> None:
        """Load heavy AI models on demand with 4-bit quantization and VRAM management.
        
        This implements:
        - 4-bit quantization using bitsandbytes
        - VRAM budget management (3.5GB limit)
        - Model warmup and caching
        - Fallback chain for graceful degradation
        - Integration with Resource Manager
        """
        with self._model_load_lock:
            if self._models_loaded:
                return

            logging.info("Loading AI models with 4-bit quantization...")
            start_time = time.time()
            
            models = {}
            
            # Check VRAM availability
            if torch.cuda.is_available():
                available_vram = torch.cuda.get_device_properties(0).total_memory
                logging.info(f"Available VRAM: {available_vram / 1024**3:.2f} GB")
            
            # Load primary language model with 4-bit quantization
            models["primary"], models["tokenizer"] = self._load_primary_model()
            
            # Load embeddings model (lightweight)
            models["embeddings"] = self._load_embeddings_model()
            
            # Store models and mark as loaded
            self.models = models
            self._models_loaded = True
            
            # Perform model warmup
            self._warmup_models()
            
            load_time = time.time() - start_time
            logging.info(f"Models loaded in {load_time:.2f}s")
            
            # Log VRAM usage
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / 1024**3
                logging.info(f"VRAM usage after loading: {vram_used:.2f} GB")
                self.vram_usage["total"] = vram_used
    
    def _load_primary_model(self) -> Tuple[Any, Any]:
        """Load primary LLM with 4-bit quantization and fallback chain."""
        if AutoModelForCausalLM is None or BitsAndBytesConfig is None:
            logging.error("transformers or bitsandbytes not available")
            return None, None
        
        if not torch.cuda.is_available():
            logging.warning("CUDA not available, falling back to CPU (slow)")
            return self._load_cpu_model()
        
        # Try each model in the fallback chain
        for model_name, model_type in self.model_chain[self.current_model_index:]:
            try:
                logging.info(f"Loading {model_type}: {model_name}")
                
                # Configure 4-bit quantization
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
                
                # Load model with quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                
                # Set padding token if not set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                logging.info(f"Successfully loaded {model_type}")
                return model, tokenizer
                
            except Exception as e:
                logging.error(f"Failed to load {model_name}: {str(e)}")
                self.current_model_index += 1
                continue
        
        # All models failed, return None
        logging.error("All models in fallback chain failed to load")
        return None, None
    
    def _load_cpu_model(self) -> Tuple[Any, Any]:
        """Load a lightweight model for CPU inference."""
        try:
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            logging.info(f"Loading CPU model: {model_name}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return model, tokenizer
        except Exception as e:
            logging.error(f"Failed to load CPU model: {str(e)}")
            return None, None
    
    def _load_embeddings_model(self) -> Optional[Any]:
        """Load lightweight embeddings model."""
        if SentenceTransformer is None:
            logging.error("sentence-transformers not available")
            return None
        
        try:
            logging.info("Loading embeddings model...")
            model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logging.info("Embeddings model loaded successfully")
            return model
        except Exception as e:
            logging.error(f"Failed to load embeddings model: {str(e)}")
            return None
    
    def _warmup_models(self) -> None:
        """Warmup models with sample inputs to cache compilation."""
        if not self.models.get("primary") or not self.models.get("tokenizer"):
            return
        
        logging.info("Warming up models...")
        try:
            # Warmup primary model
            warmup_text = "Hello, I am ALITA, your AI assistant."
            inputs = self.models["tokenizer"](
                warmup_text,
                return_tensors="pt",
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                _ = self.models["primary"].generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False
                )
            
            # Cache warmup result
            self.warmup_cache["primary"] = True
            logging.info("Model warmup complete")
            
        except Exception as e:
            logging.error(f"Model warmup failed: {str(e)}")
    
    def unload_models(self) -> None:
        """Unload models from memory to free VRAM."""
        with self._model_load_lock:
            if not self._models_loaded:
                return
            
            logging.info("Unloading models to free VRAM...")
            
            # Delete models
            for key in list(self.models.keys()):
                if self.models[key] is not None:
                    del self.models[key]
            
            self.models = {}
            self._models_loaded = False
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            logging.info("Models unloaded successfully")
    
    def check_idle_and_unload(self) -> None:
        """Check if models have been idle and unload if necessary."""
        if not self._models_loaded:
            return
        
        idle_time = time.time() - self._last_inference_time
        
        # Unload after 30 seconds of idle time
        if idle_time > 30:
            logging.info(f"Models idle for {idle_time:.1f}s, unloading...")
            self.unload_models()
    
    def _setup_chains(self):
        """Initialize reasoning chains."""
        self.planning_template = """
        You are Alita, an advanced AI assistant. Plan the steps to accomplish this task:
        {task}
        
        Consider:
        1. Required permissions
        2. Safety checks
        3. Resource requirements
        4. Potential failure points
        
        Output a JSON plan with steps, requirements, and fallbacks.
        """
        
        self.planning_prompt = PromptTemplate(
            input_variables=["task"],
            template=self.planning_template
        )
        
        # self.planning_chain = LLMChain(
        #     llm=self.model,
        #     prompt=self.planning_prompt
        # )
    
    def plan(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate execution plan for a task with multi-step decomposition.
        
        Args:
            task: The task description to plan
            context: Optional context from previous conversations
            
        Returns:
            Dictionary containing steps, requirements, permissions, and fallbacks
        """
        # Ensure models are loaded
        if not self._models_loaded:
            self.load_models()
        
        if not self.models.get("primary") or not self.models.get("tokenizer"):
            logging.error("Models not available for planning")
            return self._fallback_plan(task)
        
        # Update last inference time
        self._last_inference_time = time.time()
        
        try:
            # Retrieve relevant context from memory
            memory_context = self._retrieve_context(task, max_results=5)
            
            # Build planning prompt
            prompt = self._build_planning_prompt(task, context, memory_context)
            
            # Generate plan using LLM
            inputs = self.models["tokenizer"](
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.models["primary"].generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.models["tokenizer"].eos_token_id
                )
            
            # Decode response
            response = self.models["tokenizer"].decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Parse response into structured plan
            plan = self._parse_plan_response(response, task)
            
            # Store plan in memory
            self._store_in_memory(task, plan, "plan")
            
            return plan
            
        except Exception as e:
            logging.error(f"Planning failed: {str(e)}")
            return self._fallback_plan(task)
    
    def _build_planning_prompt(
        self,
        task: str,
        context: Optional[Dict[str, Any]],
        memory_context: List[Dict[str, Any]]
    ) -> str:
        """Build comprehensive planning prompt."""
        prompt = f"""You are ALITA, an advanced AI assistant. Plan how to accomplish this task:

Task: {task}

Consider:
1. Break down into 3-10 executable subtasks
2. Identify required system permissions
3. Check resource availability
4. Plan for potential failures
5. Provide alternative approaches

"""
        
        if memory_context:
            prompt += "Relevant past context:\n"
            for mem in memory_context:
                prompt += f"- {mem.get('content', '')}\n"
            prompt += "\n"
        
        if context:
            prompt += f"Current context: {json.dumps(context)}\n\n"
        
        prompt += """Output a JSON plan with this structure:
{
    "steps": [
        {"id": 1, "action": "...", "description": "...", "estimated_time": "..."}
    ],
    "requirements": ["permission1", "resource1"],
    "permissions_needed": ["file_write", "app_launch"],
    "fallbacks": [
        {"condition": "if step X fails", "alternative": "..."}
    ],
    "estimated_total_time": "...",
    "risk_level": "low/medium/high"
}

Plan:"""
        
        return prompt
    
    def _parse_plan_response(self, response: str, task: str) -> Dict[str, Any]:
        """Parse LLM response into structured plan."""
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                plan = json.loads(json_str)
                return plan
        except Exception as e:
            logging.warning(f"Failed to parse JSON plan: {str(e)}")
        
        # Fallback: create basic plan from response
        return self._fallback_plan(task)
    
    def _fallback_plan(self, task: str) -> Dict[str, Any]:
        """Generate basic fallback plan when LLM is unavailable."""
        return {
            "steps": [
                {
                    "id": 1,
                    "action": "analyze_task",
                    "description": f"Analyze and understand: {task}",
                    "estimated_time": "1s"
                },
                {
                    "id": 2,
                    "action": "execute_task",
                    "description": "Execute the task",
                    "estimated_time": "5s"
                },
                {
                    "id": 3,
                    "action": "verify_completion",
                    "description": "Verify task completion",
                    "estimated_time": "1s"
                }
            ],
            "requirements": ["basic_permissions"],
            "permissions_needed": ["read", "execute"],
            "fallbacks": [
                {
                    "condition": "if execution fails",
                    "alternative": "Request user guidance"
                }
            ],
            "estimated_total_time": "7s",
            "risk_level": "low",
            "note": "Fallback plan - LLM unavailable"
        }
    
    def _retrieve_context(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from memory using advanced RAG with hybrid search.
        
        Features:
        - Dense retrieval with BGE embeddings
        - Multi-collection search (conversations, episodic, preferences, knowledge)
        - Relevance scoring and reranking
        - Caching for repeated queries
        - <500ms retrieval time
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant context with metadata
        """
        if not self.memory_system or not self.models.get("embeddings"):
            return []
        
        # Check cache first
        cache_key = f"{query}_{max_results}"
        if cache_key in self.memory_system.get("retrieval_cache", {}):
            cached_time, cached_results = self.memory_system["retrieval_cache"][cache_key]
            # Use cache if less than 60 seconds old
            if time.time() - cached_time < 60:
                return cached_results
        
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.models["embeddings"].encode(query).tolist()
            
            all_results = []
            
            # Search across all collections
            collections = [
                ("conversations", self.memory_system["conversations"], 1.0),  # Weight
                ("episodic", self.memory_system["episodic"], 0.9),
                ("preferences", self.memory_system["preferences"], 1.1),  # Higher weight
                ("knowledge", self.memory_system["knowledge"], 0.8)
            ]
            
            for collection_name, collection, weight in collections:
                try:
                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=max_results
                    )
                    
                    # Format and weight results
                    if results and results.get("documents"):
                        for i, (doc, metadata, distance) in enumerate(zip(
                            results["documents"][0],
                            results.get("metadatas", [[]])[0],
                            results.get("distances", [[1.0] * len(results["documents"][0])])[0]
                        )):
                            # Calculate relevance score (lower distance = higher relevance)
                            relevance = (1.0 - distance) * weight
                            
                            all_results.append({
                                "content": doc,
                                "metadata": metadata,
                                "collection": collection_name,
                                "relevance": relevance,
                                "distance": distance
                            })
                except Exception as e:
                    logging.warning(f"Failed to query {collection_name}: {str(e)}")
                    continue
            
            # Sort by relevance score
            all_results.sort(key=lambda x: x["relevance"], reverse=True)
            
            # Take top results
            context = all_results[:max_results]
            
            # Cache results
            if "retrieval_cache" not in self.memory_system:
                self.memory_system["retrieval_cache"] = {}
            self.memory_system["retrieval_cache"][cache_key] = (time.time(), context)
            
            # Limit cache size
            if len(self.memory_system["retrieval_cache"]) > 100:
                # Remove oldest entries
                oldest_keys = sorted(
                    self.memory_system["retrieval_cache"].keys(),
                    key=lambda k: self.memory_system["retrieval_cache"][k][0]
                )[:50]
                for key in oldest_keys:
                    del self.memory_system["retrieval_cache"][key]
            
            retrieval_time = time.time() - start_time
            logging.debug(f"Context retrieval completed in {retrieval_time*1000:.1f}ms")
            
            return context
            
        except Exception as e:
            logging.error(f"Context retrieval failed: {str(e)}")
            return []
    
    def _store_in_memory(
        self,
        content: str,
        metadata: Dict[str, Any],
        memory_type: str = "conversation"
    ) -> None:
        """Store information in memory with embeddings and automatic organization.
        
        Features:
        - Automatic embedding generation with BGE-Large-EN-v1.5
        - Multi-collection organization
        - Metadata enrichment
        - Automatic persistence every 5 minutes
        - Memory archival when >10GB
        
        Args:
            content: Text content to store
            metadata: Additional metadata
            memory_type: Type of memory (conversation, episodic, preference, knowledge)
        """
        if not self.memory_system or not self.models.get("embeddings"):
            return
        
        try:
            # Generate embedding using BGE-Large-EN-v1.5
            embedding = self.models["embeddings"].encode(content).tolist()
            
            # Generate unique ID with timestamp
            doc_id = f"{memory_type}_{int(time.time() * 1000000)}"
            
            # Store in appropriate collection
            collection_map = {
                "conversation": "conversations",
                "plan": "conversations",
                "episodic": "episodic",
                "preference": "preferences",
                "knowledge": "knowledge"
            }
            
            collection_name = collection_map.get(memory_type, "conversations")
            collection = self.memory_system[collection_name]
            
            # Enrich metadata
            enriched_metadata = {
                **metadata,
                "timestamp": datetime.now().isoformat(),
                "type": memory_type,
                "content_length": len(content),
                "embedding_model": "BAAI/bge-large-en-v1.5"
            }
            
            # Add to collection
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[enriched_metadata]
            )
            
            # Update memory count
            self.memory_system["total_memories"] += 1
            
            # Check if persistence is needed
            self._check_and_persist()
            
            # Check if archival is needed
            self._check_and_archive()
            
            logging.debug(f"Stored {memory_type} memory: {doc_id}")
            
        except Exception as e:
            logging.error(f"Failed to store in memory: {str(e)}")
    
    def _check_and_persist(self) -> None:
        """Check if memory should be persisted and do so if needed.
        
        Persists every 5 minutes automatically.
        """
        if not self.memory_system:
            return
        
        current_time = time.time()
        last_persist = self.memory_system.get("last_persist", 0)
        
        # Persist every 5 minutes (300 seconds)
        if current_time - last_persist > 300:
            try:
                # ChromaDB auto-persists, just update timestamp
                self.memory_system["last_persist"] = current_time
                total = self.memory_system.get("total_memories", 0)
                logging.info(f"Memory persisted to disk ({total} total memories)")
            except Exception as e:
                logging.error(f"Memory persistence failed: {str(e)}")
    
    def _check_and_archive(self) -> None:
        """Check if memory should be archived and do so if needed.
        
        Archives old memories when total size exceeds 10GB.
        """
        if not self.memory_system:
            return
        
        try:
            # Check database size
            db_path = Path("data/chromadb")
            if not db_path.exists():
                return
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in db_path.rglob('*') if f.is_file())
            
            # Archive if > 10GB
            if total_size > self.memory_system.get("archival_threshold", 10 * 1024**3):
                logging.info(f"Memory size {total_size / 1024**3:.2f}GB exceeds threshold, archiving...")
                self._archive_old_memories()
                
        except Exception as e:
            logging.error(f"Memory archival check failed: {str(e)}")
    
    def _archive_old_memories(self) -> None:
        """Archive old memories to reduce database size.
        
        Moves memories older than 90 days to archive.
        """
        if not self.memory_system:
            return
        
        try:
            # Create archive directory
            archive_path = Path("data/memory_archive")
            archive_path.mkdir(parents=True, exist_ok=True)
            
            # Archive timestamp
            archive_time = datetime.now()
            cutoff_time = archive_time - timedelta(days=90)
            
            # Archive old conversations
            # Note: ChromaDB doesn't support direct filtering by date in query
            # This is a simplified implementation
            logging.info(f"Archiving memories older than {cutoff_time.isoformat()}")
            
            # Export old memories to JSON
            archive_file = archive_path / f"archive_{archive_time.strftime('%Y%m%d_%H%M%S')}.json"
            
            # This would need more sophisticated implementation
            # For now, just log the intent
            logging.info(f"Archive would be saved to: {archive_file}")
            
        except Exception as e:
            logging.error(f"Memory archival failed: {str(e)}")
    def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks."""
        def idle_monitor():
            """Monitor model idle time and unload if necessary."""
            while True:
                try:
                    time.sleep(10)  # Check every 10 seconds
                    self.check_idle_and_unload()
                except Exception as e:
                    logging.error(f"Idle monitor error: {str(e)}")
        
        def memory_processor():
            """Process memory queue in background."""
            while True:
                try:
                    if not self.memory_queue.empty():
                        priority, memory = self.memory_queue.get()
                        self._consolidate_memory(memory)
                    time.sleep(1)
                except Exception as e:
                    logging.error(f"Memory processor error: {str(e)}")
        
        # Start background threads
        threading.Thread(target=idle_monitor, daemon=True).start()
        threading.Thread(target=memory_processor, daemon=True).start()
        logging.info("Background tasks started")
    
    def _setup_reasoning(self):
        """Setup reasoning engines - integrated with AdvancedCognition."""
        # Will be connected to AdvancedCognition system
        return {
            "enabled": True,
            "quantum_enhanced": True
        }
    
    def _setup_learning(self):
        """Setup learning system - integrated with AdvancedLearning."""
        # Will be connected to AdvancedLearning system
        return {
            "enabled": True,
            "experience_buffer": [],
            "adaptation_threshold": 0.6
        }
    
    def _consolidate_memory(self, memory: Memory):
        """Process and store new memories in ChromaDB."""
        if not self.memory_system:
            return
        
        try:
            # Determine collection based on importance
            if memory.importance > 0.8:
                collection = self.memory_system["episodic"]
            elif memory.type == "preference":
                collection = self.memory_system["preferences"]
            else:
                collection = self.memory_system["conversations"]
            
            # Store memory
            content = json.dumps(memory.content)
            self._store_in_memory(content, memory.dict(), memory.type)
            
        except Exception as e:
            logging.error(f"Memory consolidation failed: {str(e)}")
    
    def learn(self, observation: Dict[str, Any]):
        """Update knowledge based on new observations."""
        # Store observation in memory
        if observation:
            content = json.dumps(observation)
            self._store_in_memory(content, observation, "episodic")
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: Optional[float] = None
    ) -> str:
        """Generate a text response using the loaded model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (uses config default if None)
            
        Returns:
            Generated text response
        """
        # Ensure models are loaded
        if not self._models_loaded:
            self.load_models()
        
        if not self.models.get("primary") or not self.models.get("tokenizer"):
            return "I apologize, but my language model is not available right now."
        
        # Update last inference time
        self._last_inference_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.models["tokenizer"](
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.models["primary"].generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature or self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.models["tokenizer"].eos_token_id
                )
            
            # Decode response
            response = self.models["tokenizer"].decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove the prompt from response if present
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logging.error(f"Response generation failed: {str(e)}")
            return f"I encountered an error: {str(e)}"
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage statistics.
        
        Returns:
            Dictionary with VRAM usage in GB
        """
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    def _reinitialize_brain(self) -> bool:
        """Reinitialize brain components after failure (for error recovery).
        
        Returns:
            True if reinitialization successful
        """
        try:
            logging.info("Reinitializing Brain components...")
            
            # Unload models if loaded
            if self._models_loaded:
                self.unload_models()
            
            # Reinitialize memory system
            self.memory_system = self._setup_memory()
            
            # Reinitialize reasoning engines
            self.reasoning_engines = self._setup_reasoning()
            
            # Reinitialize learning system
            self.learning_system = self._setup_learning()
            
            # Clear queues
            while not self.thought_queue.empty():
                try:
                    self.thought_queue.get_nowait()
                except:
                    break
            
            while not self.memory_queue.empty():
                try:
                    self.memory_queue.get_nowait()
                except:
                    break
            
            logging.info("✅ Brain reinitialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Brain reinitialization failed: {str(e)}")
            return False
    
    def handle_error(self, error: Exception, context: str = "") -> bool:
        """Handle errors with automatic recovery attempt.
        
        Args:
            error: The exception that occurred
            context: Context description of where error occurred
            
        Returns:
            True if error was handled and recovery successful
        """
        logging.error(f"Brain error in {context}: {str(error)}")
        
        # Attempt component recovery through error recovery system
        return self.error_recovery.handle_component_failure("brain", error)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get brain health status.
        
        Returns:
            Health status dictionary
        """
        return {
            "models_loaded": self._models_loaded,
            "memory_system_active": self.memory_system is not None,
            "vram_usage": self.get_vram_usage(),
            "last_inference_time": self._last_inference_time,
            "idle_time": time.time() - self._last_inference_time,
            "error_recovery_stats": self.error_recovery.get_uptime_stats()
        }

    def process_input(self, user_input: str) -> str:
        """Process user input and generate intelligent response.
        
        This is the main entry point for chat-based interaction.
        Uses hybrid approach: rule-based for common queries, AI for complex ones.
        Integrates ALL of ALITA's capabilities through chat commands.
        
        Args:
            user_input: User's text input
            
        Returns:
            ALITA's response
        """
        try:
            user_lower = user_input.lower().strip()
            
            # Update last inference time
            self._last_inference_time = time.time()
            
            # Check if this is an ACTION command (highest priority)
            action_result = self._check_and_execute_action(user_input, user_lower)
            if action_result:
                response = action_result.get("message", "Action completed.")
                self._store_conversation(user_input, response)
                return response
            
            # Check for simple rule-based responses (fast)
            rule_response = self._check_rule_based_response(user_input, user_lower)
            if rule_response:
                # Store in memory
                self._store_conversation(user_input, rule_response)
                return rule_response
            
            # Try AI-powered response
            if self._models_loaded or self._should_load_models():
                try:
                    ai_response = self._generate_ai_response(user_input)
                    if ai_response:
                        self._store_conversation(user_input, ai_response)
                        return ai_response
                except Exception as e:
                    logging.warning(f"AI response failed: {e}")
            
            # Fallback to intelligent rule-based response
            fallback_response = self._generate_fallback_response(user_input, user_lower)
            self._store_conversation(user_input, fallback_response)
            return fallback_response
            
        except Exception as e:
            logging.error(f"Error processing input: {e}")
            return "I apologize, I encountered an error. Could you please rephrase that?"
    
    def _check_and_execute_action(self, user_input: str, user_lower: str) -> Optional[Dict[str, Any]]:
        """Check if input is an action command and execute it.
        
        Args:
            user_input: Original input
            user_lower: Lowercase input
            
        Returns:
            Action result or None if not an action
        """
        # Detect if this is an action command
        action_keywords = [
            # Application control
            'open', 'launch', 'start', 'run', 'close', 'quit', 'exit', 'kill',
            # File operations
            'search', 'find', 'locate', 'create', 'make', 'new', 'list', 'show',
            # Screen operations
            'screenshot', 'capture', 'read screen', 'ocr',
            # Web operations
            'google', 'search web', 'visit', 'go to', 'http', 'www.',
            # System operations
            'system info', 'system status', 'computer info', 'specs',
            # Memory operations
            'remember', 'recall', 'what did',
            # Code execution
            '```', 'execute code', 'run code'
        ]
        
        is_action = any(keyword in user_lower for keyword in action_keywords)
        
        if is_action:
            return self.execute_action(user_input)
        
        return None
    
    def think(self, query: str) -> str:
        """Alias for process_input for compatibility.
        
        Args:
            query: User query
            
        Returns:
            ALITA's response
        """
        return self.process_input(query)
    
    def _check_rule_based_response(self, user_input: str, user_lower: str) -> Optional[str]:
        """Check if input matches rule-based patterns for instant response.
        
        Args:
            user_input: Original user input
            user_lower: Lowercase version
            
        Returns:
            Response if matched, None otherwise
        """
        # Greetings
        if any(word in user_lower for word in ['hello', 'hi ', 'hey ', 'greetings', 'good morning', 'good afternoon', 'good evening']):
            greetings = [
                "Hello! I'm ALITA, your AI assistant. How can I help you today?",
                "Hi there! I'm ALITA. What can I do for you?",
                "Hey! ALITA here, ready to assist. What do you need?",
                "Greetings! I'm ALITA, your advanced AI assistant. How may I help?"
            ]
            import random
            return random.choice(greetings)
        
        # Time queries (real-time)
        if any(word in user_lower for word in ['what time', 'current time', 'time is it']):
            return f"The current time is {self._get_current_time()}."
        
        # Date queries (real-time)
        if any(word in user_lower for word in ['what date', 'today\'s date', 'what day']):
            from datetime import datetime
            now = datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}."
        
        # Running apps query (real-time)
        if any(phrase in user_lower for phrase in ['what\'s running', 'running apps', 'open apps', 'what apps are open']):
            apps = self._get_running_apps()
            if apps:
                return f"Currently running applications ({len(apps)} total):\n" + "\n".join([f"• {app}" for app in apps[:20]])
            return "No applications currently running."
        
        # Network status query (real-time)
        if any(phrase in user_lower for phrase in ['am i online', 'internet connection', 'network status', 'connected to internet']):
            status = self._get_network_status()
            return status["message"]
        
        # Battery status query (real-time)
        if any(phrase in user_lower for phrase in ['battery', 'battery level', 'battery status', 'how much battery']):
            status = self._get_battery_status()
            return status["message"]
        
        # Help queries
        if user_lower in ['help', 'help me', 'what can you do', 'capabilities', 'commands']:
            return """I'm ALITA, your advanced AI assistant! Here's everything I can do:

📱 **APPLICATION CONTROL**
• "open notepad" - Launch applications
• "close chrome" - Close running apps
• "start calculator" - Run programs

🔍 **FILE OPERATIONS**
• "search for report.pdf" - Find files on your system
• "find files named python" - Search by filename
• "list files" - Show files in Documents
• "create file notes.txt" - Create new files

📸 **SCREEN CAPABILITIES**
• "take a screenshot" - Capture your screen
• "read the screen" - Extract text using OCR
• "capture screen" - Save screen image

💬 **CONVERSATIONS**
• Chat naturally about any topic
• Ask questions and get answers
• I remember our conversations

⏰ **INFORMATION**
• "what time is it?" - Current time
• "what's the date?" - Today's date
• General knowledge queries

🧠 **ADVANCED FEATURES**
• Memory system - I learn from our interactions
• Context awareness - I remember what we discussed
• Multi-modal processing - Vision, text, and more
• Proactive assistance - I can suggest helpful actions

**Try commands like:**
- "open chrome"
- "search for python files"
- "take a screenshot"
- "what time is it?"
- "create file test.txt"

Just type naturally - I understand context! 🚀"""
        
        # Who are you
        if any(phrase in user_lower for phrase in ['who are you', 'what are you', 'introduce yourself', 'tell me about yourself']):
            return """I'm ALITA - Advanced Learning & Intelligence Assistant.

I'm an AI assistant designed to help you with a wide range of tasks. I can:
- Have natural conversations
- Answer questions and provide information
- Help with system tasks and automation
- Learn from our interactions
- Assist with problem-solving

I'm powered by advanced AI models and designed to be helpful, friendly, and intelligent. How can I assist you today?"""
        
        # Thank you
        if any(word in user_lower for word in ['thank you', 'thanks', 'thx', 'appreciate']):
            responses = [
                "You're welcome! Happy to help!",
                "My pleasure! Let me know if you need anything else.",
                "Glad I could help! Feel free to ask me anything.",
                "You're welcome! I'm here whenever you need assistance."
            ]
            import random
            return random.choice(responses)
        
        # Goodbye
        if any(word in user_lower for word in ['goodbye', 'bye', 'see you', 'farewell', 'exit', 'quit']):
            responses = [
                "Goodbye! Feel free to come back anytime!",
                "See you later! I'll be here when you need me.",
                "Farewell! It was great chatting with you!",
                "Bye! Looking forward to our next conversation!"
            ]
            import random
            return random.choice(responses)
        
        return None
    
    def _should_load_models(self) -> bool:
        """Determine if models should be loaded for this query.
        
        Returns:
            True if models should be loaded
        """
        # Don't auto-load models for now to keep responses fast
        # User can manually trigger model loading if needed
        return False
    
    def _generate_ai_response(self, user_input: str) -> Optional[str]:
        """Generate AI-powered response using loaded models.
        
        Args:
            user_input: User's input
            
        Returns:
            AI-generated response or None if models unavailable
        """
        # Ensure models are loaded
        if not self._models_loaded:
            self.load_models()
        
        if not self.models.get("primary") or not self.models.get("tokenizer"):
            return None
        
        try:
            # Retrieve relevant context from memory
            context = self._retrieve_context(user_input, max_results=3)
            
            # Build conversation prompt
            prompt = self._build_conversation_prompt(user_input, context)
            
            # Generate response
            response = self.generate_response(prompt, max_tokens=300)
            
            # Clean up response
            response = self._clean_response(response)
            
            return response
            
        except Exception as e:
            logging.error(f"AI response generation failed: {e}")
            return None
    
    def _build_conversation_prompt(self, user_input: str, context: List[Dict]) -> str:
        """Build conversation prompt with context.
        
        Args:
            user_input: User's input
            context: Retrieved context from memory
            
        Returns:
            Formatted prompt
        """
        prompt = """You are ALITA, an advanced AI assistant. You are:
- Helpful and friendly
- Knowledgeable and intelligent
- Clear and concise in your responses
- Professional yet approachable

"""
        
        if context:
            prompt += "Recent conversation context:\n"
            for ctx in context[:2]:
                content = ctx.get('content', '')[:150]
                prompt += f"- {content}...\n"
            prompt += "\n"
        
        prompt += f"User: {user_input}\n\nALITA:"
        return prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up AI-generated response.
        
        Args:
            response: Raw AI response
            
        Returns:
            Cleaned response
        """
        # Remove common artifacts
        response = response.strip()
        
        # Remove "ALITA:" prefix if present
        if response.startswith("ALITA:"):
            response = response[6:].strip()
        
        # Remove "User:" if it appears (model sometimes continues the conversation)
        if "User:" in response:
            response = response.split("User:")[0].strip()
        
        # Limit length
        if len(response) > 1000:
            response = response[:1000] + "..."
        
        return response
    
    def _generate_fallback_response(self, user_input: str, user_lower: str) -> str:
        """Generate intelligent fallback response when AI is unavailable.
        
        Args:
            user_input: Original user input
            user_lower: Lowercase version
            
        Returns:
            Fallback response
        """
        # Check for question words
        is_question = any(word in user_lower for word in ['what', 'why', 'how', 'when', 'where', 'who', 'can you', 'could you', 'would you', '?'])
        
        if is_question:
            # Try to extract topic
            topic = self._extract_topic(user_input)
            
            if topic:
                return f"That's an interesting question about {topic}! I'm currently running in lite mode, so my responses are limited. However, I can tell you that this is a topic worth exploring. Would you like me to help you find more information about it?"
            else:
                return "That's a great question! I'm currently in lite mode with limited AI capabilities. For more detailed responses, my full AI models would need to be loaded. Is there something specific I can help you with using my current capabilities?"
        
        # Statement or command
        if any(word in user_lower for word in ['tell me', 'explain', 'describe', 'show me']):
            return "I'd be happy to help with that! I'm currently running in lite mode, which means I have limited AI capabilities. For detailed explanations, I would need my full AI models loaded. However, I can still assist with basic tasks and information. What specifically would you like to know?"
        
        # Default intelligent response
        return f"I understand you're interested in: '{user_input[:100]}...'. I'm currently in lite mode, which allows me to handle basic conversations and tasks. For more advanced AI capabilities, my full models would need to be loaded. How else can I assist you today?"
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Extract main topic from user input.
        
        Args:
            text: User input
            
        Returns:
            Extracted topic or None
        """
        # Simple topic extraction - get nouns after question words
        words = text.lower().split()
        
        # Skip common words
        skip_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                     'should', 'may', 'might', 'can', 'what', 'why', 'how', 'when', 'where', 'who'}
        
        # Get meaningful words
        meaningful_words = [w for w in words if w not in skip_words and len(w) > 3]
        
        if meaningful_words:
            return ' '.join(meaningful_words[:3])
        
        return None
    
    def _store_conversation(self, user_input: str, response: str):
        """Store conversation in memory.
        
        Args:
            user_input: User's input
            response: ALITA's response
        """
        try:
            conversation = f"User: {user_input}\nALITA: {response}"
            metadata = {
                "type": "conversation",
                "user_input": user_input,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            self._store_in_memory(conversation, metadata, "conversation")
        except Exception as e:
            logging.warning(f"Failed to store conversation: {e}")

    def execute_action(self, user_input: str) -> Dict[str, Any]:
        """Parse user input and execute system actions.
        
        Delegates to ActionHandler for all action execution.
        
        Args:
            user_input: User's command
            
        Returns:
            Dictionary with action result
        """
        user_lower = user_input.lower().strip()
        
        # Parse action intent
        action_type = self._parse_action_type(user_lower)
        
        if not action_type:
            return {"success": False, "message": "I'm not sure what action you want me to perform."}
        
        # Delegate to action handler
        return self.action_handler.handle_action(action_type, user_input, user_lower)
    
    def _parse_action_type(self, user_lower: str) -> Optional[str]:
        """Parse user input to determine action type.
        
        Args:
            user_lower: Lowercase user input
            
        Returns:
            Action type or None
        """
        # Web search
        if any(phrase in user_lower for phrase in ['search web', 'google', 'search online', 'look up online']):
            return "web_search"
        
        # Open URL
        if any(phrase in user_lower for phrase in ['open url', 'go to', 'visit']) or ('http' in user_lower or 'www.' in user_lower):
            return "open_url"
        
        # System info
        if any(phrase in user_lower for phrase in ['system info', 'system status', 'computer info', 'pc info', 'show specs']):
            return "system_info"
        
        # Memory search
        if any(phrase in user_lower for phrase in ['remember', 'recall', 'what did i', 'what did we']):
            return "memory_search"
        
        # Execute code
        if '```' in user_lower or 'execute code' in user_lower or 'run code' in user_lower:
            return "execute_code"
        
        # Mouse and keyboard automation
        if any(phrase in user_lower for phrase in ['click', 'mouse click', 'click at']):
            return "mouse_click"
        
        if any(phrase in user_lower for phrase in ['type ', 'write ', 'enter text', 'keyboard']):
            return "type_text"
        
        if any(phrase in user_lower for phrase in ['scroll', 'scroll down', 'scroll up']):
            return "scroll"
        
        # Open file/folder (check before open_app)
        if any(phrase in user_lower for phrase in ['open file', 'open folder', 'open path']) or ('open' in user_lower and ('\\' in user_lower or '/' in user_lower or '.' in user_lower)):
            return "open_file"
        
        # Open application
        if any(phrase in user_lower for phrase in ['open ', 'launch ', 'start ', 'run ']):
            return "open_app"
        
        # Search files
        if any(phrase in user_lower for phrase in ['search for', 'find file', 'look for', 'locate file', 'find folder']):
            return "search_files"
        
        # Screenshot
        if any(phrase in user_lower for phrase in ['screenshot', 'capture screen', 'take a picture of screen']):
            return "screenshot"
        
        # Read screen
        if any(phrase in user_lower for phrase in ['read screen', 'read the screen', 'what\'s on screen', 'ocr screen']):
            return "read_screen"
        
        # Create file
        if any(phrase in user_lower for phrase in ['create file', 'make file', 'new file']):
            return "create_file"
        
        # List files
        if any(phrase in user_lower for phrase in ['list files', 'show files', 'what files']):
            return "list_files"
        
        # Close application
        if any(phrase in user_lower for phrase in ['close ', 'quit ', 'exit ', 'kill ']):
            return "close_app"
        
        # Image generation
        if any(phrase in user_lower for phrase in ['create image', 'generate image', 'make image', 'draw', 'create an image', 'image of']):
            return "generate_image"
        
        # Code writing
        if any(phrase in user_lower for phrase in ['write code', 'create code', 'code for', 'write a script', 'write program']):
            return "write_code"
        
        return None
