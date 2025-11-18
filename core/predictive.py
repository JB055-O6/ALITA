"""
Predictive Intelligence System

Implements Task 21 requirements:
- Application pre-loading based on time patterns
- Project context detection for tool prediction
- Command prediction using sequence models
- File pre-fetching based on task context
- Proactive documentation suggestions

Integrates with:
- LearningSystem for usage patterns
- SystemMonitor for activity tracking
- ApplicationTracker for app usage data

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import asyncio

try:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
except ImportError:
    np = None
    RandomForestClassifier = None
    xgb = None

try:
    from .learning_system import get_learning_system
except ImportError:
    get_learning_system = None


class ApplicationPredictor:
    """Predict and pre-load applications based on time patterns."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/app_predictions.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Time-based patterns
        self.hourly_patterns = defaultdict(lambda: defaultdict(int))
        self.day_patterns = defaultdict(lambda: defaultdict(int))
        self.context_patterns = defaultdict(lambda: defaultdict(int))
        
        # Prediction model
        self.model = None
        self.model_trained = False
        
        # Pre-loading queue
        self.preload_queue = deque(maxlen=10)
        
        self.load_data()
        
        if RandomForestClassifier:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def load_data(self):
        """Load prediction data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.hourly_patterns = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get("hourly", {}).items()}
                    )
                    self.day_patterns = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get("daily", {}).items()}
                    )
                    self.context_patterns = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get("context", {}).items()}
                    )
            except Exception as e:
                logging.error(f"Failed to load app predictions: {e}")
    
    def save_data(self):
        """Save prediction data."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump({
                    "hourly": {k: dict(v) for k, v in self.hourly_patterns.items()},
                    "daily": {k: dict(v) for k, v in self.day_patterns.items()},
                    "context": {k: dict(v) for k, v in self.context_patterns.items()}
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save app predictions: {e}")
    
    def track_app_launch(self, app_name: str, context: str = "general"):
        """Track application launch for pattern learning."""
        now = datetime.now()
        hour = str(now.hour)
        day = str(now.weekday())
        
        # Update patterns
        self.hourly_patterns[hour][app_name] += 1
        self.day_patterns[day][app_name] += 1
        self.context_patterns[context][app_name] += 1
        
        self.save_data()
        
        # Retrain model if enough data
        if sum(len(v) for v in self.hourly_patterns.values()) >= 100:
            self._train_model()
    
    def _train_model(self):
        """Train prediction model."""
        if not self.model or not np:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for hour, apps in self.hourly_patterns.items():
                for app, count in apps.items():
                    # Features: hour, day, context indicators
                    features = [int(hour)]
                    
                    # Add day patterns
                    for day in range(7):
                        day_count = self.day_patterns[str(day)].get(app, 0)
                        features.append(day_count)
                    
                    X.append(features)
                    y.append(app)
            
            if len(X) >= 10:
                X = np.array(X)
                self.model.fit(X, y)
                self.model_trained = True
                logging.info("Application prediction model trained")
        
        except Exception as e:
            logging.error(f"Model training failed: {e}")
    
    def predict_next_apps(self, context: str = "general", top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict next likely applications.
        
        Args:
            context: Current context
            top_k: Number of predictions to return
            
        Returns:
            List of (app_name, confidence) tuples
        """
        now = datetime.now()
        hour = str(now.hour)
        day = str(now.weekday())
        
        # Get apps for current hour
        hour_apps = self.hourly_patterns.get(hour, {})
        
        # Get apps for current context
        context_apps = self.context_patterns.get(context, {})
        
        # Combine scores
        combined_scores = defaultdict(float)
        
        for app, count in hour_apps.items():
            combined_scores[app] += count * 0.6
        
        for app, count in context_apps.items():
            combined_scores[app] += count * 0.4
        
        # Sort by score
        sorted_apps = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Normalize to probabilities
        total_score = sum(score for _, score in sorted_apps)
        if total_score > 0:
            predictions = [(app, score / total_score) for app, score in sorted_apps[:top_k]]
        else:
            predictions = []
        
        return predictions
    
    def should_preload(self, app_name: str, threshold: float = 0.5) -> bool:
        """Check if app should be pre-loaded.
        
        Args:
            app_name: Application name
            threshold: Confidence threshold
            
        Returns:
            True if should pre-load
        """
        predictions = self.predict_next_apps(top_k=5)
        
        for pred_app, confidence in predictions:
            if pred_app == app_name and confidence >= threshold:
                return True
        
        return False
    
    def add_to_preload_queue(self, app_name: str, priority: float = 0.5):
        """Add application to pre-load queue.
        
        Args:
            app_name: Application name
            priority: Priority (0-1)
        """
        self.preload_queue.append({
            "app": app_name,
            "priority": priority,
            "timestamp": datetime.now().isoformat()
        })


class ProjectContextDetector:
    """Detect project context and predict associated tools."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/project_contexts.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Project → tools mapping
        self.project_tools = defaultdict(lambda: defaultdict(int))
        
        # File type → tools mapping
        self.filetype_tools = defaultdict(lambda: defaultdict(int))
        
        self.load_data()
    
    def load_data(self):
        """Load project context data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.project_tools = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get("projects", {}).items()}
                    )
                    self.filetype_tools = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get("filetypes", {}).items()}
                    )
            except Exception as e:
                logging.error(f"Failed to load project contexts: {e}")
    
    def save_data(self):
        """Save project context data."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump({
                    "projects": {k: dict(v) for k, v in self.project_tools.items()},
                    "filetypes": {k: dict(v) for k, v in self.filetype_tools.items()}
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save project contexts: {e}")
    
    def track_project_tool(self, project_path: str, tool_name: str):
        """Track tool usage in project context."""
        project_key = str(Path(project_path).name)
        self.project_tools[project_key][tool_name] += 1
        
        # Also track by file types in project
        try:
            project_dir = Path(project_path)
            if project_dir.is_dir():
                file_types = set()
                for file in project_dir.rglob("*"):
                    if file.is_file() and file.suffix:
                        file_types.add(file.suffix)
                
                for file_type in file_types:
                    self.filetype_tools[file_type][tool_name] += 1
        except Exception:
            pass
        
        self.save_data()
    
    def predict_tools_for_project(self, project_path: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict tools for project.
        
        Args:
            project_path: Project directory path
            top_k: Number of predictions
            
        Returns:
            List of (tool_name, confidence) tuples
        """
        project_key = str(Path(project_path).name)
        
        # Get tools for this project
        tools = self.project_tools.get(project_key, {})
        
        # Also check file types
        try:
            project_dir = Path(project_path)
            if project_dir.is_dir():
                for file in project_dir.rglob("*"):
                    if file.is_file() and file.suffix:
                        filetype_tools = self.filetype_tools.get(file.suffix, {})
                        for tool, count in filetype_tools.items():
                            tools[tool] = tools.get(tool, 0) + count * 0.5
        except Exception:
            pass
        
        # Sort and normalize
        sorted_tools = sorted(tools.items(), key=lambda x: x[1], reverse=True)
        
        total_score = sum(score for _, score in sorted_tools)
        if total_score > 0:
            predictions = [(tool, score / total_score) for tool, score in sorted_tools[:top_k]]
        else:
            predictions = []
        
        return predictions


class CommandPredictor:
    """Predict next likely voice command using sequence models."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/command_sequences.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Command sequences
        self.command_history = deque(maxlen=1000)
        self.command_transitions = defaultdict(lambda: defaultdict(int))
        self.command_contexts = defaultdict(lambda: defaultdict(int))
        
        # Sequence model
        self.model = None
        self.model_trained = False
        
        self.load_data()
        
        if xgb:
            self.model = xgb.XGBClassifier(n_estimators=100, max_depth=5)
    
    def load_data(self):
        """Load command sequence data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.command_history = deque(data.get("history", []), maxlen=1000)
                    self.command_transitions = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get("transitions", {}).items()}
                    )
                    self.command_contexts = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get("contexts", {}).items()}
                    )
            except Exception as e:
                logging.error(f"Failed to load command sequences: {e}")
    
    def save_data(self):
        """Save command sequence data."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump({
                    "history": list(self.command_history),
                    "transitions": {k: dict(v) for k, v in self.command_transitions.items()},
                    "contexts": {k: dict(v) for k, v in self.command_contexts.items()}
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save command sequences: {e}")
    
    def track_command(self, command: str, context: str = "general"):
        """Track command for sequence learning."""
        # Add to history
        self.command_history.append({
            "command": command,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update transitions (bigram model)
        if len(self.command_history) >= 2:
            prev_command = self.command_history[-2]["command"]
            self.command_transitions[prev_command][command] += 1
        
        # Update context patterns
        self.command_contexts[context][command] += 1
        
        self.save_data()
        
        # Retrain model if enough data
        if len(self.command_history) >= 100:
            self._train_model()
    
    def _train_model(self):
        """Train sequence prediction model."""
        if not self.model or not np:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for i in range(1, len(self.command_history)):
                prev_cmd = self.command_history[i-1]["command"]
                curr_cmd = self.command_history[i]["command"]
                context = self.command_history[i]["context"]
                
                # Simple feature encoding
                features = [
                    hash(prev_cmd) % 1000,
                    hash(context) % 100,
                    datetime.fromisoformat(self.command_history[i]["timestamp"]).hour
                ]
                
                X.append(features)
                y.append(curr_cmd)
            
            if len(X) >= 50:
                X = np.array(X)
                self.model.fit(X, y)
                self.model_trained = True
                logging.info("Command prediction model trained")
        
        except Exception as e:
            logging.error(f"Command model training failed: {e}")
    
    def predict_next_commands(self, context: str = "general", top_k: int = 3) -> List[Tuple[str, float]]:
        """Predict next likely commands.
        
        Args:
            context: Current context
            top_k: Number of predictions
            
        Returns:
            List of (command, confidence) tuples
        """
        if not self.command_history:
            return []
        
        # Get last command
        last_command = self.command_history[-1]["command"]
        
        # Get transitions from last command
        transitions = self.command_transitions.get(last_command, {})
        
        # Get commands for context
        context_commands = self.command_contexts.get(context, {})
        
        # Combine scores
        combined_scores = defaultdict(float)
        
        for cmd, count in transitions.items():
            combined_scores[cmd] += count * 0.7
        
        for cmd, count in context_commands.items():
            combined_scores[cmd] += count * 0.3
        
        # Sort and normalize
        sorted_commands = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        total_score = sum(score for _, score in sorted_commands)
        if total_score > 0:
            predictions = [(cmd, score / total_score) for cmd, score in sorted_commands[:top_k]]
        else:
            predictions = []
        
        return predictions
    
    def get_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy over recent history.
        
        Returns:
            Accuracy percentage (0-1)
        """
        if len(self.command_history) < 10:
            return 0.0
        
        correct = 0
        total = 0
        
        # Check last 50 commands
        for i in range(max(1, len(self.command_history) - 50), len(self.command_history)):
            actual_command = self.command_history[i]["command"]
            
            # Simulate prediction at i-1
            if i > 0:
                prev_command = self.command_history[i-1]["command"]
                transitions = self.command_transitions.get(prev_command, {})
                
                if transitions:
                    predicted = max(transitions, key=transitions.get)
                    if predicted == actual_command:
                        correct += 1
                    total += 1
        
        return correct / total if total > 0 else 0.0


class FilePrefetcher:
    """Pre-fetch files based on task context."""
    
    def __init__(self):
        self.file_access_patterns = defaultdict(lambda: defaultdict(int))
        self.task_file_associations = defaultdict(set)
        self.prefetch_cache = {}
    
    def track_file_access(self, file_path: str, task_context: str):
        """Track file access in task context."""
        self.file_access_patterns[task_context][file_path] += 1
        self.task_file_associations[task_context].add(file_path)
    
    def predict_files_for_task(self, task_context: str, top_k: int = 5) -> List[str]:
        """Predict files needed for task.
        
        Args:
            task_context: Task description or context
            top_k: Number of files to predict
            
        Returns:
            List of file paths
        """
        files = self.file_access_patterns.get(task_context, {})
        
        sorted_files = sorted(files.items(), key=lambda x: x[1], reverse=True)
        
        return [file for file, _ in sorted_files[:top_k]]
    
    async def prefetch_files(self, file_paths: List[str]):
        """Pre-fetch files into cache.
        
        Args:
            file_paths: List of file paths to prefetch
        """
        for file_path in file_paths:
            try:
                path = Path(file_path)
                if path.exists() and path.is_file():
                    # Read file into cache
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        self.prefetch_cache[file_path] = {
                            "content": content,
                            "timestamp": datetime.now().isoformat()
                        }
                    logging.info(f"Prefetched: {file_path}")
            except Exception as e:
                logging.warning(f"Failed to prefetch {file_path}: {e}")


class DocumentationSuggester:
    """Suggest relevant documentation proactively."""
    
    def __init__(self):
        self.code_doc_mapping = {
            "python": ["https://docs.python.org/3/", "https://pypi.org/"],
            "javascript": ["https://developer.mozilla.org/", "https://nodejs.org/docs/"],
            "typescript": ["https://www.typescriptlang.org/docs/"],
            "java": ["https://docs.oracle.com/javase/"],
            "csharp": ["https://docs.microsoft.com/dotnet/"],
            "rust": ["https://doc.rust-lang.org/"],
            "go": ["https://golang.org/doc/"]
        }
        
        self.keyword_docs = {
            "async": "asynchronous programming",
            "await": "asynchronous programming",
            "promise": "promises and async",
            "thread": "multithreading",
            "regex": "regular expressions",
            "sql": "database queries"
        }
    
    def suggest_docs_for_code(self, code: str, language: str) -> List[Dict[str, str]]:
        """Suggest documentation for code.
        
        Args:
            code: Code snippet
            language: Programming language
            
        Returns:
            List of documentation suggestions
        """
        suggestions = []
        
        # Add language-specific docs
        if language in self.code_doc_mapping:
            for doc_url in self.code_doc_mapping[language]:
                suggestions.append({
                    "type": "language_docs",
                    "url": doc_url,
                    "description": f"{language.title()} documentation"
                })
        
        # Check for keywords
        code_lower = code.lower()
        for keyword, topic in self.keyword_docs.items():
            if keyword in code_lower:
                suggestions.append({
                    "type": "topic",
                    "topic": topic,
                    "description": f"Documentation about {topic}"
                })
        
        return suggestions[:5]  # Limit to 5 suggestions


class PredictiveEngine:
    """Main predictive intelligence system."""
    
    def __init__(self):
        self.app_predictor = ApplicationPredictor()
        self.project_detector = ProjectContextDetector()
        self.command_predictor = CommandPredictor()
        self.file_prefetcher = FilePrefetcher()
        self.doc_suggester = DocumentationSuggester()
        
        # Integration with learning system
        self.learning_system = None
        if get_learning_system:
            try:
                self.learning_system = get_learning_system()
            except Exception as e:
                logging.warning(f"Learning system integration failed: {e}")
        
        # Start background prediction
        self._start_prediction_loop()
        
        logging.info("Predictive Engine initialized")
    
    def track_app_launch(self, app_name: str, context: str = "general"):
        """Track application launch."""
        self.app_predictor.track_app_launch(app_name, context)
    
    def track_project_opened(self, project_path: str, tool_name: str):
        """Track project opening with tool."""
        self.project_detector.track_project_tool(project_path, tool_name)
    
    def track_command(self, command: str, context: str = "general"):
        """Track voice command."""
        self.command_predictor.track_command(command, context)
    
    def track_file_access(self, file_path: str, task_context: str):
        """Track file access."""
        self.file_prefetcher.track_file_access(file_path, task_context)
    
    async def predict_and_preload(self, context: str = "general"):
        """Predict and pre-load applications.
        
        Args:
            context: Current context
        """
        # Get predictions
        app_predictions = self.app_predictor.predict_next_apps(context, top_k=3)
        
        # Add to preload queue
        for app, confidence in app_predictions:
            if confidence >= 0.5:
                self.app_predictor.add_to_preload_queue(app, confidence)
                logging.info(f"Queued for preload: {app} (confidence: {confidence:.2f})")
    
    async def handle_project_opened(self, project_path: str) -> List[str]:
        """Handle project opening and predict tools.
        
        Args:
            project_path: Project directory path
            
        Returns:
            List of predicted tool names
        """
        predictions = self.project_detector.predict_tools_for_project(project_path, top_k=3)
        
        predicted_tools = []
        for tool, confidence in predictions:
            if confidence >= 0.3:
                predicted_tools.append(tool)
                logging.info(f"Predicted tool for project: {tool} (confidence: {confidence:.2f})")
        
        return predicted_tools
    
    def predict_next_command(self, context: str = "general") -> Optional[str]:
        """Predict next likely command.
        
        Args:
            context: Current context
            
        Returns:
            Predicted command or None
        """
        predictions = self.command_predictor.predict_next_commands(context, top_k=1)
        
        if predictions and predictions[0][1] >= 0.5:
            return predictions[0][0]
        
        return None
    
    async def prefetch_for_task(self, task_context: str):
        """Pre-fetch files for task.
        
        Args:
            task_context: Task description
        """
        files = self.file_prefetcher.predict_files_for_task(task_context, top_k=5)
        
        if files:
            await self.file_prefetcher.prefetch_files(files)
            logging.info(f"Prefetched {len(files)} files for task: {task_context}")
    
    def suggest_documentation(self, code: str, language: str) -> List[Dict[str, str]]:
        """Suggest documentation for code.
        
        Args:
            code: Code snippet
            language: Programming language
            
        Returns:
            List of documentation suggestions
        """
        return self.doc_suggester.suggest_docs_for_code(code, language)
    
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "command_accuracy": self.command_predictor.get_prediction_accuracy(),
            "preload_queue_size": len(self.app_predictor.preload_queue),
            "tracked_projects": len(self.project_detector.project_tools),
            "command_history_size": len(self.command_predictor.command_history),
            "prefetch_cache_size": len(self.file_prefetcher.prefetch_cache)
        }
    
    def _start_prediction_loop(self):
        """Start background prediction loop."""
        async def prediction_loop():
            while True:
                try:
                    # Periodic predictions
                    await self.predict_and_preload()
                    
                    # Clean old cache
                    self._clean_prefetch_cache()
                    
                    await asyncio.sleep(300)  # Every 5 minutes
                    
                except Exception as e:
                    logging.error(f"Prediction loop error: {e}")
                    await asyncio.sleep(300)
        
        asyncio.create_task(prediction_loop())
    
    def _clean_prefetch_cache(self):
        """Clean old entries from prefetch cache."""
        now = datetime.now()
        to_remove = []
        
        for file_path, data in self.file_prefetcher.prefetch_cache.items():
            timestamp = datetime.fromisoformat(data["timestamp"])
            if (now - timestamp) > timedelta(hours=1):
                to_remove.append(file_path)
        
        for file_path in to_remove:
            del self.file_prefetcher.prefetch_cache[file_path]


# Global instance
_predictive_engine: Optional[PredictiveEngine] = None


def get_predictive_engine() -> PredictiveEngine:
    """Get global predictive engine instance."""
    global _predictive_engine
    
    if _predictive_engine is None:
        _predictive_engine = PredictiveEngine()
    
    return _predictive_engine
