"""
Learning System with User Adaptation

Implements Task 17 requirements:
- Application usage tracking
- Voice command shortcut learning
- Response verbosity adaptation based on feedback
- File preference learning (locations, naming patterns)
- 30-day user profile with interests and work patterns

Integrates with:
- AdvancedLearning (learning.py) for neural network-based meta-learning
- Experience replay and skill learning

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

try:
    from .learning import AdvancedLearning, Experience
except ImportError:
    AdvancedLearning = None
    Experience = None


class ApplicationTracker:
    """Track application usage patterns."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/app_usage.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.usage_data = defaultdict(lambda: {"count": 0, "total_time": 0, "last_used": None})
        self.load_data()
    
    def load_data(self):
        """Load usage data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.usage_data = defaultdict(lambda: {"count": 0, "total_time": 0, "last_used": None}, data)
            except Exception as e:
                logging.error(f"Failed to load app usage data: {e}")
    
    def save_data(self):
        """Save usage data."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(dict(self.usage_data), f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save app usage data: {e}")
    
    def track_usage(self, app_name: str, duration_seconds: float = 0):
        """Track application usage."""
        self.usage_data[app_name]["count"] += 1
        self.usage_data[app_name]["total_time"] += duration_seconds
        self.usage_data[app_name]["last_used"] = datetime.now().isoformat()
        self.save_data()
    
    def get_most_used(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most used applications."""
        sorted_apps = sorted(
            self.usage_data.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        
        return [
            {"app": app, **data}
            for app, data in sorted_apps[:limit]
        ]


class VoiceShortcutLearner:
    """Learn voice command shortcuts."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/voice_shortcuts.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.shortcuts = {}
        self.command_frequency = defaultdict(int)
        self.load_data()
    
    def load_data(self):
        """Load shortcut data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.shortcuts = data.get("shortcuts", {})
                    self.command_frequency = defaultdict(int, data.get("frequency", {}))
            except Exception as e:
                logging.error(f"Failed to load voice shortcuts: {e}")
    
    def save_data(self):
        """Save shortcut data."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump({
                    "shortcuts": self.shortcuts,
                    "frequency": dict(self.command_frequency)
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save voice shortcuts: {e}")
    
    def track_command(self, command: str, action: str):
        """Track command usage."""
        self.command_frequency[command] += 1
        
        # Suggest shortcut if command used frequently
        if self.command_frequency[command] >= 5 and command not in self.shortcuts:
            self.suggest_shortcut(command, action)
        
        self.save_data()
    
    def suggest_shortcut(self, command: str, action: str):
        """Suggest a shortcut for frequent command."""
        # Extract key words for shortcut
        words = command.lower().split()
        shortcut = " ".join(words[:2]) if len(words) >= 2 else words[0]
        
        self.shortcuts[shortcut] = {
            "full_command": command,
            "action": action,
            "created": datetime.now().isoformat()
        }
        
        logging.info(f"Learned shortcut: '{shortcut}' → '{command}'")
        self.save_data()
    
    def get_shortcuts(self) -> Dict[str, Any]:
        """Get all learned shortcuts."""
        return self.shortcuts


class VerbosityAdapter:
    """Adapt response verbosity based on feedback."""
    
    def __init__(self):
        self.feedback_history = deque(maxlen=100)
        self.current_verbosity = "normal"  # concise, normal, detailed
        self.verbosity_scores = {"concise": 0, "normal": 0, "detailed": 0}
    
    def record_feedback(self, response_length: int, feedback: str):
        """Record feedback on response verbosity.
        
        Args:
            response_length: Length of response in words
            feedback: "too_short", "just_right", "too_long"
        """
        self.feedback_history.append({
            "length": response_length,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update verbosity preference
        if feedback == "too_short":
            self.verbosity_scores["detailed"] += 1
        elif feedback == "too_long":
            self.verbosity_scores["concise"] += 1
        else:  # just_right
            self.verbosity_scores[self.current_verbosity] += 2
        
        # Adapt verbosity
        self._adapt_verbosity()
    
    def _adapt_verbosity(self):
        """Adapt verbosity based on feedback."""
        if len(self.feedback_history) < 10:
            return
        
        # Get highest scoring verbosity
        best_verbosity = max(self.verbosity_scores, key=self.verbosity_scores.get)
        
        if best_verbosity != self.current_verbosity:
            logging.info(f"Adapting verbosity: {self.current_verbosity} → {best_verbosity}")
            self.current_verbosity = best_verbosity
    
    def get_target_length(self) -> tuple:
        """Get target response length range.
        
        Returns:
            Tuple of (min_words, max_words)
        """
        ranges = {
            "concise": (10, 30),
            "normal": (20, 60),
            "detailed": (40, 120)
        }
        return ranges[self.current_verbosity]


class FilePreferenceLearner:
    """Learn file location and naming preferences."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/file_preferences.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.location_preferences = defaultdict(int)
        self.naming_patterns = defaultdict(int)
        self.file_type_locations = defaultdict(lambda: defaultdict(int))
        self.load_data()
    
    def load_data(self):
        """Load preference data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.location_preferences = defaultdict(int, data.get("locations", {}))
                    self.naming_patterns = defaultdict(int, data.get("naming", {}))
                    self.file_type_locations = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get("type_locations", {}).items()}
                    )
            except Exception as e:
                logging.error(f"Failed to load file preferences: {e}")
    
    def save_data(self):
        """Save preference data."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump({
                    "locations": dict(self.location_preferences),
                    "naming": dict(self.naming_patterns),
                    "type_locations": {k: dict(v) for k, v in self.file_type_locations.items()}
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save file preferences: {e}")
    
    def track_file_operation(self, file_path: str, operation: str = "create"):
        """Track file operation to learn preferences."""
        path = Path(file_path)
        
        # Track location
        location = str(path.parent)
        self.location_preferences[location] += 1
        
        # Track naming pattern
        name_pattern = self._extract_naming_pattern(path.name)
        if name_pattern:
            self.naming_patterns[name_pattern] += 1
        
        # Track file type location
        file_ext = path.suffix
        if file_ext:
            self.file_type_locations[file_ext][location] += 1
        
        self.save_data()
    
    def _extract_naming_pattern(self, filename: str) -> Optional[str]:
        """Extract naming pattern from filename."""
        # Remove extension
        name = Path(filename).stem
        
        # Detect patterns
        if "_" in name:
            return "underscore_separated"
        elif "-" in name:
            return "dash-separated"
        elif name[0].isupper() and any(c.isupper() for c in name[1:]):
            return "CamelCase"
        else:
            return "lowercase"
    
    def suggest_location(self, file_type: str = None) -> str:
        """Suggest file location based on preferences."""
        if file_type and file_type in self.file_type_locations:
            # Get most common location for this file type
            locations = self.file_type_locations[file_type]
            if locations:
                return max(locations, key=locations.get)
        
        # Return most common location overall
        if self.location_preferences:
            return max(self.location_preferences, key=self.location_preferences.get)
        
        return str(Path.home() / "Documents")
    
    def suggest_naming_pattern(self) -> str:
        """Suggest naming pattern based on preferences."""
        if self.naming_patterns:
            return max(self.naming_patterns, key=self.naming_patterns.get)
        return "underscore_separated"


class UserProfileBuilder:
    """Build 30-day user profile with interests and work patterns."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/user_profile.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        self.profile = {
            "interests": defaultdict(int),
            "work_patterns": defaultdict(lambda: defaultdict(int)),
            "skills": defaultdict(int),
            "preferences": {},
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }
        self.load_profile()
    
    def load_profile(self):
        """Load user profile."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.profile["interests"] = defaultdict(int, data.get("interests", {}))
                    self.profile["work_patterns"] = defaultdict(
                        lambda: defaultdict(int),
                        {k: defaultdict(int, v) for k, v in data.get("work_patterns", {}).items()}
                    )
                    self.profile["skills"] = defaultdict(int, data.get("skills", {}))
                    self.profile["preferences"] = data.get("preferences", {})
                    self.profile["created"] = data.get("created", datetime.now().isoformat())
            except Exception as e:
                logging.error(f"Failed to load user profile: {e}")
    
    def save_profile(self):
        """Save user profile."""
        try:
            self.profile["last_updated"] = datetime.now().isoformat()
            
            with open(self.data_file, 'w') as f:
                json.dump({
                    "interests": dict(self.profile["interests"]),
                    "work_patterns": {k: dict(v) for k, v in self.profile["work_patterns"].items()},
                    "skills": dict(self.profile["skills"]),
                    "preferences": self.profile["preferences"],
                    "created": self.profile["created"],
                    "last_updated": self.profile["last_updated"]
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save user profile: {e}")
    
    def track_interest(self, topic: str, weight: int = 1):
        """Track user interest in topic."""
        self.profile["interests"][topic.lower()] += weight
        self.save_profile()
    
    def track_work_pattern(self, activity: str, hour: int):
        """Track work pattern by hour."""
        self.profile["work_patterns"][activity][str(hour)] += 1
        self.save_profile()
    
    def track_skill(self, skill: str):
        """Track skill usage."""
        self.profile["skills"][skill.lower()] += 1
        self.save_profile()
    
    def get_top_interests(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get top interests."""
        return sorted(
            self.profile["interests"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]
    
    def get_work_pattern_summary(self) -> Dict[str, Any]:
        """Get work pattern summary."""
        summary = {}
        
        for activity, hours in self.profile["work_patterns"].items():
            if hours:
                peak_hour = max(hours, key=hours.get)
                summary[activity] = {
                    "peak_hour": int(peak_hour),
                    "total_occurrences": sum(hours.values())
                }
        
        return summary
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get complete profile summary."""
        return {
            "top_interests": self.get_top_interests(5),
            "work_patterns": self.get_work_pattern_summary(),
            "top_skills": sorted(
                self.profile["skills"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "profile_age_days": (
                datetime.now() - datetime.fromisoformat(self.profile["created"])
            ).days
        }


class LearningSystem:
    """Main learning system coordinating all adaptation.
    
    Integrates:
    - User adaptation components (app tracking, voice shortcuts, etc.)
    - AdvancedLearning for neural network-based meta-learning
    """
    
    def __init__(self):
        # User adaptation components
        self.app_tracker = ApplicationTracker()
        self.voice_learner = VoiceShortcutLearner()
        self.verbosity_adapter = VerbosityAdapter()
        self.file_learner = FilePreferenceLearner()
        self.profile_builder = UserProfileBuilder()
        
        # Advanced learning integration
        self.advanced_learning = None
        if AdvancedLearning:
            try:
                self.advanced_learning = AdvancedLearning()
                logging.info("Advanced Learning (neural network) initialized")
            except Exception as e:
                logging.warning(f"Advanced Learning initialization failed: {e}")
        
        logging.info("Learning System initialized")
    
    def track_app_usage(self, app_name: str, duration: float = 0):
        """Track application usage."""
        self.app_tracker.track_usage(app_name, duration)
        self.profile_builder.track_work_pattern("app_usage", datetime.now().hour)
    
    def track_voice_command(self, command: str, action: str):
        """Track voice command."""
        self.voice_learner.track_command(command, action)
    
    def adapt_verbosity(self, response_length: int, feedback: str):
        """Adapt response verbosity."""
        self.verbosity_adapter.record_feedback(response_length, feedback)
    
    def track_file_operation(self, file_path: str):
        """Track file operation."""
        self.file_learner.track_file_operation(file_path)
    
    def track_interest(self, topic: str):
        """Track user interest."""
        self.profile_builder.track_interest(topic)
    
    def track_skill(self, skill: str):
        """Track skill usage."""
        self.profile_builder.track_skill(skill)
    
    async def learn_from_experience(self,
                                   state: Dict[str, Any],
                                   action: Dict[str, Any],
                                   reward: float,
                                   next_state: Dict[str, Any],
                                   metadata: Dict[str, Any] = None):
        """Learn from experience using advanced learning system.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            metadata: Additional metadata
        """
        if self.advanced_learning and Experience:
            try:
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    metadata=metadata or {},
                    timestamp=datetime.now()
                )
                await self.advanced_learning.learn_from_experience(experience)
            except Exception as e:
                logging.error(f"Advanced learning failed: {e}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get complete learning summary."""
        summary = {
            "most_used_apps": self.app_tracker.get_most_used(5),
            "voice_shortcuts": len(self.voice_learner.get_shortcuts()),
            "current_verbosity": self.verbosity_adapter.current_verbosity,
            "preferred_file_location": self.file_learner.suggest_location(),
            "user_profile": self.profile_builder.get_profile_summary()
        }
        
        # Add advanced learning stats if available
        if self.advanced_learning:
            summary["advanced_learning"] = {
                "experience_buffer_size": len(self.advanced_learning.experience_replay["buffer"]),
                "skills_learned": len(self.advanced_learning.current_skills),
                "meta_learning_active": len(self.advanced_learning.experience_replay["buffer"]) >= 10000
            }
        
        return summary


# Global instance
_learning_system: Optional[LearningSystem] = None


def get_learning_system() -> LearningSystem:
    """Get global learning system instance."""
    global _learning_system
    
    if _learning_system is None:
        _learning_system = LearningSystem()
    
    return _learning_system
