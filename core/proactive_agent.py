"""
Proactive Agent for Autonomous Behavior

Implements Task 13 requirements:
- Activity pattern monitoring with time-series analysis
- Frustration detection using typing speed and facial expressions
- Morning greeting system with time-based triggers
- Calendar integration for event reminders
- Preference learning from user feedback
- Low-power mode after extended idle periods

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, time, timedelta
from collections import deque, defaultdict
import threading
import time as time_module
import json

try:
    import numpy as np
    import pandas as pd
except ImportError:
    np = None
    pd = None

from .safety import get_safety_manager


class ActivityPattern:
    """Track and analyze user activity patterns."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.activity_log = deque(maxlen=history_size)
        self.daily_patterns = defaultdict(list)
    
    def log_activity(self, activity_type: str, metadata: Dict[str, Any] = None):
        """Log user activity."""
        entry = {
            "timestamp": datetime.now(),
            "type": activity_type,
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "metadata": metadata or {}
        }
        
        self.activity_log.append(entry)
        
        # Update daily patterns
        hour_key = entry["hour"]
        self.daily_patterns[hour_key].append(activity_type)
    
    def get_typical_activities(self, hour: int) -> List[str]:
        """Get typical activities for a given hour."""
        if hour not in self.daily_patterns:
            return []
        
        # Count activity types
        activities = self.daily_patterns[hour]
        activity_counts = defaultdict(int)
        
        for activity in activities:
            activity_counts[activity] += 1
        
        # Return sorted by frequency
        return sorted(activity_counts.keys(), key=lambda x: activity_counts[x], reverse=True)
    
    def predict_next_activity(self) -> Optional[str]:
        """Predict next likely activity based on patterns."""
        current_hour = datetime.now().hour
        typical = self.get_typical_activities(current_hour)
        
        return typical[0] if typical else None
    
    def get_activity_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get activity summary for time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [a for a in self.activity_log if a["timestamp"] >= cutoff]
        
        if not recent:
            return {"total": 0, "types": {}}
        
        # Count by type
        type_counts = defaultdict(int)
        for activity in recent:
            type_counts[activity["type"]] += 1
        
        return {
            "total": len(recent),
            "types": dict(type_counts),
            "time_range": {
                "start": min(a["timestamp"] for a in recent).isoformat(),
                "end": max(a["timestamp"] for a in recent).isoformat()
            }
        }


class FrustrationDetector:
    """Detect user frustration from typing and facial expressions."""
    
    def __init__(self):
        self.typing_speeds = deque(maxlen=50)
        self.error_counts = deque(maxlen=20)
        self.facial_expressions = deque(maxlen=30)
        self.frustration_threshold = 0.6
    
    def log_typing_event(self, speed_wpm: float, errors: int = 0):
        """Log typing event."""
        self.typing_speeds.append(speed_wpm)
        self.error_counts.append(errors)
    
    def log_facial_expression(self, expression: str):
        """Log facial expression."""
        self.facial_expressions.append({
            "expression": expression,
            "timestamp": datetime.now()
        })
    
    def detect_frustration(self) -> Dict[str, Any]:
        """Detect frustration level.
        
        Returns:
            Dict with frustration_level (0-1), indicators, and suggestions
        """
        indicators = []
        frustration_score = 0.0
        
        # Check typing speed variance (erratic typing)
        if len(self.typing_speeds) >= 10:
            speeds = list(self.typing_speeds)
            avg_speed = sum(speeds) / len(speeds)
            variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
            
            if variance > 100:  # High variance
                frustration_score += 0.3
                indicators.append("erratic_typing")
        
        # Check error rate
        if len(self.error_counts) >= 5:
            recent_errors = list(self.error_counts)[-5:]
            avg_errors = sum(recent_errors) / len(recent_errors)
            
            if avg_errors > 2:
                frustration_score += 0.3
                indicators.append("high_error_rate")
        
        # Check facial expressions
        if len(self.facial_expressions) >= 5:
            recent_expressions = [e["expression"] for e in list(self.facial_expressions)[-5:]]
            negative_count = sum(1 for e in recent_expressions if e in ["angry", "sad", "frustrated"])
            
            if negative_count >= 3:
                frustration_score += 0.4
                indicators.append("negative_expressions")
        
        # Generate suggestions
        suggestions = []
        if frustration_score >= self.frustration_threshold:
            suggestions.append("Take a short break")
            suggestions.append("Try a different approach")
            if "high_error_rate" in indicators:
                suggestions.append("Slow down and focus on accuracy")
        
        return {
            "frustration_level": min(frustration_score, 1.0),
            "is_frustrated": frustration_score >= self.frustration_threshold,
            "indicators": indicators,
            "suggestions": suggestions
        }


class GreetingSystem:
    """Time-based greeting and check-in system."""
    
    def __init__(self):
        self.last_greeting_date = None
        self.greeting_times = {
            "morning": (time(6, 0), time(12, 0)),
            "afternoon": (time(12, 0), time(18, 0)),
            "evening": (time(18, 0), time(22, 0)),
            "night": (time(22, 0), time(6, 0))
        }
    
    def should_greet(self) -> bool:
        """Check if should greet user."""
        today = datetime.now().date()
        
        # Greet once per day
        if self.last_greeting_date == today:
            return False
        
        # Check if it's morning (6 AM - 10 AM)
        current_time = datetime.now().time()
        if time(6, 0) <= current_time <= time(10, 0):
            return True
        
        return False
    
    def get_greeting(self) -> str:
        """Get appropriate greeting based on time."""
        current_time = datetime.now().time()
        
        if time(6, 0) <= current_time < time(12, 0):
            greeting = "Good morning"
        elif time(12, 0) <= current_time < time(18, 0):
            greeting = "Good afternoon"
        elif time(18, 0) <= current_time < time(22, 0):
            greeting = "Good evening"
        else:
            greeting = "Hello"
        
        self.last_greeting_date = datetime.now().date()
        
        return greeting
    
    def get_time_period(self) -> str:
        """Get current time period."""
        current_time = datetime.now().time()
        
        for period, (start, end) in self.greeting_times.items():
            if period == "night":
                # Handle overnight period
                if current_time >= start or current_time < end:
                    return period
            else:
                if start <= current_time < end:
                    return period
        
        return "unknown"


class CalendarIntegration:
    """Simple calendar integration for reminders."""
    
    def __init__(self, calendar_file: Path = None):
        self.calendar_file = calendar_file or Path("data/calendar.json")
        self.calendar_file.parent.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.load_events()
    
    def load_events(self):
        """Load events from file."""
        if self.calendar_file.exists():
            try:
                with open(self.calendar_file, 'r') as f:
                    data = json.load(f)
                    self.events = data.get("events", [])
            except Exception as e:
                logging.error(f"Failed to load calendar: {e}")
                self.events = []
    
    def save_events(self):
        """Save events to file."""
        try:
            with open(self.calendar_file, 'w') as f:
                json.dump({"events": self.events}, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save calendar: {e}")
    
    def add_event(self, title: str, start_time: datetime, duration_minutes: int = 60, description: str = ""):
        """Add calendar event."""
        event = {
            "id": len(self.events) + 1,
            "title": title,
            "start_time": start_time.isoformat(),
            "duration_minutes": duration_minutes,
            "description": description,
            "reminded": False
        }
        
        self.events.append(event)
        self.save_events()
        
        logging.info(f"Event added: {title} at {start_time}")
    
    def get_upcoming_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get upcoming events."""
        now = datetime.now()
        cutoff = now + timedelta(hours=hours)
        
        upcoming = []
        for event in self.events:
            event_time = datetime.fromisoformat(event["start_time"])
            if now <= event_time <= cutoff:
                upcoming.append(event)
        
        return sorted(upcoming, key=lambda e: e["start_time"])
    
    def get_reminders(self, remind_minutes: int = 15) -> List[Dict[str, Any]]:
        """Get events that need reminders."""
        now = datetime.now()
        remind_time = now + timedelta(minutes=remind_minutes)
        
        reminders = []
        for event in self.events:
            if event.get("reminded"):
                continue
            
            event_time = datetime.fromisoformat(event["start_time"])
            if now <= event_time <= remind_time:
                event["reminded"] = True
                reminders.append(event)
        
        if reminders:
            self.save_events()
        
        return reminders


class PreferenceLearning:
    """Learn user preferences from feedback."""
    
    def __init__(self, preferences_file: Path = None):
        self.preferences_file = preferences_file or Path("data/preferences.json")
        self.preferences_file.parent.mkdir(parents=True, exist_ok=True)
        self.preferences = {}
        self.feedback_history = deque(maxlen=100)
        self.load_preferences()
    
    def load_preferences(self):
        """Load preferences from file."""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r') as f:
                    data = json.load(f)
                    self.preferences = data.get("preferences", {})
            except Exception as e:
                logging.error(f"Failed to load preferences: {e}")
    
    def save_preferences(self):
        """Save preferences to file."""
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump({"preferences": self.preferences}, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save preferences: {e}")
    
    def record_feedback(self, action: str, feedback: str, context: Dict[str, Any] = None):
        """Record user feedback on an action.
        
        Args:
            action: Action taken
            feedback: "positive", "negative", or "neutral"
            context: Additional context
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "feedback": feedback,
            "context": context or {}
        }
        
        self.feedback_history.append(entry)
        
        # Update preferences
        if action not in self.preferences:
            self.preferences[action] = {"positive": 0, "negative": 0, "neutral": 0}
        
        self.preferences[action][feedback] += 1
        self.save_preferences()
    
    def get_preference_score(self, action: str) -> float:
        """Get preference score for action (-1 to 1)."""
        if action not in self.preferences:
            return 0.0
        
        prefs = self.preferences[action]
        total = sum(prefs.values())
        
        if total == 0:
            return 0.0
        
        score = (prefs["positive"] - prefs["negative"]) / total
        return score
    
    def should_suggest(self, action: str, threshold: float = 0.3) -> bool:
        """Check if action should be suggested based on preferences."""
        score = self.get_preference_score(action)
        return score >= threshold


class ProactiveAgent:
    """Autonomous proactive agent for ALITA."""
    
    def __init__(self):
        # Components
        self.activity_monitor = ActivityPattern()
        self.frustration_detector = FrustrationDetector()
        self.greeting_system = GreetingSystem()
        self.calendar = CalendarIntegration()
        self.preferences = PreferenceLearning()
        
        # State
        self.running = False
        self.idle_time = 0
        self.last_activity_time = datetime.now()
        self.low_power_mode = False
        self.low_power_threshold = 1800  # 30 minutes
        
        # Callbacks
        self.suggestion_callbacks: List[Callable] = []
        self.greeting_callbacks: List[Callable] = []
        self.reminder_callbacks: List[Callable] = []
        
        # Safety integration
        self.safety_manager = get_safety_manager()
        
        logging.info("Proactive Agent initialized")
    
    def start(self):
        """Start proactive agent."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logging.info("✅ Proactive Agent started")
    
    def stop(self):
        """Stop proactive agent."""
        self.running = False
        logging.info("Proactive Agent stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Check emergency stop
                if self.safety_manager.emergency_stop.is_stopped():
                    time_module.sleep(5)
                    continue
                
                # Update idle time
                time_since_activity = (datetime.now() - self.last_activity_time).total_seconds()
                self.idle_time = time_since_activity
                
                # Check for low power mode
                if time_since_activity > self.low_power_threshold and not self.low_power_mode:
                    self._enter_low_power_mode()
                elif time_since_activity <= self.low_power_threshold and self.low_power_mode:
                    self._exit_low_power_mode()
                
                # Skip proactive actions in low power mode
                if self.low_power_mode:
                    time_module.sleep(60)
                    continue
                
                # Check for morning greeting
                if self.greeting_system.should_greet():
                    self._send_greeting()
                
                # Check for calendar reminders
                reminders = self.calendar.get_reminders(remind_minutes=15)
                for reminder in reminders:
                    self._send_reminder(reminder)
                
                # Check for frustration
                frustration = self.frustration_detector.detect_frustration()
                if frustration["is_frustrated"]:
                    self._handle_frustration(frustration)
                
                # Sleep before next check
                time_module.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Proactive agent error: {e}")
                time_module.sleep(60)
    
    def log_activity(self, activity_type: str, metadata: Dict[str, Any] = None):
        """Log user activity."""
        self.activity_monitor.log_activity(activity_type, metadata)
        self.last_activity_time = datetime.now()
        
        # Exit low power mode if active
        if self.low_power_mode:
            self._exit_low_power_mode()
    
    def log_typing(self, speed_wpm: float, errors: int = 0):
        """Log typing event."""
        self.frustration_detector.log_typing_event(speed_wpm, errors)
        self.log_activity("typing", {"speed": speed_wpm, "errors": errors})
    
    def log_facial_expression(self, expression: str):
        """Log facial expression."""
        self.frustration_detector.log_facial_expression(expression)
    
    def add_calendar_event(self, title: str, start_time: datetime, duration_minutes: int = 60):
        """Add calendar event."""
        self.calendar.add_event(title, start_time, duration_minutes)
    
    def record_feedback(self, action: str, feedback: str):
        """Record user feedback."""
        self.preferences.record_feedback(action, feedback)
    
    def register_suggestion_callback(self, callback: Callable):
        """Register callback for suggestions."""
        self.suggestion_callbacks.append(callback)
    
    def register_greeting_callback(self, callback: Callable):
        """Register callback for greetings."""
        self.greeting_callbacks.append(callback)
    
    def register_reminder_callback(self, callback: Callable):
        """Register callback for reminders."""
        self.reminder_callbacks.append(callback)
    
    def _send_greeting(self):
        """Send morning greeting."""
        greeting = self.greeting_system.get_greeting()
        message = f"{greeting}! I'm here to help you today."
        
        # Check for upcoming events
        upcoming = self.calendar.get_upcoming_events(hours=8)
        if upcoming:
            message += f" You have {len(upcoming)} event(s) scheduled today."
        
        logging.info(f"Sending greeting: {message}")
        
        for callback in self.greeting_callbacks:
            try:
                callback(message)
            except Exception as e:
                logging.error(f"Greeting callback failed: {e}")
    
    def _send_reminder(self, event: Dict[str, Any]):
        """Send event reminder."""
        event_time = datetime.fromisoformat(event["start_time"])
        minutes_until = int((event_time - datetime.now()).total_seconds() / 60)
        
        message = f"Reminder: '{event['title']}' starts in {minutes_until} minutes"
        
        logging.info(f"Sending reminder: {message}")
        
        for callback in self.reminder_callbacks:
            try:
                callback(message, event)
            except Exception as e:
                logging.error(f"Reminder callback failed: {e}")
    
    def _handle_frustration(self, frustration: Dict[str, Any]):
        """Handle detected frustration."""
        message = "I notice you might be feeling frustrated. "
        
        if frustration["suggestions"]:
            message += "Here are some suggestions: " + ", ".join(frustration["suggestions"])
        
        logging.info(f"Frustration detected: {frustration['frustration_level']:.2f}")
        
        for callback in self.suggestion_callbacks:
            try:
                callback(message, frustration)
            except Exception as e:
                logging.error(f"Suggestion callback failed: {e}")
    
    def _enter_low_power_mode(self):
        """Enter low power mode."""
        self.low_power_mode = True
        logging.info("⚡ Entering low power mode (idle for 30+ minutes)")
    
    def _exit_low_power_mode(self):
        """Exit low power mode."""
        self.low_power_mode = False
        logging.info("⚡ Exiting low power mode (activity detected)")
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "running": self.running,
            "idle_time_seconds": self.idle_time,
            "low_power_mode": self.low_power_mode,
            "last_activity": self.last_activity_time.isoformat(),
            "activity_summary": self.activity_monitor.get_activity_summary(hours=24),
            "upcoming_events": len(self.calendar.get_upcoming_events(hours=24)),
            "time_period": self.greeting_system.get_time_period()
        }


# Global instance
_proactive_agent: Optional[ProactiveAgent] = None


def get_proactive_agent() -> ProactiveAgent:
    """Get global proactive agent instance."""
    global _proactive_agent
    
    if _proactive_agent is None:
        _proactive_agent = ProactiveAgent()
    
    return _proactive_agent
