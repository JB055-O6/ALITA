"""
Graceful Interruption Handling System

Implements Task 18 requirements:
- Interrupt command handlers
- Action history stack for undo functionality
- Task state serialization for interruptions
- Continuation prompt after extended pauses
- Warning system for non-reversible actions

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime, timedelta
from collections import deque
import json
import pickle


class ActionHistory:
    """Track action history for undo functionality."""
    
    def __init__(self, max_history: int = 100):
        self.history: deque = deque(maxlen=max_history)
        self.redo_stack: deque = deque(maxlen=max_history)
    
    def add_action(self, action: Dict[str, Any]):
        """Add action to history.
        
        Args:
            action: Action dictionary with type, params, timestamp, reversible flag
        """
        action["timestamp"] = datetime.now().isoformat()
        self.history.append(action)
        
        # Clear redo stack when new action is added
        self.redo_stack.clear()
        
        logging.debug(f"Action added to history: {action['type']}")
    
    def undo_last(self) -> Optional[Dict[str, Any]]:
        """Undo last action.
        
        Returns:
            Action that was undone, or None
        """
        if not self.history:
            return None
        
        action = self.history.pop()
        
        # Check if action is reversible
        if not action.get("reversible", True):
            logging.warning(f"Action {action['type']} is not reversible")
            self.history.append(action)  # Put it back
            return None
        
        # Add to redo stack
        self.redo_stack.append(action)
        
        logging.info(f"Undoing action: {action['type']}")
        return action
    
    def redo_last(self) -> Optional[Dict[str, Any]]:
        """Redo last undone action.
        
        Returns:
            Action that was redone, or None
        """
        if not self.redo_stack:
            return None
        
        action = self.redo_stack.pop()
        self.history.append(action)
        
        logging.info(f"Redoing action: {action['type']}")
        return action
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent action history.
        
        Args:
            limit: Number of recent actions to return
            
        Returns:
            List of recent actions
        """
        return list(self.history)[-limit:]
    
    def clear_history(self):
        """Clear all history."""
        self.history.clear()
        self.redo_stack.clear()


class TaskStateManager:
    """Manage task state for interruption and continuation."""
    
    def __init__(self, state_dir: Path = None):
        self.state_dir = state_dir or Path("data/task_states")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.current_task: Optional[Dict[str, Any]] = None
        self.paused_tasks: Dict[str, Dict[str, Any]] = {}
    
    def save_task_state(self, task_id: str, state: Dict[str, Any]):
        """Save task state to disk.
        
        Args:
            task_id: Unique task identifier
            state: Task state dictionary
        """
        try:
            state_file = self.state_dir / f"{task_id}.json"
            
            state["saved_at"] = datetime.now().isoformat()
            state["task_id"] = task_id
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.paused_tasks[task_id] = state
            logging.info(f"Task state saved: {task_id}")
            
        except Exception as e:
            logging.error(f"Failed to save task state: {e}")
    
    def load_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Load task state from disk.
        
        Args:
            task_id: Unique task identifier
            
        Returns:
            Task state dictionary or None
        """
        try:
            state_file = self.state_dir / f"{task_id}.json"
            
            if not state_file.exists():
                return None
            
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            logging.info(f"Task state loaded: {task_id}")
            return state
            
        except Exception as e:
            logging.error(f"Failed to load task state: {e}")
            return None
    
    def delete_task_state(self, task_id: str):
        """Delete task state.
        
        Args:
            task_id: Unique task identifier
        """
        try:
            state_file = self.state_dir / f"{task_id}.json"
            
            if state_file.exists():
                state_file.unlink()
            
            if task_id in self.paused_tasks:
                del self.paused_tasks[task_id]
            
            logging.info(f"Task state deleted: {task_id}")
            
        except Exception as e:
            logging.error(f"Failed to delete task state: {e}")
    
    def list_paused_tasks(self) -> List[Dict[str, Any]]:
        """List all paused tasks.
        
        Returns:
            List of paused task summaries
        """
        paused = []
        
        for state_file in self.state_dir.glob("*.json"):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                paused.append({
                    "task_id": state.get("task_id"),
                    "description": state.get("description", "Unknown task"),
                    "saved_at": state.get("saved_at"),
                    "progress": state.get("progress", 0)
                })
            except Exception:
                continue
        
        return paused


class InterruptHandler:
    """Handle interruptions gracefully."""
    
    def __init__(self):
        self.interrupt_commands = [
            "stop",
            "cancel",
            "abort",
            "pause",
            "wait",
            "hold on",
            "never mind"
        ]
        
        self.continuation_prompts = [
            "Would you like to continue where we left off?",
            "Should I resume the previous task?",
            "Do you want to pick up where we stopped?",
            "Shall I continue with what we were doing?"
        ]
        
        self.last_activity_time = datetime.now()
        self.pause_threshold = timedelta(minutes=5)
        self.interrupted_task: Optional[Dict[str, Any]] = None
    
    def is_interrupt_command(self, command: str) -> bool:
        """Check if command is an interrupt.
        
        Args:
            command: User command
            
        Returns:
            True if command is an interrupt
        """
        command_lower = command.lower().strip()
        
        for interrupt_cmd in self.interrupt_commands:
            if interrupt_cmd in command_lower:
                return True
        
        return False
    
    def handle_interrupt(self, current_task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task interruption.
        
        Args:
            current_task: Currently executing task
            
        Returns:
            Interrupt response with saved state info
        """
        self.interrupted_task = current_task
        self.last_activity_time = datetime.now()
        
        logging.info(f"Task interrupted: {current_task.get('description')}")
        
        return {
            "status": "interrupted",
            "message": "Task paused. Say 'continue' to resume.",
            "task_id": current_task.get("task_id"),
            "can_resume": True
        }
    
    def should_prompt_continuation(self) -> bool:
        """Check if should prompt user to continue.
        
        Returns:
            True if enough time has passed since last activity
        """
        if not self.interrupted_task:
            return False
        
        time_since_activity = datetime.now() - self.last_activity_time
        return time_since_activity > self.pause_threshold
    
    def get_continuation_prompt(self) -> str:
        """Get continuation prompt.
        
        Returns:
            Continuation prompt string
        """
        import random
        return random.choice(self.continuation_prompts)
    
    def update_activity(self):
        """Update last activity time."""
        self.last_activity_time = datetime.now()


class NonReversibleWarning:
    """Warning system for non-reversible actions."""
    
    def __init__(self):
        self.non_reversible_actions = {
            "delete_file": "This will permanently delete the file.",
            "delete_folder": "This will permanently delete the folder and all its contents.",
            "system_shutdown": "This will shut down the system.",
            "format_drive": "This will erase all data on the drive.",
            "send_email": "This will send the email and cannot be unsent.",
            "publish_post": "This will publish the post publicly.",
            "execute_code": "This will execute code that may have irreversible effects."
        }
    
    def is_non_reversible(self, action_type: str) -> bool:
        """Check if action is non-reversible.
        
        Args:
            action_type: Type of action
            
        Returns:
            True if action is non-reversible
        """
        return action_type in self.non_reversible_actions
    
    def get_warning(self, action_type: str) -> Optional[str]:
        """Get warning message for action.
        
        Args:
            action_type: Type of action
            
        Returns:
            Warning message or None
        """
        return self.non_reversible_actions.get(action_type)
    
    def format_warning(self, action_type: str, action_details: Dict[str, Any]) -> str:
        """Format complete warning message.
        
        Args:
            action_type: Type of action
            action_details: Action details
            
        Returns:
            Formatted warning message
        """
        warning = self.get_warning(action_type)
        
        if not warning:
            return ""
        
        message = f"⚠️ WARNING: {warning}\n"
        message += f"Action: {action_type}\n"
        
        if action_details:
            message += "Details:\n"
            for key, value in action_details.items():
                message += f"  - {key}: {value}\n"
        
        message += "\nThis action cannot be undone. Are you sure you want to proceed?"
        
        return message


class InterruptionHandler:
    """Main interruption handling system."""
    
    def __init__(self):
        self.action_history = ActionHistory()
        self.task_state_manager = TaskStateManager()
        self.interrupt_handler = InterruptHandler()
        self.warning_system = NonReversibleWarning()
        
        # Callbacks
        self.undo_callbacks: Dict[str, Callable] = {}
        
        logging.info("Interruption Handler initialized")
    
    def register_undo_callback(self, action_type: str, callback: Callable):
        """Register undo callback for action type.
        
        Args:
            action_type: Type of action
            callback: Callback function to undo action
        """
        self.undo_callbacks[action_type] = callback
    
    def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with interruption support.
        
        Args:
            action: Action to execute
            
        Returns:
            Execution result
        """
        action_type = action.get("type")
        
        # Check if non-reversible and warn
        if self.warning_system.is_non_reversible(action_type):
            warning = self.warning_system.format_warning(
                action_type,
                action.get("details", {})
            )
            
            # In real implementation, would show confirmation dialog
            logging.warning(warning)
            
            # Mark as non-reversible
            action["reversible"] = False
        else:
            action["reversible"] = True
        
        # Add to history
        self.action_history.add_action(action)
        
        # Execute action (placeholder)
        result = {
            "status": "success",
            "action": action_type,
            "reversible": action["reversible"]
        }
        
        return result
    
    def undo_last_action(self) -> Dict[str, Any]:
        """Undo last action.
        
        Returns:
            Undo result
        """
        action = self.action_history.undo_last()
        
        if not action:
            return {
                "status": "error",
                "message": "No action to undo or action is not reversible"
            }
        
        # Execute undo callback if registered
        action_type = action.get("type")
        if action_type in self.undo_callbacks:
            try:
                self.undo_callbacks[action_type](action)
                logging.info(f"Undo callback executed for {action_type}")
            except Exception as e:
                logging.error(f"Undo callback failed: {e}")
        
        return {
            "status": "success",
            "message": f"Undid action: {action_type}",
            "action": action
        }
    
    def redo_last_action(self) -> Dict[str, Any]:
        """Redo last undone action.
        
        Returns:
            Redo result
        """
        action = self.action_history.redo_last()
        
        if not action:
            return {
                "status": "error",
                "message": "No action to redo"
            }
        
        return {
            "status": "success",
            "message": f"Redid action: {action.get('type')}",
            "action": action
        }
    
    def handle_interrupt(self, command: str, current_task: Dict[str, Any]) -> Dict[str, Any]:
        """Handle interruption command.
        
        Args:
            command: User command
            current_task: Currently executing task
            
        Returns:
            Interrupt handling result
        """
        if not self.interrupt_handler.is_interrupt_command(command):
            return {"status": "not_interrupt"}
        
        # Save task state
        task_id = current_task.get("task_id", f"task_{datetime.now().timestamp()}")
        self.task_state_manager.save_task_state(task_id, current_task)
        
        # Handle interrupt
        result = self.interrupt_handler.handle_interrupt(current_task)
        
        return result
    
    def check_continuation_prompt(self) -> Optional[str]:
        """Check if should prompt for continuation.
        
        Returns:
            Continuation prompt or None
        """
        if self.interrupt_handler.should_prompt_continuation():
            return self.interrupt_handler.get_continuation_prompt()
        
        return None
    
    def get_action_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent action history.
        
        Args:
            limit: Number of actions to return
            
        Returns:
            List of recent actions
        """
        return self.action_history.get_history(limit)
    
    def get_paused_tasks(self) -> List[Dict[str, Any]]:
        """Get list of paused tasks.
        
        Returns:
            List of paused tasks
        """
        return self.task_state_manager.list_paused_tasks()


# Global instance
_interruption_handler: Optional[InterruptionHandler] = None


def get_interruption_handler() -> InterruptionHandler:
    """Get global interruption handler instance."""
    global _interruption_handler
    
    if _interruption_handler is None:
        _interruption_handler = InterruptionHandler()
    
    return _interruption_handler
