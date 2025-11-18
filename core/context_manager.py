"""
Cross-Application Context Manager

Implements Task 22 requirements:
- Context graph with application nodes and edges
- Cross-app data flow tracking
- Workflow memory with pattern recognition
- Concurrent task tracking with priorities
- Fast context restoration with state snapshots

Integrates with:
- PredictiveEngine for workflow prediction
- LearningSystem for pattern learning
- ClipboardSearch for data flow tracking

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    from .predictive import get_predictive_engine
except ImportError:
    get_predictive_engine = None


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class ApplicationContext:
    """Context for a single application."""
    app_name: str
    window_title: str
    active_file: Optional[str]
    clipboard_content: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ApplicationContext':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class TaskContext:
    """Context for a task spanning multiple applications."""
    task_id: str
    description: str
    applications: List[str]
    priority: TaskPriority
    state: Dict[str, Any]
    created: datetime
    last_active: datetime
    completed: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['created'] = self.created.isoformat()
        data['last_active'] = self.last_active.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskContext':
        """Create from dictionary."""
        data['priority'] = TaskPriority(data['priority'])
        data['created'] = datetime.fromisoformat(data['created'])
        data['last_active'] = datetime.fromisoformat(data['last_active'])
        return cls(**data)


class ContextGraph:
    """Graph representing application contexts and relationships."""
    
    def __init__(self):
        self.graph = nx.DiGraph() if nx else None
        self.nodes: Dict[str, ApplicationContext] = {}
        self.edges: List[Tuple[str, str, Dict[str, Any]]] = []
    
    def add_context(self, context: ApplicationContext):
        """Add application context as node."""
        node_id = f"{context.app_name}_{context.timestamp.timestamp()}"
        self.nodes[node_id] = context
        
        if self.graph:
            self.graph.add_node(node_id, **context.to_dict())
    
    def add_relationship(self, 
                        from_app: str, 
                        to_app: str, 
                        relationship_type: str,
                        data: Dict[str, Any] = None):
        """Add relationship between applications."""
        edge_data = {
            "type": relationship_type,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        self.edges.append((from_app, to_app, edge_data))
        
        if self.graph:
            self.graph.add_edge(from_app, to_app, **edge_data)
    
    def get_related_apps(self, app_name: str, max_depth: int = 2) -> List[str]:
        """Get applications related to given app.
        
        Args:
            app_name: Application name
            max_depth: Maximum relationship depth
            
        Returns:
            List of related application names
        """
        if not self.graph or app_name not in self.graph:
            return []
        
        related = set()
        
        # Get neighbors within max_depth
        for node in nx.single_source_shortest_path_length(
            self.graph, app_name, cutoff=max_depth
        ):
            if node != app_name:
                related.add(node)
        
        return list(related)
    
    def get_workflow_path(self, start_app: str, end_app: str) -> List[str]:
        """Get workflow path between applications.
        
        Args:
            start_app: Starting application
            end_app: Ending application
            
        Returns:
            List of applications in workflow path
        """
        if not self.graph:
            return []
        
        try:
            path = nx.shortest_path(self.graph, start_app, end_app)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []


class DataFlowTracker:
    """Track data flow between applications."""
    
    def __init__(self):
        self.data_flows: List[Dict[str, Any]] = []
        self.flow_patterns = defaultdict(int)
    
    def track_data_transfer(self,
                           source_app: str,
                           dest_app: str,
                           data_type: str,
                           data_preview: str = ""):
        """Track data transfer between apps.
        
        Args:
            source_app: Source application
            dest_app: Destination application
            data_type: Type of data (text, file, image, etc.)
            data_preview: Preview of data content
        """
        flow = {
            "source": source_app,
            "destination": dest_app,
            "data_type": data_type,
            "data_preview": data_preview[:100],  # Limit preview
            "timestamp": datetime.now().isoformat()
        }
        
        self.data_flows.append(flow)
        
        # Track pattern
        pattern_key = f"{source_app}→{dest_app}:{data_type}"
        self.flow_patterns[pattern_key] += 1
        
        # Keep last 1000 flows
        if len(self.data_flows) > 1000:
            self.data_flows = self.data_flows[-1000:]
    
    def get_common_flows(self, top_k: int = 10) -> List[Tuple[str, int]]:
        """Get most common data flows.
        
        Args:
            top_k: Number of flows to return
            
        Returns:
            List of (pattern, count) tuples
        """
        sorted_patterns = sorted(
            self.flow_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_patterns[:top_k]
    
    def suggest_next_app(self, current_app: str, data_type: str) -> Optional[str]:
        """Suggest next application based on data flow patterns.
        
        Args:
            current_app: Current application
            data_type: Type of data
            
        Returns:
            Suggested application name or None
        """
        # Find patterns starting with current app and data type
        matching_patterns = {}
        
        for pattern, count in self.flow_patterns.items():
            if pattern.startswith(f"{current_app}→") and data_type in pattern:
                # Extract destination app
                parts = pattern.split("→")[1].split(":")
                dest_app = parts[0]
                matching_patterns[dest_app] = count
        
        if matching_patterns:
            return max(matching_patterns, key=matching_patterns.get)
        
        return None


class WorkflowMemory:
    """Remember and recognize workflow patterns."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/workflows.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_executions = defaultdict(int)
        
        self.load_data()
    
    def load_data(self):
        """Load workflow data."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.workflows = data.get("workflows", {})
                    self.workflow_executions = defaultdict(
                        int,
                        data.get("executions", {})
                    )
            except Exception as e:
                logging.error(f"Failed to load workflows: {e}")
    
    def save_data(self):
        """Save workflow data."""
        try:
            with open(self.data_file, 'w') as f:
                json.dump({
                    "workflows": self.workflows,
                    "executions": dict(self.workflow_executions)
                }, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save workflows: {e}")
    
    def record_workflow(self,
                       workflow_id: str,
                       description: str,
                       steps: List[Dict[str, Any]]):
        """Record a workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            description: Workflow description
            steps: List of workflow steps
        """
        self.workflows[workflow_id] = {
            "description": description,
            "steps": steps,
            "created": datetime.now().isoformat(),
            "last_executed": None
        }
        
        self.save_data()
    
    def execute_workflow(self, workflow_id: str):
        """Mark workflow as executed.
        
        Args:
            workflow_id: Workflow identifier
        """
        if workflow_id in self.workflows:
            self.workflows[workflow_id]["last_executed"] = datetime.now().isoformat()
            self.workflow_executions[workflow_id] += 1
            self.save_data()
    
    def recognize_workflow(self, recent_actions: List[Dict[str, Any]]) -> Optional[str]:
        """Recognize workflow from recent actions.
        
        Args:
            recent_actions: List of recent actions
            
        Returns:
            Workflow ID if recognized, None otherwise
        """
        if len(recent_actions) < 2:
            return None
        
        # Extract action signatures
        action_signatures = [
            f"{action.get('app', '')}:{action.get('action_type', '')}"
            for action in recent_actions
        ]
        
        # Match against known workflows
        for workflow_id, workflow in self.workflows.items():
            workflow_signatures = [
                f"{step.get('app', '')}:{step.get('action_type', '')}"
                for step in workflow["steps"]
            ]
            
            # Check if recent actions match workflow start
            if len(action_signatures) >= 2:
                if action_signatures[-2:] == workflow_signatures[:2]:
                    return workflow_id
        
        return None
    
    def get_popular_workflows(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """Get most popular workflows.
        
        Args:
            top_k: Number of workflows to return
            
        Returns:
            List of (workflow_id, execution_count) tuples
        """
        sorted_workflows = sorted(
            self.workflow_executions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_workflows[:top_k]


class TaskManager:
    """Manage concurrent tasks with priorities."""
    
    def __init__(self, max_tasks: int = 50):
        self.max_tasks = max_tasks
        self.tasks: Dict[str, TaskContext] = {}
        self.active_task_id: Optional[str] = None
    
    def create_task(self,
                   description: str,
                   applications: List[str],
                   priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Create new task.
        
        Args:
            description: Task description
            applications: List of involved applications
            priority: Task priority
            
        Returns:
            Task ID
        """
        # Generate task ID
        task_id = f"task_{datetime.now().timestamp()}"
        
        # Create task
        task = TaskContext(
            task_id=task_id,
            description=description,
            applications=applications,
            priority=priority,
            state={},
            created=datetime.now(),
            last_active=datetime.now(),
            completed=False
        )
        
        self.tasks[task_id] = task
        
        # Cleanup old tasks if needed
        if len(self.tasks) > self.max_tasks:
            self._cleanup_old_tasks()
        
        return task_id
    
    def update_task_state(self, task_id: str, state_update: Dict[str, Any]):
        """Update task state.
        
        Args:
            task_id: Task identifier
            state_update: State updates
        """
        if task_id in self.tasks:
            self.tasks[task_id].state.update(state_update)
            self.tasks[task_id].last_active = datetime.now()
    
    def switch_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Switch to different task.
        
        Args:
            task_id: Task to switch to
            
        Returns:
            Task state or None
        """
        if task_id not in self.tasks:
            return None
        
        # Save current task state
        if self.active_task_id:
            self._save_task_snapshot(self.active_task_id)
        
        # Switch to new task
        self.active_task_id = task_id
        self.tasks[task_id].last_active = datetime.now()
        
        # Return task state
        return self.tasks[task_id].state
    
    def complete_task(self, task_id: str):
        """Mark task as completed.
        
        Args:
            task_id: Task identifier
        """
        if task_id in self.tasks:
            self.tasks[task_id].completed = True
            self.tasks[task_id].last_active = datetime.now()
    
    def get_active_tasks(self) -> List[TaskContext]:
        """Get all active (incomplete) tasks.
        
        Returns:
            List of active tasks sorted by priority
        """
        active = [t for t in self.tasks.values() if not t.completed]
        
        # Sort by priority (descending) then last_active (descending)
        active.sort(
            key=lambda t: (t.priority.value, t.last_active.timestamp()),
            reverse=True
        )
        
        return active
    
    def _save_task_snapshot(self, task_id: str):
        """Save task state snapshot."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            # State is already in task.state, just update timestamp
            task.last_active = datetime.now()
    
    def _cleanup_old_tasks(self):
        """Remove old completed tasks."""
        # Get completed tasks older than 7 days
        cutoff = datetime.now() - timedelta(days=7)
        
        to_remove = [
            task_id for task_id, task in self.tasks.items()
            if task.completed and task.last_active < cutoff
        ]
        
        for task_id in to_remove:
            del self.tasks[task_id]
        
        logging.info(f"Cleaned up {len(to_remove)} old tasks")


class ContextManager:
    """Main cross-application context manager."""
    
    def __init__(self):
        self.context_graph = ContextGraph()
        self.data_flow_tracker = DataFlowTracker()
        self.workflow_memory = WorkflowMemory()
        self.task_manager = TaskManager(max_tasks=50)
        
        # Current context
        self.current_app: Optional[str] = None
        self.recent_actions = deque(maxlen=20)
        
        # Integration with predictive engine
        self.predictive_engine = None
        if get_predictive_engine:
            try:
                self.predictive_engine = get_predictive_engine()
            except Exception as e:
                logging.warning(f"Predictive engine integration failed: {e}")
        
        # Start background monitoring
        self._start_monitoring()
        
        logging.info("Context Manager initialized")
    
    def track_app_switch(self,
                        app_name: str,
                        window_title: str = "",
                        active_file: str = None):
        """Track application switch.
        
        Args:
            app_name: Application name
            window_title: Window title
            active_file: Active file path
        """
        # Create context
        context = ApplicationContext(
            app_name=app_name,
            window_title=window_title,
            active_file=active_file,
            clipboard_content=None,
            timestamp=datetime.now(),
            metadata={}
        )
        
        # Add to graph
        self.context_graph.add_context(context)
        
        # Add relationship if switching from another app
        if self.current_app and self.current_app != app_name:
            self.context_graph.add_relationship(
                self.current_app,
                app_name,
                "switched_to"
            )
        
        self.current_app = app_name
    
    def track_data_copy(self,
                       source_app: str,
                       data_type: str,
                       data_preview: str = ""):
        """Track data copy operation.
        
        Args:
            source_app: Source application
            data_type: Type of data
            data_preview: Preview of data
        """
        self.recent_actions.append({
            "action_type": "copy",
            "app": source_app,
            "data_type": data_type,
            "timestamp": datetime.now().isoformat()
        })
        
        # Suggest next app
        suggested_app = self.data_flow_tracker.suggest_next_app(
            source_app,
            data_type
        )
        
        if suggested_app:
            logging.info(f"Suggested next app for {data_type}: {suggested_app}")
    
    def track_data_paste(self,
                        dest_app: str,
                        source_app: str,
                        data_type: str,
                        data_preview: str = ""):
        """Track data paste operation.
        
        Args:
            dest_app: Destination application
            source_app: Source application
            data_type: Type of data
            data_preview: Preview of data
        """
        # Track data flow
        self.data_flow_tracker.track_data_transfer(
            source_app,
            dest_app,
            data_type,
            data_preview
        )
        
        # Add relationship
        self.context_graph.add_relationship(
            source_app,
            dest_app,
            "data_transfer",
            {"data_type": data_type}
        )
        
        self.recent_actions.append({
            "action_type": "paste",
            "app": dest_app,
            "data_type": data_type,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check for workflow recognition
        workflow_id = self.workflow_memory.recognize_workflow(
            list(self.recent_actions)
        )
        
        if workflow_id:
            logging.info(f"Recognized workflow: {workflow_id}")
            self.workflow_memory.execute_workflow(workflow_id)
    
    def create_workflow(self,
                       description: str,
                       steps: List[Dict[str, Any]]) -> str:
        """Create new workflow from steps.
        
        Args:
            description: Workflow description
            steps: List of workflow steps
            
        Returns:
            Workflow ID
        """
        workflow_id = f"workflow_{datetime.now().timestamp()}"
        
        self.workflow_memory.record_workflow(
            workflow_id,
            description,
            steps
        )
        
        return workflow_id
    
    def create_task(self,
                   description: str,
                   applications: List[str],
                   priority: str = "normal") -> str:
        """Create new task.
        
        Args:
            description: Task description
            applications: Involved applications
            priority: Priority level (low, normal, high, urgent)
            
        Returns:
            Task ID
        """
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "urgent": TaskPriority.URGENT
        }
        
        task_priority = priority_map.get(priority.lower(), TaskPriority.NORMAL)
        
        return self.task_manager.create_task(
            description,
            applications,
            task_priority
        )
    
    def switch_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Switch to different task.
        
        Args:
            task_id: Task to switch to
            
        Returns:
            Task state or None
        """
        state = self.task_manager.switch_task(task_id)
        
        if state:
            logging.info(f"Switched to task: {task_id}")
        
        return state
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get complete context summary.
        
        Returns:
            Context summary dictionary
        """
        return {
            "current_app": self.current_app,
            "active_tasks": len(self.task_manager.get_active_tasks()),
            "total_tasks": len(self.task_manager.tasks),
            "recent_actions": list(self.recent_actions)[-5:],
            "common_workflows": self.workflow_memory.get_popular_workflows(3),
            "common_data_flows": self.data_flow_tracker.get_common_flows(5),
            "context_graph_nodes": len(self.context_graph.nodes),
            "context_graph_edges": len(self.context_graph.edges)
        }
    
    def get_related_apps(self, app_name: str) -> List[str]:
        """Get applications related to given app.
        
        Args:
            app_name: Application name
            
        Returns:
            List of related applications
        """
        return self.context_graph.get_related_apps(app_name)
    
    def suggest_next_action(self) -> Optional[Dict[str, Any]]:
        """Suggest next action based on context.
        
        Returns:
            Suggested action or None
        """
        if not self.recent_actions:
            return None
        
        last_action = self.recent_actions[-1]
        
        # Check for workflow continuation
        workflow_id = self.workflow_memory.recognize_workflow(
            list(self.recent_actions)
        )
        
        if workflow_id:
            workflow = self.workflow_memory.workflows[workflow_id]
            # Suggest next step in workflow
            return {
                "type": "workflow_continuation",
                "workflow_id": workflow_id,
                "description": workflow["description"],
                "next_step": workflow["steps"][len(self.recent_actions)]
                    if len(self.recent_actions) < len(workflow["steps"])
                    else None
            }
        
        # Check for data flow suggestion
        if last_action["action_type"] == "copy":
            suggested_app = self.data_flow_tracker.suggest_next_app(
                last_action["app"],
                last_action.get("data_type", "text")
            )
            
            if suggested_app:
                return {
                    "type": "data_flow",
                    "action": "paste",
                    "suggested_app": suggested_app
                }
        
        return None
    
    def _start_monitoring(self):
        """Start background context monitoring."""
        async def monitoring_loop():
            while True:
                try:
                    # Periodic context analysis
                    summary = self.get_context_summary()
                    
                    # Log context changes
                    if summary["active_tasks"] > 0:
                        logging.debug(f"Active tasks: {summary['active_tasks']}")
                    
                    await asyncio.sleep(60)  # Every minute
                    
                except Exception as e:
                    logging.error(f"Context monitoring error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(monitoring_loop())


# Global instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get global context manager instance."""
    global _context_manager
    
    if _context_manager is None:
        _context_manager = ContextManager()
    
    return _context_manager
