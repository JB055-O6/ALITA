"""
User Control and Transparency Dashboard
Provides complete control and visibility over ALITA's actions
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from PyQt6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
        QLabel, QListWidget, QListWidgetItem, QTextEdit,
        QScrollArea, QFrame, QSplitter, QTabWidget,
        QCheckBox, QComboBox, QLineEdit, QTableWidget,
        QTableWidgetItem, QHeaderView, QMessageBox,
        QDialog, QDialogButtonBox
    )
    from PyQt6.QtCore import (
        Qt, QTimer, pyqtSignal, QSize, QDateTime,
        QPropertyAnimation, QEasingCurve
    )
    from PyQt6.QtGui import (
        QFont, QColor, QIcon, QPalette, QBrush, QPen, QPainter
    )
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


class ActionStatus(Enum):
    """Status of an action"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    UNDONE = "undone"


class ActionPriority(Enum):
    """Priority levels for actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Action:
    """Represents a single action"""
    id: str
    name: str
    description: str
    timestamp: float
    status: ActionStatus = ActionStatus.PENDING
    priority: ActionPriority = ActionPriority.MEDIUM
    requires_approval: bool = False
    reversible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    state_snapshot: Optional[Dict[str, Any]] = None


class ControlDashboard(QWidget):
    """
    User Control and Transparency Dashboard
    
    Features:
    - Action history viewer with timeline
    - Action preview system
    - Manual override controls
    - Action approval queue
    - Undo/redo functionality
    - Permission management
    - Real-time monitoring
    """
    
    # Signals
    action_approved = pyqtSignal(str)  # action_id
    action_rejected = pyqtSignal(str)  # action_id
    action_undone = pyqtSignal(str)  # action_id
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data storage
        self.actions: Dict[str, Action] = {}
        self.action_history: List[str] = []  # action IDs in order
        self.approval_queue: List[str] = []
        self.undo_stack: List[str] = []
        self.redo_stack: List[str] = []
        
        # Permissions
        self.permissions: Dict[str, bool] = {
            "file_operations": True,
            "network_access": True,
            "system_commands": False,
            "code_execution": False,
            "database_write": False,
        }
        
        # Setup UI
        self._init_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_displays)
        self.update_timer.start(1000)  # Update every second
    
    def _init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("Control Dashboard")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: rgba(0, 255, 255, 1.0);")
        layout.addWidget(title)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid rgba(0, 255, 255, 0.3);
                border-radius: 8px;
                background: rgba(10, 20, 40, 0.8);
            }
            QTabBar::tab {
                background: rgba(20, 30, 50, 0.8);
                color: white;
                padding: 8px 16px;
                margin: 2px;
                border-radius: 4px;
            }
            QTabBar::tab:selected {
                background: rgba(0, 255, 255, 0.3);
            }
        """)
        
        # Add tabs
        self.tabs.addTab(self._create_approval_tab(), "Approval Queue")
        self.tabs.addTab(self._create_history_tab(), "Action History")
        self.tabs.addTab(self._create_permissions_tab(), "Permissions")
        self.tabs.addTab(self._create_monitoring_tab(), "Monitoring")
        
        layout.addWidget(self.tabs)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.pause_btn = QPushButton("Pause All")
        self.pause_btn.clicked.connect(self._pause_all)
        control_layout.addWidget(self.pause_btn)
        
        self.resume_btn = QPushButton("Resume")
        self.resume_btn.clicked.connect(self._resume_all)
        control_layout.addWidget(self.resume_btn)
        
        self.emergency_stop_btn = QPushButton("Emergency Stop")
        self.emergency_stop_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 0, 0, 0.8);
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255, 50, 50, 1.0);
            }
        """)
        self.emergency_stop_btn.clicked.connect(self._emergency_stop)
        control_layout.addWidget(self.emergency_stop_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        self.setLayout(layout)
    
    def _create_approval_tab(self) -> QWidget:
        """Create approval queue tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Queue list
        self.approval_list = QListWidget()
        self.approval_list.setStyleSheet("""
            QListWidget {
                background: rgba(0, 20, 40, 0.8);
                border: 1px solid rgba(0, 255, 255, 0.3);
                border-radius: 4px;
                color: white;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid rgba(0, 255, 255, 0.1);
            }
            QListWidget::item:selected {
                background: rgba(0, 255, 255, 0.2);
            }
        """)
        self.approval_list.itemClicked.connect(self._show_action_preview)
        layout.addWidget(self.approval_list)
        
        # Action preview
        preview_label = QLabel("Action Preview:")
        preview_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        layout.addWidget(preview_label)
        
        self.action_preview = QTextEdit()
        self.action_preview.setReadOnly(True)
        self.action_preview.setMaximumHeight(150)
        self.action_preview.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 20, 40, 0.8);
                border: 1px solid rgba(0, 255, 255, 0.3);
                border-radius: 4px;
                color: white;
            }
        """)
        layout.addWidget(self.action_preview)
        
        # Approval buttons
        btn_layout = QHBoxLayout()
        
        self.approve_btn = QPushButton("✓ Approve")
        self.approve_btn.setStyleSheet("""
            QPushButton {
                background: rgba(0, 255, 0, 0.3);
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: rgba(0, 255, 0, 0.5);
            }
        """)
        self.approve_btn.clicked.connect(self._approve_selected)
        btn_layout.addWidget(self.approve_btn)
        
        self.reject_btn = QPushButton("✗ Reject")
        self.reject_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 0, 0, 0.3);
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: rgba(255, 0, 0, 0.5);
            }
        """)
        self.reject_btn.clicked.connect(self._reject_selected)
        btn_layout.addWidget(self.reject_btn)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        widget.setLayout(layout)
        return widget
    
    def _create_history_tab(self) -> QWidget:
        """Create action history tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Filter controls
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Filter:"))
        
        self.status_filter = QComboBox()
        self.status_filter.addItems(["All", "Completed", "Failed", "Undone"])
        self.status_filter.currentTextChanged.connect(self._filter_history)
        filter_layout.addWidget(self.status_filter)
        
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search actions...")
        self.search_box.textChanged.connect(self._filter_history)
        filter_layout.addWidget(self.search_box)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Time", "Action", "Status", "Priority", "Actions"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setStyleSheet("""
            QTableWidget {
                background: rgba(0, 20, 40, 0.8);
                border: 1px solid rgba(0, 255, 255, 0.3);
                border-radius: 4px;
                color: white;
            }
            QHeaderView::section {
                background: rgba(0, 255, 255, 0.2);
                color: white;
                padding: 4px;
                border: none;
            }
        """)
        layout.addWidget(self.history_table)
        
        # Undo/Redo buttons
        undo_layout = QHBoxLayout()
        
        self.undo_btn = QPushButton("⟲ Undo")
        self.undo_btn.clicked.connect(self._undo_action)
        undo_layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("⟳ Redo")
        self.redo_btn.clicked.connect(self._redo_action)
        undo_layout.addWidget(self.redo_btn)
        
        undo_layout.addStretch()
        layout.addLayout(undo_layout)
        
        widget.setLayout(layout)
        return widget
    
    def _create_permissions_tab(self) -> QWidget:
        """Create permissions management tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Permission Settings:"))
        
        # Permission checkboxes
        self.permission_checks: Dict[str, QCheckBox] = {}
        
        for perm_key, perm_value in self.permissions.items():
            check = QCheckBox(perm_key.replace("_", " ").title())
            check.setChecked(perm_value)
            check.stateChanged.connect(
                lambda state, key=perm_key: self._update_permission(key, state == Qt.CheckState.Checked.value)
            )
            self.permission_checks[perm_key] = check
            layout.addWidget(check)
        
        layout.addStretch()
        
        # Save button
        save_btn = QPushButton("Save Permissions")
        save_btn.clicked.connect(self._save_permissions)
        layout.addWidget(save_btn)
        
        widget.setLayout(layout)
        return widget
    
    def _create_monitoring_tab(self) -> QWidget:
        """Create real-time monitoring tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Statistics
        stats_layout = QHBoxLayout()
        
        self.total_actions_label = QLabel("Total Actions: 0")
        stats_layout.addWidget(self.total_actions_label)
        
        self.pending_actions_label = QLabel("Pending: 0")
        stats_layout.addWidget(self.pending_actions_label)
        
        self.completed_actions_label = QLabel("Completed: 0")
        stats_layout.addWidget(self.completed_actions_label)
        
        stats_layout.addStretch()
        layout.addLayout(stats_layout)
        
        # Activity log
        layout.addWidget(QLabel("Activity Log:"))
        
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setStyleSheet("""
            QTextEdit {
                background: rgba(0, 20, 40, 0.8);
                border: 1px solid rgba(0, 255, 255, 0.3);
                border-radius: 4px;
                color: white;
                font-family: 'Consolas', monospace;
            }
        """)
        layout.addWidget(self.activity_log)
        
        widget.setLayout(layout)
        return widget
    
    def add_action(
        self,
        action_id: str,
        name: str,
        description: str,
        priority: ActionPriority = ActionPriority.MEDIUM,
        requires_approval: bool = False,
        reversible: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Action:
        """Add a new action to the dashboard"""
        action = Action(
            id=action_id,
            name=name,
            description=description,
            timestamp=time.time(),
            priority=priority,
            requires_approval=requires_approval,
            reversible=reversible,
            metadata=metadata or {}
        )
        
        self.actions[action_id] = action
        self.action_history.append(action_id)
        
        if requires_approval:
            self.approval_queue.append(action_id)
            self._update_approval_list()
        
        self._log_activity(f"Action added: {name}")
        self._update_displays()
        
        return action
    
    def _show_action_preview(self, item: QListWidgetItem):
        """Show preview of selected action"""
        action_id = item.data(Qt.ItemDataRole.UserRole)
        if action_id in self.actions:
            action = self.actions[action_id]
            preview_text = f"""
Action: {action.name}
Description: {action.description}
Priority: {action.priority.value}
Time: {datetime.fromtimestamp(action.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
Reversible: {'Yes' if action.reversible else 'No'}

Metadata:
{json.dumps(action.metadata, indent=2)}
            """
            self.action_preview.setText(preview_text.strip())
    
    def _approve_selected(self):
        """Approve selected action"""
        current_item = self.approval_list.currentItem()
        if current_item:
            action_id = current_item.data(Qt.ItemDataRole.UserRole)
            self._approve_action(action_id)
    
    def _reject_selected(self):
        """Reject selected action"""
        current_item = self.approval_list.currentItem()
        if current_item:
            action_id = current_item.data(Qt.ItemDataRole.UserRole)
            self._reject_action(action_id)
    
    def _approve_action(self, action_id: str):
        """Approve an action"""
        if action_id in self.actions:
            action = self.actions[action_id]
            action.status = ActionStatus.APPROVED
            
            if action_id in self.approval_queue:
                self.approval_queue.remove(action_id)
            
            self._log_activity(f"Action approved: {action.name}")
            self._update_approval_list()
            self._update_displays()
            
            self.action_approved.emit(action_id)
    
    def _reject_action(self, action_id: str):
        """Reject an action"""
        if action_id in self.actions:
            action = self.actions[action_id]
            action.status = ActionStatus.REJECTED
            
            if action_id in self.approval_queue:
                self.approval_queue.remove(action_id)
            
            self._log_activity(f"Action rejected: {action.name}")
            self._update_approval_list()
            self._update_displays()
            
            self.action_rejected.emit(action_id)
    
    def _undo_action(self):
        """Undo last action"""
        if self.undo_stack:
            action_id = self.undo_stack.pop()
            if action_id in self.actions:
                action = self.actions[action_id]
                
                if action.reversible:
                    action.status = ActionStatus.UNDONE
                    self.redo_stack.append(action_id)
                    
                    self._log_activity(f"Action undone: {action.name}")
                    self._update_displays()
                    
                    self.action_undone.emit(action_id)
                else:
                    QMessageBox.warning(
                        self,
                        "Cannot Undo",
                        f"Action '{action.name}' is not reversible."
                    )
    
    def _redo_action(self):
        """Redo last undone action"""
        if self.redo_stack:
            action_id = self.redo_stack.pop()
            if action_id in self.actions:
                action = self.actions[action_id]
                action.status = ActionStatus.APPROVED
                self.undo_stack.append(action_id)
                
                self._log_activity(f"Action redone: {action.name}")
                self._update_displays()
    
    def _update_approval_list(self):
        """Update approval queue list"""
        self.approval_list.clear()
        
        for action_id in self.approval_queue:
            if action_id in self.actions:
                action = self.actions[action_id]
                item = QListWidgetItem(
                    f"[{action.priority.value.upper()}] {action.name}"
                )
                item.setData(Qt.ItemDataRole.UserRole, action_id)
                
                # Color code by priority
                if action.priority == ActionPriority.CRITICAL:
                    item.setForeground(QBrush(QColor(255, 0, 0)))
                elif action.priority == ActionPriority.HIGH:
                    item.setForeground(QBrush(QColor(255, 165, 0)))
                
                self.approval_list.addItem(item)
    
    def _filter_history(self):
        """Filter action history"""
        self._update_history_table()
    
    def _update_history_table(self):
        """Update history table"""
        self.history_table.setRowCount(0)
        
        status_filter = self.status_filter.currentText()
        search_text = self.search_box.text().lower()
        
        for action_id in reversed(self.action_history):
            if action_id in self.actions:
                action = self.actions[action_id]
                
                # Apply filters
                if status_filter != "All":
                    if status_filter.lower() != action.status.value:
                        continue
                
                if search_text and search_text not in action.name.lower():
                    continue
                
                # Add row
                row = self.history_table.rowCount()
                self.history_table.insertRow(row)
                
                # Time
                time_str = datetime.fromtimestamp(action.timestamp).strftime('%H:%M:%S')
                self.history_table.setItem(row, 0, QTableWidgetItem(time_str))
                
                # Action name
                self.history_table.setItem(row, 1, QTableWidgetItem(action.name))
                
                # Status
                status_item = QTableWidgetItem(action.status.value)
                if action.status == ActionStatus.COMPLETED:
                    status_item.setForeground(QBrush(QColor(0, 255, 0)))
                elif action.status == ActionStatus.FAILED:
                    status_item.setForeground(QBrush(QColor(255, 0, 0)))
                self.history_table.setItem(row, 2, status_item)
                
                # Priority
                self.history_table.setItem(row, 3, QTableWidgetItem(action.priority.value))
                
                # Actions column (buttons)
                self.history_table.setItem(row, 4, QTableWidgetItem(""))
    
    def _update_permission(self, key: str, value: bool):
        """Update permission setting"""
        self.permissions[key] = value
        self._log_activity(f"Permission '{key}' set to {value}")
    
    def _save_permissions(self):
        """Save permissions to file"""
        try:
            perm_file = Path("config/permissions.json")
            perm_file.parent.mkdir(exist_ok=True)
            
            with open(perm_file, 'w') as f:
                json.dump(self.permissions, f, indent=2)
            
            QMessageBox.information(
                self,
                "Permissions Saved",
                "Permissions have been saved successfully."
            )
            self._log_activity("Permissions saved to file")
        except Exception as e:
            QMessageBox.critical(
                self,
                "Save Failed",
                f"Failed to save permissions: {e}"
            )
    
    def _update_displays(self):
        """Update all displays"""
        # Update statistics
        total = len(self.actions)
        pending = sum(1 for a in self.actions.values() if a.status == ActionStatus.PENDING)
        completed = sum(1 for a in self.actions.values() if a.status == ActionStatus.COMPLETED)
        
        self.total_actions_label.setText(f"Total Actions: {total}")
        self.pending_actions_label.setText(f"Pending: {pending}")
        self.completed_actions_label.setText(f"Completed: {completed}")
        
        # Update history table
        self._update_history_table()
    
    def _log_activity(self, message: str):
        """Log activity to activity log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.activity_log.append(f"[{timestamp}] {message}")
        
        # Auto-scroll to bottom
        self.activity_log.verticalScrollBar().setValue(
            self.activity_log.verticalScrollBar().maximum()
        )
    
    def _pause_all(self):
        """Pause all actions"""
        self._log_activity("All actions paused")
        QMessageBox.information(self, "Paused", "All actions have been paused.")
    
    def _resume_all(self):
        """Resume all actions"""
        self._log_activity("All actions resumed")
        QMessageBox.information(self, "Resumed", "All actions have been resumed.")
    
    def _emergency_stop(self):
        """Emergency stop all actions"""
        reply = QMessageBox.question(
            self,
            "Emergency Stop",
            "Are you sure you want to stop all actions immediately?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._log_activity("EMERGENCY STOP activated")
            QMessageBox.warning(self, "Stopped", "All actions have been stopped.")
    
    def mark_action_completed(self, action_id: str, result: Any = None):
        """Mark action as completed"""
        if action_id in self.actions:
            action = self.actions[action_id]
            action.status = ActionStatus.COMPLETED
            action.result = result
            
            if action.reversible:
                self.undo_stack.append(action_id)
            
            self._log_activity(f"Action completed: {action.name}")
            self._update_displays()
    
    def mark_action_failed(self, action_id: str, error: str):
        """Mark action as failed"""
        if action_id in self.actions:
            action = self.actions[action_id]
            action.status = ActionStatus.FAILED
            action.error = error
            
            self._log_activity(f"Action failed: {action.name} - {error}")
            self._update_displays()
    
    def get_permission(self, permission_key: str) -> bool:
        """Check if permission is granted"""
        return self.permissions.get(permission_key, False)
    
    def clear_history(self):
        """Clear action history"""
        reply = QMessageBox.question(
            self,
            "Clear History",
            "Are you sure you want to clear all action history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.actions.clear()
            self.action_history.clear()
            self.undo_stack.clear()
            self.redo_stack.clear()
            
            self._log_activity("Action history cleared")
            self._update_displays()
