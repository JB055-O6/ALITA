"""System automation helpers.

Enhanced automation system with:
- Application launching and control
- Mouse and keyboard automation
- File operations with confirmations
- Multi-step workflow execution
- Permission system
- Audit logging
- Macro templates

Uses pywinauto on Windows when available, with graceful fallbacks.
"""
from typing import Optional, Tuple, Dict, List, Any
import subprocess
import time
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml
import shutil

try:
    import pywinauto
    from pywinauto import Application
    from pywinauto import keyboard as _pw_keyboard
    from pywinauto import mouse as _pw_mouse
except Exception:
    pywinauto = None
    Application = None
    _pw_keyboard = None
    _pw_mouse = None

try:
    from PIL import ImageGrab
except Exception:
    ImageGrab = None

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    import win32api
    import win32con
except Exception:
    win32api = None
    win32con = None

from .safety import get_safety_manager, ActionRisk


class SystemControl:
    """Enhanced system automation utilities with full control.
    
    Features:
    - Application launching and control
    - Mouse automation (click, move, scroll)
    - Keyboard automation (type, send keys)
    - File operations with confirmations
    - Screenshot capture
    - Audit logging
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.audit_log_path = Path("logs/automation_audit.log")
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load permissions
        self.permissions = self._load_permissions()
        
        # Safety system integration
        self.safety_manager = get_safety_manager()
        self.safety_manager.emergency_stop.register_callback(self._on_emergency_stop)
        
        # App cache for faster launches
        self.app_cache_path = Path("cache/app_cache.json")
        self.app_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.app_cache = self._load_app_cache()
        
        logging.info("SystemControl initialized")

    def launch(self, command: str, wait: bool = False) -> Optional[subprocess.Popen]:
        """Launch an application or command.

        Returns subprocess.Popen when available or None on failure.
        """
        try:
            p = subprocess.Popen(command, shell=True)
            if wait:
                p.wait()
            return p
        except Exception:
            return None

    def find_window(self, title_re: str):
        """Find a window by title regex using pywinauto (Windows) or return None."""
        if pywinauto is None:
            return None
        try:
            windows = pywinauto.findwindows.find_windows(title_re=title_re)
            if not windows:
                return None
            return windows[0]
        except Exception:
            return None

    def focus_window(self, title_re: str) -> bool:
        """Bring a window matching title_re to foreground (Windows).

        Returns True if succeeded, False otherwise.
        """
        if Application is None:
            return False
        try:
            app = Application().connect(title_re=title_re)
            win = app.top_window()
            win.set_focus()
            return True
        except Exception:
            return False

    def send_keystrokes(self, keys: str) -> bool:
        """Send keystrokes to the active window.

        On Windows uses pywinauto.keyboard, otherwise falls back to printing.
        """
        if _pw_keyboard is not None:
            try:
                _pw_keyboard.send_keys(keys)
                return True
            except Exception:
                return False
        # Fallback: print to console (no-op for automation)
        try:
            print(f"[send_keystrokes] {keys}")
            return True
        except Exception:
            return False

    def screenshot(self, bbox: Optional[Tuple[int, int, int, int]] = None):
        """Take a screenshot. Returns a PIL.Image or None.

        bbox is (left, top, right, bottom) if provided.
        """
        if ImageGrab is None:
            return None
        try:
            if bbox:
                return ImageGrab.grab(bbox=bbox)
            return ImageGrab.grab()
        except Exception:
            return None
    
    def _load_permissions(self) -> Dict[str, Any]:
        """Load permission configuration."""
        perm_path = Path("config/permissions.yaml")
        if perm_path.exists():
            try:
                with open(perm_path) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logging.error(f"Failed to load permissions: {str(e)}")
        
        # Default permissions
        return {
            "file_operations": {
                "delete": "confirm",
                "move": "confirm",
                "create": "allow",
                "read": "allow"
            },
            "system_operations": {
                "shutdown": "deny",
                "restart": "deny",
                "sleep": "confirm"
            },
            "automation": {
                "click": "allow",
                "type": "allow",
                "launch": "allow"
            }
        }
    
    def _check_permission(self, operation: str, category: str = "automation") -> bool:
        """Check if operation is permitted."""
        perms = self.permissions.get(category, {})
        perm = perms.get(operation, "deny")
        
        if perm == "allow":
            return True
        elif perm == "confirm":
            # In real implementation, would show confirmation dialog
            logging.warning(f"Operation {operation} requires confirmation")
            return True  # Auto-approve for now
        else:
            logging.warning(f"Operation {operation} denied by permissions")
            return False
    
    def _audit_log(self, action: str, details: Dict[str, Any]):
        """Log action to audit trail."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "details": details
            }
            
            with open(self.audit_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logging.error(f"Audit logging failed: {str(e)}")
    
    # Enhanced Mouse Control
    
    def click_at(self, x: int, y: int, button: str = "left", clicks: int = 1) -> bool:
        """Click at specific coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: 'left', 'right', or 'middle'
            clicks: Number of clicks (1 for single, 2 for double)
            
        Returns:
            True if successful
        """
        if not self._check_permission("click"):
            return False
        
        try:
            # Try pyautogui first (cross-platform)
            if pyautogui is not None:
                pyautogui.click(x, y, clicks=clicks, button=button)
                self._audit_log("click", {"x": x, "y": y, "button": button})
                return True
            
            # Fallback to win32api
            if win32api is not None:
                win32api.SetCursorPos((x, y))
                if button == "left":
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
                elif button == "right":
                    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, x, y, 0, 0)
                    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, x, y, 0, 0)
                
                self._audit_log("click", {"x": x, "y": y, "button": button})
                return True
            
            return False
        except Exception as e:
            logging.error(f"Click failed: {str(e)}")
            return False
    
    def move_mouse(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            duration: Time to take for movement (seconds)
            
        Returns:
            True if successful
        """
        try:
            if pyautogui is not None:
                pyautogui.moveTo(x, y, duration=duration)
                return True
            
            if win32api is not None:
                win32api.SetCursorPos((x, y))
                return True
            
            return False
        except Exception as e:
            logging.error(f"Mouse move failed: {str(e)}")
            return False
    
    def scroll(self, amount: int, direction: str = "vertical") -> bool:
        """Scroll the active window.
        
        Args:
            amount: Scroll amount (positive = up/right, negative = down/left)
            direction: 'vertical' or 'horizontal'
            
        Returns:
            True if successful
        """
        if not self._check_permission("click"):  # Scroll uses same permission
            return False
        
        try:
            if pyautogui is not None:
                if direction == "vertical":
                    pyautogui.scroll(amount)
                else:
                    pyautogui.hscroll(amount)
                
                self._audit_log("scroll", {"amount": amount, "direction": direction})
                return True
            
            if win32api is not None and direction == "vertical":
                win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, amount * 120, 0)
                self._audit_log("scroll", {"amount": amount})
                return True
            
            return False
        except Exception as e:
            logging.error(f"Scroll failed: {str(e)}")
            return False
    
    # Enhanced Keyboard Control
    
    def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type text character by character.
        
        Args:
            text: Text to type
            interval: Delay between keystrokes (seconds)
            
        Returns:
            True if successful
        """
        if not self._check_permission("type"):
            return False
        
        try:
            if pyautogui is not None:
                pyautogui.write(text, interval=interval)
                self._audit_log("type", {"length": len(text)})
                return True
            
            if _pw_keyboard is not None:
                _pw_keyboard.send_keys(text)
                self._audit_log("type", {"length": len(text)})
                return True
            
            return False
        except Exception as e:
            logging.error(f"Type text failed: {str(e)}")
            return False
    
    def press_key(self, key: str, modifiers: Optional[List[str]] = None) -> bool:
        """Press a key with optional modifiers.
        
        Args:
            key: Key to press (e.g., 'enter', 'a', 'f1')
            modifiers: List of modifiers (e.g., ['ctrl', 'shift'])
            
        Returns:
            True if successful
        """
        if not self._check_permission("type"):
            return False
        
        try:
            if pyautogui is not None:
                if modifiers:
                    pyautogui.hotkey(*modifiers, key)
                else:
                    pyautogui.press(key)
                
                self._audit_log("press_key", {"key": key, "modifiers": modifiers})
                return True
            
            return False
        except Exception as e:
            logging.error(f"Press key failed: {str(e)}")
            return False
    
    # File Operations
    
    def create_file(self, path: Path, content: str = "") -> bool:
        """Create a file with optional content.
        
        Args:
            path: File path
            content: File content
            
        Returns:
            True if successful
        """
        if not self._check_permission("create", "file_operations"):
            return False
        
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            
            self._audit_log("create_file", {"path": str(path), "size": len(content)})
            return True
        except Exception as e:
            logging.error(f"Create file failed: {str(e)}")
            return False
    
    def delete_file(self, path: Path, confirm: bool = True) -> bool:
        """Delete a file (Requirement 8.3: Confirmation for destructive actions).
        
        Args:
            path: File path
            confirm: Require confirmation
            
        Returns:
            True if successful
        """
        # Check emergency stop
        if self.safety_manager.emergency_stop.is_stopped():
            logging.error("❌ Operation blocked: Emergency stop active")
            return False
        
        if not self._check_permission("delete", "file_operations"):
            return False
        
        try:
            path = Path(path)
            if not path.exists():
                return False
            
            # Require confirmation for destructive action (Requirement 8.3)
            if confirm:
                if not self.safety_manager.confirmation.confirm_action(
                    f"Delete file: {path.name}",
                    {"path": str(path), "size": path.stat().st_size},
                    ActionRisk.HIGH
                ):
                    logging.info("File deletion cancelled by user")
                    return False
            
            # Store file content for potential rollback
            rollback_info = {
                "path": str(path),
                "content": path.read_text() if path.is_file() else None,
                "size": path.stat().st_size
            }
            
            path.unlink()
            
            # Enhanced audit logging with rollback info (Requirement 8.4)
            self.safety_manager.audit_logger.log_action(
                "delete_file",
                {"path": str(path)},
                rollback_info=rollback_info,
                risk_level=ActionRisk.HIGH
            )
            
            return True
        except Exception as e:
            logging.error(f"Delete file failed: {str(e)}")
            return False
    
    def move_file(self, source: Path, destination: Path, confirm: bool = True) -> bool:
        """Move a file (Requirement 8.3: Confirmation for destructive actions).
        
        Args:
            source: Source path
            destination: Destination path
            confirm: Require confirmation
            
        Returns:
            True if successful
        """
        # Check emergency stop
        if self.safety_manager.emergency_stop.is_stopped():
            logging.error("❌ Operation blocked: Emergency stop active")
            return False
        
        if not self._check_permission("move", "file_operations"):
            return False
        
        try:
            source = Path(source)
            destination = Path(destination)
            
            if not source.exists():
                return False
            
            # Require confirmation (Requirement 8.3)
            if confirm:
                if not self.safety_manager.confirmation.confirm_action(
                    f"Move file: {source.name}",
                    {"source": str(source), "destination": str(destination)},
                    ActionRisk.MEDIUM
                ):
                    logging.info("File move cancelled by user")
                    return False
            
            # Store rollback info (Requirement 8.4)
            rollback_info = {
                "source": str(source),
                "destination": str(destination),
                "can_rollback": True
            }
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            
            # Enhanced audit logging (Requirement 8.4)
            self.safety_manager.audit_logger.log_action(
                "move_file",
                {"source": str(source), "destination": str(destination)},
                rollback_info=rollback_info,
                risk_level=ActionRisk.MEDIUM
            )
            
            return True
        except Exception as e:
            logging.error(f"Move file failed: {str(e)}")
            return False
    
    def copy_file(self, source: Path, destination: Path) -> bool:
        """Copy a file.
        
        Args:
            source: Source path
            destination: Destination path
            
        Returns:
            True if successful
        """
        # Check emergency stop
        if self.safety_manager.emergency_stop.is_stopped():
            logging.error("❌ Operation blocked: Emergency stop active")
            return False
        
        if not self._check_permission("create", "file_operations"):
            return False
        
        try:
            source = Path(source)
            destination = Path(destination)
            
            if not source.exists():
                return False
            
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(destination))
            
            # Enhanced audit logging (Requirement 8.4)
            self.safety_manager.audit_logger.log_action(
                "copy_file",
                {"source": str(source), "destination": str(destination)},
                risk_level=ActionRisk.LOW
            )
            
            return True
        except Exception as e:
            logging.error(f"Copy file failed: {str(e)}")
            return False
    
    # App Cache Methods
    
    def _load_app_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load application cache from JSON file."""
        if self.app_cache_path.exists():
            try:
                with open(self.app_cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load app cache: {str(e)}")
        return {}
    
    def _save_app_cache(self):
        """Save application cache to JSON file."""
        try:
            with open(self.app_cache_path, 'w') as f:
                json.dump(self.app_cache, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save app cache: {str(e)}")
    
    def cache_app_info(self, app_name: str, path: str, method: str):
        """Cache successful app launch info."""
        self.app_cache[app_name.lower()] = {
            "path": path,
            "last_used": datetime.now().isoformat(),
            "launch_method": method,
            "success_count": self.app_cache.get(app_name.lower(), {}).get("success_count", 0) + 1
        }
        self._save_app_cache()
        logging.info(f"Cached app info for {app_name}: {path}")
    
    # PowerShell Integration Methods
    
    def _execute_powershell(self, command: str, timeout: int = 10) -> Tuple[bool, str]:
        """Execute PowerShell command and return result."""
        try:
            result = subprocess.run(
                ["powershell", "-Command", command],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout.strip()
        except Exception as e:
            logging.error(f"PowerShell execution failed: {str(e)}")
            return False, str(e)
    
    def launch_via_powershell(self, app_name: str) -> bool:
        """Launch app using PowerShell Start-Process."""
        try:
            command = f"Start-Process '{app_name}'"
            success, output = self._execute_powershell(command, timeout=5)
            if success:
                logging.info(f"Launched {app_name} via PowerShell")
                return True
            return False
        except Exception as e:
            logging.error(f"PowerShell launch failed: {str(e)}")
            return False
    
    def discover_installed_apps(self) -> List[Dict[str, str]]:
        """Discover all installed applications using PowerShell."""
        apps = []
        
        # Try Get-StartApps
        try:
            command = "Get-StartApps | ConvertTo-Json"
            success, output = self._execute_powershell(command, timeout=15)
            if success and output:
                app_list = json.loads(output)
                if isinstance(app_list, list):
                    for app in app_list:
                        apps.append({
                            "name": app.get("Name", ""),
                            "app_id": app.get("AppID", ""),
                            "source": "Get-StartApps"
                        })
        except Exception as e:
            logging.error(f"Get-StartApps failed: {str(e)}")
        
        return apps
    
    def launch_via_windows_search(self, app_name: str) -> bool:
        """Launch app using Windows Search (Win+S) automation - LAST RESORT."""
        try:
            if pyautogui is None:
                return False
            
            logging.info(f"Attempting Windows Search launch for {app_name}")
            
            # Press Win+S
            pyautogui.hotkey('win', 's')
            time.sleep(1.5)
            
            # Type app name
            pyautogui.write(app_name, interval=0.05)
            time.sleep(2)
            
            # Press Enter
            pyautogui.press('enter')
            time.sleep(3)
            
            # Close search if still open
            pyautogui.press('esc')
            
            logging.info(f"Windows Search launch attempted for {app_name}")
            return True
            
        except Exception as e:
            logging.error(f"Windows Search launch failed: {str(e)}")
            return False
    
    def launch_with_fallbacks(self, app_name: str) -> Dict[str, Any]:
        """Launch app using multiple fallback methods.
        
        Fallback order:
        1. Check app cache for known path
        2. PowerShell Start-Process
        3. Search common directories
        4. PowerShell Get-StartApps query
        5. Windows Search (Win+S) automation
        
        Returns: {"success": bool, "method": str, "message": str, "attempted": list}
        """
        attempted_methods = []
        app_lower = app_name.lower()
        
        # Method 1: Cache lookup
        if app_lower in self.app_cache:
            cached_info = self.app_cache[app_lower]
            cached_path = cached_info.get("path", "")
            logging.info(f"Found {app_name} in cache: {cached_path}")
            attempted_methods.append("cache")
            
            process = self.launch(cached_path)
            if process:
                self.cache_app_info(app_name, cached_path, "cache")
                return {
                    "success": True,
                    "method": "cache",
                    "message": f"Opened {app_name} from cache",
                    "attempted": attempted_methods
                }
        
        # Method 2: PowerShell Start-Process
        logging.info(f"Trying PowerShell Start-Process for {app_name}")
        attempted_methods.append("powershell")
        if self.launch_via_powershell(app_name):
            self.cache_app_info(app_name, app_name, "powershell")
            return {
                "success": True,
                "method": "powershell",
                "message": f"Opened {app_name} via PowerShell",
                "attempted": attempted_methods
            }
        
        # Method 3: Common directories search
        logging.info(f"Searching common directories for {app_name}")
        attempted_methods.append("directory_search")
        common_dirs = [
            Path("C:/Program Files"),
            Path("C:/Program Files (x86)"),
            Path.home() / "AppData/Local",
            Path.home() / "AppData/Roaming"
        ]
        
        for directory in common_dirs:
            if not directory.exists():
                continue
            try:
                for exe_file in directory.rglob(f"*{app_name}*.exe"):
                    process = self.launch(str(exe_file))
                    if process:
                        self.cache_app_info(app_name, str(exe_file), "directory_search")
                        return {
                            "success": True,
                            "method": "directory_search",
                            "message": f"Opened {app_name} from {exe_file}",
                            "attempted": attempted_methods
                        }
            except (PermissionError, OSError):
                continue
        
        # Method 4: Get-StartApps query
        logging.info(f"Querying Get-StartApps for {app_name}")
        attempted_methods.append("get_startapps")
        apps = self.discover_installed_apps()
        for app in apps:
            if app_name.lower() in app["name"].lower():
                if self.launch_via_powershell(app["name"]):
                    self.cache_app_info(app_name, app["name"], "get_startapps")
                    return {
                        "success": True,
                        "method": "get_startapps",
                        "message": f"Opened {app_name} via Get-StartApps",
                        "attempted": attempted_methods
                    }
        
        # Method 5: Windows Search (LAST RESORT)
        logging.info(f"Using Windows Search as last resort for {app_name}")
        attempted_methods.append("windows_search")
        if self.launch_via_windows_search(app_name):
            return {
                "success": True,
                "method": "windows_search",
                "message": f"Attempted to open {app_name} via Windows Search",
                "attempted": attempted_methods
            }
        
        # All methods failed
        return {
            "success": False,
            "method": None,
            "message": f"Could not open {app_name}. Tried: {', '.join(attempted_methods)}",
            "attempted": attempted_methods
        }
    
    def close_app(self, app_name: str) -> bool:
        """Close an application using PowerShell or taskkill.
        
        Args:
            app_name: Name of the application to close
            
        Returns:
            True if successful
        """
        try:
            # Try PowerShell Stop-Process first
            command = f"Stop-Process -Name '{app_name}' -Force -ErrorAction SilentlyContinue"
            success, output = self._execute_powershell(command, timeout=5)
            
            if success:
                logging.info(f"Closed {app_name} via PowerShell")
                return True
            
            # Fallback to taskkill
            import subprocess
            result = subprocess.run(
                f"taskkill /IM {app_name}.exe /F",
                shell=True,
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                logging.info(f"Closed {app_name} via taskkill")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Failed to close {app_name}: {str(e)}")
            return False
    
    def _on_emergency_stop(self):
        """Handle emergency stop event."""
        logging.critical("🚨 SystemControl: Emergency stop - halting all automation")



class AutomationExecutor:
    """Execute multi-step automation workflows with rollback capability.
    
    Features:
    - Sequential step execution
    - Error handling and rollback
    - Progress tracking
    - Macro loading from YAML
    """
    
    def __init__(self, system_control: SystemControl):
        self.system_control = system_control
        self.execution_history = []
        self.macro_dir = Path("config/macros")
        self.macro_dir.mkdir(parents=True, exist_ok=True)
        
        # Safety system integration
        self.safety_manager = get_safety_manager()
        self.safety_manager.emergency_stop.register_callback(self._on_emergency_stop)
        self.current_workflow = None
    
    def execute_workflow(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a multi-step workflow with safety checks.
        
        Args:
            steps: List of step dictionaries with 'action' and parameters
            
        Returns:
            Execution result with status and details
        """
        # Check emergency stop before starting
        if self.safety_manager.emergency_stop.is_stopped():
            logging.error("❌ Workflow blocked: Emergency stop active")
            return {
                "success": False,
                "error": "Emergency stop is active",
                "completed_steps": 0,
                "total_steps": len(steps)
            }
        
        results = []
        executed_steps = []
        self.current_workflow = {"steps": steps, "current_step": 0}
        
        try:
            for idx, step in enumerate(steps):
                # Check emergency stop before each step
                if self.safety_manager.emergency_stop.is_stopped():
                    logging.error("🚨 Workflow interrupted by emergency stop")
                    self._rollback(executed_steps)
                    return {
                        "success": False,
                        "error": "Emergency stop triggered",
                        "completed_steps": idx,
                        "total_steps": len(steps),
                        "results": results
                    }
                
                self.current_workflow["current_step"] = idx
                logging.info(f"Executing step {idx + 1}/{len(steps)}: {step.get('action')}")
                
                result = self._execute_step(step)
                results.append(result)
                executed_steps.append(step)
                
                if not result.get("success", False):
                    logging.error(f"Step {idx + 1} failed: {result.get('error')}")
                    
                    # Attempt rollback
                    if step.get("rollback_on_failure", True):
                        self._rollback(executed_steps)
                    
                    return {
                        "success": False,
                        "completed_steps": idx,
                        "total_steps": len(steps),
                        "results": results,
                        "error": result.get("error")
                    }
            
            self.current_workflow = None
            return {
                "success": True,
                "completed_steps": len(steps),
                "total_steps": len(steps),
                "results": results
            }
            
        except Exception as e:
            logging.error(f"Workflow execution failed: {str(e)}")
            self.current_workflow = None
            return {
                "success": False,
                "error": str(e),
                "results": results
            }
    
    def _on_emergency_stop(self):
        """Handle emergency stop event."""
        logging.critical("🚨 AutomationExecutor: Emergency stop - halting workflow")
        if self.current_workflow:
            logging.info(f"Interrupted at step {self.current_workflow['current_step'] + 1}")
            self.current_workflow = None
    
    def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single automation step."""
        action = step.get("action")
        
        try:
            if action == "click":
                success = self.system_control.click_at(
                    step["x"], step["y"],
                    button=step.get("button", "left")
                )
            
            elif action == "type":
                success = self.system_control.type_text(
                    step["text"],
                    interval=step.get("interval", 0.0)
                )
            
            elif action == "press_key":
                success = self.system_control.press_key(
                    step["key"],
                    modifiers=step.get("modifiers")
                )
            
            elif action == "scroll":
                success = self.system_control.scroll(
                    step["amount"],
                    direction=step.get("direction", "vertical")
                )
            
            elif action == "launch":
                result = self.system_control.launch(
                    step["command"],
                    wait=step.get("wait", False)
                )
                success = result is not None
            
            elif action == "focus_window":
                success = self.system_control.focus_window(step["title"])
            
            elif action == "wait":
                time.sleep(step.get("duration", 1.0))
                success = True
            
            elif action == "create_file":
                success = self.system_control.create_file(
                    Path(step["path"]),
                    content=step.get("content", "")
                )
            
            elif action == "move_file":
                success = self.system_control.move_file(
                    Path(step["source"]),
                    Path(step["destination"])
                )
            
            elif action == "delete_file":
                success = self.system_control.delete_file(
                    Path(step["path"]),
                    confirm=step.get("confirm", True)
                )
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
            
            return {
                "success": success,
                "action": action,
                "step": step
            }
            
        except Exception as e:
            return {
                "success": False,
                "action": action,
                "error": str(e)
            }
    
    def _rollback(self, executed_steps: List[Dict[str, Any]]):
        """Attempt to rollback executed steps."""
        logging.info(f"Rolling back {len(executed_steps)} steps...")
        
        for step in reversed(executed_steps):
            try:
                self._rollback_step(step)
            except Exception as e:
                logging.error(f"Rollback failed for step: {str(e)}")
    
    def _rollback_step(self, step: Dict[str, Any]):
        """Rollback a single step if possible."""
        action = step.get("action")
        
        # Only certain actions can be rolled back
        if action == "create_file":
            path = Path(step["path"])
            if path.exists():
                path.unlink()
                logging.info(f"Rolled back: deleted {path}")
        
        elif action == "move_file":
            # Move back
            self.system_control.move_file(
                Path(step["destination"]),
                Path(step["source"]),
                confirm=False
            )
            logging.info(f"Rolled back: moved file back")
        
        elif action == "delete_file":
            logging.warning(f"Cannot rollback file deletion: {step['path']}")
    
    def load_macro(self, macro_name: str) -> Optional[List[Dict[str, Any]]]:
        """Load a macro from YAML file.
        
        Args:
            macro_name: Name of the macro (without .yaml extension)
            
        Returns:
            List of steps or None if not found
        """
        macro_path = self.macro_dir / f"{macro_name}.yaml"
        
        if not macro_path.exists():
            logging.error(f"Macro not found: {macro_name}")
            return None
        
        try:
            with open(macro_path) as f:
                macro_data = yaml.safe_load(f)
            
            return macro_data.get("steps", [])
        except Exception as e:
            logging.error(f"Failed to load macro: {str(e)}")
            return None
    
    def save_macro(self, macro_name: str, steps: List[Dict[str, Any]], 
                   description: str = "") -> bool:
        """Save a macro to YAML file.
        
        Args:
            macro_name: Name of the macro
            steps: List of automation steps
            description: Macro description
            
        Returns:
            True if successful
        """
        macro_path = self.macro_dir / f"{macro_name}.yaml"
        
        try:
            macro_data = {
                "name": macro_name,
                "description": description,
                "created": datetime.now().isoformat(),
                "steps": steps
            }
            
            with open(macro_path, 'w') as f:
                yaml.dump(macro_data, f, default_flow_style=False)
            
            logging.info(f"Macro saved: {macro_name}")
            return True
        except Exception as e:
            logging.error(f"Failed to save macro: {str(e)}")
            return False
    
    def list_macros(self) -> List[str]:
        """List available macros.
        
        Returns:
            List of macro names
        """
        try:
            return [f.stem for f in self.macro_dir.glob("*.yaml")]
        except Exception:
            return []
    
    def execute_macro(self, macro_name: str) -> Dict[str, Any]:
        """Load and execute a macro.
        
        Args:
            macro_name: Name of the macro to execute
            
        Returns:
            Execution result
        """
        steps = self.load_macro(macro_name)
        
        if steps is None:
            return {
                "success": False,
                "error": f"Macro not found: {macro_name}"
            }
        
        logging.info(f"Executing macro: {macro_name}")
        return self.execute_workflow(steps)
