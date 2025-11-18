"""
Tool Integration System

Implements Task 19 requirements:
- Clipboard operations using pyperclip
- Browser automation with selenium
- Windows notification integration with win11toast
- Plugin system with dynamic module loading
- Export functionality for conversations and data

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import json
import importlib
import sys

try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False
    logging.warning("pyperclip not available - clipboard operations disabled")

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logging.warning("selenium not available - browser automation disabled")

try:
    from win11toast import toast
    WIN11TOAST_AVAILABLE = True
except ImportError:
    WIN11TOAST_AVAILABLE = False
    logging.warning("win11toast not available - notifications disabled")


class ClipboardManager:
    """Manage clipboard operations."""
    
    def __init__(self):
        self.available = PYPERCLIP_AVAILABLE
        self.history = []
        self.max_history = 100
    
    def copy(self, text: str) -> bool:
        """Copy text to clipboard.
        
        Args:
            text: Text to copy
            
        Returns:
            True if successful
        """
        if not self.available:
            logging.error("Clipboard not available")
            return False
        
        try:
            pyperclip.copy(text)
            
            # Add to history
            self.history.append({
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "action": "copy"
            })
            
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            
            logging.info(f"Copied to clipboard: {text[:50]}...")
            return True
            
        except Exception as e:
            logging.error(f"Clipboard copy failed: {e}")
            return False
    
    def paste(self) -> Optional[str]:
        """Paste text from clipboard.
        
        Returns:
            Clipboard text or None
        """
        if not self.available:
            logging.error("Clipboard not available")
            return None
        
        try:
            text = pyperclip.paste()
            
            # Add to history
            self.history.append({
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "action": "paste"
            })
            
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            
            return text
            
        except Exception as e:
            logging.error(f"Clipboard paste failed: {e}")
            return None
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get clipboard history.
        
        Args:
            limit: Number of items to return
            
        Returns:
            List of clipboard history items
        """
        return self.history[-limit:]


class BrowserAutomation:
    """Browser automation using Selenium."""
    
    def __init__(self):
        self.available = SELENIUM_AVAILABLE
        self.driver = None
        self.browser_type = "chrome"  # chrome, firefox, edge
    
    def start_browser(self, browser: str = "chrome", headless: bool = False) -> bool:
        """Start browser instance.
        
        Args:
            browser: Browser type (chrome, firefox, edge)
            headless: Run in headless mode
            
        Returns:
            True if successful
        """
        if not self.available:
            logging.error("Selenium not available")
            return False
        
        try:
            if browser == "chrome":
                from selenium.webdriver.chrome.options import Options
                options = Options()
                if headless:
                    options.add_argument("--headless")
                self.driver = webdriver.Chrome(options=options)
            elif browser == "firefox":
                from selenium.webdriver.firefox.options import Options
                options = Options()
                if headless:
                    options.add_argument("--headless")
                self.driver = webdriver.Firefox(options=options)
            elif browser == "edge":
                from selenium.webdriver.edge.options import Options
                options = Options()
                if headless:
                    options.add_argument("--headless")
                self.driver = webdriver.Edge(options=options)
            else:
                logging.error(f"Unsupported browser: {browser}")
                return False
            
            self.browser_type = browser
            logging.info(f"Browser started: {browser}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to start browser: {e}")
            return False
    
    def navigate(self, url: str) -> bool:
        """Navigate to URL.
        
        Args:
            url: URL to navigate to
            
        Returns:
            True if successful
        """
        if not self.driver:
            logging.error("Browser not started")
            return False
        
        try:
            self.driver.get(url)
            logging.info(f"Navigated to: {url}")
            return True
        except Exception as e:
            logging.error(f"Navigation failed: {e}")
            return False
    
    def find_element(self, selector: str, by: str = "css") -> Optional[Any]:
        """Find element on page.
        
        Args:
            selector: Element selector
            by: Selector type (css, xpath, id, name)
            
        Returns:
            Element or None
        """
        if not self.driver:
            return None
        
        try:
            by_type = {
                "css": By.CSS_SELECTOR,
                "xpath": By.XPATH,
                "id": By.ID,
                "name": By.NAME
            }.get(by, By.CSS_SELECTOR)
            
            element = self.driver.find_element(by_type, selector)
            return element
        except Exception as e:
            logging.error(f"Element not found: {e}")
            return None
    
    def click_element(self, selector: str, by: str = "css") -> bool:
        """Click element.
        
        Args:
            selector: Element selector
            by: Selector type
            
        Returns:
            True if successful
        """
        element = self.find_element(selector, by)
        
        if element:
            try:
                element.click()
                return True
            except Exception as e:
                logging.error(f"Click failed: {e}")
        
        return False
    
    def type_text(self, selector: str, text: str, by: str = "css") -> bool:
        """Type text into element.
        
        Args:
            selector: Element selector
            text: Text to type
            by: Selector type
            
        Returns:
            True if successful
        """
        element = self.find_element(selector, by)
        
        if element:
            try:
                element.send_keys(text)
                return True
            except Exception as e:
                logging.error(f"Type failed: {e}")
        
        return False
    
    def get_page_source(self) -> Optional[str]:
        """Get page source HTML.
        
        Returns:
            Page source or None
        """
        if not self.driver:
            return None
        
        try:
            return self.driver.page_source
        except Exception as e:
            logging.error(f"Failed to get page source: {e}")
            return None
    
    def close_browser(self):
        """Close browser instance."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
                logging.info("Browser closed")
            except Exception as e:
                logging.error(f"Failed to close browser: {e}")


class NotificationManager:
    """Windows 11 notification manager using win11toast."""
    
    def __init__(self):
        self.available = WIN11TOAST_AVAILABLE
    
    def show_notification(self,
                         title: str,
                         message: str,
                         duration: str = "short",
                         icon_path: str = None,
                         audio: str = None,
                         buttons: List[Dict[str, str]] = None) -> bool:
        """Show Windows 11 notification.
        
        Args:
            title: Notification title
            message: Notification message
            duration: Duration ("short" or "long")
            icon_path: Path to icon file (optional)
            audio: Audio type (optional: "default", "sms", "alarm", etc.)
            buttons: List of button dicts with 'label' and 'action' keys
            
        Returns:
            True if successful
        """
        if not self.available:
            logging.warning("win11toast not available - notifications disabled")
            return False
        
        try:
            # Build notification parameters
            kwargs = {
                'title': title,
                'body': message,
                'duration': duration
            }
            
            # Add optional parameters
            if icon_path:
                kwargs['icon'] = icon_path
            
            if audio:
                kwargs['audio'] = audio
            
            if buttons:
                kwargs['buttons'] = buttons
            
            # Show notification (async)
            toast(**kwargs)
            
            logging.info(f"Notification shown: {title}")
            return True
            
        except Exception as e:
            logging.error(f"Notification failed: {e}")
            return False
    
    def show_notification_with_buttons(self,
                                      title: str,
                                      message: str,
                                      buttons: List[Dict[str, str]]) -> bool:
        """Show notification with action buttons.
        
        Args:
            title: Notification title
            message: Notification message
            buttons: List of button dicts with 'label' and 'action' keys
            
        Returns:
            True if successful
        """
        return self.show_notification(title, message, buttons=buttons)


class PluginSystem:
    """Dynamic plugin loading system."""
    
    def __init__(self, plugin_dir: Path = None):
        self.plugin_dir = plugin_dir or Path("plugins")
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_metadata: Dict[str, Dict[str, Any]] = {}
    
    def load_plugin(self, plugin_name: str) -> bool:
        """Load plugin dynamically.
        
        Args:
            plugin_name: Name of plugin module
            
        Returns:
            True if successful
        """
        try:
            # Add plugin directory to path if not already there
            plugin_dir_str = str(self.plugin_dir.absolute())
            if plugin_dir_str not in sys.path:
                sys.path.insert(0, plugin_dir_str)
            
            # Import plugin module
            plugin_module = importlib.import_module(plugin_name)
            
            # Get plugin class (convention: Plugin class in module)
            if hasattr(plugin_module, 'Plugin'):
                plugin_class = getattr(plugin_module, 'Plugin')
                plugin_instance = plugin_class()
                
                self.loaded_plugins[plugin_name] = plugin_instance
                
                # Get metadata if available
                if hasattr(plugin_instance, 'get_metadata'):
                    self.plugin_metadata[plugin_name] = plugin_instance.get_metadata()
                
                logging.info(f"Plugin loaded: {plugin_name}")
                return True
            else:
                logging.error(f"Plugin {plugin_name} has no Plugin class")
                return False
                
        except Exception as e:
            logging.error(f"Failed to load plugin {plugin_name}: {e}")
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload plugin.
        
        Args:
            plugin_name: Name of plugin
            
        Returns:
            True if successful
        """
        if plugin_name in self.loaded_plugins:
            # Call cleanup if available
            plugin = self.loaded_plugins[plugin_name]
            if hasattr(plugin, 'cleanup'):
                try:
                    plugin.cleanup()
                except Exception as e:
                    logging.error(f"Plugin cleanup failed: {e}")
            
            del self.loaded_plugins[plugin_name]
            
            if plugin_name in self.plugin_metadata:
                del self.plugin_metadata[plugin_name]
            
            logging.info(f"Plugin unloaded: {plugin_name}")
            return True
        
        return False
    
    def call_plugin(self, plugin_name: str, method: str, *args, **kwargs) -> Any:
        """Call plugin method.
        
        Args:
            plugin_name: Name of plugin
            method: Method name to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:       Method return value
        """
        if plugin_name not in self.loaded_plugins:
            logging.error(f"Plugin not loaded: {plugin_name}")
            return None
        
        plugin = self.loaded_plugins[plugin_name]
        
        if notattr(plugin, method):
            logging.error(f"Plugin {plugin_name} has no method {method}")
            return None
        
        try:
            method_func = getattr(plugin, method)
            return method_func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Plugin method call failed: {e}")
            return None
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List loaded plugins.
        
        Returns:
            List of plugin info
        """
        plugins = []
        
        for name, plugin in self.loaded_plugins.items():
            info = {
                "name": name,
                "metadata": self.plugin_metadata.get(name, {}),
                "methods": [m for m in dir(plugin) if not m.startswith('_')]
            }
            plugins.append(info)
        
        return plugins


class DataExporter:
    """Export conversations and data."""
    
    def __init__(self, export_dir: Path = None):
        self.export_dir = export_dir or Path("exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_conversations(self,
                            conversations: List[Dict[str, Any]],
                            format: str = "json") -> Optional[Path]:
        """Export conversations.
        
        Args:
            conversations: List of conversation data
            format: Export format (json, txt, md)
            
        Returns:
            Path to exported file or None
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversations_{timestamp}.{format}"
            filepath = self.export_dir / filename
            
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(conversations, f, indent=2)
            
            elif format == "txt":
                with open(filepath, 'w') as f:
                    for conv in conversations:
                        f.write(f"=== Conversation {conv.get('id', 'unknown')} ===\n")
                        f.write(f"Timestamp: {conv.get('timestamp', 'unknown')}\n")
                        f.write(f"Content: {conv.get('content', '')}\n\n")
            
            elif format == "md":
                with open(filepath, 'w') as f:
                    f.write("# Conversation Export\n\n")
                    for conv in conversations:
                        f.write(f"## Conversation {conv.get('id', 'unknown')}\n\n")
                        f.write(f"**Timestamp:** {conv.get('timestamp', 'unknown')}\n\n")
                        f.write(f"{conv.get('content', '')}\n\n")
                        f.write("---\n\n")
            
            logging.info(f"Conversations exported to: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Export failed: {e}")
            return None
    
    def export_data(self,
                   data: Dict[str, Any],
                   filename: str,
                   format: str = "json") -> Optional[Path]:
        """Export generic data.
        
        Args:
            data: Data to export
            filename: Output filename
            format: Export format
            
        Returns:
            Path to exported file or None
        """
        try:
            filepath = self.export_dir / f"{filename}.{format}"
            
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                logging.error(f"Unsupported format: {format}")
                return None
            
            logging.info(f"Data exported to: {filepath}")
            return filepath
            
        except Exception as e:
            logging.error(f"Export failed: {e}")
            return None


class ToolIntegration:
    """Main tool integration system."""
    
    def __init__(self):
        self.clipboard = ClipboardManager()
        self.browser = BrowserAutomation()
        self.notifications = NotificationManager()
        self.plugins = PluginSystem()
        self.exporter = DataExporter()
        
        logging.info("Tool Integration System initialized")
    
    def get_status(self) -> Dict[str, bool]:
        """Get tool availability status.
        
        Returns:
            Dictionary of tool availability
        """
        return {
            "clipboard": self.clipboard.available,
            "browser": self.browser.available,
            "notifications": self.notifications.available,
            "plugins": True,
            "exporter": True
        }


# Global instance
_tool_integration: Optional[ToolIntegration] = None


def get_tool_integration() -> ToolIntegration:
    """Get global tool integration instance."""
    global _tool_integration
    
    if _tool_integration is None:
        _tool_integration = ToolIntegration()
    
    return _tool_integration
