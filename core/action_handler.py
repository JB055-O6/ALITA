"""Action Handler - Handles all action execution logic extracted from brain.py"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime


class ActionHandler:
    """Handles execution of all user actions."""
    
    def __init__(self, automation, vision_system, sys_controller):
        self.automation = automation
        self.vision_system = vision_system
        self.sys_controller = sys_controller
        self._image_generator = None
        self._content_generator = None
    
    @property
    def image_generator(self):
        if self._image_generator is None:
            from .image_generation import ImageGenerator
            self._image_generator = ImageGenerator()
        return self._image_generator
    
    @property
    def content_generator(self):
        if self._content_generator is None:
            from .content_generation import ContentGenerator
            self._content_generator = ContentGenerator()
        return self._content_generator
    
    def handle_action(self, action_type: str, user_input: str, user_lower: str) -> Dict[str, Any]:
        """Route action to appropriate handler."""
        handlers = {
            "system_info": lambda: self.action_system_info(),
            "memory_search": lambda: self.action_memory_search(user_input, user_lower),
            "web_search": lambda: self.action_web_search(user_input, user_lower),
            "open_url": lambda: self.action_open_url(user_input, user_lower),
            "execute_code": lambda: self.action_execute_code(user_input, user_lower),
            "mouse_click": lambda: self.action_mouse_click(user_input, user_lower),
            "type_text": lambda: self.action_type_text(user_input, user_lower),
            "scroll": lambda: self.action_scroll(user_input, user_lower),
            "open_app": lambda: self.action_open_app(user_input, user_lower),
            "close_app": lambda: self.action_close_app(user_input, user_lower),
            "open_file": lambda: self.action_open_file(user_input, user_lower),
            "search_files": lambda: self.action_search_files(user_input, user_lower),
            "create_file": lambda: self.action_create_file(user_input, user_lower),
            "list_files": lambda: self.action_list_files(user_input, user_lower),
            "screenshot": lambda: self.action_screenshot(),
            "read_screen": lambda: self.action_read_screen(),
            "generate_image": lambda: self.action_generate_image(user_input, user_lower),
            "write_code": lambda: self.action_write_code(user_input, user_lower),
        }
        
        handler = handlers.get(action_type)
        if handler:
            try:
                return handler()
            except Exception as e:
                logging.error(f"Action {action_type} error: {e}")
                return {"success": False, "message": f"Error: {str(e)}"}
        
        return {"success": False, "message": f"Unknown action: {action_type}"}
    
    def action_open_app(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        """Open application."""
        app_name = self._extract_app_name(user_lower)
        if not app_name:
            return {"success": False, "message": "Couldn't determine which app to open."}
        
        result = self.automation.launch_with_fallbacks(app_name)
        return {
            "success": result["success"],
            "message": result["message"],
            "action": "open_app",
            "app": app_name
        }
    
    def action_close_app(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        """Close application."""
        app_name = self._extract_app_name(user_lower.replace('close', '').replace('quit', ''))
        if not app_name:
            return {"success": False, "message": "Couldn't determine which app to close."}
        
        success = self.automation.close_app(app_name)
        return {
            "success": success,
            "message": f"{'Closed' if success else 'Could not close'} {app_name}",
            "action": "close_app"
        }
    
    def action_search_files(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        """Search for files."""
        query = self._extract_search_query(user_lower)
        if not query:
            return {"success": False, "message": "Couldn't determine what to search for."}
        
        try:
            from .file_search import SafeFileSearch
            searcher = SafeFileSearch()
            results = searcher.search_priority_locations(query)
            
            if results:
                result_list = "\n".join([f"- {r['name']} ({r['path']})" for r in results[:10]])
                return {
                    "success": True,
                    "message": f"Found {len(results)} items:\n{result_list}",
                    "results": results
                }
            return {"success": True, "message": f"No files found matching '{query}'", "results": []}
        except Exception as e:
            return {"success": False, "message": f"Search error: {str(e)}"}
    
    def action_screenshot(self) -> Dict[str, Any]:
        """Take screenshot."""
        try:
            screenshots_dir = Path.home() / "Pictures" / "ALITA_Screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            
            screenshot = self.automation.screenshot()
            if screenshot:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = screenshots_dir / f"screenshot_{timestamp}.png"
                screenshot.save(filepath)
                return {"success": True, "message": f"Screenshot saved to: {filepath}"}
            return {"success": False, "message": "Screenshot failed"}
        except Exception as e:
            return {"success": False, "message": f"Screenshot error: {str(e)}"}
    
    def action_read_screen(self) -> Dict[str, Any]:
        """Read screen text."""
        try:
            text = self.vision_system.read_screen_text()
            if text and text.strip():
                return {"success": True, "message": f"Screen text:\n{text[:500]}", "text": text}
            return {"success": True, "message": "No text detected", "text": ""}
        except Exception as e:
            return {"success": False, "message": f"OCR error: {str(e)}"}
    
    def action_generate_image(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        """Generate image."""
        prompt = user_input
        for phrase in ['create image', 'generate image', 'make image', 'draw']:
            prompt = prompt.replace(phrase, '').strip()
        
        if not prompt:
            return {"success": False, "message": "Please provide image description."}
        
        image_path = self.image_generator.generate_image(prompt)
        if image_path:
            return {"success": True, "message": f"Image saved to: {image_path}", "path": image_path}
        return {"success": False, "message": "Image generation failed"}
    
    def action_write_code(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        """Write code."""
        prompt = user_input
        for phrase in ['write code', 'create code', 'code for']:
            prompt = prompt.replace(phrase, '').strip()
        
        if not prompt:
            return {"success": False, "message": "Please describe the code you want."}
        
        code = self.content_generator.generate_code("python", "script", prompt)
        if code:
            return {"success": True, "message": f"Code:\n```python\n{code}\n```", "code": code}
        return {"success": False, "message": "Code generation failed"}
    
    def action_system_info(self) -> Dict[str, Any]:
        """Get system info."""
        try:
            info = self.sys_controller.get_system_info()
            return {"success": True, "message": "System info retrieved", "info": info}
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    # Stub methods for remaining actions
    def action_create_file(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        return {"success": False, "message": "Create file not implemented"}
    
    def action_list_files(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        return {"success": False, "message": "List files not implemented"}
    
    def action_mouse_click(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        return {"success": False, "message": "Mouse click not implemented"}
    
    def action_type_text(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        return {"success": False, "message": "Type text not implemented"}
    
    def action_scroll(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        return {"success": False, "message": "Scroll not implemented"}
    
    def action_open_file(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        return {"success": False, "message": "Open file not implemented"}
    
    def action_web_search(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        return {"success": False, "message": "Web search not implemented"}
    
    def action_open_url(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        return {"success": False, "message": "Open URL not implemented"}
    
    def action_execute_code(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        return {"success": False, "message": "Execute code not implemented"}
    
    def action_memory_search(self, user_input: str, user_lower: str) -> Dict[str, Any]:
        return {"success": False, "message": "Memory search not implemented"}
    
    # Helper methods
    def _extract_app_name(self, user_lower: str) -> Optional[str]:
        """Extract app name from input."""
        for word in ['open', 'launch', 'start', 'run', 'the', 'app']:
            user_lower = user_lower.replace(word, '')
        return user_lower.strip() or None
    
    def _extract_search_query(self, user_lower: str) -> Optional[str]:
        """Extract search query from input."""
        for phrase in ['search for', 'find file', 'find files', 'look for', 'locate']:
            user_lower = user_lower.replace(phrase, '')
        return user_lower.strip() or None
