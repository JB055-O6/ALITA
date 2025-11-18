"""
Spatial Desktop Awareness System

Implements Task 23 requirements:
- 2D desktop layout map using win32gui
- Window arrangement optimizer
- Activity-based layout memory
- Resolution change handler
- Multi-monitor window distribution

Integrates with:
- ContextManager for activity tracking
- LearningSystem for layout preferences

All features are FREE and run locally!
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict
import json
import asyncio

try:
    import win32gui
    import win32con
    import win32api
except ImportError:
    win32gui = None
    win32con = None
    win32api = None

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class WindowInfo:
    """Information about a window."""
    handle: int
    title: str
    class_name: str
    position: Tuple[int, int, int, int]  # (left, top, right, bottom)
    is_visible: bool
    is_minimized: bool
    is_maximized: bool
    z_order: int
    monitor_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @property
    def width(self) -> int:
        """Get window width."""
        return self.position[2] - self.position[0]
    
    @property
    def height(self) -> int:
        """Get window height."""
        return self.position[3] - self.position[1]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get window center point."""
        return (
            (self.position[0] + self.position[2]) // 2,
            (self.position[1] + self.position[3]) // 2
        )


@dataclass
class MonitorInfo:
    """Information about a monitor."""
    index: int
    position: Tuple[int, int, int, int]  # (left, top, right, bottom)
    is_primary: bool
    work_area: Tuple[int, int, int, int]  # Excluding taskbar
    
    @property
    def width(self) -> int:
        """Get monitor width."""
        return self.position[2] - self.position[0]
    
    @property
    def height(self) -> int:
        """Get monitor height."""
        return self.position[3] - self.position[1]


@dataclass
class LayoutPreset:
    """Saved layout preset."""
    name: str
    activity: str
    windows: List[Dict[str, Any]]
    monitors: List[Dict[str, Any]]
    created: datetime
    usage_count: int


class DesktopMapper:
    """Build 2D map of desktop layout."""
    
    def __init__(self):
        self.windows: Dict[int, WindowInfo] = {}
        self.monitors: List[MonitorInfo] = []
        self.spatial_map: Optional[np.ndarray] = None
        self.map_resolution = (100, 100)  # Grid resolution
    
    def scan_desktop(self) -> Dict[str, Any]:
        """Scan current desktop layout.
        
        Returns:
            Desktop layout information
        """
        if not win32gui:
            logging.warning("win32gui not available")
            return {"windows": [], "monitors": []}
        
        # Get monitors
        self.monitors = self._get_monitors()
        
        # Get windows
        self.windows = {}
        z_order = 0
        
        def enum_callback(hwnd, _):
            nonlocal z_order
            
            if win32gui.IsWindowVisible(hwnd):
                try:
                    title = win32gui.GetWindowText(hwnd)
                    class_name = win32gui.GetClassName(hwnd)
                    
                    # Skip empty titles and system windows
                    if not title or title.startswith("MSCTFIME"):
                        return True
                    
                    # Get window position
                    rect = win32gui.GetWindowRect(hwnd)
                    
                    # Get window state
                    placement = win32gui.GetWindowPlacement(hwnd)
                    is_minimized = placement[1] == win32con.SW_SHOWMINIMIZED
                    is_maximized = placement[1] == win32con.SW_SHOWMAXIMIZED
                    
                    # Determine monitor
                    monitor_index = self._get_window_monitor(rect)
                    
                    # Create window info
                    window = WindowInfo(
                        handle=hwnd,
                        title=title,
                        class_name=class_name,
                        position=rect,
                        is_visible=True,
                        is_minimized=is_minimized,
                        is_maximized=is_maximized,
                        z_order=z_order,
                        monitor_index=monitor_index
                    )
                    
                    self.windows[hwnd] = window
                    z_order += 1
                    
                except Exception as e:
                    logging.debug(f"Error processing window {hwnd}: {e}")
            
            return True
        
        try:
            win32gui.EnumWindows(enum_callback, None)
        except Exception as e:
            logging.error(f"Desktop scan failed: {e}")
        
        # Build spatial map
        self._build_spatial_map()
        
        return {
            "windows": [w.to_dict() for w in self.windows.values()],
            "monitors": [asdict(m) for m in self.monitors],
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_monitors(self) -> List[MonitorInfo]:
        """Get monitor information."""
        if not win32api:
            return []
        
        monitors = []
        
        try:
            for i, monitor in enumerate(win32api.EnumDisplayMonitors()):
                monitor_info = win32api.GetMonitorInfo(monitor[0])
                
                monitors.append(MonitorInfo(
                    index=i,
                    position=monitor_info['Monitor'],
                    is_primary=(monitor_info['Flags'] & 1) == 1,
                    work_area=monitor_info['Work']
                ))
        except Exception as e:
            logging.error(f"Failed to get monitors: {e}")
        
        return monitors
    
    def _get_window_monitor(self, window_rect: Tuple[int, int, int, int]) -> int:
        """Determine which monitor a window is on."""
        window_center = (
            (window_rect[0] + window_rect[2]) // 2,
            (window_rect[1] + window_rect[3]) // 2
        )
        
        for monitor in self.monitors:
            if (monitor.position[0] <= window_center[0] <= monitor.position[2] and
                monitor.position[1] <= window_center[1] <= monitor.position[3]):
                return monitor.index
        
        return 0  # Default to primary monitor
    
    def _build_spatial_map(self):
        """Build 2D spatial map of desktop."""
        if not np or not self.monitors:
            return
        
        # Get total desktop bounds
        if self.monitors:
            min_x = min(m.position[0] for m in self.monitors)
            min_y = min(m.position[1] for m in self.monitors)
            max_x = max(m.position[2] for m in self.monitors)
            max_y = max(m.position[3] for m in self.monitors)
        else:
            min_x, min_y, max_x, max_y = 0, 0, 1920, 1080
        
        # Create grid
        width = max_x - min_x
        height = max_y - min_y
        
        self.spatial_map = np.zeros(self.map_resolution)
        
        # Mark occupied areas
        for window in self.windows.values():
            if window.is_minimized:
                continue
            
            # Convert window position to grid coordinates
            x1 = int((window.position[0] - min_x) / width * self.map_resolution[0])
            y1 = int((window.position[1] - min_y) / height * self.map_resolution[1])
            x2 = int((window.position[2] - min_x) / width * self.map_resolution[0])
            y2 = int((window.position[3] - min_y) / height * self.map_resolution[1])
            
            # Clamp to grid bounds
            x1 = max(0, min(x1, self.map_resolution[0] - 1))
            y1 = max(0, min(y1, self.map_resolution[1] - 1))
            x2 = max(0, min(x2, self.map_resolution[0] - 1))
            y2 = max(0, min(y2, self.map_resolution[1] - 1))
            
            # Mark as occupied
            self.spatial_map[y1:y2, x1:x2] = 1
    
    def find_free_space(self, width: int, height: int, monitor_index: int = 0) -> Optional[Tuple[int, int]]:
        """Find free space for window placement.
        
        Args:
            width: Required width
            height: Required height
            monitor_index: Preferred monitor
            
        Returns:
            (x, y) position or None
        """
        if monitor_index >= len(self.monitors):
            monitor_index = 0
        
        monitor = self.monitors[monitor_index]
        work_area = monitor.work_area
        
        # Try to find free space
        for y in range(work_area[1], work_area[3] - height, 50):
            for x in range(work_area[0], work_area[2] - width, 50):
                # Check if space is free
                rect = (x, y, x + width, y + height)
                if self._is_space_free(rect):
                    return (x, y)
        
        # Fallback to top-left of work area
        return (work_area[0], work_area[1])
    
    def _is_space_free(self, rect: Tuple[int, int, int, int]) -> bool:
        """Check if space is free of windows."""
        for window in self.windows.values():
            if window.is_minimized:
                continue
            
            # Check for overlap
            if not (rect[2] < window.position[0] or
                   rect[0] > window.position[2] or
                   rect[3] < window.position[1] or
                   rect[1] > window.position[3]):
                return False
        
        return True


class WindowArranger:
    """Arrange windows intelligently."""
    
    def __init__(self, desktop_mapper: DesktopMapper):
        self.desktop_mapper = desktop_mapper
    
    def arrange_windows(self, layout_type: str = "tile") -> bool:
        """Arrange windows according to layout type.
        
        Args:
            layout_type: Layout type (tile, cascade, side_by_side)
            
        Returns:
            True if successful
        """
        if not win32gui:
            return False
        
        # Get visible windows
        windows = [w for w in self.desktop_mapper.windows.values()
                  if not w.is_minimized and not w.is_maximized]
        
        if not windows:
            return False
        
        if layout_type == "tile":
            return self._tile_windows(windows)
        elif layout_type == "cascade":
            return self._cascade_windows(windows)
        elif layout_type == "side_by_side":
            return self._side_by_side_windows(windows)
        
        return False
    
    def _tile_windows(self, windows: List[WindowInfo]) -> bool:
        """Tile windows in grid."""
        if not self.desktop_mapper.monitors:
            return False
        
        monitor = self.desktop_mapper.monitors[0]
        work_area = monitor.work_area
        
        # Calculate grid
        n = len(windows)
        cols = int(np.ceil(np.sqrt(n))) if np else 2
        rows = int(np.ceil(n / cols)) if np else 2
        
        width = (work_area[2] - work_area[0]) // cols
        height = (work_area[3] - work_area[1]) // rows
        
        # Arrange windows
        for i, window in enumerate(windows):
            row = i // cols
            col = i % cols
            
            x = work_area[0] + col * width
            y = work_area[1] + row * height
            
            try:
                win32gui.MoveWindow(
                    window.handle,
                    x, y, width, height,
                    True
                )
            except Exception as e:
                logging.warning(f"Failed to move window {window.title}: {e}")
        
        return True
    
    def _cascade_windows(self, windows: List[WindowInfo]) -> bool:
        """Cascade windows."""
        if not self.desktop_mapper.monitors:
            return False
        
        monitor = self.desktop_mapper.monitors[0]
        work_area = monitor.work_area
        
        offset = 30
        width = (work_area[2] - work_area[0]) * 2 // 3
        height = (work_area[3] - work_area[1]) * 2 // 3
        
        for i, window in enumerate(windows):
            x = work_area[0] + i * offset
            y = work_area[1] + i * offset
            
            try:
                win32gui.MoveWindow(
                    window.handle,
                    x, y, width, height,
                    True
                )
            except Exception as e:
                logging.warning(f"Failed to move window {window.title}: {e}")
        
        return True
    
    def _side_by_side_windows(self, windows: List[WindowInfo]) -> bool:
        """Arrange windows side by side."""
        if not self.desktop_mapper.monitors or not windows:
            return False
        
        monitor = self.desktop_mapper.monitors[0]
        work_area = monitor.work_area
        
        n = len(windows)
        width = (work_area[2] - work_area[0]) // n
        height = work_area[3] - work_area[1]
        
        for i, window in enumerate(windows):
            x = work_area[0] + i * width
            y = work_area[1]
            
            try:
                win32gui.MoveWindow(
                    window.handle,
                    x, y, width, height,
                    True
                )
            except Exception as e:
                logging.warning(f"Failed to move window {window.title}: {e}")
        
        return True
    
    def optimize_for_activity(self, activity: str) -> bool:
        """Optimize layout for specific activity.
        
        Args:
            activity: Activity type (coding, browsing, gaming, etc.)
            
        Returns:
            True if successful
        """
        activity_layouts = {
            "coding": "side_by_side",  # Editor + terminal/browser
            "browsing": "tile",
            "gaming": "maximize",
            "video": "maximize"
        }
        
        layout = activity_layouts.get(activity, "tile")
        
        if layout == "maximize":
            # Maximize active window
            windows = [w for w in self.desktop_mapper.windows.values()
                      if not w.is_minimized]
            if windows:
                try:
                    win32gui.ShowWindow(windows[0].handle, win32con.SW_MAXIMIZE)
                    return True
                except Exception as e:
                    logging.warning(f"Failed to maximize window: {e}")
        else:
            return self.arrange_windows(layout)
        
        return False


class LayoutMemory:
    """Remember and restore layout presets."""
    
    def __init__(self, data_file: Path = None):
        self.data_file = data_file or Path("data/layout_presets.json")
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.presets: Dict[str, LayoutPreset] = {}
        self.activity_layouts = defaultdict(list)
        
        self.load_data()
    
    def load_data(self):
        """Load layout presets."""
        if self.data_file.exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    
                    for preset_data in data.get("presets", []):
                        preset = LayoutPreset(
                            name=preset_data["name"],
                            activity=preset_data["activity"],
                            windows=preset_data["windows"],
                            monitors=preset_data["monitors"],
                            created=datetime.fromisoformat(preset_data["created"]),
                            usage_count=preset_data.get("usage_count", 0)
                        )
                        self.presets[preset.name] = preset
                        self.activity_layouts[preset.activity].append(preset.name)
            except Exception as e:
                logging.error(f"Failed to load layout presets: {e}")
    
    def save_data(self):
        """Save layout presets."""
        try:
            presets_data = []
            for preset in self.presets.values():
                presets_data.append({
                    "name": preset.name,
                    "activity": preset.activity,
                    "windows": preset.windows,
                    "monitors": preset.monitors,
                    "created": preset.created.isoformat(),
                    "usage_count": preset.usage_count
                })
            
            with open(self.data_file, 'w') as f:
                json.dump({"presets": presets_data}, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save layout presets: {e}")
    
    def save_current_layout(self, name: str, activity: str, desktop_layout: Dict[str, Any]):
        """Save current layout as preset.
        
        Args:
            name: Preset name
            activity: Activity type
            desktop_layout: Current desktop layout
        """
        preset = LayoutPreset(
            name=name,
            activity=activity,
            windows=desktop_layout.get("windows", []),
            monitors=desktop_layout.get("monitors", []),
            created=datetime.now(),
            usage_count=0
        )
        
        self.presets[name] = preset
        self.activity_layouts[activity].append(name)
        
        self.save_data()
        logging.info(f"Saved layout preset: {name}")
    
    def restore_layout(self, name: str) -> bool:
        """Restore saved layout.
        
        Args:
            name: Preset name
            
        Returns:
            True if successful
        """
        if name not in self.presets or not win32gui:
            return False
        
        preset = self.presets[name]
        preset.usage_count += 1
        
        # Restore window positions
        for window_data in preset.windows:
            try:
                # Find window by title
                hwnd = win32gui.FindWindow(None, window_data["title"])
                if hwnd:
                    pos = window_data["position"]
                    win32gui.MoveWindow(hwnd, pos[0], pos[1],
                                      pos[2] - pos[0], pos[3] - pos[1], True)
            except Exception as e:
                logging.warning(f"Failed to restore window {window_data['title']}: {e}")
        
        self.save_data()
        return True
    
    def get_preferred_layout(self, activity: str) -> Optional[str]:
        """Get preferred layout for activity.
        
        Args:
            activity: Activity type
            
        Returns:
            Preset name or None
        """
        presets = self.activity_layouts.get(activity, [])
        
        if not presets:
            return None
        
        # Return most used preset
        most_used = max(
            [self.presets[name] for name in presets],
            key=lambda p: p.usage_count
        )
        
        return most_used.name


class ResolutionHandler:
    """Handle resolution changes."""
    
    def __init__(self):
        self.current_resolution: Optional[Tuple[int, int]] = None
        self.previous_layouts: Dict[Tuple[int, int], Dict[str, Any]] = {}
    
    def detect_resolution_change(self, monitors: List[MonitorInfo]) -> bool:
        """Detect if resolution changed.
        
        Args:
            monitors: Current monitor information
            
        Returns:
            True if resolution changed
        """
        if not monitors:
            return False
        
        # Get primary monitor resolution
        primary = next((m for m in monitors if m.is_primary), monitors[0])
        resolution = (primary.width, primary.height)
        
        if self.current_resolution and self.current_resolution != resolution:
            logging.info(f"Resolution changed: {self.current_resolution} → {resolution}")
            return True
        
        self.current_resolution = resolution
        return False
    
    def save_layout_for_resolution(self, resolution: Tuple[int, int], layout: Dict[str, Any]):
        """Save layout for specific resolution."""
        self.previous_layouts[resolution] = layout
    
    def restore_layout_for_resolution(self, resolution: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Restore layout for resolution."""
        return self.previous_layouts.get(resolution)


class SpatialEngine:
    """Main spatial desktop awareness system."""
    
    def __init__(self):
        self.desktop_mapper = DesktopMapper()
        self.window_arranger = WindowArranger(self.desktop_mapper)
        self.layout_memory = LayoutMemory()
        self.resolution_handler = ResolutionHandler()
        
        # Current state
        self.current_activity: Optional[str] = None
        self.last_scan: Optional[datetime] = None
        
        # Start monitoring
        self._start_monitoring()
        
        logging.info("Spatial Engine initialized")
    
    def scan_and_map(self) -> Dict[str, Any]:
        """Scan and map desktop layout.
        
        Returns:
            Desktop layout information
        """
        layout = self.desktop_mapper.scan_desktop()
        self.last_scan = datetime.now()
        
        # Check for resolution change
        if self.resolution_handler.detect_resolution_change(self.desktop_mapper.monitors):
            self._handle_resolution_change()
        
        return layout
    
    def arrange_for_activity(self, activity: str) -> bool:
        """Arrange windows for activity.
        
        Args:
            activity: Activity type
            
        Returns:
            True if successful
        """
        self.current_activity = activity
        
        # Check for saved layout
        preferred_layout = self.layout_memory.get_preferred_layout(activity)
        
        if preferred_layout:
            return self.layout_memory.restore_layout(preferred_layout)
        else:
            # Use default optimization
            return self.window_arranger.optimize_for_activity(activity)
    
    def save_current_layout(self, name: str, activity: str):
        """Save current layout as preset.
        
        Args:
            name: Preset name
            activity: Activity type
        """
        layout = self.scan_and_map()
        self.layout_memory.save_current_layout(name, activity, layout)
    
    def distribute_across_monitors(self) -> bool:
        """Distribute windows across multiple monitors.
        
        Returns:
            True if successful
        """
        if len(self.desktop_mapper.monitors) < 2:
            return False
        
        windows = [w for w in self.desktop_mapper.windows.values()
                  if not w.is_minimized]
        
        if not windows:
            return False
        
        # Distribute evenly
        windows_per_monitor = len(windows) // len(self.desktop_mapper.monitors)
        
        for i, monitor in enumerate(self.desktop_mapper.monitors):
            start_idx = i * windows_per_monitor
            end_idx = start_idx + windows_per_monitor if i < len(self.desktop_mapper.monitors) - 1 else len(windows)
            
            monitor_windows = windows[start_idx:end_idx]
            
            # Arrange windows on this monitor
            work_area = monitor.work_area
            width = (work_area[2] - work_area[0]) // len(monitor_windows) if monitor_windows else 0
            height = work_area[3] - work_area[1]
            
            for j, window in enumerate(monitor_windows):
                x = work_area[0] + j * width
                y = work_area[1]
                
                try:
                    if win32gui:
                        win32gui.MoveWindow(window.handle, x, y, width, height, True)
                except Exception as e:
                    logging.warning(f"Failed to move window: {e}")
        
        return True
    
    def get_spatial_summary(self) -> Dict[str, Any]:
        """Get spatial awareness summary.
        
        Returns:
            Summary dictionary
        """
        return {
            "monitors": len(self.desktop_mapper.monitors),
            "windows": len(self.desktop_mapper.windows),
            "current_activity": self.current_activity,
            "saved_layouts": len(self.layout_memory.presets),
            "last_scan": self.last_scan.isoformat() if self.last_scan else None,
            "resolution": self.resolution_handler.current_resolution
        }
    
    def _handle_resolution_change(self):
        """Handle resolution change."""
        # Save current layout for old resolution
        if self.resolution_handler.current_resolution:
            layout = self.desktop_mapper.scan_desktop()
            self.resolution_handler.save_layout_for_resolution(
                self.resolution_handler.current_resolution,
                layout
            )
        
        # Try to restore layout for new resolution
        new_resolution = (
            self.desktop_mapper.monitors[0].width,
            self.desktop_mapper.monitors[0].height
        ) if self.desktop_mapper.monitors else None
        
        if new_resolution:
            saved_layout = self.resolution_handler.restore_layout_for_resolution(new_resolution)
            if saved_layout:
                logging.info("Restored layout for new resolution")
    
    def _start_monitoring(self):
        """Start background monitoring."""
        async def monitoring_loop():
            while True:
                try:
                    # Periodic scan
                    self.scan_and_map()
                    
                    await asyncio.sleep(30)  # Every 30 seconds
                    
                except Exception as e:
                    logging.error(f"Spatial monitoring error: {e}")
                    await asyncio.sleep(30)
        
        asyncio.create_task(monitoring_loop())


# Global instance
_spatial_engine: Optional[SpatialEngine] = None


def get_spatial_engine() -> SpatialEngine:
    """Get global spatial engine instance."""
    global _spatial_engine
    
    if _spatial_engine is None:
        _spatial_engine = SpatialEngine()
    
    return _spatial_engine
