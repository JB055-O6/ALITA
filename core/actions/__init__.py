"""
Action handlers - Each file handles specific action types
"""

from . import app_actions
from . import file_actions
from . import screen_actions
from . import automation_actions
from . import web_actions
from . import system_actions

__all__ = [
    'app_actions',
    'file_actions', 
    'screen_actions',
    'automation_actions',
    'web_actions',
    'system_actions'
]
