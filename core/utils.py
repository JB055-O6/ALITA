"""
Shared utilities and common exports used across core modules.

Expose commonly-used third-party modules and small helpers so we
don't duplicate heavy imports across many files.
"""
import os
import logging
from pathlib import Path
import asyncio

# Third-party libraries exposed for convenience
import numpy as np
import pandas as pd
import torch
import psutil

def ensure_dir(path: Path) -> None:
    """Ensure a directory exists."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def format_timestamp(ts=None):
    from datetime import datetime
    ts = ts or datetime.now()
    return ts.isoformat()

def setup_basic_logging(level: str = "INFO"):
    """Configure basic console logging if not already configured."""
    log = logging.getLogger()
    if not log.handlers:
        logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

def safe_cuda_percent() -> float:
    """Return a safe GPU usage percent (0.0 if no CUDA)."""
    try:
        if torch.cuda.is_available():
            used = float(torch.cuda.memory_allocated())
            total = float(torch.cuda.max_memory_allocated())
            if total <= 0:
                return 0.0
            return used / total * 100.0
    except Exception:
        return 0.0

__all__ = [
    "np",
    "pd",
    "torch",
    "psutil",
    "ensure_dir",
    "format_timestamp",
    "setup_basic_logging",
    "safe_cuda_percent",
]
