"""
Lightweight async file helpers.

Provides async read/append/write utilities. If `aiofiles` is available it will be used;
otherwise synchronous file IO is executed on a thread pool to preserve async API without
requiring the aiofiles dependency (helps keep lint/installation clean).
"""
from pathlib import Path
import asyncio
from typing import List

import importlib

def _get_aiofiles():
    # Dynamically import aiofiles if available to avoid static linter
    # complaining about missing optional dependency.
    try:
        return importlib.import_module('aiofiles')
    except Exception:
        return None

async def async_write_lines(path: Path, lines: List[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    aiofiles = _get_aiofiles()
    if aiofiles:
        async with aiofiles.open(path, 'a', encoding='utf-8') as f:
            for line in lines:
                await f.write(line + "\n")
    else:
        loop = asyncio.get_event_loop()
        def _sync_write():
            with open(path, 'a', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + "\n")
        await loop.run_in_executor(None, _sync_write)

async def async_read_lines(path: Path) -> List[str]:
    path = Path(path)
    if not path.exists():
        return []
    aiofiles = _get_aiofiles()
    if aiofiles:
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            return [line.rstrip('\n') for line in await f.readlines()]
    else:
        loop = asyncio.get_event_loop()
        def _sync_read():
            with open(path, 'r', encoding='utf-8') as f:
                return [line.rstrip('\n') for line in f.readlines()]
        return await loop.run_in_executor(None, _sync_read)

async def async_overwrite(path: Path, lines: List[str]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    aiofiles = _get_aiofiles()
    if aiofiles:
        async with aiofiles.open(path, 'w', encoding='utf-8') as f:
            for line in lines:
                await f.write(line + "\n")
    else:
        loop = asyncio.get_event_loop()
        def _sync_write():
            with open(path, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + "\n")
        await loop.run_in_executor(None, _sync_write)
