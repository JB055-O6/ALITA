"""
Advanced Logging Service with Analytics
Features:
- Structured logging
- Log analytics
- Pattern detection
- Log rotation
- Distributed tracing
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from alita.core.async_file import async_write_lines, async_read_lines
from typing import AsyncGenerator
from pathlib import Path
from alita.core.utils import pd, np, ensure_dir
from contextlib import asynccontextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import re
from collections import defaultdict
import uuid
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: str
    message: str
    source: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_id: Optional[str] = None
    context: Dict[str, Any] = None
    tags: List[str] = None

class PatternDetector:
    """Log pattern detection."""
    
    def __init__(self, min_samples: int = 3):
        self.vectorizer = TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b',
            analyzer='word'
        )
        self.clusterer = DBSCAN(
            eps=0.3,
            min_samples=min_samples,
            metric='cosine'
        )
        self.patterns = []
    
    def detect_patterns(self, logs: List[LogEntry]) -> List[Dict[str, Any]]:
        """Detect patterns in log messages."""
        if not logs:
            return []
        
        # Extract messages
        messages = [log.message for log in logs]
        
        # Vectorize
        vectors = self.vectorizer.fit_transform(messages)
        
        # Cluster
        clusters = self.clusterer.fit_predict(vectors)
        
        # Group by cluster
        patterns = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            if cluster_id != -1:  # Ignore noise
                patterns[cluster_id].append(logs[i])
        
        # Extract patterns
        results = []
        for cluster_id, cluster_logs in patterns.items():
            pattern = self._extract_pattern(cluster_logs)
            if pattern:
                results.append({
                    "pattern": pattern,
                    "count": len(cluster_logs),
                    "first_seen": min(log.timestamp for log in cluster_logs),
                    "last_seen": max(log.timestamp for log in cluster_logs),
                    "sources": list(set(log.source for log in cluster_logs)),
                    "levels": list(set(log.level for log in cluster_logs))
                })
        
        return results
    
    def _extract_pattern(self, logs: List[LogEntry]) -> Optional[str]:
        """Extract common pattern from logs."""
        if not logs:
            return None
        
        # Tokenize messages
        tokens_list = [
            re.findall(r'\b\w+\b', log.message)
            for log in logs
        ]
        
        # Find common tokens
        common_tokens = set(tokens_list[0])
        for tokens in tokens_list[1:]:
            common_tokens.intersection_update(tokens)
        
        # Build pattern
        pattern_parts = []
        for token in tokens_list[0]:
            if token in common_tokens:
                pattern_parts.append(token)
            else:
                pattern_parts.append('*')
        
        return ' '.join(pattern_parts)

class TraceContext:
    """Distributed tracing context."""
    
    def __init__(self,
                 trace_id: Optional[str] = None,
                 parent_id: Optional[str] = None):
        self.trace_id = trace_id or str(uuid.uuid4())
        self.parent_id = parent_id
        self.span_id = str(uuid.uuid4())
    
    def child(self) -> 'TraceContext':
        """Create child trace context."""
        return TraceContext(
            trace_id=self.trace_id,
            parent_id=self.span_id
        )

class LogProcessor:
    """Log processing and analysis."""
    
    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.logs_cache = []
        self.cache_limit = 1000
    
    def process_logs(self,
                    logs: List[LogEntry]
                    ) -> Dict[str, Any]:
        """Process and analyze logs."""
        # Update cache
        self.logs_cache.extend(logs)
        if len(self.logs_cache) > self.cache_limit:
            self.logs_cache = self.logs_cache[-self.cache_limit:]
        
        # Calculate statistics
        level_counts = defaultdict(int)
        source_counts = defaultdict(int)
        error_traces = []
        
        for log in logs:
            level_counts[log.level] += 1
            source_counts[log.source] += 1
            
            if log.level in ('ERROR', 'CRITICAL'):
                error_traces.append({
                    "timestamp": log.timestamp.isoformat(),
                    "message": log.message,
                    "trace_id": log.trace_id,
                    "context": log.context
                })
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(self.logs_cache)
        
        return {
            "processed_count": len(logs),
            "level_distribution": dict(level_counts),
            "source_distribution": dict(source_counts),
            "error_traces": error_traces,
            "detected_patterns": patterns
        }

class LogWriter:
    """Asynchronous log writing."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = self._get_current_file()
        self.buffer = []
        self.buffer_limit = 100
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def _get_current_file(self) -> Path:
        """Get current log file path."""
        timestamp = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"log_{timestamp}.jsonl"
    
    async def write_log(self, log: LogEntry):
        """Write log entry."""
        try:
            # Add to buffer
            with self.lock:
                self.buffer.append(log)
            
            # Flush if needed
            if len(self.buffer) >= self.buffer_limit:
                await self.flush()
            
        except Exception as e:
            logging.error(f"Log write failed: {str(e)}")
    
    async def flush(self):
        """Flush log buffer."""
        if not self.buffer:
            return
        
        try:
            # Get current buffer
            with self.lock:
                current_buffer = self.buffer
                self.buffer = []
            
            # Check file rotation
            current_file = self._get_current_file()
            if current_file != self.current_file:
                await self._rotate_logs()
                self.current_file = current_file
            
            # Write logs
            await self._write_logs(current_buffer)
            
        except Exception as e:
            logging.error(f"Log flush failed: {str(e)}")
            # Return logs to buffer
            with self.lock:
                self.buffer = current_buffer + self.buffer
    
    async def _write_logs(self, logs: List[LogEntry]):
        """Write logs to file (async). Uses fallback writer when aiofiles is not present."""
        # Convert to JSON lines
        log_lines = []
        for log in logs:
            log_dict = {
                "timestamp": log.timestamp.isoformat(),
                "level": log.level,
                "message": log.message,
                "source": log.source,
                "trace_id": log.trace_id,
                "span_id": log.span_id,
                "parent_id": log.parent_id,
                "context": log.context,
                "tags": log.tags
            }
            log_lines.append(json.dumps(log_dict))

        await async_write_lines(self.current_file, log_lines)
    
    async def _rotate_logs(self):
        """Rotate log files."""
        # Keep last 7 days of logs
        old_logs = sorted(
            self.log_dir.glob("log_*.jsonl"),
            key=lambda p: p.stat().st_mtime
        )[:-7]
        
        for log in old_logs:
            log.unlink()

class LogReader:
    """Log reading and querying."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
    
    async def query_logs(self,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        level: Optional[str] = None,
                        source: Optional[str] = None,
                        trace_id: Optional[str] = None,
                        pattern: Optional[str] = None,
                        limit: int = 100) -> List[LogEntry]:
        """Query logs with filters."""
        logs = []
        
        try:
            # Find relevant files
            log_files = sorted(
                self.log_dir.glob("log_*.jsonl"),
                key=lambda p: p.name,
                reverse=True
            )
            
            for log_file in log_files:
                if len(logs) >= limit:
                    break

                try:
                    lines = await async_read_lines(log_file)
                except Exception as e:
                    logging.error(f"Failed to read log file {log_file}: {e}")
                    continue

                for line in lines:
                    try:
                        log_dict = json.loads(line)

                        # Apply filters
                        if not self._matches_filters(
                            log_dict,
                            start_time,
                            end_time,
                            level,
                            source,
                            trace_id,
                            pattern
                        ):
                            continue

                        logs.append(LogEntry(
                            timestamp=datetime.fromisoformat(
                                log_dict["timestamp"]
                            ),
                            level=log_dict["level"],
                            message=log_dict["message"],
                            source=log_dict["source"],
                            trace_id=log_dict.get("trace_id"),
                            span_id=log_dict.get("span_id"),
                            parent_id=log_dict.get("parent_id"),
                            context=log_dict.get("context"),
                            tags=log_dict.get("tags")
                        ))

                        if len(logs) >= limit:
                            break

                    except Exception as e:
                        logging.error(f"Log parse failed: {str(e)}")
            
            return logs
            
        except Exception as e:
            logging.error(f"Log query failed: {str(e)}")
            return []
    
    def _matches_filters(self,
                        log_dict: Dict[str, Any],
                        start_time: Optional[datetime],
                        end_time: Optional[datetime],
                        level: Optional[str],
                        source: Optional[str],
                        trace_id: Optional[str],
                        pattern: Optional[str]) -> bool:
        """Check if log matches filters."""
        try:
            # Time filter
            log_time = datetime.fromisoformat(log_dict["timestamp"])
            if start_time and log_time < start_time:
                return False
            if end_time and log_time > end_time:
                return False
            
            # Level filter
            if level and log_dict["level"] != level:
                return False
            
            # Source filter
            if source and log_dict["source"] != source:
                return False
            
            # Trace filter
            if trace_id and log_dict.get("trace_id") != trace_id:
                return False
            
            # Pattern filter
            if pattern and not re.search(pattern, log_dict["message"]):
                return False
            
            return True
            
        except Exception:
            return False

class LoggingService:
    """Advanced logging service."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.writer = LogWriter(log_dir)
        self.reader = LogReader(log_dir)
        self.processor = LogProcessor()
        self.trace_contexts: Dict[str, TraceContext] = {}
    
    @asynccontextmanager
    async def trace_context(self,
                          name: str,
                          parent_context: Optional[TraceContext] = None
                          ) -> AsyncGenerator[TraceContext, None]:
        """Create trace context."""
        if parent_context:
            context = parent_context.child()
        else:
            context = TraceContext()
        
        self.trace_contexts[name] = context
        try:
            yield context
        finally:
            del self.trace_contexts[name]
    
    async def log(self,
                 level: str,
                 message: str,
                 source: str,
                 context: Optional[Dict[str, Any]] = None,
                 tags: Optional[List[str]] = None,
                 trace_name: Optional[str] = None):
        """Log message with context."""
        try:
            # Get trace context
            trace_context = None
            if trace_name and trace_name in self.trace_contexts:
                trace_context = self.trace_contexts[trace_name]
            
            # Create log entry
            log = LogEntry(
                timestamp=datetime.now(),
                level=level.upper(),
                message=message,
                source=source,
                trace_id=trace_context.trace_id if trace_context else None,
                span_id=trace_context.span_id if trace_context else None,
                parent_id=trace_context.parent_id if trace_context else None,
                context=context or {},
                tags=tags or []
            )
            
            # Write log
            await self.writer.write_log(log)
            
        except Exception as e:
            logging.error(f"Logging failed: {str(e)}")
    
    async def query(self,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   level: Optional[str] = None,
                   source: Optional[str] = None,
                   trace_id: Optional[str] = None,
                   pattern: Optional[str] = None,
                   limit: int = 100) -> List[LogEntry]:
        """Query logs."""
        return await self.reader.query_logs(
            start_time=start_time,
            end_time=end_time,
            level=level,
            source=source,
            trace_id=trace_id,
            pattern=pattern,
            limit=limit
        )
    
    async def analyze(self,
                     start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None
                     ) -> Dict[str, Any]:
        """Analyze logs."""
        try:
            # Query logs for analysis
            logs = await self.query(
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            # Process logs
            analysis = self.processor.process_logs(logs)
            
            # Add time range
            if logs:
                analysis["time_range"] = {
                    "start": min(log.timestamp for log in logs).isoformat(),
                    "end": max(log.timestamp for log in logs).isoformat()
                }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Log analysis failed: {str(e)}")
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def flush(self):
        """Flush log buffer."""
        await self.writer.flush()