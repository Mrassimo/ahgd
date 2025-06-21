"""
Performance profiling utilities for the AHGD ETL pipeline.

This module provides comprehensive profiling capabilities including:
- Memory usage profiling with memory_profiler and tracemalloc
- CPU profiling with cProfile and line_profiler
- I/O performance monitoring
- Database query performance tracking
- Custom performance metrics collection
"""

import asyncio
import cProfile
import functools
import gc
import io
import os
import pstats
import psutil
import sqlite3
import sys
import time
import tracemalloc
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import warnings

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    warnings.warn("memory_profiler not available. Install with: pip install memory-profiler")

try:
    import line_profiler
    LINE_PROFILER_AVAILABLE = True
except ImportError:
    LINE_PROFILER_AVAILABLE = False
    warnings.warn("line_profiler not available. Install with: pip install line-profiler")

from ..utils.logging import get_logger, track_lineage
from ..utils.interfaces import ProcessingMetadata, ProcessingStatus

logger = get_logger()


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    timestamp: datetime
    peak_memory_mb: float
    current_memory_mb: float
    memory_percent: float
    gc_stats: Dict[str, Any]
    process_memory: Dict[str, Any]
    top_allocations: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'peak_memory_mb': self.peak_memory_mb,
            'current_memory_mb': self.current_memory_mb,
            'memory_percent': self.memory_percent,
            'gc_stats': self.gc_stats,
            'process_memory': self.process_memory,
            'top_allocations': self.top_allocations
        }


@dataclass
class CPUSnapshot:
    """Snapshot of CPU usage at a point in time."""
    timestamp: datetime
    cpu_percent: float
    cpu_times: Dict[str, float]
    load_average: Tuple[float, float, float]
    thread_count: int
    context_switches: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'cpu_times': self.cpu_times,
            'load_average': list(self.load_average),
            'thread_count': self.thread_count,
            'context_switches': self.context_switches
        }


@dataclass
class IOSnapshot:
    """Snapshot of I/O performance at a point in time."""
    timestamp: datetime
    read_count: int
    write_count: int
    read_bytes: int
    write_bytes: int
    read_time: float
    write_time: float
    disk_usage: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'read_count': self.read_count,
            'write_count': self.write_count,
            'read_bytes': self.read_bytes,
            'write_bytes': self.write_bytes,
            'read_time': self.read_time,
            'write_time': self.write_time,
            'disk_usage': self.disk_usage
        }


@dataclass
class QueryProfile:
    """Profile information for a database query."""
    query_id: str
    query_text: str
    execution_time: float
    rows_affected: int
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    explain_plan: Optional[str] = None
    index_usage: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query_id': self.query_id,
            'query_text': self.query_text,
            'execution_time': self.execution_time,
            'rows_affected': self.rows_affected,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'explain_plan': self.explain_plan,
            'index_usage': self.index_usage
        }

class MemoryProfiler:
    """
    Memory profiling utilities for tracking memory usage during ETL operations.
    
    Features:
    - Real-time memory monitoring with tracemalloc
    - Peak memory usage tracking
    - Memory allocation hotspots identification
    - Memory leak detection
    - Process memory statistics
    - Garbage collection monitoring
    """
    
    def __init__(self, enable_tracemalloc: bool = True, max_snapshots: int = 1000):
        self.enable_tracemalloc = enable_tracemalloc
        self.max_snapshots = max_snapshots
        self.snapshots = deque(maxlen=max_snapshots)
        self.baseline_snapshot = None
        self.is_profiling = False
        self.process = psutil.Process()
        
        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def start_profiling(self):
        """Start memory profiling."""
        self.is_profiling = True
        self.baseline_snapshot = self._take_snapshot()
        logger.info("Memory profiling started", baseline_memory=self.baseline_snapshot.current_memory_mb)
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop memory profiling and return summary."""
        if not self.is_profiling:
            return {}
        
        self.is_profiling = False
        final_snapshot = self._take_snapshot()
        
        summary = self._generate_summary(final_snapshot)
        logger.info("Memory profiling stopped", 
                   final_memory=final_snapshot.current_memory_mb,
                   peak_memory=final_snapshot.peak_memory_mb,
                   memory_growth=final_snapshot.current_memory_mb - self.baseline_snapshot.current_memory_mb)
        
        return summary
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        current_time = datetime.now(timezone.utc)
        
        # Get process memory info
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # Get GC statistics
        gc_stats = {
            'generation_0': gc.get_count()[0],
            'generation_1': gc.get_count()[1],
            'generation_2': gc.get_count()[2],
            'total_collections': sum(gc.get_stats()),
            'uncollectable': len(gc.garbage)
        }
        
        # Get tracemalloc stats if available
        top_allocations = []
        current_memory_mb = memory_info.rss / 1024 / 1024
        peak_memory_mb = memory_info.peak_wset / 1024 / 1024 if hasattr(memory_info, 'peak_wset') else current_memory_mb
        
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            current_memory_mb = current / 1024 / 1024
            peak_memory_mb = peak / 1024 / 1024
            
            # Get top memory allocations
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            for stat in top_stats:
                top_allocations.append({
                    'filename': stat.traceback.format()[-1],
                    'size_mb': stat.size / 1024 / 1024,
                    'count': stat.count
                })
        
        snapshot = MemorySnapshot(
            timestamp=current_time,
            peak_memory_mb=peak_memory_mb,
            current_memory_mb=current_memory_mb,
            memory_percent=memory_percent,
            gc_stats=gc_stats,
            process_memory={
                'rss': memory_info.rss,
                'vms': memory_info.vms,
                'shared': getattr(memory_info, 'shared', 0),
                'text': getattr(memory_info, 'text', 0),
                'data': getattr(memory_info, 'data', 0)
            },
            top_allocations=top_allocations
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def _generate_summary(self, final_snapshot: MemorySnapshot) -> Dict[str, Any]:
        """Generate memory profiling summary."""
        if not self.baseline_snapshot:
            return {}
        
        memory_growth = final_snapshot.current_memory_mb - self.baseline_snapshot.current_memory_mb
        peak_memory = max(s.peak_memory_mb for s in self.snapshots)
        
        return {
            'baseline_memory_mb': self.baseline_snapshot.current_memory_mb,
            'final_memory_mb': final_snapshot.current_memory_mb,
            'peak_memory_mb': peak_memory,
            'memory_growth_mb': memory_growth,
            'memory_growth_percent': (memory_growth / self.baseline_snapshot.current_memory_mb) * 100,
            'snapshots_count': len(self.snapshots),
            'gc_collections': final_snapshot.gc_stats,
            'top_allocations': final_snapshot.top_allocations[:5]
        }
    
    @contextmanager
    def profile_memory(self, operation_name: str):
        """Context manager for memory profiling."""
        self.start_profiling()
        operation_start = time.time()
        
        try:
            yield self
        finally:
            summary = self.stop_profiling()
            operation_time = time.time() - operation_start
            
            # Log memory usage summary
            logger.info(f"Memory profile for {operation_name}",
                       operation_time=operation_time,
                       memory_summary=summary)
            
            # Track in data lineage
            track_lineage(
                f"memory_profile_{operation_name}",
                f"memory_summary_{operation_name}",
                "memory_profiling",
                memory_summary=summary
            )


class CPUProfiler:
    """
    CPU profiling utilities for tracking CPU usage and performance bottlenecks.
    
    Features:
    - CPU usage monitoring
    - Function-level profiling with cProfile
    - Line-by-line profiling support
    - Thread monitoring
    - Load average tracking
    - Context switch monitoring
    """
    
    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self.snapshots = deque(maxlen=max_snapshots)
        self.profiler = None
        self.is_profiling = False
        self.process = psutil.Process()
        self.baseline_snapshot = None
    
    def start_profiling(self):
        """Start CPU profiling."""
        self.is_profiling = True
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.baseline_snapshot = self._take_snapshot()
        logger.info("CPU profiling started")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop CPU profiling and return summary."""
        if not self.is_profiling or not self.profiler:
            return {}
        
        self.profiler.disable()
        self.is_profiling = False
        final_snapshot = self._take_snapshot()
        
        # Generate profile statistics
        stats_stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_stream)
        stats.sort_stats('cumulative')
        stats.print_stats(20)
        
        summary = self._generate_summary(final_snapshot, stats_stream.getvalue())
        logger.info("CPU profiling stopped", cpu_summary=summary)
        
        return summary    
    def _take_snapshot(self) -> CPUSnapshot:
        """Take a CPU snapshot."""
        current_time = datetime.now(timezone.utc)
        
        # Get CPU information
        cpu_percent = self.process.cpu_percent()
        cpu_times = self.process.cpu_times()._asdict()
        
        # Get system load average (Unix only)
        try:
            load_average = os.getloadavg()
        except AttributeError:
            load_average = (0.0, 0.0, 0.0)  # Windows doesn't have getloadavg
        
        # Get thread and context switch information
        num_threads = self.process.num_threads()
        try:
            ctx_switches = self.process.num_ctx_switches().voluntary
        except AttributeError:
            ctx_switches = 0
        
        snapshot = CPUSnapshot(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            cpu_times=cpu_times,
            load_average=load_average,
            thread_count=num_threads,
            context_switches=ctx_switches
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def _generate_summary(self, final_snapshot: CPUSnapshot, profile_stats: str) -> Dict[str, Any]:
        """Generate CPU profiling summary."""
        if not self.baseline_snapshot:
            return {}
        
        return {
            'cpu_usage_percent': final_snapshot.cpu_percent,
            'load_average': final_snapshot.load_average,
            'thread_count': final_snapshot.thread_count,
            'context_switches': final_snapshot.context_switches,
            'cpu_times': final_snapshot.cpu_times,
            'profile_stats': profile_stats[:2000],  # Truncate for storage
            'snapshots_count': len(self.snapshots)
        }
    
    @contextmanager
    def profile_cpu(self, operation_name: str):
        """Context manager for CPU profiling."""
        self.start_profiling()
        operation_start = time.time()
        
        try:
            yield self
        finally:
            summary = self.stop_profiling()
            operation_time = time.time() - operation_start
            
            # Log CPU usage summary
            logger.info(f"CPU profile for {operation_name}",
                       operation_time=operation_time,
                       cpu_summary=summary)
            
            # Track in data lineage
            track_lineage(
                f"cpu_profile_{operation_name}",
                f"cpu_summary_{operation_name}",
                "cpu_profiling",
                cpu_summary=summary
            )


class IOProfiler:
    """
    I/O performance monitoring for tracking disk and network operations.
    
    Features:
    - Disk I/O monitoring
    - Network I/O tracking
    - File operation profiling
    - Disk usage monitoring
    - I/O latency measurement
    """
    
    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self.snapshots = deque(maxlen=max_snapshots)
        self.is_profiling = False
        self.process = psutil.Process()
        self.baseline_snapshot = None
        self.operation_timings = []
    
    def start_profiling(self):
        """Start I/O profiling."""
        self.is_profiling = True
        self.baseline_snapshot = self._take_snapshot()
        logger.info("I/O profiling started")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop I/O profiling and return summary."""
        if not self.is_profiling:
            return {}
        
        self.is_profiling = False
        final_snapshot = self._take_snapshot()
        
        summary = self._generate_summary(final_snapshot)
        logger.info("I/O profiling stopped", io_summary=summary)
        
        return summary
    
    def _take_snapshot(self) -> IOSnapshot:
        """Take an I/O snapshot."""
        current_time = datetime.now(timezone.utc)
        
        # Get I/O counters
        io_counters = self.process.io_counters()
        
        # Get disk usage
        disk_usage = {}
        try:
            for disk in psutil.disk_partitions():
                if disk.fstype:  # Skip empty partitions
                    usage = psutil.disk_usage(disk.mountpoint)
                    disk_usage[disk.device] = {
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': (usage.used / usage.total) * 100
                    }
        except (PermissionError, OSError):
            pass  # Skip inaccessible partitions
        
        snapshot = IOSnapshot(
            timestamp=current_time,
            read_count=io_counters.read_count,
            write_count=io_counters.write_count,
            read_bytes=io_counters.read_bytes,
            write_bytes=io_counters.write_bytes,
            read_time=getattr(io_counters, 'read_time', 0),
            write_time=getattr(io_counters, 'write_time', 0),
            disk_usage=disk_usage
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def _generate_summary(self, final_snapshot: IOSnapshot) -> Dict[str, Any]:
        """Generate I/O profiling summary."""
        if not self.baseline_snapshot:
            return {}
        
        read_ops = final_snapshot.read_count - self.baseline_snapshot.read_count
        write_ops = final_snapshot.write_count - self.baseline_snapshot.write_count
        read_bytes = final_snapshot.read_bytes - self.baseline_snapshot.read_bytes
        write_bytes = final_snapshot.write_bytes - self.baseline_snapshot.write_bytes
        
        return {
            'read_operations': read_ops,
            'write_operations': write_ops,
            'bytes_read': read_bytes,
            'bytes_written': write_bytes,
            'read_mb': read_bytes / 1024 / 1024,
            'write_mb': write_bytes / 1024 / 1024,
            'total_io_mb': (read_bytes + write_bytes) / 1024 / 1024,
            'disk_usage': final_snapshot.disk_usage,
            'snapshots_count': len(self.snapshots)
        }
    
    @contextmanager
    def profile_io(self, operation_name: str):
        """Context manager for I/O profiling."""
        self.start_profiling()
        operation_start = time.time()
        
        try:
            yield self
        finally:
            summary = self.stop_profiling()
            operation_time = time.time() - operation_start
            
            # Log I/O usage summary
            logger.info(f"I/O profile for {operation_name}",
                       operation_time=operation_time,
                       io_summary=summary)
            
            # Track in data lineage
            track_lineage(
                f"io_profile_{operation_name}",
                f"io_summary_{operation_name}",
                "io_profiling",
                io_summary=summary
            )
class QueryProfiler:
    """
    Database query performance profiling for tracking SQL execution performance.
    
    Features:
    - Query execution time tracking
    - Parameter binding analysis
    - Query plan analysis
    - Index usage monitoring
    - Slow query detection
    - Connection pooling metrics
    """
    
    def __init__(self, slow_query_threshold: float = 1.0):
        self.slow_query_threshold = slow_query_threshold
        self.query_profiles = []
        self.is_profiling = False
        self.connection_stats = defaultdict(int)
    
    def start_profiling(self):
        """Start query profiling."""
        self.is_profiling = True
        self.query_profiles.clear()
        self.connection_stats.clear()
        logger.info("Query profiling started")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop query profiling and return summary."""
        if not self.is_profiling:
            return {}
        
        self.is_profiling = False
        summary = self._generate_summary()
        logger.info("Query profiling stopped", query_summary=summary)
        
        return summary
    
    def profile_query(self, query_text: str, parameters: Dict[str, Any] = None) -> Callable:
        """Decorator for profiling individual queries."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_profiling:
                    return func(*args, **kwargs)
                
                query_id = f"{func.__name__}_{id(query_text)}"
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    # Determine rows affected (heuristic)
                    rows_affected = 0
                    if hasattr(result, 'rowcount'):
                        rows_affected = result.rowcount
                    elif isinstance(result, (list, tuple)):
                        rows_affected = len(result)
                    
                    # Create query profile
                    profile = QueryProfile(
                        query_id=query_id,
                        query_text=query_text,
                        execution_time=execution_time,
                        rows_affected=rows_affected,
                        timestamp=datetime.now(timezone.utc),
                        parameters=parameters or {}
                    )
                    
                    self.query_profiles.append(profile)
                    
                    # Log slow queries
                    if execution_time > self.slow_query_threshold:
                        logger.warning(f"Slow query detected: {query_id}",
                                     execution_time=execution_time,
                                     query_text=query_text[:200])
                    
                    return result
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"Query execution failed: {query_id}",
                               execution_time=execution_time,
                               error=str(e))
                    raise
            
            return wrapper
        return decorator
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate query profiling summary."""
        if not self.query_profiles:
            return {}
        
        total_queries = len(self.query_profiles)
        total_time = sum(q.execution_time for q in self.query_profiles)
        slow_queries = [q for q in self.query_profiles if q.execution_time > self.slow_query_threshold]
        
        # Calculate percentiles
        execution_times = sorted([q.execution_time for q in self.query_profiles])
        percentiles = {}
        for p in [50, 90, 95, 99]:
            idx = int(len(execution_times) * p / 100)
            percentiles[f'p{p}'] = execution_times[idx] if idx < len(execution_times) else 0
        
        return {
            'total_queries': total_queries,
            'total_execution_time': total_time,
            'average_execution_time': total_time / total_queries,
            'slow_queries_count': len(slow_queries),
            'slow_queries_percent': (len(slow_queries) / total_queries) * 100,
            'execution_time_percentiles': percentiles,
            'top_slow_queries': [q.to_dict() for q in sorted(slow_queries, key=lambda x: x.execution_time, reverse=True)[:5]]
        }
    
    @contextmanager
    def profile_queries(self, operation_name: str):
        """Context manager for query profiling."""
        self.start_profiling()
        operation_start = time.time()
        
        try:
            yield self
        finally:
            summary = self.stop_profiling()
            operation_time = time.time() - operation_start
            
            # Log query performance summary
            logger.info(f"Query profile for {operation_name}",
                       operation_time=operation_time,
                       query_summary=summary)
            
            # Track in data lineage
            track_lineage(
                f"query_profile_{operation_name}",
                f"query_summary_{operation_name}",
                "query_profiling",
                query_summary=summary
            )


class PerformanceProfiler:
    """
    Comprehensive performance profiler that combines all profiling capabilities.
    
    Features:
    - Combined memory, CPU, I/O, and query profiling
    - Automatic profiling based on thresholds
    - Performance regression detection
    - Comprehensive reporting
    - Integration with monitoring systems
    """
    
    def __init__(self, 
                 enable_memory: bool = True,
                 enable_cpu: bool = True,
                 enable_io: bool = True,
                 enable_queries: bool = True,
                 slow_query_threshold: float = 1.0):
        self.enable_memory = enable_memory
        self.enable_cpu = enable_cpu
        self.enable_io = enable_io
        self.enable_queries = enable_queries
        
        # Initialize profilers
        self.memory_profiler = MemoryProfiler() if enable_memory else None
        self.cpu_profiler = CPUProfiler() if enable_cpu else None
        self.io_profiler = IOProfiler() if enable_io else None
        self.query_profiler = QueryProfiler(slow_query_threshold) if enable_queries else None
        
        self.is_profiling = False
        self.profile_results = {}
    
    def start_profiling(self):
        """Start all enabled profilers."""
        self.is_profiling = True
        self.profile_results.clear()
        
        if self.memory_profiler:
            self.memory_profiler.start_profiling()
        if self.cpu_profiler:
            self.cpu_profiler.start_profiling()
        if self.io_profiler:
            self.io_profiler.start_profiling()
        if self.query_profiler:
            self.query_profiler.start_profiling()
        
        logger.info("Comprehensive performance profiling started")
    
    def stop_profiling(self) -> Dict[str, Any]:
        """Stop all profilers and return comprehensive summary."""
        if not self.is_profiling:
            return {}
        
        self.is_profiling = False
        
        # Collect results from all profilers
        if self.memory_profiler:
            self.profile_results['memory'] = self.memory_profiler.stop_profiling()
        if self.cpu_profiler:
            self.profile_results['cpu'] = self.cpu_profiler.stop_profiling()
        if self.io_profiler:
            self.profile_results['io'] = self.io_profiler.stop_profiling()
        if self.query_profiler:
            self.profile_results['queries'] = self.query_profiler.stop_profiling()
        
        # Generate comprehensive summary
        summary = self._generate_comprehensive_summary()
        logger.info("Comprehensive performance profiling stopped", 
                   performance_summary=summary)
        
        return summary
    
    def _generate_comprehensive_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary."""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'profiling_enabled': {
                'memory': self.enable_memory,
                'cpu': self.enable_cpu,
                'io': self.enable_io,
                'queries': self.enable_queries
            },
            'results': self.profile_results,
            'performance_score': self._calculate_performance_score()
        }
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        score = 100.0
        
        # Memory score (penalty for high memory usage)
        if 'memory' in self.profile_results:
            memory_growth = self.profile_results['memory'].get('memory_growth_percent', 0)
            if memory_growth > 50:
                score -= 20
            elif memory_growth > 20:
                score -= 10
        
        # CPU score (penalty for high CPU usage)
        if 'cpu' in self.profile_results:
            cpu_usage = self.profile_results['cpu'].get('cpu_usage_percent', 0)
            if cpu_usage > 80:
                score -= 15
            elif cpu_usage > 60:
                score -= 8
        
        # Query score (penalty for slow queries)
        if 'queries' in self.profile_results:
            slow_queries_percent = self.profile_results['queries'].get('slow_queries_percent', 0)
            if slow_queries_percent > 20:
                score -= 25
            elif slow_queries_percent > 10:
                score -= 10
        
        return max(0.0, score)
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for comprehensive operation profiling."""
        self.start_profiling()
        operation_start = time.time()
        
        try:
            yield self
        finally:
            summary = self.stop_profiling()
            operation_time = time.time() - operation_start
            
            # Log comprehensive performance summary
            logger.info(f"Performance profile for {operation_name}",
                       operation_time=operation_time,
                       performance_summary=summary)
            
            # Track in data lineage
            track_lineage(
                f"performance_profile_{operation_name}",
                f"performance_summary_{operation_name}",
                "performance_profiling",
                performance_summary=summary
            )

# Convenience decorator functions
def profile_memory(operation_name: Optional[str] = None):
    """Decorator for memory profiling."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = MemoryProfiler()
            with profiler.profile_memory(op_name):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler = MemoryProfiler()
            with profiler.profile_memory(op_name):
                return await func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


def profile_cpu(operation_name: Optional[str] = None):
    """Decorator for CPU profiling."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = CPUProfiler()
            with profiler.profile_cpu(op_name):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler = CPUProfiler()
            with profiler.profile_cpu(op_name):
                return await func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


def profile_io(operation_name: Optional[str] = None):
    """Decorator for I/O profiling."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = IOProfiler()
            with profiler.profile_io(op_name):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler = IOProfiler()
            with profiler.profile_io(op_name):
                return await func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator


def profile_query(query_text: str, parameters: Dict[str, Any] = None):
    """Decorator for query profiling."""
    def decorator(func: Callable) -> Callable:
        profiler = QueryProfiler()
        return profiler.profile_query(query_text, parameters)(func)
    
    return decorator


def profile_performance(operation_name: Optional[str] = None, 
                       enable_memory: bool = True,
                       enable_cpu: bool = True,
                       enable_io: bool = True,
                       enable_queries: bool = True):
    """Decorator for comprehensive performance profiling."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = PerformanceProfiler(
                enable_memory=enable_memory,
                enable_cpu=enable_cpu,
                enable_io=enable_io,
                enable_queries=enable_queries
            )
            with profiler.profile_operation(op_name):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler = PerformanceProfiler(
                enable_memory=enable_memory,
                enable_cpu=enable_cpu,
                enable_io=enable_io,
                enable_queries=enable_queries
            )
            with profiler.profile_operation(op_name):
                return await func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator