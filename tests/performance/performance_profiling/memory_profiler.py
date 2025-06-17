"""
Memory Profiler - Phase 5.4

Detailed memory usage profiling and analysis for the Australian Health Analytics platform.
Provides comprehensive memory tracking, leak detection, optimization monitoring, and
memory efficiency analysis for production-scale data processing operations.

Key Features:
- Real-time memory usage tracking
- Memory leak detection algorithms
- Memory optimization effectiveness analysis
- Memory pressure simulation and testing
- Detailed memory allocation profiling
"""

import psutil
import gc
import time
import threading
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
import json
import numpy as np
from contextlib import contextmanager
import tracemalloc
import sys

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot."""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float  # Available system memory
    gc_objects: int  # Number of objects tracked by GC
    gc_collections: Tuple[int, int, int]  # GC collections per generation


@dataclass
class MemoryProfile:
    """Complete memory profiling results."""
    start_time: float
    end_time: float
    duration_seconds: float
    snapshots: List[MemorySnapshot]
    peak_memory_mb: float
    average_memory_mb: float
    memory_growth_mb: float
    memory_growth_rate_mb_per_second: float
    leak_detected: bool
    leak_rate_mb_per_hour: float
    gc_efficiency: float
    optimization_opportunities: List[str]


@dataclass
class MemoryLeakAnalysis:
    """Memory leak detection analysis."""
    leak_detected: bool
    confidence_score: float  # 0-1
    leak_rate_mb_per_hour: float
    trend_analysis: Dict[str, float]
    statistical_significance: float
    leak_sources: List[Dict[str, Any]]


class MemoryProfiler:
    """
    Comprehensive memory profiler for Australian Health Analytics platform.
    Provides detailed memory usage tracking and analysis capabilities.
    """
    
    def __init__(self, 
                 sample_interval: float = 1.0,
                 enable_tracemalloc: bool = True,
                 profile_gc: bool = True):
        """
        Initialize memory profiler.
        
        Args:
            sample_interval: Seconds between memory samples
            enable_tracemalloc: Enable detailed memory allocation tracking
            profile_gc: Enable garbage collection profiling
        """
        self.sample_interval = sample_interval
        self.enable_tracemalloc = enable_tracemalloc
        self.profile_gc = profile_gc
        
        self.snapshots: List[MemorySnapshot] = []
        self.start_time: Optional[float] = None
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Tracemalloc state
        self.tracemalloc_started = False
        self.initial_tracemalloc_snapshot = None
        
        # GC tracking
        self.initial_gc_stats = None
        
        logger.info(f"Memory profiler initialized with {sample_interval}s interval")
    
    def start_profiling(self) -> None:
        """Start memory profiling."""
        if self.monitoring_active:
            logger.warning("Memory profiling already active")
            return
        
        self.start_time = time.time()
        self.snapshots.clear()
        self.monitoring_active = True
        
        # Enable tracemalloc if requested
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            self.tracemalloc_started = True
            self.initial_tracemalloc_snapshot = tracemalloc.take_snapshot()
        
        # Record initial GC stats
        if self.profile_gc:
            self.initial_gc_stats = gc.get_stats()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_memory)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("Memory profiling started")
    
    def stop_profiling(self) -> MemoryProfile:
        """Stop memory profiling and return results."""
        if not self.monitoring_active:
            logger.warning("Memory profiling not active")
            return self._create_empty_profile()
        
        self.monitoring_active = False
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        # Analyze memory usage
        profile = self._analyze_memory_usage(end_time, duration)
        
        # Clean up tracemalloc if we started it
        if self.tracemalloc_started and tracemalloc.is_tracing():
            tracemalloc.stop()
            self.tracemalloc_started = False
        
        logger.info(f"Memory profiling stopped. Duration: {duration:.2f}s, "
                   f"Peak: {profile.peak_memory_mb:.1f}MB, "
                   f"Growth: {profile.memory_growth_mb:.1f}MB")
        
        return profile
    
    def _monitor_memory(self) -> None:
        """Monitor memory usage in background thread."""
        while self.monitoring_active:
            try:
                snapshot = self._take_memory_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a single memory usage snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # System memory info
        system_memory = psutil.virtual_memory()
        
        # GC information
        gc_objects = len(gc.get_objects()) if self.profile_gc else 0
        gc_collections = tuple(gc.get_count()) if self.profile_gc else (0, 0, 0)
        
        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=memory_percent,
            available_mb=system_memory.available / 1024 / 1024,
            gc_objects=gc_objects,
            gc_collections=gc_collections
        )
    
    def _analyze_memory_usage(self, end_time: float, duration: float) -> MemoryProfile:
        """Analyze collected memory usage data."""
        if not self.snapshots:
            return self._create_empty_profile()
        
        # Basic statistics
        memory_values = [s.rss_mb for s in self.snapshots]
        peak_memory = max(memory_values)
        average_memory = np.mean(memory_values)
        memory_growth = memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
        memory_growth_rate = memory_growth / (duration / 3600) if duration > 0 else 0  # MB/hour
        
        # Leak detection
        leak_analysis = self._detect_memory_leak()
        
        # GC efficiency analysis
        gc_efficiency = self._analyze_gc_efficiency()
        
        # Optimization opportunities
        optimization_opportunities = self._identify_optimization_opportunities()
        
        return MemoryProfile(
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=duration,
            snapshots=self.snapshots.copy(),
            peak_memory_mb=peak_memory,
            average_memory_mb=average_memory,
            memory_growth_mb=memory_growth,
            memory_growth_rate_mb_per_second=memory_growth_rate / 3600 if memory_growth_rate else 0,
            leak_detected=leak_analysis.leak_detected,
            leak_rate_mb_per_hour=leak_analysis.leak_rate_mb_per_hour,
            gc_efficiency=gc_efficiency,
            optimization_opportunities=optimization_opportunities
        )
    
    def _detect_memory_leak(self) -> MemoryLeakAnalysis:
        """Detect memory leaks using statistical analysis."""
        if len(self.snapshots) < 10:
            return MemoryLeakAnalysis(
                leak_detected=False,
                confidence_score=0.0,
                leak_rate_mb_per_hour=0.0,
                trend_analysis={},
                statistical_significance=0.0,
                leak_sources=[]
            )
        
        # Extract time series data
        times = [(s.timestamp - self.snapshots[0].timestamp) / 3600 for s in self.snapshots]  # Hours
        memories = [s.rss_mb for s in self.snapshots]
        
        # Linear regression to detect trend
        time_array = np.array(times)
        memory_array = np.array(memories)
        
        # Calculate slope (memory growth rate)
        if len(time_array) > 1:
            slope, intercept = np.polyfit(time_array, memory_array, 1)
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(time_array, memory_array)[0, 1]
            
            # Statistical significance test
            n = len(time_array)
            t_statistic = correlation * np.sqrt((n - 2) / (1 - correlation**2)) if correlation != 1 else 0
            statistical_significance = min(1.0, abs(t_statistic) / 3.0)  # Normalized
            
            # Confidence scoring
            confidence_factors = [
                min(1.0, abs(slope) / 10.0),  # Slope magnitude
                min(1.0, abs(correlation)),   # Correlation strength
                min(1.0, statistical_significance),  # Statistical significance
                min(1.0, n / 50.0)  # Sample size adequacy
            ]
            confidence_score = np.mean(confidence_factors)
            
            # Leak detection criteria
            leak_detected = (
                slope > 1.0 and  # Growing at >1MB/hour
                correlation > 0.7 and  # Strong positive correlation
                statistical_significance > 0.5 and  # Statistically significant
                confidence_score > 0.6  # High confidence
            )
            
            leak_sources = self._identify_leak_sources() if leak_detected else []
            
            return MemoryLeakAnalysis(
                leak_detected=leak_detected,
                confidence_score=confidence_score,
                leak_rate_mb_per_hour=slope,
                trend_analysis={
                    'slope_mb_per_hour': slope,
                    'intercept_mb': intercept,
                    'correlation': correlation,
                    'r_squared': correlation**2
                },
                statistical_significance=statistical_significance,
                leak_sources=leak_sources
            )
        
        return MemoryLeakAnalysis(
            leak_detected=False,
            confidence_score=0.0,
            leak_rate_mb_per_hour=0.0,
            trend_analysis={},
            statistical_significance=0.0,
            leak_sources=[]
        )
    
    def _identify_leak_sources(self) -> List[Dict[str, Any]]:
        """Identify potential memory leak sources using tracemalloc."""
        leak_sources = []
        
        if not self.tracemalloc_started or not tracemalloc.is_tracing():
            return leak_sources
        
        try:
            # Get current snapshot
            current_snapshot = tracemalloc.take_snapshot()
            
            if self.initial_tracemalloc_snapshot:
                # Compare with initial snapshot
                top_stats = current_snapshot.compare_to(
                    self.initial_tracemalloc_snapshot, 'lineno'
                )
                
                # Analyze top memory allocations
                for index, stat in enumerate(top_stats[:10]):
                    if stat.size_diff > 1024 * 1024:  # >1MB difference
                        leak_sources.append({
                            'rank': index + 1,
                            'size_diff_mb': stat.size_diff / 1024 / 1024,
                            'count_diff': stat.count_diff,
                            'traceback': str(stat.traceback)
                        })
        
        except Exception as e:
            logger.warning(f"Failed to identify leak sources: {e}")
        
        return leak_sources
    
    def _analyze_gc_efficiency(self) -> float:
        """Analyze garbage collection efficiency."""
        if not self.profile_gc or len(self.snapshots) < 2:
            return 1.0  # Assume efficient if no data
        
        try:
            # Calculate GC object growth rate
            initial_objects = self.snapshots[0].gc_objects
            final_objects = self.snapshots[-1].gc_objects
            
            if initial_objects > 0:
                object_growth_rate = (final_objects - initial_objects) / initial_objects
                
                # GC efficiency based on object growth control
                gc_efficiency = max(0.0, 1.0 - min(1.0, object_growth_rate))
                
                return gc_efficiency
        
        except Exception as e:
            logger.warning(f"GC efficiency analysis failed: {e}")
        
        return 1.0
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify memory optimization opportunities."""
        opportunities = []
        
        if not self.snapshots:
            return opportunities
        
        # Analyze memory patterns
        memory_values = [s.rss_mb for s in self.snapshots]
        peak_memory = max(memory_values)
        average_memory = np.mean(memory_values)
        memory_volatility = np.std(memory_values)
        
        # High memory usage
        if peak_memory > 1024:  # >1GB
            opportunities.append(f"High memory usage detected ({peak_memory:.1f}MB peak). Consider data streaming or chunking.")
        
        # High memory volatility
        if memory_volatility > average_memory * 0.3:
            opportunities.append(f"High memory volatility ({memory_volatility:.1f}MB std). Consider memory pooling or caching strategies.")
        
        # GC pressure
        if self.profile_gc and len(self.snapshots) > 1:
            gc_growth = self.snapshots[-1].gc_objects - self.snapshots[0].gc_objects
            if gc_growth > 100000:  # >100k new objects
                opportunities.append(f"High GC pressure ({gc_growth:,} new objects). Consider object reuse or lazy loading.")
        
        # Memory growth
        if len(memory_values) > 1:
            growth = memory_values[-1] - memory_values[0]
            if growth > 100:  # >100MB growth
                opportunities.append(f"Significant memory growth ({growth:.1f}MB). Review data retention and cleanup policies.")
        
        return opportunities
    
    def _create_empty_profile(self) -> MemoryProfile:
        """Create empty memory profile for error cases."""
        return MemoryProfile(
            start_time=time.time(),
            end_time=time.time(),
            duration_seconds=0.0,
            snapshots=[],
            peak_memory_mb=0.0,
            average_memory_mb=0.0,
            memory_growth_mb=0.0,
            memory_growth_rate_mb_per_second=0.0,
            leak_detected=False,
            leak_rate_mb_per_hour=0.0,
            gc_efficiency=1.0,
            optimization_opportunities=[]
        )
    
    @contextmanager
    def profile_operation(self, operation_name: str = "operation"):
        """Context manager for profiling a specific operation."""
        logger.info(f"Starting memory profiling for {operation_name}")
        
        self.start_profiling()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            yield self
        finally:
            profile = self.stop_profiling()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info(f"Memory profiling complete for {operation_name}:")
            logger.info(f"  Duration: {profile.duration_seconds:.2f}s")
            logger.info(f"  Memory delta: {end_memory - start_memory:.1f}MB")
            logger.info(f"  Peak memory: {profile.peak_memory_mb:.1f}MB")
            logger.info(f"  Leak detected: {profile.leak_detected}")
            
            if profile.optimization_opportunities:
                logger.info("  Optimization opportunities:")
                for opportunity in profile.optimization_opportunities:
                    logger.info(f"    - {opportunity}")
    
    def save_profile(self, profile: MemoryProfile, output_path: Path) -> None:
        """Save memory profile to file."""
        try:
            # Convert to serializable format
            profile_data = asdict(profile)
            
            # Convert numpy types to Python types
            def convert_numpy(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            # Apply conversion recursively
            import json
            profile_json = json.loads(json.dumps(profile_data, default=convert_numpy))
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(profile_json, f, indent=2)
            
            logger.info(f"Memory profile saved to {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to save memory profile: {e}")
    
    def generate_memory_report(self, profile: MemoryProfile) -> str:
        """Generate a human-readable memory profiling report."""
        report_lines = [
            "="*80,
            "MEMORY PROFILING REPORT",
            "="*80,
            f"Duration: {profile.duration_seconds:.2f} seconds",
            f"Samples: {len(profile.snapshots)}",
            "",
            "MEMORY USAGE STATISTICS:",
            f"  Peak Memory: {profile.peak_memory_mb:.1f} MB",
            f"  Average Memory: {profile.average_memory_mb:.1f} MB",
            f"  Memory Growth: {profile.memory_growth_mb:.1f} MB",
            f"  Growth Rate: {profile.memory_growth_rate_mb_per_second*3600:.2f} MB/hour",
            "",
            "MEMORY LEAK ANALYSIS:",
            f"  Leak Detected: {'YES' if profile.leak_detected else 'NO'}",
            f"  Leak Rate: {profile.leak_rate_mb_per_hour:.2f} MB/hour",
            "",
            "GARBAGE COLLECTION:",
            f"  GC Efficiency: {profile.gc_efficiency:.1%}",
            "",
            "OPTIMIZATION OPPORTUNITIES:"
        ]
        
        if profile.optimization_opportunities:
            for i, opportunity in enumerate(profile.optimization_opportunities, 1):
                report_lines.append(f"  {i}. {opportunity}")
        else:
            report_lines.append("  No optimization opportunities identified.")
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)