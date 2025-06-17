"""
CPU Profiler - Phase 5.4

Comprehensive CPU utilization profiling and bottleneck identification for the
Australian Health Analytics platform. Provides detailed analysis of CPU usage
patterns, thread utilization, and performance bottlenecks during data processing.

Key Features:
- Real-time CPU utilization monitoring
- Thread-level performance analysis
- CPU bottleneck identification
- Function-level profiling with cProfile integration
- Performance hotspot detection
"""

import psutil
import time
import threading
import cProfile
import pstats
import io
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
import json
import numpy as np
from contextlib import contextmanager
import sys
import traceback

logger = logging.getLogger(__name__)


@dataclass
class CPUSnapshot:
    """Single CPU usage snapshot."""
    timestamp: float
    total_cpu_percent: float
    per_cpu_percent: List[float]
    user_time: float
    system_time: float
    idle_time: float
    iowait_time: float
    thread_count: int
    context_switches: int
    interrupts: int


@dataclass
class CPUProfile:
    """Complete CPU profiling results."""
    start_time: float
    end_time: float
    duration_seconds: float
    snapshots: List[CPUSnapshot]
    peak_cpu_percent: float
    average_cpu_percent: float
    cpu_efficiency: float
    thread_utilization: Dict[str, float]
    bottlenecks_identified: List[Dict[str, Any]]
    hotspots: List[Dict[str, Any]]
    performance_score: float


@dataclass
class FunctionProfile:
    """Function-level profiling results."""
    function_name: str
    call_count: int
    total_time: float
    cumulative_time: float
    time_per_call: float
    percent_total_time: float
    file_path: str
    line_number: int


@dataclass
class BottleneckAnalysis:
    """CPU bottleneck analysis results."""
    bottleneck_type: str
    severity: str  # low, medium, high, critical
    description: str
    affected_functions: List[str]
    recommendation: str
    impact_score: float


class CPUProfiler:
    """
    Comprehensive CPU profiler for Australian Health Analytics platform.
    Provides detailed CPU usage tracking and performance analysis capabilities.
    """
    
    def __init__(self, 
                 sample_interval: float = 0.5,
                 enable_function_profiling: bool = True,
                 profile_threads: bool = True):
        """
        Initialize CPU profiler.
        
        Args:
            sample_interval: Seconds between CPU samples
            enable_function_profiling: Enable detailed function profiling
            profile_threads: Enable thread-level profiling
        """
        self.sample_interval = sample_interval
        self.enable_function_profiling = enable_function_profiling
        self.profile_threads = profile_threads
        
        self.snapshots: List[CPUSnapshot] = []
        self.start_time: Optional[float] = None
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Function profiling
        self.profiler: Optional[cProfile.Profile] = None
        self.function_stats: Optional[pstats.Stats] = None
        
        # Performance tracking
        self.operation_timings: Dict[str, List[float]] = {}
        
        logger.info(f"CPU profiler initialized with {sample_interval}s interval")
    
    def start_profiling(self) -> None:
        """Start CPU profiling."""
        if self.monitoring_active:
            logger.warning("CPU profiling already active")
            return
        
        self.start_time = time.time()
        self.snapshots.clear()
        self.operation_timings.clear()
        self.monitoring_active = True
        
        # Start function profiling if enabled
        if self.enable_function_profiling:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_cpu)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("CPU profiling started")
    
    def stop_profiling(self) -> CPUProfile:
        """Stop CPU profiling and return results."""
        if not self.monitoring_active:
            logger.warning("CPU profiling not active")
            return self._create_empty_profile()
        
        self.monitoring_active = False
        
        # Stop function profiling
        if self.profiler:
            self.profiler.disable()
            
            # Capture function statistics
            s = io.StringIO()
            self.function_stats = pstats.Stats(self.profiler, stream=s)
            self.function_stats.sort_stats('cumulative')
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        # Analyze CPU usage
        profile = self._analyze_cpu_usage(end_time, duration)
        
        logger.info(f"CPU profiling stopped. Duration: {duration:.2f}s, "
                   f"Peak: {profile.peak_cpu_percent:.1f}%, "
                   f"Average: {profile.average_cpu_percent:.1f}%")
        
        return profile
    
    def _monitor_cpu(self) -> None:
        """Monitor CPU usage in background thread."""
        while self.monitoring_active:
            try:
                snapshot = self._take_cpu_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"CPU monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def _take_cpu_snapshot(self) -> CPUSnapshot:
        """Take a single CPU usage snapshot."""
        # Overall CPU usage
        cpu_percent = psutil.cpu_percent(interval=None)
        per_cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
        
        # CPU times
        cpu_times = psutil.cpu_times()
        
        # System info
        process = psutil.Process()
        thread_count = process.num_threads()
        
        # System-wide context switches and interrupts (if available)
        try:
            ctx_switches = psutil.cpu_stats().ctx_switches
            interrupts = psutil.cpu_stats().interrupts
        except (AttributeError, OSError):
            ctx_switches = 0
            interrupts = 0
        
        return CPUSnapshot(
            timestamp=time.time(),
            total_cpu_percent=cpu_percent,
            per_cpu_percent=per_cpu_percent,
            user_time=cpu_times.user,
            system_time=cpu_times.system,
            idle_time=cpu_times.idle,
            iowait_time=getattr(cpu_times, 'iowait', 0),
            thread_count=thread_count,
            context_switches=ctx_switches,
            interrupts=interrupts
        )
    
    def _analyze_cpu_usage(self, end_time: float, duration: float) -> CPUProfile:
        """Analyze collected CPU usage data."""
        if not self.snapshots:
            return self._create_empty_profile()
        
        # Basic statistics
        cpu_values = [s.total_cpu_percent for s in self.snapshots]
        peak_cpu = max(cpu_values)
        average_cpu = np.mean(cpu_values)
        
        # CPU efficiency analysis
        cpu_efficiency = self._calculate_cpu_efficiency()
        
        # Thread utilization analysis
        thread_utilization = self._analyze_thread_utilization()
        
        # Bottleneck identification
        bottlenecks = self._identify_bottlenecks()
        
        # Hotspot detection
        hotspots = self._detect_hotspots()
        
        # Performance score calculation
        performance_score = self._calculate_performance_score(
            average_cpu, cpu_efficiency, bottlenecks, hotspots
        )
        
        return CPUProfile(
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=duration,
            snapshots=self.snapshots.copy(),
            peak_cpu_percent=peak_cpu,
            average_cpu_percent=average_cpu,
            cpu_efficiency=cpu_efficiency,
            thread_utilization=thread_utilization,
            bottlenecks_identified=bottlenecks,
            hotspots=hotspots,
            performance_score=performance_score
        )
    
    def _calculate_cpu_efficiency(self) -> float:
        """Calculate CPU efficiency based on usage patterns."""
        if not self.snapshots:
            return 0.0
        
        # Analyze CPU usage distribution
        cpu_values = [s.total_cpu_percent for s in self.snapshots]
        
        # Efficiency factors
        average_usage = np.mean(cpu_values)
        usage_variance = np.var(cpu_values)
        peak_usage = max(cpu_values)
        
        # Efficiency scoring (0-1)
        # Good efficiency: consistent moderate usage, low variance
        usage_efficiency = min(1.0, average_usage / 70.0)  # Optimal around 70%
        variance_penalty = min(0.5, usage_variance / 1000.0)  # Penalty for high variance
        peak_penalty = min(0.3, max(0, peak_usage - 90) / 10.0)  # Penalty for >90% usage
        
        efficiency = max(0.0, usage_efficiency - variance_penalty - peak_penalty)
        
        return efficiency
    
    def _analyze_thread_utilization(self) -> Dict[str, float]:
        """Analyze thread utilization patterns."""
        if not self.snapshots or not self.profile_threads:
            return {}
        
        thread_counts = [s.thread_count for s in self.snapshots]
        
        return {
            'average_thread_count': np.mean(thread_counts),
            'peak_thread_count': max(thread_counts),
            'thread_efficiency': min(1.0, np.mean(thread_counts) / psutil.cpu_count()),
            'thread_stability': 1.0 - (np.std(thread_counts) / max(1, np.mean(thread_counts)))
        }
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify CPU bottlenecks from profiling data."""
        bottlenecks = []
        
        # High CPU usage bottleneck
        if self.snapshots:
            cpu_values = [s.total_cpu_percent for s in self.snapshots]
            high_cpu_periods = [cpu for cpu in cpu_values if cpu > 90]
            
            if len(high_cpu_periods) > len(cpu_values) * 0.2:  # >20% of time
                bottlenecks.append({
                    'type': 'high_cpu_usage',
                    'severity': 'high',
                    'description': f'CPU usage exceeded 90% for {len(high_cpu_periods)} samples ({len(high_cpu_periods)/len(cpu_values):.1%} of time)',
                    'recommendation': 'Consider optimizing CPU-intensive operations or increasing parallelization',
                    'impact_score': len(high_cpu_periods) / len(cpu_values)
                })
        
        # I/O wait bottleneck
        if self.snapshots:
            iowait_values = [s.iowait_time for s in self.snapshots if s.iowait_time > 0]
            if iowait_values and np.mean(iowait_values) > 10:  # >10% I/O wait
                bottlenecks.append({
                    'type': 'io_wait',
                    'severity': 'medium',
                    'description': f'High I/O wait time detected (average: {np.mean(iowait_values):.1f}%)',
                    'recommendation': 'Optimize disk I/O operations, consider SSD storage or async I/O',
                    'impact_score': min(1.0, np.mean(iowait_values) / 50.0)
                })
        
        # Function-level bottlenecks
        if self.function_stats:
            function_bottlenecks = self._analyze_function_bottlenecks()
            bottlenecks.extend(function_bottlenecks)
        
        return bottlenecks
    
    def _analyze_function_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze function-level bottlenecks."""
        if not self.function_stats:
            return []
        
        bottlenecks = []
        
        try:
            # Get top time-consuming functions
            self.function_stats.sort_stats('cumulative')
            stats = self.function_stats.get_stats()
            
            total_time = sum(stat.totaltime for stat in stats.values())
            
            for func_key, stat in list(stats.items())[:10]:  # Top 10 functions
                filename, line_num, func_name = func_key
                
                # Skip built-in functions
                if '<built-in>' in filename or '<frozen' in filename:
                    continue
                
                time_percent = (stat.totaltime / total_time) * 100 if total_time > 0 else 0
                
                # Identify bottleneck functions (>5% of total time)
                if time_percent > 5.0:
                    severity = 'high' if time_percent > 20 else 'medium' if time_percent > 10 else 'low'
                    
                    bottlenecks.append({
                        'type': 'function_bottleneck',
                        'severity': severity,
                        'description': f'Function {func_name} consumes {time_percent:.1f}% of total CPU time',
                        'function_name': func_name,
                        'file_path': filename,
                        'line_number': line_num,
                        'total_time': stat.totaltime,
                        'call_count': stat.callcount,
                        'recommendation': f'Optimize {func_name} function - consider algorithmic improvements or caching',
                        'impact_score': time_percent / 100.0
                    })
        
        except Exception as e:
            logger.warning(f"Function bottleneck analysis failed: {e}")
        
        return bottlenecks
    
    def _detect_hotspots(self) -> List[Dict[str, Any]]:
        """Detect performance hotspots."""
        hotspots = []
        
        # CPU usage hotspots
        if self.snapshots:
            cpu_values = [s.total_cpu_percent for s in self.snapshots]
            
            # Find periods of sustained high CPU usage
            high_cpu_threshold = 80
            current_streak = 0
            max_streak = 0
            streak_start = 0
            
            for i, cpu in enumerate(cpu_values):
                if cpu > high_cpu_threshold:
                    if current_streak == 0:
                        streak_start = i
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    if current_streak > 0:
                        # End of streak
                        if current_streak >= 5:  # Sustained for 5+ samples
                            hotspots.append({
                                'type': 'sustained_high_cpu',
                                'start_sample': streak_start,
                                'duration_samples': current_streak,
                                'peak_cpu': max(cpu_values[streak_start:streak_start + current_streak]),
                                'average_cpu': np.mean(cpu_values[streak_start:streak_start + current_streak])
                            })
                        current_streak = 0
        
        # Function hotspots from profiling
        if self.function_stats:
            function_hotspots = self._identify_function_hotspots()
            hotspots.extend(function_hotspots)
        
        return hotspots
    
    def _identify_function_hotspots(self) -> List[Dict[str, Any]]:
        """Identify function-level performance hotspots."""
        if not self.function_stats:
            return []
        
        hotspots = []
        
        try:
            # Get functions with high call counts
            self.function_stats.sort_stats('ncalls')
            stats = self.function_stats.get_stats()
            
            for func_key, stat in list(stats.items())[:5]:  # Top 5 by call count
                filename, line_num, func_name = func_key
                
                # Skip built-in functions
                if '<built-in>' in filename or '<frozen' in filename:
                    continue
                
                if stat.callcount > 1000:  # High call count hotspot
                    hotspots.append({
                        'type': 'high_call_count',
                        'function_name': func_name,
                        'file_path': filename,
                        'line_number': line_num,
                        'call_count': stat.callcount,
                        'total_time': stat.totaltime,
                        'time_per_call': stat.totaltime / stat.callcount if stat.callcount > 0 else 0
                    })
        
        except Exception as e:
            logger.warning(f"Function hotspot identification failed: {e}")
        
        return hotspots
    
    def _calculate_performance_score(self, 
                                   average_cpu: float, 
                                   cpu_efficiency: float,
                                   bottlenecks: List[Dict[str, Any]],
                                   hotspots: List[Dict[str, Any]]) -> float:
        """Calculate overall performance score (0-10)."""
        
        # Base score from CPU usage (optimal around 50-70%)
        if 50 <= average_cpu <= 70:
            usage_score = 10.0
        elif 30 <= average_cpu < 50 or 70 < average_cpu <= 85:
            usage_score = 8.0
        elif 15 <= average_cpu < 30 or 85 < average_cpu <= 95:
            usage_score = 6.0
        else:
            usage_score = 4.0
        
        # Efficiency bonus
        efficiency_bonus = cpu_efficiency * 2.0
        
        # Bottleneck penalties
        bottleneck_penalty = 0
        for bottleneck in bottlenecks:
            if bottleneck.get('severity') == 'critical':
                bottleneck_penalty += 3.0
            elif bottleneck.get('severity') == 'high':
                bottleneck_penalty += 2.0
            elif bottleneck.get('severity') == 'medium':
                bottleneck_penalty += 1.0
            else:
                bottleneck_penalty += 0.5
        
        # Hotspot penalty
        hotspot_penalty = min(2.0, len(hotspots) * 0.5)
        
        # Calculate final score
        performance_score = max(0.0, min(10.0, 
            usage_score + efficiency_bonus - bottleneck_penalty - hotspot_penalty
        ))
        
        return performance_score
    
    def _create_empty_profile(self) -> CPUProfile:
        """Create empty CPU profile for error cases."""
        return CPUProfile(
            start_time=time.time(),
            end_time=time.time(),
            duration_seconds=0.0,
            snapshots=[],
            peak_cpu_percent=0.0,
            average_cpu_percent=0.0,
            cpu_efficiency=0.0,
            thread_utilization={},
            bottlenecks_identified=[],
            hotspots=[],
            performance_score=0.0
        )
    
    @contextmanager
    def profile_operation(self, operation_name: str = "operation"):
        """Context manager for profiling a specific operation."""
        logger.info(f"Starting CPU profiling for {operation_name}")
        
        self.start_profiling()
        operation_start = time.time()
        
        try:
            yield self
        finally:
            operation_time = time.time() - operation_start
            profile = self.stop_profiling()
            
            # Record operation timing
            if operation_name not in self.operation_timings:
                self.operation_timings[operation_name] = []
            self.operation_timings[operation_name].append(operation_time)
            
            logger.info(f"CPU profiling complete for {operation_name}:")
            logger.info(f"  Duration: {profile.duration_seconds:.2f}s")
            logger.info(f"  Average CPU: {profile.average_cpu_percent:.1f}%")
            logger.info(f"  Peak CPU: {profile.peak_cpu_percent:.1f}%")
            logger.info(f"  Performance Score: {profile.performance_score:.1f}/10")
            logger.info(f"  Bottlenecks: {len(profile.bottlenecks_identified)}")
    
    def get_function_profiles(self, top_n: int = 20) -> List[FunctionProfile]:
        """Get top N function profiles by total time."""
        if not self.function_stats:
            return []
        
        function_profiles = []
        
        try:
            self.function_stats.sort_stats('tottime')
            stats = self.function_stats.get_stats()
            total_time = sum(stat.totaltime for stat in stats.values())
            
            for func_key, stat in list(stats.items())[:top_n]:
                filename, line_num, func_name = func_key
                
                # Skip built-in functions
                if '<built-in>' in filename or '<frozen' in filename:
                    continue
                
                function_profiles.append(FunctionProfile(
                    function_name=func_name,
                    call_count=stat.callcount,
                    total_time=stat.totaltime,
                    cumulative_time=stat.cumtime,
                    time_per_call=stat.totaltime / stat.callcount if stat.callcount > 0 else 0,
                    percent_total_time=(stat.totaltime / total_time) * 100 if total_time > 0 else 0,
                    file_path=filename,
                    line_number=line_num
                ))
        
        except Exception as e:
            logger.error(f"Failed to extract function profiles: {e}")
        
        return function_profiles
    
    def save_profile(self, profile: CPUProfile, output_path: Path) -> None:
        """Save CPU profile to file."""
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
            
            logger.info(f"CPU profile saved to {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to save CPU profile: {e}")
    
    def generate_cpu_report(self, profile: CPUProfile) -> str:
        """Generate a human-readable CPU profiling report."""
        report_lines = [
            "="*80,
            "CPU PROFILING REPORT",
            "="*80,
            f"Duration: {profile.duration_seconds:.2f} seconds",
            f"Samples: {len(profile.snapshots)}",
            "",
            "CPU USAGE STATISTICS:",
            f"  Peak CPU: {profile.peak_cpu_percent:.1f}%",
            f"  Average CPU: {profile.average_cpu_percent:.1f}%",
            f"  CPU Efficiency: {profile.cpu_efficiency:.1%}",
            f"  Performance Score: {profile.performance_score:.1f}/10",
            "",
            "THREAD UTILIZATION:"
        ]
        
        for metric, value in profile.thread_utilization.items():
            report_lines.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
        
        report_lines.extend([
            "",
            "BOTTLENECKS IDENTIFIED:"
        ])
        
        if profile.bottlenecks_identified:
            for i, bottleneck in enumerate(profile.bottlenecks_identified, 1):
                report_lines.extend([
                    f"  {i}. {bottleneck['type'].upper()} ({bottleneck.get('severity', 'unknown')})",
                    f"     {bottleneck.get('description', 'No description')}",
                    f"     Recommendation: {bottleneck.get('recommendation', 'No recommendation')}",
                    ""
                ])
        else:
            report_lines.append("  No significant bottlenecks identified.")
        
        report_lines.extend([
            "",
            "PERFORMANCE HOTSPOTS:"
        ])
        
        if profile.hotspots:
            for i, hotspot in enumerate(profile.hotspots, 1):
                report_lines.append(f"  {i}. {hotspot['type'].upper()}")
                for key, value in hotspot.items():
                    if key != 'type':
                        report_lines.append(f"     {key}: {value}")
                report_lines.append("")
        else:
            report_lines.append("  No performance hotspots detected.")
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)