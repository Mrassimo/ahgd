"""
I/O Profiler - Phase 5.4

Comprehensive I/O performance profiling for the Australian Health Analytics platform.
Monitors disk I/O, network I/O, and storage operations to identify performance
bottlenecks and optimization opportunities in data processing workflows.

Key Features:
- Disk I/O monitoring and analysis
- Storage operation performance tracking
- I/O bottleneck identification
- Parquet file I/O optimization analysis
- Network I/O monitoring (if applicable)
"""

import psutil
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
import os
import traceback

logger = logging.getLogger(__name__)


@dataclass
class IOSnapshot:
    """Single I/O usage snapshot."""
    timestamp: float
    disk_read_bytes: int
    disk_write_bytes: int
    disk_read_count: int
    disk_write_count: int
    disk_read_time: float
    disk_write_time: float
    disk_busy_time: float
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    open_files: int


@dataclass
class IOOperation:
    """Single I/O operation tracking."""
    operation_type: str  # read, write, delete, etc.
    file_path: str
    file_size_bytes: int
    start_time: float
    end_time: float
    duration_seconds: float
    throughput_mb_per_second: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class IOProfile:
    """Complete I/O profiling results."""
    start_time: float
    end_time: float
    duration_seconds: float
    snapshots: List[IOSnapshot]
    operations: List[IOOperation]
    total_bytes_read: int
    total_bytes_written: int
    total_read_operations: int
    total_write_operations: int
    average_read_speed_mb_s: float
    average_write_speed_mb_s: float
    peak_read_speed_mb_s: float
    peak_write_speed_mb_s: float
    io_efficiency: float
    bottlenecks_identified: List[Dict[str, Any]]
    optimization_opportunities: List[str]


@dataclass
class StorageAnalysis:
    """Storage performance analysis results."""
    storage_device: str
    total_capacity_gb: float
    used_capacity_gb: float
    free_capacity_gb: float
    utilization_percent: float
    read_iops: float
    write_iops: float
    average_queue_depth: float
    storage_type_detected: str  # SSD, HDD, Network, etc.
    performance_tier: str  # high, medium, low


class IOProfiler:
    """
    Comprehensive I/O profiler for Australian Health Analytics platform.
    Provides detailed I/O performance tracking and analysis capabilities.
    """
    
    def __init__(self, 
                 sample_interval: float = 1.0,
                 track_file_operations: bool = True,
                 monitor_network: bool = True):
        """
        Initialize I/O profiler.
        
        Args:
            sample_interval: Seconds between I/O samples
            track_file_operations: Enable file operation tracking
            monitor_network: Enable network I/O monitoring
        """
        self.sample_interval = sample_interval
        self.track_file_operations = track_file_operations
        self.monitor_network = monitor_network
        
        self.snapshots: List[IOSnapshot] = []
        self.operations: List[IOOperation] = []
        self.start_time: Optional[float] = None
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        # Baseline measurements
        self.baseline_disk_io: Optional[psutil._common.sdiskio] = None
        self.baseline_network_io: Optional[psutil._common.snetio] = None
        
        logger.info(f"I/O profiler initialized with {sample_interval}s interval")
    
    def start_profiling(self) -> None:
        """Start I/O profiling."""
        if self.monitoring_active:
            logger.warning("I/O profiling already active")
            return
        
        self.start_time = time.time()
        self.snapshots.clear()
        self.operations.clear()
        self.monitoring_active = True
        
        # Capture baseline measurements
        try:
            self.baseline_disk_io = psutil.disk_io_counters()
            if self.monitor_network:
                self.baseline_network_io = psutil.net_io_counters()
        except Exception as e:
            logger.warning(f"Failed to capture baseline I/O metrics: {e}")
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_io)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logger.info("I/O profiling started")
    
    def stop_profiling(self) -> IOProfile:
        """Stop I/O profiling and return results."""
        if not self.monitoring_active:
            logger.warning("I/O profiling not active")
            return self._create_empty_profile()
        
        self.monitoring_active = False
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
        
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        # Analyze I/O usage
        profile = self._analyze_io_usage(end_time, duration)
        
        logger.info(f"I/O profiling stopped. Duration: {duration:.2f}s, "
                   f"Read: {profile.average_read_speed_mb_s:.1f}MB/s, "
                   f"Write: {profile.average_write_speed_mb_s:.1f}MB/s")
        
        return profile
    
    def _monitor_io(self) -> None:
        """Monitor I/O usage in background thread."""
        while self.monitoring_active:
            try:
                snapshot = self._take_io_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.error(f"I/O monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def _take_io_snapshot(self) -> IOSnapshot:
        """Take a single I/O usage snapshot."""
        try:
            # Disk I/O counters
            disk_io = psutil.disk_io_counters()
            disk_read_bytes = disk_io.read_bytes if disk_io else 0
            disk_write_bytes = disk_io.write_bytes if disk_io else 0
            disk_read_count = disk_io.read_count if disk_io else 0
            disk_write_count = disk_io.write_count if disk_io else 0
            disk_read_time = disk_io.read_time if disk_io else 0
            disk_write_time = disk_io.write_time if disk_io else 0
            disk_busy_time = getattr(disk_io, 'busy_time', 0) if disk_io else 0
            
            # Network I/O counters (if enabled)
            if self.monitor_network:
                try:
                    net_io = psutil.net_io_counters()
                    network_bytes_sent = net_io.bytes_sent if net_io else 0
                    network_bytes_recv = net_io.bytes_recv if net_io else 0
                    network_packets_sent = net_io.packets_sent if net_io else 0
                    network_packets_recv = net_io.packets_recv if net_io else 0
                except:
                    network_bytes_sent = network_bytes_recv = 0
                    network_packets_sent = network_packets_recv = 0
            else:
                network_bytes_sent = network_bytes_recv = 0
                network_packets_sent = network_packets_recv = 0
            
            # Open files count
            try:
                process = psutil.Process()
                open_files = process.num_fds() if hasattr(process, 'num_fds') else len(process.open_files())
            except:
                open_files = 0
            
            return IOSnapshot(
                timestamp=time.time(),
                disk_read_bytes=disk_read_bytes,
                disk_write_bytes=disk_write_bytes,
                disk_read_count=disk_read_count,
                disk_write_count=disk_write_count,
                disk_read_time=disk_read_time,
                disk_write_time=disk_write_time,
                disk_busy_time=disk_busy_time,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                network_packets_sent=network_packets_sent,
                network_packets_recv=network_packets_recv,
                open_files=open_files
            )
        
        except Exception as e:
            logger.error(f"Failed to take I/O snapshot: {e}")
            return IOSnapshot(
                timestamp=time.time(),
                disk_read_bytes=0, disk_write_bytes=0,
                disk_read_count=0, disk_write_count=0,
                disk_read_time=0, disk_write_time=0, disk_busy_time=0,
                network_bytes_sent=0, network_bytes_recv=0,
                network_packets_sent=0, network_packets_recv=0,
                open_files=0
            )
    
    def track_file_operation(self, 
                           operation_type: str,
                           file_path: Path,
                           file_size_bytes: int = 0) -> Callable:
        """
        Decorator/context manager for tracking file operations.
        
        Args:
            operation_type: Type of operation (read, write, delete, etc.)
            file_path: Path to the file
            file_size_bytes: Size of the file in bytes
        """
        
        @contextmanager
        def operation_tracker():
            if not self.track_file_operations:
                yield
                return
            
            start_time = time.time()
            success = True
            error_message = None
            
            try:
                yield
                
                # If file size not provided, try to get it
                actual_file_size = file_size_bytes
                if actual_file_size == 0 and file_path.exists():
                    try:
                        actual_file_size = file_path.stat().st_size
                    except:
                        pass
                
            except Exception as e:
                success = False
                error_message = str(e)
                actual_file_size = file_size_bytes
                raise
            
            finally:
                end_time = time.time()
                duration = end_time - start_time
                
                # Calculate throughput
                throughput = 0.0
                if duration > 0 and actual_file_size > 0:
                    throughput = (actual_file_size / 1024 / 1024) / duration  # MB/s
                
                # Record operation
                operation = IOOperation(
                    operation_type=operation_type,
                    file_path=str(file_path),
                    file_size_bytes=actual_file_size,
                    start_time=start_time,
                    end_time=end_time,
                    duration_seconds=duration,
                    throughput_mb_per_second=throughput,
                    success=success,
                    error_message=error_message
                )
                
                self.operations.append(operation)
        
        return operation_tracker()
    
    def _analyze_io_usage(self, end_time: float, duration: float) -> IOProfile:
        """Analyze collected I/O usage data."""
        if not self.snapshots:
            return self._create_empty_profile()
        
        # Calculate I/O deltas from baseline
        if len(self.snapshots) >= 2:
            first_snapshot = self.snapshots[0]
            last_snapshot = self.snapshots[-1]
            
            total_bytes_read = last_snapshot.disk_read_bytes - first_snapshot.disk_read_bytes
            total_bytes_written = last_snapshot.disk_write_bytes - first_snapshot.disk_write_bytes
            total_read_ops = last_snapshot.disk_read_count - first_snapshot.disk_read_count
            total_write_ops = last_snapshot.disk_write_count - first_snapshot.disk_write_count
        else:
            total_bytes_read = total_bytes_written = 0
            total_read_ops = total_write_ops = 0
        
        # Calculate average speeds
        avg_read_speed = (total_bytes_read / 1024 / 1024) / duration if duration > 0 else 0
        avg_write_speed = (total_bytes_written / 1024 / 1024) / duration if duration > 0 else 0
        
        # Calculate peak speeds from operations
        read_ops = [op for op in self.operations if 'read' in op.operation_type.lower()]
        write_ops = [op for op in self.operations if 'write' in op.operation_type.lower()]
        
        peak_read_speed = max([op.throughput_mb_per_second for op in read_ops]) if read_ops else 0
        peak_write_speed = max([op.throughput_mb_per_second for op in write_ops]) if write_ops else 0
        
        # I/O efficiency analysis
        io_efficiency = self._calculate_io_efficiency()
        
        # Bottleneck identification
        bottlenecks = self._identify_io_bottlenecks()
        
        # Optimization opportunities
        optimization_opportunities = self._identify_io_optimizations()
        
        return IOProfile(
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=duration,
            snapshots=self.snapshots.copy(),
            operations=self.operations.copy(),
            total_bytes_read=total_bytes_read,
            total_bytes_written=total_bytes_written,
            total_read_operations=total_read_ops,
            total_write_operations=total_write_ops,
            average_read_speed_mb_s=avg_read_speed,
            average_write_speed_mb_s=avg_write_speed,
            peak_read_speed_mb_s=peak_read_speed,
            peak_write_speed_mb_s=peak_write_speed,
            io_efficiency=io_efficiency,
            bottlenecks_identified=bottlenecks,
            optimization_opportunities=optimization_opportunities
        )
    
    def _calculate_io_efficiency(self) -> float:
        """Calculate I/O efficiency based on usage patterns."""
        if not self.snapshots or len(self.snapshots) < 2:
            return 1.0
        
        try:
            # Calculate I/O wait vs busy time ratio
            first_snapshot = self.snapshots[0]
            last_snapshot = self.snapshots[-1]
            
            read_time_delta = last_snapshot.disk_read_time - first_snapshot.disk_read_time
            write_time_delta = last_snapshot.disk_write_time - first_snapshot.disk_write_time
            total_io_time = read_time_delta + write_time_delta
            
            # Efficiency based on successful operations
            successful_ops = [op for op in self.operations if op.success]
            total_ops = len(self.operations)
            
            success_rate = len(successful_ops) / total_ops if total_ops > 0 else 1.0
            
            # Calculate throughput efficiency
            if successful_ops:
                avg_throughput = np.mean([op.throughput_mb_per_second for op in successful_ops])
                # Normalize throughput (100 MB/s = 1.0 efficiency)
                throughput_efficiency = min(1.0, avg_throughput / 100.0)
            else:
                throughput_efficiency = 0.0
            
            # Combined efficiency score
            efficiency = (success_rate * 0.5) + (throughput_efficiency * 0.5)
            
            return efficiency
        
        except Exception as e:
            logger.warning(f"I/O efficiency calculation failed: {e}")
            return 1.0
    
    def _identify_io_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify I/O bottlenecks from profiling data."""
        bottlenecks = []
        
        # Slow file operations
        slow_operations = [op for op in self.operations if op.throughput_mb_per_second < 10.0 and op.file_size_bytes > 1024*1024]
        if slow_operations:
            avg_slow_speed = np.mean([op.throughput_mb_per_second for op in slow_operations])
            bottlenecks.append({
                'type': 'slow_file_operations',
                'severity': 'high' if avg_slow_speed < 5.0 else 'medium',
                'description': f'{len(slow_operations)} file operations with speed <10MB/s (avg: {avg_slow_speed:.1f}MB/s)',
                'affected_operations': len(slow_operations),
                'recommendation': 'Consider using faster storage (SSD), optimizing file sizes, or implementing parallel I/O',
                'impact_score': min(1.0, len(slow_operations) / max(1, len(self.operations)))
            })
        
        # High I/O wait times
        if len(self.snapshots) >= 2:
            read_times = []
            write_times = []
            
            for i in range(1, len(self.snapshots)):
                prev = self.snapshots[i-1]
                curr = self.snapshots[i]
                
                read_time_delta = curr.disk_read_time - prev.disk_read_time
                write_time_delta = curr.disk_write_time - prev.disk_write_time
                
                read_times.append(read_time_delta)
                write_times.append(write_time_delta)
            
            if read_times and np.mean(read_times) > 100:  # >100ms average read time
                bottlenecks.append({
                    'type': 'high_read_latency',
                    'severity': 'medium',
                    'description': f'High disk read latency detected (avg: {np.mean(read_times):.1f}ms)',
                    'recommendation': 'Consider storage optimization, disk defragmentation, or SSD upgrade',
                    'impact_score': min(1.0, np.mean(read_times) / 1000.0)
                })
        
        # Too many open files
        if self.snapshots:
            max_open_files = max([s.open_files for s in self.snapshots])
            if max_open_files > 100:
                bottlenecks.append({
                    'type': 'high_open_file_count',
                    'severity': 'medium' if max_open_files > 200 else 'low',
                    'description': f'High number of open files detected (max: {max_open_files})',
                    'recommendation': 'Implement proper file handle management and close unused files',
                    'impact_score': min(1.0, max_open_files / 500.0)
                })
        
        # Failed operations
        failed_operations = [op for op in self.operations if not op.success]
        if failed_operations:
            failure_rate = len(failed_operations) / len(self.operations)
            if failure_rate > 0.05:  # >5% failure rate
                bottlenecks.append({
                    'type': 'high_io_failure_rate',
                    'severity': 'high' if failure_rate > 0.2 else 'medium',
                    'description': f'High I/O failure rate: {failure_rate:.1%} ({len(failed_operations)} failures)',
                    'recommendation': 'Investigate file system errors, permissions, and storage health',
                    'impact_score': failure_rate
                })
        
        return bottlenecks
    
    def _identify_io_optimizations(self) -> List[str]:
        """Identify I/O optimization opportunities."""
        optimizations = []
        
        if not self.operations:
            return optimizations
        
        # Analyze file operation patterns
        read_operations = [op for op in self.operations if 'read' in op.operation_type.lower()]
        write_operations = [op for op in self.operations if 'write' in op.operation_type.lower()]
        
        # Small file operations
        small_files = [op for op in self.operations if op.file_size_bytes < 1024*1024]  # <1MB
        if len(small_files) > len(self.operations) * 0.5:
            optimizations.append(f"Many small file operations detected ({len(small_files)}). Consider batching or using archive formats.")
        
        # Sequential vs random access patterns
        if read_operations:
            avg_read_throughput = np.mean([op.throughput_mb_per_second for op in read_operations])
            if avg_read_throughput < 50:  # <50MB/s suggests random access
                optimizations.append(f"Low read throughput ({avg_read_throughput:.1f}MB/s) suggests random access. Consider sequential access patterns or caching.")
        
        # Write patterns
        if write_operations:
            write_sizes = [op.file_size_bytes for op in write_operations]
            if write_sizes:
                avg_write_size = np.mean(write_sizes)
                if avg_write_size < 10*1024*1024:  # <10MB average
                    optimizations.append(f"Small average write size ({avg_write_size/1024/1024:.1f}MB). Consider buffering writes or using larger block sizes.")
        
        # Parquet-specific optimizations
        parquet_operations = [op for op in self.operations if '.parquet' in op.file_path.lower()]
        if parquet_operations:
            parquet_read_ops = [op for op in parquet_operations if 'read' in op.operation_type.lower()]
            if parquet_read_ops:
                avg_parquet_speed = np.mean([op.throughput_mb_per_second for op in parquet_read_ops])
                if avg_parquet_speed < 30:  # <30MB/s for Parquet is suboptimal
                    optimizations.append(f"Suboptimal Parquet read performance ({avg_parquet_speed:.1f}MB/s). Consider row group optimization or compression tuning.")
        
        # File handle management
        if self.snapshots:
            max_open_files = max([s.open_files for s in self.snapshots])
            if max_open_files > 50:
                optimizations.append(f"High open file count ({max_open_files}). Implement file handle pooling or lazy loading.")
        
        return optimizations
    
    def analyze_storage_performance(self, storage_path: Path) -> StorageAnalysis:
        """Analyze storage device performance characteristics."""
        try:
            # Get disk usage
            disk_usage = psutil.disk_usage(str(storage_path))
            total_gb = disk_usage.total / 1024**3
            used_gb = disk_usage.used / 1024**3
            free_gb = disk_usage.free / 1024**3
            utilization = (used_gb / total_gb) * 100
            
            # Get I/O statistics for the disk
            disk_io = psutil.disk_io_counters(perdisk=True)
            
            # Try to identify the storage device
            storage_device = "unknown"
            read_iops = write_iops = 0
            avg_queue_depth = 0
            
            # Simple storage type detection based on performance
            # This is a heuristic and may not be 100% accurate
            if self.operations:
                avg_throughput = np.mean([op.throughput_mb_per_second for op in self.operations if op.success])
                
                if avg_throughput > 200:
                    storage_type = "SSD"
                    performance_tier = "high"
                elif avg_throughput > 50:
                    storage_type = "HDD"
                    performance_tier = "medium"
                else:
                    storage_type = "Network/Remote"
                    performance_tier = "low"
            else:
                storage_type = "unknown"
                performance_tier = "unknown"
            
            return StorageAnalysis(
                storage_device=storage_device,
                total_capacity_gb=total_gb,
                used_capacity_gb=used_gb,
                free_capacity_gb=free_gb,
                utilization_percent=utilization,
                read_iops=read_iops,
                write_iops=write_iops,
                average_queue_depth=avg_queue_depth,
                storage_type_detected=storage_type,
                performance_tier=performance_tier
            )
        
        except Exception as e:
            logger.error(f"Storage analysis failed: {e}")
            return StorageAnalysis(
                storage_device="error",
                total_capacity_gb=0, used_capacity_gb=0, free_capacity_gb=0,
                utilization_percent=0, read_iops=0, write_iops=0,
                average_queue_depth=0, storage_type_detected="unknown",
                performance_tier="unknown"
            )
    
    def _create_empty_profile(self) -> IOProfile:
        """Create empty I/O profile for error cases."""
        return IOProfile(
            start_time=time.time(),
            end_time=time.time(),
            duration_seconds=0.0,
            snapshots=[],
            operations=[],
            total_bytes_read=0,
            total_bytes_written=0,
            total_read_operations=0,
            total_write_operations=0,
            average_read_speed_mb_s=0.0,
            average_write_speed_mb_s=0.0,
            peak_read_speed_mb_s=0.0,
            peak_write_speed_mb_s=0.0,
            io_efficiency=0.0,
            bottlenecks_identified=[],
            optimization_opportunities=[]
        )
    
    @contextmanager
    def profile_operation(self, operation_name: str = "operation"):
        """Context manager for profiling I/O during a specific operation."""
        logger.info(f"Starting I/O profiling for {operation_name}")
        
        self.start_profiling()
        
        try:
            yield self
        finally:
            profile = self.stop_profiling()
            
            logger.info(f"I/O profiling complete for {operation_name}:")
            logger.info(f"  Duration: {profile.duration_seconds:.2f}s")
            logger.info(f"  Read: {profile.total_bytes_read / 1024 / 1024:.1f}MB at {profile.average_read_speed_mb_s:.1f}MB/s")
            logger.info(f"  Write: {profile.total_bytes_written / 1024 / 1024:.1f}MB at {profile.average_write_speed_mb_s:.1f}MB/s")
            logger.info(f"  Operations: {len(profile.operations)}")
            logger.info(f"  Bottlenecks: {len(profile.bottlenecks_identified)}")
    
    def save_profile(self, profile: IOProfile, output_path: Path) -> None:
        """Save I/O profile to file."""
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
            
            logger.info(f"I/O profile saved to {output_path}")
        
        except Exception as e:
            logger.error(f"Failed to save I/O profile: {e}")
    
    def generate_io_report(self, profile: IOProfile) -> str:
        """Generate a human-readable I/O profiling report."""
        report_lines = [
            "="*80,
            "I/O PROFILING REPORT",
            "="*80,
            f"Duration: {profile.duration_seconds:.2f} seconds",
            f"Samples: {len(profile.snapshots)}",
            f"Operations: {len(profile.operations)}",
            "",
            "I/O STATISTICS:",
            f"  Total Read: {profile.total_bytes_read / 1024 / 1024:.1f} MB",
            f"  Total Written: {profile.total_bytes_written / 1024 / 1024:.1f} MB",
            f"  Read Operations: {profile.total_read_operations}",
            f"  Write Operations: {profile.total_write_operations}",
            "",
            "PERFORMANCE METRICS:",
            f"  Average Read Speed: {profile.average_read_speed_mb_s:.1f} MB/s",
            f"  Average Write Speed: {profile.average_write_speed_mb_s:.1f} MB/s",
            f"  Peak Read Speed: {profile.peak_read_speed_mb_s:.1f} MB/s",
            f"  Peak Write Speed: {profile.peak_write_speed_mb_s:.1f} MB/s",
            f"  I/O Efficiency: {profile.io_efficiency:.1%}",
            "",
            "BOTTLENECKS IDENTIFIED:"
        ]
        
        if profile.bottlenecks_identified:
            for i, bottleneck in enumerate(profile.bottlenecks_identified, 1):
                report_lines.extend([
                    f"  {i}. {bottleneck['type'].upper()} ({bottleneck.get('severity', 'unknown')})",
                    f"     {bottleneck.get('description', 'No description')}",
                    f"     Recommendation: {bottleneck.get('recommendation', 'No recommendation')}",
                    ""
                ])
        else:
            report_lines.append("  No significant I/O bottlenecks identified.")
        
        report_lines.extend([
            "",
            "OPTIMIZATION OPPORTUNITIES:"
        ])
        
        if profile.optimization_opportunities:
            for i, opportunity in enumerate(profile.optimization_opportunities, 1):
                report_lines.append(f"  {i}. {opportunity}")
        else:
            report_lines.append("  No I/O optimization opportunities identified.")
        
        report_lines.extend([
            "",
            "="*80
        ])
        
        return "\n".join(report_lines)