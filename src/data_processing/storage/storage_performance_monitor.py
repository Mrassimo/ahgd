"""
Storage Performance Monitor - Comprehensive monitoring and benchmarking for Australian health data storage

Provides real-time performance monitoring, benchmarking, and optimization recommendations
for the Parquet-based storage system handling 497,181+ health records.

Key Features:
- Real-time storage performance metrics
- Query performance profiling
- Compression ratio monitoring  
- Memory usage tracking
- Storage optimization recommendations
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
import time
import json
import psutil
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class StorageMetrics:
    """Storage performance metrics data structure."""
    timestamp: str
    file_path: str
    operation_type: str  # read, write, scan
    duration_seconds: float
    file_size_mb: float
    rows_processed: int
    memory_usage_mb: float
    compression_ratio: float
    throughput_mb_per_second: float
    query_complexity: str  # simple, moderate, complex


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    active_queries: int


class StoragePerformanceMonitor:
    """
    Monitor and profile storage performance for Australian health data analytics.
    Provides insights for optimization and capacity planning.
    """
    
    # Performance thresholds and targets
    PERFORMANCE_TARGETS = {
        "query_response_time_seconds": 5.0,      # Max acceptable query time
        "compression_ratio_minimum": 0.3,        # Minimum compression ratio
        "memory_efficiency_ratio": 0.5,          # Max memory vs dataset size ratio
        "throughput_mb_per_second": 50.0,        # Minimum I/O throughput
        "concurrent_queries_limit": 10,          # Max concurrent queries
    }
    
    # Monitoring configuration
    MONITORING_CONFIG = {
        "metrics_retention_hours": 24,           # How long to keep metrics
        "sampling_interval_seconds": 30,        # System metrics sampling interval
        "performance_log_threshold": 2.0,       # Log operations slower than this
        "alert_threshold_memory_percent": 85,   # Memory usage alert threshold
        "alert_threshold_cpu_percent": 90,      # CPU usage alert threshold
    }
    
    def __init__(self, metrics_dir: Optional[Path] = None):
        """Initialize storage performance monitor."""
        self.metrics_dir = metrics_dir or Path("data/metrics")
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics storage
        self.storage_metrics: deque = deque(maxlen=1000)
        self.system_metrics: deque = deque(maxlen=1000)
        self.query_profiles: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.active_operations: Dict[str, float] = {}
        self.performance_alerts: List[Dict[str, Any]] = []
        
        # System monitoring
        self.system_monitor_thread = None
        self.monitoring_active = False
        
        logger.info(f"Initialized storage performance monitor at {self.metrics_dir}")
    
    def start_monitoring(self) -> None:
        """Start continuous system monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.system_monitor_thread = threading.Thread(target=self._system_monitoring_loop, daemon=True)
        self.system_monitor_thread.start()
        
        logger.info("Started continuous performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop continuous system monitoring."""
        self.monitoring_active = False
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=5)
        
        logger.info("Stopped performance monitoring")
    
    def _system_monitoring_loop(self) -> None:
        """Continuous system metrics collection loop."""
        try:
            while self.monitoring_active:
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Check for performance alerts
                self._check_performance_alerts(system_metrics)
                
                time.sleep(self.MONITORING_CONFIG["sampling_interval_seconds"])
                
        except Exception as e:
            logger.error(f"System monitoring loop failed: {e}")
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024 ** 3),
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                active_queries=len(self.active_operations)
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now().isoformat(),
                cpu_percent=0, memory_percent=0, memory_available_gb=0,
                disk_io_read_mb=0, disk_io_write_mb=0, active_queries=0
            )
    
    def _check_performance_alerts(self, metrics: SystemMetrics) -> None:
        """Check for performance alerts and warnings."""
        try:
            alerts = []
            
            # Memory usage alerts
            if metrics.memory_percent > self.MONITORING_CONFIG["alert_threshold_memory_percent"]:
                alerts.append({
                    "type": "memory_high",
                    "severity": "warning",
                    "message": f"High memory usage: {metrics.memory_percent:.1f}%",
                    "timestamp": metrics.timestamp,
                    "value": metrics.memory_percent
                })
            
            # CPU usage alerts
            if metrics.cpu_percent > self.MONITORING_CONFIG["alert_threshold_cpu_percent"]:
                alerts.append({
                    "type": "cpu_high", 
                    "severity": "warning",
                    "message": f"High CPU usage: {metrics.cpu_percent:.1f}%",
                    "timestamp": metrics.timestamp,
                    "value": metrics.cpu_percent
                })
            
            # Active queries alerts
            if metrics.active_queries > self.PERFORMANCE_TARGETS["concurrent_queries_limit"]:
                alerts.append({
                    "type": "concurrent_queries_high",
                    "severity": "info",
                    "message": f"High concurrent queries: {metrics.active_queries}",
                    "timestamp": metrics.timestamp,
                    "value": metrics.active_queries
                })
            
            # Add alerts to queue
            for alert in alerts:
                self.performance_alerts.append(alert)
                if alert["severity"] == "warning":
                    logger.warning(alert["message"])
                    
        except Exception as e:
            logger.error(f"Performance alert check failed: {e}")
    
    def start_operation(self, operation_id: str, operation_type: str, file_path: str) -> None:
        """Start tracking a storage operation."""
        try:
            self.active_operations[operation_id] = {
                "start_time": time.time(),
                "operation_type": operation_type,
                "file_path": file_path,
                "start_memory": psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            }
            
            logger.debug(f"Started tracking operation {operation_id}: {operation_type} on {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to start operation tracking: {e}")
    
    def end_operation(self, 
                     operation_id: str,
                     rows_processed: int = 0,
                     file_size_mb: float = 0,
                     compression_ratio: float = 0) -> Optional[StorageMetrics]:
        """End tracking a storage operation and calculate metrics."""
        try:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found in active operations")
                return None
            
            operation_info = self.active_operations.pop(operation_id)
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - operation_info["start_time"]
            end_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            memory_used = end_memory - operation_info["start_memory"]
            
            # Calculate throughput
            if duration > 0 and file_size_mb > 0:
                throughput = file_size_mb / duration
            else:
                throughput = 0
            
            # Determine query complexity
            complexity = self._determine_query_complexity(duration, rows_processed, file_size_mb)
            
            # Create metrics object
            metrics = StorageMetrics(
                timestamp=datetime.now().isoformat(),
                file_path=operation_info["file_path"],
                operation_type=operation_info["operation_type"],
                duration_seconds=duration,
                file_size_mb=file_size_mb,
                rows_processed=rows_processed,
                memory_usage_mb=memory_used,
                compression_ratio=compression_ratio,
                throughput_mb_per_second=throughput,
                query_complexity=complexity
            )
            
            # Store metrics
            self.storage_metrics.append(metrics)
            
            # Log slow operations
            if duration > self.MONITORING_CONFIG["performance_log_threshold"]:
                logger.warning(f"Slow operation {operation_id}: {duration:.2f}s for {operation_info['operation_type']}")
            
            logger.debug(f"Completed operation {operation_id}: {duration:.2f}s, {throughput:.2f}MB/s")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to end operation tracking: {e}")
            return None
    
    def _determine_query_complexity(self, duration: float, rows: int, size_mb: float) -> str:
        """Determine query complexity based on performance characteristics."""
        try:
            # Simple heuristic based on duration, rows, and size
            if duration < 1.0 and rows < 10000:
                return "simple"
            elif duration < 5.0 and rows < 100000:
                return "moderate"
            else:
                return "complex"
        except:
            return "unknown"
    
    def profile_query_performance(self, 
                                 query_name: str,
                                 lazy_df: pl.LazyFrame,
                                 collect_result: bool = True) -> Dict[str, Any]:
        """Profile performance of a specific Polars query."""
        try:
            operation_id = f"profile_{query_name}_{int(time.time())}"
            
            # Start tracking
            self.start_operation(operation_id, "query_profile", query_name)
            
            # Get query plan
            query_plan = lazy_df.explain()
            plan_complexity = len(query_plan.split('\n'))
            
            # Execute query if requested
            result_df = None
            if collect_result:
                start_time = time.time()
                result_df = lazy_df.collect()
                execution_time = time.time() - start_time
                rows_processed = result_df.shape[0]
                estimated_size_mb = result_df.estimated_size("mb")
            else:
                execution_time = 0
                rows_processed = 0
                estimated_size_mb = 0
            
            # End tracking
            metrics = self.end_operation(operation_id, rows_processed, estimated_size_mb)
            
            # Create profile
            profile = {
                "query_name": query_name,
                "execution_time_seconds": execution_time,
                "query_plan_lines": plan_complexity,
                "rows_processed": rows_processed,
                "estimated_size_mb": estimated_size_mb,
                "query_plan": query_plan,
                "metrics": asdict(metrics) if metrics else None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store profile
            self.query_profiles[query_name] = profile
            
            logger.info(f"Query profile completed for {query_name}: {execution_time:.2f}s, {rows_processed} rows")
            
            return profile
            
        except Exception as e:
            logger.error(f"Query profiling failed for {query_name}: {e}")
            return {"error": str(e)}
    
    def benchmark_storage_operations(self) -> Dict[str, Any]:
        """Comprehensive benchmark of storage operations."""
        try:
            logger.info("Starting comprehensive storage benchmark...")
            
            benchmark_results = {
                "timestamp": datetime.now().isoformat(),
                "read_performance": {},
                "write_performance": {},
                "compression_performance": {},
                "system_capabilities": {}
            }
            
            # System capabilities
            benchmark_results["system_capabilities"] = {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
                "disk_free_gb": psutil.disk_usage(".").free / (1024 ** 3)
            }
            
            # Generate test data for benchmarking
            test_sizes = [1000, 10000, 100000]  # Different dataset sizes
            
            for size in test_sizes:
                test_data = self._generate_benchmark_data(size)
                
                # Write performance test
                write_metrics = self._benchmark_write_performance(test_data, f"benchmark_{size}_rows")
                benchmark_results["write_performance"][f"{size}_rows"] = write_metrics
                
                # Read performance test
                read_metrics = self._benchmark_read_performance(f"benchmark_{size}_rows")
                benchmark_results["read_performance"][f"{size}_rows"] = read_metrics
            
            logger.info("Storage benchmark completed")
            
            # Save benchmark results
            self._save_benchmark_results(benchmark_results)
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Storage benchmark failed: {e}")
            return {"error": str(e)}
    
    def _generate_benchmark_data(self, n_rows: int) -> pl.DataFrame:
        """Generate realistic test data for benchmarking."""
        np.random.seed(42)  # Reproducible benchmarks
        
        return pl.DataFrame({
            "sa2_code": np.random.choice([f"1{str(i).zfill(8)}" for i in range(1000, 3000)], n_rows),
            "state_name": np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], n_rows),
            "risk_score": np.random.uniform(1, 10, n_rows),
            "population": np.random.randint(100, 5000, n_rows),
            "prescription_count": np.random.poisson(3, n_rows),
            "total_cost": np.random.exponential(45, n_rows),
            "date": ["2023-01-01"] * n_rows
        })
    
    def _benchmark_write_performance(self, test_data: pl.DataFrame, filename: str) -> Dict[str, Any]:
        """Benchmark Parquet write performance."""
        try:
            operation_id = f"benchmark_write_{filename}"
            file_path = Path(f"/tmp/{filename}.parquet")
            
            self.start_operation(operation_id, "benchmark_write", str(file_path))
            
            start_time = time.time()
            test_data.write_parquet(file_path, compression="snappy")
            write_time = time.time() - start_time
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            metrics = self.end_operation(operation_id, test_data.shape[0], file_size_mb)
            
            # Cleanup
            if file_path.exists():
                file_path.unlink()
            
            return {
                "write_time_seconds": write_time,
                "file_size_mb": file_size_mb,
                "rows": test_data.shape[0],
                "throughput_mb_per_second": file_size_mb / write_time if write_time > 0 else 0,
                "rows_per_second": test_data.shape[0] / write_time if write_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Write benchmark failed: {e}")
            return {"error": str(e)}
    
    def _benchmark_read_performance(self, filename: str) -> Dict[str, Any]:
        """Benchmark Parquet read performance."""
        try:
            # First create the test file
            test_data = self._generate_benchmark_data(10000)
            file_path = Path(f"/tmp/{filename}_read.parquet")
            test_data.write_parquet(file_path)
            
            operation_id = f"benchmark_read_{filename}"
            
            self.start_operation(operation_id, "benchmark_read", str(file_path))
            
            start_time = time.time()
            result_df = pl.read_parquet(file_path)
            read_time = time.time() - start_time
            
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            metrics = self.end_operation(operation_id, result_df.shape[0], file_size_mb)
            
            # Cleanup
            if file_path.exists():
                file_path.unlink()
            
            return {
                "read_time_seconds": read_time,
                "file_size_mb": file_size_mb,
                "rows": result_df.shape[0],
                "throughput_mb_per_second": file_size_mb / read_time if read_time > 0 else 0,
                "rows_per_second": result_df.shape[0] / read_time if read_time > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Read benchmark failed: {e}")
            return {"error": str(e)}
    
    def _save_benchmark_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        try:
            benchmark_file = self.metrics_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(benchmark_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Benchmark results saved to {benchmark_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save benchmark results: {e}")
    
    def get_performance_summary(self, hours_back: int = 1) -> Dict[str, Any]:
        """Get comprehensive performance summary for specified time period."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Filter recent metrics
            recent_storage_metrics = [
                m for m in self.storage_metrics 
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            recent_system_metrics = [
                m for m in self.system_metrics
                if datetime.fromisoformat(m.timestamp) > cutoff_time
            ]
            
            # Calculate summary statistics
            summary = {
                "time_period_hours": hours_back,
                "storage_operations": len(recent_storage_metrics),
                "system_samples": len(recent_system_metrics),
                "performance_summary": {},
                "system_summary": {},
                "alerts": len(self.performance_alerts),
                "query_profiles": len(self.query_profiles)
            }
            
            # Storage performance summary
            if recent_storage_metrics:
                durations = [m.duration_seconds for m in recent_storage_metrics]
                throughputs = [m.throughput_mb_per_second for m in recent_storage_metrics if m.throughput_mb_per_second > 0]
                
                summary["performance_summary"] = {
                    "average_duration_seconds": statistics.mean(durations),
                    "median_duration_seconds": statistics.median(durations),
                    "max_duration_seconds": max(durations),
                    "average_throughput_mb_per_second": statistics.mean(throughputs) if throughputs else 0,
                    "total_rows_processed": sum(m.rows_processed for m in recent_storage_metrics),
                    "total_data_mb": sum(m.file_size_mb for m in recent_storage_metrics)
                }
            
            # System performance summary
            if recent_system_metrics:
                cpu_values = [m.cpu_percent for m in recent_system_metrics]
                memory_values = [m.memory_percent for m in recent_system_metrics]
                
                summary["system_summary"] = {
                    "average_cpu_percent": statistics.mean(cpu_values),
                    "max_cpu_percent": max(cpu_values),
                    "average_memory_percent": statistics.mean(memory_values),
                    "max_memory_percent": max(memory_values),
                    "min_available_memory_gb": min(m.memory_available_gb for m in recent_system_metrics)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return {"error": str(e)}
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on performance data."""
        try:
            recommendations = []
            
            # Analyze recent metrics
            if len(self.storage_metrics) > 0:
                recent_metrics = list(self.storage_metrics)[-50:]  # Last 50 operations
                
                # Check for slow queries
                slow_queries = [m for m in recent_metrics if m.duration_seconds > self.PERFORMANCE_TARGETS["query_response_time_seconds"]]
                if slow_queries:
                    recommendations.append({
                        "type": "performance",
                        "priority": "high",
                        "title": "Slow Query Optimization",
                        "description": f"Found {len(slow_queries)} slow queries. Consider query optimization or indexing.",
                        "action": "Analyze query plans and consider lazy loading strategies."
                    })
                
                # Check compression ratios
                poor_compression = [m for m in recent_metrics if m.compression_ratio < self.PERFORMANCE_TARGETS["compression_ratio_minimum"]]
                if poor_compression:
                    recommendations.append({
                        "type": "storage",
                        "priority": "medium", 
                        "title": "Poor Compression Ratio",
                        "description": f"Found {len(poor_compression)} files with poor compression. Consider data type optimization.",
                        "action": "Review data types and apply categorical encoding for string columns."
                    })
                
                # Check memory usage
                high_memory = [m for m in recent_metrics if m.memory_usage_mb > 1000]  # >1GB
                if high_memory:
                    recommendations.append({
                        "type": "memory",
                        "priority": "high",
                        "title": "High Memory Usage",
                        "description": f"Found {len(high_memory)} operations with high memory usage.",
                        "action": "Implement batch processing and lazy evaluation."
                    })
            
            # System-level recommendations
            if len(self.system_metrics) > 0:
                recent_system = list(self.system_metrics)[-20:]  # Last 20 samples
                
                avg_memory = statistics.mean(m.memory_percent for m in recent_system)
                if avg_memory > 80:
                    recommendations.append({
                        "type": "system",
                        "priority": "high",
                        "title": "High System Memory Usage",
                        "description": f"Average memory usage is {avg_memory:.1f}%.",
                        "action": "Consider increasing system memory or optimizing query memory usage."
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate optimization recommendations: {e}")
            return [{"type": "error", "description": str(e)}]


if __name__ == "__main__":
    # Development testing
    monitor = StoragePerformanceMonitor()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Run benchmark
    benchmark_results = monitor.benchmark_storage_operations()
    print(f"📊 Benchmark completed: {len(benchmark_results.get('write_performance', {}))} write tests")
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"⚡ Performance summary: {summary.get('storage_operations', 0)} operations tracked")
    
    # Get recommendations
    recommendations = monitor.get_optimization_recommendations()
    print(f"💡 Optimization recommendations: {len(recommendations)} suggestions")
    
    # Stop monitoring
    monitor.stop_monitoring()