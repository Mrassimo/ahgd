"""
Performance monitoring, benchmarking and optimisation suite for AHGD.

This package provides comprehensive performance monitoring capabilities including:
- Memory and CPU profiling
- I/O performance monitoring
- Database query performance tracking
- Real-time system monitoring
- Performance benchmarking and regression testing
- Optimisation recommendations
"""

__version__ = "0.1.0"

from .profiler import (
    MemoryProfiler,
    CPUProfiler,
    IOProfiler,
    QueryProfiler,
    PerformanceProfiler,
    profile_memory,
    profile_cpu,
    profile_io,
    profile_query
)

from .benchmarks import (
    ETLBenchmarkSuite,
    DataProcessingBenchmarks,
    ValidationBenchmarks,
    LoadingBenchmarks,
    RegressionTestFramework
)

from .monitoring import (
    SystemMonitor,
    PerformanceMonitor,
    ResourceTracker,
    AlertManager,
    PerformanceReporter
)

from .optimisation import (
    PerformanceAnalyzer,
    BottleneckDetector,
    OptimisationRecommender,
    MemoryOptimiser,
    QueryOptimiser
)

__all__ = [
    # Profiling
    "MemoryProfiler",
    "CPUProfiler", 
    "IOProfiler",
    "QueryProfiler",
    "PerformanceProfiler",
    "profile_memory",
    "profile_cpu",
    "profile_io",
    "profile_query",
    
    # Benchmarking
    "ETLBenchmarkSuite",
    "DataProcessingBenchmarks",
    "ValidationBenchmarks",
    "LoadingBenchmarks",
    "RegressionTestFramework",
    
    # Monitoring
    "SystemMonitor",
    "PerformanceMonitor",
    "ResourceTracker",
    "AlertManager",
    "PerformanceReporter",
    
    # Optimisation
    "PerformanceAnalyzer",
    "BottleneckDetector",
    "OptimisationRecommender",
    "MemoryOptimiser",
    "QueryOptimiser"
]