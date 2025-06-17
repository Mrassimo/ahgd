"""
Performance Testing Suite - Phase 5.4

Comprehensive performance and load testing for the Australian Health Analytics platform.
Tests system performance under realistic and extreme data volumes, validating the 57.5% 
memory optimization targets and ensuring the platform can handle production-scale 
Australian health data workloads.

Key Testing Domains:
- Large-Scale Data Processing Performance (1M+ records)
- Storage Optimization Performance (Parquet compression, memory optimization)
- Web Interface and Dashboard Performance (<2s load time target)
- System Resilience and Stress Testing

Performance Targets:
- Processing Speed: <5 minutes for 1M+ records end-to-end
- Memory Optimization: 57.5% memory reduction maintained at scale
- Storage Performance: 60-70% Parquet compression, <0.1s/MB read speeds
- Dashboard Load Time: <2 seconds with realistic data volumes
- Concurrent Processing: >500 records/second throughput
- System Stability: 24+ hour continuous operation without degradation
"""

from pathlib import Path

# Performance testing configuration
PERFORMANCE_CONFIG = {
    "large_scale_targets": {
        "max_processing_time_minutes": 5,
        "min_throughput_records_per_second": 500,
        "max_memory_usage_gb": 4,
        "min_integration_success_rate": 0.85
    },
    "memory_optimization_targets": {
        "min_memory_reduction_percent": 57.5,
        "max_memory_overhead_percent": 15,
        "target_compression_ratio_min": 1.6,
        "target_compression_ratio_max": 2.8
    },
    "dashboard_performance_targets": {
        "max_load_time_seconds": 2.0,
        "max_interactive_response_ms": 500,
        "min_concurrent_users": 10,
        "max_mobile_load_time_seconds": 3.0
    },
    "storage_performance_targets": {
        "min_read_speed_mb_per_second": 100,
        "min_write_speed_mb_per_second": 50,
        "parquet_compression_min_percent": 60,
        "parquet_compression_max_percent": 70
    },
    "stress_testing_targets": {
        "max_continuous_operation_hours": 24,
        "memory_leak_tolerance_mb": 100,
        "error_recovery_max_seconds": 30,
        "max_resource_exhaustion_recovery_seconds": 60
    }
}

# Australian health data scale requirements
AUSTRALIAN_DATA_SCALE = {
    "sa2_areas_total": 2454,
    "simulated_health_records": 1000000,
    "states_territories": 8,
    "concurrent_users_target": 10,
    "geographic_complexity_levels": ["state", "sa2", "sa3", "sa4", "postcode"],
    "temporal_patterns": ["daily", "weekly", "monthly", "quarterly", "yearly"]
}

__version__ = "1.0.0"
__author__ = "Australian Health Data Analytics Team"