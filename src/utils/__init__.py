"""
AHGD Utils Package

Core utility modules for the AHGD project including structured logging,
monitoring, health checks, and performance tracking capabilities.
"""

from .logging import (
    setup_logging,
    get_logger,
    log_operation,
    monitor_performance,
    track_lineage,
    AHGDLogger,
    LogContext,
    DataLineage
)

from .monitoring import (
    get_system_monitor,
    get_health_checker,
    get_error_tracker,
    SystemMonitor,
    HealthChecker,
    ErrorTracker,
    HealthCheckResult,
    PerformanceMetrics,
    Alert,
    check_database_connection,
    check_file_system_access,
    check_external_service
)

from .config_loader import (
    ConfigLoader,
    get_config_loader,
    load_logging_config,
    setup_environment_logging,
    validate_environment_config,
    create_log_directories,
    detect_environment,
    is_production,
    is_development,
    is_testing,
    setup_development_logging,
    setup_production_logging,
    setup_testing_logging,
    print_config_summary
)

__all__ = [
    # Logging
    'setup_logging',
    'get_logger',
    'log_operation',
    'monitor_performance',
    'track_lineage',
    'AHGDLogger',
    'LogContext',
    'DataLineage',
    
    # Monitoring
    'get_system_monitor',
    'get_health_checker',
    'get_error_tracker',
    'SystemMonitor',
    'HealthChecker',
    'ErrorTracker',
    'HealthCheckResult',
    'PerformanceMetrics',
    'Alert',
    'check_database_connection',
    'check_file_system_access',
    'check_external_service',
    
    # Configuration
    'ConfigLoader',
    'get_config_loader',
    'load_logging_config',
    'setup_environment_logging',
    'validate_environment_config',
    'create_log_directories',
    'detect_environment',
    'is_production',
    'is_development',
    'is_testing',
    'setup_development_logging',
    'setup_production_logging',
    'setup_testing_logging',
    'print_config_summary'
]