"""
Australian Health Analytics Dashboard - Configuration Package

This package provides centralised configuration management for the AHGD project.
"""

from .config import (
    Config,
    DatabaseConfig,
    DataSourceConfig,
    DashboardConfig,
    ProcessingConfig,
    LoggingConfig,
    Environment,
    get_config,
    get_global_config,
    get_project_root,
    setup_logging,
    reset_global_config
)

__all__ = [
    'Config',
    'DatabaseConfig',
    'DataSourceConfig', 
    'DashboardConfig',
    'ProcessingConfig',
    'LoggingConfig',
    'Environment',
    'get_config',
    'get_global_config',
    'get_project_root',
    'setup_logging',
    'reset_global_config'
]

__version__ = '1.0.0'
