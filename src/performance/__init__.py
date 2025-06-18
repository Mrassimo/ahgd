"""
Performance monitoring and optimization package for Australian Health Analytics Dashboard.

This package provides comprehensive performance monitoring, caching, and optimization
features for production deployment.

Components:
- Advanced caching with Redis compatibility
- Performance metrics collection
- Database query optimization
- Lazy loading and pagination
- Production deployment features
- Health monitoring and alerting
"""

from .cache import CacheManager, CacheConfig
from .monitoring import PerformanceMonitor, MetricsCollector
from .optimization import QueryOptimizer, DataLoader
from .health import HealthChecker, HealthStatus
from .alerts import AlertManager, AlertConfig

__all__ = [
    'CacheManager',
    'CacheConfig',
    'PerformanceMonitor',
    'MetricsCollector',
    'QueryOptimizer',
    'DataLoader',
    'HealthChecker',
    'HealthStatus',
    'AlertManager',
    'AlertConfig'
]

__version__ = "1.0.0"