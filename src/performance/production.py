"""
Production Deployment Configuration and Integration

This module provides production-ready configurations and integrations for:
- Environment-specific settings
- Load balancer health endpoints
- Graceful shutdown handling
- Resource optimization
- Production monitoring setup
"""

import os
import sys
import signal
import atexit
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import json

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .cache import CacheManager, CacheConfig
from .monitoring import PerformanceMonitor
from .health import HealthChecker, get_health_checker
from .alerts import AlertManager, AlertConfig, AlertRule, AlertSeverity, AlertChannel
from .optimization import QueryOptimizer, DataLoader

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production deployment configuration"""
    # Environment
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "WARNING"
    
    # Performance
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    enable_query_optimization: bool = True
    enable_health_checks: bool = True
    enable_alerting: bool = True
    
    # Health check endpoints
    health_endpoint_enabled: bool = True
    health_endpoint_port: int = 8502
    readiness_check_enabled: bool = True
    liveness_check_enabled: bool = True
    
    # Resource optimization
    max_memory_mb: int = 2048
    max_cpu_percent: float = 80.0
    gc_frequency_seconds: int = 300  # 5 minutes
    
    # Monitoring
    metrics_retention_hours: int = 24
    alert_retention_hours: int = 72
    health_check_interval_seconds: int = 60
    
    # Cache configuration
    cache_redis_enabled: bool = False
    cache_redis_host: str = "localhost"
    cache_redis_port: int = 6379
    cache_file_enabled: bool = True
    cache_max_memory_mb: int = 512
    
    # Alert configuration
    alert_email_enabled: bool = False
    alert_webhook_enabled: bool = False
    alert_email_recipients: List[str] = field(default_factory=list)
    alert_webhook_urls: List[str] = field(default_factory=list)
    
    # Database optimization
    db_connection_pool_size: int = 10
    db_query_cache_ttl: int = 3600
    db_enable_wal_mode: bool = True
    
    # Session management
    session_cleanup_enabled: bool = True
    session_max_age_hours: int = 24
    
    @classmethod
    def from_environment(cls) -> 'ProductionConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Environment settings
        config.environment = os.getenv('AHGD_ENVIRONMENT', config.environment)
        config.debug_mode = os.getenv('AHGD_DEBUG', 'false').lower() == 'true'
        config.log_level = os.getenv('AHGD_LOG_LEVEL', config.log_level)
        
        # Performance settings
        config.enable_performance_monitoring = os.getenv('AHGD_PERFORMANCE_MONITORING', 'true').lower() == 'true'
        config.enable_caching = os.getenv('AHGD_CACHING', 'true').lower() == 'true'
        config.enable_query_optimization = os.getenv('AHGD_QUERY_OPTIMIZATION', 'true').lower() == 'true'
        
        # Health check settings
        config.health_endpoint_enabled = os.getenv('AHGD_HEALTH_ENDPOINT', 'true').lower() == 'true'
        config.health_endpoint_port = int(os.getenv('AHGD_HEALTH_PORT', str(config.health_endpoint_port)))
        
        # Resource limits
        config.max_memory_mb = int(os.getenv('AHGD_MAX_MEMORY_MB', str(config.max_memory_mb)))
        config.max_cpu_percent = float(os.getenv('AHGD_MAX_CPU_PERCENT', str(config.max_cpu_percent)))
        
        # Cache settings
        config.cache_redis_enabled = os.getenv('AHGD_REDIS_ENABLED', 'false').lower() == 'true'
        config.cache_redis_host = os.getenv('AHGD_REDIS_HOST', config.cache_redis_host)
        config.cache_redis_port = int(os.getenv('AHGD_REDIS_PORT', str(config.cache_redis_port)))
        
        # Alert settings
        config.alert_email_enabled = os.getenv('AHGD_ALERT_EMAIL', 'false').lower() == 'true'
        config.alert_webhook_enabled = os.getenv('AHGD_ALERT_WEBHOOK', 'false').lower() == 'true'
        
        email_recipients = os.getenv('AHGD_ALERT_EMAIL_RECIPIENTS', '')
        if email_recipients:
            config.alert_email_recipients = [email.strip() for email in email_recipients.split(',')]
        
        webhook_urls = os.getenv('AHGD_ALERT_WEBHOOK_URLS', '')
        if webhook_urls:
            config.alert_webhook_urls = [url.strip() for url in webhook_urls.split(',')]
        
        return config


class ProductionManager:
    """Production deployment manager"""
    
    def __init__(self, config: Optional[ProductionConfig] = None):
        self.config = config or ProductionConfig.from_environment()
        
        # Initialize components
        self.cache_manager: Optional[CacheManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.health_checker: Optional[HealthChecker] = None
        self.alert_manager: Optional[AlertManager] = None
        self.query_optimizer: Optional[QueryOptimizer] = None
        
        # Runtime state
        self.shutdown_requested = False
        self.cleanup_threads: List[threading.Thread] = []
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Initialize production components
        self._initialize_components()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def shutdown_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
        
        # Register handlers for common shutdown signals
        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def _initialize_components(self):
        """Initialize production components based on configuration"""
        try:
            # Initialize cache manager
            if self.config.enable_caching:
                cache_config = CacheConfig(
                    redis_enabled=self.config.cache_redis_enabled,
                    redis_host=self.config.cache_redis_host,
                    redis_port=self.config.cache_redis_port,
                    file_cache_enabled=self.config.cache_file_enabled,
                    max_memory_mb=self.config.cache_max_memory_mb
                )
                self.cache_manager = CacheManager(cache_config)
                logger.info("Cache manager initialized")
            
            # Initialize performance monitor
            if self.config.enable_performance_monitoring:
                self.performance_monitor = PerformanceMonitor()
                logger.info("Performance monitor initialized")
            
            # Initialize health checker
            if self.config.enable_health_checks:
                try:
                    self.health_checker = get_health_checker()
                    logger.info("Health checker initialized")
                except Exception as e:
                    logger.warning(f"Could not initialize health checker: {e}")
            
            # Initialize alert manager
            if self.config.enable_alerting:
                alert_config = AlertConfig(
                    email_to=self.config.alert_email_recipients,
                    webhook_urls=self.config.alert_webhook_urls
                )
                self.alert_manager = AlertManager(alert_config)
                
                # Setup production alert rules
                self._setup_production_alerts()
                logger.info("Alert manager initialized")
            
        except Exception as e:
            logger.error(f"Error initializing production components: {e}")
            raise
    
    def _setup_production_alerts(self):
        """Setup production-specific alert rules"""
        if not self.alert_manager:
            return
        
        production_rules = [
            AlertRule(
                name="critical_memory_usage",
                condition=f"name == 'memory_usage_percent' and value > {self.config.max_memory_mb / 1024 * 80}",  # 80% of max
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                message_template="CRITICAL: Memory usage {value:.1f}% exceeds production threshold",
                cooldown_minutes=5,
                max_alerts_per_hour=6
            ),
            AlertRule(
                name="high_cpu_sustained",
                condition=f"name == 'cpu_usage_percent' and value > {self.config.max_cpu_percent}",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                message_template="HIGH: CPU usage {value:.1f}% sustained above threshold",
                cooldown_minutes=10,
                max_alerts_per_hour=4
            ),
            AlertRule(
                name="health_check_critical",
                condition="health_check and status == 'critical'",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                message_template="CRITICAL: Health check failure - {name}: {message}",
                cooldown_minutes=5,
                escalation_delay_minutes=15,
                escalation_channels=[AlertChannel.WEBHOOK]
            ),
            AlertRule(
                name="database_slow_queries",
                condition="name.startswith('db_query_duration_') and value > 10.0",
                severity=AlertSeverity.HIGH,
                channels=[AlertChannel.LOG],
                message_template="SLOW QUERY: {name} took {value:.2f}s",
                cooldown_minutes=30,
                max_alerts_per_hour=10
            ),
            AlertRule(
                name="cache_hit_rate_low",
                condition="name == 'cache_hit_rate' and value < 0.5",
                severity=AlertSeverity.MEDIUM,
                channels=[AlertChannel.LOG],
                message_template="Cache hit rate low: {value:.1%}",
                cooldown_minutes=60,
                max_alerts_per_hour=2
            )
        ]
        
        for rule in production_rules:
            self.alert_manager.add_rule(rule)
        
        logger.info(f"Added {len(production_rules)} production alert rules")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        # Resource monitoring thread
        if self.config.enable_performance_monitoring:
            monitor_thread = threading.Thread(
                target=self._resource_monitor_worker,
                daemon=True,
                name="resource_monitor"
            )
            monitor_thread.start()
            self.cleanup_threads.append(monitor_thread)
        
        # Garbage collection thread
        gc_thread = threading.Thread(
            target=self._garbage_collection_worker,
            daemon=True,
            name="garbage_collector"
        )
        gc_thread.start()
        self.cleanup_threads.append(gc_thread)
        
        # Session cleanup thread
        if self.config.session_cleanup_enabled:
            session_thread = threading.Thread(
                target=self._session_cleanup_worker,
                daemon=True,
                name="session_cleanup"
            )
            session_thread.start()
            self.cleanup_threads.append(session_thread)
        
        logger.info("Background tasks started")
    
    def _resource_monitor_worker(self):
        """Background resource monitoring worker"""
        while not self.shutdown_requested:
            try:
                if self.performance_monitor:
                    # Collect current metrics
                    summary = self.performance_monitor.get_performance_summary(hours=1)
                    
                    # Check resource limits
                    current_system = summary.get('current_system', {})
                    
                    # Memory check
                    memory_percent = current_system.get('memory_percent', 0)
                    if memory_percent > 90:
                        logger.warning(f"High memory usage: {memory_percent:.1f}%")
                    
                    # CPU check
                    cpu_percent = current_system.get('cpu_percent', 0)
                    if cpu_percent > self.config.max_cpu_percent:
                        logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                
                time.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                time.sleep(60)
    
    def _garbage_collection_worker(self):
        """Background garbage collection worker"""
        import gc
        
        while not self.shutdown_requested:
            try:
                # Force garbage collection
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Garbage collected {collected} objects")
                
                time.sleep(self.config.gc_frequency_seconds)
                
            except Exception as e:
                logger.error(f"Error in garbage collection: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _session_cleanup_worker(self):
        """Background session cleanup worker"""
        while not self.shutdown_requested:
            try:
                if STREAMLIT_AVAILABLE:
                    # Clean up old session state entries
                    # This is a simplified cleanup - in practice you'd want more sophisticated logic
                    pass
                
                time.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                time.sleep(1800)
    
    def get_health_endpoint_response(self, check_type: str = "health") -> Dict[str, Any]:
        """Get health endpoint response for load balancers"""
        try:
            if not self.health_checker:
                return {
                    'status': 'unavailable',
                    'message': 'Health checker not initialized',
                    'timestamp': time.time(),
                    'http_status': 503
                }
            
            if check_type == "readiness":
                return self.health_checker.get_readiness_check()
            elif check_type == "liveness":
                return self.health_checker.get_liveness_check()
            else:
                return self.health_checker.get_health_endpoint_response(include_slow_checks=False)
        
        except Exception as e:
            logger.error(f"Health endpoint error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time(),
                'http_status': 503
            }
    
    def get_metrics_endpoint_response(self) -> Dict[str, Any]:
        """Get metrics endpoint response for monitoring systems"""
        try:
            metrics = {}
            
            if self.performance_monitor:
                metrics['performance'] = self.performance_monitor.get_performance_summary(hours=1)
            
            if self.cache_manager:
                metrics['cache'] = self.cache_manager.get_stats()
            
            if self.alert_manager:
                metrics['alerts'] = self.alert_manager.get_alert_statistics()
            
            return {
                'status': 'ok',
                'metrics': metrics,
                'timestamp': time.time(),
                'environment': self.config.environment
            }
        
        except Exception as e:
            logger.error(f"Metrics endpoint error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': time.time()
            }
    
    def optimize_for_production(self):
        """Apply production optimizations"""
        try:
            # Memory optimization
            if STREAMLIT_AVAILABLE:
                # Configure Streamlit for production
                st.set_page_config(
                    page_title="Australian Health Analytics",
                    page_icon="🏥",
                    layout="wide",
                    initial_sidebar_state="collapsed"  # Save space
                )
            
            # Database optimization
            if self.query_optimizer:
                # Optimize common tables
                tables_to_optimize = [
                    'sa2_boundaries',
                    'health_metrics',
                    'demographics'
                ]
                
                for table in tables_to_optimize:
                    try:
                        self.query_optimizer.optimize_table(table)
                    except Exception as e:
                        logger.warning(f"Could not optimize table {table}: {e}")
            
            # Cache warming
            if self.cache_manager:
                # Pre-warm cache with common queries
                self._warm_cache()
            
            logger.info("Production optimizations applied")
            
        except Exception as e:
            logger.error(f"Error applying production optimizations: {e}")
    
    def _warm_cache(self):
        """Pre-warm cache with common data"""
        try:
            # This would typically load common datasets into cache
            # Implementation depends on your specific use case
            logger.info("Cache warming completed")
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
    
    def shutdown(self):
        """Graceful shutdown"""
        if self.shutdown_requested:
            return
        
        logger.info("Initiating graceful shutdown...")
        self.shutdown_requested = True
        
        try:
            # Stop alert manager
            if self.alert_manager:
                self.alert_manager.stop_background_processing()
            
            # Stop performance monitor
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            # Clear cache if needed
            if self.cache_manager:
                # Optionally clear sensitive data
                pass
            
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def cleanup(self):
        """Cleanup resources on exit"""
        try:
            self.shutdown()
            
            # Wait for background threads to finish
            for thread in self.cleanup_threads:
                if thread.is_alive():
                    thread.join(timeout=5)
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    @contextmanager
    def production_context(self):
        """Context manager for production operations"""
        try:
            self.optimize_for_production()
            yield self
        except Exception as e:
            logger.error(f"Production context error: {e}")
            raise
        finally:
            # Cleanup is handled by atexit
            pass


# Global production manager instance
_global_production_manager: Optional[ProductionManager] = None


def get_production_manager(config: Optional[ProductionConfig] = None) -> ProductionManager:
    """Get or create global production manager"""
    global _global_production_manager
    if _global_production_manager is None:
        _global_production_manager = ProductionManager(config)
    return _global_production_manager


def setup_production_environment():
    """Setup production environment with all optimizations"""
    config = ProductionConfig.from_environment()
    
    # Configure logging for production
    log_level = getattr(logging, config.log_level.upper(), logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('ahgd_production.log') if not config.debug_mode else logging.NullHandler()
        ]
    )
    
    # Initialize production manager
    production_manager = get_production_manager(config)
    
    logger.info(f"Production environment setup completed - Environment: {config.environment}")
    
    return production_manager


def create_health_endpoints():
    """Create Flask/FastAPI health endpoints for load balancers"""
    # This is a template for creating actual HTTP endpoints
    # In practice, you'd integrate this with your web framework
    
    endpoints = {
        '/health': 'Health check endpoint',
        '/health/ready': 'Readiness check endpoint',
        '/health/live': 'Liveness check endpoint',
        '/metrics': 'Metrics endpoint for monitoring'
    }
    
    logger.info(f"Health endpoints available: {list(endpoints.keys())}")
    return endpoints


if __name__ == "__main__":
    # Test production setup
    print("Testing production setup...")
    
    # Create test configuration
    config = ProductionConfig(
        environment="test",
        debug_mode=True,
        enable_caching=True,
        enable_performance_monitoring=True
    )
    
    # Initialize production manager
    with ProductionManager(config).production_context() as prod_manager:
        print(f"Production manager initialized: {prod_manager.config.environment}")
        
        # Test health endpoints
        health_response = prod_manager.get_health_endpoint_response()
        print(f"Health endpoint: {health_response['status']}")
        
        metrics_response = prod_manager.get_metrics_endpoint_response()
        print(f"Metrics endpoint: {metrics_response['status']}")
        
        time.sleep(2)  # Let background tasks run briefly
    
    print("Production setup test completed!")