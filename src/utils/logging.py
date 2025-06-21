"""
AHGD Structured Logging Framework

This module provides a comprehensive logging framework with:
- Structured logging using loguru and structlog
- Rotating file handlers for different log levels
- JSON formatting for machine-readable logs
- Context managers for operation logging
- Performance logging decorators
- Data lineage tracking capabilities
"""

import asyncio
import json
import time
import sys
import os
import functools
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from collections import defaultdict

import structlog
from loguru import logger
import yaml
from rich.console import Console
from rich.traceback import install as rich_traceback_install

# Install rich traceback for better error display
rich_traceback_install(show_locals=True)

# Thread-local storage for context
_context = threading.local()


@dataclass
class LogContext:
    """Container for logging context information"""
    operation_id: str
    session_id: str
    user_id: Optional[str] = None
    component: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class DataLineage:
    """Data lineage tracking for ETL operations"""
    source_id: str
    target_id: str
    operation: str
    timestamp: datetime
    schema_version: Optional[str] = None
    row_count: Optional[int] = None
    transformations: Optional[List[str]] = None
    validation_status: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AHGDLogger:
    """
    Structured logging framework for AHGD project
    
    Features:
    - Multiple output formats (JSON, structured text, console)
    - Rotating file handlers with compression
    - Context-aware logging with operation tracking
    - Performance monitoring and metrics
    - Data lineage tracking
    - Health checks and alerting
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        self.config_path = config_path
        self.config_dict = config_dict
        self.console = Console()
        self.metrics = defaultdict(list)
        self.lineage_records = []
        self.session_id = str(uuid.uuid4())
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup loguru
        self._setup_loguru()
        
        # Setup structlog
        self._setup_structlog()
        
        # Initialize context
        self._init_context()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load logging configuration from YAML file or config dict"""
        # Prioritize config_dict if provided
        if self.config_dict:
            return self.config_dict
        
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'log_level': 'INFO',
            'log_dir': 'logs',
            'max_file_size': '100 MB',
            'backup_count': 5,
            'json_logs': True,
            'console_logs': True,
            'performance_logging': True,
            'lineage_tracking': True,
            'compression': 'gz',
            'formats': {
                'console': '{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}',
                'file': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}',
                'json': '{message}'
            }
        }
    
    def _setup_loguru(self):
        """Configure loguru with rotating file handlers"""
        # Remove default handler
        logger.remove()
        
        log_dir = Path(self.config['log_dir'])
        log_dir.mkdir(exist_ok=True)
        
        # Console handler
        if self.config.get('console_logs', True):
            logger.add(
                sys.stdout,
                format=self.config['formats']['console'],
                level=self.config['log_level'],
                colorize=True,
                backtrace=True,
                diagnose=True
            )
        
        # File handlers for different levels
        log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in log_levels:
            # Standard text logs
            logger.add(
                log_dir / f"{level.lower()}.log",
                format=self.config['formats']['file'],
                level=level,
                rotation=self.config['max_file_size'],
                retention=self.config['backup_count'],
                compression=self.config.get('compression', 'gz'),
                backtrace=True,
                diagnose=True,
                enqueue=True
            )
            
            # JSON logs for machine processing
            if self.config.get('json_logs', True):
                logger.add(
                    log_dir / f"{level.lower()}.json",
                    format=self._json_formatter,
                    level=level,
                    rotation=self.config['max_file_size'],
                    retention=self.config['backup_count'],
                    compression=self.config.get('compression', 'gz'),
                    serialize=True,
                    enqueue=True
                )
    
    def _setup_structlog(self):
        """Configure structlog for structured logging"""
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer() if self.config.get('console_logs') else structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(structlog.stdlib, self.config['log_level'], 20)
            ),
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def _json_formatter(self, record):
        """Custom JSON formatter for loguru"""
        # Get context information
        context = getattr(_context, 'current', {})
        
        # Build log record
        log_record = {
            'timestamp': record['time'].isoformat(),
            'level': record['level'].name,
            'logger': record['name'],
            'function': record['function'],
            'line': record['line'],
            'message': record['message'],
            'module': record['module'],
            'process': record['process'].id,
            'thread': record['thread'].id,
            'session_id': self.session_id,
        }
        
        # Add context information
        if context:
            log_record['context'] = context
        
        # Add exception information if present
        if record['exception']:
            log_record['exception'] = {
                'type': record['exception'].type.__name__,
                'value': str(record['exception'].value),
                'traceback': record['exception'].traceback
            }
        
        return json.dumps(log_record, default=str, ensure_ascii=False)
    
    def _init_context(self):
        """Initialize logging context"""
        if not hasattr(_context, 'current'):
            _context.current = {}
    
    def set_context(self, **kwargs):
        """Set logging context for current thread"""
        if not hasattr(_context, 'current'):
            _context.current = {}
        _context.current.update(kwargs)
    
    def clear_context(self):
        """Clear logging context for current thread"""
        if hasattr(_context, 'current'):
            _context.current.clear()
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context"""
        return getattr(_context, 'current', {})
    
    @contextmanager
    def operation_context(self, operation_name: str, **kwargs):
        """Context manager for operation logging"""
        operation_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Set context
        old_context = self.get_context().copy()
        self.set_context(
            operation_id=operation_id,
            operation=operation_name,
            **kwargs
        )
        
        try:
            logger.info(f"Starting operation: {operation_name}", operation_id=operation_id)
            yield operation_id
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Operation failed: {operation_name}",
                operation_id=operation_id,
                duration=duration,
                error=str(e),
                exc_info=True
            )
            raise
            
        else:
            duration = time.time() - start_time
            logger.info(
                f"Operation completed: {operation_name}",
                operation_id=operation_id,
                duration=duration
            )
            
            # Store performance metrics
            if self.config.get('performance_logging', True):
                self.metrics[operation_name].append({
                    'duration': duration,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'operation_id': operation_id
                })
        
        finally:
            # Restore context
            _context.current = old_context
    
    def performance_monitor(self, operation_name: Optional[str] = None):
        """Decorator for performance monitoring"""
        def decorator(func: Callable) -> Callable:
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.operation_context(op_name):
                    return func(*args, **kwargs)
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.operation_context(op_name):
                    return await func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
        
        return decorator
    
    def track_data_lineage(self, source_id: str, target_id: str, operation: str, **kwargs):
        """Track data lineage for ETL operations"""
        if not self.config.get('lineage_tracking', True):
            return
        
        lineage = DataLineage(
            source_id=source_id,
            target_id=target_id,
            operation=operation,
            timestamp=datetime.now(timezone.utc),
            **kwargs
        )
        
        self.lineage_records.append(lineage)
        
        logger.info(
            f"Data lineage tracked: {operation}",
            source_id=source_id,
            target_id=target_id,
            lineage=lineage.to_dict()
        )
    
    def get_performance_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for operations"""
        if operation_name:
            return {
                'operation': operation_name,
                'metrics': self.metrics.get(operation_name, [])
            }
        
        return dict(self.metrics)
    
    def get_lineage_records(self, source_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get data lineage records"""
        records = [record.to_dict() for record in self.lineage_records]
        
        if source_id:
            records = [r for r in records if r['source_id'] == source_id]
        
        return records
    
    def export_metrics(self, filepath: str):
        """Export performance metrics to file"""
        metrics_data = {
            'session_id': self.session_id,
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics': self.get_performance_metrics(),
            'lineage': self.get_lineage_records()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on logging system"""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': {}
        }
        
        try:
            # Check log directory
            log_dir = Path(self.config['log_dir'])
            health_status['checks']['log_directory'] = {
                'exists': log_dir.exists(),
                'writable': os.access(log_dir, os.W_OK) if log_dir.exists() else False
            }
            
            # Check disk space
            if log_dir.exists():
                stat = os.statvfs(log_dir)
                free_space = stat.f_bavail * stat.f_frsize
                health_status['checks']['disk_space'] = {
                    'free_bytes': free_space,
                    'free_mb': free_space / (1024 * 1024)
                }
            
            # Check recent metrics
            health_status['checks']['metrics'] = {
                'operations_count': len(self.metrics),
                'lineage_records': len(self.lineage_records)
            }
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
    
    # Standard logging interface
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        logger.critical(message, **kwargs)


# Global logger instance
_logger_instance = None


def get_logger(config_path: Optional[str] = None) -> AHGDLogger:
    """Get or create global logger instance"""
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = AHGDLogger(config_path)
    
    return _logger_instance


def setup_logging(config_path: Optional[str] = None) -> AHGDLogger:
    """Setup and return configured logger"""
    return get_logger(config_path)


# Convenience functions
def log_operation(operation_name: str, **kwargs):
    """Decorator for operation logging"""
    return get_logger().operation_context(operation_name, **kwargs)


def monitor_performance(operation_name: Optional[str] = None):
    """Decorator for performance monitoring"""
    return get_logger().performance_monitor(operation_name)


def track_lineage(source_id: str, target_id: str, operation: str, **kwargs):
    """Track data lineage"""
    get_logger().track_data_lineage(source_id, target_id, operation, **kwargs)


# Export commonly used logger functions
log = logger