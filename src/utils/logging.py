"""
AHGD V3: High-Performance Logging Framework
Optimized logging for Polars-based data processing.
"""

import logging
import time
import functools
from typing import Optional, Callable, Any
from datetime import datetime


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a configured logger instance for AHGD components.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Create console handler with formatting
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger


def monitor_performance(func: Callable) -> Callable:
    """
    Decorator to monitor performance of data processing functions.
    
    Args:
        func: Function to monitor
        
    Returns:
        Decorated function with performance monitoring
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        logger = get_logger(f"performance.{func.__module__}.{func.__name__}")
        
        try:
            logger.info(f"Starting {func.__name__}")
            result = await func(*args, **kwargs)
            
            duration = time.time() - start_time
            logger.info(
                f"Completed {func.__name__}",
                extra={
                    'duration_seconds': duration,
                    'function': func.__name__,
                    'module': func.__module__
                }
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Failed {func.__name__}: {str(e)}",
                extra={
                    'duration_seconds': duration,
                    'error': str(e),
                    'function': func.__name__,
                    'module': func.__module__
                }
            )
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        logger = get_logger(f"performance.{func.__module__}.{func.__name__}")
        
        try:
            logger.info(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            logger.info(
                f"Completed {func.__name__}",
                extra={
                    'duration_seconds': duration,
                    'function': func.__name__,
                    'module': func.__module__
                }
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Failed {func.__name__}: {str(e)}",
                extra={
                    'duration_seconds': duration,
                    'error': str(e),
                    'function': func.__name__,
                    'module': func.__module__
                }
            )
            raise
    
    # Return appropriate wrapper based on function type
    if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
        return async_wrapper
    else:
        return sync_wrapper