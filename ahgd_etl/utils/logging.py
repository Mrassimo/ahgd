"""Logging utilities for AHGD ETL Pipeline."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import get_settings


def setup_logging(
    name: Optional[str] = None,
    log_level: Optional[str] = None,
    log_file: Optional[bool] = True
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        name: Logger name (defaults to root logger)
        log_level: Logging level (defaults to settings)
        log_file: Whether to also log to file
        
    Returns:
        Configured logger instance
    """
    settings = get_settings()
    
    # Use provided level or fall back to settings
    level = log_level or settings.log_level
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"ahgd_etl_{timestamp}.log"
        log_path = settings.logs_dir / log_filename
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Log the log file location
        logger.info(f"Logging to file: {log_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)