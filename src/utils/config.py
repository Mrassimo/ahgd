"""
AHGD V3: Configuration Management
Centralized configuration for high-performance data processing.
"""

import os
from typing import Any, Dict, Optional
from pathlib import Path


def get_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration for AHGD data processing.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        Configuration dictionary
    """
    
    # Default configuration
    default_config = {
        "processing": {
            "chunk_size": 50000,
            "max_workers": 4,
            "memory_limit_gb": 4,
            "enable_lazy_evaluation": True,
            "enable_streaming": True,
            "cache_results": True
        },
        "sources": {
            "abs": {
                "base_url": "https://www.abs.gov.au",
                "api_timeout": 60
            },
            "aihw": {
                "base_url": "https://api.aihw.gov.au",
                "health_indicators_url": "https://api.aihw.gov.au/health-indicators/v1",
                "mortality_url": "https://api.aihw.gov.au/mortality/v1",
                "api_timeout": 60,
                "requests_per_second": 5
            }
        },
        "storage": {
            "duckdb_path": "./duckdb_data/ahgd_v3.db",
            "parquet_cache_dir": "./data/parquet_cache",
            "raw_data_dir": "./data/raw",
            "processed_data_dir": "./data/processed"
        }
    }
    
    # Override with environment variables if available
    if os.getenv("AHGD_CHUNK_SIZE"):
        default_config["processing"]["chunk_size"] = int(os.getenv("AHGD_CHUNK_SIZE"))
    
    if os.getenv("AHGD_MAX_WORKERS"):
        default_config["processing"]["max_workers"] = int(os.getenv("AHGD_MAX_WORKERS"))
    
    if os.getenv("AHGD_MEMORY_LIMIT_GB"):
        default_config["processing"]["memory_limit_gb"] = int(os.getenv("AHGD_MEMORY_LIMIT_GB"))
        
    if os.getenv("DUCKDB_PATH"):
        default_config["storage"]["duckdb_path"] = os.getenv("DUCKDB_PATH")
    
    return default_config