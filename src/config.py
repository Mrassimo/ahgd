"""
Configuration Management System for Australian Health Analytics Dashboard

Centralised configuration management to eliminate hardcoded paths and enable
flexible deployment across different environments.

Author: AHGD Configuration System
Date: 2025-06-17
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not available, continue without it
    pass


class Environment(Enum):
    """Supported deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    name: str = "health_analytics.db"
    path: Optional[Path] = None
    connection_timeout: int = 30
    backup_enabled: bool = True
    
    def __post_init__(self):
        if self.path is None:
            self.path = get_project_root() / "data" / self.name
    
    @property
    def connection_string(self) -> str:
        """Generate SQLite connection string"""
        return f"sqlite:///{self.path}"


@dataclass
class DataSourceConfig:
    """Data source configuration settings"""
    raw_data_dir: Optional[Path] = None
    processed_data_dir: Optional[Path] = None
    
    # Data source URLs
    abs_data_base_url: str = "https://www.abs.gov.au/statistics"
    aihw_data_base_url: str = "https://www.aihw.gov.au/reports"
    
    # File processing settings
    chunk_size: int = 10000
    max_file_size_mb: int = 500
    
    def __post_init__(self):
        project_root = get_project_root()
        if self.raw_data_dir is None:
            self.raw_data_dir = project_root / "data" / "raw"
        if self.processed_data_dir is None:
            self.processed_data_dir = project_root / "data" / "processed"


@dataclass
class DashboardConfig:
    """Dashboard configuration settings"""
    host: str = "localhost"
    port: int = 8501
    debug: bool = False
    page_title: str = "Australian Health Analytics Dashboard"
    page_icon: str = "🏥"
    layout: str = "wide"
    
    # Caching settings
    cache_ttl: int = 3600  # 1 hour
    max_cache_entries: int = 100
    
    # Map settings
    default_map_center: tuple = (-25.2744, 133.7751)  # Australia center
    default_map_zoom: int = 5
    
    # Chart settings
    default_chart_height: int = 400
    default_chart_width: int = 600


@dataclass
class ProcessingConfig:
    """Data processing configuration settings"""
    # Performance settings
    max_workers: int = 4
    memory_limit_gb: int = 8
    
    # Geographic processing
    crs_code: str = "EPSG:4326"  # WGS84
    simplify_tolerance: float = 0.001
    
    # Analysis settings
    correlation_threshold: float = 0.5
    significance_level: float = 0.05


@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    console_enabled: bool = True
    log_dir: Optional[Path] = None
    max_file_size_mb: int = 10
    backup_count: int = 5
    
    def __post_init__(self):
        if self.log_dir is None:
            self.log_dir = get_project_root() / "logs"


@dataclass
class Config:
    """Main configuration class"""
    environment: Environment = Environment.DEVELOPMENT
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Load environment-specific overrides"""
        self._load_environment_config()
    
    def _load_environment_config(self):
        """Load configuration from environment variables"""
        # Environment
        env_name = os.getenv("AHGD_ENVIRONMENT", self.environment.value)
        try:
            self.environment = Environment(env_name)
        except ValueError:
            pass  # Use default
        
        # Database
        if os.getenv("AHGD_DB_NAME"):
            self.database.name = os.getenv("AHGD_DB_NAME")
        if os.getenv("AHGD_DB_PATH"):
            self.database.path = Path(os.getenv("AHGD_DB_PATH"))
        
        # Dashboard
        if os.getenv("AHGD_DASHBOARD_HOST"):
            self.dashboard.host = os.getenv("AHGD_DASHBOARD_HOST")
        if os.getenv("AHGD_DASHBOARD_PORT"):
            try:
                self.dashboard.port = int(os.getenv("AHGD_DASHBOARD_PORT"))
            except ValueError:
                pass
        if os.getenv("AHGD_DASHBOARD_DEBUG"):
            self.dashboard.debug = os.getenv("AHGD_DASHBOARD_DEBUG").lower() == "true"
        
        # Data source paths
        if os.getenv("AHGD_RAW_DATA_DIR"):
            self.data_source.raw_data_dir = Path(os.getenv("AHGD_RAW_DATA_DIR"))
        if os.getenv("AHGD_PROCESSED_DATA_DIR"):
            self.data_source.processed_data_dir = Path(os.getenv("AHGD_PROCESSED_DATA_DIR"))
        
        # Processing
        if os.getenv("AHGD_MAX_WORKERS"):
            try:
                self.processing.max_workers = int(os.getenv("AHGD_MAX_WORKERS"))
            except ValueError:
                pass
        
        # Logging
        if os.getenv("AHGD_LOG_LEVEL"):
            self.logging.level = os.getenv("AHGD_LOG_LEVEL")
        if os.getenv("AHGD_LOG_DIR"):
            self.logging.log_dir = Path(os.getenv("AHGD_LOG_DIR"))
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        issues = []
        
        # Check database path is accessible
        try:
            self.database.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Database path not accessible: {e}")
        
        # Check data directories exist
        for dir_path, name in [
            (self.data_source.raw_data_dir, "raw data"),
            (self.data_source.processed_data_dir, "processed data"),
            (self.logging.log_dir, "logging")
        ]:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"{name} directory not accessible: {e}")
        
        # Check port availability (basic check)
        if not (1024 <= self.dashboard.port <= 65535):
            issues.append(f"Dashboard port {self.dashboard.port} is not in valid range")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "environment": self.environment.value,
            "project_root": str(get_project_root())
        }


def get_project_root() -> Path:
    """
    Discover the project root directory dynamically.
    
    This function searches for the project root by looking for characteristic files
    that indicate the AHGD project directory, regardless of the current working directory.
    
    Returns:
        Path: The project root directory
        
    Raises:
        RuntimeError: If project root cannot be determined
    """
    # Start from current file's directory
    current_path = Path(__file__).resolve().parent
    
    # Look for characteristic files that indicate project root
    root_indicators = [
        "pyproject.toml",
        "uv.lock", 
        "main.py",
        "README.md",
        ("data", "processed"),  # data/processed directory
        ("scripts", "streamlit_dashboard.py")  # scripts directory with dashboard
    ]
    
    # Search upwards from current directory
    search_path = current_path
    for _ in range(10):  # Limit search depth
        for indicator in root_indicators:
            if isinstance(indicator, tuple):
                # Check for directory structure
                if all((search_path / part).exists() for part in indicator):
                    return search_path
            else:
                # Check for single file
                if (search_path / indicator).exists():
                    return search_path
        
        parent = search_path.parent
        if parent == search_path:  # Reached filesystem root
            break
        search_path = parent
    
    # If not found through file indicators, check for AHGD in path
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == "AHGD":
            return parent
    
    # Last resort: use current working directory if it looks like AHGD
    cwd = Path.cwd()
    if cwd.name == "AHGD" or "AHGD" in str(cwd):
        return cwd
    
    # Final fallback: assume project root is parent of src
    if current_path.parent.name == "src":
        return current_path.parent.parent
    
    raise RuntimeError(
        f"Could not determine project root. Current file: {__file__}, "
        f"Current working directory: {cwd}. "
        f"Please ensure you're running from within the AHGD project directory."
    )


def get_config(environment: Optional[str] = None) -> Config:
    """
    Get configuration instance for specified environment.
    
    Args:
        environment: Target environment (development, staging, production)
        
    Returns:
        Config: Configuration instance
    """
    config = Config()
    
    if environment:
        try:
            config.environment = Environment(environment)
        except ValueError:
            raise ValueError(f"Invalid environment: {environment}")
    
    return config


def setup_logging(config: Config) -> None:
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration instance
    """
    import logging
    from logging.handlers import RotatingFileHandler
    
    # Create log directory
    config.logging.log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.logging.level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    if config.logging.console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(config.logging.format))
        logger.addHandler(console_handler)
    
    # File handler
    if config.logging.file_enabled:
        log_file = config.logging.log_dir / "ahgd.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=config.logging.backup_count
        )
        file_handler.setFormatter(logging.Formatter(config.logging.format))
        logger.addHandler(file_handler)


# Global configuration instance
_config_instance = None


def get_global_config() -> Config:
    """Get the global configuration instance (singleton pattern)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = get_config()
    return _config_instance


def reset_global_config():
    """Reset the global configuration instance (useful for testing)"""
    global _config_instance
    _config_instance = None


if __name__ == "__main__":
    # Configuration validation and testing
    config = get_config()
    validation = config.validate()
    
    print("Configuration Validation Results:")
    print(f"Environment: {validation['environment']}")
    print(f"Project Root: {validation['project_root']}")
    print(f"Valid: {validation['valid']}")
    
    if validation['issues']:
        print("\nIssues found:")
        for issue in validation['issues']:
            print(f"  - {issue}")
    else:
        print("\nAll configuration checks passed!")
    
    print(f"\nDatabase path: {config.database.path}")
    print(f"Raw data directory: {config.data_source.raw_data_dir}")
    print(f"Processed data directory: {config.data_source.processed_data_dir}")
    print(f"Dashboard: {config.dashboard.host}:{config.dashboard.port}")