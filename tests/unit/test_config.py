"""
Unit tests for configuration management system.

This module tests the Config class and related configuration functions
to ensure proper configuration loading, validation, and environment handling.
"""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import FrozenInstanceError

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from config import (
    Config, DatabaseConfig, DataSourceConfig, DashboardConfig,
    ProcessingConfig, LoggingConfig, Environment,
    get_config, get_project_root, setup_logging,
    get_global_config, reset_global_config
)


class TestEnvironmentEnum:
    """Test class for Environment enum."""
    
    def test_environment_values(self):
        """Test that all expected environment values exist."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"
    
    def test_environment_from_string(self):
        """Test creating Environment from string values."""
        assert Environment("development") == Environment.DEVELOPMENT
        assert Environment("staging") == Environment.STAGING
        assert Environment("production") == Environment.PRODUCTION
    
    def test_invalid_environment_string(self):
        """Test that invalid environment strings raise ValueError."""
        with pytest.raises(ValueError):
            Environment("invalid")


class TestDatabaseConfig:
    """Test class for DatabaseConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DatabaseConfig()
        assert config.name == "health_analytics.db"
        assert config.connection_timeout == 30
        assert config.backup_enabled is True
        assert config.path is not None
    
    def test_custom_values(self, temp_dir):
        """Test configuration with custom values."""
        custom_path = temp_dir / "custom.db"
        config = DatabaseConfig(
            name="custom.db",
            path=custom_path,
            connection_timeout=60,
            backup_enabled=False
        )
        assert config.name == "custom.db"
        assert config.path == custom_path
        assert config.connection_timeout == 60
        assert config.backup_enabled is False
    
    def test_connection_string(self, temp_dir):
        """Test connection string generation."""
        db_path = temp_dir / "test.db"
        config = DatabaseConfig(path=db_path)
        expected = f"sqlite:///{db_path}"
        assert config.connection_string == expected
    
    @patch('config.get_project_root')
    def test_default_path_from_project_root(self, mock_get_root, temp_dir):
        """Test that default path uses project root."""
        mock_get_root.return_value = temp_dir
        config = DatabaseConfig()
        expected_path = temp_dir / "health_analytics.db"
        assert config.path == expected_path


class TestDataSourceConfig:
    """Test class for DataSourceConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DataSourceConfig()
        assert config.abs_data_base_url == "https://www.abs.gov.au/statistics"
        assert config.aihw_data_base_url == "https://www.aihw.gov.au/reports"
        assert config.chunk_size == 10000
        assert config.max_file_size_mb == 500
        assert config.raw_data_dir is not None
        assert config.processed_data_dir is not None
    
    def test_custom_directories(self, temp_dir):
        """Test configuration with custom directories."""
        raw_dir = temp_dir / "custom_raw"
        processed_dir = temp_dir / "custom_processed"
        
        config = DataSourceConfig(
            raw_data_dir=raw_dir,
            processed_data_dir=processed_dir
        )
        
        assert config.raw_data_dir == raw_dir
        assert config.processed_data_dir == processed_dir
    
    @patch('config.get_project_root')
    def test_default_directories_from_project_root(self, mock_get_root, temp_dir):
        """Test that default directories use project root."""
        mock_get_root.return_value = temp_dir
        config = DataSourceConfig()
        
        expected_raw = temp_dir / "data" / "raw"
        expected_processed = temp_dir / "data" / "processed"
        
        assert config.raw_data_dir == expected_raw
        assert config.processed_data_dir == expected_processed


class TestDashboardConfig:
    """Test class for DashboardConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DashboardConfig()
        assert config.host == "localhost"
        assert config.port == 8501
        assert config.debug is False
        assert config.page_title == "Australian Health Analytics Dashboard"
        assert config.page_icon == "🏥"
        assert config.layout == "wide"
        assert config.cache_ttl == 3600
        assert config.max_cache_entries == 100
        assert config.default_map_center == (-25.2744, 133.7751)
        assert config.default_map_zoom == 5
        assert config.default_chart_height == 400
        assert config.default_chart_width == 600
    
    def test_custom_values(self):
        """Test configuration with custom values."""
        config = DashboardConfig(
            host="0.0.0.0",
            port=9000,
            debug=True,
            page_title="Custom Dashboard",
            cache_ttl=7200
        )
        assert config.host == "0.0.0.0"
        assert config.port == 9000
        assert config.debug is True
        assert config.page_title == "Custom Dashboard"
        assert config.cache_ttl == 7200


class TestProcessingConfig:
    """Test class for ProcessingConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        assert config.max_workers == 4
        assert config.memory_limit_gb == 8
        assert config.crs_code == "EPSG:4326"
        assert config.simplify_tolerance == 0.001
        assert config.correlation_threshold == 0.5
        assert config.significance_level == 0.05
    
    def test_custom_values(self):
        """Test configuration with custom values."""
        config = ProcessingConfig(
            max_workers=8,
            memory_limit_gb=16,
            crs_code="EPSG:3857",
            correlation_threshold=0.7
        )
        assert config.max_workers == 8
        assert config.memory_limit_gb == 16
        assert config.crs_code == "EPSG:3857"
        assert config.correlation_threshold == 0.7


class TestLoggingConfig:
    """Test class for LoggingConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert "%(asctime)s" in config.format
        assert config.file_enabled is True
        assert config.console_enabled is True
        assert config.max_file_size_mb == 10
        assert config.backup_count == 5
        assert config.log_dir is not None
    
    @patch('config.get_project_root')
    def test_default_log_directory(self, mock_get_root, temp_dir):
        """Test that default log directory uses project root."""
        mock_get_root.return_value = temp_dir
        config = LoggingConfig()
        expected_dir = temp_dir / "logs"
        assert config.log_dir == expected_dir


class TestMainConfig:
    """Test class for main Config class."""
    
    def test_default_initialization(self):
        """Test default configuration initialization."""
        config = Config()
        assert config.environment == Environment.DEVELOPMENT
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.data_source, DataSourceConfig)
        assert isinstance(config.dashboard, DashboardConfig)
        assert isinstance(config.processing, ProcessingConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    @patch.dict(os.environ, {
        'AHGD_ENVIRONMENT': 'production',
        'AHGD_DB_NAME': 'prod_db.db',
        'AHGD_DASHBOARD_HOST': '0.0.0.0',
        'AHGD_DASHBOARD_PORT': '9000',
        'AHGD_DASHBOARD_DEBUG': 'false',
        'AHGD_MAX_WORKERS': '8',
        'AHGD_LOG_LEVEL': 'ERROR'
    })
    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables."""
        config = Config()
        assert config.environment == Environment.PRODUCTION
        assert config.database.name == 'prod_db.db'
        assert config.dashboard.host == '0.0.0.0'
        assert config.dashboard.port == 9000
        assert config.dashboard.debug is False
        assert config.processing.max_workers == 8
        assert config.logging.level == 'ERROR'
    
    @patch.dict(os.environ, {
        'AHGD_DASHBOARD_PORT': 'invalid_port',
        'AHGD_MAX_WORKERS': 'invalid_workers'
    })
    def test_invalid_environment_values_ignored(self):
        """Test that invalid environment values are ignored gracefully."""
        config = Config()
        # Should use defaults when invalid values provided
        assert config.dashboard.port == 8501  # default
        assert config.processing.max_workers == 4  # default
    
    @patch.dict(os.environ, {'AHGD_ENVIRONMENT': 'invalid_env'})
    def test_invalid_environment_uses_default(self):
        """Test that invalid environment uses default."""
        config = Config()
        assert config.environment == Environment.DEVELOPMENT
    
    def test_validate_success(self, temp_dir):
        """Test successful configuration validation."""
        # Create a config with accessible paths
        config = Config()
        config.database.path = temp_dir / "test.db"
        config.data_source.raw_data_dir = temp_dir / "raw"
        config.data_source.processed_data_dir = temp_dir / "processed"
        config.logging.log_dir = temp_dir / "logs"
        
        result = config.validate()
        assert result['valid'] is True
        assert len(result['issues']) == 0
        assert 'environment' in result
        assert 'project_root' in result
    
    def test_validate_invalid_port(self):
        """Test validation with invalid port."""
        config = Config()
        config.dashboard.port = 99999999  # Invalid port
        
        result = config.validate()
        assert result['valid'] is False
        assert any('port' in issue.lower() for issue in result['issues'])
    
    @patch('pathlib.Path.mkdir')
    def test_validate_directory_creation_error(self, mock_mkdir):
        """Test validation when directory creation fails."""
        mock_mkdir.side_effect = PermissionError("Permission denied")
        
        config = Config()
        result = config.validate()
        
        assert result['valid'] is False
        assert len(result['issues']) > 0


class TestProjectRootDiscovery:
    """Test class for project root discovery."""
    
    def test_get_project_root_with_pyproject_toml(self, temp_dir):
        """Test project root discovery with pyproject.toml."""
        # Create a pyproject.toml file
        (temp_dir / "pyproject.toml").touch()
        
        with patch('config.Path') as mock_path:
            mock_path(__file__).resolve.return_value.parent = temp_dir
            mock_path.cwd.return_value = temp_dir
            
            # Mock Path behavior for file existence checks
            def mock_truediv(self, other):
                result = Mock()
                result.exists.return_value = (other == "pyproject.toml")
                return result
            
            mock_path.return_value.__truediv__ = mock_truediv
            
            # This test would need more sophisticated mocking
            # to fully test the project root discovery logic
    
    def test_get_project_root_runtime_error(self):
        """Test that RuntimeError is raised when project root cannot be found."""
        with patch('config.Path') as mock_path:
            # Mock Path to return non-AHGD directories
            mock_path(__file__).resolve.return_value.parents = [Path("/tmp"), Path("/")]
            mock_path(__file__).resolve.return_value.parent = Path("/tmp")
            mock_path.cwd.return_value = Path("/tmp")
            
            # Mock file existence to return False for all indicators
            mock_path.return_value.__truediv__.return_value.exists.return_value = False
            
            with pytest.raises(RuntimeError, match="Could not determine project root"):
                get_project_root()


class TestConfigFunctions:
    """Test class for configuration utility functions."""
    
    def test_get_config_default(self):
        """Test get_config with default parameters."""
        config = get_config()
        assert isinstance(config, Config)
        assert config.environment == Environment.DEVELOPMENT
    
    def test_get_config_with_environment(self):
        """Test get_config with specific environment."""
        config = get_config(environment="production")
        assert config.environment == Environment.PRODUCTION
    
    def test_get_config_invalid_environment(self):
        """Test get_config with invalid environment."""
        with pytest.raises(ValueError, match="Invalid environment"):
            get_config(environment="invalid")
    
    def test_global_config_singleton(self):
        """Test that global config uses singleton pattern."""
        reset_global_config()  # Ensure clean state
        
        config1 = get_global_config()
        config2 = get_global_config()
        
        assert config1 is config2  # Same instance
    
    def test_reset_global_config(self):
        """Test global config reset functionality."""
        config1 = get_global_config()
        reset_global_config()
        config2 = get_global_config()
        
        assert config1 is not config2  # Different instances


class TestLoggingSetup:
    """Test class for logging setup functionality."""
    
    @patch('logging.handlers.RotatingFileHandler')
    @patch('logging.getLogger')
    def test_setup_logging_with_file_and_console(self, mock_get_logger, mock_file_handler, temp_dir):
        """Test logging setup with both file and console handlers."""
        config = Config()
        config.logging.log_dir = temp_dir
        config.logging.file_enabled = True
        config.logging.console_enabled = True
        
        mock_logger = Mock()
        mock_logger.handlers = []  # Initialize handlers as empty list
        mock_get_logger.return_value = mock_logger
        
        setup_logging(config)
        
        # Verify logger configuration
        mock_logger.setLevel.assert_called_once()
        assert mock_logger.addHandler.call_count == 2  # Console + File
    
    @patch('logging.getLogger')
    def test_setup_logging_console_only(self, mock_get_logger, temp_dir):
        """Test logging setup with console handler only."""
        config = Config()
        config.logging.log_dir = temp_dir
        config.logging.file_enabled = False
        config.logging.console_enabled = True
        
        mock_logger = Mock()
        mock_logger.handlers = []  # Initialize handlers as empty list
        mock_get_logger.return_value = mock_logger
        
        setup_logging(config)
        
        # Should only add console handler
        assert mock_logger.addHandler.call_count == 1
    
    @patch('logging.getLogger')
    def test_setup_logging_creates_log_directory(self, mock_get_logger, temp_dir):
        """Test that logging setup creates log directory."""
        log_dir = temp_dir / "logs"
        config = Config()
        config.logging.log_dir = log_dir
        
        mock_logger = Mock()
        mock_logger.handlers = []  # Initialize handlers as empty list
        mock_get_logger.return_value = mock_logger
        
        setup_logging(config)
        
        assert log_dir.exists()


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration system."""
    
    def test_full_configuration_lifecycle(self, temp_dir):
        """Test complete configuration lifecycle."""
        # Set up environment
        with patch.dict(os.environ, {
            'AHGD_ENVIRONMENT': 'staging',
            'AHGD_DB_NAME': 'staging.db',
            'AHGD_RAW_DATA_DIR': str(temp_dir / 'raw'),
            'AHGD_PROCESSED_DATA_DIR': str(temp_dir / 'processed')
        }):
            # Create and validate configuration
            config = get_config()
            validation = config.validate()
            
            assert config.environment == Environment.STAGING
            assert config.database.name == 'staging.db'
            assert validation['valid'] is True
            
            # Test global config
            reset_global_config()
            global_config = get_global_config()
            assert global_config.environment == Environment.STAGING
    
    def test_configuration_with_logging_setup(self):
        """Test configuration integration with logging setup."""
        config = get_config()
        
        # Test that we can call setup_logging without errors
        # We won't mock it here since we want to test integration
        try:
            setup_logging(config)
            # If we get here, the integration worked
            assert True
        except Exception as e:
            pytest.fail(f"Logging setup integration failed: {e}")


@pytest.mark.parametrize("env_name,expected_enum", [
    ("development", Environment.DEVELOPMENT),
    ("staging", Environment.STAGING),
    ("production", Environment.PRODUCTION)
])
def test_environment_parameter_mapping(env_name, expected_enum):
    """Test environment string to enum mapping."""
    assert Environment(env_name) == expected_enum


@pytest.mark.parametrize("port,should_be_valid", [
    (1024, True),
    (8501, True),
    (65535, True),
    (1023, False),
    (65536, False),
    (-1, False)
])
def test_port_validation(port, should_be_valid):
    """Test port validation logic."""
    config = Config()
    config.dashboard.port = port
    validation = config.validate()
    
    if should_be_valid:
        # Port validation should not add issues
        port_issues = [issue for issue in validation['issues'] if 'port' in issue.lower()]
        assert len(port_issues) == 0
    else:
        # Port validation should add issues
        port_issues = [issue for issue in validation['issues'] if 'port' in issue.lower()]
        assert len(port_issues) > 0
