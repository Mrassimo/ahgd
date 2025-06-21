"""
AHGD Configuration Loader

Utility functions for loading environment-specific configurations
including logging, monitoring, and application settings.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .interfaces import ConfigurationError


class ConfigLoader:
    """
    Configuration loader for AHGD project
    
    Handles loading of environment-specific configurations with
    fallback support and validation.
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.environment = self._detect_environment()
        self._config_cache = {}
    
    def _detect_environment(self) -> str:
        """Detect current environment from environment variables"""
        return os.getenv('AHGD_ENV', os.getenv('ENV', 'development')).lower()
    
    def load_logging_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Load logging configuration for specified environment
        
        Args:
            environment: Target environment (development, staging, production, testing)
                        If None, uses detected environment
        
        Returns:
            Dictionary containing logging configuration
        """
        env = environment or self.environment
        config_file = self.config_dir / "logging_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Logging config file not found: {config_file}")
        
        # Load full configuration
        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Get environment-specific config
        if env not in full_config:
            available_envs = list(full_config.keys())
            raise ValueError(f"Environment '{env}' not found in config. Available: {available_envs}")
        
        env_config = full_config[env].copy()
        
        # Merge with common configurations if they exist
        common_configs = ['monitoring', 'health_checks', 'sampling', 
                         'enrichment', 'security', 'performance', 'integrations']
        
        for common_key in common_configs:
            if common_key in full_config:
                env_config[common_key] = full_config[common_key]
        
        return env_config
    
    def setup_environment_logging(self, environment: Optional[str] = None):
        """
        Setup logging for the specified environment
        
        Args:
            environment: Target environment, if None uses detected environment
        
        Returns:
            Configured AHGDLogger instance
        """
        from .logging import AHGDLogger
        
        config = self.load_logging_config(environment)
        return AHGDLogger(config_dict=config)
    
    def get_monitoring_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Get monitoring configuration for environment"""
        config = self.load_logging_config(environment)
        return config.get('monitoring', {})
    
    def get_health_check_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Get health check configuration for environment"""
        config = self.load_logging_config(environment)
        return config.get('health_checks', {})
    
    def get_notification_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Get notification configuration for environment"""
        monitoring_config = self.get_monitoring_config(environment)
        return monitoring_config.get('notifications', {})
    
    def validate_configuration(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """
        Validate configuration for environment
        
        Returns:
            Dictionary with validation results
        """
        env = environment or self.environment
        validation_results = {
            'environment': env,
            'valid': True,
            'errors': [],
            'warnings': [],
            'config_file_exists': False,
            'required_directories': {},
            'log_levels': [],
            'integrations': {}
        }
        
        try:
            # Check config file exists
            config_file = self.config_dir / "logging_config.yaml"
            validation_results['config_file_exists'] = config_file.exists()
            
            if not config_file.exists():
                validation_results['valid'] = False
                validation_results['errors'].append(f"Configuration file not found: {config_file}")
                return validation_results
            
            # Load and validate configuration
            config = self.load_logging_config(env)
            
            # Validate log directories
            log_dir = Path(config.get('log_dir', 'logs'))
            validation_results['required_directories']['log_dir'] = {
                'path': str(log_dir),
                'exists': log_dir.exists(),
                'writable': log_dir.exists() and os.access(log_dir, os.W_OK)
            }
            
            if not log_dir.exists():
                validation_results['warnings'].append(f"Log directory does not exist: {log_dir}")
            
            # Validate log levels
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            log_level = config.get('log_level', 'INFO')
            if log_level not in valid_levels:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Invalid log level: {log_level}")
            else:
                validation_results['log_levels'] = [log_level]
            
            # Validate specific log file configurations
            log_files = config.get('log_files', {})
            for file_config_name, file_config in log_files.items():
                file_path = Path(file_config['path'])
                validation_results['required_directories'][file_config_name] = {
                    'path': str(file_path.parent),
                    'exists': file_path.parent.exists(),
                    'levels': file_config.get('levels', [])
                }
            
            # Validate integrations
            integrations = config.get('integrations', {})
            for integration_name, integration_config in integrations.items():
                enabled = integration_config.get('enabled', False)
                validation_results['integrations'][integration_name] = {
                    'enabled': enabled,
                    'configured': len(integration_config) > 1  # More than just 'enabled'
                }
        
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Configuration validation error: {str(e)}")
        
        return validation_results
    
    def create_environment_directories(self, environment: Optional[str] = None):
        """Create required directories for the environment"""
        env = environment or self.environment
        
        try:
            config = self.load_logging_config(env)
            
            # Create main log directory
            log_dir = Path(config.get('log_dir', 'logs'))
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create specific log file directories
            log_files = config.get('log_files', {})
            for file_config in log_files.values():
                file_path = Path(file_config['path'])
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            return True
        
        except Exception as e:
            raise RuntimeError(f"Failed to create directories: {str(e)}")
    
    def get_environment_summary(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive environment configuration summary"""
        env = environment or self.environment
        
        try:
            config = self.load_logging_config(env)
            validation = self.validate_configuration(env)
            
            return {
                'environment': env,
                'detected_environment': self.environment,
                'configuration_valid': validation['valid'],
                'log_level': config.get('log_level'),
                'log_directory': config.get('log_dir'),
                'console_logging': config.get('console_logs', False),
                'json_logging': config.get('json_logs', False),
                'performance_logging': config.get('performance_logging', False),
                'lineage_tracking': config.get('lineage_tracking', False),
                'monitoring_enabled': config.get('monitoring', {}).get('enabled', False),
                'health_checks_enabled': config.get('health_checks', {}).get('enabled', False),
                'log_files_configured': len(config.get('log_files', {})),
                'integrations_available': list(config.get('integrations', {}).keys()),
                'validation_errors': validation.get('errors', []),
                'validation_warnings': validation.get('warnings', [])
            }
        
        except Exception as e:
            return {
                'environment': env,
                'error': str(e),
                'configuration_valid': False
            }


# Global config loader instance
_config_loader = None


def get_config_loader(config_dir: str = "configs") -> ConfigLoader:
    """Get or create global config loader instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    return _config_loader


def load_logging_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to load logging configuration"""
    return get_config_loader().load_logging_config(environment)


def setup_environment_logging(environment: Optional[str] = None):
    """Convenience function to setup environment-specific logging"""
    return get_config_loader().setup_environment_logging(environment)


def validate_environment_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to validate environment configuration"""
    return get_config_loader().validate_configuration(environment)


def create_log_directories(environment: Optional[str] = None):
    """Convenience function to create required log directories"""
    return get_config_loader().create_environment_directories(environment)


# Environment detection utilities
def detect_environment() -> str:
    """Detect current environment from various sources"""
    # Check environment variables
    env_vars = ['AHGD_ENV', 'ENV', 'ENVIRONMENT', 'STAGE']
    for var in env_vars:
        if var in os.environ:
            return os.environ[var].lower()
    
    # Check for common deployment indicators
    if os.path.exists('/.dockerenv'):
        return 'production' if os.getenv('KUBERNETES_SERVICE_HOST') else 'staging'
    
    # Check for development indicators
    if os.path.exists('.git'):
        return 'development'
    
    # Default
    return 'development'


def is_production() -> bool:
    """Check if running in production environment"""
    return detect_environment() == 'production'


def is_development() -> bool:
    """Check if running in development environment"""
    return detect_environment() == 'development'


def is_testing() -> bool:
    """Check if running in testing environment"""
    return detect_environment() in ['test', 'testing']


# Quick setup functions for different environments
def setup_development_logging():
    """Quick setup for development environment"""
    config_loader = get_config_loader()
    config_loader.create_environment_directories('development')
    return config_loader.setup_environment_logging('development')


def setup_production_logging():
    """Quick setup for production environment"""
    config_loader = get_config_loader()
    config_loader.create_environment_directories('production')
    return config_loader.setup_environment_logging('production')


def setup_testing_logging():
    """Quick setup for testing environment"""
    config_loader = get_config_loader()
    config_loader.create_environment_directories('testing')
    return config_loader.setup_environment_logging('testing')


# Configuration validation CLI-like function
def print_config_summary(environment: Optional[str] = None):
    """Print configuration summary for debugging"""
    config_loader = get_config_loader()
    summary = config_loader.get_environment_summary(environment)
    
    print(f"AHGD Configuration Summary")
    print(f"=" * 40)
    print(f"Environment: {summary['environment']}")
    print(f"Detected Environment: {summary['detected_environment']}")
    print(f"Configuration Valid: {summary['configuration_valid']}")
    print(f"Log Level: {summary.get('log_level', 'N/A')}")
    print(f"Log Directory: {summary.get('log_directory', 'N/A')}")
    print(f"Console Logging: {summary.get('console_logging', False)}")
    print(f"JSON Logging: {summary.get('json_logging', False)}")
    print(f"Performance Logging: {summary.get('performance_logging', False)}")
    print(f"Lineage Tracking: {summary.get('lineage_tracking', False)}")
    print(f"Monitoring Enabled: {summary.get('monitoring_enabled', False)}")
    print(f"Health Checks Enabled: {summary.get('health_checks_enabled', False)}")
    print(f"Log Files Configured: {summary.get('log_files_configured', 0)}")
    
    if summary.get('integrations_available'):
        print(f"Available Integrations: {', '.join(summary['integrations_available'])}")
    
    if summary.get('validation_errors'):
        print(f"\nErrors:")
        for error in summary['validation_errors']:
            print(f"  - {error}")
    
    if summary.get('validation_warnings'):
        print(f"\nWarnings:")
        for warning in summary['validation_warnings']:
            print(f"  - {warning}")


# Generic configuration loading functions for compatibility with tests
def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ConfigurationError: If file cannot be loaded or parsed
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f) or {}
            elif config_path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {config_path.suffix}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file {config_path}: {e}")
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Failed to parse JSON file {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Error loading configuration file {config_path}: {e}")


def validate_config(config: Dict[str, Any], 
                   schema: Optional[Dict[str, Any]] = None,
                   type_constraints: Optional[Dict[str, type]] = None,
                   value_constraints: Optional[Dict[str, list]] = None) -> None:
    """
    Validate configuration against schema and constraints.
    
    Args:
        config: Configuration dictionary to validate
        schema: Schema dictionary with required/optional field definitions
        type_constraints: Dictionary mapping field paths to expected types
        value_constraints: Dictionary mapping field paths to allowed values
        
    Raises:
        ConfigurationError: If validation fails
    """
    errors = []
    
    # Schema validation
    if schema:
        errors.extend(_validate_schema(config, schema))
    
    # Type validation
    if type_constraints:
        errors.extend(_validate_types(config, type_constraints))
    
    # Value validation
    if value_constraints:
        errors.extend(_validate_values(config, value_constraints))
    
    if errors:
        raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")


def _validate_schema(config: Dict[str, Any], schema: Dict[str, Any], prefix: str = "") -> list:
    """Validate configuration against schema."""
    errors = []
    
    for key, requirement in schema.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(requirement, dict):
            # Nested schema
            if key in config and isinstance(config[key], dict):
                errors.extend(_validate_schema(config[key], requirement, full_key))
            elif "required" in str(requirement).lower():
                errors.append(f"Missing required section: {full_key}")
        elif requirement == "required":
            if key not in config:
                errors.append(f"Missing required configuration: {full_key}")
    
    return errors


def _validate_types(config: Dict[str, Any], type_constraints: Dict[str, type]) -> list:
    """Validate configuration types."""
    errors = []
    
    for path, expected_type in type_constraints.items():
        value = _get_nested_value(config, path)
        if value is not None and not isinstance(value, expected_type):
            errors.append(f"Invalid configuration type for {path}: expected {expected_type.__name__}, got {type(value).__name__}")
    
    return errors


def _validate_values(config: Dict[str, Any], value_constraints: Dict[str, list]) -> list:
    """Validate configuration values."""
    errors = []
    
    for path, allowed_values in value_constraints.items():
        value = _get_nested_value(config, path)
        if value is not None and value not in allowed_values:
            errors.append(f"Invalid configuration value for {path}: '{value}' not in {allowed_values}")
    
    return errors


def _get_nested_value(config: Dict[str, Any], path: str) -> Any:
    """Get nested value from configuration using dot notation."""
    keys = path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return None


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    
    for config in configs:
        result = _deep_merge(result, config)
    
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


if __name__ == "__main__":
    # CLI-like usage for debugging configuration
    import sys
    
    if len(sys.argv) > 1:
        environment = sys.argv[1]
        print_config_summary(environment)
    else:
        print_config_summary()
        
        print(f"\nAvailable commands:")
        print(f"  python -m src.utils.config_loader development")
        print(f"  python -m src.utils.config_loader production")
        print(f"  python -m src.utils.config_loader staging")
        print(f"  python -m src.utils.config_loader testing")