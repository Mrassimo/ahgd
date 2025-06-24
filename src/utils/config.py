"""
AHGD Comprehensive Configuration Management System

This module provides a comprehensive configuration management system with:
- Environment-specific configuration loading
- YAML/JSON/ENV file support
- Configuration validation
- Environment variable overrides
- Secrets management integration
- Configuration hot-reloading
- Type safety and validation
"""

import os
import json
import yaml
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable, Type, Generic, TypeVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging

# Type definitions
T = TypeVar('T')
ConfigDict = Dict[str, Any]

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Supported environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class ConfigFormat(Enum):
    """Supported configuration file formats"""
    YAML = "yaml"
    JSON = "json"
    ENV = "env"


@dataclass
class ConfigSource:
    """Configuration source metadata"""
    file_path: Path
    format: ConfigFormat
    priority: int = 100  # Lower number = higher priority
    required: bool = True
    loaded_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None


@dataclass
class ValidationRule:
    """Configuration validation rule"""
    path: str  # Dot notation path, e.g., "database.url"
    required: bool = False
    type_check: Optional[Type] = None
    validator: Optional[Callable[[Any], bool]] = None
    default: Any = None
    description: str = ""


class ConfigurationError(Exception):
    """Configuration-related errors"""
    pass


class ValidationError(ConfigurationError):
    """Configuration validation errors"""
    pass


class ConfigFileWatcher(FileSystemEventHandler):
    """File system watcher for configuration hot-reloading"""
    
    def __init__(self, config_manager: 'ConfigurationManager'):
        self.config_manager = config_manager
        self._last_reload = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        file_path = Path(event.src_path)
        
        # Debounce: only reload if it's been more than 1 second
        now = time.time()
        if file_path in self._last_reload:
            if now - self._last_reload[file_path] < 1.0:
                return
        
        self._last_reload[file_path] = now
        
        # Check if this is a watched config file
        for source in self.config_manager._sources:
            if source.file_path.resolve() == file_path.resolve():
                logger.info(f"Configuration file changed: {file_path}")
                self.config_manager._reload_configuration()
                break


class ConfigurationManager:
    """
    Comprehensive configuration manager with environment support,
    validation, hot-reloading, and secrets integration.
    """
    
    def __init__(
        self,
        config_dir: Union[str, Path] = "configs",
        environment: Optional[str] = None,
        enable_hot_reload: bool = False,
        enable_env_override: bool = True
    ):
        self.config_dir = Path(config_dir)
        self.environment = self._detect_environment(environment)
        self.enable_hot_reload = enable_hot_reload
        self.enable_env_override = enable_env_override
        
        # Internal state
        self._config: ConfigDict = {}
        self._sources: List[ConfigSource] = []
        self._validation_rules: List[ValidationRule] = []
        self._observers: List[Observer] = []
        self._lock = threading.RLock()
        self._loaded_at: Optional[datetime] = None
        self._reload_callbacks: List[Callable[[ConfigDict], None]] = []
        
        # Initialize secrets manager
        from .secrets import SecretsManager
        self._secrets_manager = SecretsManager()
        
        # Load initial configuration
        self._initialize()
    
    def _detect_environment(self, environment: Optional[str] = None) -> Environment:
        """Detect current environment from various sources"""
        if environment:
            return Environment(environment.lower())
        
        # Check environment variables
        env_vars = ['AHGD_ENV', 'ENV', 'ENVIRONMENT', 'STAGE']
        for var in env_vars:
            if var in os.environ:
                try:
                    return Environment(os.environ[var].lower())
                except ValueError:
                    logger.warning(f"Invalid environment value in {var}: {os.environ[var]}")
        
        # Check for deployment indicators
        if os.path.exists('/.dockerenv'):
            return Environment.PRODUCTION if os.getenv('KUBERNETES_SERVICE_HOST') else Environment.STAGING
        
        # Check for development indicators
        if os.path.exists('.git'):
            return Environment.DEVELOPMENT
        
        # Default
        return Environment.DEVELOPMENT
    
    def _initialize(self):
        """Initialize configuration system"""
        # Load .env files into environment variables first
        self._load_env_files()
        
        self._discover_config_sources()
        self._load_configuration()
        self._validate_configuration()
        
        if self.enable_hot_reload:
            self._setup_hot_reload()
    
    def _load_env_files(self):
        """Load .env files into environment variables using python-dotenv"""
        try:
            from dotenv import load_dotenv
            
            # List of .env files to try loading (in priority order)
            env_files = [
                self.config_dir / ".env",  # Project-specific .env
                Path(".env"),  # Root .env
                Path.home() / ".ahgd" / ".env",  # User-specific .env
            ]
            
            loaded_count = 0
            for env_file in env_files:
                if env_file.exists():
                    try:
                        load_dotenv(env_file, override=False)  # Don't override existing env vars
                        logger.debug(f"Loaded environment variables from: {env_file}")
                        loaded_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load .env file {env_file}: {e}")
            
            if loaded_count > 0:
                logger.info(f"Loaded {loaded_count} .env file(s) into environment variables")
            
        except ImportError:
            logger.debug("python-dotenv not available, skipping .env file loading")
        except Exception as e:
            logger.warning(f"Error loading .env files: {e}")
    
    def _discover_config_sources(self):
        """Discover and register configuration sources"""
        # Default configuration files to look for
        config_files = [
            ("default.yaml", 100, True),
            (f"{self.environment.value}.yaml", 50, False),
            ("local.yaml", 10, False),  # Highest priority, optional
            (".env", 200, False),
        ]
        
        for filename, priority, required in config_files:
            file_path = self.config_dir / filename
            if file_path.exists() or required:
                format_type = self._detect_format(file_path)
                source = ConfigSource(
                    file_path=file_path,
                    format=format_type,
                    priority=priority,
                    required=required
                )
                self._sources.append(source)
        
        # Sort by priority (lower number = higher priority)
        self._sources.sort(key=lambda x: x.priority)
    
    def _detect_format(self, file_path: Path) -> ConfigFormat:
        """Detect configuration file format"""
        suffix = file_path.suffix.lower()
        if suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        elif suffix == '.json':
            return ConfigFormat.JSON
        elif suffix == '.env' or file_path.name == '.env':
            return ConfigFormat.ENV
        else:
            # Default to YAML
            return ConfigFormat.YAML
    
    def _load_configuration(self):
        """Load configuration from all sources"""
        with self._lock:
            merged_config = {}
            
            for source in self._sources:
                if not source.file_path.exists():
                    if source.required:
                        raise ConfigurationError(f"Required configuration file not found: {source.file_path}")
                    continue
                
                try:
                    config_data = self._load_file(source)
                    source.loaded_at = datetime.now()
                    source.last_modified = datetime.fromtimestamp(source.file_path.stat().st_mtime)
                    
                    # Merge configuration (later sources override earlier ones due to sorting)
                    merged_config = self._deep_merge(merged_config, config_data)
                    
                    logger.debug(f"Loaded configuration from: {source.file_path}")
                    
                except Exception as e:
                    if source.required:
                        raise ConfigurationError(f"Failed to load required config {source.file_path}: {e}")
                    else:
                        logger.warning(f"Failed to load optional config {source.file_path}: {e}")
            
            # Apply environment variable overrides
            if self.enable_env_override:
                merged_config = self._apply_env_overrides(merged_config)
            
            # Resolve secrets
            merged_config = self._resolve_secrets(merged_config)
            
            self._config = merged_config
            self._loaded_at = datetime.now()
            
            logger.info(f"Configuration loaded successfully for environment: {self.environment.value}")
    
    def _load_file(self, source: ConfigSource) -> ConfigDict:
        """Load configuration from a single file"""
        try:
            with open(source.file_path, 'r', encoding='utf-8') as f:
                if source.format == ConfigFormat.YAML:
                    return yaml.safe_load(f) or {}
                elif source.format == ConfigFormat.JSON:
                    return json.load(f)
                elif source.format == ConfigFormat.ENV:
                    return self._parse_env_file(f)
                else:
                    raise ConfigurationError(f"Unsupported format: {source.format}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"YAML parsing error in {source.file_path}: {e}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"JSON parsing error in {source.file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading {source.file_path}: {e}")
    
    def _parse_env_file(self, file_handle) -> ConfigDict:
        """Parse .env file format"""
        config = {}
        for line_num, line in enumerate(file_handle, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if '=' not in line:
                logger.warning(f"Invalid .env format at line {line_num}: {line}")
                continue
            
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            
            # Convert dot notation to nested dict
            self._set_nested_value(config, key, value)
        
        return config
    
    def _deep_merge(self, base: ConfigDict, override: ConfigDict) -> ConfigDict:
        """Deep merge two configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: ConfigDict) -> ConfigDict:
        """Apply environment variable overrides"""
        result = config.copy()
        
        # Look for environment variables with AHGD_ prefix
        for key, value in os.environ.items():
            if key.startswith('AHGD_'):
                config_key = key[5:].lower()  # Remove AHGD_ prefix
                
                # Convert underscores to dot notation
                config_path = config_key.replace('__', '.')
                
                # Try to convert value to appropriate type
                converted_value = self._convert_env_value(value)
                
                # Set the value in nested config
                self._set_nested_value(result, config_path, converted_value)
                
                logger.debug(f"Environment override: {config_path} = {converted_value}")
        
        return result
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON values
        if value.startswith(('{', '[', '"')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Default: string
        return value
    
    def _set_nested_value(self, config: ConfigDict, path: str, value: Any):
        """Set a value in nested configuration using dot notation"""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Convert to dict if not already
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_nested_value(self, config: ConfigDict, path: str, default: Any = None) -> Any:
        """Get a value from nested configuration using dot notation"""
        keys = path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def _resolve_secrets(self, config: ConfigDict) -> ConfigDict:
        """Resolve secret references in configuration"""
        def _resolve_recursive(obj):
            if isinstance(obj, dict):
                return {k: _resolve_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_resolve_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                # Secret reference format: ${secret:key_name}
                if obj.startswith('${secret:'):
                    secret_key = obj[9:-1]  # Remove ${secret: and }
                    return self._secrets_manager.get_secret(secret_key)
                # Environment variable reference: ${env:VAR_NAME}
                elif obj.startswith('${env:'):
                    env_var = obj[6:-1]  # Remove ${env: and }
                    return os.getenv(env_var, obj)  # Return original if not found
            return obj
        
        return _resolve_recursive(config)
    
    def _validate_configuration(self):
        """Validate loaded configuration against rules"""
        errors = []
        
        for rule in self._validation_rules:
            value = self._get_nested_value(self._config, rule.path, rule.default)
            
            # Check if required
            if rule.required and value is None:
                errors.append(f"Required configuration '{rule.path}' is missing")
                continue
            
            # Skip validation if value is None and not required
            if value is None:
                continue
            
            # Type checking
            if rule.type_check and not isinstance(value, rule.type_check):
                errors.append(f"Configuration '{rule.path}' must be of type {rule.type_check.__name__}, got {type(value).__name__}")
            
            # Custom validation
            if rule.validator and not rule.validator(value):
                errors.append(f"Configuration '{rule.path}' failed validation: {rule.description}")
        
        if errors:
            raise ValidationError(f"Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def _setup_hot_reload(self):
        """Setup file system watching for hot-reload"""
        if not self._sources:
            return
        
        try:
            observer = Observer()
            handler = ConfigFileWatcher(self)
            
            # Watch each config file's directory
            watched_dirs = set()
            for source in self._sources:
                if source.file_path.exists():
                    parent_dir = source.file_path.parent
                    if parent_dir not in watched_dirs:
                        observer.schedule(handler, str(parent_dir), recursive=False)
                        watched_dirs.add(parent_dir)
            
            observer.start()
            self._observers.append(observer)
            
            logger.info(f"Hot-reload enabled for {len(watched_dirs)} directories")
            
        except Exception as e:
            logger.warning(f"Failed to setup hot-reload: {e}")
    
    def _reload_configuration(self):
        """Reload configuration from sources"""
        try:
            old_config = self._config.copy()
            self._load_configuration()
            self._validate_configuration()
            
            # Notify callbacks
            for callback in self._reload_callbacks:
                try:
                    callback(self._config)
                except Exception as e:
                    logger.error(f"Error in reload callback: {e}")
            
            logger.info("Configuration reloaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            # Restore old configuration
            self._config = old_config
    
    # Public API
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        with self._lock:
            return self._get_nested_value(self._config, path, default)
    
    def get_typed(self, path: str, type_class: Type[T], default: Optional[T] = None) -> T:
        """Get configuration value with type checking"""
        value = self.get(path, default)
        if value is None:
            if default is not None:
                return default
            raise ConfigurationError(f"Configuration '{path}' not found")
        
        if not isinstance(value, type_class):
            try:
                # Try to convert
                return type_class(value)
            except (ValueError, TypeError):
                raise ConfigurationError(f"Configuration '{path}' cannot be converted to {type_class.__name__}")
        
        return value
    
    def get_section(self, path: str) -> ConfigDict:
        """Get a configuration section as dictionary"""
        section = self.get(path, {})
        if not isinstance(section, dict):
            raise ConfigurationError(f"Configuration section '{path}' is not a dictionary")
        return section
    
    def has(self, path: str) -> bool:
        """Check if configuration path exists"""
        return self._get_nested_value(self._config, path) is not None
    
    def set(self, path: str, value: Any, persistent: bool = False):
        """Set configuration value (runtime only unless persistent=True)"""
        with self._lock:
            self._set_nested_value(self._config, path, value)
            
            if persistent:
                # Save to local.yaml for persistence
                local_config_path = self.config_dir / "local.yaml"
                try:
                    # Load existing local config
                    if local_config_path.exists():
                        with open(local_config_path, 'r') as f:
                            local_config = yaml.safe_load(f) or {}
                    else:
                        local_config = {}
                    
                    # Update with new value
                    self._set_nested_value(local_config, path, value)
                    
                    # Save back
                    local_config_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(local_config_path, 'w') as f:
                        yaml.dump(local_config, f, default_flow_style=False)
                    
                    logger.info(f"Configuration '{path}' saved persistently")
                    
                except Exception as e:
                    logger.error(f"Failed to save persistent configuration: {e}")
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a configuration validation rule"""
        self._validation_rules.append(rule)
    
    def add_reload_callback(self, callback: Callable[[ConfigDict], None]):
        """Add a callback to be called when configuration is reloaded"""
        self._reload_callbacks.append(callback)
    
    def reload(self):
        """Manually reload configuration"""
        self._reload_configuration()
    
    def get_info(self) -> Dict[str, Any]:
        """Get configuration system information"""
        with self._lock:
            return {
                'environment': self.environment.value,
                'config_dir': str(self.config_dir),
                'sources_loaded': len([s for s in self._sources if s.loaded_at]),
                'total_sources': len(self._sources),
                'loaded_at': self._loaded_at.isoformat() if self._loaded_at else None,
                'hot_reload_enabled': self.enable_hot_reload,
                'env_override_enabled': self.enable_env_override,
                'validation_rules': len(self._validation_rules),
                'sources': [
                    {
                        'path': str(source.file_path),
                        'format': source.format.value,
                        'priority': source.priority,
                        'required': source.required,
                        'loaded': source.loaded_at.isoformat() if source.loaded_at else None,
                        'exists': source.file_path.exists()
                    }
                    for source in self._sources
                ]
            }
    
    def export_config(self, format_type: str = "yaml", include_secrets: bool = False) -> str:
        """Export current configuration to string"""
        config_copy = self._config.copy()
        
        if not include_secrets:
            # Remove sensitive data (basic implementation)
            config_copy = self._sanitize_config(config_copy)
        
        if format_type.lower() == "json":
            return json.dumps(config_copy, indent=2, default=str)
        else:  # YAML
            return yaml.dump(config_copy, default_flow_style=False)
    
    def _sanitize_config(self, config: ConfigDict) -> ConfigDict:
        """Remove sensitive information from configuration"""
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credentials', 'auth']
        
        def _sanitize_recursive(obj):
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if any(sensitive in k.lower() for sensitive in sensitive_keys):
                        result[k] = "***REDACTED***"
                    else:
                        result[k] = _sanitize_recursive(v)
                return result
            elif isinstance(obj, list):
                return [_sanitize_recursive(item) for item in obj]
            return obj
        
        return _sanitize_recursive(config)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup observers"""
        for observer in self._observers:
            observer.stop()
            observer.join()


# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None


def get_config_manager(**kwargs) -> ConfigurationManager:
    """Get or create global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(**kwargs)
    return _config_manager


def get_config(path: str, default: Any = None) -> Any:
    """Convenience function to get configuration value"""
    return get_config_manager().get(path, default)


def get_typed_config(path: str, type_class: Type[T], default: Optional[T] = None) -> T:
    """Convenience function to get typed configuration value"""
    return get_config_manager().get_typed(path, type_class, default)


def get_config_section(path: str) -> ConfigDict:
    """Convenience function to get configuration section"""
    return get_config_manager().get_section(path)


def has_config(path: str) -> bool:
    """Convenience function to check if configuration exists"""
    return get_config_manager().has(path)


def set_config(path: str, value: Any, persistent: bool = False):
    """Convenience function to set configuration value"""
    get_config_manager().set(path, value, persistent)


# Environment detection utilities
def get_environment() -> Environment:
    """Get current environment"""
    return get_config_manager().environment


def is_production() -> bool:
    """Check if running in production environment"""
    return get_environment() == Environment.PRODUCTION


def is_development() -> bool:
    """Check if running in development environment"""
    return get_environment() == Environment.DEVELOPMENT


def is_testing() -> bool:
    """Check if running in testing environment"""
    return get_environment() == Environment.TESTING


def is_staging() -> bool:
    """Check if running in staging environment"""
    return get_environment() == Environment.STAGING


# Configuration decorators
def config_required(*config_paths: str):
    """Decorator to ensure required configuration is present"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            missing = []
            for path in config_paths:
                if not has_config(path):
                    missing.append(path)
            
            if missing:
                raise ConfigurationError(f"Required configuration missing: {', '.join(missing)}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def with_config(config_path: str, default: Any = None):
    """Decorator to inject configuration value as function argument"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config_value = get_config(config_path, default)
            return func(*args, config_value, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # CLI usage for testing configuration
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "info":
            config_manager = get_config_manager()
            info = config_manager.get_info()
            print(json.dumps(info, indent=2))
        
        elif command == "export":
            format_type = sys.argv[2] if len(sys.argv) > 2 else "yaml"
            config_manager = get_config_manager()
            print(config_manager.export_config(format_type))
        
        elif command == "get":
            if len(sys.argv) < 3:
                print("Usage: python config.py get <path>")
                sys.exit(1)
            path = sys.argv[2]
            value = get_config(path)
            print(f"{path}: {value}")
        
        elif command == "validate":
            try:
                config_manager = get_config_manager()
                print("Configuration validation passed")
            except ValidationError as e:
                print(f"Validation failed: {e}")
                sys.exit(1)
        
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    else:
        print("AHGD Configuration Manager")
        print("Commands:")
        print("  info     - Show configuration system info")
        print("  export   - Export current configuration")
        print("  get      - Get configuration value")
        print("  validate - Validate configuration")


# =============================================================================
# COMPATIBILITY LAYER - Functions from config_loader.py for backwards compatibility
# =============================================================================

# Legacy ConfigLoader class for backwards compatibility
class ConfigLoader:
    """Legacy compatibility class matching old config_loader.py interface"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.environment = detect_environment()
        self._config_cache = {}
    
    def _detect_environment(self) -> str:
        """Detect current environment from environment variables"""
        return detect_environment()
    
    def load_logging_config(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Load logging configuration for specified environment"""
        env = environment or self.environment
        config_file = self.config_dir / "logging_config.yaml"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Logging config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            full_config = yaml.safe_load(f)
        
        if env not in full_config:
            available_envs = list(full_config.keys())
            raise ValueError(f"Environment '{env}' not found in config. Available: {available_envs}")
        
        env_config = full_config[env].copy()
        
        # Merge with common configurations
        common_configs = ['monitoring', 'health_checks', 'sampling', 
                         'enrichment', 'security', 'performance', 'integrations']
        
        for common_key in common_configs:
            if common_key in full_config:
                env_config[common_key] = full_config[common_key]
        
        return env_config
    
    def validate_configuration(self, environment: Optional[str] = None) -> Dict[str, Any]:
        """Validate configuration for environment - stub implementation"""
        return {
            'environment': environment or self.environment,
            'valid': True,
            'errors': [],
            'warnings': []
        }
    
    def create_environment_directories(self, environment: Optional[str] = None):
        """Create environment directories - stub implementation"""
        pass


# Global config loader instance for backwards compatibility
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
    from .logging import AHGDLogger
    config = load_logging_config(environment)
    return AHGDLogger(config_dict=config)


def validate_environment_config(environment: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to validate environment configuration"""
    return get_config_loader().validate_configuration(environment)


def create_log_directories(environment: Optional[str] = None):
    """Convenience function to create required log directories"""
    return get_config_loader().create_environment_directories(environment)


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


def setup_development_logging():
    """Quick setup for development environment"""
    config_loader = get_config_loader()
    return setup_environment_logging('development')


def setup_production_logging():
    """Quick setup for production environment"""
    config_loader = get_config_loader()
    return setup_environment_logging('production')


def setup_testing_logging():
    """Quick setup for testing environment"""
    config_loader = get_config_loader()
    return setup_environment_logging('testing')


def print_config_summary(environment: Optional[str] = None):
    """Print configuration summary for debugging"""
    env = environment or detect_environment()
    
    print(f"AHGD Configuration Summary")
    print(f"=" * 40)
    print(f"Environment: {env}")
    print(f"Detected Environment: {detect_environment()}")
    print(f"Is Production: {is_production()}")
    print(f"Is Development: {is_development()}")
    print(f"Is Testing: {is_testing()}")


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from a file"""
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
    """Validate configuration against schema and constraints"""
    errors = []
    
    # Basic validation - can be extended as needed
    if schema:
        for key, requirement in schema.items():
            if requirement == "required" and key not in config:
                errors.append(f"Missing required configuration: {key}")
    
    if errors:
        raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries"""
    result = {}
    
    for config in configs:
        result = _deep_merge(result, config)
    
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result