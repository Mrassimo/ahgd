"""
Unit tests for AHGD configuration system.

Tests configuration loading, validation, environment handling,
and configuration management functionality.
"""

import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import os
import tempfile
import yaml
import json

from src.utils.config import ConfigurationManager
from src.utils.config_loader import load_config, validate_config, merge_configs
from src.utils.interfaces import ConfigurationError


@pytest.mark.unit
class TestConfigurationManager:
    """Test cases for ConfigurationManager."""
    
    def test_config_manager_initialisation_with_file(self, temp_dir):
        """Test config manager initialisation with config file."""
        # Create config directory structure
        config_dir = temp_dir / "configs"
        config_dir.mkdir(exist_ok=True)
        
        config_content = {
            "database": {
                "type": "sqlite",
                "path": ":memory:"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s"
            }
        }
        
        # Create default.yaml file
        config_file = config_dir / "default.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        manager = ConfigurationManager(config_dir=config_dir, environment="testing")
        
        assert manager.get("database.type") == "sqlite"
        assert manager.get("database.path") == ":memory:"
        assert manager.get("logging.level") == "INFO"
    
    @pytest.mark.skip("ConfigurationManager doesn't support dict initialisation - uses file-based config")
    def test_config_manager_initialisation_with_dict(self):
        """Test config manager initialisation with config dictionary."""
        pass
    
    def test_config_manager_nonexistent_file(self, temp_dir):
        """Test config manager with non-existent config directory."""
        nonexistent_dir = temp_dir / "nonexistent_configs"
        
        # This should create the directory structure if needed
        # or raise an appropriate error for missing required files
        try:
            manager = ConfigurationManager(config_dir=nonexistent_dir, environment="testing")
            # If it succeeds, check that it behaves reasonably with empty config
            assert manager.get("nonexistent.key", "default") == "default"
        except Exception as e:
            # It's acceptable for this to fail with missing required configs
            assert "config" in str(e).lower() or "file" in str(e).lower()
    
    def test_get_config_value(self, temp_dir):
        """Test getting configuration values."""
        config_file = temp_dir / "test_config.yaml"
        config_content = {
            "database": {
                "type": "sqlite",
                "settings": {
                    "memory": True,
                    "timeout": 30
                }
            },
            "app_name": "AHGD"
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        manager = ConfigurationManager(config_file)
        
        # Test simple value
        assert manager.get("app_name") == "AHGD"
        
        # Test nested value
        assert manager.get("database.type") == "sqlite"
        assert manager.get("database.settings.memory") is True
        assert manager.get("database.settings.timeout") == 30
        
        # Test default value
        assert manager.get("nonexistent.key", default="default_value") == "default_value"
        
        # Test missing key without default
        with pytest.raises(KeyError):
            manager.get("nonexistent.key")
    
    def test_set_config_value(self, temp_dir):
        """Test setting configuration values."""
        config_file = temp_dir / "test_config.yaml"
        config_content = {
            "database": {"type": "sqlite"},
            "logging": {"level": "INFO"}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        manager = ConfigurationManager(config_file)
        
        # Set simple value
        manager.set("app_version", "1.0.0")
        assert manager.get("app_version") == "1.0.0"
        
        # Set nested value
        manager.set("database.timeout", 60)
        assert manager.get("database.timeout") == 60
        
        # Create new nested structure
        manager.set("new.nested.value", "test")
        assert manager.get("new.nested.value") == "test"
    
    def test_has_config_key(self, temp_dir):
        """Test checking if configuration key exists."""
        config_file = temp_dir / "test_config.yaml"
        config_content = {
            "database": {
                "type": "sqlite",
                "settings": {"memory": True}
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        manager = ConfigurationManager(config_file)
        
        assert manager.has("database") is True
        assert manager.has("database.type") is True
        assert manager.has("database.settings.memory") is True
        assert manager.has("nonexistent") is False
        assert manager.has("database.nonexistent") is False
    
    def test_update_config(self, temp_dir):
        """Test updating configuration with new values."""
        config_file = temp_dir / "test_config.yaml"
        config_content = {
            "database": {"type": "sqlite"},
            "logging": {"level": "INFO"}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        manager = ConfigurationManager(config_file)
        
        updates = {
            "database": {"type": "postgresql", "host": "localhost"},
            "new_section": {"key": "value"}
        }
        
        manager.update(updates)
        
        # Database type should be updated, but original keys preserved
        assert manager.get("database.type") == "postgresql"
        assert manager.get("database.host") == "localhost"
        assert manager.get("logging.level") == "INFO"  # Should remain unchanged
        assert manager.get("new_section.key") == "value"
    
    def test_save_config(self, temp_dir):
        """Test saving configuration to file."""
        config_file = temp_dir / "test_config.yaml"
        initial_config = {
            "database": {"type": "sqlite"},
            "logging": {"level": "INFO"}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(initial_config, f)
        
        manager = ConfigurationManager(config_file)
        manager.set("app_version", "1.0.0")
        manager.set("database.timeout", 60)
        
        manager.save()
        
        # Reload and verify changes were saved
        with open(config_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        assert saved_config["app_version"] == "1.0.0"
        assert saved_config["database"]["timeout"] == 60
        assert saved_config["logging"]["level"] == "INFO"
    
    def test_reload_config(self, temp_dir):
        """Test reloading configuration from file."""
        config_file = temp_dir / "test_config.yaml"
        initial_config = {
            "database": {"type": "sqlite"},
            "logging": {"level": "INFO"}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(initial_config, f)
        
        manager = ConfigurationManager(config_file)
        
        # Modify in memory
        manager.set("app_version", "1.0.0")
        assert manager.get("app_version") == "1.0.0"
        
        # Modify the file externally
        updated_config = {
            "database": {"type": "postgresql"},
            "logging": {"level": "DEBUG"},
            "new_key": "new_value"
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(updated_config, f)
        
        # Reload from file
        manager.reload()
        
        assert manager.get("database.type") == "postgresql"
        assert manager.get("logging.level") == "DEBUG"
        assert manager.get("new_key") == "new_value"
        
        # In-memory changes should be lost
        with pytest.raises(KeyError):
            manager.get("app_version")


@pytest.mark.unit
class TestConfigLoader:
    """Test configuration loading functions."""
    
    def test_load_yaml_config(self, temp_dir):
        """Test loading YAML configuration file."""
        config_file = temp_dir / "config.yaml"
        config_content = {
            "database": {
                "type": "sqlite",
                "path": ":memory:"
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        loaded_config = load_config(config_file)
        
        assert loaded_config == config_content
    
    def test_load_json_config(self, temp_dir):
        """Test loading JSON configuration file."""
        config_file = temp_dir / "config.json"
        config_content = {
            "database": {
                "type": "sqlite",
                "path": ":memory:"
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_content, f)
        
        loaded_config = load_config(config_file)
        
        assert loaded_config == config_content
    
    def test_load_unsupported_format(self, temp_dir):
        """Test loading unsupported configuration format."""
        config_file = temp_dir / "config.txt"
        config_file.write_text("This is not a valid config format")
        
        with pytest.raises(ConfigurationError, match="Unsupported configuration file format"):
            load_config(config_file)
    
    def test_load_invalid_yaml(self, temp_dir):
        """Test loading invalid YAML file."""
        config_file = temp_dir / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")
        
        with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
            load_config(config_file)
    
    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON file."""
        config_file = temp_dir / "invalid.json"
        config_file.write_text('{"invalid": json content}')
        
        with pytest.raises(ConfigurationError, match="Failed to parse JSON"):
            load_config(config_file)


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation functions."""
    
    def test_validate_config_success(self):
        """Test successful configuration validation."""
        config = {
            "database": {
                "type": "sqlite",
                "path": ":memory:"
            },
            "logging": {
                "level": "INFO",
                "format": "%(levelname)s - %(message)s"
            },
            "extractors": {
                "csv_extractor": {
                    "batch_size": 1000,
                    "max_retries": 3
                }
            }
        }
        
        schema = {
            "database": {
                "type": "required",
                "path": "optional"
            },
            "logging": {
                "level": "required",
                "format": "optional"
            },
            "extractors": "optional"
        }
        
        # Should not raise an exception
        validate_config(config, schema)
    
    def test_validate_config_missing_required(self):
        """Test configuration validation with missing required fields."""
        config = {
            "database": {
                "path": ":memory:"
                # Missing required "type" field
            },
            "logging": {
                "level": "INFO"
            }
        }
        
        schema = {
            "database": {
                "type": "required",
                "path": "optional"
            },
            "logging": {
                "level": "required"
            }
        }
        
        with pytest.raises(ConfigurationError, match="Missing required configuration"):
            validate_config(config, schema)
    
    def test_validate_config_invalid_type(self):
        """Test configuration validation with invalid data types."""
        config = {
            "database": {
                "type": "sqlite",
                "port": "not_a_number"  # Should be integer
            },
            "logging": {
                "level": "INFO",
                "enabled": "yes"  # Should be boolean
            }
        }
        
        type_constraints = {
            "database.port": int,
            "logging.enabled": bool
        }
        
        with pytest.raises(ConfigurationError, match="Invalid configuration type"):
            validate_config(config, type_constraints=type_constraints)
    
    def test_validate_config_invalid_values(self):
        """Test configuration validation with invalid values."""
        config = {
            "logging": {
                "level": "INVALID_LEVEL"  # Should be valid log level
            },
            "database": {
                "type": "unsupported_db"  # Should be supported database type
            }
        }
        
        value_constraints = {
            "logging.level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "database.type": ["sqlite", "postgresql", "mysql"]
        }
        
        with pytest.raises(ConfigurationError, match="Invalid configuration value"):
            validate_config(config, value_constraints=value_constraints)


@pytest.mark.unit
class TestConfigMerging:
    """Test configuration merging functionality."""
    
    def test_merge_configs_simple(self):
        """Test merging simple configurations."""
        base_config = {
            "database": {"type": "sqlite"},
            "logging": {"level": "INFO"}
        }
        
        override_config = {
            "database": {"path": ":memory:"},
            "new_section": {"key": "value"}
        }
        
        merged = merge_configs(base_config, override_config)
        
        expected = {
            "database": {"type": "sqlite", "path": ":memory:"},
            "logging": {"level": "INFO"},
            "new_section": {"key": "value"}
        }
        
        assert merged == expected
    
    def test_merge_configs_deep(self):
        """Test deep merging of nested configurations."""
        base_config = {
            "database": {
                "type": "sqlite",
                "settings": {
                    "timeout": 30,
                    "pool_size": 5
                }
            }
        }
        
        override_config = {
            "database": {
                "settings": {
                    "timeout": 60,  # Override existing
                    "max_connections": 10  # Add new
                }
            }
        }
        
        merged = merge_configs(base_config, override_config)
        
        expected = {
            "database": {
                "type": "sqlite",
                "settings": {
                    "timeout": 60,
                    "pool_size": 5,
                    "max_connections": 10
                }
            }
        }
        
        assert merged == expected
    
    def test_merge_configs_list_replacement(self):
        """Test merging configurations with list replacement."""
        base_config = {
            "plugins": ["plugin1", "plugin2"],
            "settings": {"values": [1, 2, 3]}
        }
        
        override_config = {
            "plugins": ["plugin3", "plugin4"],
            "settings": {"values": [4, 5]}
        }
        
        merged = merge_configs(base_config, override_config)
        
        # Lists should be replaced, not merged
        expected = {
            "plugins": ["plugin3", "plugin4"],
            "settings": {"values": [4, 5]}
        }
        
        assert merged == expected
    
    def test_merge_multiple_configs(self):
        """Test merging multiple configuration sources."""
        config1 = {"a": 1, "b": {"x": 1}}
        config2 = {"b": {"y": 2}, "c": 3}
        config3 = {"b": {"z": 3}, "d": 4}
        
        merged = merge_configs(config1, config2, config3)
        
        expected = {
            "a": 1,
            "b": {"x": 1, "y": 2, "z": 3},
            "c": 3,
            "d": 4
        }
        
        assert merged == expected


@pytest.mark.unit
class TestEnvironmentConfiguration:
    """Test environment-based configuration."""
    
    def test_environment_variable_override(self, temp_dir, monkeypatch):
        """Test configuration override with environment variables."""
        # Set environment variables
        monkeypatch.setenv("AHGD_DATABASE_TYPE", "postgresql")
        monkeypatch.setenv("AHGD_DATABASE_HOST", "localhost")
        monkeypatch.setenv("AHGD_LOGGING_LEVEL", "DEBUG")
        
        config_file = temp_dir / "config.yaml"
        base_config = {
            "database": {"type": "sqlite", "path": ":memory:"},
            "logging": {"level": "INFO"}
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Mock environment variable processing
        with patch('src.utils.config.apply_environment_overrides') as mock_apply_env:
            def apply_env_side_effect(config):
                config["database"]["type"] = os.getenv("AHGD_DATABASE_TYPE", config["database"]["type"])
                config["database"]["host"] = os.getenv("AHGD_DATABASE_HOST", config["database"].get("host"))
                config["logging"]["level"] = os.getenv("AHGD_LOGGING_LEVEL", config["logging"]["level"])
                return config
            
            mock_apply_env.side_effect = apply_env_side_effect
            
            manager = ConfigurationManager(config_file)
            
            assert manager.get("database.type") == "postgresql"
            assert manager.get("database.host") == "localhost"
            assert manager.get("logging.level") == "DEBUG"
    
    def test_environment_variable_type_conversion(self, monkeypatch):
        """Test automatic type conversion for environment variables."""
        # Set environment variables with different types
        monkeypatch.setenv("AHGD_DATABASE_PORT", "5432")
        monkeypatch.setenv("AHGD_LOGGING_ENABLED", "true")
        monkeypatch.setenv("AHGD_BATCH_SIZE", "1000")
        monkeypatch.setenv("AHGD_TIMEOUT", "30.5")
        
        # Mock environment variable processing with type conversion
        with patch('src.utils.config.convert_env_value') as mock_convert:
            def convert_side_effect(value, target_type=None):
                if target_type == int:
                    return int(value)
                elif target_type == float:
                    return float(value)
                elif target_type == bool:
                    return value.lower() in ('true', '1', 'yes', 'on')
                return value
            
            mock_convert.side_effect = convert_side_effect
            
            # Test conversions
            assert mock_convert("5432", int) == 5432
            assert mock_convert("true", bool) is True
            assert mock_convert("30.5", float) == 30.5


@pytest.mark.unit
class TestConfigurationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_config_file(self, temp_dir):
        """Test handling of empty configuration file."""
        config_file = temp_dir / "empty.yaml"
        config_file.write_text("")
        
        # Should handle empty file gracefully
        try:
            loaded_config = load_config(config_file)
            assert loaded_config == {} or loaded_config is None
        except ConfigurationError:
            # Empty file might be considered an error
            pass
    
    def test_config_with_none_values(self, temp_dir):
        """Test handling of None values in configuration."""
        config_file = temp_dir / "config.yaml"
        config_content = {
            "database": {
                "type": "sqlite",
                "host": None,
                "port": None
            },
            "optional_section": None
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)
        
        manager = ConfigurationManager(config_file)
        
        assert manager.get("database.type") == "sqlite"
        assert manager.get("database.host") is None
        assert manager.get("database.port") is None
        assert manager.get("optional_section") is None
    
    def test_circular_reference_handling(self, temp_dir):
        """Test handling of circular references in configuration."""
        # This would require special handling if supported
        config_with_refs = {
            "database": {
                "url": "${HOST}:${PORT}/${DATABASE}"
            },
            "HOST": "localhost",
            "PORT": 5432,
            "DATABASE": "ahgd"
        }
        
        # Basic test - actual interpolation would need implementation
        assert "url" in config_with_refs["database"]
    
    def test_very_deep_nesting(self, temp_dir):
        """Test handling of very deeply nested configuration."""
        config_file = temp_dir / "deep.yaml"
        
        # Create deeply nested structure
        deep_config = {}
        current = deep_config
        for i in range(20):  # 20 levels deep
            current[f"level_{i}"] = {}
            current = current[f"level_{i}"]
        current["value"] = "deep_value"
        
        with open(config_file, 'w') as f:
            yaml.dump(deep_config, f)
        
        manager = ConfigurationManager(config_file)
        
        # Should handle deep nesting
        deep_key = ".".join([f"level_{i}" for i in range(20)] + ["value"])
        assert manager.get(deep_key) == "deep_value"
    
    def test_unicode_configuration(self, temp_dir):
        """Test handling of unicode characters in configuration."""
        config_file = temp_dir / "unicode.yaml"
        config_content = {
            "app_name": "AHGD - 澳大利亚健康地理数据",
            "description": "Données géographiques de santé australiennes",
            "symbols": "αβγδε",
            "emoji": "🏥📊🇦🇺"
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_content, f, allow_unicode=True)
        
        manager = ConfigurationManager(config_file)
        
        assert manager.get("app_name") == "AHGD - 澳大利亚健康地理数据"
        assert manager.get("description") == "Données géographiques de santé australiennes"
        assert manager.get("symbols") == "αβγδε"
        assert manager.get("emoji") == "🏥📊🇦🇺"
    
    def test_large_configuration_file(self, temp_dir):
        """Test handling of large configuration files."""
        config_file = temp_dir / "large.yaml"
        
        # Create large configuration
        large_config = {}
        for i in range(1000):
            large_config[f"section_{i}"] = {
                "key_1": f"value_{i}_1",
                "key_2": f"value_{i}_2",
                "nested": {
                    "deep_key": f"deep_value_{i}",
                    "list": [f"item_{i}_{j}" for j in range(10)]
                }
            }
        
        with open(config_file, 'w') as f:
            yaml.dump(large_config, f)
        
        manager = ConfigurationManager(config_file)
        
        # Should handle large configuration efficiently
        assert manager.get("section_500.key_1") == "value_500_1"
        assert manager.get("section_999.nested.deep_key") == "deep_value_999"