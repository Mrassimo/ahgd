"""
Configuration loader for AHGD ETL pipeline.

This module loads and provides access to configuration settings from YAML files
and environment variables. It serves as the central point for all configuration
in the AHGD ETL pipeline.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml
from dotenv import load_dotenv
import polars as pl

# Set up logger
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Map string data types to Polars types
TYPE_MAPPING = {
    'Int64': pl.Int64,
    'Int32': pl.Int32,
    'Int16': pl.Int16,
    'Int8': pl.Int8,
    'UInt64': pl.UInt64,
    'UInt32': pl.UInt32,
    'UInt16': pl.UInt16,
    'UInt8': pl.UInt8,
    'Float64': pl.Float64,
    'Float32': pl.Float32,
    'Boolean': pl.Boolean,
    'Utf8': pl.Utf8,
    'Categorical': pl.Categorical,
    'Date': pl.Date,
    'Datetime': pl.Datetime,
    'Time': pl.Time
}

class ConfigManager:
    """
    Configuration manager for AHGD ETL pipeline.
    
    This class loads and provides access to all configuration settings.
    It loads YAML configuration files and environment variables.
    """
    
    def __init__(self, config_dir: Optional[Path] = None, base_dir: Optional[Path] = None):
        """
        Initialize the ConfigManager.
        
        Args:
            config_dir: Directory containing YAML configuration files
            base_dir: Base directory for the ETL pipeline
        """
        # Determine base directory
        self.base_dir = Path(base_dir) if base_dir else Path(os.getenv('BASE_DIR', '.')).resolve()
        
        # Determine config directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Look for config in standard locations
            script_dir = Path(__file__).parent
            self.config_dir = script_dir / "yaml"
        
        # Initialize configuration dictionaries
        self.schemas = {}
        self.column_mappings = {}
        self.data_sources = {}
        self.paths = {}
        
        # Initialize the directories and load configurations
        self._init_directories()
        self._load_configs()
    
    def _init_directories(self):
        """Initialize directory paths."""
        # Define relative paths from base directory
        relative_paths = {
            'DATA_DIR': 'data',
            'RAW_DATA_DIR': 'data/raw',
            'OUTPUT_DIR': 'output',
            'TEMP_DIR': 'data/raw/temp',
            'LOG_DIR': 'logs',
            'GEOGRAPHIC_DIR': 'data/raw/geographic',
            'CENSUS_DIR': 'data/raw/census',
            'TEMP_ZIP_DIR': 'data/raw/temp/zips',
            'TEMP_EXTRACT_DIR': 'data/raw/temp/extract'
        }
        
        # Override with environment variables if available
        for key in relative_paths:
            env_value = os.getenv(key)
            if env_value:
                if os.path.isabs(env_value):
                    # Absolute path specified in env var
                    self.paths[key] = Path(env_value)
                else:
                    # Relative path specified in env var
                    self.paths[key] = self.base_dir / env_value
            else:
                # Use default relative path
                self.paths[key] = self.base_dir / relative_paths[key]
                
        # Ensure directories exist
        for path_name, path in self.paths.items():
            path.mkdir(parents=True, exist_ok=True)
    
    def _load_configs(self):
        """Load configuration from YAML files."""
        # Load schemas
        schema_path = self.config_dir / "schemas.yaml"
        if schema_path.exists():
            try:
                with open(schema_path, 'r') as f:
                    raw_schemas = yaml.safe_load(f)
                
                # Convert data types from strings to Polars types
                self.schemas = self._convert_schema_types(raw_schemas)
                logger.info(f"Loaded schemas from {schema_path}")
            except Exception as e:
                logger.error(f"Error loading schemas: {e}")
        else:
            logger.warning(f"Schema file not found: {schema_path}")
        
        # Load column mappings
        mappings_path = self.config_dir / "column_mappings.yaml"
        if mappings_path.exists():
            try:
                with open(mappings_path, 'r') as f:
                    self.column_mappings = yaml.safe_load(f)
                logger.info(f"Loaded column mappings from {mappings_path}")
            except Exception as e:
                logger.error(f"Error loading column mappings: {e}")
        else:
            logger.warning(f"Column mappings file not found: {mappings_path}")
        
        # Load data sources
        sources_path = self.config_dir / "data_sources.yaml"
        if sources_path.exists():
            try:
                with open(sources_path, 'r') as f:
                    self.data_sources = yaml.safe_load(f)
                logger.info(f"Loaded data sources from {sources_path}")
            except Exception as e:
                logger.error(f"Error loading data sources: {e}")
        else:
            logger.warning(f"Data sources file not found: {sources_path}")
    
    def _convert_schema_types(self, raw_schemas: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert schema data types from strings to Polars types.
        
        Args:
            raw_schemas: Raw schema dictionary loaded from YAML
            
        Returns:
            Schema dictionary with converted data types
        """
        converted_schemas = {}
        
        # Process dimension schemas
        if 'dimensions' in raw_schemas:
            converted_schemas['dimensions'] = {}
            for dim_name, dim_schema in raw_schemas['dimensions'].items():
                converted_schemas['dimensions'][dim_name] = {}
                for col_name, col_type in dim_schema.items():
                    # Extract the type string (remove comments)
                    if isinstance(col_type, str):
                        type_str = col_type.split('#')[0].strip()
                        converted_schemas['dimensions'][dim_name][col_name] = TYPE_MAPPING.get(type_str, pl.Utf8)
                    else:
                        # If already a Polars type or something else, keep as is
                        converted_schemas['dimensions'][dim_name][col_name] = col_type
        
        # Process fact schemas
        if 'facts' in raw_schemas:
            converted_schemas['facts'] = {}
            for fact_name, fact_schema in raw_schemas['facts'].items():
                converted_schemas['facts'][fact_name] = {}
                for col_name, col_type in fact_schema.items():
                    # Extract the type string (remove comments)
                    if isinstance(col_type, str):
                        type_str = col_type.split('#')[0].strip()
                        converted_schemas['facts'][fact_name][col_name] = TYPE_MAPPING.get(type_str, pl.Utf8)
                    else:
                        # If already a Polars type or something else, keep as is
                        converted_schemas['facts'][fact_name][col_name] = col_type
        
        return converted_schemas
    
    def get_schema(self, table_name: str) -> Dict[str, Any]:
        """
        Get schema for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Schema dictionary mapping column names to data types
        """
        # Determine if this is a dimension or fact table
        if table_name.startswith('dim_') or table_name == 'geo_dimension':
            return self.schemas.get('dimensions', {}).get(table_name, {})
        else:
            return self.schemas.get('facts', {}).get(table_name, {})
    
    def get_column_mapping(self, table_code: str) -> Dict[str, Any]:
        """
        Get column mapping for a specific Census table.
        
        Args:
            table_code: Census table code (e.g., 'G01', 'G19')
            
        Returns:
            Column mapping configuration
        """
        return self.column_mappings.get(table_code, {})
    
    def get_data_source_url(self, source_key: str) -> str:
        """
        Get URL for a specific data source.
        
        Args:
            source_key: Key identifying the data source
            
        Returns:
            URL string
        """
        # Check Census URLs
        census_urls = self.data_sources.get('census_urls', {})
        if source_key in census_urls:
            return census_urls[source_key]
        
        # Check ASGS URLs
        asgs_urls = self.data_sources.get('asgs2021_urls', {})
        if source_key in asgs_urls:
            return asgs_urls[source_key]
        
        # Not found
        return ""
    
    def get_path(self, path_name: str) -> Path:
        """
        Get a specific file system path.
        
        Args:
            path_name: Name of the path (e.g., 'OUTPUT_DIR', 'CENSUS_DIR')
            
        Returns:
            Path object
        """
        return self.paths.get(path_name, self.base_dir)

# Default config manager instance
_config_manager = None

def get_config_manager() -> ConfigManager:
    """
    Get the global ConfigManager instance, creating it if necessary.
    
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_schema(table_name: str) -> Dict[str, Any]:
    """
    Get schema for a specific table.
    
    Args:
        table_name: Name of the table
        
    Returns:
        Schema dictionary mapping column names to data types
    """
    return get_config_manager().get_schema(table_name)

def get_column_mapping(table_code: str) -> Dict[str, Any]:
    """
    Get column mapping for a specific Census table.
    
    Args:
        table_code: Census table code (e.g., 'G01', 'G19')
        
    Returns:
        Column mapping configuration
    """
    return get_config_manager().get_column_mapping(table_code)

def get_data_source_url(source_key: str) -> str:
    """
    Get URL for a specific data source.
    
    Args:
        source_key: Key identifying the data source
        
    Returns:
        URL string
    """
    return get_config_manager().get_data_source_url(source_key)

def get_path(path_name: str) -> Path:
    """
    Get a specific file system path.
    
    Args:
        path_name: Name of the path (e.g., 'OUTPUT_DIR', 'CENSUS_DIR')
        
    Returns:
        Path object
    """
    return get_config_manager().get_path(path_name)

# Convenience properties for accessing configurations
schemas = property(lambda: get_config_manager().schemas)
column_mappings = property(lambda: get_config_manager().column_mappings)
data_sources = property(lambda: get_config_manager().data_sources)
paths = property(lambda: get_config_manager().paths)