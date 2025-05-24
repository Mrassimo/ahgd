"""Settings module for AHGD ETL Pipeline.

This module provides centralized configuration management by loading
YAML configuration files and environment variables.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from datetime import datetime
import logging
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)


class Settings:
    """Central configuration management for AHGD ETL Pipeline."""
    
    def __init__(self):
        """Initialize settings by loading configuration files and environment."""
        # Base paths
        self.base_dir = Path(os.getenv("BASE_DIR", Path.cwd()))
        self.config_dir = self.base_dir / "ahgd_etl" / "config" / "yaml"
        
        # Output directories
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.output_dir = self.base_dir / "output"
        self.logs_dir = self.base_dir / "logs"
        
        # Ensure directories exist
        for directory in [self.raw_data_dir, self.output_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Load configuration files
        self._data_sources = self._load_yaml("data_sources.yaml")
        self._schemas = self._load_yaml("schemas.yaml")
        self._column_mappings = self._load_yaml("column_mappings.yaml")
        
        # Processing settings
        self.force_download = os.getenv("FORCE_DOWNLOAD", "false").lower() == "true"
        self.stop_on_error = os.getenv("STOP_ON_ERROR", "true").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        
        # Census date configuration
        census_config = self._data_sources.get("census", {})
        self.census_date = datetime.strptime(
            census_config.get("date", "2021-08-10"), 
            "%Y-%m-%d"
        ).date()
        self.census_year = census_config.get("year", 2021)
        
        # Time dimension configuration
        self.time_dim_start_year = int(os.getenv("TIME_DIM_START_YEAR", "2010"))
        self.time_dim_end_year = int(os.getenv("TIME_DIM_END_YEAR", "2030"))
        
        # Unknown member keys
        self.unknown_geo_sk = -1
        self.unknown_time_sk = 19000101  # 1900-01-01 as YYYYMMDD
        self.unknown_sk_prefix = "UNKNOWN_"
        
        # Snowflake configuration (optional)
        self.snowflake_config_path = os.getenv(
            "SNOWFLAKE_CONFIG", 
            str(self.base_dir / "snowflake" / "config.json")
        )
    
    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file.
        
        Args:
            filename: Name of the YAML file to load
            
        Returns:
            Dictionary containing the loaded configuration
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            yaml.YAMLError: If the YAML file is invalid
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error(f"Error loading YAML file {filename}: {e}")
            raise
    
    @property
    def geographic_sources(self) -> Dict[str, Any]:
        """Get geographic data source configurations."""
        return self._data_sources.get("geographic", {}).get("asgs_2021", {})
    
    @property
    def census_sources(self) -> Dict[str, Any]:
        """Get census data source configurations."""
        return self._data_sources.get("census", {}).get("gcp_2021", {})
    
    @property
    def census_tables(self) -> list:
        """Get list of census tables to process."""
        return self.census_sources.get("all_australia", {}).get("tables", [])
    
    def get_schema(self, table_name: str) -> Dict[str, Any]:
        """Get schema definition for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Schema definition dictionary
            
        Raises:
            KeyError: If table schema not found
        """
        # Check dimensions first
        if table_name in self._schemas.get("dimensions", {}):
            return self._schemas["dimensions"][table_name]
        # Then check facts
        elif table_name in self._schemas.get("facts", {}):
            return self._schemas["facts"][table_name]
        else:
            raise KeyError(f"Schema not found for table: {table_name}")
    
    def get_column_mapping(self, source_type: str) -> Dict[str, Any]:
        """Get column mapping configuration for a source type.
        
        Args:
            source_type: Type of source (e.g., 'G01', 'sa1')
            
        Returns:
            Column mapping configuration
        """
        # Check geographic mappings
        if source_type in self._column_mappings.get("geographic", {}):
            return self._column_mappings["geographic"][source_type]
        # Check census mappings
        elif source_type in self._column_mappings.get("census", {}):
            return self._column_mappings["census"][source_type]
        else:
            return {}
    
    def get_characteristic_types(self) -> Dict[str, Any]:
        """Get person characteristic type definitions."""
        return self._column_mappings.get("characteristic_types", {})
    
    def get_demographics(self) -> Dict[str, Any]:
        """Get demographic definitions (age groups, sex categories)."""
        return self._column_mappings.get("demographics", {})
    
    def get_geographic_url(self, level: str) -> Optional[str]:
        """Get download URL for a geographic level.
        
        Args:
            level: Geographic level (e.g., 'sa1', 'sa2')
            
        Returns:
            URL string or None if not found
        """
        geo_config = self.geographic_sources.get(level.lower(), {})
        return geo_config.get("url")
    
    def get_geographic_filename(self, level: str) -> Optional[str]:
        """Get expected filename for a geographic level.
        
        Args:
            level: Geographic level (e.g., 'sa1', 'sa2')
            
        Returns:
            Filename string or None if not found
        """
        geo_config = self.geographic_sources.get(level.lower(), {})
        return geo_config.get("filename")
    
    def get_census_url(self) -> Optional[str]:
        """Get download URL for census data pack."""
        census_config = self.census_sources.get("all_australia", {})
        return census_config.get("url")
    
    def get_census_filename(self) -> Optional[str]:
        """Get expected filename for census data pack."""
        census_config = self.census_sources.get("all_australia", {})
        return census_config.get("filename")
    
    def validate_configuration(self) -> bool:
        """Validate that all required configurations are present.
        
        Returns:
            True if valid, raises exception otherwise
        """
        required_configs = [
            ("Geographic sources", self.geographic_sources),
            ("Census sources", self.census_sources),
            ("Schemas", self._schemas),
            ("Column mappings", self._column_mappings)
        ]
        
        for name, config in required_configs:
            if not config:
                raise ValueError(f"{name} configuration is empty or missing")
        
        # Validate required geographic levels
        required_geo_levels = ["sa1", "sa2", "sa3", "sa4", "ste"]
        for level in required_geo_levels:
            if level not in self.geographic_sources:
                raise ValueError(f"Geographic level '{level}' configuration missing")
        
        # Validate census tables
        if not self.census_tables:
            raise ValueError("No census tables configured for processing")
        
        # Validate schema definitions
        required_dimensions = ["geo_dimension", "dim_time", "dim_health_condition", 
                             "dim_demographic", "dim_person_characteristic"]
        for dim in required_dimensions:
            if dim not in self._schemas.get("dimensions", {}):
                raise ValueError(f"Schema missing for dimension: {dim}")
        
        return True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Returns:
        Singleton Settings instance
    """
    settings = Settings()
    settings.validate_configuration()
    return settings


# Make settings available at module level
settings = get_settings()