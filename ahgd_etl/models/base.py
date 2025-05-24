"""
Base model classes for AHGD ETL dimensional models.

This module defines the base classes for dimension and fact tables,
providing common functionality for model operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, ClassVar, Type

import polars as pl
import yaml

class BaseModel:
    """Base class for all data warehouse models."""
    
    # Class variables to be overridden by subclasses
    table_name: ClassVar[str] = ""
    schema: ClassVar[Dict[str, str]] = {}
    
    @classmethod
    def load_schema(cls, schema_file: Path) -> Dict[str, Dict[str, str]]:
        """
        Load schema definitions from YAML file.
        
        Args:
            schema_file: Path to schema YAML file
            
        Returns:
            Dictionary of table schemas
        """
        with open(schema_file, 'r') as f:
            schema_data = yaml.safe_load(f)
        return schema_data
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> 'BaseModel':
        """
        Create model instance from a DataFrame.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement from_dataframe")
    
    def to_dataframe(self) -> pl.DataFrame:
        """
        Convert model instance to DataFrame.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement to_dataframe")

@dataclass
class DimensionModel(BaseModel):
    """Base class for dimension tables in the data warehouse."""
    
    # Common fields for all dimensions
    surrogate_key: str = ""
    is_unknown: bool = False
    etl_processed_at: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def get_unknown_member(cls) -> 'DimensionModel':
        """
        Create an unknown member for this dimension.
        
        Returns:
            DimensionModel instance representing unknown member
        """
        instance = cls()
        instance.is_unknown = True
        
        # Generate surrogate key for unknown member
        import hashlib
        instance.surrogate_key = hashlib.md5(f"UNKNOWN_{cls.table_name}".encode()).hexdigest()
        
        return instance
    
    @classmethod
    def read_from_parquet(cls, file_path: Path) -> List['DimensionModel']:
        """
        Read dimension members from parquet file.
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            List of dimension model instances
        """
        try:
            df = pl.read_parquet(file_path)
            result = []
            
            for row in df.iter_rows(named=True):
                instance = cls.from_dataframe(pl.DataFrame([row]))
                result.append(instance)
                
            return result
        except Exception as e:
            raise ValueError(f"Error reading dimension from {file_path}: {e}")
    
    def write_to_parquet(self, output_dir: Path) -> Path:
        """
        Write dimension to parquet file.
        
        Args:
            output_dir: Directory to write to
            
        Returns:
            Path to written file
        """
        df = self.to_dataframe()
        output_path = output_dir / f"{self.table_name}.parquet"
        df.write_parquet(output_path)
        return output_path

@dataclass
class FactModel(BaseModel):
    """Base class for fact tables in the data warehouse."""
    
    # Common fields for all facts
    etl_processed_at: datetime = field(default_factory=datetime.now)
    
    # Track dimension keys (to be populated by subclasses)
    dimension_keys: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def read_from_parquet(cls, file_path: Path) -> 'FactModel':
        """
        Read fact table from parquet file.
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            Fact model instance
        """
        try:
            df = pl.read_parquet(file_path)
            return cls.from_dataframe(df)
        except Exception as e:
            raise ValueError(f"Error reading fact table from {file_path}: {e}")
    
    def write_to_parquet(self, output_dir: Path) -> Path:
        """
        Write fact table to parquet file.
        
        Args:
            output_dir: Directory to write to
            
        Returns:
            Path to written file
        """
        df = self.to_dataframe()
        output_path = output_dir / f"{self.table_name}.parquet"
        df.write_parquet(output_path)
        return output_path
    
    def validate_grain(self) -> bool:
        """
        Validate that the fact table grain is correct.
        
        Returns:
            True if grain is valid, False otherwise
        """
        # To be implemented by subclasses
        return True