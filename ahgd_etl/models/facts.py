"""
Fact model classes for AHGD ETL pipeline.

This module defines the fact table models used in the AHGD data warehouse.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, ClassVar

import polars as pl

from .base import FactModel

@dataclass
class PopulationFact(FactModel):
    """Population fact table model."""
    
    table_name: ClassVar[str] = "fact_population"
    
    # Dimension keys
    geo_sk: str = ""
    time_sk: str = ""
    
    # Measures
    total_population: int = 0
    male_population: int = 0
    female_population: int = 0
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> 'PopulationFact':
        """
        Create PopulationFact instance from DataFrame.
        For fact tables, this typically returns a single instance
        representing the entire fact table.
        """
        instance = cls()
        
        # Just store the DataFrame reference for efficiency
        instance._df = df
        
        # Also set dimension keys for reference
        instance.dimension_keys = {
            "geo_sk": "geo_dimension",
            "time_sk": "dim_time"
        }
        
        return instance
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert PopulationFact to DataFrame."""
        # If we have a stored DataFrame, return it
        if hasattr(self, "_df"):
            return self._df
        
        # Otherwise create a simple one-row DataFrame
        return pl.DataFrame({
            "geo_sk": [self.geo_sk],
            "time_sk": [self.time_sk],
            "total_population": [self.total_population],
            "male_population": [self.male_population],
            "female_population": [self.female_population],
            "etl_processed_at": [self.etl_processed_at]
        })
    
    def validate_grain(self) -> bool:
        """
        Validate that the grain of the fact table is correct.
        For PopulationFact, the grain is geo_sk + time_sk.
        
        Returns:
            True if grain is valid, False otherwise
        """
        if not hasattr(self, "_df"):
            return True  # Nothing to validate
            
        # Check for duplicates in the grain columns
        df = self._df
        grain_cols = ["geo_sk", "time_sk"]
        
        # Count distinct combinations of grain columns
        distinct_grain = df.select(grain_cols).unique().height
        total_rows = df.height
        
        if distinct_grain < total_rows:
            return False
            
        return True

@dataclass
class HealthConditionFact(FactModel):
    """Health condition fact table model."""
    
    table_name: ClassVar[str] = "fact_health_conditions"
    
    # Dimension keys
    geo_sk: str = ""
    time_sk: str = ""
    condition_sk: str = ""
    demographic_sk: str = ""
    
    # Measures
    count_persons: int = 0
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> 'HealthConditionFact':
        """Create HealthConditionFact instance from DataFrame."""
        instance = cls()
        
        # Store the DataFrame reference
        instance._df = df
        
        # Set dimension keys for reference
        instance.dimension_keys = {
            "geo_sk": "geo_dimension",
            "time_sk": "dim_time",
            "condition_sk": "dim_health_condition",
            "demographic_sk": "dim_demographic"
        }
        
        return instance
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert HealthConditionFact to DataFrame."""
        # If we have a stored DataFrame, return it
        if hasattr(self, "_df"):
            return self._df
        
        # Otherwise create a simple one-row DataFrame
        return pl.DataFrame({
            "geo_sk": [self.geo_sk],
            "time_sk": [self.time_sk],
            "condition_sk": [self.condition_sk],
            "demographic_sk": [self.demographic_sk],
            "count_persons": [self.count_persons],
            "etl_processed_at": [self.etl_processed_at]
        })
    
    def validate_grain(self) -> bool:
        """
        Validate that the grain of the fact table is correct.
        For HealthConditionFact, the grain is geo_sk + time_sk + condition_sk + demographic_sk.
        
        Returns:
            True if grain is valid, False otherwise
        """
        if not hasattr(self, "_df"):
            return True  # Nothing to validate
            
        # Check for duplicates in the grain columns
        df = self._df
        grain_cols = ["geo_sk", "time_sk", "condition_sk", "demographic_sk"]
        
        # Count distinct combinations of grain columns
        distinct_grain = df.select(grain_cols).unique().height
        total_rows = df.height
        
        if distinct_grain < total_rows:
            return False
            
        return True

@dataclass
class HealthConditionRefinedFact(FactModel):
    """Refined health condition fact table model."""
    
    table_name: ClassVar[str] = "fact_health_conditions_refined"
    
    # Dimension keys
    geo_sk: str = ""
    time_sk: str = ""
    condition_sk: str = ""
    demographic_sk: str = ""
    characteristic_sk: str = ""
    
    # Measures
    count_persons: int = 0
    
    @classmethod
    def from_dataframe(cls, df: pl.DataFrame) -> 'HealthConditionRefinedFact':
        """Create HealthConditionRefinedFact instance from DataFrame."""
        instance = cls()
        
        # Store the DataFrame reference
        instance._df = df
        
        # Set dimension keys for reference
        instance.dimension_keys = {
            "geo_sk": "geo_dimension",
            "time_sk": "dim_time",
            "condition_sk": "dim_health_condition",
            "demographic_sk": "dim_demographic",
            "characteristic_sk": "dim_person_characteristic"
        }
        
        return instance
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert HealthConditionRefinedFact to DataFrame."""
        # If we have a stored DataFrame, return it
        if hasattr(self, "_df"):
            return self._df
        
        # Otherwise create a simple one-row DataFrame
        return pl.DataFrame({
            "geo_sk": [self.geo_sk],
            "time_sk": [self.time_sk],
            "condition_sk": [self.condition_sk],
            "demographic_sk": [self.demographic_sk],
            "characteristic_sk": [self.characteristic_sk],
            "count_persons": [self.count_persons],
            "etl_processed_at": [self.etl_processed_at]
        })
    
    def validate_grain(self) -> bool:
        """
        Validate that the grain of the fact table is correct.
        
        Returns:
            True if grain is valid, False otherwise
        """
        if not hasattr(self, "_df"):
            return True  # Nothing to validate
            
        # Check for duplicates in the grain columns
        df = self._df
        grain_cols = ["geo_sk", "time_sk", "condition_sk", "demographic_sk", "characteristic_sk"]
        
        # Count distinct combinations of grain columns
        distinct_grain = df.select(grain_cols).unique().height
        total_rows = df.height
        
        if distinct_grain < total_rows:
            return False
            
        return True