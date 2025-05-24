"""
Parquet file loaders for AHGD ETL pipeline.

This module provides loaders for writing data to Parquet files with schema enforcement.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import polars as pl

from ..config import settings

# Set up logger
logger = logging.getLogger('ahgd_etl.loaders.parquet')

class ParquetLoader:
    """
    Base class for loading data to Parquet files with schema enforcement.
    """
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the ParquetLoader.
        
        Args:
            schema: Dictionary mapping column names to data types
        """
        self.logger = logger
        self.schema = schema
    
    def enforce_schema(self, df: pl.DataFrame, table_name: str) -> pl.DataFrame:
        """
        Enforce the schema on a DataFrame before writing to Parquet.
        
        Args:
            df: DataFrame to enforce schema on
            table_name: Name of the table for schema lookup
            
        Returns:
            DataFrame with enforced schema
        """
        # Get schema if not provided in constructor
        schema = self.schema
        if not schema:
            schema = settings.get_schema(table_name)
        
        if not schema:
            self.logger.warning(f"No schema found for table {table_name}. Skipping schema enforcement.")
            return df
            
        self.logger.info(f"Enforcing schema for {table_name}")
        
        # Get current columns
        current_columns = set(df.columns)
        
        # Get schema columns
        schema_columns = set(schema.keys())
        
        # Find missing columns
        missing_columns = schema_columns - current_columns
        if missing_columns:
            self.logger.warning(f"Missing columns in {table_name}: {missing_columns}")
            # Add missing columns with null values
            for col in missing_columns:
                df = df.with_columns(pl.lit(None).cast(schema[col]).alias(col))
            
        # Find extra columns
        extra_columns = current_columns - schema_columns
        if extra_columns:
            self.logger.warning(f"Extra columns in {table_name} will be removed: {extra_columns}")
            # Remove extra columns
            df = df.select([col for col in df.columns if col in schema_columns])
        
        # Cast columns to correct types
        for col in df.columns:
            if col in schema:
                try:
                    df = df.with_columns(pl.col(col).cast(schema[col], strict=False))
                except Exception as e:
                    self.logger.error(f"Error casting column {col} to {schema[col]}: {e}")
        
        # Order columns according to schema
        ordered_columns = [col for col in schema.keys() if col in df.columns]
        df = df.select(ordered_columns)
        
        return df
    
    def save(self, df: pl.DataFrame, file_path: Path, table_name: Optional[str] = None) -> bool:
        """
        Save DataFrame to Parquet file with schema enforcement.
        
        Args:
            df: DataFrame to save
            file_path: Path to save the Parquet file
            table_name: Name of the table for schema lookup
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Use file name as table name if not provided
            if not table_name:
                table_name = file_path.stem
            
            # Enforce schema
            df = self.enforce_schema(df, table_name)
            
            # Create parent directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write DataFrame to Parquet file
            df.write_parquet(file_path)
            
            self.logger.info(f"Successfully wrote {len(df)} rows to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving {table_name} to {file_path}: {e}")
            return False

class DimensionLoader(ParquetLoader):
    """
    Loader for dimension tables with additional handling for unknown members.
    """
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the DimensionLoader.
        
        Args:
            schema: Dictionary mapping column names to data types
        """
        super().__init__(schema)
    
    def ensure_unknown_member(self, df: pl.DataFrame, table_name: str, key_column: str) -> pl.DataFrame:
        """
        Ensure that the dimension table has an unknown member.
        
        Args:
            df: Dimension DataFrame
            table_name: Name of the dimension table
            key_column: Name of the surrogate key column
            
        Returns:
            DataFrame with unknown member added if not already present
        """
        # Check if unknown member exists
        if 'is_unknown' in df.columns:
            unknown_exists = df.filter(pl.col('is_unknown') == True).height > 0
        else:
            # Add is_unknown column if it doesn't exist
            df = df.with_columns(pl.lit(False).alias('is_unknown'))
            unknown_exists = False
        
        if not unknown_exists:
            self.logger.info(f"Adding unknown member to {table_name}")
            
            # Get schema if not provided in constructor
            schema = self.schema
            if not schema:
                schema = settings.get_schema(table_name)
            
            # Create unknown member record
            unknown_data = {}
            
            # Set surrogate key for unknown member
            if key_column.endswith('_sk'):
                if df[key_column].dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]:
                    # For numeric surrogate keys, use -1
                    unknown_data[key_column] = -1
                else:
                    # For string surrogate keys, use "UNKNOWN"
                    unknown_data[key_column] = "UNKNOWN"
            else:
                self.logger.warning(f"Key column {key_column} doesn't follow naming convention (_sk suffix)")
                unknown_data[key_column] = "UNKNOWN"
            
            # Set is_unknown flag
            unknown_data['is_unknown'] = True
            
            # Fill other columns with appropriate unknown values
            for col in df.columns:
                if col not in unknown_data:
                    dtype = df[col].dtype
                    if dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]:
                        unknown_data[col] = -1
                    elif dtype in [pl.Float64, pl.Float32]:
                        unknown_data[col] = -1.0
                    elif dtype == pl.Boolean:
                        unknown_data[col] = False
                    else:
                        unknown_data[col] = "UNKNOWN"
            
            # Create unknown member DataFrame
            unknown_df = pl.DataFrame([unknown_data])
            
            # Combine with original DataFrame
            df = pl.concat([df, unknown_df])
            
            self.logger.info(f"Added unknown member to {table_name}")
        
        return df
    
    def save(self, df: pl.DataFrame, file_path: Path, table_name: Optional[str] = None, 
             key_column: Optional[str] = None) -> bool:
        """
        Save dimension table to Parquet file with schema enforcement and unknown member.
        
        Args:
            df: DataFrame to save
            file_path: Path to save the Parquet file
            table_name: Name of the table for schema lookup
            key_column: Name of the surrogate key column
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Use file name as table name if not provided
            if not table_name:
                table_name = file_path.stem
            
            # Determine key column if not provided
            if not key_column:
                for col in df.columns:
                    if col.endswith('_sk'):
                        key_column = col
                        break
            
            if not key_column:
                self.logger.warning(f"No key column found for {table_name}")
                return super().save(df, file_path, table_name)
            
            # Ensure unknown member exists
            df = self.ensure_unknown_member(df, table_name, key_column)
            
            # Save using parent method
            return super().save(df, file_path, table_name)
            
        except Exception as e:
            self.logger.error(f"Error saving dimension {table_name} to {file_path}: {e}")
            return False

class FactLoader(ParquetLoader):
    """
    Loader for fact tables with grain management.
    """
    
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        """
        Initialize the FactLoader.
        
        Args:
            schema: Dictionary mapping column names to data types
        """
        super().__init__(schema)
    
    def manage_grain(self, df: pl.DataFrame, table_name: str, grain_columns: List[str], 
                    measure_columns: List[str]) -> pl.DataFrame:
        """
        Manage the grain of a fact table to handle duplicate keys.
        
        Args:
            df: Fact DataFrame
            table_name: Name of the fact table
            grain_columns: List of columns that define the grain
            measure_columns: List of measure columns to aggregate
            
        Returns:
            DataFrame with managed grain
        """
        # Check for duplicate grain values
        grouped = df.group_by(grain_columns).agg(pl.count().alias('_count'))
        duplicates = grouped.filter(pl.col('_count') > 1)
        
        if duplicates.height > 0:
            self.logger.warning(f"Found {duplicates.height} duplicate grain combinations in {table_name}")
            
            # Log the first few duplicates for debugging
            self.logger.warning(f"First 5 duplicates: {duplicates.head(5)}")
            
            # Aggregate at the grain level
            aggregations = []
            for col in measure_columns:
                # Use appropriate aggregation function based on column name
                if any(indicator in col.lower() for indicator in ['count', 'population', 'persons']):
                    # Use sum for count measures
                    aggregations.append(pl.sum(col).alias(col))
                elif any(indicator in col.lower() for indicator in ['avg', 'average', 'mean']):
                    # Use average for average measures
                    aggregations.append(pl.mean(col).alias(col))
                elif any(indicator in col.lower() for indicator in ['min', 'minimum']):
                    # Use min for minimum measures
                    aggregations.append(pl.min(col).alias(col))
                elif any(indicator in col.lower() for indicator in ['max', 'maximum']):
                    # Use max for maximum measures
                    aggregations.append(pl.max(col).alias(col))
                else:
                    # Default to sum for other measures
                    aggregations.append(pl.sum(col).alias(col))
            
            # Include non-grain, non-measure columns with first value
            non_agg_columns = [col for col in df.columns if col not in grain_columns and col not in measure_columns]
            for col in non_agg_columns:
                aggregations.append(pl.first(col).alias(col))
            
            # Aggregate to resolve duplicates
            df = df.group_by(grain_columns).agg(aggregations)
            
            self.logger.info(f"Resolved {duplicates.height} duplicate grain combinations in {table_name}")
        
        return df
    
    def save(self, df: pl.DataFrame, file_path: Path, table_name: Optional[str] = None,
             grain_columns: Optional[List[str]] = None, 
             measure_columns: Optional[List[str]] = None) -> bool:
        """
        Save fact table to Parquet file with schema enforcement and grain management.
        
        Args:
            df: DataFrame to save
            file_path: Path to save the Parquet file
            table_name: Name of the table for schema lookup
            grain_columns: List of columns that define the grain
            measure_columns: List of measure columns to aggregate
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            # Use file name as table name if not provided
            if not table_name:
                table_name = file_path.stem
            
            # Auto-detect grain columns if not provided
            if not grain_columns:
                grain_columns = [col for col in df.columns if col.endswith('_sk')]
                if not grain_columns:
                    self.logger.warning(f"No grain columns detected for {table_name}")
            
            # Auto-detect measure columns if not provided
            if not measure_columns:
                # Exclude grain columns, etl_processed_at, and any other metadata
                exclude_columns = grain_columns + ['etl_processed_at']
                measure_columns = [col for col in df.columns if col not in exclude_columns]
            
            # Manage grain to handle duplicate keys
            if grain_columns and measure_columns:
                df = self.manage_grain(df, table_name, grain_columns, measure_columns)
            
            # Save using parent method
            return super().save(df, file_path, table_name)
            
        except Exception as e:
            self.logger.error(f"Error saving fact {table_name} to {file_path}: {e}")
            return False