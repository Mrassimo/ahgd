"""
Base transformer class for Census data processing.

This module provides a base class for Census table transformers,
with common functionality for processing Census data.
"""

import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union

import polars as pl

from ...config import settings
from ...loaders import DimensionLoader, FactLoader
from ...core.fix_manager import FixManager

class BaseCensusTransformer:
    """
    Base class for Census data transformers.
    
    This class provides common functionality for all Census table transformers,
    including configuration loading, geographic code handling, and data processing.
    """
    
    def __init__(self, table_code: str, fix_manager: Optional[FixManager] = None):
        """
        Initialize the transformer.
        
        Args:
            table_code: Census table code (e.g., 'G01', 'G17')
            fix_manager: Optional fix manager for inline data quality fixes
        """
        self.table_code = table_code
        self.logger = logging.getLogger(f'ahgd_etl.transformers.census.{table_code.lower()}')
        self.fix_manager = fix_manager
        
        # Load configuration
        self.config = settings.get_column_mapping(table_code)
        self.geo_column_options = settings.get_column_mapping('geo_column_options')
        
        if not self.geo_column_options:
            self.geo_column_options = [
                'SA1_CODE', 'SA1_CODE_2021', 'SA1_CODE21', 'SA1_2021_CODE',
                'SA2_CODE', 'SA2_CODE_2021', 'SA2_CODE21', 'SA2_2021_CODE'
            ]
        
        # Initialize loaders for dimensions and facts
        self.dimension_loader = DimensionLoader()
        self.fact_loader = FactLoader()
    
    def find_geo_column(self, df: pl.DataFrame) -> Optional[str]:
        """
        Find the geographic code column in a DataFrame.
        
        Args:
            df: Input DataFrame to search for geographic code column
            
        Returns:
            Name of geographic code column or None if not found
        """
        for col_option in self.geo_column_options:
            if col_option in df.columns:
                return col_option
        
        return None
    
    def process_file(self, csv_file: Path, geo_output_path: Optional[Path] = None,
                     time_sk: Optional[Union[str, int]] = None) -> Optional[pl.DataFrame]:
        """
        Process a single Census CSV file.

        Args:
            csv_file: Path to the CSV file to process
            geo_output_path: Path to the geographic dimension file (for lookups)
            time_sk: Time dimension surrogate key to use

        Returns:
            Processed DataFrame or None if processing failed
        """
        # This is an abstract method to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement process_file method")

    def process(self, zip_dir: Path, temp_extract_base: Path, output_dir: Path,
               geo_output_path: Path, time_dim_path: Path) -> bool:
        """
        Process Census data files for this table.

        Args:
            zip_dir: Directory containing downloaded ZIP files
            temp_extract_base: Base directory for temporary file extraction
            output_dir: Directory to write output files
            geo_output_path: Path to the geographic dimension file
            time_dim_path: Path to the time dimension file

        Returns:
            True if processing was successful, False otherwise
        """
        from ...utils import process_census_table

        try:
            self.logger.info(f"Processing {self.table_code} table")

            # Read time dimension to get the Census reference time_sk
            time_df = pl.read_parquet(time_dim_path)
            # Get time surrogate key for Census 2021 reference date (August 10, 2021)
            # Use day_of_month instead of day to match the time dimension schema
            census_time = time_df.filter(
                (pl.col("year") == 2021) &
                (pl.col("month") == 8) &
                (pl.col("day_of_month") == 10)
            )

            if len(census_time) == 0:
                self.logger.error("Could not find Census reference date in time dimension")
                return False

            time_sk = str(census_time[0, "time_sk"])
            self.logger.info(f"Using time_sk {time_sk} for Census 2021 reference date")

            # Process the table using the utility function
            output_filename = f"fact_{self.table_code.lower()}.parquet"

            success = process_census_table(
                self.table_code,
                self.process_file,  # Pass the process_file method as the processor function
                output_filename,
                zip_dir,
                temp_extract_base,
                output_dir,
                geo_output_path,
                time_sk
            )

            return success

        except Exception as e:
            self.logger.error(f"Error processing {self.table_code} table: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def lookup_geo_sk(self, df: pl.DataFrame, geo_code_column: str, 
                      geo_output_path: Path) -> Optional[pl.DataFrame]:
        """
        Look up geographic surrogate keys for the DataFrame.
        
        This is an enhanced version that uses the unknown geo_sk for unmapped codes.
        
        Args:
            df: DataFrame with geographic codes
            geo_code_column: Name of the column containing geographic codes
            geo_output_path: Path to the geographic dimension file
            
        Returns:
            DataFrame with geo_sk column added, or None if lookup failed
        """
        try:
            # Read the geographic dimension
            geo_df = pl.read_parquet(geo_output_path)
            self.logger.info(f"Read geographic dimension with {len(geo_df)} rows")
            
            # Find the unknown geo_sk
            unknown_geo_sk = None
            unknown_rows = geo_df.filter(pl.col('is_unknown') == True) if 'is_unknown' in geo_df.columns else None
            if unknown_rows is not None and len(unknown_rows) > 0:
                unknown_geo_sk = unknown_rows[0, 'geo_sk']
            else:
                # Try to find by geo_id = 'UNKNOWN'
                unknown_rows = geo_df.filter(pl.col('geo_id') == 'UNKNOWN')
                if unknown_rows is not None and len(unknown_rows) > 0:
                    unknown_geo_sk = unknown_rows[0, 'geo_sk']
                else:
                    # Use -1 as a fallback
                    unknown_geo_sk = -1
            
            self.logger.info(f"Unknown geo_sk is {unknown_geo_sk}")
            
            # Create a mapping dictionary for faster lookups
            geo_mapping = {}
            for row in geo_df.select(['geo_id', 'geo_sk']).iter_rows(named=True):
                geo_mapping[row['geo_id']] = row['geo_sk']
            
            # Add geo_sk column using the mapping with unknown fallback
            df = df.with_columns(
                pl.col(geo_code_column).map_elements(
                    lambda x: geo_mapping.get(str(x), unknown_geo_sk)
                ).alias('geo_sk')
            )
            
            # Count mapped to unknown in geo_sk to check for unmapped codes
            unknown_count = df.filter(pl.col('geo_sk') == unknown_geo_sk).height
            if unknown_count > 0:
                self.logger.warning(f"Found {unknown_count} rows mapped to unknown geo_sk")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error looking up geographic surrogate keys: {e}")
            return None
    
    def generate_surrogate_key(self, *args) -> str:
        """
        Generate a deterministic surrogate key using MD5 hash.
        
        Args:
            *args: One or more values to hash together
            
        Returns:
            MD5 hash as hexadecimal string
        """
        # Convert all arguments to strings
        str_args = []
        for arg in args:
            if arg is None:
                str_args.append("UNKNOWN")
            else:
                str_args.append(str(arg))
        
        # Join with underscore and hash
        key_str = "_".join(str_args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def lookup_dimension_sk(self, df: pl.DataFrame, lookup_column: str, 
                           dimension_df: pl.DataFrame, dimension_key: str, 
                           dimension_value: str) -> pl.DataFrame:
        """
        Look up surrogate keys from a dimension table.
        
        Args:
            df: DataFrame containing values to lookup
            lookup_column: Column in df containing values to lookup
            dimension_df: Dimension DataFrame to lookup in
            dimension_key: Natural key column in dimension_df
            dimension_value: Surrogate key column in dimension_df
            
        Returns:
            DataFrame with surrogate key column added
        """
        try:
            # Find the unknown surrogate key
            unknown_sk = None
            unknown_rows = dimension_df.filter(pl.col('is_unknown') == True) if 'is_unknown' in dimension_df.columns else None
            if unknown_rows is not None and len(unknown_rows) > 0:
                unknown_sk = unknown_rows[0, dimension_value]
            else:
                # Try to find by dimension_key = 'UNKNOWN'
                unknown_rows = dimension_df.filter(pl.col(dimension_key) == 'UNKNOWN')
                if unknown_rows is not None and len(unknown_rows) > 0:
                    unknown_sk = unknown_rows[0, dimension_value]
                else:
                    # Generate a fallback unknown SK
                    unknown_sk = self.generate_surrogate_key("UNKNOWN", dimension_value)
            
            # Create a mapping dictionary for faster lookups
            sk_mapping = {}
            for row in dimension_df.select([dimension_key, dimension_value]).iter_rows(named=True):
                sk_mapping[row[dimension_key]] = row[dimension_value]
            
            # Add surrogate key column using the mapping with unknown fallback
            df = df.with_columns(
                pl.col(lookup_column).map_elements(
                    lambda x: sk_mapping.get(str(x), unknown_sk)
                ).alias(dimension_value)
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error looking up dimension keys: {e}")
            # Return original DataFrame if lookup fails
            return df
    
    def add_etl_timestamp(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add ETL processing timestamp to DataFrame.
        
        Args:
            df: DataFrame to add timestamp to
            
        Returns:
            DataFrame with etl_processed_at column added
        """
        return df.with_columns(pl.lit(datetime.now()).alias('etl_processed_at'))
    
    def save_fact_table(self, df: pl.DataFrame, output_path: Path, 
                       grain_columns: Optional[List[str]] = None, 
                       measure_columns: Optional[List[str]] = None,
                       aggregation_rules: Optional[Dict[str, str]] = None,
                       dimension_refs: Optional[Dict[str, Path]] = None) -> bool:
        """
        Save fact table with schema enforcement and grain management.
        
        Args:
            df: Fact DataFrame to save
            output_path: Path to save the Parquet file
            grain_columns: List of columns that define the grain
            measure_columns: List of measure columns to aggregate
            aggregation_rules: Dict mapping columns to aggregation methods (sum, mean, max, min)
            dimension_refs: Dict mapping foreign key columns to dimension file paths
            
        Returns:
            True if save was successful, False otherwise
        """
        # Apply inline fixes if fix manager is available
        if self.fix_manager:
            # Fix referential integrity
            if dimension_refs:
                for fk_column, dim_path in dimension_refs.items():
                    if fk_column in df.columns and dim_path.exists():
                        dim_df = pl.read_parquet(dim_path)
                        # Determine the primary key column name
                        pk_column = fk_column  # Assumes same name, e.g., geo_sk in both tables
                        df = self.fix_manager.fix_referential_integrity(
                            df, dim_df, fk_column, pk_column, output_path.stem
                        )
            
            # Handle duplicates
            if grain_columns:
                if not aggregation_rules:
                    # Default aggregation rules for measures
                    aggregation_rules = {col: "sum" for col in measure_columns} if measure_columns else {}
                
                df = self.fix_manager.deduplicate_fact_table(
                    df, grain_columns, aggregation_rules, output_path.stem
                )
            
            # Enforce schema
            table_name = output_path.stem
            schema = settings.schemas.facts.get(table_name, {})
            if schema:
                df = self.fix_manager.enforce_schema(df, schema, table_name)
        
        # Add ETL timestamp if not already present
        if 'etl_processed_at' not in df.columns:
            df = self.add_etl_timestamp(df)
        
        # Save using fact loader
        table_name = output_path.stem
        return self.fact_loader.save(df, output_path, table_name, grain_columns, measure_columns)
    
    def save_dimension_table(self, df: pl.DataFrame, output_path: Path, 
                            key_column: Optional[str] = None) -> bool:
        """
        Save dimension table with schema enforcement and unknown member.
        
        Args:
            df: Dimension DataFrame to save
            output_path: Path to save the Parquet file
            key_column: Name of the surrogate key column
            
        Returns:
            True if save was successful, False otherwise
        """
        # Add ETL timestamp if not already present
        if 'etl_processed_at' not in df.columns:
            df = self.add_etl_timestamp(df)
        
        # Save using dimension loader
        table_name = output_path.stem
        return self.dimension_loader.save(df, output_path, table_name, key_column)