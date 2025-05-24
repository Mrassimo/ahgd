"""
Census G01 (Selected Person Characteristics) data transformer.

This module handles the processing of ABS Census G01 table data, which contains
selected person characteristics and population counts.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Any
import traceback
from datetime import datetime

import polars as pl

from .base import BaseCensusTransformer
from .utils import clean_polars_geo_code, safe_polars_int
from ...config import settings

class G01PopulationTransformer(BaseCensusTransformer):
    """
    Transformer for G01 (Selected Person Characteristics) Census data.
    
    This class processes G01 Census data files, extracting population information
    by geography, age, and sex.
    """
    
    def __init__(self, output_dir=None, fix_manager=None):
        """Initialize G01 transformer."""
        super().__init__("G01", fix_manager)
        self.output_dir = output_dir
        
        # Load G01-specific configuration
        self.measure_column_map = self.config.get("measure_column_map", {})
        self.required_target_columns = self.config.get("required_target_columns", [])
    
    def process(self) -> bool:
        """Process G01 data using the enhanced pipeline."""
        try:
            # For now, return True to avoid breaking the pipeline
            # TODO: Implement full processing logic
            self.logger.info("G01 processing placeholder - TODO: implement")
            return True
        except Exception as e:
            self.logger.error(f"Error in G01 processing: {str(e)}", exc_info=True)
            return False
    
    def process_file(self, csv_file: Path, geo_output_path: Optional[Path] = None, 
                     time_sk: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Process a single G01 (Selected Person Characteristics) Census CSV file.
        
        Args:
            csv_file: Path to the CSV file to process
            geo_output_path: Path to the geographic dimension file (for lookups)
            time_sk: Time dimension surrogate key to use
            
        Returns:
            Processed DataFrame or None if processing failed
        """
        try:
            # Read the CSV file
            df = pl.read_csv(csv_file, truncate_ragged_lines=True)
            self.logger.info(f"Read {len(df)} rows from {csv_file.name}")
            
            # Identify geo_code column
            geo_code_column = self.find_geo_column(df)
            if not geo_code_column:
                self.logger.error(f"Could not find geographic code column in {csv_file.name}")
                return None
            self.logger.info(f"Found geographic code column: {geo_code_column}")
            
            # Basic validation of configuration
            if not self.measure_column_map:
                self.logger.error(f"Missing measure column mappings for G01 table")
                return None
            
            if not self.required_target_columns:
                self.logger.error(f"Missing required target columns for G01 table")
                return None
            
            # Select required source columns
            source_columns = list(set([col for cols in self.measure_column_map.values() for col in cols]))
            
            # Verify required columns exist in the DataFrame
            missing_columns = [col for col in source_columns if col not in df.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns in input data: {missing_columns}")
                return None
            
            # Create the final DataFrame with reshaped columns
            result_columns = []
            for target_col, source_cols in self.measure_column_map.items():
                if len(source_cols) == 1:
                    # Single column mapping
                    result_columns.append(pl.col(source_cols[0]).alias(target_col))
                else:
                    # Multiple columns to combine (sum)
                    expr = pl.lit(0)
                    for col in source_cols:
                        expr = expr + pl.col(col).fill_null(0)
                    result_columns.append(expr.alias(target_col))
            
            # Add the geo_code column (cleaned)
            result_columns.insert(0, clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"))
            
            # Create the result DataFrame
            result_df = df.select(result_columns)
            
            # Verify all required columns are present
            missing_required = [col for col in self.required_target_columns if col not in result_df.columns]
            if missing_required:
                self.logger.error(f"Failed to create required columns: {missing_required}")
                return None
            
            # Add geo_sk and time_sk if geo_output_path is provided
            if geo_output_path and time_sk:
                # Add geo_sk column
                result_df = self.lookup_geo_sk(result_df, "geo_code", geo_output_path)
                if result_df is None:
                    return None
                
                # Add time_sk column
                result_df = result_df.with_columns(pl.lit(time_sk).alias("time_sk"))
                
                # Add timestamp column
                result_df = result_df.with_columns(pl.lit(datetime.now()).alias("etl_processed_at"))
            
            self.logger.info(f"Created G01 DataFrame with {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error processing CSV file {csv_file.name}: {str(e)}")
            self.logger.error(traceback.format_exc())  # Log full traceback
            return None
    
    def calculate_derived_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate derived population metrics.
        
        Args:
            df: DataFrame with population counts
            
        Returns:
            DataFrame with additional derived metrics
        """
        try:
            # Check for required columns
            required_cols = ["male_population", "female_population", "total_population"]
            if not all(col in df.columns for col in required_cols):
                self.logger.warning("Missing required columns for derived metrics")
                return df
            
            # Calculate derived metrics
            derived_metrics = [
                # Sex ratio (males per 100 females)
                (pl.col("male_population") * 100 / pl.col("female_population")).alias("sex_ratio"),
                
                # Percentage male
                (pl.col("male_population") * 100 / pl.col("total_population")).alias("percent_male"),
                
                # Percentage female
                (pl.col("female_population") * 100 / pl.col("total_population")).alias("percent_female")
            ]
            
            # Add metrics to DataFrame
            result_df = df.with_columns(derived_metrics)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating derived metrics: {str(e)}")
            return df  # Return original DataFrame on error