"""
Census G21 (Health Conditions by Characteristics) data transformer.

This module handles the processing of ABS Census G21 table data, which contains
information about health conditions broken down by various person characteristics
such as age, sex, country of birth, and labour force status.
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

class G21ConditionsByCharacteristicsTransformer(BaseCensusTransformer):
    """
    Transformer for G21 (Health Conditions by Characteristics) Census data.
    
    This class processes G21 Census data files, extracting health condition information
    by various person characteristics.
    """
    
    def __init__(self):
        """Initialize G21 transformer."""
        super().__init__("G21")
        
        # Load G21-specific configuration
        self.health_conditions = self.config.get("health_conditions_map", {})
        self.characteristic_types = self.config.get("characteristic_types", {})
        
    def process_file(self, csv_file: Path, geo_output_path: Optional[Path] = None, 
                     time_sk: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Process a single G21 (Health Conditions by Characteristics) Census CSV file.
        
        Args:
            csv_file: Path to the CSV file to process
            geo_output_path: Path to the geographic dimension file (for lookups)
            time_sk: Time dimension surrogate key to use
            
        Returns:
            Processed DataFrame or None if processing failed
        """
        table_code = "G21"
        self.logger.info(f"Processing G21 file: {csv_file.name}")
        
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
            
            # Define variables for unpivoting
            value_vars = []  # List of columns to unpivot
            parsed_cols = {}  # Store parsing results
            
            # Dynamically parse columns based on config mappings
            for col_name in df.columns:
                if col_name == geo_code_column:
                    continue
                
                characteristic_type = None
                characteristic_value = None
                condition = None
                remaining_part = col_name
                
                # 1. Identify Characteristic Type and Value
                matched_char_key = None
                for char_key in sorted(self.characteristic_types.keys(), key=len, reverse=True):
                    if remaining_part.startswith(f"{char_key}_"):
                        characteristic_type = self.characteristic_types[char_key]
                        remaining_part = remaining_part[len(char_key)+1:]  # Remove prefix and underscore
                        matched_char_key = char_key
                        break
                
                if not characteristic_type:
                    # Handle cases like Tot_Tot_Arth where char type might be 'Tot'
                    if remaining_part.startswith("Tot_"):
                        characteristic_type = self.characteristic_types.get("Tot", "Total")  # Use config or default
                        remaining_part = remaining_part[len("Tot_"):]
                        matched_char_key = "Tot"
                    else:
                        self.logger.debug(f"Could not determine characteristic type for column: {col_name}")
                        continue
                
                # 2. Identify Condition from the rest
                matched_cond_key = None
                for cond_key in sorted(self.health_conditions.keys(), key=len, reverse=True):
                    # Need to handle cases where condition is at end or start
                    if remaining_part.endswith(f"_{cond_key}"):
                        condition = self.health_conditions[cond_key]
                        characteristic_value = remaining_part[:-len(cond_key)-1]
                        matched_cond_key = cond_key
                        break
                    elif remaining_part == cond_key:  # Case like P_Tot_Tot (already handled char type 'Tot')
                        condition = self.health_conditions[cond_key]
                        characteristic_value = matched_char_key  # Use the char key ('Tot') as value
                        matched_cond_key = cond_key
                        break
                
                if not condition:
                    # Handle special Tot cases (e.g., COB_Aus_Tot)
                    if remaining_part.endswith("_Tot"):
                        condition = self.health_conditions.get("Tot", "total")
                        characteristic_value = remaining_part[:-len("_Tot")]
                        matched_cond_key = "Tot"
                    else:
                        self.logger.debug(f"Could not determine condition for remaining part '{remaining_part}' in column: {col_name}")
                        continue
                
                # Store parsed info
                value_vars.append(col_name)
                parsed_cols[col_name] = {
                    "characteristic_type": characteristic_type,
                    "characteristic_value": characteristic_value,
                    "condition": condition
                }
            
            if not value_vars:
                self.logger.error(f"No valid columns found to unpivot in {csv_file.name}")
                return None
            
            self.logger.info(f"Unpivoting {len(value_vars)} columns.")
            
            # Unpivot using melt
            long_df = df.melt(
                id_vars=[geo_code_column],
                value_vars=value_vars,
                variable_name="column_info",
                value_name="count"
            )
            
            # Map parsed info back
            long_df = long_df.with_columns([
                clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),  # Clean geo_code early
                pl.col("column_info").map_dict({k: v["characteristic_type"] for k, v in parsed_cols.items()}).alias("characteristic_type"),
                pl.col("column_info").map_dict({k: v["characteristic_value"] for k, v in parsed_cols.items()}).alias("characteristic_value"),
                pl.col("column_info").map_dict({k: v["condition"] for k, v in parsed_cols.items()}).alias("condition"),
                safe_polars_int(pl.col("count")).alias("count")
            ])
            
            # Select final columns and filter
            result_df = long_df.select([
                "geo_code",
                "characteristic_type",
                "characteristic_value",
                pl.col("condition").alias("health_condition"),
                "count"
            ]).filter(pl.col("count").is_not_null() & (pl.col("count") > 0))
            
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
            
            self.logger.info(f"Created DataFrame with {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error processing CSV file {csv_file.name}: {str(e)}")
            self.logger.error(traceback.format_exc())  # Log full traceback
            return None
    
    def ensure_unique_grain(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Ensure that the fact table has a unique grain.
        
        Args:
            df: Input DataFrame with potentially duplicate grain
            
        Returns:
            DataFrame with unique grain
        """
        try:
            # Define the grain columns
            grain_cols = ["geo_sk", "time_sk", "condition_sk", "characteristic_sk"]
            
            # Ensure no duplicates on grain
            unique_rows = df.group_by(grain_cols).agg([
                pl.col("count").sum().alias("count")
            ])
            
            # Check if we have actually deduplicated anything
            if len(unique_rows) < len(df):
                self.logger.info(f"Reduced {len(df)} rows to {len(unique_rows)} unique grain combinations")
            else:
                self.logger.info(f"No duplicate grain combinations found")
            
            return unique_rows
            
        except Exception as e:
            self.logger.error(f"Error ensuring unique grain: {str(e)}")
            self.logger.error(traceback.format_exc())
            # Return original DataFrame if deduplication fails
            return df