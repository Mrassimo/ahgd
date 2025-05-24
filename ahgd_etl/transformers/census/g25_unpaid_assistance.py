"""
Census G25 (Unpaid Assistance) data transformer.

This module handles the processing of ABS Census G25 table data, which contains
information about unpaid assistance to persons with a disability by age and sex.
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

class G25UnpaidAssistanceTransformer(BaseCensusTransformer):
    """
    Transformer for G25 (Unpaid Assistance) Census data.
    
    This class processes G25 Census data files, extracting information about
    unpaid assistance to persons with a disability by age and sex.
    """
    
    def __init__(self):
        """Initialize G25 transformer."""
        super().__init__("G25")
        
        # Load G25-specific configuration
        self.care_types = self.config.get("care_types", {
            "Provided_unpaid_assistance": "Provided unpaid assistance",
            "No_unpaid_assistance_provided": "No unpaid assistance provided",
            "Not_stated_unpaid_assistance": "Not stated"
        })
        self.age_sex_patterns = self.config.get("age_sex_patterns", {})
    
    def process_file(self, csv_file: Path, geo_output_path: Optional[Path] = None, 
                     time_sk: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Process a single G25 (Unpaid Assistance) Census CSV file.
        
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
            
            # Define expected columns based on G25 data structure
            # First, try to identify columns by mapping from configuration
            measure_column_map = {}
            for care_type_key, care_type_value in self.care_types.items():
                for age_sex_key, age_sex_info in self.age_sex_patterns.items():
                    # Create column name patterns like M_0_4_Provided_unpaid_assistance
                    col_pattern = f"{age_sex_key}_{care_type_key}"
                    # Map to standardized values
                    measure_column_map[col_pattern] = {
                        "sex": age_sex_info.get("sex"),
                        "age_group": age_sex_info.get("age_group"),
                        "care_type": care_type_value
                    }
            
            # Find valid columns to unpivot
            value_vars = []
            parsed_cols = {}
            
            for col_name in df.columns:
                if col_name == geo_code_column:
                    continue
                
                # Try exact match from mapping
                if col_name in measure_column_map:
                    value_vars.append(col_name)
                    parsed_cols[col_name] = measure_column_map[col_name]
                    continue
                
                # Try to parse if not in exact mapping
                parts = col_name.split('_', 1)
                if len(parts) < 2:
                    continue
                
                sex_prefix = parts[0]
                if sex_prefix not in ["M", "F", "P"]:
                    continue
                
                remaining = parts[1]
                
                # Look for care type in column name
                care_type = None
                for care_key, care_value in self.care_types.items():
                    if care_key in remaining:
                        care_type = care_value
                        remaining = remaining.replace(care_key, "").strip("_")
                        break
                
                if not care_type:
                    continue
                
                # Determine age group from remaining part
                age_group = None
                for pattern, info in self.age_sex_patterns.items():
                    if pattern.startswith(f"{sex_prefix}_") and pattern.replace(f"{sex_prefix}_", "") == remaining:
                        age_group = info.get("age_group")
                        sex = info.get("sex")
                        break
                
                if not age_group:
                    # Try some common patterns
                    if remaining == "0_4":
                        age_group = "0-4"
                    elif remaining == "5_9":
                        age_group = "5-9"
                    elif remaining == "Total":
                        age_group = "All ages"
                    else:
                        self.logger.debug(f"Could not determine age group from: {remaining} in column {col_name}")
                        continue
                
                # If we got here, we have all the required information
                value_vars.append(col_name)
                parsed_cols[col_name] = {
                    "sex": sex if 'sex' in locals() else sex_prefix,
                    "age_group": age_group,
                    "care_type": care_type
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
                clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
                pl.col("column_info").map_dict({k: v["sex"] for k, v in parsed_cols.items()}).alias("sex"),
                pl.col("column_info").map_dict({k: v["age_group"] for k, v in parsed_cols.items()}).alias("age_group"),
                pl.col("column_info").map_dict({k: v["care_type"] for k, v in parsed_cols.items()}).alias("care_type"),
                safe_polars_int(pl.col("count")).alias("count")  # Ensure count is integer
            ])
            
            # Remove rows with null or zero counts
            result_df = long_df.filter(pl.col("count").is_not_null() & (pl.col("count") > 0))
            
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
    
    def standardize_care_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Standardize care type values in the DataFrame.
        
        Args:
            df: Input DataFrame with care_type column
            
        Returns:
            DataFrame with standardized care types
        """
        # Ensure care_type column exists
        if "care_type" not in df.columns:
            self.logger.warning("No care_type column found for standardization")
            return df
        
        # Define standardized mappings
        standard_care_types = {
            "provided assistance": "Provided unpaid assistance",
            "provided unpaid assistance": "Provided unpaid assistance",
            "no assistance provided": "No unpaid assistance provided",
            "no unpaid assistance provided": "No unpaid assistance provided",
            "not stated": "Not stated",
            "assistance not stated": "Not stated"
        }
        
        # Apply mappings (case-insensitive)
        result_df = df.with_columns([
            pl.col("care_type").str.to_lowercase().map_dict(
                {k.lower(): v for k, v in standard_care_types.items()},
                default=pl.col("care_type")
            ).alias("unpaid_care_type")
        ])
        
        # Remove original care_type column if standardization was successful
        if "unpaid_care_type" in result_df.columns:
            result_df = result_df.drop("care_type")
        
        return result_df