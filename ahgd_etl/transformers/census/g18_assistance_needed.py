"""
Census G18 (Need for Assistance) data transformer.

This module handles the processing of ABS Census G18 table data, which contains
information about core activity need for assistance by age and sex.
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

class G18AssistanceNeededTransformer(BaseCensusTransformer):
    """
    Transformer for G18 (Need for Assistance) Census data.
    
    This class processes G18 Census data files, extracting information about
    core activity need for assistance by age and sex.
    """
    
    def __init__(self):
        """Initialize G18 transformer."""
        super().__init__("G18")
        
        # Load G18-specific configuration
        self.sex_prefixes = self.config.get("sex_prefixes", ["M", "F", "P"])
        self.assistance_categories = self.config.get("assistance_categories", {})
        self.age_range_patterns = self.config.get("age_range_patterns", {})
    
    def process_file(self, csv_file: Path, geo_output_path: Optional[Path] = None, 
                     time_sk: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Process a single G18 (Need for Assistance) Census CSV file.
        
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
            
            value_vars = []  # List of columns to unpivot
            parsed_cols = {}
            
            for col_name in df.columns:
                if col_name == geo_code_column:
                    continue
                
                # Parse column name: Sex_AgeRange_AssistanceCategory (order might vary slightly)
                parts = col_name.split('_', 1)  # Split only at first underscore to get sex prefix
                if len(parts) < 2:
                    continue
                
                sex = parts[0]
                if sex not in self.sex_prefixes:
                    continue
                
                remaining = parts[1]
                
                assistance_category = None
                age_range = None
                age_part = None  # Part used to determine age
                matched_assistance_key = None
                
                # First try to identify the assistance category
                for assist_cat in sorted(self.assistance_categories.keys(), key=len, reverse=True):
                    if assist_cat in remaining:
                        matched_assistance_key = assist_cat
                        break
                
                if matched_assistance_key:
                    assistance_category = self.assistance_categories[matched_assistance_key]
                    # Try to determine age part robustly
                    try:
                        idx = remaining.index(matched_assistance_key)
                        if idx == 0:
                            age_part = remaining[len(matched_assistance_key):].strip("_")
                        else:
                            age_part = remaining[:idx].strip("_")
                    except ValueError:
                        age_part = remaining.replace(matched_assistance_key, "").strip("_")
                else:
                    continue  # Could not parse assistance category
                
                # Find the age range from the age_part
                matched_age_key = None
                for age_pat in sorted(self.age_range_patterns.keys(), key=len, reverse=True):
                    if age_pat == age_part or age_pat.replace("_", "") == age_part:
                        matched_age_key = age_pat
                        break
                
                if matched_age_key:
                    age_range = self.age_range_patterns[matched_age_key]
                elif age_part == 'Tot':  # Handle total cases
                    age_range = 'total'
                else:
                    self.logger.debug(f"Could not parse age range '{age_part}' in column {col_name}")
                    continue  # Could not parse age range
                
                # Store the parsed information
                value_vars.append(col_name)
                parsed_cols[col_name] = {
                    "sex": sex,
                    "assistance_category": assistance_category,
                    "age_range": age_range
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
                pl.col("column_info").map_dict({k: v["sex"] for k, v in parsed_cols.items()}).alias("sex"),
                pl.col("column_info").map_dict({k: v["assistance_category"] for k, v in parsed_cols.items()}).alias("assistance_category"),
                pl.col("column_info").map_dict({k: v["age_range"] for k, v in parsed_cols.items()}).alias("age_range")
            ])
            
            # Clean geo_code and select final columns
            result_df = long_df.select([
                clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
                pl.col("sex"),
                pl.col("assistance_category"),
                pl.col("age_range"),
                safe_polars_int(pl.col("count")).alias("count")  # Ensure count is integer
            ]).filter(pl.col("count").is_not_null() & (pl.col("count") > 0))  # Remove rows with null or zero count
            
            if len(result_df) == 0:
                self.logger.warning(f"No non-zero data after unpivoting {csv_file.name}")
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
            
            self.logger.info(f"Created DataFrame with {len(result_df)} rows")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error processing CSV file {csv_file.name}: {str(e)}")
            self.logger.error(traceback.format_exc())  # Log full traceback
            return None
    
    def create_boolean_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert assistance categories to boolean indicator columns.
        
        Args:
            df: Input DataFrame with assistance_category column
            
        Returns:
            DataFrame with boolean indicator columns
        """
        try:
            # Create boolean indicators for each assistance category
            result_df = df.with_columns([
                (pl.col("assistance_category") == "Has need for assistance").alias("has_need_for_assistance"),
                (pl.col("assistance_category") == "Does not have need for assistance").alias("does_not_have_need_for_assistance"),
                (pl.col("assistance_category") == "Not stated").alias("assistance_not_stated")
            ])
            
            # Drop the original assistance_category column
            if "assistance_category" in result_df.columns:
                result_df = result_df.drop("assistance_category")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error creating boolean indicators: {str(e)}")
            return df  # Return original DataFrame on error