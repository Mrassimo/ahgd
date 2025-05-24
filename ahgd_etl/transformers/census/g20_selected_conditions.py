"""
Census G20 (Selected Health Conditions) data transformer.

This module handles the processing of ABS Census G20 table data, which contains
information about selected long-term health conditions and their counts.
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

class G20SelectedConditionsTransformer(BaseCensusTransformer):
    """
    Transformer for G20 (Selected Health Conditions) Census data.
    
    This class processes G20 Census data files, extracting the count of health
    conditions per person by geography, age, and sex.
    """
    
    def __init__(self):
        """Initialize G20 transformer."""
        super().__init__("G20")
        
        # Load G20-specific configuration
        self.sex_patterns = self.config.get("sex_patterns", {"M": "Male", "F": "Female", "P": "All Persons"})
        self.age_patterns = self.config.get("age_patterns", {})
        self.condition_count_map = self.config.get("condition_count_map", {
            "No_Conditions": 0,
            "One_Condition": 1,
            "Two_Conditions": 2,
            "Three_ormore_Conditions": 3,
            "Not_Stated": -1
        })
    
    def process_file(self, csv_file: Path, geo_output_path: Optional[Path] = None, 
                     time_sk: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Process a single G20 (Selected Health Conditions) Census CSV file.
        
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
            
            # Find geo_code column
            geo_code_column = self.find_geo_column(df)
            if not geo_code_column:
                self.logger.error(f"Could not find geographic code column in {csv_file.name}")
                return None
            self.logger.info(f"Found geographic code column: {geo_code_column}")
            
            value_vars = []
            parsed_cols = {}
            
            # Identify and parse value columns
            for col_name in df.columns:
                if col_name == geo_code_column:
                    continue
                
                # Basic check for structure (Sex_...)
                if not any(col_name.startswith(f"{sex}_") for sex in self.sex_patterns.keys()):
                    continue
                
                parts = col_name.split('_', 1)
                sex = self.sex_patterns.get(parts[0])
                if not sex:
                    continue
                
                remaining = parts[1]
                age_range = None
                condition_count = None
                age_part = None
                matched_condition_key = None
                
                # Try to extract condition count category first
                for cond_key in sorted(self.condition_count_map.keys(), key=len, reverse=True):
                    if cond_key in remaining:
                        matched_condition_key = cond_key
                        break
                
                if matched_condition_key:
                    condition_count = self.condition_count_map[matched_condition_key]
                    try:  # Determine age part
                        idx = remaining.index(matched_condition_key)
                        if idx == 0:
                            age_part = remaining[len(matched_condition_key):].strip("_")
                        else:
                            age_part = remaining[:idx].strip("_")
                    except ValueError:
                        age_part = remaining.replace(matched_condition_key, "").strip("_")
                else:
                    continue  # Failed to find condition count category
                
                # Determine age range from age_part
                matched_age_key = None
                for age_pat in sorted(self.age_patterns.keys(), key=len, reverse=True):
                    if age_pat == age_part or age_pat.replace("_", "") == age_part:
                        matched_age_key = age_pat
                        break
                
                if matched_age_key:
                    age_range = self.age_patterns[matched_age_key]
                elif age_part == 'Tot':
                    age_range = 'total'
                else:
                    self.logger.debug(f"Could not parse age range '{age_part}' in column {col_name}")
                    continue  # Age range not found
                
                # Store the parsed information
                value_vars.append(col_name)
                parsed_cols[col_name] = {
                    "sex": sex,
                    "condition_count": condition_count,
                    "age_range": age_range
                }
            
            if not value_vars:
                self.logger.error(f"No valid columns found to unpivot in {csv_file.name}")
                return None
            
            self.logger.info(f"Unpivoting {len(value_vars)} columns.")
            
            # Unpivot the data
            long_df = df.melt(
                id_vars=[geo_code_column],
                value_vars=value_vars,
                variable_name="column_info",
                value_name="count"
            )
            
            # Map parsed info back
            long_df = long_df.with_columns([
                pl.col("column_info").map_dict({k: v["sex"] for k, v in parsed_cols.items()}).alias("sex"),
                pl.col("column_info").map_dict({k: v["condition_count"] for k, v in parsed_cols.items()}).alias("condition_count"),
                pl.col("column_info").map_dict({k: v["age_range"] for k, v in parsed_cols.items()}).alias("age_range")
            ])
            
            # Clean geo_code and select final columns
            result_df = long_df.select([
                clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
                pl.col("sex"),
                pl.col("condition_count"),
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
    
    def refine_condition_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Refine the condition data by mapping condition counts to condition names.
        
        Args:
            df: DataFrame with condition count data
            
        Returns:
            Refined DataFrame with condition names
        """
        # Map condition counts to condition names
        condition_mapping = {
            0: "no_condition",
            1: "one_condition",
            2: "two_conditions",
            3: "three_or_more_conditions",
            -1: "not_stated"
        }
        
        # Create a mapping expression
        refined_df = df.with_columns([
            pl.col("condition_count").map_dict(condition_mapping).alias("condition")
        ])
        
        return refined_df