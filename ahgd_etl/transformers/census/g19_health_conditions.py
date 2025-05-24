"""
Census G19 (Health Conditions) data transformer.

This module handles the processing of ABS Census G19 table data, which contains
information about long-term health conditions by age and sex.
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
from ...models.facts import HealthConditionFact

class G19HealthConditionsTransformer(BaseCensusTransformer):
    """
    Transformer for G19 (Long-Term Health Conditions) Census data.
    
    This class processes G19 Census data files, extracting health condition information
    by geography, age, and sex.
    """
    
    def __init__(self):
        """Initialize G19 transformer."""
        super().__init__("G19")
        
        # Load G19-specific configuration
        self.sex_prefixes = self.config.get("sex_prefixes", ["M", "F", "P"])
        self.health_conditions = self.config.get("health_conditions_map", {})
        self.age_range_patterns = self.config.get("age_range_patterns", {})
    
    def process_file(self, csv_file: Path, geo_output_path: Optional[Path] = None, 
                     time_sk: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Process a single G19 (Long-Term Health Conditions) Census CSV file.
        
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
            
            # Identify columns to unpivot
            value_vars = []  # List of columns to unpivot
            parsed_cols = {}
            
            for col_name in df.columns:
                if col_name == geo_code_column:
                    continue
                
                # Parse column name format: Sex_Condition_AgeRange (order may vary)
                parts = col_name.split('_', 1)
                if len(parts) < 2:
                    continue
                
                sex = parts[0]
                if sex not in self.sex_prefixes:
                    continue
                
                remaining = parts[1]
                
                condition = None
                age_range = None
                age_part = None
                matched_condition_key = None
                
                # Find the health condition first (most distinct usually)
                # Iterate known condition codes from longest to shortest
                for cond_key in sorted(self.health_conditions.keys(), key=len, reverse=True):
                    if cond_key in remaining:
                        matched_condition_key = cond_key
                        break
                
                if matched_condition_key:
                    condition = self.health_conditions[matched_condition_key]
                    # Determine age part
                    try:
                        idx = remaining.index(matched_condition_key)
                        if idx == 0:
                            age_part = remaining[len(matched_condition_key):].strip("_")
                        else:
                            age_part = remaining[:idx].strip("_")
                    except ValueError:
                        age_part = remaining.replace(matched_condition_key, "").strip("_")
                else:
                    continue  # Condition not found
                
                # Find the age range from age_part
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
                    continue  # Age range not found
                
                # Store the parsed information
                value_vars.append(col_name)
                parsed_cols[col_name] = {
                    "sex": sex,
                    "condition": condition,
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
                pl.col("column_info").map_dict({k: v["condition"] for k, v in parsed_cols.items()}).alias("condition"),
                pl.col("column_info").map_dict({k: v["age_range"] for k, v in parsed_cols.items()}).alias("age_range")
            ])
            
            # Clean geo_code and select final columns
            result_df = long_df.select([
                clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
                pl.col("sex"),
                pl.col("condition").alias("health_condition"),
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
    
    def process_detailed(self, csv_file: Path, geo_output_path: Optional[Path] = None,
                        time_sk: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Process a G19 detailed file (G19A, G19B, G19C) to extract specific health condition data.
        
        Args:
            csv_file: Path to the G19 CSV file to process
            geo_output_path: Path to the geographic dimension file (for lookups)
            time_sk: Time dimension surrogate key to use
            
        Returns:
            Processed DataFrame or None if processing failed
        """
        try:
            # Read CSV file
            df = pl.read_csv(csv_file, truncate_ragged_lines=True)
            self.logger.info(f"Read {len(df)} rows from {csv_file.name}")
            
            # Identify the geo_code column
            geo_code_column = self.find_geo_column(df)
            if not geo_code_column:
                self.logger.error(f"Could not find geographic code column in {csv_file.name}")
                return None
            self.logger.info(f"Found geographic code column: {geo_code_column}")
            
            # Identify G19 health condition columns
            value_vars = []
            parsed_cols = {}
            
            # Iterate through columns to parse them
            for col_name in df.columns:
                if col_name == geo_code_column:
                    continue
                
                parts = col_name.split('_', 1)
                if len(parts) < 2:
                    continue
                
                sex = parts[0]
                if sex not in self.sex_prefixes:
                    continue
                
                remaining = parts[1]
                
                condition = None
                age_range = None
                age_part = None
                matched_condition_key = None
                
                # Find condition
                for cond_key in sorted(self.health_conditions.keys(), key=len, reverse=True):
                    if cond_key in remaining:
                        matched_condition_key = cond_key
                        break
                
                if matched_condition_key:
                    condition = self.health_conditions[matched_condition_key]
                    try:
                        idx = remaining.index(matched_condition_key)
                        if idx == 0:
                            age_part = remaining[len(matched_condition_key):].strip("_")
                        else:
                            age_part = remaining[:idx].strip("_")
                    except ValueError:
                        age_part = remaining.replace(matched_condition_key, "").strip("_")
                else:
                    continue
                
                # Find age range
                matched_age_key = None
                for age_pat in sorted(self.age_range_patterns.keys(), key=len, reverse=True):
                    if age_pat == age_part or age_pat.replace("_", "") == age_part:
                        matched_age_key = age_pat
                        break
                
                if matched_age_key:
                    age_range = self.age_range_patterns[matched_age_key]
                elif age_part == 'Tot':
                    age_range = 'total'
                else:
                    continue
                
                value_vars.append(col_name)
                parsed_cols[col_name] = {
                    "sex": sex,
                    "condition": condition,
                    "age_range": age_range
                }
            
            if not value_vars:
                self.logger.error(f"No valid condition columns found in {csv_file.name}")
                return None
            
            self.logger.info(f"Identified {len(value_vars)} condition columns for unpivoting.")
            
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
                pl.col("column_info").map_dict({k: v["condition"] for k, v in parsed_cols.items()}).alias("condition"),
                pl.col("column_info").map_dict({k: v["age_range"] for k, v in parsed_cols.items()}).alias("age_range")
            ])
            
            # Final selection and cleaning
            result_df = long_df.select([
                clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
                pl.col("sex"),
                pl.col("condition").alias("health_condition"),
                pl.col("age_range"),
                safe_polars_int(pl.col("count")).alias("count")
            ]).filter(pl.col("count").is_not_null() & (pl.col("count") > 0))
            
            if len(result_df) == 0:
                self.logger.warning(f"No non-zero condition data found after processing {csv_file.name}")
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
            
            self.logger.info(f"Created detailed DataFrame with {len(result_df)} rows")
            return result_df
        
        except Exception as e:
            self.logger.error(f"Error processing detailed file {csv_file.name}: {e}")
            self.logger.error(traceback.format_exc())  # Log full traceback
            return None

    def create_fact_table(self, df: pl.DataFrame) -> HealthConditionFact:
        """
        Create a HealthConditionFact model from the processed DataFrame.
        
        Args:
            df: Processed DataFrame with health condition data
            
        Returns:
            HealthConditionFact model
        """
        return HealthConditionFact.from_dataframe(df)