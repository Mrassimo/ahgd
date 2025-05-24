"""
Census G17 (Income) data transformer.

This module handles the processing of ABS Census G17 table data, which contains
information about personal, family, and household income.
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

class G17IncomeTransformer(BaseCensusTransformer):
    """
    Transformer for G17 (Income) Census data.
    
    This class processes G17 Census data files, extracting income information
    by geography, age, and sex.
    """
    
    def __init__(self):
        """Initialize G17 transformer."""
        super().__init__("G17")
        
        # Load G17-specific configuration
        self.sex_prefixes = self.config.get("sex_prefixes", ["M", "F", "P"])
        self.income_categories = self.config.get("income_categories", {})
        self.age_range_patterns = self.config.get("age_range_patterns", {})
    
    def process_file(self, csv_file: Path, geo_output_path: Optional[Path] = None, 
                     time_sk: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Process a single G17 (Income) Census CSV file.
        
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
            parsed_cols = {}  # Store parsing results
            
            for col_name in df.columns:
                if col_name == geo_code_column:
                    continue
                
                # Parse column name: Sex_IncomeCategory_AgeRange
                parts = col_name.split('_', 1)  # Split only at first underscore to get sex prefix
                if len(parts) < 2:
                    continue
                
                # Check if first part is a valid sex code
                sex = parts[0]
                if sex not in self.sex_prefixes:
                    continue
                
                remaining = parts[1]
                
                # Find the income category
                income_category = None
                age_part = None  # Store the part remaining after income category is found
                matched_income_key = None
                # Iterate longest keys first for better matching
                for inc_cat in sorted(self.income_categories.keys(), key=len, reverse=True):
                    if inc_cat in remaining:
                        matched_income_key = inc_cat
                        break
                
                if matched_income_key:
                    income_category = self.income_categories[matched_income_key]
                    # Determine age part robustly
                    try:
                        idx = remaining.index(matched_income_key)
                        if idx == 0:
                            age_part = remaining[len(matched_income_key):].strip("_")
                        else:
                            age_part = remaining[:idx].strip("_")
                    except ValueError:
                        age_part = remaining.replace(matched_income_key, "").strip("_")
                else:
                    continue  # Could not parse income category
                
                # Find the age range
                age_range = None
                matched_age_key = None
                for age_pat in sorted(self.age_range_patterns.keys(), key=len, reverse=True):
                    if age_pat == age_part or age_pat.replace("_", "") == age_part:
                        matched_age_key = age_pat
                        break
                
                if matched_age_key:
                    age_range = self.age_range_patterns[matched_age_key]
                elif age_part == 'Tot':  # Handle total cases if age_part is 'Tot'
                    age_range = 'total'
                else:
                    self.logger.debug(f"Could not parse age range '{age_part}' in column {col_name}")
                    continue  # Could not parse age range
                
                # Store the parsed information
                value_vars.append(col_name)
                parsed_cols[col_name] = {
                    "sex": sex,
                    "income_category": income_category,
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
            
            # Map parsed info back efficiently without intermediate DataFrame
            long_df = long_df.with_columns([
                pl.col("column_info").map_dict({k: v["sex"] for k, v in parsed_cols.items()}).alias("sex"),
                pl.col("column_info").map_dict({k: v["income_category"] for k, v in parsed_cols.items()}).alias("income_category"),
                pl.col("column_info").map_dict({k: v["age_range"] for k, v in parsed_cols.items()}).alias("age_range")
            ])
            
            # Clean geo_code and select final columns
            result_df = long_df.select([
                clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
                pl.col("sex"),
                pl.col("income_category"),
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
    
    def calculate_statistics(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate income statistics by demographic group.
        
        Args:
            df: Processed income data with counts by income category
            
        Returns:
            DataFrame with calculated income statistics
        """
        try:
            # Ensure required columns exist
            required_cols = ["geo_sk", "time_sk", "sex", "age_range", "income_category", "count"]
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns for income statistics calculation")
                return df
            
            # Define income category values for calculations
            income_midpoints = {
                "Negative/Nil": 0,
                "$1-$149": 75,
                "$150-$299": 225,
                "$300-$399": 350,
                "$400-$499": 450,
                "$500-$649": 575,
                "$650-$799": 725,
                "$800-$999": 900,
                "$1,000-$1,249": 1125,
                "$1,250-$1,499": 1375,
                "$1,500-$1,749": 1625,
                "$1,750-$1,999": 1875,
                "$2,000-$2,499": 2250,
                "$2,500-$2,999": 2750,
                "$3,000 or more": 4000,  # Estimate
                "Not stated": None,
                "Not applicable": None
            }
            
            # Create a grouping key for demographic groups
            result_df = df.with_columns([
                pl.lit(df["sex"] + "_" + df["age_range"]).alias("demographic_group")
            ])
            
            # Add income midpoint column
            result_df = result_df.with_columns([
                pl.col("income_category").map_dict(income_midpoints).alias("income_midpoint")
            ])
            
            # Calculate statistics by geographic region and demographic group
            stats_df = result_df.filter(pl.col("income_midpoint").is_not_null()).group_by(
                ["geo_sk", "time_sk", "demographic_group"]
            ).agg([
                pl.sum("count").alias("total_persons"),
                (pl.sum(pl.col("count") * pl.col("income_midpoint")) / pl.sum("count")).alias("mean_income"),
                # Add median income calculation if possible (complex in Polars)
            ])
            
            return stats_df
            
        except Exception as e:
            self.logger.error(f"Error calculating income statistics: {str(e)}")
            self.logger.error(traceback.format_exc())
            return df  # Return original DataFrame on error