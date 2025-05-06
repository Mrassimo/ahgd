"""Census G19 (Health Conditions) data processing module.

This module handles the processing of ABS Census G19 table data, which contains
information about long-term health conditions by age and sex.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import traceback

import polars as pl

from .. import config
from .. import utils
from ..census import process_census_table
from ..config import CensusConfig

logger = logging.getLogger('ahgd_etl')

def process_g19_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G19 (Long-Term Health Conditions) Census CSV file.

    Args:
        csv_file: Path to the CSV file to process.

    Returns:
        Optional[pl.DataFrame]: Processed DataFrame or None if processing failed.
    """
    # Define table code
    table_code = "G19"
    logger.debug(f"[{table_code}] Processing file: {csv_file.name} using specific G19 processor.")

    # Define geographic column options
    geo_column_options = config.GEO_COLUMN_OPTIONS

    try:
        # Read the CSV file
        df = pl.read_csv(csv_file, truncate_ragged_lines=True)
        logger.info(f"[{table_code}] Read {len(df)} rows from {csv_file.name}")

        # Identify geo_code column
        geo_code_column = utils.find_geo_column(df, geo_column_options)
        if not geo_code_column:
            logger.error(f"[{table_code}] Could not find geographic code column in {csv_file.name}")
            return None
        logger.info(f"[{table_code}] Found geographic code column: {geo_code_column}")

        # Define patterns for column interpretation
        sex_prefixes = config.G19_UNPIVOT["sex_prefixes"]
    except Exception as e:
        logger.error(f"[{table_code}] Error processing CSV file {csv_file.name}: {str(e)}", exc_info=True)
        return None
        health_conditions = config.G19_UNPIVOT["health_conditions_map"]
        age_range_patterns = config.G19_UNPIVOT["age_range_patterns"]

        value_vars = [] # List of columns to unpivot
        parsed_cols = {}

        for col_name in df.columns:
            if col_name == geo_code_column: continue

            # Parse column name format: Sex_Condition_AgeRange (order may vary)
            parts = col_name.split('_', 1)
            if len(parts) < 2: continue

            sex = parts[0]
            if sex not in sex_prefixes: continue

            remaining = parts[1]

            condition = None
            age_range = None
            age_part = None
            matched_condition_key = None

            # Find the health condition first (most distinct usually)
            # Iterate known condition codes from longest to shortest
            for cond_key in sorted(health_conditions.keys(), key=len, reverse=True):
                 if cond_key in remaining:
                     matched_condition_key = cond_key
                     break

            if matched_condition_key:
                condition = health_conditions[matched_condition_key]
                # Determine age part
                try:
                     idx = remaining.index(matched_condition_key)
                     if idx == 0: age_part = remaining[len(matched_condition_key):].strip("_")
                     else: age_part = remaining[:idx].strip("_")
                except ValueError:
                     age_part = remaining.replace(matched_condition_key, "").strip("_")
            else:
                 continue # Condition not found

            # Find the age range from age_part
            matched_age_key = None
            for age_pat in sorted(age_range_patterns.keys(), key=len, reverse=True):
                 if age_pat == age_part or age_pat.replace("_", "") == age_part:
                     matched_age_key = age_pat
                     break

            if matched_age_key:
                 age_range = age_range_patterns[matched_age_key]
            elif age_part == 'Tot': # Handle total cases
                 age_range = 'total'
            else:
                 logger.debug(f"[{table_code}] Could not parse age range '{age_part}' in column {col_name}")
                 continue # Age range not found

            # Store the parsed information
            value_vars.append(col_name)
            parsed_cols[col_name] = {
                "sex": sex,
                "condition": condition,
                "age_range": age_range
            }

        if not value_vars:
            logger.error(f"[{table_code}] No valid columns found to unpivot in {csv_file.name}")
            return None

        logger.info(f"[{table_code}] Unpivoting {len(value_vars)} columns.")

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
            utils.clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
            pl.col("sex"),
            pl.col("condition").alias("health_condition"),
            pl.col("age_range"),
            utils.safe_polars_int(pl.col("count")).alias("count") # Ensure count is integer
        ]).filter(pl.col("count").is_not_null() & (pl.col("count") > 0)) # Remove rows with null or zero count

        if len(result_df) == 0:
            logger.warning(f"[{table_code}] No non-zero data after unpivoting {csv_file.name}")
            return None
    
        logger.info(f"[{table_code}] Created unpivoted DataFrame with {len(result_df)} rows")
        return result_df

def process_g19_census_data(config: CensusConfig, census_data_path: Path, output_path: Path) -> bool:
        """Process all G19 Census data files and create health conditions fact table.
        
        This function orchestrates the processing of all G19 Census files by:
        - Finding and extracting relevant CSV files
        - Processing each file using process_g19_file
        - Combining results into a standardized fact table
        - Writing final output as Parquet
        
        Args:
            config: Census configuration object
            census_data_path: Directory containing Census data files
            output_path: Directory where output files should be written
            
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        return process_census_table(
            table_code="G19",
            process_file_function=process_g19_file,
            output_filename="fact_health_conditions.parquet",
            zip_dir=census_data_path,
            temp_extract_base=config.extract_path,
            output_dir=output_path,
            geo_output_path=config.geography_output_path,
            time_sk=config.time_sk
        )
def process_g19_detailed_csv(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a G19 detailed file (G19A, G19B, G19C) to extract specific health condition data.

    Args:
        csv_file: Path to the G19 CSV file to process.

    Returns:
        Optional[pl.DataFrame]: Processed DataFrame with unpivoted health condition data,
                               or None if processing failed.
    """
    table_code = "G19_detailed" # Use distinct code for logging
    logger.info(f"[{table_code}] Processing G19 detailed file: {csv_file.name}")

    try:
        # Read CSV file
        df = pl.read_csv(csv_file, truncate_ragged_lines=True) # Added truncate_ragged_lines
        logger.info(f"[{table_code}] Read {len(df)} rows from {csv_file.name}")

        # Log column names for debugging
        logger.debug(f"[{table_code}] Available columns: {df.columns}") # Log all for debug

        # Identify the geo_code column - handle various geographic codes
        geo_cols = config.GEO_COLUMN_OPTIONS

        geo_code_column = utils.find_geo_column(df, geo_cols)

        if not geo_code_column:
            logger.error(f"[{table_code}] Could not find geographic code column in {csv_file.name}")
            return None
        logger.info(f"[{table_code}] Found geographic code column: {geo_code_column}")

        # Identify G19 health condition columns
        known_conditions = config.G19_UNPIVOT["health_conditions_map"]
        sex_prefixes = config.G19_UNPIVOT["sex_prefixes"]
        age_groups = config.G19_UNPIVOT["age_range_patterns"]

        value_vars = []
        parsed_cols = {}

        # Iterate through columns to parse them
        for col_name in df.columns:
             if col_name == geo_code_column: continue

             parts = col_name.split('_', 1)
             if len(parts) < 2: continue

             sex = parts[0]
             if sex not in sex_prefixes: continue

             remaining = parts[1]

             condition = None
             age_range = None
             age_part = None
             matched_condition_key = None

             # Find condition
             for cond_key in sorted(known_conditions.keys(), key=len, reverse=True):
                  if cond_key in remaining:
                      matched_condition_key = cond_key
                      break

             if matched_condition_key:
                 condition = known_conditions[matched_condition_key]
                 try:
                      idx = remaining.index(matched_condition_key)
                      if idx == 0: age_part = remaining[len(matched_condition_key):].strip("_")
                      else: age_part = remaining[:idx].strip("_")
                 except ValueError:
                      age_part = remaining.replace(matched_condition_key, "").strip("_")
             else:
                  continue

             # Find age range
             matched_age_key = None
             for age_pat in sorted(age_groups.keys(), key=len, reverse=True):
                  if age_pat == age_part or age_pat.replace("_", "") == age_part:
                     matched_age_key = age_pat
                     break

             if matched_age_key:
                  age_range = age_groups[matched_age_key]
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
            logger.error(f"[{table_code}] No valid condition columns found in {csv_file.name}")
            return None

        logger.info(f"[{table_code}] Identified {len(value_vars)} condition columns for unpivoting.")

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
             utils.clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
             pl.col("sex"),
             pl.col("condition").alias("health_condition"),
             pl.col("age_range"),
             utils.safe_polars_int(pl.col("count")).alias("count")
        ]).filter(pl.col("count").is_not_null() & (pl.col("count") > 0))

        if len(result_df) == 0:
            logger.warning(f"[{table_code}] No non-zero condition data found after processing {csv_file.name}")
            return None

        logger.info(f"[{table_code}] Created unpivoted DataFrame with {len(result_df)} rows")
        return result_df

    except Exception as e:
        logger.error(f"[{table_code}] Error processing {csv_file.name}: {e}")
        logger.error(traceback.format_exc()) # Log full traceback
        return None
