"""Census G18 (Need for Assistance) data processing module.

This module handles the processing of ABS Census G18 table data, which contains
information about core activity need for assistance by age and sex.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import traceback

import polars as pl

from .. import config
from .. import utils
from ..config import G18_UNPIVOT

logger = logging.getLogger('ahgd_etl')

def process_g18_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G18 (Need for Assistance) Census CSV file.

    Args:
        csv_file: Path to the CSV file to process.

    Returns:
        Optional[pl.DataFrame]: Processed DataFrame or None if processing failed.
    """
    # Define table code
    table_code = "G18"
    logger.debug(f"[{table_code}] Processing file: {csv_file.name} using specific G18 processor.")

    # Define geographic column options (Consider moving to config)
    geo_column_options = ["region_id", "SA1_CODE_2021", "SA2_CODE_2021", "SA3_CODE_2021", "SA4_CODE_2021"]

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
        sex_prefixes = G18_UNPIVOT["sex_prefixes"]
        assistance_categories = G18_UNPIVOT["assistance_categories"]
        age_range_patterns = G18_UNPIVOT["age_range_patterns"]

        value_vars = [] # List of columns to unpivot
        parsed_cols = {}

        for col_name in df.columns:
            if col_name == geo_code_column:
                continue

            # Parse column name: Sex_AgeRange_AssistanceCategory (order might vary slightly)
            parts = col_name.split('_', 1)  # Split only at first underscore to get sex prefix
            if len(parts) < 2: continue

            sex = parts[0]
            if sex not in sex_prefixes: continue

            remaining = parts[1]

            assistance_category = None
            age_range = None
            age_part = None # Part used to determine age
            matched_assistance_key = None

            # First try to identify the assistance category
            for assist_cat in sorted(assistance_categories.keys(), key=len, reverse=True):
                if assist_cat in remaining:
                     matched_assistance_key = assist_cat
                     break

            if matched_assistance_key:
                 assistance_category = assistance_categories[matched_assistance_key]
                 # Try to determine age part robustly
                 try:
                     idx = remaining.index(matched_assistance_key)
                     if idx == 0: age_part = remaining[len(matched_assistance_key):].strip("_")
                     else: age_part = remaining[:idx].strip("_")
                 except ValueError:
                     age_part = remaining.replace(matched_assistance_key, "").strip("_")
            else:
                 continue # Could not parse assistance category

            # Find the age range from the age_part
            matched_age_key = None
            for age_pat in sorted(age_range_patterns.keys(), key=len, reverse=True):
                 if age_pat == age_part or age_pat.replace("_","") == age_part:
                     matched_age_key = age_pat
                     break

            if matched_age_key:
                 age_range = age_range_patterns[matched_age_key]
            elif age_part == 'Tot': # Handle total cases
                 age_range = 'total'
            else:
                 logger.debug(f"[{table_code}] Could not parse age range '{age_part}' in column {col_name}")
                 continue # Could not parse age range

            # Store the parsed information
            value_vars.append(col_name)
            parsed_cols[col_name] = {
                "sex": sex,
                "assistance_category": assistance_category,
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
             pl.col("column_info").map_dict({k: v["assistance_category"] for k, v in parsed_cols.items()}).alias("assistance_category"),
             pl.col("column_info").map_dict({k: v["age_range"] for k, v in parsed_cols.items()}).alias("age_range")
        ])

        # Clean geo_code and select final columns
        result_df = long_df.select([
            utils.clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
            pl.col("sex"),
            pl.col("assistance_category"),
            pl.col("age_range"),
            utils.safe_polars_int(pl.col("count")).alias("count") # Ensure count is integer
        ]).filter(pl.col("count").is_not_null() & (pl.col("count") > 0)) # Remove rows with null or zero count

        if len(result_df) == 0:
            logger.warning(f"[{table_code}] No non-zero data after unpivoting {csv_file.name}")
            return None

        logger.info(f"[{table_code}] Created unpivoted DataFrame with {len(result_df)} rows")
        return result_df

    except Exception as e:
        logger.error(f"[{table_code}] Error processing {table_code} file {csv_file.name}: {str(e)}")
        logger.error(traceback.format_exc()) # Log full traceback
        return None

def process_census_g18_data(zip_dir: Path, temp_extract_base: Path, output_dir: Path,
                           geo_output_path: Path, time_sk: Optional[int] = None) -> bool:
    """Process G18 Census data files and create assistance need fact table.
    
    Args:
        zip_dir (Path): Directory containing Census zip files
        temp_extract_base (Path): Base directory for temporary file extraction
        output_dir (Path): Directory to write output files
        geo_output_path (Path): Path to geographic dimension file
        time_sk (Optional[int]): Time dimension surrogate key
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    return utils.process_census_table(
        table_code="G18",
        process_file_function=process_g18_file,
        output_filename="fact_unpaid_care.parquet",
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )
