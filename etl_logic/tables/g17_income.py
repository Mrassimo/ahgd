"""Census G17 (Income) data processing module.

This module handles the processing of ABS Census G17 table data, which contains
information about personal, family, and household income.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import traceback

import polars as pl

from .. import config
from .. import utils

logger = logging.getLogger('ahgd_etl')

def process_g17_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a G17 Census CSV file (income data by age and sex).

    Args:
        csv_file (Path): Path to CSV file.

    Returns:
        Optional[pl.DataFrame]: Processed data or None if error.
    """
    # Define table code
    table_code = "G17"
    logger.debug(f"[{table_code}] Processing file: {csv_file.name} using specific G17 processor.")

    # Define geographic column options (Consider moving to config)
    geo_column_options = ['region_id', 'SA1_CODE21', 'SA2_CODE21', 'SA1_CODE_2021', 'SA2_CODE_2021',
                          'SA3_CODE_2021', 'SA4_CODE_2021', 'LGA_CODE_2021', 'STE_CODE_2021', 'AUS_CODE_2021']

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

        # Define patterns for the actual G17 data format (income data)
        sex_prefixes = ["M", "F", "P"]  # Male, Female, Person

        # Define income categories and their standardized names
        income_categories = config.G17_UNPIVOT["income_categories"]

        # Define age range patterns and their standardized format
        age_range_patterns = config.G17_UNPIVOT["age_range_patterns"]

        value_vars = [] # List of columns to unpivot
        parsed_cols = {} # Store parsing results

        for col_name in df.columns:
            if col_name == geo_code_column:
                continue

            # Parse column name: Sex_IncomeCategory_AgeRange
            parts = col_name.split('_', 1)  # Split only at first underscore to get sex prefix
            if len(parts) < 2:
                continue

            # Check if first part is a valid sex code
            sex = parts[0]
            if sex not in sex_prefixes:
                continue

            remaining = parts[1]

            # Find the income category
            income_category = None
            age_part = None # Store the part remaining after income category is found
            matched_income_key = None
            # Iterate longest keys first for better matching
            for inc_cat in sorted(income_categories.keys(), key=len, reverse=True):
                if inc_cat in remaining:
                    matched_income_key = inc_cat
                    break

            if matched_income_key:
                 income_category = income_categories[matched_income_key]
                 # Determine age part robustly
                 try:
                     idx = remaining.index(matched_income_key)
                     if idx == 0: age_part = remaining[len(matched_income_key):].strip("_")
                     else: age_part = remaining[:idx].strip("_")
                 except ValueError:
                     age_part = remaining.replace(matched_income_key, "").strip("_")
            else:
                 continue # Could not parse income category

            # Find the age range
            age_range = None
            matched_age_key = None
            for age_pat in sorted(age_range_patterns.keys(), key=len, reverse=True):
                 if age_pat == age_part or age_pat.replace("_","") == age_part:
                     matched_age_key = age_pat
                     break

            if matched_age_key:
                 age_range = age_range_patterns[matched_age_key]
            elif age_part == 'Tot': # Handle total cases if age_part is 'Tot'
                 age_range = 'total'
            else:
                 logger.debug(f"[{table_code}] Could not parse age range '{age_part}' in column {col_name}")
                 continue # Could not parse age range

            # Store the parsed information
            value_vars.append(col_name)
            parsed_cols[col_name] = {
                "sex": sex,
                "income_category": income_category,
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

        # Map parsed info back efficiently without intermediate DataFrame
        long_df = long_df.with_columns([
            pl.col("column_info").map_dict({k: v["sex"] for k, v in parsed_cols.items()}).alias("sex"),
            pl.col("column_info").map_dict({k: v["income_category"] for k, v in parsed_cols.items()}).alias("income_category"),
            pl.col("column_info").map_dict({k: v["age_range"] for k, v in parsed_cols.items()}).alias("age_range")
        ])

        # Clean geo_code and select final columns
        result_df = long_df.select([
            utils.clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
            pl.col("sex"),
            pl.col("income_category"),
            pl.col("age_range"),
            utils.safe_polars_int(pl.col("count")).alias("count") # Ensure count is integer
        ]).filter(pl.col("count").is_not_null() & (pl.col("count") > 0)) # Remove rows with null or zero count

        if len(result_df) == 0:
            logger.warning(f"[{table_code}] No non-zero data after unpivoting {csv_file.name}")
            return None

        logger.info(f"[{table_code}] Created unpivoted DataFrame with {len(result_df)} rows")
        return result_df

    except Exception as e:
        logger.error(f"[{table_code}] Error processing G17 file {csv_file.name}: {str(e)}")
        logger.error(traceback.format_exc()) # Log full traceback


from ..census import process_census_table

def process_g17_census_data(config: config.CensusConfig, census_data_path: Path, output_path: Path) -> bool:
    """Process all G17 Census data files and create income fact table.
    
    This function orchestrates the processing of all G17 Census files by:
    - Finding and extracting relevant CSV files
    - Processing each file using process_g17_file
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
        table_code="G17",
        process_file_function=process_g17_file,
        output_filename="fact_income.parquet",
        zip_dir=census_data_path,
        temp_extract_base=config.extract_path,
        output_dir=output_path,
        geo_output_path=config.geography_output_path,
        time_sk=config.time_sk
    )
    return None
