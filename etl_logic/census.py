"""Census data processing module for the AHGD ETL pipeline.

This module handles the processing of ABS Census data, specifically the G01
(Selected Person Characteristics) table, transforming it into a standardised
Parquet format and linking it with geographic boundaries.
"""

import logging
import re
import zipfile
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime

import pandas as pd
import geopandas as gpd
import polars as pl
import pyarrow
import pyarrow.parquet as pq

from . import config
from . import utils

logger = logging.getLogger('ahgd_etl')

def find_census_files(zip_dir: Path, pattern: str) -> List[Path]:
    """Find ZIP files containing Census CSV files matching the pattern.
    
    Args:
        zip_dir (Path): Directory to search for ZIP files.
        pattern (str): Regex pattern to match filenames.
        
    Returns:
        List[Path]: List of matching ZIP files.
    """
    logger.info(f"Looking for Census CSV files with pattern: {pattern}")
    matching_files = []
    
    for zip_path in zip_dir.glob("*.zip"):
        logger.info(f"Checking zip file: {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path) as zf:
                matching_csv_files = []
                for name in zf.namelist():
                    logger.debug(f"Checking file in zip: {name}")
                    match = re.search(pattern, name, re.IGNORECASE)
                    if match:
                        logger.info(f"Found matching file: {name}")
                        matching_csv_files.append(name)
                
                if matching_csv_files:
                    matching_files.append(zip_path)
                    logger.info(f"Zip file {zip_path.name} contains {len(matching_csv_files)} matching files")
                else:
                    logger.warning(f"No matching files found in {zip_path.name}")
                    # Print actual file list to help diagnose issues
                    csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                    if len(csv_files) > 0:
                        logger.info(f"CSV files in zip: {', '.join(csv_files[:5])}{'...' if len(csv_files) > 5 else ''}")
        except Exception as e:
            logger.error(f"Error scanning {zip_path}: {str(e)}")
    
    logger.info(f"Found {len(matching_files)} matching zip files: {[f.name for f in matching_files]}")
    return matching_files

def extract_census_files(zip_file: Path, pattern: str, extract_dir: Path) -> List[Path]:
    """Extract Census CSV files from a ZIP file.
    
    Args:
        zip_file (Path): Path to ZIP file.
        pattern (str): Regex pattern to match filenames.
        extract_dir (Path): Directory to extract to.
        
    Returns:
        List[Path]: List of extracted CSV files.
    """
    extracted_files = []
    try:
        with zipfile.ZipFile(zip_file) as zf:
            for name in zf.namelist():
                if re.search(pattern, name, re.IGNORECASE):
                    try:
                        zf.extract(name, extract_dir)
                        extracted_files.append(extract_dir / name)
                    except Exception as e:
                        logger.error(f"Error extracting {name}: {str(e)}")
    except Exception as e:
        logger.error(f"Error opening {zip_file}: {str(e)}")
    return extracted_files

def _process_single_census_csv(csv_file: Path, geo_column_options: List[str], 
                              measure_column_map: Dict[str, List[str]], 
                              required_target_columns: List[str],
                              table_code: str) -> Optional[pl.DataFrame]:
    """Process a single Census CSV file using standardised column mapping.
    
    This helper function implements a generic approach to process Census CSV files
    using flexible column mapping. It handles:
    - Reading the CSV file with Polars
    - Finding the appropriate geographic column
    - Mapping source columns to target columns based on provided mapping
    - Cleaning and standardising data types
    - Data quality validation
    
    Args:
        csv_file (Path): Path to the Census CSV file to process
        geo_column_options (List[str]): List of possible geographic column names to look for
        measure_column_map (Dict[str, List[str]]): Mapping of target column names to possible 
                                                  source column names
        required_target_columns (List[str]): List of required columns that must be found
        table_code (str): Census table code (e.g., "G17", "G18") for logging context
        
    Returns:
        Optional[pl.DataFrame]: Processed DataFrame with standardised column names and
                               data types if successful, or None if processing failed
    """
    logger.info(f"[{table_code}] Processing file: {csv_file.name}")
    
    try:
        # Read CSV with Polars - added truncate_ragged_lines to handle files with ragged lines
        df = pl.read_csv(csv_file, truncate_ragged_lines=True)
        
        # Find geographic code column
        geo_col = utils.find_geo_column(df, geo_column_options)
        if not geo_col:
            logger.error(f"[{table_code}] No geographic code column found in {csv_file}")
            return None
        
        # Initialise result dictionary to build the output DataFrame
        result_cols = {'geo_code': utils.clean_polars_geo_code(pl.col(geo_col))}
        
        # Find and map measure columns
        found_columns = {}
        for target_col, source_options in measure_column_map.items():
            found = False
            for source_col in source_options:
                if source_col in df.columns:
                    found_columns[target_col] = source_col
                    # Apply safe integer conversion for numeric columns
                    result_cols[target_col] = utils.safe_polars_int(pl.col(source_col))
                    found = True
                    logger.debug(f"[{table_code}] Mapped '{source_col}' to '{target_col}'")
                    break
            
            if not found and target_col in required_target_columns:
                logger.error(f"[{table_code}] Required column '{target_col}' not found in {csv_file}.")
                logger.error(f"[{table_code}] Available columns: {df.columns}")
                return None
        
        # Create result DataFrame
        result_df = df.select([expr.alias(col_name) for col_name, expr in result_cols.items()])
        
        # Drop rows with invalid codes
        result_df = result_df.drop_nulls(subset=['geo_code'])
        
        # Check if we have reasonable data
        if len(result_df) == 0:
            logger.warning(f"[{table_code}] No valid data rows in {csv_file} after processing")
            return None
        
        # Data Quality Check: Validate the processed dataframe
        if not validate_processed_df(result_df, table_code, csv_file):
            logger.warning(f"[{table_code}] Data validation issues found in {csv_file.name}, but processing will continue")
        
        logger.info(f"[{table_code}] Successfully processed {csv_file.name}: {len(result_df)} rows")
        return result_df
        
    except Exception as e:
        logger.error(f"[{table_code}] Error processing {csv_file}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_census_table(
    table_code: str,
    process_file_function: callable,
    output_filename: str,
    zip_dir: Path,
    temp_extract_base: Path,
    output_dir: Path,
    geo_output_path: Path,
    time_sk: Optional[int] = None
) -> bool:
    """Process a Census table by finding, extracting, and processing relevant files, then linking with dimensions.
    
    This function serves as the main workflow controller for processing any Census table. It handles:
    1. Finding ZIP files containing the specified Census table data
    2. Extracting relevant CSV files from each ZIP file
    3. Processing each CSV file using the provided processing function
    4. Combining results into a single standardized DataFrame
    5. Joining with geographic dimension to add geo_sk surrogate keys
    6. Adding time dimension surrogate key if provided
    7. Adding ETL processing timestamp
    8. Writing the final fact table to Parquet format
    
    Args:
        table_code (str): Census table code (e.g., "G17", "G18") used to identify the
                         specific Census table pattern in ZIP files.
        process_file_function (callable): Table-specific function that processes individual
                                        CSV files and returns a standardized DataFrame.
        output_filename (str): Name of the output Parquet file to be saved in output_dir.
        zip_dir (Path): Directory containing Census ZIP files to search.
        temp_extract_base (Path): Base directory for temporary file extraction.
        output_dir (Path): Directory where the final Parquet output will be saved.
        geo_output_path (Path): Path to the geographic dimension Parquet file for joining.
        time_sk (Optional[int]): Time dimension surrogate key to associate with all 
                                records in the fact table. If None, no time_sk column is added.
        
    Returns:
        bool: True if the entire process completed successfully and data was written to
             the output file, False if any critical error occurred during processing.
    
    Note:
        This function performs several data quality checks throughout the process:
        - Validates that required ZIP files are found
        - Ensures extracted CSV files contain expected data
        - Checks for null/missing geographic codes
        - Validates the final combined dataset has a reasonable number of rows
    """
    logger.info(f"=== Starting Census {table_code} Data Processing ===")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find ZIP files containing table data using pattern from config
    table_pattern = config.CENSUS_TABLE_PATTERNS.get(table_code, fr"2021.*Census.*{table_code}.*?(SA1|SA2)\.csv$")
    logger.info(f"[{table_code}] Searching for Census files with pattern: {table_pattern}")
    table_zips = find_census_files(zip_dir, table_pattern)
    
    if not table_zips:
        logger.error(f"[{table_code}] No ZIP files found containing {table_code} data")
        return False
    logger.info(f"[{table_code}] Found {len(table_zips)} ZIP files containing potential {table_code} data")
    
    # Create temporary extraction directory
    temp_extract_dir = temp_extract_base / f"census_{table_code}"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract all matching files from each ZIP
    total_extracted = 0
    for zip_file in table_zips:
        logger.info(f"[{table_code}] Extracting ZIP file: {zip_file.name}")
        extracted_files = extract_census_files(zip_file, table_pattern, temp_extract_dir)
        if extracted_files:
            logger.info(f"[{table_code}] Extracted {len(extracted_files)} CSV files from {zip_file.name}")
            total_extracted += len(extracted_files)
    
    if total_extracted == 0:
        logger.error(f"[{table_code}] No CSV files were extracted from any ZIP files")
        return False
    
    # Find all extracted CSV files
    csv_files = find_census_csv_files(temp_extract_dir, table_code)
    if not csv_files:
        logger.error(f"[{table_code}] No CSV files found in extracted files")
        return False
    logger.info(f"[{table_code}] Found {len(csv_files)} CSV files")
    
    # Process each CSV file
    processed_dfs = []
    success_count = 0
    
    for csv_file in csv_files:
        df = process_file_function(csv_file)
        if df is not None and len(df) > 0:
            processed_dfs.append(df)
            success_count += 1
    
    if not processed_dfs:
        logger.error(f"[{table_code}] No CSV files were successfully processed")
        return False
    logger.info(f"[{table_code}] Successfully processed {success_count} of {len(csv_files)} CSV files")
    
    # Combine all processed DataFrames
    try:
        combined_df = pl.concat(processed_dfs)
        logger.info(f"[{table_code}] Combined data has {len(combined_df)} rows")
        
        # Data Quality Check: Ensure we have a reasonable amount of data
        if len(combined_df) < 100:  # Arbitrary threshold, adjust based on expectations
            logger.warning(
                f"[{table_code}] Data quality check: Combined data has only {len(combined_df)} rows, "
                f"which is fewer than expected (< 100)"
            )
        else:
            logger.info(f"[{table_code}] Successfully combined data into {len(combined_df)} rows")
        
        # Filter out special geo codes before joining
        original_count = len(combined_df)
        combined_df = utils.filter_special_geo_codes(combined_df, "geo_code")
        if original_count > len(combined_df):
            logger.info(f"[{table_code}] Filtered out {original_count - len(combined_df)} rows with special geo codes")
        
        # --- Surrogate Key & Geo-code Verification Logs ---
        # Log unique and duplicate geo_codes in fact table before join
        n_fact_geo = combined_df['geo_code'].n_unique()
        n_fact_geo_dupes = (combined_df['geo_code'].value_counts().filter(pl.col('counts') > 1).height)
        logger.info(f"[{table_code}] Fact table: {n_fact_geo} unique geo_codes, {n_fact_geo_dupes} duplicate geo_codes before join.")

        # Log unique and duplicate geo_codes in dimension table before join
        n_dim_geo = geo_df['geo_code'].n_unique()
        n_dim_geo_dupes = (geo_df['geo_code'].value_counts().filter(pl.col('counts') > 1).height)
        logger.info(f"[{table_code}] Dimension table: {n_dim_geo} unique geo_codes, {n_dim_geo_dupes} duplicate geo_codes before join.")

        # Log unmatched geo_codes (fact not in dimension)
        unmatched_geo = set(combined_df['geo_code'].unique()) - set(geo_df['geo_code'].unique())
        logger.info(f"[{table_code}] {len(unmatched_geo)} geo_codes in fact table not present in dimension table.")

        # Log row count before join
        logger.info(f"[{table_code}] Row count before join: {len(combined_df)}")

        # Join with geographic dimension to add surrogate key - using left join to preserve all fact records
        logger.info(f"[{table_code}] Joining with geographic dimension using left join...")
        geo_df = pl.read_parquet(geo_output_path)
        combined_df = combined_df.join(
            geo_df.select(['geo_code', 'geo_sk']),
            on='geo_code',
            how='left'
        )

        # Log detailed join statistics
        total_rows = len(combined_df)
        matched_rows = combined_df.filter(pl.col('geo_sk').is_not_null()).height
        unmatched_rows = total_rows - matched_rows
        unmatched_percent = (unmatched_rows / total_rows * 100) if total_rows > 0 else 0
        
        logger.info(f"[{table_code}] Join results - Total rows: {total_rows}, Matched: {matched_rows}, "
                   f"Unmatched: {unmatched_rows} ({unmatched_percent:.2f}%)")
        
        if unmatched_rows > 0:
            logger.warning(f"[{table_code}] Found {unmatched_rows} rows with unmatched geo_codes: "
                         f"{combined_df.filter(pl.col('geo_sk').is_null()).select('geo_code').unique().to_series().to_list()}")
            
            # Handle unmatched records by setting geo_sk to special value
            combined_df = combined_df.with_columns(
                pl.when(pl.col('geo_sk').is_null())
                .then(-1)
                .otherwise(pl.col('geo_sk'))
                .alias('geo_sk')
            )
            logger.info(f"[{table_code}] Set geo_sk to -1 for unmatched records")

        # Log uniqueness of geo_sk in output
        n_geo_sk = combined_df['geo_sk'].n_unique()
        n_geo_sk_dupes = (combined_df['geo_sk'].value_counts().filter(pl.col('counts') > 1).height)
        logger.info(f"[{table_code}] Output contains {n_geo_sk} unique geo_sk values ({n_geo_sk_dupes} duplicates)")

        # --- End Verification Logs ---

        # Add time dimension surrogate key if provided
        if time_sk is not None:
            combined_df = combined_df.with_columns(pl.lit(time_sk).alias('time_sk'))
        
        # Add ETL processing timestamp
        logger.info(f"[{table_code}] Adding ETL processing timestamp")
        combined_df = combined_df.with_columns(pl.lit(datetime.now()).alias('etl_processed_at'))
        
        # Write to Parquet
        output_path = output_dir / output_filename
        combined_df.write_parquet(output_path)
        logger.info(f"[{table_code}] Successfully wrote {len(combined_df)} rows to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"[{table_code}] CRITICAL: Error combining {table_code} data: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def validate_processed_df(df: pl.DataFrame, table_code: str, source_file: Path) -> bool:
    """Validate the structure and content of a processed DataFrame.
    
    Args:
        df (pl.DataFrame): The DataFrame to validate.
        table_code (str): Census table code for logging.
        source_file (Path): Original source file for logging.
        
    Returns:
        bool: True if validation passes, False otherwise.
    """
    logger.debug(f"[{table_code}] Validating DataFrame from {source_file.name}: {df.shape} rows x columns")
    
    is_valid = True
    
    # Check for required columns (assuming at least geo_code is always needed)
    if 'geo_code' not in df.columns:
        logger.error(f"[{table_code}] Missing 'geo_code' column in processed data from {source_file.name}")
        is_valid = False
        
    # Check for excessive nulls in geo_code
    null_geo_count = df.filter(pl.col('geo_code').is_null()).height
    if null_geo_count > 0:
        logger.warning(f"[{table_code}] Found {null_geo_count} rows with null 'geo_code' in {source_file.name}")
        # Depending on requirements, this might be an error: is_valid = False
        
    # Check if any numeric columns are entirely null (suggests mapping/parsing issue)
    numeric_cols = df.select(pl.col(pl.NUMERIC_DTYPES)).columns
    for col in numeric_cols:
        if col != 'geo_code': # Exclude geo_code if it happens to be numeric
            null_count = df.filter(pl.col(col).is_null()).height
            if null_count == df.height:
                logger.warning(f"[{table_code}] Column '{col}' is entirely null in {source_file.name}")
                # Depending on requirements, this might be an error: is_valid = False
                
    # Add more specific validation checks as needed
    
    if is_valid:
        logger.debug(f"[{table_code}] Validation passed for {source_file.name}")
    else:
        logger.warning(f"[{table_code}] Validation issues detected for {source_file.name}")
        
    return is_valid

def find_census_csv_files(extract_dir: Path, table_code: str) -> List[Path]:
    """Find all census CSV files in the extract directory.
    
    Args:
        extract_dir (Path): Directory where files were extracted
        table_code (str): Census table code (e.g., "G17", "G18")
        
    Returns:
        List[Path]: List of paths to CSV files
    """
    # Recursively find all CSV files
    csv_files = list(extract_dir.glob("**/*.csv"))
    logger.info(f"[{table_code}] Found {len(csv_files)} CSV files in extraction directory")
    return csv_files