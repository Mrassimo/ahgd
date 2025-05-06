"""Utility functions for the AHGD ETL pipeline.

This module contains helper functions for logging, downloading, extracting,
and data cleaning operations used throughout the ETL process.
"""

import logging
import zipfile
import requests
import tempfile
import shutil
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import pandas as pd
import geopandas as gpd
import polars as pl
import pyarrow
import pyarrow.parquet as pq
from shapely.geometry import mapping
from shapely.validation import make_valid
from tqdm.notebook import tqdm

import functools


# Custom exceptions
def log_function_timing(func):
    """Decorator that logs the execution time of the decorated function.
    
    Args:
        func: The function to be decorated.
        
    Returns:
        The wrapped function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        logger = logging.getLogger('ahgd_etl')
        logger.info(f"Function '{func.__name__}' executed in {duration:.4f} seconds")
        return result
    
    return wrapper


class DownloadError(Exception):
    """Raised when a file download fails."""
    pass

class ExtractionError(Exception):
    """Raised when zip file extraction fails."""
    pass

def setup_logging(log_directory: Path = None) -> logging.Logger:
    """Set up logging configuration.

    Args:
        log_directory (Path, optional): Directory for log files. Defaults to None.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger('ahgd_etl')
    # Avoid adding handlers multiple times if called repeatedly
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    # Create formatter - using a single consistent format for both handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if log_directory provided)
    if log_directory:
        try:
            log_directory.mkdir(parents=True, exist_ok=True)
            
            # Create standard log file (for backward compatibility)
            standard_log_file = log_directory / 'ahgd_etl.log'
            file_handler = logging.FileHandler(standard_log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Create timestamped log file
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            timestamped_log_file = log_directory / f'ahgd_etl_{timestamp}.log'
            timestamped_handler = logging.FileHandler(timestamped_log_file)
            timestamped_handler.setFormatter(formatter)
            logger.addHandler(timestamped_handler)
            
            logger.info(f"Logging to timestamped file: {timestamped_log_file}")
        except Exception as e:
            # Log error about file handler creation but don't crash
            logger.error(f"Could not create file handler at {log_directory}: {e}")

    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    return logger

def download_file(url: str, dest_file: Path, desc: str = None, max_retries: int = 3) -> bool:
    """Download a file from a URL with progress bar and retries.

    Args:
        url (str): URL to download from.
        dest_file (Path): Destination file path.
        desc (str, optional): Description for progress bar. Defaults to None.
        max_retries (int, optional): Maximum number of retry attempts. Defaults to 3.

    Returns:
        bool: True if download successful, False otherwise.
    """
    logger = logging.getLogger('ahgd_etl') # Ensure logger is available
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60) # Added timeout
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192

            # Ensure tqdm works outside notebooks too
            try:
                from tqdm import tqdm as tqdm_cli
                tqdm_instance = tqdm_cli
            except ImportError:
                tqdm_instance = tqdm # Fallback to notebook version

            with tqdm_instance(total=total_size, unit='B', unit_scale=True, desc=desc or Path(url).name) as pbar:
                with open(dest_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify file size if possible
            if total_size != 0 and dest_file.stat().st_size != total_size:
                logger.warning(
                    f"Downloaded file size mismatch for {dest_file.name}. "
                    f"Expected {total_size}, got {dest_file.stat().st_size}"
                )
                # Decide if this should be a failure or just a warning
                # return False # Uncomment to make size mismatch a failure

            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Download attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)  # Exponential backoff
                continue
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts.")
                return False
        except Exception as e: # Catch other potential errors (e.g., file write)
            logger.error(f"An unexpected error occurred during download of {url}: {str(e)}", exc_info=True)
            return False # Treat unexpected errors as failure

    return False # Should not be reached if loop completes, but added for safety

def extract_zipfile(zip_file: Path, extract_dir: Path, desc: str = None) -> bool:
    """Extract a ZIP file with progress indication.

    Args:
        zip_file (Path): Path to ZIP file.
        extract_dir (Path): Directory to extract to.
        desc (str, optional): Description for progress bar. Defaults to None.

    Returns:
        bool: True if extraction successful, False otherwise.
    """
    logger = logging.getLogger('ahgd_etl') # Ensure logger is available
    try:
        # Ensure tqdm works outside notebooks too
        try:
            from tqdm import tqdm as tqdm_cli
            tqdm_instance = tqdm_cli
        except ImportError:
            tqdm_instance = tqdm # Fallback to notebook version

        with zipfile.ZipFile(zip_file) as zf:
            # Create extract_dir only if zip opens successfully
            extract_dir.mkdir(parents=True, exist_ok=True)
            file_list = zf.infolist()
            for member in tqdm_instance(file_list, desc=desc or f"Extracting {zip_file.name}"):
                try:
                    # Basic protection against zip slip - refuse absolute paths or paths navigating up
                    if member.is_dir() or member.filename.startswith('/') or '..' in member.filename:
                        continue
                    zf.extract(member, extract_dir)
                except Exception as e:
                    logger.error(f"Failed to extract {member.filename} from {zip_file.name}: {str(e)}")
                    # Decide if one failed extraction fails the whole process
                    # return False # Uncomment to fail on first extraction error
            logger.info(f"Successfully extracted {len(file_list)} items from {zip_file.name}")
        return True
    except zipfile.BadZipFile:
        logger.error(f"File is not a valid zip file: {zip_file}")
        return False
    except Exception as e:
        logger.error(f"Failed to open/process zip file {zip_file}: {str(e)}", exc_info=True)
        return False

def extract_files_from_zip(zip_file: Path, extract_dir: Path, pattern: str = None) -> bool:
    """Extract specific files from a ZIP file based on a regex pattern.
    
    Args:
        zip_file (Path): Path to ZIP file.
        extract_dir (Path): Directory to extract to.
        pattern (str, optional): Regex pattern to match filenames. Defaults to None (all files).
        
    Returns:
        bool: True if extraction successful, False otherwise.
    """
    logger = logging.getLogger('ahgd_etl')
    try:
        # Compile regex pattern if provided
        regex = re.compile(pattern) if pattern else None
        
        # Ensure extract directory exists
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_count = 0
        skipped_count = 0
        
        with zipfile.ZipFile(zip_file) as zf:
            # Get list of all files in the ZIP
            file_list = zf.infolist()
            logger.info(f"Found {len(file_list)} files in {zip_file}")
            
            # Try to use tqdm for progress indication
            try:
                from tqdm import tqdm as tqdm_cli
                tqdm_instance = tqdm_cli
            except ImportError:
                tqdm_instance = tqdm  # Fallback to notebook version
                
            # Extract only matching files
            for member in tqdm_instance(file_list, desc=f"Extracting from {zip_file.name}"):
                # Skip directories and files with problematic paths
                if member.is_dir() or member.filename.startswith('/') or '..' in member.filename:
                    skipped_count += 1
                    continue
                
                # Check if file matches pattern
                if regex and not regex.search(member.filename):
                    skipped_count += 1
                    continue
                
                try:
                    # Extract the file
                    zf.extract(member, extract_dir)
                    extracted_count += 1
                except Exception as e:
                    logger.error(f"Failed to extract {member.filename}: {str(e)}")
            
        logger.info(f"Extracted {extracted_count} files from {zip_file} (skipped {skipped_count})")
        return True
        
    except zipfile.BadZipFile:
        logger.error(f"File is not a valid ZIP file: {zip_file}")
        return False
    except Exception as e:
        logger.error(f"Error extracting files from {zip_file}: {str(e)}")
        return False

def download_data(urls_dict: Dict[str, str], download_dir: Path, force_download: bool = False) -> bool:
    """Download multiple files from URLs.

    Args:
        urls_dict (Dict[str, str]): Dictionary mapping filenames to URLs.
        download_dir (Path): Directory to download files to.
        force_download (bool, optional): Force redownload if file exists. Defaults to False.

    Returns:
        bool: True if all downloads successful, False otherwise.
    """
    logger = logging.getLogger('ahgd_etl') # Ensure logger is available
    download_dir.mkdir(parents=True, exist_ok=True)
    overall_success = True

    if not urls_dict:
        logger.warning("No URLs provided for download.")
        return True # Nothing to download is considered success

    for filename, url in urls_dict.items():
        dest_file = download_dir / filename
        if dest_file.exists() and not force_download:
            logger.info(f"File {filename} already exists, skipping download.")
            continue

        logger.info(f"Attempting to download {filename} from {url}")
        if not download_file(url, dest_file, desc=f"Downloading {filename}"):
            logger.error(f"Failed to download {filename}.")
            overall_success = False
            # Decide if one failure should stop all downloads
            # return False # Uncomment to stop on first download failure

    return overall_success

def find_geo_column(df: Union[gpd.GeoDataFrame, pd.DataFrame, pl.DataFrame, pl.LazyFrame, Dict[str, Any]],
                    possible_names: List[str]) -> Optional[str]:
    """Find geographic code column in a dataframe.

    Args:
        df: Input dataframe (supports multiple types).
        possible_names (List[str]): List of possible column names.

    Returns:
        Optional[str]: Found column name or None.
    """
    if isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
        cols = df.columns
    elif isinstance(df, pl.DataFrame):
        cols = df.columns
    elif isinstance(df, pl.LazyFrame):
        # Need to access columns differently for LazyFrame
        cols = df.columns # This works in recent Polars versions
    elif isinstance(df, dict):
        cols = df.keys()
    else:
        return None

    # Extend possible names based on common ABS patterns for different geographic levels
    extended_names = list(possible_names)  # Start with the original list
    
    # Check if we're looking for STATE level codes and add common variations
    state_level_patterns = [
        name for name in possible_names 
        if 'state' in name.lower() or 'ste' in name.lower()
    ]
    
    if state_level_patterns:
        # These are common variations of STATE/STE code column names in ABS data
        state_variations = [
            'STATE_CODE21', 'STATE_CODE_2021', 'STE_CODE21', 'STE_CODE_2021',
            'STATE_CODE', 'STE_CODE', 'STATE_CODE_2016', 'STE_CODE16',
            'STE21', 'STATE21', 'STE_2021', 'STATE_2021',
            'STE_ID', 'STATE_ID', 'STE', 'STATE'
        ]
        extended_names.extend(state_variations)
    
    # Case-insensitive comparison might be safer
    cols_lower = {col.lower(): col for col in cols}
    for name in extended_names:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()] # Return original case name
    return None

def clean_geo_code(code_val: Any) -> Optional[str]:
    """Clean and validate geographic code values.

    Args:
        code_val: Input code value.

    Returns:
        Optional[str]: Cleaned code as string or None if invalid.
    """
    if pd.isna(code_val):
        return None

    try:
        # Convert to string and strip whitespace
        code_str = str(code_val).strip()
        
        # Special handling for STATE codes which can be single digits (1-9)
        # STATE codes are typically 1 (NSW), 2 (VIC), etc.
        if len(code_str) == 1 and code_str.isdigit():
            return code_str
            
        # Handle possible 'AUS' code for Australia
        if code_str.upper() == 'AUS':
            return 'AUS'
            
        # For other geographic levels (SA1, SA2, etc.), expect all digits
        if not code_str.isdigit():
            return None
            
        # Handle SA1 codes which should be 11 digits
        if len(code_str) == 11 and code_str.isdigit():
            return code_str
            
        # Handle SA2 codes which should be 9 digits
        if len(code_str) == 9 and code_str.isdigit():
            return code_str
            
        # Handle SA3 codes which should be 5 digits
        if len(code_str) == 5 and code_str.isdigit():
            return code_str
            
        # Handle SA4 codes which should be 3 digits
        if len(code_str) == 3 and code_str.isdigit():
            return code_str
            
        # If we get here, it's a numeric code but not one of the expected formats
        # Return it anyway, but log a warning
        logger = logging.getLogger('ahgd_etl')
        logger.warning(f"Unusual geographic code format: {code_str}")
        return code_str

    except (ValueError, TypeError):
        # Catch potential errors during string conversion etc.
        return None

def safe_float(value: Any) -> Optional[float]:
    """Safely convert value to float.

    Args:
        value: Input value.

    Returns:
        Optional[float]: Converted float or None if invalid.
    """
    if pd.isna(value):
        return None
    try:
        # Handle potential strings with commas
        value_str = str(value).strip().replace(',', '')
        return float(value_str)
    except (ValueError, TypeError):
        return None

def geometry_to_wkt(geometry: Any) -> Optional[str]:
    """Convert geometry to WKT string with validation.

    Args:
        geometry: Input geometry.

    Returns:
        Optional[str]: WKT string or None if invalid.
    """
    logger = logging.getLogger('ahgd_etl') # Ensure logger is available
    if pd.isna(geometry) or geometry is None:
        return None
    try:
        # Ensure input is a shapely geometry object if possible
        if not hasattr(geometry, 'geom_type'):
            # Attempt conversion if it looks like geo-interface
            if hasattr(geometry, '__geo_interface__'):
                from shapely.geometry import shape
                geometry = shape(geometry.__geo_interface__)
            else:
                # Cannot determine geometry type
                logger.warning(f"Cannot convert unrecognised type to geometry: {type(geometry)}")
                return None

        if not geometry.is_valid:
            logger.debug(f"Attempting to fix invalid geometry: {geometry.wkt[:100]}...") # Log snippet
            valid_geom = make_valid(geometry)
            if not valid_geom.is_valid:
                logger.warning(f"Failed to fix invalid geometry: {geometry.wkt[:100]}...")
                return None # Return None if fixing fails
            logger.debug("Geometry fixed successfully.")
            return valid_geom.wkt
        else:
            return geometry.wkt

    except ImportError:
        logger.error("Shapely is required for geometry_to_wkt function.")
        return None
    except Exception as e:
        # Log error but try to return None gracefully
        logger.error(f"Error converting geometry to WKT: {str(e)}", exc_info=True)
        return None

def clean_polars_geo_code(series_expr: pl.Expr) -> pl.Expr:
    """Clean geographic codes in a Polars expression (ensure integer-like strings).

    Args:
        series_expr (pl.Expr): Input Polars expression.

    Returns:
        pl.Expr: Cleaned expression (string type, digits only or None).
    """
    return (
        series_expr
        .cast(pl.String) # Ensure it's string first
        .str.strip_chars()
        # Remove all non-digit characters
        .str.replace_all(r"[^0-9]", "")
        # Set to null if the result is an empty string, otherwise keep the digits
        .pipe(lambda s: pl.when(s != "").then(s).otherwise(pl.lit(None, dtype=pl.String)))
    )


def safe_polars_int(series_expr: pl.Expr) -> pl.Expr:
    """Safely convert Polars expression to Int64, handling common non-numeric chars.

    Args:
        series_expr (pl.Expr): Input Polars expression.

    Returns:
        pl.Expr: Int64 expression, with None for unparseable values.
    """
    logger = logging.getLogger('ahgd_etl') # Ensure logger is available

    # Ensure input is string, replacing errors with null
    str_expr = series_expr.cast(pl.String, strict=False)

    # Clean the string: remove $, comma; handle parentheses for negatives
    # Add handling for potential decimals before casting to Int
    cleaned_expr = (
        str_expr
        .str.strip_chars() # Remove leading/trailing whitespace
        .str.replace_all(r"[$,]", "") # Remove $ and ,
        .str.replace(r"^\(\s*(.*?)\s*\)$", "-$1") # Handle (num) -> -num
        # Attempt to cast to Float first to handle decimals, then to Int
        .cast(pl.Float64, strict=False)
        .round(0) # Round the float value to nearest integer
        .cast(pl.Int64, strict=False) # Cast to Int64, errors become null
    )

    # Keep original name in alias
    return cleaned_expr.alias(series_expr.meta.output_name() if hasattr(series_expr, 'meta') else 'unknown')

def filter_special_geo_codes(df: pl.DataFrame, geo_code_col: str) -> pl.DataFrame:
    """
    Filter special geo codes that aren't actual geographic areas.
    
    Args:
        df: The DataFrame containing geographic codes
        geo_code_col: The name of the column containing the geographic codes
        
    Returns:
        DataFrame with special geocodes filtered out
    """
    logger = logging.getLogger('ahgd_etl')
    
    # Define patterns for special geo codes
    special_patterns = [
        # Remove codes with length over 11 characters
        pl.col(geo_code_col).str.len_chars() > 11,
        
        # Remove codes with only repeated digits (e.g., 11111, 22222, etc.)
        pl.col(geo_code_col).str.contains(r'1{5,}'),
        pl.col(geo_code_col).str.contains(r'2{5,}'),
        pl.col(geo_code_col).str.contains(r'3{5,}'),
        pl.col(geo_code_col).str.contains(r'4{5,}'),
        pl.col(geo_code_col).str.contains(r'5{5,}'),
        pl.col(geo_code_col).str.contains(r'6{5,}'),
        pl.col(geo_code_col).str.contains(r'7{5,}'),
        pl.col(geo_code_col).str.contains(r'8{5,}'),
        pl.col(geo_code_col).str.contains(r'9{5,}'),
        pl.col(geo_code_col).str.contains(r'0{5,}'),
        
        # Remove codes that end with 9's (ABS uses these for "not elsewhere classified")
        pl.col(geo_code_col).str.contains(r'9{4,}$'),
        
        # Remove codes that include specific pattern (e.g., 797979)
        pl.col(geo_code_col).str.contains(r'797979')
    ]
    
    # Combine all patterns with OR logic
    filter_expr = special_patterns[0]
    for pattern in special_patterns[1:]:
        filter_expr = filter_expr | pattern
    
    # Log number of rows filtered out
    original_count = df.shape[0]
    filtered_df = df.filter(~filter_expr)
    filtered_count = filtered_df.shape[0]
    logger.info(f"Filtered out {original_count - filtered_count} rows with special geo codes")
    
    return filtered_df

def extract_census_files(zip_file: Path, extract_dir: Path, file_pattern: str) -> int:
    """Extract Census CSV files from a ZIP file that match the given pattern.
    
    Args:
        zip_file (Path): Path to the ZIP file to extract from
        extract_dir (Path): Directory to extract files into
        file_pattern (str): Regex pattern to match filenames against
        
    Returns:
        int: Number of files extracted
        
    Raises:
        Exception: If there are any errors opening or extracting from the ZIP file
    """
    try:
        pattern = re.compile(file_pattern)
        extracted_count = 0
        
        with zipfile.ZipFile(zip_file, 'r') as zf:
            # Find all matching files in the ZIP
            for info in zf.infolist():
                if pattern.search(info.filename):
                    # Extract the file
                    zf.extract(info, extract_dir)
                    extracted_count += 1
                    
        return extracted_count
        
    except Exception as e:
        logger.error(f"Error opening {zip_file}: {str(e)}")
        raise

def process_census_table(table_data: Union[pd.DataFrame, pl.DataFrame], table_name: str) -> Optional[bool]:
    """Process a Census table data into a standardized format.
    
    Args:
        table_data (Union[pd.DataFrame, pl.DataFrame]): The raw Census table data.
        table_name (str): The name of the table being processed.
    
    Returns:
        Optional[bool]: True if processing was successful, None or False otherwise.
    """
    logger = logging.getLogger('ahgd_etl')
    try:
        logger.info(f"Processing Census table: {table_name}")
        # Convert to Polars if Pandas DataFrame
        if isinstance(table_data, pd.DataFrame):
            table_data = pl.from_pandas(table_data)
        
        # Basic processing logic placeholder
        # Add specific cleaning, transformation, or validation steps here as needed
        logger.info(f"Successfully processed {table_name} with {len(table_data)} rows")
        return True
    except Exception as e:
        logger.error(f"Error processing {table_name}: {str(e)}")
        return None