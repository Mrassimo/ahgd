"""
Utility functions for the AHGD ETL pipeline.

This module contains helper functions for downloading, extracting, logging,
and processing data used throughout the ETL process.
"""

import logging
import os
import zipfile
import requests
import tempfile
import shutil
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable

import polars as pl
import pandas as pd

from .config import settings

def setup_logging(log_dir: Optional[Path] = None) -> logging.Logger:
    """
    Set up logging for the ETL pipeline.
    
    Args:
        log_dir: Directory to write log files to
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('ahgd_etl')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if log_dir is provided
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        log_file = log_dir / f"ahgd_etl_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    return logger

def download_data(url: str, target_path: Path, force_download: bool = False, max_retries: int = 3) -> bool:
    """
    Download data from a URL to a target path.

    Args:
        url: URL to download from
        target_path: Path to save downloaded file to
        force_download: Whether to download even if the file exists
        max_retries: Maximum number of retry attempts for transient errors

    Returns:
        True if download succeeded, False otherwise
    """
    logger = logging.getLogger('ahgd_etl.utils')

    # Check if file exists and force_download is False
    if target_path.exists() and not force_download:
        logger.info(f"File already exists: {target_path}")
        return True

    # Create parent directories if they don't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Try downloading with retries
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading {url} to {target_path} (attempt {attempt}/{max_retries})")

            # Make request with progress tracking
            with requests.get(url, stream=True, timeout=30) as response:
                response.raise_for_status()  # Will raise for 4xx/5xx responses

                # Get total file size
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0

                # Download in chunks
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress periodically
                            if total_size > 0 and downloaded % (1024 * 1024) == 0:  # Log every 1 MB
                                percent = (downloaded / total_size) * 100
                                logger.debug(f"Download progress: {percent:.1f}% ({downloaded}/{total_size} bytes)")

                logger.info(f"Download complete: {target_path}")
                return True

        except requests.exceptions.HTTPError as e:
            # For HTTP errors like 404, retrying won't help
            logger.error(f"HTTP error downloading {url}: {e}")

            # Clean up partial download
            if target_path.exists():
                target_path.unlink()

            # Don't retry for 4xx client errors, but do retry for 5xx server errors
            if 400 <= e.response.status_code < 500:
                return False

        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            # These are potentially transient errors worth retrying
            logger.warning(f"Error downloading {url} (attempt {attempt}/{max_retries}): {e}")

            # Clean up partial download
            if target_path.exists():
                target_path.unlink()

            # Wait before retrying (exponential backoff)
            if attempt < max_retries:
                wait_time = 2 ** attempt  # 2, 4, 8 seconds
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)

        except Exception as e:
            # Unexpected errors
            logger.error(f"Unexpected error downloading {url}: {e}")

            # Clean up partial download
            if target_path.exists():
                target_path.unlink()

            return False

    # If we get here, all retries have failed
    logger.error(f"All download attempts failed for {url}")
    return False

def extract_zipfile(zip_path: Path, extract_dir: Path, desc: str = "") -> bool:
    """
    Extract a ZIP file to a directory.
    
    Args:
        zip_path: Path to ZIP file
        extract_dir: Directory to extract to
        desc: Optional description for logging
        
    Returns:
        True if extraction succeeded, False otherwise
    """
    logger = logging.getLogger('ahgd_etl.utils')
    
    # Check if file exists
    if not zip_path.exists():
        logger.error(f"ZIP file does not exist: {zip_path}")
        return False
    
    # Create extract directory if it doesn't exist
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if desc:
            logger.info(f"{desc}: {zip_path} to {extract_dir}")
        else:
            logger.info(f"Extracting {zip_path} to {extract_dir}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Create a temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract all files to temp directory
                zip_ref.extractall(temp_dir)
                
                # Move files to final location
                for item in Path(temp_dir).iterdir():
                    target = extract_dir / item.name
                    if target.exists():
                        if target.is_dir():
                            shutil.rmtree(target)
                        else:
                            target.unlink()
                    shutil.move(str(item), str(extract_dir))
        
        logger.info(f"Extraction complete: {zip_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False

def geometry_to_wkt(geometry) -> Optional[str]:
    """
    Convert a shapely geometry to WKT representation.
    
    Args:
        geometry: Shapely geometry object
        
    Returns:
        WKT string or None if conversion fails
    """
    if geometry is None:
        return None
        
    try:
        return geometry.wkt
    except Exception as e:
        logger = logging.getLogger('ahgd_etl.utils')
        logger.error(f"Error converting geometry to WKT: {e}")
        return None

def clean_geo_code(code) -> str:
    """
    Clean a geographic code.
    
    Args:
        code: Geographic code to clean
        
    Returns:
        Cleaned code as string
    """
    if code is None:
        return "UNKNOWN"
        
    # Convert to string
    code_str = str(code)
    
    # Remove any non-alphanumeric characters except underscores
    code_str = re.sub(r'[^\w]', '', code_str)
    
    return code_str

def process_census_table(table_code: str, process_file_function: Callable, output_filename: str,
                         zip_dir: Path, temp_extract_base: Path, output_dir: Path,
                         geo_output_path: Optional[Path] = None, time_sk: Optional[str] = None) -> bool:
    """
    Process a Census table by finding, extracting, and processing all relevant files.
    
    Args:
        table_code: Census table code (e.g., 'G01', 'G19')
        process_file_function: Function to process individual CSV files
        output_filename: Name of the output file
        zip_dir: Directory containing Census zip files
        temp_extract_base: Base directory for temporary file extraction
        output_dir: Directory to write output files
        geo_output_path: Path to the geographic dimension file (for lookups)
        time_sk: Time dimension surrogate key to use
        
    Returns:
        True if processing succeeded, False otherwise
    """
    logger = logging.getLogger('ahgd_etl.utils')
    
    # Get the Census table pattern
    census_table_patterns = settings.get_column_mapping('census_table_patterns')
    if not census_table_patterns:
        # Fallback if not in YAML
        census_table_patterns = {
            "G01": r"2021\s*Census_G01[_\s].*?(SA1|SA2)\.csv$",
            "G17": r"2021\s*Census_G17[A-C]_.*?(SA1|SA2)\.csv$",
            "G18": r"2021\s*Census_G18[_\s].*?(SA1|SA2)\.csv$", 
            "G19": r"2021\s*Census_G19[A-C]_.*?(SA1|SA2)\.csv$",
            "G20": r"2021\s*Census_G20[A-B]_.*?(SA1|SA2)\.csv$",
            "G21": r"2021\s*Census_G21[A-C]_.*?(SA1|SA2)\.csv$",
            "G25": r"2021\s*Census_G25[_\s].*?(SA1|SA2)\.csv$"
        }
    
    # Define pattern for finding ZIP files - expanded to match GCP_ALL.zip and G*.zip
    census_zip_pattern = r".*(Census|GCP|G[0-9]+).*\.zip$"
    
    # Define extract directory for this table
    extract_dir = temp_extract_base / f"{table_code.lower()}_extract"
    
    # Define output path
    output_path = output_dir / output_filename
    
    try:
        # Create directories if they don't exist
        extract_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find and extract ZIP files
        file_extracted = False
        zip_files = list(zip_dir.glob("*.zip"))
        logger.info(f"Found {len(zip_files)} ZIP files in {zip_dir}: {[f.name for f in zip_files]}")

        matched_files = []
        for zip_file in zip_files:
            if re.match(census_zip_pattern, zip_file.name, re.IGNORECASE):
                matched_files.append(zip_file.name)
                logger.info(f"Extracting Census ZIP file: {zip_file}")
                extract_success = extract_zipfile(zip_file, extract_dir)
                if extract_success:
                    file_extracted = True

        if not matched_files:
            logger.error(f"No census ZIP files matched pattern '{census_zip_pattern}'")
            return False

        if not file_extracted:
            logger.error(f"Census ZIP files found ({matched_files}) but extraction failed")
            return False
        
        # Find CSV files matching the table pattern
        table_pattern = census_table_patterns.get(table_code)
        if not table_pattern:
            logger.error(f"No pattern defined for table {table_code}")
            return False
        
        # Process all matching CSV files
        all_results = []
        csv_pattern = re.compile(table_pattern, re.IGNORECASE)
        
        for csv_file in extract_dir.glob("**/*.csv"):
            if csv_pattern.search(csv_file.name):
                logger.info(f"Processing CSV file: {csv_file.name}")
                result_df = process_file_function(csv_file, geo_output_path, time_sk)
                
                if result_df is not None:
                    all_results.append(result_df)
                    logger.info(f"Successfully processed {csv_file.name}")
        
        if not all_results:
            logger.error(f"No {table_code} CSV files found or processing failed")
            return False
        
        # Combine all results
        combined_df = pl.concat(all_results)
        
        # Write combined results to parquet
        combined_df.write_parquet(output_path)
        logger.info(f"Wrote {len(combined_df)} rows to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing table {table_code}: {e}")
        logger.error("Traceback:", exc_info=True)
        return False