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

import pandas as pd
import geopandas as gpd
import polars as pl
import pyarrow
import pyarrow.parquet as pq
from shapely.geometry import mapping
from shapely.validation import make_valid
from tqdm.notebook import tqdm

# Custom exceptions
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
    logger.setLevel(logging.INFO)
    
    # Create formatter
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
        log_directory.mkdir(parents=True, exist_ok=True)
        log_file = log_directory / 'ahgd_colab_run.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
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
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                with open(dest_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            logging.error(f"Failed to download {url}: {str(e)}")
            return False
    return False

def extract_zipfile(zip_file: Path, extract_dir: Path, desc: str = None) -> bool:
    """Extract a ZIP file with progress indication.
    
    Args:
        zip_file (Path): Path to ZIP file.
        extract_dir (Path): Directory to extract to.
        desc (str, optional): Description for progress bar. Defaults to None.
        
    Returns:
        bool: True if extraction successful, False otherwise.
    """
    try:
        with zipfile.ZipFile(zip_file) as zf:
            for member in tqdm(zf.infolist(), desc=desc):
                try:
                    zf.extract(member, extract_dir)
                except Exception as e:
                    logging.error(f"Failed to extract {member.filename}: {str(e)}")
                    return False
        return True
    except Exception as e:
        logging.error(f"Failed to open/process zip file {zip_file}: {str(e)}")
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
    download_dir.mkdir(parents=True, exist_ok=True)
    success = True
    
    for filename, url in urls_dict.items():
        dest_file = download_dir / filename
        if dest_file.exists() and not force_download:
            logging.info(f"File {filename} already exists, skipping download.")
            continue
            
        logging.info(f"Downloading {filename} from {url}")
        if not download_file(url, dest_file, desc=f"Downloading {filename}"):
            success = False
            
    return success

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
        cols = df.columns
    elif isinstance(df, dict):
        cols = df.keys()
    else:
        return None
        
    for name in possible_names:
        if name in cols:
            return name
    return None

def clean_geo_code(code_val: Any) -> Optional[str]:
    """Clean and validate geographic code values.
    
    Args:
        code_val: Input code value.
        
    Returns:
        Optional[str]: Cleaned code or None if invalid.
    """
    if pd.isna(code_val):
        return None
    try:
        return str(int(float(str(code_val).strip())))
    except (ValueError, TypeError):
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
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None

def geometry_to_wkt(geometry: Any) -> Optional[str]:
    """Convert geometry to WKT string with validation.
    
    Args:
        geometry: Input geometry.
        
    Returns:
        Optional[str]: WKT string or None if invalid.
    """
    if pd.isna(geometry):
        return None
    try:
        if hasattr(geometry, 'wkt'):
            # Try to validate/repair the geometry
            valid_geom = make_valid(geometry)
            return valid_geom.wkt
        elif hasattr(geometry, '__geo_interface__'):
            # Convert to shapely and validate
            from shapely.geometry import shape
            geom = shape(geometry.__geo_interface__)
            valid_geom = make_valid(geom)
            return valid_geom.wkt
        return None  # Unrecognized geometry type
    except Exception as e:
        logging.error(f"Error converting geometry to WKT: {str(e)}")
        return None

def clean_polars_geo_code(series_expr: pl.Expr) -> pl.Expr:
    """Clean geographic codes in a Polars expression.
    
    Args:
        series_expr (pl.Expr): Input Polars expression.
        
    Returns:
        pl.Expr: Cleaned expression.
    """
    return (series_expr
            .str.strip()
            .str.replace(r'[^0-9A-Za-z]', '')  # Keep only alphanumeric
            .str.to_uppercase()  # For codes like 'AUS'
            .map_elements(lambda x: None if x == '' else x))  # Remove empty strings

def safe_polars_int(series_expr: pl.Expr) -> pl.Expr:
    """Safely convert Polars expression to integer.
    
    Args:
        series_expr (pl.Expr): Input Polars expression.
        
    Returns:
        pl.Expr: Integer expression.
    """
    return (series_expr
            .str.strip()
            .str.replace(r'[($),]', '')  # Remove currency and grouping symbols
            .str.replace(r'\((-?\d+)\)', '$1')  # Convert (123) to -123
            .cast(pl.Float64)
            .round(0)  # Round to nearest integer
            .cast(pl.Int64)) 