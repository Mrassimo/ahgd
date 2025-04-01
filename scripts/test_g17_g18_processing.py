#!/usr/bin/env python

import logging
import polars as pl
from pathlib import Path
import sys

# Add the root directory to the path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from etl_logic import census, utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_g17_processing():
    """Test the processing of G17 files (Need for Assistance)."""
    logger.info("Testing G17 (Need for Assistance) processing...")
    
    # Find G17 files
    data_dir = Path("data/raw/Census")
    g17_files = list(data_dir.glob("**/G17*"))
    
    if not g17_files:
        logger.error("No G17 files found for testing")
        return
    
    logger.info(f"Found {len(g17_files)} G17 files")
    
    # Process each file
    for file_path in g17_files:
        logger.info(f"Processing {file_path.name}...")
        
        # Before processing, list the columns in the file
        try:
            df_preview = pl.read_csv(file_path, n_rows=1)
            logger.info(f"Columns in file: {df_preview.columns}")
        except Exception as e:
            logger.error(f"Error reading preview: {str(e)}")
            continue
        
        # Try processing the file
        result_df = census.process_g17_file(file_path)
        
        if result_df is not None:
            logger.info(f"Successfully processed {file_path.name}")
            logger.info(f"Resulting DataFrame: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
            logger.info(f"First 5 rows: {result_df.head(5)}")
        else:
            logger.error(f"Failed to process {file_path.name}")

def test_g18_processing():
    """Test the processing of G18 files (Unpaid Care)."""
    logger.info("Testing G18 (Unpaid Care) processing...")
    
    # Find G18 files
    data_dir = Path("data/raw/Census")
    g18_files = list(data_dir.glob("**/G18*"))
    
    if not g18_files:
        logger.error("No G18 files found for testing")
        return
    
    logger.info(f"Found {len(g18_files)} G18 files")
    
    # Process each file
    for file_path in g18_files:
        logger.info(f"Processing {file_path.name}...")
        
        # Before processing, list the columns in the file
        try:
            df_preview = pl.read_csv(file_path, n_rows=1)
            logger.info(f"Columns in file: {df_preview.columns}")
        except Exception as e:
            logger.error(f"Error reading preview: {str(e)}")
            continue
        
        # Try processing the file
        result_df = census.process_g18_file(file_path)
        
        if result_df is not None:
            logger.info(f"Successfully processed {file_path.name}")
            logger.info(f"Resulting DataFrame: {result_df.shape[0]} rows, {result_df.shape[1]} columns")
            logger.info(f"First 5 rows: {result_df.head(5)}")
        else:
            logger.error(f"Failed to process {file_path.name}")

if __name__ == "__main__":
    logger.info("Starting G17 and G18 processing test...")
    test_g17_processing()
    test_g18_processing()
    logger.info("Tests completed.") 