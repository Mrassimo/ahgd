"""Census data processing module for the AHGD ETL pipeline.

This module handles the processing of ABS Census data, specifically the G01
(Selected Person Characteristics) table, transforming it into a standardized
Parquet format and linking it with geographic boundaries.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import polars as pl
import pyarrow
import pyarrow.parquet as pq
import zipfile

from . import config
from . import utils

logger = logging.getLogger('ahgd_etl')

def find_census_files(zip_dir: Path, pattern: str) -> List[Path]:
    """Find Census CSV files in ZIP files matching a pattern.
    
    Args:
        zip_dir (Path): Directory containing ZIP files.
        pattern (str): Regex pattern to match filenames.
        
    Returns:
        List[Path]: List of matching ZIP files.
    """
    matching_files = []
    for zip_path in zip_dir.glob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if re.search(pattern, name, re.IGNORECASE):
                        matching_files.append(zip_path)
                        break
        except Exception as e:
            logger.error(f"Error scanning {zip_path}: {str(e)}")
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

def process_g01_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a G01 Census CSV file.
    
    Args:
        csv_file (Path): Path to CSV file.
        
    Returns:
        Optional[pl.DataFrame]: Processed data or None if error.
    """
    try:
        # Read CSV with Polars
        df = pl.read_csv(csv_file)
        
        # Find geographic code column
        geo_col = utils.find_geo_column(df, ['region_id', 'SA1_CODE21', 'SA2_CODE21'])
        if not geo_col:
            logger.error(f"No geographic code column found in {csv_file}")
            return None
            
        # Clean and select columns
        selected_cols = {
            geo_col: 'geo_code',
            'Tot_P_P': 'total_persons',
            'Tot_M_P': 'total_male',
            'Tot_F_P': 'total_female',
            'Indigenous_P': 'total_indigenous'
        }
        
        # Check if all required columns exist
        missing_cols = [col for col in selected_cols.keys() if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in {csv_file}: {missing_cols}")
            return None
            
        # Select and rename columns
        df = df.select([
            pl.col(old).alias(new)
            for old, new in selected_cols.items()
        ])
        
        # Clean geographic codes
        df = df.with_columns([
            utils.clean_polars_geo_code(pl.col('geo_code')).alias('geo_code')
        ])
        
        # Convert count columns to integers
        for col in ['total_persons', 'total_male', 'total_female', 'total_indigenous']:
            df = df.with_columns([
                utils.safe_polars_int(pl.col(col)).alias(col)
            ])
            
        # Drop rows with invalid codes or counts
        df = df.drop_nulls()
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing {csv_file}: {str(e)}")
        return None

def process_census_data(zip_dir: Path, temp_extract_base: Path, output_dir: Path,
                       geo_output_path: Path) -> bool:
    """Process Census G01 data and link with geographic boundaries.
    
    Args:
        zip_dir (Path): Directory containing ZIP files.
        temp_extract_base (Path): Base directory for temporary extraction.
        output_dir (Path): Directory for output files.
        geo_output_path (Path): Path to geographic boundaries Parquet file.
        
    Returns:
        bool: True if processing successful, False otherwise.
    """
    logger.info("=== Starting Census Data Processing ===")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find ZIP files containing G01 data
    g01_pattern = config.CENSUS_TABLE_PATTERNS['G01']
    g01_zips = find_census_files(zip_dir, g01_pattern)
    
    if not g01_zips:
        logger.error("No ZIP files containing G01 data found")
        return False
    
    # Process each ZIP file
    all_census_data = []
    success = True
    
    for zip_file in g01_zips:
        logger.info(f"Processing {zip_file.name}...")
        
        try:
            with zipfile.ZipFile(zip_file) as zf:
                # Find matching CSV files in zip
                csv_files = [name for name in zf.namelist() 
                           if re.search(g01_pattern, name, re.IGNORECASE)]
                
                if not csv_files:
                    logger.error(f"No G01 CSV files found in {zip_file}")
                    success = False
                    continue
                
                # Process each CSV file in memory
                for csv_name in csv_files:
                    logger.info(f"Processing {csv_name}...")
                    
                    try:
                        # Read CSV directly from zip into memory
                        with zf.open(csv_name) as csv_file:
                            df = pl.read_csv(csv_file)
                            
                            # Find geographic code column
                            geo_col = utils.find_geo_column(df, ['region_id', 'SA1_CODE21', 'SA2_CODE21'])
                            if not geo_col:
                                logger.error(f"No geographic code column found in {csv_name}")
                                continue
                            
                            # Clean and select columns
                            selected_cols = {
                                geo_col: 'geo_code',
                                'Tot_P_P': 'total_persons',
                                'Tot_M_P': 'total_male',
                                'Tot_F_P': 'total_female',
                                'Indigenous_P': 'total_indigenous'
                            }
                            
                            # Check if all required columns exist
                            missing_cols = [col for col in selected_cols.keys() if col not in df.columns]
                            if missing_cols:
                                logger.error(f"Missing columns in {csv_name}: {missing_cols}")
                                continue
                            
                            # Select and rename columns
                            df = df.select([
                                pl.col(old).alias(new)
                                for old, new in selected_cols.items()
                            ])
                            
                            # Clean geographic codes
                            df = df.with_columns([
                                utils.clean_polars_geo_code(pl.col('geo_code')).alias('geo_code')
                            ])
                            
                            # Convert count columns to integers
                            for col in ['total_persons', 'total_male', 'total_female', 'total_indigenous']:
                                df = df.with_columns([
                                    utils.safe_polars_int(pl.col(col)).alias(col)
                                ])
                            
                            # Drop rows with invalid codes or counts
                            df = df.drop_nulls()
                            
                            all_census_data.append(df)
                            logger.info(f"Successfully processed {len(df)} rows from {csv_name}")
                            
                    except Exception as e:
                        logger.error(f"Error processing {csv_name}: {str(e)}")
                        success = False
                        
        except Exception as e:
            logger.error(f"Error opening {zip_file}: {str(e)}")
            success = False
    
    if not all_census_data:
        logger.error("No census data was successfully processed")
        return False
    
    try:
        # Combine all census data
        logger.info("Combining all census data...")
        combined_df = pl.concat(all_census_data)
        
        # Load geographic boundaries for validation
        logger.info("Loading geographic boundaries for validation...")
        geo_df = pl.read_parquet(geo_output_path)
        
        # Join with geographic boundaries to validate codes
        logger.info("Validating against geographic boundaries...")
        validated_df = combined_df.join(
            geo_df.select(['geo_code']).unique(),
            on='geo_code',
            how='inner'
        )
        
        # Write to Parquet
        output_file = output_dir / "population_dimension.parquet"
        validated_df.write_parquet(output_file)
        logger.info(f"Successfully wrote population data to {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error combining/writing census data: {str(e)}")
        return False 