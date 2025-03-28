"""Geographic data processing module for the AHGD ETL pipeline.

This module handles the processing of ABS ASGS geographic boundary files,
including downloading, extracting, and transforming shapefile data into
a standardized Parquet format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd
import geopandas as gpd
import polars as pl
import pyarrow
import pyarrow.parquet as pq
from shapely.validation import make_valid

from . import config
from . import utils

logger = logging.getLogger('ahgd_etl')

def process_geography(zip_dir: Path, temp_extract_base: Path, output_dir: Path) -> bool:
    """Process geographic boundary files from ABS ASGS.
    
    Args:
        zip_dir (Path): Directory containing downloaded ZIP files.
        temp_extract_base (Path): Base directory for temporary extraction.
        output_dir (Path): Directory for output files.
        
    Returns:
        bool: True if processing successful, False otherwise.
    """
    logger.info("=== Starting Geographic Data Processing ===")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track overall success
    success = True
    all_geo_data = []
    
    # Process each geographic level
    for level_name, prefix in config.GEO_LEVELS_SHP_PROCESS.items():
        logger.info(f"Processing {level_name} boundaries...")
        
        # Construct zip filename and path
        zip_filename = f"{prefix}_SHP.zip"
        zip_path = zip_dir / zip_filename
        
        if not zip_path.exists():
            logger.error(f"ZIP file not found: {zip_path}")
            success = False
            continue
            
        # Create temporary extraction directory
        extract_dir = temp_extract_base / level_name
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract ZIP file
        if not utils.extract_zipfile(zip_path, extract_dir, desc=f"Extracting {level_name} ZIP"):
            logger.error(f"Failed to extract {zip_filename}")
            success = False
            continue
            
        try:
            # Find and read shapefile
            shp_files = list(extract_dir.glob("*.shp"))
            if not shp_files:
                logger.error(f"No shapefile found in {extract_dir}")
                success = False
                continue
                
            # Read shapefile with geopandas
            gdf = gpd.read_file(shp_files[0])
            
            # Find the geographic code column
            possible_names = [f"{level_name}_CODE_2021", f"{level_name}_CODE21"]
            geo_col = utils.find_geo_column(gdf, possible_names)
            
            if not geo_col:
                logger.error(f"Could not find geographic code column for {level_name}")
                success = False
                continue
                
            # Basic cleaning and validation
            gdf['geometry'] = gdf['geometry'].apply(lambda g: make_valid(g) if g else None)
            gdf = gdf.dropna(subset=['geometry'])
            
            # Convert to WKT for Polars/Parquet compatibility
            gdf['geometry_wkt'] = gdf['geometry'].apply(utils.geometry_to_wkt)
            
            # Select and rename columns
            df = pd.DataFrame({
                'geo_code': gdf[geo_col].apply(utils.clean_geo_code),
                'geo_level': level_name,
                'geometry': gdf['geometry_wkt']
            })
            
            # Drop any rows with invalid codes or geometries
            df = df.dropna()
            
            # Convert to Polars for efficient processing
            pl_df = pl.from_pandas(df)
            
            # Add to collection
            all_geo_data.append(pl_df)
            
            logger.info(f"Successfully processed {len(pl_df)} {level_name} boundaries")
            
        except Exception as e:
            logger.error(f"Error processing {level_name}: {str(e)}")
            success = False
            continue
            
    if not all_geo_data:
        logger.error("No geographic data was successfully processed")
        return False
        
    try:
        # Combine all geographic levels
        logger.info("Combining all geographic levels...")
        combined_df = pl.concat(all_geo_data)
        
        # Write to Parquet
        output_file = output_dir / "geo_dimension.parquet"
        combined_df.write_parquet(output_file)
        logger.info(f"Successfully wrote combined geographic data to {output_file}")
        
    except Exception as e:
        logger.error(f"Error combining/writing geographic data: {str(e)}")
        return False
        
    return success 