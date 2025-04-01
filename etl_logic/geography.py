"""Geographic data processing module for the AHGD ETL pipeline.

This module handles the processing of ABS ASGS geographic boundary files,
including downloading, extracting, and transforming shapefile data into
a standardized Parquet format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import datetime

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
            
            # Find the geographic code column with appropriate names for the level
            if level_name == 'STATE':
                # For STATE level, include additional possible column names used by ABS
                possible_names = [f"{level_name}_CODE_2021", f"{level_name}_CODE21", 
                                 "STE_CODE_2021", "STE_CODE21", "STATE_CODE21", "STATE_CODE_2021"]
            else:
                possible_names = [f"{level_name}_CODE_2021", f"{level_name}_CODE21"]
                
            geo_col = utils.find_geo_column(gdf, possible_names)
            
            if not geo_col:
                logger.error(f"Could not find geographic code column for {level_name}")
                success = False
                continue
                
            # Basic cleaning and validation
            gdf['geometry'] = gdf['geometry'].apply(lambda g: make_valid(g) if g else None)
            gdf = gdf.dropna(subset=['geometry'])
            
            # Ensure the geometry column is active and project to a suitable CRS for Australia
            gdf = gdf.set_geometry("geometry")
            # Project to GDA2020 / MGA zone 55 (EPSG:7855) which is suitable for most of Australia
            gdf = gdf.to_crs(epsg=7855)
            # Calculate geometric centroids
            gdf['centroid_longitude'] = gdf.geometry.centroid.x
            gdf['centroid_latitude'] = gdf.geometry.centroid.y
            logger.info(f"[{level_name}] Calculated and added geometric centroids (longitude, latitude)")
            
            # Project back to GDA2020 geographic (EPSG:7844) for storage
            gdf = gdf.to_crs(epsg=7844)
            
            # Convert to WKT for Polars/Parquet compatibility
            gdf['geometry_wkt'] = gdf['geometry'].apply(utils.geometry_to_wkt)
            
            # Select and rename columns
            df = pd.DataFrame({
                'geo_code': gdf[geo_col].apply(utils.clean_geo_code),
                'geo_level': level_name,
                'geometry': gdf['geometry_wkt'],
                'centroid_longitude': gdf['centroid_longitude'],
                'centroid_latitude': gdf['centroid_latitude']
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
        
        # Add surrogate key column
        logger.info("Adding surrogate key to combined geographic data...")
        final_geo_df = combined_df.with_row_index(name='geo_sk')  # Add unique integer SK
        
        # Add ETL processed timestamp
        final_geo_df = final_geo_df.with_columns(pl.lit(datetime.datetime.now()).alias('etl_processed_at'))
        
        # Write to Parquet
        output_file = output_dir / "geo_dimension.parquet"
        # Ensure geo_sk is the first column followed by the other columns
        final_geo_df = final_geo_df.select(['geo_sk', 'geo_code', 'geo_level', 'geometry', 'centroid_longitude', 'centroid_latitude', 'etl_processed_at'])
        final_geo_df.write_parquet(output_file)
        logger.info(f"Successfully wrote combined geographic data to {output_file}")
        
    except Exception as e:
        logger.error(f"Error combining/writing geographic data: {str(e)}")
        return False
        
    return success 

def update_population_weighted_centroids(geo_output_path: Path, population_fact_path: Path) -> bool:
    """Update geographic dimension with population-weighted centroids.
    
    Args:
        geo_output_path (Path): Path to the geographic dimension Parquet file
        population_fact_path (Path): Path to the population fact table Parquet file
        
    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        logger.info("Updating population-weighted centroids...")
        
        # Load geographic dimension
        geo_df = pl.read_parquet(geo_output_path)
        
        # Load population fact table
        pop_df = pl.read_parquet(population_fact_path)
        
        # Join with population data
        joined_df = geo_df.join(
            pop_df.select(['geo_sk', 'total_persons']),
            on='geo_sk',
            how='left'
        )
        
        # Convert to GeoPandas for centroid calculation
        gdf = gpd.GeoDataFrame(
            joined_df.to_pandas(),
            geometry=gpd.GeoSeries.from_wkt(joined_df['geometry'].to_list())
        )
        
        # Calculate population-weighted centroids
        def calculate_weighted_centroid(group):
            if group['total_persons'].sum() > 0:
                # Weight by population
                weights = group['total_persons'] / group['total_persons'].sum()
                # Calculate weighted average of centroids
                weighted_lon = (group['centroid_lon'] * weights).sum()
                weighted_lat = (group['centroid_lat'] * weights).sum()
                return pd.Series({'pop_weighted_lon': weighted_lon, 'pop_weighted_lat': weighted_lat})
            else:
                # If no population data, use geometric centroid
                return pd.Series({'pop_weighted_lon': group['centroid_lon'].iloc[0], 
                                'pop_weighted_lat': group['centroid_lat'].iloc[0]})
        
        # Calculate weighted centroids for each geographic level
        weighted_centroids = gdf.groupby('geo_level').apply(calculate_weighted_centroid).reset_index()
        
        # Update the geographic dimension
        updated_geo_df = geo_df.join(
            pl.from_pandas(weighted_centroids),
            on='geo_level',
            how='left'
        )
        
        # Write updated geographic dimension
        updated_geo_df.write_parquet(geo_output_path)
        logger.info("Successfully updated population-weighted centroids")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating population-weighted centroids: {str(e)}")
        logger.exception(e)
        return False 