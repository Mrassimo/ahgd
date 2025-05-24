"""
Geographic data processing for the AHGD ETL pipeline.

This module handles the processing of ABS ASGS geographic boundary files,
including extracting shapefile data and transforming it into a standardized
Parquet format for the geographic dimension.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import datetime
import hashlib

import pandas as pd
import geopandas as gpd
import polars as pl
from shapely.validation import make_valid

from ...config import settings
from ...utils import extract_zipfile, geometry_to_wkt, clean_geo_code

logger = logging.getLogger('ahgd_etl.transformers.geo')

class GeoTransformer:
    """
    Transformer for geographic boundary data.
    
    This class handles the processing of ABS ASGS geographic boundary files 
    into the geographic dimension for the AHGD data warehouse.
    """
    
    def __init__(self):
        """Initialize the GeoTransformer."""
        self.logger = logger
        
        # Get geo levels from settings
        self.geo_levels = settings.get_column_mapping('geo_levels')
        if not self.geo_levels:
            # Fallback if not in YAML
            self.geo_levels = {
                'SA1': 'SA1_2021_AUST_GDA2020',
                'SA2': 'SA2_2021_AUST_GDA2020',
                'SA3': 'SA3_2021_AUST_GDA2020',
                'SA4': 'SA4_2021_AUST_GDA2020',
                'STATE': 'STE_2021_AUST_GDA2020',
                'POA': 'POA_2021_AUST_GDA2020'
            }
    
    def process_level(self, level_name: str, extract_dir: Path) -> Optional[pl.DataFrame]:
        """
        Process a single geographic level.
        
        Args:
            level_name: Name of the geographic level (e.g., 'SA1', 'SA2')
            extract_dir: Directory containing extracted shapefile
            
        Returns:
            Processed Polars DataFrame or None if processing failed
        """
        try:
            # Find and read shapefile
            shp_files = list(extract_dir.glob("*.shp"))
            if not shp_files:
                self.logger.error(f"No shapefile found in {extract_dir}")
                return None
                
            # Read shapefile with geopandas
            gdf = gpd.read_file(shp_files[0])
            
            # Find the geographic code column
            if level_name == 'STATE':
                # For STATE level, include additional possible column names used by ABS
                possible_names = [f"{level_name}_CODE_2021", f"{level_name}_CODE21", 
                                "STE_CODE_2021", "STE_CODE21", "STATE_CODE21", "STATE_CODE_2021"]
            else:
                possible_names = [f"{level_name}_CODE_2021", f"{level_name}_CODE21"]
                
            geo_col = None
            for name in possible_names:
                if name in gdf.columns:
                    geo_col = name
                    break
            
            if not geo_col:
                self.logger.error(f"Could not find geographic code column for {level_name}")
                return None
                
            # Basic cleaning and validation
            gdf['geometry'] = gdf['geometry'].apply(lambda g: make_valid(g) if g else None)
            gdf = gdf.dropna(subset=['geometry'])
            
            # Ensure the geometry column is active and project to a suitable CRS for Australia
            gdf = gdf.set_geometry("geometry")
            # Project to GDA2020 / MGA zone 55 (EPSG:7855) which is suitable for most of Australia
            gdf = gdf.to_crs(epsg=7855)
            # Calculate geometric centroids
            gdf['longitude'] = gdf.geometry.centroid.x
            gdf['latitude'] = gdf.geometry.centroid.y
            self.logger.info(f"[{level_name}] Calculated and added centroids (longitude, latitude)")
            
            # Project back to GDA2020 geographic (EPSG:7844) for storage
            gdf = gdf.to_crs(epsg=7844)
            
            # Convert to WKT for Polars/Parquet compatibility
            gdf['geometry_wkt'] = gdf['geometry'].apply(geometry_to_wkt)
            
            # Get name column if available
            name_col = None
            for name_option in [f"{level_name}_NAME_2021", f"{level_name}_NAME21"]:
                if name_option in gdf.columns:
                    name_col = name_option
                    break
            
            # Get state code column if available
            state_col = None
            for state_option in ["STE_CODE_2021", "STE_CODE21", "STATE_CODE_2021", "STATE_CODE21"]:
                if state_option in gdf.columns:
                    state_col = state_option
                    break
            
            # Get state name column if available
            state_name_col = None
            for state_name_option in ["STE_NAME_2021", "STE_NAME21", "STATE_NAME_2021", "STATE_NAME21"]:
                if state_name_option in gdf.columns:
                    state_name_col = state_name_option
                    break
            
            # Select and rename columns
            data = {
                'geo_id': gdf[geo_col].apply(clean_geo_code),
                'geo_level': level_name,
                'geom': gdf['geometry_wkt'],
                'longitude': gdf['longitude'],
                'latitude': gdf['latitude']
            }
            
            # Add name if available
            if name_col:
                data['geo_name'] = gdf[name_col]
            else:
                data['geo_name'] = "Unknown"
                
            # Add state code and name if available
            if state_col:
                data['state_code'] = gdf[state_col]
            else:
                data['state_code'] = "Unknown"
                
            if state_name_col:
                data['state_name'] = gdf[state_name_col]
            else:
                data['state_name'] = "Unknown"
            
            df = pd.DataFrame(data)
            
            # Drop any rows with invalid codes or geometries
            df = df.dropna(subset=['geo_id', 'geom'])
            
            # Generate surrogate keys using MD5 hash of geo_id and geo_level
            df['geo_sk'] = df.apply(
                lambda row: int(hashlib.md5(f"{row['geo_id']}_{row['geo_level']}".encode()).hexdigest(), 16) % (2**31-1), 
                axis=1
            )
            
            # Add parent geo_sk if applicable
            # For SA1, find parent SA2, etc.
            if level_name == 'SA1' and 'SA2_CODE_2021' in gdf.columns:
                # Link to parent SA2
                df['parent_geo_level'] = 'SA2'
                df['parent_geo_id'] = gdf['SA2_CODE_2021'].apply(clean_geo_code)
            elif level_name == 'SA2' and 'SA3_CODE_2021' in gdf.columns:
                # Link to parent SA3
                df['parent_geo_level'] = 'SA3'
                df['parent_geo_id'] = gdf['SA3_CODE_2021'].apply(clean_geo_code)
            elif level_name == 'SA3' and 'SA4_CODE_2021' in gdf.columns:
                # Link to parent SA4
                df['parent_geo_level'] = 'SA4'
                df['parent_geo_id'] = gdf['SA4_CODE_2021'].apply(clean_geo_code)
            elif level_name == 'SA4' and 'STE_CODE_2021' in gdf.columns:
                # Link to parent STATE
                df['parent_geo_level'] = 'STATE'
                df['parent_geo_id'] = gdf['STE_CODE_2021'].apply(clean_geo_code)
            else:
                df['parent_geo_level'] = None
                df['parent_geo_id'] = None
            
            # Add ETL timestamp
            df['etl_processed_at'] = datetime.datetime.now()
            
            # Convert to Polars for efficient processing
            pl_df = pl.from_pandas(df)
            
            self.logger.info(f"Successfully processed {len(pl_df)} {level_name} boundaries")
            return pl_df
            
        except Exception as e:
            self.logger.error(f"Error processing {level_name}: {str(e)}")
            self.logger.exception(e)
            return None
    
    def process_all_levels(self, zip_dir: Path, temp_extract_base: Path) -> pl.DataFrame:
        """
        Process all geographic levels.
        
        Args:
            zip_dir: Directory containing downloaded ZIP files
            temp_extract_base: Base directory for temporary extraction
            
        Returns:
            Combined Polars DataFrame with all geographic levels
        """
        all_geo_data = []
        
        for level_name, prefix in self.geo_levels.items():
            self.logger.info(f"Processing {level_name} boundaries...")
            
            # Construct zip filename
            zip_path = zip_dir / f"{prefix}.zip"
            
            if not zip_path.exists():
                self.logger.error(f"ZIP file not found: {zip_path}")
                continue
                
            # Create temporary extraction directory
            extract_dir = temp_extract_base / level_name
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract ZIP file
            if not extract_zipfile(zip_path, extract_dir, desc=f"Extracting {level_name} ZIP"):
                self.logger.error(f"Failed to extract {zip_path}")
                continue
            
            # Process the level
            level_df = self.process_level(level_name, extract_dir)
            
            if level_df is not None:
                all_geo_data.append(level_df)
        
        if not all_geo_data:
            self.logger.error("No geographic data was successfully processed")
            raise ValueError("No geographic data was successfully processed")
        
        # Combine all geographic levels
        self.logger.info("Combining all geographic levels...")
        combined_df = pl.concat(all_geo_data)
        
        # Add unknown record
        unknown_geo_record = pl.DataFrame({
            'geo_sk': [-1],
            'geo_id': ['UNKNOWN'],
            'geo_level': ['UNKNOWN'],
            'geo_name': ['Unknown'],
            'state_code': ['Unknown'],
            'state_name': ['Unknown'],
            'geom': [None],
            'longitude': [None],
            'latitude': [None],
            'parent_geo_level': [None],
            'parent_geo_id': [None],
            'etl_processed_at': [datetime.datetime.now()]
        })
        
        # Ensure schema alignment
        for col in combined_df.columns:
            if col not in unknown_geo_record.columns:
                unknown_geo_record = unknown_geo_record.with_columns(pl.lit(None).alias(col))

        # Ensure column order exactly matches
        unknown_geo_record = unknown_geo_record.select(combined_df.columns)

        # Debug logging for column names
        logger.info(f"Combined DF columns: {combined_df.columns}")
        logger.info(f"Unknown record columns: {unknown_geo_record.columns}")

        # Combine the unknown record with the main data
        final_geo_df = pl.concat([combined_df, unknown_geo_record], how="vertical_relaxed")
        self.logger.info("Added 'UNKNOWN' record with geo_sk = -1")
        
        return final_geo_df
    
    def update_parent_references(self, geo_df: pl.DataFrame) -> pl.DataFrame:
        """
        Update parent surrogate key references.
        
        Args:
            geo_df: Geographic dimension DataFrame
            
        Returns:
            Updated DataFrame with parent_geo_sk filled in
        """
        # Create a mapping of geo_id and level to geo_sk
        geo_map = {}
        for row in geo_df.select(['geo_sk', 'geo_id', 'geo_level']).iter_rows(named=True):
            key = (row['geo_id'], row['geo_level'])
            geo_map[key] = row['geo_sk']
        
        # Map parent geo_id to parent_geo_sk
        def map_parent_sk(parent_id, parent_level):
            if parent_id and parent_level:
                return geo_map.get((parent_id, parent_level), -1)
            return None
        
        # Apply the mapping
        parent_sk_list = []
        for row in geo_df.select(['parent_geo_id', 'parent_geo_level']).iter_rows(named=True):
            parent_sk_list.append(map_parent_sk(row['parent_geo_id'], row['parent_geo_level']))
        
        # Add the parent_geo_sk column
        updated_df = geo_df.with_columns(pl.Series(parent_sk_list).alias('parent_geo_sk'))
        
        return updated_df
    
    def save_to_parquet(self, geo_df: pl.DataFrame, output_path: Path) -> None:
        """
        Save geographic dimension to Parquet.
        
        Args:
            geo_df: Geographic dimension DataFrame
            output_path: Path to save the Parquet file
        """
        # Ensure geo_sk is the first column followed by the other columns
        ordered_columns = ['geo_sk', 'geo_id', 'geo_level', 'geo_name', 
                         'state_code', 'state_name', 'geom', 'longitude', 
                         'latitude', 'parent_geo_sk', 'etl_processed_at']
        
        # Only select columns that exist in the DataFrame
        columns_to_select = [col for col in ordered_columns if col in geo_df.columns]
        
        # Reorder columns
        geo_df = geo_df.select(columns_to_select)
        
        # Write to Parquet
        geo_df.write_parquet(output_path)
        self.logger.info(f"Successfully wrote geographic dimension to {output_path}")


def process_geography(shp_dir: Optional[Path] = None,
                     output_path: Optional[Path] = None,
                     geo_levels: Optional[List[str]] = None,
                     temp_extract_base: Optional[Path] = None) -> bool:
    """
    Process geographic boundary files from ABS ASGS.

    This function serves as a compatibility wrapper for the old API.

    Args:
        shp_dir: Directory containing downloaded ZIP files
        output_path: Path to save the output Parquet file
        geo_levels: List of geographic levels to process
        temp_extract_base: Base directory for temporary file extraction

    Returns:
        True if processing successful, False otherwise
    """
    try:
        # Use default paths if not provided
        if shp_dir is None:
            shp_dir = settings.get_path('GEOGRAPHIC_DIR')
        
        if output_path is None:
            output_path = settings.get_path('OUTPUT_DIR') / "geo_dimension.parquet"
        
        # Use provided temp extraction directory or get from settings
        if temp_extract_base is None:
            temp_extract_base = settings.get_path('TEMP_EXTRACT_DIR')
        
        # Create transformer
        transformer = GeoTransformer()
        
        # Override geo_levels if provided
        if geo_levels:
            # Filter to only include the specified levels
            transformer.geo_levels = {k: v for k, v in transformer.geo_levels.items() if k in geo_levels}
        
        # Process all levels
        geo_df = transformer.process_all_levels(shp_dir, temp_extract_base)
        
        # Update parent references
        geo_df = transformer.update_parent_references(geo_df)
        
        # Save to Parquet
        transformer.save_to_parquet(geo_df, output_path)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing geography: {str(e)}")
        logger.exception(e)
        return False


def update_population_weighted_centroids(geo_output_path: Path, population_fact_path: Path) -> bool:
    """
    Update geographic dimension with population-weighted centroids.
    
    Args:
        geo_output_path: Path to the geographic dimension Parquet file
        population_fact_path: Path to the population fact table Parquet file
        
    Returns:
        True if update successful, False otherwise
    """
    try:
        logger.info("Updating population-weighted centroids...")
        
        # Load geographic dimension
        geo_df = pl.read_parquet(geo_output_path)
        
        # Load population fact table
        pop_df = pl.read_parquet(population_fact_path)
        
        # Join with population data
        joined_df = geo_df.join(
            pop_df.select(['geo_sk', 'total_population']),
            on='geo_sk',
            how='left'
        )
        
        # Convert to GeoPandas for centroid calculation
        gdf = gpd.GeoDataFrame(
            joined_df.to_pandas(),
            geometry=gpd.GeoSeries.from_wkt(joined_df['geom'].to_list())
        )
        
        # Calculate population-weighted centroids
        def calculate_weighted_centroid(group):
            if 'total_population' in group.columns and group['total_population'].sum() > 0:
                # Weight by population
                weights = group['total_population'] / group['total_population'].sum()
                # Calculate weighted average of centroids
                weighted_lon = (group['longitude'] * weights).sum()
                weighted_lat = (group['latitude'] * weights).sum()
                return pd.Series({'pop_weighted_lon': weighted_lon, 'pop_weighted_lat': weighted_lat})
            else:
                # If no population data, use geometric centroid
                return pd.Series({'pop_weighted_lon': group['longitude'].iloc[0], 
                                'pop_weighted_lat': group['latitude'].iloc[0]})
        
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


# Alias for backward compatibility
GeographyTransformer = GeoTransformer