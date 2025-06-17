"""
ABS Boundary Processor for Australian Statistical Geography Standard (ASGS) data.

Processes SA2 boundary shapefiles from Australian Bureau of Statistics:
- Downloads and extracts SA2 boundary ZIP files
- Converts shapefiles to GeoJSON for web mapping
- Links geographic boundaries with SEIFA socio-economic data
- Optimises file sizes for dashboard performance

Based on ASGS Edition 3 (2021-2026) SA2 boundaries.
"""

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import polars as pl
from loguru import logger
from rich.console import Console
from shapely.geometry import mapping

console = Console()

# SA2 Boundary processing configuration
BOUNDARY_CONFIG = {
    "filename": "SA2_2021_AUST_SHP_GDA94.zip",
    "shapefile_pattern": "SA2_2021_AUST_GDA94.shp", 
    "expected_records": 2368,  # Total SA2 areas in Australia
    "coordinate_system": "GDA94",  # Geocentric Datum of Australia 1994
    "geometry_type": "MultiPolygon",
    "simplification_tolerance": 0.001,  # Degrees for file size optimization
}

# SA2 shapefile schema mapping  
SA2_BOUNDARY_SCHEMA = {
    "sa2_code_2021": "SA2_CODE21",      # SA2 Code 2021
    "sa2_name_2021": "SA2_NAME21",      # SA2 Name 2021
    "gcc_code_2021": "GCC_CODE21",      # Greater Capital City Code
    "gcc_name_2021": "GCC_NAME21",      # Greater Capital City Name
    "state_code_2021": "STE_CODE21",    # State/Territory Code
    "state_name_2021": "STE_NAME21",    # State/Territory Name
    "area_sqkm": "AREASQKM21",          # Area in square kilometres
    "geometry": "geometry"              # Geographic boundaries
}

# GeoJSON export configuration
GEOJSON_CONFIG = {
    "precision": 6,  # Decimal places for coordinates
    "validate": True,  # Validate geometries
    "drop_invalid": True,  # Remove invalid geometries
    "ensure_valid": True,  # Fix invalid geometries where possible
}


class BoundaryProcessor:
    """
    High-performance boundary processor for Australian SA2 geographic data.
    
    Converts ABS shapefiles to analysis-ready GeoJSON with SEIFA integration.
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.geojson_dir = self.processed_dir / "geojson"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.geojson_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Boundary Processor initialized")
        logger.info(f"Raw data directory: {self.raw_dir}")
        logger.info(f"GeoJSON output directory: {self.geojson_dir}")
    
    def validate_boundary_file(self, file_path: Path) -> bool:
        """
        Validate SA2 boundary ZIP file structure.
        
        Args:
            file_path: Path to SA2 boundary ZIP file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            if not file_path.exists():
                logger.error(f"Boundary file not found: {file_path}")
                return False
            
            # Check file size (should be ~47MB)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 30 or file_size_mb > 80:
                logger.warning(f"Unexpected boundary file size: {file_size_mb:.1f}MB")
            
            # Validate ZIP structure
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Check for required shapefile components
                required_extensions = ['.shp', '.shx', '.dbf', '.prj']
                found_extensions = set()
                
                for filename in file_list:
                    for ext in required_extensions:
                        if filename.endswith(ext):
                            found_extensions.add(ext)
                
                missing_extensions = set(required_extensions) - found_extensions
                if missing_extensions:
                    logger.error(f"Missing shapefile components: {missing_extensions}")
                    return False
                
                # Check for main shapefile
                shapefile_found = any(
                    BOUNDARY_CONFIG["shapefile_pattern"] in filename 
                    for filename in file_list
                )
                
                if not shapefile_found:
                    logger.error(f"Main shapefile not found: {BOUNDARY_CONFIG['shapefile_pattern']}")
                    return False
            
            logger.info(f"✓ Boundary file validation passed: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Boundary file validation failed: {e}")
            return False
    
    def extract_boundary_data(self, zip_path: Path) -> gpd.GeoDataFrame:
        """
        Extract SA2 boundary data from ZIP file to GeoDataFrame.
        
        Args:
            zip_path: Path to SA2 boundary ZIP file
            
        Returns:
            GeoDataFrame with standardized SA2 boundary schema
        """
        if not self.validate_boundary_file(zip_path):
            raise ValueError(f"Invalid boundary file: {zip_path}")
        
        logger.info(f"Extracting boundary data from {zip_path.name}")
        
        try:
            # Extract ZIP to temporary directory
            extract_dir = self.raw_dir / "temp_boundaries"
            extract_dir.mkdir(exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the main shapefile
            shapefile_path = None
            for file_path in extract_dir.rglob("*.shp"):
                if BOUNDARY_CONFIG["shapefile_pattern"] in str(file_path):
                    shapefile_path = file_path
                    break
            
            if not shapefile_path:
                raise FileNotFoundError(f"Shapefile not found: {BOUNDARY_CONFIG['shapefile_pattern']}")
            
            # Read shapefile with GeoPandas
            logger.info(f"Reading shapefile: {shapefile_path}")
            gdf = gpd.read_file(shapefile_path)
            
            logger.info(f"Loaded {len(gdf)} boundary records with {len(gdf.columns)} columns")
            
            # Standardize column names and schema
            standardized_gdf = self._standardize_boundary_columns(gdf)
            
            # Validate and clean geometries
            validated_gdf = self._validate_boundary_geometries(standardized_gdf)
            
            # Clean up temporary extraction
            import shutil
            shutil.rmtree(extract_dir)
            
            logger.info(f"✓ Successfully processed boundary data: {len(validated_gdf)} SA2 areas")
            return validated_gdf
            
        except Exception as e:
            logger.error(f"Failed to extract boundary data: {e}")
            raise
    
    def _standardize_boundary_columns(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Standardize SA2 boundary column names and data types.
        
        Maps shapefile column names to our standard schema.
        """
        logger.info("Standardizing boundary column names")
        
        # Create reverse mapping from schema
        column_mapping = {}
        for standard_col, shapefile_col in SA2_BOUNDARY_SCHEMA.items():
            if shapefile_col in gdf.columns:
                column_mapping[shapefile_col] = standard_col
        
        # Apply column mapping
        standardized_gdf = gdf.rename(columns=column_mapping)
        
        # Select only columns we need
        available_columns = [col for col in SA2_BOUNDARY_SCHEMA.keys() if col in standardized_gdf.columns]
        final_gdf = standardized_gdf[available_columns].copy()
        
        logger.info(f"Standardized {len(available_columns)} columns: {available_columns}")
        return final_gdf
    
    def _validate_boundary_geometries(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Validate and clean SA2 boundary geometries.
        
        Fixes invalid geometries and filters out problematic records.
        """
        logger.info("Validating boundary geometries")
        
        initial_count = len(gdf)
        
        # Check for valid geometries
        if GEOJSON_CONFIG["validate"]:
            valid_mask = gdf.geometry.is_valid
            invalid_count = (~valid_mask).sum()
            
            if invalid_count > 0:
                logger.warning(f"Found {invalid_count} invalid geometries")
                
                if GEOJSON_CONFIG["ensure_valid"]:
                    # Try to fix invalid geometries
                    gdf.loc[~valid_mask, 'geometry'] = gdf.loc[~valid_mask, 'geometry'].buffer(0)
                    valid_mask = gdf.geometry.is_valid
                    fixed_count = invalid_count - (~valid_mask).sum()
                    logger.info(f"Fixed {fixed_count} invalid geometries")
                
                if GEOJSON_CONFIG["drop_invalid"]:
                    gdf = gdf[valid_mask].copy()
                    logger.info(f"Removed {(~valid_mask).sum()} invalid geometries")
        
        # Validate SA2 codes (should be 9-digit strings)
        if "sa2_code_2021" in gdf.columns:
            valid_sa2_mask = gdf["sa2_code_2021"].astype(str).str.len() == 9
            invalid_sa2_count = (~valid_sa2_mask).sum()
            
            if invalid_sa2_count > 0:
                logger.warning(f"Found {invalid_sa2_count} invalid SA2 codes")
                gdf = gdf[valid_sa2_mask].copy()
        
        # Check coordinate system
        if gdf.crs is None:
            logger.warning("No CRS found, assuming GDA94 (EPSG:4283)")
            gdf = gdf.set_crs("EPSG:4283")
        
        final_count = len(gdf)
        logger.info(f"Geometry validation complete: {initial_count} → {final_count} records")
        
        if final_count < BOUNDARY_CONFIG["expected_records"] * 0.9:
            logger.warning(f"Significant data loss during validation: {final_count} vs expected {BOUNDARY_CONFIG['expected_records']}")
        
        return gdf
    
    def simplify_boundaries(self, gdf: gpd.GeoDataFrame, tolerance: float = None) -> gpd.GeoDataFrame:
        """
        Simplify boundary geometries for performance optimization.
        
        Args:
            gdf: GeoDataFrame with boundary data
            tolerance: Simplification tolerance in degrees
            
        Returns:
            GeoDataFrame with simplified geometries
        """
        if tolerance is None:
            tolerance = BOUNDARY_CONFIG["simplification_tolerance"]
        
        logger.info(f"Simplifying geometries with tolerance: {tolerance}")
        
        # Calculate original size
        original_size = gdf.memory_usage(deep=True).sum() / (1024 * 1024)
        
        # Simplify geometries
        simplified_gdf = gdf.copy()
        simplified_gdf['geometry'] = gdf.geometry.simplify(tolerance, preserve_topology=True)
        
        # Calculate new size
        new_size = simplified_gdf.memory_usage(deep=True).sum() / (1024 * 1024)
        reduction = ((original_size - new_size) / original_size) * 100
        
        logger.info(f"Geometry simplification: {original_size:.1f}MB → {new_size:.1f}MB ({reduction:.1f}% reduction)")
        
        return simplified_gdf
    
    def export_geojson(self, gdf: gpd.GeoDataFrame, filename: str = "sa2_boundaries_2021.geojson") -> Path:
        """
        Export boundary data to GeoJSON format.
        
        Args:
            gdf: GeoDataFrame with boundary data
            filename: Output filename
            
        Returns:
            Path to exported GeoJSON file
        """
        output_path = self.geojson_dir / filename
        
        logger.info(f"Exporting GeoJSON to {output_path}")
        
        try:
            # Export to GeoJSON with optimized settings
            gdf.to_file(
                output_path,
                driver="GeoJSON",
                index=False
            )
            
            # Get file size
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Exported GeoJSON: {file_size_mb:.1f}MB")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export GeoJSON: {e}")
            raise
    
    def process_boundary_file(self, filename: Optional[str] = None) -> gpd.GeoDataFrame:
        """
        Process SA2 boundary ZIP file to analysis-ready GeoDataFrame.
        
        Args:
            filename: Boundary ZIP filename (default: from config)
            
        Returns:
            Processed GeoDataFrame with SA2 boundary data
        """
        if filename is None:
            filename = BOUNDARY_CONFIG["filename"]
        
        file_path = self.raw_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Boundary file not found: {file_path}")
        
        # Extract and process boundary data
        boundary_gdf = self.extract_boundary_data(file_path)
        
        # Simplify geometries for performance
        simplified_gdf = self.simplify_boundaries(boundary_gdf)
        
        # Export to GeoJSON
        geojson_path = self.export_geojson(simplified_gdf)
        logger.info(f"Exported boundary data to {geojson_path}")
        
        return simplified_gdf
    
    def link_with_seifa(self, boundary_gdf: gpd.GeoDataFrame, seifa_df: pl.DataFrame) -> gpd.GeoDataFrame:
        """
        Link boundary geometries with SEIFA socio-economic data.
        
        Args:
            boundary_gdf: GeoDataFrame with SA2 boundaries
            seifa_df: Polars DataFrame with SEIFA data
            
        Returns:
            GeoDataFrame with boundaries and SEIFA attributes
        """
        logger.info("Linking boundary data with SEIFA socio-economic indicators")
        
        # Convert SEIFA data to pandas for GeoPandas compatibility
        seifa_pandas = seifa_df.to_pandas()
        
        # Merge on SA2 code
        merged_gdf = boundary_gdf.merge(
            seifa_pandas,
            on="sa2_code_2021",
            how="left"
        )
        
        # Check merge success
        matched_count = merged_gdf["irsd_score"].notna().sum()
        total_boundaries = len(boundary_gdf)
        match_rate = (matched_count / total_boundaries) * 100
        
        logger.info(f"SEIFA linkage: {matched_count}/{total_boundaries} boundaries matched ({match_rate:.1f}%)")
        
        if match_rate < 90:
            logger.warning(f"Low SEIFA match rate: {match_rate:.1f}%")
        
        return merged_gdf
    
    def get_boundary_summary(self, boundary_gdf: gpd.GeoDataFrame) -> Dict:
        """
        Generate summary statistics for boundary data.
        
        Args:
            boundary_gdf: GeoDataFrame with boundary data
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "total_sa2_areas": len(boundary_gdf),
            "states_covered": [],
            "total_area_sqkm": 0,
            "coordinate_system": str(boundary_gdf.crs),
            "geometry_type": boundary_gdf.geom_type.iloc[0] if len(boundary_gdf) > 0 else None
        }
        
        # Extract state information
        if "state_name_2021" in boundary_gdf.columns:
            states = boundary_gdf["state_name_2021"].unique()
            summary["states_covered"] = sorted([state for state in states if state])
        
        # Calculate total area
        if "area_sqkm" in boundary_gdf.columns:
            summary["total_area_sqkm"] = float(boundary_gdf["area_sqkm"].sum())
        
        return summary
    
    def process_complete_pipeline(self) -> gpd.GeoDataFrame:
        """
        Execute complete boundary processing pipeline.
        
        Returns:
            Fully processed GeoDataFrame ready for analysis
        """
        console.print("🗺️  [bold blue]Starting boundary processing pipeline...[/bold blue]")
        
        try:
            # Process boundary data
            boundary_gdf = self.process_boundary_file()
            console.print(f"✅ Processed {len(boundary_gdf)} SA2 boundary areas")
            
            # Generate summary
            summary = self.get_boundary_summary(boundary_gdf)
            console.print(f"✅ Coverage: {summary['states_covered']}")
            console.print(f"✅ Total area: {summary['total_area_sqkm']:,.0f} km²")
            console.print(f"✅ Coordinate system: {summary['coordinate_system']}")
            
            console.print("🎉 [bold green]Boundary processing pipeline complete![/bold green]")
            return boundary_gdf
            
        except Exception as e:
            console.print(f"❌ [bold red]Boundary processing failed: {e}[/bold red]")
            raise