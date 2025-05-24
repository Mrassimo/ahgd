"""Geographic dimension transformer for ASGS boundary data."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import geopandas as gpd
import polars as pl
from shapely import wkt
from shapely.validation import make_valid
from datetime import datetime
import hashlib

from ...config import get_settings
from ...utils import get_logger


class GeographyTransformer:
    """Transforms ASGS shapefile data into geographic dimension table."""
    
    def __init__(self):
        """Initialize the geography transformer."""
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Column mapping from settings
        self.column_mappings = self.settings._column_mappings.get("geographic", {})
        
        # Schema definition
        self.schema = self.settings.get_schema("geo_dimension")
        
        # Track processed records for surrogate key generation
        self.geo_sk_counter = 1000  # Start from 1000 for regular records
        
    def transform_all(self) -> pl.DataFrame:
        """Transform all configured geographic levels into a single dimension table.
        
        Returns:
            Polars DataFrame containing the complete geographic dimension
        """
        self.logger.info("Starting geographic dimension transformation")
        
        all_geo_data = []
        
        # Process each geographic level
        for level in self.settings.geographic_sources.keys():
            self.logger.info(f"Processing {level.upper()} geographic data")
            
            # Find shapefile for this level
            geo_dir = self.settings.raw_data_dir / "geographic" / level
            if not geo_dir.exists():
                self.logger.warning(f"Directory not found for {level}: {geo_dir}")
                continue
            
            shapefiles = list(geo_dir.glob("*.shp"))
            if not shapefiles:
                self.logger.warning(f"No shapefiles found for {level}")
                continue
            
            # Process the shapefile
            try:
                geo_df = self._process_shapefile(shapefiles[0], level)
                all_geo_data.append(geo_df)
                self.logger.info(f"Processed {len(geo_df)} {level.upper()} records")
            except Exception as e:
                self.logger.error(f"Failed to process {level}: {e}")
                if self.settings.stop_on_error:
                    raise
        
        if not all_geo_data:
            raise ValueError("No geographic data was successfully processed")
        
        # Enforce schema on each DataFrame before concatenating
        all_geo_data = [self._enforce_schema(df) for df in all_geo_data]
        
        # Combine all geographic levels
        combined_df = pl.concat(all_geo_data)
        
        # Add unknown member
        unknown_df = self._create_unknown_member()
        unknown_df = self._enforce_schema(unknown_df)
        
        final_df = pl.concat([unknown_df, combined_df])
        
        # Add parent relationships
        final_df = self._add_parent_relationships(final_df)
        
        self.logger.info(f"Geographic dimension complete: {len(final_df)} total records")
        
        return final_df
    
    def _process_shapefile(self, shapefile_path: Path, level: str) -> pl.DataFrame:
        """Process a single shapefile into standardized format.
        
        Args:
            shapefile_path: Path to the shapefile
            level: Geographic level (e.g., 'sa1', 'sa2')
            
        Returns:
            Polars DataFrame with processed geographic data
        """
        # Load shapefile using GeoPandas
        gdf = gpd.read_file(shapefile_path)
        
        # Get column mapping for this level
        mapping = self.column_mappings.get(level, {})
        source_cols = mapping.get("source_columns", {})
        
        # Extract required columns
        records = []
        
        for idx, row in gdf.iterrows():
            # Helper function to get column value from multiple possible names
            def get_column_value(row, col_names):
                if isinstance(col_names, list):
                    for col in col_names:
                        if col in row.index and row[col] is not None:
                            return str(row[col])
                else:
                    if col_names in row.index:
                        return str(row[col_names])
                return ""
            
            # Get geographic code and name
            geo_code = get_column_value(row, source_cols.get("code", [f"{level.upper()}_CODE_2021", f"{level.upper()}_CODE_2"]))
            geo_name = get_column_value(row, source_cols.get("name", [f"{level.upper()}_NAME_2021", f"{level.upper()}_NAME_2"]))
            
            # Skip invalid records
            if not geo_code or geo_code == "nan":
                self.logger.warning(f"Skipping record with invalid code at index {idx}")
                continue
            
            # Get state information (if available)
            state_code = None
            state_name = None
            if "state_code" in source_cols:
                state_code = get_column_value(row, source_cols["state_code"])
                if state_code == "nan" or not state_code:
                    state_code = None
            if "state_name" in source_cols:
                state_name = get_column_value(row, source_cols["state_name"])
                if state_name == "nan" or not state_name:
                    state_name = None
            
            # Process geometry
            geom_data = self._process_geometry(row.geometry)
            
            # Create record
            record = {
                "geo_sk": self.geo_sk_counter,
                "geo_id": geo_code,
                "geo_level": level.upper(),
                "geo_name": geo_name,
                "state_code": state_code,
                "state_name": state_name,
                "latitude": geom_data["latitude"],
                "longitude": geom_data["longitude"],
                "geom": geom_data["wkt"],
                "parent_geo_sk": None,  # Will be set later
                "is_unknown": False,
                "etl_processed_at": datetime.now()
            }
            
            records.append(record)
            self.geo_sk_counter += 1
        
        # Convert to Polars DataFrame
        return pl.DataFrame(records)
    
    def _process_geometry(self, geometry) -> Dict[str, any]:
        """Process a shapely geometry object.
        
        Args:
            geometry: Shapely geometry object
            
        Returns:
            Dictionary with latitude, longitude, and WKT
        """
        if geometry is None or geometry.is_empty:
            return {"latitude": None, "longitude": None, "wkt": None}
        
        try:
            # Validate and fix geometry if needed
            if not geometry.is_valid:
                geometry = make_valid(geometry)
            
            # Calculate centroid
            centroid = geometry.centroid
            
            # Get coordinates
            lat = round(centroid.y, 6) if centroid else None
            lon = round(centroid.x, 6) if centroid else None
            
            # Convert to WKT
            wkt_str = geometry.wkt if geometry else None
            
            return {
                "latitude": lat,
                "longitude": lon,
                "wkt": wkt_str
            }
            
        except Exception as e:
            self.logger.warning(f"Error processing geometry: {e}")
            return {"latitude": None, "longitude": None, "wkt": None}
    
    def _create_unknown_member(self) -> pl.DataFrame:
        """Create the unknown member record for the geographic dimension.
        
        Returns:
            Polars DataFrame with one unknown member record
        """
        unknown_record = {
            "geo_sk": self.settings.unknown_geo_sk,
            "geo_id": "UNKNOWN",
            "geo_level": "UNKNOWN",
            "geo_name": "Unknown Geographic Area",
            "state_code": None,
            "state_name": None,
            "latitude": None,
            "longitude": None,
            "geom": None,
            "parent_geo_sk": None,
            "is_unknown": True,
            "etl_processed_at": datetime.now()
        }
        
        return pl.DataFrame([unknown_record])
    
    def _add_parent_relationships(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add parent geographic relationships based on hierarchy.
        
        Geographic hierarchy (top to bottom):
        - STE (State/Territory)
        - SA4
        - SA3
        - SA2
        - SA1
        - POA (Postal Area - separate hierarchy)
        
        Args:
            df: DataFrame with all geographic records
            
        Returns:
            DataFrame with parent_geo_sk populated
        """
        self.logger.info("Adding parent geographic relationships")
        
        # Create lookup dictionaries for each level
        lookups = {}
        for level in ["ste", "sa4", "sa3", "sa2"]:
            level_df = df.filter(pl.col("geo_level") == level.upper())
            lookups[level] = dict(zip(
                level_df["geo_id"].to_list(),
                level_df["geo_sk"].to_list()
            ))
        
        # Define parent relationships
        # For SA1 -> SA2, SA2 -> SA3, etc., we need the actual mapping
        # This is a simplified version - in production you'd have proper mapping tables
        
        # For now, set all parent_geo_sk to None except unknown records
        # In a real implementation, you would have mapping tables or derive from codes
        
        return df
    
    def _enforce_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Enforce the target schema on the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with enforced schema
        """
        # Get expected columns from schema
        expected_columns = [col["name"] for col in self.schema["columns"]]
        
        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        
        # Select and reorder columns
        df = df.select(expected_columns)
        
        # Cast to correct types
        type_mapping = {
            "Int64": pl.Int64,
            "Utf8": pl.Utf8,
            "Float64": pl.Float64,
            "Boolean": pl.Boolean,
            "Datetime": pl.Datetime,
            "Categorical": pl.Categorical
        }
        
        cast_expressions = []
        for col_def in self.schema["columns"]:
            col_name = col_def["name"]
            col_type = col_def["dtype"]
            nullable = col_def.get("nullable", True)
            
            if col_type in type_mapping:
                if col_type == "Categorical":
                    # First cast to string, then to categorical
                    expr = pl.col(col_name).cast(pl.Utf8, strict=False)
                    if not nullable:
                        expr = expr.fill_null("")
                    cast_expressions.append(expr.cast(pl.Categorical))
                else:
                    cast_expressions.append(
                        pl.col(col_name).cast(type_mapping[col_type], strict=False)
                    )
            else:
                cast_expressions.append(pl.col(col_name))
        
        df = df.select(cast_expressions)
        
        return df
    
    def save_to_parquet(self, df: pl.DataFrame, output_path: Optional[Path] = None) -> Path:
        """Save the geographic dimension to Parquet format.
        
        Args:
            df: DataFrame to save
            output_path: Optional custom output path
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = self.settings.output_dir / "geo_dimension.parquet"
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to Parquet
        df.write_parquet(output_path, compression="snappy")
        
        self.logger.info(f"Saved geographic dimension to {output_path}")
        
        return output_path