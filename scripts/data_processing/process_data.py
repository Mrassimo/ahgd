#!/usr/bin/env python3
"""
Australian Health Data Analytics - Data Processing Pipeline

This script creates the foundational data processing pipeline for Australian health data analytics:
1. Downloads Census 2021 SA2 demographic data and SEIFA 2021 indexes
2. Performs data quality checks and processing using Polars
3. Sets up DuckDB workspace for analytics
4. Creates initial geographic visualizations

Author: Generated for AHGD Project
Date: 2025-06-17
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple
import zipfile
import tempfile
import os
import sys

# Add src to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_global_config, setup_logging

import polars as pl
import pandas as pd
import duckdb
import geopandas as gpd
import folium
import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.logging import RichHandler

# Get configuration
config = get_global_config()

# Setup logging with Rich
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

# Project paths from configuration
PROJECT_ROOT = config.data_source.raw_data_dir.parent.parent  # Go up from data/raw to project root
DATA_RAW = config.data_source.raw_data_dir
DATA_PROCESSED = config.data_source.processed_data_dir
DOCS_DIR = PROJECT_ROOT / "docs"

# Data URLs from verified sources
DATA_SOURCES = {
    "seifa_2021": {
        "url": "https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/2021/Statistical%20Area%20Level%202%2C%20Indexes%2C%20SEIFA%202021.xlsx",
        "filename": "seifa_2021_sa2.xlsx",
        "description": "SEIFA 2021 indexes by Statistical Area Level 2"
    },
    "sa2_boundaries_gda2020": {
        "url": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip",
        "filename": "sa2_2021_boundaries_gda2020.zip",
        "description": "SA2 2021 Digital Boundaries - GDA2020"
    }
}


class DataProcessor:
    """Main data processing class for Australian health analytics."""
    
    def __init__(self):
        self.db_path = PROJECT_ROOT / "health_analytics.db"
        self.conn = None
        self.stats = {
            "download_time": 0,
            "processing_time": 0,
            "sa2_count": 0,
            "seifa_records": 0,
            "data_quality_issues": []
        }
    
    async def download_data(self) -> Dict[str, Path]:
        """Download all required datasets asynchronously."""
        console.print("\n[bold blue]🌏 Downloading Australian Bureau of Statistics data...[/bold blue]")
        
        start_time = time.time()
        downloaded_files = {}
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            tasks = []
            for key, source in DATA_SOURCES.items():
                tasks.append(self._download_file(client, key, source))
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                download_task = progress.add_task("Downloading datasets...", total=len(tasks))
                
                results = await asyncio.gather(*tasks)
                for key, filepath in results:
                    downloaded_files[key] = filepath
                    progress.update(download_task, advance=1)
        
        self.stats["download_time"] = time.time() - start_time
        console.print(f"✅ Downloads completed in {self.stats['download_time']:.1f} seconds")
        
        return downloaded_files
    
    async def _download_file(self, client: httpx.AsyncClient, key: str, source: Dict) -> Tuple[str, Path]:
        """Download a single file."""
        filepath = DATA_RAW / source["filename"]
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath.name}")
            return key, filepath
        
        logger.info(f"Downloading {source['description']}...")
        
        try:
            response = await client.get(source["url"], follow_redirects=True)
            response.raise_for_status()
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            logger.info(f"✅ Downloaded: {filepath.name} ({len(response.content) / 1024 / 1024:.1f} MB)")
            return key, filepath
            
        except Exception as e:
            logger.error(f"❌ Failed to download {source['description']}: {e}")
            raise
    
    def process_seifa_data(self, filepath: Path) -> pl.DataFrame:
        """Process SEIFA 2021 data using Polars."""
        console.print("\n[bold green]📊 Processing SEIFA 2021 data with Polars...[/bold green]")
        
        try:
            # Read Excel file using pandas first (better Excel handling)
            seifa_pandas = pd.read_excel(
                filepath,
                sheet_name="Table 2",  # Table 2 contains SA2 level IRSD data
                skiprows=5  # Skip header rows
            )
            
            # Clean the data - remove rows with copyright text or other non-data content
            sa2_code_col = None
            for col in seifa_pandas.columns:
                if "SA2" in str(col) and "Code" in str(col):
                    sa2_code_col = col
                    break
            
            if sa2_code_col:
                # Filter to only rows with valid SA2 codes (9-digit numbers)
                seifa_pandas = seifa_pandas[
                    seifa_pandas[sa2_code_col].astype(str).str.match(r'^\d{9}$', na=False)
                ]
            
            # Convert to Polars
            seifa_df = pl.from_pandas(seifa_pandas)
            
            # Data quality checks
            initial_rows = len(seifa_df)
            console.print(f"📊 Initial data shape: {seifa_df.shape}")
            console.print(f"📋 Columns: {seifa_df.columns[:5]}...")  # Show first 5 columns
            
            # Map actual column names to standard names
            column_mapping = {}
            for col in seifa_df.columns:
                col_str = str(col).strip()
                if "SA2" in col_str and "Code" in col_str:
                    column_mapping[col] = "SA2_Code_2021"
                elif "SA2" in col_str and "Name" in col_str:
                    column_mapping[col] = "SA2_Name_2021"
                elif col_str == "State":
                    column_mapping[col] = "State_Name_2021"
                elif col_str == "Score":
                    column_mapping[col] = "IRSD_Score"
                elif col_str == "Rank":
                    column_mapping[col] = "IRSD_Rank_Australia"
                elif col_str == "Decile":
                    column_mapping[col] = "IRSD_Decile_Australia"
                elif col_str == "Percentile":
                    column_mapping[col] = "IRSD_Percentile_Australia"
                elif col_str == "Usual Resident Population":
                    column_mapping[col] = "Population"
            
            # Rename columns
            seifa_df = seifa_df.rename(column_mapping)
            
            # Remove any rows where SA2_Code_2021 is null or invalid
            if "SA2_Code_2021" in seifa_df.columns:
                seifa_df = seifa_df.filter(
                    pl.col("SA2_Code_2021").is_not_null()
                )
                
                # Convert SA2 codes to string format
                seifa_df = seifa_df.with_columns([
                    pl.col("SA2_Code_2021").cast(pl.Utf8),
                ])
                
                # Add SA2_Name_2021 if it exists
                if "SA2_Name_2021" in seifa_df.columns:
                    seifa_df = seifa_df.with_columns([
                        pl.col("SA2_Name_2021").cast(pl.Utf8).fill_null("Unknown")
                    ])
                
                # Select relevant columns
                key_columns = ["SA2_Code_2021"]
                if "SA2_Name_2021" in seifa_df.columns:
                    key_columns.append("SA2_Name_2021")
                if "State_Name_2021" in seifa_df.columns:
                    key_columns.append("State_Name_2021")
                if "Population" in seifa_df.columns:
                    key_columns.append("Population")
                
                # Add SEIFA columns
                seifa_columns = [col for col in seifa_df.columns if col.startswith("IRSD_")]
                
                available_columns = [col for col in key_columns + seifa_columns if col in seifa_df.columns]
                seifa_df = seifa_df.select(available_columns)
                
                # Data quality reporting
                final_rows = len(seifa_df)
                null_sa2_codes = seifa_df.filter(pl.col("SA2_Code_2021").is_null()).height
                
                if null_sa2_codes > 0:
                    self.stats["data_quality_issues"].append(f"Found {null_sa2_codes} records with null SA2 codes")
                
                self.stats["seifa_records"] = final_rows
                
                console.print(f"✅ Processed SEIFA data: {final_rows} SA2 areas (filtered from {initial_rows} total rows)")
                console.print(f"📊 Selected columns: {seifa_df.columns}")
                
                return seifa_df
            else:
                raise ValueError("Could not find SA2 Code column in the data")
            
        except Exception as e:
            logger.error(f"❌ Failed to process SEIFA data: {e}")
            raise
    
    def process_boundaries(self, zip_filepath: Path) -> gpd.GeoDataFrame:
        """Process SA2 boundary shapefiles."""
        console.print("\n[bold green]🗺️ Processing SA2 boundary shapefiles...[/bold green]")
        
        try:
            # Extract shapefile from zip
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find the shapefile
                shp_files = list(Path(temp_dir).rglob("*.shp"))
                if not shp_files:
                    raise FileNotFoundError("No shapefile found in the zip archive")
                
                shp_file = shp_files[0]
                
                # Read with geopandas
                boundaries_gdf = gpd.read_file(shp_file)
                
                # Basic data quality checks
                initial_count = len(boundaries_gdf)
                
                # Clean up SA2 codes and names
                boundaries_gdf['SA2_CODE21'] = boundaries_gdf['SA2_CODE21'].astype(str)
                boundaries_gdf['SA2_NAME21'] = boundaries_gdf['SA2_NAME21'].astype(str)
                
                # Remove invalid geometries
                valid_geom_mask = boundaries_gdf.geometry.is_valid
                invalid_count = (~valid_geom_mask).sum()
                
                if invalid_count > 0:
                    console.print(f"⚠️ Found {invalid_count} invalid geometries, fixing...")
                    boundaries_gdf.loc[~valid_geom_mask, 'geometry'] = boundaries_gdf.loc[~valid_geom_mask, 'geometry'].buffer(0)
                    self.stats["data_quality_issues"].append(f"Fixed {invalid_count} invalid geometries")
                
                # Filter out any null geometries
                boundaries_gdf = boundaries_gdf[boundaries_gdf.geometry.notna()]
                
                final_count = len(boundaries_gdf)
                self.stats["sa2_count"] = final_count
                
                console.print(f"✅ Processed SA2 boundaries: {final_count} areas")
                if final_count != initial_count:
                    console.print(f"⚠️ Filtered out {initial_count - final_count} invalid areas")
                
                return boundaries_gdf
                
        except Exception as e:
            logger.error(f"❌ Failed to process boundary data: {e}")
            raise
    
    def setup_duckdb(self) -> duckdb.DuckDBPyConnection:
        """Set up DuckDB workspace with spatial extensions."""
        console.print("\n[bold cyan]🦆 Setting up DuckDB workspace...[/bold cyan]")
        
        try:
            # Create DuckDB connection
            conn = duckdb.connect(str(self.db_path))
            
            # Install and load spatial extension
            try:
                conn.execute("INSTALL spatial")
                conn.execute("LOAD spatial")
                console.print("✅ Spatial extension loaded successfully")
            except Exception as e:
                console.print(f"⚠️ Could not load spatial extension: {e}")
                console.print("Continuing without spatial features...")
            
            self.conn = conn
            return conn
            
        except Exception as e:
            logger.error(f"❌ Failed to set up DuckDB: {e}")
            raise
    
    def load_data_to_duckdb(self, seifa_df: pl.DataFrame, boundaries_gdf: gpd.GeoDataFrame):
        """Load processed data into DuckDB tables."""
        console.print("\n[bold cyan]📥 Loading data into DuckDB...[/bold cyan]")
        
        try:
            # Save processed data as Parquet for efficient loading
            seifa_path = DATA_PROCESSED / "seifa_2021_sa2.parquet"
            seifa_df.write_parquet(seifa_path)
            
            boundaries_path = DATA_PROCESSED / "sa2_boundaries_2021.parquet"
            boundaries_gdf.to_parquet(boundaries_path)
            
            # Load into DuckDB
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE seifa_2021 AS 
                SELECT * FROM read_parquet('{seifa_path}')
            """)
            
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE sa2_boundaries AS 
                SELECT * FROM read_parquet('{boundaries_path}')
            """)
            
            # Create a combined analysis table
            self.conn.execute("""
                CREATE OR REPLACE TABLE sa2_analysis AS
                SELECT 
                    b.SA2_CODE21,
                    b.SA2_NAME21,
                    b.STE_NAME21 as state_name,
                    b.AREASQKM21 as area_sqkm,
                    s.*
                FROM sa2_boundaries b
                LEFT JOIN seifa_2021 s ON b.SA2_CODE21 = s.SA2_Code_2021
            """)
            
            # Validate data loading
            seifa_count = self.conn.execute("SELECT COUNT(*) FROM seifa_2021").fetchone()[0]
            boundaries_count = self.conn.execute("SELECT COUNT(*) FROM sa2_boundaries").fetchone()[0]
            analysis_count = self.conn.execute("SELECT COUNT(*) FROM sa2_analysis").fetchone()[0]
            
            console.print(f"✅ Loaded data into DuckDB:")
            console.print(f"   - SEIFA records: {seifa_count}")
            console.print(f"   - SA2 boundaries: {boundaries_count}")
            console.print(f"   - Combined analysis table: {analysis_count}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load data into DuckDB: {e}")
            raise
    
    def create_initial_visualization(self, boundaries_gdf: gpd.GeoDataFrame, seifa_df: pl.DataFrame):
        """Create an initial choropleth map visualization."""
        console.print("\n[bold magenta]🎨 Creating initial visualization...[/bold magenta]")
        
        try:
            # Convert to regular pandas for easier integration with folium
            boundaries_df = boundaries_gdf.to_crs('EPSG:4326')  # Ensure WGS84 for web mapping
            
            # Merge with SEIFA data
            seifa_pandas = seifa_df.to_pandas()
            
            # Find a suitable SEIFA index column for visualization
            seifa_cols = [col for col in seifa_pandas.columns if 'IRSD_Score' in col or 'IRSD_Decile' in col]
            if not seifa_cols:
                seifa_cols = [col for col in seifa_pandas.columns if any(x in col.upper() for x in ['SCORE', 'DECILE', 'RANK'])]
            
            viz_column = seifa_cols[0] if seifa_cols else None
            
            if viz_column:
                # Merge data
                merge_columns = ['SA2_Code_2021', viz_column]
                viz_data = boundaries_df.merge(
                    seifa_pandas[merge_columns], 
                    left_on='SA2_CODE21', 
                    right_on='SA2_Code_2021',
                    how='left'
                )
                
                # Create folium map centred on Australia
                m = folium.Map(
                    location=[-25.2744, 133.7751],  # Centre of Australia
                    zoom_start=6,
                    tiles='OpenStreetMap'
                )
                
                # Add choropleth layer
                if viz_column in viz_data.columns and not viz_data[viz_column].isna().all():
                    folium.Choropleth(
                        geo_data=viz_data,
                        name='SEIFA Index',
                        data=viz_data,
                        columns=['SA2_CODE21', viz_column],
                        key_on='feature.properties.SA2_CODE21',
                        fill_color='RdYlBu',
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name=f'SEIFA {viz_column}',
                        bins=9
                    ).add_to(m)
                
                # Add layer control
                folium.LayerControl().add_to(m)
                
                # Save map
                map_path = DOCS_DIR / "initial_map.html"
                m.save(str(map_path))
                
                console.print(f"✅ Created visualization: {map_path}")
                
            else:
                console.print("⚠️ No suitable SEIFA columns found for visualization")
                
        except Exception as e:
            logger.error(f"❌ Failed to create visualization: {e}")
            # Create a simple boundaries-only map as fallback
            try:
                boundaries_simple = boundaries_gdf.to_crs('EPSG:4326')
                
                m = folium.Map(
                    location=[-25.2744, 133.7751],
                    zoom_start=6,
                    tiles='OpenStreetMap'
                )
                
                # Add just the boundaries
                folium.GeoJson(
                    boundaries_simple.iloc[:100],  # Limit to first 100 for performance
                    style_function=lambda x: {
                        'fillColor': 'lightblue',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.5,
                    }
                ).add_to(m)
                
                map_path = DOCS_DIR / "initial_map.html"
                m.save(str(map_path))
                
                console.print(f"✅ Created basic boundary map: {map_path}")
                
            except Exception as fallback_error:
                logger.error(f"❌ Fallback visualization also failed: {fallback_error}")
    
    def generate_report(self):
        """Generate a summary report of the data processing."""
        console.print("\n[bold yellow]📋 Data Processing Summary[/bold yellow]")
        
        # Create summary table
        table = Table(title="Australian Health Data Processing Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Download Time", f"{self.stats['download_time']:.1f} seconds")
        table.add_row("Processing Time", f"{self.stats['processing_time']:.1f} seconds")
        table.add_row("SA2 Areas Processed", str(self.stats['sa2_count']))
        table.add_row("SEIFA Records", str(self.stats['seifa_records']))
        
        if self.stats['data_quality_issues']:
            table.add_row("Data Quality Issues", str(len(self.stats['data_quality_issues'])))
        
        console.print(table)
        
        # Print data quality issues if any
        if self.stats['data_quality_issues']:
            console.print("\n[bold red]⚠️ Data Quality Issues Found:[/bold red]")
            for issue in self.stats['data_quality_issues']:
                console.print(f"  • {issue}")
        
        # Database validation queries
        if self.conn:
            console.print("\n[bold cyan]🔍 Database Validation Queries:[/bold cyan]")
            
            try:
                # Basic stats
                result = self.conn.execute("""
                    SELECT 
                        COUNT(*) as total_sa2s,
                        COUNT(DISTINCT STE_NAME21) as states_territories,
                        AVG(AREASQKM21) as avg_area_sqkm
                    FROM sa2_boundaries
                """).fetchone()
                
                console.print(f"📊 Geographic Coverage:")
                console.print(f"   • Total SA2s: {result[0]}")
                console.print(f"   • States/Territories: {result[1]}")
                console.print(f"   • Average Area: {result[2]:.1f} km²")
                
            except Exception as e:
                console.print(f"❌ Could not run validation queries: {e}")


async def main():
    """Main processing function."""
    console.print("[bold blue]🇦🇺 Australian Health Data Analytics - Data Processing Pipeline[/bold blue]")
    console.print("Creating foundational data processing pipeline for health analytics...\n")
    
    processor = DataProcessor()
    start_time = time.time()
    
    try:
        # Download data
        downloaded_files = await processor.download_data()
        
        # Set up DuckDB
        processor.setup_duckdb()
        
        # Process SEIFA data
        if "seifa_2021" in downloaded_files:
            seifa_df = processor.process_seifa_data(downloaded_files["seifa_2021"])
        else:
            raise FileNotFoundError("SEIFA data not downloaded successfully")
        
        # Process boundaries
        if "sa2_boundaries_gda2020" in downloaded_files:
            boundaries_gdf = processor.process_boundaries(downloaded_files["sa2_boundaries_gda2020"])
        else:
            raise FileNotFoundError("SA2 boundaries not downloaded successfully")
        
        # Load into DuckDB
        processor.load_data_to_duckdb(seifa_df, boundaries_gdf)
        
        # Create visualization
        processor.create_initial_visualization(boundaries_gdf, seifa_df)
        
        # Update processing time
        processor.stats["processing_time"] = time.time() - start_time
        
        # Generate final report
        processor.generate_report()
        
        console.print(f"\n[bold green]🎉 Pipeline completed successfully in {processor.stats['processing_time']:.1f} seconds![/bold green]")
        console.print(f"📁 Database created: {processor.db_path}")
        console.print(f"🗺️ Visualization saved: {DOCS_DIR / 'initial_map.html'}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise
    
    finally:
        if processor.conn:
            processor.conn.close()


if __name__ == "__main__":
    asyncio.run(main())