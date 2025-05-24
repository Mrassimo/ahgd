#!/usr/bin/env python3
"""Unified ETL runner for AHGD pipeline - works with real or mock data."""

import click
import sys
from pathlib import Path
from datetime import datetime
import polars as pl

from ahgd_etl.utils import setup_logging
from ahgd_etl.config import get_settings
from ahgd_etl.extractors import DataDownloader
from ahgd_etl.transformers.geo import GeographyTransformer
from ahgd_etl.models import (
    TimeDimensionBuilder,
    HealthConditionDimensionBuilder,
    DemographicDimensionBuilder,
    PersonCharacteristicDimensionBuilder
)


class ETLPipeline:
    """Main ETL pipeline orchestrator."""
    
    def __init__(self, force_download=False):
        """Initialize the pipeline."""
        self.settings = get_settings()
        self.logger = setup_logging("unified_etl")
        self.force_download = force_download
        self.start_time = datetime.now()
        
    def run_download(self):
        """Download ABS data files."""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Data Download")
        self.logger.info("=" * 60)
        
        downloader = DataDownloader(force_download=self.force_download)
        
        # Check if we need to download
        verification = downloader.verify_downloads()
        missing = [k for k, v in verification.items() if not v]
        
        if not missing:
            self.logger.info("✅ All required data files already present")
            return True
        
        self.logger.warning(f"Missing files: {missing}")
        self.logger.info("\n" + "!" * 60)
        self.logger.info("! MANUAL DOWNLOAD REQUIRED")
        self.logger.info("!" * 60)
        self.logger.info("\nThe ABS data files require authentication to download.")
        self.logger.info("\nFor GEOGRAPHIC data:")
        self.logger.info("1. Visit: https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files")
        self.logger.info("2. Download the following ZIP files:")
        self.logger.info("   - Statistical Area Level 1 (SA1) ASGS Ed 3 2021")
        self.logger.info("   - Statistical Area Level 2 (SA2) ASGS Ed 3 2021")
        self.logger.info("   - Statistical Area Level 3 (SA3) ASGS Ed 3 2021")
        self.logger.info("   - Statistical Area Level 4 (SA4) ASGS Ed 3 2021")
        self.logger.info("   - State and Territory (STE) ASGS Ed 3 2021")
        self.logger.info("3. Save to: data/raw/geographic/")
        
        self.logger.info("\nFor CENSUS data:")
        self.logger.info("1. Visit: https://www.abs.gov.au/census/find-census-data/datapacks")
        self.logger.info("2. Download: '2021 Census GCP All Geographies for AUS'")
        self.logger.info("3. Save to: data/raw/census/")
        self.logger.info("4. Extract the ZIP file")
        
        self.logger.info("\nThen run this script again.")
        
        # Check if mock data exists as alternative
        mock_geo = list(Path("data/raw/geographic").glob("*/mock_*.shp"))
        mock_census = list(Path("data/raw/census/extracted").glob("*/2021Census_*.csv"))
        
        if mock_geo and mock_census:
            self.logger.info("\n✅ Mock data detected! Continuing with mock data...")
            return True
        else:
            self.logger.info("\n💡 TIP: Run 'python create_mock_data.py' to use mock data for testing")
            return False
    
    def run_geographic_dimension(self):
        """Process geographic dimension."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 2: Geographic Dimension")
        self.logger.info("=" * 60)
        
        transformer = GeographyTransformer()
        geo_df = transformer.transform_all()
        output_path = transformer.save_to_parquet(geo_df)
        
        self.logger.info(f"✅ Geographic dimension complete: {len(geo_df)} records")
        self.logger.info(f"   Saved to: {output_path}")
        
        return geo_df
    
    def run_time_dimension(self):
        """Process time dimension."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 3: Time Dimension")
        self.logger.info("=" * 60)
        
        builder = TimeDimensionBuilder()
        time_df = builder.build()
        output_path = builder.save_to_parquet(time_df)
        
        self.logger.info(f"✅ Time dimension complete: {len(time_df)} records")
        self.logger.info(f"   Saved to: {output_path}")
        
        return time_df
    
    def run_core_dimensions(self):
        """Process core dimensions."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 4: Core Dimensions")
        self.logger.info("=" * 60)
        
        results = {}
        
        # Health condition dimension
        self.logger.info("\nProcessing health condition dimension...")
        health_builder = HealthConditionDimensionBuilder()
        health_df = health_builder.build()
        health_path = health_builder.save_to_parquet(health_df, "dim_health_condition.parquet")
        results['health_condition'] = health_df
        self.logger.info(f"✅ Health conditions: {len(health_df)} records")
        
        # Demographic dimension
        self.logger.info("\nProcessing demographic dimension...")
        demo_builder = DemographicDimensionBuilder()
        demo_df = demo_builder.build()
        demo_path = demo_builder.save_to_parquet(demo_df, "dim_demographic.parquet")
        results['demographic'] = demo_df
        self.logger.info(f"✅ Demographics: {len(demo_df)} records")
        
        # Person characteristic dimension
        self.logger.info("\nProcessing person characteristic dimension...")
        char_builder = PersonCharacteristicDimensionBuilder()
        char_df = char_builder.build()
        char_path = char_builder.save_to_parquet(char_df, "dim_person_characteristic.parquet")
        results['person_characteristic'] = char_df
        self.logger.info(f"✅ Person characteristics: {len(char_df)} records")
        
        return results
    
    def run_census_facts(self):
        """Process census fact tables."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 5: Census Fact Tables")
        self.logger.info("=" * 60)
        
        # Check if census data exists
        census_dir = Path("data/raw/census/extracted")
        if not census_dir.exists() or not list(census_dir.glob("*/*.csv")):
            self.logger.warning("No census data found. Skipping fact tables.")
            self.logger.info("Run with real census data to generate fact tables.")
            return {}
        
        # TODO: Implement fact table transformers
        self.logger.info("⚠️  Fact table transformers not yet implemented")
        self.logger.info("   Dimensions are ready for fact table processing")
        
        return {}
    
    def run_validation(self):
        """Run data quality validation."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STEP 6: Data Validation")
        self.logger.info("=" * 60)
        
        # Basic validation of output files
        output_dir = Path("output")
        parquet_files = list(output_dir.glob("*.parquet"))
        
        self.logger.info(f"Found {len(parquet_files)} output files:")
        
        total_size = 0
        for file in parquet_files:
            size = file.stat().st_size
            total_size += size
            
            # Read and get basic stats
            df = pl.read_parquet(file)
            self.logger.info(f"\n  📊 {file.name}:")
            self.logger.info(f"     Records: {len(df):,}")
            self.logger.info(f"     Columns: {len(df.columns)}")
            self.logger.info(f"     Size: {size / 1024 / 1024:.2f} MB")
            
            # Check for nulls in key columns
            if 'is_unknown' in df.columns:
                unknown_count = df.filter(pl.col('is_unknown')).height
                self.logger.info(f"     Unknown members: {unknown_count}")
        
        self.logger.info(f"\n📦 Total output size: {total_size / 1024 / 1024:.2f} MB")
        
        return True
    
    def print_summary(self):
        """Print pipeline execution summary."""
        duration = datetime.now() - self.start_time
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"Execution time: {duration}")
        self.logger.info(f"Output directory: {self.settings.output_dir}")
        
        # List all outputs
        output_files = sorted(Path("output").glob("*.parquet"))
        if output_files:
            self.logger.info("\nGenerated files:")
            for f in output_files:
                self.logger.info(f"  ✅ {f.name}")
        
        self.logger.info("\n🎉 ETL Pipeline completed successfully!")


@click.command()
@click.option('--steps', default='all', help='Steps to run (all, download, geo, time, dimensions, facts, validate)')
@click.option('--force-download', is_flag=True, help='Force re-download of data files')
@click.option('--stop-on-error', is_flag=True, help='Stop pipeline on first error')
def main(steps, force_download, stop_on_error):
    """Run the AHGD ETL pipeline."""
    pipeline = ETLPipeline(force_download=force_download)
    
    # Determine which steps to run
    all_steps = ['download', 'geo', 'time', 'dimensions', 'facts', 'validate']
    
    if steps == 'all':
        steps_to_run = all_steps
    else:
        steps_to_run = [s.strip() for s in steps.split(',')]
    
    pipeline.logger.info("Australian Healthcare Geographic Database ETL Pipeline")
    pipeline.logger.info("=" * 60)
    pipeline.logger.info(f"Steps to run: {', '.join(steps_to_run)}")
    pipeline.logger.info(f"Force download: {force_download}")
    pipeline.logger.info(f"Stop on error: {stop_on_error}")
    
    try:
        # Run each step
        if 'download' in steps_to_run:
            if not pipeline.run_download():
                if stop_on_error:
                    pipeline.logger.error("Download step failed. Exiting.")
                    sys.exit(1)
        
        if 'geo' in steps_to_run:
            pipeline.run_geographic_dimension()
        
        if 'time' in steps_to_run:
            pipeline.run_time_dimension()
        
        if 'dimensions' in steps_to_run:
            pipeline.run_core_dimensions()
        
        if 'facts' in steps_to_run:
            pipeline.run_census_facts()
        
        if 'validate' in steps_to_run:
            pipeline.run_validation()
        
        # Print summary
        pipeline.print_summary()
        
    except Exception as e:
        pipeline.logger.error(f"Pipeline failed: {e}")
        if stop_on_error:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()