#!/usr/bin/env python3
"""
Main entry point and orchestrator for the AHGD ETL pipeline.

This script handles command-line arguments to control the ETL process,
including downloading data, processing geographic boundaries, and processing
Census data.

Usage examples:
  python run_etl.py --steps all          # Run all steps
  python run_etl.py --steps download geo # Run download and geography steps
  python run_etl.py --steps census       # Run only the census step
  python run_etl.py --steps time         # Run only the time dimension creation step
"""

import argparse
import logging
import sys
from pathlib import Path
import polars as pl
from typing import Optional
import time
import os
import traceback
import re
from datetime import date, datetime
import zipfile
import shutil

# Ensure etl_logic is importable
# Assuming run_etl.py is at the project root, and etl_logic is a subdirectory
try:
    from etl_logic import config, utils, geography, census, time_dimension, dimensions
except ImportError:
    # If run_etl.py is moved or the structure changes, this might need adjustment
    # Or, consider installing etl_logic as a package (`pip install .`)
    print("Error: Could not import etl_logic package.")
    print("Ensure run_etl.py is in the project root or etl_logic is installed.")
    sys.exit(1)

# Setup logging (using the utility function)
# Log file will be placed in the LOG_DIR defined in config.PATHS
logger = utils.setup_logging(config.PATHS.get('LOG_DIR'))

def get_time_dimension_sk(date_value):
    """Get the time dimension surrogate key for a given date."""
    try:
        time_dim_path = config.PATHS['OUTPUT_DIR'] / "dim_time.parquet"
        if not os.path.exists(time_dim_path):
            logger.error(f"Time dimension not found at {time_dim_path}")
            return None
            
        # Load time dimension
        time_dim = pl.read_parquet(time_dim_path)
        
        # Filter for the specified date
        census_date_df = time_dim.filter(
            (pl.col("year") == date_value.year) & 
            (pl.col("month") == date_value.month) & 
            (pl.col("day_of_month") == date_value.day)
        )
        
        if len(census_date_df) == 0:
            logger.error(f"Date {date_value} not found in time dimension")
            return None
            
        # Get the time_sk
        time_sk = census_date_df.select("time_sk").item()
        logger.info(f"Using time_sk {time_sk} for date {date_value}")
        return time_sk
        
    except Exception as e:
        logger.error(f"Error getting time dimension SK: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def run_download(force_download: bool):
    """Runs the data download step."""
    logger.info("--- Starting Data Download Step ---")
    success = True

    # Get required zip files and their URLs
    geo_zips = config.get_required_geo_zips()
    census_zips = config.get_required_census_zips()

    # Download geographic data
    logger.info("Downloading Geographic data...")
    geo_download_dir = config.PATHS['GEOGRAPHIC_DIR']
    if not utils.download_data(geo_zips, geo_download_dir, force_download):
        logger.error("Geographic data download failed.")
        success = False
    else:
        logger.info("Geographic data download completed.")

    # Download census data
    logger.info("Downloading Census data...")
    census_download_dir = config.PATHS['CENSUS_DIR']
    if not utils.download_data(census_zips, census_download_dir, force_download):
        logger.error("Census data download failed.")
        success = False
    else:
        logger.info("Census data download completed.")

    if success:
        logger.info("--- Data Download Step Finished Successfully ---")
    else:
        logger.error("--- Data Download Step Finished with Errors ---")
    return success

def run_geographic_processing():
    """Process geographic boundary files."""
    logger.info("=== Starting Geographic Processing Step ===")
    
    from etl_logic import geography
    
    # Get configuration paths
    paths = config.PATHS
    zip_dir = paths['GEOGRAPHIC_DIR']
    temp_extract_base = paths['TEMP_EXTRACT_DIR']
    output_dir = paths['OUTPUT_DIR']
    
    # Process geographic boundaries
    success = geography.process_geography(
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir
    )
    
    if success:
        logger.info("=== Geographic Processing Step Finished Successfully ===")
    else:
        logger.error("=== Geographic Processing Step Finished with Errors ===")
    
    return success

def run_population_processing():
    """Process population data and update geographic dimension with weighted centroids."""
    logger.info("=== Starting Population Processing Step ===")
    
    from etl_logic import census
    from etl_logic import geography
    
    # Get configuration paths
    paths = config.PATHS
    zip_dir = paths['CENSUS_DIR']
    temp_extract_base = paths['TEMP_EXTRACT_DIR']
    output_dir = paths['OUTPUT_DIR']
    
    # Get time dimension surrogate key for Census 2021
    time_sk = get_time_dimension_sk(date(2021, 8, 10))
    
    # Process population data
    success = census.process_census_g01_data(
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=output_dir / "geo_dimension.parquet",
        time_sk=time_sk
    )
    
    if success:
        # Update geographic dimension with population-weighted centroids
        success = geography.update_population_weighted_centroids(
            geo_output_path=output_dir / "geo_dimension.parquet",
            population_fact_path=output_dir / "fact_population.parquet"
        )
    
    if success:
        logger.info("=== Population Processing Step Finished Successfully ===")
    else:
        logger.error("=== Population Processing Step Finished with Errors ===")
    
    return success

def run_census_g01_processing():
    """Runs the census G01 (population) data processing step."""
    logger.info("--- Starting Census G01 Processing Step ---")
    zip_dir = config.PATHS['CENSUS_DIR']
    temp_extract_base = config.PATHS['TEMP_EXTRACT_DIR']
    output_dir = config.PATHS['OUTPUT_DIR']
    geo_output_path = config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet"
    time_dim_path = config.PATHS['OUTPUT_DIR'] / "dim_time.parquet"
    
    # Check if geographic dimension exists first
    if not geo_output_path.exists():
        logger.error(f"Geographic dimension file not found: {geo_output_path}")
        logger.error("Please run the 'geo' step first.")
        return False
    
    # Check if time dimension exists and get a Census year time_sk
    time_sk = None
    if time_dim_path.exists():
        try:
            # Find time_sk for 2021 Census reference date (August 10, 2021)
            time_df = pl.read_parquet(time_dim_path)
            census_date_row = time_df.filter(pl.col('full_date') == pl.lit('2021-08-10').cast(pl.Date))
            if len(census_date_row) > 0:
                time_sk = census_date_row.select('time_sk').item()
                logger.info(f"Using time_sk {time_sk} for Census date 2021-08-10")
            else:
                # Fallback to the first day of 2021
                fallback_row = time_df.filter(pl.col('full_date') == pl.lit('2021-01-01').cast(pl.Date))
                if len(fallback_row) > 0:
                    time_sk = fallback_row.select('time_sk').item()
                    logger.info(f"Using fallback time_sk {time_sk} for 2021-01-01")
        except Exception as e:
            logger.warning(f"Could not read time dimension for time_sk: {e}")
            logger.warning("Proceeding without time dimension integration")
    
    # Ensure temp extract dir exists
    temp_extract_base.mkdir(parents=True, exist_ok=True)
    
    success = census.process_census_g01_data(
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )
    
    if success:
        logger.info("--- Census G01 Processing Step Finished Successfully ---")
    else:
        logger.error("--- Census G01 Processing Step Finished with Errors ---")
    return success

def run_census_g17_processing():
    """Runs the census G17 (income) data processing step."""
    logger.info("--- Starting Census G17 Processing Step ---")
    zip_dir = config.PATHS['CENSUS_DIR']
    temp_extract_base = config.PATHS['TEMP_EXTRACT_DIR']
    output_dir = config.PATHS['OUTPUT_DIR']
    geo_output_path = config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet"
    time_dim_path = config.PATHS['OUTPUT_DIR'] / "dim_time.parquet"
    
    # Check if geographic dimension exists first
    if not geo_output_path.exists():
        logger.error(f"Geographic dimension file not found: {geo_output_path}")
        logger.error("Please run the 'geo' step first.")
        return False
    
    # Check if time dimension exists and get a Census year time_sk
    time_sk = None
    if time_dim_path.exists():
        try:
            # Find time_sk for 2021 Census reference date (August 10, 2021)
            time_df = pl.read_parquet(time_dim_path)
            census_date_row = time_df.filter(pl.col('full_date') == pl.lit('2021-08-10').cast(pl.Date))
            if len(census_date_row) > 0:
                time_sk = census_date_row.select('time_sk').item()
                logger.info(f"Using time_sk {time_sk} for Census date 2021-08-10")
            else:
                # Fallback to the first day of 2021
                fallback_row = time_df.filter(pl.col('full_date') == pl.lit('2021-01-01').cast(pl.Date))
                if len(fallback_row) > 0:
                    time_sk = fallback_row.select('time_sk').item()
                    logger.info(f"Using fallback time_sk {time_sk} for 2021-01-01")
        except Exception as e:
            logger.warning(f"Could not read time dimension for time_sk: {e}")
            logger.warning("Proceeding without time dimension integration")

    # Ensure temp extract dir exists
    temp_extract_base.mkdir(parents=True, exist_ok=True)

    success = census.process_g17_census_data(
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )

    if success:
        logger.info("--- Census G17 Processing Step Finished Successfully ---")
    else:
        logger.error("--- Census G17 Processing Step Finished with Errors ---")
    return success

def run_census_g18_processing():
    """Runs the census G18 (Unpaid Care) data processing step."""
    logger.info("--- Starting Census G18 Processing Step ---")
    zip_dir = config.PATHS['CENSUS_DIR']
    temp_extract_base = config.PATHS['TEMP_EXTRACT_DIR']
    output_dir = config.PATHS['OUTPUT_DIR']
    geo_output_path = config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet"
    time_dim_path = config.PATHS['OUTPUT_DIR'] / "dim_time.parquet"

    # Check if geographic dimension exists first
    if not geo_output_path.exists():
        logger.error(f"Geographic dimension file not found: {geo_output_path}")
        logger.error("Please run the 'geo' step first.")
        return False

    # Check if time dimension exists and get a Census year time_sk
    time_sk = None
    if time_dim_path.exists():
        try:
            # Find time_sk for 2021 Census reference date (August 10, 2021)
            time_df = pl.read_parquet(time_dim_path)
            census_date_row = time_df.filter(pl.col('full_date') == pl.lit('2021-08-10').cast(pl.Date))
            if len(census_date_row) > 0:
                time_sk = census_date_row.select('time_sk').item()
                logger.info(f"Using time_sk {time_sk} for Census date 2021-08-10")
            else:
                # Fallback to the first day of 2021
                fallback_row = time_df.filter(pl.col('full_date') == pl.lit('2021-01-01').cast(pl.Date))
                if len(fallback_row) > 0:
                    time_sk = fallback_row.select('time_sk').item()
                    logger.info(f"Using fallback time_sk {time_sk} for 2021-01-01")
        except Exception as e:
            logger.warning(f"Could not read time dimension for time_sk: {e}")
            logger.warning("Proceeding without time dimension integration")

    # Ensure temp extract dir exists
    temp_extract_base.mkdir(parents=True, exist_ok=True)

    success = census.process_g18_census_data(
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )

    if success:
        logger.info("--- Census G18 Processing Step Finished Successfully ---")
    else:
        logger.error("--- Census G18 Processing Step Finished with Errors ---")
    return success

def run_census_g19_processing():
    """Runs the census G19 (Long-Term Health Conditions) data processing step."""
    logger.info("--- Starting Census G19 Processing Step ---")
    
    from etl_logic import census
    
    # Get configuration paths
    paths = config.PATHS
    zip_dir = paths['CENSUS_DIR']
    temp_extract_base = paths['TEMP_EXTRACT_DIR']
    output_dir = paths['OUTPUT_DIR']
    
    # Check path exists
    if not zip_dir.exists():
        logger.error(f"Census data directory not found: {zip_dir}")
        return False
        
    # Get time dimension surrogate key, if available
    time_sk = get_time_dimension_sk(date(2021, 8, 10))
    
    # Load geographic boundaries
    geo_output_path = output_dir / "geo_dimension.parquet"
    if not geo_output_path.exists():
        logger.error(f"Geographic boundaries file not found: {geo_output_path}")
        return False
    
    logger.info(f"Using geographic boundaries from: {geo_output_path}")
    
    # Process census G19 data
    success = census.process_g19_census_data(
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )
    
    if success:
        logger.info("--- Census G19 Processing Step Finished Successfully ---")
    else:
        logger.error("--- Census G19 Processing Step Finished with Errors ---")
    return success

def run_census_g19_detailed_processing():
    """Runs the detailed census G19 (Specific Health Conditions) data processing step."""
    logger.info("--- Starting Detailed Census G19 Processing Step ---")
    
    from etl_logic import census
    
    # Get configuration paths
    paths = config.PATHS
    zip_dir = paths['CENSUS_DIR']
    temp_extract_base = paths['TEMP_EXTRACT_DIR']
    output_dir = paths['OUTPUT_DIR']
    
    # Check path exists
    if not zip_dir.exists():
        logger.error(f"Census data directory not found: {zip_dir}")
        return False
        
    # Get time dimension surrogate key, if available
    time_sk = get_time_dimension_sk(date(2021, 8, 10))
    
    # Load geographic boundaries
    geo_output_path = output_dir / "geo_dimension.parquet"
    if not geo_output_path.exists():
        logger.error(f"Geographic boundaries file not found: {geo_output_path}")
        return False
    
    logger.info(f"Using geographic boundaries from: {geo_output_path}")
    
    # Process detailed census G19 data (G19A, G19B, G19C)
    success = census.process_g19_detailed_census_data(
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )
    
    if success:
        logger.info("--- Detailed Census G19 Processing Step Finished Successfully ---")
    else:
        logger.error("--- Detailed Census G19 Processing Step Finished with Errors ---")
    return success

def run_census_g20_processing():
    """Runs the census G20 (detailed medical conditions) data processing step."""
    logger.info("--- Starting Census G20 Processing Step ---")
    zip_dir = config.PATHS['CENSUS_DIR']
    temp_extract_base = config.PATHS['TEMP_EXTRACT_DIR']
    output_dir = config.PATHS['OUTPUT_DIR']
    geo_output_path = config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet"
    time_dim_path = config.PATHS['OUTPUT_DIR'] / "dim_time.parquet"

    if not geo_output_path.exists():
        logger.error(f"Geographic dimension file not found: {geo_output_path}. Run 'geo' step first.")
        return False

    time_sk = None
    if time_dim_path.exists():
        try:
            # Find time_sk for 2021 Census reference date (August 10, 2021)
            time_df = pl.read_parquet(time_dim_path)
            census_date_row = time_df.filter(pl.col('full_date') == pl.lit('2021-08-10').cast(pl.Date))
            if len(census_date_row) > 0:
                time_sk = census_date_row.select('time_sk').item()
                logger.info(f"Using time_sk {time_sk} for Census date 2021-08-10")
            else:
                # Fallback to the first day of 2021
                fallback_row = time_df.filter(pl.col('full_date') == pl.lit('2021-01-01').cast(pl.Date))
                if len(fallback_row) > 0:
                    time_sk = fallback_row.select('time_sk').item()
                    logger.info(f"Using fallback time_sk {time_sk} for 2021-01-01")
        except Exception as e:
            logger.warning(f"Could not read time dimension for time_sk: {e}")
            logger.warning("Proceeding without time dimension integration")

    temp_extract_base.mkdir(parents=True, exist_ok=True)

    success = census.process_g20_census_data(
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )

    if success: logger.info("--- Census G20 Processing Step Finished Successfully ---")
    else: logger.error("--- Census G20 Processing Step Finished with Errors ---")
    return success

def run_time_dimension_generation():
    """Runs the time dimension creation step."""
    logger.info("--- Starting Time Dimension Creation Step ---")
    output_dir = config.PATHS['OUTPUT_DIR']
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default years (adjust as needed for project requirements)
    start_year = 2011  # Start at 2011 (2011 Census)
    end_year = 2031    # End at 2031 (covers a few census periods)
    
    success = time_dimension.generate_time_dimension(
        output_dir=output_dir,
        start_year=start_year,
        end_year=end_year
    )
    
    if success:
        logger.info("--- Time Dimension Creation Step Finished Successfully ---")
    else:
        logger.error("--- Time Dimension Creation Step Finished with Errors ---")
    return success

def run_health_condition_dimension_generation():
    """Runs the health condition dimension generation step."""
    logger.info("--- Starting Health Condition Dimension Generation Step ---")
    output_dir = config.PATHS['OUTPUT_DIR']
    
    # Try to use G21 data for comprehensive condition list if available
    g21_path = output_dir / "fact_health_conditions_by_characteristic.parquet"
    g21_data_available = g21_path.exists()
    
    if g21_data_available:
        logger.info(f"G21 data found at {g21_path}, will use for comprehensive condition list")
    else:
        logger.info("G21 data not found, using standard condition list")
        g21_path = None
    
    # Run the generation function
    success = dimensions.generate_health_condition_dimension(
        output_dir=output_dir,
        g21_path=g21_path
    )
    
    if success:
        logger.info("--- Health Condition Dimension Generation Step Finished Successfully ---")
    else:
        logger.error("--- Health Condition Dimension Generation Step Finished with Errors ---")
        
    return success

def run_demographic_dimension_generation():
    """Runs the demographic dimension creation step."""
    logger.info("--- Starting Demographic Dimension Creation Step ---")
    output_dir = config.PATHS['OUTPUT_DIR']
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Look for existing fact_health_conditions_detailed file
    fact_health_conditions_path = output_dir / "fact_health_conditions_detailed.parquet"
    
    # Check if fact table exists
    if not fact_health_conditions_path.exists():
        logger.warning(f"Fact table {fact_health_conditions_path} not found. Will use predefined demographics.")
        fact_health_conditions_path = None
    
    success = dimensions.create_demographic_dimension(
        output_dir=output_dir,
        source_data_path=fact_health_conditions_path
    )
    
    if success is not None:
        logger.info("--- Demographic Dimension Creation Step Finished Successfully ---")
    else:
        logger.error("--- Demographic Dimension Creation Step Finished with Errors ---")
    return success is not None

def run_refined_g20_processing():
    """Process G20 health conditions data using streaming."""
    logger.info("=== Starting Refined G20 Processing Step (Streaming) ===")
    start_time = time.time()

    try:
        # Define paths
        # Ensure this staging path matches where process_g20_census_data saves its output
        staging_path = config.PATHS['TEMP_DIR'] / "staging_G20_detailed.parquet"
        dim_health_condition_path = config.PATHS['OUTPUT_DIR'] / "dim_health_condition.parquet"
        dim_demographic_path = config.PATHS['OUTPUT_DIR'] / "dim_demographic.parquet"
        refined_output_path = config.PATHS['OUTPUT_DIR'] / "fact_health_conditions_refined.parquet"

        # --- Input Validation ---
        required_files = {
            "G20 Staging Data": staging_path,
            "Health Condition Dimension": dim_health_condition_path,
            "Demographic Dimension": dim_demographic_path
        }
        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(f"{name} at {path}")

        if missing_files:
            for msg in missing_files:
                logger.error(f"Required file not found: {msg}")
            logger.error("Cannot proceed with Refined G20 processing.")
            return False
        # --- End Input Validation ---

        logger.info(f"Reading dimensions: {dim_health_condition_path.name}, {dim_demographic_path.name}")
        # Read dimensions into memory
        dim_health = pl.read_parquet(dim_health_condition_path)
        dim_demo = pl.read_parquet(dim_demographic_path)

        logger.info(f"Scanning staging data: {staging_path.name}")
        # Scan the large staging fact table lazily
        lazy_fact = pl.scan_parquet(staging_path)

        # Log columns before join for debugging
        logger.info(f"Staging G20 columns: {lazy_fact.columns}")
        logger.info(f"Health Dim columns: {dim_health.columns}")
        logger.info(f"Demo Dim columns: {dim_demo.columns}")

        # Get available condition values in the health dimension
        health_condition_values = dim_health.select("condition").unique().to_series().to_list()
        logger.info(f"Available health condition values: {health_condition_values}")
        
        # Get available age_group values in the demographic dimension
        demo_age_groups = dim_demo.select("age_group").unique().to_series().to_list()
        logger.info(f"Available demographic age_group values: {demo_age_groups}")

        # Since we don't have direct condition mappings, we'll use a generic mapping to "has_condition" or "no_condition"
        # Map specific condition_count values to condition values in the dim table
        # The "has_condition" and "no_condition" values must exist in the dim_health_condition table
        
        # Create a new LazyFrame with our mapping
        lazy_fact_with_condition = lazy_fact.with_columns([
            # Map condition_count to condition value expected by dimension table
            pl.when(pl.col("condition_count") == "no_conditions").then(pl.lit("no_condition"))
            .when(pl.col("condition_count") == "conditions_not_stated").then(pl.lit("not_stated"))
            .otherwise(pl.lit("other_condition"))  # Use a catch-all value that exists in dimension
            .alias("condition"),
            
            # Fix the age_range format to match age_group in demographic dimension
            # Convert "0-14" to "0_14", "15-24" to "15_24", etc.
            pl.when(pl.col("age_range") == "0-14").then(pl.lit("0_14"))
            .when(pl.col("age_range") == "15-24").then(pl.lit("15_24"))
            .when(pl.col("age_range") == "25-34").then(pl.lit("25_34"))
            .when(pl.col("age_range") == "35-44").then(pl.lit("35_44"))
            .when(pl.col("age_range") == "45-54").then(pl.lit("45_54"))
            .when(pl.col("age_range") == "55-64").then(pl.lit("55_64"))
            .when(pl.col("age_range") == "65-74").then(pl.lit("65_74"))
            .when(pl.col("age_range") == "75-84").then(pl.lit("75_84"))
            .when(pl.col("age_range") == "85+").then(pl.lit("85_plus"))
            .when(pl.col("age_range") == "total").then(pl.lit("Tot"))
            .otherwise(pl.lit("Tot"))  # Default to "Tot" for anything else
            .alias("age_group")
        ])

        # Prepare dimensions for joining
        dim_health_join = dim_health.select(['condition', 'condition_sk'])
        dim_demo_join = dim_demo.select(['age_group', 'sex', 'demographic_sk'])

        # Log our join columns for debugging
        logger.info(f"Will join on condition values: {dim_health_join['condition'].unique()}")
        logger.info(f"Will join on age_group values: {dim_demo_join['age_group'].unique()}")
        logger.info(f"Will join on sex values: {dim_demo_join['sex'].unique()}")

        # Perform joins lazily
        refined_lazy = (
            lazy_fact_with_condition
            .join(
                dim_health_join.lazy(), # Convert dimension to LazyFrame for join
                on='condition',
                how='inner'  # Only keep rows where condition matches
            )
            .join(
                dim_demo_join.lazy(), # Convert dimension to LazyFrame for join
                on=['age_group', 'sex'],
                how='inner'  # Only keep rows where both age_group and sex match
            )
            .select([ # Select and order the final columns
                pl.col('geo_sk'),
                pl.col('time_sk'),
                pl.col('condition_sk'), # Use surrogate key from join
                pl.col('demographic_sk'), # Use surrogate key from join
                pl.col('count'), # Ensure 'count' exists in the staging file
                pl.lit(datetime.now()).alias('etl_processed_at')
            ])
        )

        logger.info(f"Executing streaming join and writing to {refined_output_path.name}")
        # Execute the lazy plan and stream the result directly to a Parquet file
        refined_lazy.sink_parquet(refined_output_path, compression="snappy")

        # Optional: Log row count after writing
        try:
             output_df = pl.read_parquet(refined_output_path)
             logger.info(f"Successfully wrote {len(output_df):,} rows to {refined_output_path}")
        except Exception as read_e:
             logger.warning(f"Could not read output file to verify row count: {read_e}")
             logger.info(f"Streaming write operation to {refined_output_path} completed without error.")

        end_time = time.time()
        logger.info(f"--- Refined G20 Processing Step Finished Successfully in {end_time - start_time:.2f} seconds ---")
        logger.info(f"=== Refined G20 Completed Successfully ===")
        return True

    except Exception as e:
        logger.error(f"Error in refined G20 processing: {e}")
        logger.error(traceback.format_exc())
        logger.error("=== Refined G20 Failed ===")
        return False

def run_census_g21_processing():
    """Runs the census G21 (Type of Health Condition by Characteristics) data processing step."""
    logger.info("--- Starting Census G21 Processing Step ---")
    zip_dir = config.PATHS['CENSUS_DIR']
    temp_extract_base = config.PATHS['TEMP_EXTRACT_DIR']
    output_dir = config.PATHS['OUTPUT_DIR']
    geo_output_path = config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet"
    time_dim_path = config.PATHS['OUTPUT_DIR'] / "dim_time.parquet"

    # Dependencies: Geo and Time dimensions must exist
    if not geo_output_path.exists():
        logger.error(f"Geographic dimension file not found: {geo_output_path}. Run 'geo' step first.")
        return False
    if not time_dim_path.exists():
        logger.error(f"Time dimension file not found: {time_dim_path}. Run 'time' step first.")
        return False

    # Get time_sk
    time_sk = get_time_dimension_sk(date(2021, 8, 10))
    if time_sk is None:
        logger.error("Could not determine time_sk for Census. Cannot proceed with G21.")
        return False

    # Ensure temp extract dir exists
    temp_extract_base.mkdir(parents=True, exist_ok=True)

    # Call the G21 processing function from census module
    success = census.process_g21_census_data(
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
        # Note: This function will need access to dim_health_condition and dim_demographic
        # if we implement the refinement step within it.
    )

    if success: logger.info("--- Census G21 Processing Step Finished Successfully (Placeholder Logic) ---")
    else: logger.error("--- Census G21 Processing Step Finished with Errors ---")
    return success

def run_person_characteristic_dimension_generation():
    """Runs the person characteristic dimension generation step."""
    logger.info("--- Starting Person Characteristic Dimension Generation Step ---")
    output_dir = config.PATHS['OUTPUT_DIR']
    
    # Get the G21 source data if available
    g21_file = output_dir / "fact_health_conditions_by_characteristic.parquet"
    source_path = g21_file if g21_file.exists() else None
    
    # Create dimension table
    try:
        path = dimensions.create_person_characteristic_dimension(source_path, output_dir)
        logger.info(f"Person characteristic dimension created at: {path}")
        logger.info("--- Person Characteristic Dimension Generation Step Finished Successfully ---")
        return True
    except Exception as e:
        logger.error(f"Failed to create person characteristic dimension: {e}")
        logger.error("--- Person Characteristic Dimension Generation Step Finished with Errors ---")
        return False

def run_refined_g21_processing():
    """Process G21 health conditions by characteristic data using streaming."""
    logger.info("=== Starting Refined G21 Processing Step (Streaming) ===")
    start_time = time.time()

    try:
        # Define paths
        staging_path = config.PATHS['TEMP_DIR'] / "staging_G21_characteristic.parquet"
        health_condition_dim_path = config.PATHS['OUTPUT_DIR'] / "dim_health_condition.parquet"
        person_characteristic_dim_path = config.PATHS['OUTPUT_DIR'] / "dim_person_characteristic.parquet"
        refined_output_path = config.PATHS['OUTPUT_DIR'] / "fact_health_conditions_by_characteristic_refined.parquet" # Changed filename

        # --- Input Validation ---
        required_files = {
            "G21 Staging Data": staging_path,
            "Health Condition Dimension": health_condition_dim_path,
            "Person Characteristic Dimension": person_characteristic_dim_path
        }
        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(f"{name} at {path}")

        if missing_files:
            for msg in missing_files:
                logger.error(f"Required file not found: {msg}")
            logger.error("Cannot proceed with Refined G21 processing.")
            return False
        # --- End Input Validation ---

        logger.info(f"Reading dimensions: {health_condition_dim_path.name}, {person_characteristic_dim_path.name}")
        # Read dimensions into memory (usually small enough)
        dim_health = pl.read_parquet(health_condition_dim_path)
        dim_person = pl.read_parquet(person_characteristic_dim_path)

        logger.info(f"Scanning staging data: {staging_path.name}")
        # First read a single row to get column names without loading full dataset
        staging_sample = pl.read_parquet(staging_path, n_rows=1)
        staging_columns = staging_sample.columns
        logger.info(f"Staging G21 columns: {staging_columns}")
        
        # Now scan lazily for processing
        lazy_fact = pl.scan_parquet(staging_path)

        logger.info(f"Health Dim columns: {dim_health.columns}")
        logger.info(f"Person Dim columns: {dim_person.columns}")

        # Prepare dimensions for joining (select only necessary columns)
        dim_health_join = dim_health.select(['condition', 'condition_sk'])
        dim_person_join = dim_person.select(['characteristic_type', 'characteristic_code', 'characteristic_sk'])

        # Check if required columns exist in the staging file
        has_characteristic_code = 'characteristic_code' in staging_columns
        has_condition = 'condition' in staging_columns
        
        if not has_characteristic_code:
            logger.error(f"G21 staging data missing required 'characteristic_code' column.")
            logger.error(f"Available columns: {staging_columns}")
            return False
            
        if not has_condition:
            logger.error(f"G21 staging data missing required 'condition' column")
            logger.error(f"Available columns: {staging_columns}")
            return False

        # Perform joins lazily
        refined_lazy = (
            lazy_fact
            .join(
                dim_health_join.lazy(), # Join with lazy version of dimension
                on='condition',
                how='inner' # Use inner join to drop rows with conditions not in dimension
            )
            .join(
                dim_person_join.lazy(), # Join with lazy version of dimension
                on=['characteristic_type', 'characteristic_code'],
                how='inner' # Use inner join to drop rows with characteristics not in dimension
            )
            .select([ # Select and order the final columns for the fact table
                pl.col('geo_sk'),
                pl.col('time_sk'),
                pl.col('condition_sk'),
                pl.col('characteristic_sk'),
                pl.col('count') if 'count' in staging_columns else pl.lit(1).alias('count'), # Make sure 'count' column exists or provide default
                pl.lit(datetime.now()).alias('etl_processed_at')
            ])
        )

        logger.info(f"Executing streaming join and writing to {refined_output_path.name}")
        # Execute the lazy plan and stream the result directly to a Parquet file
        # sink_parquet handles writing in chunks efficiently
        refined_lazy.sink_parquet(refined_output_path, compression="snappy")

        # Optional: Log row count after writing (requires reading the output file)
        try:
             output_df = pl.read_parquet(refined_output_path)
             logger.info(f"Successfully wrote {len(output_df):,} rows to {refined_output_path}")
        except Exception as read_e:
             logger.warning(f"Could not read output file to verify row count: {read_e}")
             # Log that the sink operation itself didn't raise an error
             logger.info(f"Streaming write operation to {refined_output_path} completed without error.")


        end_time = time.time()
        logger.info(f"--- Refined G21 Processing Step Finished Successfully in {end_time - start_time:.2f} seconds ---")
        logger.info(f"=== Refined G21 Completed Successfully ===")
        return True

    except Exception as e:
        logger.error(f"Error in refined G21 processing: {e}")
        logger.error(traceback.format_exc())
        logger.error("=== Refined G21 Failed ===")
        return False

def run_census_g25_processing():
    """Process unpaid assistance (G25) census data."""
    logger.info("Starting G25 processing for unpaid assistance data")
    logger.info("=== Starting Census G25 Data Processing ===")
    
    # Look up the time dimension surrogate key for Census 2021
    logger.info("Looking up time dimension surrogate key for Census 2021")
    time_sk = get_time_dimension_sk(date(2021, 8, 10))
    logger.info(f"Using time_sk {time_sk} for Census date 2021-08-10")
    
    # Define pattern to match G25 Census files
    file_pattern = r"2021\s*Census_G25[_\s].*?(SA1|SA2)\.csv$"
    logger.info(f"[G25] Searching for Census files with pattern: {file_pattern}")
    
    try:
        # Use the same extract function as used for G20 and G21
        g25_csv_files = extract_g25_census_files(file_pattern)
        
        if not g25_csv_files:
            logger.error("[G25] No G25 Census files found")
            return False
        
        # Process each file
        results = []
        for csv_file in g25_csv_files:
            logger.info(f"[G25] Processing file: {csv_file}")
            
            # Read the CSV file
            df = pl.read_csv(csv_file)
            logger.info(f"[G25] Read {len(df)} rows from {csv_file}")
            
            # Identify the geographic code column (could be SA1_CODE_2021, SA2_CODE_2021, etc.)
            geo_code_col = None
            for col in df.columns:
                if "_CODE_2021" in col:
                    geo_code_col = col
                    break
                    
            if not geo_code_col:
                logger.error(f"[G25] Could not identify geographic code column in {csv_file}")
                continue
                
            logger.info(f"[G25] Found geographic code column: {geo_code_col}")
            logger.info(f"[G25] Available columns: {df.columns}")
            
            # Look for columns related to no assistance provided
            no_assistance_cols = [col for col in df.columns if "No_unpad_asst_prvided" in col]
            
            if not no_assistance_cols:
                logger.error(f"[G25] No 'No_unpad_asst_prvided' columns found in {csv_file}")
                continue
                
            logger.info(f"[G25] Found {len(no_assistance_cols)} no assistance columns")
            
            # Extract demographic information from the column names
            file_results = []
            
            # Vectorised operations instead of row-by-row filtering
            for col in no_assistance_cols:
                # Use group_by to efficiently aggregate data by geo_code
                summary_df = df.group_by(geo_code_col).agg(
                    pl.sum(col).alias("no_assistance_provided_count")
                )
                
                # Filter out null and zero values
                filtered_df = summary_df.filter(
                    (pl.col("no_assistance_provided_count").is_not_null()) & 
                    (pl.col("no_assistance_provided_count") > 0)
                )
                
                # Convert geo_code to string
                filtered_df = filtered_df.with_columns(
                    pl.col(geo_code_col).cast(pl.Utf8).alias("geo_code")
                )
                
                # Select only the columns we need
                result_records = filtered_df.select(
                    ["geo_code", "no_assistance_provided_count"]
                ).to_dicts()
                
                file_results.extend(result_records)
            
            logger.info(f"[G25] Extracted {len(file_results)} records from {csv_file}")
            results.extend(file_results)
        
        # Convert results to DataFrame
        if not results:
            logger.error("[G25] No valid data extracted from any file")
            return False
            
        result_df = pl.DataFrame(results)
        
        # Join with geographic dimension to get geo_sk
        geo_dim = pl.read_parquet(config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet")
        joined_df = result_df.join(
            geo_dim,
            left_on="geo_code",
            right_on="geo_code",
            how="left"
        )
        
        # Check for unmatched geo codes
        unmatched_df = joined_df.filter(pl.col("geo_sk").is_null())
        if len(unmatched_df) > 0:
            unmatched_pct = len(unmatched_df) / len(joined_df) * 100
            # Break up long line for better readability
            logger.warning(
                f"[G25] Data quality check: {len(unmatched_df)} rows "
                f"({unmatched_pct:.2f}%) have geo codes that don't match the geographic dimension"
            )
            
            # Sample some unmatched codes for debugging
            sample_unmatched = unmatched_df["geo_code"].unique()[:5].to_list()
            logger.warning(f"[G25] Sample of unmatched geo codes: {sample_unmatched}")
            
            # Filter out unmatched rows
            joined_df = joined_df.filter(pl.col("geo_sk").is_not_null())
            logger.info(f"[G25] Filtered out {len(unmatched_df)} rows with unmatched geo codes")
        
        # Add time dimension surrogate key and ETL timestamp
        joined_df = joined_df.with_columns([
            pl.lit(time_sk).alias("time_sk"),
            pl.lit(None).cast(pl.Datetime).alias("etl_processed_at")
        ])
        
        # Select final columns and save to parquet
        output_df = joined_df.select([
            "geo_sk", 
            "time_sk", 
            "no_assistance_provided_count", 
            "etl_processed_at"
        ])
        
        output_path = config.PATHS['OUTPUT_DIR'] / "fact_no_assistance.parquet"
        output_df.write_parquet(output_path)
        
        logger.info(f"[G25] Successfully saved {len(output_df)} records to {output_path}")
        logger.info("=== Census G25 Processing Step Finished Successfully ===")
        return True
        
    except Exception as e:
        logger.error(f"Failed to process G25 data: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def extract_g25_census_files(file_pattern):
    """Extract G25 census files matching the pattern from zip files."""
    logger.info(f"Searching for Census CSV files matching pattern: {file_pattern}")
    
    # Create extraction directory
    extract_dir = config.PATHS['TEMP_EXTRACT_DIR'] / "g25"
    os.makedirs(extract_dir, exist_ok=True)
    
    # Get all zip files in the raw data directory
    zip_files = []
    for root, _, files in os.walk(config.PATHS['RAW_DATA_DIR']):
        for file in files:
            if file.endswith('.zip'):
                zip_files.append(os.path.join(root, file))
    
    extracted_files = []
    
    for zip_path in zip_files:
        try:
            # Check if this zip file contains any G25 files
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                g25_files = [f for f in zip_ref.namelist() if re.search(file_pattern, f, re.IGNORECASE)]
                
                if g25_files:
                    logger.info(f"Found {len(g25_files)} G25 files in {zip_path}")
                    
                    # Extract only the G25 files
                    for g25_file in g25_files:
                        # Create subdirectories if needed
                        dirs = os.path.dirname(g25_file)
                        if dirs:
                            os.makedirs(os.path.join(extract_dir, dirs), exist_ok=True)
                        
                        # Extract the file
                        with zip_ref.open(g25_file) as source:
                            target_path = os.path.join(extract_dir, g25_file)
                            with open(target_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                            extracted_files.append(target_path)
        
        except Exception as e:
            logger.error(f"Error processing zip file {zip_path}: {str(e)}")
    
    if not extracted_files:
        logger.error("No G25 census files were found in any zip files.")
    else:
        logger.info(f"Extracted {len(extracted_files)} G25 census files to {extract_dir}")
    
    return extracted_files

def run_etl_pipeline(continue_on_error=True):
    """Run the complete ETL pipeline.
    
    Args:
        continue_on_error: If True, continue processing all steps even if one fails.
                           If False, stop processing after the first failure.
    """
    steps = [
        ('Download Census Data', lambda: run_download(force_download=False)), 
        ('Time Dimension', run_time_dimension_generation),
        ('Geographic Dimension', run_geographic_processing),
        ('Census G17 (Income)', run_census_g17_processing),
        ('Census G18', run_census_g18_processing),
        ('Census G19 (Health Condition)', run_census_g19_processing),
        ('Census G19 Detailed (Health Conditions Detailed)', run_census_g19_detailed_processing),
        ('Census G20 (Count of Health Conditions)', run_census_g20_processing),
        ('Census G21 (Health Conditions by Characteristic)', run_census_g21_processing),
        ('Health Condition Dimension', run_health_condition_dimension_generation),
        ('Demographic Dimension', run_demographic_dimension_generation),
        ('Person Characteristic Dimension', run_person_characteristic_dimension_generation),
        ('Refined G20', run_refined_g20_processing),
        ('Refined G21', run_refined_g21_processing),
        ('Census G25', run_census_g25_processing)
    ]
    
    # Track overall success
    overall_success = True
    
    # Run each step
    for step_name, step_func in steps:
        logger.info(f"=== Starting {step_name} ===")
        try:
            success = step_func()
            # Ensure step functions return a boolean
            if success is None:
                logger.error(f"=== {step_name} returned None instead of a boolean status ===")
                overall_success = False
            elif not success:
                logger.error(f"=== {step_name} Failed ===")
                overall_success = False
            else:
                logger.info(f"=== {step_name} Completed Successfully ===")
                
            # If a step fails and continue_on_error is False, break the loop
            if not success and not continue_on_error:
                logger.error(f"=== Stopping pipeline after failure in {step_name} ===")
                break
                
        except Exception as e:
            logger.error(f"=== {step_name} Failed with Exception: {e} ===")
            import traceback
            logger.error(traceback.format_exc())
            overall_success = False
            
            # If an exception occurs and continue_on_error is False, break the loop
            if not continue_on_error:
                logger.error(f"=== Stopping pipeline after exception in {step_name} ===")
                break
    
    # Capture status BEFORE validation
    pipeline_status = overall_success
    
    # Get validation flag from args
    args = parse_args()
    
    # Skip validation if flag is set
    if args.skip_validation:
        logger.info("=== Skipping validation as requested ===")
        return pipeline_status
        
    # Log start of validation with timing
    logger.info(
        f"=== Pipeline Processing Complete (Success: {pipeline_status}). "
        f"Starting Final Validation... ==="
    )
    validation_start_time = time.time()
    
    # Final validation checks
    # Note: We'll always run validation checks even if overall_success is False,
    # to provide a complete report on the state of the data
    try:
        # Required output files that should exist after successful pipeline run
        required_files = [
            'geo_dimension.parquet',
            'dim_time.parquet',
            'dim_health_condition.parquet',
            'dim_demographic.parquet',
            'dim_person_characteristic.parquet',
            'fact_health_conditions_refined.parquet',
            'fact_health_conditions_by_characteristic_refined.parquet',
            'fact_no_assistance.parquet'
        ]
        
        # Check each required file
        missing_files = []
        empty_files = []
        for filename in required_files:
            file_path = config.PATHS['OUTPUT_DIR'] / filename
            if not file_path.exists():
                missing_files.append(filename)
            elif file_path.stat().st_size == 0:
                empty_files.append(filename)
        
        if missing_files:
            logger.error(f"Missing critical output files: {missing_files}")
            overall_success = False
        
        if empty_files:
            logger.error(f"Empty output files detected: {empty_files}")
            overall_success = False
            
        # Additional data quality checks
        try:
            # Check if fact tables have reasonable row counts
            fact_files = {
                'fact_health_conditions_refined.parquet': 1000,  # Minimum expected rows
                'fact_health_conditions_by_characteristic_refined.parquet': 1000,
                'fact_no_assistance.parquet': 1000
            }
            
            for fact_file, min_rows in fact_files.items():
                file_path = config.PATHS['OUTPUT_DIR'] / fact_file
                if file_path.exists():
                    try:
                        # OPTIMISED READ: Get row count without loading data
                        scan = pl.scan_parquet(file_path)
                        # Check if the file actually has rows before collecting length
                        if scan.select(pl.first()).collect().height > 0:
                            row_count = scan.select(pl.len()).collect().item()
                        else:
                            row_count = 0
                            
                        logger.info(f"Validation: {fact_file} has {row_count:,} rows.")
                        if row_count < min_rows:
                            logger.warning(
                                f"Validation Warning: {fact_file} has only {row_count} rows, "
                                f"which is fewer than expected minimum of {min_rows}"
                            )
                    except Exception as e:
                        logger.error(f"Error getting row count for {fact_file}: {e}")
                        if not continue_on_error:  # Only affect overall success if we're not continuing on error
                            overall_success = False
                else:
                    # If the file doesn't exist, log it but don't fail if we're continuing on errors
                    logger.warning(f"Validation Warning: Required fact file {fact_file} not found.")
                    if not continue_on_error:
                        overall_success = False
        
        except Exception as e:
            logger.error(f"Error during fact table validation: {e}")
            if not continue_on_error:
                overall_success = False
            
    except Exception as e:
        logger.error(f"Error during final validation setup: {e}")
        if not continue_on_error:
            overall_success = False
    
    # Log validation timing
    validation_end_time = time.time()
    logger.info(
        f"=== Final Validation Checks Completed in {validation_end_time - validation_start_time:.2f} seconds "
        f"(Success: {overall_success}) ==="
    )
    
    # Final status - if continue_on_error is True, we'll consider it successful even with errors
    if overall_success or continue_on_error:
        logger.info("=== Full ETL Pipeline Completed Successfully ===")
        return True
    else:
        logger.error("=== Full ETL Pipeline Completed with Errors ===")
        logger.error("AHGD ETL Pipeline completed with errors. Check logs for details.")
        return False

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ETL processes for the AHGD')
    parser.add_argument('--step', type=str, default='all',
                        choices=['all', 'download', 'time', 'geo', 'g17', 'g18', 'g19', 'g19detailed', 'g20', 'g21',
                                'health_dim', 'demo_dim', 'characteristic_dim', 
                                'refine_g20', 'refine_g21', 'g25'],
                        help='Step to run (default: all)')
    parser.add_argument('--force-download', action='store_true',
                        help='Force download even if files exist')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip final validation checks (useful if they are too slow)')
    parser.add_argument('--stop-on-error', action='store_true',
                        help='Stop processing after the first failed step')
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialise directories
    config.initialise_directories()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.PATHS['LOG_DIR'] / 'etl.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set the log level (this can be adjusted as needed)
    logger.setLevel(logging.INFO)
    
    # Choose what to run based on arguments
    if args.step == 'all':
        success = run_etl_pipeline(continue_on_error=not args.stop_on_error)
    elif args.step == 'download':
        logger.info("=== Starting Download Step ===")
        success = run_download(args.force_download)
        if success:
            logger.info("=== Download Step Completed Successfully ===")
        else:
            logger.error("=== Download Step Failed ===")
    elif args.step == 'time':
        success = run_time_dimension_generation()
    elif args.step == 'geo':
        success = run_geographic_processing()
    elif args.step == 'g17':
        success = run_census_g17_processing()
    elif args.step == 'g18':
        success = run_census_g18_processing()
    elif args.step == 'g19':
        success = run_census_g19_processing()
    elif args.step == 'g19detailed':
        success = run_census_g19_detailed_processing()
    elif args.step == 'g20':
        success = run_census_g20_processing()
    elif args.step == 'g21':
        success = run_census_g21_processing()
    elif args.step == 'health_dim':
        success = run_health_condition_dimension_generation()
    elif args.step == 'demo_dim':
        success = run_demographic_dimension_generation()
    elif args.step == 'characteristic_dim':
        success = run_person_characteristic_dimension_generation()
    elif args.step == 'refine_g20':
        success = run_refined_g20_processing()
    elif args.step == 'refine_g21':
        success = run_refined_g21_processing()
    elif args.step == 'g25':
        success = run_census_g25_processing()
    else:
        logger.error(f"Unknown step: {args.step}")
        return 1

    # Final status message - if running just a single step, report its specific result
    # If running all steps with continue_on_error=True, we'll always report success
    if args.step == 'all' and not args.stop_on_error:
        logger.info("AHGD ETL Pipeline completed.")
        return 0
    elif success:
        logger.info("AHGD ETL Pipeline completed successfully.")
        return 0
    else:
        logger.error("AHGD ETL Pipeline completed with errors. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())