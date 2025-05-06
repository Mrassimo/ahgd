#!/usr/bin/env python3
"""
Main entry point and orchestrator for the AHGD ETL pipeline.

This script handles command-line arguments to control the ETL process,
including downloading data, processing geographic boundaries, and processing
Census data.

Usage examples:
  python run_etl.py --steps all          # Run all steps
  python run_etl.py --steps download geo # Run download and geography steps
  python run_etl.py --steps census g01   # Run all census processing + G01 specifically
  python run_etl.py --steps time         # Run only the time dimension creation step
  python run_etl.py --steps validate     # Run data validation checks
  python run_etl.py --steps g01 g17 g18  # Run only specific census tables
"""

import argparse
import logging
import sys
from pathlib import Path
import polars as pl
from typing import Dict, List, Optional, Callable, Any, Tuple
import time
import os
import traceback
import re
from datetime import date, datetime

# Import configuration and utility modules
from etl_logic import config
from etl_logic import utils, geography, census, time_dimension, dimensions, validation
from etl_logic.utils import download_data
from etl_logic.time_dimension import generate_time_dimension
from etl_logic.geography import process_geography
from etl_logic.census import process_census_table

# Import all table processors
from etl_logic.tables.g01_population import process_g01_file
from etl_logic.tables.g17_income import process_g17_file
from etl_logic.tables.g18_assistance_needed import process_g18_file
from etl_logic.tables.g19_health_conditions import process_g19_file, process_g19_detailed_csv
from etl_logic.tables.g20_selected_conditions import process_g20_unpivot_csv
from etl_logic.tables.g21_conditions_by_characteristics import process_g21_unpivot_csv
from etl_logic.tables.g25_unpaid_assistance import process_g25_file

# Setup logging
logger = utils.setup_logging(config.PATHS.get('LOG_DIR'))

def get_time_dimension_sk(date_value):
    """Get the time dimension surrogate key for a given date."""
    try:
        time_dim_path = config.PATHS['OUTPUT_DIR'] / "dim_time.parquet"
        if not os.path.exists(time_dim_path):
            logger.error(f"Time dimension not found at {time_dim_path}")
            return None
            
        # Read the time dimension parquet file
        time_dim = pl.read_parquet(time_dim_path)
        
        if not isinstance(date_value, str):
            date_str = date_value.isoformat()
        else:
            date_str = date_value
            
        # Find the matching date and return the surrogate key
        matching_row = time_dim.filter(pl.col("full_date").cast(pl.Utf8).str.contains(date_str))
        
        if len(matching_row) > 0:
            return matching_row["time_sk"][0]
        else:
            # If no exact match, try to find the closest date
            # For Census 2021, use an approximate date if needed (August 10, 2021)
            if "2021" in date_str:
                census_rows = time_dim.filter(pl.col("full_date").dt.year() == 2021)
                if len(census_rows) > 0:
                    # Get August 2021 or the closest available date
                    august_rows = census_rows.filter(pl.col("full_date").dt.month() == 8)
                    if len(august_rows) > 0:
                        return august_rows["time_sk"][0]
                    else:
                        return census_rows["time_sk"][0]
            
            logger.warning(f"No matching date found in time dimension for {date_str}")
            return None
    except Exception as e:
        logger.error(f"Exception in get_time_dimension_sk: {e}")
        return None

def run_download_step(force_download: bool = False) -> bool:
    """Download required geographic and census data files.
    
    Args:
        force_download (bool): Whether to force re-download of existing files
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    logger.info("=== Starting Data Download ===")
    
    try:
        # Initialize directories if they don't exist
        config.initialise_directories()
        
        # Get geographic data to download
        geo_zips = config.get_required_geo_zips()
        logger.info(f"Geographic files to download: {list(geo_zips.keys())}")
        
        # Get census data to download
        census_zips = config.get_required_census_zips()
        logger.info(f"Census files to download: {list(census_zips.keys())}")
        
        # Combine all files to download
        all_zips = {**geo_zips, **census_zips}
        
        # Set target directories
        geo_dir = config.PATHS['GEOGRAPHIC_DIR']
        census_dir = config.PATHS['CENSUS_DIR']
        
        # Download geographic files
        for filename, url in geo_zips.items():
            target_path = geo_dir / filename
            download_data(url, target_path, force_download=force_download)
        
        # Download census files
        for filename, url in census_zips.items():
            target_path = census_dir / filename
            download_data(url, target_path, force_download=force_download)
        
        logger.info("=== Data Download Complete ===")
        return True
    
    except Exception as e:
        logger.error(f"Error in download step: {str(e)}", exc_info=True)
        return False

def run_geography_step() -> bool:
    """Process geographic files to create the geographic dimension.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("=== Starting Geographic Processing ===")
    
    try:
        geo_result = process_geography(
            shp_dir=config.PATHS['GEOGRAPHIC_DIR'],
            output_path=config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet",
            geo_levels=list(config.GEO_LEVELS_SHP_PROCESS.keys())
        )
        
        logger.info("=== Geographic Processing Complete ===")
        return geo_result
    
    except Exception as e:
        logger.error(f"Error in geography step: {str(e)}", exc_info=True)
        return False

def run_time_dimension_step() -> bool:
    """Create time dimension.
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("=== Starting Time Dimension Creation ===")
    
    try:
        time_result = generate_time_dimension(
            start_year=2010,
            end_year=2030,
            output_path=config.PATHS['OUTPUT_DIR'] / "dim_time.parquet"
        )
        
        logger.info("=== Time Dimension Creation Complete ===")
        return time_result
    
    except Exception as e:
        logger.error(f"Error in time dimension step: {str(e)}", exc_info=True)
        return False

def run_g01_step() -> bool:
    """Process G01 census data."""
    logger.info("=== Starting G01 Census Processing ===")
    table_code = "G01"
    try:
        time_sk = get_time_dimension_sk(date(2021, 8, 10)) # Assuming Census 2021 date
        logger.info(f"[{table_code}] Using time_sk={time_sk} for Census date 2021-08-10")
        if time_sk is None:
             logger.error(f"[{table_code}] Could not determine time_sk. Aborting G01 step.")
             return False

        result = process_census_table(
            table_code=table_code,
            process_file_function=process_g01_file, # Use the function from tables.g01_population
            output_filename="fact_population.parquet", # Define output filename directly
            zip_dir=config.PATHS['CENSUS_DIR'],
            temp_extract_base=config.PATHS['TEMP_EXTRACT_DIR'],
            output_dir=config.PATHS['OUTPUT_DIR'],
            geo_output_path=config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet",
            time_sk=time_sk
        )

        logger.info(f"=== G01 Census Processing Complete (Success: {result}) ===")
        return result
    except Exception as e:
        logger.error(f"Error in G01 step: {str(e)}", exc_info=True)
        return False

def run_g17_step() -> bool:
    """Process G17 census data."""
    logger.info("=== Starting G17 Census Processing ===")
    table_code = "G17"
    try:
        time_sk = get_time_dimension_sk(date(2021, 8, 10))
        logger.info(f"[{table_code}] Using time_sk={time_sk} for Census date 2021-08-10")
        if time_sk is None:
             logger.error(f"[{table_code}] Could not determine time_sk. Aborting G17 step.")
             return False

        result = process_census_table(
            table_code=table_code,
            process_file_function=process_g17_file, # Use the function from tables.g17_income
            output_filename="fact_income.parquet",
            zip_dir=config.PATHS['CENSUS_DIR'],
            temp_extract_base=config.PATHS['TEMP_EXTRACT_DIR'],
            output_dir=config.PATHS['OUTPUT_DIR'],
            geo_output_path=config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet",
            time_sk=time_sk
        )

        logger.info(f"=== G17 Census Processing Complete (Success: {result}) ===")
        return result
    except Exception as e:
        logger.error(f"Error in G17 step: {str(e)}", exc_info=True)
        return False

def run_g18_step() -> bool:
    """Process G18 census data."""
    logger.info("=== Starting G18 Census Processing ===")
    table_code = "G18"
    try:
        time_sk = get_time_dimension_sk(date(2021, 8, 10))
        logger.info(f"[{table_code}] Using time_sk={time_sk} for Census date 2021-08-10")
        if time_sk is None:
             logger.error(f"[{table_code}] Could not determine time_sk. Aborting G18 step.")
             return False

        result = process_census_table(
            table_code=table_code,
            process_file_function=process_g18_file, # Use the function from tables.g18_assistance_needed
            output_filename="fact_assistance_needed.parquet", # Corrected filename?
            zip_dir=config.PATHS['CENSUS_DIR'],
            temp_extract_base=config.PATHS['TEMP_EXTRACT_DIR'],
            output_dir=config.PATHS['OUTPUT_DIR'],
            geo_output_path=config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet",
            time_sk=time_sk
        )

        logger.info(f"=== G18 Census Processing Complete (Success: {result}) ===")
        return result
    except Exception as e:
        logger.error(f"Error in G18 step: {str(e)}", exc_info=True)
        return False

def run_g19_step() -> bool:
    """Process G19 census data (Long-Term Health Conditions)."""
    logger.info("=== Starting G19 Census Processing ===")
    table_code = "G19"
    try:
        time_sk = get_time_dimension_sk(date(2021, 8, 10))
        logger.info(f"[{table_code}] Using time_sk={time_sk} for Census date 2021-08-10")
        if time_sk is None:
             logger.error(f"[{table_code}] Could not determine time_sk. Aborting G19 step.")
             return False

        # Using the basic process_g19_file here
        result = process_census_table(
            table_code=table_code,
            process_file_function=process_g19_file, # Use the function from tables.g19_health_conditions
            output_filename="fact_health_condition_basic.parquet", # Consider distinct name
            zip_dir=config.PATHS['CENSUS_DIR'],
            temp_extract_base=config.PATHS['TEMP_EXTRACT_DIR'],
            output_dir=config.PATHS['OUTPUT_DIR'],
            geo_output_path=config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet",
            time_sk=time_sk
        )
        # Optionally run detailed processing as well or make it a separate step?
        # result_detailed = process_census_table(
        #     table_code="G19", # Still uses G19 pattern
        #     process_file_function=process_g19_detailed_csv, 
        #     output_filename="fact_health_condition_detailed.parquet", 
        #     ...
        # )

        logger.info(f"=== G19 Census Processing Complete (Success: {result}) ===")
        return result # Return status of basic processing for now
    except Exception as e:
        logger.error(f"Error in G19 step: {str(e)}", exc_info=True)
        return False

def run_g20_step() -> bool:
    """Process G20 census data (Count of Conditions)."""
    logger.info("=== Starting G20 Census Processing ===")
    table_code = "G20"
    try:
        time_sk = get_time_dimension_sk(date(2021, 8, 10))
        logger.info(f"[{table_code}] Using time_sk={time_sk} for Census date 2021-08-10")
        if time_sk is None:
             logger.error(f"[{table_code}] Could not determine time_sk. Aborting G20 step.")
             return False

        result = process_census_table(
            table_code=table_code,
            process_file_function=process_g20_unpivot_csv, # Use the unpivot function from tables.g20_selected_conditions
            output_filename="fact_condition_counts.parquet", # e.g. count of conditions per person
            zip_dir=config.PATHS['CENSUS_DIR'],
            temp_extract_base=config.PATHS['TEMP_EXTRACT_DIR'],
            output_dir=config.PATHS['OUTPUT_DIR'],
            geo_output_path=config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet",
            time_sk=time_sk
        )

        logger.info(f"=== G20 Census Processing Complete (Success: {result}) ===")
        return result
    except Exception as e:
        logger.error(f"Error in G20 step: {str(e)}", exc_info=True)
        return False

def run_g21_step() -> bool:
    """Process G21 census data (Condition by Characteristic)."""
    logger.info("=== Starting G21 Census Processing ===")
    table_code = "G21"
    try:
        time_sk = get_time_dimension_sk(date(2021, 8, 10))
        logger.info(f"[{table_code}] Using time_sk={time_sk} for Census date 2021-08-10")
        if time_sk is None:
             logger.error(f"[{table_code}] Could not determine time_sk. Aborting G21 step.")
             return False

        result = process_census_table(
            table_code=table_code,
            process_file_function=process_g21_unpivot_csv, # Use the unpivot function from tables.g21_...
            output_filename="fact_health_condition_by_characteristic.parquet", # Needs refinement step later
            zip_dir=config.PATHS['CENSUS_DIR'],
            temp_extract_base=config.PATHS['TEMP_EXTRACT_DIR'],
            output_dir=config.PATHS['OUTPUT_DIR'],
            geo_output_path=config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet",
            time_sk=time_sk
        )
        # Note: The original process_g21_file included a post-processing step for uniqueness.
        # This should ideally be moved into process_g21_unpivot_csv or a subsequent refinement step.

        logger.info(f"=== G21 Census Processing Complete (Success: {result}) ===")
        return result
    except Exception as e:
        logger.error(f"Error in G21 step: {str(e)}", exc_info=True)
        return False

def run_g25_step() -> bool:
    """Process G25 census data (Unpaid Assistance)."""
    logger.info("=== Starting G25 Census Processing ===")
    table_code = "G25"
    try:
        time_sk = get_time_dimension_sk(date(2021, 8, 10))
        logger.info(f"[{table_code}] Using time_sk={time_sk} for Census date 2021-08-10")
        if time_sk is None:
             logger.error(f"[{table_code}] Could not determine time_sk. Aborting G25 step.")
             return False

        result = process_census_table(
            table_code=table_code,
            process_file_function=process_g25_file, # Use the function from tables.g25_unpaid_assistance
            output_filename="fact_unpaid_assistance.parquet",
            zip_dir=config.PATHS['CENSUS_DIR'],
            temp_extract_base=config.PATHS['TEMP_EXTRACT_DIR'],
            output_dir=config.PATHS['OUTPUT_DIR'],
            geo_output_path=config.PATHS['OUTPUT_DIR'] / "geo_dimension.parquet",
            time_sk=time_sk
        )

        logger.info(f"=== G25 Census Processing Complete (Success: {result}) ===")
        return result
    except Exception as e:
        logger.error(f"Error in G25 step: {str(e)}", exc_info=True)
        return False

def run_refine_health_facts_step() -> bool:
    """Wrapper to trigger health fact refinement logic (likely in dimensions or a dedicated module)."""
    logger.info("=== Starting Health Facts Refinement ===")
    try:
        # Assuming the core logic is moved elsewhere, e.g., dimensions.refine_health_facts()
        # This function now just orchestrates the call and handles exceptions
        # success = dimensions.refine_health_facts(config.PATHS['OUTPUT_DIR'])

        # Placeholder: Replace with actual call to refinement logic
        logger.warning("Placeholder: Health fact refinement logic needs to be implemented/called.")
        success = True # Assume success for now until logic is implemented

        if success:
            logger.info("=== Health Facts Refinement Complete (Success: True) ===")
        else:
            logger.error("=== Health Facts Refinement Failed ===")
        return success
    except Exception as e:
        logger.error(f"Error in refine_health_facts step: {str(e)}", exc_info=True)
        return False

def run_dimensions_step() -> bool:
    """Generate all required dimension tables (excluding time and geo)."""
    logger.info("=== Starting Dimension Generation ===")
    all_success = True
    try:
        # Assuming dimension generation logic is in the dimensions module
        output_dir = config.PATHS['OUTPUT_DIR']

        logger.info("Generating Health Condition dimension...")
        success_health = dimensions.generate_health_condition_dimension(
            config.HEALTH_CONDITIONS, output_dir / "dim_health_condition.parquet"
        )
        if not success_health:
            logger.error("Failed to generate Health Condition dimension.")
            all_success = False

        logger.info("Generating Demographic dimension...")
        success_demo = dimensions.generate_demographic_dimension(
            config.DEMOGRAPHIC_CATEGORIES, output_dir / "dim_demographic.parquet"
        )
        if not success_demo:
            logger.error("Failed to generate Demographic dimension.")
            all_success = False

        logger.info("Generating Person Characteristic dimension...")
        success_char = dimensions.generate_person_characteristic_dimension(
            config.PERSON_CHARACTERISTICS, output_dir / "dim_person_characteristic.parquet"
        )
        if not success_char:
            logger.error("Failed to generate Person Characteristic dimension.")
            all_success = False

        # Add calls for other dimensions if needed

        logger.info(f"=== Dimension Generation Complete (Overall Success: {all_success}) ===")
        return all_success

    except Exception as e:
        logger.error(f"Error in dimensions step: {str(e)}", exc_info=True)
        return False

def run_validation_step() -> bool:
    """Run data validation checks on the generated Parquet files."""
    logger.info("=== Starting Data Validation ===")
    try:
        output_dir = config.PATHS['OUTPUT_DIR']
        # Assuming validation.run_all_validations returns a boolean overall status
        # and logs details internally
        report = validation.run_all_validations(output_dir)

        # Determine overall validity from the report structure seen previously
        all_valid = all(result["passed"] for result in report.values())

        # Log summary based on the report
        logger.info("--- Validation Report Summary ---")
        passed_count = sum(1 for result in report.values() if result["passed"])
        failed_count = len(report) - passed_count

        for check_name, result in report.items():
            status = "PASSED" if result["passed"] else "FAILED"
            logger.info(f"Check: {check_name:<40} Status: {status}")
            if not result["passed"]:
                if "details" in result:
                     logger.warning(f"  Details: {result['details']}")
                if "failed_rows_sample" in result and result["failed_rows_sample"] is not None:
                     # Convert Polars DataFrame sample to string for logging if needed
                     sample_str = str(result["failed_rows_sample"])
                     logger.warning(f"  Failed Rows Sample:\n{sample_str}")

        logger.info("---------------------------------")
        logger.info(f"Total Checks: {len(report)}, Passed: {passed_count}, Failed: {failed_count}")
        logger.info(f"=== Data Validation Complete (Overall Result: {'PASSED' if all_valid else 'FAILED'}) ===")
        return all_valid

    except Exception as e:
        logger.error(f"Error during validation step: {str(e)}", exc_info=True)
        return False

# Mapping from step names to functions
# Uses the refactored step functions
ETL_STEPS: Dict[str, Dict[str, Any]] = {
    "download": {
        "func": run_download_step,
        "description": "Download ASGS and Census data ZIP files"
    },
    "geo": {
        "func": run_geography_step,
        "description": "Process ASGS Shapefiles into geo_dimension.parquet"
    },
    "time": {
        "func": run_time_dimension_step,
        "description": "Generate dim_time.parquet"
    },
    "dimensions": {
        "func": run_dimensions_step,
        "description": "Generate other dimensions (health, demo, characteristic)"
    },
    "g01": {
        "func": run_g01_step,
        "description": "Process G01 (Population) census data"
    },
    "g17": {
        "func": run_g17_step,
        "description": "Process G17 (Income) census data"
    },
    "g18": {
        "func": run_g18_step,
        "description": "Process G18 (Assistance Needed) census data"
    },
    "g19": {
        "func": run_g19_step,
        "description": "Process G19 (Health Conditions) census data"
    },
    "g20": {
        "func": run_g20_step,
        "description": "Process G20 (Selected Conditions) census data"
    },
    "g21": {
        "func": run_g21_step,
        "description": "Process G21 (Conditions by Characteristics) census data"
    },
    "g25": {
        "func": run_g25_step,
        "description": "Process G25 (Unpaid Assistance) census data"
    },
    "refine_health": { # Renamed from refine_health_facts for consistency
        "func": run_refine_health_facts_step,
        "description": "Refine health facts by combining G19, G20, G21 outputs"
    },
    "validate": {
        "func": run_validation_step,
        "description": "Run data quality and validation checks on output Parquet files"
    },
}

# Define step groups based on logical execution order and dependencies
STEP_GROUPS = {
    # Defines the default execution order when 'all' is specified
    "all": [
        "download",
        "geo",
        "time",
        "dimensions", # Must run before census tables using these dims
        "g01",
        "g17",
        "g18",
        "g19",
        "g20",
        "g21",
        "g25",
        "refine_health", # Must run after G19, G20, G21 and dimensions
        "validate",      # Should generally run last
    ],
    # Group of all census table processing steps
    "census": ["g01", "g17", "g18", "g19", "g20", "g21", "g25"],
    # Foundational steps required before most census processing
    "base_dims": ["download", "geo", "time", "dimensions"],
}

def run_etl_pipeline(steps: List[str], stop_on_error: bool = False, force_download: bool = False) -> bool:
    """Runs the specified steps of the ETL pipeline.
    Args:
        steps (List[str]): A list of step names to execute.
                           Can include 'all', 'census', specific step names (e.g., 'download', 'g01').
        stop_on_error (bool): If True, stops the pipeline immediately on the first error.
                              If False, logs the error and continues with subsequent steps.
        force_download (bool): If True, forces re-download of files in the 'download' step.
    Returns:
        bool: True if all executed steps were successful, False otherwise.
    """
    logger.info(f"Starting ETL pipeline run for steps: {steps}")
    pipeline_start_time = time.perf_counter()
    overall_success = True
    executed_steps_summary: List[Dict[str, Any]] = [] # Store results for summary

    # Determine the actual steps to run
    steps_to_run = []
    if "all" in steps:
        steps_to_run = STEP_GROUPS["all"]
    else:
        requested_steps = set(steps)
        if "census" in requested_steps:
            # Add all census table steps if 'census' is requested
            requested_steps.remove("census")
            requested_steps.update(STEP_GROUPS["census"])
            # Also add dimension/refinement steps often needed with census
            if "dimensions" not in requested_steps:
                requested_steps.add("dimensions")
            if "refine_health" not in requested_steps:
                requested_steps.add("refine_health")

        # Ensure fundamental steps run if census steps are requested but base steps aren't explicitly mentioned
        needs_census_prereqs = any(step in STEP_GROUPS["census"] for step in requested_steps) or \
                               "refine_health" in requested_steps
        if needs_census_prereqs:
             # Add base dimensions if any census step or refine step is requested
             for base_step in STEP_GROUPS["base_dims"]:
                 if base_step not in requested_steps:
                    requested_steps.add(base_step)

        # Ensure validate runs last if requested
        run_validate_last = "validate" in requested_steps
        if run_validate_last:
            requested_steps.remove("validate")

        # Order the requested steps based on the canonical order in STEP_GROUPS["all"]
        steps_to_run = [step for step in STEP_GROUPS["all"] if step in requested_steps]

        # Add validate back at the end if it was requested
        if run_validate_last:
            # Ensure validate isn't accidentally added twice if 'all' was also specified somehow
            if "validate" not in steps_to_run:
                 steps_to_run.append("validate")

    logger.info(f"Resolved execution order: {steps_to_run}")

    for step_name in steps_to_run:
        if step_name not in ETL_STEPS:
            logger.warning(f"Skipping unknown step: {step_name}")
            continue

        step_info = ETL_STEPS[step_name]
        step_func = step_info["func"]
        logger.info(f"--- Executing Step: {step_name.upper()} --- ({step_info.get('description', '')})") # Added description log
        step_start_time = time.perf_counter()
        success = False
        try:
            # Special handling for download step to pass force_download flag
            if step_name == "download":
                success = step_func(force_download=force_download)
            else:
                success = step_func()

        except Exception as e:
            logger.error(f"Unexpected error occurred while trying to run step '{step_name}': {str(e)}", exc_info=True)
            success = False # Ensure step is marked as failed

        step_end_time = time.perf_counter()
        step_duration = step_end_time - step_start_time
        status_str = "Success" if success else "Failure"

        executed_steps_summary.append({
            "name": step_name,
            "status": status_str,
            "duration_seconds": step_duration
        })

        if not success:
            overall_success = False
            logger.error(f"--- Step: {step_name.upper()} FAILED (Duration: {step_duration:.2f}s) ---")
            if stop_on_error:
                logger.warning("Stopping pipeline due to error.")
                break
        else:
             logger.info(f"--- Step: {step_name.upper()} COMPLETED (Duration: {step_duration:.2f}s) ---")


    pipeline_end_time = time.perf_counter()
    total_duration = pipeline_end_time - pipeline_start_time

    # --- Summary Report ---
    logger.info("===========================================")
    logger.info("           ETL Pipeline Summary            ")
    logger.info("===========================================")
    logger.info(f"Requested Steps: {steps}")
    logger.info(f"Executed Steps Order: {steps_to_run}")
    logger.info("-------------------------------------------")
    logger.info("Step Results:")
    for step_summary in executed_steps_summary:
        logger.info(f"  - {step_summary['name']:<15} | Status: {step_summary['status']:<8} | Duration: {step_summary['duration_seconds']:.2f}s")
    logger.info("-------------------------------------------")
    logger.info(f"Total Pipeline Execution Time: {total_duration:.2f}s")
    logger.info(f"Overall Status: {'SUCCESS' if overall_success else 'FAILURE'}")
    logger.info("===========================================")

    # Ensure final log messages are written out
    logging.shutdown()

    return overall_success

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AHGD ETL Pipeline")
    
    parser.add_argument("--steps", nargs="+", default=["all"],
                        help=f"ETL steps to run. Available steps: {', '.join(ETL_STEPS.keys())} or groups: {', '.join(STEP_GROUPS.keys())}")
    
    parser.add_argument("--stop-on-error", action="store_true",
                        help="Stop pipeline execution on first error")
    
    parser.add_argument("--force-download", action="store_true",
                        help="Force re-download of data files even if they exist")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize config and logging
    config.initialise_directories()
    
    # Run ETL pipeline
    result = run_etl_pipeline(args.steps, args.stop_on_error, args.force_download)
    
    # Exit with appropriate status code
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()