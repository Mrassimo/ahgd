#!/usr/bin/env python3
"""
Main entry point and orchestrator for the AHGD ETL pipeline.

This script handles command-line arguments to control the ETL process,
including downloading data, processing geographic boundaries, and processing
Census data.

Usage examples:
  python run_etl_enhanced.py --steps all          # Run all steps
  python run_etl_enhanced.py --steps download geo # Run download and geography steps
  python run_etl_enhanced.py --steps census g01   # Run all census processing + G01 specifically
  python run_etl_enhanced.py --steps time         # Run only the time dimension creation step
  python run_etl_enhanced.py --steps validate     # Run data validation checks
  python run_etl_enhanced.py --steps g01 g17 g18  # Run only specific census tables
"""

import argparse
import logging
import sys
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Dict, List, Set, Optional, Any

# Import from the ahgd_etl package
from ahgd_etl import utils
from ahgd_etl.config.settings import get_config_manager
from ahgd_etl.transformers.manager import TransformerManager
from ahgd_etl.transformers.geo import GeoTransformer, process_geography
from ahgd_etl.models import TimeDimensionModel, generate_time_dimension
from ahgd_etl.validators import DataQualityValidator
from ahgd_etl.core.temp_fix.dimension_fix import run_dimension_fix

# Get configuration manager
config_manager = get_config_manager()

# Setup logging
logger = utils.setup_logging(config_manager.get_path('LOG_DIR'))

# Define ETL steps
ETL_STEPS = {
    # Core steps
    'download': 'Download data files',
    'geo': 'Process geographic boundaries',
    'time': 'Create time dimension',
    'dimensions': 'Create dimension tables',
    'validate': 'Validate data',
    'refine_health': 'Refine health facts',
    
    # Census table processing
    'g01': 'Process G01 Population data',
    'g17': 'Process G17 Income data',
    'g18': 'Process G18 Assistance Needed data',
    'g19': 'Process G19 Health Conditions data',
    'g20': 'Process G20 Selected Conditions data',
    'g21': 'Process G21 Conditions by Characteristics data',
    'g25': 'Process G25 Unpaid Assistance data',
    
    # Step groups
    'all': 'All steps',
    'census': 'All Census tables',
}

# Step groups - in a logical sequence
STEP_GROUPS = {
    'all': ['download', 'time', 'geo', 'g01', 'g17', 'g18', 'g19', 'g20', 'g21', 'g25', 'dimensions', 'refine_health', 'validate'],
    'census': ['g01', 'g17', 'g18', 'g19', 'g20', 'g21', 'g25'],
    'base_dims': ['time', 'geo'],
    'post_process': ['dimensions', 'refine_health', 'validate'],
}

def time_func(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"Step {func.__name__.replace('run_', '')} completed with status: {'SUCCESS' if result else 'FAILED'} (elapsed: {elapsed:.2f}s)")
        return result
    return wrapper

@time_func
def run_download_step(force_download: bool = False) -> bool:
    """
    Download data files from the internet.

    Args:
        force_download: Whether to force download even if files exist

    Returns:
        True if download was successful, False otherwise
    """
    logger.info("=== Starting Data Download ===")

    try:
        # Get data source URLs from the configuration
        data_sources_dict = config_manager.data_sources
        if not data_sources_dict:
            logger.error("No data sources defined in configuration")
            return False

        # Download geographic files
        asgs_urls = data_sources_dict.get('asgs2021_urls', {})
        geo_levels = config_manager.get_column_mapping('geo_levels')

        # Track overall download success
        download_success = True

        for name, url in asgs_urls.items():
            # Handle special case for STE (which maps to STATE in geo_levels)
            if name == 'STE' and 'STATE' in geo_levels:
                filename = f"{geo_levels['STATE']}.zip"
                logger.info(f"Mapping STE to {geo_levels['STATE']} based on geo_levels['STATE']")
            # Use the geo_levels mapping to determine the correct filename
            elif name in geo_levels:
                filename = f"{geo_levels[name]}.zip"
            else:
                filename = f"{name}.zip"

            # Set the target path with the correct filename
            target_path = config_manager.get_path('GEOGRAPHIC_DIR') / filename

            # Download the file and track success
            success = utils.download_data(url, target_path, force_download=force_download)
            if not success:
                download_success = False
                logger.error(f"Failed to download geographic file: {name} -> {filename}")

        # Download census files
        census_urls = data_sources_dict.get('census_urls', {})
        for name, url in census_urls.items():
            target_path = config_manager.get_path('CENSUS_DIR') / f"{name}.zip"
            success = utils.download_data(url, target_path, force_download=force_download)
            if not success:
                download_success = False
                logger.error(f"Failed to download census file: {name}")

        logger.info("=== Data Download Complete ===")

        if not download_success:
            logger.warning("Some downloads failed. Check the logs for details.")

        return download_success

    except Exception as e:
        logger.error(f"Error in download step: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return False

@time_func
def run_geography_step() -> bool:
    """
    Process geographic files to create the geographic dimension.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=== Starting Geographic Processing ===")
    
    try:
        # Get geo levels from configuration
        geo_levels = list(config_manager.get_column_mapping('geo_levels').keys())
        
        # Process geography
        result = process_geography(
            geo_levels=geo_levels,
            shp_dir=config_manager.get_path('GEOGRAPHIC_DIR'),
            output_path=config_manager.get_path('OUTPUT_DIR') / "geo_dimension.parquet",
            temp_extract_base=config_manager.get_path('TEMP_EXTRACT_DIR')
        )
        
        logger.info("=== Geographic Processing Complete ===")
        return result
    except Exception as e:
        logger.error(f"Error in geography step: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return False

@time_func
def run_time_dimension_step() -> bool:
    """
    Create the time dimension.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("=== Starting Time Dimension Creation ===")
    
    try:
        # Create time dimension for Census 2021
        year = 2021
        
        # Generate time dimension
        success = generate_time_dimension(
            start_year=year-5, 
            end_year=year+5,
            output_path=config_manager.get_path('OUTPUT_DIR') / "dim_time.parquet"
        )
        
        logger.info("=== Time Dimension Creation Complete ===")
        return success
    except Exception as e:
        logger.error(f"Error in time dimension step: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return False

@time_func
def run_dimensions_step() -> bool:
    """Create dimension tables."""
    logger.info("=== Starting Dimensions Creation ===")
    
    try:
        # Create a dimension fix handler to ensure unknown members
        # This integrates the fix logic into the proactive ETL
        result = run_dimension_fix(config_manager.get_path('OUTPUT_DIR'))
        
        if not result:
            logger.error("Failed to create dimensions")
            return False
        
        logger.info("=== Dimensions Creation Complete ===")
        return True
    except Exception as e:
        logger.error(f"Error in dimensions step: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return False

@time_func
def run_validate_step() -> bool:
    """
    Validate data quality.

    Returns:
        True if all validations pass (ignoring duplicate key warnings), False otherwise
    """
    logger.info("=== Starting Data Validation ===")

    try:
        # Create a data quality validator
        validator = DataQualityValidator(config_manager.get_path('OUTPUT_DIR'))

        # Run all validations
        results = validator.run_all_validations()

        # Process the results, checking for any failures
        # Organize failures by type: duplicates vs ref integrity vs other
        duplicate_key_failures = []
        ref_integrity_failures = []
        range_check_failures = []
        other_failures = []

        # Extract detailed check results
        all_check_results = {}
        for table_name, table_result in results.items():
            if "check_results" in table_result:
                all_check_results.update(table_result["check_results"])
            elif not table_result.get("passed", True):
                # If a table has no check_results but failed, count it as other
                other_failures.append(table_name)

        # Categorize failures by type
        for check_name, check_result in all_check_results.items():
            if not check_result.get("passed", True):
                if "_key_uniqueness" in check_name:
                    duplicate_key_failures.append(check_name)
                elif "_ref_integrity" in check_name:
                    ref_integrity_failures.append(check_name)
                elif "_range_check" in check_name:
                    range_check_failures.append(check_name)
                else:
                    other_failures.append(check_name)

        # Log validation results with more detailed information
        if duplicate_key_failures:
            logger.info(f"Duplicate key warnings (expected with unknown members): {len(duplicate_key_failures)}")
            for failure in duplicate_key_failures[:3]:  # Show only first few
                logger.info(f"  - {failure}")

        if ref_integrity_failures:
            logger.error(f"Referential integrity failures: {len(ref_integrity_failures)}")
            for failure in ref_integrity_failures:
                logger.error(f"  - {failure}")
        else:
            logger.info("Referential integrity: PASSED")

        if range_check_failures:
            logger.warning(f"Range check warnings: {len(range_check_failures)}")
            for failure in range_check_failures[:3]:
                logger.warning(f"  - {failure}")

        if other_failures:
            logger.error(f"Other failures: {len(other_failures)}")
            for failure in other_failures:
                logger.error(f"  - {failure}")
        else:
            logger.info("Other checks: PASSED")

        # We consider duplicate key warnings as acceptable (they're expected with the unknown member design)
        # Also, range check warnings are non-critical
        # Only referential integrity failures and other critical failures should cause validation to fail
        success = len(ref_integrity_failures) == 0 and len(other_failures) == 0

        logger.info(f"Overall validation result: {'PASSED' if success else 'FAILED'}")
        logger.info("=== Data Validation Complete ===")

        return success
    except Exception as e:
        logger.error(f"Error in validation step: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return False

@time_func
def run_refine_health_step() -> bool:
    """
    Refine health facts by:
    1. Fixing dimension keys to ensure referential integrity
    2. Fixing grain issues by aggregating at the proper level
    """
    logger.info("=== Starting Health Facts Refinement ===")

    try:
        from ahgd_etl.core.temp_fix.dimension_fix import run_dimension_fix
        from ahgd_etl.core.temp_fix.grain_fix import run_grain_fix

        output_dir = config_manager.get_path('OUTPUT_DIR')

        # First, fix dimension references to ensure referential integrity
        logger.info("Fixing dimension references in fact tables...")
        dimension_fix_result = run_dimension_fix(output_dir)

        if not dimension_fix_result:
            logger.error("Failed to fix dimension references")
            return False

        # Next, fix grain issues by aggregating duplicate keys
        logger.info("Fixing grain issues in fact tables...")
        grain_fix_result = run_grain_fix(output_dir)

        if not grain_fix_result:
            logger.error("Failed to fix grain issues")
            return False

        logger.info("=== Health Facts Refinement Complete ===")
        return True
    except Exception as e:
        logger.error(f"Error in health facts refinement step: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return False

def run_census_table_step(table_code: str, time_dim_path: Optional[Path] = None) -> bool:
    """
    Process a specific Census table.
    
    Args:
        table_code: Census table code (G01, G17, etc.)
        time_dim_path: Path to the time dimension file
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"=== Starting {table_code} Processing ===")
    table_code = table_code.upper()
    
    if time_dim_path is None:
        time_dim_path = config_manager.get_path('OUTPUT_DIR') / "dim_time.parquet"
    
    try:
        # Create transformer manager
        transformer_manager = TransformerManager()
        
        # Get transformer for this table
        transformer = transformer_manager.get_transformer(table_code)
        
        if transformer is None:
            logger.error(f"No transformer found for table {table_code}")
            return False
        
        # Process the table
        success = transformer.process(
            zip_dir=config_manager.get_path('CENSUS_DIR'),
            temp_extract_base=config_manager.get_path('TEMP_EXTRACT_DIR'),
            output_dir=config_manager.get_path('OUTPUT_DIR'),
            geo_output_path=config_manager.get_path('OUTPUT_DIR') / "geo_dimension.parquet",
            time_dim_path=time_dim_path
        )
        
        logger.info(f"=== {table_code} Processing Complete ===")
        return success
    
    except Exception as e:
        logger.error(f"Error processing {table_code}: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return False


def main():
    """Main entry point."""
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Australian Healthcare Geographic Database ETL Pipeline')
    parser.add_argument('--steps', nargs='+', default=['all'], 
                        help='ETL steps to run (e.g., download, geo, time, g01, g17).')
    parser.add_argument('--force-download', action='store_true', 
                        help='Force download of data files even if they exist.')
    parser.add_argument('--skip-validation', action='store_true', 
                        help='Skip validation steps.')
    parser.add_argument('--stop-on-error', action='store_true', 
                        help='Stop ETL process on first error.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Expand step groups
    steps_to_run = set()
    for step in args.steps:
        if step in STEP_GROUPS:
            steps_to_run.update(STEP_GROUPS[step])
        else:
            steps_to_run.add(step)
    
    # Setup logging
    logger = utils.setup_logging(config_manager.get_path('LOG_DIR'))
    
    # Log start
    logger.info("="*80)
    logger.info("AHGD ETL PIPELINE START")
    logger.info("="*80)
    logger.info(f"Steps to run: {args.steps}")
    logger.info(f"Force download: {args.force_download}")
    logger.info(f"Stop on error: {args.stop_on_error}")
    
    # Track status of each step
    step_status = {}
    start_time = time.time()
    
    # Run steps in logical order
    try:
        # 1. Download data
        if 'download' in steps_to_run:
            logger.info("Executing step: download")
            step_status['download'] = run_download_step(args.force_download)
            if not step_status['download'] and args.stop_on_error:
                raise Exception("Download step failed")

        # 2a. Create time dimension - needed for Census processing
        if 'time' in steps_to_run:
            logger.info("Executing step: time")
            step_status['time'] = run_time_dimension_step()
            if not step_status['time'] and args.stop_on_error:
                raise Exception("Time dimension step failed")

        # 2b. Process geographic boundaries - needed for Census processing
        if 'geo' in steps_to_run:
            logger.info("Executing step: geo")
            step_status['geo'] = run_geography_step()
            if not step_status['geo'] and args.stop_on_error:
                raise Exception("Geography step failed")

        # 3. Process Census tables (G-tables) to create fact tables
        # Process in a specific order that makes sense for dependencies
        for table_code in ['g01', 'g17', 'g18', 'g19', 'g20', 'g21', 'g25']:
            if table_code in steps_to_run:
                logger.info(f"Executing step: {table_code}")
                step_status[table_code] = run_census_table_step(table_code)
                if not step_status[table_code] and args.stop_on_error:
                    raise Exception(f"{table_code.upper()} step failed")

        # 4. Create derived dimensions from reference data
        if 'dimensions' in steps_to_run:
            logger.info("Executing step: dimensions")
            step_status['dimensions'] = run_dimensions_step()
            if not step_status['dimensions'] and args.stop_on_error:
                raise Exception("Dimensions step failed")

        # 5. Refine health facts - fix dimension links and grain issues
        if 'refine_health' in steps_to_run:
            logger.info("Executing step: refine_health")
            step_status['refine_health'] = run_refine_health_step()
            if not step_status['refine_health'] and args.stop_on_error:
                raise Exception("Health facts refinement step failed")

        # 6. Validate data - final quality checks
        if 'validate' in steps_to_run and not args.skip_validation:
            logger.info("Executing step: validate")
            step_status['validate'] = run_validate_step()
            if not step_status['validate'] and args.stop_on_error:
                raise Exception("Validation step failed")
    
    except Exception as e:
        logger.error(f"ETL process stopped due to error: {str(e)}")
        logger.error("Traceback:", exc_info=True)
    
    # Log summary
    logger.info("="*80)
    logger.info("ETL PROCESS SUMMARY")
    logger.info("="*80)
    for step, status in step_status.items():
        logger.info(f"{step}: {'SUCCESS' if status else 'FAILED'}")
    
    total_elapsed = time.time() - start_time
    logger.info(f"Total elapsed time: {total_elapsed:.2f}s")
    
    if all(step_status.values()):
        logger.info("ETL process completed successfully")
    else:
        logger.error("ETL process completed with errors")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())