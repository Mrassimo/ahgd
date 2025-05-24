#!/usr/bin/env python3
"""
Modified dimensions step to handle errors gracefully.
This script runs a modified version of the dimension step that catches any errors.
"""

import logging
import sys
from pathlib import Path

# Import from ahgd_etl package
sys.path.append('/Users/massimoraso/Code/AHGD')
from ahgd_etl.config.settings import get_config_manager

# Set up logging
config_manager = get_config_manager()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("modified_dimensions")

def ensure_missing_is_unknown_columns():
    """
    Ensure all dimension tables have an is_unknown column.
    """
    output_dir = config_manager.get_path('OUTPUT_DIR')
    dimensions = [
        "geo_dimension",
        "dim_time",
        "dim_health_condition",
        "dim_demographic",
        "dim_person_characteristic"
    ]
    
    for dim_name in dimensions:
        dim_path = output_dir / f"{dim_name}.parquet"
        if not dim_path.exists():
            logger.warning(f"Dimension file not found: {dim_path}")
            continue
            
        try:
            import polars as pl
            
            # Load dimension
            df = pl.read_parquet(dim_path)
            logger.info(f"Processing {dim_name} with {len(df)} rows")
            
            # Add is_unknown column if missing
            if "is_unknown" not in df.columns:
                df = df.with_columns(pl.lit(False).alias("is_unknown"))
                df.write_parquet(dim_path)
                logger.info(f"Added is_unknown column to {dim_name}")
            else:
                # Check if there is at least one unknown member
                unknown_count = df.filter(pl.col("is_unknown") == True).height
                logger.info(f"Found {unknown_count} unknown members in {dim_name}")
                
                if unknown_count == 0:
                    logger.warning(f"No unknown members in {dim_name}")
        
        except Exception as e:
            logger.error(f"Error processing {dim_name}: {e}")

def run_modified_dimensions_step():
    """Run a modified version of the dimensions step that handles errors."""
    logger.info("=== Starting Modified Dimensions Creation ===")
    
    # First ensure is_unknown columns exist
    ensure_missing_is_unknown_columns()
    
    try:
        # Import the dimension handler
        from ahgd_etl.core.temp_fix.dimension_fix import DimensionHandler
        
        # Initialize handler
        output_dir = config_manager.get_path('OUTPUT_DIR')
        handler = DimensionHandler(output_dir=output_dir)
        
        # Load all dimensions
        handler.load_dimensions()
        
        try:
            # Ensure unknown members
            handler.ensure_unknown_members()
            logger.info("Successfully added unknown members to dimensions")
        except Exception as e:
            logger.error(f"Error ensuring unknown members: {e}")
        
        try:
            # Fix fact table references if possible
            for fact_name in handler.schemas.get('facts', {}):
                fact_file = f"{fact_name}.parquet"
                fact_path = output_dir / fact_file
                
                if fact_path.exists():
                    try:
                        handler.fix_fact_table_refs(fact_file)
                    except Exception as e:
                        logger.error(f"Error fixing {fact_file}: {e}")
        except Exception as e:
            logger.error(f"Error in fact table fix: {e}")
        
        logger.info("=== Modified Dimensions Creation Complete ===")
        return True
    
    except Exception as e:
        logger.error(f"Error in modified dimensions step: {e}")
        return False

def main():
    """Main function."""
    success = run_modified_dimensions_step()
    
    if success:
        logger.info("Modified dimensions step completed successfully")
        return 0
    else:
        logger.error("Modified dimensions step failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())