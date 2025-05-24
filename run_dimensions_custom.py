#!/usr/bin/env python3
"""
Custom script to run dimensions step with a modified version of dimension_fix.
This avoids the type mismatch error in the dimension_fix module.
"""

import logging
import sys
from pathlib import Path

# Import run_etl_enhanced module for pipeline components
sys.path.append('/Users/massimoraso/Code/AHGD')
from ahgd_etl.config.settings import get_config_manager
from ahgd_etl import utils

# Set up logging
config_manager = get_config_manager()
logger = utils.setup_logging(config_manager.get_path('LOG_DIR'))

def modified_run_dimensions_step() -> bool:
    """Create dimension tables - modified version that skips fact table references."""
    logger.info("=== Starting Dimensions Creation (Custom) ===")
    
    try:
        from ahgd_etl.core.temp_fix.dimension_fix import DimensionHandler
        
        # Initialize handler
        output_dir = config_manager.get_path('OUTPUT_DIR')
        handler = DimensionHandler(output_dir=output_dir)
        
        # Load all dimensions
        handler.load_dimensions()
        
        # Ensure unknown members exist in all dimensions
        handler.ensure_unknown_members()
        
        # Skip fact table reference fixing since we've already done it separately
        logger.info("=== Dimensions Creation Complete (Custom) ===")
        return True
    
    except Exception as e:
        logger.error(f"Error in custom dimensions step: {str(e)}")
        logger.error("Traceback:", exc_info=True)
        return False

def main():
    """Main function."""
    # Run our custom dimensions step
    success = modified_run_dimensions_step()
    
    # Log result
    if success:
        logger.info("Dimensions step completed successfully")
    else:
        logger.error("Dimensions step failed")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())