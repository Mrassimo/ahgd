#!/usr/bin/env python3
"""
AHGD ETL Data Architecture Fix Master Script

This script orchestrates the execution of multiple fix modules to address 
data architecture issues in the AHGD ETL pipeline:

1. Schema Validation and Enforcement: Ensures all tables have correct columns and data types
2. Dimension Handling: Fixes dimension tables and surrogate key issues
3. Fact Table Grain: Resolves duplicate key issues in fact tables

Usage:
    python fix_all.py --output-dir output [--steps all|schema|dimension|grain] [--log-file fix_all.log]

Each step can be run individually or all together, and progress is logged
to both console and the specified log file.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import fix modules
from ahgd_etl.core.temp_fix.schema_fix import run_schema_fix
from ahgd_etl.core.temp_fix.dimension_fix import run_dimension_fix
from ahgd_etl.core.temp_fix.grain_fix import run_grain_fix

def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging to both console and file.
    
    Args:
        log_file: Path to log file (optional)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)  # More detailed in file
        file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

def run_fix_steps(steps: List[str], output_dir: Path) -> Dict[str, bool]:
    """
    Run the specified fix steps.
    
    Args:
        steps: List of steps to run ('schema', 'dimension', 'grain', or 'all')
        output_dir: Directory containing data files to fix
        
    Returns:
        Dictionary mapping step names to success status
    """
    # Validate output directory
    if not output_dir.exists():
        logger.error(f"Output directory {output_dir} does not exist")
        return {step: False for step in steps}
    
    results = {}
    run_all = 'all' in steps
    
    # Step 1: Schema Validation and Fix
    if run_all or 'schema' in steps:
        logger.info("=" * 80)
        logger.info("STEP 1: Schema Validation and Fix")
        logger.info("=" * 80)
        
        start_time = time.time()
        schema_result = run_schema_fix(output_dir)
        elapsed = time.time() - start_time
        
        results['schema'] = schema_result
        status = "SUCCESS" if schema_result else "FAILED"
        logger.info(f"Schema step completed with status: {status} (elapsed: {elapsed:.2f}s)")
    
    # Step 2: Dimension Handling
    if run_all or 'dimension' in steps:
        logger.info("=" * 80)
        logger.info("STEP 2: Dimension Handling")
        logger.info("=" * 80)
        
        start_time = time.time()
        dimension_result = run_dimension_fix(output_dir)
        elapsed = time.time() - start_time
        
        results['dimension'] = dimension_result
        status = "SUCCESS" if dimension_result else "FAILED"
        logger.info(f"Dimension step completed with status: {status} (elapsed: {elapsed:.2f}s)")
    
    # Step 3: Fact Table Grain Fix
    if run_all or 'grain' in steps:
        logger.info("=" * 80)
        logger.info("STEP 3: Fact Table Grain Fix")
        logger.info("=" * 80)
        
        start_time = time.time()
        grain_result = run_grain_fix(output_dir)
        elapsed = time.time() - start_time
        
        results['grain'] = grain_result
        status = "SUCCESS" if grain_result else "FAILED"
        logger.info(f"Grain step completed with status: {status} (elapsed: {elapsed:.2f}s)")
    
    # Log overall summary
    logger.info("=" * 80)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 80)
    
    all_success = all(results.values())
    for step, success in results.items():
        logger.info(f"{step.upper()}: {'SUCCESS' if success else 'FAILED'}")
    
    logger.info("-" * 80)
    logger.info(f"OVERALL: {'SUCCESS' if all_success else 'FAILED'}")
    
    return results

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="AHGD ETL Data Architecture Fix Script")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="output",
        help="Directory containing data files to fix (default: 'output')"
    )
    parser.add_argument(
        "--steps", 
        nargs="+", 
        choices=["all", "schema", "dimension", "grain"],
        default=["all"], 
        help="Steps to run (default: 'all')"
    )
    parser.add_argument(
        "--log-file", 
        type=str, 
        default="fix_all.log",
        help="Log file path (default: 'fix_all.log')"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    logger.info("=" * 80)
    logger.info("AHGD ETL DATA ARCHITECTURE FIX")
    logger.info("=" * 80)
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Steps to Run: {', '.join(args.steps)}")
    logger.info(f"Log File: {args.log_file}")
    logger.info("=" * 80)
    
    # Run the fix steps
    output_dir = Path(args.output_dir)
    results = run_fix_steps(args.steps, output_dir)
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)