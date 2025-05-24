#!/usr/bin/env python3
"""
AHGD Unified ETL Runner

This is the new single entry point for running the AHGD ETL pipeline.
It replaces all the legacy runners and includes inline data quality fixes.

Usage:
    python run_unified_etl.py [options]
    
Examples:
    # Run full pipeline
    python run_unified_etl.py
    
    # Run specific steps
    python run_unified_etl.py --steps geo time dimensions
    
    # Export to Snowflake
    python run_unified_etl.py --mode export --snowflake-config snowflake/config.json
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the unified CLI
from ahgd_etl.cli.main import main

if __name__ == "__main__":
    main()