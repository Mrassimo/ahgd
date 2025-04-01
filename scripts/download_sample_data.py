#!/usr/bin/env python

import logging
import requests
import zipfile
import io
from pathlib import Path
import os
import pandas as pd
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Sample data URLs - these would normally be real ABS data URLs
# For this demo, we'll create sample data locally
SAMPLE_DATA_DIR = Path("data/raw/Census/sample")

def ensure_dirs():
    """Ensure necessary directories exist."""
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    logger.info(f"Created directory: {SAMPLE_DATA_DIR}")

def create_sample_g17_data():
    """Create a sample G17 (Need for Assistance) dataset."""
    logger.info("Creating sample G17 (Need for Assistance) data...")
    
    # Sample data for G17 based on existing patterns
    g17_data = {
        "SA2_CODE_2021": [10101, 10102, 10103, 10104, 10105],
        "Has_Need_for_Assistance_P": [120, 150, 89, 210, 95],
        "Does_Not_Have_Need_for_Assistance_P": [3000, 3500, 2800, 4100, 2600],
        "Need_for_Assistance_Not_Stated_P": [45, 80, 32, 96, 28]
    }
    
    # Create a sample DataFrame
    g17_df = pd.DataFrame(g17_data)
    
    # Save to CSV
    g17_path = SAMPLE_DATA_DIR / "G17_Sample_SA2.csv"
    g17_df.to_csv(g17_path, index=False)
    logger.info(f"Saved sample G17 data to {g17_path}")
    
    # Create an alternative version with different column names
    g17_alt_data = {
        "SA2_CODE_2021": [10101, 10102, 10103, 10104, 10105],
        "P_Tot_Has_need_for_assistance": [120, 150, 89, 210, 95],
        "P_Tot_Does_not_have_need_for_assistance": [3000, 3500, 2800, 4100, 2600],
        "P_Tot_Need_for_assistance_not_stated": [45, 80, 32, 96, 28]
    }
    
    g17_alt_df = pd.DataFrame(g17_alt_data)
    g17_alt_path = SAMPLE_DATA_DIR / "G17_Sample_Alt_SA2.csv"
    g17_alt_df.to_csv(g17_alt_path, index=False)
    logger.info(f"Saved alternative sample G17 data to {g17_alt_path}")

def create_sample_g18_data():
    """Create a sample G18 (Unpaid Care) dataset."""
    logger.info("Creating sample G18 (Unpaid Care) data...")
    
    # Sample data for G18
    g18_data = {
        "SA2_CODE_2021": [10101, 10102, 10103, 10104, 10105],
        "Provided_Unpaid_Care_P": [520, 610, 480, 730, 390],
        "Did_Not_Provide_Unpaid_Care_P": [2500, 2900, 2300, 3500, 2200],
        "Unpaid_Care_Not_Stated_P": [145, 220, 141, 176, 133]
    }
    
    # Create a sample DataFrame
    g18_df = pd.DataFrame(g18_data)
    
    # Save to CSV
    g18_path = SAMPLE_DATA_DIR / "G18_Sample_SA2.csv"
    g18_df.to_csv(g18_path, index=False)
    logger.info(f"Saved sample G18 data to {g18_path}")
    
    # Create an alternative version with different column names
    g18_alt_data = {
        "SA2_CODE_2021": [10101, 10102, 10103, 10104, 10105],
        "P_Tot_Provided_unpaid_care": [520, 610, 480, 730, 390],
        "P_Tot_No_unpaid_care_provided": [2500, 2900, 2300, 3500, 2200],
        "P_Tot_Unpaid_care_not_stated": [145, 220, 141, 176, 133]
    }
    
    g18_alt_df = pd.DataFrame(g18_alt_data)
    g18_alt_path = SAMPLE_DATA_DIR / "G18_Sample_Alt_SA2.csv"
    g18_alt_df.to_csv(g18_alt_path, index=False)
    logger.info(f"Saved alternative sample G18 data to {g18_alt_path}")

def main():
    """Main function to download and set up sample data."""
    logger.info("Setting up sample data for G17 and G18 testing...")
    
    try:
        # Ensure directories exist
        ensure_dirs()
        
        # Create sample data
        create_sample_g17_data()
        create_sample_g18_data()
        
        logger.info("Sample data setup complete. You can now run the test scripts.")
        
    except Exception as e:
        logger.error(f"Error setting up sample data: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 