#!/usr/bin/env python

import logging
import requests
import zipfile
import io
from pathlib import Path
import os
import pandas as pd
import time
import sys

# Add project root to sys.path to allow importing etl_logic
project_root = Path(__file__).resolve().parents[2] # Assuming scripts/test_utilities/ is two levels down
sys.path.insert(0, str(project_root))

# Import config and utils after setting path
from etl_logic import config
from etl_logic import utils # Assuming setup_logging is in utils

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the directory for sample data using config paths
# Create a subdirectory within the CENSUS_DIR or a dedicated TEST_DATA_DIR if configured
SAMPLE_DATA_BASE_DIR = config.PATHS.get('TEST_DATA_DIR', config.PATHS['CENSUS_DIR'])
SAMPLE_DATA_DIR = SAMPLE_DATA_BASE_DIR / "sample"

def ensure_dirs():
    """Ensure necessary directories exist."""
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
    logger.info(f"Created directory: {SAMPLE_DATA_DIR}")

def create_g17_sample_data(output_dir: Path):
    """Creates sample G17 CSV files."""
    logger.info(f"Creating sample G17 data in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample Data 1 (Standard Headers)
    data1 = {
        'SA2_CODE_2021': ['20101', '20102', '20103'],
        'Median_Tot_weekly_personal_income_A': [650, 720, 800],
        'Median_Tot_weekly_family_income_A': [1500, 1650, 1800],
        'Median_Tot_weekly_household_income_A': [1400, 1550, 1700]
    }
    df1 = pd.DataFrame(data1)
    file1 = output_dir / "G17_Sample_SA2.csv"
    df1.to_csv(file1, index=False)
    logger.info(f"Created: {file1.name}")

    # Sample Data 2 (Alternative Headers)
    data2 = {
        'SA2_CODE21': ['20201', '20202'],
        'Median_positive_zero_tot_inc_weekly_A': [700, 750],
        'Median_Tot_Wkly_Fam_Income': [1600, 1750],
        'Median_Tot_Wkly_Hhold_Income': [1500, 1650]
    }
    df2 = pd.DataFrame(data2)
    file2 = output_dir / "G17_Sample_Alt_SA2.csv"
    df2.to_csv(file2, index=False)
    logger.info(f"Created: {file2.name}")

def create_g18_sample_data(output_dir: Path):
    """Creates sample G18 CSV files."""
    logger.info(f"Creating sample G18 data in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample Data 1 (Standard Headers)
    data1 = {
        'SA2_CODE_2021': ['20101', '20102', '20103'],
        'P_Need_for_assistance': [100, 150, 120],
        'M_Need_for_assistance': [40, 60, 50],
        'F_Need_for_assistance': [60, 90, 70],
        'P_No_need_for_assistance': [2000, 2500, 2200],
        'M_No_need_for_assistance': [950, 1200, 1050],
        'F_No_need_for_assistance': [1050, 1300, 1150],
        'P_Need_for_assistance_ns': [10, 15, 12],
        'M_Need_for_assistance_ns': [5, 7, 6],
        'F_Need_for_assistance_ns': [5, 8, 6]
    }
    df1 = pd.DataFrame(data1)
    file1 = output_dir / "G18_Sample_SA2.csv"
    df1.to_csv(file1, index=False)
    logger.info(f"Created: {file1.name}")

    # Sample Data 2 (Alternative Headers / Structure - if applicable)
    # If G18 has known variations, add another sample here
    # data2 = { ... }
    # df2 = pd.DataFrame(data2)
    # file2 = output_dir / "G18_Sample_Alt_SA2.csv"
    # df2.to_csv(file2, index=False)
    # logger.info(f"Created: {file2.name}")

def main():
    """Main function to generate all sample data files."""
    logger.info(f"=== Generating Sample Census Data Files ==~")
    logger.info(f"Target directory: {SAMPLE_DATA_DIR}")

    # Create sample data for each table
    create_g17_sample_data(SAMPLE_DATA_DIR)
    create_g18_sample_data(SAMPLE_DATA_DIR)
    # Add calls to create other sample data (G19, G20, etc.) here
    # create_g19_sample_data(SAMPLE_DATA_DIR)
    # create_g20_sample_data(SAMPLE_DATA_DIR)
    # ...

    logger.info("=== Sample Data Generation Complete ===")

if __name__ == "__main__":
    main() 