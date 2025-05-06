#!/usr/bin/env python3
import pandas as pd
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing etl_logic
project_root = Path(__file__).resolve().parents[2] # Assuming scripts/analysis/ is two levels down
sys.path.insert(0, str(project_root))

# Import config and utils after setting path
from etl_logic import config
from etl_logic import utils # Assuming setup_logging is in utils

# Setup logging (optional, adjust level as needed)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths using config
METADATA_DIR = config.PATHS.get('METADATA_DIR', config.PATHS['RAW_DATA_DIR'] / "Metadata")
METADATA_FILE = METADATA_DIR / "Metadata_2021_GCP_DataPack_R1_R2.xlsx"
TEMPLATE_FILE = METADATA_DIR / "2021_GCP_Sequential_Template_R2.xlsx"

def explore_excel_file(file_path: Path, description: str):
    """Explores an Excel file, printing sheet names and head of each sheet."""
    logger.info(f"\n--- Exploring {description}: {file_path.name} ---")
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return

    try:
        xlsx = pd.ExcelFile(file_path)
        sheet_names = xlsx.sheet_names
        logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")

        for sheet in sheet_names[:5]: # Limit to first 5 sheets for brevity
            logger.info(f"\n  --- Sheet: {sheet} (Head) ---")
            try:
                df = pd.read_excel(xlsx, sheet_name=sheet, nrows=10)
                print(df.to_string())
            except Exception as e:
                logger.error(f"    Could not read sheet '{sheet}': {e}")

    except Exception as e:
        logger.error(f"Error reading Excel file '{file_path.name}': {e}")

def main():
    """Main function to explore metadata files."""
    logger.info("=== Starting Metadata Exploration Script ===")

    # Explore the main metadata file
    explore_excel_file(METADATA_FILE, "Main Metadata File")

    # Explore the template file
    explore_excel_file(TEMPLATE_FILE, "Template File")

    logger.info("=== Metadata Exploration Complete ===")

if __name__ == "__main__":
    main()