#!/usr/bin/env python3
import pandas as pd
import os
import zipfile
import re
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

# Define the directory containing the metadata Excel files
METADATA_DIR = config.PATHS.get('METADATA_DIR', config.PATHS['RAW_DATA_DIR'] / "Metadata")

# Define the primary metadata file
METADATA_FILE = METADATA_DIR / "Metadata_2021_GCP_DataPack_R1_R2.xlsx"


def extract_table_info(metadata_file_path: Path, table_codes: list):
    """Extract information about specific tables from the 'List of tables' sheet."""
    logger.info(f"Extracting info for tables {table_codes} from: {metadata_file_path}")
    if not metadata_file_path.exists():
        logger.error(f"Metadata file not found: {metadata_file_path}")
        return

    try:
        tables_df = pd.read_excel(metadata_file_path, sheet_name="List of tables")
        logger.info(f"Columns in 'List of tables': {tables_df.columns.tolist()}")

        for code in table_codes:
            logger.info(f"\n--- Information for Table {code} ---")
            # Search all columns for the table code
            table_info = tables_df[tables_df.apply(lambda row: row.astype(str).str.contains(code, na=False, case=False).any(), axis=1)]

            if not table_info.empty:
                print(table_info.to_string(index=False))
            else:
                logger.warning(f"Table {code} not found in 'List of tables' sheet.")

    except Exception as e:
        logger.error(f"Error reading or processing 'List of tables' sheet: {e}")

def examine_csv_in_zip(zip_filepath: Path, csv_pattern: str):
    """Examines the structure of CSV files matching a pattern within a ZIP file."""
    logger.info(f"\n--- Examining CSVs matching '{csv_pattern}' in: {zip_filepath.name} ---")
    if not zip_filepath.exists():
        logger.error(f"ZIP file not found: {zip_filepath}")
        return

    found_csv = False
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if re.search(csv_pattern, file_info.filename, re.IGNORECASE):
                    found_csv = True
                    logger.info(f"Found matching CSV: {file_info.filename}")
                    try:
                        # Read header and first few rows directly from ZIP
                        with zip_ref.open(file_info.filename) as csv_file:
                            # Use low_memory=False for potentially mixed types
                            df = pd.read_csv(csv_file, nrows=5, low_memory=False)
                            logger.info("First 5 rows (including header):")
                            print(df.to_string())
                            logger.info(f"Column Names ({len(df.columns)} total): {df.columns.tolist()}")
                            logger.info("Data Types (Pandas inferred):")
                            print(df.dtypes)
                            # Only process the first matching CSV found for brevity
                            break
                    except Exception as e:
                        logger.error(f"Error reading CSV '{file_info.filename}' from ZIP: {e}")
                        break # Stop if error reading the first match
            if not found_csv:
                logger.warning(f"No CSV files matching pattern '{csv_pattern}' found in the ZIP.")
    except Exception as e:
        logger.error(f"Error opening or reading ZIP file {zip_filepath.name}: {e}")

def main():
    """Main function to run extraction and examination."""
    logger.info("=== Starting G18/G19 Metadata and Structure Analysis ===")

    # Extract info from metadata Excel file
    extract_table_info(METADATA_FILE, ['G18', 'G19'])

    # Define path to the main Census ZIP file using config
    # Assuming the filename is known or defined in config (e.g., under DATA_URLS key)
    # Construct the expected filename
    gcp_all_filename = "2021_GCP_all_for_AUS_short-header.zip" # Assuming this is the standard name
    census_zip_path = config.PATHS['CENSUS_DIR'] / gcp_all_filename

    # Examine structure of G18 and G19 CSVs within the ZIP
    examine_csv_in_zip(census_zip_path, r"G18.*\.csv$")
    examine_csv_in_zip(census_zip_path, r"G19.*\.csv$")

    logger.info("=== Analysis Complete ===")

if __name__ == "__main__":
    main() 