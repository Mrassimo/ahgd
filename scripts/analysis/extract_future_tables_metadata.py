#!/usr/bin/env python3
import pandas as pd
import os
import sys
from pathlib import Path
import zipfile
import re
import json

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
# Use METADATA_DIR from config if defined, otherwise construct from RAW_DATA_DIR
METADATA_DIR = config.PATHS.get('METADATA_DIR', config.PATHS['RAW_DATA_DIR'] / "Metadata")

# List of known metadata files (can be updated)
METADATA_FILES = [
    METADATA_DIR / "Metadata_2021_GCP_DataPack_R1_R2.xlsx",
    # Add other potential metadata file names here if needed
]

# List of tables to extract (add future table codes here)
TARGET_TABLES = [
    'G22', 'G23', 'G24', 'G26', 'G27', 'G28', 'G29', 'G30', 'G31', 'G32', 'G33', 'G34'
    # Add other tables of interest, e.g., G02, G03, ...
]

def find_metadata_file() -> Optional[Path]:
    """Finds the first existing metadata file from the list."""
    for file_path in METADATA_FILES:
        if file_path.exists():
            logger.info(f"Using metadata file: {file_path}")
            return file_path
    logger.error("No metadata file found in the expected locations.")
    logger.error(f"Expected locations: {[str(p) for p in METADATA_FILES]}")
    return None

def extract_table_metadata(metadata_file_path: Path, table_code: str) -> Optional[dict]:
    """Extracts metadata for a specific table code from the metadata Excel file."""
    logger.info(f"Extracting metadata for table: {table_code}")
    try:
        # Check the 'List of tables' sheet first
        try:
            tables_df = pd.read_excel(metadata_file_path, sheet_name="List of tables")
            # Search all columns for the table code (flexible matching)
            table_info = tables_df[tables_df.apply(lambda row: row.astype(str).str.contains(table_code, na=False, case=False).any(), axis=1)]
            if not table_info.empty:
                # Try to extract a meaningful description
                # Common description columns: 'Table Title', 'Description', 'Table name'
                desc_col = next((col for col in ['Table Title', 'Description', 'Table name', tables_df.columns[1]] if col in table_info.columns), None)
                description = table_info.iloc[0][desc_col] if desc_col else "No description found"
                logger.debug(f"  Found {table_code} in 'List of tables': {description}")
                return {
                    'table_code': table_code,
                    'description': description,
                    'found_in': 'List of tables'
                }
            else:
                logger.debug(f"  {table_code} not found in 'List of tables' sheet.")
        except Exception as e:
            logger.warning(f"Could not process 'List of tables' sheet: {e}")

        # If not found in 'List of tables', check if there's a dedicated sheet
        try:
            xlsx = pd.ExcelFile(metadata_file_path)
            if table_code in xlsx.sheet_names:
                logger.info(f"  Found dedicated sheet for {table_code}. Reading structure...")
                df = pd.read_excel(metadata_file_path, sheet_name=table_code, nrows=10)
                # Extract title (often in row 2 or 3, column 0)
                title = "Unknown Title"
                for i in range(min(5, df.shape[0])):
                    cell_value = str(df.iloc[i, 0])
                    if table_code in cell_value or 'table' in cell_value.lower():
                        title = cell_value
                        break
                # Basic column structure preview
                columns = list(df.columns)
                sample_row = list(df.iloc[df.first_valid_index() + 1]) if df.first_valid_index() is not None and df.shape[0] > df.first_valid_index() + 1 else []

                logger.debug(f"  Sheet title guess: {title}")
                logger.debug(f"  Columns: {columns[:10]}...")
                return {
                    'table_code': table_code,
                    'description': title, # Use sheet title as description
                    'found_in': f'Sheet: {table_code}',
                    'columns_preview': columns,
                    'sample_row_preview': sample_row
                }
            else:
                 logger.debug(f"  No dedicated sheet found for {table_code}.")
        except Exception as e:
            logger.warning(f"Could not process sheet '{table_code}': {e}")

        logger.warning(f"Could not find metadata for {table_code} in the file.")
        return None

    except Exception as e:
        logger.error(f"Failed to read or process metadata file {metadata_file_path}: {e}")
        return None

def main():
    """Main function to extract metadata for target future tables."""
    logger.info("=== Starting Future Tables Metadata Extraction ===")

    metadata_file = find_metadata_file()
    if not metadata_file:
        return # Error logged in find_metadata_file

    extracted_data = {}
    for table in TARGET_TABLES:
        metadata = extract_table_metadata(metadata_file, table)
        if metadata:
            extracted_data[table] = metadata

    # Save the extracted metadata to a JSON file in the output directory
    output_file = config.PATHS['OUTPUT_DIR'] / "future_tables_metadata.json"
    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=4)
        logger.info(f"Successfully saved future tables metadata to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving metadata to JSON: {e}")

    logger.info("=== Future Tables Metadata Extraction Complete ===")

if __name__ == "__main__":
    main() 