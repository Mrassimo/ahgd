"""
Script for extracting metadata from all G-tables in the ABS Census data.

This script reads the metadata Excel file, identifies all G-tables, and extracts their structure, including headers, variables, and notes, based on the template file. It outputs the metadata to a Markdown file for documentation purposes.

Usage:
   Run this script to generate a comprehensive metadata document for all census tables.
"""

#!/usr/bin/env python3
#!/usr/bin/env python3
import pandas as pd
import os
import sys
from pathlib import Path
import re
import logging
from typing import List, Tuple, Dict, Any, Optional
import openpyxl # Needed to read xlsx
import zipfile
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

# Define paths using config
METADATA_DIR = config.PATHS.get('METADATA_DIR', config.PATHS['RAW_DATA_DIR'] / "Metadata")
CENSUS_ZIP_DIR = config.PATHS['CENSUS_DIR']
OUTPUT_DIR = config.PATHS['OUTPUT_DIR']

# Constants
METADATA_FILENAME = "Metadata_2021_GCP_DataPack_R1_R2.xlsx"
TEMPLATE_FILENAME = "2021_GCP_Sequential_Template_R2.xlsx"
LIST_OF_TABLES_SHEET = "List of tables"
OUTPUT_JSON_FILENAME = "all_tables_metadata_analysis.json"

def find_census_zip() -> Optional[Path]:
    """Finds the main Census GCP ZIP file."""
    patterns = [
        "2021_GCP_all_for_AUS_short-header.zip",
        "2021_GCP_*.zip"
    ]
    for pattern in patterns:
        zip_files = list(CENSUS_ZIP_DIR.glob(pattern))
        if zip_files:
            found_zip = zip_files[0]
            logger.info(f"Using Census ZIP file: {found_zip}")
            return found_zip
    logger.error(f"Could not find Census GCP ZIP file in {CENSUS_ZIP_DIR}")
    return None

def get_all_g_table_codes(metadata_file_path: Path) -> List[str]:
    """Extracts all table codes starting with 'G' from the metadata file."""
    if not metadata_file_path.exists():
        logger.error(f"Metadata file not found: {metadata_file_path}")
        return []
    try:
        tables_df = pd.read_excel(metadata_file_path, sheet_name=LIST_OF_TABLES_SHEET)
        # Assuming 'Table Number' column exists
        if 'Table Number' in tables_df.columns:
            g_tables = tables_df[tables_df['Table Number'].astype(str).str.startswith('G', na=False)]
            codes = sorted(g_tables['Table Number'].unique().tolist())
            logger.info(f"Found {len(codes)} G-table codes in metadata: {codes}")
            return codes
        else:
            logger.warning(f"'Table Number' column not found in '{LIST_OF_TABLES_SHEET}' sheet.")
            return []
    except Exception as e:
        logger.error(f"Error reading '{LIST_OF_TABLES_SHEET}' sheet from {metadata_file_path}: {e}")
        return []

def extract_metadata_from_excel(metadata_file_path: Path, template_file_path: Path, table_code: str) -> dict:
    """Extracts metadata for a table from both the main metadata file and the template file."""
    logger.info(f"Extracting metadata for {table_code} from Excel files...")
    metadata = {
        'table_code': table_code,
        'description': "Not found",
        'population': "Unknown",
        'source_sheet': "Not found",
        'template_columns': [],
        'template_sample_row': []
    }

    # 1. Extract from main metadata file ('List of tables')
    if metadata_file_path.exists():
        try:
            # Read only once if possible, maybe pass df as arg?
            tables_df = pd.read_excel(metadata_file_path, sheet_name=LIST_OF_TABLES_SHEET)
            table_info = tables_df[tables_df.apply(lambda row: row.astype(str).str.contains(f'^{table_code}(?:[A-Z])?$', na=False, case=False, regex=True).any(), axis=1)] # Match GXX or GXXA etc.
            if not table_info.empty:
                metadata['source_sheet'] = LIST_OF_TABLES_SHEET
                desc_col = next((col for col in ['Table Title', 'Description', 'Table Name', tables_df.columns[1]] if col in table_info.columns), None)
                pop_col = next((col for col in ['Population', 'Pop', tables_df.columns[2]] if col in table_info.columns), None)
                metadata['description'] = table_info.iloc[0][desc_col] if desc_col and pd.notna(table_info.iloc[0][desc_col]) else "Description unavailable"
                metadata['population'] = table_info.iloc[0][pop_col] if pop_col and pd.notna(table_info.iloc[0][pop_col]) else "Population unavailable"
                logger.debug(f"  Found basic info for {table_code} in {metadata_file_path.name}")
            else:
                logger.debug(f"  {table_code} not found in '{LIST_OF_TABLES_SHEET}' sheet.")
        except Exception as e:
            logger.warning(f"Could not process '{LIST_OF_TABLES_SHEET}' in {metadata_file_path.name}: {e}")
    else:
        logger.warning(f"Main metadata file not found: {metadata_file_path}")

    # 2. Extract structure from template file (Sheet named like table_code)
    if template_file_path.exists():
        try:
            xlsx = pd.ExcelFile(template_file_path)
            if table_code in xlsx.sheet_names:
                logger.debug(f"  Found sheet '{table_code}' in template file. Reading structure...")
                df_template = pd.read_excel(template_file_path, sheet_name=table_code, nrows=20)
                if metadata['source_sheet'] == 'Not found':
                    metadata['source_sheet'] = f'Template Sheet: {table_code}'
                if metadata['description'] == "Not found":
                    for i in range(min(5, df_template.shape[0])):
                        cell_value = str(df_template.iloc[i, 0])
                        if table_code in cell_value or 'table' in cell_value.lower():
                            metadata['description'] = cell_value
                            break
                header_row_idx = -1
                for i in range(5, 15):
                     if i < df_template.shape[0]:
                         row_vals = df_template.iloc[i].dropna().astype(str).str.lower().tolist()
                         if any(term in rv for term in ['tot', 'male', 'female', 'median', 'assist', 'arth', 'asth', 'code', '_key', 'years'] for rv in row_vals):
                             header_row_idx = i
                             break
                if header_row_idx != -1:
                    metadata['template_columns'] = [str(col) for col in df_template.iloc[header_row_idx].dropna().tolist()]
                    logger.debug(f"  Extracted {len(metadata['template_columns'])} template columns from row {header_row_idx + 1}")
                else:
                     logger.debug(f"  Could not identify template column row for {table_code}.")
                first_data_idx = df_template.iloc[:, 0].first_valid_index()
                if first_data_idx is not None and first_data_idx + 1 < df_template.shape[0]:
                     metadata['template_sample_row'] = [str(item) for item in df_template.iloc[first_data_idx + 1].dropna().tolist()]
                     logger.debug(f"  Extracted template sample row from row {first_data_idx + 2}")
            else:
                logger.debug(f"  Sheet '{table_code}' not found in template file.")
        except Exception as e:
            logger.warning(f"Could not process template file {template_file_path.name} for sheet {table_code}: {e}")
    else:
        logger.warning(f"Template file not found: {template_file_path}")

    return metadata

def extract_csv_structure_from_zip(zip_filepath: Path, table_code: str) -> dict:
    """Extracts actual column names and types from the first matching CSV in a ZIP."""
    logger.info(f"Extracting CSV structure for {table_code} from {zip_filepath.name}...")
    csv_structure = {
        'actual_columns': [],
        'actual_dtypes': {},
        'found_csv_filename': None
    }
    if not zip_filepath.exists():
        logger.error(f"ZIP file not found: {zip_filepath}")
        return csv_structure

    # Pattern specific to the table code, allowing for A/B/C suffixes
    csv_pattern = re.compile(rf"2021.*?Census.*?{table_code}[A-Z]?_.*?.(SA[1-4]|STE|AUS|POA|LGA|GCC).csv", re.IGNORECASE)

    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            found_files = []
            for file_info in zip_ref.infolist():
                if csv_pattern.search(file_info.filename):
                     found_files.append(file_info.filename)

            if not found_files:
                logger.warning(f"No CSV files matching pattern for {table_code} found in the ZIP.")
                return csv_structure

            # Process the first matching file found
            first_match_filename = sorted(found_files)[0]
            logger.debug(f"  Found {len(found_files)} matching CSV(s). Processing first: {first_match_filename}")
            csv_structure['found_csv_filename'] = first_match_filename
            try:
                with zip_ref.open(first_match_filename) as csv_file:
                    df = pd.read_csv(csv_file, nrows=5, low_memory=False)
                    csv_structure['actual_columns'] = df.columns.tolist()
                    csv_structure['actual_dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
                    logger.debug(f"  Extracted {len(df.columns)} columns from {first_match_filename}")
            except Exception as e:
                logger.error(f"  Error reading CSV '{first_match_filename}' from ZIP: {e}")
                # Return partial info even on read error
    except Exception as e:
        logger.error(f"Error opening or reading ZIP file {zip_filepath.name}: {e}")

    return csv_structure

def main():
    """Main function to drive the metadata extraction for all G-tables."""
    logger.info("=== Starting All Tables Metadata Analysis ===")

    metadata_file = METADATA_DIR / METADATA_FILENAME
    template_file = METADATA_DIR / TEMPLATE_FILENAME

    # Find the Census ZIP file
    census_zip = find_census_zip()
    if not census_zip:
        logger.warning("Cannot proceed without Census ZIP file to check actual CSVs.")
        # Continue with Excel metadata only?
        # return

    # Get all G-table codes
    all_g_codes = get_all_g_table_codes(metadata_file)
    if not all_g_codes:
        logger.error("No G-table codes found. Cannot proceed.")
        return

    all_tables_metadata = {}

    for table in all_g_codes:
        # 1. Extract metadata from Excel files
        excel_metadata = extract_metadata_from_excel(metadata_file, template_file, table)

        # 2. Extract structure from actual CSV file in ZIP (if zip found)
        csv_metadata = {}
        if census_zip:
            csv_metadata = extract_csv_structure_from_zip(census_zip, table)
        else:
             logger.warning(f"Skipping actual CSV check for {table} as ZIP file was not found.")

        # Combine metadata
        # Prioritize actual CSV columns/types if available
        combined_meta = excel_metadata.copy()
        combined_meta.update(csv_metadata)
        all_tables_metadata[table] = combined_meta

    # Save the combined metadata to a JSON file
    output_json_file = OUTPUT_DIR / OUTPUT_JSON_FILENAME
    try:
        output_json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_file, 'w') as f:
            json.dump(all_tables_metadata, f, indent=4)
        logger.info(f"Full table metadata analysis saved to: {output_json_file}")
    except Exception as e:
        logger.error(f"Error saving metadata analysis JSON file: {e}")

    logger.info("=== All Tables Metadata Analysis Complete ===")

if __name__ == "__main__":
    main() 