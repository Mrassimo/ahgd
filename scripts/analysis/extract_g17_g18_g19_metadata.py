#!/usr/bin/env python3
import pandas as pd
import os
import sys
from pathlib import Path
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
import openpyxl # Needed to read xlsx
import zipfile
from collections import defaultdict

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
CENSUS_ZIP_DIR = config.PATHS['CENSUS_DIR']

# Define target tables
TARGET_TABLES = ["G17", "G18", "G19"]

def find_census_zip() -> Optional[Path]:
    """Finds the main Census GCP ZIP file."""
    # Try common naming patterns or a specific name if known
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
            tables_df = pd.read_excel(metadata_file_path, sheet_name="List of tables")
            table_info = tables_df[tables_df.apply(lambda row: row.astype(str).str.contains(table_code, na=False, case=False).any(), axis=1)]
            if not table_info.empty:
                metadata['source_sheet'] = 'List of tables'
                # Extract description and population (flexible column naming)
                desc_col = next((col for col in ['Table Title', 'Description', 'Table Name', tables_df.columns[1]] if col in table_info.columns), None)
                pop_col = next((col for col in ['Population', 'Pop', tables_df.columns[2]] if col in table_info.columns), None)
                metadata['description'] = table_info.iloc[0][desc_col] if desc_col else "Description unavailable"
                metadata['population'] = table_info.iloc[0][pop_col] if pop_col else "Population unavailable"
                logger.debug(f"  Found basic info for {table_code} in {metadata_file_path.name}")
            else:
                logger.debug(f"  {table_code} not found in 'List of tables' sheet.")
        except Exception as e:
            logger.warning(f"Could not process 'List of tables' in {metadata_file_path.name}: {e}")
    else:
        logger.warning(f"Main metadata file not found: {metadata_file_path}")

    # 2. Extract structure from template file (Sheet named like table_code)
    if template_file_path.exists():
        try:
            xlsx = pd.ExcelFile(template_file_path)
            if table_code in xlsx.sheet_names:
                logger.debug(f"  Found sheet '{table_code}' in template file. Reading structure...")
                df_template = pd.read_excel(template_file_path, sheet_name=table_code, nrows=20) # Read more rows for structure
                metadata['source_sheet'] = f'Template Sheet: {table_code}'
                # Try to refine description from template title if not found earlier
                if metadata['description'] == "Not found":
                    for i in range(min(5, df_template.shape[0])):
                        cell_value = str(df_template.iloc[i, 0])
                        if table_code in cell_value or 'table' in cell_value.lower():
                            metadata['description'] = cell_value
                            break
                # Extract column headers (find likely row)
                header_row_idx = -1
                for i in range(5, 15): # Search common header rows
                     if i < df_template.shape[0]:
                         row_vals = df_template.iloc[i].dropna().astype(str).str.lower().tolist()
                         # Heuristic: look for common terms like 'tot', 'male', 'female', 'median', specific conditions
                         if any(term in rv for term in ['tot', 'male', 'female', 'median', 'assist', 'arth', 'asth'] for rv in row_vals):
                             header_row_idx = i
                             break
                if header_row_idx != -1:
                    metadata['template_columns'] = df_template.iloc[header_row_idx].dropna().tolist()
                    logger.debug(f"  Extracted template columns from row {header_row_idx + 1}")
                else:
                     logger.debug(f"  Could not identify template column row for {table_code}.")

                 # Extract a sample data row
                first_data_idx = df_template.iloc[:, 0].first_valid_index()
                if first_data_idx is not None and first_data_idx + 1 < df_template.shape[0]:
                     metadata['template_sample_row'] = df_template.iloc[first_data_idx + 1].dropna().tolist()
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

    # Pattern specific to the table code
    csv_pattern = re.compile(rf"2021.*?Census.*?{table_code}[A-Z]?_.*?.(SA[1-4]|STE|AUS).csv", re.IGNORECASE)

    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                if csv_pattern.search(file_info.filename):
                    logger.debug(f"  Found matching CSV: {file_info.filename}")
                    csv_structure['found_csv_filename'] = file_info.filename
                    try:
                        with zip_ref.open(file_info.filename) as csv_file:
                            # Read header and infer types
                            df = pd.read_csv(csv_file, nrows=5, low_memory=False)
                            csv_structure['actual_columns'] = df.columns.tolist()
                            csv_structure['actual_dtypes'] = {col: str(dtype) for col, dtype in df.dtypes.items()}
                            logger.debug(f"  Extracted {len(df.columns)} columns from {file_info.filename}")
                            return csv_structure # Process only the first match
                    except Exception as e:
                        logger.error(f"  Error reading CSV '{file_info.filename}' from ZIP: {e}")
                        return csv_structure # Return partial data on error
            logger.warning(f"No CSV files matching pattern for {table_code} found in the ZIP.")
    except Exception as e:
        logger.error(f"Error opening or reading ZIP file {zip_filepath.name}: {e}")

    return csv_structure

def compare_and_summarize(excel_meta: dict, csv_meta: dict) -> str:
    """Compares metadata from Excel and CSV and creates a summary."""
    summary = []
    summary.append(f"## Metadata Summary for: {excel_meta['table_code']}")
    summary.append(f"- **Description**: {excel_meta.get('description', 'N/A')}")
    summary.append(f"- **Population**: {excel_meta.get('population', 'N/A')}")
    summary.append(f"- **Metadata Source**: {excel_meta.get('source_sheet', 'N/A')}")
    summary.append(f"- **Actual CSV File Found**: {csv_meta.get('found_csv_filename', 'None')}")

    # Compare column counts
    template_cols = excel_meta.get('template_columns', [])
    actual_cols = csv_meta.get('actual_columns', [])
    summary.append(f"- **Column Count (Template vs Actual)**: {len(template_cols)} vs {len(actual_cols)}")

    # List columns present in actual CSV but not obviously in template header (simple check)
    if actual_cols and template_cols:
        actual_set = set(actual_cols)
        template_set = set(str(col) for col in template_cols) # Convert template cols to string for comparison
        diff_cols = actual_set - template_set
        # Further refine diff: ignore geo code variations if present in both
        geo_codes_actual = {col for col in actual_set if any(gc in col for gc in ['SA1', 'SA2', 'SA3', 'SA4', 'STE', 'POA', 'LGA', 'AUS', 'GCC'])}
        geo_codes_template = {col for col in template_set if any(gc in col for gc in ['SA1', 'SA2', 'SA3', 'SA4', 'STE', 'POA', 'LGA', 'AUS', 'GCC'])}
        if geo_codes_actual and geo_codes_template:
            diff_cols = diff_cols - geo_codes_actual # Remove geo codes if template also had one

        if diff_cols:
            summary.append(f"- **Columns in Actual CSV potentially missing/different from Template Header ({len(diff_cols)})**: `{', '.join(sorted(list(diff_cols)))}`")
        else:
            summary.append("- **Column Alignment**: Actual CSV columns seem to align well with template header (based on simple check).")

    summary.append("\n### Actual CSV Columns and Types")
    summary.append("| Column Name | Inferred Type |")
    summary.append("|-------------|---------------|")
    if actual_cols:
        for col in actual_cols:
            dtype = csv_meta.get('actual_dtypes', {}).get(col, 'N/A')
            summary.append(f"| `{col}` | `{dtype}` |")
    else:
        summary.append("| *No CSV columns found* | - |")

    summary.append("\n")
    return "\n".join(summary)

def main():
    """Main function to drive the metadata extraction and comparison."""
    logger.info("=== Starting G17/G18/G19 Metadata Analysis ===")

    # Find the Census ZIP file
    census_zip = find_census_zip()
    if not census_zip:
        return

    all_summaries = ["# Census Table Metadata Analysis (G17, G18, G19)\n"]

    for table in TARGET_TABLES:
        # 1. Extract metadata from Excel files
        excel_metadata = extract_metadata_from_excel(METADATA_FILE, TEMPLATE_FILE, table)

        # 2. Extract structure from actual CSV file in ZIP
        csv_metadata = extract_csv_structure_from_zip(census_zip, table)

        # 3. Compare and summarize
        summary = compare_and_summarize(excel_metadata, csv_metadata)
        all_summaries.append(summary)

    # Save the combined summary to a Markdown file
    output_md_file = config.PATHS['OUTPUT_DIR'] / "g17_g18_g19_metadata_analysis.md"
    try:
        output_md_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_md_file, 'w') as f:
            f.write("\n".join(all_summaries))
        logger.info(f"Metadata analysis summary saved to: {output_md_file}")
    except Exception as e:
        logger.error(f"Error saving summary Markdown file: {e}")

    logger.info("=== Metadata Analysis Complete ===")

if __name__ == "__main__":
    main() 