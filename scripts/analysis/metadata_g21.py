#!/usr/bin/env python3
"""
Module for analyzing the G21 Census data files, focusing on health conditions by characteristics.

This script extracts and logs detailed information about the file structure, column patterns, and sample data to aid in debugging and improving the ETL processing functions for G21 data.
"""

import sys
from pathlib import Path
import pandas as pd
import logging
import re

# Add project root to sys.path to allow importing etl_logic
project_root = Path(__file__).resolve().parents[2] # Assuming scripts/analysis/ is two levels down
sys.path.insert(0, str(project_root))

# Import config and utils after setting path
from etl_logic import config
from etl_logic import utils # Assuming setup_logging is in utils

# Setup logging (optional, adjust level as needed)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_g21_files():
   """Analyze the structure of G21 files in the extract directory."""
   # Path to G21 extract directory
   g21_path = config.PATHS['TEMP_EXTRACT_DIR'] / "g21/2021 Census GCP All Geographies for AUS/SA1/AUS/"
   
   if not g21_path.exists():
       logger.error(f"G21 directory not found: {g21_path}")
       return
   
   # Get all G21 CSV files
   g21_files = list(g21_path.glob("2021Census_G21*.csv"))
   logger.info(f"Found {len(g21_files)} G21 files: {[f.name for f in g21_files]}")
   
   for file_path in g21_files:
       logger.info(f"Analyzing {file_path.name}...")
       
       # Read just the headers to analyze columns
       try:
           df = pd.read_csv(file_path, nrows=1)
           
           # Get column details
           columns = df.columns.tolist()
           logger.info(f"File has {len(columns)} columns")
           
           # Analyze column patterns
           col_patterns = {}
           for col in columns:
               if col.startswith("SA1_CODE") or col == "region_id":
                   logger.info(f"Geographic column: {col}")
                   continue
               
               # Try to detect patterns in column names
               parts = col.split('_')
               if len(parts) >= 2:
                   prefix = parts[0]
                   if prefix not in col_patterns:
                       col_patterns[prefix] = []
                   col_patterns[prefix].append(col)
           
           # Report patterns
           logger.info(f"Column patterns identified:")
           for prefix, cols in col_patterns.items():
               logger.info(f"  - {prefix}: {len(cols)} columns (example: {cols[0]})")
           
           # Analyze a sample of values to understand data types and distributions
           df_sample = pd.read_csv(file_path, nrows=5)
           logger.info(f"Sample data (first 5 rows):")
           logger.info(f"\n{df_sample.head().to_string()}")
           
           # Calculate column name frequency to find patterns
           suffix_counts = {}
           for col in columns:
               if col.startswith("SA1_CODE") or col == "region_id":
                   continue
               # Extract condition suffixes (last part after underscore)
               parts = col.split('_')
               if len(parts) >= 2:
                   suffix = parts[-1]
                   if suffix not in suffix_counts:
                       suffix_counts[suffix] = 0
                   suffix_counts[suffix] += 1
           # Report suffixes (likely representing conditions)
           logger.info(f"Detected condition suffixes:")
           for suffix, count in sorted(suffix_counts.items(), key=lambda x: x[1], reverse=True):
               logger.info(f"  - {suffix}: {count} occurrences")
           
       except Exception as e:
           logger.error(f"Error analyzing {file_path.name}: {e}")

def examine_g21_metadata_sheet(metadata_file_path: Path):
    """Examines the G21 sheet within the main metadata Excel file."""
    logger.info(f"Examining G21 sheet in metadata file: {metadata_file_path}")
    if not metadata_file_path.exists():
        logger.error(f"Metadata file not found: {metadata_file_path}")
        return

    try:
        xlsx = pd.ExcelFile(metadata_file_path)
        if 'G21' in xlsx.sheet_names:
            logger.info("Found G21 sheet. Reading first 50 rows...")
            df_g21 = pd.read_excel(metadata_file_path, sheet_name='G21', nrows=50)

            # Print basic info
            print("\n--- G21 Sheet Structure (from Metadata) ---")
            print(f"Shape: {df_g21.shape}")
            print("First few rows:")
            print(df_g21.head().to_string())
            # Extract Title (common location)
            title = df_g21.iloc[2, 0] if df_g21.shape[0] > 2 and pd.notna(df_g21.iloc[2, 0]) else "Title not found"
            print(f"\nTable Title (from A3): {title}")

            # Find potential header row (looking for condition names)
            header_row_index = -1
            for i in range(5, 20): # Search common header rows
                if i < df_g21.shape[0] and any(cond in str(df_g21.iloc[i, 1]).lower() for cond in ['arth', 'asth', 'cancer', 'diab']):
                    header_row_index = i
                    break
            if header_row_index != -1:
                print(f"\nPotential Header Row ({header_row_index + 1}):")
                print(df_g21.iloc[header_row_index].dropna().tolist())
            else:
                print("\nCould not reliably identify header row in the first 50 rows.")

            # Find potential data start (looking for Age groups or Sex)
            data_start_index = -1
            for i in range(header_row_index + 1, df_g21.shape[0]):
                cell_val = str(df_g21.iloc[i, 0]).lower()
                if any(term in cell_val for term in ['years', 'male', 'female', 'person', '0_4', '5_14', 'total']):
                    data_start_index = i
                    break
            if data_start_index != -1:
                print(f"\nPotential Data Start Row ({data_start_index + 1}):")
                print(df_g21.iloc[data_start_index].dropna().tolist())
            else:
                print("\nCould not reliably identify data start row.")

        else:
            logger.warning("G21 sheet not found in the metadata file.")
    except Exception as e:
        logger.error(f"Error reading or processing metadata file '{metadata_file_path.name}': {e}")

def examine_actual_g21_csv(census_dir: Path):
    """Finds an actual G21 CSV file and examines its header."""
    logger.info(f"\nSearching for an actual G21 CSV file in {census_dir}...")
    # Use find_g21_files logic (simplified or reused)
    g21_files = []
    pattern = re.compile(r"2021.?Census.?G21[A-C]?_.*?.(SA[1-4]|STE|AUS).csv", re.IGNORECASE)
    search_dir = config.PATHS.get('TEMP_EXTRACT_DIR', census_dir / "temp/extract")
    if not search_dir.exists():
        search_dir = census_dir # Fallback
    if search_dir.exists():
        for item in search_dir.rglob('*.csv'):
            if pattern.search(item.name):
                g21_files.append(item)

    if not g21_files:
        logger.warning("No actual G21 CSV files found.")
        return

    # Examine the first found file
    file_path = g21_files[0]
    logger.info(f"Examining header of actual G21 CSV file: {file_path.name}")
    try:
        df_actual = pd.read_csv(file_path, nrows=5, low_memory=False)
        print("\n--- Actual G21 CSV Header ---")
        print(f"Total Columns: {len(df_actual.columns)}")
        print("Column Names:")
        for i, col in enumerate(df_actual.columns):
            print(f"  {i+1}. {col}")
    except Exception as e:
        logger.error(f"Error reading actual G21 CSV file '{file_path.name}': {e}")

def main():
    """Main function to run the G21 metadata examination."""
    logger.info("=== Starting G21 Metadata Examination Script ===")

    # Define paths using config
    metadata_dir = config.PATHS.get('METADATA_DIR', config.PATHS['RAW_DATA_DIR'] / "Metadata")
    metadata_file = metadata_dir / "Metadata_2021_GCP_DataPack_R1_R2.xlsx" # Assuming standard filename
    census_dir = config.PATHS['CENSUS_DIR']

    # Examine the metadata sheet
    examine_g21_metadata_sheet(metadata_file)

    # Examine an actual CSV file
    examine_actual_g21_csv(census_dir)

    logger.info("=== G21 Metadata Examination Complete ===")

if __name__ == "__main__":
    # Need to import re for the pattern matching in examine_actual_g21_csv
    import re
    main()