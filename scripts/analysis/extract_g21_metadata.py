#!/usr/bin/env python3
import zipfile
import re
import sys
from pathlib import Path
import polars as pl

# Add project root to sys.path to allow importing etl_logic
project_root = Path(__file__).resolve().parents[2] # Assuming scripts/analysis/ is two levels down
sys.path.insert(0, str(project_root))

# Import config and utils after setting path
from etl_logic import config
from etl_logic import utils # Assuming setup_logging is in utils

# Setup logging (optional, adjust level as needed)
# Configure logger if you want output from this script
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_g21_files(census_data_dir: Path) -> list:
    """Finds all potential G21 CSV files within extracted Census ZIP contents."""
    g21_files = []
    pattern = re.compile(r"2021.?Census.?G21[A-C]?_.*?.(SA[1-4]|STE|AUS).csv", re.IGNORECASE)
    logger.info(f"Searching for G21 files in: {census_data_dir}")
    # Assuming data is extracted into subdirectories within census_data_dir
    # or directly within temp_extract_dir defined in config
    search_dir = config.PATHS.get('TEMP_EXTRACT_DIR', census_data_dir / "temp/extract")

    if not search_dir.exists():
        logger.warning(f"Search directory does not exist: {search_dir}")
        # Fallback to searching the raw census dir if temp extract doesn't exist
        search_dir = census_data_dir
        if not search_dir.exists():
            logger.error(f"Census data directory not found: {search_dir}")
            return []

    for item in search_dir.rglob('*.csv'):
        if pattern.search(item.name):
            g21_files.append(item)
    logger.info(f"Found {len(g21_files)} potential G21 files.")
    return g21_files

def extract_metadata_from_g21(file_path: Path) -> dict:
    """Extracts metadata (column names, types, potential keys) from a G21 CSV."""
    logger.info(f"Extracting metadata from: {file_path.name}")
    metadata = {
        'file_name': file_path.name,
        'columns': {},
        'potential_geo_key': None,
        'potential_measure_cols': []
    }
    try:
        # Read only header and first few rows to infer schema quickly
        df = pl.read_csv(file_path, n_rows=5, truncate_ragged_lines=True)
        col_names = df.columns

        # Identify potential geographic key
        geo_col = utils.find_geo_column(df, config.GEO_COLUMN_OPTIONS)
        if geo_col:
            metadata['potential_geo_key'] = geo_col
            logger.debug(f"  Potential Geo Key: {geo_col}")
        else:
            logger.warning(f"  No standard geo key found in {file_path.name}")

        # Store column names and basic type inference (Polars dtype)
        for col in col_names:
            col_type = str(df[col].dtype)
            metadata['columns'][col] = col_type
            # Basic heuristic to identify measure columns (numeric, not the geo key)
            if col != geo_col and df[col].dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8, pl.Float64, pl.Float32]:
                metadata['potential_measure_cols'].append(col)

        logger.debug(f"  Found {len(metadata['potential_measure_cols'])} potential measure columns.")

    except Exception as e:
        logger.error(f"Error processing file {file_path.name}: {e}")
        metadata['error'] = str(e)

    return metadata

def main():
    """Main function to find G21 files and extract metadata."""
    logger.info("=== Starting G21 Metadata Extraction Script ===")

    # Use CENSUS_DIR from config
    census_dir = config.PATHS['CENSUS_DIR']

    # Find G21 files (searches within TEMP_EXTRACT_DIR or CENSUS_DIR)
    g21_files = find_g21_files(census_dir)

    if not g21_files:
        logger.warning("No G21 files found. Exiting.")
        return

    all_metadata = []
    for file_path in g21_files:
        metadata = extract_metadata_from_g21(file_path)
        all_metadata.append(metadata)

    # Optional: Save metadata to a file (e.g., JSON)
    output_file = config.PATHS['OUTPUT_DIR'] / "g21_metadata_analysis.json"
    try:
        import json
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_metadata, f, indent=4)
        logger.info(f"Metadata analysis saved to: {output_file}")
    except Exception as e:
        logger.error(f"Could not save metadata JSON: {e}")

    # Print summary
    logger.info("\n=== Metadata Extraction Summary ===")
    successful_files = sum(1 for m in all_metadata if 'error' not in m)
    failed_files = len(all_metadata) - successful_files
    logger.info(f"Successfully processed: {successful_files} files")
    if failed_files > 0:
        logger.warning(f"Failed to process: {failed_files} files")
        for m in all_metadata:
            if 'error' in m:
                logger.warning(f"  - {m['file_name']}: {m['error']}")

    logger.info("=== G21 Metadata Extraction Complete ===")

if __name__ == "__main__":
    main()