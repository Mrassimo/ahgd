import zipfile
import re
import pandas as pd
from pathlib import Path
import sys

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

def inspect_zip_contents(zip_filepath: Path):
    """Lists files within a ZIP archive, focusing on CSVs."""
    logger.info(f"Inspecting contents of: {zip_filepath.name}")
    if not zip_filepath.exists():
        logger.error(f"ZIP file not found: {zip_filepath}")
        return

    try:
        with zipfile.ZipFile(zip_filepath, 'r') as z:
            all_files = z.namelist()
            logger.info(f"Total files in archive: {len(all_files)}")

            csv_files = [f for f in all_files if f.lower().endswith('.csv')]
            logger.info(f"Found {len(csv_files)} CSV files.")

            if not csv_files:
                logger.warning("No CSV files found in the archive.")
                return

            logger.info("First 10 CSV files found:")
            for csv_name in csv_files[:10]:
                print(f"  - {csv_name}")

            # Examine the first CSV found in detail
            first_csv = csv_files[0]
            logger.info(f"\n--- Examining first CSV file: {first_csv} ---")
            try:
                with z.open(first_csv) as csv_file:
                    # Read header and first few rows using Pandas
                    df = pd.read_csv(csv_file, nrows=5, low_memory=False)
                    logger.info("First 5 rows (including header):")
                    print(df.to_string())
                    logger.info(f"Column Names ({len(df.columns)} total): {df.columns.tolist()}")
                    logger.info("Data Types (Pandas inferred):")
                    print(df.dtypes)
            except Exception as e:
                logger.error(f"Error reading CSV '{first_csv}' from ZIP: {e}")

    except Exception as e:
        logger.error(f"Error opening or reading ZIP file {zip_filepath.name}: {e}")

def main():
    """Main function to inspect the Census ZIP file."""
    logger.info("=== Starting Census ZIP Inspection Script ===")

    # Define path to the main Census ZIP file using config
    # Assuming the filename is known or defined in config
    gcp_all_filename = "2021_GCP_all_for_AUS_short-header.zip" # Assuming standard name
    census_zip_path = config.PATHS['CENSUS_DIR'] / gcp_all_filename

    # Inspect the contents
    inspect_zip_contents(census_zip_path)

    logger.info("=== Inspection Complete ===")

if __name__ == "__main__":
    main()