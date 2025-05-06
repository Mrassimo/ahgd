"""
Script to generate data profiling reports for Parquet files in the output directory.

This script uses the ydata-profiling library to create detailed HTML reports
for each Parquet file found in the configured output directory.
"""

import logging
import sys
from pathlib import Path
import polars as pl
from ydata_profiling import ProfileReport

# Add project root to sys.path to allow importing etl_logic
project_root = Path(__file__).resolve().parents[1] # Assuming scripts/ is one level down from project root
sys.path.insert(0, str(project_root))

# Import config after setting path
from etl_logic import config
from etl_logic import utils # Assuming setup_logging is in utils

# Setup logging
logger = utils.setup_logging(config.PATHS.get('LOG_DIR', project_root / 'logs'))

def generate_profile_report(parquet_path: Path, output_html_path: Path):
    """Generates a ydata-profiling report for a given Parquet file.

    Args:
        parquet_path (Path): Path to the input Parquet file.
        output_html_path (Path): Path to save the output HTML report.
    """
    logger.info(f"Generating profiling report for: {parquet_path.name}")
    try:
        # Read Parquet file using Polars
        df = pl.read_parquet(parquet_path)

        # Convert Polars DataFrame to Pandas DataFrame for profiling
        # Note: This can be memory-intensive for large datasets.
        # Consider sampling if needed: df_pandas = df.sample(n=10000).to_pandas()
        df_pandas = df.to_pandas()

        # Generate the profile report
        # Provide a title for the report
        profile_title = f"Profiling Report: {parquet_path.stem}"
        profile = ProfileReport(df_pandas, title=profile_title, explorative=True)

        # Save the report to HTML
        profile.to_file(output_html_path)
        logger.info(f"Successfully generated report: {output_html_path}")

    except FileNotFoundError:
        logger.error(f"Error: Parquet file not found at {parquet_path}")
    except Exception as e:
        logger.error(f"Error generating report for {parquet_path.name}: {e}")
        logger.exception("Detailed traceback:")

def main():
    """Main function to find Parquet files and generate reports for each."""
    logger.info("=== Starting Data Profiling Report Generation ===")

    # Use output directory from config
    output_dir = config.PATHS['OUTPUT_DIR']
    profiling_reports_dir = output_dir / "profiling_reports"

    # Create the profiling reports directory if it doesn't exist
    profiling_reports_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving reports to: {profiling_reports_dir}")

    # Find all Parquet files in the output directory
    try:
        # Use output_dir from config
        parquet_files = list(output_dir.glob('*.parquet'))

        if not parquet_files:
            logger.warning(f"No Parquet files found in {output_dir}. Cannot generate reports.")
            return

        logger.info(f"Found {len(parquet_files)} Parquet files to profile.")

        # Generate report for each Parquet file
        for parquet_file in parquet_files:
            # Construct output HTML file path
            # Use .stem to get filename without extension
            report_filename = f"{parquet_file.stem}_profile_report.html"
            output_html_path = profiling_reports_dir / report_filename

            # Generate the report
            generate_profile_report(parquet_file, output_html_path)

        logger.info("=== Data Profiling Report Generation Complete ===")

    except Exception as e:
        logger.critical(f"An unexpected error occurred during the main process: {e}")
        logger.exception("Detailed traceback:")

if __name__ == "__main__":
    # Ensure directories are initialised (optional, depends on workflow)
    # config.initialise_directories() 
    main() 