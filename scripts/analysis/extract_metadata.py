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
OUTPUT_MD_FILE = config.PATHS['OUTPUT_DIR'] / "metadata_summary.md"

def extract_sheet_summary(file_path: Path) -> dict:
    """Extracts sheet names and a brief head from each sheet in an Excel file."""
    summary = {'file_name': file_path.name, 'sheets': {}}
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        summary['error'] = "File not found"
        return summary

    try:
        xlsx = pd.ExcelFile(file_path)
        sheet_names = xlsx.sheet_names
        summary['sheet_count'] = len(sheet_names)
        logger.info(f"Found {len(sheet_names)} sheets in {file_path.name}")

        for sheet in sheet_names:
            try:
                # Read only a few rows to get a sense of content
                df_head = pd.read_excel(xlsx, sheet_name=sheet, nrows=5)
                summary['sheets'][sheet] = {
                    'head_preview': df_head.to_string()
                }
            except Exception as e:
                 summary['sheets'][sheet] = {'error': f"Could not read sheet: {e}"}
                 logger.warning(f"Could not read sheet '{sheet}' from {file_path.name}: {e}")
    except Exception as e:
        logger.error(f"Error reading Excel file {file_path.name}: {e}")
        summary['error'] = f"Error reading file: {e}"
    return summary

def write_summary_to_markdown(summary_data: dict, output_path: Path):
    """Writes the extracted summary data to a Markdown file."""
    logger.info(f"Writing metadata summary to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("# Metadata Files Summary\n\n")
        for file_key, data in summary_data.items():
            f.write(f"## File: {data['file_name']}\n\n")
            if data.get('error'):
                f.write(f"**Error:** {data['error']}\n\n")
                continue
            f.write(f"- **Sheet Count:** {data.get('sheet_count', 'N/A')}\n")
            f.write("- **Sheets Found:**\n")
            for sheet_name, sheet_data in data.get('sheets', {}).items():
                f.write(f"  - **{sheet_name}**\n")
                if sheet_data.get('error'):
                    f.write(f"    - Error reading sheet: {sheet_data['error']}\n")
                else:
                    f.write(f"    - Preview (first 5 rows):\n```\n{sheet_data.get('head_preview', 'N/A')}\n```\n")
            f.write("\n")
    logger.info("Summary written successfully.")

def main():
    """Main function to extract summaries and write Markdown."""
    logger.info("=== Starting Metadata File Summary Extraction ===")
    all_summaries = {}
    all_summaries['metadata_main'] = extract_sheet_summary(METADATA_FILE)
    all_summaries['metadata_template'] = extract_sheet_summary(TEMPLATE_FILE)

    write_summary_to_markdown(all_summaries, OUTPUT_MD_FILE)
    logger.info("=== Metadata Summary Extraction Complete ===")

if __name__ == "__main__":
    main() 