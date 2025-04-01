#!/usr/bin/env python3
import polars as pl
import pandas as pd
import os
import sys
from pathlib import Path
import re
import logging
from typing import Dict, List, Optional, Tuple, Any

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Base directories
BASE_DIR = Path(os.getenv('PROJECT_DIR', '/Users/massimoraso/AHGD3'))
RAW_DATA_DIR = BASE_DIR / "data/raw"
EXTRACT_DIR = RAW_DATA_DIR / "temp/extract"
OUTPUT_DIR = BASE_DIR / "output"
MEMORY_BANK_DIR = BASE_DIR / "memory-bank"

# Ensure the memory bank directory exists
MEMORY_BANK_DIR.mkdir(parents=True, exist_ok=True)

def find_csv_files(base_dir: Path, pattern: str) -> List[Path]:
    """Find CSV files matching a pattern in a directory tree."""
    logger.info(f"Searching for {pattern} files in {base_dir}")
    
    # Use glob to find all CSV files
    all_csv_files = list(base_dir.glob("**/*.csv"))
    
    # Filter for pattern
    matching_files = [f for f in all_csv_files if re.search(pattern, f.name, re.IGNORECASE)]
    logger.info(f"Found {len(matching_files)} files matching pattern '{pattern}'")
    
    return matching_files

def analyze_csv_columns(csv_file: Path) -> Dict[str, Any]:
    """Analyze the column structure of a CSV file."""
    logger.info(f"Analyzing columns in {csv_file}")
    
    # Initialize result dictionary
    result = {
        "file_name": csv_file.name,
        "file_path": str(csv_file),
        "total_columns": 0,
        "geo_columns": [],
        "data_columns": [],
        "column_prefixes": set(),
        "column_patterns": {},
        "sample_rows": [],
        "error": None
    }
    
    try:
        # Read CSV file with Polars to handle larger files efficiently
        df = pl.read_csv(csv_file, infer_schema_length=1000)
        
        # Basic info
        result["total_columns"] = len(df.columns)
        
        # Identify geographic columns
        geo_patterns = ["CODE", "SA1", "SA2", "SA3", "SA4", "SUA", "region_id"]
        result["geo_columns"] = [col for col in df.columns if any(pattern.lower() in col.lower() for pattern in geo_patterns)]
        
        # Collect other data columns
        result["data_columns"] = [col for col in df.columns if col not in result["geo_columns"]]
        
        # Extract column prefixes (first part before underscore)
        for col in result["data_columns"]:
            parts = col.split("_")
            if len(parts) > 1:
                result["column_prefixes"].add(parts[0])
        
        # Identify column patterns
        for prefix in result["column_prefixes"]:
            pattern_cols = [col for col in result["data_columns"] if col.startswith(f"{prefix}_")]
            if pattern_cols:
                result["column_patterns"][prefix] = pattern_cols[:5]  # First 5 examples
        
        # Save sample rows
        if len(df) > 0:
            try:
                # Convert first few rows to string representation for display
                sample_rows = df.head(3).to_pandas().to_dict('records')
                result["sample_rows"] = sample_rows
            except Exception as e:
                result["sample_rows"] = [{"error": f"Could not convert sample rows: {str(e)}"}]
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error analyzing {csv_file}: {e}")
    
    return result

def analyze_all_files(table_codes: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Analyze all CSV files for the specified table codes."""
    results = {}
    
    for code in table_codes:
        # Define the directory to look in
        extract_dir = EXTRACT_DIR / f"g{code.lower()}" 
        if not extract_dir.exists():
            # Try alternate path formats
            alternates = [
                EXTRACT_DIR / f"g{code.lower()}_extract_2021_GCP_all_for_AUS_short-header",
                EXTRACT_DIR,  # Try the main extract dir
                EXTRACT_DIR / f"g{code}"
            ]
            
            for alt_dir in alternates:
                if alt_dir.exists():
                    extract_dir = alt_dir
                    break
            else:
                logger.warning(f"Could not find extract directory for G{code}")
                continue
        
        # Find CSV files for this table
        pattern = f"g{code}"
        csv_files = find_csv_files(extract_dir, pattern)
        
        if not csv_files:
            logger.warning(f"No G{code} files found in {extract_dir}")
            continue
        
        # Analyze a sample of files (first 3)
        sample_files = csv_files[:3]
        results[f"G{code}"] = [analyze_csv_columns(f) for f in sample_files]
    
    return results

def write_results_to_markdown(results: Dict[str, List[Dict[str, Any]]]):
    """Write the analysis results to a Markdown file."""
    output_file = MEMORY_BANK_DIR / "census_file_structure_analysis.md"
    
    with open(output_file, "w") as f:
        f.write("# Census File Structure Analysis\n\n")
        f.write("This document provides an analysis of the structure of G17, G18, and G19 CSV files to help debug processing issues.\n\n")
        
        for table_code, file_results in results.items():
            f.write(f"## {table_code} Files\n\n")
            
            if not file_results:
                f.write(f"No {table_code} files were found for analysis.\n\n")
                continue
            
            for i, result in enumerate(file_results):
                f.write(f"### File {i+1}: {result['file_name']}\n\n")
                
                if result.get('error'):
                    f.write(f"**Error analyzing file:** {result['error']}\n\n")
                    continue
                
                f.write(f"**Path:** {result['file_path']}\n\n")
                f.write(f"**Total columns:** {result['total_columns']}\n\n")
                
                f.write("**Geographic columns:**\n")
                for col in result['geo_columns']:
                    f.write(f"- `{col}`\n")
                f.write("\n")
                
                f.write("**Column prefixes:** ")
                f.write(", ".join(f"`{p}`" for p in sorted(result['column_prefixes'])))
                f.write("\n\n")
                
                f.write("**Column patterns:**\n")
                for prefix, cols in result['column_patterns'].items():
                    f.write(f"- Prefix `{prefix}`: ")
                    f.write(", ".join(f"`{c}`" for c in cols))
                    f.write(" ...\n")
                f.write("\n")
                
                if result['sample_rows']:
                    f.write("**Sample data (first few rows):**\n\n")
                    f.write("```\n")
                    for row in result['sample_rows']:
                        f.write(f"{row}\n")
                    f.write("```\n\n")
                
                f.write("---\n\n")
        
        f.write("\n## Recommendations for Processing Functions\n\n")
        f.write("Based on the analysis above, here are recommendations for fixing the processing functions:\n\n")
        
        # Add specific recommendations for each table
        for table_code in results.keys():
            f.write(f"### {table_code} Processing\n\n")
            f.write(f"1. Update the `process_{table_code.lower()}_file` function to handle these column patterns:\n")
            
            # Extract patterns seen across all files for this table code
            all_prefixes = set()
            all_patterns = {}
            
            for result in results.get(table_code, []):
                all_prefixes.update(result.get('column_prefixes', set()))
                for prefix, cols in result.get('column_patterns', {}).items():
                    if prefix not in all_patterns:
                        all_patterns[prefix] = []
                    all_patterns[prefix].extend(cols)
            
            for prefix in sorted(all_prefixes):
                f.write(f"   - For prefix `{prefix}`, look for columns like: ")
                pattern_examples = all_patterns.get(prefix, [])[:5]  # First 5 examples
                f.write(", ".join(f"`{c}`" for c in pattern_examples))
                f.write("\n")
            
            f.write("2. Use a flexible column detection approach similar to what we implemented for G21.\n")
            f.write("3. Ensure the geographic column detection can handle all variations seen in these files.\n\n")
    
    logger.info(f"Analysis written to {output_file}")

if __name__ == "__main__":
    logger.info("Starting analysis of G17, G18, and G19 files...")
    
    # Specify the table codes to analyze
    table_codes = ["17", "18", "19"]
    
    # Analyze the files
    results = analyze_all_files(table_codes)
    
    # Write the results to a Markdown file
    write_results_to_markdown(results)
    
    logger.info("Analysis complete!") 