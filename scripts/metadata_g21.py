#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_g21_files():
    """Analyze the structure of G21 files in the extract directory."""
    # Path to G21 extract directory
    g21_path = Path('data/raw/temp/extract/g21/2021 Census GCP All Geographies for AUS/SA1/AUS/')
    
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

if __name__ == "__main__":
    logger.info("Starting G21 file analysis")
    analyze_g21_files()
    logger.info("G21 analysis complete") 