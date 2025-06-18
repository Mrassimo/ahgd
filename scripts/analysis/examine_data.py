#!/usr/bin/env python3
"""
Quick data examination script to understand file structures and contents.
"""

import zipfile
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def examine_zip_file(zip_path: Path, max_files_to_show: int = 10):
    """Examine contents of a ZIP file"""
    logger.info(f"📦 Examining: {zip_path.name}")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            file_list = zip_file.namelist()
            total_files = len(file_list)
            
            logger.info(f"   Total files: {total_files}")
            logger.info(f"   First {min(max_files_to_show, total_files)} files:")
            
            for file_name in file_list[:max_files_to_show]:
                file_info = zip_file.getinfo(file_name)
                size_mb = file_info.file_size / (1024 * 1024)
                logger.info(f"     - {file_name} ({size_mb:.2f} MB)")
            
            if total_files > max_files_to_show:
                logger.info(f"     ... and {total_files - max_files_to_show} more files")
                
    except Exception as e:
        logger.error(f"   Error examining {zip_path.name}: {str(e)}")

def examine_excel_file(excel_path: Path):
    """Quick look at Excel file structure"""
    logger.info(f"📊 Examining: {excel_path.name}")
    
    try:
        # Just read the first few rows to understand structure
        df = pd.read_excel(excel_path, nrows=5)
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)}")
        logger.info(f"   First few rows:")
        for idx, row in df.head(3).iterrows():
            logger.info(f"     Row {idx}: {dict(row)}")
            
    except Exception as e:
        logger.error(f"   Error examining {excel_path.name}: {str(e)}")

def examine_csv_file(csv_path: Path):
    """Quick look at CSV file structure"""
    logger.info(f"📄 Examining: {csv_path.name}")
    
    try:
        # Just read the first few rows to understand structure  
        df = pd.read_csv(csv_path, nrows=5)
        logger.info(f"   Shape: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)}")
        logger.info(f"   First few rows:")
        for idx, row in df.head(3).iterrows():
            logger.info(f"     Row {idx}: {dict(row)}")
            
    except Exception as e:
        logger.error(f"   Error examining {csv_path.name}: {str(e)}")

def main():
    """Examine all downloaded data files"""
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    
    logger.info("🔍 Examining downloaded data files...")
    
    # Look at all categories
    for category_dir in data_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        logger.info(f"\n📁 Category: {category_dir.name}")
        
        for file_path in category_dir.iterdir():
            if file_path.suffix.lower() == '.zip':
                examine_zip_file(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                examine_excel_file(file_path)
            elif file_path.suffix.lower() == '.csv':
                examine_csv_file(file_path)
    
    # Also examine files in the root data directory
    logger.info(f"\n📁 Root files:")
    for file_path in data_dir.iterdir():
        if file_path.is_file():
            if file_path.suffix.lower() == '.zip':
                examine_zip_file(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                examine_excel_file(file_path)
            elif file_path.suffix.lower() == '.csv':
                examine_csv_file(file_path)

if __name__ == "__main__":
    main()