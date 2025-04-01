#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path

# Path to the metadata files
metadata_dir = Path("/Users/massimoraso/AHGD3/data/raw/Metadata")
metadata_file = metadata_dir / "Metadata_2021_GCP_DataPack_R1_R2.xlsx"
template_file = metadata_dir / "2021_GCP_Sequential_Template_R2.xlsx"

def explore_excel_file(file_path):
    """Explore an Excel file to understand its structure"""
    print(f"\nExploring file: {file_path}")
    
    try:
        # Get sheet names
        xlsx = pd.ExcelFile(file_path)
        sheets = xlsx.sheet_names
        print(f"Sheets in the file: {sheets}")
        
        # Examine each sheet
        for sheet_name in sheets:
            print(f"\n--- Sheet: {sheet_name} ---")
            
            # Read a few rows to see structure
            df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=10)
            
            # Print shape
            print(f"Shape: {df.shape}")
            
            # Print column names
            print("Column names:")
            for i, col in enumerate(df.columns):
                print(f"  {i}: {col}")
            
            # Print first few rows
            print("\nFirst few rows:")
            for i in range(min(5, df.shape[0])):
                row_values = []
                for j in range(min(5, df.shape[1])):
                    val = df.iloc[i, j]
                    if pd.notna(val):
                        row_values.append(f"({j}: {val})")
                if row_values:
                    print(f"  Row {i}: {', '.join(row_values)}")
            
            # Look for table info in first column
            print("\nLooking for table information in first column:")
            first_col = df.iloc[:, 0]
            for i, val in enumerate(first_col):
                if pd.notna(val) and isinstance(val, str) and val.startswith('G'):
                    print(f"  Row {i}: {val}")
            
    except Exception as e:
        print(f"Error exploring file: {e}")

if __name__ == "__main__":
    print("Exploring metadata files...")
    explore_excel_file(metadata_file)
    explore_excel_file(template_file) 