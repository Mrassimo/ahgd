#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path

# Path to the metadata files
metadata_dir = Path("/Users/massimoraso/AHGD3/data/raw/Metadata")
metadata_file = metadata_dir / "Metadata_2021_GCP_DataPack_R1_R2.xlsx"
template_file = metadata_dir / "2021_GCP_Sequential_Template_R2.xlsx"

def extract_g21_metadata():
    """Extract metadata specifically for G21 table"""
    print(f"Extracting G21 metadata from: {metadata_file}")
    
    # Try to find info in the "List of tables" sheet first
    try:
        tables_df = pd.read_excel(
            metadata_file, 
            sheet_name="List of tables"
        )
        
        print("\nSearching for G21 in List of tables sheet")
        # Search for G21 in any column
        for col in tables_df.columns:
            g21_rows = tables_df[tables_df[col].astype(str).str.contains('G21', na=False)]
            if not g21_rows.empty:
                print(f"\nFound G21 references in column {col}:")
                print(g21_rows.to_string(index=False))
                
    except Exception as e:
        print(f"Error searching List of tables sheet: {e}")

def extract_g21_structure():
    """Extract structure information for G21 from the template file"""
    print(f"\nExtracting G21 structure details from: {template_file}")
    
    # Get all sheet names
    try:
        xlsx = pd.ExcelFile(template_file)
        sheets = xlsx.sheet_names
        print(f"Available sheets: {sheets}")
        
        if 'G21' in sheets:
            print("\n" + "="*80)
            print("Structure for G21 table")
            print("="*80)
            
            # Read more rows to capture the structure
            df = pd.read_excel(template_file, sheet_name='G21', nrows=100)
            
            # Extract useful information
            try:
                if df.shape[0] > 2:
                    table_title = df.iloc[2, 0]  # Usually in cell A3
                    print(f"Table Title: {table_title}")
                
                # Look for column headers (row with health conditions)
                condition_row = None
                for i in range(df.shape[0]):
                    if 'Arthritis' in str(df.iloc[i, 1]):
                        condition_row = i
                        break
                
                if condition_row is not None:
                    print(f"\nHealth Conditions (Row {condition_row+1}):")
                    conditions = []
                    for col in range(1, 14):  # Columns B to N typically contain conditions
                        if pd.notna(df.iloc[condition_row, col]):
                            conditions.append(df.iloc[condition_row, col])
                    print(", ".join(conditions))
                
                # Look for characteristics (first column)
                print("\nPerson Characteristics:")
                characteristics = set()
                for i in range(15, min(50, df.shape[0])):
                    if pd.notna(df.iloc[i, 0]) and len(str(df.iloc[i, 0])) > 3:
                        characteristics.add(df.iloc[i, 0])
                
                for char in sorted(characteristics):
                    print(f"- {char}")
                
                # Show sample of data rows to understand structure
                print("\nSample Data Structure (first characteristic):")
                if len(characteristics) > 0:
                    first_char = sorted(characteristics)[0]
                    for i in range(15, df.shape[0]):
                        if str(df.iloc[i, 0]).startswith(first_char):
                            values = []
                            for col in range(df.shape[1]):
                                if pd.notna(df.iloc[i, col]):
                                    values.append(f"{col}:{df.iloc[i, col]}")
                            if values:
                                print(f"Row {i+1}: {', '.join(values)}")
                                if len(values) > 3:  # Stop after showing one complete row
                                    break
                
            except Exception as e:
                print(f"Error analyzing G21 sheet: {e}")
        else:
            print("\nNo G21 template available in the file")
    
    except Exception as e:
        print(f"Error reading template file: {e}")

def extract_actual_g21_file_structure():
    """Try to find an actual G21 file that was extracted and get its column structure"""
    print("\n" + "="*80)
    print("Looking for extracted G21 files")
    print("="*80)
    
    extract_dir = Path("/Users/massimoraso/AHGD3/data/raw/temp/extract/g21")
    if not extract_dir.exists():
        print(f"Extract directory {extract_dir} does not exist. Unable to check actual G21 files.")
        return
    
    g21_files = list(extract_dir.glob("**/2021Census_G21*.csv"))
    if not g21_files:
        print("No G21 CSV files found in the extract directory.")
        return
    
    print(f"Found {len(g21_files)} G21 files.")
    
    # Check the first file
    g21_file = g21_files[0]
    print(f"\nExamining file: {g21_file.name}")
    
    try:
        # Read the first few lines to get the column names
        df = pd.read_csv(g21_file, nrows=2)
        
        print("\nColumn names:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}: {col}")
        
        # Count the total number of columns
        print(f"\nTotal columns: {len(df.columns)}")
        
        # Try to categorize columns based on patterns
        condition_cols = [col for col in df.columns if any(c in col for c in ['Arth', 'Asth', 'Cancer', 'Diabetes', 'Heart', 'Kidney', 'Lung', 'Mental', 'Stroke'])]
        
        print(f"\nHealth condition columns identified: {len(condition_cols)}")
        print("Examples:", condition_cols[:5])
        
        # Try to identify characteristic patterns
        characteristics = set()
        for col in df.columns:
            parts = col.split('_')
            if len(parts) > 2:
                characteristic_part = '_'.join(parts[1:-1])  # Middle part is usually characteristic
                characteristics.add(characteristic_part)
        
        print("\nPossible characteristics identified:")
        for char in sorted(characteristics):
            if len(char) > 3:  # Filter out noise
                print(f"- {char}")
    
    except Exception as e:
        print(f"Error examining G21 file: {e}")

if __name__ == "__main__":
    extract_g21_metadata()
    extract_g21_structure()
    extract_actual_g21_file_structure() 