#!/usr/bin/env python3
import pandas as pd
import os

def examine_metadata_file(file_path, search_terms=None):
    """
    Examine an Excel file for information about specific tables or columns.
    
    Args:
        file_path: Path to the Excel file
        search_terms: List of terms to search for (e.g., ['G17', 'G18', 'G19'])
    """
    print(f"\n\nExamining file: {os.path.basename(file_path)}")
    print("-" * 80)
    
    # Get all sheet names
    xl = pd.ExcelFile(file_path)
    print(f"Sheets in the file: {xl.sheet_names}")
    
    results = {}
    
    # Process each sheet
    for sheet_name in xl.sheet_names:
        print(f"\nReading sheet: {sheet_name}")
        try:
            # Read the first few rows to get a sense of the data
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            print(f"Sheet dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Column names: {list(df.columns)[:10]}...")
            
            # If search terms provided, look for them in the sheet
            if search_terms:
                matches = []
                for term in search_terms:
                    # Search in all string columns
                    for col in df.select_dtypes(include=['object']).columns:
                        matches_in_col = df[df[col].astype(str).str.contains(term, na=False)]
                        if not matches_in_col.empty:
                            print(f"Found '{term}' in column '{col}'")
                            # Store the matching rows for this term
                            if term not in results:
                                results[term] = []
                            results[term].append({
                                'sheet': sheet_name,
                                'column': col,
                                'rows': matches_in_col
                            })
        except Exception as e:
            print(f"Error reading sheet {sheet_name}: {e}")
    
    # Print detailed results for each term
    if search_terms:
        for term in search_terms:
            if term in results:
                print(f"\n{'='*40}\nDetailed results for '{term}':\n{'='*40}")
                for match in results[term]:
                    print(f"\nSheet: {match['sheet']}, Column: {match['column']}")
                    print("-" * 60)
                    print(match['rows'].head(10))  # Show first 10 matching rows
    
    return results

# Path to the metadata files
metadata_dir = "/Users/massimoraso/AHGD3/data/raw/Metadata"
metadata_files = [
    os.path.join(metadata_dir, "Metadata_2021_GCP_DataPack_R1_R2.xlsx"),
    os.path.join(metadata_dir, "2021_GCP_Sequential_Template_R2.xlsx"),
    os.path.join(metadata_dir, "2021Census_geog_desc_1st_2nd_3rd_release.xlsx")
]

# Search terms for the tables we're interested in
search_terms = ["G17", "G18", "G19"]

# Examine each metadata file
for file_path in metadata_files:
    examine_metadata_file(file_path, search_terms) 