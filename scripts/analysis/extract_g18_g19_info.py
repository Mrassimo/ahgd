#!/usr/bin/env python3
import pandas as pd
import os

# Path to the metadata files
metadata_dir = "/Users/massimoraso/AHGD3/data/raw/Metadata"
template_file = os.path.join(metadata_dir, "2021_GCP_Sequential_Template_R2.xlsx")

def extract_table_structure(sheet_name):
    """Extract the column structure of a specific Census table."""
    print(f"\n{'='*80}\nExtracting table structure for {sheet_name}\n{'='*80}")
    
    try:
        # Read the sheet
        df = pd.read_excel(template_file, sheet_name=sheet_name)
        
        # Print table information
        table_title = df.iloc[2, 0]  # Extract table title from cell A3
        print(f"Table Title: {table_title}")
        
        # Print the first few rows to see the header structure
        print("\nTable Header Structure:")
        print(df.head(10))
        
        # Look for column names - typically in row 11-15
        print("\nPotential Column Names:")
        for i in range(10, 20):  # Check rows 11-20
            try:
                row = df.iloc[i]
                non_empty_values = [val for val in row if pd.notna(val)]
                if non_empty_values:
                    print(f"Row {i+1}: {non_empty_values}")
            except:
                pass
        
        # Extract all column headers
        print("\nAll Column Values (first 5 columns):")
        for col_idx in range(min(5, len(df.columns))):
            col_values = df.iloc[:, col_idx].dropna().unique()
            print(f"Column {col_idx+1}: {list(col_values)[:10]}...")
        
        # Now try to identify the variable names (columns) for later use in our code
        print("\nAttempting to identify variable names for data processing:")
        for row_idx in range(15, 30):  # Search around rows 15-30
            try:
                row = df.iloc[row_idx]
                if any('Total' in str(val) for val in row if isinstance(val, str)):
                    print(f"Found row with 'Total' at row {row_idx+1}:")
                    print(row.tolist())
            except:
                pass
            
        return df
    except Exception as e:
        print(f"Error reading sheet {sheet_name}: {e}")
        return None

# Extract structure for G18 and G19
g18_df = extract_table_structure("G18")
g19_df = extract_table_structure("G19")

# Now look at the exact column values in the actual CSV files
print("\n\n" + "="*80)
print("Examining actual Census ZIP files to find G18 and G19 column names")
print("="*80)

# Function to check a CSV file for its column names
def examine_csv_in_zip(zip_path, search_pattern):
    import zipfile
    import re
    import csv
    from io import StringIO
    
    if not os.path.exists(zip_path):
        print(f"ZIP file not found: {zip_path}")
        return
    
    pattern = re.compile(search_pattern, re.IGNORECASE)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all files in the ZIP
        file_list = zip_ref.namelist()
        
        # Filter files matching our pattern
        matching_files = [f for f in file_list if pattern.search(f)]
        
        if not matching_files:
            print(f"No files matching pattern '{search_pattern}' found in {zip_path}")
            return
        
        print(f"Found {len(matching_files)} matching files. Examining the first few:")
        
        # Examine the first few matching files
        for file_name in matching_files[:3]:
            print(f"\nExamining file: {file_name}")
            
            try:
                # Read the CSV file
                with zip_ref.open(file_name) as f:
                    # Read a sample (first few lines) to identify column names
                    sample = StringIO(f.read(5000).decode('utf-8', errors='ignore'))
                    
                    # Use CSV reader to parse the header
                    csv_reader = csv.reader(sample)
                    headers = next(csv_reader)
                    
                    print(f"CSV Headers: {headers}")
                    
                    # Read a few rows for sample data
                    print("Sample data (first 3 rows):")
                    for _ in range(3):
                        try:
                            row = next(csv_reader)
                            print(row)
                        except StopIteration:
                            break
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

# Path to the Census ZIP file
census_zip = "/Users/massimoraso/AHGD3/data/raw/census/2021_GCP_all_for_AUS_short-header.zip"

# Check for G18 files
print("\nSearching for G18 files (Core Activity Need for Assistance):")
examine_csv_in_zip(census_zip, r"G18.*\.csv$")

# Check for G19 files
print("\nSearching for G19 files (Long-Term Health Conditions):")
examine_csv_in_zip(census_zip, r"G19.*\.csv$") 