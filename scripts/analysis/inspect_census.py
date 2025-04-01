import zipfile
import os
from pathlib import Path
import pandas as pd

# Path to the census zip file
zip_path = Path('data/raw/census/2021_GCP_all_for_AUS_short-header.zip')

# Create temp directory if it doesn't exist
os.makedirs('temp_extract', exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as z:
    # Find files matching G01 and SA1/SA2
    g01_files = [f for f in z.namelist() if 'G01' in f and ('SA1' in f or 'SA2' in f)]
    
    print(f"Found {len(g01_files)} G01 files:")
    for i, file in enumerate(g01_files):
        print(f"{i+1}. {file}")
    
    # Extract and inspect the first few files
    for i, file in enumerate(g01_files[:2]):  # Limit to first 2 files
        print(f"\nExamining file: {file}")
        z.extract(file, 'temp_extract')
        
        # Read first few rows with pandas
        try:
            df = pd.read_csv(Path('temp_extract') / file, nrows=2)
            print(f"Shape: {df.shape}")
            
            # Look for the columns we need
            needed_columns = {
                "geo_code": ["SA1_CODE_2021", "SA2_CODE_2021", "region_id", "SA1_CODE21", "SA2_CODE21"],
                "total_persons": ["Tot_P_P"],
                "total_male": ["Tot_M_P", "Tot_P_M"],  # Check both possibilities
                "total_female": ["Tot_F_P", "Tot_P_F"],  # Check both possibilities
                "total_indigenous": ["Indigenous_P", "Indigenous_P_Tot_P"]  # Check both possibilities
            }
            
            found_columns = {}
            for our_col, possible_names in needed_columns.items():
                for col_name in possible_names:
                    if col_name in df.columns:
                        found_columns[our_col] = col_name
                        break
            
            print(f"Found matching columns:")
            for our_col, actual_col in found_columns.items():
                print(f"  {our_col} -> {actual_col}")
                # Print sample values
                print(f"    Sample: {df[actual_col].head(2).tolist()}")
            
            # Check what columns are missing
            missing = [our_col for our_col in needed_columns.keys() if our_col not in found_columns]
            if missing:
                print(f"Missing columns: {missing}")
                
        except Exception as e:
            print(f"Error reading file: {e}") 