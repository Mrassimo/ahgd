import os
import polars as pl
import pandas as pd
from pathlib import Path
import gc
import sys

# Define the output directory
OUTPUT_DIR = "/Users/massimoraso/AHGD3/output"
REPORTS_DIR = os.path.join(OUTPUT_DIR, "profiling_reports")

def generate_basic_html_report(df, table_name, output_file):
    """Generate a basic HTML report with table statistics."""
    # Get basic statistics
    num_rows = len(df)
    num_cols = len(df.columns)
    
    # Start HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Basic Data Profile - {table_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        h1, h2 {{ color: #333; }}
    </style>
</head>
<body>
    <h1>Data Profile: {table_name}</h1>
    
    <h2>Basic Information</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Number of Rows</td><td>{num_rows}</td></tr>
        <tr><td>Number of Columns</td><td>{num_cols}</td></tr>
    </table>
    
    <h2>Column Information</h2>
    <table>
        <tr><th>Column Name</th><th>Data Type</th><th>Sample Values</th></tr>
"""
    
    # Add each column
    for col in df.columns:
        col_type = str(df[col].dtype)
        sample_values = ", ".join([str(x) for x in df[col].head(3).to_list()])
        html += f"""        <tr><td>{col}</td><td>{col_type}</td><td>{sample_values}</td></tr>\n"""
    
    # Close HTML
    html += """    </table>
</body>
</html>
"""
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(html)
    
    return output_file

def generate_data_profiling_reports():
    """Generate simple data profiling reports for all Parquet files in the output directory."""
    # Create the reports directory if it doesn't exist
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    parquet_files = list(Path(OUTPUT_DIR).glob('*.parquet'))
    
    if not parquet_files:
        print(f"No Parquet files found in {OUTPUT_DIR}")
        return
    
    print(f"\n===== GENERATING SIMPLE DATA PROFILES =====\n")
    
    for i, parquet_file in enumerate(parquet_files):
        file_name = parquet_file.name
        file_path = str(parquet_file)
        table_name = file_name.replace('.parquet', '')
        
        report_file = os.path.join(REPORTS_DIR, f"{table_name}_profile.html")
        
        try:
            print(f"[{i+1}/{len(parquet_files)}] Processing {file_name}...")
            
            # Read Parquet file using Polars with memory optimization
            print(f"  Reading file...")
            df_pl = pl.scan_parquet(file_path).limit(1000).collect()  # Limit to 1000 rows
            
            # Generate simple HTML report
            print(f"  Generating simple HTML report...")
            output_file = generate_basic_html_report(df_pl, table_name, report_file)
            print(f"  Report saved to {output_file}")
            
            # Free up memory
            del df_pl
            gc.collect()
            
        except Exception as e:
            print(f"  Error generating profile for {file_name}: {e}")
    
    print(f"\nAll basic profiling reports have been saved to {REPORTS_DIR}")
    print(f"Note: Reports are based on samples of 1000 rows each to avoid memory issues.")

if __name__ == "__main__":
    # Generate data profiling reports
    generate_data_profiling_reports()
    
    print("\nProcess completed successfully!") 