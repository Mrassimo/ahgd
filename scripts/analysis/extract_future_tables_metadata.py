#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path

# Path to the metadata files
metadata_dir = Path("/Users/massimoraso/AHGD3/data/raw/Metadata")
metadata_file = metadata_dir / "Metadata_2021_GCP_DataPack_R1_R2.xlsx"
template_file = metadata_dir / "2021_GCP_Sequential_Template_R2.xlsx"

# Define the set of tables we already have implemented
implemented_tables = {"G17", "G18", "G19"}

# Define potential tables of interest for health demographics
# We'll look for tables related to health, disability, care, education, employment, etc.
health_keywords = [
    "health", "disability", "assistance", "care", "medical", "hospital", 
    "disease", "condition", "unpaid", "education", "employment", "occupation",
    "income", "housing", "internet", "indigenous", "age", "gender"
]

def extract_table_metadata():
    """Extract metadata for potentially useful tables beyond G17, G18, G19"""
    print(f"Extracting metadata from: {metadata_file}")
    print(f"Looking for tables beyond the already implemented: {', '.join(implemented_tables)}")
    
    # First, read the Table Number, Name, Population sheet
    try:
        tables_df = pd.read_excel(
            metadata_file, 
            sheet_name="Table Number, Name, Population"
        )
        
        # Print the column names to understand the structure
        print("\nColumns in the Table Number, Name, Population sheet:")
        print(tables_df.columns.tolist())
        
        # Based on column names, filter to relevant columns
        if 'Table Number' in tables_df.columns and 'Table Name' in tables_df.columns:
            # Keep only rows where Table Number starts with 'G'
            g_tables = tables_df[tables_df['Table Number'].str.startswith('G', na=False)]
            
            # Filter out already implemented tables
            future_tables = g_tables[~g_tables['Table Number'].isin(implemented_tables)]
            
            # Find tables that match health-related keywords
            relevant_tables = []
            for _, row in future_tables.iterrows():
                table_name = row['Table Name'].lower() if isinstance(row['Table Name'], str) else ""
                table_num = row['Table Number']
                
                if any(keyword in table_name for keyword in health_keywords):
                    relevant_tables.append((table_num, row['Table Name'], row.get('Population', 'All')))
            
            # Print the relevant tables
            print(f"\nFound {len(relevant_tables)} potential tables for future implementation:")
            for table_num, table_name, population in sorted(relevant_tables):
                print(f"{table_num}: {table_name} [{population}]")
            
            # Now extract detailed information about these tables from the template file
            extract_table_structure(relevant_tables)
            
            return relevant_tables
        else:
            print("Could not find expected columns in the metadata file.")
            return []
            
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return []

def extract_table_structure(relevant_tables):
    """Extract structure information for the relevant tables from the template file"""
    print(f"\nExtracting structure details from: {template_file}")
    
    # Get all sheet names
    try:
        xlsx = pd.ExcelFile(template_file)
        sheets = xlsx.sheet_names
        print(f"Available sheets: {sheets}")
        
        for table_num, table_name, _ in relevant_tables:
            if table_num in sheets:
                print(f"\n{'='*80}")
                print(f"Structure for {table_num}: {table_name}")
                print(f"{'='*80}")
                
                # Read just enough rows to understand the structure
                df = pd.read_excel(template_file, sheet_name=table_num, nrows=25)
                
                # Extract useful information
                try:
                    if df.shape[0] > 2:
                        table_title = df.iloc[2, 0]  # Usually in cell A3
                        print(f"Table Title: {table_title}")
                    
                    # Look for column headers and structure
                    print("\nTable Structure:")
                    for i in range(min(15, df.shape[0])):
                        row = df.iloc[i]
                        # Print non-empty values in this row
                        non_empty = [(col, val) for col, val in enumerate(row) if pd.notna(val)]
                        if non_empty:
                            print(f"Row {i+1}: {non_empty}")
                    
                    # Try to identify variables (columns) 
                    print("\nPotential Variables:")
                    for col in range(min(5, df.shape[1])):
                        unique_vals = df.iloc[:, col].dropna().unique()
                        if len(unique_vals) > 0:
                            print(f"Column {col+1}: {unique_vals[:5]}...")
                except Exception as e:
                    print(f"Error analyzing sheet {table_num}: {e}")
            else:
                print(f"\nNo template available for {table_num}: {table_name}")
    
    except Exception as e:
        print(f"Error reading template file: {e}")

def write_metadata_to_file(tables):
    """Write the extracted metadata to a markdown file in the memory-bank directory"""
    output_file = Path("/Users/massimoraso/AHGD3/memory-bank/futureTables.md")
    
    content = [
        "# Future Census Tables\n",
        "This document contains metadata about additional ABS Census tables that could be ",
        "implemented in the future for health demographic analysis.\n",
        "## Potentially Relevant Tables\n"
    ]
    
    # Group tables by category
    categories = {}
    for table_num, table_name, population in tables:
        # Extract category from name (e.g., "HEALTH CONDITION" from "TYPE OF LONG-TERM HEALTH CONDITION")
        keywords_found = [k for k in health_keywords if k in table_name.lower()]
        category = keywords_found[0].title() if keywords_found else "Other"
        
        if category not in categories:
            categories[category] = []
        categories[category].append((table_num, table_name, population))
    
    # Write tables by category
    for category, tables in sorted(categories.items()):
        content.append(f"### {category} Related Tables\n")
        for table_num, table_name, population in sorted(tables):
            content.append(f"- **{table_num}**: {table_name}")
            if population != "All":
                content.append(f" [{population}]")
            content.append("\n")
        content.append("\n")
    
    # Add section for implementation recommendations
    content.append("## Implementation Recommendations\n")
    content.append("When implementing these tables, consider the following:\n\n")
    content.append("1. **Column Naming Conventions**: Like G17, G18, and G19, these tables may have inconsistent column naming\n")
    content.append("2. **Split Files**: Large tables may be split across multiple files (e.g., G##A, G##B)\n")
    content.append("3. **Metadata Analysis**: Always analyze the metadata Excel files first to understand the structure\n")
    content.append("4. **Pattern Implementation**: Follow the established pattern for flexible column mapping\n")
    content.append("5. **Documentation**: Update the Memory Bank with any new insights about table structures\n")
    
    # Write to file
    with open(output_file, "w") as f:
        f.writelines(content)
    
    print(f"\nMetadata written to: {output_file}")

if __name__ == "__main__":
    tables = extract_table_metadata()
    if tables:
        write_metadata_to_file(tables)
    else:
        print("No tables extracted, skipping output file creation.") 