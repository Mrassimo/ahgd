#!/usr/bin/env python3
import pandas as pd
import os
from pathlib import Path

# Path to the metadata files
metadata_dir = config.PATHS['RAW_DATA_DIR'] / "Metadata"
template_file = metadata_dir / "2021_GCP_Sequential_Template_R2.xlsx"

# Define the set of tables we already have implemented
implemented_tables = {"G17", "G18", "G19"}

# Define tables of potential interest for health demographics
health_tables = {
    "G20": "COUNT OF SELECTED LONG-TERM HEALTH CONDITIONS BY AGE BY SEX",
    "G21": "TYPE OF LONG-TERM HEALTH CONDITION BY SELECTED PERSON CHARACTERISTICS",
    "G23": "VOLUNTARY WORK FOR AN ORGANISATION OR GROUP BY AGE BY SEX", 
    "G25": "UNPAID ASSISTANCE TO A PERSON WITH DISABILITY, HEALTH CONDITION OR DUE TO OLD AGE BY AGE BY SEX",
    "G46": "LABOUR FORCE STATUS BY AGE BY SEX",
    "G49": "HIGHEST NON-SCHOOL QUALIFICATION: LEVEL OF EDUCATION BY AGE BY SEX",
}

def extract_table_structure(table_id):
    """Extract structure information for a specific table"""
    print(f"\n{'='*80}")
    print(f"Extracting structure for {table_id}")
    print(f"{'='*80}")
    
    try:
        # Read the sheet for this table
        df = pd.read_excel(template_file, sheet_name=table_id, nrows=30)
        
        # Extract the official title from row 2, column 0
        if df.shape[0] > 2:
            title = df.iloc[2, 0]
            print(f"Table Title: {title}")
        else:
            title = f"G{table_id}"
            print(f"Could not extract title, using: {title}")
        
        # Look for column headers and structure
        print("\nTable Structure:")
        for i in range(min(10, df.shape[0])):
            row = df.iloc[i]
            non_empty = [(j, val) for j, val in enumerate(row) if pd.notna(val)]
            if non_empty:
                print(f"Row {i+1}: {non_empty}")
        
        # Find variable descriptions (often starts around row 10-15)
        print("\nPotential Variables:")
        variable_rows = []
        for i in range(10, min(30, df.shape[0])):
            row = df.iloc[i]
            non_empty = [(j, val) for j, val in enumerate(row) if pd.notna(val)]
            if non_empty and len(non_empty) > 1:  # At least two values in the row
                print(f"Row {i+1}: {non_empty}")
                variable_rows.append((i, non_empty))
        
        return {
            "table_id": table_id,
            "title": title,
            "variables": variable_rows
        }
    except Exception as e:
        print(f"Error analyzing {table_id}: {e}")
        return {
            "table_id": table_id,
            "title": f"G{table_id}",
            "error": str(e)
        }

def write_to_memory_bank(table_info):
    """Write all the table information to the memory bank"""
    output_file = Path("/Users/massimoraso/AHGD3/memory-bank/futureTables.md")
    
    content = [
        "# Future Census Tables for Implementation\n",
        "This document contains information about future ABS Census tables that could be implemented ",
        "in our ETL pipeline, based on analysis of the ABS metadata files.\n",
        "\n## Health-Related Tables\n\n"
    ]
    
    # Add information about each table
    for table in table_info:
        table_id = table["table_id"]
        title = table["title"]
        
        content.append(f"### {table_id}: {title}\n\n")
        
        # Add a description based on the table name
        if "HEALTH CONDITION" in title:
            content.append("This table contains data about health conditions across the population. ")
        elif "UNPAID ASSISTANCE" in title:
            content.append("This table contains data about unpaid care provided to people with disabilities or health conditions. ")
        elif "VOLUNTARY WORK" in title:
            content.append("This table contains data about voluntary work participation. ")
        elif "LABOUR FORCE" in title:
            content.append("This table contains data about employment status. ")
        elif "QUALIFICATION" in title:
            content.append("This table contains data about education levels. ")
        
        # Add information about table structure
        content.append("It follows ABS's standard structure for Census tables:\n\n")
        content.append("- Contains data broken down by age groups and sex\n")
        content.append("- Includes totals for each category\n")
        content.append("- Uses standard ABS column naming conventions\n\n")
        
        # Add implementation notes
        content.append("**Implementation Notes:**\n\n")
        content.append("1. Like G17, G18, and G19, column names may have inconsistencies and typos\n")
        content.append("2. The data may be split across multiple files (e.g., G20A, G20B) if it contains many columns\n")
        content.append("3. Implement using the flexible column mapping pattern established for G18 and G19\n")
        content.append("4. Follow the standardized output naming convention: `g{table_id}_{gender}_{characteristic}_{age_group}`\n\n")
        
        # Add information about potential applications
        content.append("**Potential Applications:**\n\n")
        if "HEALTH CONDITION" in title:
            content.append("- Analysis of health condition prevalence across different demographics\n")
            content.append("- Correlation of health conditions with other socioeconomic factors\n")
            content.append("- Health service planning based on condition prevalence\n\n")
        elif "UNPAID ASSISTANCE" in title:
            content.append("- Identification of carer populations and their demographics\n")
            content.append("- Analysis of unpaid care distribution across communities\n")
            content.append("- Planning for carer support services\n\n")
        elif "VOLUNTARY WORK" in title:
            content.append("- Analysis of volunteer demographics and distribution\n")
            content.append("- Correlation between volunteering and health outcomes\n")
            content.append("- Community engagement assessment\n\n")
        elif "LABOUR FORCE" in title:
            content.append("- Correlation between employment status and health outcomes\n")
            content.append("- Economic analysis of health-related employment patterns\n")
            content.append("- Identification of at-risk populations based on employment status\n\n")
        elif "QUALIFICATION" in title:
            content.append("- Correlation between education levels and health outcomes\n")
            content.append("- Analysis of healthcare workforce qualifications\n")
            content.append("- Targeting health education programs\n\n")
        
        content.append("---\n\n")
    
    # Add implementation strategy section
    content.append("## Implementation Strategy\n\n")
    content.append("When implementing these tables, we recommend the following approach:\n\n")
    content.append("1. **Start with G20 and G21**: These tables directly extend our existing health condition data (G19)\n")
    content.append("2. **Implement G25 next**: Unpaid care assistance is closely related to health conditions\n")
    content.append("3. **Add G23**: Volunteer work can provide context for community health support\n")
    content.append("4. **Follow with G46 and G49**: Employment and education status provide socioeconomic context\n\n")
    
    content.append("For each implementation:\n\n")
    content.append("1. Add the table pattern to `config.py`\n")
    content.append("2. Create a dedicated processing module (`process_g##.py`)\n")
    content.append("3. Implement flexible column mapping like in G18 and G19\n")
    content.append("4. Create comprehensive tests\n")
    content.append("5. Update documentation\n\n")
    
    # Write to file
    with open(output_file, "w") as f:
        f.writelines(content)
    
    print(f"\nWritten future tables information to: {output_file}")

if __name__ == "__main__":
    print(f"Extracting structure for future health-related tables: {', '.join(health_tables.keys())}")
    
    table_info = []
    for table_id in health_tables:
        info = extract_table_structure(table_id)
        table_info.append(info)
    
    write_to_memory_bank(table_info) 