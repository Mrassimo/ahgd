#!/usr/bin/env python3
import pandas as pd
import os
import sys
from pathlib import Path
import re
import logging
from typing import List, Tuple, Dict, Any, Optional
import openpyxl # Needed to read xlsx

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Adjust these paths if your metadata files are elsewhere
# Use environment variables or default to relative paths from the script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT_DIR = SCRIPT_DIR.parent.parent # Assumes scripts/analysis is two levels down from project root
METADATA_BASE_DIR = Path(os.getenv('METADATA_DIR', PROJECT_ROOT_DIR / 'data/raw/Metadata'))
MEMORY_BANK_DIR = Path(os.getenv('MEMORY_BANK_DIR', PROJECT_ROOT_DIR / 'memory-bank'))

METADATA_FILE = METADATA_BASE_DIR / "Metadata_2021_GCP_DataPack_R1_R2.xlsx"
TEMPLATE_FILE = METADATA_BASE_DIR / "2021_GCP_Sequential_Template_R2.xlsx"
OUTPUT_MD_FILE = MEMORY_BANK_DIR / "all_census_tables_metadata.md"
# ---

def get_all_g_tables() -> List[Tuple[str, str]]:
    """Extracts all G-table numbers and names from the main metadata file."""
    g_tables = []
    logger.info(f"Attempting to read metadata from: {METADATA_FILE}")
    if not METADATA_FILE.exists():
        logger.error(f"Metadata file not found: {METADATA_FILE}")
        return []
      
    try:
        # Find the sheet containing the list of tables
        xlsx = pd.ExcelFile(METADATA_FILE)
        list_sheet_name = None
        # More robust sheet finding
        possible_sheet_names = [
            "Table Number, Name, Population", 
            "List of tables", 
            "Tables" # Add other potential names if needed
        ]
        for sheet in xlsx.sheet_names:
             if any(possible_name.lower() in sheet.lower() for possible_name in possible_sheet_names):
                 list_sheet_name = sheet
                 logger.info(f"Found table list sheet: '{list_sheet_name}'")
                 break
      
        if not list_sheet_name:
            logger.error(f"Could not find a 'List of tables' sheet in {METADATA_FILE.name}. Sheets found: {xlsx.sheet_names}")
            return []

        # Determine header row by looking for 'Table Number'
        df_peek = pd.read_excel(METADATA_FILE, sheet_name=list_sheet_name, header=None, nrows=15)
        header_row = 0
        for idx, row in df_peek.iterrows():
            if any('table number' in str(cell).lower() for cell in row):
                header_row = idx
                logger.info(f"Detected header row at index {header_row}")
                break
        else:
             logger.warning("Could not reliably detect header row, assuming 7 (0-indexed).")
             header_row = 7 # Default if not found

        logger.info(f"Reading table list from sheet: '{list_sheet_name}', skipping {header_row} rows.")
        df = pd.read_excel(METADATA_FILE, sheet_name=list_sheet_name, skiprows=header_row)

        # Find columns likely containing table number and name (case-insensitive)
        table_num_col = next((col for col in df.columns if 'number' in str(col).lower()), None)
        table_name_col = next((col for col in df.columns if 'name' in str(col).lower()), None)
      
        if not table_num_col or not table_name_col:
            logger.error(f"Could not identify 'Table Number' or 'Table Name' columns in sheet '{list_sheet_name}'. Found: {df.columns.tolist()}")
            return []
          
        logger.info(f"Using columns: '{table_num_col}' (Number), '{table_name_col}' (Name)")

        # Filter for G-tables (handle potential float conversion if numbers read as float)
        df[table_num_col] = df[table_num_col].astype(str).str.replace(r'\.0$', '', regex=True) # Remove trailing .0
        g_table_rows = df[df[table_num_col].str.match(r'^G\d+[A-Z]?$', na=False)]
      
        g_tables = list(g_table_rows[[table_num_col, table_name_col]].itertuples(index=False, name=None))
        logger.info(f"Found {len(g_tables)} G-tables in metadata.")
      
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {METADATA_FILE}")
    except Exception as e:
        logger.error(f"Error reading metadata file {METADATA_FILE.name}: {e}", exc_info=True)
      
    return sorted(g_tables)

def extract_table_structure(table_id: str, xlsx: pd.ExcelFile) -> Dict[str, Any]:
    """Extracts structure information for a specific table from the template file."""
    structure = {"id": table_id, "title": f"G{table_id} (Title not found)", "headers": [], "variables": [], "notes": []}
  
    if table_id not in xlsx.sheet_names:
        logger.warning(f"Sheet '{table_id}' not found in template file.")
        structure["error"] = "Sheet not found in template."
        return structure

    try:
        logger.debug(f"Reading sheet: {table_id}")
        # Read more rows to better capture structure, don't assume header row
        df = pd.read_excel(xlsx, sheet_name=table_id, header=None, nrows=50) 

        # --- Extract Title (Usually around row 3, col A) ---
        for i in range(min(5, df.shape[0])): # Check first few rows
             val = df.iloc[i, 0]
             if pd.notna(val) and isinstance(val, str) and val.startswith(f'G{table_id.replace("G","").split("A")[0]}'): # More robust title check
                 structure["title"] = " ".join(str(val).split()) # Clean extra whitespace
                 logger.debug(f"Found title: {structure['title']}")
                 break
      
        # --- Extract Headers (Rows before the main data variables) ---
        # Look for rows with multiple non-null values, often defining dimensions like Age/Sex/Condition
        header_end_row = 0
        potential_header_rows = []
        for r in range(5, 20): # Search range for headers
            if r >= df.shape[0]: break
            # Get non-null string values, stripping whitespace
            row_vals = [str(v).strip() for v in df.iloc[r].tolist() if pd.notna(v) and str(v).strip()]
            # Crude check: if a row has several short-ish items, it might be a header
            if len(row_vals) > 2 and all(len(v) < 50 for v in row_vals):
                 potential_header_rows.append(f"Row {r+1}: {', '.join(row_vals)}")
                 header_end_row = r
            elif len(row_vals) > 0 and any(kw in str(row_vals[0]).lower() for kw in ['total', 'persons', 'males', 'females']):
                 # Stop if we hit something looking like data totals
                 logger.debug(f"Stopping header search at row {r+1} due to potential data totals.")
                 break
        structure["headers"] = potential_header_rows
               
        # --- Extract Variables (Often start after headers, look for definitions in early columns) ---
        potential_var_start = header_end_row + 1
        variable_candidates = []
        for r in range(potential_var_start, min(potential_var_start + 30, df.shape[0])):
             # Look for non-empty cells in the first few columns
             row_start = [str(df.iloc[r, c]).strip() for c in range(min(3, df.shape[1])) if pd.notna(df.iloc[r, c]) and str(df.iloc[r, c]).strip()]
             if row_start:
                 # Check if it looks like a variable definition (not just 'Total' or numeric)
                 is_total = all(str(v).lower() == 'total' for v in row_start)
                 is_numeric = all(re.match(r'^-?\d+(\.\d+)?$', v) for v in row_start)
                 if not is_total and not is_numeric:
                      variable_candidates.append(f"Row {r+1}: {' | '.join(row_start)}")
        structure["variables"] = variable_candidates

        # --- Extract Notes (Often at the bottom) ---
        notes_candidates = []
        for r in range(df.shape[0] - 1, max(0, df.shape[0] - 10), -1):
             val = df.iloc[r, 0]
             # Check if it looks like a footnote (e.g., starts with (a), (b), *)
             if pd.notna(val) and isinstance(val, str) and re.match(r'^\s*(\(\w+\)|\*|\d+\.)', val.strip()):
                 notes_candidates.insert(0, f"- {val.strip()}") # Prepend notes
             elif pd.notna(val) and isinstance(val, str) and len(val) > 20: # Catch longer text notes
                 notes_candidates.insert(0, f"- {val.strip()}")
        structure["notes"] = notes_candidates

    except Exception as e:
        logger.error(f"Error parsing sheet '{table_id}': {e}", exc_info=True)
        structure["error"] = str(e)
      
    return structure

def write_metadata_to_markdown(all_tables_metadata: List[Dict[str, Any]]):
    """Writes the extracted metadata for all tables to a markdown file."""
    OUTPUT_MD_FILE.parent.mkdir(parents=True, exist_ok=True)
  
    content = ["# ABS Census 2021 GCP - All G-Table Metadata\n\n"]
    content.append(f"*Source Metadata File:* `{METADATA_FILE.name}`\n")
    content.append(f"*Source Template File:* `{TEMPLATE_FILE.name}`\n\n")
    content.append("This document summarizes the structure of all G-tables found in the metadata template file. Use this as a reference when implementing processing logic for specific tables.\n\n")

    for table_info in all_tables_metadata:
        content.append(f"## {table_info['id']}: {table_info['title']}\n\n")
      
        if table_info.get("error"):
            content.append(f"**Error processing this table:** {table_info['error']}\n\n")
            content.append("---\n\n")
            continue

        if table_info["headers"]:
            content.append("**Potential Headers (from Template Rows ~5-20):**\n")
            content.extend([f"  - {h}\n" for h in table_info["headers"]])
            content.append("\n")
        else:
            content.append("**Potential Headers:** (None detected in typical range)\n\n")
          
        if table_info["variables"]:
            content.append("**Potential Variables/Characteristics (from Template Rows ~20-50, first few columns):**\n")
            content.extend([f"  - {v}\n" for v in table_info["variables"]])
            content.append("\n")
        else:
            content.append("**Potential Variables/Characteristics:** (None detected in typical range)\n\n")


        if table_info["notes"]:
            content.append("**Notes (from bottom of template sheet):**\n")
            content.extend([f"  {n}\n" for n in table_info["notes"]]) # Indent notes
            content.append("\n")
        else:
            content.append("**Notes:** (None detected)\n\n")
          
        content.append("---\n\n")

    try:
        with open(OUTPUT_MD_FILE, "w", encoding="utf-8") as f:
            f.writelines(content)
        logger.info(f"Successfully wrote all table metadata to: {OUTPUT_MD_FILE}")
    except Exception as e:
        logger.error(f"Failed to write markdown file {OUTPUT_MD_FILE}: {e}")


if __name__ == "__main__":
    logger.info("Starting metadata extraction for all G-tables...")
  
    if not METADATA_FILE.exists():
         logger.error(f"Metadata file listing tables not found: {METADATA_FILE}")
         sys.exit(1)
    if not TEMPLATE_FILE.exists():
         logger.error(f"Template file with structures not found: {TEMPLATE_FILE}")
         sys.exit(1)
       
    g_tables = get_all_g_tables()
  
    if not g_tables:
        logger.error("No G-tables found in metadata list. Exiting.")
        sys.exit(1)
      
    all_metadata = []
    try:
        logger.info(f"Opening template file: {TEMPLATE_FILE.name}")
        # Use openpyxl engine explicitly if needed, though default should work
        xlsx_template = pd.ExcelFile(TEMPLATE_FILE, engine='openpyxl') 
      
        found_sheets = xlsx_template.sheet_names
        logger.info(f"Found {len(found_sheets)} sheets in template file.")
      
        processed_ids = set()
        for table_id, table_name in g_tables:
             if table_id in processed_ids: continue # Skip duplicates like G19A/B/C if G19 was listed
           
             logger.info(f"Extracting structure for: {table_id} - {table_name}")
             # Handle potential suffixes like G19A, G19B - check if base G19 exists first
             base_table_id = re.match(r'(G\d+)', table_id).group(1) if re.match(r'(G\d+)', table_id) else table_id
           
             # If the specific sheet (e.g., G19A) exists, use it. Otherwise, try the base (e.g., G19).
             sheet_to_process = table_id if table_id in found_sheets else base_table_id
           
             if sheet_to_process in found_sheets:
                 metadata = extract_table_structure(sheet_to_process, xlsx_template)
                 # Use the original ID from the list for the output dict
                 metadata['id'] = table_id 
                 metadata['title'] = f"{table_id}: {metadata['title']}" # Prepend ID to title
                 all_metadata.append(metadata)
                 processed_ids.add(base_table_id) # Mark base ID as processed
             else:
                 logger.warning(f"Sheet for {table_id} or base {base_table_id} not found in template.")
                 all_metadata.append({"id": table_id, "title": table_name, "error": "Sheet not found in template."})

        write_metadata_to_markdown(all_metadata)
      
    except Exception as e:
        logger.error(f"An error occurred during template file processing: {e}", exc_info=True)
        sys.exit(1)
      
    logger.info("Metadata extraction complete.") 