#!/usr/bin/env python3
"""
Direct fix for dimension tables to add unknown members robustly.
This script bypasses type compatibility issues in the dimension handler.
"""

import logging
import sys
import hashlib
from pathlib import Path
from datetime import datetime
import polars as pl

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("direct_fix")

def generate_surrogate_key(*args) -> str:
    """Generate a surrogate key from args using MD5 hash."""
    key_str = '_'.join([str(arg) for arg in args])
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def fix_dimension(dim_path: Path, dim_name: str, is_int_sk: bool = False):
    """Fix a dimension table by adding an unknown member."""
    try:
        # Read the dimension
        dim_df = pl.read_parquet(dim_path)
        logger.info(f"Processing {dim_name} with {len(dim_df)} rows")
        
        # Check if unknown member exists
        if "is_unknown" in dim_df.columns:
            unknown_rows = dim_df.filter(pl.col("is_unknown") == True)
            if len(unknown_rows) > 0:
                logger.info(f"Unknown member already exists in {dim_name}")
                return True
        else:
            # Add is_unknown column if missing
            dim_df = dim_df.with_columns(pl.lit(False).alias("is_unknown"))
        
        # Find surrogate key column
        sk_cols = [col for col in dim_df.columns if col.endswith("_sk")]
        if not sk_cols:
            logger.warning(f"No surrogate key found in {dim_name}")
            return False
        
        sk_col = sk_cols[0]
        
        # Get the first row as a template and modify for unknown member
        if len(dim_df) > 0:
            # Convert to dictionary 
            rows = dim_df.to_dicts()
            first_row = rows[0].copy()
            
            # Modify for unknown member
            for col, value in first_row.items():
                if col == sk_col:
                    # For surrogate key, use a special value
                    if is_int_sk or str(dim_df[col].dtype).startswith("Int"):
                        # For integer keys, use a negative number
                        first_row[col] = -9999
                    else:
                        # For string keys, use a hash
                        first_row[col] = generate_surrogate_key("UNKNOWN", dim_name)
                elif col == "is_unknown":
                    first_row[col] = True
                else:
                    # For other columns, use appropriate unknowns
                    if isinstance(value, int):
                        first_row[col] = -1
                    elif isinstance(value, float):
                        first_row[col] = -1.0
                    elif isinstance(value, bool):
                        first_row[col] = False
                    elif isinstance(value, datetime) or str(type(value)).find("datetime") >= 0:
                        # Keep the same datetime for unknown member
                        pass
                    elif isinstance(value, str):
                        first_row[col] = "UNKNOWN"
            
            # Add the unknown member to rows
            rows.append(first_row)
            
            # Create new dataframe from all rows
            new_df = pl.DataFrame(rows)
            
            # Save the dataframe back
            new_df.write_parquet(dim_path)
            logger.info(f"Added unknown member to {dim_name}")
            return True
        else:
            logger.warning(f"Cannot add unknown member to empty table {dim_name}")
            return False
            
    except Exception as e:
        logger.error(f"Error fixing {dim_name}: {e}")
        return False

def main():
    """Main function to fix all dimensions."""
    # Path to output directory
    output_dir = Path("./output")
    
    # List of dimensions to fix
    dimensions = [
        ("geo_dimension", True),       # geo_sk is integer
        ("dim_time", True),            # time_sk is integer
        ("dim_health_condition", False),  # condition_sk is string
        ("dim_demographic", True),     # demographic_sk is integer (after our fix)
        ("dim_person_characteristic", False)  # characteristic_sk is string
    ]
    
    # Fix each dimension
    for dim_name, is_int_sk in dimensions:
        dim_path = output_dir / f"{dim_name}.parquet"
        if dim_path.exists():
            fix_dimension(dim_path, dim_name, is_int_sk)
        else:
            logger.warning(f"Dimension file not found: {dim_path}")
    
    logger.info("Dimension fix complete")
    
if __name__ == "__main__":
    main()