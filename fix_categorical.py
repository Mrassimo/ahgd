#!/usr/bin/env python3
"""
Fix for the categorical type issues in dimension tables.

This script recreates dimension tables with unknown members in a way that preserves
the categorical data type integrity. It should be run once to fix dimension tables
with incompatible categorical types.
"""

import logging
from pathlib import Path
import polars as pl

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_categorical")

def fix_dimension_tables(output_dir: Path):
    """
    Fix dimension tables with categorical type issues.
    
    Args:
        output_dir: Directory containing dimension Parquet files
    """
    # List of dimension tables to fix
    dimensions = [
        "geo_dimension",
        "dim_time",
        "dim_health_condition",
        "dim_demographic",
        "dim_person_characteristic"
    ]
    
    for dim_name in dimensions:
        dim_path = output_dir / f"{dim_name}.parquet"
        
        if not dim_path.exists():
            logger.warning(f"Dimension {dim_name} not found at {dim_path}")
            continue
        
        logger.info(f"Processing dimension: {dim_name}")
        
        try:
            # Read the original dimension
            dim_df = pl.read_parquet(dim_path)
            logger.info(f"Loaded {dim_name} with {len(dim_df)} rows")
            
            # Check if unknown member already exists
            has_unknown = False
            if "is_unknown" in dim_df.columns:
                unknown_rows = dim_df.filter(pl.col("is_unknown") == True)
                has_unknown = len(unknown_rows) > 0
                
                if has_unknown:
                    logger.info(f"Dimension {dim_name} already has unknown member")
                    continue
            else:
                # Add is_unknown column if missing
                dim_df = dim_df.with_columns(pl.lit(False).alias("is_unknown"))
            
            # Get surrogate key column
            sk_cols = [col for col in dim_df.columns if col.endswith("_sk")]
            if not sk_cols:
                logger.warning(f"No surrogate key column found for {dim_name}")
                continue
                
            sk_col = sk_cols[0]
            
            # Get all rows as dictionaries
            rows = [row for row in dim_df.iter_rows(named=True)]
            
            # Create unknown member 
            unknown_row = {}
            for col in dim_df.columns:
                if col == sk_col:
                    # Use a special unknown surrogate key
                    import hashlib
                    key_str = f"UNKNOWN_{dim_name}"
                    unknown_row[col] = hashlib.md5(key_str.encode('utf-8')).hexdigest()
                elif col == "is_unknown":
                    unknown_row[col] = True
                else:
                    # Get a sample value to determine type
                    sample_value = dim_df[col][0]
                    if isinstance(sample_value, int):
                        unknown_row[col] = -1
                    elif isinstance(sample_value, float):
                        unknown_row[col] = -1.0
                    elif isinstance(sample_value, bool):
                        unknown_row[col] = False
                    else:
                        # Default for strings or categorical
                        unknown_row[col] = "UNKNOWN"
            
            # Add unknown row
            rows.append(unknown_row)
            
            # Create a new dataframe from all rows
            # This will automatically handle conversion of categorical types
            new_df = pl.DataFrame(rows)
            
            # Save the fixed dimension
            new_df.write_parquet(dim_path)
            logger.info(f"Saved fixed dimension {dim_name} with unknown member")
            
        except Exception as e:
            logger.error(f"Error fixing dimension {dim_name}: {e}", exc_info=True)

if __name__ == "__main__":
    # Fix dimensions in the output directory
    output_dir = Path("./output")
    
    # Make sure output directory exists
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        exit(1)
    
    fix_dimension_tables(output_dir)
    logger.info("Dimension fix completed")