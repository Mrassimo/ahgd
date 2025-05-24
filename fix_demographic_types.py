#!/usr/bin/env python3
"""
Fix demographic dimension and related fact tables to ensure compatible types.

This script:
1. Updates dim_demographic to use integer surrogate keys
2. Updates fact tables to use corresponding surrogate keys
"""

import logging
import hashlib
from pathlib import Path
import polars as pl

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_demographic_types")

def fix_demographic_dimension(output_dir: Path):
    """
    Fix demographic dimension to use integer surrogate keys.
    
    Args:
        output_dir: Path to the output directory
    """
    dim_path = output_dir / "dim_demographic.parquet"
    if not dim_path.exists():
        logger.error(f"Dimension file not found: {dim_path}")
        return False
        
    try:
        # Load dimension
        dim_df = pl.read_parquet(dim_path)
        logger.info(f"Loaded dim_demographic with {len(dim_df)} rows")
        
        # Check if unknown member exists
        unknown_row = dim_df.filter(pl.col("is_unknown") == True)
        if len(unknown_row) == 0:
            logger.warning("No unknown member found in dim_demographic")
            return False
            
        # Create a mapping from string surrogate keys to integer surrogate keys
        sk_mapping = {}
        for i, row in enumerate(dim_df.iter_rows(named=True)):
            old_sk = row["demographic_sk"]
            # Use an incremental integer for normal rows
            if not row["is_unknown"]:
                new_sk = i + 1  # Start at 1
            else:
                # Use -9999 for unknown member
                new_sk = -9999
                
            sk_mapping[old_sk] = new_sk
        
        # Update dimension with integer surrogate keys
        # First create a new string column with the integer keys
        dim_df = dim_df.with_columns(
            pl.Series(
                [sk_mapping[sk] for sk in dim_df["demographic_sk"]],
                dtype=pl.Int64
            ).alias("new_demographic_sk")
        )
        
        # Then replace the original column and drop the temporary column
        dim_df = dim_df.drop("demographic_sk").rename({"new_demographic_sk": "demographic_sk"})
        
        # Save updated dimension
        dim_df.write_parquet(dim_path)
        logger.info(f"Updated dim_demographic with integer surrogate keys")
        
        return sk_mapping
        
    except Exception as e:
        logger.error(f"Error fixing dim_demographic: {e}")
        return False

def fix_fact_table(fact_path: Path, sk_mapping: dict):
    """
    Fix fact table to use integer surrogate keys for demographic_sk.
    
    Args:
        fact_path: Path to the fact table
        sk_mapping: Mapping from old string keys to new integer keys
    """
    if not fact_path.exists():
        logger.warning(f"Fact table not found: {fact_path}")
        return False
        
    try:
        # Load fact table
        fact_df = pl.read_parquet(fact_path)
        fact_name = fact_path.stem
        logger.info(f"Loaded fact table {fact_name} with {len(fact_df)} rows")
        
        # Check if demographic_sk exists
        if "demographic_sk" not in fact_df.columns:
            logger.info(f"No demographic_sk column in {fact_name}")
            return True
            
        # We don't need to update if values are already integers and unknown member is -9999
        # which is what we're aiming for
        sample_value = fact_df["demographic_sk"][0]
        if isinstance(sample_value, int) and -9999 in fact_df["demographic_sk"].to_list():
            logger.info(f"{fact_name} already has correct demographic_sk types")
            return True
        
        # Update with the mapped values if needed
        # If demographic keys in the fact are strings, convert them
        if not str(fact_df["demographic_sk"].dtype).startswith("Int"):
            # Create mapping from fact keys to new integer keys
            # If key doesn't exist in mapping, use -9999 (unknown)
            new_values = []
            for sk in fact_df["demographic_sk"]:
                if sk in sk_mapping:
                    new_values.append(sk_mapping[sk])
                else:
                    new_values.append(-9999)
                    
            # Update fact table
            fact_df = fact_df.with_columns(
                pl.Series(new_values, dtype=pl.Int64).alias("new_demographic_sk")
            )
            
            # Replace original column
            fact_df = fact_df.drop("demographic_sk").rename({"new_demographic_sk": "demographic_sk"})
            
            # Save updated fact table
            fact_df.write_parquet(fact_path)
            logger.info(f"Updated {fact_name} with integer demographic_sk values")
        
        return True
        
    except Exception as e:
        logger.error(f"Error fixing {fact_path.name}: {e}")
        return False

def main():
    """Main function."""
    # Path to output directory
    output_dir = Path("./output")
    
    # First fix the demographic dimension
    logger.info("Fixing demographic dimension...")
    sk_mapping = fix_demographic_dimension(output_dir)
    
    if not sk_mapping:
        logger.error("Failed to fix demographic dimension")
        return 1
    
    # Fix fact tables
    logger.info("Fixing fact tables with demographic_sk...")
    
    # List of fact tables that might reference demographic_sk
    fact_tables = [
        "fact_population",
        "fact_income",
        "fact_assistance_needed",
        "fact_health_conditions",
        "fact_health_conditions_refined",
        "fact_health_conditions_by_characteristic_refined",
        "fact_no_assistance"
    ]
    
    for fact_name in fact_tables:
        fact_path = output_dir / f"{fact_name}.parquet"
        fix_fact_table(fact_path, sk_mapping)
    
    logger.info("Demographic key type fixes complete!")
    return 0

if __name__ == "__main__":
    main()