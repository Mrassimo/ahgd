#!/usr/bin/env python3
"""
Improved fix for fact table references with robust type handling.

This script identifies fact tables with invalid foreign keys and updates them
to reference the unknown members we created in the dimension tables.
"""

import logging
import polars as pl
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("improved_fix_fact_refs")

# Mapping of foreign key column names to dimension tables
FK_TO_DIM = {
    'geo_sk': 'geo_dimension',
    'time_sk': 'dim_time',
    'condition_sk': 'dim_health_condition',
    'demographic_sk': 'dim_demographic',
    'characteristic_sk': 'dim_person_characteristic'
}

def fix_fact_table(fact_path: Path, output_dir: Path) -> bool:
    """
    Fix foreign key references in a fact table.
    
    Args:
        fact_path: Path to the fact table file
        output_dir: Directory containing dimension files
        
    Returns:
        bool: True if changes were made, False otherwise
    """
    if not fact_path.exists():
        logger.warning(f"Fact table not found: {fact_path}")
        return False
    
    try:
        # Load fact table
        fact_df = pl.read_parquet(fact_path)
        fact_name = fact_path.stem
        logger.info(f"Loaded fact table {fact_name} with {len(fact_df)} rows")
        
        # Identify foreign key columns in fact table
        fk_columns = [col for col in fact_df.columns if col.endswith('_sk')]
        
        # Fix each foreign key
        changes_made = False
        for fk in fk_columns:
            # Determine the dimension table
            dim_prefix = fk.split('_sk')[0]
            dim_name = None
            
            # Look up the dimension name
            if fk in FK_TO_DIM:
                dim_name = FK_TO_DIM[fk]
            else:
                logger.warning(f"Unknown foreign key column: {fk}")
                continue
            
            # Load dimension table
            dim_path = output_dir / f"{dim_name}.parquet"
            if not dim_path.exists():
                logger.warning(f"Dimension table not found: {dim_path}")
                continue
                
            dim_df = pl.read_parquet(dim_path)
            logger.info(f"Loaded dimension {dim_name} with {len(dim_df)} rows")
            
            # Find primary key column in dimension table (usually same name as foreign key)
            pk_col = fk
            
            # Get all valid values from dimension table
            valid_keys = set(dim_df[pk_col].to_list())
            
            # Find unknown member surrogate key
            unknown_sk = None
            if "is_unknown" in dim_df.columns:
                unknown_rows = dim_df.filter(pl.col("is_unknown") == True)
                if len(unknown_rows) > 0:
                    unknown_sk = unknown_rows[0, pk_col]
            
            if not unknown_sk:
                logger.warning(f"Unknown member not found in {dim_name}, skipping {fk}")
                continue
            
            # Handle type compatibility issues
            fact_type = fact_df[fk].dtype
            dim_type = dim_df[pk_col].dtype
            
            try:
                # Check for type mismatches and handle accordingly
                if str(fact_type) != str(dim_type):
                    logger.warning(f"Type mismatch in {fact_name}.{fk} ({fact_type}) vs {dim_name}.{pk_col} ({dim_type})")
                
                # Convert list of valid keys to match fact table type if needed
                if str(fact_type).startswith('Int') and not all(isinstance(k, int) for k in valid_keys):
                    # For integer facts, ensure dim keys are integers
                    valid_keys_converted = set()
                    for key in valid_keys:
                        if isinstance(key, str):
                            try:
                                valid_keys_converted.add(int(key))
                            except ValueError:
                                # Skip non-numeric strings
                                pass
                        else:
                            valid_keys_converted.add(key)
                    
                    # Use converted keys
                    valid_keys = valid_keys_converted
                    
                elif str(fact_type).startswith('Utf') and not all(isinstance(k, str) for k in valid_keys):
                    # For string facts, ensure dim keys are strings
                    valid_keys = set(str(k) for k in valid_keys)
                
                # Convert unknown surrogate key to match fact table type
                if str(fact_type).startswith('Int') and not isinstance(unknown_sk, int):
                    if isinstance(unknown_sk, str):
                        try:
                            unknown_sk = int(unknown_sk)
                        except ValueError:
                            # Use special value if conversion fails
                            unknown_sk = -9999
                elif str(fact_type).startswith('Utf') and not isinstance(unknown_sk, str):
                    unknown_sk = str(unknown_sk)
                
                # Now identify invalid foreign keys
                if str(fact_type).startswith('Int'):
                    # For integer keys, use Python comparison
                    invalid_indices = [i for i, val in enumerate(fact_df[fk]) 
                                      if val not in valid_keys or val is None]
                    n_invalid = len(invalid_indices)
                    
                    if n_invalid > 0:
                        # Create a boolean mask for replacement
                        mask = pl.Series([i in invalid_indices for i in range(len(fact_df))])
                        
                        # Replace invalid keys
                        fact_df = fact_df.with_columns(
                            pl.when(mask)
                            .then(pl.lit(unknown_sk))
                            .otherwise(pl.col(fk))
                            .alias(fk)
                        )
                        changes_made = True
                        logger.info(f"Fixed {n_invalid} invalid references in {fact_name}.{fk}")
                else:
                    # For string keys, use normal filter
                    invalid_keys = fact_df.filter(
                        ~pl.col(fk).is_in(valid_keys) | pl.col(fk).is_null()
                    )
                    n_invalid = len(invalid_keys)
                    
                    if n_invalid > 0:
                        # Replace invalid keys
                        fact_df = fact_df.with_columns(
                            pl.when(~pl.col(fk).is_in(valid_keys) | pl.col(fk).is_null())
                            .then(pl.lit(unknown_sk))
                            .otherwise(pl.col(fk))
                            .alias(fk)
                        )
                        changes_made = True
                        logger.info(f"Fixed {n_invalid} invalid references in {fact_name}.{fk}")
            
            except Exception as e:
                logger.error(f"Error fixing references in {fact_name}.{fk}: {e}")
                # Continue with other foreign keys
        
        # Save updated fact table if changes were made
        if changes_made:
            fact_df.write_parquet(fact_path)
            logger.info(f"Saved updated fact table {fact_name}")
        
        return changes_made
    
    except Exception as e:
        logger.error(f"Error fixing fact table {fact_path.name}: {e}")
        return False

def fix_all_fact_tables(output_dir: Path):
    """
    Fix all fact tables in the output directory.
    
    Args:
        output_dir: Directory containing Parquet files
    """
    # List of fact tables to check
    fact_tables = [
        "fact_population.parquet",
        "fact_income.parquet",
        "fact_assistance_needed.parquet",
        "fact_health_conditions.parquet",
        "fact_health_conditions_refined.parquet",
        "fact_health_conditions_by_characteristic_refined.parquet",
        "fact_no_assistance.parquet"
    ]
    
    # Process each fact table
    fixed_count = 0
    for fact_file in fact_tables:
        fact_path = output_dir / fact_file
        if fact_path.exists():
            if fix_fact_table(fact_path, output_dir):
                fixed_count += 1
        else:
            logger.info(f"Fact table {fact_file} not found")
    
    logger.info(f"Fixed {fixed_count} fact tables")

def main():
    """Main function."""
    # Path to output directory
    output_dir = Path("./output")
    
    # Check if output directory exists
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return
    
    # Fix all fact tables
    logger.info("Starting fact table reference fix process")
    fix_all_fact_tables(output_dir)
    logger.info("Fact table fix complete")

if __name__ == "__main__":
    main()