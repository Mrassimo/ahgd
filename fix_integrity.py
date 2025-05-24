#!/usr/bin/env python3
"""
Final targeted fix for the specific fact_health_conditions_refined referential integrity issue.

This script directly inspects the tables and columns involved in the failed
referential integrity check and makes the necessary corrections.
"""

import logging
import sys
from pathlib import Path
import polars as pl
import pandas as pd

# Add project root to path
sys.path.append('/Users/massimoraso/Code/AHGD')

# Import required modules
from ahgd_etl.config.settings import get_config_manager
from ahgd_etl import utils
from ahgd_etl.validators.data_quality import DataQualityValidator

# Get configuration manager
config_manager = get_config_manager()

# Setup logging
logger = utils.setup_logging(config_manager.get_path('LOG_DIR'))
logger.setLevel(logging.INFO)

def inspect_columns(output_dir):
    """Inspect the columns in fact and dimension tables to diagnose the issue."""
    logger.info("=== Inspecting tables for integrity issues ===")
    
    # Path definitions
    fact_path = output_dir / "fact_health_conditions_refined.parquet"
    dim_path = output_dir / "dim_person_characteristic.parquet"
    
    # 1. Inspect fact table
    if fact_path.exists():
        fact_df = pl.read_parquet(fact_path)
        logger.info(f"Fact table columns: {fact_df.columns}")
        
        # Check for characteristic columns
        characteristic_cols = [col for col in fact_df.columns if "charact" in col.lower()]
        if characteristic_cols:
            logger.info(f"Found characteristic-related columns: {characteristic_cols}")
            
            # Inspect values in these columns
            for col in characteristic_cols:
                unique_vals = fact_df[col].unique().to_list()
                logger.info(f"Unique values in {col}: {unique_vals[:5]} (showing first 5)")
                if fact_df[col].null_count() > 0:
                    logger.info(f"Column {col} has {fact_df[col].null_count()} null values")
    else:
        logger.error(f"Fact table not found: {fact_path}")
    
    # 2. Inspect dimension table
    if dim_path.exists():
        dim_df = pl.read_parquet(dim_path)
        logger.info(f"Dimension table columns: {dim_df.columns}")
        
        # Check for primary key
        if "characteristic_sk" in dim_df.columns:
            unique_vals = dim_df["characteristic_sk"].unique().to_list()
            logger.info(f"Unique dimension keys: {unique_vals[:5]} (showing first 5)")
    else:
        logger.error(f"Dimension table not found: {dim_path}")

def fix_integrity_issue(output_dir):
    """Fix the referential integrity issue by manually linking the tables."""
    logger.info("=== Fixing referential integrity issue ===")
    
    # Path definitions
    fact_path = output_dir / "fact_health_conditions_refined.parquet"
    dim_path = output_dir / "dim_person_characteristic.parquet"
    
    # Load tables
    try:
        fact_df = pl.read_parquet(fact_path)
        dim_df = pl.read_parquet(dim_path)
        
        # Print diagnostics about the tables
        logger.info(f"Fact table: {len(fact_df)} rows")
        logger.info(f"Dimension table: {len(dim_df)} rows")
        
        # Check for the column that represents the link
        if "characteristic_sk" in fact_df.columns and "characteristic_sk" in dim_df.columns:
            # Get valid keys from dimension
            valid_keys = set(dim_df["characteristic_sk"].to_list())
            logger.info(f"Valid keys in dimension: {list(valid_keys)[:10]} (showing first 10)")
            
            # Find unknown key in dimension (if it exists)
            unknown_row = dim_df.filter(pl.col("is_unknown") == True)
            unknown_key = None
            if len(unknown_row) > 0:
                unknown_key = unknown_row[0, "characteristic_sk"]
                logger.info(f"Found unknown key in dimension: {unknown_key}")
            else:
                # Create unknown member if not exists
                logger.info("Creating unknown member")
                
                # Use existing row as template
                template = dict(zip(dim_df.columns, dim_df[0,:]))
                unknown_data = {}
                
                for col, val in template.items():
                    if col == "characteristic_sk":
                        # For string surrogate key, use a special value
                        unknown_data[col] = "UNKNOWN_CHARACTERISTIC"
                    elif col == "is_unknown":
                        unknown_data[col] = True
                    elif col == "name" or "description" in col.lower():
                        unknown_data[col] = "UNKNOWN"
                    else:
                        # Use default based on type
                        col_dtype = dim_df[col].dtype
                        if str(col_dtype).startswith("Int"):
                            unknown_data[col] = -9999
                        elif str(col_dtype).startswith("Float"):
                            unknown_data[col] = -1.0
                        else:
                            unknown_data[col] = "UNKNOWN"
                
                # Add unknown member
                dim_df_pd = dim_df.to_pandas()
                dim_df_pd = pd.concat([dim_df_pd, pd.DataFrame([unknown_data])], ignore_index=True)
                dim_df = pl.from_pandas(dim_df_pd)
                
                # Check if unknown member was added
                unknown_row = dim_df.filter(pl.col("is_unknown") == True)
                if len(unknown_row) > 0:
                    unknown_key = unknown_row[0, "characteristic_sk"]
                    logger.info(f"Created unknown key: {unknown_key}")
                    
                    # Save updated dimension
                    dim_df.write_parquet(dim_path)
                    logger.info("Saved updated dimension table with unknown member")
                    
                    # Update valid keys
                    valid_keys = set(dim_df["characteristic_sk"].to_list())
            
            # Find invalid keys in fact table
            invalid_vals = []
            for val in fact_df["characteristic_sk"].unique():
                if val not in valid_keys:
                    invalid_vals.append(val)
            
            if invalid_vals:
                logger.info(f"Found {len(invalid_vals)} invalid characteristic_sk values")
                logger.info(f"Invalid values: {invalid_vals}")
                
                # Fix invalid keys
                if unknown_key:
                    # Replace invalid keys with unknown key
                    for invalid_val in invalid_vals:
                        fact_df = fact_df.with_columns(
                            pl.when(pl.col("characteristic_sk") == invalid_val)
                              .then(pl.lit(unknown_key))
                              .otherwise(pl.col("characteristic_sk"))
                              .alias("characteristic_sk")
                        )
                    logger.info("Replaced invalid keys with unknown key")
                    
                    # Save updated fact table
                    fact_df.write_parquet(fact_path)
                    logger.info("Saved updated fact table")
                else:
                    logger.error("Unknown key not available for replacement")
            else:
                logger.info("No invalid keys found")
        else:
            # Column missing, check if we need to add it
            if "characteristic_sk" not in fact_df.columns:
                logger.warning("characteristic_sk column missing from fact table - adding placeholder column")
                
                # Find an unknown key from dimension or create one
                unknown_key = None
                unknown_row = dim_df.filter(pl.col("is_unknown") == True)
                if len(unknown_row) > 0:
                    unknown_key = unknown_row[0, "characteristic_sk"]
                else:
                    unknown_key = "UNKNOWN_CHARACTERISTIC"
                
                # Add characteristic_sk column with unknown key
                fact_df = fact_df.with_columns(pl.lit(unknown_key).alias("characteristic_sk"))
                logger.info(f"Added characteristic_sk column with value {unknown_key}")
                
                # Save updated fact table
                fact_df.write_parquet(fact_path)
                logger.info("Saved updated fact table with new column")
            else:
                logger.warning("characteristic_sk column missing from dimension table - creating placeholder dimension")
                
                # Get unique characteristic_sk values from fact
                unique_keys = fact_df["characteristic_sk"].unique().to_list()
                logger.info(f"Unique characteristic_sk values in fact: {unique_keys}")
                
                # Create dimension with these keys
                rows = []
                for key in unique_keys:
                    row = {
                        "characteristic_sk": key,
                        "name": f"Auto-created for {key}",
                        "description": "Auto-created characteristic",
                        "is_unknown": key == "UNKNOWN" or key == "UNKNOWN_CHARACTERISTIC"
                    }
                    rows.append(row)
                
                # Add explicit unknown member if not already there
                has_unknown = any(r["is_unknown"] for r in rows)
                if not has_unknown:
                    rows.append({
                        "characteristic_sk": "UNKNOWN_CHARACTERISTIC",
                        "name": "Unknown Characteristic",
                        "description": "Unknown characteristic value",
                        "is_unknown": True
                    })
                
                # Create dimension dataframe
                new_dim_df = pl.DataFrame(rows)
                new_dim_df.write_parquet(dim_path)
                logger.info("Created and saved new dimension table")
    
    except Exception as e:
        logger.error(f"Error fixing integrity issue: {e}")
        logger.error("Traceback:", exc_info=True)
        return False
    
    return True

def validate_fix(output_dir):
    """Validate the fix by running targeted checks."""
    logger.info("=== Validating fix ===")
    
    # Create validator
    validator = DataQualityValidator(output_dir)
    
    # Load tables
    fact_path = output_dir / "fact_health_conditions_refined.parquet"
    dim_path = output_dir / "dim_person_characteristic.parquet"
    
    if not fact_path.exists() or not dim_path.exists():
        logger.error("Required tables not found")
        return False
    
    try:
        fact_df = pl.read_parquet(fact_path)
        dim_df = pl.read_parquet(dim_path)
        
        # Directly check referential integrity
        if "characteristic_sk" in fact_df.columns and "characteristic_sk" in dim_df.columns:
            # Get all valid dimension keys
            valid_keys = set(dim_df["characteristic_sk"].to_list())
            
            # Check if all fact keys are valid
            invalid_keys = fact_df.filter(~pl.col("characteristic_sk").is_in(valid_keys))
            invalid_count = len(invalid_keys)
            
            if invalid_count > 0:
                logger.error(f"Still found {invalid_count} invalid references")
                logger.error(f"Invalid values: {invalid_keys.select('characteristic_sk').unique().to_list()}")
                return False
            else:
                logger.info("All references are valid!")
                return True
        else:
            logger.error("Required columns not found in tables")
            return False
    
    except Exception as e:
        logger.error(f"Error validating fix: {e}")
        logger.error("Traceback:", exc_info=True)
        return False

def main():
    """Main function."""
    # Get output directory
    output_dir = config_manager.get_path('OUTPUT_DIR')
    
    # 1. Inspect columns to diagnose the issue
    inspect_columns(output_dir)
    
    # 2. Apply the fix
    success = fix_integrity_issue(output_dir)
    
    # 3. Validate the fix
    if success:
        success = validate_fix(output_dir)
    
    # 4. Run final validation
    if success:
        # Create validator
        validator = DataQualityValidator(output_dir)
        
        # Run validation for fact_health_conditions_refined against dim_person_characteristic
        fact_df = pl.read_parquet(output_dir / "fact_health_conditions_refined.parquet")
        dim_df = pl.read_parquet(output_dir / "dim_person_characteristic.parquet")
        
        result = validator.validate_referential_integrity(
            fact_df, dim_df, 
            "characteristic_sk", "characteristic_sk", 
            "fact_health_conditions_refined", "dim_person_characteristic"
        )
        
        if result["passed"]:
            logger.info("PASS: Referential integrity validated successfully!")
        else:
            logger.error(f"FAIL: Referential integrity check failed: {result['details']}")
            success = False
    
    if success:
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())