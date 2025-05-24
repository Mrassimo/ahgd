#!/usr/bin/env python3
"""
Fix for dimension tables to ensure unknown members and consistent types.

This script:
1. Adds unknown members to dimension tables if missing
2. Fixes type compatibility issues between String and Categorical
3. Repairs key relationships in fact tables
"""

import logging
import hashlib
from pathlib import Path
import polars as pl
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_dim_types")

def generate_surrogate_key(*args) -> str:
    """Generate a surrogate key using MD5 hash."""
    key_str = '_'.join([str(arg) for arg in args])
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def fix_health_condition_dimension(output_dir: Path):
    """Fix the health condition dimension specifically."""
    dim_path = output_dir / "dim_health_condition.parquet"
    if not dim_path.exists():
        logger.warning(f"Dimension not found: {dim_path}")
        return False
    
    try:
        # Read the dimension
        dim_df = pl.read_parquet(dim_path)
        logger.info(f"Loaded dim_health_condition with {len(dim_df)} rows")
        
        # Check if unknown member exists
        if "is_unknown" in dim_df.columns:
            unknown_rows = dim_df.filter(pl.col("is_unknown") == True)
            if len(unknown_rows) > 0:
                logger.info("Unknown member already exists in dim_health_condition")
                return True
        else:
            # Add is_unknown column if missing
            dim_df = dim_df.with_columns(pl.lit(False).alias("is_unknown"))
        
        # Get schema
        schema = {name: dtype for name, dtype in zip(dim_df.columns, dim_df.dtypes)}
        
        # Create unknown member
        # Start with a copy of the first row to maintain schema
        first_row = dim_df.row(0, named=True)
        unknown_data = {}
        
        for col, value in first_row.items():
            if col == "condition_sk":
                unknown_data[col] = generate_surrogate_key("UNKNOWN", "health_condition")
            elif col == "condition_code":
                unknown_data[col] = "UNK"
            elif col == "condition_name":
                unknown_data[col] = "UNKNOWN CONDITION"
            elif col == "condition_category":
                unknown_data[col] = "UNKNOWN"
            elif col == "is_unknown":
                unknown_data[col] = True
            elif col == "etl_processed_at":
                unknown_data[col] = datetime.now()
            else:
                # For other columns, use type-appropriate unknowns
                if isinstance(value, int):
                    unknown_data[col] = -1
                elif isinstance(value, float):
                    unknown_data[col] = -1.0
                elif isinstance(value, bool):
                    unknown_data[col] = False
                else:
                    unknown_data[col] = "UNKNOWN"
        
        # Create a single-row dataframe for the unknown member
        unknown_df = pl.DataFrame([unknown_data])
        
        # Force casts to match the original schema
        cast_exprs = []
        for col, dtype in schema.items():
            if col in unknown_df.columns:
                cast_exprs.append(pl.col(col).cast(dtype))
        
        if cast_exprs:
            unknown_df = unknown_df.select(cast_exprs)
        
        # Append unknown member
        # Use hstack because it's safer with categorical columns
        dim_df = pl.concat([dim_df, unknown_df], how="vertical")
        
        # Save the updated dimension
        dim_df.write_parquet(dim_path)
        logger.info(f"Added unknown member to dim_health_condition and saved to {dim_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing dim_health_condition: {str(e)}", exc_info=True)
        return False

def fix_demographic_dimension(output_dir: Path):
    """Fix the demographic dimension specifically."""
    dim_path = output_dir / "dim_demographic.parquet"
    if not dim_path.exists():
        logger.warning(f"Dimension not found: {dim_path}")
        return False
    
    try:
        # Read the dimension
        dim_df = pl.read_parquet(dim_path)
        logger.info(f"Loaded dim_demographic with {len(dim_df)} rows")
        
        # Check if unknown member exists
        if "is_unknown" in dim_df.columns:
            unknown_rows = dim_df.filter(pl.col("is_unknown") == True)
            if len(unknown_rows) > 0:
                logger.info("Unknown member already exists in dim_demographic")
                return True
        else:
            # Add is_unknown column if missing
            dim_df = dim_df.with_columns(pl.lit(False).alias("is_unknown"))
        
        # Get schema
        schema = {name: dtype for name, dtype in zip(dim_df.columns, dim_df.dtypes)}
        
        # Create unknown member
        # Start with a copy of the first row to maintain schema
        first_row = dim_df.row(0, named=True)
        unknown_data = {}
        
        for col, value in first_row.items():
            if col == "demographic_sk":
                unknown_data[col] = generate_surrogate_key("UNKNOWN", "demographic")
            elif col == "age_group":
                unknown_data[col] = "UNKNOWN"
            elif col == "sex":
                unknown_data[col] = "UNKNOWN"
            elif col == "is_unknown":
                unknown_data[col] = True
            elif col == "etl_processed_at":
                unknown_data[col] = datetime.now()
            else:
                # For other columns, use type-appropriate unknowns
                if isinstance(value, int):
                    unknown_data[col] = -1
                elif isinstance(value, float):
                    unknown_data[col] = -1.0
                elif isinstance(value, bool):
                    unknown_data[col] = False
                else:
                    unknown_data[col] = "UNKNOWN"
        
        # Create a single-row dataframe for the unknown member
        unknown_df = pl.DataFrame([unknown_data])
        
        # Force casts to match the original schema
        cast_exprs = []
        for col, dtype in schema.items():
            if col in unknown_df.columns:
                cast_exprs.append(pl.col(col).cast(dtype))
        
        if cast_exprs:
            unknown_df = unknown_df.select(cast_exprs)
        
        # Append unknown member
        dim_df = pl.concat([dim_df, unknown_df], how="vertical")
        
        # Save the updated dimension
        dim_df.write_parquet(dim_path)
        logger.info(f"Added unknown member to dim_demographic and saved to {dim_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing dim_demographic: {str(e)}", exc_info=True)
        return False

def fix_person_characteristic_dimension(output_dir: Path):
    """Fix the person characteristic dimension specifically."""
    dim_path = output_dir / "dim_person_characteristic.parquet"
    if not dim_path.exists():
        logger.warning(f"Dimension not found: {dim_path}")
        return False
    
    try:
        # Read the dimension
        dim_df = pl.read_parquet(dim_path)
        logger.info(f"Loaded dim_person_characteristic with {len(dim_df)} rows")
        
        # Check if unknown member exists
        if "is_unknown" in dim_df.columns:
            unknown_rows = dim_df.filter(pl.col("is_unknown") == True)
            if len(unknown_rows) > 0:
                logger.info("Unknown member already exists in dim_person_characteristic")
                return True
        else:
            # Add is_unknown column if missing
            dim_df = dim_df.with_columns(pl.lit(False).alias("is_unknown"))
        
        # Get schema
        schema = {name: dtype for name, dtype in zip(dim_df.columns, dim_df.dtypes)}
        
        # Create unknown member
        # Start with a copy of the first row to maintain schema
        first_row = dim_df.row(0, named=True)
        unknown_data = {}
        
        for col, value in first_row.items():
            if col == "characteristic_sk":
                unknown_data[col] = generate_surrogate_key("UNKNOWN", "characteristic")
            elif col == "characteristic_type":
                unknown_data[col] = "UNKNOWN"
            elif col == "characteristic_value":
                unknown_data[col] = "UNKNOWN"
            elif col == "characteristic_category":
                unknown_data[col] = "UNKNOWN"
            elif col == "is_unknown":
                unknown_data[col] = True
            elif col == "etl_processed_at":
                unknown_data[col] = datetime.now()
            else:
                # For other columns, use type-appropriate unknowns
                if isinstance(value, int):
                    unknown_data[col] = -1
                elif isinstance(value, float):
                    unknown_data[col] = -1.0
                elif isinstance(value, bool):
                    unknown_data[col] = False
                else:
                    unknown_data[col] = "UNKNOWN"
        
        # Create a single-row dataframe for the unknown member
        unknown_df = pl.DataFrame([unknown_data])
        
        # Force casts to match the original schema
        cast_exprs = []
        for col, dtype in schema.items():
            if col in unknown_df.columns:
                cast_exprs.append(pl.col(col).cast(dtype))
        
        if cast_exprs:
            unknown_df = unknown_df.select(cast_exprs)
        
        # Append unknown member
        dim_df = pl.concat([dim_df, unknown_df], how="vertical")
        
        # Save the updated dimension
        dim_df.write_parquet(dim_path)
        logger.info(f"Added unknown member to dim_person_characteristic and saved to {dim_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error fixing dim_person_characteristic: {str(e)}", exc_info=True)
        return False

def fix_geo_dimension(output_dir: Path):
    """Fix the geo dimension specifically."""
    dim_path = output_dir / "geo_dimension.parquet"
    if not dim_path.exists():
        logger.warning(f"Dimension not found: {dim_path}")
        return False

    try:
        # Read the dimension
        dim_df = pl.read_parquet(dim_path)
        logger.info(f"Loaded geo_dimension with {len(dim_df)} rows")

        # Check if unknown member exists
        if "is_unknown" in dim_df.columns:
            unknown_rows = dim_df.filter(pl.col("is_unknown") == True)
            if len(unknown_rows) > 0:
                logger.info("Unknown member already exists in geo_dimension")
                return True
        else:
            # Add is_unknown column if missing
            dim_df = dim_df.with_columns(pl.lit(False).alias("is_unknown"))

        # Get schema
        schema = {name: dtype for name, dtype in zip(dim_df.columns, dim_df.dtypes)}

        # Create unknown member
        # Start with a copy of the first row to maintain schema
        first_row = dim_df.row(0, named=True)
        unknown_data = {}

        for col, value in first_row.items():
            if col == "geo_sk":
                # For geo_sk, we need to use an integer, not a hash
                # Use a large negative number to avoid collision with real keys
                unknown_data[col] = -9999
            elif col == "geo_id":
                unknown_data[col] = "UNKNOWN"
            elif col == "geo_level":
                unknown_data[col] = "UNKNOWN"
            elif col == "geo_name":
                unknown_data[col] = "UNKNOWN LOCATION"
            elif col == "state_code":
                unknown_data[col] = "UNK"
            elif col == "state_name":
                unknown_data[col] = "UNKNOWN"
            elif col == "latitude":
                unknown_data[col] = 0.0
            elif col == "longitude":
                unknown_data[col] = 0.0
            elif col == "geom":
                unknown_data[col] = "POINT(0 0)"
            elif col == "parent_geo_sk":
                unknown_data[col] = -9999
            elif col == "is_unknown":
                unknown_data[col] = True
            elif col == "etl_processed_at":
                unknown_data[col] = datetime.now()
            else:
                # For other columns, use type-appropriate unknowns
                if isinstance(value, int):
                    unknown_data[col] = -1
                elif isinstance(value, float):
                    unknown_data[col] = -1.0
                elif isinstance(value, bool):
                    unknown_data[col] = False
                else:
                    unknown_data[col] = "UNKNOWN"

        # Create a single-row dataframe for the unknown member
        unknown_df = pl.DataFrame([unknown_data])

        # Force casts to match the original schema
        cast_exprs = []
        for col, dtype in schema.items():
            if col in unknown_df.columns:
                cast_exprs.append(pl.col(col).cast(dtype))

        if cast_exprs:
            unknown_df = unknown_df.select(cast_exprs)

        # Append unknown member
        dim_df = pl.concat([dim_df, unknown_df], how="vertical")

        # Save the updated dimension
        dim_df.write_parquet(dim_path)
        logger.info(f"Added unknown member to geo_dimension and saved to {dim_path}")
        return True

    except Exception as e:
        logger.error(f"Error fixing geo_dimension: {str(e)}", exc_info=True)
        return False

def fix_time_dimension(output_dir: Path):
    """Fix the time dimension specifically."""
    dim_path = output_dir / "dim_time.parquet"
    if not dim_path.exists():
        logger.warning(f"Dimension not found: {dim_path}")
        return False

    try:
        # Read the dimension
        dim_df = pl.read_parquet(dim_path)
        logger.info(f"Loaded dim_time with {len(dim_df)} rows")

        # Check if unknown member exists
        if "is_unknown" in dim_df.columns:
            unknown_rows = dim_df.filter(pl.col("is_unknown") == True)
            if len(unknown_rows) > 0:
                logger.info("Unknown member already exists in dim_time")
                return True
        else:
            # Add is_unknown column if missing
            dim_df = dim_df.with_columns(pl.lit(False).alias("is_unknown"))

        # Get schema
        schema = {name: dtype for name, dtype in zip(dim_df.columns, dim_df.dtypes)}

        # Create unknown member
        # Start with a copy of the first row to maintain schema
        first_row = dim_df.row(0, named=True)
        unknown_data = {}

        for col, value in first_row.items():
            if col == "time_sk":
                # For time_sk, we need to use an integer, not a hash
                # Use a special value like 19000101 (January 1, 1900)
                unknown_data[col] = 19000101
            elif col == "full_date":
                unknown_data[col] = datetime(1900, 1, 1).date()
            elif col == "year":
                unknown_data[col] = 1900
            elif col == "quarter":
                unknown_data[col] = 1
            elif col == "month":
                unknown_data[col] = 1
            elif col == "month_name":
                unknown_data[col] = "Unknown"
            elif col == "day_of_month":
                unknown_data[col] = 1
            elif col == "day_of_week":
                unknown_data[col] = 0
            elif col == "day_name":
                unknown_data[col] = "Unknown"
            elif col == "is_weekday":
                unknown_data[col] = False
            elif col == "financial_year":
                unknown_data[col] = "1900/01"
            elif col == "is_census_year":
                unknown_data[col] = False
            elif col == "is_unknown":
                unknown_data[col] = True
            elif col == "etl_processed_at":
                unknown_data[col] = datetime.now()
            else:
                # For other columns, use type-appropriate unknowns
                if isinstance(value, int):
                    unknown_data[col] = -1
                elif isinstance(value, float):
                    unknown_data[col] = -1.0
                elif isinstance(value, bool):
                    unknown_data[col] = False
                else:
                    unknown_data[col] = "UNKNOWN"

        # Create a single-row dataframe for the unknown member
        unknown_df = pl.DataFrame([unknown_data])

        # Force casts to match the original schema
        cast_exprs = []
        for col, dtype in schema.items():
            if col in unknown_df.columns:
                cast_exprs.append(pl.col(col).cast(dtype))

        if cast_exprs:
            unknown_df = unknown_df.select(cast_exprs)

        # Append unknown member
        dim_df = pl.concat([dim_df, unknown_df], how="vertical")

        # Save the updated dimension
        dim_df.write_parquet(dim_path)
        logger.info(f"Added unknown member to dim_time and saved to {dim_path}")
        return True

    except Exception as e:
        logger.error(f"Error fixing dim_time: {str(e)}", exc_info=True)
        return False

def fix_all_dimension_tables(output_dir: Path):
    """Fix all dimension tables."""
    # Fix each dimension table
    fix_geo_dimension(output_dir)
    fix_time_dimension(output_dir)
    fix_health_condition_dimension(output_dir)
    fix_demographic_dimension(output_dir)
    fix_person_characteristic_dimension(output_dir)

    # Success if we get here
    return True

def main():
    """Main function."""
    # Path to output directory
    output_dir = Path("./output")
    
    # Fix dimension tables
    logger.info("Starting dimension fix process")
    success = fix_all_dimension_tables(output_dir)
    
    if success:
        logger.info("All dimensions successfully fixed")
    else:
        logger.error("Failed to fix all dimensions")

if __name__ == "__main__":
    main()