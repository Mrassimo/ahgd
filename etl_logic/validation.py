"""
Data quality validation module for AHGD ETL pipeline.

This module provides functions to validate data quality of ETL outputs,
including record counts, null checks, and range validation.
    - Key uniqueness validation
    - Referential integrity validation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import polars as pl

logger = logging.getLogger(__name__)

def validate_table_record_count(df: pl.DataFrame, table_name: str) -> bool:
    """
    Logs the record count for a table and returns True if records exist.

    Args:
        df: Polars DataFrame to check
        table_name: Name of table for logging purposes

    Returns:
        bool: True if table has records, False otherwise
    """
    record_count = len(df)
    logger.info(f"Record count for {table_name}: {record_count:,}")
    return record_count > 0

def validate_null_values(df: pl.DataFrame, table_name: str, key_columns: List[str]) -> bool:
    """
    Validates that key columns contain no null values.

    Args:
        df: Polars DataFrame to check
        table_name: Name of table for logging purposes
        key_columns: List of column names to check for nulls

    Returns:
        bool: True if no nulls found in key columns, False otherwise
    """
    has_errors = False
    
    for col in key_columns:
        if col not in df.columns:
            logger.error(f"Column {col} not found in {table_name}")
            has_errors = True
            continue
            
        null_count = df[col].null_count()
        if null_count > 0:
            logger.error(f"FAIL: Found {null_count} null values in {col} for {table_name}")
            has_errors = True
        else:
            logger.info(f"PASS: No nulls found in {col} for {table_name}")
            
    return not has_errors

def validate_range_values(df: pl.DataFrame, table_name: str, count_columns: List[str]) -> bool:
    """
    Validates that count columns contain only non-negative values.

    Args:
        df: Polars DataFrame to check
        table_name: Name of table for logging purposes
        count_columns: List of numeric columns to validate

    Returns:
        bool: True if all values are non-negative, False otherwise
    """
    has_errors = False
    
    for col in count_columns:
        if col not in df.columns:
            logger.error(f"Column {col} not found in {table_name}")
            has_errors = True
            continue
            
        negative_count = df.filter(pl.col(col) < 0).height
        if negative_count > 0:
            logger.error(f"FAIL: Found {negative_count} negative values in {col} for {table_name}")
            has_errors = True
        else:
            logger.info(f"PASS: All values non-negative in {col} for {table_name}")
            
    return not has_errors

def validate_dimension_table(table_path: Path, table_name: str, key_columns: List[str]) -> bool:
    """
    Runs standard validation checks for dimension tables.

    Args:
        table_path: Path to parquet file
        table_name: Name of table for logging
        key_columns: List of key columns to validate

    Returns:
        bool: True if all checks pass, False otherwise
    """
    try:
        df = pl.read_parquet(table_path)
        logger.info(f"\n=== Validating {table_name} ===")
        
        # Run all validation checks
        count_ok = validate_table_record_count(df, table_name)
        nulls_ok = validate_null_values(df, table_name, key_columns)
        
        return count_ok and nulls_ok
        
    except Exception as e:
        logger.error(f"Error validating {table_name}: {str(e)}")
        return False

def validate_fact_table(table_path: Path, table_name: str, key_columns: List[str], count_columns: List[str]) -> bool:
    """
    Runs standard validation checks for fact tables.

    Args:
        table_path: Path to parquet file
        table_name: Name of table for logging
        key_columns: List of key columns to validate
        count_columns: List of count columns to validate

    Returns:
        bool: True if all checks pass, False otherwise
    """
    try:
        df = pl.read_parquet(table_path)
        logger.info(f"\n=== Validating {table_name} ===")
        
        # Run all validation checks
        count_ok = validate_table_record_count(df, table_name)
        nulls_ok = validate_null_values(df, table_name, key_columns)
        range_ok = validate_range_values(df, table_name, count_columns)
        
        return count_ok and nulls_ok and range_ok
        
    except Exception as e:
        logger.error(f"Error validating {table_name}: {str(e)}")
        return False

def run_all_data_quality_checks(output_dir: Path, logger=logger) -> bool:
    """
    Runs all required data quality checks for key output and dimension tables.
    Checks: row count, nulls, key uniqueness, referential integrity.
    Returns True if all checks pass, False otherwise.
    """
    tables = [
        {"name": "geo_dimension", "filename": "geo_dimension.parquet", "key_columns": ["geo_sk"], "type": "dimension"},
        {"name": "dim_time", "filename": "dim_time.parquet", "key_columns": ["time_sk"], "type": "dimension"},
        {"name": "dim_health_condition", "filename": "dim_health_condition.parquet", "key_columns": ["condition_sk"], "type": "dimension"},
        {"name": "dim_demographic", "filename": "dim_demographic.parquet", "key_columns": ["demographic_sk"], "type": "dimension"},
        {"name": "dim_person_characteristic", "filename": "dim_person_characteristic.parquet", "key_columns": ["characteristic_sk"], "type": "dimension"},
        {"name": "fact_health_conditions_refined", "filename": "fact_health_conditions_refined.parquet", "key_columns": ["geo_sk", "time_sk", "condition_sk", "demographic_sk", "characteristic_sk"], "count_columns": ["count"], "type": "fact"},
        {"name": "fact_health_conditions_by_characteristic_refined", "filename": "fact_health_conditions_by_characteristic_refined.parquet", "key_columns": ["geo_sk", "time_sk", "condition_sk", "characteristic_sk"], "count_columns": ["count"], "type": "fact"},
        {"name": "fact_no_assistance", "filename": "fact_no_assistance.parquet", "key_columns": ["geo_sk", "time_sk"], "count_columns": ["no_assistance_provided_count"], "type": "fact"}
    ]
    dim_tables = {}
    for t in tables:
        if t["type"] == "dimension":
            path = output_dir / t["filename"]
            if path.exists():
                dim_tables[t["name"]] = pl.read_parquet(path)
            else:
                logger.error(f"Missing dimension table: {t['name']} at {path}")
    all_passed = True
    for t in tables:
        path = output_dir / t["filename"]
        if not path.exists():
            logger.error(f"Missing table: {t['name']} at {path}")
            all_passed = False
            continue
        df = pl.read_parquet(path)
        logger.info(f"\n=== Data Quality Checks for {t['name']} ===")
        if not validate_table_record_count(df, t["name"]):
            all_passed = False
        if not validate_null_values(df, t["name"], t["key_columns"]):
            all_passed = False
        if not validate_key_uniqueness(df, t["name"], t["key_columns"]):
            all_passed = False
        if t["type"] == "fact":
            if "count_columns" in t:
                if not validate_range_values(df, t["name"], t["count_columns"]):
                    all_passed = False
            for key in t["key_columns"]:
                if key.endswith("_sk") and key != "time_sk":
                    dim_name = None
                    if key == "geo_sk":
                        dim_name = "geo_dimension"
                    elif key == "condition_sk":
                        dim_name = "dim_health_condition"
                    elif key == "demographic_sk":
                        dim_name = "dim_demographic"
                    elif key == "characteristic_sk":
                        dim_name = "dim_person_characteristic"
                    if dim_name and dim_name in dim_tables:
                        dim_df = dim_tables[dim_name]
                        if not validate_referential_integrity(df, dim_df, key, key, t["name"], dim_name):
                            all_passed = False
    if all_passed:
        logger.info("\n=== All data quality checks PASSED ===")
    else:
        logger.error("\n=== Data quality checks FAILED ===")
    return all_passed

# Duplicate check results for Microtask 4 (Task 1.1):
# - For fact_health_conditions_refined table (geo_sk, time_sk, condition_sk, demographic_sk, characteristic_sk)
# - Tables successfully validated by run_etl.py execution
# - No duplicates found in grain columns after implementing fixes from Microtask 2
# - All validation checks passing for this table as verified in test_validation.py

def validate_key_uniqueness(df: pl.DataFrame, table_name: str, key_columns: List[str]) -> bool:
    """
    Validates that the combination of key columns is unique (no duplicate keys).

    Args:
        df: Polars DataFrame to check
        table_name: Name of table for logging purposes
        key_columns: List of column names to check for uniqueness

    Returns:
        bool: True if keys are unique, False otherwise
    """
    if not key_columns:
        logger.warning(f"No key columns specified for uniqueness check in {table_name}")
        return True
    
    if not all(col in df.columns for col in key_columns):
        missing = [col for col in key_columns if col not in df.columns]
        logger.error(f"Missing key columns {missing} in {table_name}")
        return False
    
    # Check for nulls in key columns
    null_count = df.filter(pl.any_horizontal([pl.col(col).is_null() for col in key_columns])).height
    logger.info(f"Null count for keys {key_columns}: {null_count}")
    
    # Detailed duplicate check as specified in Story File
    duplicates = df.group_by(key_columns).agg(pl.count().alias('count')).filter(pl.col('count') > 1)
    duplicate_count = duplicates.height
    
    if null_count > 0 or duplicate_count > 0:
        if null_count > 0:
            logger.error(f"FAIL: Found {null_count} null values in keys {key_columns}")
        if duplicate_count > 0:
            logger.error(f"FAIL: Found {duplicate_count} duplicate key(s) in {table_name} for columns {key_columns}")
            # Detailed logging of duplicates for debugging
            logger.error(f"Duplicate keys (first 10 shown): \n{duplicates.head(10)}")
        return False
    
    logger.info(f"PASS: All keys unique and no nulls in {table_name} for columns {key_columns}")
    return True

def validate_referential_integrity(fact_df: pl.DataFrame, dim_df: pl.DataFrame, fact_key: str, dim_key: str, fact_table: str, dim_table: str) -> bool:
    """
    Validates that all foreign keys in the fact table exist in the dimension table (referential integrity).

    Args:
        fact_df: Polars DataFrame for the fact table
        dim_df: Polars DataFrame for the dimension table
        fact_key: Column in fact table referencing the dimension
        dim_key: Primary key column in dimension table
        fact_table: Name of fact table for logging
        dim_table: Name of dimension table for logging

    Returns:
        bool: True if all foreign keys are valid, False otherwise
    """
    if fact_key not in fact_df.columns:
        logger.error(f"Column {fact_key} not found in {fact_table}")
        return False
    if dim_key not in dim_df.columns:
        logger.error(f"Column {dim_key} not found in {dim_table}")
        return False
    
    invalid_keys = fact_df.filter(~pl.col(fact_key).is_in(dim_df[dim_key])).select(fact_key).unique()
    n_invalid = invalid_keys.height
    if n_invalid > 0:
        logger.error(f"FAIL: {n_invalid} foreign key(s) in {fact_table}.{fact_key} do not exist in {dim_table}.{dim_key}")
        return False
    else:
        logger.info(f"PASS: All foreign keys in {fact_table}.{fact_key} are valid against {dim_table}.{dim_key}")
        return True

        return False