"""
Verifies surrogate key relationships between fact and dimension tables.
Uses Polars for all data operations for consistency with validation.py.
"""

import logging
from pathlib import Path
import json
import polars as pl

logger = logging.getLogger(__name__)

def verify_surrogate_keys():
    """
    Verifies referential integrity between fact and dimension tables using Polars.
    Checks that all foreign keys in fact tables exist in their dimension tables.
    Results are written to surrogate_key_verification.json.
    """
    output_dir = Path("output")
    result_file = Path("surrogate_key_verification.json")
    
    # Define all tables to verify
    tables = [
        {
            "type": "fact",
            "name": "fact_health_conditions_refined",
            "path": "fact_health_conditions_refined.parquet",
            "dimensions": [
                {"name": "geo_dimension", "fact_key": "geo_sk", "dim_key": "geo_sk"},
                {"name": "dim_time", "fact_key": "time_sk", "dim_key": "time_sk"},
                {"name": "dim_health_condition", "fact_key": "condition_sk", "dim_key": "condition_sk"},
                {"name": "dim_demographic", "fact_key": "demographic_sk", "dim_key": "demographic_sk"},
                {"name": "dim_person_characteristic", "fact_key": "characteristic_sk", "dim_key": "characteristic_sk"}
            ]
        },
        {
            "type": "fact",
            "name": "fact_health_conditions_by_characteristic_refined",
            "path": "fact_health_conditions_by_characteristic_refined.parquet",
            "dimensions": [
                {"name": "geo_dimension", "fact_key": "geo_sk", "dim_key": "geo_sk"},
                {"name": "dim_time", "fact_key": "time_sk", "dim_key": "time_sk"},
                {"name": "dim_health_condition", "fact_key": "condition_sk", "dim_key": "condition_sk"},
                {"name": "dim_person_characteristic", "fact_key": "characteristic_sk", "dim_key": "characteristic_sk"}
            ]
        }
    ]
    
    results = {
        "status": "incomplete",
        "missing_files": [],
        "verification": {}
    }

    try:
        # Check if output directory exists
        if not output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        # Load all required files
        loaded_tables = {}
        for table in tables:
            path = output_dir / table["path"]
            if not path.exists():
                results["missing_files"].append(table["path"])
                continue
                
            loaded_tables[table["name"]] = pl.read_parquet(path)
            
            # Also load all dimension tables
            for dim in table["dimensions"]:
                dim_path = output_dir / f"{dim['name']}.parquet"
                if dim["name"] not in loaded_tables and dim_path.exists():
                    loaded_tables[dim["name"]] = pl.read_parquet(dim_path)

        if results["missing_files"]:
            results["status"] = "failed"
            logger.error(f"Missing required files: {results['missing_files']}")
            return False

        # Perform verification for each fact table
        all_valid = True
        for table in tables:
            if table["name"] not in loaded_tables:
                continue
                
            fact_df = loaded_tables[table["name"]]
            for dim in table["dimensions"]:
                if dim["name"] not in loaded_tables:
                    continue
                    
                dim_df = loaded_tables[dim["name"]]
                invalid_keys = fact_df.filter(
                    ~pl.col(dim["fact_key"]).is_in(dim_df[dim["dim_key"]])
                ).select(dim["fact_key"]).unique()
                
                n_invalid = invalid_keys.height
                if n_invalid > 0:
                    all_valid = False
                    logger.error(
                        f"FAIL: {n_invalid} foreign key(s) in {table['name']}.{dim['fact_key']} "
                        f"do not exist in {dim['name']}.{dim['dim_key']}"
                    )
                else:
                    logger.info(
                        f"PASS: All foreign keys in {table['name']}.{dim['fact_key']} "
                        f"are valid against {dim['name']}.{dim['dim_key']}"
                    )
                    
                # Store results
                key = f"{table['name']}.{dim['fact_key']}"
                results["verification"][key] = {
                    "invalid_count": n_invalid,
                    "invalid_keys": invalid_keys.to_dicts() if n_invalid > 0 else []
                }

        results["status"] = "success" if all_valid else "failed"
        
        # Write results to file
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
            
        return all_valid
        
    except Exception as e:
        logger.error(f"Error verifying surrogate keys: {str(e)}")
        results["status"] = "error"
        results["error"] = str(e)
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        return False

if __name__ == "__main__":
    verify_surrogate_keys()