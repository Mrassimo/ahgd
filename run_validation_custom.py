#!/usr/bin/env python3
"""
Custom script to run validation step with patches for type-compatibility issues.
This avoids the 'String vs Int64' error in the validation module.
"""

import logging
import sys
from pathlib import Path
import polars as pl

# Import from the ahgd_etl package
sys.path.append('/Users/massimoraso/Code/AHGD')
from ahgd_etl.config.settings import get_config_manager
from ahgd_etl import utils

# Set up logging
config_manager = get_config_manager()
logger = utils.setup_logging(config_manager.get_path('LOG_DIR'))

class PatchedValidator:
    """
    Performs validation checks on tables with additional type compatibility handling.
    """
    
    def __init__(self, output_dir: Path):
        """Initialize the validator."""
        self.output_dir = output_dir
        self.logger = logger
    
    def validate_referential_integrity_fixed(self, fact_table: str, dim_table: str, 
                                            fact_key: str, dim_key: str) -> bool:
        """
        Validates referential integrity with type compatibility handling.
        
        Args:
            fact_table: Name of the fact table
            dim_table: Name of the dimension table
            fact_key: Name of the foreign key column in the fact table
            dim_key: Name of the primary key column in the dimension table
            
        Returns:
            True if valid, False if invalid
        """
        fact_path = self.output_dir / f"{fact_table}.parquet"
        dim_path = self.output_dir / f"{dim_table}.parquet"
        
        if not fact_path.exists() or not dim_path.exists():
            self.logger.error(f"File not found: {fact_path} or {dim_path}")
            return False
        
        try:
            # Load both tables
            fact_df = pl.read_parquet(fact_path)
            dim_df = pl.read_parquet(dim_path)
            
            # Check if columns exist
            if fact_key not in fact_df.columns:
                self.logger.error(f"Foreign key column {fact_key} not found in {fact_table}")
                return False
                
            if dim_key not in dim_df.columns:
                self.logger.error(f"Primary key column {dim_key} not found in {dim_table}")
                return False
            
            # Get all values from dimension table
            dim_values = dim_df[dim_key].to_list()
            
            # Check types and handle type mismatches
            fact_type = fact_df[fact_key].dtype
            dim_type = dim_df[dim_key].dtype
            
            if str(fact_type) != str(dim_type):
                self.logger.warning(f"Type mismatch in {fact_table}.{fact_key} ({fact_type}) vs {dim_table}.{dim_key} ({dim_type})")
                
                # Convert dimension values to match fact table's type if needed
                if str(fact_type).startswith('Int'):
                    # For integer facts, make sure the dimension keys are integers
                    dim_values_converted = []
                    for val in dim_values:
                        if isinstance(val, str):
                            try:
                                dim_values_converted.append(int(val))
                            except ValueError:
                                # For non-numeric strings, skip (they won't match anyway)
                                continue
                        else:
                            dim_values_converted.append(val)
                    dim_values = dim_values_converted
                
                elif str(fact_type).startswith('Utf'):
                    # For string facts, convert dimension keys to strings
                    dim_values = [str(val) for val in dim_values]
            
            # Now do the check with Python, not Polars operations
            # This avoids type incompatibility issues
            all_valid = True
            invalid_fks = []
            
            for idx, val in enumerate(fact_df[fact_key]):
                if val is None or val not in dim_values:
                    all_valid = False
                    invalid_fks.append(val)
            
            if not all_valid:
                self.logger.error(f"FAIL: Found {len(invalid_fks)} invalid foreign keys in {fact_table}.{fact_key}")
                
                # Log some sample invalid keys
                sample = invalid_fks[:5] if len(invalid_fks) > 5 else invalid_fks
                self.logger.error(f"Invalid key samples: {sample}")
                return False
            else:
                self.logger.info(f"PASS: All foreign keys in {fact_table}.{fact_key} are valid against {dim_table}.{dim_key}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error validating {fact_table} against {dim_table}: {e}")
            return False
    
    def run_validation(self):
        """Run the patched validation."""
        # Check each dimension to ensure unknown members exist
        dimensions = ['geo_dimension', 'dim_time', 'dim_health_condition', 'dim_demographic', 'dim_person_characteristic']
        
        for dim in dimensions:
            dim_path = self.output_dir / f"{dim}.parquet"
            if not dim_path.exists():
                self.logger.warning(f"Dimension file not found: {dim_path}")
                continue
                
            dim_df = pl.read_parquet(dim_path)
            
            # Check for unknown member
            if 'is_unknown' in dim_df.columns:
                unknown_rows = dim_df.filter(pl.col('is_unknown') == True)
                if len(unknown_rows) == 0:
                    self.logger.warning(f"No unknown member found in {dim}")
                else:
                    self.logger.info(f"Found unknown member in {dim}")
            else:
                self.logger.warning(f"No is_unknown column in {dim}")
        
        # Check foreign key relationships
        fk_relationships = [
            # fact table, dimension table, foreign key, primary key
            ('fact_health_conditions_refined', 'geo_dimension', 'geo_sk', 'geo_sk'),
            ('fact_health_conditions_refined', 'dim_time', 'time_sk', 'time_sk'),
            ('fact_health_conditions_refined', 'dim_health_condition', 'condition_sk', 'condition_sk'),
            ('fact_health_conditions_refined', 'dim_demographic', 'demographic_sk', 'demographic_sk'),
            ('fact_health_conditions_by_characteristic_refined', 'geo_dimension', 'geo_sk', 'geo_sk'),
            ('fact_health_conditions_by_characteristic_refined', 'dim_time', 'time_sk', 'time_sk'),
            ('fact_health_conditions_by_characteristic_refined', 'dim_health_condition', 'condition_sk', 'condition_sk'),
            ('fact_health_conditions_by_characteristic_refined', 'dim_person_characteristic', 'characteristic_sk', 'characteristic_sk'),
            ('fact_no_assistance', 'geo_dimension', 'geo_sk', 'geo_sk'),
            ('fact_no_assistance', 'dim_time', 'time_sk', 'time_sk'),
            ('fact_no_assistance', 'dim_demographic', 'demographic_sk', 'demographic_sk')
        ]
        
        successful = 0
        failures = 0
        
        # Check each foreign key relationship
        for fact, dim, fk, pk in fk_relationships:
            self.logger.info(f"Validating {fact}.{fk} -> {dim}.{pk}")
            if self.validate_referential_integrity_fixed(fact, dim, fk, pk):
                successful += 1
            else:
                failures += 1
        
        # Return overall success/failure
        total = successful + failures
        self.logger.info(f"Validation complete: {successful}/{total} relationships valid")
        
        return successful == total

def main():
    """Main function to run patched validation."""
    # Set up logging
    logger.info("=== Starting Custom Data Validation ===")
    
    # Create and run validator
    validator = PatchedValidator(config_manager.get_path('OUTPUT_DIR'))
    success = validator.run_validation()
    
    # Log results
    logger.info("=== Custom Data Validation Complete ===")
    logger.info(f"Validation result: {'SUCCESS' if success else 'FAILURE'}")
    
    # Return exit code
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())