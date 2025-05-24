"""
Schema validation and enforcement module for AHGD ETL.

This module handles schema-related fixes for the AHGD data warehouse:
1. Validates table schemas against the schema definitions
2. Adds missing columns with default values
3. Coerces data types to match schema
4. Validates referential integrity between tables

Usage:
    from ahgd_etl.core.temp_fix.schema_fix import SchemaValidator
    validator = SchemaValidator(output_dir=Path('output'))
    validator.validate_and_fix_all_tables()
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union

import polars as pl

# Set up logger
logger = logging.getLogger(__name__)

class SchemaValidator:
    """
    Schema validation and enforcement handler for AHGD ETL.
    Ensures all tables conform to their defined schemas and fixes issues.
    """
    
    def __init__(self, output_dir: Path, config_dir: Optional[Path] = None):
        """
        Initialize the SchemaValidator.
        
        Args:
            output_dir: Directory containing data files to validate/fix
            config_dir: Directory containing YAML configuration files
        """
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / "config" / "yaml"
        self.schemas = {}
        self.tables = {}
        self._load_schemas()
    
    def _load_schemas(self):
        """Load schemas from YAML config file."""
        try:
            schema_path = self.config_dir / "schemas.yaml"
            with open(schema_path, 'r') as f:
                self.schemas = yaml.safe_load(f)
            logger.info(f"Loaded schemas from {schema_path}")
        except Exception as e:
            logger.error(f"Error loading schemas: {e}")
            raise
    
    def _get_polars_type(self, type_str: str) -> Any:
        """
        Convert schema type string to Polars data type.
        
        Args:
            type_str: Type string from schema definition
            
        Returns:
            Polars data type
        """
        type_map = {
            'Int64': pl.Int64,
            'Int32': pl.Int32,
            'Float64': pl.Float64,
            'Float32': pl.Float32,
            'Utf8': pl.Utf8,
            'Boolean': pl.Boolean,
            'Date': pl.Date,
            'Datetime': pl.Datetime
        }
        
        return type_map.get(type_str, pl.Utf8)
    
    def _get_default_value(self, type_str: str) -> Any:
        """
        Get default value for a given data type.
        
        Args:
            type_str: Type string from schema definition
            
        Returns:
            Default value for the type
        """
        default_values = {
            'Int64': 0,
            'Int32': 0,
            'Float64': 0.0,
            'Float32': 0.0,
            'Utf8': '',
            'Boolean': False,
            'Date': None,  # This will need special handling
            'Datetime': None  # This will need special handling
        }
        
        return default_values.get(type_str, None)
    
    def load_tables(self):
        """Load all dimension and fact tables from output directory."""
        # Load dimension tables
        dimension_schemas = self.schemas.get('dimensions', {})
        for dim_name in dimension_schemas.keys():
            dim_path = self.output_dir / f"{dim_name}.parquet"
            if dim_path.exists():
                try:
                    self.tables[dim_name] = pl.read_parquet(dim_path)
                    logger.info(f"Loaded dimension {dim_name} with {len(self.tables[dim_name])} rows")
                except Exception as e:
                    logger.error(f"Error loading dimension {dim_name}: {e}")
        
        # Load fact tables
        fact_schemas = self.schemas.get('facts', {})
        for fact_name in fact_schemas.keys():
            fact_path = self.output_dir / f"{fact_name}.parquet"
            if fact_path.exists():
                try:
                    self.tables[fact_name] = pl.read_parquet(fact_path)
                    logger.info(f"Loaded fact table {fact_name} with {len(self.tables[fact_name])} rows")
                except Exception as e:
                    logger.error(f"Error loading fact table {fact_name}: {e}")
        
        logger.info(f"Loaded {len(self.tables)} tables")
    
    def enforce_schema(self, table_name: str) -> bool:
        """
        Enforce schema on a specific table. Adds missing columns and coerces data types.
        
        Args:
            table_name: Name of the table to validate and fix
            
        Returns:
            bool: True if changes were made, False otherwise
        """
        if table_name not in self.tables:
            logger.warning(f"Table {table_name} not loaded")
            return False
        
        # Determine if this is a dimension or fact table
        schema_section = 'dimensions' if table_name.startswith('dim_') or table_name == 'geo_dimension' else 'facts'
        schema = self.schemas.get(schema_section, {}).get(table_name, {})
        
        if not schema:
            logger.warning(f"No schema found for {table_name}")
            return False
        
        df = self.tables[table_name]
        changes_made = False
        
        # Check for missing columns
        missing_columns = [col for col in schema.keys() if col not in df.columns]
        if missing_columns:
            logger.info(f"Adding {len(missing_columns)} missing columns to {table_name}")
            
            for col in missing_columns:
                type_str = schema[col]
                default_value = self._get_default_value(type_str)
                
                # Handle date/datetime types specially
                if type_str == 'Date' or type_str == 'Datetime':
                    # For now, leave these as null
                    df = df.with_columns(pl.lit(None).cast(self._get_polars_type(type_str)).alias(col))
                else:
                    df = df.with_columns(pl.lit(default_value).cast(self._get_polars_type(type_str)).alias(col))
            
            changes_made = True
        
        # Check for type mismatches and perform conversion
        for col, type_str in schema.items():
            if col in df.columns:
                target_type = self._get_polars_type(type_str)
                current_type = df.schema[col]
                
                # Only convert if types don't match
                if str(current_type) != str(target_type):
                    logger.info(f"Converting column {col} from {current_type} to {target_type} in {table_name}")
                    
                    try:
                        # Try to convert the column, handling null values
                        df = df.with_columns(pl.col(col).cast(target_type))
                        changes_made = True
                    except Exception as e:
                        logger.error(f"Error converting {col} to {target_type}: {e}")
                        
                        # Fall back to setting null values for the problematic column
                        if type_str == 'Int64' or type_str == 'Int32':
                            df = df.with_columns(pl.lit(0).cast(target_type).alias(col))
                        elif type_str == 'Float64' or type_str == 'Float32':
                            df = df.with_columns(pl.lit(0.0).cast(target_type).alias(col))
                        elif type_str == 'Boolean':
                            df = df.with_columns(pl.lit(False).cast(target_type).alias(col))
                        else:
                            df = df.with_columns(pl.lit('').cast(target_type).alias(col))
                        
                        changes_made = True
        
        # Update table in memory and on disk if changes were made
        if changes_made:
            self.tables[table_name] = df
            table_path = self.output_dir / f"{table_name}.parquet"
            df.write_parquet(table_path)
            logger.info(f"Saved schema-fixed {table_name} to {table_path}")
        
        return changes_made
    
    def validate_referential_integrity(self, fact_table_name: str) -> Dict[str, int]:
        """
        Validate referential integrity for a fact table.
        
        Args:
            fact_table_name: Name of the fact table to validate
            
        Returns:
            Dict mapping foreign key columns to count of invalid keys
        """
        if fact_table_name not in self.tables:
            logger.warning(f"Fact table {fact_table_name} not loaded")
            return {}
        
        fact_df = self.tables[fact_table_name]
        
        # Identify dimension foreign keys in fact table
        fk_columns = [col for col in fact_df.columns if col.endswith('_sk')]
        
        # Create mapping of foreign key columns to their dimension tables
        fk_to_dim = {}
        for fk in fk_columns:
            # Extract dimension name from foreign key column (e.g., geo_sk -> geo_dimension)
            dim_prefix = fk.split('_sk')[0]
            
            # Handle special cases
            if dim_prefix == 'geo':
                fk_to_dim[fk] = 'geo_dimension'
            elif dim_prefix == 'time':
                fk_to_dim[fk] = 'dim_time'
            # Handle other dimension keys (condition, demographic, characteristic)
            else:
                potential_dims = [d for d in self.tables.keys() if dim_prefix in d]
                if potential_dims:
                    fk_to_dim[fk] = potential_dims[0]
        
        # Validate each foreign key
        invalid_counts = {}
        for fk, dim_name in fk_to_dim.items():
            if dim_name not in self.tables:
                logger.warning(f"Dimension {dim_name} not loaded, skipping {fk}")
                continue
                
            dim_df = self.tables[dim_name]
            
            # Find primary key column in dimension table
            pk_col = fk  # Usually the same name (e.g., geo_sk in geo_dimension)
            
            # Get all valid values from dimension table
            valid_keys = set(dim_df[pk_col].to_list())
            
            # Find rows with invalid keys
            invalid_rows = fact_df.filter(
                ~pl.col(fk).is_in(valid_keys) | pl.col(fk).is_null()
            )
            
            n_invalid = len(invalid_rows)
            invalid_counts[fk] = n_invalid
            
            if n_invalid > 0:
                logger.warning(f"Found {n_invalid} invalid or null values in {fact_table_name}.{fk}")
            else:
                logger.info(f"All {fk} values in {fact_table_name} are valid")
        
        return invalid_counts
    
    def validate_and_fix_all_tables(self) -> Dict[str, bool]:
        """
        Validate and fix all tables in the output directory.
        
        Returns:
            Dictionary of table names and whether they were modified
        """
        # First load all tables
        self.load_tables()
        
        results = {}
        
        # Validate and fix dimension tables first
        dimension_schemas = self.schemas.get('dimensions', {})
        for dim_name in dimension_schemas.keys():
            if dim_name in self.tables:
                results[dim_name] = self.enforce_schema(dim_name)
        
        # Then validate and fix fact tables
        fact_schemas = self.schemas.get('facts', {})
        for fact_name in fact_schemas.keys():
            if fact_name in self.tables:
                results[fact_name] = self.enforce_schema(fact_name)
                
                # Also validate referential integrity
                invalid_counts = self.validate_referential_integrity(fact_name)
                
                # Log summary of referential integrity validation
                total_invalid = sum(invalid_counts.values())
                if total_invalid > 0:
                    logger.warning(f"Found total of {total_invalid} referential integrity issues in {fact_name}")
                else:
                    logger.info(f"All foreign keys in {fact_name} are valid")
        
        return results

def run_schema_fix(output_dir: Path) -> bool:
    """
    Run the schema validation and fix process on all tables.
    
    Args:
        output_dir: Directory containing data files to fix
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Starting schema validation and fix process")
        
        # Initialize validator
        validator = SchemaValidator(output_dir=output_dir)
        
        # Validate and fix all tables
        results = validator.validate_and_fix_all_tables()
        
        # Log summary
        fixed_count = sum(1 for fixed in results.values() if fixed)
        logger.info(f"Fixed {fixed_count} of {len(results)} tables")
        
        return True
    except Exception as e:
        logger.error(f"Error in schema fix process: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Set up logging when run directly
    logging.basicConfig(level=logging.INFO)
    
    # Run the fix on the default output directory
    run_schema_fix(Path("output"))