"""
Dimension handling fix for AHGD ETL data architecture issues.

This module handles issues related to dimensions and surrogate key relationships:
1. Ensures all dimension tables have unknown members
2. Fixes fact table foreign key references to dimension tables
3. Ensures consistent surrogate key generation

Usage:
    from ahgd_etl.core.temp_fix.dimension_fix import DimensionHandler
    handler = DimensionHandler(output_dir=Path('output'))
    handler.load_dimensions()
    handler.ensure_unknown_members()
    handler.fix_fact_table_refs('fact_health_conditions_refined.parquet')
"""

import logging
import yaml
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import polars as pl

# Set up logger
logger = logging.getLogger(__name__)

class DimensionHandler:
    """
    Handler for dimension-related fixes in the AHGD ETL pipeline.
    Manages dimension table loading, unknown member creation, and fact table reference fixing.
    """
    
    def __init__(self, output_dir: Path, config_dir: Optional[Path] = None):
        """
        Initialize the DimensionHandler.
        
        Args:
            output_dir: Directory containing data files to fix
            config_dir: Directory containing YAML configuration files (defaults to ahgd_etl/config/yaml)
        """
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / "config" / "yaml"
        self.schemas = {}
        self.dimensions = {}
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
    
    def generate_surrogate_key(self, *args) -> str:
        """
        Generate a surrogate key from business keys using MD5 hash.
        
        Args:
            *args: Business key components to hash
            
        Returns:
            MD5 hash of the concatenated business keys
        """
        if not args:
            raise ValueError("At least one argument required for surrogate key generation")
            
        # Convert all arguments to strings and handle NULL/None values
        str_args = []
        for arg in args:
            if arg is None:
                # Use a special placeholder for NULL values
                str_args.append("UNKNOWN")
            else:
                str_args.append(str(arg))
        
        # Create combined key string and generate hash
        key_str = '_'.join(str_args)
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()
    
    def load_dimensions(self):
        """Load all dimension tables from output directory."""
        dimension_schemas = self.schemas.get('dimensions', {})
        
        for dim_name, _ in dimension_schemas.items():
            dim_path = self.output_dir / f"{dim_name}.parquet"
            
            if dim_path.exists():
                try:
                    self.dimensions[dim_name] = pl.read_parquet(dim_path)
                    logger.info(f"Loaded dimension {dim_name} with {len(self.dimensions[dim_name])} rows")
                except Exception as e:
                    logger.error(f"Error loading dimension {dim_name}: {e}")
            else:
                logger.warning(f"Dimension file not found: {dim_path}")
        
        logger.info(f"Loaded {len(self.dimensions)} dimension tables")
    
    def ensure_unknown_members(self):
        """
        Ensure all dimension tables have unknown members for handling missing references.
        Creates or updates dimension tables with unknown members.
        """
        for dim_name, dim_df in self.dimensions.items():
            schema = self.schemas.get('dimensions', {}).get(dim_name, {})
            if not schema:
                logger.warning(f"No schema found for dimension {dim_name}")
                continue
            
            # Determine the surrogate key column
            sk_col = None
            for col, dtype in schema.items():
                if col.endswith('_sk'):
                    sk_col = col
                    break
            
            if not sk_col:
                logger.warning(f"No surrogate key column found for {dim_name}")
                continue
            
            # Check if unknown member exists
            if "is_unknown" in dim_df.columns:
                unknown_exists = dim_df.filter(pl.col("is_unknown") == True).height > 0
            else:
                # If is_unknown column doesn't exist, add it
                dim_df = dim_df.with_columns(pl.lit(False).alias("is_unknown"))
                unknown_exists = False
            
            if not unknown_exists:
                # Create unknown member
                unknown_data = {}

                # Fill in required columns based on schema
                for col in dim_df.columns:
                    if col == sk_col:
                        # Generate special unknown surrogate key based on column type
                        if str(dim_df[col].dtype).startswith("Int"):
                            # For integer surrogate keys, use a large negative number
                            unknown_data[col] = -9999
                        else:
                            # For string surrogate keys, use a hash
                            unknown_data[col] = self.generate_surrogate_key("UNKNOWN", dim_name)
                    elif col == "is_unknown":
                        unknown_data[col] = True
                    else:
                        # For other columns, use the column name as the value (to make it identifiable)
                        data_type = schema.get(col)
                        # Use the column's actual type for creating the unknown value
                        if str(dim_df[col].dtype).startswith("Int"):
                            unknown_data[col] = -1
                        elif str(dim_df[col].dtype).startswith("Float"):
                            unknown_data[col] = -1.0
                        elif str(dim_df[col].dtype).startswith("Boolean"):
                            unknown_data[col] = False
                        elif str(dim_df[col].dtype).startswith("Date") or str(dim_df[col].dtype).startswith("Datetime"):
                            # For date/datetime, use a safe default
                            unknown_data[col] = dim_df[col][0]  # Use first row's date as template
                        else:
                            unknown_data[col] = "UNKNOWN"

                # Create unknown member DataFrame with explicit typing
                # Match the types of the original dimension
                unknown_df = pl.DataFrame([unknown_data])

                # Ensure each column has the correct type
                for col_name, col_type in dim_df.schema.items():
                    if col_name in unknown_df.columns:
                        try:
                            unknown_df = unknown_df.with_columns(
                                pl.col(col_name).cast(col_type).alias(col_name)
                            )
                        except Exception as e:
                            # If casting fails, log it but don't fail the process
                            logger.warning(f"Could not cast {col_name} to {col_type}: {e}")

                # A more complete solution for handling categorical columns
                for col_name, col_type in dim_df.schema.items():
                    if col_name in unknown_df.columns:
                        # For categorical columns, we need special handling
                        if str(col_type).startswith('Categorical'):
                            # Get the string representation of the unknown value
                            unknown_value = unknown_data[col_name]

                            # First, convert the categorical values to strings
                            orig_values = dim_df[col_name].to_list()

                            # Create a new categorical type with the original values plus the unknown value
                            if unknown_value not in orig_values:
                                # Extract the unknown value as string and add it directly to the DataFrame
                                unknown_df = unknown_df.with_columns(
                                    pl.lit(unknown_value).cast(pl.Utf8).alias(f"{col_name}_temp")
                                )

                                # Create categorical column with the same categories as the dimension
                                categories = dim_df[col_name].cast(pl.Utf8).unique().to_list()
                                if unknown_value not in categories:
                                    categories.append(unknown_value)

                                # Use the categories to create a new categorical column in unknown_df
                                unknown_df = unknown_df.with_columns(
                                    pl.col(f"{col_name}_temp").cast(pl.Categorical).alias(col_name)
                                )

                                # Drop the temporary column
                                unknown_df = unknown_df.drop(f"{col_name}_temp")
                        else:
                            # For other column types, make sure the types match
                            unknown_df = unknown_df.with_columns(
                                pl.col(col_name).cast(col_type).alias(col_name)
                            )

                try:
                    # Try to concatenate with vertical alignment
                    dim_df = pl.concat([dim_df, unknown_df], how="vertical")
                except Exception as e:
                    logger.warning(f"Error in standard concat: {e}")
                    # Fallback: convert to Python dictionaries and rebuild
                    try:
                        # Get all rows as dictionaries
                        rows = [row for row in dim_df.iter_rows(named=True)]
                        # Add unknown row
                        unknown_row = unknown_df.row(0, named=True)
                        rows.append(unknown_row)
                        # Rebuild from rows
                        dim_df = pl.DataFrame(rows)
                    except Exception as e2:
                        logger.error(f"Failed to add unknown member: {e2}")
                        # If all else fails, skip this dimension but don't crash
                        continue

                # Save updated dimension
                dim_path = self.output_dir / f"{dim_name}.parquet"
                dim_df.write_parquet(dim_path)
                logger.info(f"Added unknown member to {dim_name} and saved to {dim_path}")

                # Update in-memory dimension
                self.dimensions[dim_name] = dim_df
        
        logger.info("Completed ensuring unknown members in all dimensions")
    
    def fix_fact_table_refs(self, fact_table_name: str) -> bool:
        """
        Fix foreign key references in a fact table.
        Replaces null or invalid surrogate keys with references to unknown dimension members.
        
        Args:
            fact_table_name: Name of the fact table file to fix
            
        Returns:
            bool: True if fixes were applied, False otherwise
        """
        fact_path = self.output_dir / fact_table_name
        if not fact_path.exists():
            logger.warning(f"Fact table not found: {fact_path}")
            return False
        
        # Load fact table
        try:
            fact_df = pl.read_parquet(fact_path)
            logger.info(f"Loaded fact table {fact_table_name} with {len(fact_df)} rows")
        except Exception as e:
            logger.error(f"Error loading fact table {fact_table_name}: {e}")
            return False
        
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
                potential_dims = [d for d in self.dimensions.keys() if dim_prefix in d]
                if potential_dims:
                    fk_to_dim[fk] = potential_dims[0]
        
        # Fix each foreign key
        changes_made = False
        for fk, dim_name in fk_to_dim.items():
            if dim_name not in self.dimensions:
                logger.warning(f"Dimension {dim_name} not loaded, skipping {fk}")
                continue
                
            dim_df = self.dimensions[dim_name]
            
            # Find primary key column in dimension table
            # Special handling for various column mappings
            if fk == 'health_condition_sk':
                pk_col = 'condition_sk'
            elif fk == 'condition_sk' and dim_name == 'dim_health_condition':
                # Already aligned
                pk_col = fk
            elif fk == 'demo_sk':
                # Map demo_sk to demographic_sk
                pk_col = 'demographic_sk'
                dim_name = 'dim_demographic'
                # Update the dimension reference
                dim_df = self.dimensions[dim_name]
            else:
                # Usually the same name (e.g., geo_sk in geo_dimension)
                pk_col = fk
            
            # Get all valid values from dimension table
            valid_keys = set(dim_df[pk_col].to_list())
            
            # Find unknown member surrogate key
            unknown_sk = None
            if "is_unknown" in dim_df.columns:
                unknown_row = dim_df.filter(pl.col("is_unknown") == True)
                if len(unknown_row) > 0:
                    unknown_sk = unknown_row[0, pk_col]
            
            if not unknown_sk:
                logger.warning(f"Unknown member not found in {dim_name}, skipping {fk}")
                continue
                
            # Count invalid foreign keys
            invalid_keys = fact_df.filter(
                ~pl.col(fk).is_in(valid_keys) | pl.col(fk).is_null()
            )
            
            n_invalid = len(invalid_keys)
            if n_invalid > 0:
                logger.info(f"Found {n_invalid} invalid or null values in {fk}")
                
                # Replace invalid keys with unknown surrogate key
                fact_df = fact_df.with_columns(
                    pl.when(~pl.col(fk).is_in(valid_keys) | pl.col(fk).is_null())
                    .then(pl.lit(unknown_sk))
                    .otherwise(pl.col(fk))
                    .alias(fk)
                )
                
                changes_made = True
        
        # Save updated fact table if changes were made
        if changes_made:
            fact_df.write_parquet(fact_path)
            logger.info(f"Saved fixed fact table {fact_table_name}")
        else:
            logger.info(f"No changes needed for {fact_table_name}")
        
        return changes_made

    def fix_all_fact_tables(self) -> Dict[str, bool]:
        """
        Fix all fact tables in the output directory.
        
        Returns:
            Dictionary of fact table names and whether they were modified
        """
        result = {}
        fact_schemas = self.schemas.get('facts', {})
        
        for fact_name in fact_schemas.keys():
            fact_file = f"{fact_name}.parquet"
            fact_path = self.output_dir / fact_file
            
            if fact_path.exists():
                result[fact_name] = self.fix_fact_table_refs(fact_file)
            else:
                logger.warning(f"Fact table not found: {fact_path}")
                result[fact_name] = False
        
        return result
        
def run_dimension_fix(output_dir: Path) -> bool:
    """
    Run the dimension fix process on all tables.
    
    Args:
        output_dir: Directory containing data files to fix
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Starting dimension fix process")
        
        # Initialize handler
        handler = DimensionHandler(output_dir=output_dir)
        
        # Load all dimensions
        handler.load_dimensions()
        
        # Ensure unknown members exist in all dimensions
        handler.ensure_unknown_members()
        
        # Fix all fact tables
        results = handler.fix_all_fact_tables()
        
        # Log summary
        fixed_count = sum(1 for fixed in results.values() if fixed)
        logger.info(f"Fixed {fixed_count} of {len(results)} fact tables")
        
        return True
    except Exception as e:
        logger.error(f"Error in dimension fix process: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Set up logging when run directly
    logging.basicConfig(level=logging.INFO)
    
    # Run the fix on the default output directory
    run_dimension_fix(Path("output"))