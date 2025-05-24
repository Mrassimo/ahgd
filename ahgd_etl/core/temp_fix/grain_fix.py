"""
Fact table grain management module for AHGD ETL.

This module handles grain-related fixes for fact tables:
1. Identifies and analyzes duplicate keys in fact tables
2. Resolves duplicate keys by aggregating measure values
3. Ensures proper grain definition and enforcement

Usage:
    from ahgd_etl.core.temp_fix.grain_fix import GrainHandler
    handler = GrainHandler(output_dir=Path('output'))
    handler.analyze_grain_issues('fact_health_conditions_refined.parquet')
    handler.fix_grain('fact_health_conditions_refined.parquet')
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import polars as pl

# Set up logger
logger = logging.getLogger(__name__)

class GrainHandler:
    """
    Handler for fact table grain issues in the AHGD ETL pipeline.
    Manages analysis and resolution of duplicate keys through aggregation.
    """
    
    def __init__(self, output_dir: Path, config_dir: Optional[Path] = None):
        """
        Initialize the GrainHandler.
        
        Args:
            output_dir: Directory containing data files to fix
            config_dir: Directory containing YAML configuration files
        """
        self.output_dir = Path(output_dir)
        self.config_dir = Path(config_dir) if config_dir else Path(__file__).parent.parent.parent / "config" / "yaml"
        self.schemas = {}
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
    
    def _identify_grain_columns(self, fact_name: str) -> List[str]:
        """
        Identify grain-defining columns for a fact table.
        
        Args:
            fact_name: Name of the fact table
            
        Returns:
            List of column names that define the grain
        """
        fact_schema = self.schemas.get('facts', {}).get(fact_name, {})
        if not fact_schema:
            logger.warning(f"No schema found for {fact_name}")
            return []
        
        # Grain is defined by all _sk (surrogate key) columns that link to dimensions
        grain_cols = [col for col in fact_schema.keys() if col.endswith('_sk')]
        
        # Add any other business key columns that might be part of the grain
        # This would be domain-specific and depends on the fact table
        
        return grain_cols
    
    def _identify_measure_columns(self, fact_name: str) -> List[str]:
        """
        Identify measure columns for a fact table.
        
        Args:
            fact_name: Name of the fact table
            
        Returns:
            List of column names that contain measures
        """
        fact_schema = self.schemas.get('facts', {}).get(fact_name, {})
        if not fact_schema:
            logger.warning(f"No schema found for {fact_name}")
            return []
        
        # Exclude grain columns, timestamps, and other non-measure columns
        grain_cols = self._identify_grain_columns(fact_name)
        excluded_cols = grain_cols + ['etl_processed_at']
        
        # Measures are typically numeric columns not in the grain
        measure_cols = []
        for col, type_str in fact_schema.items():
            if col not in excluded_cols and type_str in ('Int64', 'Int32', 'Float64', 'Float32'):
                measure_cols.append(col)
        
        return measure_cols
    
    def analyze_grain_issues(self, fact_file: str) -> Dict[str, Any]:
        """
        Analyze grain issues in a fact table.
        
        Args:
            fact_file: Name of the fact table file
            
        Returns:
            Dictionary with analysis results
        """
        fact_path = self.output_dir / fact_file
        if not fact_path.exists():
            logger.warning(f"Fact table not found: {fact_path}")
            return {"status": "file_not_found"}
        
        fact_name = fact_file.replace('.parquet', '')
        
        try:
            # Load fact table
            fact_df = pl.read_parquet(fact_path)
            logger.info(f"Loaded fact table {fact_file} with {len(fact_df)} rows")
            
            # Identify grain columns
            grain_cols = self._identify_grain_columns(fact_name)
            if not grain_cols:
                logger.warning(f"Could not identify grain columns for {fact_name}")
                return {"status": "no_grain_columns"}
            
            # Count rows in the original table
            total_rows = len(fact_df)
            
            # Count distinct combinations of grain columns
            distinct_combinations = fact_df.select(grain_cols).unique().height
            
            # If distinct combinations equals total rows, there's no grain issue
            if distinct_combinations == total_rows:
                logger.info(f"No duplicate keys found in {fact_name}")
                return {
                    "status": "no_duplicates",
                    "total_rows": total_rows,
                    "distinct_combinations": distinct_combinations,
                    "duplicate_count": 0
                }
            
            # Calculate duplicate count
            duplicate_count = total_rows - distinct_combinations
            
            # Find which combinations have duplicates - using Polars group_by instead of pandas value_counts
            duplicate_analysis = fact_df.select(grain_cols).group_by(grain_cols).agg(
                pl.len().alias("count")
            )

            # Filter to only combinations with count > 1
            duplicates = duplicate_analysis.filter(pl.col("count") > 1)
            duplicate_keys_count = len(duplicates)
            
            # Calculate some statistics on the duplicates
            if duplicate_keys_count > 0:
                max_duplicates = duplicates["count"].max()
                avg_duplicates = duplicates["count"].mean()
            else:
                max_duplicates = 0
                avg_duplicates = 0
            
            logger.warning(
                f"Found {duplicate_count} duplicate rows in {fact_name} "
                f"({duplicate_keys_count} distinct combinations have duplicates)"
            )
            
            return {
                "status": "duplicates_found",
                "total_rows": total_rows,
                "distinct_combinations": distinct_combinations,
                "duplicate_count": duplicate_count,
                "duplicate_keys_count": duplicate_keys_count,
                "max_duplicates": max_duplicates,
                "avg_duplicates": avg_duplicates
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {fact_file}: {e}")
            return {"status": "error", "error": str(e)}
    
    def fix_grain(self, fact_file: str) -> bool:
        """
        Fix grain issues in a fact table by aggregating duplicate key combinations.
        
        Args:
            fact_file: Name of the fact table file
            
        Returns:
            bool: True if fixes were applied, False otherwise
        """
        fact_path = self.output_dir / fact_file
        if not fact_path.exists():
            logger.warning(f"Fact table not found: {fact_path}")
            return False
        
        fact_name = fact_file.replace('.parquet', '')
        
        try:
            # Load fact table
            fact_df = pl.read_parquet(fact_path)
            logger.info(f"Loaded fact table {fact_file} with {len(fact_df)} rows")
            
            # Identify grain and measure columns
            grain_cols = self._identify_grain_columns(fact_name)
            measure_cols = self._identify_measure_columns(fact_name)
            
            if not grain_cols:
                logger.warning(f"Could not identify grain columns for {fact_name}")
                return False
                
            if not measure_cols:
                logger.warning(f"Could not identify measure columns for {fact_name}")
                return False
            
            # First check if there are actually duplicates
            original_row_count = len(fact_df)
            distinct_keys = fact_df.select(grain_cols).unique().height
            
            if distinct_keys == original_row_count:
                logger.info(f"No duplicate keys found in {fact_name}, no fixes needed")
                return False
            
            # Create aggregation expressions for each measure column
            agg_exprs = []
            for col in measure_cols:
                # Use sum aggregation for measures - could be configurable based on semantics
                agg_exprs.append(pl.sum(col).alias(col))
            
            # For non-grain, non-measure columns, we'll take the first value
            # First identify these columns
            other_cols = [
                col for col in fact_df.columns 
                if col not in grain_cols and col not in measure_cols and col != 'etl_processed_at'
            ]
            
            # Add first() aggregation for these columns
            for col in other_cols:
                agg_exprs.append(pl.first(col).alias(col))
            
            # Add timestamp column updated to current time
            from datetime import datetime
            agg_exprs.append(pl.lit(datetime.now()).alias('etl_processed_at'))
            
            # Perform the aggregation
            aggregated_df = fact_df.group_by(grain_cols).agg(agg_exprs)
            
            # Calculate row reduction
            new_row_count = len(aggregated_df)
            rows_reduced = original_row_count - new_row_count
            
            if rows_reduced > 0:
                logger.info(f"Reduced {fact_name} from {original_row_count} to {new_row_count} rows "
                           f"({rows_reduced} duplicates resolved)")
                
                # Save aggregated fact table
                aggregated_df.write_parquet(fact_path)
                logger.info(f"Saved grain-fixed {fact_name} to {fact_path}")
                return True
            else:
                logger.info(f"No changes made to {fact_name}")
                return False
            
        except Exception as e:
            logger.error(f"Error fixing grain in {fact_file}: {e}")
            return False
    
    def fix_all_fact_tables(self) -> Dict[str, bool]:
        """
        Fix grain issues in all fact tables.
        
        Returns:
            Dictionary mapping fact table names to whether they were fixed
        """
        results = {}
        fact_schemas = self.schemas.get('facts', {})
        
        for fact_name in fact_schemas.keys():
            fact_file = f"{fact_name}.parquet"
            fact_path = self.output_dir / fact_file
            
            if fact_path.exists():
                # Analyze first to log information
                analysis = self.analyze_grain_issues(fact_file)
                
                if analysis.get("status") == "duplicates_found":
                    # Fix grain issues
                    results[fact_name] = self.fix_grain(fact_file)
                else:
                    results[fact_name] = False
            else:
                logger.warning(f"Fact table not found: {fact_path}")
                results[fact_name] = False
        
        return results

def run_grain_fix(output_dir: Path) -> bool:
    """
    Run the grain fix process on all fact tables.
    
    Args:
        output_dir: Directory containing data files to fix
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Starting fact table grain fix process")
        
        # Initialize handler
        handler = GrainHandler(output_dir=output_dir)
        
        # Fix all fact tables
        results = handler.fix_all_fact_tables()
        
        # Log summary
        fixed_count = sum(1 for fixed in results.values() if fixed)
        logger.info(f"Fixed grain issues in {fixed_count} of {len(results)} fact tables")
        
        return True
    except Exception as e:
        logger.error(f"Error in grain fix process: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Set up logging when run directly
    logging.basicConfig(level=logging.INFO)
    
    # Run the fix on the default output directory
    run_grain_fix(Path("output"))