"""
Fix Manager - Integrates data quality fixes into the main pipeline

This module consolidates all the fix logic from the temporary fix scripts
and integrates them directly into the transformation pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import polars as pl
import hashlib

from ahgd_etl.config import settings


class FixManager:
    """Manages inline data quality fixes during ETL processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.unknown_sk = -1  # Standard unknown surrogate key
        
    def add_unknown_dimension_member(
        self, 
        df: pl.DataFrame, 
        dimension_name: str,
        sk_column: str,
        additional_columns: Optional[Dict[str, Any]] = None
    ) -> pl.DataFrame:
        """
        Add unknown member to dimension table.
        
        Args:
            df: Dimension dataframe
            dimension_name: Name of the dimension
            sk_column: Surrogate key column name
            additional_columns: Additional column values for unknown member
            
        Returns:
            DataFrame with unknown member added
        """
        # Check if unknown member already exists
        if len(df.filter(pl.col(sk_column) == self.unknown_sk)) > 0:
            self.logger.debug(f"Unknown member already exists in {dimension_name}")
            return df
            
        # Build unknown member row
        unknown_row = {sk_column: self.unknown_sk}
        
        # Add standard columns based on dimension type
        if dimension_name == "geo_dimension":
            unknown_row.update({
                "geo_id": "UNKNOWN",
                "geo_level": "UNKNOWN",
                "geo_name": "Unknown Geography",
                "state_code": "UNK",
                "state_name": "Unknown",
                "latitude": None,
                "longitude": None,
                "parent_geo_sk": None
            })
        elif dimension_name == "dim_health_condition":
            unknown_row.update({
                "condition_code": "UNKNOWN",
                "condition_name": "Unknown Condition",
                "condition_category": "Unknown",
                "is_unknown": True
            })
        elif dimension_name == "dim_demographic":
            unknown_row.update({
                "age_group": "Unknown",
                "sex": "U",
                "is_unknown": True
            })
        elif dimension_name == "dim_person_characteristic":
            unknown_row.update({
                "characteristic_type": "Unknown",
                "characteristic_value": "Unknown",
                "characteristic_category": "Unknown",
                "is_unknown": True
            })
            
        # Add any additional columns
        if additional_columns:
            unknown_row.update(additional_columns)
            
        # Add timestamp
        unknown_row["etl_processed_at"] = pl.datetime.now()
        
        # Create unknown member dataframe
        unknown_df = pl.DataFrame([unknown_row])
        
        # Ensure schema matches
        for col in df.columns:
            if col not in unknown_df.columns:
                unknown_df = unknown_df.with_columns(pl.lit(None).alias(col))
                
        # Reorder columns to match original
        unknown_df = unknown_df.select(df.columns)
        
        # Concatenate
        result = pl.concat([unknown_df, df])
        
        self.logger.info(f"Added unknown member to {dimension_name}")
        return result
    
    def fix_referential_integrity(
        self,
        fact_df: pl.DataFrame,
        dimension_df: pl.DataFrame,
        fact_key_column: str,
        dimension_key_column: str,
        table_name: str
    ) -> pl.DataFrame:
        """
        Fix referential integrity by replacing invalid foreign keys with unknown member.
        
        Args:
            fact_df: Fact table dataframe
            dimension_df: Dimension table dataframe
            fact_key_column: Foreign key column in fact table
            dimension_key_column: Primary key column in dimension table
            table_name: Name of the fact table for logging
            
        Returns:
            Fact dataframe with fixed referential integrity
        """
        # Get valid keys from dimension
        valid_keys = set(dimension_df[dimension_key_column].to_list())
        
        # Check if unknown member exists
        if self.unknown_sk not in valid_keys:
            self.logger.warning(f"Unknown member not found in dimension for {table_name}")
            valid_keys.add(self.unknown_sk)
            
        # Find invalid references
        invalid_mask = ~fact_df[fact_key_column].is_in(valid_keys)
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            self.logger.warning(
                f"Found {invalid_count} invalid {fact_key_column} references in {table_name}. "
                f"Replacing with unknown member."
            )
            
            # Replace invalid references with unknown
            fact_df = fact_df.with_columns(
                pl.when(invalid_mask)
                .then(self.unknown_sk)
                .otherwise(pl.col(fact_key_column))
                .alias(fact_key_column)
            )
            
        return fact_df
    
    def deduplicate_fact_table(
        self,
        df: pl.DataFrame,
        key_columns: List[str],
        aggregation_columns: Dict[str, str],
        table_name: str
    ) -> pl.DataFrame:
        """
        Remove duplicates from fact table by aggregating measures.
        
        Args:
            df: Fact table dataframe
            key_columns: Columns that define the grain
            aggregation_columns: Dict mapping column names to aggregation methods
            table_name: Name of the table for logging
            
        Returns:
            Deduplicated dataframe
        """
        # Check for duplicates
        duplicate_count = df.group_by(key_columns).agg(pl.count()).filter(pl.col("count") > 1).shape[0]
        
        if duplicate_count > 0:
            self.logger.warning(
                f"Found {duplicate_count} duplicate key combinations in {table_name}. "
                f"Aggregating using specified rules."
            )
            
            # Build aggregation expressions
            agg_exprs = []
            for col, method in aggregation_columns.items():
                if method == "sum":
                    agg_exprs.append(pl.sum(col).alias(col))
                elif method == "mean":
                    agg_exprs.append(pl.mean(col).alias(col))
                elif method == "max":
                    agg_exprs.append(pl.max(col).alias(col))
                elif method == "min":
                    agg_exprs.append(pl.min(col).alias(col))
                elif method == "first":
                    agg_exprs.append(pl.first(col).alias(col))
                else:
                    self.logger.warning(f"Unknown aggregation method {method} for {col}, using sum")
                    agg_exprs.append(pl.sum(col).alias(col))
                    
            # Add non-aggregated columns (like timestamps)
            for col in df.columns:
                if col not in key_columns and col not in aggregation_columns:
                    agg_exprs.append(pl.first(col).alias(col))
                    
            # Perform aggregation
            df = df.group_by(key_columns).agg(agg_exprs)
            
        return df
    
    def enforce_schema(
        self,
        df: pl.DataFrame,
        schema_definition: Dict[str, str],
        table_name: str
    ) -> pl.DataFrame:
        """
        Enforce schema on dataframe, adding missing columns and fixing types.
        
        Args:
            df: Input dataframe
            schema_definition: Expected schema from configuration
            table_name: Name of the table for logging
            
        Returns:
            Dataframe with enforced schema
        """
        # Map string types to Polars types
        type_mapping = {
            "Int64": pl.Int64,
            "Int32": pl.Int32,
            "Float64": pl.Float64,
            "Utf8": pl.Utf8,
            "Boolean": pl.Boolean,
            "Date": pl.Date,
            "Datetime": pl.Datetime
        }
        
        # Add missing columns
        for col_name, col_type in schema_definition.items():
            if col_name not in df.columns:
                self.logger.info(f"Adding missing column {col_name} to {table_name}")
                
                polars_type = type_mapping.get(col_type, pl.Utf8)
                if col_type in ["Int64", "Int32"]:
                    default_value = 0
                elif col_type == "Float64":
                    default_value = 0.0
                elif col_type == "Boolean":
                    default_value = False
                else:
                    default_value = None
                    
                df = df.with_columns(pl.lit(default_value).alias(col_name).cast(polars_type))
                
        # Fix data types
        for col_name, col_type in schema_definition.items():
            if col_name in df.columns:
                expected_type = type_mapping.get(col_type)
                if expected_type and df[col_name].dtype != expected_type:
                    self.logger.info(
                        f"Converting {col_name} from {df[col_name].dtype} to {expected_type} in {table_name}"
                    )
                    try:
                        df = df.with_columns(pl.col(col_name).cast(expected_type))
                    except Exception as e:
                        self.logger.error(f"Failed to convert {col_name}: {str(e)}")
                        
        # Reorder columns to match schema
        ordered_columns = list(schema_definition.keys())
        extra_columns = [col for col in df.columns if col not in ordered_columns]
        final_columns = ordered_columns + extra_columns
        
        df = df.select([col for col in final_columns if col in df.columns])
        
        return df
    
    def generate_surrogate_key(self, *args) -> int:
        """
        Generate a deterministic surrogate key from business keys.
        
        Args:
            *args: Business key components
            
        Returns:
            Integer surrogate key
        """
        # Concatenate all arguments
        key_string = "|".join(str(arg) for arg in args)
        
        # Generate hash
        hash_value = hashlib.md5(key_string.encode()).hexdigest()
        
        # Convert to integer (use first 15 hex chars to avoid overflow)
        return int(hash_value[:15], 16)
    
    def run_all_fixes(self, output_dir: Path) -> bool:
        """
        Run all fixes on existing files (standalone mode).
        
        Args:
            output_dir: Directory containing parquet files
            
        Returns:
            bool: True if all fixes succeeded
        """
        self.logger.info("Running standalone fixes on all tables...")
        
        success = True
        
        # Fix dimensions first
        dimension_files = [
            ("geo_dimension.parquet", "geo_dimension", "geo_sk"),
            ("dim_health_condition.parquet", "dim_health_condition", "condition_sk"),
            ("dim_demographic.parquet", "dim_demographic", "demographic_sk"),
            ("dim_person_characteristic.parquet", "dim_person_characteristic", "characteristic_sk")
        ]
        
        for file_name, dim_name, sk_column in dimension_files:
            file_path = output_dir / file_name
            if file_path.exists():
                self.logger.info(f"Fixing {file_name}...")
                try:
                    df = pl.read_parquet(file_path)
                    df = self.add_unknown_dimension_member(df, dim_name, sk_column)
                    df.write_parquet(file_path)
                except Exception as e:
                    self.logger.error(f"Failed to fix {file_name}: {str(e)}")
                    success = False
                    
        # Then fix fact tables
        # Implementation depends on specific fact table requirements
        
        return success