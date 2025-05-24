"""Core dimension builders for the data warehouse."""

from typing import List, Dict, Any, Optional
import polars as pl
from pathlib import Path
from datetime import datetime
import hashlib

from ..config import get_settings
from ..utils import get_logger


class DimensionBuilder:
    """Base class for dimension builders."""
    
    def __init__(self):
        """Initialize the dimension builder."""
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.etl_timestamp = datetime.now()
    
    def _generate_hash_sk(self, *args) -> str:
        """Generate a hash-based surrogate key.
        
        Args:
            *args: Values to hash together
            
        Returns:
            MD5 hash string
        """
        # Concatenate all arguments with a delimiter
        key_string = "|".join(str(arg) for arg in args if arg is not None)
        
        # Generate MD5 hash
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _enforce_schema(self, df: pl.DataFrame, schema_name: str) -> pl.DataFrame:
        """Enforce the target schema on the DataFrame.
        
        Args:
            df: Input DataFrame
            schema_name: Name of the schema to enforce
            
        Returns:
            DataFrame with enforced schema
        """
        schema = self.settings.get_schema(schema_name)
        expected_columns = [col["name"] for col in schema["columns"]]
        
        # Ensure all expected columns exist
        for col in expected_columns:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).alias(col))
        
        # Select and reorder columns
        df = df.select(expected_columns)
        
        # Cast to correct types
        type_mapping = {
            "Int64": pl.Int64,
            "Utf8": pl.Utf8,
            "Boolean": pl.Boolean,
            "Datetime": pl.Datetime,
            "Categorical": pl.Categorical
        }
        
        cast_expressions = []
        for col_def in schema["columns"]:
            col_name = col_def["name"]
            col_type = col_def["dtype"]
            
            if col_type in type_mapping:
                if col_type == "Categorical":
                    expr = pl.col(col_name).cast(pl.Utf8, strict=False).cast(pl.Categorical)
                else:
                    expr = pl.col(col_name).cast(type_mapping[col_type], strict=False)
                cast_expressions.append(expr)
            else:
                cast_expressions.append(pl.col(col_name))
        
        return df.select(cast_expressions)
    
    def save_to_parquet(self, df: pl.DataFrame, output_filename: str) -> Path:
        """Save dimension to Parquet format.
        
        Args:
            df: DataFrame to save
            output_filename: Name of output file
            
        Returns:
            Path to saved file
        """
        output_path = self.settings.output_dir / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.write_parquet(output_path, compression="snappy")
        self.logger.info(f"Saved {output_filename} with {len(df)} records")
        
        return output_path


class HealthConditionDimensionBuilder(DimensionBuilder):
    """Builds the health condition dimension."""
    
    def build(self) -> pl.DataFrame:
        """Build the health condition dimension.
        
        Returns:
            Polars DataFrame containing health condition dimension
        """
        self.logger.info("Building health condition dimension")
        
        # Get condition definitions from column mappings
        mappings = self.settings.get_column_mapping("G19")
        conditions = mappings.get("conditions", [])
        
        # Add conditions from G20 (condition counts)
        g20_mappings = self.settings.get_column_mapping("G20")
        condition_counts = g20_mappings.get("condition_counts", [])
        
        # Build records
        records = []
        
        # Add unknown member
        records.append({
            "condition_sk": self._generate_hash_sk("UNKNOWN"),
            "condition_code": "UNKNOWN",
            "condition_name": "Unknown Condition",
            "condition_category": "Unknown",
            "is_unknown": True,
            "etl_processed_at": self.etl_timestamp
        })
        
        # Add health conditions from G19
        for condition in conditions:
            records.append({
                "condition_sk": self._generate_hash_sk(condition["code"]),
                "condition_code": condition["code"],
                "condition_name": condition["name"],
                "condition_category": condition.get("category", "Other"),
                "is_unknown": False,
                "etl_processed_at": self.etl_timestamp
            })
        
        # Add condition count categories from G20
        for count_item in condition_counts:
            code = f"CONDITION_COUNT_{count_item['value'].replace(' ', '_').upper()}"
            records.append({
                "condition_sk": self._generate_hash_sk(code),
                "condition_code": code,
                "condition_name": count_item["value"],
                "condition_category": "Condition Count",
                "is_unknown": False,
                "etl_processed_at": self.etl_timestamp
            })
        
        # Create DataFrame and enforce schema
        df = pl.DataFrame(records)
        df = self._enforce_schema(df, "dim_health_condition")
        
        self.logger.info(f"Health condition dimension complete: {len(df)} records")
        
        return df


class DemographicDimensionBuilder(DimensionBuilder):
    """Builds the demographic dimension (age groups and sex)."""
    
    def build(self) -> pl.DataFrame:
        """Build the demographic dimension.
        
        Returns:
            Polars DataFrame containing demographic dimension
        """
        self.logger.info("Building demographic dimension")
        
        # Get demographic definitions
        demographics = self.settings.get_demographics()
        age_groups = demographics.get("age_groups", [])
        sex_categories = demographics.get("sex_categories", [])
        
        records = []
        
        # Add unknown member
        records.append({
            "demographic_sk": self._generate_hash_sk("UNKNOWN", "UNKNOWN"),
            "age_group": "Unknown",
            "sex": "Unknown",
            "is_unknown": True,
            "etl_processed_at": self.etl_timestamp
        })
        
        # Add all combinations of age groups and sex
        for age_group in age_groups:
            for sex in sex_categories:
                records.append({
                    "demographic_sk": self._generate_hash_sk(age_group, sex),
                    "age_group": age_group,
                    "sex": sex,
                    "is_unknown": False,
                    "etl_processed_at": self.etl_timestamp
                })
        
        # Add totals for each age group (Persons)
        # These are already included in the loop above
        
        # Add totals for each sex (all ages)
        for sex in sex_categories:
            if sex != "Persons":  # Avoid duplicate
                records.append({
                    "demographic_sk": self._generate_hash_sk("All ages", sex),
                    "age_group": "All ages",
                    "sex": sex,
                    "is_unknown": False,
                    "etl_processed_at": self.etl_timestamp
                })
        
        # Add overall total
        records.append({
            "demographic_sk": self._generate_hash_sk("All ages", "Persons"),
            "age_group": "All ages",
            "sex": "Persons",
            "is_unknown": False,
            "etl_processed_at": self.etl_timestamp
        })
        
        # Create DataFrame and enforce schema
        df = pl.DataFrame(records)
        df = self._enforce_schema(df, "dim_demographic")
        
        # Remove duplicates if any
        df = df.unique(subset=["demographic_sk"])
        
        self.logger.info(f"Demographic dimension complete: {len(df)} records")
        
        return df


class PersonCharacteristicDimensionBuilder(DimensionBuilder):
    """Builds the person characteristic dimension."""
    
    def build(self) -> pl.DataFrame:
        """Build the person characteristic dimension.
        
        Returns:
            Polars DataFrame containing person characteristic dimension
        """
        self.logger.info("Building person characteristic dimension")
        
        # Get characteristic type definitions
        characteristic_types = self.settings.get_characteristic_types()
        
        records = []
        
        # Add unknown member
        records.append({
            "characteristic_sk": self._generate_hash_sk("UNKNOWN", "UNKNOWN"),
            "characteristic_type": "Unknown",
            "characteristic_value": "Unknown",
            "characteristic_category": "Unknown",
            "is_unknown": True,
            "etl_processed_at": self.etl_timestamp
        })
        
        # Process each characteristic type
        for char_type, char_config in characteristic_types.items():
            category = char_config.get("category", "Other")
            values = char_config.get("values", [])
            
            # Convert type name to readable format
            type_name = char_type.replace("_", " ").title()
            
            for value in values:
                records.append({
                    "characteristic_sk": self._generate_hash_sk(type_name, value),
                    "characteristic_type": type_name,
                    "characteristic_value": value,
                    "characteristic_category": category,
                    "is_unknown": False,
                    "etl_processed_at": self.etl_timestamp
                })
        
        # Create DataFrame and enforce schema
        df = pl.DataFrame(records)
        df = self._enforce_schema(df, "dim_person_characteristic")
        
        self.logger.info(f"Person characteristic dimension complete: {len(df)} records")
        
        return df