"""
Schema Validation Utilities

Comprehensive schema validation framework for Australian health data including:
- Schema drift detection across Bronze-Silver-Gold layers
- Column type enforcement and validation
- Required field presence validation
- Data lineage and provenance tracking
- Schema versioning and backward compatibility
- Cross-dataset schema consistency validation

This module ensures schema integrity throughout the data processing pipeline
and detects any changes that might affect data quality or processing.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import hashlib

import polars as pl
from loguru import logger


class SchemaCompatibility(Enum):
    """Schema compatibility levels."""
    COMPATIBLE = "compatible"
    BACKWARD_COMPATIBLE = "backward_compatible"
    FORWARD_COMPATIBLE = "forward_compatible"
    INCOMPATIBLE = "incompatible"


class SchemaChangeType(Enum):
    """Types of schema changes."""
    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_RENAMED = "column_renamed"
    TYPE_CHANGED = "type_changed"
    CONSTRAINT_ADDED = "constraint_added"
    CONSTRAINT_REMOVED = "constraint_removed"
    INDEX_ADDED = "index_added"
    INDEX_REMOVED = "index_removed"


class SchemaValidator:
    """Core schema validation and comparison utilities."""
    
    def __init__(self, schema_registry_path: Optional[Path] = None):
        """
        Initialize schema validator.
        
        Args:
            schema_registry_path: Path to schema registry directory
        """
        self.schema_registry_path = schema_registry_path or Path("data/metadata/schemas")
        self.schema_registry_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(component="schema_validator")
    
    def extract_schema(self, df: pl.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Extract comprehensive schema information from DataFrame.
        
        Args:
            df: Polars DataFrame
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary containing schema information
        """
        schema_info = {
            "dataset_name": dataset_name,
            "extraction_timestamp": datetime.now().isoformat(),
            "row_count": df.height,
            "column_count": df.width,
            "columns": {},
            "constraints": {},
            "metadata": {}
        }
        
        # Extract column information
        for col_name in df.columns:
            col_data = df[col_name]
            
            schema_info["columns"][col_name] = {
                "data_type": str(col_data.dtype),
                "null_count": col_data.null_count(),
                "null_percentage": (col_data.null_count() / df.height) * 100,
                "unique_count": col_data.n_unique(),
                "unique_percentage": (col_data.n_unique() / df.height) * 100,
            }
            
            # Add type-specific statistics
            if col_data.dtype.is_numeric():
                non_null_data = col_data.drop_nulls()
                if len(non_null_data) > 0:
                    schema_info["columns"][col_name].update({
                        "min_value": non_null_data.min(),
                        "max_value": non_null_data.max(),
                        "mean_value": non_null_data.mean(),
                        "std_dev": non_null_data.std()
                    })
            elif col_data.dtype == pl.Utf8:
                non_null_data = col_data.drop_nulls()
                if len(non_null_data) > 0:
                    lengths = non_null_data.str.len_chars()
                    schema_info["columns"][col_name].update({
                        "min_length": lengths.min(),
                        "max_length": lengths.max(),
                        "avg_length": lengths.mean()
                    })
        
        # Calculate schema hash for change detection
        schema_info["schema_hash"] = self._calculate_schema_hash(schema_info["columns"])
        
        return schema_info
    
    def validate_schema_against_expected(self, 
                                       actual_schema: Dict[str, Any], 
                                       expected_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate actual schema against expected schema.
        
        Args:
            actual_schema: Actual schema extracted from data
            expected_schema: Expected schema definition
            
        Returns:
            Validation results with errors and warnings
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "missing_columns": [],
            "extra_columns": [],
            "type_mismatches": [],
            "constraint_violations": []
        }
        
        actual_columns = set(actual_schema["columns"].keys())
        expected_columns = set(expected_schema.get("columns", {}).keys())
        
        # Check for missing required columns
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            validation_result["missing_columns"] = list(missing_columns)
            validation_result["errors"].append(f"Missing required columns: {missing_columns}")
            validation_result["valid"] = False
        
        # Check for extra columns
        extra_columns = actual_columns - expected_columns
        if extra_columns:
            validation_result["extra_columns"] = list(extra_columns)
            validation_result["warnings"].append(f"Extra columns found: {extra_columns}")
        
        # Check column types
        for col_name in actual_columns & expected_columns:
            actual_type = actual_schema["columns"][col_name]["data_type"]
            expected_type = expected_schema["columns"][col_name].get("data_type")
            
            if expected_type and actual_type != expected_type:
                validation_result["type_mismatches"].append({
                    "column": col_name,
                    "expected": expected_type,
                    "actual": actual_type
                })
                validation_result["errors"].append(
                    f"Type mismatch in {col_name}: expected {expected_type}, got {actual_type}"
                )
                validation_result["valid"] = False
        
        # Check constraints
        for col_name, constraints in expected_schema.get("constraints", {}).items():
            if col_name in actual_schema["columns"]:
                col_info = actual_schema["columns"][col_name]
                
                # Check null constraints
                if constraints.get("not_null", False) and col_info["null_count"] > 0:
                    validation_result["constraint_violations"].append({
                        "column": col_name,
                        "constraint": "not_null",
                        "violation": f"{col_info['null_count']} null values found"
                    })
                    validation_result["errors"].append(
                        f"Null values found in {col_name} (not_null constraint violated)"
                    )
                    validation_result["valid"] = False
                
                # Check unique constraints
                if constraints.get("unique", False) and col_info["unique_percentage"] < 100:
                    validation_result["constraint_violations"].append({
                        "column": col_name,
                        "constraint": "unique",
                        "violation": f"Only {col_info['unique_percentage']:.2f}% unique values"
                    })
                    validation_result["errors"].append(
                        f"Duplicate values found in {col_name} (unique constraint violated)"
                    )
                    validation_result["valid"] = False
                
                # Check range constraints
                if "min_value" in constraints and "min_value" in col_info:
                    if col_info["min_value"] < constraints["min_value"]:
                        validation_result["constraint_violations"].append({
                            "column": col_name,
                            "constraint": "min_value",
                            "violation": f"Minimum value {col_info['min_value']} < {constraints['min_value']}"
                        })
                        validation_result["errors"].append(
                            f"Minimum value constraint violated in {col_name}"
                        )
                        validation_result["valid"] = False
                
                if "max_value" in constraints and "max_value" in col_info:
                    if col_info["max_value"] > constraints["max_value"]:
                        validation_result["constraint_violations"].append({
                            "column": col_name,
                            "constraint": "max_value",
                            "violation": f"Maximum value {col_info['max_value']} > {constraints['max_value']}"
                        })
                        validation_result["errors"].append(
                            f"Maximum value constraint violated in {col_name}"
                        )
                        validation_result["valid"] = False
        
        return validation_result
    
    def compare_schemas(self, schema1: Dict[str, Any], schema2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two schemas and identify differences.
        
        Args:
            schema1: First schema (often older version)
            schema2: Second schema (often newer version)
            
        Returns:
            Dictionary containing comparison results
        """
        comparison_result = {
            "identical": False,
            "compatibility": SchemaCompatibility.COMPATIBLE,
            "changes": [],
            "summary": {
                "columns_added": 0,
                "columns_removed": 0,
                "columns_modified": 0,
                "type_changes": 0
            }
        }
        
        # Check if schemas are identical
        if schema1.get("schema_hash") == schema2.get("schema_hash"):
            comparison_result["identical"] = True
            return comparison_result
        
        columns1 = set(schema1.get("columns", {}).keys())
        columns2 = set(schema2.get("columns", {}).keys())
        
        # Columns added
        added_columns = columns2 - columns1
        for col in added_columns:
            comparison_result["changes"].append({
                "type": SchemaChangeType.COLUMN_ADDED.value,
                "column": col,
                "details": schema2["columns"][col]
            })
            comparison_result["summary"]["columns_added"] += 1
        
        # Columns removed
        removed_columns = columns1 - columns2
        for col in removed_columns:
            comparison_result["changes"].append({
                "type": SchemaChangeType.COLUMN_REMOVED.value,
                "column": col,
                "details": schema1["columns"][col]
            })
            comparison_result["summary"]["columns_removed"] += 1
        
        # Columns modified
        common_columns = columns1 & columns2
        for col in common_columns:
            col1_info = schema1["columns"][col]
            col2_info = schema2["columns"][col]
            
            # Check type changes
            if col1_info["data_type"] != col2_info["data_type"]:
                comparison_result["changes"].append({
                    "type": SchemaChangeType.TYPE_CHANGED.value,
                    "column": col,
                    "old_type": col1_info["data_type"],
                    "new_type": col2_info["data_type"]
                })
                comparison_result["summary"]["type_changes"] += 1
                comparison_result["summary"]["columns_modified"] += 1
        
        # Determine compatibility
        compatibility = self._determine_compatibility(comparison_result["changes"])
        comparison_result["compatibility"] = compatibility
        
        return comparison_result
    
    def detect_schema_drift(self, 
                          current_schema: Dict[str, Any], 
                          dataset_name: str,
                          layer: str = "bronze") -> Dict[str, Any]:
        """
        Detect schema drift by comparing current schema with historical versions.
        
        Args:
            current_schema: Current schema to check
            dataset_name: Name of the dataset
            layer: Data layer (bronze, silver, gold)
            
        Returns:
            Schema drift detection results
        """
        drift_result = {
            "drift_detected": False,
            "drift_severity": "none",
            "historical_comparisons": [],
            "recommendations": []
        }
        
        # Load historical schemas
        historical_schemas = self._load_historical_schemas(dataset_name, layer)
        
        if not historical_schemas:
            self.logger.info(f"No historical schemas found for {dataset_name}:{layer}")
            return drift_result
        
        # Compare with most recent historical schema
        latest_historical = historical_schemas[-1]
        comparison = self.compare_schemas(latest_historical, current_schema)
        
        drift_result["historical_comparisons"].append({
            "historical_version": latest_historical.get("version", "unknown"),
            "comparison": comparison
        })
        
        if not comparison["identical"]:
            drift_result["drift_detected"] = True
            
            # Determine drift severity
            total_changes = sum(comparison["summary"].values())
            if total_changes > 10 or comparison["summary"]["columns_removed"] > 0:
                drift_result["drift_severity"] = "high"
            elif total_changes > 5 or comparison["summary"]["type_changes"] > 0:
                drift_result["drift_severity"] = "medium"
            else:
                drift_result["drift_severity"] = "low"
            
            # Generate recommendations
            drift_result["recommendations"] = self._generate_drift_recommendations(comparison)
        
        return drift_result
    
    def register_schema(self, schema: Dict[str, Any], dataset_name: str, layer: str = "bronze") -> str:
        """
        Register a schema in the schema registry.
        
        Args:
            schema: Schema to register
            dataset_name: Name of the dataset
            layer: Data layer (bronze, silver, gold)
            
        Returns:
            Schema version identifier
        """
        # Create version identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version = f"{dataset_name}_{layer}_{timestamp}"
        
        # Add metadata
        schema_copy = schema.copy()
        schema_copy["version"] = version
        schema_copy["registration_timestamp"] = datetime.now().isoformat()
        schema_copy["dataset_name"] = dataset_name
        schema_copy["layer"] = layer
        
        # Save to registry
        registry_file = self.schema_registry_path / f"{version}.json"
        with open(registry_file, "w") as f:
            json.dump(schema_copy, f, indent=2, default=str)
        
        self.logger.info(f"Schema registered: {version}")
        return version
    
    def load_schema(self, version: str) -> Optional[Dict[str, Any]]:
        """
        Load a schema from the registry.
        
        Args:
            version: Schema version identifier
            
        Returns:
            Schema dictionary or None if not found
        """
        registry_file = self.schema_registry_path / f"{version}.json"
        
        if not registry_file.exists():
            return None
        
        with open(registry_file, "r") as f:
            return json.load(f)
    
    def validate_cross_layer_consistency(self, 
                                       bronze_schema: Dict[str, Any],
                                       silver_schema: Dict[str, Any],
                                       gold_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate schema consistency across Bronze-Silver-Gold layers.
        
        Args:
            bronze_schema: Bronze layer schema
            silver_schema: Silver layer schema
            gold_schema: Gold layer schema
            
        Returns:
            Cross-layer validation results
        """
        validation_result = {
            "consistent": True,
            "errors": [],
            "warnings": [],
            "layer_comparisons": {}
        }
        
        # Bronze to Silver comparison
        bronze_silver = self.compare_schemas(bronze_schema, silver_schema)
        validation_result["layer_comparisons"]["bronze_to_silver"] = bronze_silver
        
        # Silver to Gold comparison
        silver_gold = self.compare_schemas(silver_schema, gold_schema)
        validation_result["layer_comparisons"]["silver_to_gold"] = silver_gold
        
        # Check for acceptable transformations
        for layer_name, comparison in validation_result["layer_comparisons"].items():
            if comparison["compatibility"] == SchemaCompatibility.INCOMPATIBLE:
                validation_result["errors"].append(
                    f"Incompatible schema changes detected in {layer_name} transformation"
                )
                validation_result["consistent"] = False
            
            # Warn about unexpected column removals
            removed_columns = comparison["summary"]["columns_removed"]
            if removed_columns > 0:
                validation_result["warnings"].append(
                    f"{removed_columns} columns removed in {layer_name} transformation"
                )
        
        return validation_result
    
    def _calculate_schema_hash(self, columns: Dict[str, Any]) -> str:
        """Calculate hash of schema for change detection."""
        # Create a stable representation of the schema
        schema_repr = {}
        for col_name, col_info in sorted(columns.items()):
            schema_repr[col_name] = {
                "data_type": col_info["data_type"],
                # Don't include statistics in hash as they change with data
            }
        
        schema_str = json.dumps(schema_repr, sort_keys=True)
        return hashlib.sha256(schema_str.encode()).hexdigest()[:16]
    
    def _determine_compatibility(self, changes: List[Dict[str, Any]]) -> SchemaCompatibility:
        """Determine schema compatibility based on changes."""
        has_removals = any(change["type"] == SchemaChangeType.COLUMN_REMOVED.value for change in changes)
        has_type_changes = any(change["type"] == SchemaChangeType.TYPE_CHANGED.value for change in changes)
        has_additions = any(change["type"] == SchemaChangeType.COLUMN_ADDED.value for change in changes)
        
        if has_removals or has_type_changes:
            return SchemaCompatibility.INCOMPATIBLE
        elif has_additions:
            return SchemaCompatibility.FORWARD_COMPATIBLE
        else:
            return SchemaCompatibility.COMPATIBLE
    
    def _load_historical_schemas(self, dataset_name: str, layer: str) -> List[Dict[str, Any]]:
        """Load historical schemas for a dataset and layer."""
        historical_schemas = []
        
        # Find all schema files for this dataset and layer
        pattern = f"{dataset_name}_{layer}_*.json"
        for schema_file in self.schema_registry_path.glob(pattern):
            with open(schema_file, "r") as f:
                schema = json.load(f)
                historical_schemas.append(schema)
        
        # Sort by registration timestamp
        historical_schemas.sort(key=lambda x: x.get("registration_timestamp", ""))
        
        return historical_schemas
    
    def _generate_drift_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on schema drift."""
        recommendations = []
        
        if comparison["summary"]["columns_removed"] > 0:
            recommendations.append(
                "Critical: Column removals detected. Verify data pipeline compatibility."
            )
        
        if comparison["summary"]["type_changes"] > 0:
            recommendations.append(
                "Warning: Data type changes detected. Update downstream transformations."
            )
        
        if comparison["summary"]["columns_added"] > 0:
            recommendations.append(
                "Info: New columns added. Consider updating data quality checks."
            )
        
        return recommendations


class DataLineageTracker:
    """Track data lineage and provenance through the processing pipeline."""
    
    def __init__(self, lineage_path: Optional[Path] = None):
        """
        Initialize lineage tracker.
        
        Args:
            lineage_path: Path to store lineage information
        """
        self.lineage_path = lineage_path or Path("data/metadata/lineage")
        self.lineage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(component="data_lineage_tracker")
    
    def record_transformation(self, 
                            source_dataset: str,
                            target_dataset: str,
                            transformation_type: str,
                            transformation_details: Dict[str, Any],
                            schema_before: Dict[str, Any],
                            schema_after: Dict[str, Any]) -> str:
        """
        Record a data transformation in the lineage.
        
        Args:
            source_dataset: Source dataset identifier
            target_dataset: Target dataset identifier
            transformation_type: Type of transformation
            transformation_details: Details of the transformation
            schema_before: Schema before transformation
            schema_after: Schema after transformation
            
        Returns:
            Lineage record identifier
        """
        lineage_record = {
            "record_id": f"lineage_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "timestamp": datetime.now().isoformat(),
            "source_dataset": source_dataset,
            "target_dataset": target_dataset,
            "transformation_type": transformation_type,
            "transformation_details": transformation_details,
            "schema_before": schema_before,
            "schema_after": schema_after,
            "schema_changes": self._calculate_schema_changes(schema_before, schema_after)
        }
        
        # Save lineage record
        record_file = self.lineage_path / f"{lineage_record['record_id']}.json"
        with open(record_file, "w") as f:
            json.dump(lineage_record, f, indent=2, default=str)
        
        self.logger.info(f"Lineage recorded: {lineage_record['record_id']}")
        return lineage_record["record_id"]
    
    def get_dataset_lineage(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        Get complete lineage for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of lineage records
        """
        lineage_records = []
        
        for record_file in self.lineage_path.glob("lineage_*.json"):
            with open(record_file, "r") as f:
                record = json.load(f)
                
            if (record["source_dataset"] == dataset_name or 
                record["target_dataset"] == dataset_name):
                lineage_records.append(record)
        
        # Sort by timestamp
        lineage_records.sort(key=lambda x: x["timestamp"])
        
        return lineage_records
    
    def validate_lineage_integrity(self, dataset_name: str) -> Dict[str, Any]:
        """
        Validate lineage integrity for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Lineage integrity validation results
        """
        lineage_records = self.get_dataset_lineage(dataset_name)
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "lineage_gaps": [],
            "schema_inconsistencies": []
        }
        
        # Check for lineage gaps
        for i in range(1, len(lineage_records)):
            prev_record = lineage_records[i-1]
            curr_record = lineage_records[i]
            
            if prev_record["target_dataset"] != curr_record["source_dataset"]:
                validation_result["lineage_gaps"].append({
                    "gap_between": prev_record["record_id"],
                    "and": curr_record["record_id"],
                    "missing_link": f"{prev_record['target_dataset']} -> {curr_record['source_dataset']}"
                })
        
        # Check schema consistency
        for record in lineage_records:
            if record["schema_changes"]["has_breaking_changes"]:
                validation_result["schema_inconsistencies"].append({
                    "record_id": record["record_id"],
                    "breaking_changes": record["schema_changes"]["breaking_changes"]
                })
        
        if validation_result["lineage_gaps"] or validation_result["schema_inconsistencies"]:
            validation_result["valid"] = False
        
        return validation_result
    
    def _calculate_schema_changes(self, schema_before: Dict[str, Any], schema_after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate schema changes between two schemas."""
        validator = SchemaValidator()
        comparison = validator.compare_schemas(schema_before, schema_after)
        
        # Determine if changes are breaking
        breaking_changes = []
        for change in comparison["changes"]:
            if change["type"] in [SchemaChangeType.COLUMN_REMOVED.value, SchemaChangeType.TYPE_CHANGED.value]:
                breaking_changes.append(change)
        
        return {
            "has_changes": not comparison["identical"],
            "has_breaking_changes": len(breaking_changes) > 0,
            "breaking_changes": breaking_changes,
            "total_changes": sum(comparison["summary"].values()),
            "compatibility": comparison["compatibility"].value
        }


if __name__ == "__main__":
    # Example usage
    validator = SchemaValidator()
    
    # Create sample DataFrames
    df1 = pl.DataFrame({
        "sa2_code_2021": ["101021007", "201011001"],
        "sa2_name_2021": ["Sydney", "Melbourne"],
        "population": [15000, 12000]
    })
    
    df2 = pl.DataFrame({
        "sa2_code_2021": ["101021007", "201011001"],
        "sa2_name_2021": ["Sydney", "Melbourne"],
        "population": [15000, 12000],
        "new_column": ["A", "B"]  # Schema change
    })
    
    # Extract schemas
    schema1 = validator.extract_schema(df1, "test_dataset")
    schema2 = validator.extract_schema(df2, "test_dataset")
    
    # Compare schemas
    comparison = validator.compare_schemas(schema1, schema2)
    print(f"Schema comparison: {comparison}")
    
    # Register schemas
    version1 = validator.register_schema(schema1, "test_dataset", "bronze")
    version2 = validator.register_schema(schema2, "test_dataset", "silver")
    
    print(f"Registered schemas: {version1}, {version2}")