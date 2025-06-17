"""
Schema Evolution and Validation Testing

Comprehensive testing suite for schema validation framework including:
- Schema drift detection across Bronze-Silver-Gold layers
- Column type enforcement and validation
- Required field presence validation
- Data lineage and provenance tracking
- Schema versioning and backward compatibility
- Cross-dataset schema consistency validation

This test suite ensures schema integrity throughout the data processing pipeline
and detects any changes that might affect data quality or processing.
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, patch

import polars as pl
import numpy as np
from loguru import logger

from tests.data_quality.validators.schema_validators import (
    SchemaValidator, 
    DataLineageTracker,
    SchemaCompatibility,
    SchemaChangeType
)


class TestSchemaValidation:
    """Test suite for schema validation and evolution."""
    
    @pytest.fixture
    def temp_registry_path(self):
        """Create temporary schema registry path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "schema_registry"
    
    @pytest.fixture
    def temp_lineage_path(self):
        """Create temporary lineage tracking path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "lineage"
    
    @pytest.fixture
    def schema_validator(self, temp_registry_path):
        """Create schema validator with temporary registry."""
        return SchemaValidator(temp_registry_path)
    
    @pytest.fixture
    def lineage_tracker(self, temp_lineage_path):
        """Create lineage tracker with temporary path."""
        return DataLineageTracker(temp_lineage_path)
    
    @pytest.fixture
    def bronze_seifa_df(self):
        """Bronze layer SEIFA DataFrame."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "sa2_name_2021": ["Sydney", "Melbourne", "Brisbane"],
            "state_name_2021": ["NSW", "VIC", "QLD"],
            "irsd_score": [1050, 950, 1100],
            "irsd_decile": [8, 5, 9],
            "irsad_score": [1080, 920, 1120],
            "irsad_decile": [7, 4, 8],
            "ier_score": [1000, 900, 1050],
            "ier_decile": [6, 3, 7],
            "ieo_score": [1150, 850, 1180],
            "ieo_decile": [9, 2, 10],
            "usual_resident_population": [15000, 12000, 18000],
            "source_file": ["seifa_2021.xlsx", "seifa_2021.xlsx", "seifa_2021.xlsx"],
            "extraction_timestamp": ["2023-01-01", "2023-01-01", "2023-01-01"],
        })
    
    @pytest.fixture
    def silver_seifa_df(self):
        """Silver layer SEIFA DataFrame (cleaned and validated)."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "sa2_name_2021": ["Sydney", "Melbourne", "Brisbane"],
            "state_name_2021": ["NSW", "VIC", "QLD"],
            "irsd_score": [1050, 950, 1100],
            "irsd_decile": [8, 5, 9],
            "irsad_score": [1080, 920, 1120],
            "irsad_decile": [7, 4, 8],
            "ier_score": [1000, 900, 1050],
            "ier_decile": [6, 3, 7],
            "ieo_score": [1150, 850, 1180],
            "ieo_decile": [9, 2, 10],
            "usual_resident_population": [15000, 12000, 18000],
            "quality_score": [95.5, 92.3, 97.8],  # Added quality metric
            "validation_timestamp": ["2023-01-01T10:00:00", "2023-01-01T10:00:00", "2023-01-01T10:00:00"],
        })
    
    @pytest.fixture
    def gold_seifa_df(self):
        """Gold layer SEIFA DataFrame (aggregated and business-ready)."""
        return pl.DataFrame({
            "sa2_code_2021": ["101021007", "201011001", "301011002"],
            "sa2_name_2021": ["Sydney", "Melbourne", "Brisbane"],
            "state_name_2021": ["NSW", "VIC", "QLD"],
            "seifa_composite_score": [1070.0, 930.0, 1112.5],  # Composite of all SEIFA indices
            "seifa_composite_decile": [8, 4, 9],
            "disadvantage_category": ["Low", "High", "Low"],  # Business classification
            "usual_resident_population": [15000, 12000, 18000],
            "last_updated": ["2023-01-01T12:00:00", "2023-01-01T12:00:00", "2023-01-01T12:00:00"],
        })
    
    def test_schema_extraction_basic(self, schema_validator, bronze_seifa_df):
        """Test basic schema extraction from DataFrame."""
        schema = schema_validator.extract_schema(bronze_seifa_df, "bronze_seifa")
        
        # Validate schema structure
        assert "dataset_name" in schema
        assert "extraction_timestamp" in schema
        assert "row_count" in schema
        assert "column_count" in schema
        assert "columns" in schema
        assert "schema_hash" in schema
        
        # Validate basic metrics
        assert schema["dataset_name"] == "bronze_seifa"
        assert schema["row_count"] == 3
        assert schema["column_count"] == bronze_seifa_df.width
        
        # Validate column information
        assert "sa2_code_2021" in schema["columns"]
        assert "irsd_score" in schema["columns"]
        
        # Check column details
        sa2_col = schema["columns"]["sa2_code_2021"]
        assert "data_type" in sa2_col
        assert "null_count" in sa2_col
        assert "null_percentage" in sa2_col
        assert "unique_count" in sa2_col
        
        # Check numeric column statistics
        irsd_col = schema["columns"]["irsd_score"]
        assert "min_value" in irsd_col
        assert "max_value" in irsd_col
        assert "mean_value" in irsd_col
        
        # Check string column statistics
        name_col = schema["columns"]["sa2_name_2021"]
        assert "min_length" in name_col
        assert "max_length" in name_col
        assert "avg_length" in name_col
    
    def test_schema_comparison_identical(self, schema_validator, bronze_seifa_df):
        """Test schema comparison with identical schemas."""
        schema1 = schema_validator.extract_schema(bronze_seifa_df, "seifa_v1")
        schema2 = schema_validator.extract_schema(bronze_seifa_df, "seifa_v2")
        
        comparison = schema_validator.compare_schemas(schema1, schema2)
        
        assert comparison["identical"] is True
        assert comparison["compatibility"] == SchemaCompatibility.COMPATIBLE
        assert len(comparison["changes"]) == 0
        assert comparison["summary"]["columns_added"] == 0
        assert comparison["summary"]["columns_removed"] == 0
        assert comparison["summary"]["type_changes"] == 0
    
    def test_schema_comparison_column_added(self, schema_validator, bronze_seifa_df, silver_seifa_df):
        """Test schema comparison with column additions."""
        bronze_schema = schema_validator.extract_schema(bronze_seifa_df, "bronze_seifa")
        silver_schema = schema_validator.extract_schema(silver_seifa_df, "silver_seifa")
        
        comparison = schema_validator.compare_schemas(bronze_schema, silver_schema)
        
        assert comparison["identical"] is False
        assert comparison["compatibility"] == SchemaCompatibility.FORWARD_COMPATIBLE
        assert comparison["summary"]["columns_added"] > 0
        assert comparison["summary"]["columns_removed"] > 0  # Some bronze columns removed
        
        # Check for specific added columns
        added_changes = [change for change in comparison["changes"] 
                        if change["type"] == SchemaChangeType.COLUMN_ADDED.value]
        added_column_names = [change["column"] for change in added_changes]
        
        assert "quality_score" in added_column_names
        assert "validation_timestamp" in added_column_names
    
    def test_schema_comparison_major_transformation(self, schema_validator, silver_seifa_df, gold_seifa_df):
        """Test schema comparison with major transformations (Silver to Gold)."""
        silver_schema = schema_validator.extract_schema(silver_seifa_df, "silver_seifa")
        gold_schema = schema_validator.extract_schema(gold_seifa_df, "gold_seifa")
        
        comparison = schema_validator.compare_schemas(silver_schema, gold_schema)
        
        assert comparison["identical"] is False
        # Major transformations may be incompatible due to removed columns
        assert comparison["compatibility"] in [SchemaCompatibility.INCOMPATIBLE, SchemaCompatibility.FORWARD_COMPATIBLE]
        
        # Should have both additions and removals
        assert comparison["summary"]["columns_added"] > 0
        assert comparison["summary"]["columns_removed"] > 0
        
        # Check for specific transformations
        changes_by_type = {}
        for change in comparison["changes"]:
            change_type = change["type"]
            if change_type not in changes_by_type:
                changes_by_type[change_type] = []
            changes_by_type[change_type].append(change)
        
        # Should have added composite columns
        if SchemaChangeType.COLUMN_ADDED.value in changes_by_type:
            added_columns = [change["column"] for change in changes_by_type[SchemaChangeType.COLUMN_ADDED.value]]
            assert "seifa_composite_score" in added_columns
            assert "disadvantage_category" in added_columns
    
    def test_schema_validation_against_expected(self, schema_validator, bronze_seifa_df):
        """Test schema validation against expected schema definition."""
        # Extract actual schema
        actual_schema = schema_validator.extract_schema(bronze_seifa_df, "bronze_seifa")
        
        # Define expected schema
        expected_schema = {
            "columns": {
                "sa2_code_2021": {"data_type": "String"},
                "sa2_name_2021": {"data_type": "String"},
                "irsd_score": {"data_type": "Int64"},
                "irsd_decile": {"data_type": "Int64"},
                "usual_resident_population": {"data_type": "Int64"},
            },
            "constraints": {
                "sa2_code_2021": {"not_null": True, "unique": True},
                "irsd_score": {"min_value": 800, "max_value": 1200},
                "irsd_decile": {"min_value": 1, "max_value": 10},
                "usual_resident_population": {"min_value": 0},
            }
        }
        
        validation_result = schema_validator.validate_schema_against_expected(
            actual_schema, expected_schema
        )
        
        # Check validation structure
        assert "valid" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result
        assert "missing_columns" in validation_result
        assert "extra_columns" in validation_result
        assert "type_mismatches" in validation_result
        assert "constraint_violations" in validation_result
        
        # Should have extra columns (more than expected)
        assert len(validation_result["extra_columns"]) > 0
        
        # Should not have missing required columns
        assert len(validation_result["missing_columns"]) == 0
        
        # Type validation depends on actual Polars types
        if validation_result["type_mismatches"]:
            logger.info(f"Type mismatches found: {validation_result['type_mismatches']}")
    
    def test_schema_registry_operations(self, schema_validator, bronze_seifa_df):
        """Test schema registry registration and retrieval."""
        # Extract schema
        schema = schema_validator.extract_schema(bronze_seifa_df, "test_dataset")
        
        # Register schema
        version = schema_validator.register_schema(schema, "test_dataset", "bronze")
        
        assert version is not None
        assert "test_dataset_bronze_" in version
        
        # Load schema back
        loaded_schema = schema_validator.load_schema(version)
        
        assert loaded_schema is not None
        assert loaded_schema["version"] == version
        assert loaded_schema["dataset_name"] == "test_dataset"
        assert loaded_schema["layer"] == "bronze"
        assert "registration_timestamp" in loaded_schema
        
        # Schema content should match
        assert loaded_schema["schema_hash"] == schema["schema_hash"]
        assert len(loaded_schema["columns"]) == len(schema["columns"])
    
    def test_schema_drift_detection_no_drift(self, schema_validator, bronze_seifa_df):
        """Test schema drift detection with no drift."""
        # Register initial schema
        schema1 = schema_validator.extract_schema(bronze_seifa_df, "test_dataset")
        version1 = schema_validator.register_schema(schema1, "test_dataset", "bronze")
        
        # Extract same schema again
        schema2 = schema_validator.extract_schema(bronze_seifa_df, "test_dataset")
        
        # Detect drift
        drift_result = schema_validator.detect_schema_drift(schema2, "test_dataset", "bronze")
        
        assert drift_result["drift_detected"] is False
        assert drift_result["drift_severity"] == "none"
        assert len(drift_result["historical_comparisons"]) == 1
        assert len(drift_result["recommendations"]) == 0
    
    def test_schema_drift_detection_with_drift(self, schema_validator, bronze_seifa_df):
        """Test schema drift detection with schema changes."""
        # Register initial schema
        schema1 = schema_validator.extract_schema(bronze_seifa_df, "test_dataset")
        version1 = schema_validator.register_schema(schema1, "test_dataset", "bronze")
        
        # Create modified DataFrame
        modified_df = bronze_seifa_df.with_columns([
            pl.lit("new_value").alias("new_column"),
            pl.col("irsd_score").cast(pl.Float64).alias("irsd_score")  # Type change
        ])
        
        # Extract modified schema
        schema2 = schema_validator.extract_schema(modified_df, "test_dataset")
        
        # Detect drift
        drift_result = schema_validator.detect_schema_drift(schema2, "test_dataset", "bronze")
        
        assert drift_result["drift_detected"] is True
        assert drift_result["drift_severity"] in ["low", "medium", "high"]
        assert len(drift_result["historical_comparisons"]) == 1
        assert len(drift_result["recommendations"]) > 0
        
        # Check that recommendations contain relevant advice
        recommendations_text = " ".join(drift_result["recommendations"])
        assert any(keyword in recommendations_text.lower() 
                  for keyword in ["column", "type", "added", "update"])
    
    def test_cross_layer_schema_consistency(self, schema_validator, bronze_seifa_df, silver_seifa_df, gold_seifa_df):
        """Test schema consistency validation across Bronze-Silver-Gold layers."""
        # Extract schemas for all layers
        bronze_schema = schema_validator.extract_schema(bronze_seifa_df, "seifa")
        silver_schema = schema_validator.extract_schema(silver_seifa_df, "seifa")
        gold_schema = schema_validator.extract_schema(gold_seifa_df, "seifa")
        
        # Validate cross-layer consistency
        validation_result = schema_validator.validate_cross_layer_consistency(
            bronze_schema, silver_schema, gold_schema
        )
        
        # Check validation structure
        assert "consistent" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result
        assert "layer_comparisons" in validation_result
        
        # Should have comparisons for both transitions
        assert "bronze_to_silver" in validation_result["layer_comparisons"]
        assert "silver_to_gold" in validation_result["layer_comparisons"]
        
        # Each comparison should have results
        bronze_silver_comp = validation_result["layer_comparisons"]["bronze_to_silver"]
        assert "compatibility" in bronze_silver_comp
        assert "changes" in bronze_silver_comp
        
        silver_gold_comp = validation_result["layer_comparisons"]["silver_to_gold"]
        assert "compatibility" in silver_gold_comp
        assert "changes" in silver_gold_comp
        
        # Log results for analysis
        logger.info(f"Cross-layer validation consistent: {validation_result['consistent']}")
        logger.info(f"Bronze->Silver compatibility: {bronze_silver_comp['compatibility']}")
        logger.info(f"Silver->Gold compatibility: {silver_gold_comp['compatibility']}")
    
    def test_data_lineage_tracking_basic(self, lineage_tracker, bronze_seifa_df, silver_seifa_df):
        """Test basic data lineage tracking."""
        # Extract schemas
        bronze_schema = {"columns": {"col1": {"data_type": "String"}}, "schema_hash": "abc123"}
        silver_schema = {"columns": {"col1": {"data_type": "String"}, "col2": {"data_type": "Int64"}}, "schema_hash": "def456"}
        
        # Record transformation
        transformation_details = {
            "transformation_type": "bronze_to_silver",
            "operations": ["validation", "quality_scoring"],
            "timestamp": datetime.now().isoformat()
        }
        
        record_id = lineage_tracker.record_transformation(
            source_dataset="bronze_seifa",
            target_dataset="silver_seifa",
            transformation_type="bronze_to_silver",
            transformation_details=transformation_details,
            schema_before=bronze_schema,
            schema_after=silver_schema
        )
        
        assert record_id is not None
        assert record_id.startswith("lineage_")
        
        # Retrieve lineage
        lineage_records = lineage_tracker.get_dataset_lineage("bronze_seifa")
        
        assert len(lineage_records) == 1
        assert lineage_records[0]["record_id"] == record_id
        assert lineage_records[0]["source_dataset"] == "bronze_seifa"
        assert lineage_records[0]["target_dataset"] == "silver_seifa"
        assert lineage_records[0]["transformation_type"] == "bronze_to_silver"
    
    def test_data_lineage_integrity_validation(self, lineage_tracker):
        """Test data lineage integrity validation."""
        # Create a chain of transformations
        schemas = [
            {"columns": {"col1": {"data_type": "String"}}, "schema_hash": f"hash{i}"}
            for i in range(4)
        ]
        
        transformations = [
            ("raw_data", "bronze_data", "extraction"),
            ("bronze_data", "silver_data", "validation"),
            ("silver_data", "gold_data", "aggregation"),
        ]
        
        # Record transformations
        for i, (source, target, trans_type) in enumerate(transformations):
            lineage_tracker.record_transformation(
                source_dataset=source,
                target_dataset=target,
                transformation_type=trans_type,
                transformation_details={"step": i},
                schema_before=schemas[i],
                schema_after=schemas[i + 1]
            )
        
        # Validate lineage integrity
        validation_result = lineage_tracker.validate_lineage_integrity("gold_data")
        
        assert "valid" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result
        assert "lineage_gaps" in validation_result
        assert "schema_inconsistencies" in validation_result
        
        # Should be valid since we created a proper chain
        assert validation_result["valid"] is True
        assert len(validation_result["lineage_gaps"]) == 0
    
    def test_data_lineage_gap_detection(self, lineage_tracker):
        """Test detection of gaps in data lineage."""
        # Create transformations with a gap
        schema1 = {"columns": {"col1": {"data_type": "String"}}, "schema_hash": "hash1"}
        schema2 = {"columns": {"col1": {"data_type": "String"}}, "schema_hash": "hash2"}
        schema3 = {"columns": {"col1": {"data_type": "String"}}, "schema_hash": "hash3"}
        
        # Record first transformation
        lineage_tracker.record_transformation(
            source_dataset="raw_data",
            target_dataset="bronze_data",
            transformation_type="extraction",
            transformation_details={"step": 1},
            schema_before=schema1,
            schema_after=schema2
        )
        
        # Skip silver layer - create gap
        lineage_tracker.record_transformation(
            source_dataset="silver_data",  # Gap: bronze_data -> silver_data missing
            target_dataset="gold_data",
            transformation_type="aggregation",
            transformation_details={"step": 3},
            schema_before=schema2,
            schema_after=schema3
        )
        
        # Validate lineage integrity
        validation_result = lineage_tracker.validate_lineage_integrity("gold_data")
        
        # Should detect the gap
        assert validation_result["valid"] is False
        assert len(validation_result["lineage_gaps"]) > 0
        
        # Check gap details
        gap = validation_result["lineage_gaps"][0]
        assert "missing_link" in gap
        assert "bronze_data -> silver_data" in gap["missing_link"]
    
    def test_schema_versioning_compatibility(self, schema_validator, bronze_seifa_df):
        """Test schema versioning and backward compatibility."""
        # Register multiple versions
        schema_v1 = schema_validator.extract_schema(bronze_seifa_df, "test_dataset")
        version_v1 = schema_validator.register_schema(schema_v1, "test_dataset", "bronze")
        
        # Create v2 with additional column
        df_v2 = bronze_seifa_df.with_columns([
            pl.lit("new_value").alias("new_column")
        ])
        schema_v2 = schema_validator.extract_schema(df_v2, "test_dataset")
        version_v2 = schema_validator.register_schema(schema_v2, "test_dataset", "bronze")
        
        # Create v3 with removed column (breaking change)
        df_v3 = bronze_seifa_df.drop("irsd_score")
        schema_v3 = schema_validator.extract_schema(df_v3, "test_dataset")
        version_v3 = schema_validator.register_schema(schema_v3, "test_dataset", "bronze")
        
        # Load and compare versions
        loaded_v1 = schema_validator.load_schema(version_v1)
        loaded_v2 = schema_validator.load_schema(version_v2)
        loaded_v3 = schema_validator.load_schema(version_v3)
        
        # v1 to v2 should be forward compatible (addition only)
        comp_v1_v2 = schema_validator.compare_schemas(loaded_v1, loaded_v2)
        assert comp_v1_v2["compatibility"] == SchemaCompatibility.FORWARD_COMPATIBLE
        
        # v1 to v3 should be incompatible (removal)
        comp_v1_v3 = schema_validator.compare_schemas(loaded_v1, loaded_v3)
        assert comp_v1_v3["compatibility"] == SchemaCompatibility.INCOMPATIBLE
        
        # v2 to v3 should be incompatible (removals)
        comp_v2_v3 = schema_validator.compare_schemas(loaded_v2, loaded_v3)
        assert comp_v2_v3["compatibility"] == SchemaCompatibility.INCOMPATIBLE
    
    def test_schema_validation_edge_cases(self, schema_validator):
        """Test schema validation with edge cases."""
        # Empty DataFrame
        empty_df = pl.DataFrame()
        empty_schema = schema_validator.extract_schema(empty_df, "empty_dataset")
        
        assert empty_schema["row_count"] == 0
        assert empty_schema["column_count"] == 0
        assert len(empty_schema["columns"]) == 0
        
        # DataFrame with null values
        null_df = pl.DataFrame({
            "col_with_nulls": [1, None, 3],
            "col_all_nulls": [None, None, None],
            "col_no_nulls": [1, 2, 3]
        })
        null_schema = schema_validator.extract_schema(null_df, "null_dataset")
        
        assert null_schema["columns"]["col_with_nulls"]["null_count"] == 1
        assert null_schema["columns"]["col_with_nulls"]["null_percentage"] == pytest.approx(33.33, rel=1e-2)
        assert null_schema["columns"]["col_all_nulls"]["null_count"] == 3
        assert null_schema["columns"]["col_all_nulls"]["null_percentage"] == 100.0
        assert null_schema["columns"]["col_no_nulls"]["null_count"] == 0
        assert null_schema["columns"]["col_no_nulls"]["null_percentage"] == 0.0
        
        # DataFrame with mixed types
        mixed_df = pl.DataFrame({
            "integers": [1, 2, 3],
            "floats": [1.1, 2.2, 3.3],
            "strings": ["a", "bb", "ccc"],
            "booleans": [True, False, True],
            "dates": ["2023-01-01", "2023-01-02", "2023-01-03"]
        })
        mixed_schema = schema_validator.extract_schema(mixed_df, "mixed_dataset")
        
        # Should extract different statistics for different types
        assert "min_value" in mixed_schema["columns"]["integers"]
        assert "min_value" in mixed_schema["columns"]["floats"]
        assert "min_length" in mixed_schema["columns"]["strings"]
        assert "unique_count" in mixed_schema["columns"]["booleans"]
    
    def test_end_to_end_schema_pipeline_validation(self, schema_validator, lineage_tracker, 
                                                  bronze_seifa_df, silver_seifa_df, gold_seifa_df):
        """End-to-end test of complete schema validation pipeline."""
        dataset_name = "seifa_pipeline_test"
        
        # Step 1: Extract and register schemas for all layers
        bronze_schema = schema_validator.extract_schema(bronze_seifa_df, dataset_name)
        bronze_version = schema_validator.register_schema(bronze_schema, dataset_name, "bronze")
        
        silver_schema = schema_validator.extract_schema(silver_seifa_df, dataset_name)
        silver_version = schema_validator.register_schema(silver_schema, dataset_name, "silver")
        
        gold_schema = schema_validator.extract_schema(gold_seifa_df, dataset_name)
        gold_version = schema_validator.register_schema(gold_schema, dataset_name, "gold")
        
        # Step 2: Record lineage for transformations
        bronze_to_silver_id = lineage_tracker.record_transformation(
            source_dataset=f"{dataset_name}_bronze",
            target_dataset=f"{dataset_name}_silver",
            transformation_type="bronze_to_silver_validation",
            transformation_details={
                "operations": ["data_validation", "quality_scoring", "null_handling"],
                "timestamp": datetime.now().isoformat()
            },
            schema_before=bronze_schema,
            schema_after=silver_schema
        )
        
        silver_to_gold_id = lineage_tracker.record_transformation(
            source_dataset=f"{dataset_name}_silver",
            target_dataset=f"{dataset_name}_gold",
            transformation_type="silver_to_gold_aggregation",
            transformation_details={
                "operations": ["seifa_composite_calculation", "business_categorization"],
                "timestamp": datetime.now().isoformat()
            },
            schema_before=silver_schema,
            schema_after=gold_schema
        )
        
        # Step 3: Validate cross-layer consistency
        consistency_result = schema_validator.validate_cross_layer_consistency(
            bronze_schema, silver_schema, gold_schema
        )
        
        # Step 4: Validate lineage integrity
        lineage_integrity = lineage_tracker.validate_lineage_integrity(f"{dataset_name}_gold")
        
        # Step 5: Detect any schema drift (simulate new data)
        new_bronze_df = bronze_seifa_df.with_columns([
            pl.lit("2023-06-01").alias("data_refresh_date")  # New column
        ])
        new_bronze_schema = schema_validator.extract_schema(new_bronze_df, dataset_name)
        drift_result = schema_validator.detect_schema_drift(new_bronze_schema, dataset_name, "bronze")
        
        # Validate end-to-end results
        logger.info("End-to-end schema validation results:")
        logger.info(f"  Bronze version: {bronze_version}")
        logger.info(f"  Silver version: {silver_version}")
        logger.info(f"  Gold version: {gold_version}")
        logger.info(f"  Lineage records: {bronze_to_silver_id}, {silver_to_gold_id}")
        logger.info(f"  Cross-layer consistent: {consistency_result['consistent']}")
        logger.info(f"  Lineage integrity valid: {lineage_integrity['valid']}")
        logger.info(f"  Schema drift detected: {drift_result['drift_detected']}")
        
        # Assertions for end-to-end validation
        assert bronze_version is not None
        assert silver_version is not None
        assert gold_version is not None
        assert bronze_to_silver_id is not None
        assert silver_to_gold_id is not None
        
        # Lineage should be valid
        assert lineage_integrity["valid"] is True
        
        # Drift should be detected (new column added)
        assert drift_result["drift_detected"] is True
        assert drift_result["drift_severity"] in ["low", "medium"]  # Adding column is not severe
        
        # Should have recommendations for drift
        assert len(drift_result["recommendations"]) > 0
        
        # Cross-layer validation may have warnings but should capture all transformations
        assert "layer_comparisons" in consistency_result
        assert len(consistency_result["layer_comparisons"]) == 2


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])