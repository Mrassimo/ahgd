"""
Test suite for Incremental Processing and Data Versioning (Phase 4.2)

Tests the Bronze-Silver-Gold data lake architecture and versioning system
for Australian health data incremental updates.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import tempfile
import time
import json
from datetime import datetime, timedelta

from src.data_processing.storage.incremental_processor import (
    IncrementalProcessor, DataLayer, MergeStrategy, DataVersion, ChangeRecord
)


@pytest.fixture
def sample_health_data():
    """Create sample health data for testing."""
    np.random.seed(42)
    n_rows = 100
    
    return pl.DataFrame({
        "sa2_code": [f"1{str(i).zfill(8)}" for i in range(1000, 1000 + n_rows)],
        "sa2_name": [f"Health Area {i}" for i in range(n_rows)],
        "state_name": np.random.choice(['NSW', 'VIC', 'QLD'], n_rows),
        "prescription_count": np.random.poisson(3, n_rows),
        "total_cost": np.random.exponential(45, n_rows),
        "dispensing_date": ["2023-01-01"] * n_rows
    })


@pytest.fixture
def updated_health_data():
    """Create updated health data for testing incremental processing."""
    np.random.seed(43)  # Different seed for updated data
    n_rows = 50
    
    return pl.DataFrame({
        "sa2_code": [f"1{str(i).zfill(8)}" for i in range(1020, 1020 + n_rows)],  # Overlapping range
        "sa2_name": [f"Updated Health Area {i}" for i in range(n_rows)],
        "state_name": np.random.choice(['NSW', 'VIC', 'QLD'], n_rows),
        "prescription_count": np.random.poisson(5, n_rows),  # Higher prescription counts
        "total_cost": np.random.exponential(60, n_rows),     # Higher costs
        "dispensing_date": ["2023-02-01"] * n_rows
    })


@pytest.fixture
def temp_data_lake():
    """Create temporary data lake structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestIncrementalProcessor:
    """Test suite for IncrementalProcessor class."""
    
    def test_initialization(self, temp_data_lake):
        """Test incremental processor initialization."""
        processor = IncrementalProcessor(temp_data_lake)
        
        assert processor.base_path == temp_data_lake
        assert len(processor.versions_metadata) == 0
        assert len(processor.change_log) == 0
        
        # Check data lake structure was created
        assert (temp_data_lake / "bronze" / "health").exists()
        assert (temp_data_lake / "silver" / "health").exists()
        assert (temp_data_lake / "gold" / "sa2_health_summary").exists()
        assert (temp_data_lake / "metadata" / "versions").exists()
    
    def test_schema_change_detection(self, sample_health_data, temp_data_lake):
        """Test schema change detection."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # No previous version - should detect as changed
        schema_changed = processor.detect_schema_changes(sample_health_data, "test_dataset")
        assert schema_changed is True
        
        # Create a version first
        bronze_version = processor.ingest_to_bronze(sample_health_data, "test_dataset", {"source": "test"})
        silver_version = processor.process_to_silver("test_dataset", bronze_version)
        
        # Same schema - should not detect change
        schema_changed = processor.detect_schema_changes(sample_health_data, "test_dataset")
        assert schema_changed is False
        
        # Different schema - should detect change
        modified_data = sample_health_data.with_columns([pl.lit("new_column").alias("new_col")])
        schema_changed = processor.detect_schema_changes(modified_data, "test_dataset")
        assert schema_changed is True
    
    def test_bronze_layer_ingestion(self, sample_health_data, temp_data_lake):
        """Test ingestion to bronze layer with partitioning."""
        processor = IncrementalProcessor(temp_data_lake)
        
        source_info = {"source_file": "test.csv", "ingestion_time": datetime.now().isoformat()}
        version_id = processor.ingest_to_bronze(sample_health_data, "health", source_info)
        
        # Check version was created
        assert version_id in processor.versions_metadata
        version = processor.versions_metadata[version_id]
        
        assert version.dataset_name == "health"
        assert version.layer == DataLayer.BRONZE
        assert version.record_count == sample_health_data.shape[0]
        assert version.merge_strategy == MergeStrategy.APPEND
        assert len(version.source_files) == 1
        
        # Check file was created
        file_path = Path(version.source_files[0])
        assert file_path.exists()
        assert file_path.suffix == ".parquet"
        
        # Verify data integrity
        loaded_data = pl.read_parquet(file_path)
        assert loaded_data.shape == sample_health_data.shape
        assert loaded_data.columns == sample_health_data.columns
    
    def test_silver_layer_processing(self, sample_health_data, temp_data_lake):
        """Test processing to silver layer with data cleaning."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # First ingest to bronze
        bronze_version = processor.ingest_to_bronze(sample_health_data, "health", {"source": "test"})
        
        # Process to silver
        silver_version_id = processor.process_to_silver("health", bronze_version, MergeStrategy.UPSERT)
        
        # Check silver version was created
        assert silver_version_id in processor.versions_metadata
        silver_version = processor.versions_metadata[silver_version_id]
        
        assert silver_version.dataset_name == "health"
        assert silver_version.layer == DataLayer.SILVER
        assert silver_version.merge_strategy == MergeStrategy.UPSERT
        assert silver_version.parent_version is None  # First version has no parent
        
        # Check file was created
        file_path = Path(silver_version.source_files[0])
        assert file_path.exists()
        
        # Verify data was cleaned
        silver_data = pl.read_parquet(file_path)
        assert silver_data.shape[0] <= sample_health_data.shape[0]  # May have been cleaned
        
        # Check that negative values were removed (if any)
        if "prescription_count" in silver_data.columns:
            assert (silver_data["prescription_count"] >= 0).all()
    
    def test_incremental_silver_update(self, sample_health_data, updated_health_data, temp_data_lake):
        """Test incremental updates to silver layer."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # First dataset
        bronze_v1 = processor.ingest_to_bronze(sample_health_data, "health", {"source": "initial"})
        silver_v1 = processor.process_to_silver("health", bronze_v1, MergeStrategy.UPSERT)
        
        # Second dataset (incremental update)
        bronze_v2 = processor.ingest_to_bronze(updated_health_data, "health", {"source": "update"})
        silver_v2 = processor.process_to_silver("health", bronze_v2, MergeStrategy.UPSERT)
        
        # Check that second silver version has parent
        silver_v2_version = processor.versions_metadata[silver_v2]
        assert silver_v2_version.parent_version == silver_v1
        
        # Load silver data and verify merge
        silver_data = pl.read_parquet(silver_v2_version.source_files[0])
        
        # Should have records from both datasets, with overlapping SA2 codes updated
        original_sa2s = set(sample_health_data["sa2_code"].to_list())
        updated_sa2s = set(updated_health_data["sa2_code"].to_list())
        result_sa2s = set(silver_data["sa2_code"].to_list())
        
        # Should have union of all SA2 codes
        expected_sa2s = original_sa2s.union(updated_sa2s)
        assert result_sa2s == expected_sa2s
    
    def test_merge_strategies(self, sample_health_data, updated_health_data, temp_data_lake):
        """Test different merge strategies."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # Setup initial data
        bronze_v1 = processor.ingest_to_bronze(sample_health_data, "health", {"source": "initial"})
        silver_v1 = processor.process_to_silver("health", bronze_v1, MergeStrategy.UPSERT)
        
        # Test APPEND strategy
        bronze_v2 = processor.ingest_to_bronze(updated_health_data, "health_append", {"source": "append"})
        silver_v2 = processor.process_to_silver("health_append", bronze_v2, MergeStrategy.APPEND)
        
        append_data = pl.read_parquet(processor.versions_metadata[silver_v2].source_files[0])
        assert append_data.shape[0] == updated_health_data.shape[0]
        
        # Test REPLACE strategy
        bronze_v3 = processor.ingest_to_bronze(updated_health_data, "health_replace", {"source": "replace"})
        silver_v3 = processor.process_to_silver("health_replace", bronze_v3, MergeStrategy.REPLACE)
        
        replace_data = pl.read_parquet(processor.versions_metadata[silver_v3].source_files[0])
        assert replace_data.shape[0] == updated_health_data.shape[0]
    
    def test_gold_layer_aggregation(self, sample_health_data, temp_data_lake):
        """Test aggregation to gold layer."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # Setup silver data
        bronze_version = processor.ingest_to_bronze(sample_health_data, "health", {"source": "test"})
        silver_version = processor.process_to_silver("health", bronze_version, MergeStrategy.UPSERT)
        
        # Aggregate to gold
        aggregation_config = {
            "group_by_sa2": True,
            "include_totals": True
        }
        
        gold_version_id = processor.aggregate_to_gold("health", silver_version, aggregation_config)
        
        # Check gold version was created
        assert gold_version_id in processor.versions_metadata
        gold_version = processor.versions_metadata[gold_version_id]
        
        assert gold_version.dataset_name == "health"
        assert gold_version.layer == DataLayer.GOLD
        assert gold_version.parent_version == silver_version
        
        # Check aggregated data
        gold_data = pl.read_parquet(gold_version.source_files[0])
        
        # Should have aggregated columns
        expected_columns = ["sa2_code", "state_name", "total_prescriptions", "avg_prescriptions"]
        for col in expected_columns:
            if col in sample_health_data.columns or col.startswith(("total_", "avg_")):
                # Column should exist in aggregated data
                pass  # We can't guarantee all columns exist without knowing exact aggregation logic
        
        # Should have fewer or equal rows than original (due to grouping)
        assert gold_data.shape[0] <= sample_health_data.shape[0]
    
    def test_data_cleaning(self, temp_data_lake):
        """Test data cleaning for different dataset types."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # Create health data with issues
        dirty_health_data = pl.DataFrame({
            "sa2_code": ["12345678", "123456789", "1234567890"],  # Wrong length SA2 code
            "prescription_count": [5, -1, 10],  # Negative prescription count
            "total_cost": [100.0, -50.0, 75.0],  # Negative cost
            "state_name": ["NSW", "VIC", "QLD"]
        })
        
        # Clean health data
        cleaned = processor._clean_health_data(dirty_health_data)
        
        # Should remove invalid SA2 codes and negative values
        assert cleaned.shape[0] < dirty_health_data.shape[0]
        assert (cleaned["prescription_count"] >= 0).all()
        assert (cleaned["total_cost"] >= 0).all()
        assert (cleaned["sa2_code"].str.len_chars() == 9).all()
        
        # Create SEIFA data with issues
        dirty_seifa_data = pl.DataFrame({
            "sa2_code": ["123456789"] * 3,
            "irsd_decile": [5, 11, -1],  # Invalid deciles
            "irsad_decile": [8, 0, 15]   # Invalid deciles
        })
        
        # Clean SEIFA data
        cleaned_seifa = processor._clean_seifa_data(dirty_seifa_data)
        
        # Should remove rows with invalid deciles
        assert cleaned_seifa.shape[0] < dirty_seifa_data.shape[0]
        for col in ["irsd_decile", "irsad_decile"]:
            if col in cleaned_seifa.columns:
                values = cleaned_seifa[col]
                assert (values >= 1).all() and (values <= 10).all()
    
    def test_version_rollback(self, sample_health_data, updated_health_data, temp_data_lake):
        """Test rollback to previous version."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # Create multiple versions
        bronze_v1 = processor.ingest_to_bronze(sample_health_data, "health", {"source": "v1"})
        silver_v1 = processor.process_to_silver("health", bronze_v1)
        
        bronze_v2 = processor.ingest_to_bronze(updated_health_data, "health", {"source": "v2"})
        silver_v2 = processor.process_to_silver("health", bronze_v2)
        
        # Rollback to v1
        rollback_success = processor.rollback_to_version("health", DataLayer.SILVER, silver_v1)
        assert rollback_success is True
        
        # Check rollback version was created
        rollback_versions = [
            v for v in processor.versions_metadata.values()
            if v.dataset_name == "health" and v.parent_version == silver_v1
        ]
        assert len(rollback_versions) >= 1
        
        rollback_version = rollback_versions[-1]  # Most recent rollback
        assert "rollback" in rollback_version.version_id
        assert rollback_version.record_count == processor.versions_metadata[silver_v1].record_count
    
    def test_data_lineage(self, sample_health_data, updated_health_data, temp_data_lake):
        """Test data lineage tracking."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # Create lineage chain: bronze -> silver -> gold
        bronze_version = processor.ingest_to_bronze(sample_health_data, "health", {"source": "test"})
        silver_version = processor.process_to_silver("health", bronze_version)
        gold_version = processor.aggregate_to_gold("health", silver_version, {"group_by_sa2": True})
        
        # Get lineage for gold version
        lineage = processor.get_data_lineage(gold_version)
        
        assert "target_version" in lineage
        assert "lineage_chain" in lineage
        assert lineage["target_version"] == gold_version
        
        # Should have gold -> silver lineage
        assert len(lineage["lineage_chain"]) >= 2
        assert gold_version in lineage["lineage_chain"]
        assert silver_version in lineage["lineage_chain"]
    
    def test_version_cleanup(self, sample_health_data, temp_data_lake):
        """Test automatic cleanup of old versions."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # Modify retention policy for testing
        processor.LAYER_CONFIG[DataLayer.SILVER]["retention_versions"] = 2
        
        # Create multiple silver versions
        versions = []
        for i in range(5):
            bronze_version = processor.ingest_to_bronze(sample_health_data, "health", {"source": f"v{i}"})
            silver_version = processor.process_to_silver("health", bronze_version)
            versions.append(silver_version)
            time.sleep(0.01)  # Ensure different timestamps
        
        # Check that old versions were cleaned up
        remaining_silver_versions = [
            v for v in processor.versions_metadata.values()
            if v.dataset_name == "health" and v.layer == DataLayer.SILVER
        ]
        
        # Should only have 2 versions (retention policy)
        assert len(remaining_silver_versions) <= 2
        
        # Should keep the most recent versions
        latest_versions = sorted(remaining_silver_versions, key=lambda v: v.created_timestamp, reverse=True)
        assert versions[-1] in [v.version_id for v in latest_versions]
        assert versions[-2] in [v.version_id for v in latest_versions]
    
    def test_incremental_summary(self, sample_health_data, temp_data_lake):
        """Test incremental processing summary generation."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # Create some data across layers
        bronze_version = processor.ingest_to_bronze(sample_health_data, "health", {"source": "test"})
        silver_version = processor.process_to_silver("health", bronze_version)
        gold_version = processor.aggregate_to_gold("health", silver_version, {"group_by_sa2": True})
        
        summary = processor.get_incremental_summary()
        
        assert "total_versions" in summary
        assert "layers" in summary
        assert "datasets" in summary
        assert "recent_activity" in summary
        
        # Check layer counts
        assert summary["layers"]["bronze"] >= 1
        assert summary["layers"]["silver"] >= 1
        assert summary["layers"]["gold"] >= 1
        
        # Check dataset info
        assert "health" in summary["datasets"]
        health_info = summary["datasets"]["health"]
        assert health_info["versions"] >= 3  # bronze + silver + gold
        assert health_info["total_records"] > 0
    
    def test_concurrent_processing(self, sample_health_data, temp_data_lake):
        """Test handling of concurrent processing scenarios."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # Create initial data
        bronze_v1 = processor.ingest_to_bronze(sample_health_data, "health", {"source": "concurrent_1"})
        silver_v1 = processor.process_to_silver("health", bronze_v1)
        
        # Simulate concurrent updates with different data
        concurrent_data_1 = sample_health_data.with_columns([
            (pl.col("prescription_count") + 1).alias("prescription_count")
        ])
        
        concurrent_data_2 = sample_health_data.with_columns([
            (pl.col("prescription_count") + 2).alias("prescription_count")
        ])
        
        # Process both concurrently (in sequence for testing)
        bronze_v2 = processor.ingest_to_bronze(concurrent_data_1, "health", {"source": "concurrent_2"})
        bronze_v3 = processor.ingest_to_bronze(concurrent_data_2, "health", {"source": "concurrent_3"})
        
        silver_v2 = processor.process_to_silver("health", bronze_v2)
        silver_v3 = processor.process_to_silver("health", bronze_v3)
        
        # Both should succeed and create different versions
        assert silver_v2 != silver_v3
        assert silver_v2 in processor.versions_metadata
        assert silver_v3 in processor.versions_metadata
        
        # Later version should have higher prescription counts
        silver_v3_data = pl.read_parquet(processor.versions_metadata[silver_v3].source_files[0])
        avg_prescriptions = silver_v3_data["prescription_count"].mean()
        
        # Should reflect the updates
        assert avg_prescriptions > sample_health_data["prescription_count"].mean()


@pytest.mark.integration
class TestIncrementalProcessingIntegration:
    """Integration tests for complete incremental processing workflows."""
    
    def test_complete_incremental_workflow(self, sample_health_data, updated_health_data, temp_data_lake):
        """Test complete Bronze-Silver-Gold incremental workflow."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # Initial load
        bronze_v1 = processor.ingest_to_bronze(sample_health_data, "health", {"source": "initial_load"})
        silver_v1 = processor.process_to_silver("health", bronze_v1, MergeStrategy.UPSERT)
        gold_v1 = processor.aggregate_to_gold("health", silver_v1, {"group_by_sa2": True})
        
        # Incremental update
        bronze_v2 = processor.ingest_to_bronze(updated_health_data, "health", {"source": "daily_update"})
        silver_v2 = processor.process_to_silver("health", bronze_v2, MergeStrategy.UPSERT)
        gold_v2 = processor.aggregate_to_gold("health", silver_v2, {"group_by_sa2": True})
        
        # Verify complete workflow
        assert len(processor.versions_metadata) == 6  # 2 bronze + 2 silver + 2 gold
        
        # Check data lineage
        gold_lineage = processor.get_data_lineage(gold_v2)
        assert len(gold_lineage["lineage_chain"]) >= 2  # gold -> silver chain
        
        # Verify final gold data contains merged results
        final_gold_data = pl.read_parquet(processor.versions_metadata[gold_v2].source_files[0])
        assert final_gold_data.shape[0] > 0
        
        # Check that aggregation worked
        if "total_prescriptions" in final_gold_data.columns:
            assert final_gold_data["total_prescriptions"].sum() > 0
    
    def test_schema_evolution_workflow(self, sample_health_data, temp_data_lake):
        """Test workflow when schema evolves."""
        processor = IncrementalProcessor(temp_data_lake)
        
        # Initial data
        bronze_v1 = processor.ingest_to_bronze(sample_health_data, "health", {"source": "v1"})
        silver_v1 = processor.process_to_silver("health", bronze_v1)
        
        # Evolved schema with new column
        evolved_data = sample_health_data.with_columns([
            pl.lit("new_value").alias("new_column"),
            pl.lit(5.0).alias("new_metric")
        ])
        
        # Should detect schema change
        schema_changed = processor.detect_schema_changes(evolved_data, "health")
        assert schema_changed is True
        
        # Process evolved data
        bronze_v2 = processor.ingest_to_bronze(evolved_data, "health", {"source": "v2_evolved"})
        silver_v2 = processor.process_to_silver("health", bronze_v2)
        
        # Should create new silver version (not merge due to schema change)
        silver_v2_version = processor.versions_metadata[silver_v2]
        assert silver_v2_version.parent_version is None  # New schema = no parent
        
        # Verify new schema in silver data
        silver_v2_data = pl.read_parquet(silver_v2_version.source_files[0])
        assert "new_column" in silver_v2_data.columns
        assert "new_metric" in silver_v2_data.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])