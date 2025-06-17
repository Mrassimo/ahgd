"""
Comprehensive integration tests for Australian Health Analytics platform.

Tests end-to-end integration across all major components:
- Data processing pipeline integration (SEIFA → Health → Boundary → Risk)
- Storage optimization integration (Processing → Optimization → Storage)
- Analysis pipeline integration (Data → Risk Assessment → Results)
- Cross-component data flow and consistency
- Real-world scenario simulation

Validates complete system functionality with Australian health data patterns.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil
import time
from datetime import datetime, date

from src.data_processing.seifa_processor import SEIFAProcessor
from src.data_processing.health_processor import HealthDataProcessor
from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor
from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from src.analysis.risk.health_risk_calculator import HealthRiskCalculator


class TestEndToEndIntegration:
    """End-to-end integration tests across all major components."""
    
    def test_complete_data_processing_pipeline(self, mock_excel_seifa_file, mock_health_data, 
                                             mock_boundary_data, mock_data_paths):
        """Test complete data processing pipeline from raw data to analysis-ready datasets."""
        
        # Step 1: Initialize all processors
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Step 2: Create and process SEIFA data
        excel_file = mock_excel_seifa_file(num_areas=100)
        expected_seifa_path = seifa_processor.raw_dir / "SEIFA_2021_SA2_Indexes.xlsx"
        shutil.copy(excel_file, expected_seifa_path)
        
        seifa_df = seifa_processor.process_seifa_file()
        
        # Step 3: Create and process health data
        health_df = mock_health_data(num_records=500, num_sa2_areas=100)
        health_csv_path = health_processor.raw_dir / "health_data.csv"
        health_df.write_csv(health_csv_path)
        
        # Process health data through validation and aggregation
        validated_health = health_processor._validate_health_data(health_df)
        aggregated_health = health_processor._aggregate_by_sa2(validated_health)
        
        # Step 4: Create and process boundary data
        boundary_df = mock_boundary_data(num_areas=100)
        boundary_csv_path = boundary_processor.raw_dir / "boundaries.csv"
        boundary_df.write_csv(boundary_csv_path)
        
        processed_boundary = boundary_processor._validate_boundary_data(boundary_df)
        enhanced_boundary = boundary_processor._calculate_population_density(processed_boundary)
        
        # Step 5: Integrate all datasets
        # Ensure consistent SA2 codes across datasets
        common_sa2_codes = list(set(seifa_df["sa2_code_2021"].to_list()) & 
                               set(enhanced_boundary["sa2_code_2021"].to_list()))[:50]
        
        # Filter datasets to common SA2 codes
        integrated_seifa = seifa_df.filter(pl.col("sa2_code_2021").is_in(common_sa2_codes))
        integrated_boundary = enhanced_boundary.filter(pl.col("sa2_code_2021").is_in(common_sa2_codes))
        integrated_health = aggregated_health.filter(pl.col("sa2_code").is_in(common_sa2_codes))
        
        # Step 6: Save integrated datasets
        seifa_parquet = storage_manager.save_optimized_parquet(
            integrated_seifa, 
            mock_data_paths["parquet_dir"] / "integrated_seifa.parquet",
            data_type="seifa"
        )
        
        boundary_parquet = storage_manager.save_optimized_parquet(
            integrated_boundary,
            mock_data_paths["parquet_dir"] / "integrated_boundary.parquet", 
            data_type="geographic"
        )
        
        health_parquet = storage_manager.save_optimized_parquet(
            integrated_health,
            mock_data_paths["parquet_dir"] / "integrated_health.parquet",
            data_type="health"
        )
        
        # Step 7: Verify integration results
        assert seifa_parquet.exists()
        assert boundary_parquet.exists()
        assert health_parquet.exists()
        
        # Load and verify integrated data
        loaded_seifa = pl.read_parquet(seifa_parquet)
        loaded_boundary = pl.read_parquet(boundary_parquet)
        loaded_health = pl.read_parquet(health_parquet)
        
        assert len(loaded_seifa) > 0
        assert len(loaded_boundary) > 0
        assert len(loaded_health) > 0
        
        # Verify data consistency
        seifa_codes = set(loaded_seifa["sa2_code_2021"].to_list())
        boundary_codes = set(loaded_boundary["sa2_code_2021"].to_list())
        health_codes = set(loaded_health["sa2_code"].to_list())
        
        # Should have overlapping SA2 codes
        common_codes = seifa_codes & boundary_codes & health_codes
        assert len(common_codes) > 10, "Should have significant overlap in SA2 codes"
        
        # Verify data quality
        assert loaded_seifa["irsd_decile"].min() >= 1
        assert loaded_seifa["irsd_decile"].max() <= 10
        assert loaded_boundary["population_density"].min() >= 0
        assert loaded_health["total_prescriptions"].min() >= 0
    
    def test_storage_optimization_integration(self, mock_seifa_data, mock_health_data, mock_data_paths):
        """Test integration between data processing and storage optimization."""
        
        # Initialize components
        memory_optimizer = MemoryOptimizer()
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create test datasets with suboptimal types
        seifa_df = mock_seifa_data(num_areas=200)
        health_df = mock_health_data(num_records=1000, num_sa2_areas=200)
        
        # Force inefficient data types
        seifa_df = seifa_df.with_columns([
            pl.col("irsd_decile").cast(pl.Int64),
            pl.col("sa2_code_2021").cast(pl.Utf8)
        ])
        
        health_df = health_df.with_columns([
            pl.col("prescription_count").cast(pl.Int64),
            pl.col("sa2_code").cast(pl.Utf8),
            pl.col("state").cast(pl.Utf8)
        ])
        
        # Step 1: Memory optimization
        initial_seifa_memory = seifa_df.estimated_size("mb")
        initial_health_memory = health_df.estimated_size("mb")
        
        optimized_seifa = memory_optimizer.optimize_data_types(seifa_df, data_category="seifa")
        optimized_health = memory_optimizer.optimize_data_types(health_df, data_category="health")
        
        optimized_seifa_memory = optimized_seifa.estimated_size("mb")
        optimized_health_memory = optimized_health.estimated_size("mb")
        
        # Verify memory optimization
        seifa_reduction = (initial_seifa_memory - optimized_seifa_memory) / initial_seifa_memory
        health_reduction = (initial_health_memory - optimized_health_memory) / initial_health_memory
        
        assert seifa_reduction > 0, "SEIFA memory optimization should reduce memory usage"
        assert health_reduction > 0, "Health memory optimization should reduce memory usage"
        
        # Step 2: Storage optimization
        seifa_path = mock_data_paths["parquet_dir"] / "optimized_seifa.parquet"
        health_path = mock_data_paths["parquet_dir"] / "optimized_health.parquet"
        
        # Save unoptimized for comparison
        seifa_unopt_path = mock_data_paths["parquet_dir"] / "unoptimized_seifa.csv"
        health_unopt_path = mock_data_paths["parquet_dir"] / "unoptimized_health.csv"
        
        seifa_df.write_csv(seifa_unopt_path)
        health_df.write_csv(health_unopt_path)
        
        unopt_seifa_size = seifa_unopt_path.stat().st_size
        unopt_health_size = health_unopt_path.stat().st_size
        
        # Save optimized
        storage_manager.save_optimized_parquet(optimized_seifa, seifa_path, data_type="seifa")
        storage_manager.save_optimized_parquet(optimized_health, health_path, data_type="health")
        
        opt_seifa_size = seifa_path.stat().st_size
        opt_health_size = health_path.stat().st_size
        
        # Verify storage optimization
        seifa_compression = opt_seifa_size / unopt_seifa_size
        health_compression = opt_health_size / unopt_health_size
        
        assert seifa_compression < 0.8, "SEIFA storage should achieve significant compression"
        assert health_compression < 0.8, "Health storage should achieve significant compression"
        
        # Step 3: Verify data integrity after full optimization pipeline
        loaded_seifa = pl.read_parquet(seifa_path)
        loaded_health = pl.read_parquet(health_path)
        
        assert len(loaded_seifa) == len(seifa_df)
        assert len(loaded_health) == len(health_df)
        
        # Verify optimizations were preserved
        assert loaded_seifa["sa2_code_2021"].dtype == pl.Categorical
        assert loaded_seifa["irsd_decile"].dtype == pl.Int8
        assert loaded_health["sa2_code"].dtype == pl.Categorical
        assert loaded_health["state"].dtype == pl.Categorical
    
    def test_risk_assessment_integration(self, integration_test_data, mock_data_paths):
        """Test integration of complete risk assessment pipeline."""
        
        # Initialize risk calculator
        risk_calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Create comprehensive integrated dataset
        integrated_data = integration_test_data(num_sa2_areas=100, num_health_records=500)
        
        seifa_df = integrated_data["seifa"]
        health_df = integrated_data["health"]
        boundary_df = integrated_data["boundaries"]
        
        # Save datasets as expected by risk calculator
        seifa_path = mock_data_paths["processed_dir"] / "seifa_2021_sa2.csv"
        boundary_path = mock_data_paths["processed_dir"] / "sa2_boundaries_processed.csv"
        
        seifa_df.write_csv(seifa_path)
        boundary_df.write_csv(boundary_path)
        
        # Step 1: Load data
        load_success = risk_calculator.load_processed_data()
        assert load_success is True
        
        # Step 2: Calculate SEIFA risk component
        seifa_risk = risk_calculator._calculate_seifa_risk_score(seifa_df)
        assert "seifa_risk_score" in seifa_risk.columns
        assert len(seifa_risk) > 0
        
        # Step 3: Process health data for risk assessment
        health_aggregated = health_df.group_by("sa2_code").agg([
            pl.col("prescription_count").sum().alias("total_prescriptions"),
            pl.col("chronic_medication").mean().alias("chronic_rate"),
            pl.col("cost_government").sum().alias("total_cost")
        ])
        
        health_risk = risk_calculator._calculate_health_utilisation_risk(health_aggregated)
        assert "health_utilisation_risk" in health_risk.columns
        
        # Step 4: Calculate geographic risk
        geographic_risk = risk_calculator._calculate_geographic_accessibility_risk(boundary_df)
        assert "geographic_risk" in geographic_risk.columns
        
        # Step 5: Integrate all risk components
        comprehensive_risk = seifa_risk.join(
            health_risk, 
            left_on="sa2_code_2021", 
            right_on="sa2_code", 
            how="inner"
        ).join(
            geographic_risk, 
            on="sa2_code_2021", 
            how="inner"
        )
        
        # Step 6: Calculate composite risk scores
        composite_risk = risk_calculator._calculate_composite_risk_score(comprehensive_risk)
        
        assert "composite_risk_score" in composite_risk.columns
        assert len(composite_risk) > 0
        
        # Step 7: Classify risk categories
        classified_risk = risk_calculator._classify_risk_categories(composite_risk)
        
        assert "risk_category" in classified_risk.columns
        
        # Verify risk classification distribution
        risk_categories = classified_risk["risk_category"].value_counts()
        assert len(risk_categories) > 1, "Should have multiple risk categories"
        
        # Step 8: Generate summary statistics
        risk_summary = risk_calculator._generate_risk_summary(classified_risk)
        
        expected_summary_keys = ["total_sa2_areas", "risk_distribution", "average_risk_score"]
        for key in expected_summary_keys:
            assert key in risk_summary
        
        assert risk_summary["total_sa2_areas"] > 0
        assert 0 <= risk_summary["average_risk_score"] <= 100
    
    def test_cross_component_data_consistency(self, integration_test_data, mock_data_paths):
        """Test data consistency across all integrated components."""
        
        # Create integrated test environment
        integrated_data = integration_test_data(num_sa2_areas=50, num_health_records=200)
        
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        risk_calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Process all datasets
        seifa_df = integrated_data["seifa"]
        health_df = integrated_data["health"]
        boundary_df = integrated_data["boundaries"]
        
        # Validate through each processor
        validated_seifa = seifa_processor._validate_seifa_data(seifa_df)
        validated_health = health_processor._validate_health_data(health_df)
        validated_boundary = boundary_processor._validate_boundary_data(boundary_df)
        
        # Check SA2 code consistency
        seifa_codes = set(validated_seifa["sa2_code_2021"].to_list())
        boundary_codes = set(validated_boundary["sa2_code_2021"].to_list())
        health_codes = set(validated_health["sa2_code"].to_list())
        
        # Verify overlap
        seifa_boundary_overlap = len(seifa_codes & boundary_codes) / len(seifa_codes)
        seifa_health_overlap = len(seifa_codes & health_codes) / len(seifa_codes)
        
        assert seifa_boundary_overlap > 0.8, "SEIFA and boundary data should have high SA2 code overlap"
        assert seifa_health_overlap > 0.7, "SEIFA and health data should have reasonable SA2 code overlap"
        
        # Check data value consistency
        # SEIFA deciles should be 1-10
        for col in validated_seifa.columns:
            if "decile" in col:
                decile_values = validated_seifa[col].drop_nulls()
                if len(decile_values) > 0:
                    assert decile_values.min() >= 1
                    assert decile_values.max() <= 10
        
        # Health utilisation should be non-negative
        if "prescription_count" in validated_health.columns:
            prescription_counts = validated_health["prescription_count"].drop_nulls()
            if len(prescription_counts) > 0:
                assert prescription_counts.min() >= 0
        
        # Population should be positive
        if "population_2021" in validated_boundary.columns:
            populations = validated_boundary["population_2021"].drop_nulls()
            if len(populations) > 0:
                assert populations.min() > 0
        
        # Test risk calculation consistency
        # Calculate risks using integrated data
        seifa_risk = risk_calculator._calculate_seifa_risk_score(validated_seifa)
        
        # Risk scores should be within valid range
        risk_scores = seifa_risk["seifa_risk_score"].drop_nulls()
        if len(risk_scores) > 0:
            assert risk_scores.min() >= 0
            assert risk_scores.max() <= 100
            
            # Should have reasonable distribution
            risk_std = risk_scores.std()
            assert risk_std > 5, "Risk scores should show meaningful variation"
    
    def test_performance_integration_pipeline(self, mock_excel_seifa_file, mock_health_data, 
                                            mock_boundary_data, mock_data_paths):
        """Test performance of complete integrated pipeline."""
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Initialize all components
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        memory_optimizer = MemoryOptimizer()
        risk_calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Create realistic-sized datasets
        excel_file = mock_excel_seifa_file(num_areas=500)
        expected_seifa_path = seifa_processor.raw_dir / "SEIFA_2021_SA2_Indexes.xlsx"
        shutil.copy(excel_file, expected_seifa_path)
        
        health_df = mock_health_data(num_records=2500, num_sa2_areas=500)
        boundary_df = mock_boundary_data(num_areas=500)
        
        # Execute complete pipeline
        # Step 1: Data processing
        seifa_df = seifa_processor.process_seifa_file()
        validated_health = health_processor._validate_health_data(health_df)
        aggregated_health = health_processor._aggregate_by_sa2(validated_health)
        processed_boundary = boundary_processor._validate_boundary_data(boundary_df)
        
        # Step 2: Memory optimization
        optimized_seifa = memory_optimizer.optimize_data_types(seifa_df, data_category="seifa")
        optimized_health = memory_optimizer.optimize_data_types(aggregated_health, data_category="health")
        optimized_boundary = memory_optimizer.optimize_data_types(processed_boundary, data_category="geographic")
        
        # Step 3: Storage optimization
        seifa_path = storage_manager.save_optimized_parquet(
            optimized_seifa, 
            mock_data_paths["parquet_dir"] / "pipeline_seifa.parquet",
            data_type="seifa"
        )
        health_path = storage_manager.save_optimized_parquet(
            optimized_health,
            mock_data_paths["parquet_dir"] / "pipeline_health.parquet", 
            data_type="health"
        )
        boundary_path = storage_manager.save_optimized_parquet(
            optimized_boundary,
            mock_data_paths["parquet_dir"] / "pipeline_boundary.parquet",
            data_type="geographic"
        )
        
        # Step 4: Risk assessment
        seifa_risk = risk_calculator._calculate_seifa_risk_score(optimized_seifa)
        health_risk = risk_calculator._calculate_health_utilisation_risk(optimized_health)
        geographic_risk = risk_calculator._calculate_geographic_accessibility_risk(optimized_boundary)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        total_time = end_time - start_time
        memory_usage = end_memory - start_memory
        
        # Performance assertions for complete pipeline
        assert total_time < 60.0, f"Complete pipeline took {total_time:.1f}s, expected <60s"
        assert memory_usage < 1000, f"Pipeline used {memory_usage:.1f}MB, expected <1GB"
        
        # Verify all outputs exist and are valid
        assert seifa_path.exists()
        assert health_path.exists() 
        assert boundary_path.exists()
        
        assert len(seifa_risk) > 400  # Should retain most SA2 areas
        assert len(health_risk) > 0
        assert len(geographic_risk) > 400
        
        # Verify data quality maintained throughout pipeline
        assert "seifa_risk_score" in seifa_risk.columns
        assert "health_utilisation_risk" in health_risk.columns
        assert "geographic_risk" in geographic_risk.columns
    
    def test_error_handling_integration(self, mock_data_paths):
        """Test error handling across integrated components."""
        
        # Initialize components
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        risk_calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Test 1: Missing input files
        with pytest.raises(FileNotFoundError):
            seifa_processor.process_seifa_file()
        
        # Test 2: Invalid data formats
        invalid_df = pl.DataFrame({
            "invalid_column": ["invalid_data", "more_invalid"],
            "another_column": [1, 2]
        })
        
        # Should handle gracefully or raise appropriate errors
        try:
            validated_invalid = health_processor._validate_health_data(invalid_df)
            # If it succeeds, should return empty or filtered DataFrame
            assert len(validated_invalid) == 0
        except (ValueError, pl.ComputeError, KeyError):
            # Acceptable to raise validation errors
            pass
        
        # Test 3: Risk calculator with missing data
        load_result = risk_calculator.load_processed_data()
        assert load_result is False  # Should handle missing files gracefully
        
        # Test 4: Empty datasets
        empty_df = pl.DataFrame()
        
        try:
            empty_result = health_processor._validate_health_data(empty_df)
            assert len(empty_result) == 0
        except Exception as e:
            # Should either handle gracefully or raise informative error
            assert isinstance(e, (ValueError, pl.ComputeError))
    
    def test_data_versioning_integration(self, mock_seifa_data, mock_data_paths):
        """Test data versioning and compatibility across pipeline versions."""
        
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create v1 dataset
        v1_data = mock_seifa_data(num_areas=100)
        v1_path = mock_data_paths["parquet_dir"] / "seifa_v1.parquet"
        
        # Add version metadata
        v1_metadata = {
            "version": "1.0",
            "schema_version": "2021.1",
            "created_date": datetime.now().isoformat()
        }
        
        storage_manager.save_with_metadata(v1_data, v1_path, v1_metadata, data_type="seifa")
        
        # Create v2 dataset with additional columns
        v2_data = v1_data.with_columns([
            pl.lit("2024").alias("data_year"),
            pl.Series("quality_score", np.random.uniform(0.8, 1.0, len(v1_data)))
        ])
        
        v2_path = mock_data_paths["parquet_dir"] / "seifa_v2.parquet"
        v2_metadata = {
            "version": "2.0", 
            "schema_version": "2024.1",
            "created_date": datetime.now().isoformat(),
            "backwards_compatible": True
        }
        
        storage_manager.save_with_metadata(v2_data, v2_path, v2_metadata, data_type="seifa")
        
        # Test compatibility
        loaded_v1 = pl.read_parquet(v1_path)
        loaded_v2 = pl.read_parquet(v2_path)
        
        # V2 should be superset of V1
        v1_columns = set(loaded_v1.columns)
        v2_columns = set(loaded_v2.columns)
        
        assert v1_columns.issubset(v2_columns), "V2 should be backwards compatible with V1"
        
        # Core columns should be identical
        common_columns = list(v1_columns & v2_columns)
        v1_subset = loaded_v1.select(common_columns)
        v2_subset = loaded_v2.select(common_columns)
        
        # Should have same core data
        assert len(v1_subset) == len(v2_subset)
    
    def test_concurrent_integration_operations(self, integration_test_data, mock_data_paths):
        """Test concurrent operations across integrated components."""
        import concurrent.futures
        
        integrated_data = integration_test_data(num_sa2_areas=100, num_health_records=400)
        
        def process_seifa_component():
            processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
            return processor._validate_seifa_data(integrated_data["seifa"])
        
        def process_health_component():
            processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
            return processor._validate_health_data(integrated_data["health"])
        
        def process_boundary_component():
            processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
            return processor._validate_boundary_data(integrated_data["boundaries"])
        
        def calculate_risk_component():
            calculator = HealthRiskCalculator()
            return calculator._calculate_seifa_risk_score(integrated_data["seifa"])
        
        # Run components concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_seifa_component),
                executor.submit(process_health_component),
                executor.submit(process_boundary_component),
                executor.submit(calculate_risk_component)
            ]
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Verify all concurrent operations succeeded
        assert len(results) == 4
        
        for result in results:
            assert isinstance(result, pl.DataFrame)
            assert len(result) > 0