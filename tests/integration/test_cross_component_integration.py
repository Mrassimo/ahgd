"""
Cross-component integration tests for Australian Health Analytics platform.

Tests the interaction and data consistency between different components:
- SEIFA processor ↔ Health processor ↔ Boundary processor integration
- Risk calculator ↔ Data processors coordination  
- Storage manager ↔ All data processing components
- Memory optimizer ↔ Pipeline components
- Incremental processor ↔ Storage systems

Validates that components work together seamlessly with consistent data flows,
proper error propagation, and maintained data integrity across component boundaries.
"""

import pytest
import polars as pl
import numpy as np
import time
import psutil
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import concurrent.futures
import logging
from unittest.mock import Mock, patch

from src.data_processing.seifa_processor import SEIFAProcessor
from src.data_processing.health_processor import HealthDataProcessor
from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor
from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from src.data_processing.storage.incremental_processor import IncrementalProcessor
from src.analysis.risk.health_risk_calculator import HealthRiskCalculator
from src.analysis.risk.healthcare_access_scorer import HealthcareAccessScorer


class TestCrossComponentIntegration:
    """Tests for component interaction and data consistency across the platform."""
    
    def test_seifa_health_boundary_processor_integration(self, integration_test_data, mock_data_paths):
        """Test seamless integration between SEIFA, Health, and Boundary processors."""
        
        # Create coordinated test dataset
        integrated_data = integration_test_data(num_sa2_areas=200, num_health_records=1000)
        
        # Initialize all processors
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent) 
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Process each dataset through its respective processor
        seifa_result = seifa_processor._validate_seifa_data(integrated_data["seifa"])
        health_result = health_processor._validate_health_data(integrated_data["health"])
        boundary_result = boundary_processor._validate_boundary_data(integrated_data["boundaries"])
        
        # Test 1: Data consistency across processors
        # All processors should maintain SA2 code integrity
        seifa_sa2_codes = set(seifa_result["sa2_code_2021"].drop_nulls().to_list())
        health_sa2_codes = set(health_result["sa2_code"].drop_nulls().to_list())
        boundary_sa2_codes = set(boundary_result["sa2_code_2021"].drop_nulls().to_list())
        
        # Calculate overlap rates
        seifa_health_overlap = len(seifa_sa2_codes & health_sa2_codes) / len(seifa_sa2_codes)
        seifa_boundary_overlap = len(seifa_sa2_codes & boundary_sa2_codes) / len(seifa_sa2_codes)
        health_boundary_overlap = len(health_sa2_codes & boundary_sa2_codes) / len(health_sa2_codes)
        
        assert seifa_health_overlap > 0.85, f"SEIFA-Health overlap {seifa_health_overlap:.1%} should be >85%"
        assert seifa_boundary_overlap > 0.90, f"SEIFA-Boundary overlap {seifa_boundary_overlap:.1%} should be >90%"
        assert health_boundary_overlap > 0.85, f"Health-Boundary overlap {health_boundary_overlap:.1%} should be >85%"
        
        # Test 2: Cross-processor data integration
        # Should be able to join datasets seamlessly
        seifa_health_integrated = seifa_result.join(
            health_result.group_by("sa2_code").first(),
            left_on="sa2_code_2021",
            right_on="sa2_code",
            how="inner"
        )
        
        comprehensive_integrated = seifa_health_integrated.join(
            boundary_result,
            on="sa2_code_2021", 
            how="inner"
        )
        
        # Integration should retain substantial data
        integration_retention = len(comprehensive_integrated) / len(seifa_result)
        assert integration_retention > 0.75, f"Integration retention {integration_retention:.1%} should be >75%"
        
        # Test 3: Data quality consistency across processors
        # SEIFA deciles should be valid
        seifa_deciles = seifa_result["irsd_decile"].drop_nulls()
        if len(seifa_deciles) > 0:
            assert seifa_deciles.min() >= 1 and seifa_deciles.max() <= 10, "SEIFA deciles should be 1-10"
        
        # Health utilisation should be non-negative
        if "prescription_count" in health_result.columns:
            health_counts = health_result["prescription_count"].drop_nulls()
            if len(health_counts) > 0:
                assert health_counts.min() >= 0, "Health utilisation should be non-negative"
        
        # Geographic data should be reasonable
        if "population_2021" in boundary_result.columns:
            populations = boundary_result["population_2021"].drop_nulls()
            if len(populations) > 0:
                assert populations.min() > 0, "Population should be positive"
                assert populations.max() < 100000, "Population should be reasonable for SA2 areas"
        
        # Test 4: Processor performance coordination
        # All processors should complete within reasonable time when run together
        start_time = time.time()
        
        # Parallel processing simulation
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(seifa_processor._validate_seifa_data, integrated_data["seifa"]),
                executor.submit(health_processor._validate_health_data, integrated_data["health"]),
                executor.submit(boundary_processor._validate_boundary_data, integrated_data["boundaries"])
            ]
            
            concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        concurrent_time = time.time() - start_time
        
        assert concurrent_time < 30.0, f"Concurrent processing took {concurrent_time:.1f}s, expected <30s"
        assert len(concurrent_results) == 3, "All processors should complete successfully"
        
        # Test 5: Error propagation between processors
        # Corrupt data in one processor should be handled gracefully by others
        corrupted_seifa = integrated_data["seifa"].with_columns([
            pl.col("sa2_code_2021").map_elements(
                lambda x: "INVALID" if np.random.random() < 0.1 else x,
                return_dtype=pl.Utf8
            )
        ])
        
        # Should handle corruption gracefully
        try:
            corrupted_result = seifa_processor._validate_seifa_data(corrupted_seifa)
            # If successful, should filter out invalid codes
            valid_codes = corrupted_result["sa2_code_2021"].drop_nulls().to_list()
            assert all(len(code) == 9 and code.isdigit() for code in valid_codes[:10])
        except Exception as e:
            # Or raise appropriate error
            assert isinstance(e, (ValueError, pl.ComputeError))
        
        # Generate integration report
        integration_report = {
            "seifa_health_overlap_rate": seifa_health_overlap,
            "seifa_boundary_overlap_rate": seifa_boundary_overlap,
            "health_boundary_overlap_rate": health_boundary_overlap,
            "comprehensive_integration_retention": integration_retention,
            "concurrent_processing_time": concurrent_time,
            "data_quality_validation_passed": True,
            "error_handling_tested": True,
            "total_sa2_areas_integrated": len(comprehensive_integrated)
        }
        
        logging.info(f"Cross-Component Integration Report: {integration_report}")
        
        return integration_report
    
    def test_risk_calculator_data_processor_coordination(self, integration_test_data, mock_data_paths):
        """Test coordination between risk calculator and all data processors."""
        
        # Initialize components
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        risk_calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Create test dataset
        integrated_data = integration_test_data(num_sa2_areas=150, num_health_records=750)
        
        # Process data through processors
        processed_seifa = seifa_processor._validate_seifa_data(integrated_data["seifa"])
        processed_health = health_processor._validate_health_data(integrated_data["health"])
        processed_boundaries = boundary_processor._validate_boundary_data(integrated_data["boundaries"])
        
        # Test 1: Risk calculator should accept processed data from all processors
        seifa_risk = risk_calculator._calculate_seifa_risk_score(processed_seifa)
        
        # Aggregate health data for risk calculation
        aggregated_health = health_processor._aggregate_by_sa2(processed_health)
        health_risk = risk_calculator._calculate_health_utilisation_risk(aggregated_health)
        
        # Calculate geographic risk
        geographic_risk = risk_calculator._calculate_geographic_accessibility_risk(processed_boundaries)
        
        # All risk calculations should succeed
        assert len(seifa_risk) > 0, "SEIFA risk calculation should produce results"
        assert len(health_risk) > 0, "Health risk calculation should produce results" 
        assert len(geographic_risk) > 0, "Geographic risk calculation should produce results"
        
        # Test 2: Risk score integration should work seamlessly
        # Join risk components
        integrated_risk = seifa_risk.join(
            health_risk,
            left_on="sa2_code_2021",
            right_on="sa2_code",
            how="inner"
        ).join(
            geographic_risk,
            on="sa2_code_2021",
            how="inner"
        )
        
        # Should retain substantial overlap
        risk_integration_rate = len(integrated_risk) / len(seifa_risk)
        assert risk_integration_rate > 0.70, f"Risk integration rate {risk_integration_rate:.1%} should be >70%"
        
        # Test 3: Composite risk calculation coordination
        composite_risk = risk_calculator._calculate_composite_risk_score(integrated_risk)
        classified_risk = risk_calculator._classify_risk_categories(composite_risk)
        
        assert "composite_risk_score" in composite_risk.columns
        assert "risk_category" in classified_risk.columns
        
        # Risk scores should be valid
        risk_scores = composite_risk["composite_risk_score"].drop_nulls()
        if len(risk_scores) > 0:
            assert risk_scores.min() >= 0 and risk_scores.max() <= 100, "Risk scores should be 0-100"
        
        # Risk categories should be valid
        risk_categories = classified_risk["risk_category"].drop_nulls().unique().to_list()
        expected_categories = ["Very Low", "Low", "Medium", "High", "Very High"]
        assert all(cat in expected_categories for cat in risk_categories), "Risk categories should be valid"
        
        # Test 4: Data flow consistency across component chain
        # Start with raw data and follow through entire chain
        chain_start_time = time.time()
        
        # Raw → Processed → Risk assessment chain
        chain_seifa = seifa_processor._validate_seifa_data(integrated_data["seifa"])
        chain_health = health_processor._aggregate_by_sa2(
            health_processor._validate_health_data(integrated_data["health"])
        )
        chain_boundaries = boundary_processor._calculate_population_density(
            boundary_processor._validate_boundary_data(integrated_data["boundaries"])
        )
        
        # Calculate risks on chain-processed data
        chain_seifa_risk = risk_calculator._calculate_seifa_risk_score(chain_seifa)
        chain_health_risk = risk_calculator._calculate_health_utilisation_risk(chain_health)
        chain_geographic_risk = risk_calculator._calculate_geographic_accessibility_risk(chain_boundaries)
        
        # Create final risk assessment
        final_risk = risk_calculator._classify_risk_categories(
            risk_calculator._calculate_composite_risk_score(
                chain_seifa_risk.join(
                    chain_health_risk, left_on="sa2_code_2021", right_on="sa2_code", how="inner"
                ).join(
                    chain_geographic_risk, on="sa2_code_2021", how="inner"
                )
            )
        )
        
        chain_time = time.time() - chain_start_time
        
        # Validate chain processing
        assert len(final_risk) > 0, "Complete chain should produce final results"
        assert chain_time < 60.0, f"Complete chain took {chain_time:.1f}s, expected <60s"
        
        # Final data quality validation
        final_sa2_codes = final_risk["sa2_code_2021"].drop_nulls().to_list()
        assert all(len(code) == 9 and code.isdigit() for code in final_sa2_codes[:10])
        
        # Test 5: Component error coordination
        # Error in one component should be handled gracefully by risk calculator
        empty_seifa = pl.DataFrame({"sa2_code_2021": [], "irsd_decile": []})
        
        try:
            empty_risk = risk_calculator._calculate_seifa_risk_score(empty_seifa)
            assert len(empty_risk) == 0, "Empty input should produce empty output"
        except Exception as e:
            # Should raise appropriate error, not crash
            assert isinstance(e, (ValueError, pl.ComputeError, KeyError))
        
        # Generate coordination report
        coordination_report = {
            "seifa_risk_records": len(seifa_risk),
            "health_risk_records": len(health_risk),
            "geographic_risk_records": len(geographic_risk),
            "risk_integration_rate": risk_integration_rate,
            "final_risk_assessments": len(final_risk),
            "complete_chain_time": chain_time,
            "data_quality_maintained": True,
            "error_handling_validated": True,
            "component_coordination_success": True
        }
        
        logging.info(f"Risk Calculator Coordination Report: {coordination_report}")
        
        return coordination_report
    
    def test_storage_manager_processor_integration(self, mock_seifa_data, mock_health_data, 
                                                  mock_boundary_data, mock_data_paths):
        """Test integration between storage manager and all data processors."""
        
        # Initialize components
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        memory_optimizer = MemoryOptimizer()
        
        # Create test datasets
        seifa_data = mock_seifa_data(num_areas=300)
        health_data = mock_health_data(num_records=1500, num_sa2_areas=300)
        boundary_data = mock_boundary_data(num_areas=300)
        
        # Test 1: Storage manager should handle outputs from all processors
        # Process data through each processor
        processed_seifa = seifa_processor._validate_seifa_data(seifa_data)
        processed_health = health_processor._aggregate_by_sa2(
            health_processor._validate_health_data(health_data)
        )
        processed_boundaries = boundary_processor._calculate_population_density(
            boundary_processor._validate_boundary_data(boundary_data)
        )
        
        # Storage manager should save all processed datasets
        seifa_path = storage_manager.save_optimized_parquet(
            processed_seifa,
            mock_data_paths["parquet_dir"] / "integrated_seifa.parquet",
            data_type="seifa"
        )
        
        health_path = storage_manager.save_optimized_parquet(
            processed_health,
            mock_data_paths["parquet_dir"] / "integrated_health.parquet",
            data_type="health"
        )
        
        boundaries_path = storage_manager.save_optimized_parquet(
            processed_boundaries,
            mock_data_paths["parquet_dir"] / "integrated_boundaries.parquet",
            data_type="geographic"
        )
        
        # All saves should succeed
        assert seifa_path.exists() and health_path.exists() and boundaries_path.exists()
        
        # Test 2: Memory optimizer integration with processors
        # Memory optimizer should work with processor outputs
        optimized_seifa = memory_optimizer.optimize_data_types(processed_seifa, data_category="seifa")
        optimized_health = memory_optimizer.optimize_data_types(processed_health, data_category="health")
        optimized_boundaries = memory_optimizer.optimize_data_types(processed_boundaries, data_category="geographic")
        
        # Verify optimization worked
        seifa_memory_reduction = (processed_seifa.estimated_size("mb") - optimized_seifa.estimated_size("mb")) / processed_seifa.estimated_size("mb")
        health_memory_reduction = (processed_health.estimated_size("mb") - optimized_health.estimated_size("mb")) / processed_health.estimated_size("mb")
        boundaries_memory_reduction = (processed_boundaries.estimated_size("mb") - optimized_boundaries.estimated_size("mb")) / processed_boundaries.estimated_size("mb")
        
        assert seifa_memory_reduction >= 0, "Memory optimizer should not increase SEIFA memory usage"
        assert health_memory_reduction >= 0, "Memory optimizer should not increase health memory usage"
        assert boundaries_memory_reduction >= 0, "Memory optimizer should not increase boundaries memory usage"
        
        # Test 3: Storage optimization with memory optimization
        # Storage manager should handle memory-optimized data efficiently
        opt_seifa_path = storage_manager.save_optimized_parquet(
            optimized_seifa,
            mock_data_paths["parquet_dir"] / "memory_opt_seifa.parquet",
            data_type="seifa"
        )
        
        opt_health_path = storage_manager.save_optimized_parquet(
            optimized_health,
            mock_data_paths["parquet_dir"] / "memory_opt_health.parquet", 
            data_type="health"
        )
        
        opt_boundaries_path = storage_manager.save_optimized_parquet(
            optimized_boundaries,
            mock_data_paths["parquet_dir"] / "memory_opt_boundaries.parquet",
            data_type="geographic"
        )
        
        # Compare file sizes - memory optimized versions should be more efficient
        seifa_size_improvement = seifa_path.stat().st_size / opt_seifa_path.stat().st_size
        health_size_improvement = health_path.stat().st_size / opt_health_path.stat().st_size
        boundaries_size_improvement = boundaries_path.stat().st_size / opt_boundaries_path.stat().st_size
        
        # Should see some improvement (or at least no degradation)
        assert seifa_size_improvement >= 0.9, f"SEIFA storage optimization ratio {seifa_size_improvement:.2f} should be ≥0.9"
        assert health_size_improvement >= 0.9, f"Health storage optimization ratio {health_size_improvement:.2f} should be ≥0.9"
        assert boundaries_size_improvement >= 0.9, f"Boundaries storage optimization ratio {boundaries_size_improvement:.2f} should be ≥0.9"
        
        # Test 4: Data integrity through storage roundtrip
        # Load saved data and verify integrity
        loaded_seifa = pl.read_parquet(opt_seifa_path)
        loaded_health = pl.read_parquet(opt_health_path)
        loaded_boundaries = pl.read_parquet(opt_boundaries_path)
        
        # Data should be identical after roundtrip
        assert len(loaded_seifa) == len(optimized_seifa)
        assert len(loaded_health) == len(optimized_health)
        assert len(loaded_boundaries) == len(optimized_boundaries)
        
        # Key columns should be preserved
        if "sa2_code_2021" in optimized_seifa.columns:
            assert "sa2_code_2021" in loaded_seifa.columns
            original_codes = set(optimized_seifa["sa2_code_2021"].drop_nulls().to_list())
            loaded_codes = set(loaded_seifa["sa2_code_2021"].drop_nulls().to_list())
            assert original_codes == loaded_codes, "SA2 codes should be preserved through storage"
        
        # Test 5: Metadata integration
        # Storage manager should preserve metadata from processors
        metadata = {
            "processor": "seifa_processor",
            "validation_date": datetime.now().isoformat(),
            "record_count": len(optimized_seifa),
            "data_quality_score": 0.95
        }
        
        meta_path = storage_manager.save_with_metadata(
            optimized_seifa,
            mock_data_paths["parquet_dir"] / "seifa_with_metadata.parquet",
            metadata,
            data_type="seifa"
        )
        
        assert meta_path.exists()
        
        # Should be able to load with metadata preserved
        loaded_with_meta = pl.read_parquet(meta_path)
        assert len(loaded_with_meta) == len(optimized_seifa)
        
        # Test 6: Concurrent storage operations
        # Multiple processors should be able to save simultaneously
        def save_seifa():
            return storage_manager.save_optimized_parquet(
                optimized_seifa,
                mock_data_paths["parquet_dir"] / f"concurrent_seifa_{time.time()}.parquet",
                data_type="seifa"
            )
        
        def save_health():
            return storage_manager.save_optimized_parquet(
                optimized_health,
                mock_data_paths["parquet_dir"] / f"concurrent_health_{time.time()}.parquet",
                data_type="health"
            )
        
        def save_boundaries():
            return storage_manager.save_optimized_parquet(
                optimized_boundaries,
                mock_data_paths["parquet_dir"] / f"concurrent_boundaries_{time.time()}.parquet",
                data_type="geographic"
            )
        
        # Run concurrent save operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            save_futures = [
                executor.submit(save_seifa),
                executor.submit(save_health), 
                executor.submit(save_boundaries)
            ]
            
            concurrent_paths = [future.result() for future in concurrent.futures.as_completed(save_futures)]
        
        # All concurrent saves should succeed
        assert len(concurrent_paths) == 3
        assert all(path.exists() for path in concurrent_paths)
        
        # Generate storage integration report
        storage_integration_report = {
            "seifa_memory_reduction_rate": seifa_memory_reduction,
            "health_memory_reduction_rate": health_memory_reduction,
            "boundaries_memory_reduction_rate": boundaries_memory_reduction,
            "seifa_storage_optimization": seifa_size_improvement,
            "health_storage_optimization": health_size_improvement,
            "boundaries_storage_optimization": boundaries_size_improvement,
            "data_integrity_maintained": True,
            "metadata_integration_tested": True,
            "concurrent_operations_successful": True,
            "total_files_created": len(concurrent_paths) + 6  # 3 concurrent + 6 regular saves
        }
        
        logging.info(f"Storage Integration Report: {storage_integration_report}")
        
        return storage_integration_report
    
    def test_incremental_processor_component_coordination(self, mock_seifa_data, mock_health_data, mock_data_paths):
        """Test incremental processor coordination with other components."""
        
        # Initialize components
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        incremental_processor = IncrementalProcessor(base_path=mock_data_paths["parquet_dir"])
        memory_optimizer = MemoryOptimizer()
        
        # Test 1: Incremental processing with SEIFA processor
        # Create initial SEIFA dataset
        initial_seifa = mock_seifa_data(num_areas=200)
        processed_initial_seifa = seifa_processor._validate_seifa_data(initial_seifa)
        
        # Save initial version
        initial_path = storage_manager.save_with_metadata(
            processed_initial_seifa,
            mock_data_paths["parquet_dir"] / "incremental_seifa_v1.parquet",
            {"version": "1.0", "date": datetime.now().isoformat()},
            data_type="seifa"
        )
        
        # Create incremental update
        update_seifa = mock_seifa_data(num_areas=50)  # Additional SA2 areas
        processed_update_seifa = seifa_processor._validate_seifa_data(update_seifa)
        
        # Merge using incremental processor
        merged_seifa = incremental_processor.merge_incremental_data(
            existing_data=pl.read_parquet(initial_path),
            new_data=processed_update_seifa,
            key_column="sa2_code_2021",
            strategy="upsert"
        )
        
        # Should have grown
        assert len(merged_seifa) >= len(processed_initial_seifa), "Incremental SEIFA should grow dataset"
        
        # Test 2: Incremental processing with health processor
        # Create initial health dataset
        initial_health = mock_health_data(num_records=1000, num_sa2_areas=200)
        processed_initial_health = health_processor._validate_health_data(initial_health)
        aggregated_initial_health = health_processor._aggregate_by_sa2(processed_initial_health)
        
        # Save initial version
        health_initial_path = storage_manager.save_with_metadata(
            aggregated_initial_health,
            mock_data_paths["parquet_dir"] / "incremental_health_v1.parquet",
            {"version": "1.0", "date": datetime.now().isoformat()},
            data_type="health"
        )
        
        # Create incremental health update
        update_health = mock_health_data(num_records=500, num_sa2_areas=100)
        processed_update_health = health_processor._validate_health_data(update_health)
        aggregated_update_health = health_processor._aggregate_by_sa2(processed_update_health)
        
        # Merge health data
        merged_health = incremental_processor.merge_incremental_data(
            existing_data=pl.read_parquet(health_initial_path),
            new_data=aggregated_update_health,
            key_column="sa2_code",
            strategy="append"  # Health data can be appended
        )
        
        # Should have grown
        assert len(merged_health) >= len(aggregated_initial_health), "Incremental health should grow dataset"
        
        # Test 3: Memory optimization with incremental processing
        # Memory optimizer should work with incremental results
        optimized_merged_seifa = memory_optimizer.optimize_data_types(merged_seifa, data_category="seifa")
        optimized_merged_health = memory_optimizer.optimize_data_types(merged_health, data_category="health")
        
        # Should maintain data integrity
        assert len(optimized_merged_seifa) == len(merged_seifa)
        assert len(optimized_merged_health) == len(merged_health)
        
        # Test 4: Storage manager with incremental results
        # Should be able to save incremental results efficiently
        incremental_seifa_path = storage_manager.save_with_metadata(
            optimized_merged_seifa,
            mock_data_paths["parquet_dir"] / "incremental_seifa_v2.parquet",
            {"version": "2.0", "date": datetime.now().isoformat(), "incremental": True},
            data_type="seifa"
        )
        
        incremental_health_path = storage_manager.save_with_metadata(
            optimized_merged_health,
            mock_data_paths["parquet_dir"] / "incremental_health_v2.parquet",
            {"version": "2.0", "date": datetime.now().isoformat(), "incremental": True},
            data_type="health"
        )
        
        # Files should exist and be larger than initial versions
        assert incremental_seifa_path.exists() and incremental_health_path.exists()
        assert incremental_seifa_path.stat().st_size >= initial_path.stat().st_size
        assert incremental_health_path.stat().st_size >= health_initial_path.stat().st_size
        
        # Test 5: Rollback coordination
        # Should be able to rollback incremental changes
        rollback_seifa = incremental_processor.rollback_to_version(
            current_data=merged_seifa,
            previous_data=processed_initial_seifa,
            key_column="sa2_code_2021"
        )
        
        # Rollback should restore original size
        assert len(rollback_seifa) == len(processed_initial_seifa), "Rollback should restore original dataset size"
        
        # Test 6: Component chain with incremental processing
        # Full pipeline: Process → Optimize → Incremental → Store
        chain_start_time = time.time()
        
        # Create new update
        final_update_seifa = mock_seifa_data(num_areas=30)
        
        # Process through full chain
        chain_processed = seifa_processor._validate_seifa_data(final_update_seifa)
        chain_optimized = memory_optimizer.optimize_data_types(chain_processed, data_category="seifa")
        chain_incremental = incremental_processor.merge_incremental_data(
            existing_data=optimized_merged_seifa,
            new_data=chain_optimized,
            key_column="sa2_code_2021",
            strategy="upsert"
        )
        chain_stored_path = storage_manager.save_optimized_parquet(
            chain_incremental,
            mock_data_paths["parquet_dir"] / "chain_incremental_final.parquet",
            data_type="seifa"
        )
        
        chain_time = time.time() - chain_start_time
        
        # Chain should complete efficiently
        assert chain_time < 15.0, f"Component chain took {chain_time:.1f}s, expected <15s"
        assert chain_stored_path.exists()
        
        # Final result should be larger than previous version
        final_result = pl.read_parquet(chain_stored_path)
        assert len(final_result) >= len(optimized_merged_seifa)
        
        # Generate incremental coordination report
        incremental_coordination_report = {
            "initial_seifa_records": len(processed_initial_seifa),
            "final_seifa_records": len(final_result),
            "initial_health_records": len(aggregated_initial_health),
            "final_health_records": len(merged_health),
            "incremental_growth_rate": (len(final_result) - len(processed_initial_seifa)) / len(processed_initial_seifa),
            "component_chain_time": chain_time,
            "rollback_functionality_tested": True,
            "memory_optimization_compatible": True,
            "storage_integration_successful": True,
            "versions_created": 4  # v1 initial, v2 incremental, rollback, final
        }
        
        logging.info(f"Incremental Coordination Report: {incremental_coordination_report}")
        
        return incremental_coordination_report
    
    def test_end_to_end_component_workflow(self, integration_test_data, mock_data_paths):
        """Test complete workflow across all components working together."""
        
        # Initialize all components
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        memory_optimizer = MemoryOptimizer()
        incremental_processor = IncrementalProcessor(base_path=mock_data_paths["parquet_dir"])
        risk_calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Create comprehensive test dataset
        integrated_data = integration_test_data(num_sa2_areas=250, num_health_records=1250)
        
        # WORKFLOW STAGE 1: Data Processing
        stage1_start = time.time()
        
        # Process all datasets through their respective processors
        processed_seifa = seifa_processor._validate_seifa_data(integrated_data["seifa"])
        processed_health = health_processor._validate_health_data(integrated_data["health"])
        processed_boundaries = boundary_processor._validate_boundary_data(integrated_data["boundaries"])
        
        stage1_time = time.time() - stage1_start
        
        # WORKFLOW STAGE 2: Memory Optimization
        stage2_start = time.time()
        
        optimized_seifa = memory_optimizer.optimize_data_types(processed_seifa, data_category="seifa")
        optimized_health = memory_optimizer.optimize_data_types(processed_health, data_category="health")
        optimized_boundaries = memory_optimizer.optimize_data_types(processed_boundaries, data_category="geographic")
        
        stage2_time = time.time() - stage2_start
        
        # WORKFLOW STAGE 3: Storage and Versioning
        stage3_start = time.time()
        
        # Save optimized datasets with versioning
        seifa_v1_path = storage_manager.save_with_metadata(
            optimized_seifa,
            mock_data_paths["parquet_dir"] / "workflow_seifa_v1.parquet",
            {"version": "1.0", "stage": "optimized", "workflow": "end_to_end"},
            data_type="seifa"
        )
        
        health_v1_path = storage_manager.save_with_metadata(
            optimized_health,
            mock_data_paths["parquet_dir"] / "workflow_health_v1.parquet", 
            {"version": "1.0", "stage": "optimized", "workflow": "end_to_end"},
            data_type="health"
        )
        
        boundaries_v1_path = storage_manager.save_with_metadata(
            optimized_boundaries,
            mock_data_paths["parquet_dir"] / "workflow_boundaries_v1.parquet",
            {"version": "1.0", "stage": "optimized", "workflow": "end_to_end"},
            data_type="geographic"
        )
        
        stage3_time = time.time() - stage3_start
        
        # WORKFLOW STAGE 4: Analytics Processing
        stage4_start = time.time()
        
        # Aggregate health data for analytics
        aggregated_health = health_processor._aggregate_by_sa2(optimized_health)
        enhanced_boundaries = boundary_processor._calculate_population_density(optimized_boundaries)
        
        # Calculate risk assessments
        seifa_risk = risk_calculator._calculate_seifa_risk_score(optimized_seifa)
        health_risk = risk_calculator._calculate_health_utilisation_risk(aggregated_health)
        geographic_risk = risk_calculator._calculate_geographic_accessibility_risk(enhanced_boundaries)
        
        # Integrate risk components
        comprehensive_risk = seifa_risk.join(
            health_risk, left_on="sa2_code_2021", right_on="sa2_code", how="inner"
        ).join(
            geographic_risk, on="sa2_code_2021", how="inner"
        )
        
        composite_risk = risk_calculator._calculate_composite_risk_score(comprehensive_risk)
        final_risk = risk_calculator._classify_risk_categories(composite_risk)
        
        stage4_time = time.time() - stage4_start
        
        # WORKFLOW STAGE 5: Final Storage and Incremental Setup
        stage5_start = time.time()
        
        # Save analytics results
        analytics_path = storage_manager.save_with_metadata(
            final_risk,
            mock_data_paths["parquet_dir"] / "workflow_analytics_v1.parquet",
            {"version": "1.0", "stage": "analytics", "workflow": "end_to_end"},
            data_type="analytics"
        )
        
        # Test incremental capability
        additional_seifa = integrated_data["seifa"].tail(50)  # Take subset for incremental
        processed_additional = seifa_processor._validate_seifa_data(additional_seifa)
        optimized_additional = memory_optimizer.optimize_data_types(processed_additional, data_category="seifa")
        
        incremental_seifa = incremental_processor.merge_incremental_data(
            existing_data=optimized_seifa,
            new_data=optimized_additional,
            key_column="sa2_code_2021", 
            strategy="upsert"
        )
        
        stage5_time = time.time() - stage5_start
        
        total_workflow_time = stage1_time + stage2_time + stage3_time + stage4_time + stage5_time
        
        # WORKFLOW VALIDATION
        
        # 1. All stages should complete successfully
        assert seifa_v1_path.exists() and health_v1_path.exists() and boundaries_v1_path.exists()
        assert analytics_path.exists()
        
        # 2. Data integrity maintained throughout workflow
        final_analytics = pl.read_parquet(analytics_path)
        assert len(final_analytics) > 0
        assert "risk_category" in final_analytics.columns
        assert "composite_risk_score" in final_analytics.columns
        
        # 3. Performance targets met
        assert total_workflow_time < 120.0, f"Total workflow took {total_workflow_time:.1f}s, expected <120s"
        assert stage1_time < 30.0, f"Processing stage took {stage1_time:.1f}s, expected <30s"
        assert stage2_time < 15.0, f"Optimization stage took {stage2_time:.1f}s, expected <15s"
        assert stage3_time < 30.0, f"Storage stage took {stage3_time:.1f}s, expected <30s"
        assert stage4_time < 30.0, f"Analytics stage took {stage4_time:.1f}s, expected <30s"
        assert stage5_time < 15.0, f"Final stage took {stage5_time:.1f}s, expected <15s"
        
        # 4. Component coordination validation
        # Risk assessment should have reasonable success rate
        risk_success_rate = len(final_analytics) / len(processed_seifa)
        assert risk_success_rate > 0.70, f"Risk assessment success rate {risk_success_rate:.1%} should be >70%"
        
        # 5. Incremental functionality
        assert len(incremental_seifa) >= len(optimized_seifa), "Incremental processing should maintain or grow dataset"
        
        # 6. Data quality validation
        risk_categories = final_analytics["risk_category"].drop_nulls().unique().to_list()
        expected_categories = ["Very Low", "Low", "Medium", "High", "Very High"]
        assert all(cat in expected_categories for cat in risk_categories)
        
        risk_scores = final_analytics["composite_risk_score"].drop_nulls()
        if len(risk_scores) > 0:
            assert risk_scores.min() >= 0 and risk_scores.max() <= 100
        
        # Generate comprehensive workflow report
        workflow_report = {
            "total_workflow_time": total_workflow_time,
            "stage_1_processing_time": stage1_time,
            "stage_2_optimization_time": stage2_time,
            "stage_3_storage_time": stage3_time,
            "stage_4_analytics_time": stage4_time,
            "stage_5_incremental_time": stage5_time,
            "initial_records_processed": len(processed_seifa),
            "final_analytics_generated": len(final_analytics),
            "risk_assessment_success_rate": risk_success_rate,
            "incremental_capability_tested": True,
            "all_components_coordinated": True,
            "data_integrity_maintained": True,
            "performance_targets_met": True,
            "files_created": 4,  # seifa, health, boundaries, analytics
            "workflow_success": True
        }
        
        logging.info(f"End-to-End Workflow Report: {workflow_report}")
        
        return workflow_report