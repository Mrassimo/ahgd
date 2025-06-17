"""
Complete pipeline integration tests for Australian Health Analytics platform.

Tests the full end-to-end data processing pipeline from raw Australian government
data through to analytics-ready outputs, validating all components work together
seamlessly with real data volumes and patterns.

Key integration scenarios:
- Raw data ingestion → Processing → Storage optimization → Analytics
- ABS Census, SEIFA, PBS, and geographic boundary data integration
- Bronze-Silver-Gold data lake transitions with 497,181+ records
- Data lineage tracking and incremental processing
- Performance validation under realistic Australian health data loads
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

from src.data_processing.seifa_processor import SEIFAProcessor
from src.data_processing.health_processor import HealthDataProcessor
from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor
from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from src.data_processing.storage.incremental_processor import IncrementalProcessor
from src.analysis.risk.health_risk_calculator import HealthRiskCalculator
from src.analysis.risk.healthcare_access_scorer import HealthcareAccessScorer


class TestCompletePipelineIntegration:
    """End-to-end pipeline integration tests with real Australian data patterns."""
    
    def test_complete_health_analytics_pipeline(self, mock_excel_seifa_file, mock_health_data, 
                                              mock_boundary_data, mock_data_paths):
        """Test complete pipeline: raw data → bronze → silver → gold → analytics."""
        
        # Initialize pipeline components
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        memory_optimizer = MemoryOptimizer()
        risk_calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Create data lake structure
        bronze_dir = mock_data_paths["parquet_dir"] / "bronze"
        silver_dir = mock_data_paths["parquet_dir"] / "silver"
        gold_dir = mock_data_paths["parquet_dir"] / "gold"
        
        for dir_path in [bronze_dir, silver_dir, gold_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # BRONZE LAYER: Raw data ingestion and basic validation
        # Create realistic-sized datasets (simulating real Australian data volumes)
        start_time = time.time()
        
        # Process SEIFA data (2,454 SA2 areas)
        excel_file = mock_excel_seifa_file(num_areas=2454, include_errors=True)
        expected_seifa_path = seifa_processor.raw_dir / "SEIFA_2021_SA2_Indexes.xlsx"
        shutil.copy(excel_file, expected_seifa_path)
        
        bronze_seifa = seifa_processor.process_seifa_file()
        bronze_seifa_path = storage_manager.save_optimized_parquet(
            bronze_seifa, 
            bronze_dir / "seifa_2021_sa2.parquet",
            data_type="seifa"
        )
        
        # Process health data (492,434 PBS prescription records)
        bronze_health = mock_health_data(num_records=50000, num_sa2_areas=2454)  # Scaled for testing
        bronze_health_path = storage_manager.save_optimized_parquet(
            bronze_health,
            bronze_dir / "pbs_prescriptions_2023.parquet",
            data_type="health"
        )
        
        # Process geographic boundaries (96MB+ shapefiles equivalent)
        bronze_boundaries = mock_boundary_data(num_areas=2454)
        bronze_boundaries_path = storage_manager.save_optimized_parquet(
            bronze_boundaries,
            bronze_dir / "sa2_boundaries_2021.parquet",
            data_type="geographic"
        )
        
        bronze_time = time.time() - start_time
        
        # SILVER LAYER: Data quality improvements and standardisation
        silver_start = time.time()
        
        # Load bronze data
        bronze_seifa_df = pl.read_parquet(bronze_seifa_path)
        bronze_health_df = pl.read_parquet(bronze_health_path)
        bronze_boundaries_df = pl.read_parquet(bronze_boundaries_path)
        
        # Apply data quality improvements
        silver_seifa = seifa_processor._validate_seifa_data(bronze_seifa_df)
        silver_health = health_processor._validate_health_data(bronze_health_df)
        silver_boundaries = boundary_processor._validate_boundary_data(bronze_boundaries_df)
        
        # Standardise SA2 codes across datasets
        valid_sa2_codes = list(set(silver_seifa["sa2_code_2021"].to_list()) & 
                              set(silver_boundaries["sa2_code_2021"].to_list()))
        
        # Filter to consistent SA2 codes
        silver_seifa = silver_seifa.filter(pl.col("sa2_code_2021").is_in(valid_sa2_codes))
        silver_boundaries = silver_boundaries.filter(pl.col("sa2_code_2021").is_in(valid_sa2_codes))
        silver_health = silver_health.filter(pl.col("sa2_code").is_in(valid_sa2_codes))
        
        # Apply memory optimizations
        silver_seifa = memory_optimizer.optimize_data_types(silver_seifa, data_category="seifa")
        silver_health = memory_optimizer.optimize_data_types(silver_health, data_category="health")
        silver_boundaries = memory_optimizer.optimize_data_types(silver_boundaries, data_category="geographic")
        
        # Save silver layer
        silver_seifa_path = storage_manager.save_optimized_parquet(
            silver_seifa, silver_dir / "seifa_validated.parquet", data_type="seifa"
        )
        silver_health_path = storage_manager.save_optimized_parquet(
            silver_health, silver_dir / "health_validated.parquet", data_type="health"
        )
        silver_boundaries_path = storage_manager.save_optimized_parquet(
            silver_boundaries, silver_dir / "boundaries_validated.parquet", data_type="geographic"
        )
        
        silver_time = time.time() - silver_start
        
        # GOLD LAYER: Analytics-ready datasets and derived metrics
        gold_start = time.time()
        
        # Aggregate health data by SA2
        gold_health = health_processor._aggregate_by_sa2(silver_health)
        
        # Calculate population density
        gold_boundaries = boundary_processor._calculate_population_density(silver_boundaries)
        
        # Create integrated dataset
        integrated_data = silver_seifa.join(
            gold_health, 
            left_on="sa2_code_2021", 
            right_on="sa2_code", 
            how="left"
        ).join(
            gold_boundaries, 
            on="sa2_code_2021", 
            how="left"
        )
        
        # Calculate risk assessments
        seifa_risk = risk_calculator._calculate_seifa_risk_score(silver_seifa)
        health_risk = risk_calculator._calculate_health_utilisation_risk(gold_health)
        geographic_risk = risk_calculator._calculate_geographic_accessibility_risk(gold_boundaries)
        
        # Create comprehensive risk assessment
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
        
        composite_risk = risk_calculator._calculate_composite_risk_score(comprehensive_risk)
        classified_risk = risk_calculator._classify_risk_categories(composite_risk)
        
        # Save gold layer
        gold_integrated_path = storage_manager.save_optimized_parquet(
            integrated_data, gold_dir / "integrated_health_analytics.parquet", data_type="analytics"
        )
        gold_risk_path = storage_manager.save_optimized_parquet(
            classified_risk, gold_dir / "health_risk_assessment.parquet", data_type="analytics"
        )
        
        gold_time = time.time() - gold_start
        total_time = time.time() - start_time
        
        # VALIDATION: End-to-end pipeline validation
        
        # 1. Verify all layers exist and contain data
        assert bronze_seifa_path.exists() and bronze_health_path.exists() and bronze_boundaries_path.exists()
        assert silver_seifa_path.exists() and silver_health_path.exists() and silver_boundaries_path.exists()
        assert gold_integrated_path.exists() and gold_risk_path.exists()
        
        # Load final outputs for validation
        final_integrated = pl.read_parquet(gold_integrated_path)
        final_risk = pl.read_parquet(gold_risk_path)
        
        # 2. Verify data quality and completeness
        assert len(final_integrated) > 2000, "Should retain majority of SA2 areas through pipeline"
        assert len(final_risk) > 1000, "Should have substantial risk assessments"
        
        # 3. Verify data consistency across pipeline
        expected_seifa_cols = ["sa2_code_2021", "irsd_decile", "irsd_score", "irsad_decile", "usual_resident_population"]
        for col in expected_seifa_cols:
            if col in final_integrated.columns:
                assert final_integrated[col].null_count() < len(final_integrated) * 0.1, f"Too many nulls in {col}"
        
        # 4. Verify Australian health data compliance
        # SA2 codes should be valid 9-digit Australian codes
        sa2_codes = final_integrated["sa2_code_2021"].drop_nulls().to_list()
        valid_sa2_pattern = all(
            len(code) == 9 and code.isdigit() and code[0] in "12345678" 
            for code in sa2_codes[:100]  # Sample check
        )
        assert valid_sa2_pattern, "SA2 codes should follow Australian 9-digit pattern"
        
        # SEIFA deciles should be 1-10
        seifa_deciles = final_integrated["irsd_decile"].drop_nulls()
        if len(seifa_deciles) > 0:
            assert seifa_deciles.min() >= 1 and seifa_deciles.max() <= 10
        
        # Risk categories should be valid
        risk_categories = final_risk["risk_category"].drop_nulls().unique().to_list()
        expected_categories = ["Very Low", "Low", "Medium", "High", "Very High"]
        assert all(cat in expected_categories for cat in risk_categories)
        
        # 5. Verify performance targets
        assert total_time < 300.0, f"Complete pipeline took {total_time:.1f}s, expected <5 minutes"
        assert bronze_time < 120.0, f"Bronze layer took {bronze_time:.1f}s, expected <2 minutes"
        assert silver_time < 90.0, f"Silver layer took {silver_time:.1f}s, expected <90 seconds"
        assert gold_time < 90.0, f"Gold layer took {gold_time:.1f}s, expected <90 seconds"
        
        # 6. Verify data integration success rate (target: 92.9%)
        integration_success_rate = len(final_integrated) / len(bronze_seifa_df)
        assert integration_success_rate > 0.85, f"Integration success rate {integration_success_rate:.1%}, expected >85%"
        
        # 7. Verify storage optimization
        bronze_size = sum(path.stat().st_size for path in [bronze_seifa_path, bronze_health_path, bronze_boundaries_path])
        gold_size = sum(path.stat().st_size for path in [gold_integrated_path, gold_risk_path])
        
        # Should have reasonable compression (not necessarily smaller due to derived metrics)
        compression_ratio = gold_size / bronze_size
        assert compression_ratio < 2.0, f"Storage growth {compression_ratio:.1f}x, should be <2x for analytics datasets"
        
        # 8. Generate pipeline summary report
        pipeline_report = {
            "pipeline_execution_time": total_time,
            "bronze_processing_time": bronze_time,
            "silver_processing_time": silver_time,
            "gold_processing_time": gold_time,
            "total_sa2_areas_processed": len(final_integrated),
            "health_records_processed": len(bronze_health_df),
            "integration_success_rate": integration_success_rate,
            "data_quality_score": 1.0 - (final_integrated.null_count().sum() / (len(final_integrated) * len(final_integrated.columns))),
            "storage_compression_ratio": compression_ratio,
            "risk_assessments_generated": len(final_risk)
        }
        
        # Log pipeline performance metrics
        logging.info(f"Pipeline Integration Report: {pipeline_report}")
        
        return pipeline_report
    
    def test_cross_dataset_sa2_integration(self, integration_test_data, mock_data_paths):
        """Validate SA2 codes align consistently across all datasets."""
        
        # Create comprehensive test dataset
        integrated_data = integration_test_data(num_sa2_areas=500, num_health_records=2500)
        
        seifa_df = integrated_data["seifa"]
        health_df = integrated_data["health"]
        boundaries_df = integrated_data["boundaries"]
        expected_sa2_codes = set(integrated_data["sa2_codes"])
        
        # Initialize processors
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Process through validation pipelines
        validated_seifa = seifa_processor._validate_seifa_data(seifa_df)
        validated_health = health_processor._validate_pbs_data(health_df)
        validated_boundaries = boundary_processor._validate_boundary_data(boundaries_df)
        
        # Extract SA2 codes from each dataset
        seifa_sa2_codes = set(validated_seifa["sa2_code_2021"].drop_nulls().to_list())
        health_sa2_codes = set(validated_health["sa2_code"].drop_nulls().to_list())
        boundary_sa2_codes = set(validated_boundaries["sa2_code_2021"].drop_nulls().to_list())
        
        # Validate SA2 code consistency
        # 1. All datasets should have substantial overlap
        seifa_health_overlap = len(seifa_sa2_codes & health_sa2_codes) / len(seifa_sa2_codes)
        seifa_boundary_overlap = len(seifa_sa2_codes & boundary_sa2_codes) / len(seifa_sa2_codes)
        health_boundary_overlap = len(health_sa2_codes & boundary_sa2_codes) / len(health_sa2_codes)
        
        assert seifa_health_overlap > 0.90, f"SEIFA-Health SA2 overlap {seifa_health_overlap:.1%}, expected >90%"
        assert seifa_boundary_overlap > 0.95, f"SEIFA-Boundary SA2 overlap {seifa_boundary_overlap:.1%}, expected >95%"
        assert health_boundary_overlap > 0.90, f"Health-Boundary SA2 overlap {health_boundary_overlap:.1%}, expected >90%"
        
        # 2. SA2 codes should follow Australian standards
        all_sa2_codes = seifa_sa2_codes | health_sa2_codes | boundary_sa2_codes
        
        for sa2_code in list(all_sa2_codes)[:50]:  # Sample validation
            assert len(sa2_code) == 9, f"SA2 code {sa2_code} should be 9 digits"
            assert sa2_code.isdigit(), f"SA2 code {sa2_code} should be numeric"
            assert sa2_code[0] in "12345678", f"SA2 code {sa2_code} should start with valid state prefix"
        
        # 3. Cross-dataset integration should work seamlessly
        # Join datasets by SA2 code
        seifa_health_join = validated_seifa.join(
            validated_health.group_by("sa2_code").first(),
            left_on="sa2_code_2021",
            right_on="sa2_code",
            how="inner"
        )
        
        seifa_boundary_join = validated_seifa.join(
            validated_boundaries,
            on="sa2_code_2021",
            how="inner"
        )
        
        # Should retain substantial data after joins
        assert len(seifa_health_join) > len(validated_seifa) * 0.8, "SEIFA-Health join should retain >80% of data"
        assert len(seifa_boundary_join) > len(validated_seifa) * 0.9, "SEIFA-Boundary join should retain >90% of data"
        
        # 4. Validate geographic distribution
        # SA2 codes should represent different states
        state_prefixes = {code[0] for code in all_sa2_codes}
        assert len(state_prefixes) >= 3, "Should have SA2 codes from multiple Australian states"
        
        # 5. Create comprehensive SA2 mapping report
        sa2_integration_report = {
            "total_unique_sa2_codes": len(all_sa2_codes),
            "seifa_sa2_count": len(seifa_sa2_codes),
            "health_sa2_count": len(health_sa2_codes),
            "boundary_sa2_count": len(boundary_sa2_codes),
            "seifa_health_overlap_rate": seifa_health_overlap,
            "seifa_boundary_overlap_rate": seifa_boundary_overlap,
            "health_boundary_overlap_rate": health_boundary_overlap,
            "states_represented": list(state_prefixes),
            "integration_feasibility": min(seifa_health_overlap, seifa_boundary_overlap, health_boundary_overlap)
        }
        
        logging.info(f"SA2 Integration Report: {sa2_integration_report}")
        
        return sa2_integration_report
    
    def test_pipeline_with_incremental_updates(self, mock_seifa_data, mock_health_data, mock_data_paths):
        """Test adding new data to existing pipeline with incremental processing."""
        
        # Initialize incremental processor
        incremental_processor = IncrementalProcessor(base_path=mock_data_paths["parquet_dir"])
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        
        # Create initial dataset (baseline)
        initial_seifa = mock_seifa_data(num_areas=1000)
        initial_health = mock_health_data(num_records=5000, num_sa2_areas=1000)
        
        # Save initial datasets with version metadata
        initial_seifa_path = storage_manager.save_with_metadata(
            initial_seifa,
            mock_data_paths["parquet_dir"] / "seifa_v1.parquet",
            {"version": "1.0", "created_date": datetime.now().isoformat()},
            data_type="seifa"
        )
        
        initial_health_path = storage_manager.save_with_metadata(
            initial_health,
            mock_data_paths["parquet_dir"] / "health_v1.parquet",
            {"version": "1.0", "created_date": datetime.now().isoformat()},
            data_type="health"
        )
        
        # Simulate time passing
        time.sleep(0.1)
        
        # Create incremental update (new data)
        # Add new SA2 areas and update existing ones
        new_sa2_codes = [f"8{str(i).zfill(8)}" for i in range(1001, 1201)]  # 200 new SA2 areas
        updated_seifa = mock_seifa_data(num_areas=200)
        updated_seifa = updated_seifa.with_columns(
            pl.Series("sa2_code_2021", new_sa2_codes)
        )
        
        # Add health data for new and existing SA2 areas
        existing_sa2_codes = initial_seifa["sa2_code_2021"].to_list()[:500]  # Update half of existing
        all_update_codes = new_sa2_codes + existing_sa2_codes
        
        incremental_health = mock_health_data(num_records=3000, num_sa2_areas=len(all_update_codes))
        incremental_health = incremental_health.with_columns(
            pl.col("sa2_code").map_elements(
                lambda _: np.random.choice(all_update_codes), 
                return_dtype=pl.Utf8
            )
        )
        
        # Process incremental updates
        start_time = time.time()
        
        # Update SEIFA data
        combined_seifa = incremental_processor.merge_incremental_data(
            existing_data=pl.read_parquet(initial_seifa_path),
            new_data=updated_seifa,
            key_column="sa2_code_2021",
            strategy="upsert"
        )
        
        # Update health data
        combined_health = incremental_processor.merge_incremental_data(
            existing_data=pl.read_parquet(initial_health_path),
            new_data=incremental_health,
            key_column="sa2_code",
            strategy="append"
        )
        
        # Save updated datasets
        updated_seifa_path = storage_manager.save_with_metadata(
            combined_seifa,
            mock_data_paths["parquet_dir"] / "seifa_v2.parquet",
            {"version": "2.0", "created_date": datetime.now().isoformat(), "incremental_update": True},
            data_type="seifa"
        )
        
        updated_health_path = storage_manager.save_with_metadata(
            combined_health,
            mock_data_paths["parquet_dir"] / "health_v2.parquet",
            {"version": "2.0", "created_date": datetime.now().isoformat(), "incremental_update": True},
            data_type="health"
        )
        
        processing_time = time.time() - start_time
        
        # Validate incremental processing
        # 1. Verify data growth
        assert len(combined_seifa) > len(initial_seifa), "SEIFA data should grow with incremental updates"
        assert len(combined_health) > len(initial_health), "Health data should grow with incremental updates"
        
        # 2. Verify new SA2 areas were added
        final_sa2_codes = set(combined_seifa["sa2_code_2021"].to_list())
        initial_sa2_codes = set(initial_seifa["sa2_code_2021"].to_list())
        added_sa2_codes = final_sa2_codes - initial_sa2_codes
        
        assert len(added_sa2_codes) == 200, f"Should add exactly 200 new SA2 areas, added {len(added_sa2_codes)}"
        
        # 3. Verify existing data integrity
        # Check that existing SA2 areas still exist
        retained_sa2_codes = final_sa2_codes & initial_sa2_codes
        assert len(retained_sa2_codes) >= len(initial_sa2_codes) * 0.95, "Should retain >95% of existing SA2 areas"
        
        # 4. Verify incremental performance
        assert processing_time < 30.0, f"Incremental processing took {processing_time:.1f}s, expected <30s"
        
        # 5. Test data lineage and versioning
        # Should be able to load both versions
        v1_seifa = pl.read_parquet(initial_seifa_path)
        v2_seifa = pl.read_parquet(updated_seifa_path)
        
        assert len(v2_seifa) > len(v1_seifa), "V2 should have more records than V1"
        
        # 6. Test rollback capability
        # Should be able to revert to previous version
        rollback_data = incremental_processor.rollback_to_version(
            current_data=combined_seifa,
            previous_data=initial_seifa,
            key_column="sa2_code_2021"
        )
        
        assert len(rollback_data) == len(initial_seifa), "Rollback should restore original data size"
        
        # Generate incremental processing report
        incremental_report = {
            "processing_time": processing_time,
            "initial_seifa_records": len(initial_seifa),
            "final_seifa_records": len(combined_seifa),
            "initial_health_records": len(initial_health),
            "final_health_records": len(combined_health),
            "new_sa2_areas_added": len(added_sa2_codes),
            "sa2_retention_rate": len(retained_sa2_codes) / len(initial_sa2_codes),
            "data_growth_rate": (len(combined_seifa) - len(initial_seifa)) / len(initial_seifa),
            "versioning_enabled": True,
            "rollback_tested": True
        }
        
        logging.info(f"Incremental Processing Report: {incremental_report}")
        
        return incremental_report
    
    def test_pipeline_performance_at_scale(self, mock_excel_seifa_file, mock_health_data, 
                                         mock_boundary_data, mock_data_paths):
        """Test pipeline performance with realistic Australian health data volumes."""
        
        # Create realistic-scale datasets
        # Target: Process 497,181+ records end-to-end in <5 minutes
        
        # Initialize performance monitoring
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create large-scale datasets
        large_seifa = mock_excel_seifa_file(num_areas=2454)  # All Australian SA2 areas
        large_health = mock_health_data(num_records=100000, num_sa2_areas=2454)  # Scaled health data
        large_boundaries = mock_boundary_data(num_areas=2454)  # All SA2 boundaries
        
        # Initialize all pipeline components
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        memory_optimizer = MemoryOptimizer()
        risk_calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        
        # Stage 1: Data ingestion and validation (Bronze layer)
        stage1_start = time.time()
        
        # Set up SEIFA file
        expected_seifa_path = seifa_processor.raw_dir / "SEIFA_2021_SA2_Indexes.xlsx"
        shutil.copy(large_seifa, expected_seifa_path)
        
        # Process all datasets
        processed_seifa = seifa_processor.process_seifa_file()
        validated_health = health_processor._validate_health_data(large_health)
        validated_boundaries = boundary_processor._validate_boundary_data(large_boundaries)
        
        stage1_time = time.time() - stage1_start
        stage1_memory = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
        
        # Stage 2: Data optimization and standardisation (Silver layer)
        stage2_start = time.time()
        
        # Apply memory optimizations
        optimized_seifa = memory_optimizer.optimize_data_types(processed_seifa, data_category="seifa")
        optimized_health = memory_optimizer.optimize_data_types(validated_health, data_category="health")
        optimized_boundaries = memory_optimizer.optimize_data_types(validated_boundaries, data_category="geographic")
        
        # Aggregate health data
        aggregated_health = health_processor._aggregate_by_sa2(optimized_health)
        
        # Calculate enhanced metrics
        enhanced_boundaries = boundary_processor._calculate_population_density(optimized_boundaries)
        
        stage2_time = time.time() - stage2_start
        stage2_memory = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
        
        # Stage 3: Analytics and risk assessment (Gold layer)
        stage3_start = time.time()
        
        # Calculate risk components
        seifa_risk = risk_calculator._calculate_seifa_risk_score(optimized_seifa)
        health_risk = risk_calculator._calculate_health_utilisation_risk(aggregated_health)
        geographic_risk = risk_calculator._calculate_geographic_accessibility_risk(enhanced_boundaries)
        
        # Create comprehensive risk assessment
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
        
        composite_risk = risk_calculator._calculate_composite_risk_score(comprehensive_risk)
        classified_risk = risk_calculator._classify_risk_categories(composite_risk)
        
        stage3_time = time.time() - stage3_start
        stage3_memory = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
        
        # Stage 4: Storage optimization and persistence
        stage4_start = time.time()
        
        # Save optimized datasets
        seifa_path = storage_manager.save_optimized_parquet(
            optimized_seifa, 
            mock_data_paths["parquet_dir"] / "scale_test_seifa.parquet",
            data_type="seifa"
        )
        health_path = storage_manager.save_optimized_parquet(
            aggregated_health,
            mock_data_paths["parquet_dir"] / "scale_test_health.parquet",
            data_type="health"
        )
        boundaries_path = storage_manager.save_optimized_parquet(
            enhanced_boundaries,
            mock_data_paths["parquet_dir"] / "scale_test_boundaries.parquet", 
            data_type="geographic"
        )
        risk_path = storage_manager.save_optimized_parquet(
            classified_risk,
            mock_data_paths["parquet_dir"] / "scale_test_risk.parquet",
            data_type="analytics"
        )
        
        stage4_time = time.time() - stage4_start
        
        total_time = time.time() - start_time
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
        
        # Performance validation
        # 1. Total pipeline execution time should be <5 minutes
        assert total_time < 300.0, f"Pipeline took {total_time:.1f}s, expected <300s (5 minutes)"
        
        # 2. Each stage should meet performance targets
        assert stage1_time < 120.0, f"Stage 1 (ingestion) took {stage1_time:.1f}s, expected <120s"
        assert stage2_time < 90.0, f"Stage 2 (optimization) took {stage2_time:.1f}s, expected <90s"
        assert stage3_time < 60.0, f"Stage 3 (analytics) took {stage3_time:.1f}s, expected <60s"
        assert stage4_time < 30.0, f"Stage 4 (storage) took {stage4_time:.1f}s, expected <30s"
        
        # 3. Memory usage should be reasonable
        assert peak_memory < 2048, f"Peak memory usage {peak_memory:.1f}MB, expected <2GB"
        
        # 4. Data volume validation
        total_records_processed = len(processed_seifa) + len(validated_health) + len(validated_boundaries)
        assert total_records_processed > 100000, f"Processed {total_records_processed} records, expected >100k"
        
        # 5. Integration success rate should be high
        integration_success_rate = len(classified_risk) / len(processed_seifa)
        assert integration_success_rate > 0.80, f"Integration success rate {integration_success_rate:.1%}, expected >80%"
        
        # 6. Storage efficiency validation
        total_file_size = sum(path.stat().st_size for path in [seifa_path, health_path, boundaries_path, risk_path])
        storage_efficiency = total_records_processed / (total_file_size / 1024 / 1024)  # Records per MB
        assert storage_efficiency > 100, f"Storage efficiency {storage_efficiency:.1f} records/MB, expected >100"
        
        # Generate comprehensive performance report
        performance_report = {
            "total_execution_time": total_time,
            "stage_1_ingestion_time": stage1_time,
            "stage_2_optimization_time": stage2_time, 
            "stage_3_analytics_time": stage3_time,
            "stage_4_storage_time": stage4_time,
            "peak_memory_usage_mb": peak_memory,
            "total_records_processed": total_records_processed,
            "seifa_records": len(processed_seifa),
            "health_records": len(validated_health),
            "boundary_records": len(validated_boundaries),
            "risk_assessments": len(classified_risk),
            "integration_success_rate": integration_success_rate,
            "storage_efficiency_records_per_mb": storage_efficiency,
            "total_storage_size_mb": total_file_size / 1024 / 1024,
            "throughput_records_per_second": total_records_processed / total_time,
            "meets_performance_targets": True
        }
        
        logging.info(f"Scale Performance Report: {performance_report}")
        
        return performance_report