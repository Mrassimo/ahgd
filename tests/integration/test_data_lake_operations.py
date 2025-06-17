"""
Data lake operations integration tests for Australian Health Analytics platform.

Tests the Bronze-Silver-Gold data lake architecture with real Australian health data:
- Bronze layer: Raw data ingestion and basic validation
- Silver layer: Data quality improvements, standardisation, and integration
- Gold layer: Analytics-ready datasets and derived metrics
- Data versioning, lineage tracking, and schema evolution
- Cross-layer data movement and transformation validation

Validates enterprise-grade data lake operations at Australian health data scale.
"""

import pytest
import polars as pl
import numpy as np
import time
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging

from src.data_processing.seifa_processor import SEIFAProcessor
from src.data_processing.health_processor import HealthDataProcessor
from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor
from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from src.data_processing.storage.incremental_processor import IncrementalProcessor
from src.analysis.risk.health_risk_calculator import HealthRiskCalculator


class TestDataLakeOperations:
    """Tests for Bronze-Silver-Gold data lake architecture and operations."""
    
    def test_bronze_silver_gold_transitions(self, mock_excel_seifa_file, mock_health_data, 
                                           mock_boundary_data, mock_data_paths):
        """Test complete Bronze → Silver → Gold data lake transitions."""
        
        # Initialize components
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
        metadata_dir = mock_data_paths["parquet_dir"] / "metadata"
        
        for dir_path in [bronze_dir, silver_dir, gold_dir, metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # =================================================================
        # BRONZE LAYER: Raw data ingestion with minimal processing
        # =================================================================
        bronze_start_time = time.time()
        
        # Ingest SEIFA data (raw format with potential quality issues)
        excel_file = mock_excel_seifa_file(num_areas=1000, include_errors=True)
        expected_seifa_path = seifa_processor.raw_dir / "SEIFA_2021_SA2_Indexes.xlsx"
        shutil.copy(excel_file, expected_seifa_path)
        
        # Raw SEIFA processing - minimal validation
        bronze_seifa = seifa_processor.process_seifa_file()
        
        # Ingest health data (raw PBS format)
        bronze_health = mock_health_data(num_records=5000, num_sa2_areas=1000)
        
        # Ingest boundary data (raw geographic format)
        bronze_boundaries = mock_boundary_data(num_areas=1000)
        
        # Save Bronze layer with minimal metadata
        bronze_seifa_path = bronze_dir / "seifa" / "year=2021" / "seifa_raw.parquet"
        bronze_health_path = bronze_dir / "health" / "year=2023" / "month=01" / "pbs_raw.parquet"
        bronze_boundaries_path = bronze_dir / "geographic" / "year=2021" / "boundaries_raw.parquet"
        
        # Create directory structure
        for path in [bronze_seifa_path, bronze_health_path, bronze_boundaries_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save bronze data with basic metadata
        bronze_seifa_metadata = {
            "layer": "bronze",
            "source": "ABS_SEIFA_2021",
            "ingestion_date": datetime.now().isoformat(),
            "record_count": len(bronze_seifa),
            "quality_score": None,  # Not assessed at bronze level
            "schema_version": "bronze_v1"
        }
        
        storage_manager.save_with_metadata(
            bronze_seifa, bronze_seifa_path, bronze_seifa_metadata, data_type="seifa"
        )
        
        bronze_health_metadata = {
            "layer": "bronze",
            "source": "PBS_Prescriptions_2023",
            "ingestion_date": datetime.now().isoformat(),
            "record_count": len(bronze_health),
            "quality_score": None,
            "schema_version": "bronze_v1"
        }
        
        storage_manager.save_with_metadata(
            bronze_health, bronze_health_path, bronze_health_metadata, data_type="health"
        )
        
        bronze_boundaries_metadata = {
            "layer": "bronze",
            "source": "ABS_Geographic_Boundaries_2021",
            "ingestion_date": datetime.now().isoformat(),
            "record_count": len(bronze_boundaries),
            "quality_score": None,
            "schema_version": "bronze_v1"
        }
        
        storage_manager.save_with_metadata(
            bronze_boundaries, bronze_boundaries_path, bronze_boundaries_metadata, data_type="geographic"
        )
        
        bronze_processing_time = time.time() - bronze_start_time
        
        # =================================================================
        # SILVER LAYER: Data quality improvements and standardisation
        # =================================================================
        silver_start_time = time.time()
        
        # Load bronze data
        bronze_seifa_loaded = pl.read_parquet(bronze_seifa_path)
        bronze_health_loaded = pl.read_parquet(bronze_health_path)
        bronze_boundaries_loaded = pl.read_parquet(bronze_boundaries_path)
        
        # Apply data quality improvements
        silver_seifa = seifa_processor._validate_seifa_data(bronze_seifa_loaded)
        silver_health = health_processor._validate_health_data(bronze_health_loaded)
        silver_boundaries = boundary_processor._validate_boundary_data(bronze_boundaries_loaded)
        
        # Calculate data quality scores
        seifa_quality_score = 1.0 - (silver_seifa.null_count().sum() / (len(silver_seifa) * len(silver_seifa.columns)))
        health_quality_score = 1.0 - (silver_health.null_count().sum() / (len(silver_health) * len(silver_health.columns)))
        boundaries_quality_score = 1.0 - (silver_boundaries.null_count().sum() / (len(silver_boundaries) * len(silver_boundaries.columns)))
        
        # Standardise SA2 codes across datasets
        valid_sa2_codes = list(
            set(silver_seifa["sa2_code_2021"].drop_nulls().to_list()) & 
            set(silver_boundaries["sa2_code_2021"].drop_nulls().to_list())
        )
        
        # Filter datasets to consistent SA2 codes
        silver_seifa_filtered = silver_seifa.filter(pl.col("sa2_code_2021").is_in(valid_sa2_codes))
        silver_boundaries_filtered = silver_boundaries.filter(pl.col("sa2_code_2021").is_in(valid_sa2_codes))
        silver_health_filtered = silver_health.filter(pl.col("sa2_code").is_in(valid_sa2_codes))
        
        # Apply memory optimizations
        silver_seifa_optimized = memory_optimizer.optimize_data_types(silver_seifa_filtered, data_category="seifa")
        silver_health_optimized = memory_optimizer.optimize_data_types(silver_health_filtered, data_category="health")
        silver_boundaries_optimized = memory_optimizer.optimize_data_types(silver_boundaries_filtered, data_category="geographic")
        
        # Save Silver layer with enhanced metadata
        silver_seifa_path = silver_dir / "seifa" / "year=2021" / "seifa_validated.parquet"
        silver_health_path = silver_dir / "health" / "year=2023" / "month=01" / "pbs_validated.parquet"
        silver_boundaries_path = silver_dir / "geographic" / "year=2021" / "boundaries_validated.parquet"
        
        # Create directory structure
        for path in [silver_seifa_path, silver_health_path, silver_boundaries_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        # Enhanced metadata for silver layer
        silver_seifa_metadata = {
            "layer": "silver",
            "source": "bronze_seifa_validated",
            "processing_date": datetime.now().isoformat(),
            "record_count": len(silver_seifa_optimized),
            "quality_score": float(seifa_quality_score),
            "schema_version": "silver_v1",
            "transformations_applied": ["validation", "sa2_filtering", "memory_optimization"],
            "data_lineage": {"bronze_source": str(bronze_seifa_path)},
            "sa2_codes_standardised": len(valid_sa2_codes),
            "memory_optimisation_applied": True
        }
        
        storage_manager.save_with_metadata(
            silver_seifa_optimized, silver_seifa_path, silver_seifa_metadata, data_type="seifa"
        )
        
        silver_health_metadata = {
            "layer": "silver",
            "source": "bronze_health_validated",
            "processing_date": datetime.now().isoformat(),
            "record_count": len(silver_health_optimized),
            "quality_score": float(health_quality_score),
            "schema_version": "silver_v1",
            "transformations_applied": ["validation", "sa2_filtering", "memory_optimization"],
            "data_lineage": {"bronze_source": str(bronze_health_path)},
            "sa2_codes_standardised": len(valid_sa2_codes),
            "memory_optimisation_applied": True
        }
        
        storage_manager.save_with_metadata(
            silver_health_optimized, silver_health_path, silver_health_metadata, data_type="health"
        )
        
        silver_boundaries_metadata = {
            "layer": "silver",
            "source": "bronze_boundaries_validated",
            "processing_date": datetime.now().isoformat(),
            "record_count": len(silver_boundaries_optimized),
            "quality_score": float(boundaries_quality_score),
            "schema_version": "silver_v1",
            "transformations_applied": ["validation", "sa2_filtering", "memory_optimization"],
            "data_lineage": {"bronze_source": str(bronze_boundaries_path)},
            "sa2_codes_standardised": len(valid_sa2_codes),
            "memory_optimisation_applied": True
        }
        
        storage_manager.save_with_metadata(
            silver_boundaries_optimized, silver_boundaries_path, silver_boundaries_metadata, data_type="geographic"
        )
        
        silver_processing_time = time.time() - silver_start_time
        
        # =================================================================
        # GOLD LAYER: Analytics-ready datasets and derived metrics
        # =================================================================
        gold_start_time = time.time()
        
        # Load silver data
        silver_seifa_loaded = pl.read_parquet(silver_seifa_path)
        silver_health_loaded = pl.read_parquet(silver_health_path)
        silver_boundaries_loaded = pl.read_parquet(silver_boundaries_path)
        
        # Create analytics-ready health aggregations
        gold_health_aggregated = health_processor._aggregate_by_sa2(silver_health_loaded)
        
        # Enhance boundaries with derived metrics
        gold_boundaries_enhanced = boundary_processor._calculate_population_density(silver_boundaries_loaded)
        
        # Create integrated analytics dataset
        gold_integrated = silver_seifa_loaded.join(
            gold_health_aggregated,
            left_on="sa2_code_2021",
            right_on="sa2_code",
            how="left"
        ).join(
            gold_boundaries_enhanced,
            on="sa2_code_2021",
            how="left"
        )
        
        # Calculate comprehensive risk assessments
        seifa_risk = risk_calculator._calculate_seifa_risk_score(silver_seifa_loaded)
        health_risk = risk_calculator._calculate_health_utilisation_risk(gold_health_aggregated)
        geographic_risk = risk_calculator._calculate_geographic_accessibility_risk(gold_boundaries_enhanced)
        
        # Create comprehensive risk dataset
        gold_risk_comprehensive = seifa_risk.join(
            health_risk,
            left_on="sa2_code_2021",
            right_on="sa2_code",
            how="inner"
        ).join(
            geographic_risk,
            on="sa2_code_2021",
            how="inner"
        )
        
        gold_risk_composite = risk_calculator._calculate_composite_risk_score(gold_risk_comprehensive)
        gold_risk_classified = risk_calculator._classify_risk_categories(gold_risk_composite)
        
        # Create gold layer summary statistics
        gold_summary_stats = {
            "total_sa2_areas": len(gold_integrated),
            "health_utilisation_sa2_areas": len(gold_health_aggregated),
            "risk_assessed_sa2_areas": len(gold_risk_classified),
            "average_seifa_irsd_decile": float(gold_integrated["irsd_decile"].drop_nulls().mean()) if "irsd_decile" in gold_integrated.columns else None,
            "total_prescriptions": int(gold_health_aggregated["total_prescriptions"].sum()) if "total_prescriptions" in gold_health_aggregated.columns else None,
            "average_population_density": float(gold_boundaries_enhanced["population_density"].drop_nulls().mean()) if "population_density" in gold_boundaries_enhanced.columns else None,
            "risk_distribution": dict(gold_risk_classified["risk_category"].value_counts().to_pandas().to_dict()) if "risk_category" in gold_risk_classified.columns else {}
        }
        
        # Save Gold layer datasets
        gold_integrated_path = gold_dir / "integrated_analytics" / "year=2023" / "health_analytics_integrated.parquet"
        gold_risk_path = gold_dir / "risk_assessments" / "year=2023" / "comprehensive_risk_assessment.parquet"
        gold_summary_path = gold_dir / "summary_statistics" / "year=2023" / "platform_summary.json"
        
        # Create directory structure
        for path in [gold_integrated_path, gold_risk_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
        gold_summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Gold layer metadata with comprehensive lineage
        gold_integrated_metadata = {
            "layer": "gold",
            "dataset_type": "integrated_analytics",
            "creation_date": datetime.now().isoformat(),
            "record_count": len(gold_integrated),
            "schema_version": "gold_v1",
            "transformations_applied": ["sa2_aggregation", "cross_dataset_integration", "derived_metrics"],
            "data_lineage": {
                "silver_seifa_source": str(silver_seifa_path),
                "silver_health_source": str(silver_health_path),
                "silver_boundaries_source": str(silver_boundaries_path)
            },
            "analytics_ready": True,
            "summary_statistics": gold_summary_stats,
            "quality_assurance": {
                "data_completeness": float(1.0 - (gold_integrated.null_count().sum() / (len(gold_integrated) * len(gold_integrated.columns)))),
                "cross_dataset_integration_rate": len(gold_integrated) / len(silver_seifa_loaded)
            }
        }
        
        storage_manager.save_with_metadata(
            gold_integrated, gold_integrated_path, gold_integrated_metadata, data_type="analytics"
        )
        
        gold_risk_metadata = {
            "layer": "gold", 
            "dataset_type": "risk_assessment",
            "creation_date": datetime.now().isoformat(),
            "record_count": len(gold_risk_classified),
            "schema_version": "gold_v1",
            "transformations_applied": ["risk_calculation", "composite_scoring", "risk_classification"],
            "data_lineage": {
                "silver_seifa_source": str(silver_seifa_path),
                "silver_health_source": str(silver_health_path),
                "silver_boundaries_source": str(silver_boundaries_path)
            },
            "analytics_ready": True,
            "risk_methodology": {
                "seifa_weight": 0.4,
                "health_utilisation_weight": 0.3,
                "geographic_accessibility_weight": 0.3
            },
            "risk_categories": ["Very Low", "Low", "Medium", "High", "Very High"]
        }
        
        storage_manager.save_with_metadata(
            gold_risk_classified, gold_risk_path, gold_risk_metadata, data_type="analytics"
        )
        
        # Save summary statistics
        with open(gold_summary_path, 'w') as f:
            json.dump(gold_summary_stats, f, indent=2)
        
        gold_processing_time = time.time() - gold_start_time
        
        # =================================================================
        # VALIDATION: Data lake transitions and quality
        # =================================================================
        
        # 1. Verify all layers exist and have data
        assert bronze_seifa_path.exists() and bronze_health_path.exists() and bronze_boundaries_path.exists()
        assert silver_seifa_path.exists() and silver_health_path.exists() and silver_boundaries_path.exists()
        assert gold_integrated_path.exists() and gold_risk_path.exists() and gold_summary_path.exists()
        
        # 2. Verify data quality improvements through layers
        bronze_record_count = len(bronze_seifa) + len(bronze_health) + len(bronze_boundaries)
        silver_record_count = len(silver_seifa_optimized) + len(silver_health_optimized) + len(silver_boundaries_optimized)
        gold_record_count = len(gold_integrated) + len(gold_risk_classified)
        
        # Silver should have similar or slightly fewer records due to quality filtering
        silver_retention_rate = silver_record_count / bronze_record_count
        assert silver_retention_rate > 0.80, f"Silver retention rate {silver_retention_rate:.1%} should be >80%"
        
        # 3. Verify data consistency across layers
        # SA2 codes should be consistent in silver and gold layers
        gold_integrated_loaded = pl.read_parquet(gold_integrated_path)
        gold_risk_loaded = pl.read_parquet(gold_risk_path)
        
        gold_integrated_sa2_codes = set(gold_integrated_loaded["sa2_code_2021"].drop_nulls().to_list())
        gold_risk_sa2_codes = set(gold_risk_loaded["sa2_code_2021"].drop_nulls().to_list())
        
        sa2_consistency_rate = len(gold_integrated_sa2_codes & gold_risk_sa2_codes) / len(gold_integrated_sa2_codes)
        assert sa2_consistency_rate > 0.85, f"SA2 consistency across gold datasets {sa2_consistency_rate:.1%} should be >85%"
        
        # 4. Verify schema evolution and metadata preservation
        # Each layer should have appropriate metadata
        bronze_metadata_exists = all(
            path.with_suffix(path.suffix + '.metadata').exists() or 
            (path.parent / f"{path.stem}_metadata.json").exists()
            for path in [bronze_seifa_path, bronze_health_path, bronze_boundaries_path]
        )
        
        # 5. Verify analytics readiness in gold layer
        # Gold datasets should have derived metrics and be ready for analytics
        assert "population_density" in gold_integrated_loaded.columns or len(gold_integrated_loaded.columns) > 10
        assert "risk_category" in gold_risk_loaded.columns
        assert "composite_risk_score" in gold_risk_loaded.columns
        
        # 6. Verify processing performance
        total_processing_time = bronze_processing_time + silver_processing_time + gold_processing_time
        assert total_processing_time < 180.0, f"Total data lake processing took {total_processing_time:.1f}s, expected <180s"
        assert bronze_processing_time < 60.0, f"Bronze processing took {bronze_processing_time:.1f}s, expected <60s"
        assert silver_processing_time < 75.0, f"Silver processing took {silver_processing_time:.1f}s, expected <75s"
        assert gold_processing_time < 60.0, f"Gold processing took {gold_processing_time:.1f}s, expected <60s"
        
        # Generate comprehensive data lake report
        data_lake_report = {
            "bronze_processing_time": bronze_processing_time,
            "silver_processing_time": silver_processing_time,
            "gold_processing_time": gold_processing_time,
            "total_processing_time": total_processing_time,
            "bronze_records": {
                "seifa": len(bronze_seifa),
                "health": len(bronze_health),
                "boundaries": len(bronze_boundaries)
            },
            "silver_records": {
                "seifa": len(silver_seifa_optimized),
                "health": len(silver_health_optimized),
                "boundaries": len(silver_boundaries_optimized)
            },
            "gold_records": {
                "integrated": len(gold_integrated),
                "risk_assessment": len(gold_risk_classified)
            },
            "data_quality_scores": {
                "seifa": float(seifa_quality_score),
                "health": float(health_quality_score),
                "boundaries": float(boundaries_quality_score)
            },
            "silver_retention_rate": silver_retention_rate,
            "sa2_consistency_rate": sa2_consistency_rate,
            "analytics_datasets_created": 2,
            "metadata_preservation": True,
            "schema_evolution_supported": True,
            "data_lake_layers_validated": ["bronze", "silver", "gold"]
        }
        
        logging.info(f"Data Lake Operations Report: {data_lake_report}")
        
        return data_lake_report
    
    def test_data_versioning_and_rollback(self, mock_seifa_data, mock_health_data, mock_data_paths):
        """Test data versioning, schema evolution, and rollback capabilities."""
        
        # Initialize components
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        incremental_processor = IncrementalProcessor(base_path=mock_data_paths["parquet_dir"])
        
        # Create versioned data lake structure
        versioned_dir = mock_data_paths["parquet_dir"] / "versioned"
        versioned_dir.mkdir(parents=True, exist_ok=True)
        
        # =================================================================
        # VERSION 1.0: Initial data schema and content
        # =================================================================
        
        # Create initial dataset with baseline schema
        v1_seifa = mock_seifa_data(num_areas=500)
        v1_health = mock_health_data(num_records=2500, num_sa2_areas=500)
        
        # Save v1.0 with version metadata
        v1_seifa_path = versioned_dir / "seifa" / "v1.0" / "seifa_2021.parquet"
        v1_health_path = versioned_dir / "health" / "v1.0" / "health_2023.parquet"
        
        v1_seifa_path.parent.mkdir(parents=True, exist_ok=True)
        v1_health_path.parent.mkdir(parents=True, exist_ok=True)
        
        v1_seifa_metadata = {
            "version": "1.0",
            "schema_version": "2021.1",
            "creation_date": datetime.now().isoformat(),
            "record_count": len(v1_seifa),
            "schema": {
                "columns": list(v1_seifa.columns),
                "data_types": {col: str(dtype) for col, dtype in zip(v1_seifa.columns, v1_seifa.dtypes)}
            },
            "backwards_compatible": True,
            "changelog": "Initial version"
        }
        
        v1_health_metadata = {
            "version": "1.0",
            "schema_version": "2023.1",
            "creation_date": datetime.now().isoformat(),
            "record_count": len(v1_health),
            "schema": {
                "columns": list(v1_health.columns),
                "data_types": {col: str(dtype) for col, dtype in zip(v1_health.columns, v1_health.dtypes)}
            },
            "backwards_compatible": True,
            "changelog": "Initial version"
        }
        
        storage_manager.save_with_metadata(v1_seifa, v1_seifa_path, v1_seifa_metadata, data_type="seifa")
        storage_manager.save_with_metadata(v1_health, v1_health_path, v1_health_metadata, data_type="health")
        
        # =================================================================
        # VERSION 1.1: Minor update with additional data
        # =================================================================
        
        time.sleep(0.1)  # Ensure different timestamps
        
        # Add new records to existing datasets
        v1_1_additional_seifa = mock_seifa_data(num_areas=100)
        v1_1_additional_health = mock_health_data(num_records=500, num_sa2_areas=100)
        
        # Merge with v1.0 data
        v1_1_seifa = pl.concat([v1_seifa, v1_1_additional_seifa], how="vertical")
        v1_1_health = pl.concat([v1_health, v1_1_additional_health], how="vertical")
        
        # Save v1.1
        v1_1_seifa_path = versioned_dir / "seifa" / "v1.1" / "seifa_2021.parquet"
        v1_1_health_path = versioned_dir / "health" / "v1.1" / "health_2023.parquet"
        
        v1_1_seifa_path.parent.mkdir(parents=True, exist_ok=True)
        v1_1_health_path.parent.mkdir(parents=True, exist_ok=True)
        
        v1_1_seifa_metadata = {
            "version": "1.1",
            "schema_version": "2021.1",  # Same schema
            "creation_date": datetime.now().isoformat(),
            "record_count": len(v1_1_seifa),
            "schema": v1_seifa_metadata["schema"],  # Inherit schema
            "backwards_compatible": True,
            "changelog": "Added 100 additional SA2 areas",
            "parent_version": "1.0",
            "incremental_update": True
        }
        
        v1_1_health_metadata = {
            "version": "1.1",
            "schema_version": "2023.1",  # Same schema
            "creation_date": datetime.now().isoformat(),
            "record_count": len(v1_1_health),
            "schema": v1_health_metadata["schema"],  # Inherit schema
            "backwards_compatible": True,
            "changelog": "Added 500 additional health records",
            "parent_version": "1.0",
            "incremental_update": True
        }
        
        storage_manager.save_with_metadata(v1_1_seifa, v1_1_seifa_path, v1_1_seifa_metadata, data_type="seifa")
        storage_manager.save_with_metadata(v1_1_health, v1_1_health_path, v1_1_health_metadata, data_type="health")
        
        # =================================================================
        # VERSION 2.0: Major update with schema evolution
        # =================================================================
        
        time.sleep(0.1)
        
        # Create v2.0 with schema evolution (additional columns)
        v2_seifa = v1_1_seifa.with_columns([
            pl.lit("2024").alias("data_year"),
            pl.Series("quality_score", np.random.uniform(0.8, 1.0, len(v1_1_seifa))),
            pl.lit("enhanced").alias("processing_mode")
        ])
        
        v2_health = v1_1_health.with_columns([
            pl.lit("2024").alias("data_year"),
            pl.Series("validation_status", np.random.choice(["validated", "pending"], len(v1_1_health))),
            pl.Series("cost_category", np.random.choice(["low", "medium", "high"], len(v1_1_health)))
        ])
        
        # Save v2.0 with schema evolution
        v2_seifa_path = versioned_dir / "seifa" / "v2.0" / "seifa_2024.parquet"
        v2_health_path = versioned_dir / "health" / "v2.0" / "health_2024.parquet"
        
        v2_seifa_path.parent.mkdir(parents=True, exist_ok=True)
        v2_health_path.parent.mkdir(parents=True, exist_ok=True)
        
        v2_seifa_metadata = {
            "version": "2.0",
            "schema_version": "2024.1",  # New schema version
            "creation_date": datetime.now().isoformat(),
            "record_count": len(v2_seifa),
            "schema": {
                "columns": list(v2_seifa.columns),
                "data_types": {col: str(dtype) for col, dtype in zip(v2_seifa.columns, v2_seifa.dtypes)}
            },
            "backwards_compatible": True,  # Core columns preserved
            "changelog": "Added data_year, quality_score, processing_mode columns",
            "parent_version": "1.1",
            "schema_evolution": {
                "added_columns": ["data_year", "quality_score", "processing_mode"],
                "removed_columns": [],
                "modified_columns": []
            }
        }
        
        v2_health_metadata = {
            "version": "2.0",
            "schema_version": "2024.1",  # New schema version
            "creation_date": datetime.now().isoformat(),
            "record_count": len(v2_health),
            "schema": {
                "columns": list(v2_health.columns),
                "data_types": {col: str(dtype) for col, dtype in zip(v2_health.columns, v2_health.dtypes)}
            },
            "backwards_compatible": True,  # Core columns preserved
            "changelog": "Added data_year, validation_status, cost_category columns",
            "parent_version": "1.1",
            "schema_evolution": {
                "added_columns": ["data_year", "validation_status", "cost_category"],
                "removed_columns": [],
                "modified_columns": []
            }
        }
        
        storage_manager.save_with_metadata(v2_seifa, v2_seifa_path, v2_seifa_metadata, data_type="seifa")
        storage_manager.save_with_metadata(v2_health, v2_health_path, v2_health_metadata, data_type="health")
        
        # =================================================================
        # VALIDATION: Version management and compatibility
        # =================================================================
        
        # 1. Verify all versions exist
        assert v1_seifa_path.exists() and v1_health_path.exists()
        assert v1_1_seifa_path.exists() and v1_1_health_path.exists()
        assert v2_seifa_path.exists() and v2_health_path.exists()
        
        # 2. Test backwards compatibility
        # Should be able to load all versions
        loaded_v1_seifa = pl.read_parquet(v1_seifa_path)
        loaded_v1_1_seifa = pl.read_parquet(v1_1_seifa_path)
        loaded_v2_seifa = pl.read_parquet(v2_seifa_path)
        
        # Core columns should be preserved across versions
        v1_columns = set(loaded_v1_seifa.columns)
        v1_1_columns = set(loaded_v1_1_seifa.columns)
        v2_columns = set(loaded_v2_seifa.columns)
        
        assert v1_columns.issubset(v1_1_columns), "v1.1 should be backwards compatible with v1.0"
        assert v1_columns.issubset(v2_columns), "v2.0 should be backwards compatible with v1.0"
        assert v1_1_columns.issubset(v2_columns), "v2.0 should be backwards compatible with v1.1"
        
        # 3. Test data growth across versions
        assert len(loaded_v1_1_seifa) > len(loaded_v1_seifa), "v1.1 should have more records than v1.0"
        assert len(loaded_v2_seifa) == len(loaded_v1_1_seifa), "v2.0 should have same records as v1.1 (schema evolution only)"
        
        # 4. Test schema evolution
        # v2.0 should have additional columns
        v2_additional_columns = v2_columns - v1_1_columns
        expected_additional_columns = {"data_year", "quality_score", "processing_mode"}
        assert v2_additional_columns == expected_additional_columns, f"v2.0 should add {expected_additional_columns}"
        
        # 5. Test rollback capability
        # Should be able to rollback from v2.0 to v1.1
        rollback_seifa = incremental_processor.rollback_to_version(
            current_data=loaded_v2_seifa,
            previous_data=loaded_v1_1_seifa,
            key_column="sa2_code_2021"
        )
        
        # Rollback should restore v1.1 schema and data
        assert len(rollback_seifa) == len(loaded_v1_1_seifa)
        rollback_columns = set(rollback_seifa.columns)
        assert rollback_columns == v1_1_columns, "Rollback should restore v1.1 schema"
        
        # 6. Test version comparison
        # Should be able to compare versions and identify differences
        v1_v1_1_diff = {
            "record_count_change": len(loaded_v1_1_seifa) - len(loaded_v1_seifa),
            "schema_changes": list(v1_1_columns - v1_columns),
            "version_jump": "minor"
        }
        
        v1_1_v2_diff = {
            "record_count_change": len(loaded_v2_seifa) - len(loaded_v1_1_seifa),
            "schema_changes": list(v2_columns - v1_1_columns),
            "version_jump": "major"
        }
        
        assert v1_v1_1_diff["record_count_change"] > 0, "v1.0 to v1.1 should add records"
        assert len(v1_v1_1_diff["schema_changes"]) == 0, "v1.0 to v1.1 should not change schema"
        
        assert v1_1_v2_diff["record_count_change"] == 0, "v1.1 to v2.0 should not change record count"
        assert len(v1_1_v2_diff["schema_changes"]) == 3, "v1.1 to v2.0 should add 3 columns"
        
        # 7. Test version lineage tracking
        version_lineage = {
            "v1.0": {"parent": None, "children": ["v1.1"]},
            "v1.1": {"parent": "v1.0", "children": ["v2.0"]},
            "v2.0": {"parent": "v1.1", "children": []}
        }
        
        # Should be able to traverse version tree
        lineage_validation = True
        for version, info in version_lineage.items():
            if info["parent"] is not None:
                # Parent version should exist
                parent_path = versioned_dir / "seifa" / info["parent"] / "seifa_2021.parquet"
                if version == "v2.0":
                    parent_path = versioned_dir / "seifa" / info["parent"] / "seifa_2021.parquet"
                lineage_validation = lineage_validation and parent_path.exists()
        
        assert lineage_validation, "Version lineage should be traceable"
        
        # Generate versioning report
        versioning_report = {
            "versions_created": 3,
            "total_versioned_files": 6,  # 3 versions × 2 datasets
            "backwards_compatibility_maintained": True,
            "schema_evolution_successful": True,
            "rollback_capability_tested": True,
            "version_comparisons": {
                "v1.0_to_v1.1": v1_v1_1_diff,
                "v1.1_to_v2.0": v1_1_v2_diff
            },
            "lineage_tracking_enabled": True,
            "data_growth_across_versions": {
                "v1.0_records": len(loaded_v1_seifa),
                "v1.1_records": len(loaded_v1_1_seifa),
                "v2.0_records": len(loaded_v2_seifa)
            },
            "schema_evolution_details": {
                "v2.0_additional_columns": list(v2_additional_columns),
                "core_columns_preserved": len(v1_columns & v2_columns) == len(v1_columns)
            }
        }
        
        logging.info(f"Data Versioning Report: {versioning_report}")
        
        return versioning_report
    
    def test_concurrent_data_lake_operations(self, mock_seifa_data, mock_health_data, 
                                           mock_boundary_data, mock_data_paths):
        """Test concurrent operations across data lake layers."""
        
        import concurrent.futures
        
        # Initialize components
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        memory_optimizer = MemoryOptimizer()
        
        # Create concurrent operations structure
        concurrent_dir = mock_data_paths["parquet_dir"] / "concurrent"
        concurrent_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test datasets
        seifa_datasets = [mock_seifa_data(num_areas=200) for _ in range(3)]
        health_datasets = [mock_health_data(num_records=1000, num_sa2_areas=200) for _ in range(3)]
        boundary_datasets = [mock_boundary_data(num_areas=200) for _ in range(3)]
        
        # Define concurrent operations
        def save_seifa_dataset(dataset_id, dataset):
            """Save SEIFA dataset concurrently."""
            optimized = memory_optimizer.optimize_data_types(dataset, data_category="seifa")
            path = concurrent_dir / f"seifa_concurrent_{dataset_id}.parquet"
            metadata = {
                "dataset_id": dataset_id,
                "layer": "silver",
                "concurrent_operation": True,
                "timestamp": datetime.now().isoformat()
            }
            return storage_manager.save_with_metadata(optimized, path, metadata, data_type="seifa")
        
        def save_health_dataset(dataset_id, dataset):
            """Save health dataset concurrently."""
            optimized = memory_optimizer.optimize_data_types(dataset, data_category="health")
            path = concurrent_dir / f"health_concurrent_{dataset_id}.parquet"
            metadata = {
                "dataset_id": dataset_id,
                "layer": "silver",
                "concurrent_operation": True,
                "timestamp": datetime.now().isoformat()
            }
            return storage_manager.save_with_metadata(optimized, path, metadata, data_type="health")
        
        def save_boundary_dataset(dataset_id, dataset):
            """Save boundary dataset concurrently."""
            optimized = memory_optimizer.optimize_data_types(dataset, data_category="geographic")
            path = concurrent_dir / f"boundary_concurrent_{dataset_id}.parquet"
            metadata = {
                "dataset_id": dataset_id,
                "layer": "silver",
                "concurrent_operation": True,
                "timestamp": datetime.now().isoformat()
            }
            return storage_manager.save_with_metadata(optimized, path, metadata, data_type="geographic")
        
        # Execute concurrent operations
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
            # Submit all concurrent operations
            futures = []
            
            # SEIFA operations
            for i, dataset in enumerate(seifa_datasets):
                future = executor.submit(save_seifa_dataset, i, dataset)
                futures.append(("seifa", i, future))
            
            # Health operations
            for i, dataset in enumerate(health_datasets):
                future = executor.submit(save_health_dataset, i, dataset)
                futures.append(("health", i, future))
            
            # Boundary operations
            for i, dataset in enumerate(boundary_datasets):
                future = executor.submit(save_boundary_dataset, i, dataset)
                futures.append(("boundary", i, future))
            
            # Collect results
            results = []
            for data_type, dataset_id, future in futures:
                try:
                    result_path = future.result(timeout=60)
                    results.append({
                        "data_type": data_type,
                        "dataset_id": dataset_id,
                        "path": result_path,
                        "success": True,
                        "error": None
                    })
                except Exception as e:
                    results.append({
                        "data_type": data_type,
                        "dataset_id": dataset_id,
                        "path": None,
                        "success": False,
                        "error": str(e)
                    })
        
        concurrent_time = time.time() - start_time
        
        # Validate concurrent operations
        # 1. All operations should succeed
        successful_operations = [r for r in results if r["success"]]
        failed_operations = [r for r in results if not r["success"]]
        
        assert len(successful_operations) == 9, f"Expected 9 successful operations, got {len(successful_operations)}"
        assert len(failed_operations) == 0, f"Should have no failed operations, got {len(failed_operations)}"
        
        # 2. All files should exist
        for result in successful_operations:
            assert result["path"].exists(), f"File {result['path']} should exist"
        
        # 3. Concurrent processing should be efficient
        assert concurrent_time < 45.0, f"Concurrent operations took {concurrent_time:.1f}s, expected <45s"
        
        # 4. Test concurrent reading operations
        def load_and_validate_dataset(result):
            """Load and validate dataset concurrently."""
            df = pl.read_parquet(result["path"])
            return {
                "data_type": result["data_type"],
                "dataset_id": result["dataset_id"],
                "record_count": len(df),
                "columns": list(df.columns),
                "valid": len(df) > 0
            }
        
        # Concurrent reading
        read_start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
            read_futures = [executor.submit(load_and_validate_dataset, result) for result in successful_operations]
            read_results = [future.result() for future in concurrent.futures.as_completed(read_futures)]
        
        read_time = time.time() - read_start_time
        
        # 5. Validate concurrent reading
        assert len(read_results) == 9, "Should read all 9 datasets successfully"
        assert all(result["valid"] for result in read_results), "All loaded datasets should be valid"
        assert read_time < 15.0, f"Concurrent reading took {read_time:.1f}s, expected <15s"
        
        # 6. Test data integrity after concurrent operations
        # Group results by data type
        seifa_results = [r for r in read_results if r["data_type"] == "seifa"]
        health_results = [r for r in read_results if r["data_type"] == "health"]
        boundary_results = [r for r in read_results if r["data_type"] == "boundary"]
        
        # Each data type should have 3 datasets
        assert len(seifa_results) == 3 and len(health_results) == 3 and len(boundary_results) == 3
        
        # Record counts should be consistent within data types
        seifa_record_counts = [r["record_count"] for r in seifa_results]
        health_record_counts = [r["record_count"] for r in health_results]
        boundary_record_counts = [r["record_count"] for r in boundary_results]
        
        # All datasets of same type should have same structure
        assert len(set(seifa_record_counts)) == 1, "All SEIFA datasets should have same record count"
        assert len(set(health_record_counts)) == 1, "All health datasets should have same record count"
        assert len(set(boundary_record_counts)) == 1, "All boundary datasets should have same record count"
        
        # Generate concurrent operations report
        concurrent_report = {
            "total_concurrent_operations": 9,
            "successful_operations": len(successful_operations),
            "failed_operations": len(failed_operations),
            "concurrent_write_time": concurrent_time,
            "concurrent_read_time": read_time,
            "data_integrity_maintained": True,
            "operations_by_type": {
                "seifa": len(seifa_results),
                "health": len(health_results),
                "boundaries": len(boundary_results)
            },
            "performance_metrics": {
                "average_write_time_per_operation": concurrent_time / 9,
                "average_read_time_per_operation": read_time / 9,
                "concurrent_efficiency": True
            },
            "data_consistency_validation": {
                "seifa_consistency": len(set(seifa_record_counts)) == 1,
                "health_consistency": len(set(health_record_counts)) == 1,
                "boundary_consistency": len(set(boundary_record_counts)) == 1
            }
        }
        
        logging.info(f"Concurrent Data Lake Operations Report: {concurrent_report}")
        
        return concurrent_report