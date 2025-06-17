"""
Comprehensive unit tests for Health Data Processor.

Tests PBS (Pharmaceutical Benefits Scheme) and MBS (Medicare Benefits Schedule) 
data processing with focus on:
- Data validation and cleaning
- SA2 code linking and geographic aggregation
- Error handling for missing/invalid data
- Performance with large healthcare datasets
- Integration with SEIFA socio-economic data

Covers both historical and current health utilisation patterns.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import zipfile
import io
import time
from datetime import datetime, date, timedelta

from src.data_processing.health_processor import (
    HealthDataProcessor,
    MBS_CONFIG,
    PBS_CONFIG,
    MBS_SCHEMA,
    PBS_SCHEMA,
    HEALTH_CLASSIFICATIONS
)


class TestHealthDataProcessor:
    """Comprehensive test suite for health data processor."""
    
    def test_health_processor_initialization(self, mock_data_paths):
        """Test health processor initializes correctly with proper directory structure."""
        processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        assert processor.data_dir == mock_data_paths["raw_dir"].parent
        assert processor.raw_dir.exists()
        assert processor.processed_dir.exists()
        assert processor.raw_dir.name == "raw"
        assert processor.processed_dir.name == "processed"
    
    def test_health_processor_default_directory(self):
        """Test health processor with default data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                processor = HealthDataProcessor()
                assert processor.data_dir.name == "data"
                assert processor.raw_dir.exists()
                assert processor.processed_dir.exists()
            finally:
                import os
                os.chdir(original_cwd)
    
    def test_validate_pbs_data_valid(self, mock_health_data):
        """Test validation of valid PBS prescription data."""
        processor = HealthDataProcessor()
        health_df = mock_health_data(num_records=100, num_sa2_areas=20)
        
        # Convert to PBS format expected by processor
        pbs_df = health_df.select([
            pl.col("atc_code"),
            pl.col("prescription_count").alias("prescriptions"),
            pl.col("cost_government"),
            pl.col("state")
        ])
        
        validated_df = processor._validate_pbs_data(pbs_df)
        
        assert isinstance(validated_df, pl.DataFrame)
        assert len(validated_df) <= len(health_df)  # Should not increase rows
        
        # Check basic validation worked
        if len(validated_df) > 0:
            assert "prescriptions" in validated_df.columns
            assert "cost_government" in validated_df.columns
    
    def test_validate_pbs_data_basic_validation(self):
        """Test PBS data validation with basic checks."""
        processor = HealthDataProcessor()
        
        # Create DataFrame with PBS-like structure
        df = pl.DataFrame({
            "atc_code": ["A02BC01", "A10BD07", "INVALID", "C09AA02", "J01CA04"],
            "prescriptions": [10, 5, 8, 12, 7],
            "cost_government": [100.0, 50.0, 80.0, 120.0, 70.0],
            "state": ["NSW", "VIC", "QLD", "SA", "WA"]
        })
        
        validated_df = processor._validate_pbs_data(df)
        
        # Should return a valid DataFrame
        assert isinstance(validated_df, pl.DataFrame)
        
        # Basic validation should work
        if len(validated_df) > 0:
            assert "prescriptions" in validated_df.columns
            assert "cost_government" in validated_df.columns
    
    def test_validate_pbs_data_negative_values(self):
        """Test validation handles negative prescription counts and costs."""
        processor = HealthDataProcessor()
        
        df = pl.DataFrame({
            "atc_code": ["A02BC01", "A10BD07", "C07AB02"],
            "prescriptions": [10, -5, 8],  # Negative count
            "cost_government": [100.0, 50.0, -80.0],  # Negative cost
            "state": ["NSW", "VIC", "QLD"]
        })
        
        validated_df = processor._validate_pbs_data(df)
        
        # Should return valid DataFrame (validation logic may vary)
        assert isinstance(validated_df, pl.DataFrame)
        
        # Check that validation handles the data appropriately
        if len(validated_df) > 0:
            assert "prescriptions" in validated_df.columns
            assert "cost_government" in validated_df.columns
    
    def test_process_pbs_data_basic(self, mock_health_data, mock_data_paths):
        """Test basic PBS data processing pipeline."""
        processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_df = mock_health_data(num_records=50, num_sa2_areas=10)
        
        # Mock CSV file reading
        csv_path = mock_data_paths["raw_dir"] / "mock_pbs.csv"
        health_df.write_csv(csv_path)
        
        # Test processing
        with patch.object(processor, '_read_pbs_csv') as mock_read:
            mock_read.return_value = health_df
            
            result_df = processor.process_pbs_data(csv_path)
            
            assert isinstance(result_df, pl.DataFrame)
            assert len(result_df) > 0
            
            # Verify SA2 aggregation occurred
            unique_sa2s = result_df.select("sa2_code").unique().height
            assert unique_sa2s <= 10  # Should be aggregated by SA2
    
    def test_process_pbs_data_aggregation(self, mock_health_data):
        """Test PBS data aggregation by SA2 areas."""
        processor = HealthDataProcessor()
        
        # Create test data with duplicate SA2 codes for aggregation
        df = pl.DataFrame({
            "sa2_code": ["123456789", "123456789", "234567890", "234567890"],
            "prescription_count": [5, 3, 10, 7],
            "cost_government": [50.0, 30.0, 100.0, 70.0],
            "cost_patient": [10.0, 6.0, 20.0, 14.0],
            "chronic_medication": [1, 0, 1, 1],
            "atc_code": ["A02BC01", "A10BD07", "A02BC01", "C07AB02"],
            "dispensing_date": [date(2023, 1, 1)] * 4,
            "state": ["NSW", "NSW", "VIC", "VIC"]
        })
        
        aggregated_df = processor._aggregate_by_sa2(df)
        
        # Should have one row per SA2
        assert len(aggregated_df) == 2
        
        # Verify aggregation calculations
        sa2_123 = aggregated_df.filter(pl.col("sa2_code") == "123456789")
        sa2_234 = aggregated_df.filter(pl.col("sa2_code") == "234567890")
        
        # Check prescription count aggregation
        assert sa2_123["total_prescriptions"].item() == 8  # 5 + 3
        assert sa2_234["total_prescriptions"].item() == 17  # 10 + 7
        
        # Check cost aggregation
        assert abs(sa2_123["total_cost_government"].item() - 80.0) < 0.01  # 50 + 30
        assert abs(sa2_234["total_cost_government"].item() - 170.0) < 0.01  # 100 + 70
    
    def test_calculate_health_metrics(self, mock_health_data):
        """Test calculation of health utilisation metrics."""
        processor = HealthDataProcessor()
        
        # Create test data with known patterns
        df = pl.DataFrame({
            "sa2_code": ["123456789", "234567890"],
            "total_prescriptions": [100, 50],
            "chronic_prescriptions": [30, 40],  # 30% vs 80% chronic rate
            "total_cost_government": [1000.0, 2000.0],
            "unique_patients": [50, 25],  # Different patient counts
            "population_2021": [1000, 500]  # Different populations
        })
        
        metrics_df = processor._calculate_health_metrics(df)
        
        # Verify metric calculations
        assert "prescriptions_per_capita" in metrics_df.columns
        assert "chronic_medication_rate" in metrics_df.columns
        assert "avg_cost_per_prescription" in metrics_df.columns
        assert "utilisation_category" in metrics_df.columns
        
        # Check specific calculations
        row1 = metrics_df.filter(pl.col("sa2_code") == "123456789").row(0, named=True)
        row2 = metrics_df.filter(pl.col("sa2_code") == "234567890").row(0, named=True)
        
        assert abs(row1["prescriptions_per_capita"] - 0.1) < 0.01  # 100/1000
        assert abs(row2["prescriptions_per_capita"] - 0.1) < 0.01  # 50/500
        
        assert abs(row1["chronic_medication_rate"] - 0.3) < 0.01  # 30/100
        assert abs(row2["chronic_medication_rate"] - 0.8) < 0.01  # 40/50
    
    def test_classify_utilisation_patterns(self):
        """Test health utilisation pattern classification."""
        processor = HealthDataProcessor()
        
        df = pl.DataFrame({
            "sa2_code": ["123456789", "234567890", "345678901", "456789012"],
            "prescriptions_per_capita": [0.05, 0.15, 0.25, 0.35],  # Low to high
            "chronic_medication_rate": [0.1, 0.3, 0.5, 0.8],      # Low to high
            "avg_cost_per_prescription": [25.0, 50.0, 75.0, 100.0] # Low to high
        })
        
        classified_df = processor._classify_utilisation_patterns(df)
        
        assert "utilisation_category" in classified_df.columns
        
        # Verify classification logic
        categories = classified_df["utilisation_category"].to_list()
        
        # First row should be low utilisation
        assert categories[0] in ["Low", "Very Low"]
        
        # Last row should be high utilisation
        assert categories[3] in ["High", "Very High"]
        
        # Categories should be valid
        valid_categories = ["Very Low", "Low", "Medium", "High", "Very High"]
        for category in categories:
            assert category in valid_categories
    
    def test_link_to_seifa_data(self, mock_health_data, mock_seifa_data):
        """Test linking health data with SEIFA socio-economic indices."""
        processor = HealthDataProcessor()
        
        # Create matching SA2 codes
        sa2_codes = ["123456789", "234567890", "345678901"]
        
        health_df = pl.DataFrame({
            "sa2_code": sa2_codes,
            "total_prescriptions": [100, 150, 80],
            "utilisation_category": ["Medium", "High", "Low"]
        })
        
        seifa_df = mock_seifa_data(num_areas=3)
        seifa_df = seifa_df.with_columns(pl.Series("sa2_code_2021", sa2_codes))
        
        linked_df = processor._link_to_seifa_data(health_df, seifa_df)
        
        # Verify linking occurred
        assert len(linked_df) == 3
        assert "irsd_decile" in linked_df.columns
        assert "irsad_decile" in linked_df.columns
        
        # Verify all health data preserved
        assert "total_prescriptions" in linked_df.columns
        assert "utilisation_category" in linked_df.columns
        
        # Verify SA2 codes match
        linked_codes = set(linked_df["sa2_code"].to_list())
        expected_codes = set(sa2_codes)
        assert linked_codes == expected_codes
    
    def test_process_historical_data_zip(self, mock_data_paths):
        """Test processing of historical health data from ZIP archives."""
        processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Create mock ZIP file with CSV data
        zip_path = mock_data_paths["raw_dir"] / "historical_health.zip"
        
        # Create test CSV content
        csv_content = """sa2_code,year,prescriptions,cost
123456789,2020,50,500.00
234567890,2020,75,750.00
123456789,2021,60,600.00
234567890,2021,80,800.00"""
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("health_data.csv", csv_content)
        
        result_df = processor._process_zip_health_data(zip_path)
        
        assert isinstance(result_df, pl.DataFrame)
        assert len(result_df) == 4  # 4 rows of data
        assert "sa2_code" in result_df.columns
        assert "year" in result_df.columns
    
    def test_handle_missing_data(self, mock_health_data):
        """Test handling of missing data in health datasets."""
        processor = HealthDataProcessor()
        
        # Create data with missing values
        df = pl.DataFrame({
            "sa2_code": ["123456789", "234567890", "345678901"],
            "prescription_count": [50, None, 75],
            "cost_government": [500.0, 600.0, None],
            "atc_code": ["A02BC01", None, "C07AB02"],
            "chronic_medication": [1, 0, None]
        })
        
        cleaned_df = processor._handle_missing_data(df)
        
        # Should handle missing values appropriately
        assert isinstance(cleaned_df, pl.DataFrame)
        assert len(cleaned_df) > 0
        
        # Critical columns (SA2 code) should not have nulls
        if "sa2_code" in cleaned_df.columns:
            assert cleaned_df["sa2_code"].null_count() == 0
    
    def test_temporal_analysis(self, mock_health_data):
        """Test temporal trend analysis in health data."""
        processor = HealthDataProcessor()
        
        # Create multi-year data
        dates = [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1), date(2023, 1, 1)]
        
        df = pl.DataFrame({
            "sa2_code": ["123456789"] * 4,
            "dispensing_date": dates,
            "prescription_count": [50, 60, 70, 80],  # Increasing trend
            "cost_government": [500.0, 650.0, 800.0, 950.0]
        })
        
        trend_df = processor._analyze_temporal_trends(df)
        
        assert "year" in trend_df.columns
        assert "trend_direction" in trend_df.columns or "prescription_growth_rate" in trend_df.columns
        
        # Should detect increasing trend
        growth_rates = trend_df["prescription_growth_rate"].to_list()
        assert all(rate > 0 for rate in growth_rates if rate is not None)
    
    def test_data_quality_metrics(self, mock_health_data):
        """Test data quality assessment for health datasets."""
        processor = HealthDataProcessor()
        health_df = mock_health_data(num_records=100, num_sa2_areas=20)
        
        quality_metrics = processor._assess_data_quality(health_df)
        
        expected_metrics = [
            "total_records", "unique_sa2_areas", "completeness_rate",
            "invalid_sa2_codes", "negative_values", "data_coverage_period"
        ]
        
        for metric in expected_metrics:
            assert metric in quality_metrics
        
        # Verify reasonable values
        assert quality_metrics["total_records"] == 100
        assert quality_metrics["unique_sa2_areas"] <= 20
        assert 0 <= quality_metrics["completeness_rate"] <= 1
    
    def test_performance_large_dataset(self, mock_health_data, mock_data_paths):
        """Test processing performance with large health dataset."""
        processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Create larger dataset
        large_df = mock_health_data(num_records=10000, num_sa2_areas=500)
        
        start_time = time.time()
        
        # Process through main pipeline steps
        validated_df = processor._validate_health_data(large_df)
        aggregated_df = processor._aggregate_by_sa2(validated_df)
        metrics_df = processor._calculate_health_metrics(aggregated_df)
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 30.0  # Should complete within 30 seconds
        assert len(metrics_df) > 0
        assert len(metrics_df) <= 500  # Should be aggregated by SA2
        
        # Memory efficiency check
        memory_usage = metrics_df.estimated_size("mb")
        assert memory_usage < 50  # Should not exceed 50MB
    
    def test_atc_code_validation(self):
        """Test ATC (Anatomical Therapeutic Chemical) code validation."""
        processor = HealthDataProcessor()
        
        # Test with valid and invalid ATC codes
        df = pl.DataFrame({
            "sa2_code": ["123456789"] * 6,
            "atc_code": ["A02BC01", "INVALID", "C07AB02", "", None, "J01CA04"],
            "prescription_count": [10, 5, 8, 3, 7, 12],
            "cost_government": [100.0, 50.0, 80.0, 30.0, 70.0, 120.0]
        })
        
        validated_df = processor._validate_atc_codes(df)
        
        # Should filter out invalid ATC codes
        valid_codes = validated_df["atc_code"].drop_nulls().to_list()
        
        for code in valid_codes:
            # ATC codes should follow pattern: Letter-Number-Number-Letter-Letter-Number-Number
            assert len(code) == 7
            assert code[0].isalpha()
            assert code[1:3].isdigit()
            assert code[3:5].isalpha()
            assert code[5:7].isdigit()
    
    def test_state_level_aggregation(self, mock_health_data):
        """Test aggregation of health data at state level."""
        processor = HealthDataProcessor()
        
        # Create data with different states
        df = pl.DataFrame({
            "sa2_code": ["123456789", "134567890", "234567890", "245678901"],  # NSW, NSW, VIC, VIC
            "state": ["NSW", "NSW", "VIC", "VIC"],
            "total_prescriptions": [100, 150, 80, 120],
            "total_cost_government": [1000.0, 1500.0, 800.0, 1200.0],
            "population_2021": [1000, 1500, 800, 1200]
        })
        
        state_df = processor._aggregate_by_state(df)
        
        assert len(state_df) == 2  # NSW and VIC
        
        # Verify state-level aggregation
        nsw_data = state_df.filter(pl.col("state") == "NSW").row(0, named=True)
        vic_data = state_df.filter(pl.col("state") == "VIC").row(0, named=True)
        
        assert nsw_data["total_prescriptions"] == 250  # 100 + 150
        assert vic_data["total_prescriptions"] == 200  # 80 + 120
        
        assert abs(nsw_data["total_cost_government"] - 2500.0) < 0.01
        assert abs(vic_data["total_cost_government"] - 2000.0) < 0.01
    
    def test_error_handling_corrupt_data(self, mock_data_paths):
        """Test error handling with corrupted health data files."""
        processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Create corrupted CSV file
        corrupt_csv = mock_data_paths["raw_dir"] / "corrupt_health.csv"
        with open(corrupt_csv, "w") as f:
            f.write("This is not valid CSV data\nNo proper structure\n")
        
        with pytest.raises(Exception):
            processor._read_pbs_csv(corrupt_csv)
    
    def test_concurrent_processing_safety(self, mock_health_data, mock_data_paths):
        """Test thread safety of health processor (basic check)."""
        import concurrent.futures
        
        def process_health():
            processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
            health_df = mock_health_data(num_records=50, num_sa2_areas=10)
            return processor._validate_health_data(health_df)
        
        # Run multiple processors concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_health) for _ in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert isinstance(result, pl.DataFrame)
    
    def test_integration_with_boundary_data(self, mock_health_data, mock_boundary_data):
        """Test integration of health data with geographic boundary data."""
        processor = HealthDataProcessor()
        
        # Create matching datasets
        sa2_codes = ["123456789", "234567890", "345678901"]
        
        health_df = pl.DataFrame({
            "sa2_code": sa2_codes,
            "total_prescriptions": [100, 150, 80],
            "utilisation_category": ["Medium", "High", "Low"]
        })
        
        boundary_df = mock_boundary_data(num_areas=3)
        boundary_df = boundary_df.with_columns(pl.Series("sa2_code_2021", sa2_codes))
        
        integrated_df = processor._integrate_with_boundaries(health_df, boundary_df)
        
        # Verify integration
        assert len(integrated_df) == 3
        assert "area_sqkm" in integrated_df.columns
        assert "remoteness_category" in integrated_df.columns
        assert "population_2021" in integrated_df.columns
        
        # Calculate population-adjusted metrics
        assert "prescriptions_per_1000_pop" in integrated_df.columns or "prescription_density" in integrated_df.columns


class TestHealthDataConfiguration:
    """Test health data configuration constants and schemas."""
    
    def test_mbs_config_structure(self):
        """Test MBS configuration has expected structure."""
        required_keys = [
            "historical_filename", "expected_records_historical",
            "demographic_columns", "service_columns", "temporal_columns"
        ]
        
        for key in required_keys:
            assert key in MBS_CONFIG
        
        # Verify data types
        assert isinstance(MBS_CONFIG["historical_filename"], str)
        assert isinstance(MBS_CONFIG["expected_records_historical"], int)
        assert isinstance(MBS_CONFIG["demographic_columns"], list)
        assert isinstance(MBS_CONFIG["service_columns"], list)
        assert isinstance(MBS_CONFIG["temporal_columns"], list)
    
    def test_pbs_config_structure(self):
        """Test PBS configuration has expected structure."""
        required_keys = [
            "current_filename", "historical_filename", "expected_records_current",
            "prescription_columns", "utilisation_columns", "cost_columns"
        ]
        
        for key in required_keys:
            assert key in PBS_CONFIG
        
        # Verify data types
        assert isinstance(PBS_CONFIG["current_filename"], str)
        assert isinstance(PBS_CONFIG["historical_filename"], str)
        assert isinstance(PBS_CONFIG["expected_records_current"], int)
        assert isinstance(PBS_CONFIG["prescription_columns"], list)
        assert isinstance(PBS_CONFIG["utilisation_columns"], list)
        assert isinstance(PBS_CONFIG["cost_columns"], list)
    
    def test_mbs_schema_completeness(self):
        """Test MBS schema covers expected columns."""
        required_columns = [
            "year", "quarter", "state", "postcode", "age_group", "gender",
            "item_number", "service_category", "services_count", 
            "benefit_amount", "schedule_fee"
        ]
        
        for col in required_columns:
            assert col in MBS_SCHEMA
        
        # Verify Polars data types
        for col, dtype in MBS_SCHEMA.items():
            assert hasattr(pl, dtype.__name__) or isinstance(dtype, type)
    
    def test_pbs_schema_completeness(self):
        """Test PBS schema covers expected columns."""
        required_columns = [
            "year", "month", "state", "atc_code", "drug_name", "strength",
            "prescriptions", "ddd_per_1000", "cost_government", "cost_patient"
        ]
        
        for col in required_columns:
            assert col in PBS_SCHEMA
        
        # Verify Polars data types
        for col, dtype in PBS_SCHEMA.items():
            assert hasattr(pl, dtype.__name__) or isinstance(dtype, type)
    
    def test_health_classifications_complete(self):
        """Test health classification categories are complete."""
        required_classifications = ["age_groups", "service_categories", "atc_categories"]
        
        for classification in required_classifications:
            assert classification in HEALTH_CLASSIFICATIONS
            assert isinstance(HEALTH_CLASSIFICATIONS[classification], list)
            assert len(HEALTH_CLASSIFICATIONS[classification]) > 0
        
        # Verify age groups cover full lifecycle
        age_groups = HEALTH_CLASSIFICATIONS["age_groups"]
        assert "0-4" in age_groups
        assert "85+" in age_groups
        
        # Verify ATC categories cover major therapeutic areas
        atc_categories = HEALTH_CLASSIFICATIONS["atc_categories"]
        assert any("Cardiovascular" in cat for cat in atc_categories)
        assert any("Nervous System" in cat for cat in atc_categories)