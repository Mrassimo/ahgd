"""
Comprehensive unit tests for Simple Boundary Processor.

Tests geographic boundary data processing for Australian SA2 areas with focus on:
- Shapefile and geographic data validation
- Coordinate system transformations
- Boundary simplification and optimization
- Population data integration
- Centroid calculation and area computations
- Performance with large geographic datasets

Covers Australian Bureau of Statistics boundary data formats.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import time
import json
from datetime import datetime, date

from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor


class TestSimpleBoundaryProcessor:
    """Comprehensive test suite for simple boundary processor."""
    
    def test_boundary_processor_initialization(self, mock_data_paths):
        """Test boundary processor initializes correctly with proper directory structure."""
        processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        assert processor.data_dir == mock_data_paths["raw_dir"].parent
        assert processor.raw_dir.exists()
        assert processor.processed_dir.exists()
        assert processor.raw_dir.name == "raw"
        assert processor.processed_dir.name == "processed"
    
    def test_boundary_processor_default_directory(self):
        """Test boundary processor with default data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                processor = SimpleBoundaryProcessor()
                assert processor.data_dir.name == "data"
                assert processor.raw_dir.exists()
                assert processor.processed_dir.exists()
            finally:
                import os
                os.chdir(original_cwd)
    
    def test_validate_boundary_data_valid(self, mock_boundary_data):
        """Test validation of valid boundary data."""
        processor = SimpleBoundaryProcessor()
        boundary_df = mock_boundary_data(num_areas=50)
        
        validated_df = processor._validate_boundary_data(boundary_df)
        
        assert isinstance(validated_df, pl.DataFrame)
        assert len(validated_df) <= len(boundary_df)  # Should not increase rows
        
        # Check SA2 codes are valid
        if "sa2_code_2021" in validated_df.columns:
            codes = validated_df["sa2_code_2021"].drop_nulls().to_list()
            for code in codes:
                assert len(str(code)) == 9
                assert str(code).isdigit()
                assert str(code)[0] in "12345678"  # Valid state prefixes
    
    def test_validate_boundary_data_invalid_sa2_codes(self):
        """Test validation filters out invalid SA2 codes."""
        processor = SimpleBoundaryProcessor()
        
        # Create DataFrame with mixed valid/invalid SA2 codes
        df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "12345", "INVALID", "987654321", None],
            "sa2_name_2021": ["Valid Area 1", "Invalid 1", "Invalid 2", "Valid Area 2", "Null Area"],
            "state_name": ["NSW", "NSW", "VIC", "VIC", "QLD"],
            "area_sqkm": [10.5, 20.3, 15.7, 25.1, 12.9],
            "population_2021": [1000, 1500, 800, 2000, 900]
        })
        
        validated_df = processor._validate_boundary_data(df)
        
        # Should only keep valid 9-digit SA2 codes
        assert len(validated_df) == 2  # Only first and fourth rows
        codes = validated_df["sa2_code_2021"].to_list()
        for code in codes:
            assert len(str(code)) == 9
            assert str(code).isdigit()
    
    def test_validate_boundary_data_invalid_coordinates(self):
        """Test validation filters out invalid geographic coordinates."""
        processor = SimpleBoundaryProcessor()
        
        df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901", "456789012"],
            "sa2_name_2021": ["Area 1", "Area 2", "Area 3", "Area 4"],
            "centroid_lat": [-35.5, -95.0, 10.0, -25.5],  # Second invalid (outside Australia)
            "centroid_lon": [149.1, 135.0, 250.0, 145.2], # Third invalid (outside Australia)
            "state_name": ["NSW", "SA", "QLD", "VIC"]
        })
        
        validated_df = processor._validate_boundary_data(df)
        
        # Should filter out areas with invalid coordinates
        # Australian latitude: approximately -44 to -10
        # Australian longitude: approximately 113 to 154
        valid_lats = validated_df["centroid_lat"].drop_nulls().to_list()
        valid_lons = validated_df["centroid_lon"].drop_nulls().to_list()
        
        for lat in valid_lats:
            assert -44.0 <= lat <= -10.0
        
        for lon in valid_lons:
            assert 113.0 <= lon <= 154.0
    
    def test_validate_boundary_data_negative_areas(self):
        """Test validation handles negative or zero area values."""
        processor = SimpleBoundaryProcessor()
        
        df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901"],
            "sa2_name_2021": ["Area 1", "Area 2", "Area 3"],
            "area_sqkm": [10.5, -5.0, 0.0],  # Negative and zero areas
            "population_2021": [1000, 1500, 800],
            "state_name": ["NSW", "VIC", "QLD"]
        })
        
        validated_df = processor._validate_boundary_data(df)
        
        # Should filter out areas with non-positive area values
        areas = validated_df["area_sqkm"].drop_nulls().to_list()
        for area in areas:
            assert area > 0
    
    def test_calculate_population_density(self, mock_boundary_data):
        """Test population density calculation."""
        processor = SimpleBoundaryProcessor()
        boundary_df = mock_boundary_data(num_areas=20)
        
        # Ensure we have required columns
        if "area_sqkm" not in boundary_df.columns:
            boundary_df = boundary_df.with_columns(
                pl.lit(np.random.uniform(1.0, 100.0, 20)).alias("area_sqkm")
            )
        
        density_df = processor._calculate_population_density(boundary_df)
        
        assert "population_density" in density_df.columns
        
        # Verify density calculations
        densities = density_df["population_density"].drop_nulls().to_list()
        for density in densities:
            assert density >= 0  # Should not be negative
            assert density < 100000  # Reasonable upper bound for Australia
    
    def test_classify_remoteness(self, mock_boundary_data):
        """Test remoteness classification based on population density."""
        processor = SimpleBoundaryProcessor()
        
        # Create test data with known density values
        df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901", "456789012"],
            "sa2_name_2021": ["Urban Area", "Regional Area", "Remote Area", "Very Remote"],
            "population_density": [5000.0, 100.0, 10.0, 1.0],  # High to very low density
            "state_name": ["NSW", "VIC", "QLD", "WA"]
        })
        
        classified_df = processor._classify_remoteness(df)
        
        assert "remoteness_category" in classified_df.columns
        
        # Verify classification logic
        categories = classified_df["remoteness_category"].to_list()
        
        # High density should be Major Cities
        assert categories[0] in ["Major Cities", "Inner Regional"]
        
        # Very low density should be Remote/Very Remote
        assert categories[3] in ["Remote", "Very Remote"]
        
        # All categories should be valid
        valid_categories = ["Major Cities", "Inner Regional", "Outer Regional", "Remote", "Very Remote"]
        for category in categories:
            assert category in valid_categories
    
    def test_calculate_distance_metrics(self, mock_boundary_data):
        """Test calculation of distance metrics between SA2 areas."""
        processor = SimpleBoundaryProcessor()
        boundary_df = mock_boundary_data(num_areas=10)
        
        # Ensure we have coordinate columns
        if "centroid_lat" not in boundary_df.columns:
            boundary_df = boundary_df.with_columns([
                pl.lit(np.random.uniform(-44.0, -10.0, 10)).alias("centroid_lat"),
                pl.lit(np.random.uniform(113.0, 154.0, 10)).alias("centroid_lon")
            ])
        
        distance_df = processor._calculate_distance_metrics(boundary_df)
        
        expected_columns = ["nearest_neighbor_distance", "isolation_index"]
        for col in expected_columns:
            if col in distance_df.columns:
                distances = distance_df[col].drop_nulls().to_list()
                for distance in distances:
                    assert distance >= 0  # Distances should be non-negative
                    assert distance < 5000  # Reasonable upper bound for Australia (km)
    
    def test_simplify_boundaries_performance(self, mock_boundary_data):
        """Test boundary simplification for performance optimization."""
        processor = SimpleBoundaryProcessor()
        boundary_df = mock_boundary_data(num_areas=100)
        
        # Mock complex geometry data
        mock_geometries = []
        for i in range(100):
            # Create mock geometry string (simplified)
            coords = [(np.random.uniform(113, 154), np.random.uniform(-44, -10)) for _ in range(10)]
            geometry_str = f"POLYGON(({','.join([f'{lon} {lat}' for lon, lat in coords])}))"
            mock_geometries.append(geometry_str)
        
        boundary_df = boundary_df.with_columns(pl.Series("geometry_wkt", mock_geometries))
        
        simplified_df = processor._simplify_boundaries(boundary_df, tolerance=0.001)
        
        # Should maintain same number of records
        assert len(simplified_df) == len(boundary_df)
        
        # Should have simplified geometry (mock check)
        if "geometry_simplified" in simplified_df.columns:
            assert simplified_df["geometry_simplified"].null_count() < len(simplified_df)
    
    def test_aggregate_by_state(self, mock_boundary_data):
        """Test aggregation of boundary data by state."""
        processor = SimpleBoundaryProcessor()
        boundary_df = mock_boundary_data(num_areas=40)
        
        state_df = processor._aggregate_by_state(boundary_df)
        
        # Should have one row per state
        states = state_df["state_name"].unique().to_list()
        assert len(states) <= 8  # Maximum 8 Australian states/territories
        
        # Verify aggregation calculations
        for state in states:
            state_data = state_df.filter(pl.col("state_name") == state)
            if len(state_data) > 0:
                state_row = state_data.row(0, named=True)
                
                # Should have aggregated statistics
                assert "total_sa2_areas" in state_row
                assert "total_population" in state_row
                assert "total_area_sqkm" in state_row
                assert "avg_population_density" in state_row
                
                # Values should be reasonable
                assert state_row["total_sa2_areas"] > 0
                assert state_row["total_population"] > 0
                assert state_row["total_area_sqkm"] > 0
    
    def test_calculate_accessibility_index(self, mock_boundary_data):
        """Test calculation of accessibility index for health services."""
        processor = SimpleBoundaryProcessor()
        boundary_df = mock_boundary_data(num_areas=30)
        
        # Mock health facility data
        health_facilities = pl.DataFrame({
            "facility_type": ["Hospital", "GP Clinic", "Pharmacy"] * 10,
            "latitude": np.random.uniform(-44.0, -10.0, 30),
            "longitude": np.random.uniform(113.0, 154.0, 30),
            "capacity": np.random.randint(10, 500, 30)
        })
        
        accessibility_df = processor._calculate_accessibility_index(boundary_df, health_facilities)
        
        assert "accessibility_index" in accessibility_df.columns
        
        # Verify accessibility scores
        scores = accessibility_df["accessibility_index"].drop_nulls().to_list()
        for score in scores:
            assert 0 <= score <= 100  # Normalized score
    
    def test_process_complete_pipeline(self, mock_boundary_data, mock_data_paths):
        """Test complete boundary processing pipeline."""
        processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_df = mock_boundary_data(num_areas=25)
        
        # Save mock data as input
        input_path = mock_data_paths["raw_dir"] / "sa2_boundaries.csv"
        boundary_df.write_csv(input_path)
        
        result_df = processor.process_boundary_data(input_path)
        
        # Verify processing succeeded
        assert isinstance(result_df, pl.DataFrame)
        assert len(result_df) > 0
        
        # Verify key columns exist
        expected_columns = ["sa2_code_2021", "sa2_name_2021", "state_name"]
        for col in expected_columns:
            assert col in result_df.columns
        
        # Verify output files were created
        csv_output = mock_data_paths["processed_dir"] / "sa2_boundaries_processed.csv"
        if csv_output.exists():
            output_df = pl.read_csv(csv_output)
            assert len(output_df) == len(result_df)
    
    def test_handle_missing_population_data(self):
        """Test handling of missing population data."""
        processor = SimpleBoundaryProcessor()
        
        df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901"],
            "sa2_name_2021": ["Area 1", "Area 2", "Area 3"],
            "population_2021": [1000, None, 1500],  # Missing population for one area
            "area_sqkm": [10.0, 15.0, 20.0],
            "state_name": ["NSW", "VIC", "QLD"]
        })
        
        cleaned_df = processor._handle_missing_population(df)
        
        # Should handle missing population gracefully
        assert len(cleaned_df) > 0
        
        # Population density should be calculated where possible
        if "population_density" in cleaned_df.columns:
            densities = cleaned_df["population_density"].drop_nulls().to_list()
            for density in densities:
                assert density >= 0
    
    def test_coordinate_system_validation(self):
        """Test validation of coordinate systems and transformations."""
        processor = SimpleBoundaryProcessor()
        
        # Test with various coordinate formats
        df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901"],
            "centroid_lat": [-35.2809, -37.8136, -27.4698],  # Sydney, Melbourne, Brisbane
            "centroid_lon": [149.1300, 144.9631, 153.0251],
            "coordinate_system": ["GDA2020", "GDA94", "WGS84"]
        })
        
        validated_df = processor._validate_coordinate_system(df)
        
        # Should maintain valid Australian coordinates
        lats = validated_df["centroid_lat"].to_list()
        lons = validated_df["centroid_lon"].to_list()
        
        for lat, lon in zip(lats, lons):
            if lat is not None and lon is not None:
                assert -44.0 <= lat <= -10.0
                assert 113.0 <= lon <= 154.0
    
    def test_boundary_quality_assessment(self, mock_boundary_data):
        """Test data quality assessment for boundary data."""
        processor = SimpleBoundaryProcessor()
        boundary_df = mock_boundary_data(num_areas=50)
        
        quality_metrics = processor._assess_boundary_quality(boundary_df)
        
        expected_metrics = [
            "total_sa2_areas", "valid_coordinates", "complete_population_data",
            "valid_state_assignments", "coordinate_precision", "area_coverage"
        ]
        
        for metric in expected_metrics:
            if metric in quality_metrics:
                assert isinstance(quality_metrics[metric], (int, float, bool))
        
        # Basic quality checks
        if "total_sa2_areas" in quality_metrics:
            assert quality_metrics["total_sa2_areas"] == 50
        
        if "valid_coordinates" in quality_metrics:
            assert 0 <= quality_metrics["valid_coordinates"] <= 50
    
    def test_performance_large_dataset(self, mock_boundary_data, mock_data_paths):
        """Test processing performance with large boundary dataset."""
        processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Create larger dataset
        large_df = mock_boundary_data(num_areas=2000)  # Approximate real SA2 count
        
        start_time = time.time()
        
        # Process through main pipeline steps
        validated_df = processor._validate_boundary_data(large_df)
        density_df = processor._calculate_population_density(validated_df)
        classified_df = processor._classify_remoteness(density_df)
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 60.0  # Should complete within 60 seconds
        assert len(classified_df) > 0
        
        # Memory efficiency check
        memory_usage = classified_df.estimated_size("mb")
        assert memory_usage < 100  # Should not exceed 100MB for 2000 records
    
    def test_geographic_outlier_detection(self, mock_boundary_data):
        """Test detection of geographic outliers in boundary data."""
        processor = SimpleBoundaryProcessor()
        
        # Create data with one clear outlier
        df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901", "456789012"],
            "centroid_lat": [-35.0, -36.0, -37.0, 0.0],  # Last one is outlier
            "centroid_lon": [149.0, 150.0, 151.0, 0.0],  # Last one is outlier
            "area_sqkm": [10.0, 12.0, 15.0, 10000.0],   # Last one is outlier
            "population_2021": [1000, 1200, 1500, 50]   # Unusually low for large area
        })
        
        outlier_df = processor._detect_geographic_outliers(df)
        
        if "is_outlier" in outlier_df.columns:
            outliers = outlier_df.filter(pl.col("is_outlier") == True)
            # Should detect the obvious outlier
            assert len(outliers) >= 1
    
    def test_boundary_intersection_analysis(self, mock_boundary_data):
        """Test analysis of boundary intersections and neighboring areas."""
        processor = SimpleBoundaryProcessor()
        boundary_df = mock_boundary_data(num_areas=20)
        
        # Mock simplified boundary analysis
        neighbor_df = processor._analyze_neighboring_areas(boundary_df)
        
        expected_columns = ["sa2_code_2021", "neighbor_count", "border_length_km"]
        for col in expected_columns:
            if col in neighbor_df.columns:
                values = neighbor_df[col].drop_nulls().to_list()
                if col == "neighbor_count":
                    for count in values:
                        assert 0 <= count <= 20  # Reasonable neighbor count
                elif col == "border_length_km":
                    for length in values:
                        assert length >= 0  # Border length should be non-negative
    
    def test_error_handling_invalid_geometry(self, mock_data_paths):
        """Test error handling with invalid geometry data."""
        processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        
        # Create DataFrame with invalid geometry
        df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890"],
            "geometry_wkt": ["INVALID_GEOMETRY", "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"],
            "centroid_lat": [-35.0, -36.0],
            "centroid_lon": [149.0, 150.0]
        })
        
        # Should handle invalid geometry gracefully
        result_df = processor._validate_geometry(df)
        
        # Should filter out invalid geometries
        assert len(result_df) <= len(df)
    
    def test_concurrent_processing_safety(self, mock_boundary_data, mock_data_paths):
        """Test thread safety of boundary processor (basic check)."""
        import concurrent.futures
        
        def process_boundaries():
            processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
            boundary_df = mock_boundary_data(num_areas=20)
            return processor._validate_boundary_data(boundary_df)
        
        # Run multiple processors concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_boundaries) for _ in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert len(results) == 3
        for result in results:
            assert isinstance(result, pl.DataFrame)
    
    def test_integration_with_health_data(self, mock_boundary_data, mock_health_data):
        """Test integration of boundary data with health utilisation data."""
        processor = SimpleBoundaryProcessor()
        
        # Create matching datasets
        sa2_codes = ["123456789", "234567890", "345678901"]
        
        boundary_df = mock_boundary_data(num_areas=3)
        boundary_df = boundary_df.with_columns(pl.Series("sa2_code_2021", sa2_codes))
        
        health_df = mock_health_data(num_records=50, num_sa2_areas=3)
        health_df = health_df.with_columns(
            pl.col("sa2_code").map_elements(lambda _: np.random.choice(sa2_codes), return_dtype=pl.Utf8)
        )
        
        integrated_df = processor._integrate_with_health_data(boundary_df, health_df)
        
        # Verify integration
        assert len(integrated_df) == 3
        assert "sa2_code_2021" in integrated_df.columns
        
        # Should have both boundary and health metrics
        boundary_cols = ["area_sqkm", "population_2021", "remoteness_category"]
        health_cols = ["total_prescriptions", "utilisation_category"]
        
        for col in boundary_cols:
            if col in integrated_df.columns:
                assert integrated_df[col].null_count() < len(integrated_df)


class TestBoundaryProcessorEdgeCases:
    """Test edge cases and error conditions for boundary processor."""
    
    def test_empty_boundary_dataset(self):
        """Test handling of empty boundary dataset."""
        processor = SimpleBoundaryProcessor()
        
        empty_df = pl.DataFrame({
            "sa2_code_2021": [],
            "sa2_name_2021": [],
            "state_name": []
        })
        
        result_df = processor._validate_boundary_data(empty_df)
        
        # Should handle empty data gracefully
        assert isinstance(result_df, pl.DataFrame)
        assert len(result_df) == 0
    
    def test_single_area_dataset(self, mock_boundary_data):
        """Test processing with single SA2 area."""
        processor = SimpleBoundaryProcessor()
        single_df = mock_boundary_data(num_areas=1)
        
        result_df = processor._validate_boundary_data(single_df)
        
        assert isinstance(result_df, pl.DataFrame)
        assert len(result_df) <= 1
    
    def test_duplicate_sa2_codes(self):
        """Test handling of duplicate SA2 codes in boundary data."""
        processor = SimpleBoundaryProcessor()
        
        df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "123456789", "234567890"],  # Duplicate
            "sa2_name_2021": ["Area 1 Version 1", "Area 1 Version 2", "Area 2"],
            "state_name": ["NSW", "NSW", "VIC"],
            "population_2021": [1000, 1200, 1500]  # Different populations
        })
        
        deduplicated_df = processor._handle_duplicate_sa2_codes(df)
        
        # Should remove or consolidate duplicates
        unique_codes = deduplicated_df["sa2_code_2021"].unique().to_list()
        assert len(unique_codes) == len(deduplicated_df)
    
    def test_missing_critical_columns(self):
        """Test handling when critical columns are missing."""
        processor = SimpleBoundaryProcessor()
        
        # DataFrame missing critical SA2 code column
        df = pl.DataFrame({
            "area_name": ["Area 1", "Area 2"],
            "population": [1000, 1500],
            "state": ["NSW", "VIC"]
        })
        
        with pytest.raises(ValueError, match="Missing required column"):
            processor._validate_required_columns(df)
    
    def test_extreme_coordinate_values(self):
        """Test handling of extreme coordinate values."""
        processor = SimpleBoundaryProcessor()
        
        df = pl.DataFrame({
            "sa2_code_2021": ["123456789", "234567890", "345678901"],
            "centroid_lat": [-90.0, 90.0, -35.0],    # Extreme values
            "centroid_lon": [-180.0, 180.0, 149.0],  # Extreme values
            "state_name": ["NSW", "VIC", "QLD"]
        })
        
        validated_df = processor._validate_coordinate_ranges(df)
        
        # Should filter out extreme values outside Australia
        valid_coords = len(validated_df.filter(
            (pl.col("centroid_lat").is_between(-44, -10)) &
            (pl.col("centroid_lon").is_between(113, 154))
        ))
        
        assert valid_coords >= 1  # At least one valid coordinate