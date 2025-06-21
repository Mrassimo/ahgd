"""Test suite for target output validation.

This module implements Test-Driven Development for output formats,
validating data warehouse exports, API responses, and web platform data.
"""

import pytest
import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
import pyarrow.parquet as pq
from io import StringIO

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DataWarehouseTableSpec:
    """Specification for data warehouse table structure."""
    table_name: str
    primary_key: List[str]
    required_columns: List[str]
    column_types: Dict[str, str]
    indexes: List[str]
    partitioning: Optional[str]
    retention_policy: Optional[str]


@dataclass
class APIResponseSpec:
    """Specification for API response format."""
    endpoint: str
    response_format: str  # json, geojson, csv
    required_fields: List[str]
    optional_fields: List[str]
    pagination_support: bool
    max_response_size_mb: int
    response_time_ms: int


@dataclass
class WebPlatformDataSpec:
    """Specification for web dashboard data structure."""
    data_type: str  # map_data, chart_data, summary_stats
    format: str  # json, geojson
    required_properties: List[str]
    performance_requirements: Dict[str, Any]
    caching_strategy: str


@dataclass
class ExportFormatSpec:
    """Specification for multi-format export requirements."""
    format: str  # parquet, csv, json, geojson
    compression: Optional[str]
    max_file_size_mb: int
    column_order: List[str]
    metadata_included: bool
    quality_checks: List[str]


class TestDataWarehouseExport:
    """Test data warehouse table structure validation."""
    
    @pytest.fixture
    def data_warehouse_specs(self):
        """Define data warehouse table specifications."""
        return [
            DataWarehouseTableSpec(
                table_name="master_health_records",
                primary_key=["sa2_code"],
                required_columns=[
                    "sa2_code", "sa2_name", "sa3_code", "sa4_code", "state_code",
                    "total_population", "seifa_irsad_score", "life_expectancy",
                    "geometry", "centroid_lat", "centroid_lon", "data_version", "last_updated"
                ],
                column_types={
                    "sa2_code": "VARCHAR(9)",
                    "sa2_name": "VARCHAR(255)",
                    "total_population": "INTEGER",
                    "seifa_irsad_score": "INTEGER",
                    "life_expectancy": "DECIMAL(5,2)",
                    "centroid_lat": "DECIMAL(10,7)",
                    "centroid_lon": "DECIMAL(10,7)",
                    "geometry": "GEOMETRY",
                    "last_updated": "TIMESTAMP"
                },
                indexes=["idx_sa2_code", "idx_state_code", "idx_seifa_decile"],
                partitioning="state_code",
                retention_policy="7_years"
            ),
            DataWarehouseTableSpec(
                table_name="health_indicators",
                primary_key=["sa2_code", "indicator_code"],
                required_columns=[
                    "sa2_code", "indicator_code", "indicator_value", "indicator_unit",
                    "reference_period", "data_source", "quality_flag"
                ],
                column_types={
                    "sa2_code": "VARCHAR(9)",
                    "indicator_code": "VARCHAR(50)",
                    "indicator_value": "DECIMAL(15,4)",
                    "indicator_unit": "VARCHAR(100)",
                    "reference_period": "VARCHAR(20)",
                    "data_source": "VARCHAR(100)",
                    "quality_flag": "VARCHAR(10)"
                },
                indexes=["idx_sa2_indicator", "idx_indicator_code", "idx_reference_period"],
                partitioning="reference_period",
                retention_policy="10_years"
            ),
            DataWarehouseTableSpec(
                table_name="geographic_boundaries",
                primary_key=["sa2_code"],
                required_columns=[
                    "sa2_code", "geometry", "area_sqkm", "perimeter_km",
                    "centroid_lat", "centroid_lon", "boundary_source", "accuracy_meters"
                ],
                column_types={
                    "sa2_code": "VARCHAR(9)",
                    "geometry": "GEOMETRY",
                    "area_sqkm": "DECIMAL(12,6)",
                    "perimeter_km": "DECIMAL(12,6)",
                    "centroid_lat": "DECIMAL(10,7)",
                    "centroid_lon": "DECIMAL(10,7)",
                    "boundary_source": "VARCHAR(100)",
                    "accuracy_meters": "INTEGER"
                },
                indexes=["idx_sa2_geo", "spatial_idx_geometry"],
                partitioning=None,
                retention_policy="permanent"
            )
        ]
    
    def test_data_warehouse_export(self, data_warehouse_specs):
        """Test validation of data warehouse table structure.
        
        Validates that exported data warehouse tables conform to
        expected schema, indexing, and partitioning requirements.
        """
        from src.etl.data_warehouse import DataWarehouse
        from src.testing.target_validation import TargetSchemaValidator
        
        warehouse = DataWarehouse()
        validator = TargetSchemaValidator()
        
        for spec in data_warehouse_specs:
            # Test table structure
            table_validation = validator.validate_table_structure(
                table_name=spec.table_name,
                expected_schema=spec
            )
            
            assert table_validation.table_exists, \
                f"Table {spec.table_name} does not exist in data warehouse"
            
            # Validate primary key
            assert table_validation.primary_key_valid, \
                f"Primary key validation failed for table {spec.table_name}"
            
            # Validate required columns exist
            missing_columns = table_validation.missing_required_columns
            assert len(missing_columns) == 0, \
                f"Missing required columns in {spec.table_name}: {missing_columns}"
            
            # Validate column types
            type_mismatches = table_validation.column_type_mismatches
            assert len(type_mismatches) == 0, \
                f"Column type mismatches in {spec.table_name}: {type_mismatches}"
            
            # Validate indexes
            missing_indexes = table_validation.missing_indexes
            assert len(missing_indexes) == 0, \
                f"Missing indexes in {spec.table_name}: {missing_indexes}"
            
            # Test data quality in table
            data_quality = warehouse.assess_table_data_quality(spec.table_name)
            assert data_quality.completeness_score >= 0.95, \
                f"Data quality too low in {spec.table_name}: {data_quality.completeness_score}"
    
    def test_table_performance_characteristics(self, data_warehouse_specs):
        """Test performance characteristics of data warehouse tables."""
        from src.etl.data_warehouse import DataWarehouse
        from src.performance.monitoring import PerformanceMonitor
        
        warehouse = DataWarehouse()
        monitor = PerformanceMonitor()
        
        for spec in data_warehouse_specs:
            # Test query performance
            performance_result = monitor.measure_table_performance(
                table_name=spec.table_name,
                operations=['select', 'insert', 'update']
            )
            
            # Basic queries should be fast
            assert performance_result.select_time_ms < 1000, \
                f"Select queries too slow on {spec.table_name}: {performance_result.select_time_ms}ms"
            
            # Inserts should be reasonable
            assert performance_result.insert_time_ms < 500, \
                f"Insert operations too slow on {spec.table_name}: {performance_result.insert_time_ms}ms"
    
    def test_data_warehouse_backup_recovery(self):
        """Test data warehouse backup and recovery procedures."""
        from src.etl.data_warehouse import DataWarehouse
        from src.utils.backup import BackupManager
        
        warehouse = DataWarehouse()
        backup_manager = BackupManager()
        
        # Test backup creation
        backup_result = backup_manager.create_full_backup()
        assert backup_result.success, "Backup creation failed"
        assert backup_result.backup_size_mb > 0, "Backup appears to be empty"
        
        # Test backup integrity
        integrity_check = backup_manager.verify_backup_integrity(backup_result.backup_id)
        assert integrity_check.valid, "Backup integrity check failed"
        
        # Test point-in-time recovery capability
        recovery_test = backup_manager.test_point_in_time_recovery()
        assert recovery_test.success, "Point-in-time recovery test failed"


class TestAPIResponseFormat:
    """Test API response schema compliance."""
    
    @pytest.fixture
    def api_specs(self):
        """Define API endpoint specifications."""
        return [
            APIResponseSpec(
                endpoint="/api/v1/sa2/{sa2_code}",
                response_format="json",
                required_fields=[
                    "sa2_code", "sa2_name", "total_population", "seifa_irsad_score",
                    "life_expectancy", "health_indicators", "geographic_info", "metadata"
                ],
                optional_fields=["related_areas", "historical_data", "forecasts"],
                pagination_support=False,
                max_response_size_mb=2,
                response_time_ms=500
            ),
            APIResponseSpec(
                endpoint="/api/v1/sa2/search",
                response_format="json",
                required_fields=[
                    "results", "total_count", "page", "page_size", "has_next", "has_previous"
                ],
                optional_fields=["filters_applied", "search_metadata"],
                pagination_support=True,
                max_response_size_mb=5,
                response_time_ms=1000
            ),
            APIResponseSpec(
                endpoint="/api/v1/boundaries/{sa2_code}",
                response_format="geojson",
                required_fields=[
                    "type", "features", "properties", "geometry"
                ],
                optional_fields=["crs", "bbox"],
                pagination_support=False,
                max_response_size_mb=10,
                response_time_ms=1500
            ),
            APIResponseSpec(
                endpoint="/api/v1/export/csv",
                response_format="csv",
                required_fields=[],  # CSV structure validated separately
                optional_fields=[],
                pagination_support=True,
                max_response_size_mb=50,
                response_time_ms=5000
            )
        ]
    
    def test_api_response_format(self, api_specs):
        """Test API response schema compliance.
        
        Validates that API endpoints return properly formatted responses
        with all required fields and correct data types.
        """
        from src.web.api import HealthDataAPI
        from src.testing.target_validation import TargetSchemaValidator
        
        api = HealthDataAPI()
        validator = TargetSchemaValidator()
        
        for spec in api_specs:
            # Test response structure
            if spec.endpoint.startswith("/api/v1/sa2/") and "{sa2_code}" in spec.endpoint:
                test_sa2_code = "101011007"
                endpoint = spec.endpoint.replace("{sa2_code}", test_sa2_code)
                response = api.get(endpoint)
            else:
                response = api.get(spec.endpoint)
            
            assert response.status_code == 200, \
                f"API endpoint {spec.endpoint} returned status {response.status_code}"
            
            # Validate response format
            if spec.response_format == "json":
                response_data = response.json()
                
                # Check required fields
                for field in spec.required_fields:
                    assert field in response_data, \
                        f"Required field {field} missing from {spec.endpoint} response"
                
                # Validate data types and structure
                validation_result = validator.validate_api_response(
                    response_data, spec
                )
                assert validation_result.is_valid, \
                    f"API response validation failed for {spec.endpoint}: {validation_result.errors}"
            
            elif spec.response_format == "geojson":
                geojson_data = response.json()
                
                # Validate GeoJSON structure
                assert "type" in geojson_data, "GeoJSON missing 'type' property"
                assert geojson_data["type"] in ["FeatureCollection", "Feature"], \
                    f"Invalid GeoJSON type: {geojson_data['type']}"
                
                if geojson_data["type"] == "FeatureCollection":
                    assert "features" in geojson_data, "FeatureCollection missing 'features'"
                    for feature in geojson_data["features"]:
                        assert "geometry" in feature, "Feature missing 'geometry'"
                        assert "properties" in feature, "Feature missing 'properties'"
            
            elif spec.response_format == "csv":
                csv_content = response.text
                
                # Validate CSV structure
                df = pd.read_csv(StringIO(csv_content))
                assert len(df.columns) > 0, "CSV response has no columns"
                assert len(df) > 0, "CSV response has no data rows"
    
    def test_api_performance_requirements(self, api_specs):
        """Test API performance requirements."""
        from src.web.api import HealthDataAPI
        from src.performance.monitoring import PerformanceMonitor
        import time
        
        api = HealthDataAPI()
        monitor = PerformanceMonitor()
        
        for spec in api_specs:
            # Measure response time
            start_time = time.time()
            
            if "{sa2_code}" in spec.endpoint:
                endpoint = spec.endpoint.replace("{sa2_code}", "101011007")
            else:
                endpoint = spec.endpoint
                
            response = api.get(endpoint)
            end_time = time.time()
            
            response_time_ms = (end_time - start_time) * 1000
            
            assert response_time_ms <= spec.response_time_ms, \
                f"API endpoint {spec.endpoint} response time {response_time_ms:.0f}ms " \
                f"exceeds requirement of {spec.response_time_ms}ms"
            
            # Test response size
            response_size_mb = len(response.content) / (1024 * 1024)
            assert response_size_mb <= spec.max_response_size_mb, \
                f"API endpoint {spec.endpoint} response size {response_size_mb:.2f}MB " \
                f"exceeds limit of {spec.max_response_size_mb}MB"
    
    def test_api_pagination_support(self, api_specs):
        """Test API pagination functionality."""
        from src.web.api import HealthDataAPI
        
        api = HealthDataAPI()
        
        paginated_specs = [spec for spec in api_specs if spec.pagination_support]
        
        for spec in paginated_specs:
            # Test first page
            first_page = api.get(f"{spec.endpoint}?page=1&page_size=10")
            assert first_page.status_code == 200
            
            first_data = first_page.json()
            assert "page" in first_data
            assert "page_size" in first_data
            assert "total_count" in first_data
            assert "has_next" in first_data
            assert "has_previous" in first_data
            
            # Validate pagination logic
            assert first_data["page"] == 1
            assert first_data["has_previous"] is False
            
            if first_data["total_count"] > 10:
                assert first_data["has_next"] is True
                
                # Test second page
                second_page = api.get(f"{spec.endpoint}?page=2&page_size=10")
                second_data = second_page.json()
                assert second_data["page"] == 2
                assert second_data["has_previous"] is True


class TestWebPlatformDataStructure:
    """Test web dashboard data formats."""
    
    @pytest.fixture
    def web_platform_specs(self):
        """Define web platform data specifications."""
        return [
            WebPlatformDataSpec(
                data_type="map_data",
                format="geojson",
                required_properties=[
                    "sa2_code", "sa2_name", "health_score", "seifa_decile",
                    "population", "color_value", "tooltip_data"
                ],
                performance_requirements={
                    "max_features": 2473,  # All SA2s in Australia
                    "max_file_size_mb": 25,
                    "load_time_ms": 3000
                },
                caching_strategy="browser_cache_24h"
            ),
            WebPlatformDataSpec(
                data_type="chart_data",
                format="json",
                required_properties=[
                    "labels", "datasets", "chart_type", "axis_config", "legend"
                ],
                performance_requirements={
                    "max_data_points": 10000,
                    "max_file_size_mb": 5,
                    "load_time_ms": 1000
                },
                caching_strategy="api_cache_1h"
            ),
            WebPlatformDataSpec(
                data_type="summary_stats",
                format="json",
                required_properties=[
                    "total_sa2s", "total_population", "health_indicators_summary",
                    "geographic_coverage", "data_freshness", "quality_metrics"
                ],
                performance_requirements={
                    "max_file_size_mb": 1,
                    "load_time_ms": 500
                },
                caching_strategy="api_cache_15min"
            )
        ]
    
    def test_web_platform_data_structure(self, web_platform_specs):
        """Test web dashboard data format validation.
        
        Validates that web platform data structures meet frontend
        requirements for maps, charts, and summary statistics.
        """
        from src.web.data_export_engine import WebDataExporter
        from src.testing.target_validation import TargetSchemaValidator
        
        exporter = WebDataExporter()
        validator = TargetSchemaValidator()
        
        for spec in web_platform_specs:
            # Generate web platform data
            web_data = exporter.generate_web_data(spec.data_type)
            
            # Validate format
            if spec.format == "geojson":
                assert "type" in web_data
                assert "features" in web_data
                
                # Validate feature properties
                for feature in web_data["features"]:
                    properties = feature["properties"]
                    for prop in spec.required_properties:
                        assert prop in properties, \
                            f"Required property {prop} missing from {spec.data_type} feature"
            
            elif spec.format == "json":
                # Validate required properties at root level
                for prop in spec.required_properties:
                    assert prop in web_data, \
                        f"Required property {prop} missing from {spec.data_type} data"
            
            # Validate performance requirements
            data_validation = validator.validate_web_platform_data(web_data, spec)
            
            assert data_validation.meets_performance_requirements, \
                f"Performance requirements not met for {spec.data_type}: {data_validation.performance_issues}"
    
    def test_web_data_caching_strategy(self, web_platform_specs):
        """Test caching strategy implementation for web data."""
        from src.web.data_export_engine import WebDataExporter
        from src.web.caching import CacheManager
        
        exporter = WebDataExporter()
        cache_manager = CacheManager()
        
        for spec in web_platform_specs:
            # Test cache implementation
            cache_config = cache_manager.get_cache_config(spec.data_type)
            
            assert cache_config.strategy == spec.caching_strategy, \
                f"Caching strategy mismatch for {spec.data_type}"
            
            # Test cache effectiveness
            cache_performance = cache_manager.measure_cache_performance(spec.data_type)
            
            assert cache_performance.hit_rate >= 0.80, \
                f"Cache hit rate too low for {spec.data_type}: {cache_performance.hit_rate}"
    
    def test_web_data_responsiveness(self, web_platform_specs):
        """Test responsive data loading for different device types."""
        from src.web.data_export_engine import WebDataExporter
        
        exporter = WebDataExporter()
        
        device_types = ['desktop', 'tablet', 'mobile']
        
        for spec in web_platform_specs:
            for device_type in device_types:
                # Generate device-optimised data
                optimised_data = exporter.generate_responsive_data(
                    spec.data_type, device_type
                )
                
                # Validate size optimisation
                if device_type == 'mobile':
                    # Mobile should have reduced data size
                    mobile_size = len(json.dumps(optimised_data))
                    desktop_data = exporter.generate_web_data(spec.data_type)
                    desktop_size = len(json.dumps(desktop_data))
                    
                    assert mobile_size <= desktop_size * 0.7, \
                        f"Mobile data not sufficiently optimised for {spec.data_type}"


class TestMultiFormatExports:
    """Test Parquet, CSV, JSON, GeoJSON exports."""
    
    @pytest.fixture
    def export_format_specs(self):
        """Define export format specifications."""
        return [
            ExportFormatSpec(
                format="parquet",
                compression="snappy",
                max_file_size_mb=100,
                column_order=[
                    "sa2_code", "sa2_name", "state_code", "total_population",
                    "seifa_irsad_score", "life_expectancy", "last_updated"
                ],
                metadata_included=True,
                quality_checks=["schema_validation", "data_completeness", "referential_integrity"]
            ),
            ExportFormatSpec(
                format="csv",
                compression="gzip",
                max_file_size_mb=50,
                column_order=[
                    "sa2_code", "sa2_name", "state_name", "total_population",
                    "seifa_irsad_score", "seifa_irsad_decile", "life_expectancy"
                ],
                metadata_included=False,
                quality_checks=["data_completeness", "format_validation"]
            ),
            ExportFormatSpec(
                format="json",
                compression="gzip",
                max_file_size_mb=75,
                column_order=[],  # JSON doesn't enforce column order
                metadata_included=True,
                quality_checks=["json_schema_validation", "data_completeness"]
            ),
            ExportFormatSpec(
                format="geojson",
                compression=None,  # GeoJSON typically not compressed for web use
                max_file_size_mb=200,
                column_order=[],  # GeoJSON has standardised structure
                metadata_included=True,
                quality_checks=["geojson_validation", "geometry_validation", "crs_validation"]
            )
        ]
    
    def test_multi_format_exports(self, export_format_specs):
        """Test validation of Parquet, CSV, JSON, GeoJSON exports.
        
        Validates that data can be exported in multiple formats while
        maintaining data integrity and meeting format-specific requirements.
        """
        from src.web.data_export_engine import MultiFormatExporter
        from src.testing.target_validation import TargetSchemaValidator
        
        exporter = MultiFormatExporter()
        validator = TargetSchemaValidator()
        
        # Get test dataset
        test_data = exporter.get_master_dataset()
        
        for spec in export_format_specs:
            # Export in specified format
            export_result = exporter.export_data(
                data=test_data,
                format=spec.format,
                compression=spec.compression
            )
            
            assert export_result.success, \
                f"Export failed for format {spec.format}: {export_result.error_message}"
            
            # Validate file size
            file_size_mb = export_result.file_size_bytes / (1024 * 1024)
            assert file_size_mb <= spec.max_file_size_mb, \
                f"Export file size {file_size_mb:.2f}MB exceeds limit for {spec.format}"
            
            # Format-specific validation
            if spec.format == "parquet":
                # Validate Parquet file structure
                parquet_file = pq.ParquetFile(export_result.file_path)
                schema = parquet_file.schema_arrow
                
                # Check column order
                actual_columns = [field.name for field in schema]
                expected_columns = spec.column_order
                for i, col in enumerate(expected_columns):
                    if i < len(actual_columns):
                        assert actual_columns[i] == col, \
                            f"Column order mismatch in Parquet: expected {col} at position {i}"
                
                # Check metadata
                if spec.metadata_included:
                    metadata = parquet_file.metadata
                    assert metadata is not None, "Parquet metadata missing"
            
            elif spec.format == "csv":
                # Validate CSV structure
                import csv
                with open(export_result.file_path, 'r') as f:
                    csv_reader = csv.reader(f)
                    header = next(csv_reader)
                    
                    # Check column order
                    for i, col in enumerate(spec.column_order):
                        if i < len(header):
                            assert header[i] == col, \
                                f"CSV column order mismatch: expected {col} at position {i}"
                    
                    # Check data rows exist
                    first_row = next(csv_reader, None)
                    assert first_row is not None, "CSV file has no data rows"
            
            elif spec.format == "json":
                # Validate JSON structure
                with open(export_result.file_path, 'r') as f:
                    json_data = json.load(f)
                    
                    assert isinstance(json_data, (list, dict)), "Invalid JSON structure"
                    
                    if spec.metadata_included:
                        if isinstance(json_data, dict):
                            assert "metadata" in json_data, "JSON metadata missing"
                        elif isinstance(json_data, list) and len(json_data) > 0:
                            # Check if first item contains metadata
                            assert "data_version" in json_data[0] or "metadata" in json_data[0], \
                                "JSON metadata missing from data items"
            
            elif spec.format == "geojson":
                # Validate GeoJSON structure
                with open(export_result.file_path, 'r') as f:
                    geojson_data = json.load(f)
                    
                    assert "type" in geojson_data, "GeoJSON missing 'type' property"
                    assert geojson_data["type"] == "FeatureCollection", \
                        f"Invalid GeoJSON type: {geojson_data['type']}"
                    
                    assert "features" in geojson_data, "GeoJSON missing 'features' array"
                    
                    # Validate each feature
                    for feature in geojson_data["features"]:
                        assert "type" in feature and feature["type"] == "Feature", \
                            "Invalid feature type in GeoJSON"
                        assert "geometry" in feature, "Feature missing geometry"
                        assert "properties" in feature, "Feature missing properties"
                        
                        # Validate geometry structure
                        geometry = feature["geometry"]
                        assert "type" in geometry, "Geometry missing type"
                        assert "coordinates" in geometry, "Geometry missing coordinates"
            
            # Run quality checks
            for check in spec.quality_checks:
                quality_result = validator.run_quality_check(
                    export_result.file_path, check, spec.format
                )
                assert quality_result.passed, \
                    f"Quality check {check} failed for {spec.format}: {quality_result.message}"
    
    def test_export_data_consistency(self, export_format_specs):
        """Test that data remains consistent across different export formats."""
        from src.web.data_export_engine import MultiFormatExporter
        
        exporter = MultiFormatExporter()
        test_data = exporter.get_master_dataset()
        
        # Export to all formats
        exports = {}
        for spec in export_format_specs:
            export_result = exporter.export_data(
                data=test_data,
                format=spec.format
            )
            exports[spec.format] = export_result
        
        # Compare data consistency
        # Convert all formats back to pandas DataFrame for comparison
        dataframes = {}
        
        for format_name, export_result in exports.items():
            if format_name == "parquet":
                df = pd.read_parquet(export_result.file_path)
            elif format_name == "csv":
                df = pd.read_csv(export_result.file_path)
            elif format_name == "json":
                df = pd.read_json(export_result.file_path)
            elif format_name == "geojson":
                gdf = gpd.read_file(export_result.file_path)
                df = pd.DataFrame(gdf.drop('geometry', axis=1))
            
            dataframes[format_name] = df
        
        # Compare key fields across formats
        reference_df = dataframes['parquet']  # Use Parquet as reference
        key_columns = ['sa2_code', 'total_population', 'seifa_irsad_score']
        
        for format_name, df in dataframes.items():
            if format_name != 'parquet':
                for col in key_columns:
                    if col in df.columns and col in reference_df.columns:
                        # Allow for minor floating point differences
                        if df[col].dtype in ['float64', 'float32']:
                            diff = abs(df[col] - reference_df[col]).max()
                            assert diff < 0.001, \
                                f"Data inconsistency in {col} between parquet and {format_name}: max diff {diff}"
                        else:
                            assert df[col].equals(reference_df[col]), \
                                f"Data inconsistency in {col} between parquet and {format_name}"
    
    def test_export_compression_efficiency(self, export_format_specs):
        """Test compression efficiency for different export formats."""
        from src.web.data_export_engine import MultiFormatExporter
        import os
        
        exporter = MultiFormatExporter()
        test_data = exporter.get_master_dataset()
        
        for spec in export_format_specs:
            if spec.compression:
                # Export with compression
                compressed_result = exporter.export_data(
                    data=test_data,
                    format=spec.format,
                    compression=spec.compression
                )
                
                # Export without compression
                uncompressed_result = exporter.export_data(
                    data=test_data,
                    format=spec.format,
                    compression=None
                )
                
                compressed_size = os.path.getsize(compressed_result.file_path)
                uncompressed_size = os.path.getsize(uncompressed_result.file_path)
                
                compression_ratio = compressed_size / uncompressed_size
                
                # Compression should achieve at least 20% reduction
                assert compression_ratio < 0.80, \
                    f"Compression efficiency too low for {spec.format} with {spec.compression}: {compression_ratio:.2%}"
