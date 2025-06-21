"""
Performance tests for spatial operations in the AHGD geographic pipeline.

Tests the performance characteristics of coordinate transformations, boundary processing,
spatial indexing, and geographic standardisation operations to ensure they meet
performance requirements for production use.
"""

import unittest
import time
import psutil
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import statistics
import pytest

# Import modules under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.transformers.geographic_standardiser import GeographicStandardiser, SA2MappingEngine
from src.transformers.coordinate_transformer import CoordinateSystemTransformer
from src.transformers.boundary_processor import BoundaryProcessor, SpatialIndexBuilder
from src.utils.geographic_utils import DistanceCalculator, PostcodeValidator
from src.utils.interfaces import ValidationSeverity


class PerformanceBenchmark:
    """Utility class for performance benchmarking."""
    
    def __init__(self):
        self.measurements = {}
    
    def start_measurement(self, operation_name: str):
        """Start timing an operation."""
        self.measurements[operation_name] = {
            'start_time': time.time(),
            'start_memory': psutil.Process().memory_info().rss / 1024 / 1024,  # MB
            'start_cpu': psutil.Process().cpu_percent()
        }
    
    def end_measurement(self, operation_name: str, record_count: int = None):
        """End timing an operation and record results."""
        if operation_name not in self.measurements:
            raise ValueError(f"No measurement started for {operation_name}")
        
        measurement = self.measurements[operation_name]
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        measurement.update({
            'end_time': end_time,
            'duration_seconds': end_time - measurement['start_time'],
            'end_memory': end_memory,
            'memory_delta_mb': end_memory - measurement['start_memory'],
            'record_count': record_count
        })
        
        if record_count:
            measurement['records_per_second'] = record_count / measurement['duration_seconds']
            measurement['memory_per_record_kb'] = (measurement['memory_delta_mb'] * 1024) / record_count
        
        return measurement
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        return {
            operation: {
                'duration_seconds': data['duration_seconds'],
                'memory_delta_mb': data['memory_delta_mb'],
                'records_per_second': data.get('records_per_second', 0),
                'memory_per_record_kb': data.get('memory_per_record_kb', 0)
            }
            for operation, data in self.measurements.items()
        }


class TestGeographicStandardisationPerformance(unittest.TestCase):
    """Test performance of geographic standardisation operations."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.benchmark = PerformanceBenchmark()
        self.config = {
            'batch_size': 1000,
            'enable_caching': True,
            'strict_validation': False,
            'handle_invalid_codes': 'warn'
        }
        self.standardiser = GeographicStandardiser(config=self.config)
        
        # Performance thresholds
        self.max_processing_time_per_1000_records = 10.0  # seconds
        self.max_memory_per_record = 10.0  # KB
        self.min_throughput_records_per_second = 100
    
    def create_test_data(self, size: int, data_type: str = "postcode") -> List[Dict[str, Any]]:
        """Create test data of specified size."""
        test_data = []
        
        if data_type == "postcode":
            base_postcodes = ["2000", "3000", "4000", "5000", "6000", "7000"]
            for i in range(size):
                base_postcode = base_postcodes[i % len(base_postcodes)]
                postcode = f"{int(base_postcode) + (i // len(base_postcodes)):04d}"
                test_data.append({
                    'location_code': postcode,
                    'location_type': 'postcode',
                    'id': i,
                    'value': i * 100
                })
        
        elif data_type == "sa2":
            base_sa2 = "101021007"
            for i in range(size):
                sa2_code = f"{int(base_sa2) + i:09d}"
                test_data.append({
                    'location_code': sa2_code,
                    'location_type': 'sa2',
                    'id': i,
                    'value': i * 50
                })
        
        elif data_type == "mixed":
            types = ["postcode", "sa2", "sa3"]
            for i in range(size):
                data_type_selected = types[i % len(types)]
                
                if data_type_selected == "postcode":
                    code = f"{2000 + (i % 1000):04d}"
                elif data_type_selected == "sa2":
                    code = f"{101021007 + i:09d}"
                else:  # sa3
                    code = f"{10102 + (i % 100):05d}"
                
                test_data.append({
                    'location_code': code,
                    'location_type': data_type_selected,
                    'id': i,
                    'value': i * 75
                })
        
        return test_data
    
    def test_postcode_mapping_performance_small_dataset(self):
        """Test postcode mapping performance with small dataset (1,000 records)."""
        test_data = self.create_test_data(1000, "postcode")
        
        self.benchmark.start_measurement("postcode_mapping_1k")
        result = self.standardiser.transform(test_data)
        measurement = self.benchmark.end_measurement("postcode_mapping_1k", len(test_data))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 5.0, "Should process 1k postcodes in under 5 seconds")
        self.assertGreater(measurement['records_per_second'], 200, "Should process at least 200 records/second")
        self.assertLess(measurement['memory_delta_mb'], 50, "Should use less than 50MB additional memory")
        
        # Correctness assertions
        self.assertGreater(len(result), len(test_data))  # May have 1:many mappings
        self.assertTrue(all('sa2_code' in record for record in result))
    
    def test_postcode_mapping_performance_medium_dataset(self):
        """Test postcode mapping performance with medium dataset (10,000 records)."""
        test_data = self.create_test_data(10000, "postcode")
        
        self.benchmark.start_measurement("postcode_mapping_10k")
        result = self.standardiser.transform(test_data)
        measurement = self.benchmark.end_measurement("postcode_mapping_10k", len(test_data))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 30.0, "Should process 10k postcodes in under 30 seconds")
        self.assertGreater(measurement['records_per_second'], 300, "Should process at least 300 records/second")
        self.assertLess(measurement['memory_delta_mb'], 200, "Should use less than 200MB additional memory")
        
        # Get processing statistics
        stats = self.standardiser.get_transformation_statistics()
        self.assertEqual(stats['total_records'], 10000)
        self.assertGreater(stats['successful_mappings'], 9000)  # Allow some failures
    
    def test_postcode_mapping_performance_large_dataset(self):
        """Test postcode mapping performance with large dataset (100,000 records)."""
        test_data = self.create_test_data(100000, "postcode")
        
        self.benchmark.start_measurement("postcode_mapping_100k")
        result = self.standardiser.transform(test_data)
        measurement = self.benchmark.end_measurement("postcode_mapping_100k", len(test_data))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 300.0, "Should process 100k postcodes in under 5 minutes")
        self.assertGreater(measurement['records_per_second'], 300, "Should maintain at least 300 records/second")
        self.assertLess(measurement['memory_delta_mb'], 1000, "Should use less than 1GB additional memory")
        
        # Memory efficiency
        self.assertLess(measurement['memory_per_record_kb'], 10, "Should use less than 10KB per record")
    
    def test_mixed_geographic_types_performance(self):
        """Test performance with mixed geographic data types."""
        test_data = self.create_test_data(50000, "mixed")
        
        self.benchmark.start_measurement("mixed_types_50k")
        result = self.standardiser.transform(test_data)
        measurement = self.benchmark.end_measurement("mixed_types_50k", len(test_data))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 180.0, "Should process 50k mixed records in under 3 minutes")
        self.assertGreater(measurement['records_per_second'], 250, "Should process at least 250 records/second")
        
        # Get statistics by mapping method
        stats = self.standardiser.get_transformation_statistics()
        self.assertIn('mapping_methods', stats)
        self.assertGreater(len(stats['mapping_methods']), 1)  # Should have multiple methods
    
    def test_caching_performance_improvement(self):
        """Test that caching improves performance for repeated operations."""
        # Create data with repeated postcodes
        test_data = []
        repeated_postcodes = ["2000", "3000", "4000", "5000"]
        
        for i in range(10000):
            postcode = repeated_postcodes[i % len(repeated_postcodes)]
            test_data.append({
                'location_code': postcode,
                'location_type': 'postcode',
                'id': i
            })
        
        # Test with caching enabled
        self.benchmark.start_measurement("with_caching")
        result_cached = self.standardiser.transform(test_data)
        measurement_cached = self.benchmark.end_measurement("with_caching", len(test_data))
        
        # Test with caching disabled
        no_cache_config = self.config.copy()
        no_cache_config['enable_caching'] = False
        standardiser_no_cache = GeographicStandardiser(config=no_cache_config)
        
        self.benchmark.start_measurement("without_caching")
        result_no_cache = standardiser_no_cache.transform(test_data)
        measurement_no_cache = self.benchmark.end_measurement("without_caching", len(test_data))
        
        # Caching should improve performance
        self.assertLess(measurement_cached['duration_seconds'], measurement_no_cache['duration_seconds'])
        self.assertGreater(measurement_cached['records_per_second'], measurement_no_cache['records_per_second'])
        
        # Results should be identical
        self.assertEqual(len(result_cached), len(result_no_cache))


class TestCoordinateTransformationPerformance(unittest.TestCase):
    """Test performance of coordinate transformation operations."""
    
    def setUp(self):
        """Set up coordinate transformation performance tests."""
        self.benchmark = PerformanceBenchmark()
        self.config = {
            'target_system': 'GDA2020',
            'auto_determine_mga_zone': True,
            'enable_validation': True
        }
        self.transformer = CoordinateSystemTransformer(config=self.config)
    
    def create_coordinate_data(self, size: int, coordinate_systems: List[str] = None) -> List[Dict[str, Any]]:
        """Create coordinate test data."""
        if coordinate_systems is None:
            coordinate_systems = ['WGS84', 'GDA94', 'GDA2020']
        
        test_data = []
        
        # Australian coordinate bounds for realistic test data
        lon_range = (110.0, 155.0)
        lat_range = (-45.0, -9.0)
        
        for i in range(size):
            # Generate coordinates within Australian bounds
            lon = lon_range[0] + (lon_range[1] - lon_range[0]) * (i % 1000) / 1000
            lat = lat_range[0] + (lat_range[1] - lat_range[0]) * ((i // 1000) % 1000) / 1000
            
            coord_system = coordinate_systems[i % len(coordinate_systems)]
            
            test_data.append({
                'longitude': lon,
                'latitude': lat,
                'coordinate_system': coord_system,
                'id': i
            })
        
        return test_data
    
    def test_coordinate_transformation_performance_small(self):
        """Test coordinate transformation with small dataset (1,000 points)."""
        test_data = self.create_coordinate_data(1000)
        
        self.benchmark.start_measurement("coord_transform_1k")
        result = self.transformer.transform(test_data)
        measurement = self.benchmark.end_measurement("coord_transform_1k", len(test_data))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 3.0, "Should transform 1k coordinates in under 3 seconds")
        self.assertGreater(measurement['records_per_second'], 300, "Should process at least 300 coordinates/second")
        
        # Correctness assertions
        self.assertEqual(len(result), len(test_data))
        for record in result:
            self.assertIn('longitude_gda2020', record)
            self.assertIn('latitude_gda2020', record)
    
    def test_coordinate_transformation_performance_large(self):
        """Test coordinate transformation with large dataset (100,000 points)."""
        test_data = self.create_coordinate_data(100000)
        
        self.benchmark.start_measurement("coord_transform_100k")
        result = self.transformer.transform(test_data)
        measurement = self.benchmark.end_measurement("coord_transform_100k", len(test_data))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 120.0, "Should transform 100k coordinates in under 2 minutes")
        self.assertGreater(measurement['records_per_second'], 800, "Should process at least 800 coordinates/second")
        self.assertLess(measurement['memory_delta_mb'], 500, "Should use less than 500MB additional memory")
    
    def test_mga_zone_determination_performance(self):
        """Test performance of MGA zone determination."""
        # Create coordinates spread across different MGA zones
        test_data = []
        for i in range(50000):
            lon = 112 + (44 * i / 50000)  # Spread across zones 49-56
            lat = -35 + (20 * (i % 1000) / 1000)  # Vary latitude
            
            test_data.append({
                'longitude': lon,
                'latitude': lat,
                'coordinate_system': 'GDA2020',
                'id': i
            })
        
        self.benchmark.start_measurement("mga_zone_determination")
        result = self.transformer.transform(test_data)
        measurement = self.benchmark.end_measurement("mga_zone_determination", len(test_data))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 60.0, "Should determine MGA zones in under 1 minute")
        self.assertGreater(measurement['records_per_second'], 800, "Should process at least 800 coordinates/second")
        
        # Verify MGA zones were determined
        mga_zones_found = set()
        for record in result:
            if record.get('mga_zone'):
                mga_zones_found.add(record['mga_zone'])
        
        self.assertGreater(len(mga_zones_found), 3, "Should find multiple MGA zones")


class TestBoundaryProcessingPerformance(unittest.TestCase):
    """Test performance of boundary processing operations."""
    
    def setUp(self):
        """Set up boundary processing performance tests."""
        self.benchmark = PerformanceBenchmark()
        self.config = {
            'enable_validation': True,
            'enable_simplification': True,
            'enable_spatial_indexing': True,
            'simplification_level': 'medium'
        }
        self.processor = BoundaryProcessor(config=self.config)
    
    def create_boundary_data(self, size: int, complexity: str = "medium") -> List[Dict[str, Any]]:
        """Create boundary test data with varying complexity."""
        test_data = []
        
        # Define complexity levels
        point_counts = {
            "simple": 10,
            "medium": 100,
            "complex": 1000
        }
        
        points_per_boundary = point_counts.get(complexity, 100)
        
        for i in range(size):
            # Create a rectangular boundary with specified number of points
            base_lon = 150.0 + (i % 100) * 0.01
            base_lat = -35.0 + (i // 100) * 0.01
            
            # Generate points around a rectangular boundary
            coordinates = []
            for j in range(points_per_boundary):
                angle = 2 * 3.14159 * j / points_per_boundary
                lon = base_lon + 0.01 * (angle / (2 * 3.14159))
                lat = base_lat + 0.005 * ((j % 2) * 2 - 1)
                coordinates.append([lon, lat])
            
            # Close the polygon
            coordinates.append(coordinates[0])
            
            test_data.append({
                'area_code': f'{101021007 + i:09d}',
                'area_type': 'SA2',
                'area_name': f'Test Area {i}',
                'state_code': '1',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coordinates]
                }
            })
        
        return test_data
    
    def test_boundary_validation_performance(self):
        """Test boundary validation performance."""
        test_data = self.create_boundary_data(1000, "medium")
        
        self.benchmark.start_measurement("boundary_validation")
        result = self.processor.transform(test_data)
        measurement = self.benchmark.end_measurement("boundary_validation", len(test_data))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 30.0, "Should validate 1k boundaries in under 30 seconds")
        self.assertGreater(measurement['records_per_second'], 30, "Should process at least 30 boundaries/second")
        
        # Verify processing occurred
        stats = self.processor.get_processing_statistics()
        self.assertEqual(stats['boundaries_processed'], 1000)
    
    def test_geometry_simplification_performance(self):
        """Test geometry simplification performance."""
        # Create complex boundaries for simplification testing
        test_data = self.create_boundary_data(500, "complex")
        
        self.benchmark.start_measurement("geometry_simplification")
        result = self.processor.transform(test_data)
        measurement = self.benchmark.end_measurement("geometry_simplification", len(test_data))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 60.0, "Should simplify 500 complex boundaries in under 1 minute")
        self.assertGreater(measurement['records_per_second'], 8, "Should process at least 8 complex boundaries/second")
        
        # Verify simplification occurred
        stats = self.processor.get_processing_statistics()
        self.assertGreater(stats['simplification_ratio'], 0.0)
        self.assertLess(stats['simplification_ratio'], 1.0)
    
    def test_spatial_indexing_performance(self):
        """Test spatial indexing performance."""
        test_data = self.create_boundary_data(2000, "simple")
        
        # Process boundaries to build spatial index
        self.benchmark.start_measurement("spatial_indexing")
        result = self.processor.transform(test_data)
        measurement = self.benchmark.end_measurement("spatial_indexing", len(test_data))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 45.0, "Should index 2k boundaries in under 45 seconds")
        self.assertGreater(measurement['records_per_second'], 40, "Should process at least 40 boundaries/second")
        
        # Verify spatial index was built
        stats = self.processor.get_processing_statistics()
        self.assertTrue(stats['spatial_index_built'])


class TestSpatialIndexQueryPerformance(unittest.TestCase):
    """Test performance of spatial index queries."""
    
    def setUp(self):
        """Set up spatial index query performance tests."""
        self.benchmark = PerformanceBenchmark()
        self.config = {'index_type': 'rtree'}
        self.spatial_indexer = SpatialIndexBuilder(self.config)
        
        # Create test boundaries for indexing
        self.test_boundaries = self.create_test_boundaries(10000)
        
        # Build spatial index
        self.index_info = self.spatial_indexer.build_spatial_index(self.test_boundaries)
    
    def create_test_boundaries(self, count: int):
        """Create test boundary data for spatial indexing."""
        from src.transformers.boundary_processor import BoundaryRecord, GeometryInfo
        
        boundaries = []
        for i in range(count):
            # Create simple rectangular boundaries across Australia
            base_lon = 115.0 + (30.0 * (i % 100) / 100)  # Spread across longitude
            base_lat = -40.0 + (25.0 * (i // 100) / 100)  # Spread across latitude
            
            bbox = (base_lon, base_lat, base_lon + 0.1, base_lat + 0.1)
            
            geometry_info = GeometryInfo(
                geometry_type="Polygon",
                coordinate_count=5,
                bbox=bbox,
                area_sqkm=100.0
            )
            
            boundary = BoundaryRecord(
                area_code=f'{101000000 + i:09d}',
                area_type='SA2',
                area_name=f'Test Area {i}',
                state_code='1',
                geometry={'type': 'Polygon', 'coordinates': []},
                geometry_info=geometry_info
            )
            
            boundaries.append(boundary)
        
        return boundaries
    
    def test_point_query_performance(self):
        """Test point-in-polygon query performance."""
        # Prepare test points
        test_points = []
        for i in range(1000):
            lon = 115.0 + (30.0 * (i % 100) / 100)
            lat = -40.0 + (25.0 * (i // 100) / 100)
            test_points.append((lon, lat))
        
        # Benchmark point queries
        self.benchmark.start_measurement("point_queries")
        
        results = []
        for lon, lat in test_points:
            candidates = self.spatial_indexer.query_point(lon, lat, 'SA2')
            results.append(candidates)
        
        measurement = self.benchmark.end_measurement("point_queries", len(test_points))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 1.0, "Should query 1k points in under 1 second")
        self.assertGreater(measurement['records_per_second'], 1000, "Should query at least 1000 points/second")
        
        # Verify some results were found
        non_empty_results = [r for r in results if r]
        self.assertGreater(len(non_empty_results), 0, "Should find some intersecting boundaries")
    
    def test_bbox_query_performance(self):
        """Test bounding box query performance."""
        # Prepare test bounding boxes
        test_bboxes = []
        for i in range(500):
            min_lon = 115.0 + (29.0 * (i % 50) / 50)
            min_lat = -40.0 + (24.0 * (i // 50) / 50)
            max_lon = min_lon + 1.0
            max_lat = min_lat + 1.0
            test_bboxes.append((min_lon, min_lat, max_lon, max_lat))
        
        # Benchmark bbox queries
        self.benchmark.start_measurement("bbox_queries")
        
        results = []
        for min_lon, min_lat, max_lon, max_lat in test_bboxes:
            candidates = self.spatial_indexer.query_bbox(min_lon, min_lat, max_lon, max_lat, 'SA2')
            results.append(candidates)
        
        measurement = self.benchmark.end_measurement("bbox_queries", len(test_bboxes))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 0.5, "Should query 500 bboxes in under 0.5 seconds")
        self.assertGreater(measurement['records_per_second'], 1000, "Should query at least 1000 bboxes/second")


class TestDistanceCalculationPerformance(unittest.TestCase):
    """Test performance of distance calculation operations."""
    
    def setUp(self):
        """Set up distance calculation performance tests."""
        self.benchmark = PerformanceBenchmark()
        self.calculator = DistanceCalculator()
    
    def test_great_circle_distance_performance(self):
        """Test great circle distance calculation performance."""
        # Create coordinate pairs for distance calculation
        coordinates = []
        for i in range(100000):
            lon1 = 115.0 + (30.0 * (i % 1000) / 1000)
            lat1 = -40.0 + (25.0 * (i % 1000) / 1000)
            lon2 = lon1 + 0.1
            lat2 = lat1 + 0.1
            coordinates.append((lon1, lat1, lon2, lat2))
        
        # Benchmark distance calculations
        self.benchmark.start_measurement("great_circle_distances")
        
        distances = []
        for lon1, lat1, lon2, lat2 in coordinates:
            distance = self.calculator.calculate_great_circle_distance(lon1, lat1, lon2, lat2)
            distances.append(distance)
        
        measurement = self.benchmark.end_measurement("great_circle_distances", len(coordinates))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 10.0, "Should calculate 100k distances in under 10 seconds")
        self.assertGreater(measurement['records_per_second'], 10000, "Should calculate at least 10k distances/second")
        
        # Verify reasonable results
        self.assertEqual(len(distances), len(coordinates))
        avg_distance = statistics.mean(distances)
        self.assertGreater(avg_distance, 0, "Distances should be positive")
        self.assertLess(avg_distance, 50, "Average distance should be reasonable")
    
    def test_haversine_distance_performance(self):
        """Test haversine distance calculation performance."""
        coordinates = []
        for i in range(50000):
            lon1 = 115.0 + (30.0 * (i % 500) / 500)
            lat1 = -40.0 + (25.0 * (i % 500) / 500)
            lon2 = lon1 + 0.2
            lat2 = lat1 + 0.2
            coordinates.append((lon1, lat1, lon2, lat2))
        
        self.benchmark.start_measurement("haversine_distances")
        
        distances = []
        for lon1, lat1, lon2, lat2 in coordinates:
            distance = self.calculator.calculate_haversine_distance(lon1, lat1, lon2, lat2)
            distances.append(distance)
        
        measurement = self.benchmark.end_measurement("haversine_distances", len(coordinates))
        
        # Performance assertions
        self.assertLess(measurement['duration_seconds'], 8.0, "Should calculate 50k haversine distances in under 8 seconds")
        self.assertGreater(measurement['records_per_second'], 6000, "Should calculate at least 6k distances/second")


class TestOverallPipelinePerformance(unittest.TestCase):
    """Test overall pipeline performance with realistic workloads."""
    
    def setUp(self):
        """Set up overall pipeline performance tests."""
        self.benchmark = PerformanceBenchmark()
        
        # Configure pipeline components
        self.geographic_standardiser = GeographicStandardiser(config={
            'batch_size': 5000,
            'enable_caching': True
        })
        
        self.coordinate_transformer = CoordinateSystemTransformer(config={
            'target_system': 'GDA2020'
        })
    
    def test_realistic_workload_performance(self):
        """Test performance with realistic production workload."""
        # Create realistic mixed workload
        total_records = 50000
        
        geographic_data = []
        coordinate_data = []
        
        # 70% postcodes, 20% SA2s, 10% SA3s
        for i in range(total_records):
            record_type = "postcode" if i < total_records * 0.7 else ("sa2" if i < total_records * 0.9 else "sa3")
            
            if record_type == "postcode":
                code = f"{2000 + (i % 8000):04d}"
            elif record_type == "sa2":
                code = f"{101021007 + (i % 2000):09d}"
            else:  # sa3
                code = f"{10102 + (i % 300):05d}"
            
            geographic_data.append({
                'location_code': code,
                'location_type': record_type,
                'population': 1000 + (i % 10000),
                'id': i
            })
            
            # Add coordinate data for subset of records
            if i % 3 == 0:  # Every 3rd record has coordinates
                coordinate_data.append({
                    'longitude': 115.0 + (30.0 * (i % 1000) / 1000),
                    'latitude': -40.0 + (25.0 * (i % 1000) / 1000),
                    'coordinate_system': 'WGS84' if i % 2 == 0 else 'GDA94',
                    'id': i
                })
        
        # Test geographic standardisation
        self.benchmark.start_measurement("realistic_geographic_standardisation")
        geographic_result = self.geographic_standardiser.transform(geographic_data)
        geo_measurement = self.benchmark.end_measurement("realistic_geographic_standardisation", len(geographic_data))
        
        # Test coordinate transformation
        self.benchmark.start_measurement("realistic_coordinate_transformation")
        coordinate_result = self.coordinate_transformer.transform(coordinate_data)
        coord_measurement = self.benchmark.end_measurement("realistic_coordinate_transformation", len(coordinate_data))
        
        # Overall performance assertions
        total_time = geo_measurement['duration_seconds'] + coord_measurement['duration_seconds']
        total_records_processed = len(geographic_data) + len(coordinate_data)
        
        self.assertLess(total_time, 300.0, "Should process realistic workload in under 5 minutes")
        self.assertGreater(total_records_processed / total_time, 200, "Should maintain at least 200 records/second overall")
        
        # Memory efficiency
        total_memory = geo_measurement['memory_delta_mb'] + coord_measurement['memory_delta_mb']
        self.assertLess(total_memory, 1000, "Should use less than 1GB total memory")
        
        print(f"\nRealistic Workload Performance Summary:")
        print(f"Geographic Standardisation: {len(geographic_data):,} records in {geo_measurement['duration_seconds']:.2f}s")
        print(f"Coordinate Transformation: {len(coordinate_data):,} records in {coord_measurement['duration_seconds']:.2f}s")
        print(f"Total Throughput: {total_records_processed / total_time:.0f} records/second")
        print(f"Total Memory Usage: {total_memory:.1f} MB")
    
    def test_memory_usage_scaling(self):
        """Test memory usage scaling with increasing dataset sizes."""
        dataset_sizes = [1000, 5000, 10000, 25000]
        memory_usage = []
        
        for size in dataset_sizes:
            # Create test data
            test_data = []
            for i in range(size):
                test_data.append({
                    'location_code': f"{2000 + (i % 1000):04d}",
                    'location_type': 'postcode',
                    'id': i
                })
            
            # Measure memory usage
            self.benchmark.start_measurement(f"memory_test_{size}")
            result = self.geographic_standardiser.transform(test_data)
            measurement = self.benchmark.end_measurement(f"memory_test_{size}", size)
            
            memory_usage.append((size, measurement['memory_delta_mb']))
            
            # Clean up to prevent memory accumulation
            del test_data
            del result
        
        # Verify reasonable memory scaling
        for i in range(1, len(memory_usage)):
            prev_size, prev_memory = memory_usage[i-1]
            curr_size, curr_memory = memory_usage[i]
            
            # Memory should scale roughly linearly (allow for some overhead)
            memory_ratio = curr_memory / prev_memory if prev_memory > 0 else 1
            size_ratio = curr_size / prev_size
            
            self.assertLess(memory_ratio, size_ratio * 2, "Memory usage should scale reasonably with dataset size")
        
        print(f"\nMemory Scaling Results:")
        for size, memory in memory_usage:
            print(f"{size:,} records: {memory:.1f} MB ({memory*1024/size:.2f} KB/record)")


if __name__ == '__main__':
    # Configure test runner for performance tests
    import argparse
    
    parser = argparse.ArgumentParser(description='Run geographic pipeline performance tests')
    parser.add_argument('--quick', action='store_true', help='Run quick performance tests only')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Run performance tests
    if '--quick' in sys.argv:
        # Run only small dataset tests for quick feedback
        suite = unittest.TestSuite()
        suite.addTest(TestGeographicStandardisationPerformance('test_postcode_mapping_performance_small_dataset'))
        suite.addTest(TestCoordinateTransformationPerformance('test_coordinate_transformation_performance_small'))
        suite.addTest(TestBoundaryProcessingPerformance('test_boundary_validation_performance'))
        
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    else:
        # Run all performance tests
        unittest.main(verbosity=2)