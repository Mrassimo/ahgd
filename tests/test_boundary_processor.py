"""
Test suite for boundary processor implementation.

Tests SA2 shapefile processing, geometry validation, and GeoJSON export.
"""

import tempfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from src.data_processing.boundary_processor import BoundaryProcessor, BOUNDARY_CONFIG, SA2_BOUNDARY_SCHEMA


class TestBoundaryProcessorImplementation:
    """Test boundary processor implementation with mock and real data."""
    
    @pytest.fixture
    def mock_boundary_gdf(self):
        """Create mock boundary GeoDataFrame for testing."""
        # Create simple polygon geometries
        polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])
        ]
        
        # Create mock data
        data = {
            'sa2_code_2021': ['101021007', '101021008', '101021009'],
            'sa2_name_2021': ['Braidwood', 'Karabar', 'Queanbeyan'],
            'state_code_2021': ['1', '1', '1'],
            'state_name_2021': ['New South Wales', 'New South Wales', 'New South Wales'],
            'area_sqkm': [125.5, 89.2, 67.8],
            'geometry': polygons
        }
        
        return gpd.GeoDataFrame(data, crs="EPSG:4283")
    
    def test_boundary_processor_initialization(self):
        """Test boundary processor initializes correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = BoundaryProcessor(tmp_dir)
            
            assert processor.data_dir == Path(tmp_dir)
            assert processor.raw_dir.exists()
            assert processor.processed_dir.exists()
            assert processor.geojson_dir.exists()
    
    def test_boundary_column_standardization(self, mock_boundary_gdf):
        """Test boundary column name standardization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = BoundaryProcessor(tmp_dir)
            
            # Test standardization (mock data already has standard names)
            standardized_gdf = processor._standardize_boundary_columns(mock_boundary_gdf)
            
            # Check all expected columns are present
            expected_columns = list(SA2_BOUNDARY_SCHEMA.keys())
            for col in expected_columns:
                if col in mock_boundary_gdf.columns:
                    assert col in standardized_gdf.columns, f"Missing standardized column: {col}"
    
    def test_boundary_geometry_validation(self, mock_boundary_gdf):
        """Test boundary geometry validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = BoundaryProcessor(tmp_dir)
            
            # Test with valid geometries
            validated_gdf = processor._validate_boundary_geometries(mock_boundary_gdf)
            
            assert len(validated_gdf) == 3, "All valid records should be retained"
            assert validated_gdf.geometry.is_valid.all(), "All geometries should be valid"
            
            # Check SA2 codes are valid
            for code in validated_gdf['sa2_code_2021']:
                assert len(str(code)) == 9, f"Invalid SA2 code: {code}"
    
    def test_boundary_simplification(self, mock_boundary_gdf):
        """Test boundary geometry simplification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = BoundaryProcessor(tmp_dir)
            
            # Test simplification
            simplified_gdf = processor.simplify_boundaries(mock_boundary_gdf, tolerance=0.001)
            
            assert len(simplified_gdf) == len(mock_boundary_gdf), "Record count should be preserved"
            assert simplified_gdf.geometry.is_valid.all(), "Simplified geometries should be valid"
    
    def test_geojson_export(self, mock_boundary_gdf):
        """Test GeoJSON export functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = BoundaryProcessor(tmp_dir)
            
            # Test export
            output_path = processor.export_geojson(mock_boundary_gdf, "test_boundaries.geojson")
            
            assert output_path.exists(), "GeoJSON file should be created"
            assert output_path.suffix == ".geojson", "File should have .geojson extension"
            assert output_path.stat().st_size > 100, "GeoJSON file should not be empty"
            
            # Test that file can be read back
            reimported_gdf = gpd.read_file(output_path)
            assert len(reimported_gdf) == len(mock_boundary_gdf), "Record count should match"
    
    def test_boundary_summary_generation(self, mock_boundary_gdf):
        """Test boundary summary statistics generation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = BoundaryProcessor(tmp_dir)
            
            # Generate summary
            summary = processor.get_boundary_summary(mock_boundary_gdf)
            
            # Validate summary structure
            assert isinstance(summary, dict)
            assert "total_sa2_areas" in summary
            assert "states_covered" in summary
            assert "total_area_sqkm" in summary
            assert "coordinate_system" in summary
            
            # Check values
            assert summary["total_sa2_areas"] == 3
            assert "New South Wales" in summary["states_covered"]
            assert summary["total_area_sqkm"] > 0
    
    def test_seifa_linkage(self, mock_boundary_gdf):
        """Test linking boundary data with SEIFA data."""
        import polars as pl
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = BoundaryProcessor(tmp_dir)
            
            # Create mock SEIFA data
            seifa_data = {
                'sa2_code_2021': ['101021007', '101021008', '101021009'],
                'irsd_score': [1024, 994, 1010],
                'irsd_decile': [6, 5, 5],
                'irsad_score': [1015, 985, 1001],
                'irsad_decile': [6, 5, 5]
            }
            seifa_df = pl.DataFrame(seifa_data)
            
            # Test linkage
            linked_gdf = processor.link_with_seifa(mock_boundary_gdf, seifa_df)
            
            assert len(linked_gdf) == len(mock_boundary_gdf), "Record count should be preserved"
            assert "irsd_score" in linked_gdf.columns, "SEIFA columns should be added"
            assert linked_gdf["irsd_score"].notna().all(), "All records should have SEIFA data"


class TestBoundaryProcessorConfiguration:
    """Test boundary processor configuration and constants."""
    
    def test_boundary_config_structure(self):
        """Test boundary configuration has required fields."""
        required_fields = [
            "filename", "shapefile_pattern", "expected_records", 
            "coordinate_system", "geometry_type", "simplification_tolerance"
        ]
        
        for field in required_fields:
            assert field in BOUNDARY_CONFIG, f"Missing config field: {field}"
        
        # Validate specific values
        assert BOUNDARY_CONFIG["expected_records"] == 2368
        assert BOUNDARY_CONFIG["coordinate_system"] == "GDA94"
        assert BOUNDARY_CONFIG["geometry_type"] == "MultiPolygon"
    
    def test_boundary_schema_completeness(self):
        """Test boundary schema covers all required columns."""
        required_columns = [
            "sa2_code_2021", "sa2_name_2021", "geometry",
            "state_code_2021", "state_name_2021", "area_sqkm"
        ]
        
        for column in required_columns:
            assert column in SA2_BOUNDARY_SCHEMA, f"Missing schema column: {column}"
        
        # Check that schema maps to shapefile columns
        for standard_col, shapefile_col in SA2_BOUNDARY_SCHEMA.items():
            assert isinstance(shapefile_col, str), f"Schema mapping should be string: {standard_col}"


@pytest.mark.asyncio 
class TestBoundaryProcessorErrors:
    """Test boundary processor error handling."""
    
    async def test_missing_file_handling(self):
        """Test handling of missing boundary file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = BoundaryProcessor(tmp_dir)
            
            # Should raise FileNotFoundError for missing file
            with pytest.raises(FileNotFoundError):
                processor.process_boundary_file("nonexistent_file.zip")
    
    async def test_invalid_file_handling(self):
        """Test handling of invalid file format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = BoundaryProcessor(tmp_dir)
            
            # Create dummy file
            dummy_file = processor.raw_dir / "dummy.zip"
            dummy_file.parent.mkdir(parents=True, exist_ok=True)
            dummy_file.write_text("Not a ZIP file")
            
            # Should fail validation
            is_valid = processor.validate_boundary_file(dummy_file)
            assert not is_valid, "Should reject invalid ZIP file"


if __name__ == "__main__":
    # Run boundary processor tests
    pytest.main([__file__, "-v"])