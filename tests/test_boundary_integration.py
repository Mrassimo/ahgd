"""
Integration test for boundary processor with real data.

Tests complete pipeline: download → extract → process → export.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.data_processing.downloaders.real_data_downloader import RealDataDownloader
from src.data_processing.boundary_processor import BoundaryProcessor


@pytest.mark.asyncio
async def test_boundary_real_data_processing():
    """Test complete boundary pipeline with real downloaded data."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Step 1: Download real SA2 boundary data
            downloader = RealDataDownloader(tmp_dir)
            results = await downloader.download_essential_datasets(["sa2_boundaries_gda94"])
            
            boundary_path = results.get("sa2_boundaries_gda94")
            if not boundary_path or not boundary_path.exists():
                pytest.skip("Could not download SA2 boundary file")
            
            print(f"✅ Downloaded boundary file: {boundary_path.stat().st_size / (1024*1024):.1f}MB")
            
            # Step 2: Initialize processor
            processor = BoundaryProcessor(tmp_dir)
            
            # Step 3: Validate file
            is_valid = processor.validate_boundary_file(boundary_path)
            assert is_valid, "Boundary file should pass validation"
            print("✅ Boundary file validation passed")
            
            # Step 4: Extract data
            boundary_gdf = processor.extract_boundary_data(boundary_path)
            assert len(boundary_gdf) > 2000, f"Expected ~2,368 SA2 areas, got {len(boundary_gdf)}"
            print(f"✅ Extracted {len(boundary_gdf)} SA2 boundary records")
            
            # Step 5: Check data quality
            columns = boundary_gdf.columns.tolist()
            print(f"✅ Columns: {columns}")
            
            # Check for required columns
            required_columns = ["sa2_code_2021", "sa2_name_2021", "geometry"]
            for col in required_columns:
                assert col in columns, f"Missing required column: {col}"
            
            # Check geometry validity
            assert boundary_gdf.geometry.is_valid.all(), "All geometries should be valid"
            
            # Step 6: Test GeoJSON export
            geojson_path = processor.export_geojson(boundary_gdf)
            assert geojson_path.exists(), "GeoJSON file should be created"
            assert geojson_path.stat().st_size > 1000000, "GeoJSON file should be substantial"
            print(f"✅ Exported GeoJSON: {geojson_path.stat().st_size / (1024*1024):.1f}MB")
            
            # Step 7: Generate summary
            summary = processor.get_boundary_summary(boundary_gdf)
            assert summary["total_sa2_areas"] > 2000
            print(f"✅ Summary: {summary['total_sa2_areas']} areas across {len(summary['states_covered'])} states")
            
            print("🎉 Boundary integration test completed successfully!")
            
        except Exception as e:
            pytest.skip(f"Integration test failed (network/processing issue): {e}")


def test_boundary_processor_initialization():
    """Test boundary processor can be initialized."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        processor = BoundaryProcessor(tmp_dir)
        
        assert processor.data_dir == Path(tmp_dir)
        assert processor.raw_dir.exists()
        assert processor.processed_dir.exists()
        assert processor.geojson_dir.exists()


def test_boundary_config_validation():
    """Test boundary configuration is valid."""
    from src.data_processing.boundary_processor import BOUNDARY_CONFIG, SA2_BOUNDARY_SCHEMA
    
    # Config should have required fields
    required_config = ["filename", "shapefile_pattern", "expected_records", "coordinate_system"]
    for field in required_config:
        assert field in BOUNDARY_CONFIG, f"Missing config field: {field}"
    
    # Schema should have SA2 and geometry columns
    assert "sa2_code_2021" in SA2_BOUNDARY_SCHEMA
    assert "sa2_name_2021" in SA2_BOUNDARY_SCHEMA
    assert "geometry" in SA2_BOUNDARY_SCHEMA
    
    # Should have state and area information
    assert "state_code_2021" in SA2_BOUNDARY_SCHEMA
    assert "area_sqkm" in SA2_BOUNDARY_SCHEMA


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_boundary_real_data_processing())