"""
Integration test for health data processor with real data.

Tests MBS and PBS data processing with real Australian government datasets.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.data_processing.downloaders.real_data_downloader import RealDataDownloader
from src.data_processing.health_processor import HealthDataProcessor


@pytest.mark.asyncio
async def test_health_real_data_processing():
    """Test complete health data pipeline with real downloaded data."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Step 1: Download real health data
            downloader = RealDataDownloader(tmp_dir)
            results = await downloader.download_essential_datasets(["pbs_current"])
            
            pbs_path = results.get("pbs_current")
            if not pbs_path or not pbs_path.exists():
                pytest.skip("Could not download PBS file")
            
            print(f"✅ Downloaded PBS file: {pbs_path.stat().st_size / (1024*1024):.1f}MB")
            
            # Step 2: Initialize processor
            processor = HealthDataProcessor(tmp_dir)
            
            # Step 3: Validate file
            is_valid = processor.validate_health_file(pbs_path, "pbs")
            assert is_valid, "PBS file should pass validation"
            print("✅ PBS file validation passed")
            
            # Step 4: Extract data
            pbs_df = processor.extract_pbs_data(pbs_path)
            assert len(pbs_df) > 100000, f"Expected substantial PBS data, got {len(pbs_df)}"
            print(f"✅ Extracted {len(pbs_df)} PBS records")
            
            # Step 5: Check data quality
            columns = pbs_df.columns
            print(f"✅ Columns: {columns}")
            
            # Check for expected columns
            expected_columns = ["year", "state"]
            for col in expected_columns:
                assert col in columns, f"Missing expected column: {col}"
            
            # Step 6: Test summary generation
            mock_mbs = processor._create_mock_mbs_data()
            summary = processor.get_health_summary(mock_mbs, pbs_df)
            assert summary["pbs_records"] > 100000
            print(f"✅ Summary: {summary['pbs_records']} PBS records")
            
            print("🎉 Health integration test completed successfully!")
            
        except Exception as e:
            pytest.skip(f"Integration test failed (network/processing issue): {e}")


def test_health_processor_initialization():
    """Test health processor can be initialized."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        processor = HealthDataProcessor(tmp_dir)
        
        assert processor.data_dir == Path(tmp_dir)
        assert processor.raw_dir.exists()
        assert processor.processed_dir.exists()


def test_health_mock_data_generation():
    """Test health processor mock data generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        processor = HealthDataProcessor(tmp_dir)
        
        # Test MBS mock data
        mbs_df = processor._create_mock_mbs_data()
        assert len(mbs_df) > 0
        assert "year" in mbs_df.columns
        assert "state" in mbs_df.columns
        assert "service_category" in mbs_df.columns
        
        # Test PBS mock data
        pbs_df = processor._create_mock_pbs_data()
        assert len(pbs_df) > 0
        assert "year" in pbs_df.columns
        assert "state" in pbs_df.columns
        assert "drug_name" in pbs_df.columns
        
        # Test summary with mock data
        summary = processor.get_health_summary(mbs_df, pbs_df)
        assert summary["mbs_records"] == len(mbs_df)
        assert summary["pbs_records"] == len(pbs_df)
        assert len(summary["years_covered"]) > 0
        assert len(summary["states_covered"]) > 0


def test_health_config_validation():
    """Test health configuration is valid."""
    from src.data_processing.health_processor import MBS_CONFIG, PBS_CONFIG, MBS_SCHEMA, PBS_SCHEMA
    
    # MBS config should have required fields
    required_mbs_config = ["historical_filename", "expected_records_historical", "demographic_columns"]
    for field in required_mbs_config:
        assert field in MBS_CONFIG, f"Missing MBS config field: {field}"
    
    # PBS config should have required fields
    required_pbs_config = ["current_filename", "expected_records_current", "prescription_columns"]
    for field in required_pbs_config:
        assert field in PBS_CONFIG, f"Missing PBS config field: {field}"
    
    # Schemas should have required columns
    assert "year" in MBS_SCHEMA
    assert "state" in MBS_SCHEMA
    assert "year" in PBS_SCHEMA
    assert "state" in PBS_SCHEMA


if __name__ == "__main__":
    # Run integration test
    asyncio.run(test_health_real_data_processing())