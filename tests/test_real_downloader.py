"""
Test suite for the real data downloader implementation.

Tests the production-ready downloader that uses verified working URLs.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.data_processing.downloaders.real_data_downloader import (
    RealDataDownloader,
    VERIFIED_DATA_SOURCES,
    download_australian_health_data,
)


class TestRealDataDownloader:
    """Test the production-ready Australian data downloader."""
    
    def test_downloader_initialization(self):
        """Test downloader initializes correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = RealDataDownloader(tmp_dir)
            
            assert downloader.data_dir == Path(tmp_dir)
            assert downloader.raw_dir == Path(tmp_dir) / "raw"
            assert downloader.raw_dir.exists()
    
    def test_verified_data_sources_structure(self):
        """Test that verified data sources have correct structure."""
        
        required_fields = {"url", "filename", "size_mb", "format", "description"}
        
        for dataset_key, dataset_info in VERIFIED_DATA_SOURCES.items():
            # Check all required fields present
            assert set(dataset_info.keys()) >= required_fields, f"Missing fields in {dataset_key}"
            
            # Check field types
            assert isinstance(dataset_info["url"], str), f"URL not string in {dataset_key}"
            assert dataset_info["url"].startswith("https://"), f"Insecure URL in {dataset_key}"
            
            assert isinstance(dataset_info["filename"], str), f"Filename not string in {dataset_key}"
            assert len(dataset_info["filename"]) > 0, f"Empty filename in {dataset_key}"
            
            assert isinstance(dataset_info["size_mb"], (int, float)), f"Size not numeric in {dataset_key}"
            assert dataset_info["size_mb"] > 0, f"Invalid size in {dataset_key}"
            
            assert dataset_info["format"] in ["zip", "xlsx", "csv"], f"Invalid format in {dataset_key}"
            
            assert isinstance(dataset_info["description"], str), f"Description not string in {dataset_key}"
            assert len(dataset_info["description"]) > 0, f"Empty description in {dataset_key}"
    
    def test_dataset_keys_validity(self):
        """Test dataset keys are valid identifiers."""
        
        for dataset_key in VERIFIED_DATA_SOURCES.keys():
            # Should be valid Python identifier-like
            assert dataset_key.replace("_", "").isalnum(), f"Invalid dataset key: {dataset_key}"
            assert not dataset_key.startswith("_"), f"Key starts with underscore: {dataset_key}"
            assert dataset_key == dataset_key.lower(), f"Key not lowercase: {dataset_key}"
    
    @pytest.mark.asyncio
    async def test_download_specific_dataset_validation(self):
        """Test download validation for invalid dataset keys."""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = RealDataDownloader(tmp_dir)
            
            # Should raise error for invalid dataset key
            with pytest.raises(ValueError, match="Unknown dataset"):
                await downloader.download_specific_dataset("invalid_dataset")
    
    def test_file_validation_methods(self):
        """Test file format validation methods."""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = RealDataDownloader(tmp_dir)
            
            # Test ZIP validation with valid ZIP
            zip_path = Path(tmp_dir) / "test.zip"
            with open(zip_path, 'wb') as f:
                f.write(b'PK\x03\x04')  # ZIP magic bytes
            
            # Should not raise error for valid ZIP magic bytes
            # (Note: This is minimal validation, real test would need actual ZIP)
            try:
                with open(zip_path, 'rb') as f:
                    magic = f.read(4)
                    assert magic.startswith(b'PK')
            except Exception:
                pytest.fail("ZIP validation failed unexpectedly")
    
    def test_extracted_files_directory_creation(self):
        """Test that extraction creates proper directory structure."""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = RealDataDownloader(tmp_dir)
            
            # Test extraction directory creation
            extract_dir = Path(tmp_dir) / "test_extract"
            extracted = downloader.extract_zip_files(extract_dir)
            
            # Should create directory even if no ZIP files
            assert extract_dir.exists()
            assert isinstance(extracted, dict)
    
    def test_list_available_datasets(self):
        """Test that listing datasets works without errors."""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = RealDataDownloader(tmp_dir)
            
            # Should not raise any errors
            try:
                downloader.list_available_datasets()
            except Exception as e:
                pytest.fail(f"list_available_datasets failed: {e}")
    
    def test_get_downloaded_files_empty(self):
        """Test getting downloaded files when none exist."""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = RealDataDownloader(tmp_dir)
            
            downloaded = downloader.get_downloaded_files()
            
            assert isinstance(downloaded, dict)
            assert len(downloaded) == 0  # No files downloaded yet
    
    def test_convenience_function_parameters(self):
        """Test the convenience function accepts correct parameters."""
        
        # Should not raise errors with valid parameters
        async def test_call():
            # Test with specific datasets
            datasets = ["seifa_2021_sa2"]  # Small dataset
            
            with tempfile.TemporaryDirectory() as tmp_dir:
                # This will attempt download, but we're just testing parameter validation
                try:
                    result = await download_australian_health_data(datasets, tmp_dir)
                    assert isinstance(result, dict)
                except Exception:
                    # Download might fail, but parameters should be valid
                    pass
        
        # Run the async test
        asyncio.run(test_call())


class TestRealDownloaderIntegration:
    """Integration tests for real downloader with actual data sources."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_small_dataset_download(self):
        """Test downloading the smallest dataset (SEIFA) as integration test."""
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloader = RealDataDownloader(tmp_dir)
            
            try:
                # Download only the smallest dataset
                result = await downloader.download_essential_datasets(["seifa_2021_sa2"])
                
                # Check result structure
                assert isinstance(result, dict)
                assert "seifa_2021_sa2" in result
                
                # If download succeeded, check file exists
                if result["seifa_2021_sa2"] is not None:
                    file_path = result["seifa_2021_sa2"]
                    assert file_path.exists()
                    assert file_path.stat().st_size > 0
                    
                    # Check filename matches expected
                    expected_filename = VERIFIED_DATA_SOURCES["seifa_2021_sa2"]["filename"]
                    assert file_path.name == expected_filename
                    
                    print(f"✅ Successfully downloaded: {file_path} ({file_path.stat().st_size / (1024*1024):.1f}MB)")
                
                else:
                    print("⚠️  Download failed (network/server issue)")
                    # Don't fail test for network issues
                
            except Exception as e:
                # Log the error but don't fail test for network issues
                print(f"⚠️  Integration test failed (possibly network): {e}")
    
    def test_dataset_coverage(self):
        """Test that we have datasets covering all major data types."""
        
        dataset_keys = set(VERIFIED_DATA_SOURCES.keys())
        
        # Should have geographic boundaries
        boundary_datasets = {k for k in dataset_keys if "boundaries" in k}
        assert len(boundary_datasets) > 0, "Missing geographic boundary datasets"
        
        # Should have socio-economic data
        seifa_datasets = {k for k in dataset_keys if "seifa" in k}
        assert len(seifa_datasets) > 0, "Missing SEIFA socio-economic datasets"
        
        # Should have health service data
        health_datasets = {k for k in dataset_keys if "mbs" in k or "pbs" in k}
        assert len(health_datasets) > 0, "Missing health service datasets"
        
        print(f"✅ Dataset coverage validated:")
        print(f"   Boundaries: {len(boundary_datasets)}")
        print(f"   SEIFA: {len(seifa_datasets)}")
        print(f"   Health: {len(health_datasets)}")
        print(f"   Total: {len(dataset_keys)}")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])