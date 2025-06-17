"""
Test suite for SEIFA processor implementation.

Tests the actual SEIFAProcessor class with real data processing.
"""

import tempfile
from pathlib import Path
from typing import Optional

import polars as pl
import pytest

from src.data_processing.downloaders.real_data_downloader import RealDataDownloader
from src.data_processing.seifa_processor import SEIFAProcessor, SEIFA_CONFIG, SEIFA_SCHEMA


class TestSEIFAProcessorImplementation:
    """Test SEIFA processor implementation with real data."""
    
    @pytest.fixture
    @pytest.mark.asyncio
    async def test_seifa_setup(self):
        """Setup test environment with downloaded SEIFA data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download real SEIFA data
            downloader = RealDataDownloader(tmp_dir)
            
            try:
                results = await downloader.download_essential_datasets(["seifa_2021_sa2"])
                seifa_path = results.get("seifa_2021_sa2")
                
                if seifa_path and seifa_path.exists():
                    # Initialize processor
                    processor = SEIFAProcessor(tmp_dir)
                    yield processor, seifa_path
                else:
                    pytest.skip("Could not download SEIFA file for testing")
                    
            except Exception as e:
                pytest.skip(f"Network error: {e}")
    
    @pytest.mark.asyncio
    async def test_seifa_processor_initialization(self):
        """Test SEIFA processor initializes correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = SEIFAProcessor(tmp_dir)
            
            assert processor.data_dir == Path(tmp_dir)
            assert processor.raw_dir.exists()
            assert processor.processed_dir.exists()
    
    @pytest.mark.asyncio 
    async def test_seifa_file_validation(self, test_seifa_setup):
        """Test SEIFA file validation with real file."""
        processor, seifa_path = test_seifa_setup
        
        # Should pass validation for real SEIFA file
        is_valid = processor.validate_seifa_file(seifa_path)
        assert is_valid, "Real SEIFA file should pass validation"
    
    @pytest.mark.asyncio
    async def test_seifa_data_extraction(self, test_seifa_setup):
        """Test SEIFA data extraction from real Excel file."""
        processor, seifa_path = test_seifa_setup
        
        # Extract data
        seifa_df = processor.extract_seifa_data(seifa_path)
        
        # Validate extracted data
        assert isinstance(seifa_df, pl.DataFrame)
        assert len(seifa_df) > 2000, f"Expected ~2,368 SA2 areas, got {len(seifa_df)}"
        assert len(seifa_df) < 2500, f"Too many SA2 areas: {len(seifa_df)}"
        
        # Check key columns exist
        columns = seifa_df.columns
        assert any("sa2" in col.lower() for col in columns), "Missing SA2 code column"
        
        # Check for SEIFA index data
        has_seifa_data = any(
            index_name in " ".join(columns).lower() 
            for index_name in ["irsd", "irsad", "ier", "ieo", "score", "decile"]
        )
        assert has_seifa_data, "Missing SEIFA index data"
    
    @pytest.mark.asyncio
    async def test_seifa_column_standardization(self, test_seifa_setup):
        """Test SEIFA column name standardization."""
        processor, seifa_path = test_seifa_setup
        
        # Extract and standardize
        seifa_df = processor.extract_seifa_data(seifa_path)
        
        # Check standardized columns exist
        standardized_columns = [col for col in SEIFA_SCHEMA.keys() if col in seifa_df.columns]
        assert len(standardized_columns) > 5, f"Too few standardized columns: {standardized_columns}"
        
        # Should have SA2 identification
        if "sa2_code_2021" in seifa_df.columns:
            sa2_codes = seifa_df["sa2_code_2021"].to_list()[:5]
            for code in sa2_codes:
                assert len(str(code)) == 9, f"Invalid SA2 code length: {code}"
    
    @pytest.mark.asyncio
    async def test_seifa_data_validation(self, test_seifa_setup):
        """Test SEIFA data quality validation."""
        processor, seifa_path = test_seifa_setup
        
        # Process with validation
        seifa_df = processor.extract_seifa_data(seifa_path)
        
        # Check SEIFA score ranges if present
        score_columns = [col for col in seifa_df.columns if "score" in col]
        for score_col in score_columns:
            scores = seifa_df[score_col].drop_nulls().to_list()
            if scores:
                min_score = min(scores)
                max_score = max(scores)
                
                # SEIFA scores should be in reasonable range
                assert min_score >= 700, f"SEIFA score too low: {min_score}"
                assert max_score <= 1300, f"SEIFA score too high: {max_score}"
        
        # Check decile ranges if present  
        decile_columns = [col for col in seifa_df.columns if "decile" in col]
        for decile_col in decile_columns:
            deciles = seifa_df[decile_col].drop_nulls().to_list()
            if deciles:
                min_decile = min(deciles)
                max_decile = max(deciles)
                
                assert min_decile >= 1, f"Decile too low: {min_decile}"
                assert max_decile <= 10, f"Decile too high: {max_decile}"
    
    @pytest.mark.asyncio
    async def test_seifa_complete_pipeline(self, test_seifa_setup):
        """Test complete SEIFA processing pipeline."""
        processor, seifa_path = test_seifa_setup
        
        # Run complete pipeline
        result_df = processor.process_complete_pipeline()
        
        # Validate results
        assert isinstance(result_df, pl.DataFrame)
        assert len(result_df) > 1000, "Pipeline should produce substantial data"
        
        # Check output files were created
        csv_output = processor.processed_dir / "seifa_2021_sa2.csv"
        parquet_output = processor.processed_dir / "seifa_2021_sa2.parquet"
        
        assert csv_output.exists(), "CSV output file should be created"
        assert parquet_output.exists(), "Parquet output file should be created"
        
        # Validate output file sizes
        assert csv_output.stat().st_size > 100000, "CSV output too small"
        assert parquet_output.stat().st_size > 50000, "Parquet output too small"
    
    @pytest.mark.asyncio
    async def test_seifa_summary_generation(self, test_seifa_setup):
        """Test SEIFA summary statistics generation."""
        processor, seifa_path = test_seifa_setup
        
        # Process data
        seifa_df = processor.extract_seifa_data(seifa_path)
        
        # Generate summary
        summary = processor.get_seifa_summary(seifa_df)
        
        # Validate summary structure
        assert isinstance(summary, dict)
        assert "total_sa2_areas" in summary
        assert "states_covered" in summary
        assert "seifa_statistics" in summary
        
        # Check data completeness
        assert summary["total_sa2_areas"] > 2000
        assert len(summary["states_covered"]) >= 6  # Should cover most Australian states
        
        # Check SEIFA statistics
        if summary["seifa_statistics"]:
            for index_name, stats in summary["seifa_statistics"].items():
                assert "min_score" in stats
                assert "max_score" in stats
                assert "mean_score" in stats


class TestSEIFAProcessorConfiguration:
    """Test SEIFA processor configuration and constants."""
    
    def test_seifa_config_structure(self):
        """Test SEIFA configuration has required fields."""
        required_fields = [
            "filename", "primary_sheet", "header_row", 
            "data_start_row", "expected_records", "expected_sheets"
        ]
        
        for field in required_fields:
            assert field in SEIFA_CONFIG, f"Missing config field: {field}"
        
        # Validate specific values
        assert SEIFA_CONFIG["primary_sheet"] == "Table 1"
        assert SEIFA_CONFIG["expected_records"] == 2368
        assert isinstance(SEIFA_CONFIG["expected_sheets"], list)
    
    def test_seifa_schema_completeness(self):
        """Test SEIFA schema covers all required columns."""
        required_columns = [
            "sa2_code_2021", "sa2_name_2021",
            "irsd_score", "irsd_decile",
            "irsad_score", "irsad_decile", 
            "ier_score", "ier_decile",
            "ieo_score", "ieo_decile",
            "usual_resident_population"
        ]
        
        for column in required_columns:
            assert column in SEIFA_SCHEMA, f"Missing schema column: {column}"
        
        # Validate data types
        assert SEIFA_SCHEMA["sa2_code_2021"] == pl.Utf8
        assert SEIFA_SCHEMA["irsd_score"] == pl.Int32
        assert SEIFA_SCHEMA["irsd_decile"] == pl.Int8


@pytest.mark.asyncio 
class TestSEIFAProcessorErrors:
    """Test SEIFA processor error handling."""
    
    async def test_missing_file_handling(self):
        """Test handling of missing SEIFA file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = SEIFAProcessor(tmp_dir)
            
            # Should raise FileNotFoundError for missing file
            with pytest.raises(FileNotFoundError):
                processor.process_seifa_file("nonexistent_file.xlsx")
    
    async def test_invalid_file_handling(self):
        """Test handling of invalid file format."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            processor = SEIFAProcessor(tmp_dir)
            
            # Create dummy file
            dummy_file = processor.raw_dir / "dummy.xlsx"
            dummy_file.parent.mkdir(parents=True, exist_ok=True)
            dummy_file.write_text("Not an Excel file")
            
            # Should fail validation
            is_valid = processor.validate_seifa_file(dummy_file)
            assert not is_valid, "Should reject invalid Excel file"


if __name__ == "__main__":
    # Run SEIFA implementation tests
    pytest.main([__file__, "-v"])