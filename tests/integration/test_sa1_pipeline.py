"""
Integration tests for SA1-focused AHGD ETL pipeline.

Tests end-to-end SA1 processing functionality including extraction,
SA1 transformation, validation, and loading with realistic SA1 data flows.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest

from src.pipelines.core_etl_pipeline import CoreETLPipeline
from src.pipelines.core_etl_pipeline import PipelineStage
from src.pipelines.core_etl_pipeline import PipelineStatus
from src.transformers.sa1_processor import SA1GeographicTransformer
from src.utils.interfaces import ExtractionError
from src.validators.core_validator import CoreValidator
from tests.fixtures.sa1_data.sa1_test_fixtures import SA1TestDataGenerator


class TestSA1Pipeline:
    """Integration tests for SA1-focused ETL pipeline."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            temp_path = tmp.name
        # Delete the file so DuckDB can create a fresh database
        Path(temp_path).unlink(missing_ok=True)
        yield temp_path
        # Clean up after test
        Path(temp_path).unlink(missing_ok=True)

    @pytest.fixture
    def test_pipeline(self, temp_db_path):
        """Create test pipeline with temporary database."""
        config = {
            "batch_size": 100,
            "max_memory_gb": 1,
            "validation": {"quality_threshold": 80.0},
        }
        return CoreETLPipeline(name="test_sa1_pipeline", db_path=temp_db_path, config=config)

    @pytest.fixture
    def sample_sa1_data(self):
        """Generate sample SA1 data for testing."""
        generator = SA1TestDataGenerator(seed=42)
        return generator.generate_polars_dataframe(count=50)

    def test_pipeline_initialisation(self, test_pipeline):
        """Test that SA1 pipeline initialises correctly."""
        assert test_pipeline.name == "test_sa1_pipeline"
        assert isinstance(test_pipeline.sa1_transformer, SA1GeographicTransformer)
        assert isinstance(test_pipeline.validator, CoreValidator)
        assert test_pipeline.batch_size == 100

        # Test pipeline stages
        expected_stages = ["extract", "transform", "validate", "load"]
        actual_stages = test_pipeline.define_stages()
        assert actual_stages == expected_stages

    def test_sa1_extraction_stage(self, test_pipeline, sample_sa1_data):
        """Test SA1 data extraction stage."""
        # Mock extractor to return our sample data
        mock_extractor = Mock()
        mock_extractor.extract.return_value = [sample_sa1_data.to_dicts()]

        test_pipeline.extractor_registry.get_extractor = Mock(return_value=mock_extractor)

        # Create context
        context = test_pipeline._create_context()
        context.metadata["source_config"] = {"type": "test"}

        # Execute extraction
        test_pipeline._execute_extraction_stage(context)

        # Verify results
        extraction_result = test_pipeline.stage_results.get(PipelineStage.EXTRACT)
        assert extraction_result is not None
        assert extraction_result.status == PipelineStatus.COMPLETED
        assert extraction_result.records_processed == 50
        assert extraction_result.output_table == "extracted_data"

    def test_sa1_transformation_stage(self, test_pipeline, sample_sa1_data):
        """Test SA1 geographic transformation stage."""
        # Set up data in pipeline
        test_pipeline.con.register("extracted_data", sample_sa1_data)
        test_pipeline.current_table = "extracted_data"

        # Create context
        context = test_pipeline._create_context()

        # Execute transformation
        test_pipeline._execute_transformation_stage(context)

        # Verify results
        transform_result = test_pipeline.stage_results.get(PipelineStage.TRANSFORM)
        assert transform_result is not None
        assert transform_result.status == PipelineStatus.COMPLETED
        assert transform_result.records_processed == 50
        assert transform_result.output_table == "transformed_data"

        # Verify transformed data has SA1 structure
        transformed_data = test_pipeline.con.table("transformed_data").pl()
        assert "sa1_code" in transformed_data.columns
        assert "processing_method" in transformed_data.columns
        assert "processing_status" in transformed_data.columns

    def test_sa1_validation_stage(self, test_pipeline, sample_sa1_data):
        """Test SA1 data validation stage."""
        # Set up data in pipeline
        test_pipeline.con.register("transformed_data", sample_sa1_data)
        test_pipeline.current_table = "transformed_data"

        # Create context
        context = test_pipeline._create_context()

        # Execute validation
        test_pipeline._execute_validation_stage(context)

        # Verify results
        validation_result = test_pipeline.stage_results.get(PipelineStage.VALIDATE)
        assert validation_result is not None
        assert validation_result.status == PipelineStatus.COMPLETED
        assert validation_result.records_processed == 50

        # Check validation metadata
        assert "overall_valid" in validation_result.metadata
        assert "quality_score" in validation_result.metadata
        assert validation_result.metadata["quality_score"] >= 80.0

    def test_sa1_loading_stage(self, test_pipeline, sample_sa1_data):
        """Test SA1 data loading stage."""
        # Set up data in pipeline
        test_pipeline.con.register("transformed_data", sample_sa1_data)
        test_pipeline.current_table = "transformed_data"

        # Create context with target config
        context = test_pipeline._create_context()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_sa1_output.parquet"
            context.metadata["target_config"] = {
                "output_path": str(output_path),
                "format": "parquet",
            }

            # Execute loading
            test_pipeline._execute_loading_stage(context)

            # Verify results
            loading_result = test_pipeline.stage_results.get(PipelineStage.LOAD)
            assert loading_result is not None
            assert loading_result.status == PipelineStatus.COMPLETED
            assert loading_result.records_processed == 50
            assert loading_result.output_table == "final_sa1_data"

            # Verify output file exists
            assert output_path.exists()

            # Verify output data structure
            output_data = pl.read_parquet(output_path)
            assert len(output_data) == 50
            assert "sa1_code" in output_data.columns

    def test_complete_sa1_etl_execution(self, test_pipeline, sample_sa1_data):
        """Test complete SA1 ETL pipeline execution."""
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = [sample_sa1_data.to_dicts()]
        test_pipeline.extractor_registry.get_extractor = Mock(return_value=mock_extractor)

        # Configure pipeline
        source_config = {"type": "test"}
        with tempfile.TemporaryDirectory() as temp_dir:
            target_config = {
                "output_path": str(Path(temp_dir) / "complete_sa1_test.parquet"),
                "format": "parquet",
            }

            # Execute complete pipeline
            results = test_pipeline.run_complete_etl(source_config, target_config)

            # Verify overall results
            assert results["status"] == "completed"
            assert results["total_records"] == 50
            assert results["final_table"] == "final_sa1_data"

            # Verify all stages completed
            stage_results = results["stage_results"]
            for stage in ["extract", "transform", "validate", "load"]:
                assert stage in stage_results
                assert stage_results[stage]["status"] == "completed"
                assert stage_results[stage]["records_processed"] == 50

            # Verify execution summary
            summary = results["execution_summary"]
            assert summary["total_stages"] == 4
            assert summary["completed_stages"] == 4
            assert summary["failed_stages"] == 0
            assert summary["success_rate"] == 100.0

    def test_sa1_code_validation_in_pipeline(self, test_pipeline):
        """Test that pipeline properly validates SA1 codes."""
        # Create test data with invalid SA1 codes
        invalid_data = pl.DataFrame(
            {
                "sa1_code": [
                    "12345",
                    "1234567890123",
                    "invalid",
                    "12345678901",
                ],  # Mix of invalid and valid
                "sa1_name": ["Test SA1 1", "Test SA1 2", "Test SA1 3", "Test SA1 4"],
                "population": [400, 500, 300, 450],
                "dwellings": [180, 225, 135, 200],
            }
        )

        # Set up pipeline
        test_pipeline.con.register("extracted_data", invalid_data)
        test_pipeline.current_table = "extracted_data"

        # Execute transformation and validation
        context = test_pipeline._create_context()
        test_pipeline._execute_transformation_stage(context)
        test_pipeline._execute_validation_stage(context)

        # Check validation caught the invalid codes
        validation_result = test_pipeline.stage_results.get(PipelineStage.VALIDATE)
        assert validation_result is not None

        # Should have warnings about invalid SA1 codes
        validation_metadata = validation_result.metadata
        assert validation_metadata["error_count"] > 0
        assert not validation_metadata["overall_valid"]  # Should fail due to invalid codes

    def test_sa1_hierarchy_validation(self, test_pipeline):
        """Test SA1 geographic hierarchy validation."""
        # Create test data with hierarchy issues
        hierarchy_data = pl.DataFrame(
            {
                "sa1_code": ["10102100701", "20203200801"],
                "sa1_name": ["Sydney SA1", "Melbourne SA1"],
                "sa2_code": [
                    "101021007",
                    "999999999",
                ],  # Second one is inconsistent with SA1
                "sa3_code": ["10102", "20203"],
                "sa4_code": ["101", "202"],
                "state_code": ["NSW", "VIC"],
                "population": [400, 500],
                "dwellings": [180, 225],
            }
        )

        # Set up pipeline
        test_pipeline.con.register("extracted_data", hierarchy_data)
        test_pipeline.current_table = "extracted_data"

        # Execute transformation and validation
        context = test_pipeline._create_context()
        test_pipeline._execute_transformation_stage(context)
        test_pipeline._execute_validation_stage(context)

        # Check validation caught hierarchy issues
        validation_result = test_pipeline.stage_results.get(PipelineStage.VALIDATE)
        validation_details = validation_result.metadata.get("validation_details", {})
        hierarchy_results = validation_details.get("hierarchy", {})

        # Should detect inconsistent hierarchy
        assert hierarchy_results.get("inconsistent_hierarchies", 0) > 0

    def test_pipeline_error_handling(self, test_pipeline):
        """Test pipeline error handling and recovery."""
        # Test extraction error
        mock_extractor = Mock()
        mock_extractor.extract.side_effect = ExtractionError("Test extraction error")
        test_pipeline.extractor_registry.get_extractor = Mock(return_value=mock_extractor)

        context = test_pipeline._create_context()
        context.metadata["source_config"] = {"type": "test"}

        # Should handle extraction error gracefully
        with pytest.raises(ExtractionError):
            test_pipeline._execute_extraction_stage(context)

        # Check error was recorded
        extraction_result = test_pipeline.stage_results.get(PipelineStage.EXTRACT)
        assert extraction_result is not None
        assert extraction_result.status == PipelineStatus.FAILED
        assert extraction_result.error is not None

    def test_pipeline_performance_with_large_sa1_dataset(self, test_pipeline):
        """Test pipeline performance with larger SA1 dataset."""
        # Generate larger dataset
        generator = SA1TestDataGenerator(seed=42)
        large_dataset = generator.generate_polars_dataframe(count=1000)

        # Mock extractor
        mock_extractor = Mock()
        mock_extractor.extract.return_value = [large_dataset.to_dicts()]
        test_pipeline.extractor_registry.get_extractor = Mock(return_value=mock_extractor)

        # Execute pipeline with timing
        start_time = datetime.now()

        source_config = {"type": "test"}
        with tempfile.TemporaryDirectory() as temp_dir:
            target_config = {
                "output_path": str(Path(temp_dir) / "large_sa1_test.parquet"),
                "format": "parquet",
            }

            results = test_pipeline.run_complete_etl(source_config, target_config)

        execution_time = datetime.now() - start_time

        # Verify results
        assert results["status"] == "completed"
        assert results["total_records"] == 1000
        assert execution_time.total_seconds() < 60  # Should complete within 1 minute

        # Verify performance is logged
        assert "total_duration" in results
        assert results["total_duration"] > 0


class TestSA1GeographicProcessing:
    """Test SA1-specific geographic processing in integration scenarios."""

    @pytest.fixture
    def sa1_transformer(self):
        """Create SA1 geographic transformer."""
        return SA1GeographicTransformer(config={})

    @pytest.fixture
    def mixed_geographic_data(self):
        """Create test data with mixed geographic codes."""
        return pl.DataFrame(
            {
                "postcode": ["2000", "3000", "4000"],
                "sa2_code": ["101021007", "202032008", "305045009"],
                "address": [
                    "1 Test St Sydney",
                    "2 Test St Melbourne",
                    "3 Test St Brisbane",
                ],
                "health_indicator": ["diabetes_rate", "obesity_rate", "smoking_rate"],
                "value": [8.5, 7.2, 9.1],
            }
        )

    def test_sa1_transformation_from_mixed_inputs(self, sa1_transformer, mixed_geographic_data):
        """Test SA1 transformation from mixed geographic inputs."""
        # Transform data to SA1 framework
        result = sa1_transformer.transform(mixed_geographic_data)

        # Verify SA1 columns are added
        assert "sa1_code" in result.columns
        assert "processing_method" in result.columns
        assert "processing_status" in result.columns

        # Verify original data is preserved
        assert "health_indicator" in result.columns
        assert "value" in result.columns
        assert len(result) == 3

    def test_sa1_aggregation_to_higher_levels(self, sa1_transformer):
        """Test aggregation from SA1 to SA2/SA3/SA4 levels."""
        # Create SA1 data
        sa1_data = pl.DataFrame(
            {
                "sa1_code": ["10102100701", "10102100702", "20203200801"],
                "population": [400, 350, 465],
                "health_score": [85.2, 82.1, 88.5],
            }
        )

        # Test aggregation to SA2
        sa2_aggregated = sa1_transformer.sa1_engine.aggregate_sa1_to_sa2(
            sa1_data, ["population", "health_score"]
        )

        # Verify aggregation
        assert "sa2_code" in sa2_aggregated.columns
        assert "sa1_count" in sa2_aggregated.columns
        assert len(sa2_aggregated) == 2  # Two different SA2s

        # Check aggregated values
        sa2_101021007 = sa2_aggregated.filter(pl.col("sa2_code") == "101021007")
        assert len(sa2_101021007) == 1
        assert sa2_101021007.get_column("population").sum() == 750  # 400 + 350


@pytest.mark.integration
class TestSA1ValidationIntegration:
    """Integration tests for SA1 validation in pipeline context."""

    @pytest.fixture
    def core_validator(self):
        """Create core validator for testing."""
        return CoreValidator({"quality_threshold": 85.0})

    def test_comprehensive_sa1_validation(self, core_validator):
        """Test comprehensive SA1 validation with realistic data."""
        generator = SA1TestDataGenerator(seed=42)
        test_data = generator.generate_polars_dataframe(count=20)

        # Run full validation
        results = core_validator.validate_sa1_data(test_data)

        # Verify validation results
        assert results["overall_valid"] is True
        assert results["quality_score"] >= 85.0
        assert results["total_records"] == 20
        assert results["error_count"] == 0

        # Verify validation details
        details = results["validation_details"]
        assert "sa1_codes" in details
        assert "hierarchy" in details
        assert "data_quality" in details
        assert "statistics" in details

        # Check SA1-specific validation
        sa1_details = details["sa1_codes"]
        assert sa1_details["valid_codes"] == 20
        assert sa1_details["invalid_codes"] == 0

    def test_british_english_error_messages(self, core_validator):
        """Test that validation uses British English in error messages."""
        # Create data with validation issues
        problem_data = pl.DataFrame(
            {
                "sa1_code": ["12345"],  # Invalid format
                "sa1_name": ["Test SA1"],
                "population": [999999],  # Too high
            }
        )

        results = core_validator.validate_sa1_data(problem_data)

        # Check that error messages use British English
        assert not results["overall_valid"]
        warnings = results.get("warnings", [])

        # Should contain British English terms
        warning_text = " ".join(warnings)
        # Look for British spellings in validation messages
        british_terms_found = any(
            term in warning_text.lower()
            for term in ["colour", "centre", "optimise", "standardise", "analyse"]
        )

        # At minimum, should not contain American spellings
        american_terms = ["optimize", "standardize", "analyze", "color", "center"]
        assert not any(term in warning_text.lower() for term in american_terms)
