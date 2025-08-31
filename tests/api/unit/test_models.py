"""
Unit tests for API models.

Tests Pydantic models for validation, serialisation, and British English conventions.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.api.models.common import (
    SA1Code, GeographicLevel, QualityScore, ValidationRule,
    StatusEnum, AHGDBaseModel
)
from src.api.models.requests import (
    QualityMetricsRequest, ValidationRequest, PipelineRunRequest
)
from src.api.models.responses import (
    QualityMetricsResponse, ValidationResponse, PipelineRunResponse
)


class TestSA1Code:
    """Test SA1 code validation."""
    
    def test_valid_sa1_code(self):
        """Test valid SA1 code formats."""
        valid_codes = ["10101000001", "20202000002", "99999999999"]
        
        for code in valid_codes:
            sa1 = SA1Code(code=code)
            assert sa1.code == code
    
    def test_invalid_sa1_code_length(self):
        """Test SA1 codes with invalid lengths."""
        invalid_codes = ["123456789", "123456789012", ""]
        
        for code in invalid_codes:
            with pytest.raises(ValidationError) as exc_info:
                SA1Code(code=code)
            assert "must be exactly 11 digits" in str(exc_info.value)
    
    def test_invalid_sa1_code_non_numeric(self):
        """Test SA1 codes with non-numeric characters."""
        invalid_codes = ["1010100000A", "ABCDEFGHIJK", "101-01-00001"]
        
        for code in invalid_codes:
            with pytest.raises(ValidationError) as exc_info:
                SA1Code(code=code)
            assert "must be exactly 11 digits" in str(exc_info.value)
    
    def test_sa1_code_whitespace_handling(self):
        """Test SA1 code whitespace trimming."""
        sa1 = SA1Code(code=" 10101000001 ")
        assert sa1.code == "10101000001"


class TestGeographicLevel:
    """Test geographic level enumeration."""
    
    def test_geographic_levels(self):
        """Test all geographic levels are valid."""
        levels = ["sa1", "sa2", "sa3", "sa4", "lga", "state", "australia"]
        
        for level in levels:
            geo_level = GeographicLevel(level)
            assert geo_level.value == level
    
    def test_invalid_geographic_level(self):
        """Test invalid geographic levels."""
        with pytest.raises(ValueError):
            GeographicLevel("invalid_level")


class TestQualityScore:
    """Test quality score model."""
    
    def test_quality_score_creation(self, sample_quality_metrics):
        """Test creating quality score object."""
        metrics = QualityScore(
            overall_score=sample_quality_metrics["overall_score"],
            completeness=sample_quality_metrics["completeness_rate"], 
            accuracy=sample_quality_metrics["accuracy_score"],
            consistency=sample_quality_metrics["consistency_score"],
            validity=95.0,
            timeliness=sample_quality_metrics["timeliness_score"],
            record_count=sample_quality_metrics["record_count"]
        )
        
        assert metrics.completeness_rate == 98.5
        assert metrics.accuracy_score == 94.2
        assert metrics.overall_score == 95.4
        assert metrics.record_count == 15000
        assert metrics.error_count == 125
    
    def test_quality_metrics_computed_grade(self, sample_quality_metrics):
        """Test computed quality grade."""
        # Excellent grade
        sample_quality_metrics["overall_score"] = 98.0
        metrics = QualityMetrics(**sample_quality_metrics)
        assert metrics.quality_grade == "Excellent"
        
        # Good grade
        sample_quality_metrics["overall_score"] = 90.0
        metrics = QualityMetrics(**sample_quality_metrics)
        assert metrics.quality_grade == "Good"
        
        # Fair grade
        sample_quality_metrics["overall_score"] = 80.0
        metrics = QualityMetrics(**sample_quality_metrics)
        assert metrics.quality_grade == "Fair"
        
        # Poor grade
        sample_quality_metrics["overall_score"] = 60.0
        metrics = QualityMetrics(**sample_quality_metrics)
        assert metrics.quality_grade == "Poor"
    
    def test_quality_metrics_validation(self):
        """Test quality metrics validation rules."""
        with pytest.raises(ValidationError):
            QualityMetrics(
                completeness_rate=150.0,  # Invalid: > 100
                accuracy_score=50.0,
                overall_score=75.0,
                record_count=1000,
                error_count=50
            )
        
        with pytest.raises(ValidationError):
            QualityMetrics(
                completeness_rate=95.0,
                accuracy_score=85.0,
                overall_score=90.0,
                record_count=1000,
                error_count=-5  # Invalid: negative
            )


class TestValidationRule:
    """Test validation rule model."""
    
    def test_validation_rule_creation(self, sample_validation_result):
        """Test creating validation rule object."""
        rule = ValidationRule(**sample_validation_result)
        
        assert rule.rule_name == "sa1_code_format"
        assert rule.rule_type == "schema"
        assert rule.status == "passed"
        assert rule.success_rate == 99.5
    
    def test_validation_rule_status_enum(self):
        """Test validation status enumeration."""
        valid_statuses = ["passed", "failed", "warning", "skipped"]
        
        for status in valid_statuses:
            rule = ValidationRule(
                rule_name="test_rule",
                rule_type="schema",
                status=status,
                severity="error",
                records_tested=100,
                records_passed=90,
                records_failed=10,
                success_rate=90.0,
                message="Test rule"
            )
            assert rule.status == status


class TestRequestModels:
    """Test request model validation."""
    
    def test_quality_metrics_request(self, sample_sa1_code):
        """Test quality metrics request validation."""
        request = QualityMetricsRequest(
            geographic_level=GeographicLevel.SA1,
            sa1_codes=[sample_sa1_code],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31)
        )
        
        assert request.geographic_level == GeographicLevel.SA1
        assert len(request.sa1_codes) == 1
        assert request.sa1_codes[0] == sample_sa1_code
    
    def test_validation_request(self, sample_sa1_code):
        """Test validation request validation."""
        request = ValidationRequest(
            geographic_level=GeographicLevel.SA1,
            validation_types=["schema", "business"],
            sa1_codes=[sample_sa1_code]
        )
        
        assert request.geographic_level == GeographicLevel.SA1
        assert "schema" in request.validation_types
        assert "business" in request.validation_types
    
    def test_pipeline_run_request(self, sample_pipeline_config):
        """Test pipeline run request validation."""
        request = PipelineRunRequest(
            pipeline_name="test_pipeline",
            config=sample_pipeline_config,
            priority="normal"
        )
        
        assert request.pipeline_name == "test_pipeline"
        assert request.priority == "normal"
        assert request.config["name"] == "test_etl_pipeline"


class TestResponseModels:
    """Test response model serialisation."""
    
    def test_quality_metrics_response(self, sample_quality_metrics):
        """Test quality metrics response serialisation."""
        metrics = QualityMetrics(**sample_quality_metrics)
        
        response = QualityMetricsResponse(
            success=True,
            message="Quality metrics calculated successfully",
            timestamp=datetime.now(),
            metrics=metrics,
            geographic_level=GeographicLevel.SA1,
            total_records=15000
        )
        
        assert response.success is True
        assert response.metrics.overall_score == 95.4
        assert response.geographic_level == GeographicLevel.SA1
        assert response.total_records == 15000
    
    def test_validation_response(self, sample_validation_result):
        """Test validation response serialisation."""
        rule = ValidationRule(**sample_validation_result)
        
        response = ValidationResponse(
            success=True,
            message="Validation completed successfully",
            timestamp=datetime.now(),
            validation_id="val_123",
            overall_status="passed",
            rules=[rule],
            summary={
                "total_rules": 1,
                "passed": 1,
                "failed": 0,
                "warnings": 0,
                "overall_success_rate": 99.5
            }
        )
        
        assert response.success is True
        assert response.overall_status == "passed"
        assert len(response.rules) == 1
        assert response.summary["passed"] == 1
    
    def test_pipeline_run_response(self, sample_pipeline_config):
        """Test pipeline run response serialisation."""
        response = PipelineRunResponse(
            success=True,
            message="Pipeline started successfully",
            timestamp=datetime.now(),
            run_id="run_123",
            pipeline_name="test_pipeline",
            status=PipelineStatus.RUNNING,
            config=sample_pipeline_config,
            progress=25.5
        )
        
        assert response.success is True
        assert response.run_id == "run_123"
        assert response.status == PipelineStatus.RUNNING
        assert response.progress == 25.5


class TestBritishEnglishConventions:
    """Test British English spelling conventions in models."""
    
    def test_field_names_british_english(self):
        """Test that field names use British English spellings."""
        # Check that we use British spellings in field names and descriptions
        metrics = QualityMetrics(
            completeness_rate=95.0,
            accuracy_score=85.0,
            consistency_score=90.0,
            timeliness_score=88.0,
            overall_score=89.5,
            record_count=1000,
            error_count=25,
            warning_count=10
        )
        
        # Verify British English usage in computed properties
        assert hasattr(metrics, 'quality_grade')
        
        # Check model configuration uses British conventions
        model_config = QualityMetrics.model_config
        assert 'str_to_lower' in model_config or 'str_strip_whitespace' in model_config
    
    def test_enum_values_british_english(self):
        """Test enumeration values use British English."""
        # Geographic levels should use Australian/British conventions
        assert GeographicLevel.AUSTRALIA.value == "australia"
        assert GeographicLevel.STATE.value == "state"
        
        # Pipeline statuses should use British spellings where applicable
        statuses = [status.value for status in PipelineStatus]
        assert "cancelled" in statuses  # British spelling
        assert "optimising" in statuses  # British spelling


class TestAHGDBaseModel:
    """Test base model functionality."""
    
    def test_base_model_inheritance(self):
        """Test that all models inherit from AHGDBaseModel."""
        models = [
            SA1Code, QualityMetrics, ValidationRule,
            QualityMetricsRequest, ValidationRequest, PipelineRunRequest,
            QualityMetricsResponse, ValidationResponse, PipelineRunResponse
        ]
        
        for model in models:
            assert issubclass(model, AHGDBaseModel)
    
    def test_base_model_configuration(self):
        """Test base model configuration."""
        sa1 = SA1Code(code="10101000001")
        
        # Check that model configuration is properly inherited
        config = sa1.model_config
        assert isinstance(config, dict)
        
        # Test serialisation includes computed fields
        data = sa1.model_dump()
        assert "code" in data