"""
Unit tests for API services.

Tests service layer functionality including quality metrics, validation, and pipeline management.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from src.api.services.quality_service import QualityMetricsService
from src.api.services.validation_service import ValidationService
from src.api.services.pipeline_service import PipelineService
from src.api.models.requests import (
    QualityMetricsRequest, ValidationRequest, PipelineRunRequest
)
from src.api.models.common import (
    GeographicLevel, PipelineStatus, QualityMetrics, ValidationRule
)


class TestQualityMetricsService:
    """Test quality metrics service functionality."""
    
    @pytest.fixture
    def service(self):
        """Create quality metrics service instance."""
        return QualityMetricsService()
    
    @pytest.fixture
    def quality_request(self, sample_sa1_code):
        """Create sample quality metrics request."""
        return QualityMetricsRequest(
            geographic_level=GeographicLevel.SA1,
            sa1_codes=[sample_sa1_code],
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_get_quality_metrics_success(self, service, quality_request, sample_quality_metrics):
        """Test successful quality metrics retrieval."""
        with patch('src.api.services.quality_service.QualityChecker') as mock_checker:
            mock_checker.return_value.calculate_quality_metrics = AsyncMock(
                return_value=sample_quality_metrics
            )
            
            response = await service.get_quality_metrics(quality_request)
            
            assert response.success is True
            assert response.metrics.overall_score == 95.4
            assert response.geographic_level == GeographicLevel.SA1
    
    @pytest.mark.asyncio
    async def test_get_quality_metrics_with_cache(self, service, quality_request, sample_quality_metrics):
        """Test quality metrics retrieval with caching."""
        mock_cache = AsyncMock()
        mock_cache.get.return_value = sample_quality_metrics
        
        response = await service.get_quality_metrics(quality_request, cache_manager=mock_cache)
        
        assert response.success is True
        mock_cache.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_quality_metrics_filtering(self, service, quality_request):
        """Test quality metrics with geographic filtering."""
        quality_request.geographic_bounds = {
            "min_lat": -37.8,
            "max_lat": -37.7,
            "min_lon": 144.9,
            "max_lon": 145.0
        }
        
        with patch('src.api.services.quality_service.QualityChecker') as mock_checker:
            mock_checker.return_value.calculate_quality_metrics = AsyncMock()
            
            await service.get_quality_metrics(quality_request)
            
            mock_checker.return_value.calculate_quality_metrics.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_historical_trends(self, service, sample_sa1_code):
        """Test historical quality trends retrieval."""
        with patch('src.api.services.quality_service.QualityChecker') as mock_checker:
            mock_trends = [
                {"date": "2023-01", "score": 94.5},
                {"date": "2023-02", "score": 95.2},
                {"date": "2023-03", "score": 95.8}
            ]
            mock_checker.return_value.get_historical_trends = AsyncMock(
                return_value=mock_trends
            )
            
            response = await service.get_historical_trends(
                geographic_level=GeographicLevel.SA1,
                sa1_codes=[sample_sa1_code],
                time_period="3months"
            )
            
            assert response.success is True
            assert len(response.trends) == 3
            assert response.trends[0]["score"] == 94.5


class TestValidationService:
    """Test validation service functionality."""
    
    @pytest.fixture
    def service(self):
        """Create validation service instance."""
        return ValidationService()
    
    @pytest.fixture
    def validation_request(self, sample_sa1_code):
        """Create sample validation request."""
        return ValidationRequest(
            geographic_level=GeographicLevel.SA1,
            validation_types=["schema", "business"],
            sa1_codes=[sample_sa1_code],
            severity_filter=["error", "warning"]
        )
    
    @pytest.mark.asyncio
    async def test_validate_data_success(self, service, validation_request, sample_validation_result):
        """Test successful data validation."""
        with patch('src.api.services.validation_service.ValidationOrchestrator') as mock_orchestrator:
            mock_result = ValidationRule(**sample_validation_result)
            mock_orchestrator.return_value.run_validation = AsyncMock(
                return_value=[mock_result]
            )
            
            response = await service.validate_data(validation_request)
            
            assert response.success is True
            assert len(response.rules) == 1
            assert response.rules[0].rule_name == "sa1_code_format"
            assert response.overall_status == "passed"
    
    @pytest.mark.asyncio
    async def test_validate_data_with_failures(self, service, validation_request):
        """Test validation with failed rules."""
        failed_result = {
            "rule_name": "completeness_check",
            "rule_type": "business",
            "status": "failed",
            "severity": "error",
            "records_tested": 1000,
            "records_passed": 800,
            "records_failed": 200,
            "success_rate": 80.0,
            "message": "Data completeness below threshold",
            "details": {"threshold": 95.0, "actual": 80.0}
        }
        
        with patch('src.api.services.validation_service.ValidationOrchestrator') as mock_orchestrator:
            mock_result = ValidationRule(**failed_result)
            mock_orchestrator.return_value.run_validation = AsyncMock(
                return_value=[mock_result]
            )
            
            response = await service.validate_data(validation_request)
            
            assert response.success is True  # Service call succeeded
            assert response.overall_status == "failed"  # But validation failed
            assert response.summary["failed"] == 1
    
    @pytest.mark.asyncio
    async def test_validate_data_filtering(self, service, validation_request):
        """Test validation with type and severity filtering."""
        validation_request.validation_types = ["schema"]
        validation_request.severity_filter = ["error"]
        
        with patch('src.api.services.validation_service.ValidationOrchestrator') as mock_orchestrator:
            mock_orchestrator.return_value.run_validation = AsyncMock(return_value=[])
            
            await service.validate_data(validation_request)
            
            # Verify filtering was applied
            call_args = mock_orchestrator.return_value.run_validation.call_args
            assert "schema" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_get_validation_history(self, service, sample_sa1_code):
        """Test validation history retrieval."""
        mock_history = [
            {
                "validation_id": "val_123",
                "timestamp": datetime.now().isoformat(),
                "status": "passed",
                "rule_count": 25
            }
        ]
        
        with patch('src.api.services.validation_service.ValidationOrchestrator') as mock_orchestrator:
            mock_orchestrator.return_value.get_validation_history = AsyncMock(
                return_value=mock_history
            )
            
            response = await service.get_validation_history(
                geographic_level=GeographicLevel.SA1,
                sa1_codes=[sample_sa1_code],
                limit=10
            )
            
            assert response.success is True
            assert len(response.history) == 1
            assert response.history[0]["validation_id"] == "val_123"


class TestPipelineService:
    """Test pipeline service functionality."""
    
    @pytest.fixture
    def service(self):
        """Create pipeline service instance."""
        return PipelineService()
    
    @pytest.fixture
    def pipeline_request(self, sample_pipeline_config):
        """Create sample pipeline run request."""
        return PipelineRunRequest(
            pipeline_name="test_etl_pipeline",
            config=sample_pipeline_config,
            priority="normal"
        )
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_success(self, service, pipeline_request):
        """Test successful pipeline execution."""
        with patch('src.api.services.pipeline_service.PipelineMonitor') as mock_monitor:
            mock_monitor.return_value.start_pipeline = AsyncMock(
                return_value="run_123"
            )
            
            response = await service.execute_pipeline(pipeline_request)
            
            assert response.success is True
            assert response.run_id == "run_123"
            assert response.status == PipelineStatus.RUNNING
    
    @pytest.mark.asyncio
    async def test_execute_pipeline_concurrency_limit(self, service, pipeline_request):
        """Test pipeline execution with concurrency limits."""
        # Mock active runs exceeding limit
        service.active_runs = {"run_1": {}, "run_2": {}, "run_3": {}}
        service.max_concurrent_runs = 3
        
        response = await service.execute_pipeline(pipeline_request)
        
        assert response.success is False
        assert "concurrency limit" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_get_pipeline_status(self, service):
        """Test pipeline status retrieval."""
        run_id = "run_123"
        
        with patch('src.api.services.pipeline_service.PipelineMonitor') as mock_monitor:
            mock_status = {
                "run_id": run_id,
                "status": "running",
                "progress": 75.5,
                "start_time": datetime.now().isoformat(),
                "stages_completed": ["extract", "transform"],
                "current_stage": "validate"
            }
            mock_monitor.return_value.get_run_status = AsyncMock(
                return_value=mock_status
            )
            
            response = await service.get_pipeline_status(run_id)
            
            assert response.success is True
            assert response.run_id == run_id
            assert response.status == PipelineStatus.RUNNING
            assert response.progress == 75.5
    
    @pytest.mark.asyncio
    async def test_cancel_pipeline(self, service):
        """Test pipeline cancellation."""
        run_id = "run_123"
        
        with patch('src.api.services.pipeline_service.PipelineMonitor') as mock_monitor:
            mock_monitor.return_value.cancel_pipeline = AsyncMock(
                return_value=True
            )
            
            response = await service.cancel_pipeline(run_id)
            
            assert response.success is True
            assert response.message == "Pipeline cancelled successfully"
    
    @pytest.mark.asyncio
    async def test_list_active_pipelines(self, service):
        """Test listing active pipelines."""
        mock_pipelines = [
            {
                "run_id": "run_123",
                "pipeline_name": "etl_pipeline",
                "status": "running",
                "progress": 45.0,
                "start_time": datetime.now().isoformat()
            },
            {
                "run_id": "run_456",
                "pipeline_name": "validation_pipeline",
                "status": "queued",
                "progress": 0.0,
                "start_time": None
            }
        ]
        
        with patch('src.api.services.pipeline_service.PipelineMonitor') as mock_monitor:
            mock_monitor.return_value.list_active_runs = AsyncMock(
                return_value=mock_pipelines
            )
            
            response = await service.list_active_pipelines()
            
            assert response.success is True
            assert len(response.pipelines) == 2
            assert response.pipelines[0]["run_id"] == "run_123"
    
    @pytest.mark.asyncio
    async def test_get_pipeline_metrics(self, service):
        """Test pipeline performance metrics retrieval."""
        mock_metrics = {
            "total_runs": 150,
            "success_rate": 94.7,
            "average_duration": 1800,  # 30 minutes
            "failure_rate": 5.3,
            "throughput_per_hour": 3.2,
            "resource_utilisation": {
                "cpu": 65.5,
                "memory": 78.2,
                "disk_io": 45.8
            }
        }
        
        with patch('src.api.services.pipeline_service.PipelineMonitor') as mock_monitor:
            mock_monitor.return_value.get_performance_metrics = AsyncMock(
                return_value=mock_metrics
            )
            
            response = await service.get_pipeline_metrics(days=30)
            
            assert response.success is True
            assert response.metrics["success_rate"] == 94.7
            assert response.metrics["total_runs"] == 150


class TestServiceIntegration:
    """Test integration between services."""
    
    @pytest.fixture
    def quality_service(self):
        return QualityMetricsService()
    
    @pytest.fixture
    def validation_service(self):
        return ValidationService()
    
    @pytest.fixture
    def pipeline_service(self):
        return PipelineService()
    
    @pytest.mark.asyncio
    async def test_service_error_handling(self, quality_service, quality_request):
        """Test service error handling patterns."""
        with patch('src.api.services.quality_service.QualityChecker') as mock_checker:
            mock_checker.return_value.calculate_quality_metrics = AsyncMock(
                side_effect=Exception("Database connection error")
            )
            
            response = await quality_service.get_quality_metrics(quality_request)
            
            assert response.success is False
            assert "error" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_service_british_english_usage(self, validation_service, validation_request):
        """Test that services use British English in responses."""
        with patch('src.api.services.validation_service.ValidationOrchestrator') as mock_orchestrator:
            mock_orchestrator.return_value.run_validation = AsyncMock(return_value=[])
            
            response = await validation_service.validate_data(validation_request)
            
            # Check that British English is used in messages
            assert "optimised" in response.message or "optimized" not in response.message
            assert "analysed" in response.message or "analyzed" not in response.message
    
    @pytest.mark.asyncio
    async def test_service_performance_monitoring(self, quality_service, quality_request):
        """Test that services have performance monitoring decorators."""
        # Verify that services use the @monitor_performance decorator
        assert hasattr(quality_service.get_quality_metrics, '__wrapped__')
        
        with patch('src.api.services.quality_service.QualityChecker') as mock_checker:
            mock_checker.return_value.calculate_quality_metrics = AsyncMock(
                return_value={}
            )
            
            await quality_service.get_quality_metrics(quality_request)
    
    def test_service_configuration(self, quality_service, validation_service, pipeline_service):
        """Test service configuration and initialisation."""
        # Verify services are properly configured
        assert quality_service.cache_ttl > 0
        assert validation_service.default_severity_levels is not None
        assert pipeline_service.max_concurrent_runs > 0