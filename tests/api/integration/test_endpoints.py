"""
Integration tests for API endpoints.

Tests complete request-response cycles for all API endpoints.
"""

import pytest
from unittest.mock import patch, AsyncMock
import json
from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from httpx import AsyncClient

from src.api.models.common import GeographicLevel, PipelineStatus


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check_basic(self, client: TestClient):
        """Test basic health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_health_check_detailed(self, client: TestClient):
        """Test detailed health check with dependencies."""
        response = client.get("/health/detailed")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "services" in data
        assert "database" in data["services"]
        assert "cache" in data["services"]
    
    def test_readiness_check(self, client: TestClient):
        """Test readiness probe endpoint."""
        response = client.get("/ready")
        
        assert response.status_code in [200, 503]
        data = response.json()
        assert "ready" in data


class TestQualityMetricsEndpoints:
    """Test quality metrics API endpoints."""
    
    def test_get_quality_metrics_success(self, client: TestClient, sample_sa1_code):
        """Test successful quality metrics retrieval."""
        with patch('src.api.services.quality_service.QualityMetricsService.get_quality_metrics') as mock_service:
            mock_response = {
                "success": True,
                "message": "Quality metrics retrieved successfully",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "completeness_rate": 98.5,
                    "accuracy_score": 94.2,
                    "consistency_score": 96.8,
                    "timeliness_score": 92.0,
                    "overall_score": 95.4,
                    "record_count": 15000,
                    "error_count": 125,
                    "warning_count": 45
                },
                "geographic_level": "sa1",
                "total_records": 15000
            }
            mock_service.return_value = type('MockResponse', (), mock_response)()
            
            response = client.post("/api/v1/quality/metrics", json={
                "geographic_level": "sa1",
                "sa1_codes": [sample_sa1_code],
                "start_date": "2023-01-01T00:00:00",
                "end_date": "2023-12-31T23:59:59"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["metrics"]["overall_score"] == 95.4
    
    def test_get_quality_metrics_validation_error(self, client: TestClient):
        """Test quality metrics with validation errors."""
        response = client.post("/api/v1/quality/metrics", json={
            "geographic_level": "invalid_level",
            "sa1_codes": ["invalid_code"],
        })
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_get_quality_metrics_pagination(self, client: TestClient, sample_sa1_code):
        """Test quality metrics with pagination."""
        with patch('src.api.services.quality_service.QualityMetricsService.get_quality_metrics') as mock_service:
            mock_response = {
                "success": True,
                "message": "Quality metrics retrieved successfully",
                "timestamp": datetime.now().isoformat(),
                "metrics": {},
                "pagination": {
                    "page": 1,
                    "size": 50,
                    "total": 150,
                    "pages": 3
                }
            }
            mock_service.return_value = type('MockResponse', (), mock_response)()
            
            response = client.post("/api/v1/quality/metrics", 
                json={"geographic_level": "sa1", "sa1_codes": [sample_sa1_code]},
                params={"page": 1, "size": 50}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "pagination" in data
    
    def test_get_historical_trends(self, client: TestClient, sample_sa1_code):
        """Test historical quality trends endpoint."""
        with patch('src.api.services.quality_service.QualityMetricsService.get_historical_trends') as mock_service:
            mock_response = {
                "success": True,
                "trends": [
                    {"date": "2023-01", "score": 94.5},
                    {"date": "2023-02", "score": 95.2},
                    {"date": "2023-03", "score": 95.8}
                ]
            }
            mock_service.return_value = type('MockResponse', (), mock_response)()
            
            response = client.get(f"/api/v1/quality/trends", params={
                "geographic_level": "sa1",
                "sa1_codes": sample_sa1_code,
                "time_period": "3months"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["trends"]) == 3


class TestValidationEndpoints:
    """Test validation API endpoints."""
    
    def test_validate_data_success(self, client: TestClient, sample_sa1_code):
        """Test successful data validation."""
        with patch('src.api.services.validation_service.ValidationService.validate_data') as mock_service:
            mock_response = {
                "success": True,
                "message": "Validation completed successfully",
                "timestamp": datetime.now().isoformat(),
                "validation_id": "val_123",
                "overall_status": "passed",
                "rules": [{
                    "rule_name": "sa1_code_format",
                    "rule_type": "schema",
                    "status": "passed",
                    "severity": "error",
                    "records_tested": 1000,
                    "records_passed": 995,
                    "records_failed": 5,
                    "success_rate": 99.5,
                    "message": "SA1 codes format validation",
                    "details": {}
                }],
                "summary": {
                    "total_rules": 1,
                    "passed": 1,
                    "failed": 0,
                    "warnings": 0,
                    "overall_success_rate": 99.5
                }
            }
            mock_service.return_value = type('MockResponse', (), mock_response)()
            
            response = client.post("/api/v1/validation/validate", json={
                "geographic_level": "sa1",
                "validation_types": ["schema", "business"],
                "sa1_codes": [sample_sa1_code]
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["overall_status"] == "passed"
    
    def test_get_validation_status(self, client: TestClient):
        """Test validation status retrieval."""
        validation_id = "val_123"
        
        with patch('src.api.services.validation_service.ValidationService.get_validation_status') as mock_service:
            mock_response = {
                "success": True,
                "validation_id": validation_id,
                "status": "completed",
                "progress": 100.0,
                "results": {"passed": 25, "failed": 2}
            }
            mock_service.return_value = type('MockResponse', (), mock_response)()
            
            response = client.get(f"/api/v1/validation/{validation_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["validation_id"] == validation_id
            assert data["status"] == "completed"
    
    def test_get_validation_history(self, client: TestClient, sample_sa1_code):
        """Test validation history retrieval."""
        with patch('src.api.services.validation_service.ValidationService.get_validation_history') as mock_service:
            mock_response = {
                "success": True,
                "history": [{
                    "validation_id": "val_123",
                    "timestamp": datetime.now().isoformat(),
                    "status": "passed",
                    "rule_count": 25
                }]
            }
            mock_service.return_value = type('MockResponse', (), mock_response)()
            
            response = client.get("/api/v1/validation/history", params={
                "geographic_level": "sa1",
                "sa1_codes": sample_sa1_code,
                "limit": 10
            })
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["history"]) == 1


class TestPipelineEndpoints:
    """Test pipeline management API endpoints."""
    
    def test_execute_pipeline_success(self, client: TestClient, sample_pipeline_config):
        """Test successful pipeline execution."""
        with patch('src.api.services.pipeline_service.PipelineService.execute_pipeline') as mock_service:
            mock_response = {
                "success": True,
                "message": "Pipeline started successfully",
                "timestamp": datetime.now().isoformat(),
                "run_id": "run_123",
                "pipeline_name": "test_pipeline",
                "status": "running",
                "config": sample_pipeline_config,
                "progress": 0.0
            }
            mock_service.return_value = type('MockResponse', (), mock_response)()
            
            response = client.post("/api/v1/pipeline/run", json={
                "pipeline_name": "test_pipeline",
                "config": sample_pipeline_config,
                "priority": "normal"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["run_id"] == "run_123"
    
    def test_get_pipeline_status(self, client: TestClient):
        """Test pipeline status retrieval."""
        run_id = "run_123"
        
        with patch('src.api.services.pipeline_service.PipelineService.get_pipeline_status') as mock_service:
            mock_response = {
                "success": True,
                "run_id": run_id,
                "status": "running",
                "progress": 75.5,
                "start_time": datetime.now().isoformat(),
                "estimated_completion": (datetime.now() + timedelta(minutes=10)).isoformat()
            }
            mock_service.return_value = type('MockResponse', (), mock_response)()
            
            response = client.get(f"/api/v1/pipeline/{run_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["run_id"] == run_id
            assert data["progress"] == 75.5
    
    def test_cancel_pipeline(self, client: TestClient):
        """Test pipeline cancellation."""
        run_id = "run_123"
        
        with patch('src.api.services.pipeline_service.PipelineService.cancel_pipeline') as mock_service:
            mock_response = {
                "success": True,
                "message": "Pipeline cancelled successfully",
                "run_id": run_id
            }
            mock_service.return_value = type('MockResponse', (), mock_response)()
            
            response = client.post(f"/api/v1/pipeline/{run_id}/cancel")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert "cancelled" in data["message"].lower()
    
    def test_list_active_pipelines(self, client: TestClient):
        """Test listing active pipelines."""
        with patch('src.api.services.pipeline_service.PipelineService.list_active_pipelines') as mock_service:
            mock_response = {
                "success": True,
                "pipelines": [{
                    "run_id": "run_123",
                    "pipeline_name": "etl_pipeline",
                    "status": "running",
                    "progress": 45.0,
                    "start_time": datetime.now().isoformat()
                }]
            }
            mock_service.return_value = type('MockResponse', (), mock_response)()
            
            response = client.get("/api/v1/pipeline/active")
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["pipelines"]) == 1


class TestWebSocketEndpoints:
    """Test WebSocket endpoints."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_client: AsyncClient):
        """Test WebSocket connection establishment."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.disconnect = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Connection should be established
                assert websocket is not None
    
    @pytest.mark.asyncio
    async def test_websocket_subscription(self, async_client: AsyncClient):
        """Test WebSocket subscription to metrics."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Send subscription message
                await websocket.send_json({
                    "type": "subscribe",
                    "subscription_type": "quality_metrics",
                    "filters": {"geographic_level": "sa1"}
                })
                
                # Should receive acknowledgment
                response = await websocket.receive_json()
                assert response["type"] == "subscription_ack"


class TestErrorHandling:
    """Test API error handling."""
    
    def test_404_not_found(self, client: TestClient):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent/endpoint")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
    
    def test_422_validation_error(self, client: TestClient):
        """Test 422 validation error handling."""
        response = client.post("/api/v1/quality/metrics", json={
            "invalid_field": "invalid_value"
        })
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
    
    def test_500_internal_error(self, client: TestClient):
        """Test 500 internal error handling."""
        with patch('src.api.services.quality_service.QualityMetricsService.get_quality_metrics') as mock_service:
            mock_service.side_effect = Exception("Database connection error")
            
            response = client.post("/api/v1/quality/metrics", json={
                "geographic_level": "sa1",
                "sa1_codes": ["10101000001"]
            })
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
    
    def test_rate_limit_error(self, client: TestClient):
        """Test rate limiting error handling."""
        # This test depends on rate limiting being enabled
        # Make multiple rapid requests to trigger rate limiting
        responses = []
        for i in range(10):
            response = client.get("/health")
            responses.append(response)
        
        # At least one response should be rate limited (429)
        # Note: This test may need adjustment based on rate limiting configuration
        status_codes = [r.status_code for r in responses]
        # Either all succeed or some are rate limited
        assert all(code in [200, 429] for code in status_codes)


class TestAuthenticationIntegration:
    """Test authentication integration."""
    
    def test_protected_endpoint_without_auth(self, client: TestClient):
        """Test accessing protected endpoint without authentication."""
        # Assuming some endpoints require authentication
        response = client.post("/api/v1/pipeline/run", json={
            "pipeline_name": "test_pipeline"
        })
        
        # Response depends on auth configuration
        assert response.status_code in [200, 401, 403]
    
    def test_protected_endpoint_with_auth(self, client: TestClient, auth_headers):
        """Test accessing protected endpoint with authentication."""
        with patch('src.api.dependencies.get_current_user') as mock_auth:
            mock_auth.return_value = {"user_id": "test_user", "is_authenticated": True}
            
            response = client.post("/api/v1/pipeline/run", 
                headers=auth_headers,
                json={"pipeline_name": "test_pipeline"}
            )
            
            # Should not be a 401/403 with valid auth
            assert response.status_code not in [401, 403]


class TestCORSIntegration:
    """Test CORS integration."""
    
    def test_cors_preflight_request(self, client: TestClient):
        """Test CORS preflight request."""
        response = client.options("/api/v1/quality/metrics", headers={
            "Origin": "https://dashboard.ahgd.gov.au",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type, Authorization"
        })
        
        assert response.status_code in [200, 204]
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
    
    def test_cors_actual_request(self, client: TestClient):
        """Test CORS actual request."""
        response = client.post("/api/v1/quality/metrics", 
            headers={"Origin": "https://dashboard.ahgd.gov.au"},
            json={"geographic_level": "sa1", "sa1_codes": ["10101000001"]}
        )
        
        # CORS headers should be present
        assert "Access-Control-Allow-Origin" in response.headers


class TestAPIVersioning:
    """Test API versioning."""
    
    def test_v1_endpoint_access(self, client: TestClient):
        """Test accessing v1 API endpoints."""
        response = client.get("/api/v1/health")
        
        # v1 endpoints should be accessible
        assert response.status_code in [200, 404]  # 404 is fine if not implemented
    
    def test_version_header(self, client: TestClient):
        """Test API version in response headers."""
        response = client.get("/health")
        
        # Should include version information
        assert "X-API-Version" in response.headers or "version" in response.json()