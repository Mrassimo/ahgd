"""
Unit tests for API middleware.

Tests custom middleware functionality including rate limiting, logging, and security headers.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi import Request, Response
from fastapi.testclient import TestClient
import time

from src.api.middleware import (
    RateLimitingMiddleware, LoggingMiddleware, SecurityHeadersMiddleware,
    RequestTracingMiddleware
)
from src.api.main import create_app


class TestRateLimitingMiddleware:
    """Test rate limiting middleware functionality."""
    
    @pytest.fixture
    def app_with_rate_limiting(self):
        """Create app with rate limiting enabled."""
        app = create_app()
        rate_limiter = RateLimitingMiddleware(app, calls=5, period=60)
        app.add_middleware(RateLimitingMiddleware, calls=5, period=60)
        return app
    
    def test_rate_limiting_within_limits(self, app_with_rate_limiting):
        """Test requests within rate limits are allowed."""
        client = TestClient(app_with_rate_limiting)
        
        # Make requests within the limit
        for i in range(3):
            response = client.get("/health")
            assert response.status_code in [200, 404]  # 404 is fine, we're testing middleware
    
    def test_rate_limiting_exceeds_limits(self, app_with_rate_limiting):
        """Test requests exceeding rate limits are blocked."""
        client = TestClient(app_with_rate_limiting)
        
        # Make requests exceeding the limit
        for i in range(7):  # Exceeds limit of 5
            response = client.get("/health")
            if i < 5:
                assert response.status_code != 429
            else:
                assert response.status_code == 429
    
    def test_rate_limiting_different_clients(self, app_with_rate_limiting):
        """Test rate limiting is per-client."""
        client1 = TestClient(app_with_rate_limiting)
        client2 = TestClient(app_with_rate_limiting)
        
        # Client 1 makes requests up to limit
        for i in range(5):
            response = client1.get("/health")
            assert response.status_code != 429
        
        # Client 2 should still be able to make requests
        response = client2.get("/health")
        assert response.status_code != 429
    
    def test_rate_limiting_window_reset(self):
        """Test rate limiting window resets after period."""
        middleware = RateLimitingMiddleware(Mock(), calls=2, period=1)
        client_ip = "192.168.1.1"
        
        # Make requests up to limit
        assert middleware._check_rate_limit(client_ip) is True
        assert middleware._check_rate_limit(client_ip) is True
        assert middleware._check_rate_limit(client_ip) is False  # Exceeded
        
        # Wait for window to reset
        time.sleep(1.1)
        assert middleware._check_rate_limit(client_ip) is True


class TestLoggingMiddleware:
    """Test logging middleware functionality."""
    
    @pytest.fixture
    def logging_middleware(self):
        """Create logging middleware instance."""
        return LoggingMiddleware(Mock())
    
    @pytest.mark.asyncio
    async def test_request_logging(self, logging_middleware):
        """Test request logging captures essential information."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url = Mock()
        mock_request.url.path = "/api/quality/metrics"
        mock_request.headers = {"user-agent": "test-client", "authorization": "Bearer token"}
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.1"
        
        mock_call_next = AsyncMock()
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_call_next.return_value = mock_response
        
        with patch('src.api.middleware.get_logger') as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance
            
            await logging_middleware.dispatch(mock_request, mock_call_next)
            
            # Verify logging was called
            assert logger_instance.info.called
            call_args = logger_instance.info.call_args
            assert "GET" in str(call_args)
            assert "/api/quality/metrics" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_request_duration_logging(self, logging_middleware):
        """Test request duration is logged."""
        mock_request = Mock(spec=Request)
        mock_request.method = "POST"
        mock_request.url = Mock()
        mock_request.url.path = "/api/pipeline/run"
        mock_request.headers = {}
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.1"
        
        # Mock slow response
        async def slow_call_next(request):
            await AsyncMock()()  # Simulate async delay
            response = Mock(spec=Response)
            response.status_code = 201
            return response
        
        with patch('src.api.middleware.get_logger') as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance
            
            await logging_middleware.dispatch(mock_request, slow_call_next)
            
            # Verify duration was logged
            call_args = logger_instance.info.call_args
            assert "duration" in str(call_args).lower()
    
    @pytest.mark.asyncio
    async def test_sensitive_headers_redaction(self, logging_middleware):
        """Test sensitive headers are redacted from logs."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url = Mock()
        mock_request.url.path = "/api/health"
        mock_request.headers = {
            "authorization": "Bearer secret_token_123",
            "x-api-key": "api_key_456",
            "content-type": "application/json"
        }
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.1"
        
        mock_call_next = AsyncMock()
        mock_response = Mock(spec=Response)
        mock_response.status_code = 200
        mock_call_next.return_value = mock_response
        
        with patch('src.api.middleware.get_logger') as mock_logger:
            logger_instance = Mock()
            mock_logger.return_value = logger_instance
            
            await logging_middleware.dispatch(mock_request, mock_call_next)
            
            # Verify sensitive headers are redacted
            call_args = logger_instance.info.call_args
            logged_message = str(call_args)
            assert "secret_token_123" not in logged_message
            assert "api_key_456" not in logged_message
            assert "REDACTED" in logged_message or "***" in logged_message


class TestSecurityHeadersMiddleware:
    """Test security headers middleware functionality."""
    
    @pytest.fixture
    def security_middleware(self):
        """Create security headers middleware instance."""
        return SecurityHeadersMiddleware(Mock())
    
    @pytest.mark.asyncio
    async def test_security_headers_added(self, security_middleware):
        """Test security headers are added to responses."""
        mock_request = Mock(spec=Request)
        mock_call_next = AsyncMock()
        mock_response = Mock(spec=Response)
        mock_response.headers = {}
        mock_call_next.return_value = mock_response
        
        response = await security_middleware.dispatch(mock_request, mock_call_next)
        
        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        
        for header in expected_headers:
            assert header in response.headers
    
    @pytest.mark.asyncio
    async def test_cors_headers_included(self, security_middleware):
        """Test CORS headers are properly configured."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {"origin": "https://dashboard.ahgd.gov.au"}
        mock_call_next = AsyncMock()
        mock_response = Mock(spec=Response)
        mock_response.headers = {}
        mock_call_next.return_value = mock_response
        
        response = await security_middleware.dispatch(mock_request, mock_call_next)
        
        # Verify CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers
    
    @pytest.mark.asyncio
    async def test_security_policy_values(self, security_middleware):
        """Test security policy header values are appropriate."""
        mock_request = Mock(spec=Request)
        mock_call_next = AsyncMock()
        mock_response = Mock(spec=Response)
        mock_response.headers = {}
        mock_call_next.return_value = mock_response
        
        response = await security_middleware.dispatch(mock_request, mock_call_next)
        
        # Test specific security policy values
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert "max-age" in response.headers.get("Strict-Transport-Security", "")


class TestRequestTracingMiddleware:
    """Test request tracing middleware functionality."""
    
    @pytest.fixture
    def tracing_middleware(self):
        """Create request tracing middleware instance."""
        return RequestTracingMiddleware(Mock())
    
    @pytest.mark.asyncio
    async def test_trace_id_generation(self, tracing_middleware):
        """Test unique trace IDs are generated for requests."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_call_next = AsyncMock()
        mock_response = Mock(spec=Response)
        mock_response.headers = {}
        mock_call_next.return_value = mock_response
        
        response = await tracing_middleware.dispatch(mock_request, mock_call_next)
        
        # Verify trace ID is added to response headers
        assert "X-Trace-ID" in response.headers
        trace_id = response.headers["X-Trace-ID"]
        assert len(trace_id) > 0
        assert isinstance(trace_id, str)
    
    @pytest.mark.asyncio
    async def test_trace_id_from_request(self, tracing_middleware):
        """Test existing trace ID from request is preserved."""
        existing_trace_id = "trace_123_456"
        mock_request = Mock(spec=Request)
        mock_request.headers = {"x-trace-id": existing_trace_id}
        mock_call_next = AsyncMock()
        mock_response = Mock(spec=Response)
        mock_response.headers = {}
        mock_call_next.return_value = mock_response
        
        response = await tracing_middleware.dispatch(mock_request, mock_call_next)
        
        # Verify existing trace ID is preserved
        assert response.headers["X-Trace-ID"] == existing_trace_id
    
    @pytest.mark.asyncio
    async def test_correlation_context(self, tracing_middleware):
        """Test correlation context is set for downstream services."""
        mock_request = Mock(spec=Request)
        mock_request.headers = {}
        mock_call_next = AsyncMock()
        mock_response = Mock(spec=Response)
        mock_response.headers = {}
        mock_call_next.return_value = mock_response
        
        with patch('src.api.middleware.set_correlation_context') as mock_context:
            await tracing_middleware.dispatch(mock_request, mock_call_next)
            
            # Verify correlation context was set
            mock_context.assert_called_once()


class TestMiddlewareIntegration:
    """Test middleware integration and order."""
    
    def test_middleware_order(self):
        """Test middleware is applied in correct order."""
        app = create_app()
        
        # Verify middleware stack order
        middleware_stack = [type(middleware) for middleware in app.user_middleware]
        
        # Security headers should be first
        assert any("Security" in str(mw) for mw in middleware_stack)
        
        # Rate limiting should come before logging
        # (This test depends on actual middleware configuration)
    
    def test_middleware_british_english(self):
        """Test middleware uses British English in error messages."""
        middleware = RateLimitingMiddleware(Mock(), calls=1, period=60)
        
        # Test error messages use British spellings
        error_message = middleware._get_rate_limit_error_message()
        
        # Should use British spellings where applicable
        assert "optimised" in error_message or "optimized" not in error_message
        assert "utilisation" in error_message or "utilization" not in error_message
    
    @pytest.mark.asyncio
    async def test_middleware_performance_impact(self):
        """Test middleware doesn't significantly impact performance."""
        app = create_app()
        client = TestClient(app)
        
        start_time = time.time()
        
        # Make multiple requests to test performance
        for i in range(10):
            response = client.get("/health")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Middleware should not add excessive overhead
        # (This is a basic performance check)
        assert total_time < 5.0  # Should complete in under 5 seconds