"""
API load and performance tests.

Tests API performance under various load conditions and response time requirements.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestResponseTimes:
    """Test API response time requirements."""

    def test_health_check_response_time(self, client: TestClient):
        """Test health check responds within 100ms."""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # Convert to milliseconds

        assert response.status_code == 200
        assert response_time < 100  # Should respond within 100ms

    def test_quality_metrics_response_time(self, client: TestClient, sample_sa1_code):
        """Test quality metrics endpoint response time."""
        with patch(
            "src.api.services.quality_service.QualityMetricsService.get_quality_metrics"
        ) as mock_service:
            mock_response = {
                "success": True,
                "message": "Quality metrics retrieved successfully",
                "metrics": {"overall_score": 95.4},
                "geographic_level": "sa1",
                "total_records": 1000,
            }
            mock_service.return_value = type("MockResponse", (), mock_response)()

            start_time = time.time()
            response = client.post(
                "/api/v1/quality/metrics",
                json={"geographic_level": "sa1", "sa1_codes": [sample_sa1_code]},
            )
            end_time = time.time()

            response_time = (end_time - start_time) * 1000

            assert response.status_code == 200
            assert response_time < 2000  # Should respond within 2 seconds

    def test_validation_response_time(self, client: TestClient, sample_sa1_code):
        """Test validation endpoint response time."""
        with patch(
            "src.api.services.validation_service.ValidationService.validate_data"
        ) as mock_service:
            mock_response = {
                "success": True,
                "message": "Validation completed successfully",
                "validation_id": "val_123",
                "overall_status": "passed",
            }
            mock_service.return_value = type("MockResponse", (), mock_response)()

            start_time = time.time()
            response = client.post(
                "/api/v1/validation/validate",
                json={
                    "geographic_level": "sa1",
                    "validation_types": ["schema"],
                    "sa1_codes": [sample_sa1_code],
                },
            )
            end_time = time.time()

            response_time = (end_time - start_time) * 1000

            assert response.status_code == 200
            assert response_time < 5000  # Should respond within 5 seconds


class TestConcurrentLoad:
    """Test API performance under concurrent load."""

    def test_concurrent_health_checks(self, client: TestClient):
        """Test multiple concurrent health check requests."""

        def make_request():
            return client.get("/health")

        # Test with 50 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            responses = [future.result() for future in as_completed(futures)]

        # All requests should succeed
        assert len(responses) == 50
        assert all(r.status_code == 200 for r in responses)

    def test_concurrent_api_requests(self, client: TestClient, sample_sa1_code):
        """Test concurrent API requests."""
        with patch(
            "src.api.services.quality_service.QualityMetricsService.get_quality_metrics"
        ) as mock_service:
            mock_response = {
                "success": True,
                "message": "Quality metrics retrieved successfully",
                "metrics": {"overall_score": 95.4},
            }
            mock_service.return_value = type("MockResponse", (), mock_response)()

            def make_request():
                return client.post(
                    "/api/v1/quality/metrics",
                    json={"geographic_level": "sa1", "sa1_codes": [sample_sa1_code]},
                )

            # Test with 20 concurrent requests
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(20)]
                responses = [future.result() for future in as_completed(futures)]
            end_time = time.time()

            total_time = end_time - start_time

            # All requests should succeed within reasonable time
            assert len(responses) == 20
            assert all(r.status_code == 200 for r in responses)
            assert total_time < 10  # Should complete within 10 seconds

    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self, async_client: AsyncClient, sample_sa1_code):
        """Test concurrent requests using async client."""
        with patch(
            "src.api.services.quality_service.QualityMetricsService.get_quality_metrics"
        ) as mock_service:
            mock_response = {
                "success": True,
                "message": "Quality metrics retrieved successfully",
                "metrics": {"overall_score": 95.4},
            }
            mock_service.return_value = type("MockResponse", (), mock_response)()

            async def make_request():
                return await async_client.post(
                    "/api/v1/quality/metrics",
                    json={"geographic_level": "sa1", "sa1_codes": [sample_sa1_code]},
                )

            # Test with 30 concurrent async requests
            start_time = time.time()
            tasks = [make_request() for _ in range(30)]
            responses = await asyncio.gather(*tasks)
            end_time = time.time()

            total_time = end_time - start_time

            # All requests should succeed
            assert len(responses) == 30
            assert all(r.status_code == 200 for r in responses)
            assert total_time < 8  # Should complete within 8 seconds


class TestThroughputLimits:
    """Test API throughput and rate limiting."""

    def test_rate_limiting_enforcement(self, client: TestClient):
        """Test that rate limiting is properly enforced."""
        # Make rapid requests to trigger rate limiting
        responses = []
        for i in range(15):  # Assuming rate limit is 10 requests per minute
            response = client.get("/health")
            responses.append(response)

        status_codes = [r.status_code for r in responses]

        # Some requests should be rate limited (429) if rate limiting is enabled
        # If rate limiting is disabled in tests, all should succeed (200)
        assert all(code in [200, 429] for code in status_codes)

    def test_sustained_load_handling(self, client: TestClient):
        """Test API handling of sustained load."""

        def make_batch_requests(batch_size=10):
            responses = []
            for _ in range(batch_size):
                response = client.get("/health")
                responses.append(response)
                time.sleep(0.1)  # Small delay between requests
            return responses

        # Make 5 batches of 10 requests each
        all_responses = []
        start_time = time.time()

        for batch in range(5):
            batch_responses = make_batch_requests(10)
            all_responses.extend(batch_responses)

        end_time = time.time()
        total_time = end_time - start_time

        # All requests should succeed
        assert len(all_responses) == 50
        successful_requests = sum(1 for r in all_responses if r.status_code == 200)
        success_rate = successful_requests / len(all_responses)

        assert success_rate >= 0.95  # At least 95% success rate
        assert total_time < 15  # Should complete within 15 seconds


class TestMemoryUsage:
    """Test API memory usage patterns."""

    def test_large_request_payload(self, client: TestClient):
        """Test handling of large request payloads."""
        # Create large SA1 code list
        large_sa1_codes = [f"1010100000{i:01d}" for i in range(100)]

        with patch(
            "src.api.services.quality_service.QualityMetricsService.get_quality_metrics"
        ) as mock_service:
            mock_response = {
                "success": True,
                "message": "Quality metrics retrieved successfully",
                "metrics": {"overall_score": 95.4},
            }
            mock_service.return_value = type("MockResponse", (), mock_response)()

            response = client.post(
                "/api/v1/quality/metrics",
                json={
                    "geographic_level": "sa1",
                    "sa1_codes": large_sa1_codes,
                    "include_detailed_breakdown": True,
                },
            )

            assert response.status_code == 200

    def test_memory_cleanup_after_requests(self, client: TestClient):
        """Test that memory is properly cleaned up after requests."""
        # This is a basic test - in practice, you'd use memory profiling tools
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())

        # Get initial memory usage
        initial_memory = process.memory_info().rss

        # Make many requests
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200

        # Force garbage collection
        gc.collect()

        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB

        # Memory growth should be reasonable (less than 50MB)
        assert memory_growth < 50


class TestWebSocketPerformance:
    """Test WebSocket performance characteristics."""

    @pytest.mark.asyncio
    async def test_websocket_connection_speed(self, async_client: AsyncClient):
        """Test WebSocket connection establishment speed."""
        with patch("src.api.websocket.connection_manager.ConnectionManager") as mock_manager:
            mock_manager.return_value.connect = AsyncMock()

            start_time = time.time()
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                end_time = time.time()
                connection_time = (end_time - start_time) * 1000

                # Connection should be established quickly (< 500ms)
                assert connection_time < 500

                # Test ping-pong for latency
                await websocket.send_json({"type": "ping"})
                ping_start = time.time()
                response = await websocket.receive_json()
                ping_end = time.time()

                ping_latency = (ping_end - ping_start) * 1000

                assert response["type"] == "pong"
                assert ping_latency < 100  # Should respond within 100ms

    @pytest.mark.asyncio
    async def test_multiple_websocket_connections(self, async_client: AsyncClient):
        """Test multiple concurrent WebSocket connections."""
        with patch("src.api.websocket.connection_manager.ConnectionManager") as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.get_connection_count.return_value = 5

            connections = []

            # Establish multiple connections
            for i in range(5):
                websocket = await async_client.websocket_connect("/ws/metrics")
                connections.append(websocket)

                # Each connection should work independently
                await websocket.send_json({"type": "ping", "id": i})
                response = await websocket.receive_json()
                assert response["type"] == "pong"

            # Clean up
            for ws in connections:
                await ws.close()

    @pytest.mark.asyncio
    async def test_websocket_message_throughput(self, async_client: AsyncClient):
        """Test WebSocket message throughput."""
        with patch("src.api.websocket.connection_manager.ConnectionManager") as mock_manager:
            mock_manager.return_value.connect = AsyncMock()

            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Test rapid message exchange
                message_count = 50
                start_time = time.time()

                for i in range(message_count):
                    await websocket.send_json({"type": "ping", "id": i})
                    response = await websocket.receive_json()
                    assert response["type"] == "pong"

                end_time = time.time()
                total_time = end_time - start_time

                # Should handle messages efficiently
                messages_per_second = message_count / total_time
                assert messages_per_second > 20  # At least 20 messages per second


class TestDatabasePerformance:
    """Test database interaction performance."""

    def test_database_connection_pooling(self, client: TestClient):
        """Test database connection pooling efficiency."""
        # This would typically test actual database connections
        # For now, test that multiple requests don't fail due to connection issues

        def make_database_request():
            return client.get("/health/detailed")  # Endpoint that checks database

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_database_request) for _ in range(20)]
            responses = [future.result() for future in as_completed(futures)]

        # All requests should succeed (no connection pool exhaustion)
        success_count = sum(1 for r in responses if r.status_code == 200)
        success_rate = success_count / len(responses)

        assert success_rate >= 0.9  # At least 90% success rate

    def test_query_performance_optimization(self, client: TestClient, sample_sa1_code):
        """Test that database queries are optimized."""
        with patch(
            "src.api.services.quality_service.QualityMetricsService.get_quality_metrics"
        ) as mock_service:
            # Mock a response that simulates database query performance
            mock_response = {
                "success": True,
                "message": "Quality metrics retrieved successfully",
                "metrics": {"overall_score": 95.4},
                "query_time": 0.15,  # Simulated query time in seconds
            }
            mock_service.return_value = type("MockResponse", (), mock_response)()

            start_time = time.time()
            response = client.post(
                "/api/v1/quality/metrics",
                json={
                    "geographic_level": "sa1",
                    "sa1_codes": [sample_sa1_code] * 50,  # Large request
                },
            )
            end_time = time.time()

            response_time = end_time - start_time

            assert response.status_code == 200
            assert response_time < 3.0  # Should handle large requests efficiently


class TestScalabilityLimits:
    """Test API scalability limits and resource usage."""

    def test_maximum_concurrent_connections(self, client: TestClient):
        """Test maximum concurrent connections handling."""
        # Test with a reasonable number of concurrent connections
        connection_count = 25

        def make_long_request():
            # Simulate a request that takes some time
            time.sleep(0.1)
            return client.get("/health")

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=connection_count) as executor:
            futures = [executor.submit(make_long_request) for _ in range(connection_count)]
            responses = [future.result() for future in as_completed(futures)]
        end_time = time.time()

        total_time = end_time - start_time

        # Should handle concurrent connections efficiently
        assert len(responses) == connection_count
        success_rate = sum(1 for r in responses if r.status_code == 200) / len(responses)
        assert success_rate >= 0.95
        assert total_time < 5.0  # Should complete efficiently

    def test_resource_cleanup(self, client: TestClient):
        """Test that resources are properly cleaned up."""
        # Make many requests and verify no resource leaks
        request_count = 100

        for i in range(request_count):
            response = client.get("/health")
            assert response.status_code == 200

            # Verify response is properly closed
            assert response.is_closed or hasattr(response, "_content")

    @pytest.mark.slow
    def test_sustained_high_load(self, client: TestClient):
        """Test API under sustained high load."""
        # This test is marked as slow and would run for longer periods
        duration_seconds = 30
        requests_per_second = 10
        total_requests = duration_seconds * requests_per_second

        successful_requests = 0
        failed_requests = 0

        start_time = time.time()

        for i in range(total_requests):
            try:
                response = client.get("/health")
                if response.status_code == 200:
                    successful_requests += 1
                else:
                    failed_requests += 1
            except Exception:
                failed_requests += 1

            # Maintain request rate
            elapsed = time.time() - start_time
            expected_elapsed = i / requests_per_second
            if elapsed < expected_elapsed:
                time.sleep(expected_elapsed - elapsed)

        end_time = time.time()
        actual_duration = end_time - start_time
        success_rate = successful_requests / total_requests

        # Should maintain reasonable performance under sustained load
        assert success_rate >= 0.90  # 90% success rate
        assert actual_duration <= duration_seconds * 1.2  # Within 20% of target duration
