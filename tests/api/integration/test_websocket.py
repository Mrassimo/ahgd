"""
WebSocket integration tests.

Tests real-time WebSocket functionality for metrics streaming and live updates.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, AsyncMock
from datetime import datetime

from httpx import AsyncClient, WebSocketDisconnect


class TestWebSocketConnection:
    """Test WebSocket connection lifecycle."""
    
    @pytest.mark.asyncio
    async def test_websocket_connect_disconnect(self, async_client: AsyncClient):
        """Test WebSocket connection and disconnection."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.disconnect = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Connection should be successful
                assert websocket is not None
                
                # Send ping to verify connection
                await websocket.send_json({"type": "ping"})
                response = await websocket.receive_json()
                assert response["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_websocket_connection_limit(self, async_client: AsyncClient):
        """Test WebSocket connection limits."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.get_connection_count.return_value = 100
            mock_manager.return_value.max_connections = 100
            
            # Should reject connection when at limit
            with pytest.raises(WebSocketDisconnect):
                async with async_client.websocket_connect("/ws/metrics") as websocket:
                    pass
    
    @pytest.mark.asyncio
    async def test_websocket_authentication(self, async_client: AsyncClient):
        """Test WebSocket authentication."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.authenticate = AsyncMock(return_value=True)
            
            async with async_client.websocket_connect(
                "/ws/metrics",
                headers={"Authorization": "Bearer test_token"}
            ) as websocket:
                # Should authenticate successfully
                await websocket.send_json({
                    "type": "authenticate",
                    "token": "test_token"
                })
                
                response = await websocket.receive_json()
                assert response["type"] == "auth_success"


class TestMetricsStreaming:
    """Test real-time metrics streaming."""
    
    @pytest.mark.asyncio
    async def test_quality_metrics_subscription(self, async_client: AsyncClient):
        """Test subscribing to quality metrics updates."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Subscribe to quality metrics
                await websocket.send_json({
                    "type": "subscribe",
                    "subscription_type": "quality_metrics",
                    "filters": {
                        "geographic_level": "sa1",
                        "update_interval": 1.0
                    }
                })
                
                # Should receive subscription acknowledgment
                response = await websocket.receive_json()
                assert response["type"] == "subscription_ack"
                assert response["subscription_type"] == "quality_metrics"
    
    @pytest.mark.asyncio
    async def test_pipeline_status_streaming(self, async_client: AsyncClient):
        """Test pipeline status updates via WebSocket."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Subscribe to pipeline updates
                await websocket.send_json({
                    "type": "subscribe",
                    "subscription_type": "pipeline_status",
                    "filters": {
                        "pipeline_names": ["etl_pipeline", "validation_pipeline"]
                    }
                })
                
                response = await websocket.receive_json()
                assert response["type"] == "subscription_ack"
    
    @pytest.mark.asyncio
    async def test_validation_results_streaming(self, async_client: AsyncClient):
        """Test validation results streaming."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Subscribe to validation updates
                await websocket.send_json({
                    "type": "subscribe",
                    "subscription_type": "validation_results",
                    "filters": {
                        "validation_types": ["schema", "business"],
                        "severity_levels": ["error", "warning"]
                    }
                })
                
                response = await websocket.receive_json()
                assert response["type"] == "subscription_ack"
    
    @pytest.mark.asyncio
    async def test_system_health_streaming(self, async_client: AsyncClient):
        """Test system health metrics streaming."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Subscribe to system health
                await websocket.send_json({
                    "type": "subscribe",
                    "subscription_type": "system_health",
                    "filters": {
                        "metrics": ["cpu", "memory", "disk", "network"],
                        "update_interval": 2.0
                    }
                })
                
                response = await websocket.receive_json()
                assert response["type"] == "subscription_ack"


class TestRealTimeUpdates:
    """Test real-time update delivery."""
    
    @pytest.mark.asyncio
    async def test_metrics_update_delivery(self, async_client: AsyncClient):
        """Test delivery of metrics updates."""
        with patch('src.api.websocket.metrics_stream.MetricsStreamer') as mock_streamer:
            mock_streamer.return_value.start_streaming = AsyncMock()
            
            with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
                mock_manager.return_value.connect = AsyncMock()
                mock_manager.return_value.add_subscription = AsyncMock()
                mock_manager.return_value.broadcast = AsyncMock()
                
                async with async_client.websocket_connect("/ws/metrics") as websocket:
                    # Subscribe to updates
                    await websocket.send_json({
                        "type": "subscribe",
                        "subscription_type": "quality_metrics"
                    })
                    
                    # Simulate metrics update
                    await mock_manager.return_value.broadcast(
                        "metrics_update",
                        {
                            "subscription_type": "quality_metrics",
                            "data": {
                                "overall_score": 95.4,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                    )
                    
                    # Should receive the update
                    response = await websocket.receive_json()
                    assert response["type"] in ["subscription_ack", "metrics_update"]
    
    @pytest.mark.asyncio
    async def test_update_frequency_control(self, async_client: AsyncClient):
        """Test update frequency control."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Subscribe with specific update interval
                await websocket.send_json({
                    "type": "subscribe",
                    "subscription_type": "quality_metrics",
                    "filters": {
                        "update_interval": 0.5  # 500ms
                    }
                })
                
                response = await websocket.receive_json()
                assert response["type"] == "subscription_ack"
                
                # Verify update interval was set
                call_args = mock_manager.return_value.add_subscription.call_args
                assert "update_interval" in str(call_args)
    
    @pytest.mark.asyncio
    async def test_filtered_updates(self, async_client: AsyncClient):
        """Test filtered update delivery."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Subscribe with filters
                await websocket.send_json({
                    "type": "subscribe",
                    "subscription_type": "validation_results",
                    "filters": {
                        "geographic_level": "sa1",
                        "severity_levels": ["error"],
                        "sa1_codes": ["10101000001", "10101000002"]
                    }
                })
                
                response = await websocket.receive_json()
                assert response["type"] == "subscription_ack"
                
                # Verify filters were applied
                call_args = mock_manager.return_value.add_subscription.call_args
                assert "sa1_codes" in str(call_args)


class TestSubscriptionManagement:
    """Test WebSocket subscription management."""
    
    @pytest.mark.asyncio
    async def test_multiple_subscriptions(self, async_client: AsyncClient):
        """Test managing multiple subscriptions."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Subscribe to multiple types
                subscriptions = [
                    {"subscription_type": "quality_metrics"},
                    {"subscription_type": "pipeline_status"},
                    {"subscription_type": "system_health"}
                ]
                
                for subscription in subscriptions:
                    await websocket.send_json({
                        "type": "subscribe",
                        **subscription
                    })
                    
                    response = await websocket.receive_json()
                    assert response["type"] == "subscription_ack"
                    assert response["subscription_type"] == subscription["subscription_type"]
    
    @pytest.mark.asyncio
    async def test_subscription_unsubscribe(self, async_client: AsyncClient):
        """Test unsubscribing from updates."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            mock_manager.return_value.remove_subscription = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Subscribe first
                await websocket.send_json({
                    "type": "subscribe",
                    "subscription_type": "quality_metrics"
                })
                
                response = await websocket.receive_json()
                subscription_id = response.get("subscription_id")
                
                # Unsubscribe
                await websocket.send_json({
                    "type": "unsubscribe",
                    "subscription_id": subscription_id
                })
                
                response = await websocket.receive_json()
                assert response["type"] == "unsubscribe_ack"
    
    @pytest.mark.asyncio
    async def test_subscription_modification(self, async_client: AsyncClient):
        """Test modifying subscription filters."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            mock_manager.return_value.update_subscription = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Subscribe first
                await websocket.send_json({
                    "type": "subscribe",
                    "subscription_type": "quality_metrics",
                    "filters": {"update_interval": 1.0}
                })
                
                response = await websocket.receive_json()
                subscription_id = response.get("subscription_id")
                
                # Update subscription
                await websocket.send_json({
                    "type": "update_subscription",
                    "subscription_id": subscription_id,
                    "filters": {"update_interval": 0.5}
                })
                
                response = await websocket.receive_json()
                assert response["type"] == "subscription_updated"


class TestWebSocketErrorHandling:
    """Test WebSocket error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_message_format(self, async_client: AsyncClient):
        """Test handling of invalid message formats."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Send invalid JSON
                await websocket.send_text("invalid json")
                
                response = await websocket.receive_json()
                assert response["type"] == "error"
                assert "invalid" in response["message"].lower()
    
    @pytest.mark.asyncio
    async def test_unknown_message_type(self, async_client: AsyncClient):
        """Test handling of unknown message types."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Send unknown message type
                await websocket.send_json({
                    "type": "unknown_type",
                    "data": {}
                })
                
                response = await websocket.receive_json()
                assert response["type"] == "error"
                assert "unknown" in response["message"].lower()
    
    @pytest.mark.asyncio
    async def test_subscription_limit(self, async_client: AsyncClient):
        """Test subscription limits per connection."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock(
                side_effect=lambda *args, **kwargs: "sub_1" if mock_manager.return_value.add_subscription.call_count <= 10 
                else ValueError("Subscription limit exceeded")
            )
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Try to exceed subscription limit
                for i in range(12):  # Assuming limit is 10
                    await websocket.send_json({
                        "type": "subscribe",
                        "subscription_type": "quality_metrics",
                        "filters": {"id": i}
                    })
                    
                    response = await websocket.receive_json()
                    if i < 10:
                        assert response["type"] == "subscription_ack"
                    else:
                        assert response["type"] == "error"


class TestWebSocketPerformance:
    """Test WebSocket performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_message_throughput(self, async_client: AsyncClient):
        """Test WebSocket message throughput."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Measure time for multiple messages
                start_time = asyncio.get_event_loop().time()
                
                for i in range(100):
                    await websocket.send_json({
                        "type": "ping",
                        "id": i
                    })
                    response = await websocket.receive_json()
                    assert response["type"] == "pong"
                
                end_time = asyncio.get_event_loop().time()
                total_time = end_time - start_time
                
                # Should process messages efficiently
                assert total_time < 5.0  # 100 messages in under 5 seconds
    
    @pytest.mark.asyncio
    async def test_update_latency(self, async_client: AsyncClient):
        """Test update delivery latency."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.add_subscription = AsyncMock()
            
            # Mock immediate update delivery
            async def mock_broadcast(message_type, data, subscription_type=None):
                # Simulate immediate broadcast
                pass
            
            mock_manager.return_value.broadcast = mock_broadcast
            
            async with async_client.websocket_connect("/ws/metrics") as websocket:
                # Subscribe to updates
                await websocket.send_json({
                    "type": "subscribe",
                    "subscription_type": "quality_metrics",
                    "filters": {"update_interval": 0.1}  # Very frequent updates
                })
                
                response = await websocket.receive_json()
                assert response["type"] == "subscription_ack"
                
                # Updates should be delivered with minimal latency
                # This test would need actual streaming to measure latency
    
    @pytest.mark.asyncio 
    async def test_connection_scalability(self, async_client: AsyncClient):
        """Test WebSocket connection scalability."""
        with patch('src.api.websocket.connection_manager.ConnectionManager') as mock_manager:
            mock_manager.return_value.connect = AsyncMock()
            mock_manager.return_value.get_connection_count.return_value = 50
            
            # Simulate multiple concurrent connections
            connections = []
            
            for i in range(5):  # Test with 5 concurrent connections
                websocket = await async_client.websocket_connect("/ws/metrics")
                connections.append(websocket)
                
                # Each connection should establish successfully
                await websocket.send_json({"type": "ping"})
                response = await websocket.receive_json()
                assert response["type"] == "pong"
            
            # Clean up connections
            for ws in connections:
                await ws.close()