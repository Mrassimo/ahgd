"""
WebSocket connection manager for the AHGD Data Quality API.

This module provides real-time WebSocket connection management for live dashboard
updates, pipeline monitoring, and metrics streaming with <100ms update latency.
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Optional

from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi.websockets import WebSocketState

from ...utils.config import get_config
from ...utils.logging import get_logger
from ..exceptions import ValidationException
from ..models.requests import SubscriptionRequest
from ..models.responses import SubscriptionResponse
from ..models.responses import WebSocketResponse

logger = get_logger(__name__)


class ConnectionState(str, Enum):
    """WebSocket connection states."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class SubscriptionType(str, Enum):
    """Supported subscription types."""

    QUALITY_METRICS = "quality_metrics"
    VALIDATION_RESULTS = "validation_results"
    PIPELINE_STATUS = "pipeline_status"
    SYSTEM_HEALTH = "system_health"
    ALERTS = "alerts"
    ALL = "all"


class WebSocketConnection:
    """Individual WebSocket connection wrapper."""

    def __init__(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None):
        self.websocket = websocket
        self.connection_id = connection_id
        self.user_id = user_id or "anonymous"
        self.state = ConnectionState.CONNECTING
        self.connected_at = datetime.now()
        self.last_ping = datetime.now()
        self.subscriptions: set[str] = set()
        self.message_count = 0
        self.error_count = 0

        # Connection metadata
        self.metadata = {"user_agent": None, "client_ip": None, "api_version": "v1"}

    async def send_message(self, message: dict[str, Any]) -> bool:
        """
        Send message to WebSocket connection.

        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            if self.websocket.client_state != WebSocketState.CONNECTED:
                logger.warning(
                    "Cannot send message to disconnected WebSocket",
                    connection_id=self.connection_id,
                )
                return False

            # Add message metadata
            message_with_meta = {
                **message,
                "connection_id": self.connection_id,
                "timestamp": datetime.now().isoformat(),
                "sequence": self.message_count,
            }

            await self.websocket.send_text(json.dumps(message_with_meta))
            self.message_count += 1

            return True

        except WebSocketDisconnect:
            logger.debug("WebSocket disconnected during send", connection_id=self.connection_id)
            self.state = ConnectionState.DISCONNECTED
            return False
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}", connection_id=self.connection_id)
            self.error_count += 1
            return False

    async def send_error(self, error_code: str, error_message: str) -> bool:
        """Send error message to client."""
        error_msg = WebSocketResponse(
            message_type="error", data={"error_code": error_code, "error_message": error_message}
        )
        return await self.send_message(error_msg.model_dump())

    async def ping(self) -> bool:
        """Send ping to keep connection alive."""
        ping_msg = WebSocketResponse(
            message_type="ping", data={"server_time": datetime.now().isoformat()}
        )

        if await self.send_message(ping_msg.model_dump()):
            self.last_ping = datetime.now()
            return True
        return False

    def is_healthy(self) -> bool:
        """Check if connection is healthy."""
        # Connection is unhealthy if:
        # 1. Too many errors
        # 2. No ping response for too long
        # 3. WebSocket state is not connected

        if self.error_count > 10:
            return False

        if (datetime.now() - self.last_ping).total_seconds() > 300:  # 5 minutes
            return False

        if self.websocket.client_state != WebSocketState.CONNECTED:
            return False

        return True

    def add_subscription(self, subscription_id: str) -> None:
        """Add subscription to connection."""
        self.subscriptions.add(subscription_id)

    def remove_subscription(self, subscription_id: str) -> None:
        """Remove subscription from connection."""
        self.subscriptions.discard(subscription_id)


class Subscription:
    """WebSocket subscription configuration."""

    def __init__(
        self,
        subscription_id: str,
        connection_id: str,
        subscription_type: SubscriptionType,
        filters: Optional[dict[str, Any]] = None,
        update_frequency: int = 5,
    ):
        self.subscription_id = subscription_id
        self.connection_id = connection_id
        self.subscription_type = subscription_type
        self.filters = filters or {}
        self.update_frequency = max(1, min(60, update_frequency))  # 1-60 seconds
        self.created_at = datetime.now()
        self.last_update = None
        self.message_count = 0
        self.active = True

    def should_update(self) -> bool:
        """Check if subscription is due for update."""
        if not self.active:
            return False

        if self.last_update is None:
            return True

        elapsed = (datetime.now() - self.last_update).total_seconds()
        return elapsed >= self.update_frequency

    def matches_data(self, data: dict[str, Any]) -> bool:
        """Check if data matches subscription filters."""
        if not self.filters:
            return True

        # Apply basic filtering logic
        for filter_key, filter_value in self.filters.items():
            data_value = data.get(filter_key)

            if isinstance(filter_value, list):
                if data_value not in filter_value:
                    return False
            elif isinstance(filter_value, dict):
                # Range filtering
                if "min" in filter_value and data_value < filter_value["min"]:
                    return False
                if "max" in filter_value and data_value > filter_value["max"]:
                    return False
            else:
                if data_value != filter_value:
                    return False

        return True


class ConnectionManager:
    """
    WebSocket connection manager for real-time communications.

    Manages WebSocket connections, subscriptions, and provides
    real-time updates with <100ms latency for dashboard functionality.
    """

    def __init__(self):
        """Initialise the connection manager."""
        self.config = get_config("websocket", {})
        self.max_connections = self.config.get("max_connections", 1000)
        self.ping_interval = self.config.get("ping_interval", 30)  # seconds
        self.cleanup_interval = self.config.get("cleanup_interval", 60)  # seconds

        # Connection storage
        self.connections: dict[str, WebSocketConnection] = {}
        self.subscriptions: dict[str, Subscription] = {}
        self.subscription_by_type: dict[SubscriptionType, set[str]] = {
            sub_type: set() for sub_type in SubscriptionType
        }

        # Background tasks
        self._background_tasks: set[asyncio.Task] = set()
        self._running = False

        # Statistics
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "total_subscriptions": 0,
            "messages_sent": 0,
            "errors_count": 0,
        }

        logger.info("WebSocket connection manager initialised")

    async def start(self) -> None:
        """Start the connection manager background tasks."""
        if self._running:
            return

        self._running = True

        # Start background tasks
        ping_task = asyncio.create_task(self._ping_connections_task())
        cleanup_task = asyncio.create_task(self._cleanup_connections_task())

        self._background_tasks.add(ping_task)
        self._background_tasks.add(cleanup_task)

        logger.info("Connection manager background tasks started")

    async def stop(self) -> None:
        """Stop the connection manager and close all connections."""
        logger.info("Stopping connection manager")

        self._running = False

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        # Close all connections
        await self._close_all_connections()

        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.info("Connection manager stopped")

    async def connect(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Establish new WebSocket connection.

        Args:
            websocket: FastAPI WebSocket instance
            user_id: Optional user identifier
            metadata: Optional connection metadata

        Returns:
            Connection ID
        """

        # Check connection limits
        if len(self.connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Maximum connections exceeded")
            raise Exception("Maximum WebSocket connections exceeded")

        # Accept connection
        await websocket.accept()

        # Create connection
        connection_id = str(uuid.uuid4())
        connection = WebSocketConnection(websocket, connection_id, user_id)
        connection.state = ConnectionState.CONNECTED

        # Add metadata
        if metadata:
            connection.metadata.update(metadata)

        # Store connection
        self.connections[connection_id] = connection

        # Update statistics
        self.stats["total_connections"] += 1
        self.stats["active_connections"] = len(self.connections)

        logger.info(
            "WebSocket connection established",
            connection_id=connection_id,
            user_id=user_id,
            total_connections=len(self.connections),
        )

        # Send welcome message
        welcome_msg = WebSocketResponse(
            message_type="welcome",
            data={
                "connection_id": connection_id,
                "server_time": datetime.now().isoformat(),
                "supported_subscriptions": [t.value for t in SubscriptionType],
            },
        )
        await connection.send_message(welcome_msg.model_dump())

        return connection_id

    async def disconnect(self, connection_id: str) -> None:
        """
        Disconnect WebSocket connection.

        Args:
            connection_id: Connection identifier
        """

        if connection_id not in self.connections:
            return

        connection = self.connections[connection_id]

        try:
            # Remove all subscriptions for this connection
            subscriptions_to_remove = [
                sub_id
                for sub_id, subscription in self.subscriptions.items()
                if subscription.connection_id == connection_id
            ]

            for sub_id in subscriptions_to_remove:
                await self.unsubscribe(connection_id, sub_id)

            # Close WebSocket if still connected
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                await connection.websocket.close()

            connection.state = ConnectionState.DISCONNECTED

        except Exception as e:
            logger.warning(f"Error during disconnect cleanup: {e}")
        finally:
            # Remove from connections
            del self.connections[connection_id]
            self.stats["active_connections"] = len(self.connections)

            logger.info(
                "WebSocket connection closed",
                connection_id=connection_id,
                total_connections=len(self.connections),
            )

    async def subscribe(
        self, connection_id: str, request: SubscriptionRequest
    ) -> SubscriptionResponse:
        """
        Create new subscription for connection.

        Args:
            connection_id: Connection identifier
            request: Subscription request parameters

        Returns:
            Subscription response with details
        """

        if connection_id not in self.connections:
            raise ValidationException("Connection not found", field="connection_id")

        connection = self.connections[connection_id]

        # Validate subscription type
        try:
            subscription_type = SubscriptionType(request.subscription_type)
        except ValueError:
            raise ValidationException(
                f"Invalid subscription type: {request.subscription_type}", field="subscription_type"
            )

        # Create subscription
        subscription_id = str(uuid.uuid4())
        subscription = Subscription(
            subscription_id=subscription_id,
            connection_id=connection_id,
            subscription_type=subscription_type,
            filters=request.filters,
            update_frequency=request.update_frequency,
        )

        # Store subscription
        self.subscriptions[subscription_id] = subscription
        self.subscription_by_type[subscription_type].add(subscription_id)

        # Add to connection
        connection.add_subscription(subscription_id)

        # Update statistics
        self.stats["total_subscriptions"] = len(self.subscriptions)

        logger.info(
            "WebSocket subscription created",
            connection_id=connection_id,
            subscription_id=subscription_id,
            subscription_type=subscription_type.value,
        )

        return SubscriptionResponse(
            subscription_id=subscription_id,
            subscription_type=request.subscription_type,
            filters_applied=request.filters or {},
            update_frequency=request.update_frequency,
        )

    async def unsubscribe(self, connection_id: str, subscription_id: str) -> bool:
        """
        Remove subscription.

        Args:
            connection_id: Connection identifier
            subscription_id: Subscription identifier

        Returns:
            True if subscription was removed
        """

        if subscription_id not in self.subscriptions:
            return False

        subscription = self.subscriptions[subscription_id]

        # Verify ownership
        if subscription.connection_id != connection_id:
            return False

        # Remove from type index
        self.subscription_by_type[subscription.subscription_type].discard(subscription_id)

        # Remove from connection
        if connection_id in self.connections:
            self.connections[connection_id].remove_subscription(subscription_id)

        # Remove subscription
        del self.subscriptions[subscription_id]

        # Update statistics
        self.stats["total_subscriptions"] = len(self.subscriptions)

        logger.info(
            "WebSocket subscription removed",
            connection_id=connection_id,
            subscription_id=subscription_id,
        )

        return True

    async def broadcast(
        self,
        message_type: str,
        data: dict[str, Any],
        subscription_type: Optional[SubscriptionType] = None,
    ) -> int:
        """
        Broadcast message to all relevant connections.

        Args:
            message_type: Type of message
            data: Message data
            subscription_type: Optional subscription type filter

        Returns:
            Number of connections message was sent to
        """

        if not self.connections:
            return 0

        message = WebSocketResponse(message_type=message_type, data=data)

        sent_count = 0
        target_subscriptions = set()

        # Get target subscriptions
        if subscription_type:
            target_subscriptions = self.subscription_by_type.get(subscription_type, set())
        else:
            # Broadcast to all subscriptions
            target_subscriptions = set(self.subscriptions.keys())

        # Send to relevant connections
        for sub_id in target_subscriptions:
            subscription = self.subscriptions.get(sub_id)
            if not subscription or not subscription.active:
                continue

            # Check if data matches subscription filters
            if not subscription.matches_data(data):
                continue

            # Get connection
            connection = self.connections.get(subscription.connection_id)
            if not connection or not connection.is_healthy():
                continue

            # Send message
            if await connection.send_message(message.model_dump()):
                sent_count += 1
                subscription.message_count += 1
                subscription.last_update = datetime.now()

        # Update statistics
        self.stats["messages_sent"] += sent_count

        if sent_count > 0:
            logger.debug(
                "Broadcasted WebSocket message",
                message_type=message_type,
                sent_to=sent_count,
                subscription_type=subscription_type.value if subscription_type else "all",
            )

        return sent_count

    async def send_to_connection(
        self, connection_id: str, message_type: str, data: dict[str, Any]
    ) -> bool:
        """
        Send message to specific connection.

        Args:
            connection_id: Target connection ID
            message_type: Message type
            data: Message data

        Returns:
            True if message was sent successfully
        """

        connection = self.connections.get(connection_id)
        if not connection or not connection.is_healthy():
            return False

        message = WebSocketResponse(message_type=message_type, data=data)

        success = await connection.send_message(message.model_dump())
        if success:
            self.stats["messages_sent"] += 1

        return success

    def get_connection_info(self, connection_id: str) -> Optional[dict[str, Any]]:
        """Get connection information."""

        connection = self.connections.get(connection_id)
        if not connection:
            return None

        return {
            "connection_id": connection_id,
            "user_id": connection.user_id,
            "state": connection.state.value,
            "connected_at": connection.connected_at,
            "last_ping": connection.last_ping,
            "message_count": connection.message_count,
            "error_count": connection.error_count,
            "subscriptions": list(connection.subscriptions),
            "metadata": connection.metadata,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Get connection manager statistics."""

        active_subscriptions_by_type = {
            sub_type.value: len(sub_ids) for sub_type, sub_ids in self.subscription_by_type.items()
        }

        return {
            **self.stats,
            "max_connections": self.max_connections,
            "subscriptions_by_type": active_subscriptions_by_type,
            "average_subscriptions_per_connection": (
                len(self.subscriptions) / max(1, len(self.connections))
            ),
        }

    async def _ping_connections_task(self) -> None:
        """Background task to ping connections and maintain health."""

        while self._running:
            try:
                await asyncio.sleep(self.ping_interval)

                if not self.connections:
                    continue

                # Ping all connections
                ping_tasks = []
                for connection in self.connections.values():
                    if connection.is_healthy():
                        ping_tasks.append(connection.ping())

                if ping_tasks:
                    results = await asyncio.gather(*ping_tasks, return_exceptions=True)
                    failed_pings = sum(
                        1 for result in results if result is False or isinstance(result, Exception)
                    )

                    if failed_pings > 0:
                        logger.debug(f"Failed to ping {failed_pings} connections")

            except Exception as e:
                logger.error(f"Error in ping connections task: {e}")

    async def _cleanup_connections_task(self) -> None:
        """Background task to cleanup unhealthy connections."""

        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)

                # Find unhealthy connections
                unhealthy_connections = [
                    conn_id for conn_id, conn in self.connections.items() if not conn.is_healthy()
                ]

                # Disconnect unhealthy connections
                for conn_id in unhealthy_connections:
                    logger.info("Cleaning up unhealthy connection", connection_id=conn_id)
                    await self.disconnect(conn_id)

                # Clean up inactive subscriptions
                inactive_subscriptions = [
                    sub_id
                    for sub_id, sub in self.subscriptions.items()
                    if sub.connection_id not in self.connections
                ]

                for sub_id in inactive_subscriptions:
                    subscription = self.subscriptions[sub_id]
                    self.subscription_by_type[subscription.subscription_type].discard(sub_id)
                    del self.subscriptions[sub_id]

                if inactive_subscriptions:
                    logger.info(f"Cleaned up {len(inactive_subscriptions)} orphaned subscriptions")
                    self.stats["total_subscriptions"] = len(self.subscriptions)

            except Exception as e:
                logger.error(f"Error in cleanup connections task: {e}")

    async def _close_all_connections(self) -> None:
        """Close all active connections."""

        if not self.connections:
            return

        logger.info(f"Closing {len(self.connections)} WebSocket connections")

        close_tasks = []
        for connection in self.connections.values():
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                close_tasks.append(connection.websocket.close())

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self.connections.clear()
        self.subscriptions.clear()
        for sub_set in self.subscription_by_type.values():
            sub_set.clear()


# Global connection manager instance
connection_manager = ConnectionManager()


async def get_connection_manager() -> ConnectionManager:
    """Get connection manager instance."""
    return connection_manager


@asynccontextmanager
async def websocket_lifespan():
    """WebSocket connection manager lifespan context."""
    await connection_manager.start()
    try:
        yield connection_manager
    finally:
        await connection_manager.stop()
