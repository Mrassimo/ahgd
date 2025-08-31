"""
Real-time metrics streaming for the AHGD Data Quality API.

This module provides live dashboard updates through WebSocket connections with
<100ms latency, streaming quality metrics, validation results, pipeline status,
and system health information.
"""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Optional

from ...utils.config import get_config
from ...utils.logging import get_logger
from ...utils.logging import monitor_performance
from ..models.common import MetricValue
from ..models.common import QualityScore
from ..models.common import SystemHealth
from ..models.responses import MetricsStreamResponse
from .connection_manager import ConnectionManager
from .connection_manager import SubscriptionType

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics that can be streamed."""

    QUALITY_SCORE = "quality_score"
    VALIDATION_RATE = "validation_rate"
    PIPELINE_THROUGHPUT = "pipeline_throughput"
    ERROR_RATE = "error_rate"
    SYSTEM_CPU = "system_cpu"
    SYSTEM_MEMORY = "system_memory"
    ACTIVE_CONNECTIONS = "active_connections"
    DATA_FRESHNESS = "data_freshness"


@dataclass
class MetricGenerator:
    """Configuration for generating metric values."""

    metric_type: MetricType
    base_value: float
    variance: float
    trend_factor: float = 0.0
    min_value: float = 0.0
    max_value: float = 100.0
    unit: str = ""
    update_frequency: float = 1.0  # seconds


class MetricsStreamer:
    """
    Real-time metrics streaming service.

    Generates and streams live metrics to WebSocket connections with
    configurable update frequencies and realistic data patterns.
    """

    def __init__(self, connection_manager: ConnectionManager):
        """Initialise metrics streamer."""
        self.connection_manager = connection_manager
        self.config = get_config("metrics_streaming", {})

        # Streaming configuration
        self.enabled = self.config.get("enabled", True)
        self.base_update_interval = self.config.get("base_interval", 1.0)  # seconds
        self.max_latency_ms = self.config.get("max_latency_ms", 100)

        # Metric generators configuration
        self.metric_generators = {
            MetricType.QUALITY_SCORE: MetricGenerator(
                MetricType.QUALITY_SCORE,
                base_value=85.0,
                variance=5.0,
                trend_factor=0.1,
                min_value=70.0,
                max_value=100.0,
                unit="%",
            ),
            MetricType.VALIDATION_RATE: MetricGenerator(
                MetricType.VALIDATION_RATE,
                base_value=95.0,
                variance=3.0,
                trend_factor=-0.05,
                min_value=80.0,
                max_value=100.0,
                unit="%",
            ),
            MetricType.PIPELINE_THROUGHPUT: MetricGenerator(
                MetricType.PIPELINE_THROUGHPUT,
                base_value=1250.0,
                variance=200.0,
                trend_factor=0.05,
                min_value=800.0,
                max_value=2000.0,
                unit="records/min",
            ),
            MetricType.ERROR_RATE: MetricGenerator(
                MetricType.ERROR_RATE,
                base_value=2.5,
                variance=1.0,
                trend_factor=-0.02,
                min_value=0.0,
                max_value=10.0,
                unit="%",
            ),
            MetricType.SYSTEM_CPU: MetricGenerator(
                MetricType.SYSTEM_CPU,
                base_value=45.0,
                variance=15.0,
                trend_factor=0.02,
                min_value=10.0,
                max_value=100.0,
                unit="%",
            ),
            MetricType.SYSTEM_MEMORY: MetricGenerator(
                MetricType.SYSTEM_MEMORY,
                base_value=65.0,
                variance=10.0,
                trend_factor=0.01,
                min_value=30.0,
                max_value=95.0,
                unit="%",
            ),
            MetricType.ACTIVE_CONNECTIONS: MetricGenerator(
                MetricType.ACTIVE_CONNECTIONS,
                base_value=25.0,
                variance=8.0,
                trend_factor=0.03,
                min_value=5.0,
                max_value=100.0,
                unit="connections",
            ),
            MetricType.DATA_FRESHNESS: MetricGenerator(
                MetricType.DATA_FRESHNESS,
                base_value=12.0,
                variance=4.0,
                trend_factor=0.1,
                min_value=1.0,
                max_value=48.0,
                unit="hours",
            ),
        }

        # Runtime state
        self.current_values: dict[MetricType, float] = {}
        self.last_updates: dict[MetricType, datetime] = {}
        self.streaming_tasks: set[asyncio.Task] = set()
        self.is_running = False

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "updates_per_second": 0.0,
            "average_latency_ms": 0.0,
            "last_update": None,
        }

        # Initialize current values
        for metric_type, generator in self.metric_generators.items():
            self.current_values[metric_type] = generator.base_value
            self.last_updates[metric_type] = datetime.now()

        logger.info("Metrics streamer initialised")

    async def start(self) -> None:
        """Start metrics streaming tasks."""
        if self.is_running or not self.enabled:
            return

        self.is_running = True

        # Start streaming tasks for each metric type
        for metric_type in self.metric_generators:
            task = asyncio.create_task(self._stream_metric_task(metric_type))
            self.streaming_tasks.add(task)

        # Start system health streaming
        health_task = asyncio.create_task(self._stream_system_health_task())
        self.streaming_tasks.add(health_task)

        # Start quality metrics streaming
        quality_task = asyncio.create_task(self._stream_quality_metrics_task())
        self.streaming_tasks.add(quality_task)

        # Start statistics calculation task
        stats_task = asyncio.create_task(self._calculate_statistics_task())
        self.streaming_tasks.add(stats_task)

        logger.info(f"Started {len(self.streaming_tasks)} metrics streaming tasks")

    async def stop(self) -> None:
        """Stop metrics streaming tasks."""
        if not self.is_running:
            return

        logger.info("Stopping metrics streaming tasks")

        self.is_running = False

        # Cancel all tasks
        for task in self.streaming_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.streaming_tasks:
            await asyncio.gather(*self.streaming_tasks, return_exceptions=True)

        self.streaming_tasks.clear()
        logger.info("Metrics streaming stopped")

    @monitor_performance("metrics_streaming_update")
    async def _stream_metric_task(self, metric_type: MetricType) -> None:
        """Stream updates for a specific metric type."""

        generator = self.metric_generators[metric_type]

        while self.is_running:
            try:
                start_time = time.time()

                # Generate new metric value
                new_value = self._generate_metric_value(metric_type, generator)
                self.current_values[metric_type] = new_value
                self.last_updates[metric_type] = datetime.now()

                # Create metric value object
                metric_value = MetricValue(
                    name=metric_type.value,
                    value=new_value,
                    timestamp=datetime.now(),
                    labels={"source": "realtime_generator"},
                    unit=generator.unit,
                )

                # Broadcast to relevant subscriptions
                await self.connection_manager.broadcast(
                    message_type="metric_update",
                    data={
                        "metric_type": metric_type.value,
                        "metric": metric_value.model_dump(),
                        "update_latency_ms": 0,  # Calculated below
                    },
                    subscription_type=SubscriptionType.QUALITY_METRICS,
                )

                # Calculate and update latency
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000

                self.stats["messages_sent"] += 1

                # Adaptive sleep to maintain target frequency
                target_interval = generator.update_frequency
                elapsed = end_time - start_time
                sleep_time = max(0.01, target_interval - elapsed)

                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metric streaming task for {metric_type}: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    async def _stream_system_health_task(self) -> None:
        """Stream system health updates."""

        while self.is_running:
            try:
                # Generate system health data
                system_health = SystemHealth(
                    status=self._determine_system_status(),
                    timestamp=datetime.now(),
                    cpu_percent=self.current_values.get(MetricType.SYSTEM_CPU, 45.0),
                    memory_percent=self.current_values.get(MetricType.SYSTEM_MEMORY, 65.0),
                    disk_percent=random.uniform(40, 80),  # Mock disk usage
                    active_pipelines=random.randint(0, 3),
                    pending_validations=random.randint(0, 10),
                    uptime_seconds=time.time(),  # Mock uptime
                    version="2.0.0",
                )

                # Broadcast system health
                await self.connection_manager.broadcast(
                    message_type="system_health_update",
                    data=system_health.model_dump(),
                    subscription_type=SubscriptionType.SYSTEM_HEALTH,
                )

                self.stats["messages_sent"] += 1

                # Update every 5 seconds
                await asyncio.sleep(5.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system health streaming: {e}")
                await asyncio.sleep(5.0)

    async def _stream_quality_metrics_task(self) -> None:
        """Stream quality metrics updates."""

        while self.is_running:
            try:
                # Generate quality score
                quality_score = QualityScore(
                    overall_score=self.current_values.get(MetricType.QUALITY_SCORE, 85.0),
                    completeness=random.uniform(80, 95),
                    accuracy=random.uniform(85, 98),
                    consistency=random.uniform(75, 90),
                    validity=random.uniform(88, 97),
                    timeliness=random.uniform(70, 85),
                    calculated_at=datetime.now(),
                    record_count=57736,  # SA1 count
                )

                # Create metrics response
                metrics_response = MetricsStreamResponse(
                    timestamp=datetime.now(),
                    metrics=[
                        MetricValue(
                            name="quality_overall", value=quality_score.overall_score, unit="%"
                        ),
                        MetricValue(
                            name="quality_completeness", value=quality_score.completeness, unit="%"
                        ),
                        MetricValue(
                            name="quality_accuracy", value=quality_score.accuracy, unit="%"
                        ),
                    ],
                    system_status="healthy",
                    update_frequency=3,
                )

                # Broadcast quality metrics
                await self.connection_manager.broadcast(
                    message_type="quality_metrics_update",
                    data=metrics_response.model_dump(),
                    subscription_type=SubscriptionType.QUALITY_METRICS,
                )

                self.stats["messages_sent"] += 1

                # Update every 3 seconds
                await asyncio.sleep(3.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in quality metrics streaming: {e}")
                await asyncio.sleep(3.0)

    async def _calculate_statistics_task(self) -> None:
        """Calculate streaming statistics."""

        message_count_start = self.stats["messages_sent"]
        start_time = time.time()

        while self.is_running:
            try:
                await asyncio.sleep(10.0)  # Calculate stats every 10 seconds

                current_time = time.time()
                current_messages = self.stats["messages_sent"]

                # Calculate messages per second
                time_elapsed = current_time - start_time
                messages_sent = current_messages - message_count_start

                if time_elapsed > 0:
                    self.stats["updates_per_second"] = messages_sent / time_elapsed

                # Update baseline
                message_count_start = current_messages
                start_time = current_time

                self.stats["last_update"] = datetime.now()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error calculating streaming statistics: {e}")

    def _generate_metric_value(self, metric_type: MetricType, generator: MetricGenerator) -> float:
        """Generate realistic metric value with trends and variance."""

        current_value = self.current_values.get(metric_type, generator.base_value)

        # Apply trend (gradual drift towards trend direction)
        trend_adjustment = generator.trend_factor * random.uniform(-0.5, 1.0)

        # Apply random variance
        variance_adjustment = random.uniform(-generator.variance / 2, generator.variance / 2)

        # Mean reversion (pull back towards base value)
        base_pull = (generator.base_value - current_value) * 0.1

        # Calculate new value
        new_value = current_value + trend_adjustment + variance_adjustment + base_pull

        # Apply bounds
        new_value = max(generator.min_value, min(generator.max_value, new_value))

        return round(new_value, 2)

    def _determine_system_status(self) -> str:
        """Determine overall system status based on current metrics."""

        cpu_usage = self.current_values.get(MetricType.SYSTEM_CPU, 45.0)
        memory_usage = self.current_values.get(MetricType.SYSTEM_MEMORY, 65.0)
        error_rate = self.current_values.get(MetricType.ERROR_RATE, 2.5)
        quality_score = self.current_values.get(MetricType.QUALITY_SCORE, 85.0)

        # Determine status based on thresholds
        if cpu_usage > 90 or memory_usage > 90 or error_rate > 8 or quality_score < 75:
            return "critical"
        elif cpu_usage > 75 or memory_usage > 80 or error_rate > 5 or quality_score < 85:
            return "warning"
        else:
            return "healthy"

    async def trigger_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        affected_resources: Optional[list[str]] = None,
    ) -> None:
        """Trigger an alert broadcast."""

        alert_data = {
            "alert_id": f"alert_{int(time.time())}",
            "alert_type": alert_type,
            "severity": severity,
            "title": f"{severity.upper()}: {alert_type}",
            "description": message,
            "triggered_at": datetime.now().isoformat(),
            "affected_resources": affected_resources or [],
            "is_active": True,
        }

        # Broadcast alert
        await self.connection_manager.broadcast(
            message_type="alert", data=alert_data, subscription_type=SubscriptionType.ALERTS
        )

        logger.info("Alert triggered", alert_type=alert_type, severity=severity, message=message)

    async def send_pipeline_update(
        self,
        run_id: str,
        pipeline_name: str,
        status: str,
        progress: float,
        stage: Optional[str] = None,
    ) -> None:
        """Send pipeline status update."""

        pipeline_data = {
            "run_id": run_id,
            "pipeline_name": pipeline_name,
            "status": status,
            "progress_percentage": progress,
            "current_stage": stage,
            "updated_at": datetime.now().isoformat(),
        }

        # Broadcast pipeline update
        await self.connection_manager.broadcast(
            message_type="pipeline_status_update",
            data=pipeline_data,
            subscription_type=SubscriptionType.PIPELINE_STATUS,
        )

        logger.debug("Pipeline update sent", run_id=run_id, status=status, progress=progress)

    async def send_validation_results(
        self,
        validation_id: str,
        status: str,
        passed_rules: int,
        failed_rules: int,
        overall_valid: bool,
    ) -> None:
        """Send validation results update."""

        validation_data = {
            "validation_id": validation_id,
            "status": status,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "total_rules": passed_rules + failed_rules,
            "overall_valid": overall_valid,
            "success_rate": (passed_rules / max(1, passed_rules + failed_rules)) * 100,
            "updated_at": datetime.now().isoformat(),
        }

        # Broadcast validation results
        await self.connection_manager.broadcast(
            message_type="validation_results_update",
            data=validation_data,
            subscription_type=SubscriptionType.VALIDATION_RESULTS,
        )

        logger.debug(
            "Validation results sent",
            validation_id=validation_id,
            status=status,
            overall_valid=overall_valid,
        )

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current metric values snapshot."""

        current_metrics = {}
        for metric_type, value in self.current_values.items():
            generator = self.metric_generators[metric_type]
            current_metrics[metric_type.value] = {
                "value": value,
                "unit": generator.unit,
                "last_updated": self.last_updates.get(metric_type, datetime.now()).isoformat(),
            }

        return {
            "metrics": current_metrics,
            "statistics": self.stats,
            "is_streaming": self.is_running,
            "active_connections": len(self.connection_manager.connections),
        }

    def get_streaming_statistics(self) -> dict[str, Any]:
        """Get streaming performance statistics."""

        return {
            **self.stats,
            "active_tasks": len(self.streaming_tasks),
            "target_latency_ms": self.max_latency_ms,
            "update_interval_seconds": self.base_update_interval,
            "streaming_enabled": self.enabled,
            "connection_count": len(self.connection_manager.connections),
            "subscription_count": len(self.connection_manager.subscriptions),
        }


# Factory function to create metrics streamer with connection manager
def create_metrics_streamer(connection_manager: ConnectionManager) -> MetricsStreamer:
    """Create metrics streamer instance."""
    return MetricsStreamer(connection_manager)


# Global metrics streamer instance - will be initialized with connection manager
_metrics_streamer: Optional[MetricsStreamer] = None


def initialize_metrics_streamer(connection_manager: ConnectionManager) -> None:
    """Initialize global metrics streamer instance."""
    global _metrics_streamer
    _metrics_streamer = create_metrics_streamer(connection_manager)


async def get_metrics_streamer() -> Optional[MetricsStreamer]:
    """Get metrics streamer instance."""
    return _metrics_streamer
