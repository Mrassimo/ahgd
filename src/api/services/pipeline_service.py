"""
Pipeline management service for the AHGD Data Quality API.

This service integrates with the existing AHGD ETL pipeline infrastructure
to provide pipeline execution, monitoring, and management capabilities through the API.
"""

import asyncio
import uuid
from datetime import datetime
from datetime import timedelta
from enum import Enum
from typing import Any
from typing import Optional

from ...utils.config import get_config
from ...utils.interfaces import AHGDException
from ...utils.logging import get_logger
from ...utils.logging import monitor_performance
from ...utils.logging import track_lineage
from ..exceptions import PipelineException
from ..exceptions import ResourceNotFoundException
from ..exceptions import ServiceUnavailableException
from ..models.common import PipelineRun
from ..models.common import PipelineStageResult
from ..models.common import StatusEnum
from ..models.requests import PaginationRequest
from ..models.requests import PipelineRunRequest
from ..models.responses import PipelineRunResponse
from ..models.responses import StatusResponse

logger = get_logger(__name__)


class PipelineStatus(str, Enum):
    """Extended pipeline status for API operations."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class PipelineService:
    """
    Service for pipeline management and execution operations.

    Integrates with the existing AHGD ETL infrastructure while providing
    API-specific functionality for pipeline orchestration and monitoring.
    """

    def __init__(self):
        """Initialise the pipeline management service."""
        self.config = get_config("pipeline_service", {})
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes default
        self.max_concurrent_pipelines = self.config.get("max_concurrent", 3)

        # Pipeline configurations
        self.available_pipelines = {
            "master_etl": {
                "name": "Master ETL Pipeline",
                "description": "Complete data extraction, transformation, and loading pipeline",
                "stages": ["extract", "transform", "validate", "load"],
                "estimated_duration": 3600,  # seconds
                "max_parallel": False,
            },
            "validation_only": {
                "name": "Validation Pipeline",
                "description": "Data quality validation without processing",
                "stages": ["validate"],
                "estimated_duration": 600,
                "max_parallel": True,
            },
            "extract_transform": {
                "name": "Extract & Transform Pipeline",
                "description": "Data extraction and transformation only",
                "stages": ["extract", "transform"],
                "estimated_duration": 1800,
                "max_parallel": False,
            },
            "quality_metrics": {
                "name": "Quality Metrics Pipeline",
                "description": "Calculate comprehensive quality metrics",
                "stages": ["validate", "analyse"],
                "estimated_duration": 900,
                "max_parallel": True,
            },
        }

        # Active pipeline runs tracking
        self._active_runs = {}
        self._run_history = []
        self._max_history = 1000

        logger.info("Pipeline service initialised")

    @monitor_performance("pipeline_execution")
    async def execute_pipeline(
        self, request: PipelineRunRequest, cache_manager=None
    ) -> PipelineRunResponse:
        """
        Execute a pipeline based on the request parameters.

        Args:
            request: Pipeline execution request
            cache_manager: Optional cache manager

        Returns:
            Pipeline run response with execution details
        """

        try:
            logger.info(
                "Starting pipeline execution",
                pipeline_name=request.pipeline_name,
                stage=request.stage,
                force_rerun=request.force_rerun,
            )

            # Validate pipeline exists
            if request.pipeline_name not in self.available_pipelines:
                raise ResourceNotFoundException("pipeline", request.pipeline_name)

            pipeline_config = self.available_pipelines[request.pipeline_name]

            # Check if recent successful run exists and force_rerun is False
            if not request.force_rerun:
                recent_run = await self._check_recent_successful_run(request.pipeline_name)
                if recent_run:
                    logger.info(
                        "Recent successful run found, returning existing results",
                        run_id=recent_run["run_id"],
                    )
                    return await self._get_pipeline_run_response(recent_run["run_id"])

            # Check concurrent pipeline limits
            await self._check_concurrency_limits(request.pipeline_name, pipeline_config)

            # Create new pipeline run
            pipeline_run = await self._create_pipeline_run(request, pipeline_config)

            # Start pipeline execution (async)
            asyncio.create_task(
                self._execute_pipeline_async(pipeline_run, request, pipeline_config)
            )

            # Build initial response
            response = PipelineRunResponse(
                pipeline_run=pipeline_run,
                stage_results=[],
                logs_url=f"/api/v1/pipelines/{pipeline_run.run_id}/logs",
                artifacts_url=f"/api/v1/pipelines/{pipeline_run.run_id}/artifacts",
                next_actions=[
                    "Monitor pipeline progress via WebSocket",
                    "Check logs for detailed execution information",
                ],
            )

            logger.info(
                "Pipeline execution initiated",
                run_id=pipeline_run.run_id,
                estimated_completion=response.estimated_completion,
            )

            return response

        except Exception as e:
            logger.error(f"Failed to execute pipeline: {e}")
            if isinstance(e, (AHGDException, ResourceNotFoundException)):
                raise
            raise ServiceUnavailableException(
                "pipeline_service", f"Pipeline execution failed: {e!s}"
            )

    @monitor_performance("pipeline_status_check")
    async def get_pipeline_status(self, run_id: str, cache_manager=None) -> StatusResponse:
        """
        Get the current status of a pipeline run.

        Args:
            run_id: Pipeline run identifier
            cache_manager: Optional cache manager

        Returns:
            Current pipeline status
        """

        try:
            logger.debug("Retrieving pipeline status", run_id=run_id)

            # Check active runs first
            if run_id in self._active_runs:
                run_info = self._active_runs[run_id]
                pipeline_run = run_info["pipeline_run"]

                # Calculate progress
                progress = self._calculate_progress(pipeline_run)

                # Estimate completion
                estimated_completion = None
                if pipeline_run.status in [StatusEnum.PENDING, StatusEnum.IN_PROGRESS]:
                    estimated_completion = self._estimate_completion_time(pipeline_run)

                return StatusResponse(
                    operation_id=run_id,
                    status=pipeline_run.status.value,
                    progress_percentage=progress,
                    current_step=self._get_current_step(pipeline_run),
                    estimated_completion=estimated_completion,
                    result_url=f"/api/v1/pipelines/{run_id}"
                    if pipeline_run.status == StatusEnum.COMPLETED
                    else None,
                    error_message=pipeline_run.error_message,
                )

            # Check historical runs
            historical_run = self._find_historical_run(run_id)
            if historical_run:
                return StatusResponse(
                    operation_id=run_id,
                    status=historical_run["status"],
                    progress_percentage=100.0 if historical_run["status"] == "completed" else 0.0,
                    current_step="Completed"
                    if historical_run["status"] == "completed"
                    else "Failed",
                    estimated_completion=None,
                    result_url=f"/api/v1/pipelines/{run_id}"
                    if historical_run["status"] == "completed"
                    else None,
                    error_message=historical_run.get("error_message"),
                )

            # Run not found
            raise ResourceNotFoundException("pipeline_run", run_id)

        except Exception as e:
            logger.error(f"Failed to get pipeline status: {e}")
            if isinstance(e, ResourceNotFoundException):
                raise
            raise ServiceUnavailableException("pipeline_service", f"Status retrieval failed: {e!s}")

    @monitor_performance("pipeline_listing")
    async def list_pipeline_runs(
        self,
        pagination: PaginationRequest,
        status_filter: Optional[str] = None,
        pipeline_name_filter: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        List pipeline runs with filtering and pagination.

        Args:
            pagination: Pagination parameters
            status_filter: Optional status filter
            pipeline_name_filter: Optional pipeline name filter

        Returns:
            Paginated list of pipeline runs
        """

        try:
            logger.debug(
                "Listing pipeline runs",
                status_filter=status_filter,
                pipeline_filter=pipeline_name_filter,
            )

            # Combine active and historical runs
            all_runs = []

            # Add active runs
            for run_id, run_info in self._active_runs.items():
                pipeline_run = run_info["pipeline_run"]
                all_runs.append(
                    {
                        "run_id": run_id,
                        "pipeline_name": pipeline_run.pipeline_name,
                        "status": pipeline_run.status.value,
                        "start_time": pipeline_run.start_time,
                        "end_time": pipeline_run.end_time,
                        "duration_seconds": pipeline_run.duration_seconds,
                        "records_processed": pipeline_run.records_processed,
                        "success_rate": pipeline_run.success_rate,
                    }
                )

            # Add historical runs
            all_runs.extend(self._run_history)

            # Apply filters
            filtered_runs = all_runs
            if status_filter:
                filtered_runs = [run for run in filtered_runs if run["status"] == status_filter]
            if pipeline_name_filter:
                filtered_runs = [
                    run for run in filtered_runs if run["pipeline_name"] == pipeline_name_filter
                ]

            # Sort by start time (newest first)
            filtered_runs.sort(key=lambda x: x["start_time"], reverse=True)

            # Apply pagination
            total_count = len(filtered_runs)
            start_idx = (pagination.page - 1) * pagination.page_size
            end_idx = start_idx + pagination.page_size
            paginated_runs = filtered_runs[start_idx:end_idx]

            return {
                "runs": paginated_runs,
                "total_count": total_count,
                "page": pagination.page,
                "page_size": pagination.page_size,
                "total_pages": max(
                    1, (total_count + pagination.page_size - 1) // pagination.page_size
                ),
                "has_next": end_idx < total_count,
                "has_previous": pagination.page > 1,
            }

        except Exception as e:
            logger.error(f"Failed to list pipeline runs: {e}")
            raise ServiceUnavailableException("pipeline_service", f"Pipeline listing failed: {e!s}")

    async def cancel_pipeline_run(
        self, run_id: str, user_id: Optional[str] = None
    ) -> StatusResponse:
        """
        Cancel a running pipeline.

        Args:
            run_id: Pipeline run identifier
            user_id: User requesting cancellation

        Returns:
            Updated pipeline status
        """

        try:
            logger.info("Cancelling pipeline run", run_id=run_id, user_id=user_id)

            if run_id not in self._active_runs:
                raise ResourceNotFoundException("pipeline_run", run_id)

            run_info = self._active_runs[run_id]
            pipeline_run = run_info["pipeline_run"]

            # Can only cancel running or pending pipelines
            if pipeline_run.status not in [StatusEnum.PENDING, StatusEnum.IN_PROGRESS]:
                raise PipelineException(
                    f"Cannot cancel pipeline in status: {pipeline_run.status.value}",
                    pipeline_run.pipeline_name,
                )

            # Update status to cancelled
            pipeline_run.status = StatusEnum.CANCELLED
            pipeline_run.end_time = datetime.now()
            if pipeline_run.end_time:
                pipeline_run.duration_seconds = (
                    pipeline_run.end_time - pipeline_run.start_time
                ).total_seconds()

            # Mark current stage as cancelled
            if run_info.get("stage_results"):
                current_stage = run_info["stage_results"][-1]
                if current_stage.status == StatusEnum.IN_PROGRESS:
                    current_stage.status = StatusEnum.CANCELLED
                    current_stage.end_time = datetime.now()

            # Move to history
            await self._move_to_history(run_id)

            logger.info("Pipeline run cancelled successfully", run_id=run_id)

            return StatusResponse(
                operation_id=run_id,
                status="cancelled",
                progress_percentage=self._calculate_progress(pipeline_run),
                current_step="Cancelled",
                estimated_completion=None,
                result_url=None,
                error_message="Pipeline cancelled by user",
            )

        except Exception as e:
            logger.error(f"Failed to cancel pipeline: {e}")
            if isinstance(e, (ResourceNotFoundException, PipelineException)):
                raise
            raise ServiceUnavailableException(
                "pipeline_service", f"Pipeline cancellation failed: {e!s}"
            )

    async def get_available_pipelines(self) -> dict[str, Any]:
        """Get list of available pipelines and their configurations."""

        return {
            "pipelines": {
                name: {
                    "name": config["name"],
                    "description": config["description"],
                    "stages": config["stages"],
                    "estimated_duration_minutes": config["estimated_duration"] // 60,
                    "supports_parallel_execution": config["max_parallel"],
                }
                for name, config in self.available_pipelines.items()
            },
            "system_limits": {
                "max_concurrent_pipelines": self.max_concurrent_pipelines,
                "currently_running": len(self._active_runs),
            },
        }

    async def _check_recent_successful_run(
        self, pipeline_name: str, hours_threshold: int = 24
    ) -> Optional[dict[str, Any]]:
        """Check if there's a recent successful run of the pipeline."""

        cutoff_time = datetime.now() - timedelta(hours=hours_threshold)

        # Check active runs first
        for run_id, run_info in self._active_runs.items():
            pipeline_run = run_info["pipeline_run"]
            if (
                pipeline_run.pipeline_name == pipeline_name
                and pipeline_run.status == StatusEnum.COMPLETED
                and pipeline_run.start_time >= cutoff_time
            ):
                return {"run_id": run_id, "start_time": pipeline_run.start_time}

        # Check historical runs
        for run in self._run_history:
            if (
                run["pipeline_name"] == pipeline_name
                and run["status"] == "completed"
                and run["start_time"] >= cutoff_time
            ):
                return run

        return None

    async def _check_concurrency_limits(
        self, pipeline_name: str, pipeline_config: dict[str, Any]
    ) -> None:
        """Check if pipeline can be run considering concurrency limits."""

        # Check global concurrency limit
        if len(self._active_runs) >= self.max_concurrent_pipelines:
            raise PipelineException(
                f"Maximum concurrent pipelines limit reached ({self.max_concurrent_pipelines})",
                pipeline_name,
            )

        # Check pipeline-specific limits
        if not pipeline_config.get("max_parallel", True):
            # Check if same pipeline is already running
            for run_info in self._active_runs.values():
                if (
                    run_info["pipeline_run"].pipeline_name == pipeline_name
                    and run_info["pipeline_run"].status == StatusEnum.IN_PROGRESS
                ):
                    raise PipelineException(
                        f"Pipeline '{pipeline_name}' is already running and doesn't support parallel execution",
                        pipeline_name,
                    )

    async def _create_pipeline_run(
        self, request: PipelineRunRequest, pipeline_config: dict[str, Any]
    ) -> PipelineRun:
        """Create a new pipeline run instance."""

        run_id = str(uuid.uuid4())

        # Determine stages to execute
        if request.stage:
            stages = [request.stage.value]
        else:
            stages = pipeline_config["stages"]

        pipeline_run = PipelineRun(
            run_id=run_id,
            pipeline_name=request.pipeline_name,
            status=StatusEnum.PENDING,
            start_time=datetime.now(),
            total_stages=len(stages),
            completed_stages=0,
            failed_stages=0,
            records_processed=0,
            metadata={
                "requested_by": "api_user",  # Would be actual user from auth
                "parameters": request.parameters,
                "stages": stages,
                "notification_email": request.notification_email,
            },
        )

        return pipeline_run

    async def _execute_pipeline_async(
        self,
        pipeline_run: PipelineRun,
        request: PipelineRunRequest,
        pipeline_config: dict[str, Any],
    ) -> None:
        """Execute pipeline asynchronously."""

        run_id = pipeline_run.run_id
        stage_results = []

        try:
            # Add to active runs
            self._active_runs[run_id] = {
                "pipeline_run": pipeline_run,
                "stage_results": stage_results,
                "request": request,
            }

            # Update status to running
            pipeline_run.status = StatusEnum.IN_PROGRESS

            # Execute stages
            stages = pipeline_run.metadata.get("stages", pipeline_config["stages"])

            for stage_name in stages:
                logger.info("Executing pipeline stage", run_id=run_id, stage=stage_name)

                stage_result = await self._execute_pipeline_stage(
                    pipeline_run, stage_name, request.parameters
                )

                stage_results.append(stage_result)

                if stage_result.status == StatusEnum.FAILED:
                    pipeline_run.failed_stages += 1
                    pipeline_run.error_message = stage_result.error_message
                    break
                elif stage_result.status == StatusEnum.COMPLETED:
                    pipeline_run.completed_stages += 1
                    pipeline_run.records_processed += stage_result.records_processed

                # Check for cancellation
                if pipeline_run.status == StatusEnum.CANCELLED:
                    logger.info("Pipeline execution cancelled", run_id=run_id)
                    return

            # Determine final status
            if pipeline_run.failed_stages > 0:
                pipeline_run.status = StatusEnum.FAILED
                pipeline_run.error_message = (
                    pipeline_run.error_message or "One or more stages failed"
                )
            else:
                pipeline_run.status = StatusEnum.COMPLETED

            pipeline_run.end_time = datetime.now()
            if pipeline_run.end_time:
                pipeline_run.duration_seconds = (
                    pipeline_run.end_time - pipeline_run.start_time
                ).total_seconds()

            logger.info(
                "Pipeline execution completed",
                run_id=run_id,
                status=pipeline_run.status.value,
                duration=pipeline_run.duration_seconds,
                records_processed=pipeline_run.records_processed,
            )

            # Send notification if email provided
            if request.notification_email:
                await self._send_completion_notification(
                    request.notification_email, pipeline_run, stage_results
                )

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", run_id=run_id)

            pipeline_run.status = StatusEnum.FAILED
            pipeline_run.error_message = str(e)
            pipeline_run.end_time = datetime.now()
            if pipeline_run.end_time:
                pipeline_run.duration_seconds = (
                    pipeline_run.end_time - pipeline_run.start_time
                ).total_seconds()

        finally:
            # Move completed run to history
            await self._move_to_history(run_id)

    async def _execute_pipeline_stage(
        self, pipeline_run: PipelineRun, stage_name: str, parameters: dict[str, Any]
    ) -> PipelineStageResult:
        """Execute a single pipeline stage."""

        stage_start = datetime.now()

        stage_result = PipelineStageResult(
            stage_name=stage_name,
            status=StatusEnum.IN_PROGRESS,
            start_time=stage_start,
            records_processed=0,
        )

        try:
            # Mock stage execution - in real implementation, this would
            # integrate with existing AHGD ETL infrastructure

            execution_time = self._get_mock_stage_duration(stage_name)
            records_to_process = self._get_mock_records_count(stage_name)

            # Simulate processing with progress updates
            processed_records = 0
            while processed_records < records_to_process:
                # Simulate work
                await asyncio.sleep(0.1)

                batch_size = min(100, records_to_process - processed_records)
                processed_records += batch_size
                stage_result.records_processed = processed_records

                # Check for cancellation
                if pipeline_run.status == StatusEnum.CANCELLED:
                    stage_result.status = StatusEnum.CANCELLED
                    stage_result.end_time = datetime.now()
                    return stage_result

            # Complete stage
            stage_result.status = StatusEnum.COMPLETED
            stage_result.end_time = datetime.now()
            stage_result.performance_metrics = {
                "records_per_second": processed_records
                / max(1, stage_result.duration_seconds or 1),
                "memory_peak_mb": 256,  # Mock value
                "cpu_avg_percent": 45,  # Mock value
            }

            logger.info(
                "Pipeline stage completed",
                run_id=pipeline_run.run_id,
                stage=stage_name,
                records_processed=processed_records,
                duration=stage_result.duration_seconds,
            )

            # Track data lineage
            track_lineage(
                f"pipeline_{pipeline_run.run_id}_input",
                f"pipeline_{pipeline_run.run_id}_{stage_name}_output",
                f"pipeline_stage_{stage_name}",
            )

        except Exception as e:
            logger.error(
                f"Pipeline stage failed: {e}", run_id=pipeline_run.run_id, stage=stage_name
            )

            stage_result.status = StatusEnum.FAILED
            stage_result.error_message = str(e)
            stage_result.end_time = datetime.now()

        return stage_result

    def _get_mock_stage_duration(self, stage_name: str) -> int:
        """Get mock execution duration for stage (in seconds)."""
        durations = {
            "extract": 300,  # 5 minutes
            "transform": 600,  # 10 minutes
            "validate": 180,  # 3 minutes
            "load": 240,  # 4 minutes
            "analyse": 120,  # 2 minutes
        }
        return durations.get(stage_name, 60)

    def _get_mock_records_count(self, stage_name: str) -> int:
        """Get mock records count for stage processing."""
        counts = {
            "extract": 57736,  # SA1 count
            "transform": 57736,
            "validate": 57736,
            "load": 57736,
            "analyse": 57736,
        }
        return counts.get(stage_name, 1000)

    def _calculate_progress(self, pipeline_run: PipelineRun) -> float:
        """Calculate pipeline progress percentage."""
        if pipeline_run.total_stages == 0:
            return 0.0

        progress = (pipeline_run.completed_stages / pipeline_run.total_stages) * 100
        return min(100.0, max(0.0, progress))

    def _get_current_step(self, pipeline_run: PipelineRun) -> str:
        """Get current pipeline step description."""
        if pipeline_run.status == StatusEnum.COMPLETED:
            return "Completed"
        elif pipeline_run.status == StatusEnum.FAILED:
            return f"Failed at stage {pipeline_run.completed_stages + 1}"
        elif pipeline_run.status == StatusEnum.CANCELLED:
            return "Cancelled"
        elif pipeline_run.status == StatusEnum.PENDING:
            return "Pending execution"
        else:
            return f"Processing stage {pipeline_run.completed_stages + 1} of {pipeline_run.total_stages}"

    def _estimate_completion_time(self, pipeline_run: PipelineRun) -> Optional[datetime]:
        """Estimate pipeline completion time."""
        if pipeline_run.completed_stages == 0:
            # Use default estimate
            pipeline_config = self.available_pipelines.get(pipeline_run.pipeline_name)
            if pipeline_config:
                estimated_seconds = pipeline_config["estimated_duration"]
                return pipeline_run.start_time + timedelta(seconds=estimated_seconds)
        else:
            # Calculate based on completed stages
            elapsed = (datetime.now() - pipeline_run.start_time).total_seconds()
            avg_stage_time = elapsed / pipeline_run.completed_stages
            remaining_stages = pipeline_run.total_stages - pipeline_run.completed_stages
            estimated_remaining = avg_stage_time * remaining_stages
            return datetime.now() + timedelta(seconds=estimated_remaining)

        return None

    async def _move_to_history(self, run_id: str) -> None:
        """Move completed pipeline run to history."""

        if run_id in self._active_runs:
            run_info = self._active_runs[run_id]
            pipeline_run = run_info["pipeline_run"]

            # Create history record
            history_record = {
                "run_id": run_id,
                "pipeline_name": pipeline_run.pipeline_name,
                "status": pipeline_run.status.value,
                "start_time": pipeline_run.start_time,
                "end_time": pipeline_run.end_time,
                "duration_seconds": pipeline_run.duration_seconds,
                "records_processed": pipeline_run.records_processed,
                "success_rate": pipeline_run.success_rate,
                "error_message": pipeline_run.error_message,
            }

            # Add to history
            self._run_history.insert(0, history_record)

            # Maintain history size limit
            if len(self._run_history) > self._max_history:
                self._run_history = self._run_history[: self._max_history]

            # Remove from active runs
            del self._active_runs[run_id]

            logger.debug("Pipeline run moved to history", run_id=run_id)

    def _find_historical_run(self, run_id: str) -> Optional[dict[str, Any]]:
        """Find a pipeline run in history."""
        for run in self._run_history:
            if run["run_id"] == run_id:
                return run
        return None

    async def _get_pipeline_run_response(self, run_id: str) -> PipelineRunResponse:
        """Get full pipeline run response for a run ID."""

        if run_id in self._active_runs:
            run_info = self._active_runs[run_id]
            pipeline_run = run_info["pipeline_run"]
            stage_results = run_info["stage_results"]
        else:
            # Would need to reconstruct from history/database
            # For now, return minimal response
            historical = self._find_historical_run(run_id)
            if not historical:
                raise ResourceNotFoundException("pipeline_run", run_id)

            pipeline_run = PipelineRun(
                run_id=run_id,
                pipeline_name=historical["pipeline_name"],
                status=StatusEnum[historical["status"].upper()],
                start_time=historical["start_time"],
                end_time=historical["end_time"],
                total_stages=1,  # Mock
                completed_stages=1 if historical["status"] == "completed" else 0,
                failed_stages=1 if historical["status"] == "failed" else 0,
                records_processed=historical.get("records_processed", 0),
            )
            stage_results = []

        return PipelineRunResponse(
            pipeline_run=pipeline_run,
            stage_results=stage_results,
            logs_url=f"/api/v1/pipelines/{run_id}/logs",
            artifacts_url=f"/api/v1/pipelines/{run_id}/artifacts",
            next_actions=self._get_next_actions(pipeline_run),
        )

    def _get_next_actions(self, pipeline_run: PipelineRun) -> list[str]:
        """Get recommended next actions based on pipeline status."""

        if pipeline_run.status == StatusEnum.COMPLETED:
            return [
                "Review pipeline results and quality metrics",
                "Export processed data if needed",
                "Schedule next pipeline run",
            ]
        elif pipeline_run.status == StatusEnum.FAILED:
            return [
                "Review error logs for failure cause",
                "Check data source availability",
                "Retry pipeline execution after resolving issues",
            ]
        elif pipeline_run.status == StatusEnum.IN_PROGRESS:
            return [
                "Monitor pipeline progress via WebSocket",
                "Check logs for detailed progress information",
            ]
        else:
            return []

    async def _send_completion_notification(
        self, email: str, pipeline_run: PipelineRun, stage_results: list[PipelineStageResult]
    ) -> None:
        """Send pipeline completion notification email."""

        # Mock email notification - would integrate with actual email service
        logger.info(
            "Sending pipeline completion notification",
            email=email,
            run_id=pipeline_run.run_id,
            status=pipeline_run.status.value,
        )

        # In real implementation, would send actual email
        pass


# Singleton instance for dependency injection
pipeline_service = PipelineService()


async def get_pipeline_service() -> PipelineService:
    """Get pipeline service instance."""
    return pipeline_service
