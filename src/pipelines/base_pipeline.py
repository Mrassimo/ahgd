"""
Base pipeline class with checkpoint and recovery capabilities.

This module provides the foundation for all ETL pipelines with built-in support
for checkpointing, state persistence, and error recovery.
"""

import json
import pickle
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from concurrent.futures import ThreadPoolExecutor, Future
import threading

from src.utils.logging import get_logger
from src.utils.interfaces import AHGDException

logger = get_logger(__name__)


class PipelineState(Enum):
    """Pipeline execution states."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    RECOVERING = "recovering"


class StageState(Enum):
    """Individual stage execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class PipelineContext:
    """Context object passed between pipeline stages."""
    pipeline_id: str
    run_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    stage_outputs: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_output(self, stage_name: str, output: Any) -> None:
        """Add stage output to context."""
        self.stage_outputs[stage_name] = output
        
    def get_output(self, stage_name: str) -> Optional[Any]:
        """Retrieve stage output from context."""
        return self.stage_outputs.get(stage_name)


@dataclass
class StageResult:
    """Result of a stage execution."""
    stage_name: str
    state: StageState
    start_time: datetime
    end_time: Optional[datetime] = None
    output: Optional[Any] = None
    error: Optional[Exception] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Calculate stage execution duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class PipelineError(AHGDException):
    """Base exception for pipeline errors."""
    pass


class CheckpointError(PipelineError):
    """Exception for checkpoint-related errors."""
    pass


class StageExecutionError(PipelineError):
    """Exception for stage execution errors."""
    pass


class BasePipeline(ABC):
    """
    Abstract base class for ETL pipelines with checkpoint support.
    
    Provides infrastructure for:
    - Stage management and execution
    - Checkpoint creation and recovery
    - State persistence
    - Error handling and retry logic
    - Progress tracking
    """
    
    def __init__(
        self,
        name: str,
        checkpoint_dir: Optional[Path] = None,
        max_retries: int = 3,
        enable_checkpoints: bool = True,
        parallel_stages: bool = False,
        max_workers: int = 4
    ):
        """
        Initialise pipeline.
        
        Args:
            name: Pipeline name
            checkpoint_dir: Directory for storing checkpoints
            max_retries: Maximum retry attempts for failed stages
            enable_checkpoints: Whether to enable checkpointing
            parallel_stages: Whether to execute independent stages in parallel
            max_workers: Maximum worker threads for parallel execution
        """
        self.name = name
        self.checkpoint_dir = Path(checkpoint_dir or f"checkpoints/{name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = max_retries
        self.enable_checkpoints = enable_checkpoints
        self.parallel_stages = parallel_stages
        self.max_workers = max_workers
        
        self.state = PipelineState.PENDING
        self.context: Optional[PipelineContext] = None
        self.stage_results: Dict[str, StageResult] = {}
        self.current_stage: Optional[str] = None
        
        self._lock = threading.RLock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[str, Future] = {}
        
        logger.info(
            "Pipeline initialised",
            pipeline=name,
            checkpoint_dir=str(self.checkpoint_dir),
            parallel_stages=parallel_stages
        )
    
    @abstractmethod
    def define_stages(self) -> List[str]:
        """
        Define pipeline stages in execution order.
        
        Returns:
            List of stage names
        """
        pass
    
    @abstractmethod
    def execute_stage(self, stage_name: str, context: PipelineContext) -> Any:
        """
        Execute a specific pipeline stage.
        
        Args:
            stage_name: Name of the stage to execute
            context: Pipeline context
            
        Returns:
            Stage output
        """
        pass
    
    def get_stage_dependencies(self, stage_name: str) -> Set[str]:
        """
        Get dependencies for a stage.
        
        Override to define complex stage dependencies.
        
        Args:
            stage_name: Stage name
            
        Returns:
            Set of dependent stage names
        """
        # Default: linear dependency on previous stage
        stages = self.define_stages()
        if stage_name in stages:
            idx = stages.index(stage_name)
            if idx > 0:
                return {stages[idx - 1]}
        return set()
    
    def can_retry_stage(self, stage_name: str, attempt: int) -> bool:
        """
        Determine if a stage can be retried.
        
        Override for custom retry logic.
        
        Args:
            stage_name: Stage name
            attempt: Current attempt number
            
        Returns:
            Whether the stage can be retried
        """
        return attempt < self.max_retries
    
    def run(
        self,
        resume_from: Optional[str] = None,
        skip_stages: Optional[Set[str]] = None
    ) -> PipelineContext:
        """
        Run the pipeline with optional resume capability.
        
        Args:
            resume_from: Stage to resume from (loads checkpoint if exists)
            skip_stages: Stages to skip during execution
            
        Returns:
            Pipeline context with results
        """
        with self._lock:
            try:
                # Initialise or restore context
                if resume_from:
                    self.context = self._load_checkpoint(resume_from)
                    self.state = PipelineState.RECOVERING
                    logger.info(
                        "Resuming pipeline from checkpoint",
                        pipeline=self.name,
                        resume_stage=resume_from
                    )
                else:
                    self.context = self._create_context()
                    self.state = PipelineState.RUNNING
                
                # Get stages to execute
                stages = self.define_stages()
                start_idx = 0
                
                if resume_from:
                    if resume_from not in stages:
                        raise PipelineError(f"Unknown resume stage: {resume_from}")
                    start_idx = stages.index(resume_from)
                
                # Execute stages
                if self.parallel_stages:
                    self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
                    try:
                        self._execute_parallel(
                            stages[start_idx:],
                            skip_stages or set()
                        )
                    finally:
                        self._executor.shutdown(wait=True)
                        self._executor = None
                else:
                    self._execute_sequential(
                        stages[start_idx:],
                        skip_stages or set()
                    )
                
                self.state = PipelineState.COMPLETED
                logger.info(
                    "Pipeline completed successfully",
                    pipeline=self.name,
                    duration=self._calculate_total_duration()
                )
                
                return self.context
                
            except Exception as e:
                self.state = PipelineState.FAILED
                logger.error(
                    "Pipeline failed",
                    pipeline=self.name,
                    error=str(e),
                    current_stage=self.current_stage
                )
                raise PipelineError(f"Pipeline {self.name} failed: {str(e)}") from e
    
    def pause(self) -> None:
        """Pause pipeline execution."""
        with self._lock:
            if self.state == PipelineState.RUNNING:
                self.state = PipelineState.PAUSED
                logger.info("Pipeline paused", pipeline=self.name)
    
    def resume(self) -> None:
        """Resume paused pipeline."""
        with self._lock:
            if self.state == PipelineState.PAUSED:
                self.state = PipelineState.RUNNING
                logger.info("Pipeline resumed", pipeline=self.name)
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get pipeline execution progress.
        
        Returns:
            Progress information including completed stages
        """
        with self._lock:
            total_stages = len(self.define_stages())
            completed_stages = sum(
                1 for result in self.stage_results.values()
                if result.state == StageState.COMPLETED
            )
            
            return {
                "pipeline": self.name,
                "state": self.state.value,
                "current_stage": self.current_stage,
                "total_stages": total_stages,
                "completed_stages": completed_stages,
                "progress_percentage": (completed_stages / total_stages * 100) if total_stages > 0 else 0,
                "stage_results": {
                    name: {
                        "state": result.state.value,
                        "duration": result.duration,
                        "error": str(result.error) if result.error else None
                    }
                    for name, result in self.stage_results.items()
                }
            }
    
    def _create_context(self) -> PipelineContext:
        """Create new pipeline context."""
        run_id = f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return PipelineContext(
            pipeline_id=self.name,
            run_id=run_id,
            metadata={
                "start_time": datetime.now().isoformat(),
                "checkpoint_enabled": self.enable_checkpoints
            }
        )
    
    def _execute_sequential(
        self,
        stages: List[str],
        skip_stages: Set[str]
    ) -> None:
        """Execute stages sequentially."""
        for stage_name in stages:
            if self.state == PipelineState.PAUSED:
                self._wait_for_resume()
            
            if stage_name in skip_stages:
                self._record_stage_result(
                    stage_name,
                    StageResult(
                        stage_name=stage_name,
                        state=StageState.SKIPPED,
                        start_time=datetime.now()
                    )
                )
                continue
            
            self._execute_stage_with_retry(stage_name)
    
    def _execute_parallel(
        self,
        stages: List[str],
        skip_stages: Set[str]
    ) -> None:
        """Execute independent stages in parallel."""
        stage_graph = self._build_dependency_graph(stages)
        completed = set()
        
        while len(completed) < len(stages):
            if self.state == PipelineState.PAUSED:
                self._wait_for_resume()
            
            # Find stages ready to execute
            ready_stages = [
                stage for stage in stages
                if stage not in completed
                and stage not in self._futures
                and all(dep in completed for dep in self.get_stage_dependencies(stage))
            ]
            
            # Submit ready stages
            for stage_name in ready_stages:
                if stage_name in skip_stages:
                    self._record_stage_result(
                        stage_name,
                        StageResult(
                            stage_name=stage_name,
                            state=StageState.SKIPPED,
                            start_time=datetime.now()
                        )
                    )
                    completed.add(stage_name)
                else:
                    future = self._executor.submit(
                        self._execute_stage_with_retry,
                        stage_name
                    )
                    self._futures[stage_name] = future
            
            # Check completed futures
            for stage_name, future in list(self._futures.items()):
                if future.done():
                    try:
                        future.result()  # Raise any exceptions
                        completed.add(stage_name)
                    except Exception as e:
                        logger.error(
                            "Stage failed in parallel execution",
                            stage=stage_name,
                            error=str(e)
                        )
                        raise
                    finally:
                        del self._futures[stage_name]
            
            # Small sleep to avoid busy waiting
            if self._futures:
                threading.Event().wait(0.1)
    
    def _execute_stage_with_retry(self, stage_name: str) -> None:
        """Execute a stage with retry logic."""
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                self.current_stage = stage_name
                result = StageResult(
                    stage_name=stage_name,
                    state=StageState.RUNNING if attempt == 0 else StageState.RETRYING,
                    start_time=datetime.now()
                )
                
                logger.info(
                    "Executing stage",
                    stage=stage_name,
                    attempt=attempt + 1
                )
                
                # Execute stage
                output = self.execute_stage(stage_name, self.context)
                
                # Record success
                result.state = StageState.COMPLETED
                result.end_time = datetime.now()
                result.output = output
                
                self.context.add_output(stage_name, output)
                self._record_stage_result(stage_name, result)
                
                # Create checkpoint
                if self.enable_checkpoints:
                    self._create_checkpoint(stage_name)
                
                return
                
            except Exception as e:
                last_error = e
                logger.warning(
                    "Stage execution failed",
                    stage=stage_name,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if not self.can_retry_stage(stage_name, attempt + 1):
                    break
                
                attempt += 1
        
        # Record failure
        result = StageResult(
            stage_name=stage_name,
            state=StageState.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            error=last_error
        )
        self._record_stage_result(stage_name, result)
        
        raise StageExecutionError(
            f"Stage {stage_name} failed after {attempt + 1} attempts: {str(last_error)}"
        ) from last_error
    
    def _record_stage_result(self, stage_name: str, result: StageResult) -> None:
        """Record stage execution result."""
        with self._lock:
            self.stage_results[stage_name] = result
            logger.info(
                "Stage result recorded",
                stage=stage_name,
                state=result.state.value,
                duration=result.duration
            )
    
    def _create_checkpoint(self, stage_name: str) -> None:
        """Create checkpoint after stage completion."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{self.context.run_id}_{stage_name}.checkpoint"
            
            checkpoint_data = {
                "pipeline_name": self.name,
                "stage_name": stage_name,
                "timestamp": datetime.now().isoformat(),
                "context": self._serialise_context(),
                "stage_results": self._serialise_stage_results()
            }
            
            # Calculate checksum
            checksum = hashlib.sha256(
                json.dumps(checkpoint_data, sort_keys=True).encode()
            ).hexdigest()
            checkpoint_data["checksum"] = checksum
            
            # Save checkpoint
            with open(checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data, f)
            
            self.context.checkpoints.append(str(checkpoint_file))
            
            logger.debug(
                "Checkpoint created",
                stage=stage_name,
                file=str(checkpoint_file)
            )
            
        except Exception as e:
            logger.error(
                "Failed to create checkpoint",
                stage=stage_name,
                error=str(e)
            )
            if self.enable_checkpoints:
                raise CheckpointError(f"Checkpoint creation failed: {str(e)}") from e
    
    def _load_checkpoint(self, stage_name: str) -> PipelineContext:
        """Load checkpoint for a specific stage."""
        checkpoint_files = list(self.checkpoint_dir.glob(f"*_{stage_name}.checkpoint"))
        
        if not checkpoint_files:
            raise CheckpointError(f"No checkpoint found for stage: {stage_name}")
        
        # Use most recent checkpoint
        checkpoint_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)
            
            # Verify checksum
            saved_checksum = checkpoint_data.pop("checksum", None)
            calculated_checksum = hashlib.sha256(
                json.dumps(checkpoint_data, sort_keys=True).encode()
            ).hexdigest()
            
            if saved_checksum != calculated_checksum:
                raise CheckpointError("Checkpoint checksum verification failed")
            
            # Restore context
            context = self._deserialise_context(checkpoint_data["context"])
            self.stage_results = self._deserialise_stage_results(
                checkpoint_data["stage_results"]
            )
            
            logger.info(
                "Checkpoint loaded",
                stage=stage_name,
                file=str(checkpoint_file)
            )
            
            return context
            
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {str(e)}") from e
    
    def _serialise_context(self) -> Dict[str, Any]:
        """Serialise pipeline context for checkpointing."""
        return {
            "pipeline_id": self.context.pipeline_id,
            "run_id": self.context.run_id,
            "metadata": self.context.metadata,
            "stage_outputs": {
                k: v for k, v in self.context.stage_outputs.items()
                if self._is_serialisable(v)
            },
            "checkpoints": self.context.checkpoints,
            "created_at": self.context.created_at.isoformat()
        }
    
    def _deserialise_context(self, data: Dict[str, Any]) -> PipelineContext:
        """Deserialise pipeline context from checkpoint."""
        context = PipelineContext(
            pipeline_id=data["pipeline_id"],
            run_id=data["run_id"],
            metadata=data["metadata"],
            stage_outputs=data["stage_outputs"],
            checkpoints=data["checkpoints"],
            created_at=datetime.fromisoformat(data["created_at"])
        )
        return context
    
    def _serialise_stage_results(self) -> Dict[str, Any]:
        """Serialise stage results for checkpointing."""
        return {
            name: {
                "stage_name": result.stage_name,
                "state": result.state.value,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "metrics": result.metrics,
                "error": str(result.error) if result.error else None
            }
            for name, result in self.stage_results.items()
        }
    
    def _deserialise_stage_results(self, data: Dict[str, Any]) -> Dict[str, StageResult]:
        """Deserialise stage results from checkpoint."""
        results = {}
        for name, result_data in data.items():
            results[name] = StageResult(
                stage_name=result_data["stage_name"],
                state=StageState(result_data["state"]),
                start_time=datetime.fromisoformat(result_data["start_time"]),
                end_time=datetime.fromisoformat(result_data["end_time"]) if result_data["end_time"] else None,
                metrics=result_data["metrics"]
            )
        return results
    
    def _is_serialisable(self, obj: Any) -> bool:
        """Check if an object can be serialised."""
        try:
            pickle.dumps(obj)
            return True
        except (pickle.PicklingError, TypeError):
            return False
    
    def _build_dependency_graph(self, stages: List[str]) -> Dict[str, Set[str]]:
        """Build stage dependency graph."""
        graph = {}
        for stage in stages:
            graph[stage] = self.get_stage_dependencies(stage)
        return graph
    
    def _wait_for_resume(self) -> None:
        """Wait for pipeline to resume from paused state."""
        while self.state == PipelineState.PAUSED:
            threading.Event().wait(1.0)
    
    def _calculate_total_duration(self) -> float:
        """Calculate total pipeline execution duration."""
        total = 0.0
        for result in self.stage_results.values():
            if result.duration:
                total += result.duration
        return total
    
    def cleanup_checkpoints(self, keep_last: int = 5) -> None:
        """
        Clean up old checkpoints.
        
        Args:
            keep_last: Number of recent checkpoints to keep
        """
        try:
            checkpoint_files = sorted(
                self.checkpoint_dir.glob("*.checkpoint"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            for checkpoint_file in checkpoint_files[keep_last:]:
                checkpoint_file.unlink()
                logger.debug("Removed old checkpoint", file=str(checkpoint_file))
                
        except Exception as e:
            logger.error("Failed to clean up checkpoints", error=str(e))