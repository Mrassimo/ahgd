"""
Pipeline stage management with configuration and monitoring.

This module provides the PipelineStage class for individual stage management
with input/output validation, conditional execution, and monitoring.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import threading
import psutil

from ..utils.logging import get_logger, monitor_performance
from ..utils.interfaces import AHGDException, ValidationError
from ..pipelines.base_pipeline import PipelineContext, StageState, StageResult

logger = get_logger(__name__)


class StageExecutionMode(Enum):
    """Stage execution modes."""
    BLOCKING = "blocking"  # Wait for stage completion
    ASYNC = "async"  # Run asynchronously
    CONDITIONAL = "conditional"  # Run based on conditions


class ResourceUsage(Enum):
    """Resource usage categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    name: str
    description: str = ""
    enabled: bool = True
    timeout: Optional[timedelta] = None
    retry_attempts: int = 3
    retry_delay: timedelta = field(default_factory=lambda: timedelta(seconds=5))
    execution_mode: StageExecutionMode = StageExecutionMode.BLOCKING
    resource_usage: ResourceUsage = ResourceUsage.MEDIUM
    prerequisites: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    checkpoint_enabled: bool = True
    tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageMetrics:
    """Metrics collected during stage execution."""
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    disk_io_mb: float = 0.0
    network_io_mb: float = 0.0
    input_records: int = 0
    output_records: int = 0
    processed_records: int = 0
    error_records: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Calculate stage duration."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def records_per_second(self) -> float:
        """Calculate processing rate."""
        if self.duration and self.processed_records > 0:
            return self.processed_records / self.duration.total_seconds()
        return 0.0


class StageValidationError(ValidationError):
    """Exception for stage validation errors."""
    pass


class StageTimeoutError(AHGDException):
    """Exception for stage timeout."""
    pass


class StagePrerequisiteError(AHGDException):
    """Exception for missing prerequisites."""
    pass


class PipelineStage(ABC):
    """
    Base class for pipeline stages with comprehensive monitoring and validation.
    
    Features:
    - Configurable execution parameters
    - Input/output validation
    - Resource monitoring
    - Conditional execution
    - Retry logic
    - Performance metrics
    """
    
    def __init__(self, config: StageConfig):
        """
        Initialise pipeline stage.
        
        Args:
            config: Stage configuration
        """
        self.config = config
        self.state = StageState.PENDING
        self.metrics: Optional[StageMetrics] = None
        self.context: Optional[PipelineContext] = None
        
        self._start_time: Optional[datetime] = None
        self._process: Optional[psutil.Process] = None
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        logger.info(
            "Pipeline stage initialised",
            stage=config.name,
            enabled=config.enabled
        )
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> Any:
        """
        Execute the stage logic.
        
        Args:
            context: Pipeline context
            
        Returns:
            Stage output
        """
        pass
    
    def validate_input(self, context: PipelineContext) -> None:
        """
        Validate stage input data.
        
        Override in subclasses for custom validation.
        
        Args:
            context: Pipeline context
            
        Raises:
            StageValidationError: If validation fails
        """
        if not context:
            raise StageValidationError("Pipeline context is required")
        
        # Check prerequisites
        for prerequisite in self.config.prerequisites:
            if prerequisite not in context.stage_outputs:
                raise StagePrerequisiteError(
                    f"Missing prerequisite stage: {prerequisite}"
                )
        
        # Apply validation rules
        self._apply_validation_rules(context)
    
    def validate_output(self, output: Any) -> None:
        """
        Validate stage output data.
        
        Override in subclasses for custom validation.
        
        Args:
            output: Stage output
            
        Raises:
            StageValidationError: If validation fails
        """
        if output is None and self.config.validation_rules.get("allow_none", False):
            raise StageValidationError("Stage output cannot be None")
    
    def should_execute(self, context: PipelineContext) -> bool:
        """
        Determine if stage should execute.
        
        Override for conditional execution logic.
        
        Args:
            context: Pipeline context
            
        Returns:
            Whether stage should execute
        """
        # Check if enabled
        if not self.config.enabled:
            return False
        
        # Check conditional execution parameters
        conditions = self.config.parameters.get("conditions", {})
        
        for condition_name, condition_value in conditions.items():
            if condition_name == "skip_if_exists":
                # Skip if output already exists
                if condition_value and self.config.name in context.stage_outputs:
                    return False
            
            elif condition_name == "run_if_changed":
                # Run only if input data has changed
                if condition_value:
                    # This would require checksum comparison
                    # Implementation depends on specific requirements
                    pass
            
            elif condition_name == "schedule":
                # Run based on schedule
                if condition_value:
                    # Implementation would check schedule
                    pass
        
        return True
    
    def run(self, context: PipelineContext) -> StageResult:
        """
        Run the stage with monitoring and error handling.
        
        Args:
            context: Pipeline context
            
        Returns:
            Stage execution result
        """
        self.context = context
        
        # Check if stage should execute
        if not self.should_execute(context):
            logger.info(
                "Stage skipped due to conditions",
                stage=self.config.name
            )
            return StageResult(
                stage_name=self.config.name,
                state=StageState.SKIPPED,
                start_time=datetime.now()
            )
        
        # Initialise metrics
        self._start_time = datetime.now()
        self.metrics = StageMetrics(
            stage_name=self.config.name,
            start_time=self._start_time
        )
        
        # Start monitoring
        self._start_monitoring()
        
        attempt = 0
        last_error = None
        
        try:
            while attempt < self.config.retry_attempts:
                try:
                    logger.info(
                        "Starting stage execution",
                        stage=self.config.name,
                        attempt=attempt + 1
                    )
                    
                    self.state = StageState.RUNNING
                    
                    # Validate input
                    self.validate_input(context)
                    
                    # Execute with timeout
                    output = self._execute_with_timeout(context)
                    
                    # Validate output
                    self.validate_output(output)
                    
                    # Record success
                    self.state = StageState.COMPLETED
                    self.metrics.end_time = datetime.now()
                    
                    logger.info(
                        "Stage completed successfully",
                        stage=self.config.name,
                        duration=self.metrics.duration.total_seconds() if self.metrics.duration else 0,
                        records_per_second=self.metrics.records_per_second
                    )
                    
                    return StageResult(
                        stage_name=self.config.name,
                        state=StageState.COMPLETED,
                        start_time=self._start_time,
                        end_time=self.metrics.end_time,
                        output=output,
                        metrics=self._export_metrics()
                    )
                
                except (StageValidationError, StageTimeoutError) as e:
                    # These errors should not be retried
                    raise e
                
                except Exception as e:
                    last_error = e
                    attempt += 1
                    
                    if attempt < self.config.retry_attempts:
                        logger.warning(
                            "Stage execution failed, retrying",
                            stage=self.config.name,
                            attempt=attempt,
                            error=str(e),
                            retry_in_seconds=self.config.retry_delay.total_seconds()
                        )
                        
                        self.state = StageState.RETRYING
                        time.sleep(self.config.retry_delay.total_seconds())
                    else:
                        break
            
            # All retries exhausted
            self.state = StageState.FAILED
            self.metrics.end_time = datetime.now()
            
            return StageResult(
                stage_name=self.config.name,
                state=StageState.FAILED,
                start_time=self._start_time,
                end_time=self.metrics.end_time,
                error=last_error,
                metrics=self._export_metrics()
            )
            
        finally:
            # Stop monitoring
            self._stop_monitoring_thread()
    
    def pause(self) -> None:
        """Pause stage execution if supported."""
        logger.info("Stage pause requested", stage=self.config.name)
        # Override in subclasses that support pausing
    
    def resume(self) -> None:
        """Resume stage execution if supported."""
        logger.info("Stage resume requested", stage=self.config.name)
        # Override in subclasses that support resuming
    
    def cancel(self) -> None:
        """Cancel stage execution."""
        logger.info("Stage cancellation requested", stage=self.config.name)
        self.state = StageState.FAILED
        self._stop_monitoring.set()
    
    @monitor_performance("stage_execution")
    def _execute_with_timeout(self, context: PipelineContext) -> Any:
        """Execute stage with timeout handling."""
        if self.config.timeout:
            import signal
            
            def timeout_handler(signum, frame):
                raise StageTimeoutError(
                    f"Stage {self.config.name} timed out after {self.config.timeout}"
                )
            
            # Set timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.timeout.total_seconds()))
            
            try:
                return self.execute(context)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            return self.execute(context)
    
    def _apply_validation_rules(self, context: PipelineContext) -> None:
        """Apply configured validation rules."""
        rules = self.config.validation_rules
        
        # Required inputs
        if "required_inputs" in rules:
            for input_name in rules["required_inputs"]:
                if input_name not in context.stage_outputs:
                    raise StageValidationError(f"Required input missing: {input_name}")
        
        # Data type validation
        if "input_types" in rules:
            for input_name, expected_type in rules["input_types"].items():
                if input_name in context.stage_outputs:
                    actual_value = context.stage_outputs[input_name]
                    if not isinstance(actual_value, expected_type):
                        raise StageValidationError(
                            f"Input {input_name} has wrong type: "
                            f"expected {expected_type.__name__}, got {type(actual_value).__name__}"
                        )
        
        # Custom validation functions
        if "custom_validators" in rules:
            for validator_name, validator_func in rules["custom_validators"].items():
                if callable(validator_func):
                    try:
                        validator_func(context)
                    except Exception as e:
                        raise StageValidationError(
                            f"Custom validation {validator_name} failed: {str(e)}"
                        ) from e
    
    def _start_monitoring(self) -> None:
        """Start resource monitoring thread."""
        if not self.config.tags.get("monitor_resources", True):
            return
        
        try:
            self._process = psutil.Process()
            self._monitoring_thread = threading.Thread(
                target=self._monitor_resources,
                daemon=True
            )
            self._monitoring_thread.start()
            
        except Exception as e:
            logger.warning(
                "Failed to start resource monitoring",
                stage=self.config.name,
                error=str(e)
            )
    
    def _monitor_resources(self) -> None:
        """Monitor resource usage during execution."""
        if not self._process or not self.metrics:
            return
        
        try:
            while not self._stop_monitoring.is_set():
                try:
                    # CPU usage
                    cpu_percent = self._process.cpu_percent()
                    if cpu_percent > self.metrics.cpu_percent:
                        self.metrics.cpu_percent = cpu_percent
                    
                    # Memory usage
                    memory_info = self._process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    if memory_mb > self.metrics.memory_mb:
                        self.metrics.memory_mb = memory_mb
                    
                    # I/O counters
                    io_counters = self._process.io_counters()
                    disk_io_mb = (io_counters.read_bytes + io_counters.write_bytes) / 1024 / 1024
                    self.metrics.disk_io_mb = disk_io_mb
                    
                except psutil.NoSuchProcess:
                    break
                except Exception as e:
                    logger.debug(
                        "Resource monitoring error",
                        stage=self.config.name,
                        error=str(e)
                    )
                
                # Monitor every second
                self._stop_monitoring.wait(1.0)
                
        except Exception as e:
            logger.error(
                "Resource monitoring failed",
                stage=self.config.name,
                error=str(e)
            )
    
    def _stop_monitoring_thread(self) -> None:
        """Stop resource monitoring thread."""
        self._stop_monitoring.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
    
    def _export_metrics(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        if not self.metrics:
            return {}
        
        return {
            "stage_name": self.metrics.stage_name,
            "start_time": self.metrics.start_time.isoformat(),
            "end_time": self.metrics.end_time.isoformat() if self.metrics.end_time else None,
            "duration_seconds": self.metrics.duration.total_seconds() if self.metrics.duration else None,
            "cpu_percent": self.metrics.cpu_percent,
            "memory_mb": self.metrics.memory_mb,
            "disk_io_mb": self.metrics.disk_io_mb,
            "network_io_mb": self.metrics.network_io_mb,
            "input_records": self.metrics.input_records,
            "output_records": self.metrics.output_records,
            "processed_records": self.metrics.processed_records,
            "error_records": self.metrics.error_records,
            "records_per_second": self.metrics.records_per_second,
            "custom_metrics": self.metrics.custom_metrics
        }


class ExtractorStage(PipelineStage):
    """Stage for data extraction operations."""
    
    def __init__(
        self,
        config: StageConfig,
        extractor_class: type,
        **extractor_kwargs
    ):
        """
        Initialise extractor stage.
        
        Args:
            config: Stage configuration
            extractor_class: Extractor class to instantiate
            **extractor_kwargs: Arguments for extractor
        """
        super().__init__(config)
        self.extractor = extractor_class(**extractor_kwargs)
    
    def execute(self, context: PipelineContext) -> Any:
        """Execute extraction."""
        logger.info(
            "Running data extraction",
            stage=self.config.name,
            extractor=self.extractor.__class__.__name__
        )
        
        # Extract data
        data = self.extractor.extract()
        
        # Update metrics
        if self.metrics and hasattr(data, '__len__'):
            self.metrics.output_records = len(data)
            self.metrics.processed_records = len(data)
        
        return data


class TransformerStage(PipelineStage):
    """Stage for data transformation operations."""
    
    def __init__(
        self,
        config: StageConfig,
        transformer_class: type,
        **transformer_kwargs
    ):
        """
        Initialise transformer stage.
        
        Args:
            config: Stage configuration
            transformer_class: Transformer class to instantiate
            **transformer_kwargs: Arguments for transformer
        """
        super().__init__(config)
        self.transformer = transformer_class(**transformer_kwargs)
    
    def execute(self, context: PipelineContext) -> Any:
        """Execute transformation."""
        logger.info(
            "Running data transformation",
            stage=self.config.name,
            transformer=self.transformer.__class__.__name__
        )
        
        # Get input data
        input_data = None
        for prerequisite in self.config.prerequisites:
            if prerequisite in context.stage_outputs:
                input_data = context.stage_outputs[prerequisite]
                break
        
        if input_data is None:
            raise StageValidationError("No input data found for transformation")
        
        # Update input metrics
        if self.metrics and hasattr(input_data, '__len__'):
            self.metrics.input_records = len(input_data)
        
        # Transform data
        transformed_data = self.transformer.transform(input_data)
        
        # Update metrics
        if self.metrics and hasattr(transformed_data, '__len__'):
            self.metrics.output_records = len(transformed_data)
            self.metrics.processed_records = len(transformed_data)
        
        return transformed_data


class ValidatorStage(PipelineStage):
    """Stage for data validation operations."""
    
    def __init__(
        self,
        config: StageConfig,
        validator_class: type,
        **validator_kwargs
    ):
        """
        Initialise validator stage.
        
        Args:
            config: Stage configuration
            validator_class: Validator class to instantiate
            **validator_kwargs: Arguments for validator
        """
        super().__init__(config)
        self.validator = validator_class(**validator_kwargs)
    
    def execute(self, context: PipelineContext) -> Any:
        """Execute validation."""
        logger.info(
            "Running data validation",
            stage=self.config.name,
            validator=self.validator.__class__.__name__
        )
        
        # Get input data
        input_data = None
        for prerequisite in self.config.prerequisites:
            if prerequisite in context.stage_outputs:
                input_data = context.stage_outputs[prerequisite]
                break
        
        if input_data is None:
            raise StageValidationError("No input data found for validation")
        
        # Update input metrics
        if self.metrics and hasattr(input_data, '__len__'):
            self.metrics.input_records = len(input_data)
        
        # Validate data
        validation_result = self.validator.validate(input_data)
        
        # Update metrics
        if self.metrics:
            self.metrics.processed_records = self.metrics.input_records
            if hasattr(validation_result, 'errors'):
                self.metrics.error_records = len(validation_result.errors)
        
        # Return original data (validation doesn't transform)
        return input_data


class LoaderStage(PipelineStage):
    """Stage for data loading operations."""
    
    def __init__(
        self,
        config: StageConfig,
        loader_class: type,
        **loader_kwargs
    ):
        """
        Initialise loader stage.
        
        Args:
            config: Stage configuration
            loader_class: Loader class to instantiate
            **loader_kwargs: Arguments for loader
        """
        super().__init__(config)
        self.loader = loader_class(**loader_kwargs)
    
    def execute(self, context: PipelineContext) -> Any:
        """Execute loading."""
        logger.info(
            "Running data loading",
            stage=self.config.name,
            loader=self.loader.__class__.__name__
        )
        
        # Get input data
        input_data = None
        for prerequisite in self.config.prerequisites:
            if prerequisite in context.stage_outputs:
                input_data = context.stage_outputs[prerequisite]
                break
        
        if input_data is None:
            raise StageValidationError("No input data found for loading")
        
        # Update input metrics
        if self.metrics and hasattr(input_data, '__len__'):
            self.metrics.input_records = len(input_data)
        
        # Load data
        result = self.loader.load(input_data)
        
        # Update metrics
        if self.metrics:
            self.metrics.processed_records = self.metrics.input_records
            self.metrics.output_records = self.metrics.input_records
        
        return result