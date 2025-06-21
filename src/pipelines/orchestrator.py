"""
Pipeline orchestrator for managing complex ETL workflows.

This module provides orchestration capabilities for running multiple pipelines
with dependency management, resource allocation, and monitoring.
"""

import asyncio
import json
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import networkx as nx

from src.utils.logging import get_logger, monitor_performance
from src.utils.config import get_config
from src.pipelines.base_pipeline import BasePipeline, PipelineContext, PipelineState
from src.pipelines.monitoring import PipelineMonitor

logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Pipeline execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class PipelineDefinition:
    """Definition of a pipeline for orchestration."""
    name: str
    pipeline_class: type[BasePipeline]
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0
    resource_requirements: Dict[ResourceType, float] = field(default_factory=dict)
    timeout: Optional[timedelta] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Result of pipeline orchestration."""
    total_pipelines: int
    successful_pipelines: int
    failed_pipelines: int
    skipped_pipelines: int
    start_time: datetime
    end_time: datetime
    pipeline_results: Dict[str, PipelineContext]
    execution_order: List[str]
    errors: Dict[str, Exception] = field(default_factory=dict)
    
    @property
    def duration(self) -> timedelta:
        """Calculate total orchestration duration."""
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        """Calculate pipeline success rate."""
        if self.total_pipelines == 0:
            return 0.0
        return self.successful_pipelines / self.total_pipelines * 100


class ResourceManager:
    """Manages resource allocation for pipeline execution."""
    
    def __init__(self, resource_limits: Optional[Dict[ResourceType, float]] = None):
        """
        Initialise resource manager.
        
        Args:
            resource_limits: Maximum available resources
        """
        self.resource_limits = resource_limits or self._get_system_resources()
        self.allocated_resources: Dict[str, Dict[ResourceType, float]] = {}
        self._lock = threading.RLock()
        
    def _get_system_resources(self) -> Dict[ResourceType, float]:
        """Get system resource limits."""
        import psutil
        
        return {
            ResourceType.CPU: psutil.cpu_count(),
            ResourceType.MEMORY: psutil.virtual_memory().total / (1024 ** 3),  # GB
            ResourceType.DISK: psutil.disk_usage('/').free / (1024 ** 3),  # GB
            ResourceType.NETWORK: 1000.0  # Mbps (placeholder)
        }
    
    def allocate(
        self,
        pipeline_name: str,
        requirements: Dict[ResourceType, float]
    ) -> bool:
        """
        Allocate resources for a pipeline.
        
        Args:
            pipeline_name: Pipeline requesting resources
            requirements: Resource requirements
            
        Returns:
            Whether allocation was successful
        """
        with self._lock:
            # Check availability
            available = self._get_available_resources()
            
            for resource_type, required in requirements.items():
                if available.get(resource_type, 0) < required:
                    logger.warning(
                        "Insufficient resources",
                        pipeline=pipeline_name,
                        resource=resource_type.value,
                        required=required,
                        available=available.get(resource_type, 0)
                    )
                    return False
            
            # Allocate resources
            self.allocated_resources[pipeline_name] = requirements
            logger.debug(
                "Resources allocated",
                pipeline=pipeline_name,
                resources=requirements
            )
            return True
    
    def release(self, pipeline_name: str) -> None:
        """Release resources allocated to a pipeline."""
        with self._lock:
            if pipeline_name in self.allocated_resources:
                released = self.allocated_resources.pop(pipeline_name)
                logger.debug(
                    "Resources released",
                    pipeline=pipeline_name,
                    resources=released
                )
    
    def _get_available_resources(self) -> Dict[ResourceType, float]:
        """Calculate currently available resources."""
        available = self.resource_limits.copy()
        
        for allocated in self.allocated_resources.values():
            for resource_type, amount in allocated.items():
                available[resource_type] = available.get(resource_type, 0) - amount
        
        return available


class PipelineOrchestrator:
    """
    Orchestrates execution of multiple pipelines with dependency management.
    
    Features:
    - Dependency resolution and topological sorting
    - Parallel and distributed execution
    - Resource management
    - Dynamic pipeline construction
    - Comprehensive monitoring
    """
    
    def __init__(
        self,
        name: str = "default",
        execution_mode: ExecutionMode = ExecutionMode.PARALLEL,
        max_workers: int = 4,
        checkpoint_dir: Optional[Path] = None,
        enable_monitoring: bool = True,
        resource_limits: Optional[Dict[ResourceType, float]] = None
    ):
        """
        Initialise pipeline orchestrator.
        
        Args:
            name: Orchestrator name
            execution_mode: How to execute pipelines
            max_workers: Maximum concurrent pipelines
            checkpoint_dir: Directory for orchestration checkpoints
            enable_monitoring: Whether to enable monitoring
            resource_limits: Resource allocation limits
        """
        self.name = name
        self.execution_mode = execution_mode
        self.max_workers = max_workers
        self.checkpoint_dir = Path(checkpoint_dir or f"checkpoints/orchestration/{name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.pipelines: Dict[str, PipelineDefinition] = {}
        self.pipeline_instances: Dict[str, BasePipeline] = {}
        self.dependency_graph = nx.DiGraph()
        
        self.resource_manager = ResourceManager(resource_limits)
        self.monitor = PipelineMonitor() if enable_monitoring else None
        
        self._executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self._futures: Dict[str, Future] = {}
        self._completed_pipelines: Set[str] = set()
        self._failed_pipelines: Set[str] = set()
        self._lock = threading.RLock()
        
        logger.info(
            "Pipeline orchestrator initialised",
            name=name,
            mode=execution_mode.value,
            max_workers=max_workers
        )
    
    def register_pipeline(
        self,
        definition: PipelineDefinition
    ) -> None:
        """
        Register a pipeline for orchestration.
        
        Args:
            definition: Pipeline definition
        """
        with self._lock:
            self.pipelines[definition.name] = definition
            self.dependency_graph.add_node(definition.name)
            
            # Add dependencies to graph
            for dependency in definition.dependencies:
                self.dependency_graph.add_edge(dependency, definition.name)
            
            logger.info(
                "Pipeline registered",
                pipeline=definition.name,
                dependencies=definition.dependencies
            )
    
    def register_from_config(self, config_path: Path) -> None:
        """
        Register pipelines from configuration file.
        
        Args:
            config_path: Path to pipeline configuration
        """
        import yaml
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        for pipeline_config in config.get("pipelines", []):
            # Dynamic class loading
            module_path, class_name = pipeline_config["class"].rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            pipeline_class = getattr(module, class_name)
            
            definition = PipelineDefinition(
                name=pipeline_config["name"],
                pipeline_class=pipeline_class,
                config=pipeline_config.get("config", {}),
                dependencies=pipeline_config.get("dependencies", []),
                priority=pipeline_config.get("priority", 0),
                resource_requirements={
                    ResourceType[k.upper()]: v
                    for k, v in pipeline_config.get("resources", {}).items()
                },
                timeout=timedelta(**pipeline_config["timeout"]) if "timeout" in pipeline_config else None,
                retry_policy=pipeline_config.get("retry_policy", {})
            )
            
            self.register_pipeline(definition)
    
    @monitor_performance("orchestration")
    def run(
        self,
        target_pipelines: Optional[List[str]] = None,
        skip_pipelines: Optional[Set[str]] = None,
        force_sequential: bool = False
    ) -> OrchestrationResult:
        """
        Run registered pipelines with dependency resolution.
        
        Args:
            target_pipelines: Specific pipelines to run (with dependencies)
            skip_pipelines: Pipelines to skip
            force_sequential: Force sequential execution
            
        Returns:
            Orchestration result
        """
        start_time = datetime.now()
        skip_pipelines = skip_pipelines or set()
        
        try:
            # Validate and resolve dependencies
            execution_order = self._resolve_execution_order(target_pipelines)
            
            # Filter skipped pipelines
            execution_order = [
                p for p in execution_order
                if p not in skip_pipelines
            ]
            
            logger.info(
                "Starting pipeline orchestration",
                total_pipelines=len(execution_order),
                execution_order=execution_order,
                mode=self.execution_mode.value if not force_sequential else "sequential"
            )
            
            # Create pipeline instances
            self._create_pipeline_instances(execution_order)
            
            # Execute pipelines
            if force_sequential or self.execution_mode == ExecutionMode.SEQUENTIAL:
                pipeline_results = self._execute_sequential(execution_order)
            else:
                pipeline_results = self._execute_parallel(execution_order)
            
            # Compile results
            end_time = datetime.now()
            
            result = OrchestrationResult(
                total_pipelines=len(execution_order),
                successful_pipelines=len(self._completed_pipelines),
                failed_pipelines=len(self._failed_pipelines),
                skipped_pipelines=len(skip_pipelines),
                start_time=start_time,
                end_time=end_time,
                pipeline_results=pipeline_results,
                execution_order=execution_order,
                errors={
                    name: self._futures.get(name, Future()).exception()
                    for name in self._failed_pipelines
                    if name in self._futures
                }
            )
            
            logger.info(
                "Pipeline orchestration completed",
                duration=result.duration.total_seconds(),
                success_rate=result.success_rate
            )
            
            # Save orchestration checkpoint
            self._save_orchestration_state(result)
            
            return result
            
        except Exception as e:
            logger.error(
                "Orchestration failed",
                error=str(e)
            )
            raise
        finally:
            # Cleanup
            self._cleanup()
    
    def run_pipeline(
        self,
        pipeline_name: str,
        context: Optional[PipelineContext] = None
    ) -> PipelineContext:
        """
        Run a single pipeline.
        
        Args:
            pipeline_name: Pipeline to run
            context: Optional context to pass
            
        Returns:
            Pipeline execution context
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
        
        definition = self.pipelines[pipeline_name]
        
        # Allocate resources
        if not self.resource_manager.allocate(
            pipeline_name,
            definition.resource_requirements
        ):
            raise RuntimeError(f"Insufficient resources for pipeline: {pipeline_name}")
        
        try:
            # Create pipeline instance if needed
            if pipeline_name not in self.pipeline_instances:
                self._create_pipeline_instance(pipeline_name)
            
            pipeline = self.pipeline_instances[pipeline_name]
            
            # Monitor pipeline
            if self.monitor:
                self.monitor.start_pipeline_monitoring(pipeline_name)
            
            # Run pipeline
            logger.info("Running pipeline", pipeline=pipeline_name)
            result = pipeline.run()
            
            self._completed_pipelines.add(pipeline_name)
            
            return result
            
        except Exception as e:
            self._failed_pipelines.add(pipeline_name)
            logger.error(
                "Pipeline execution failed",
                pipeline=pipeline_name,
                error=str(e)
            )
            raise
        finally:
            # Release resources
            self.resource_manager.release(pipeline_name)
            
            # Stop monitoring
            if self.monitor:
                self.monitor.stop_pipeline_monitoring(pipeline_name)
    
    def pause_all(self) -> None:
        """Pause all running pipelines."""
        with self._lock:
            for pipeline in self.pipeline_instances.values():
                if pipeline.state == PipelineState.RUNNING:
                    pipeline.pause()
            logger.info("All pipelines paused")
    
    def resume_all(self) -> None:
        """Resume all paused pipelines."""
        with self._lock:
            for pipeline in self.pipeline_instances.values():
                if pipeline.state == PipelineState.PAUSED:
                    pipeline.resume()
            logger.info("All pipelines resumed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        with self._lock:
            return {
                "name": self.name,
                "execution_mode": self.execution_mode.value,
                "total_pipelines": len(self.pipelines),
                "completed_pipelines": len(self._completed_pipelines),
                "failed_pipelines": len(self._failed_pipelines),
                "running_pipelines": len([
                    p for p in self.pipeline_instances.values()
                    if p.state == PipelineState.RUNNING
                ]),
                "resource_usage": self.resource_manager.allocated_resources,
                "pipeline_states": {
                    name: instance.state.value
                    for name, instance in self.pipeline_instances.items()
                }
            }
    
    def _resolve_execution_order(
        self,
        target_pipelines: Optional[List[str]] = None
    ) -> List[str]:
        """Resolve pipeline execution order using topological sort."""
        if target_pipelines:
            # Get all dependencies of target pipelines
            required_pipelines = set()
            for target in target_pipelines:
                if target not in self.pipelines:
                    raise ValueError(f"Unknown pipeline: {target}")
                required_pipelines.add(target)
                required_pipelines.update(
                    nx.ancestors(self.dependency_graph, target)
                )
            
            # Create subgraph
            subgraph = self.dependency_graph.subgraph(required_pipelines)
        else:
            subgraph = self.dependency_graph
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(subgraph):
            cycles = list(nx.simple_cycles(subgraph))
            raise ValueError(f"Circular dependencies detected: {cycles}")
        
        # Topological sort
        execution_order = list(nx.topological_sort(subgraph))
        
        # Sort by priority within dependency constraints
        def priority_key(pipeline_name):
            return (
                -self.pipelines[pipeline_name].priority,
                execution_order.index(pipeline_name)
            )
        
        return sorted(execution_order, key=priority_key)
    
    def _create_pipeline_instances(self, pipeline_names: List[str]) -> None:
        """Create instances for all pipelines to be executed."""
        for name in pipeline_names:
            self._create_pipeline_instance(name)
    
    def _create_pipeline_instance(self, pipeline_name: str) -> None:
        """Create a single pipeline instance."""
        definition = self.pipelines[pipeline_name]
        
        # Merge config with defaults
        config = {
            "name": pipeline_name,
            "checkpoint_dir": self.checkpoint_dir / pipeline_name,
            **definition.config
        }
        
        # Create instance
        self.pipeline_instances[pipeline_name] = definition.pipeline_class(**config)
        
        logger.debug(
            "Pipeline instance created",
            pipeline=pipeline_name,
            class_name=definition.pipeline_class.__name__
        )
    
    def _execute_sequential(
        self,
        execution_order: List[str]
    ) -> Dict[str, PipelineContext]:
        """Execute pipelines sequentially."""
        results = {}
        
        for pipeline_name in execution_order:
            try:
                results[pipeline_name] = self.run_pipeline(pipeline_name)
            except Exception as e:
                logger.error(
                    "Pipeline failed in sequential execution",
                    pipeline=pipeline_name,
                    error=str(e)
                )
                
                # Check if we should continue
                if not self._should_continue_after_failure(pipeline_name):
                    raise
        
        return results
    
    def _execute_parallel(
        self,
        execution_order: List[str]
    ) -> Dict[str, PipelineContext]:
        """Execute pipelines in parallel with dependency resolution."""
        results = {}
        completed = set()
        
        # Create executor
        if self.execution_mode == ExecutionMode.DISTRIBUTED:
            self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        try:
            while len(completed) < len(execution_order):
                # Find pipelines ready to execute
                ready = self._find_ready_pipelines(
                    execution_order,
                    completed,
                    set(self._futures.keys())
                )
                
                # Submit ready pipelines
                for pipeline_name in ready:
                    future = self._executor.submit(
                        self.run_pipeline,
                        pipeline_name
                    )
                    self._futures[pipeline_name] = future
                    
                    logger.debug(
                        "Pipeline submitted for execution",
                        pipeline=pipeline_name
                    )
                
                # Wait for completions
                if self._futures:
                    done_futures = []
                    for name, future in self._futures.items():
                        if future.done():
                            done_futures.append(name)
                            try:
                                results[name] = future.result()
                                completed.add(name)
                            except Exception as e:
                                logger.error(
                                    "Pipeline failed in parallel execution",
                                    pipeline=name,
                                    error=str(e)
                                )
                                if not self._should_continue_after_failure(name):
                                    raise
                    
                    # Remove completed futures
                    for name in done_futures:
                        del self._futures[name]
                
                # Small sleep to avoid busy waiting
                if self._futures and not done_futures:
                    threading.Event().wait(0.1)
            
            return results
            
        finally:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def _find_ready_pipelines(
        self,
        execution_order: List[str],
        completed: Set[str],
        running: Set[str]
    ) -> List[str]:
        """Find pipelines ready for execution."""
        ready = []
        
        for pipeline_name in execution_order:
            if pipeline_name in completed or pipeline_name in running:
                continue
            
            # Check dependencies
            dependencies = set(self.dependency_graph.predecessors(pipeline_name))
            if dependencies.issubset(completed):
                # Check resource availability
                requirements = self.pipelines[pipeline_name].resource_requirements
                if self.resource_manager.allocate(pipeline_name, requirements):
                    ready.append(pipeline_name)
                else:
                    # Release allocation since we're not running yet
                    self.resource_manager.release(pipeline_name)
        
        return ready
    
    def _should_continue_after_failure(self, failed_pipeline: str) -> bool:
        """Determine if orchestration should continue after a pipeline failure."""
        # Check if any pending pipelines depend on the failed one
        for pipeline_name in self.pipelines:
            if pipeline_name in self._completed_pipelines:
                continue
            
            dependencies = set(self.dependency_graph.predecessors(pipeline_name))
            if failed_pipeline in dependencies:
                logger.warning(
                    "Skipping pipeline due to failed dependency",
                    pipeline=pipeline_name,
                    failed_dependency=failed_pipeline
                )
                return True
        
        return True
    
    def _save_orchestration_state(self, result: OrchestrationResult) -> None:
        """Save orchestration state for recovery."""
        state_file = self.checkpoint_dir / f"orchestration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        state = {
            "name": self.name,
            "timestamp": datetime.now().isoformat(),
            "execution_mode": self.execution_mode.value,
            "total_pipelines": result.total_pipelines,
            "successful_pipelines": result.successful_pipelines,
            "failed_pipelines": result.failed_pipelines,
            "duration": result.duration.total_seconds(),
            "execution_order": result.execution_order,
            "pipeline_states": {
                name: instance.state.value
                for name, instance in self.pipeline_instances.items()
            }
        }
        
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
        
        logger.debug(
            "Orchestration state saved",
            file=str(state_file)
        )
    
    def _cleanup(self) -> None:
        """Clean up resources after orchestration."""
        # Clear instance tracking
        self._completed_pipelines.clear()
        self._failed_pipelines.clear()
        self._futures.clear()
        
        # Clean up pipeline instances
        for pipeline in self.pipeline_instances.values():
            if hasattr(pipeline, "cleanup_checkpoints"):
                pipeline.cleanup_checkpoints()
        
        self.pipeline_instances.clear()
        
        logger.debug("Orchestration cleanup completed")