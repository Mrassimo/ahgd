"""
Validation Orchestrator

This module provides a coordinated validation pipeline that manages rule dependency,
validation reporting, performance monitoring, and parallel validation execution
across all validation components in the AHGD data quality framework.
"""

import logging
import asyncio
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
import time
import json
import hashlib

from ..utils.interfaces import (
    DataBatch, 
    DataRecord, 
    ValidationResult, 
    ValidationSeverity,
    ProcessingStatus,
    DataQualityError
)
from .base import BaseValidator
from .quality_checker import QualityChecker
from .business_rules import AustralianHealthBusinessRulesValidator
from .statistical_validator import StatisticalValidator
from .advanced_statistical import AdvancedStatisticalValidator
from .geographic_validator import GeographicValidator
from .enhanced_geographic import EnhancedGeographicValidator


@dataclass
class ValidationTask:
    """Individual validation task definition."""
    task_id: str
    validator_class: str
    validator_config: Dict[str, Any]
    data: DataBatch
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    results: List[ValidationResult] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ValidationPipelineConfig:
    """Configuration for validation pipeline."""
    enable_parallel_execution: bool = True
    max_workers: int = 4
    timeout_seconds: int = 300
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    enable_dependency_management: bool = True
    enable_performance_monitoring: bool = True
    batch_size: int = 1000
    retry_failed_validations: bool = True
    max_retries: int = 2


@dataclass
class ValidationPipelineResult:
    """Result of validation pipeline execution."""
    pipeline_id: str
    status: ProcessingStatus
    total_validations: int
    successful_validations: int
    failed_validations: int
    total_records_processed: int
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    validation_results: List[ValidationResult] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    error_summary: Dict[str, int] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for validation execution."""
    validator_name: str
    execution_time_seconds: float
    records_processed: int
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    cache_hit_rate: Optional[float] = None


class ValidationOrchestrator:
    """
    Coordinated validation pipeline orchestrator.
    
    This class manages the execution of multiple validators in a coordinated fashion,
    handling dependencies, parallel execution, caching, performance monitoring,
    and comprehensive reporting.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the validation orchestrator.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Pipeline configuration
        self.pipeline_config = ValidationPipelineConfig(**self.config.get('pipeline', {}))
        
        # Validator registry
        self.validator_registry: Dict[str, type] = {
            'quality_checker': QualityChecker,
            'business_rules': AustralianHealthBusinessRulesValidator,
            'statistical_validator': StatisticalValidator,
            'geographic_validator': GeographicValidator,
            'enhanced_geographic_validator': EnhancedGeographicValidator
        }
        
        # Active validator instances
        self.active_validators: Dict[str, BaseValidator] = {}
        
        # Validation cache
        self.validation_cache: Dict[str, Tuple[List[ValidationResult], datetime]] = {}
        
        # Performance monitoring
        self.performance_metrics: List[PerformanceMetrics] = []
        
        # Dependency graph
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Task execution tracking
        self.task_queue: deque = deque()
        self.completed_tasks: Dict[str, ValidationTask] = {}
        self.failed_tasks: Dict[str, ValidationTask] = {}
        
        # Thread pool for parallel execution
        self.executor: Optional[ThreadPoolExecutor] = None
        
        # Pipeline statistics
        self.pipeline_statistics = defaultdict(int)
        
    def register_validator(self, name: str, validator_class: type) -> None:
        """
        Register a custom validator.
        
        Args:
            name: Validator name
            validator_class: Validator class
        """
        self.validator_registry[name] = validator_class
        self.logger.info(f"Registered validator: {name}")
    
    def create_validation_pipeline(
        self, 
        data: DataBatch,
        validators_config: List[Dict[str, Any]],
        pipeline_id: Optional[str] = None
    ) -> str:
        """
        Create a validation pipeline with specified validators.
        
        Args:
            data: Data to validate
            validators_config: List of validator configurations
            pipeline_id: Optional pipeline identifier
            
        Returns:
            str: Pipeline identifier
        """
        if not pipeline_id:
            pipeline_id = self._generate_pipeline_id(data)
        
        # Clear previous pipeline state
        self.task_queue.clear()
        self.completed_tasks.clear()
        self.failed_tasks.clear()
        self.dependency_graph.clear()
        
        # Create validation tasks
        for validator_config in validators_config:
            task = self._create_validation_task(data, validator_config)
            self.task_queue.append(task)
            
            # Build dependency graph
            dependencies = validator_config.get('dependencies', [])
            self.dependency_graph[task.task_id] = set(dependencies)
        
        self.logger.info(f"Created validation pipeline {pipeline_id} with {len(self.task_queue)} tasks")
        return pipeline_id
    
    def execute_pipeline(self, pipeline_id: str) -> ValidationPipelineResult:
        """
        Execute the validation pipeline.
        
        Args:
            pipeline_id: Pipeline identifier
            
        Returns:
            ValidationPipelineResult: Pipeline execution results
        """
        start_time = datetime.now()
        
        pipeline_result = ValidationPipelineResult(
            pipeline_id=pipeline_id,
            status=ProcessingStatus.RUNNING,
            total_validations=len(self.task_queue),
            successful_validations=0,
            failed_validations=0,
            total_records_processed=0,
            start_time=start_time
        )
        
        try:
            if self.pipeline_config.enable_parallel_execution:
                self._execute_pipeline_parallel(pipeline_result)
            else:
                self._execute_pipeline_sequential(pipeline_result)
            
            # Finalise pipeline result
            pipeline_result.end_time = datetime.now()
            pipeline_result.duration_seconds = (
                pipeline_result.end_time - pipeline_result.start_time
            ).total_seconds()
            pipeline_result.status = ProcessingStatus.COMPLETED
            
            # Collect all validation results
            all_results = []
            for task in self.completed_tasks.values():
                all_results.extend(task.results)
            
            pipeline_result.validation_results = all_results
            pipeline_result.performance_metrics = self._collect_performance_metrics()
            pipeline_result.error_summary = self._generate_error_summary(all_results)
            
            self.logger.info(
                f"Pipeline {pipeline_id} completed in {pipeline_result.duration_seconds:.2f}s: "
                f"{pipeline_result.successful_validations} successful, "
                f"{pipeline_result.failed_validations} failed"
            )
            
        except Exception as e:
            pipeline_result.status = ProcessingStatus.FAILED
            pipeline_result.end_time = datetime.now()
            pipeline_result.duration_seconds = (
                pipeline_result.end_time - pipeline_result.start_time
            ).total_seconds()
            
            self.logger.error(f"Pipeline {pipeline_id} failed: {e}")
            raise DataQualityError(f"Validation pipeline failed: {e}")
        
        return pipeline_result
    
    def validate_data(
        self, 
        data: DataBatch,
        validator_names: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[ValidationResult]:
        """
        Validate data using specified validators.
        
        Args:
            data: Data to validate
            validator_names: List of validator names to use (all if None)
            use_cache: Whether to use validation cache
            
        Returns:
            List[ValidationResult]: Validation results
        """
        if not validator_names:
            validator_names = list(self.validator_registry.keys())
        
        # Check cache first
        if use_cache and self.pipeline_config.enable_caching:
            cache_key = self._generate_cache_key(data, validator_names)
            cached_results = self._get_cached_results(cache_key)
            if cached_results:
                self.logger.info(f"Using cached validation results for {len(data)} records")
                return cached_results
        
        # Create validation configuration
        validators_config = []
        for validator_name in validator_names:
            validators_config.append({
                'validator': validator_name,
                'config': self.config.get(f'{validator_name}_config', {}),
                'priority': self._get_validator_priority(validator_name),
                'dependencies': self._get_validator_dependencies(validator_name)
            })
        
        # Execute pipeline
        pipeline_id = self.create_validation_pipeline(data, validators_config)
        pipeline_result = self.execute_pipeline(pipeline_id)
        
        # Cache results
        if use_cache and self.pipeline_config.enable_caching:
            cache_key = self._generate_cache_key(data, validator_names)
            self._cache_results(cache_key, pipeline_result.validation_results)
        
        return pipeline_result.validation_results
    
    def get_validator_dependencies(self, validator_name: str) -> List[str]:
        """
        Get dependencies for a validator.
        
        Args:
            validator_name: Name of the validator
            
        Returns:
            List[str]: List of dependency validator names
        """
        # Define default dependencies
        default_dependencies = {
            'statistical_validator': ['quality_checker'],
            'business_rules': ['quality_checker'],
            'geographic_validator': ['business_rules'],
        }
        
        return default_dependencies.get(validator_name, [])
    
    def clear_cache(self) -> None:
        """Clear the validation cache."""
        self.validation_cache.clear()
        self.logger.info("Validation cache cleared")
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        if not self.performance_metrics:
            return {}
        
        stats = {
            'total_executions': len(self.performance_metrics),
            'average_execution_time': sum(m.execution_time_seconds for m in self.performance_metrics) / len(self.performance_metrics),
            'total_records_processed': sum(m.records_processed for m in self.performance_metrics),
            'validator_performance': {}
        }
        
        # Per-validator statistics
        validator_metrics = defaultdict(list)
        for metric in self.performance_metrics:
            validator_metrics[metric.validator_name].append(metric)
        
        for validator_name, metrics in validator_metrics.items():
            stats['validator_performance'][validator_name] = {
                'executions': len(metrics),
                'average_time': sum(m.execution_time_seconds for m in metrics) / len(metrics),
                'total_records': sum(m.records_processed for m in metrics),
                'records_per_second': sum(m.records_processed for m in metrics) / sum(m.execution_time_seconds for m in metrics)
            }
        
        return stats
    
    def shutdown(self) -> None:
        """Shutdown the orchestrator and clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        self.clear_cache()
        self.logger.info("Validation orchestrator shutdown complete")
    
    # Private methods
    
    def _create_validation_task(self, data: DataBatch, validator_config: Dict[str, Any]) -> ValidationTask:
        """Create a validation task from configuration."""
        validator_name = validator_config['validator']
        task_id = f"{validator_name}_{self._generate_task_id()}"
        
        return ValidationTask(
            task_id=task_id,
            validator_class=validator_name,
            validator_config=validator_config.get('config', {}),
            data=data,
            priority=validator_config.get('priority', 1),
            dependencies=validator_config.get('dependencies', [])
        )
    
    def _execute_pipeline_parallel(self, pipeline_result: ValidationPipelineResult) -> None:
        """Execute pipeline with parallel processing."""
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=self.pipeline_config.max_workers)
        
        # Track running tasks
        running_tasks: Dict[str, Future] = {}
        ready_tasks: List[ValidationTask] = []
        
        # Initial ready tasks (no dependencies)
        for task in list(self.task_queue):
            if not task.dependencies:
                ready_tasks.append(task)
                self.task_queue.remove(task)
        
        while ready_tasks or running_tasks or self.task_queue:
            # Submit ready tasks
            while ready_tasks and len(running_tasks) < self.pipeline_config.max_workers:
                task = ready_tasks.pop(0)
                future = self.executor.submit(self._execute_validation_task, task)
                running_tasks[task.task_id] = future
                task.status = ProcessingStatus.RUNNING
                task.start_time = datetime.now()
            
            # Check for completed tasks
            completed_futures = []
            for task_id, future in running_tasks.items():
                if future.done():
                    completed_futures.append(task_id)
            
            # Process completed tasks
            for task_id in completed_futures:
                future = running_tasks.pop(task_id)
                task = self._find_task_by_id(task_id)
                
                try:
                    results = future.result(timeout=1)
                    task.results = results
                    task.status = ProcessingStatus.COMPLETED
                    task.end_time = datetime.now()
                    self.completed_tasks[task_id] = task
                    pipeline_result.successful_validations += 1
                    
                    # Check for newly ready tasks
                    newly_ready = self._find_ready_tasks(task_id)
                    ready_tasks.extend(newly_ready)
                    
                except Exception as e:
                    task.status = ProcessingStatus.FAILED
                    task.end_time = datetime.now()
                    task.error_message = str(e)
                    self.failed_tasks[task_id] = task
                    pipeline_result.failed_validations += 1
                    
                    self.logger.error(f"Task {task_id} failed: {e}")
            
            # Brief sleep to avoid busy waiting
            if running_tasks:
                time.sleep(0.1)
    
    def _execute_pipeline_sequential(self, pipeline_result: ValidationPipelineResult) -> None:
        """Execute pipeline sequentially."""
        # Topological sort of tasks based on dependencies
        sorted_tasks = self._topological_sort_tasks()
        
        for task in sorted_tasks:
            try:
                task.status = ProcessingStatus.RUNNING
                task.start_time = datetime.now()
                
                results = self._execute_validation_task(task)
                
                task.results = results
                task.status = ProcessingStatus.COMPLETED
                task.end_time = datetime.now()
                self.completed_tasks[task.task_id] = task
                pipeline_result.successful_validations += 1
                
            except Exception as e:
                task.status = ProcessingStatus.FAILED
                task.end_time = datetime.now()
                task.error_message = str(e)
                self.failed_tasks[task.task_id] = task
                pipeline_result.failed_validations += 1
                
                self.logger.error(f"Task {task.task_id} failed: {e}")
                
                # Decide whether to continue or fail fast
                if not self.pipeline_config.retry_failed_validations:
                    break
    
    def _execute_validation_task(self, task: ValidationTask) -> List[ValidationResult]:
        """Execute a single validation task."""
        start_time = time.time()
        
        # Get or create validator instance
        validator = self._get_validator_instance(task.validator_class, task.validator_config)
        
        # Execute validation
        results = validator.validate(task.data)
        
        # Record performance metrics
        execution_time = time.time() - start_time
        self.performance_metrics.append(PerformanceMetrics(
            validator_name=task.validator_class,
            execution_time_seconds=execution_time,
            records_processed=len(task.data)
        ))
        
        return results
    
    def _get_validator_instance(self, validator_class: str, config: Dict[str, Any]) -> BaseValidator:
        """Get or create validator instance."""
        if validator_class not in self.active_validators:
            if validator_class not in self.validator_registry:
                raise ValueError(f"Unknown validator: {validator_class}")
            
            validator_type = self.validator_registry[validator_class]
            self.active_validators[validator_class] = validator_type(
                validator_id=validator_class,
                config=config,
                logger=self.logger
            )
        
        return self.active_validators[validator_class]
    
    def _find_task_by_id(self, task_id: str) -> Optional[ValidationTask]:
        """Find task by ID."""
        # Check all task collections
        all_tasks = list(self.task_queue) + list(self.completed_tasks.values()) + list(self.failed_tasks.values())
        
        for task in all_tasks:
            if task.task_id == task_id:
                return task
        
        return None
    
    def _find_ready_tasks(self, completed_task_id: str) -> List[ValidationTask]:
        """Find tasks that are ready to run after a task completes."""
        ready_tasks = []
        
        for task in list(self.task_queue):
            if completed_task_id in task.dependencies:
                task.dependencies.remove(completed_task_id)
                
                if not task.dependencies:  # All dependencies satisfied
                    ready_tasks.append(task)
                    self.task_queue.remove(task)
        
        return ready_tasks
    
    def _topological_sort_tasks(self) -> List[ValidationTask]:
        """Topologically sort tasks based on dependencies."""
        # Simple topological sort implementation
        sorted_tasks = []
        remaining_tasks = list(self.task_queue)
        
        while remaining_tasks:
            # Find tasks with no dependencies
            ready_tasks = [task for task in remaining_tasks if not task.dependencies]
            
            if not ready_tasks:
                # Circular dependency or missing dependency
                self.logger.warning("Circular dependency detected or missing dependencies")
                break
            
            # Process ready tasks
            for task in ready_tasks:
                sorted_tasks.append(task)
                remaining_tasks.remove(task)
                
                # Remove this task from other tasks' dependencies
                for other_task in remaining_tasks:
                    if task.task_id in other_task.dependencies:
                        other_task.dependencies.remove(task.task_id)
        
        return sorted_tasks
    
    def _get_validator_priority(self, validator_name: str) -> int:
        """Get priority for validator (lower number = higher priority)."""
        priorities = {
            'quality_checker': 1,
            'business_rules': 2,
            'geographic_validator': 3,
            'statistical_validator': 4
        }
        return priorities.get(validator_name, 5)
    
    def _get_validator_dependencies(self, validator_name: str) -> List[str]:
        """Get dependencies for validator."""
        dependencies = {
            'statistical_validator': ['quality_checker'],
            'business_rules': [],  # Can run independently
            'geographic_validator': []  # Can run independently
        }
        return dependencies.get(validator_name, [])
    
    def _generate_pipeline_id(self, data: DataBatch) -> str:
        """Generate unique pipeline ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        data_hash = hashlib.md5(str(len(data)).encode()).hexdigest()[:8]
        return f"pipeline_{timestamp}_{data_hash}"
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        return str(int(time.time() * 1000))
    
    def _generate_cache_key(self, data: DataBatch, validator_names: List[str]) -> str:
        """Generate cache key for validation results."""
        data_sample = str(data[:min(10, len(data))])
        data_hash = hashlib.md5(data_sample.encode()).hexdigest()
        validators_hash = hashlib.md5('_'.join(sorted(validator_names)).encode()).hexdigest()
        return f"{data_hash}_{validators_hash}_{len(data)}"
    
    def _get_cached_results(self, cache_key: str) -> Optional[List[ValidationResult]]:
        """Get cached validation results."""
        if cache_key in self.validation_cache:
            results, timestamp = self.validation_cache[cache_key]
            
            # Check if cache is still valid
            if datetime.now() - timestamp < timedelta(minutes=self.pipeline_config.cache_ttl_minutes):
                return results
            else:
                # Remove expired cache entry
                del self.validation_cache[cache_key]
        
        return None
    
    def _cache_results(self, cache_key: str, results: List[ValidationResult]) -> None:
        """Cache validation results."""
        self.validation_cache[cache_key] = (results, datetime.now())
        
        # Clean up old cache entries if cache is getting large
        if len(self.validation_cache) > 100:
            self._cleanup_cache()
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        cutoff_time = datetime.now() - timedelta(minutes=self.pipeline_config.cache_ttl_minutes)
        
        expired_keys = [
            key for key, (_, timestamp) in self.validation_cache.items()
            if timestamp < cutoff_time
        ]
        
        for key in expired_keys:
            del self.validation_cache[key]
        
        self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from completed tasks."""
        metrics = {
            'total_execution_time': 0,
            'validator_metrics': {},
            'task_metrics': []
        }
        
        for task in self.completed_tasks.values():
            if task.start_time and task.end_time:
                duration = (task.end_time - task.start_time).total_seconds()
                metrics['total_execution_time'] += duration
                
                metrics['task_metrics'].append({
                    'task_id': task.task_id,
                    'validator': task.validator_class,
                    'duration_seconds': duration,
                    'records_processed': len(task.data),
                    'results_count': len(task.results)
                })
                
                # Aggregate by validator
                if task.validator_class not in metrics['validator_metrics']:
                    metrics['validator_metrics'][task.validator_class] = {
                        'total_time': 0,
                        'total_records': 0,
                        'executions': 0
                    }
                
                validator_metrics = metrics['validator_metrics'][task.validator_class]
                validator_metrics['total_time'] += duration
                validator_metrics['total_records'] += len(task.data)
                validator_metrics['executions'] += 1
        
        return metrics
    
    def _generate_error_summary(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Generate error summary from validation results."""
        summary = {
            'total_issues': len(results),
            'errors': 0,
            'warnings': 0,
            'info': 0,
            'by_rule': defaultdict(int)
        }
        
        for result in results:
            summary['by_rule'][result.rule_id] += 1
            
            if result.severity == ValidationSeverity.ERROR:
                summary['errors'] += 1
            elif result.severity == ValidationSeverity.WARNING:
                summary['warnings'] += 1
            else:
                summary['info'] += 1
        
        summary['by_rule'] = dict(summary['by_rule'])
        return summary