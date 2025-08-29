"""
Master ETL pipeline orchestrator for complete end-to-end data processing.

This module provides comprehensive ETL pipeline orchestration with integrated
validation, quality assurance, and data flow control throughout the entire
data processing lifecycle.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
import polars as pl
import duckdb

from .base_pipeline import BasePipeline, PipelineContext, StageResult, StageState, AHGDException, PipelineError
from .validation_pipeline import (
    ValidationPipeline, StageValidator, QualityGatekeeper,
    ValidationAction, ValidationMode, StageValidationResult
)
from ..extractors import ExtractorRegistry
from ..transformers.base import BaseTransformer
from ..transformers.geographic_standardiser import GeographicStandardiser
from ..transformers.data_integrator import MasterDataIntegrator
from ..loaders.base import BaseLoader
from ..validators import ValidationOrchestrator, ValidationReporter
from ..utils.logging import get_logger, monitor_performance, track_lineage
from ..utils.config import get_config
from ..utils.interfaces import (
    ExtractionError, TransformationError, ValidationError, LoadingError, AHGDException
)


logger = get_logger(__name__)


class PipelineStageType(str, Enum):
    """Types of pipeline stages."""
    EXTRACTION = "extraction"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    INTEGRATION = "integration"
    LOADING = "loading"
    QUALITY_ASSURANCE = "quality_assurance"


class DataFlowStatus(str, Enum):
    """Data flow status between stages."""
    PENDING = "pending"
    FLOWING = "flowing"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


class QualityLevel(str, Enum):
    """Quality assurance levels."""
    MINIMAL = "minimal"     # Basic validation only
    STANDARD = "standard"   # Standard validation suite
    COMPREHENSIVE = "comprehensive"  # Full validation framework
    AUDIT = "audit"         # Audit-level validation


@dataclass
class PipelineStageDefinition:
    """Definition of a pipeline stage."""
    stage_id: str
    stage_name: str
    stage_type: PipelineStageType
    stage_class: str  # Fully qualified class name
    dependencies: List[str] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    validation_required: bool = True
    quality_level: QualityLevel = QualityLevel.STANDARD
    timeout_seconds: int = 1800  # 30 minutes default
    retry_attempts: int = 3
    parallel_capable: bool = False


@dataclass
class DataFlowCheckpoint:
    """Data flow checkpoint between stages."""
    checkpoint_id: str
    source_stage: str
    target_stage: str
    data_schema: Optional[str] = None
    validation_rules: List[str] = field(default_factory=list)
    transformation_rules: List[str] = field(default_factory=list)
    quality_threshold: float = 95.0


@dataclass
class QualityAssuranceConfig:
    """Quality assurance configuration."""
    enabled: bool = True
    quality_level: QualityLevel = QualityLevel.STANDARD
    validation_mode: ValidationMode = ValidationMode.SELECTIVE
    halt_on_critical_errors: bool = True
    generate_quality_reports: bool = True
    monitor_performance: bool = True
    track_data_lineage: bool = True
    compliance_standards: List[str] = field(default_factory=list)


class PipelineStageManager:
    """
    Manages pipeline stages with validation integration.
    
    Orchestrates the execution of individual pipeline stages while
    ensuring proper validation, error handling, and quality assurance.
    """
    
    def __init__(
        self,
        stage_definitions: List[PipelineStageDefinition],
        quality_config: Optional[QualityAssuranceConfig] = None
    ):
        """
        Initialise pipeline stage manager.
        
        Args:
            stage_definitions: List of stage definitions
            quality_config: Quality assurance configuration
        """
        self.stage_definitions = {stage.stage_id: stage for stage in stage_definitions}
        self.quality_config = quality_config or QualityAssuranceConfig()
        
        # Initialise stage instances
        self.stage_instances: Dict[str, Any] = {}
        self.stage_validators: Dict[str, StageValidator] = {}
        
        # Execution tracking
        self.stage_execution_order: List[str] = []
        self.stage_results: Dict[str, StageResult] = {}
        self.data_flow_status: Dict[str, DataFlowStatus] = {}
        
        self._build_execution_order()
        self._initialise_stage_instances()
        
        logger.info(
            "Pipeline stage manager initialised",
            stages_count=len(self.stage_definitions),
            quality_level=self.quality_config.quality_level.value
        )
    
    @monitor_performance("stage_execution")
    def execute_stage(
        self,
        stage_id: str,
        context: PipelineContext,
        con: duckdb.DuckDBPyConnection,
        input_table: Optional[str] = None
    ) -> Tuple[str, StageResult]:
        """
        Execute a specific pipeline stage.
        
        Args:
            stage_id: Stage identifier
            context: Pipeline context
            con: DuckDB connection
            input_table: Input table name for the stage
            
        Returns:
            Tuple of (output_table_name, stage_result)
        """
        if stage_id not in self.stage_definitions:
            raise PipelineError(f"Unknown stage: {stage_id}")
        
        stage_def = self.stage_definitions[stage_id]
        start_time = datetime.now()
        output_table = f"{stage_id}_output"

        logger.info(
            "Executing pipeline stage",
            stage_id=stage_id,
            stage_type=stage_def.stage_type.value
        )
        
        try:
            input_data = None
            if input_table:
                input_data = con.table(input_table).pl()

            # Pre-stage validation
            if stage_def.validation_required and input_data is not None:
                validation_result = self._validate_stage_input(stage_id, input_data, context)
                if not validation_result.gate_status.value == "passed":
                    if self.quality_config.halt_on_critical_errors:
                        raise ValidationError(f"Pre-stage validation failed for {stage_id}")
            
            # Execute stage
            stage_instance = self.stage_instances[stage_id]
            output_data = self._execute_stage_instance(stage_instance, input_data, context)
            
            # Write output to duckdb
            if output_data is not None:
                con.register(output_table, output_data)

            # Post-stage validation
            if stage_def.validation_required and output_data is not None:
                validation_result = self._validate_stage_output(stage_id, output_data, context)
                if not validation_result.gate_status.value == "passed":
                    if self.quality_config.halt_on_critical_errors:
                        raise ValidationError(f"Post-stage validation failed for {stage_id}")
            
            # Create successful result
            stage_result = StageResult(
                stage_name=stage_id,
                state=StageState.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                output=output_table,
                metrics={
                    "input_records": len(input_data) if input_data is not None else 0,
                    "output_records": len(output_data) if output_data is not None else 0,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            )
            
            self.stage_results[stage_id] = stage_result
            self.data_flow_status[stage_id] = DataFlowStatus.COMPLETED
            
            # Track data lineage
            if self.quality_config.track_data_lineage:
                track_lineage(f"{stage_id}_input", f"{stage_id}_output", stage_def.stage_type.value)
            
            logger.info(
                "Stage execution completed",
                stage_id=stage_id,
                duration=stage_result.duration,
                output_table=output_table
            )
            
            return output_table, stage_result
            
        except Exception as e:
            # Create failed result
            stage_result = StageResult(
                stage_name=stage_id,
                state=StageState.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                error=e,
                metrics={
                    "input_records": len(input_data) if input_data is not None else 0,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            )
            
            self.stage_results[stage_id] = stage_result
            self.data_flow_status[stage_id] = DataFlowStatus.FAILED
            
            logger.error(
                "Stage execution failed",
                stage_id=stage_id,
                error=str(e),
                duration=stage_result.duration
            )
            
            raise PipelineError(f"Stage {stage_id} failed: {str(e)}") from e
    
    def get_execution_plan(self) -> List[Dict[str, Any]]:
        """Get the execution plan for all stages."""
        return [
            {
                "stage_id": stage_id,
                "stage_name": self.stage_definitions[stage_id].stage_name,
                "stage_type": self.stage_definitions[stage_id].stage_type.value,
                "dependencies": self.stage_definitions[stage_id].dependencies,
                "parallel_capable": self.stage_definitions[stage_id].parallel_capable
            }
            for stage_id in self.stage_execution_order
        ]
    
    def _build_execution_order(self) -> None:
        """Build stage execution order based on dependencies."""
        # Topological sort of stages based on dependencies
        visited = set()
        temp_visited = set()
        execution_order = []
        
        def visit(stage_id: str) -> None:
            if stage_id in temp_visited:
                raise PipelineError(f"Circular dependency detected involving stage: {stage_id}")
            if stage_id in visited:
                return
            
            temp_visited.add(stage_id)
            
            # Visit dependencies first
            for dep_id in self.stage_definitions[stage_id].dependencies:
                if dep_id not in self.stage_definitions:
                    raise PipelineError(f"Unknown dependency {dep_id} for stage {stage_id}")
                visit(dep_id)
            
            temp_visited.remove(stage_id)
            visited.add(stage_id)
            execution_order.append(stage_id)
        
        # Visit all stages
        for stage_id in self.stage_definitions:
            if stage_id not in visited:
                visit(stage_id)
        
        self.stage_execution_order = execution_order
        
        logger.info(
            "Stage execution order determined",
            execution_order=self.stage_execution_order
        )
    
    def _initialise_stage_instances(self) -> None:
        """Initialise stage instances."""
        for stage_id, stage_def in self.stage_definitions.items():
            try:
                # Import and instantiate stage class
                module_path, class_name = stage_def.stage_class.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                stage_class = getattr(module, class_name)
                
                # Create instance with configuration
                stage_instance = stage_class(**stage_def.configuration)
                self.stage_instances[stage_id] = stage_instance
                
                # Initialise validator if required
                if stage_def.validation_required:
                    self._initialise_stage_validator(stage_id, stage_def)
                
                logger.debug(
                    "Stage instance initialised",
                    stage_id=stage_id,
                    stage_class=stage_def.stage_class
                )
                
            except Exception as e:
                raise PipelineError(f"Failed to initialise stage {stage_id}: {str(e)}") from e
    
    def _initialise_stage_validator(self, stage_id: str, stage_def: PipelineStageDefinition) -> None:
        """Initialise validator for a stage."""
        try:
            from .validation_pipeline import QualityGate, ValidationRule
            
            # Create validation rules based on stage type and quality level
            validation_rules = self._create_validation_rules(stage_def)
            
            # Create quality gate
            quality_gate = QualityGate(
                gate_id=f"{stage_id}_gate",
                gate_name=f"{stage_def.stage_name} Quality Gate",
                stage_name=stage_id,
                validation_rules=validation_rules,
                mode=self.quality_config.validation_mode
            )
            
            # Create stage validator
            validator = StageValidator(
                stage_name=stage_id,
                quality_gate=quality_gate
            )
            
            self.stage_validators[stage_id] = validator
            
        except Exception as e:
            logger.error(
                "Failed to initialise stage validator",
                stage_id=stage_id,
                error=str(e)
            )
    
    def _create_validation_rules(self, stage_def: PipelineStageDefinition) -> List:
        """Create validation rules based on stage definition."""
        from .validation_pipeline import ValidationRule
        
        rules = []
        
        # Add rules based on stage type
        if stage_def.stage_type == PipelineStageType.EXTRACTION:
            rules.extend([
                ValidationRule(
                    rule_id=f"{stage_def.stage_id}_schema_validation",
                    rule_name="Schema Validation",
                    rule_type="schema",
                    severity="critical",
                    action=ValidationAction.HALT
                ),
                ValidationRule(
                    rule_id=f"{stage_def.stage_id}_completeness_check",
                    rule_name="Data Completeness",
                    rule_type="business",
                    severity="high",
                    action=ValidationAction.WARNING,
                    threshold=95.0
                )
            ])
        
        elif stage_def.stage_type == PipelineStageType.TRANSFORMATION:
            rules.extend([
                ValidationRule(
                    rule_id=f"{stage_def.stage_id}_transformation_integrity",
                    rule_name="Transformation Integrity",
                    rule_type="business",
                    severity="critical",
                    action=ValidationAction.HALT
                ),
                ValidationRule(
                    rule_id=f"{stage_def.stage_id}_statistical_consistency",
                    rule_name="Statistical Consistency",
                    rule_type="statistical",
                    severity="medium",
                    action=ValidationAction.WARNING,
                    threshold=90.0
                )
            ])
        
        elif stage_def.stage_type == PipelineStageType.INTEGRATION:
            rules.extend([
                ValidationRule(
                    rule_id=f"{stage_def.stage_id}_integration_consistency",
                    rule_name="Integration Consistency",
                    rule_type="business",
                    severity="critical",
                    action=ValidationAction.HALT
                ),
                ValidationRule(
                    rule_id=f"{stage_def.stage_id}_geographic_validation",
                    rule_name="Geographic Validation",
                    rule_type="geographic",
                    severity="high",
                    action=ValidationAction.WARNING
                )
            ])
        
        # Add quality level specific rules
        if stage_def.quality_level == QualityLevel.COMPREHENSIVE:
            rules.extend([
                ValidationRule(
                    rule_id=f"{stage_def.stage_id}_comprehensive_quality",
                    rule_name="Comprehensive Quality Check",
                    rule_type="statistical",
                    severity="medium",
                    action=ValidationAction.WARNING,
                    threshold=95.0
                )
            ])
        
        return rules
    
    def _validate_stage_input(
        self,
        stage_id: str,
        input_data: pl.DataFrame,
        context: PipelineContext
    ) -> StageValidationResult:
        """Validate stage input data."""
        if stage_id not in self.stage_validators:
            return self._create_dummy_validation_result(stage_id, True)
        
        validator = self.stage_validators[stage_id]
        return validator.validate_stage_data(input_data, context)
    
    def _validate_stage_output(
        self,
        stage_id: str,
        output_data: pl.DataFrame,
        context: PipelineContext
    ) -> StageValidationResult:
        """Validate stage output data."""
        if stage_id not in self.stage_validators:
            return self._create_dummy_validation_result(stage_id, True)
        
        validator = self.stage_validators[stage_id]
        return validator.validate_stage_data(output_data, context)
    
    def _create_dummy_validation_result(self, stage_id: str, passed: bool) -> StageValidationResult:
        """Create a dummy validation result when validation is disabled."""
        from .validation_pipeline import ValidationMetrics, QualityGateStatus
        
        return StageValidationResult(
            stage_name=stage_id,
            gate_status=QualityGateStatus.PASSED if passed else QualityGateStatus.FAILED,
            validation_metrics=ValidationMetrics(
                total_records=0,
                validated_records=0,
                passed_records=0,
                failed_records=0,
                warning_records=0,
                validation_time_seconds=0.0,
                rules_executed=0,
                rules_passed=0,
                rules_failed=0,
                quality_score=100.0 if passed else 0.0,
                completeness_score=100.0 if passed else 0.0,
                accuracy_score=100.0 if passed else 0.0,
                consistency_score=100.0 if passed else 0.0
            ),
            rule_results={},
            errors=[],
            warnings=[],
            recommendations=[]
        )
    
    def _execute_stage_instance(
        self,
        stage_instance: Any,
        input_data: Optional[pl.DataFrame],
        context: PipelineContext
    ) -> Optional[pl.DataFrame]:
        """Execute a stage instance."""
        # Determine execution method based on stage type
        if hasattr(stage_instance, 'extract'):
            # Extractor - provide source configuration from context
            source_config = context.metadata.get('source_config', {})
            input_path = source_config.get('input_path', 'data_raw')
            
            # Create source specification for extractor
            source = {'path': input_path, 'type': 'directory'}
            
            # Execute extraction and collect results into DataFrame
            extraction_results = []
            for batch in stage_instance.extract(source):
                if isinstance(batch, list):
                    extraction_results.extend(batch)
                else:
                    extraction_results.append(batch)
            
            # Convert results to DataFrame if we have data
            if extraction_results:
                return pl.DataFrame(extraction_results)
            else:
                # Return empty DataFrame with basic structure if no results
                return pl.DataFrame()
                
        elif hasattr(stage_instance, 'transform'):
            # Transformer
            if input_data is None or input_data.height == 0:
                logger.warning("Transformer received empty input data")
                return pl.DataFrame()
            return stage_instance.transform(input_data)
            
        elif hasattr(stage_instance, 'load'):
            # Loader
            if input_data is None or input_data.height == 0:
                logger.warning("Loader received empty input data")
                return input_data
            
            # Get target configuration from context
            target_config = context.metadata.get('target_config', {})
            output_path = target_config.get('output_path', 'output/processed_data.parquet')
            
            stage_instance.load(input_data, output_path)
            return input_data  # Pass through data
            
        elif hasattr(stage_instance, 'execute'):
            # Generic execution
            return stage_instance.execute(input_data, context)
        else:
            raise PipelineError(f"Unknown stage execution method for {type(stage_instance)}")


class DataFlowController:
    """
    Controls data flow between extraction, transformation, and loading stages.
    
    Manages data flow with validation checkpoints, transformation pipelines,
    and quality gates between each stage of the ETL process.
    """
    
    def __init__(
        self,
        checkpoints: List[DataFlowCheckpoint],
        buffer_size: int = 10000
    ):
        """
        Initialise data flow controller.
        
        Args:
            checkpoints: List of data flow checkpoints
            buffer_size: Maximum buffer size for data flow
        """
        self.checkpoints = {cp.checkpoint_id: cp for cp in checkpoints}
        self.buffer_size = buffer_size
        
        # Data flow tracking
        self.data_buffers: Dict[str, pd.DataFrame] = {}
        self.flow_metrics: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            "Data flow controller initialised",
            checkpoints_count=len(self.checkpoints),
            buffer_size=buffer_size
        )
    
    @monitor_performance("data_flow_transfer")
    def transfer_data(
        self,
        source_stage: str,
        target_stage: str,
        data: pd.DataFrame,
        context: PipelineContext
    ) -> pd.DataFrame:
        """
        Transfer data between stages with validation and transformation.
        
        Args:
            source_stage: Source stage identifier
            target_stage: Target stage identifier
            data: Data to transfer
            context: Pipeline context
            
        Returns:
            Processed data ready for target stage
        """
        checkpoint_id = f"{source_stage}_to_{target_stage}"
        
        logger.info(
            "Transferring data between stages",
            source_stage=source_stage,
            target_stage=target_stage,
            records_count=len(data)
        )
        
        try:
            start_time = datetime.now()
            
            # Find relevant checkpoint
            checkpoint = self._find_checkpoint(source_stage, target_stage)
            
            # Validate data against checkpoint rules
            if checkpoint and checkpoint.validation_rules:
                validated_data = self._validate_checkpoint_data(checkpoint, data, context)
            else:
                validated_data = data
            
            # Apply transformation rules
            if checkpoint and checkpoint.transformation_rules:
                transformed_data = self._apply_transformation_rules(checkpoint, validated_data)
            else:
                transformed_data = validated_data
            
            # Buffer data if necessary
            if len(transformed_data) > self.buffer_size:
                buffered_data = self._buffer_large_dataset(checkpoint_id, transformed_data)
            else:
                buffered_data = transformed_data
            
            # Record flow metrics
            transfer_time = (datetime.now() - start_time).total_seconds()
            self.flow_metrics[checkpoint_id] = {
                "source_stage": source_stage,
                "target_stage": target_stage,
                "input_records": len(data),
                "output_records": len(buffered_data),
                "transfer_time": transfer_time,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(
                "Data transfer completed",
                checkpoint_id=checkpoint_id,
                input_records=len(data),
                output_records=len(buffered_data),
                transfer_time=transfer_time
            )
            
            return buffered_data
            
        except Exception as e:
            logger.error(
                "Data transfer failed",
                source_stage=source_stage,
                target_stage=target_stage,
                error=str(e)
            )
            raise PipelineError(f"Data flow transfer failed: {str(e)}") from e
    
    def get_flow_status(self) -> Dict[str, Any]:
        """Get current data flow status."""
        return {
            "active_buffers": len(self.data_buffers),
            "buffer_sizes": {
                buffer_id: len(buffer_data)
                for buffer_id, buffer_data in self.data_buffers.items()
            },
            "flow_metrics": self.flow_metrics,
            "checkpoints": {
                cp_id: {
                    "source_stage": cp.source_stage,
                    "target_stage": cp.target_stage,
                    "validation_rules_count": len(cp.validation_rules),
                    "transformation_rules_count": len(cp.transformation_rules)
                }
                for cp_id, cp in self.checkpoints.items()
            }
        }
    
    def _find_checkpoint(self, source_stage: str, target_stage: str) -> Optional[DataFlowCheckpoint]:
        """Find checkpoint for stage transition."""
        for checkpoint in self.checkpoints.values():
            if checkpoint.source_stage == source_stage and checkpoint.target_stage == target_stage:
                return checkpoint
        return None
    
    def _validate_checkpoint_data(
        self,
        checkpoint: DataFlowCheckpoint,
        data: pd.DataFrame,
        context: PipelineContext
    ) -> pd.DataFrame:
        """Validate data at checkpoint."""
        # Simplified validation - in practice, would use validation framework
        validation_errors = []
        
        for rule in checkpoint.validation_rules:
            if rule == "non_empty":
                if data.empty:
                    validation_errors.append("Data is empty")
            elif rule == "no_duplicates":
                if data.duplicated().any():
                    validation_errors.append("Duplicate records found")
            elif rule.startswith("min_records:"):
                min_records = int(rule.split(":")[1])
                if len(data) < min_records:
                    validation_errors.append(f"Insufficient records: {len(data)} < {min_records}")
        
        if validation_errors:
            raise ValidationError(f"Checkpoint validation failed: {validation_errors}")
        
        return data
    
    def _apply_transformation_rules(
        self,
        checkpoint: DataFlowCheckpoint,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply transformation rules at checkpoint."""
        transformed_data = data.copy()
        
        for rule in checkpoint.transformation_rules:
            if rule == "remove_duplicates":
                transformed_data = transformed_data.drop_duplicates()
            elif rule == "fill_nulls":
                transformed_data = transformed_data.fillna('')
            elif rule.startswith("rename_column:"):
                old_name, new_name = rule.split(":")[1].split("->")
                transformed_data = transformed_data.rename(columns={old_name.strip(): new_name.strip()})
        
        return transformed_data
    
    def _buffer_large_dataset(self, checkpoint_id: str, data: pd.DataFrame) -> pd.DataFrame:
        """Buffer large dataset for efficient processing."""
        # For now, just store in memory buffer
        # In production, would use more sophisticated buffering (disk, distributed storage)
        self.data_buffers[checkpoint_id] = data
        
        logger.info(
            "Large dataset buffered",
            checkpoint_id=checkpoint_id,
            records_count=len(data)
        )
        
        return data


class QualityAssuranceManager:
    """
    Ensures quality standards throughout the pipeline.
    
    Manages comprehensive quality assurance including validation orchestration,
    compliance monitoring, and quality reporting across all pipeline stages.
    """
    
    def __init__(
        self,
        quality_config: QualityAssuranceConfig,
        compliance_standards: Optional[List[str]] = None
    ):
        """
        Initialise quality assurance manager.
        
        Args:
            quality_config: Quality assurance configuration
            compliance_standards: List of compliance standards to enforce
        """
        self.quality_config = quality_config
        self.compliance_standards = compliance_standards or []
        
        # Initialise quality components
        self.validation_orchestrator = ValidationOrchestrator()
        self.validation_reporter = ValidationReporter()
        self.quality_gatekeeper = QualityGatekeeper()
        
        # Quality tracking
        self.quality_metrics: Dict[str, Any] = {}
        self.compliance_status: Dict[str, str] = {}
        self.quality_reports: List[str] = []
        
        logger.info(
            "Quality assurance manager initialised",
            quality_level=quality_config.quality_level.value,
            compliance_standards=self.compliance_standards
        )
    
    @monitor_performance("quality_assessment")
    def assess_pipeline_quality(
        self,
        pipeline_results: Dict[str, StageResult],
        context: PipelineContext
    ) -> Dict[str, Any]:
        """
        Assess overall pipeline quality.
        
        Args:
            pipeline_results: Results from all pipeline stages
            context: Pipeline context
            
        Returns:
            Comprehensive quality assessment
        """
        logger.info("Assessing pipeline quality")
        
        try:
            # Calculate overall metrics
            total_stages = len(pipeline_results)
            successful_stages = sum(
                1 for result in pipeline_results.values()
                if result.state == StageState.COMPLETED
            )
            
            # Calculate quality scores
            quality_scores = []
            for stage_id, result in pipeline_results.values():
                stage_quality = self._calculate_stage_quality(result)
                quality_scores.append(stage_quality)
            
            overall_quality_score = np.mean(quality_scores) if quality_scores else 0
            
            # Assess compliance
            compliance_assessment = self._assess_compliance_status(pipeline_results)
            
            # Generate quality metrics
            quality_assessment = {
                "overall_quality_score": overall_quality_score,
                "pipeline_success_rate": (successful_stages / total_stages) * 100,
                "stage_quality_scores": {
                    stage_id: self._calculate_stage_quality(result)
                    for stage_id, result in pipeline_results.items()
                },
                "compliance_status": compliance_assessment,
                "quality_grade": self._determine_quality_grade(overall_quality_score),
                "recommendations": self._generate_quality_recommendations(pipeline_results),
                "assessment_timestamp": datetime.now().isoformat()
            }
            
            # Store metrics
            self.quality_metrics = quality_assessment
            
            # Generate quality report if enabled
            if self.quality_config.generate_quality_reports:
                report_path = self._generate_quality_report(quality_assessment, context)
                quality_assessment["report_path"] = report_path
            
            logger.info(
                "Pipeline quality assessment completed",
                overall_quality_score=overall_quality_score,
                quality_grade=quality_assessment["quality_grade"]
            )
            
            return quality_assessment
            
        except Exception as e:
            logger.error(
                "Quality assessment failed",
                error=str(e)
            )
            raise PipelineError(f"Quality assessment failed: {str(e)}") from e
    
    def monitor_quality_trends(self, assessment_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Monitor quality trends over time."""
        if len(assessment_history) < 2:
            return {"status": "insufficient_data"}
        
        # Calculate trends
        recent_scores = [assessment["overall_quality_score"] for assessment in assessment_history[-5:]]
        trend_direction = "stable"
        
        if len(recent_scores) >= 2:
            if recent_scores[-1] > recent_scores[0]:
                trend_direction = "improving"
            elif recent_scores[-1] < recent_scores[0]:
                trend_direction = "declining"
        
        # Identify quality issues
        quality_issues = []
        latest_assessment = assessment_history[-1]
        
        if latest_assessment["overall_quality_score"] < 80:
            quality_issues.append("Overall quality below threshold")
        
        if latest_assessment["pipeline_success_rate"] < 95:
            quality_issues.append("Pipeline success rate below target")
        
        return {
            "trend_direction": trend_direction,
            "average_quality_score": np.mean(recent_scores),
            "quality_variance": np.var(recent_scores),
            "quality_issues": quality_issues,
            "recommendations": self._generate_trend_recommendations(trend_direction, quality_issues)
        }
    
    def _calculate_stage_quality(self, stage_result: StageResult) -> float:
        """Calculate quality score for a stage."""
        if stage_result.state != StageState.COMPLETED:
            return 0.0
        
        # Base quality score
        base_score = 100.0
        
        # Deduct points for errors
        if stage_result.error:
            base_score -= 50.0
        
        # Factor in metrics if available
        if stage_result.metrics:
            processing_time = stage_result.metrics.get("processing_time", 0)
            # Deduct points for excessive processing time (> 1 hour)
            if processing_time > 3600:
                base_score -= 10.0
            
            # Factor in record counts
            input_records = stage_result.metrics.get("input_records", 0)
            output_records = stage_result.metrics.get("output_records", 0)
            
            if input_records > 0:
                data_preservation_rate = output_records / input_records
                if data_preservation_rate < 0.95:  # Lost more than 5% of data
                    base_score -= 20.0
        
        return max(0.0, min(100.0, base_score))
    
    def _assess_compliance_status(self, pipeline_results: Dict[str, StageResult]) -> Dict[str, str]:
        """Assess compliance status against standards."""
        compliance_status = {}
        
        for standard in self.compliance_standards:
            # Simplified compliance check
            overall_success = all(
                result.state == StageState.COMPLETED
                for result in pipeline_results.values()
            )
            
            if overall_success:
                compliance_status[standard] = "COMPLIANT"
            else:
                compliance_status[standard] = "NON_COMPLIANT"
        
        return compliance_status
    
    def _determine_quality_grade(self, quality_score: float) -> str:
        """Determine quality grade based on score."""
        if quality_score >= 95:
            return "A"
        elif quality_score >= 85:
            return "B"
        elif quality_score >= 75:
            return "C"
        elif quality_score >= 65:
            return "D"
        else:
            return "F"
    
    def _generate_quality_recommendations(self, pipeline_results: Dict[str, StageResult]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Analyse failed stages
        failed_stages = [
            stage_id for stage_id, result in pipeline_results.items()
            if result.state == StageState.FAILED
        ]
        
        if failed_stages:
            recommendations.append(f"Investigate and fix failures in stages: {failed_stages}")
        
        # Analyse performance issues
        slow_stages = [
            stage_id for stage_id, result in pipeline_results.items()
            if result.duration and result.duration > 1800  # 30 minutes
        ]
        
        if slow_stages:
            recommendations.append(f"Optimise performance for slow stages: {slow_stages}")
        
        # Data quality recommendations
        recommendations.append("Implement comprehensive data validation at each stage")
        recommendations.append("Monitor data quality metrics continuously")
        recommendations.append("Establish quality gates with appropriate thresholds")
        
        return recommendations
    
    def _generate_trend_recommendations(self, trend_direction: str, quality_issues: List[str]) -> List[str]:
        """Generate recommendations based on quality trends."""
        recommendations = []
        
        if trend_direction == "declining":
            recommendations.append("Quality is declining - implement immediate corrective measures")
            recommendations.append("Review recent changes to pipeline configuration")
            recommendations.append("Increase validation frequency and coverage")
        elif trend_direction == "improving":
            recommendations.append("Quality is improving - maintain current practices")
            recommendations.append("Document successful quality improvements")
        
        if quality_issues:
            recommendations.append("Address identified quality issues immediately")
            for issue in quality_issues:
                recommendations.append(f"Fix: {issue}")
        
        return recommendations
    
    def _generate_quality_report(self, quality_assessment: Dict[str, Any], context: PipelineContext) -> str:
        """Generate comprehensive quality report."""
        try:
            report_data = {
                "pipeline_id": context.pipeline_id,
                "run_id": context.run_id,
                "quality_assessment": quality_assessment,
                "generated_at": datetime.now().isoformat()
            }
            
            # Save report
            report_filename = f"quality_report_{context.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = Path(f"logs/quality_reports/{report_filename}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.quality_reports.append(str(report_path))
            
            logger.info(
                "Quality report generated",
                report_path=str(report_path)
            )
            
            return str(report_path)
            
        except Exception as e:
            logger.error(
                "Failed to generate quality report",
                error=str(e)
            )
            return ""


class MasterETLPipeline(BasePipeline):
    """
    Complete end-to-end ETL pipeline orchestrator.
    
    Orchestrates the entire ETL process with integrated validation,
    quality assurance, and comprehensive monitoring throughout
    the data processing lifecycle.
    """
    
    def __init__(
        self,
        name: str = "master_etl_pipeline",
        stage_definitions: Optional[List[PipelineStageDefinition]] = None,
        quality_config: Optional[QualityAssuranceConfig] = None,
        db_path: str = "ahgd.db",
        **kwargs
    ):
        """
        Initialise master ETL pipeline.
        
        Args:
            name: Pipeline name
            stage_definitions: List of stage definitions
            quality_config: Quality assurance configuration
            db_path: Path to the DuckDB database file
            **kwargs: Additional pipeline arguments
        """
        super().__init__(name, **kwargs)
        
        # Load default configurations if not provided
        self.stage_definitions = stage_definitions or self._load_default_stage_definitions()
        self.quality_config = quality_config or QualityAssuranceConfig()
        
        # Initialise pipeline components
        self.stage_manager = PipelineStageManager(self.stage_definitions, self.quality_config)
        self.data_flow_controller = DataFlowController(self._create_default_checkpoints())
        self.quality_manager = QualityAssuranceManager(self.quality_config)
        self.validation_pipeline = ValidationPipeline("master_validation")
        
        # DuckDB connection
        self.db_path = db_path
        self.con = duckdb.connect(database=self.db_path, read_only=False)

        # Pipeline state
        self.current_table: Optional[str] = None
        self.stage_outputs: Dict[str, str] = {}
        self.validation_results: Dict[str, StageValidationResult] = {}
        
        logger.info(
            "Master ETL pipeline initialised",
            name=name,
            stages_count=len(self.stage_definitions),
            quality_level=self.quality_config.quality_level.value
        )

    def cleanup(self):
        """Clean up resources, like closing the database connection."""
        self.con.close()
    
    def define_stages(self) -> List[str]:
        """Define pipeline stages."""
        return [stage.stage_id for stage in self.stage_definitions]
    
    @monitor_performance("master_etl_execution")
    def execute_stage(self, stage_name: str, context: PipelineContext) -> Any:
        """
        Execute a pipeline stage with integrated validation.
        
        Args:
            stage_name: Name of the stage to execute
            context: Pipeline context
            
        Returns:
            Stage output
        """
        logger.info("Executing master ETL stage", stage=stage_name)
        
        try:
            # Get input table for stage
            input_table = self._get_stage_input_table(stage_name, context)
            
            # Execute stage through stage manager
            output_table, stage_result = self.stage_manager.execute_stage(
                stage_name, context, self.con, input_table
            )
            
            # Store stage output
            if output_table is not None:
                self.stage_outputs[stage_name] = output_table
                context.add_output(f"{stage_name}_output", output_table)
            
            # Update current data pointer
            if output_table is not None:
                self.current_table = output_table
            
            # Track data flow if moving to next stage
            next_stages = self._get_next_stages(stage_name)
            for next_stage in next_stages:
                if output_table is not None:
                    # In the new model, data flow is managed by dbt, so we just pass the table name
                    context.add_output(f"{stage_name}_to_{next_stage}", output_table)
            
            logger.info(
                "Master ETL stage completed",
                stage=stage_name,
                output_table=output_table
            )
            
            return stage_result
            
        except Exception as e:
            logger.error(
                "Master ETL stage failed",
                stage=stage_name,
                error=str(e)
            )
            raise
    
    def run_complete_etl(
        self,
        source_config: Optional[Dict[str, Any]] = None,
        target_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete ETL pipeline from start to finish.
        
        Args:
            source_config: Source configuration
            target_config: Target configuration
            
        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete ETL pipeline execution")
        
        try:
            # Set up pipeline context
            context = self._create_context()
            if source_config:
                context.metadata["source_config"] = source_config
            if target_config:
                context.metadata["target_config"] = target_config
            
            # Run pipeline
            pipeline_context = self.run()
            
            # Assess final quality
            quality_assessment = self.quality_manager.assess_pipeline_quality(
                self.stage_manager.stage_results, pipeline_context
            )
            
            # Generate comprehensive results
            results = {
                "pipeline_id": pipeline_context.pipeline_id,
                "run_id": pipeline_context.run_id,
                "execution_status": self.state.value,
                "stage_results": {
                    name: {
                        "state": result.state.value,
                        "duration": result.duration,
                        "output_table": result.output,
                        "error": str(result.error) if result.error else None
                    }
                    for name, result in self.stage_manager.stage_results.items()
                },
                "quality_assessment": quality_assessment,
                "validation_results": {
                    name: {
                        "gate_status": result.gate_status.value,
                        "quality_score": result.validation_metrics.quality_score
                    }
                    for name, result in self.validation_results.items()
                },
                "final_table": self.current_table,
                "execution_summary": self._generate_execution_summary()
            }
            
            logger.info(
                "Complete ETL pipeline execution finished",
                status=self.state.value,
                overall_quality_score=quality_assessment.get("overall_quality_score", 0),
                final_table=results["final_table"]
            )
            
            return results
            
        except Exception as e:
            logger.error(
                "Complete ETL pipeline execution failed",
                error=str(e)
            )
            self.cleanup()
            raise PipelineError(f"Complete ETL execution failed: {str(e)}") from e
        finally:
            self.cleanup()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status."""
        return {
            "pipeline_name": self.name,
            "pipeline_state": self.state.value,
            "current_stage": self.current_stage,
            "progress": self.get_progress(),
            "stage_manager_status": {
                "execution_order": self.stage_manager.stage_execution_order,
                "stage_results": {
                    stage_id: result.state.value
                    for stage_id, result in self.stage_manager.stage_results.items()
                },
            },
            "quality_metrics": self.quality_manager.quality_metrics,
            "validation_summary": {
                name: result.gate_status.value
                for name, result in self.validation_results.items()
            },
            "current_table": self.current_table
        }
    
    def _load_default_stage_definitions(self) -> List[PipelineStageDefinition]:
        """Load default stage definitions."""
        # Load default configuration for extractors/transformers/loaders
        default_extractor_config = get_config('extractors', {'batch_size': 1000})
        default_transformer_config = get_config('transformers', {'batch_size': 1000})
        default_loader_config = get_config('loaders', {'batch_size': 1000})
        
        return [
            PipelineStageDefinition(
                stage_id="data_extraction",
                stage_name="Data Extraction",
                stage_type=PipelineStageType.EXTRACTION,
                stage_class="src.extractors.aihw_extractor.AIHWMortalityExtractor",
                dependencies=[],
                configuration={'config': default_extractor_config},
                validation_required=True,
                quality_level=QualityLevel.STANDARD
            ),
            PipelineStageDefinition(
                stage_id="data_transformation",
                stage_name="Data Transformation",
                stage_type=PipelineStageType.TRANSFORMATION,
                stage_class="src.transformers.geographic_standardiser.GeographicStandardiser",
                dependencies=["data_extraction"],
                configuration={'config': default_transformer_config},
                validation_required=True,
                quality_level=QualityLevel.COMPREHENSIVE
            ),
            PipelineStageDefinition(
                stage_id="data_integration",
                stage_name="Data Integration",
                stage_type=PipelineStageType.INTEGRATION,
                stage_class="src.transformers.data_integrator.MasterDataIntegrator",
                dependencies=["data_transformation"],
                configuration={'config': default_transformer_config},
                validation_required=True,
                quality_level=QualityLevel.COMPREHENSIVE
            ),
            PipelineStageDefinition(
                stage_id="data_loading",
                stage_name="Data Loading",
                stage_type=PipelineStageType.LOADING,
                stage_class="src.loaders.production_loader.ProductionLoader",
                dependencies=["data_integration"],
                configuration={'config': default_loader_config},
                validation_required=True,
                quality_level=QualityLevel.STANDARD
            )
        ]
    
    def _create_default_checkpoints(self) -> List[DataFlowCheckpoint]:
        """Create default data flow checkpoints."""
        return [
            DataFlowCheckpoint(
                checkpoint_id="extraction_to_transformation",
                source_stage="data_extraction",
                target_stage="data_transformation",
                validation_rules=["non_empty", "no_duplicates"],
                transformation_rules=["remove_duplicates"],
                quality_threshold=95.0
            ),
            DataFlowCheckpoint(
                checkpoint_id="transformation_to_integration",
                source_stage="data_transformation",
                target_stage="data_integration",
                validation_rules=["non_empty", "min_records:100"],
                transformation_rules=["fill_nulls"],
                quality_threshold=95.0
            ),
            DataFlowCheckpoint(
                checkpoint_id="integration_to_loading",
                source_stage="data_integration",
                target_stage="data_loading",
                validation_rules=["non_empty"],
                transformation_rules=[],
                quality_threshold=98.0
            )
        ]
    
    def _get_stage_input_table(self, stage_name: str, context: PipelineContext) -> Optional[str]:
        """Get input table for a stage."""
        # First stage gets no input table (extracts from source)
        if not self.stage_manager.stage_definitions[stage_name].dependencies:
            return None
        
        # Get table from previous stage or current table
        if self.current_table is not None:
            return self.current_table
        
        # Look for table in context from previous stages
        for dep_stage in self.stage_manager.stage_definitions[stage_name].dependencies:
            output_key = f"{dep_stage}_output"
            if output_key in context.stage_outputs:
                return context.stage_outputs[output_key]
        
        return None
    
    def _get_next_stages(self, current_stage: str) -> List[str]:
        """Get stages that depend on the current stage."""
        next_stages = []
        for stage_id, stage_def in self.stage_manager.stage_definitions.items():
            if current_stage in stage_def.dependencies:
                next_stages.append(stage_id)
        return next_stages
    
    def _generate_execution_summary(self) -> Dict[str, Any]:
        """Generate execution summary."""
        total_stages = len(self.stage_results)
        successful_stages = sum(
            1 for result in self.stage_results.values()
            if result.state == StageState.COMPLETED
        )
        failed_stages = sum(
            1 for result in self.stage_results.values()
            if result.state == StageState.FAILED
        )
        
        total_duration = sum(
            result.duration for result in self.stage_results.values()
            if result.duration
        )
        
        return {
            "total_stages": total_stages,
            "successful_stages": successful_stages,
            "failed_stages": failed_stages,
            "success_rate": (successful_stages / total_stages * 100) if total_stages > 0 else 0,
            "total_duration_seconds": total_duration,
            "average_stage_duration": total_duration / total_stages if total_stages > 0 else 0,
            "pipeline_efficiency": "high" if successful_stages == total_stages else "medium" if failed_stages == 0 else "low"
        }