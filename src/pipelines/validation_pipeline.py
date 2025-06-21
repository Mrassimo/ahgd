"""
Validation pipeline for integrating data quality validation into ETL processes.

This module provides comprehensive validation integration at each stage of the
ETL pipeline, implementing quality gates that can halt processing on critical
validation failures.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
import pandas as pd
import numpy as np

from .base_pipeline import BasePipeline, PipelineContext, StageResult, StageState, PipelineError
from ..validators import (
    ValidationOrchestrator, QualityChecker, GeographicValidator,
    StatisticalValidator, AustralianHealthBusinessRulesValidator,
    ValidationReporter, ValidationReport, ValidationSummary
)
from ..utils.logging import get_logger, monitor_performance, track_lineage
from ..utils.config import get_config
from ..utils.interfaces import ValidationError, DataQualityError


logger = get_logger(__name__)


class ValidationAction(str, Enum):
    """Actions to take when validation fails."""
    HALT = "halt"           # Stop pipeline immediately
    WARNING = "warning"     # Log warning and continue
    QUARANTINE = "quarantine"  # Move data to quarantine and continue
    RETRY = "retry"         # Retry stage after validation fixes
    SKIP = "skip"           # Skip current data and continue with next


class ValidationMode(str, Enum):
    """Validation execution modes."""
    STRICT = "strict"       # Halt on any validation error
    PERMISSIVE = "permissive"  # Continue with warnings
    SELECTIVE = "selective"  # Halt only on critical errors
    AUDIT_ONLY = "audit_only"  # Validate but don't affect pipeline


class QualityGateStatus(str, Enum):
    """Quality gate evaluation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    BYPASSED = "bypassed"


@dataclass
class ValidationRule:
    """Individual validation rule configuration."""
    rule_id: str
    rule_name: str
    rule_type: str  # schema, business, statistical, geographic
    severity: str   # critical, high, medium, low
    action: ValidationAction
    threshold: Optional[float] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    bypass_conditions: List[str] = field(default_factory=list)


@dataclass
class QualityGate:
    """Quality gate configuration for pipeline stages."""
    gate_id: str
    gate_name: str
    stage_name: str
    validation_rules: List[ValidationRule]
    pass_threshold: float = 95.0  # Minimum percentage to pass
    critical_rule_threshold: float = 100.0  # Critical rules must be 100%
    mode: ValidationMode = ValidationMode.SELECTIVE
    timeout_seconds: int = 300
    retry_attempts: int = 3


@dataclass
class ValidationMetrics:
    """Validation performance and quality metrics."""
    total_records: int
    validated_records: int
    passed_records: int
    failed_records: int
    warning_records: int
    validation_time_seconds: float
    rules_executed: int
    rules_passed: int
    rules_failed: int
    quality_score: float
    completeness_score: float
    accuracy_score: float
    consistency_score: float


@dataclass
class StageValidationResult:
    """Result of stage validation."""
    stage_name: str
    gate_status: QualityGateStatus
    validation_metrics: ValidationMetrics
    rule_results: Dict[str, bool]
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    quarantined_data: Optional[pd.DataFrame] = None
    validation_report: Optional[ValidationReport] = None
    execution_time: float = 0.0


class StageValidator:
    """
    Validates data at individual pipeline stages.
    
    Provides stage-level validation with configurable quality gates,
    supporting different validation modes and actions.
    """
    
    def __init__(
        self,
        stage_name: str,
        quality_gate: QualityGate,
        validation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise stage validator.
        
        Args:
            stage_name: Name of the pipeline stage
            quality_gate: Quality gate configuration
            validation_config: Additional validation configuration
        """
        self.stage_name = stage_name
        self.quality_gate = quality_gate
        self.validation_config = validation_config or {}
        
        # Initialise validators
        self.orchestrator = ValidationOrchestrator()
        self.quality_checker = QualityChecker()
        self.business_validator = AustralianHealthBusinessRulesValidator()
        self.statistical_validator = StatisticalValidator()
        self.geographic_validator = GeographicValidator()
        self.reporter = ValidationReporter()
        
        logger.log.info(
            "Stage validator initialised",
            stage=stage_name,
            gate_id=quality_gate.gate_id,
            mode=quality_gate.mode.value
        )
    
    @monitor_performance("stage_validation")
    def validate_stage_data(
        self,
        data: pd.DataFrame,
        context: PipelineContext,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StageValidationResult:
        """
        Validate data for a pipeline stage.
        
        Args:
            data: Data to validate
            context: Pipeline context
            metadata: Additional metadata
            
        Returns:
            Stage validation result
        """
        start_time = datetime.now()
        
        try:
            # Initialise result
            result = StageValidationResult(
                stage_name=self.stage_name,
                gate_status=QualityGateStatus.PASSED,
                validation_metrics=ValidationMetrics(
                    total_records=len(data),
                    validated_records=0,
                    passed_records=0,
                    failed_records=0,
                    warning_records=0,
                    validation_time_seconds=0.0,
                    rules_executed=0,
                    rules_passed=0,
                    rules_failed=0,
                    quality_score=0.0,
                    completeness_score=0.0,
                    accuracy_score=0.0,
                    consistency_score=0.0
                ),
                rule_results={},
                errors=[],
                warnings=[],
                recommendations=[]
            )
            
            # Execute validation rules
            for rule in self.quality_gate.validation_rules:
                if not rule.enabled:
                    continue
                
                rule_result = self._execute_validation_rule(rule, data, context)
                result.rule_results[rule.rule_id] = rule_result.passed
                result.validation_metrics.rules_executed += 1
                
                if rule_result.passed:
                    result.validation_metrics.rules_passed += 1
                else:
                    result.validation_metrics.rules_failed += 1
                    
                    if rule.severity == "critical":
                        result.errors.extend(rule_result.errors)
                        if rule.action == ValidationAction.HALT:
                            result.gate_status = QualityGateStatus.FAILED
                    else:
                        result.warnings.extend(rule_result.errors)
            
            # Calculate overall metrics
            self._calculate_validation_metrics(result, data)
            
            # Evaluate quality gate
            gate_passed = self._evaluate_quality_gate(result)
            if not gate_passed and result.gate_status != QualityGateStatus.FAILED:
                result.gate_status = QualityGateStatus.WARNING
            
            # Generate validation report
            result.validation_report = self._generate_validation_report(result, data)
            
            # Calculate execution time
            result.execution_time = (datetime.now() - start_time).total_seconds()
            result.validation_metrics.validation_time_seconds = result.execution_time
            
            # Log validation result
            logger.log.info(
                "Stage validation completed",
                stage=self.stage_name,
                gate_status=result.gate_status.value,
                quality_score=result.validation_metrics.quality_score,
                execution_time=result.execution_time
            )
            
            # Track data lineage
            track_lineage(
                f"{self.stage_name}_input",
                f"{self.stage_name}_validated",
                "validation"
            )
            
            return result
            
        except Exception as e:
            logger.log.error(
                "Stage validation failed",
                stage=self.stage_name,
                error=str(e)
            )
            
            # Return failed result
            execution_time = (datetime.now() - start_time).total_seconds()
            return StageValidationResult(
                stage_name=self.stage_name,
                gate_status=QualityGateStatus.FAILED,
                validation_metrics=ValidationMetrics(
                    total_records=len(data) if data is not None else 0,
                    validated_records=0,
                    passed_records=0,
                    failed_records=0,
                    warning_records=0,
                    validation_time_seconds=execution_time,
                    rules_executed=0,
                    rules_passed=0,
                    rules_failed=0,
                    quality_score=0.0,
                    completeness_score=0.0,
                    accuracy_score=0.0,
                    consistency_score=0.0
                ),
                rule_results={},
                errors=[f"Validation execution failed: {str(e)}"],
                warnings=[],
                recommendations=[],
                execution_time=execution_time
            )
    
    def _execute_validation_rule(
        self,
        rule: ValidationRule,
        data: pd.DataFrame,
        context: PipelineContext
    ) -> 'ValidationRuleResult':
        """Execute a single validation rule."""
        try:
            if rule.rule_type == "schema":
                return self._execute_schema_validation(rule, data)
            elif rule.rule_type == "business":
                return self._execute_business_validation(rule, data)
            elif rule.rule_type == "statistical":
                return self._execute_statistical_validation(rule, data)
            elif rule.rule_type == "geographic":
                return self._execute_geographic_validation(rule, data)
            else:
                return ValidationRuleResult(
                    rule_id=rule.rule_id,
                    passed=False,
                    errors=[f"Unknown rule type: {rule.rule_type}"]
                )
                
        except Exception as e:
            return ValidationRuleResult(
                rule_id=rule.rule_id,
                passed=False,
                errors=[f"Rule execution failed: {str(e)}"]
            )
    
    def _execute_schema_validation(
        self,
        rule: ValidationRule,
        data: pd.DataFrame
    ) -> 'ValidationRuleResult':
        """Execute schema validation rule."""
        errors = []
        
        # Basic schema checks
        if 'required_columns' in rule.parameters:
            missing_cols = set(rule.parameters['required_columns']) - set(data.columns)
            if missing_cols:
                errors.append(f"Missing required columns: {missing_cols}")
        
        if 'data_types' in rule.parameters:
            for col, expected_type in rule.parameters['data_types'].items():
                if col in data.columns:
                    actual_type = str(data[col].dtype)
                    if expected_type not in actual_type:
                        errors.append(f"Column {col} has type {actual_type}, expected {expected_type}")
        
        return ValidationRuleResult(
            rule_id=rule.rule_id,
            passed=len(errors) == 0,
            errors=errors
        )
    
    def _execute_business_validation(
        self,
        rule: ValidationRule,
        data: pd.DataFrame
    ) -> 'ValidationRuleResult':
        """Execute business rule validation."""
        try:
            # Use the business rules validator
            validation_result = self.business_validator.validate_data_frame(data)
            
            # Extract errors relevant to this rule
            errors = []
            if hasattr(validation_result, 'errors'):
                errors = validation_result.errors
            
            return ValidationRuleResult(
                rule_id=rule.rule_id,
                passed=len(errors) == 0,
                errors=errors
            )
            
        except Exception as e:
            return ValidationRuleResult(
                rule_id=rule.rule_id,
                passed=False,
                errors=[f"Business validation failed: {str(e)}"]
            )
    
    def _execute_statistical_validation(
        self,
        rule: ValidationRule,
        data: pd.DataFrame
    ) -> 'ValidationRuleResult':
        """Execute statistical validation rule."""
        try:
            # Use the statistical validator
            validation_result = self.statistical_validator.validate_data_frame(data)
            
            errors = []
            if hasattr(validation_result, 'outlier_count'):
                if validation_result.outlier_count > rule.threshold:
                    errors.append(f"Too many outliers: {validation_result.outlier_count}")
            
            return ValidationRuleResult(
                rule_id=rule.rule_id,
                passed=len(errors) == 0,
                errors=errors
            )
            
        except Exception as e:
            return ValidationRuleResult(
                rule_id=rule.rule_id,
                passed=False,
                errors=[f"Statistical validation failed: {str(e)}"]
            )
    
    def _execute_geographic_validation(
        self,
        rule: ValidationRule,
        data: pd.DataFrame
    ) -> 'ValidationRuleResult':
        """Execute geographic validation rule."""
        try:
            # Use the geographic validator
            validation_result = self.geographic_validator.validate_data_frame(data)
            
            errors = []
            if hasattr(validation_result, 'invalid_coordinates_count'):
                if validation_result.invalid_coordinates_count > 0:
                    errors.append(f"Invalid coordinates found: {validation_result.invalid_coordinates_count}")
            
            return ValidationRuleResult(
                rule_id=rule.rule_id,
                passed=len(errors) == 0,
                errors=errors
            )
            
        except Exception as e:
            return ValidationRuleResult(
                rule_id=rule.rule_id,
                passed=False,
                errors=[f"Geographic validation failed: {str(e)}"]
            )
    
    def _calculate_validation_metrics(
        self,
        result: StageValidationResult,
        data: pd.DataFrame
    ) -> None:
        """Calculate validation metrics."""
        total_records = len(data)
        result.validation_metrics.total_records = total_records
        result.validation_metrics.validated_records = total_records
        
        # Calculate quality scores based on rule results
        total_rules = len(result.rule_results)
        if total_rules > 0:
            passed_rules = sum(1 for passed in result.rule_results.values() if passed)
            result.validation_metrics.quality_score = (passed_rules / total_rules) * 100
        
        # Calculate completeness score
        if total_records > 0:
            completeness_scores = []
            for col in data.columns:
                completeness = (1 - data[col].isnull().sum() / total_records) * 100
                completeness_scores.append(completeness)
            result.validation_metrics.completeness_score = np.mean(completeness_scores)
        
        # Set accuracy and consistency scores (simplified)
        result.validation_metrics.accuracy_score = result.validation_metrics.quality_score
        result.validation_metrics.consistency_score = result.validation_metrics.quality_score
        
        # Calculate record-level statistics
        result.validation_metrics.passed_records = int(
            total_records * (result.validation_metrics.quality_score / 100)
        )
        result.validation_metrics.failed_records = (
            total_records - result.validation_metrics.passed_records
        )
    
    def _evaluate_quality_gate(self, result: StageValidationResult) -> bool:
        """Evaluate whether the quality gate passes."""
        # Check overall quality score
        if result.validation_metrics.quality_score < self.quality_gate.pass_threshold:
            return False
        
        # Check critical rules
        critical_rules = [
            rule for rule in self.quality_gate.validation_rules 
            if rule.severity == "critical"
        ]
        
        for rule in critical_rules:
            if rule.rule_id in result.rule_results:
                if not result.rule_results[rule.rule_id]:
                    return False
        
        return True
    
    def _generate_validation_report(
        self,
        result: StageValidationResult,
        data: pd.DataFrame
    ) -> ValidationReport:
        """Generate comprehensive validation report."""
        try:
            # Use the validation reporter
            return self.reporter.generate_comprehensive_report(
                data,
                validation_results={
                    "stage_validation": result
                },
                metadata={
                    "stage_name": self.stage_name,
                    "gate_id": self.quality_gate.gate_id,
                    "validation_timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.log.error(
                "Failed to generate validation report",
                stage=self.stage_name,
                error=str(e)
            )
            return None


@dataclass
class ValidationRuleResult:
    """Result of a single validation rule execution."""
    rule_id: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class QualityGatekeeper:
    """
    Implements quality gates that halt processing on critical validation failures.
    
    Manages quality gate evaluation, escalation procedures, and bypass mechanisms
    for different validation scenarios.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise quality gatekeeper.
        
        Args:
            config: Quality gate configuration
        """
        self.config = config or {}
        self.bypass_tokens: Set[str] = set()
        
        # Load quality gates from configuration
        self.quality_gates: Dict[str, QualityGate] = self._load_quality_gates()
        
        logger.log.info(
            "Quality gatekeeper initialised",
            gates_count=len(self.quality_gates)
        )
    
    def evaluate_quality_gate(
        self,
        stage_name: str,
        validation_result: StageValidationResult,
        context: PipelineContext
    ) -> Tuple[bool, str]:
        """
        Evaluate quality gate for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            validation_result: Validation result to evaluate
            context: Pipeline context
            
        Returns:
            Tuple of (gate_passed, action_message)
        """
        if stage_name not in self.quality_gates:
            logger.log.warning(
                "No quality gate configured for stage",
                stage=stage_name
            )
            return True, "No quality gate configured"
        
        gate = self.quality_gates[stage_name]
        
        # Check for bypass conditions
        if self._check_bypass_conditions(gate, validation_result, context):
            logger.log.info(
                "Quality gate bypassed",
                stage=stage_name,
                gate_id=gate.gate_id
            )
            return True, "Quality gate bypassed due to conditions"
        
        # Evaluate gate based on mode
        if gate.mode == ValidationMode.STRICT:
            return self._evaluate_strict_mode(gate, validation_result)
        elif gate.mode == ValidationMode.PERMISSIVE:
            return self._evaluate_permissive_mode(gate, validation_result)
        elif gate.mode == ValidationMode.SELECTIVE:
            return self._evaluate_selective_mode(gate, validation_result)
        elif gate.mode == ValidationMode.AUDIT_ONLY:
            return True, "Audit-only mode - gate always passes"
        else:
            logger.log.error(
                "Unknown validation mode",
                mode=gate.mode.value
            )
            return False, f"Unknown validation mode: {gate.mode.value}"
    
    def register_bypass_token(self, token: str, reason: str) -> None:
        """
        Register a bypass token for emergency gate bypass.
        
        Args:
            token: Bypass token
            reason: Reason for bypass
        """
        self.bypass_tokens.add(token)
        logger.log.warning(
            "Bypass token registered",
            token=token[:8] + "...",  # Only log partial token
            reason=reason
        )
    
    def _load_quality_gates(self) -> Dict[str, QualityGate]:
        """Load quality gate configurations."""
        gates = {}
        
        # Load from configuration files
        try:
            gate_config = get_config("quality_gates", {})
            
            for gate_data in gate_config.get("gates", []):
                gate = QualityGate(
                    gate_id=gate_data["gate_id"],
                    gate_name=gate_data["gate_name"],
                    stage_name=gate_data["stage_name"],
                    validation_rules=[
                        ValidationRule(**rule_data)
                        for rule_data in gate_data.get("validation_rules", [])
                    ],
                    pass_threshold=gate_data.get("pass_threshold", 95.0),
                    critical_rule_threshold=gate_data.get("critical_rule_threshold", 100.0),
                    mode=ValidationMode(gate_data.get("mode", "selective")),
                    timeout_seconds=gate_data.get("timeout_seconds", 300),
                    retry_attempts=gate_data.get("retry_attempts", 3)
                )
                gates[gate.stage_name] = gate
                
        except Exception as e:
            logger.log.error(
                "Failed to load quality gates configuration",
                error=str(e)
            )
        
        return gates
    
    def _check_bypass_conditions(
        self,
        gate: QualityGate,
        validation_result: StageValidationResult,
        context: PipelineContext
    ) -> bool:
        """Check if bypass conditions are met."""
        # Check bypass tokens
        bypass_token = context.metadata.get("bypass_token")
        if bypass_token and bypass_token in self.bypass_tokens:
            return True
        
        # Check other bypass conditions specific to the gate
        for rule in gate.validation_rules:
            for condition in rule.bypass_conditions:
                if self._evaluate_bypass_condition(condition, validation_result, context):
                    return True
        
        return False
    
    def _evaluate_bypass_condition(
        self,
        condition: str,
        validation_result: StageValidationResult,
        context: PipelineContext
    ) -> bool:
        """Evaluate a specific bypass condition."""
        try:
            # Simple condition evaluation (can be extended)
            if condition.startswith("quality_score >"):
                threshold = float(condition.split(">")[1].strip())
                return validation_result.validation_metrics.quality_score > threshold
            elif condition.startswith("emergency_mode"):
                return context.metadata.get("emergency_mode", False)
            elif condition.startswith("dev_environment"):
                return get_config("environment") == "development"
        except Exception as e:
            logger.log.error(
                "Failed to evaluate bypass condition",
                condition=condition,
                error=str(e)
            )
        
        return False
    
    def _evaluate_strict_mode(
        self,
        gate: QualityGate,
        validation_result: StageValidationResult
    ) -> Tuple[bool, str]:
        """Evaluate quality gate in strict mode."""
        if validation_result.gate_status == QualityGateStatus.FAILED:
            return False, "Quality gate failed - validation errors detected"
        
        if validation_result.validation_metrics.quality_score < gate.pass_threshold:
            return False, f"Quality score {validation_result.validation_metrics.quality_score:.1f}% below threshold {gate.pass_threshold}%"
        
        return True, "Quality gate passed in strict mode"
    
    def _evaluate_permissive_mode(
        self,
        gate: QualityGate,
        validation_result: StageValidationResult
    ) -> Tuple[bool, str]:
        """Evaluate quality gate in permissive mode."""
        # Permissive mode only fails on critical errors
        critical_failures = [
            rule_id for rule_id, passed in validation_result.rule_results.items()
            if not passed and any(
                rule.rule_id == rule_id and rule.severity == "critical"
                for rule in gate.validation_rules
            )
        ]
        
        if critical_failures:
            return False, f"Critical validation rules failed: {critical_failures}"
        
        return True, "Quality gate passed in permissive mode"
    
    def _evaluate_selective_mode(
        self,
        gate: QualityGate,
        validation_result: StageValidationResult
    ) -> Tuple[bool, str]:
        """Evaluate quality gate in selective mode."""
        # Check critical rules first
        critical_failures = [
            rule_id for rule_id, passed in validation_result.rule_results.items()
            if not passed and any(
                rule.rule_id == rule_id and rule.severity == "critical"
                for rule in gate.validation_rules
            )
        ]
        
        if critical_failures:
            return False, f"Critical validation rules failed: {critical_failures}"
        
        # Check overall quality score
        if validation_result.validation_metrics.quality_score < gate.pass_threshold:
            # Allow warnings but don't fail
            return True, f"Quality gate passed with warnings - score {validation_result.validation_metrics.quality_score:.1f}%"
        
        return True, "Quality gate passed in selective mode"


class ValidationPipeline(BasePipeline):
    """
    Main validation pipeline that integrates validation at each ETL stage.
    
    Orchestrates validation across the entire ETL pipeline, managing
    quality gates, validation reporting, and data quality assurance.
    """
    
    def __init__(
        self,
        name: str = "validation_pipeline",
        validation_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialise validation pipeline.
        
        Args:
            name: Pipeline name
            validation_config: Validation configuration
            **kwargs: Additional pipeline arguments
        """
        super().__init__(name, **kwargs)
        
        self.validation_config = validation_config or {}
        
        # Initialise components
        self.stage_validators: Dict[str, StageValidator] = {}
        self.quality_gatekeeper = QualityGatekeeper(validation_config)
        self.validation_reporter = ValidationReporter()
        
        # Validation results storage
        self.stage_validation_results: Dict[str, StageValidationResult] = {}
        self.pipeline_validation_summary: Optional[ValidationSummary] = None
        
        logger.log.info(
            "Validation pipeline initialised",
            name=name,
            config_keys=list(self.validation_config.keys())
        )
    
    def define_stages(self) -> List[str]:
        """Define validation pipeline stages."""
        return [
            "pre_extraction_validation",
            "post_extraction_validation",
            "pre_transformation_validation",
            "post_transformation_validation",
            "pre_integration_validation",
            "post_integration_validation",
            "pre_loading_validation",
            "post_loading_validation",
            "final_quality_assessment"
        ]
    
    def execute_stage(self, stage_name: str, context: PipelineContext) -> Any:
        """
        Execute a validation stage.
        
        Args:
            stage_name: Name of the validation stage
            context: Pipeline context
            
        Returns:
            Validation results for the stage
        """
        logger.log.info("Executing validation stage", stage=stage_name)
        
        # Get data from context
        data = self._get_stage_data(stage_name, context)
        if data is None:
            logger.log.warning("No data available for validation", stage=stage_name)
            return None
        
        # Get or create stage validator
        validator = self._get_stage_validator(stage_name)
        
        # Execute validation
        validation_result = validator.validate_stage_data(data, context)
        
        # Store result
        self.stage_validation_results[stage_name] = validation_result
        
        # Evaluate quality gate
        gate_passed, gate_message = self.quality_gatekeeper.evaluate_quality_gate(
            stage_name, validation_result, context
        )
        
        if not gate_passed:
            if validation_result.gate_status == QualityGateStatus.FAILED:
                raise ValidationError(f"Quality gate failed for stage {stage_name}: {gate_message}")
            else:
                logger.log.warning(
                    "Quality gate warning",
                    stage=stage_name,
                    message=gate_message
                )
        
        # Generate stage report
        self._generate_stage_report(stage_name, validation_result)
        
        return validation_result
    
    def generate_pipeline_validation_summary(self) -> ValidationSummary:
        """Generate comprehensive validation summary for the entire pipeline."""
        try:
            # Calculate overall metrics
            total_records = sum(
                result.validation_metrics.total_records
                for result in self.stage_validation_results.values()
            )
            
            total_rules = sum(
                result.validation_metrics.rules_executed
                for result in self.stage_validation_results.values()
            )
            
            passed_rules = sum(
                result.validation_metrics.rules_passed
                for result in self.stage_validation_results.values()
            )
            
            overall_quality_score = (passed_rules / total_rules * 100) if total_rules > 0 else 0
            
            # Create summary
            summary = ValidationSummary(
                summary_id=f"pipeline_{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                validation_timestamp=datetime.now(),
                total_records_processed=total_records,
                total_validation_rules=total_rules,
                passed_validation_rules=passed_rules,
                failed_validation_rules=total_rules - passed_rules,
                overall_quality_score=overall_quality_score,
                stage_summaries={
                    stage: {
                        "quality_score": result.validation_metrics.quality_score,
                        "gate_status": result.gate_status.value,
                        "errors_count": len(result.errors),
                        "warnings_count": len(result.warnings)
                    }
                    for stage, result in self.stage_validation_results.items()
                },
                recommendations=self._generate_pipeline_recommendations(),
                compliance_status=self._assess_compliance_status()
            )
            
            self.pipeline_validation_summary = summary
            
            logger.log.info(
                "Pipeline validation summary generated",
                overall_quality_score=overall_quality_score,
                stages_validated=len(self.stage_validation_results)
            )
            
            return summary
            
        except Exception as e:
            logger.log.error(
                "Failed to generate pipeline validation summary",
                error=str(e)
            )
            raise ValidationError(f"Summary generation failed: {str(e)}") from e
    
    def _get_stage_data(self, stage_name: str, context: PipelineContext) -> Optional[pd.DataFrame]:
        """Get data for validation stage."""
        # Map validation stages to data sources
        data_mapping = {
            "pre_extraction_validation": "source_data",
            "post_extraction_validation": "extracted_data",
            "pre_transformation_validation": "extracted_data",
            "post_transformation_validation": "transformed_data",
            "pre_integration_validation": "transformed_data",
            "post_integration_validation": "integrated_data",
            "pre_loading_validation": "integrated_data",
            "post_loading_validation": "loaded_data",
            "final_quality_assessment": "final_data"
        }
        
        data_key = data_mapping.get(stage_name, "current_data")
        return context.get_output(data_key)
    
    def _get_stage_validator(self, stage_name: str) -> StageValidator:
        """Get or create stage validator."""
        if stage_name not in self.stage_validators:
            # Create quality gate for stage
            quality_gate = self._create_default_quality_gate(stage_name)
            
            # Create validator
            validator = StageValidator(
                stage_name=stage_name,
                quality_gate=quality_gate,
                validation_config=self.validation_config
            )
            
            self.stage_validators[stage_name] = validator
        
        return self.stage_validators[stage_name]
    
    def _create_default_quality_gate(self, stage_name: str) -> QualityGate:
        """Create default quality gate for a stage."""
        # Default validation rules based on stage
        default_rules = [
            ValidationRule(
                rule_id=f"{stage_name}_schema_check",
                rule_name="Schema Validation",
                rule_type="schema",
                severity="critical",
                action=ValidationAction.HALT
            ),
            ValidationRule(
                rule_id=f"{stage_name}_completeness_check",
                rule_name="Data Completeness",
                rule_type="business",
                severity="high",
                action=ValidationAction.WARNING,
                threshold=95.0
            ),
            ValidationRule(
                rule_id=f"{stage_name}_quality_check",
                rule_name="Data Quality",
                rule_type="statistical",
                severity="medium",
                action=ValidationAction.WARNING,
                threshold=90.0
            )
        ]
        
        return QualityGate(
            gate_id=f"{stage_name}_gate",
            gate_name=f"{stage_name.title()} Quality Gate",
            stage_name=stage_name,
            validation_rules=default_rules,
            pass_threshold=90.0,
            mode=ValidationMode.SELECTIVE
        )
    
    def _generate_stage_report(
        self,
        stage_name: str,
        validation_result: StageValidationResult
    ) -> None:
        """Generate validation report for a stage."""
        try:
            report_data = {
                "stage_name": stage_name,
                "validation_result": validation_result,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save report to file
            report_path = Path(f"logs/validation_reports/{stage_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.log.debug(
                "Stage validation report generated",
                stage=stage_name,
                report_path=str(report_path)
            )
            
        except Exception as e:
            logger.log.error(
                "Failed to generate stage report",
                stage=stage_name,
                error=str(e)
            )
    
    def _generate_pipeline_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Analyse validation results
        for stage_name, result in self.stage_validation_results.items():
            if result.validation_metrics.quality_score < 95:
                recommendations.append(
                    f"Improve data quality for {stage_name} - current score: {result.validation_metrics.quality_score:.1f}%"
                )
            
            if result.validation_metrics.completeness_score < 90:
                recommendations.append(
                    f"Address data completeness issues in {stage_name} - current score: {result.validation_metrics.completeness_score:.1f}%"
                )
            
            if len(result.errors) > 0:
                recommendations.append(
                    f"Resolve {len(result.errors)} validation errors in {stage_name}"
                )
        
        return recommendations
    
    def _assess_compliance_status(self) -> Dict[str, str]:
        """Assess compliance status against standards."""
        compliance_status = {}
        
        # Check against Australian health data standards
        standards = ["AIHW", "ABS", "Medicare", "PBS"]
        
        for standard in standards:
            # Simplified compliance assessment
            overall_quality = np.mean([
                result.validation_metrics.quality_score
                for result in self.stage_validation_results.values()
            ])
            
            if overall_quality >= 95:
                compliance_status[standard] = "COMPLIANT"
            elif overall_quality >= 90:
                compliance_status[standard] = "MINOR_ISSUES"
            else:
                compliance_status[standard] = "NON_COMPLIANT"
        
        return compliance_status