"""
Data Validators Module

This module contains comprehensive validation frameworks for ensuring data quality
and consistency in Australian health geography datasets.

Available Validators:
- BaseValidator: Abstract base class for all validators
- QualityChecker: Comprehensive data quality assessment framework
- AustralianHealthBusinessRulesValidator: Australian health data business rules
- StatisticalValidator: Statistical validation methods and outlier detection
- AdvancedStatisticalValidator: Advanced statistical validation for Australian health data
- GeographicValidator: Geographic and spatial validation
- EnhancedGeographicValidator: Enhanced geographic validation for Australian SA2 data
- ValidationOrchestrator: Coordinated validation pipeline management
- ValidationReporter: Comprehensive validation reporting

Example Usage:
    from src.validators import QualityChecker, ValidationOrchestrator
    
    # Create quality checker
    quality_checker = QualityChecker()
    
    # Validate data
    results = quality_checker.validate(data)
    
    # Or use orchestrator for comprehensive validation
    orchestrator = ValidationOrchestrator()
    all_results = orchestrator.validate_data(data)
"""

from .base import BaseValidator
from .quality_checker import QualityChecker, QualityScore, QualityRule, AnomalyDetectionResult
from .business_rules import AustralianHealthBusinessRulesValidator, BusinessRule, ReferenceDataSet
from .statistical_validator import (
    StatisticalValidator, 
    OutlierDetectionResult, 
    DistributionTestResult, 
    CorrelationAnalysisResult, 
    TrendAnalysisResult
)
from .advanced_statistical import (
    AdvancedStatisticalValidator,
    RangeValidationConfig,
    StatisticalReport
)
from .geographic_validator import (
    GeographicValidator,
    CoordinateValidationResult,
    BoundaryValidationResult,
    TopologyValidationResult,
    CoverageValidationResult
)
from .enhanced_geographic import (
    EnhancedGeographicValidator,
    SA2CoverageResult,
    BoundaryTopologyResult,
    CRSValidationResult,
    SpatialHierarchyResult,
    GeographicConsistencyResult
)
from .validation_orchestrator import (
    ValidationOrchestrator,
    ValidationTask,
    ValidationPipelineConfig,
    ValidationPipelineResult,
    PerformanceMetrics
)
from .reporting import (
    ValidationReporter,
    ValidationReport,
    ValidationSummary,
    DataProfileSummary,
    QualityDimension
)

__all__ = [
    # Base classes
    'BaseValidator',
    
    # Quality checking
    'QualityChecker',
    'QualityScore',
    'QualityRule',
    'AnomalyDetectionResult',
    
    # Business rules
    'AustralianHealthBusinessRulesValidator',
    'BusinessRule',
    'ReferenceDataSet',
    
    # Statistical validation
    'StatisticalValidator',
    'OutlierDetectionResult',
    'DistributionTestResult',
    'CorrelationAnalysisResult',
    'TrendAnalysisResult',
    
    # Advanced statistical validation
    'AdvancedStatisticalValidator',
    'RangeValidationConfig',
    'StatisticalReport',
    
    # Geographic validation
    'GeographicValidator',
    'CoordinateValidationResult',
    'BoundaryValidationResult',
    'TopologyValidationResult',
    'CoverageValidationResult',
    
    # Enhanced geographic validation
    'EnhancedGeographicValidator',
    'SA2CoverageResult',
    'BoundaryTopologyResult',
    'CRSValidationResult',
    'SpatialHierarchyResult',
    'GeographicConsistencyResult',
    
    # Orchestration
    'ValidationOrchestrator',
    'ValidationTask',
    'ValidationPipelineConfig',
    'ValidationPipelineResult',
    'PerformanceMetrics',
    
    # Reporting
    'ValidationReporter',
    'ValidationReport',
    'ValidationSummary',
    'DataProfileSummary',
    'QualityDimension'
]