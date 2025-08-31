"""
Common Pydantic models for the AHGD Data Quality API.

This module defines shared data models used across the API, following
British English conventions and integrating with the existing AHGD
validation and monitoring infrastructure.
"""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.types import PositiveFloat, PositiveInt, NonNegativeInt


class AHGDBaseModel(BaseModel):
    """Base model with common AHGD configuration and British English conventions."""
    
    model_config = {
        # British English configuration
        "use_enum_values": True,
        "validate_assignment": True,
        "str_strip_whitespace": True,
        "str_to_lower": False,
        # Optimisations for performance
        "validate_default": True,
    }


class StatusEnum(str, Enum):
    """Common status enumeration following British English."""
    
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SeverityEnum(str, Enum):
    """Severity levels for alerts and validation."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class GeographicLevel(str, Enum):
    """Australian geographic levels supported by AHGD."""
    
    SA1 = "sa1"  # Primary focus for AHGD
    SA2 = "sa2"  # Legacy support
    SA3 = "sa3"
    SA4 = "sa4"
    LGA = "lga"
    STATE = "state"
    POSTCODE = "postcode"


class DataFormat(str, Enum):
    """Supported data formats."""
    
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet" 
    AVRO = "avro"
    XML = "xml"


class PipelineStage(str, Enum):
    """ETL pipeline stages."""
    
    EXTRACT = "extract"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    LOAD = "load"


# Base response models
class APIResponse(AHGDBaseModel):
    """Standard API response wrapper."""
    
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class PaginatedResponse(APIResponse):
    """Paginated response model."""
    
    total_count: NonNegativeInt
    page_size: PositiveInt
    current_page: PositiveInt
    total_pages: PositiveInt
    has_next: bool
    has_previous: bool
    
    @model_validator(mode='after')
    def calculate_total_pages(self):
        """Calculate total pages based on total_count and page_size."""
        total_count = self.total_count or 0
        page_size = self.page_size or 1
        self.total_pages = max(1, (total_count + page_size - 1) // page_size)
        return self
    
    @model_validator(mode='after')
    def calculate_has_next(self):
        """Calculate if there is a next page."""
        current_page = self.current_page or 1
        total_pages = self.total_pages or 1
        self.has_next = current_page < total_pages
        return self
    
    @model_validator(mode='after')
    def calculate_has_previous(self):
        """Calculate if there is a previous page."""
        current_page = self.current_page or 1
        self.has_previous = current_page > 1
        return self


class ErrorDetail(AHGDBaseModel):
    """Detailed error information."""
    
    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(APIResponse):
    """Error response model."""
    
    success: bool = False
    error: ErrorDetail
    trace_id: Optional[str] = None


# Geographic models
class SA1Code(AHGDBaseModel):
    """SA1 geographic code validation model."""
    
    code: str = Field(..., description="11-digit SA1 code")
    name: Optional[str] = Field(None, description="SA1 area name")
    state: Optional[str] = Field(None, description="State/Territory code")
    
    @field_validator('code')
    @classmethod
    def validate_sa1_code(cls, v):
        """Validate SA1 code format (11 digits)."""
        if not re.match(r'^\d{11}$', str(v).strip()):
            raise ValueError('SA1 code must be exactly 11 digits')
        return str(v).strip()
    
    @field_validator('state')
    @classmethod
    def validate_state_code(cls, v):
        """Validate Australian state/territory codes."""
        if v is not None:
            valid_states = {'NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'}
            if v.upper() not in valid_states:
                raise ValueError(f'Invalid state code. Must be one of: {valid_states}')
            return v.upper()
        return v


class GeographicCoordinates(AHGDBaseModel):
    """Geographic coordinate model."""
    
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees") 
    accuracy_metres: Optional[PositiveFloat] = Field(None, description="Coordinate accuracy in metres")
    source: Optional[str] = Field(None, description="Coordinate source")


# Quality metrics models
class QualityScore(AHGDBaseModel):
    """Data quality score model."""
    
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score (0-100)")
    completeness: float = Field(..., ge=0, le=100, description="Completeness score")
    accuracy: float = Field(..., ge=0, le=100, description="Accuracy score")
    consistency: float = Field(..., ge=0, le=100, description="Consistency score")
    validity: float = Field(..., ge=0, le=100, description="Validity score")
    timeliness: float = Field(..., ge=0, le=100, description="Timeliness score")
    
    calculated_at: datetime = Field(default_factory=datetime.now)
    record_count: PositiveInt = Field(..., description="Number of records assessed")


class ValidationRule(AHGDBaseModel):
    """Data validation rule definition."""
    
    rule_id: str = Field(..., description="Unique rule identifier")
    rule_type: str = Field(..., description="Type of validation rule")
    description: str = Field(..., description="Human-readable rule description")
    severity: SeverityEnum = Field(..., description="Rule violation severity")
    enabled: bool = Field(True, description="Whether rule is active")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Rule parameters")


class ValidationResult(AHGDBaseModel):
    """Individual validation result."""
    
    rule_id: str = Field(..., description="Rule that generated this result")
    is_valid: bool = Field(..., description="Whether validation passed")
    severity: SeverityEnum = Field(..., description="Result severity")
    message: str = Field(..., description="Validation message")
    affected_records: List[int] = Field(default_factory=list, description="Record indices affected")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
    timestamp: datetime = Field(default_factory=datetime.now)


class ValidationSummary(AHGDBaseModel):
    """Summary of validation results."""
    
    total_rules: PositiveInt = Field(..., description="Total rules executed")
    passed_rules: NonNegativeInt = Field(..., description="Rules that passed")
    failed_rules: NonNegativeInt = Field(..., description="Rules that failed")
    error_count: NonNegativeInt = Field(..., description="Total error count")
    warning_count: NonNegativeInt = Field(..., description="Total warning count")
    info_count: NonNegativeInt = Field(..., description="Total info count")
    overall_valid: bool = Field(..., description="Whether validation passed overall")
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="Calculated quality score")
    validation_time: datetime = Field(default_factory=datetime.now)
    
    @model_validator(mode='after')
    def validate_counts(self):
        """Ensure rule counts are consistent."""
        total = self.total_rules or 0
        passed = self.passed_rules or 0 
        failed = self.failed_rules or 0
        
        if passed + failed != total:
            raise ValueError('Passed + failed rules must equal total rules')
        
        return self


# Pipeline models
class PipelineRun(AHGDBaseModel):
    """Pipeline execution run information."""
    
    run_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique run identifier")
    pipeline_name: str = Field(..., description="Pipeline name")
    status: StatusEnum = Field(StatusEnum.PENDING, description="Current status")
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(None, description="Completion time")
    total_stages: PositiveInt = Field(..., description="Total pipeline stages")
    completed_stages: NonNegativeInt = Field(0, description="Completed stages")
    failed_stages: NonNegativeInt = Field(0, description="Failed stages")
    records_processed: NonNegativeInt = Field(0, description="Total records processed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate run duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_stages == 0:
            return 0.0
        return ((self.total_stages - self.failed_stages) / self.total_stages) * 100


class PipelineStageResult(AHGDBaseModel):
    """Individual pipeline stage result."""
    
    stage_name: str = Field(..., description="Stage name")
    status: StatusEnum = Field(..., description="Stage status")
    start_time: datetime = Field(..., description="Stage start time")
    end_time: Optional[datetime] = Field(None, description="Stage end time")
    records_processed: NonNegativeInt = Field(0, description="Records processed")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate stage duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


# Monitoring models
class MetricValue(AHGDBaseModel):
    """Individual metric data point."""
    
    name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(default_factory=datetime.now)
    labels: Optional[Dict[str, str]] = Field(None, description="Metric labels")
    unit: Optional[str] = Field(None, description="Metric unit")


class SystemHealth(AHGDBaseModel):
    """System health status."""
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now)
    cpu_percent: Optional[float] = Field(None, ge=0, le=100, description="CPU utilisation")
    memory_percent: Optional[float] = Field(None, ge=0, le=100, description="Memory utilisation")
    disk_percent: Optional[float] = Field(None, ge=0, le=100, description="Disk utilisation")
    active_pipelines: NonNegativeInt = Field(0, description="Number of active pipelines")
    pending_validations: NonNegativeInt = Field(0, description="Pending validation jobs")
    uptime_seconds: Optional[PositiveFloat] = Field(None, description="System uptime")
    version: Optional[str] = Field(None, description="API version")


# WebSocket models
class WebSocketMessage(AHGDBaseModel):
    """WebSocket message structure."""
    
    message_type: str = Field(..., description="Message type identifier")
    data: Optional[Dict[str, Any]] = Field(None, description="Message payload")
    timestamp: datetime = Field(default_factory=datetime.now)
    sequence: Optional[int] = Field(None, description="Message sequence number")


class LiveMetricsUpdate(AHGDBaseModel):
    """Live metrics update for WebSocket streaming."""
    
    pipeline_name: Optional[str] = Field(None, description="Pipeline name if pipeline-specific")
    metrics: List[MetricValue] = Field(..., description="Updated metrics")
    quality_scores: Optional[QualityScore] = Field(None, description="Latest quality scores")
    system_health: Optional[SystemHealth] = Field(None, description="System health status")
    alerts: Optional[List[Dict[str, Any]]] = Field(None, description="Active alerts")
    update_frequency: Optional[str] = Field(None, description="Update frequency indicator")


# Configuration models
class APIConfiguration(AHGDBaseModel):
    """API configuration model."""
    
    rate_limiting: bool = Field(True, description="Enable rate limiting")
    max_requests_per_minute: PositiveInt = Field(100, description="Max requests per minute")
    enable_cors: bool = Field(True, description="Enable CORS")
    enable_compression: bool = Field(True, description="Enable response compression")
    log_level: str = Field("INFO", description="Logging level")
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    websocket_enabled: bool = Field(True, description="Enable WebSocket endpoints")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()


# Export commonly used models
__all__ = [
    'AHGDBaseModel',
    'StatusEnum', 
    'SeverityEnum',
    'GeographicLevel',
    'DataFormat',
    'PipelineStage',
    'APIResponse',
    'PaginatedResponse', 
    'ErrorResponse',
    'SA1Code',
    'GeographicCoordinates',
    'QualityScore',
    'ValidationRule',
    'ValidationResult',
    'ValidationSummary',
    'PipelineRun',
    'PipelineStageResult',
    'MetricValue',
    'SystemHealth',
    'WebSocketMessage',
    'LiveMetricsUpdate',
    'APIConfiguration'
]