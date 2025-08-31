"""
Request models for the AHGD Data Quality API.

This module defines all request DTOs (Data Transfer Objects) used by API endpoints,
following British English conventions and SA1-centric geographic structure.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import Field, field_validator, model_validator
from pydantic.types import PositiveInt, NonNegativeInt

from .common import AHGDBaseModel, GeographicLevel, DataFormat, PipelineStage, SeverityEnum


class QualityMetricsRequest(AHGDBaseModel):
    """Request model for quality metrics endpoint."""
    
    geographic_level: GeographicLevel = Field(
        GeographicLevel.SA1,
        description="Geographic level for analysis (default: SA1)"
    )
    start_date: Optional[datetime] = Field(
        None, 
        description="Start date for analysis period"
    )
    end_date: Optional[datetime] = Field(
        None, 
        description="End date for analysis period"
    )
    include_trends: bool = Field(
        False,
        description="Include trend analysis over time"
    )
    group_by_source: bool = Field(
        False,
        description="Group metrics by data source"
    )
    
    @model_validator(mode='after')
    def validate_date_range(self):
        """Validate that end_date is after start_date if both provided."""
        if self.start_date and self.end_date and self.end_date <= self.start_date:
            raise ValueError('end_date must be after start_date')
        return self


class ValidationRequest(AHGDBaseModel):
    """Request model for data validation endpoint."""
    
    dataset_id: Optional[str] = Field(
        None,
        description="Specific dataset identifier (optional)"
    )
    validation_types: List[str] = Field(
        default=["schema", "business_rules", "geographic"],
        description="Types of validation to perform"
    )
    severity_threshold: SeverityEnum = Field(
        SeverityEnum.WARNING,
        description="Minimum severity level to report"
    )
    include_summary: bool = Field(
        True,
        description="Include validation summary"
    )
    max_errors: PositiveInt = Field(
        1000,
        description="Maximum number of errors to return per rule"
    )
    
    @field_validator('validation_types')
    @classmethod
    def validate_validation_types(cls, v):
        """Validate validation types."""
        valid_types = {
            'schema', 'business_rules', 'geographic', 
            'statistical', 'completeness', 'consistency'
        }
        invalid_types = set(v) - valid_types
        if invalid_types:
            raise ValueError(f'Invalid validation types: {invalid_types}. '
                           f'Valid types: {valid_types}')
        return v


class PipelineRunRequest(AHGDBaseModel):
    """Request model for pipeline execution."""
    
    pipeline_name: str = Field(..., description="Pipeline identifier")
    stage: Optional[PipelineStage] = Field(
        None,
        description="Specific stage to run (optional - runs all if not specified)"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline-specific parameters"
    )
    force_rerun: bool = Field(
        False,
        description="Force rerun even if recent successful run exists"
    )
    notification_email: Optional[str] = Field(
        None,
        description="Email for completion notification"
    )
    
    @field_validator('notification_email')
    @classmethod
    def validate_email(cls, v):
        """Validate email format if provided."""
        if v and '@' not in v:
            raise ValueError('Invalid email format')
        return v


class DataExportRequest(AHGDBaseModel):
    """Request model for data export."""
    
    format: DataFormat = Field(DataFormat.CSV, description="Export format")
    geographic_level: GeographicLevel = Field(
        GeographicLevel.SA1,
        description="Geographic level for export"
    )
    include_metadata: bool = Field(
        True,
        description="Include metadata in export"
    )
    compress: bool = Field(
        False,
        description="Compress export file"
    )
    date_range: Optional[Dict[str, datetime]] = Field(
        None,
        description="Date range filter (start_date, end_date)"
    )
    columns: Optional[List[str]] = Field(
        None,
        description="Specific columns to export (optional)"
    )
    max_records: Optional[PositiveInt] = Field(
        None,
        description="Maximum number of records to export"
    )
    
    @model_validator(mode='after')
    def validate_date_range_dict(self):
        """Validate date range dictionary if provided."""
        if self.date_range:
            start = self.date_range.get('start_date')
            end = self.date_range.get('end_date')
            if start and end and end <= start:
                raise ValueError('end_date must be after start_date in date_range')
        return self


class GeographicQuery(AHGDBaseModel):
    """Geographic query parameters."""
    
    sa1_codes: Optional[List[str]] = Field(
        None,
        description="Specific SA1 codes to include"
    )
    state_codes: Optional[List[str]] = Field(
        None, 
        description="State/Territory codes to filter by"
    )
    postcode_filter: Optional[List[str]] = Field(
        None,
        description="Postcode filter"
    )
    bounding_box: Optional[Dict[str, float]] = Field(
        None,
        description="Geographic bounding box (lat_min, lat_max, lon_min, lon_max)"
    )
    
    @field_validator('sa1_codes')
    @classmethod
    def validate_sa1_codes(cls, v):
        """Validate SA1 code format."""
        if v:
            import re
            for code in v:
                if not re.match(r'^\d{11}$', str(code).strip()):
                    raise ValueError(f'Invalid SA1 code format: {code}. Must be 11 digits.')
        return v
    
    @field_validator('state_codes')
    @classmethod
    def validate_state_codes(cls, v):
        """Validate Australian state codes."""
        if v:
            valid_states = {'NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'}
            invalid_states = set(str(code).upper() for code in v) - valid_states
            if invalid_states:
                raise ValueError(f'Invalid state codes: {invalid_states}. '
                               f'Valid codes: {valid_states}')
        return [code.upper() for code in v]
    
    @field_validator('bounding_box')
    @classmethod
    def validate_bounding_box(cls, v):
        """Validate bounding box coordinates."""
        if v:
            required_keys = {'lat_min', 'lat_max', 'lon_min', 'lon_max'}
            if not all(key in v for key in required_keys):
                raise ValueError(f'Bounding box must contain: {required_keys}')
            
            if v['lat_min'] >= v['lat_max']:
                raise ValueError('lat_min must be less than lat_max')
            if v['lon_min'] >= v['lon_max']:
                raise ValueError('lon_min must be less than lon_max')
            
            # Validate coordinate ranges for Australia
            if not (-54 <= v['lat_min'] <= -9 and -54 <= v['lat_max'] <= -9):
                raise ValueError('Latitude must be within Australian bounds (-54 to -9)')
            if not (96 <= v['lon_min'] <= 168 and 96 <= v['lon_max'] <= 168):
                raise ValueError('Longitude must be within Australian bounds (96 to 168)')
                
        return v


class QualityAnalysisRequest(AHGDBaseModel):
    """Request for detailed quality analysis."""
    
    geographic_query: Optional[GeographicQuery] = Field(
        None,
        description="Geographic filtering parameters"
    )
    analysis_type: str = Field(
        "comprehensive",
        description="Type of analysis to perform"
    )
    include_visualisations: bool = Field(
        False,
        description="Include chart data in response"
    )
    benchmark_against: Optional[str] = Field(
        None,
        description="Benchmark dataset for comparison"
    )
    custom_rules: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Custom validation rules to apply"
    )
    
    @field_validator('analysis_type')
    @classmethod
    def validate_analysis_type(cls, v):
        """Validate analysis type."""
        valid_types = {'comprehensive', 'summary', 'trends', 'comparative'}
        if v not in valid_types:
            raise ValueError(f'analysis_type must be one of: {valid_types}')
        return v


class MonitoringConfigRequest(AHGDBaseModel):
    """Request for monitoring configuration updates."""
    
    alert_thresholds: Optional[Dict[str, float]] = Field(
        None,
        description="Alert threshold configurations"
    )
    notification_channels: Optional[List[str]] = Field(
        None,
        description="Notification channel configurations"
    )
    monitoring_frequency: Optional[str] = Field(
        None,
        description="Monitoring frequency (hourly, daily, weekly)"
    )
    enabled_checks: Optional[List[str]] = Field(
        None,
        description="List of monitoring checks to enable"
    )
    
    @field_validator('monitoring_frequency')
    @classmethod
    def validate_frequency(cls, v):
        """Validate monitoring frequency."""
        if v:
            valid_frequencies = {'hourly', 'daily', 'weekly'}
            if v not in valid_frequencies:
                raise ValueError(f'monitoring_frequency must be one of: {valid_frequencies}')
        return v


class DataIntegrationRequest(AHGDBaseModel):
    """Request for data integration operations."""
    
    source_datasets: List[str] = Field(
        ...,
        description="List of source dataset identifiers"
    )
    integration_method: str = Field(
        "standard",
        description="Integration method to use"
    )
    target_schema: Optional[str] = Field(
        None,
        description="Target schema version"
    )
    conflict_resolution: str = Field(
        "latest",
        description="How to resolve data conflicts"
    )
    validation_level: str = Field(
        "standard", 
        description="Level of validation to apply"
    )
    
    @field_validator('integration_method')
    @classmethod
    def validate_integration_method(cls, v):
        """Validate integration method."""
        valid_methods = {'standard', 'merge', 'append', 'replace'}
        if v not in valid_methods:
            raise ValueError(f'integration_method must be one of: {valid_methods}')
        return v
    
    @field_validator('conflict_resolution')
    @classmethod
    def validate_conflict_resolution(cls, v):
        """Validate conflict resolution strategy."""
        valid_strategies = {'latest', 'oldest', 'highest_quality', 'manual'}
        if v not in valid_strategies:
            raise ValueError(f'conflict_resolution must be one of: {valid_strategies}')
        return v
    
    @field_validator('validation_level')
    @classmethod
    def validate_validation_level(cls, v):
        """Validate validation level."""
        valid_levels = {'minimal', 'standard', 'comprehensive', 'strict'}
        if v not in valid_levels:
            raise ValueError(f'validation_level must be one of: {valid_levels}')
        return v


# Pagination and filtering requests
class PaginationRequest(AHGDBaseModel):
    """Standard pagination parameters."""
    
    page: PositiveInt = Field(1, description="Page number (1-based)")
    page_size: PositiveInt = Field(
        50, 
        le=1000,
        description="Number of items per page (max 1000)"
    )
    sort_by: Optional[str] = Field(
        None,
        description="Field to sort by"
    )
    sort_order: str = Field(
        "asc",
        description="Sort order (asc/desc)"
    )
    
    @field_validator('sort_order')
    @classmethod
    def validate_sort_order(cls, v):
        """Validate sort order."""
        if v.lower() not in ['asc', 'desc']:
            raise ValueError('sort_order must be "asc" or "desc"')
        return v.lower()


class FilterRequest(AHGDBaseModel):
    """Standard filtering parameters."""
    
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Field-based filters"
    )
    search_term: Optional[str] = Field(
        None,
        description="General search term"
    )
    date_from: Optional[datetime] = Field(
        None,
        description="Filter records from this date"
    )
    date_to: Optional[datetime] = Field(
        None,
        description="Filter records until this date"
    )
    
    @model_validator(mode='after')
    def validate_date_filter_range(self):
        """Validate date filter range."""
        if self.date_from and self.date_to and self.date_to <= self.date_from:
            raise ValueError('date_to must be after date_from')
        return self


# WebSocket subscription requests
class SubscriptionRequest(AHGDBaseModel):
    """WebSocket subscription request."""
    
    subscription_type: str = Field(..., description="Type of subscription")
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Subscription filters"
    )
    update_frequency: Optional[int] = Field(
        5,
        ge=1,
        le=60,
        description="Update frequency in seconds (1-60)"
    )
    
    @field_validator('subscription_type')
    @classmethod
    def validate_subscription_type(cls, v):
        """Validate subscription type."""
        valid_types = {
            'quality_metrics', 'validation_results', 'pipeline_status',
            'system_health', 'alerts'
        }
        if v not in valid_types:
            raise ValueError(f'subscription_type must be one of: {valid_types}')
        return v


# Export commonly used request models
__all__ = [
    'QualityMetricsRequest',
    'ValidationRequest', 
    'PipelineRunRequest',
    'DataExportRequest',
    'GeographicQuery',
    'QualityAnalysisRequest',
    'MonitoringConfigRequest',
    'DataIntegrationRequest',
    'PaginationRequest',
    'FilterRequest',
    'SubscriptionRequest'
]