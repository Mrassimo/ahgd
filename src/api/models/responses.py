"""
Response models for the AHGD Data Quality API.

This module defines all response DTOs (Data Transfer Objects) returned by API endpoints,
following British English conventions and providing comprehensive data structures.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

from pydantic import Field, computed_field
from pydantic.types import PositiveInt, NonNegativeInt, PositiveFloat

from .common import (
    AHGDBaseModel, PaginatedResponse, QualityScore, ValidationResult, 
    ValidationSummary, PipelineRun, PipelineStageResult, MetricValue, 
    SystemHealth, SA1Code, GeographicCoordinates
)


class QualityMetricsResponse(PaginatedResponse):
    """Response model for quality metrics endpoint."""
    
    metrics: QualityScore = Field(..., description="Overall quality metrics")
    geographic_breakdown: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Quality metrics by geographic region"
    )
    source_breakdown: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Quality metrics by data source"
    )
    trends: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Quality trends over time"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Data quality improvement recommendations"
    )
    
    @computed_field
    @property
    def quality_grade(self) -> str:
        """Compute overall quality grade based on score."""
        score = self.metrics.overall_score
        if score >= 95:
            return "Excellent"
        elif score >= 85:
            return "Good"
        elif score >= 70:
            return "Satisfactory"
        elif score >= 50:
            return "Needs Improvement"
        else:
            return "Poor"


class ValidationResponse(PaginatedResponse):
    """Response model for data validation endpoint."""
    
    validation_summary: ValidationSummary = Field(
        ...,
        description="Summary of validation results"
    )
    validation_results: List[ValidationResult] = Field(
        default_factory=list,
        description="Detailed validation results"
    )
    dataset_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata about validated dataset"
    )
    geographic_coverage: Optional[Dict[str, Any]] = Field(
        None,
        description="Geographic coverage analysis"
    )
    
    @computed_field
    @property
    def validation_status(self) -> str:
        """Overall validation status."""
        return "PASSED" if self.validation_summary.overall_valid else "FAILED"


class PipelineRunResponse(AHGDBaseModel):
    """Response model for pipeline execution."""
    
    pipeline_run: PipelineRun = Field(..., description="Pipeline run information")
    stage_results: List[PipelineStageResult] = Field(
        default_factory=list,
        description="Results for each pipeline stage"
    )
    logs_url: Optional[str] = Field(
        None,
        description="URL to access detailed logs"
    )
    artifacts_url: Optional[str] = Field(
        None,
        description="URL to access pipeline artifacts"
    )
    next_actions: List[str] = Field(
        default_factory=list,
        description="Recommended next actions"
    )
    
    @computed_field
    @property
    def estimated_completion(self) -> Optional[datetime]:
        """Estimate completion time based on current progress."""
        if self.pipeline_run.status.value in ['completed', 'failed', 'cancelled']:
            return self.pipeline_run.end_time
        
        # Simple estimation based on completed stages
        if self.pipeline_run.completed_stages > 0:
            avg_stage_time = (
                self.pipeline_run.duration_seconds or 0
            ) / self.pipeline_run.completed_stages
            remaining_stages = (
                self.pipeline_run.total_stages - self.pipeline_run.completed_stages
            )
            estimated_seconds = avg_stage_time * remaining_stages
            
            if self.pipeline_run.start_time:
                from datetime import timedelta
                return self.pipeline_run.start_time + timedelta(seconds=estimated_seconds)
        
        return None


class DataExportResponse(AHGDBaseModel):
    """Response model for data export."""
    
    export_id: str = Field(..., description="Unique export identifier")
    download_url: str = Field(..., description="URL to download export file")
    file_size: PositiveInt = Field(..., description="Export file size in bytes")
    record_count: NonNegativeInt = Field(..., description="Number of exported records")
    format: str = Field(..., description="Export file format")
    expires_at: datetime = Field(..., description="Download URL expiration")
    checksum: str = Field(..., description="File integrity checksum")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Export metadata"
    )
    
    @computed_field
    @property
    def file_size_mb(self) -> float:
        """File size in megabytes."""
        return round(self.file_size / (1024 * 1024), 2)


class GeographicAnalysisResponse(AHGDBaseModel):
    """Response for geographic analysis queries."""
    
    sa1_regions: List[SA1Code] = Field(
        default_factory=list,
        description="SA1 regions included in analysis"
    )
    coverage_statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Geographic coverage statistics"
    )
    coordinate_bounds: Optional[GeographicCoordinates] = Field(
        None,
        description="Bounding coordinates of analysis area"
    )
    population_coverage: Optional[int] = Field(
        None,
        description="Estimated population covered"
    )
    quality_by_region: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Quality metrics by geographic region"
    )


class QualityAnalysisResponse(PaginatedResponse):
    """Response for detailed quality analysis."""
    
    overall_assessment: QualityScore = Field(
        ...,
        description="Overall quality assessment"
    )
    dimensional_analysis: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Analysis by quality dimension"
    )
    geographic_analysis: Optional[GeographicAnalysisResponse] = Field(
        None,
        description="Geographic analysis results"
    )
    temporal_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Temporal quality trends"
    )
    comparative_analysis: Optional[Dict[str, Any]] = Field(
        None,
        description="Comparative analysis results"
    )
    visualisation_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Data for quality visualisations"
    )
    improvement_recommendations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Prioritised improvement recommendations"
    )
    
    @computed_field
    @property
    def risk_level(self) -> str:
        """Overall data quality risk level."""
        score = self.overall_assessment.overall_score
        if score >= 90:
            return "Low"
        elif score >= 75:
            return "Medium"
        elif score >= 60:
            return "High"
        else:
            return "Critical"


class MonitoringConfigResponse(AHGDBaseModel):
    """Response for monitoring configuration."""
    
    current_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current monitoring configuration"
    )
    available_metrics: List[str] = Field(
        default_factory=list,
        description="Available metrics for monitoring"
    )
    alert_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent alert history"
    )
    system_status: SystemHealth = Field(
        ...,
        description="Current system health status"
    )
    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="Configuration last updated timestamp"
    )


class DataIntegrationResponse(AHGDBaseModel):
    """Response for data integration operations."""
    
    integration_id: str = Field(..., description="Integration operation ID")
    status: str = Field(..., description="Integration status")
    source_summary: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Summary of source datasets"
    )
    integration_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Integration operation summary"
    )
    conflict_resolution_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of resolved data conflicts"
    )
    validation_results: Optional[ValidationSummary] = Field(
        None,
        description="Post-integration validation results"
    )
    output_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Output dataset metadata"
    )
    
    @computed_field
    @property
    def integration_success_rate(self) -> float:
        """Calculate integration success rate as percentage."""
        if not self.source_summary:
            return 0.0
        
        successful = sum(
            1 for source in self.source_summary 
            if source.get('status') == 'success'
        )
        return round((successful / len(self.source_summary)) * 100, 2)


class MetricsStreamResponse(AHGDBaseModel):
    """Response for real-time metrics streaming."""
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Metrics timestamp"
    )
    metrics: List[MetricValue] = Field(
        default_factory=list,
        description="Current metric values"
    )
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active alerts"
    )
    system_status: str = Field(
        "healthy",
        description="Overall system status"
    )
    update_frequency: int = Field(
        5,
        description="Update frequency in seconds"
    )


class SearchResponse(PaginatedResponse):
    """Generic search response model."""
    
    results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Search results"
    )
    search_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Search operation metadata"
    )
    facets: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None,
        description="Search facets for filtering"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Search suggestions"
    )
    
    @computed_field
    @property
    def search_quality(self) -> str:
        """Assess search result quality."""
        if self.total_count == 0:
            return "No Results"
        elif self.total_count <= 5:
            return "Limited Results"
        elif self.total_count <= 50:
            return "Good Results"
        else:
            return "Comprehensive Results"


class StatusResponse(AHGDBaseModel):
    """Generic status response for long-running operations."""
    
    operation_id: str = Field(..., description="Operation identifier")
    status: str = Field(..., description="Current status")
    progress_percentage: float = Field(
        0.0,
        ge=0,
        le=100,
        description="Completion percentage"
    )
    current_step: Optional[str] = Field(
        None,
        description="Current operation step"
    )
    estimated_completion: Optional[datetime] = Field(
        None,
        description="Estimated completion time"
    )
    result_url: Optional[str] = Field(
        None,
        description="URL to access results when complete"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if failed"
    )


class BulkOperationResponse(AHGDBaseModel):
    """Response for bulk operations."""
    
    operation_id: str = Field(..., description="Bulk operation ID")
    total_items: NonNegativeInt = Field(..., description="Total items to process")
    processed_items: NonNegativeInt = Field(0, description="Items processed")
    successful_items: NonNegativeInt = Field(0, description="Successfully processed items")
    failed_items: NonNegativeInt = Field(0, description="Failed items")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Processing errors"
    )
    status: str = Field("processing", description="Operation status")
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="Operation start time"
    )
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.processed_items == 0:
            return 0.0
        return round((self.successful_items / self.processed_items) * 100, 2)
    
    @computed_field
    @property
    def progress_percentage(self) -> float:
        """Calculate progress as percentage."""
        if self.total_items == 0:
            return 0.0
        return round((self.processed_items / self.total_items) * 100, 2)


class AlertResponse(AHGDBaseModel):
    """Response model for alerts and notifications."""
    
    alert_id: str = Field(..., description="Alert identifier")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    triggered_at: datetime = Field(..., description="When alert was triggered")
    resolved_at: Optional[datetime] = Field(None, description="When alert was resolved")
    affected_resources: List[str] = Field(
        default_factory=list,
        description="Resources affected by this alert"
    )
    recommended_actions: List[str] = Field(
        default_factory=list,
        description="Recommended actions to resolve alert"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional alert metadata"
    )
    
    @computed_field
    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.resolved_at is None
    
    @computed_field
    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate alert duration in minutes."""
        if self.resolved_at:
            delta = self.resolved_at - self.triggered_at
            return round(delta.total_seconds() / 60, 2)
        else:
            delta = datetime.now() - self.triggered_at
            return round(delta.total_seconds() / 60, 2)


# WebSocket message responses
class WebSocketResponse(AHGDBaseModel):
    """Base WebSocket message response."""
    
    message_type: str = Field(..., description="Message type")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Message timestamp"
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message payload"
    )
    subscription_id: Optional[str] = Field(
        None,
        description="Associated subscription ID"
    )


class SubscriptionResponse(AHGDBaseModel):
    """WebSocket subscription response."""
    
    subscription_id: str = Field(..., description="Subscription identifier")
    subscription_type: str = Field(..., description="Type of subscription")
    status: str = Field("active", description="Subscription status")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="Subscription creation time"
    )
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="Applied subscription filters"
    )
    update_frequency: int = Field(
        5,
        description="Update frequency in seconds"
    )


# Export commonly used response models
__all__ = [
    'QualityMetricsResponse',
    'ValidationResponse',
    'PipelineRunResponse', 
    'DataExportResponse',
    'GeographicAnalysisResponse',
    'QualityAnalysisResponse',
    'MonitoringConfigResponse',
    'DataIntegrationResponse',
    'MetricsStreamResponse',
    'SearchResponse',
    'StatusResponse',
    'BulkOperationResponse',
    'AlertResponse',
    'WebSocketResponse',
    'SubscriptionResponse'
]