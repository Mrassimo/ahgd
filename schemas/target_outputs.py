"""
Target output schemas for AHGD project data exports.

This module defines the final export formats, data warehouse schemas,
API response structures, and web platform data formats.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator

from .base_schema import VersionedSchema, DataSource


class ExportFormat(str, Enum):
    """Supported export formats."""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    GEOJSON = "geojson"
    XLSX = "xlsx"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    API_JSON = "api_json"


class CompressionType(str, Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    SNAPPY = "snappy"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"


class APIVersion(str, Enum):
    """API version specifications."""
    V1_0 = "v1.0"
    V1_1 = "v1.1"
    V2_0 = "v2.0"
    LATEST = "latest"


class DataWarehouseTable(VersionedSchema):
    """
    Schema for data warehouse table definitions.
    
    Defines the structure and properties of tables in the target
    data warehouse for analytics and reporting.
    """
    
    # === TABLE IDENTIFICATION ===
    table_name: str = Field(..., description="Table name in data warehouse")
    schema_name: str = Field(..., description="Database schema name")
    table_type: str = Field(..., description="Table type (fact, dimension, aggregate)")
    
    # === TABLE STRUCTURE ===
    columns: List[Dict[str, Any]] = Field(
        ...,
        description="Column definitions with data types and constraints"
    )
    primary_keys: List[str] = Field(
        ...,
        description="Primary key column names"
    )
    foreign_keys: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Foreign key relationships"
    )
    indexes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Index definitions for performance"
    )
    
    # === PARTITIONING ===
    partition_strategy: Optional[str] = Field(
        None,
        description="Partitioning strategy (date, range, hash)"
    )
    partition_columns: List[str] = Field(
        default_factory=list,
        description="Columns used for partitioning"
    )
    
    # === DATA MANAGEMENT ===
    retention_period_days: Optional[int] = Field(
        None,
        ge=1,
        description="Data retention period in days"
    )
    archival_strategy: Optional[str] = Field(
        None,
        description="Data archival strategy"
    )
    
    # === QUALITY AND GOVERNANCE ===
    data_lineage: List[str] = Field(
        default_factory=list,
        description="Source tables/systems feeding this table"
    )
    refresh_frequency: str = Field(
        ...,
        description="Data refresh frequency (daily, weekly, monthly)"
    )
    quality_checks: List[str] = Field(
        default_factory=list,
        description="Data quality checks applied"
    )
    
    # === ACCESS CONTROL ===
    access_level: str = Field(
        ...,
        description="Access level (public, internal, restricted)"
    )
    authorized_users: List[str] = Field(
        default_factory=list,
        description="Authorized user groups"
    )
    
    @field_validator('table_type')
    @classmethod
    def validate_table_type(cls, v: str) -> str:
        """Validate table type."""
        valid_types = {'fact', 'dimension', 'aggregate', 'staging', 'lookup'}
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid table type: {v}")
        return v.lower()
    
    @field_validator('refresh_frequency')
    @classmethod
    def validate_refresh_frequency(cls, v: str) -> str:
        """Validate refresh frequency."""
        valid_frequencies = {
            'real-time', 'hourly', 'daily', 'weekly', 
            'monthly', 'quarterly', 'annually', 'on-demand'
        }
        if v.lower() not in valid_frequencies:
            raise ValueError(f"Invalid refresh frequency: {v}")
        return v.lower()
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "DataWarehouseTable"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate data warehouse table definition."""
        errors = []
        
        # Check primary key columns exist
        column_names = [col['name'] for col in self.columns]
        for pk in self.primary_keys:
            if pk not in column_names:
                errors.append(f"Primary key column '{pk}' not found in columns")
        
        # Check foreign key references
        for fk in self.foreign_keys:
            if 'column' not in fk or 'references' not in fk:
                errors.append("Foreign key missing required fields")
            elif fk['column'] not in column_names:
                errors.append(f"Foreign key column '{fk['column']}' not found")
        
        # Check partition columns exist
        for part_col in self.partition_columns:
            if part_col not in column_names:
                errors.append(f"Partition column '{part_col}' not found")
        
        return errors


class ExportSpecification(VersionedSchema):
    """
    Specification for data export operations.
    
    Defines how data should be exported including format,
    compression, filtering, and destination details.
    """
    
    # === EXPORT IDENTIFICATION ===
    export_name: str = Field(..., description="Name of the export")
    export_description: str = Field(..., description="Description of export purpose")
    export_type: str = Field(..., description="Type of export (full, incremental, filtered)")
    
    # === SOURCE SPECIFICATION ===
    source_tables: List[str] = Field(
        ...,
        description="Source tables/views to export"
    )
    join_conditions: List[str] = Field(
        default_factory=list,
        description="SQL join conditions if multiple tables"
    )
    filter_conditions: List[str] = Field(
        default_factory=list,
        description="Filter conditions to apply"
    )
    
    # === OUTPUT FORMAT ===
    output_format: ExportFormat = Field(..., description="Output format")
    compression: CompressionType = Field(
        default=CompressionType.GZIP,
        description="Compression type"
    )
    encoding: str = Field(
        default="utf-8",
        description="Character encoding"
    )
    
    # === FILE SPECIFICATION ===
    file_naming_pattern: str = Field(
        ...,
        description="File naming pattern with placeholders"
    )
    max_file_size_mb: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum file size before splitting"
    )
    include_headers: bool = Field(
        default=True,
        description="Include column headers (for CSV/Excel)"
    )
    
    # === COLUMN SELECTION ===
    included_columns: List[str] = Field(
        default_factory=list,
        description="Specific columns to include (empty = all)"
    )
    excluded_columns: List[str] = Field(
        default_factory=list,
        description="Columns to exclude"
    )
    column_mappings: Dict[str, str] = Field(
        default_factory=dict,
        description="Column name mappings (internal -> export)"
    )
    
    # === FORMATTING OPTIONS ===
    date_format: str = Field(
        default="%Y-%m-%d",
        description="Date format pattern"
    )
    decimal_places: Optional[int] = Field(
        None,
        ge=0,
        le=10,
        description="Decimal places for numeric fields"
    )
    null_value_representation: str = Field(
        default="",
        description="How to represent null values"
    )
    
    # === DESTINATION ===
    destination_type: str = Field(
        ...,
        description="Destination type (local, s3, azure, http)"
    )
    destination_path: str = Field(
        ...,
        description="Destination path or URL"
    )
    destination_credentials: Optional[str] = Field(
        None,
        description="Credentials reference (not the actual credentials)"
    )
    
    # === SCHEDULING ===
    schedule_expression: Optional[str] = Field(
        None,
        description="Cron expression for scheduled exports"
    )
    timezone: str = Field(
        default="Australia/Sydney",
        description="Timezone for scheduling"
    )
    
    # === QUALITY CONTROLS ===
    row_count_validation: bool = Field(
        default=True,
        description="Validate row count against source"
    )
    data_validation_rules: List[str] = Field(
        default_factory=list,
        description="Data validation rules to apply"
    )
    
    @field_validator('export_type')
    @classmethod
    def validate_export_type(cls, v: str) -> str:
        """Validate export type."""
        valid_types = {'full', 'incremental', 'filtered', 'sample'}
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid export type: {v}")
        return v.lower()
    
    @field_validator('destination_type')
    @classmethod
    def validate_destination_type(cls, v: str) -> str:
        """Validate destination type."""
        valid_types = {'local', 's3', 'azure', 'gcs', 'ftp', 'http', 'database'}
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid destination type: {v}")
        return v.lower()
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "ExportSpecification"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate export specification."""
        errors = []
        
        # Check column selection consistency
        if self.included_columns and self.excluded_columns:
            overlap = set(self.included_columns) & set(self.excluded_columns)
            if overlap:
                errors.append(f"Columns in both included and excluded lists: {overlap}")
        
        # Validate file naming pattern
        if '{' not in self.file_naming_pattern:
            errors.append("File naming pattern should include placeholders")
        
        # Check schedule expression format (basic validation)
        if self.schedule_expression:
            parts = self.schedule_expression.split()
            if len(parts) not in [5, 6]:  # Standard cron or with seconds
                errors.append("Invalid cron expression format")
        
        return errors


class APIResponseSchema(BaseModel):
    """
    Schema for API response structures.
    
    Defines standardised API response formats for different
    types of health data queries and endpoints.
    """
    
    # === RESPONSE METADATA ===
    api_version: APIVersion = Field(..., description="API version")
    response_id: str = Field(..., description="Unique response identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response generation timestamp"
    )
    
    # === REQUEST CONTEXT ===
    endpoint: str = Field(..., description="API endpoint called")
    method: str = Field(..., description="HTTP method used")
    query_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query parameters provided"
    )
    
    # === RESPONSE STATUS ===
    status: str = Field(..., description="Response status (success, error, partial)")
    status_code: int = Field(..., ge=100, le=599, description="HTTP status code")
    message: Optional[str] = Field(None, description="Status message")
    
    # === DATA PAYLOAD ===
    data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        None,
        description="Response data payload"
    )
    
    # === PAGINATION ===
    pagination: Optional[Dict[str, Any]] = Field(
        None,
        description="Pagination information for large result sets"
    )
    
    # === METADATA ===
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata"
    )
    
    # === LINKS ===
    links: Dict[str, str] = Field(
        default_factory=dict,
        description="Related API endpoints and resources"
    )
    
    # === ERROR DETAILS ===
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Error details if status is error"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate response status."""
        valid_statuses = {'success', 'error', 'partial', 'timeout'}
        if v.lower() not in valid_statuses:
            raise ValueError(f"Invalid status: {v}")
        return v.lower()
    
    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate HTTP method."""
        valid_methods = {'GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'HEAD'}
        if v.upper() not in valid_methods:
            raise ValueError(f"Invalid HTTP method: {v}")
        return v.upper()


class WebPlatformDataStructure(VersionedSchema):
    """
    Data structure for web platform consumption.
    
    Defines optimised data structures for web dashboard
    and interactive visualisation components.
    """
    
    # === CONTENT IDENTIFICATION ===
    content_type: str = Field(..., description="Type of content (map, chart, table, summary)")
    content_id: str = Field(..., description="Unique content identifier")
    content_title: str = Field(..., description="Display title")
    content_description: str = Field(..., description="Content description")
    
    # === DATA PAYLOAD ===
    data_structure: str = Field(
        ...,
        description="Data structure type (geojson, timeseries, matrix, hierarchical)"
    )
    data_payload: Dict[str, Any] = Field(
        ...,
        description="Optimised data payload for web consumption"
    )
    
    # === VISUALISATION CONFIGURATION ===
    chart_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Chart.js or similar visualisation configuration"
    )
    map_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Leaflet or similar map configuration"
    )
    table_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Table display configuration"
    )
    
    # === INTERACTIVITY ===
    interactive_elements: List[str] = Field(
        default_factory=list,
        description="Available interactive features"
    )
    filter_options: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Available filter controls"
    )
    drill_down_paths: List[str] = Field(
        default_factory=list,
        description="Available drill-down navigation paths"
    )
    
    # === PERFORMANCE OPTIMISATION ===
    cache_duration_seconds: int = Field(
        default=3600,
        ge=0,
        description="Recommended cache duration"
    )
    lazy_loading: bool = Field(
        default=False,
        description="Whether content supports lazy loading"
    )
    compression_applied: bool = Field(
        default=True,
        description="Whether data is compressed"
    )
    
    # === RESPONSIVE DESIGN ===
    responsive_breakpoints: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Responsive design configurations"
    )
    mobile_optimised: bool = Field(
        default=True,
        description="Whether optimised for mobile devices"
    )
    
    # === ACCESSIBILITY ===
    accessibility_features: List[str] = Field(
        default_factory=list,
        description="Implemented accessibility features"
    )
    alt_text_descriptions: Dict[str, str] = Field(
        default_factory=dict,
        description="Alternative text descriptions for screen readers"
    )
    
    # === DATA FRESHNESS ===
    data_last_updated: datetime = Field(..., description="When underlying data was last updated")
    content_generated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this content structure was generated"
    )
    refresh_trigger: Optional[str] = Field(
        None,
        description="What triggers content refresh"
    )
    
    @field_validator('content_type')
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """Validate content type."""
        valid_types = {
            'map', 'chart', 'table', 'summary', 'dashboard',
            'report', 'widget', 'indicator', 'comparison'
        }
        if v.lower() not in valid_types:
            raise ValueError(f"Invalid content type: {v}")
        return v.lower()
    
    @field_validator('data_structure')
    @classmethod
    def validate_data_structure(cls, v: str) -> str:
        """Validate data structure type."""
        valid_structures = {
            'geojson', 'timeseries', 'matrix', 'hierarchical',
            'graph', 'tree', 'network', 'table', 'key-value'
        }
        if v.lower() not in valid_structures:
            raise ValueError(f"Invalid data structure: {v}")
        return v.lower()
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "WebPlatformDataStructure"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate web platform data structure."""
        errors = []
        
        # Check configuration consistency
        if self.content_type == 'map' and not self.map_config:
            errors.append("Map content type requires map_config")
        elif self.content_type == 'chart' and not self.chart_config:
            errors.append("Chart content type requires chart_config")
        elif self.content_type == 'table' and not self.table_config:
            errors.append("Table content type requires table_config")
        
        # Validate data structure alignment
        if self.content_type == 'map' and self.data_structure != 'geojson':
            errors.append("Map content should use geojson data structure")
        
        # Check cache duration reasonableness
        if self.cache_duration_seconds > 86400:  # More than 24 hours
            errors.append("Cache duration unusually long")
        
        return errors


class DataQualityReport(VersionedSchema):
    """
    Data quality report for exported datasets.
    
    Provides comprehensive quality assessment and metrics
    for exported data to ensure fitness for purpose.
    """
    
    # === REPORT IDENTIFICATION ===
    report_id: str = Field(..., description="Unique report identifier")
    dataset_name: str = Field(..., description="Name of the assessed dataset")
    assessment_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Date of quality assessment"
    )
    
    # === SCOPE ===
    assessment_scope: str = Field(
        ...,
        description="Scope of assessment (full, sample, targeted)"
    )
    records_assessed: int = Field(..., ge=0, description="Number of records assessed")
    columns_assessed: int = Field(..., ge=0, description="Number of columns assessed")
    
    # === COMPLETENESS METRICS ===
    overall_completeness: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Overall data completeness percentage"
    )
    column_completeness: Dict[str, float] = Field(
        ...,
        description="Completeness percentage by column"
    )
    critical_field_completeness: Dict[str, float] = Field(
        default_factory=dict,
        description="Completeness for business-critical fields"
    )
    
    # === ACCURACY METRICS ===
    accuracy_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Overall accuracy score"
    )
    validation_rules_passed: int = Field(..., ge=0, description="Number of validation rules passed")
    validation_rules_failed: int = Field(..., ge=0, description="Number of validation rules failed")
    accuracy_issues: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Identified accuracy issues"
    )
    
    # === CONSISTENCY METRICS ===
    consistency_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Data consistency score"
    )
    duplicate_records: int = Field(..., ge=0, description="Number of duplicate records found")
    inconsistent_formats: Dict[str, int] = Field(
        default_factory=dict,
        description="Format inconsistencies by column"
    )
    referential_integrity_issues: int = Field(
        ..., 
        ge=0, 
        description="Referential integrity violations"
    )
    
    # === VALIDITY METRICS ===
    validity_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Data validity score"
    )
    invalid_values: Dict[str, int] = Field(
        default_factory=dict,
        description="Invalid values count by column"
    )
    range_violations: Dict[str, int] = Field(
        default_factory=dict,
        description="Range constraint violations by column"
    )
    format_violations: Dict[str, int] = Field(
        default_factory=dict,
        description="Format constraint violations by column"
    )
    
    # === TIMELINESS METRICS ===
    timeliness_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Data timeliness score"
    )
    data_freshness_days: float = Field(
        ..., 
        ge=0, 
        description="Average data age in days"
    )
    outdated_records: int = Field(..., ge=0, description="Number of outdated records")
    
    # === OVERALL ASSESSMENT ===
    overall_quality_score: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Overall data quality score"
    )
    quality_grade: str = Field(
        ...,
        description="Quality grade (A, B, C, D, F)"
    )
    fitness_for_purpose: str = Field(
        ...,
        description="Fitness assessment (excellent, good, adequate, poor)"
    )
    
    # === RECOMMENDATIONS ===
    improvement_recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for quality improvement"
    )
    priority_issues: List[str] = Field(
        default_factory=list,
        description="High-priority issues requiring attention"
    )
    
    # === COMPLIANCE ===
    compliance_standards: List[str] = Field(
        default_factory=list,
        description="Compliance standards assessed against"
    )
    compliance_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Compliance status by standard"
    )
    
    @field_validator('quality_grade')
    @classmethod
    def validate_quality_grade(cls, v: str) -> str:
        """Validate quality grade."""
        valid_grades = {'A', 'B', 'C', 'D', 'F'}
        if v.upper() not in valid_grades:
            raise ValueError(f"Invalid quality grade: {v}")
        return v.upper()
    
    @field_validator('fitness_for_purpose')
    @classmethod
    def validate_fitness_assessment(cls, v: str) -> str:
        """Validate fitness for purpose assessment."""
        valid_assessments = {'excellent', 'good', 'adequate', 'poor', 'unfit'}
        if v.lower() not in valid_assessments:
            raise ValueError(f"Invalid fitness assessment: {v}")
        return v.lower()
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "DataQualityReport"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate data quality report integrity."""
        errors = []
        
        # Check score consistency
        if self.overall_quality_score > 100 or self.overall_quality_score < 0:
            errors.append("Overall quality score outside valid range")
        
        # Validate validation rules total
        total_rules = self.validation_rules_passed + self.validation_rules_failed
        if total_rules == 0:
            errors.append("No validation rules assessed")
        
        # Check grade consistency with score
        expected_grade = 'F'
        if self.overall_quality_score >= 90:
            expected_grade = 'A'
        elif self.overall_quality_score >= 80:
            expected_grade = 'B'
        elif self.overall_quality_score >= 70:
            expected_grade = 'C'
        elif self.overall_quality_score >= 60:
            expected_grade = 'D'
        
        if self.quality_grade != expected_grade:
            errors.append(f"Quality grade {self.quality_grade} inconsistent with score {self.overall_quality_score}")
        
        return errors


# Export format specification functions

def get_parquet_export_config() -> Dict[str, Any]:
    """Get optimised Parquet export configuration."""
    return {
        "compression": "snappy",
        "row_group_size": 50000,
        "page_size": 1024 * 1024,  # 1MB
        "use_dictionary": True,
        "write_statistics": True,
        "data_page_version": "2.0"
    }


def get_csv_export_config() -> Dict[str, Any]:
    """Get standardised CSV export configuration."""
    return {
        "delimiter": ",",
        "quotechar": '"',
        "quoting": "minimal",
        "line_terminator": "\n",
        "encoding": "utf-8-sig",  # UTF-8 with BOM for Excel compatibility
        "include_index": False
    }


def get_geojson_export_config() -> Dict[str, Any]:
    """Get GeoJSON export configuration."""
    return {
        "coordinate_precision": 6,
        "ensure_ascii": False,
        "validate_geometry": True,
        "simplify_tolerance": None,  # No simplification by default
        "crs": "EPSG:4326"  # WGS84
    }


def validate_export_compatibility(
    source_schema: type,
    target_format: ExportFormat
) -> List[str]:
    """
    Validate compatibility between source schema and target export format.
    
    Args:
        source_schema: Source schema class
        target_format: Target export format
        
    Returns:
        List of compatibility issues
    """
    issues = []
    
    # Check for format-specific requirements
    if target_format == ExportFormat.GEOJSON:
        # GeoJSON requires geometry fields
        if not hasattr(source_schema, 'geometry') and not hasattr(source_schema, 'boundary_data'):
            issues.append("GeoJSON export requires geometry fields")
    
    elif target_format == ExportFormat.CSV:
        # CSV doesn't handle nested structures well
        schema_fields = getattr(source_schema, '__fields__', {})
        for field_name, field_info in schema_fields.items():
            if hasattr(field_info, 'type_') and str(field_info.type_).startswith('typing.Dict'):
                issues.append(f"CSV export may not handle nested field '{field_name}' properly")
    
    return issues