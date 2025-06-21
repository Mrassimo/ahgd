"""
Common interfaces, protocols, and data structures for the AHGD ETL pipeline.

This module provides standardised interfaces and data structures that are used
across all components of the ETL pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union
import hashlib


class ProcessingStatus(Enum):
    """Status of a processing operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class DataFormat(Enum):
    """Supported data formats."""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    XLSX = "xlsx"
    GEOJSON = "geojson"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


@dataclass
class SourceMetadata:
    """Metadata about a data source."""
    source_id: str
    source_type: str
    source_url: Optional[str] = None
    source_file: Optional[Path] = None
    last_modified: Optional[datetime] = None
    file_size: Optional[int] = None
    checksum: Optional[str] = None
    encoding: Optional[str] = None
    delimiter: Optional[str] = None
    headers: Optional[List[str]] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    schema_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


@dataclass
class ProcessingMetadata:
    """Metadata about a processing operation."""
    operation_id: str
    operation_type: str
    status: ProcessingStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    records_processed: int = 0
    records_failed: int = 0
    error_message: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def mark_completed(self) -> None:
        """Mark the operation as completed."""
        self.end_time = datetime.now()
        self.status = ProcessingStatus.COMPLETED
        if self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def mark_failed(self, error_message: str) -> None:
        """Mark the operation as failed."""
        self.end_time = datetime.now()
        self.status = ProcessingStatus.FAILED
        self.error_message = error_message
        if self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    severity: ValidationSeverity
    rule_id: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_records: List[int] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ColumnMapping:
    """Mapping configuration for column transformations."""
    source_column: str
    target_column: str
    data_type: str
    transformation: Optional[str] = None
    validation_rules: List[str] = field(default_factory=list)
    is_required: bool = True
    default_value: Optional[Any] = None


@dataclass
class DataPartition:
    """Information about a data partition."""
    partition_key: str
    partition_value: str
    file_path: Path
    record_count: int
    file_size: int
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AuditTrail:
    """Audit trail for data lineage."""
    operation_id: str
    operation_type: str
    source_metadata: SourceMetadata
    processing_metadata: ProcessingMetadata
    validation_results: List[ValidationResult] = field(default_factory=list)
    output_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)


class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""
    
    def __call__(self, current: int, total: int, message: str = "") -> None:
        """Called to report progress."""
        ...


class DataProcessor(Protocol):
    """Protocol for data processing components."""
    
    def process(self, data: Any) -> Any:
        """Process data and return the result."""
        ...
    
    def validate(self, data: Any) -> List[ValidationResult]:
        """Validate data and return validation results."""
        ...


# Exception classes
class AHGDException(Exception):
    """Base exception for AHGD ETL pipeline."""
    pass


class ExtractionError(AHGDException):
    """Exception raised during data extraction."""
    pass


class TransformationError(AHGDException):
    """Exception raised during data transformation."""
    pass


class ValidationError(AHGDException):
    """Exception raised during data validation."""
    pass


class LoadingError(AHGDException):
    """Exception raised during data loading."""
    pass


class ConfigurationError(AHGDException):
    """Exception raised for configuration issues."""
    pass


class DataQualityError(AHGDException):
    """Exception raised for data quality issues."""
    pass


class GeographicValidationError(ValidationError):
    """Exception raised for geographic validation issues."""
    pass


# Type aliases for better readability
DataRecord = Dict[str, Any]
DataBatch = List[DataRecord]
ConfigDict = Dict[str, Any]
MetadataDict = Dict[str, Any]