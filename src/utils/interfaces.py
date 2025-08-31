"""
AHGD V3: Core Interfaces and Data Models
Minimal interfaces for high-performance Polars extractors.
"""

from collections.abc import Callable
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Optional

from pydantic import BaseModel


class ProcessingStatus(str, Enum):
    """Status enumeration for data processing operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExtractionError(Exception):
    """Custom exception for data extraction operations."""

    pass


class ValidationError(Exception):
    """Custom exception for data validation operations."""

    pass


class SourceMetadata(BaseModel):
    """Metadata about a data source."""

    source_id: str
    source_name: str
    description: str
    url: Optional[str] = None
    update_frequency: Optional[str] = None
    coverage_area: Optional[str] = None
    data_format: Optional[str] = None
    last_updated: Optional[datetime] = None
    schema_version: Optional[str] = None
    quality_indicators: Optional[dict[str, float]] = None
    processing_notes: Optional[list[str]] = None


class ProcessingMetadata(BaseModel):
    """Metadata about data processing operations."""

    processing_id: str
    source_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: ProcessingStatus
    records_processed: int = 0
    errors_encountered: int = 0
    quality_score: Optional[float] = None
    processing_notes: Optional[list[str]] = None


class DataRecord(BaseModel):
    """Individual data record with metadata."""

    record_id: str
    source_id: str
    data: dict[str, Any]
    extracted_at: datetime
    quality_score: Optional[float] = None
    validation_errors: Optional[list[str]] = None


class DataBatch(BaseModel):
    """Collection of data records with batch metadata."""

    batch_id: str
    source_id: str
    records: list[DataRecord]
    batch_metadata: ProcessingMetadata
    total_records: int = 0

    def __post_init__(self):
        self.total_records = len(self.records)


class AuditTrail(BaseModel):
    """Audit trail for data processing operations."""

    operation_id: str
    timestamp: datetime
    operation_type: str
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    user_id: Optional[str] = None
    details: Optional[dict[str, Any]] = None
    status: ProcessingStatus
    error_message: Optional[str] = None


# Type aliases for callbacks and progress reporting
ProgressCallback = Callable[[int, int, str], None]  # (current, total, message)
