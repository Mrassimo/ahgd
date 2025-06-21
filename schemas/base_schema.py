"""
Base Pydantic models for AHGD data structures with schema versioning support.

This module provides the foundation for all data schemas in the AHGD project,
including version management, field validation, and migration capabilities.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from pydantic.version import VERSION as PYDANTIC_VERSION
from pydantic_settings import BaseSettings
import uuid
import semver
import threading
from typing import ClassVar


# Type variable for schema classes
T = TypeVar('T', bound='VersionedSchema')


class SchemaVersion(str, Enum):
    """Enumeration of schema versions."""
    V1_0_0 = "1.0.0"
    V1_1_0 = "1.1.0"
    V1_2_0 = "1.2.0"
    V2_0_0 = "2.0.0"
    
    @classmethod
    def from_string(cls, version_str: str) -> 'SchemaVersion':
        """Convert string to SchemaVersion enum."""
        for version in cls:
            if version.value == version_str:
                return version
        raise ValueError(f"Unknown schema version: {version_str}")
    
    def is_compatible_with(self, other: 'SchemaVersion') -> bool:
        """Check if this version is compatible with another version."""
        self_ver = semver.VersionInfo.parse(self.value)
        other_ver = semver.VersionInfo.parse(other.value)
        
        # Major version must match for compatibility
        return self_ver.major == other_ver.major


class DataQualityLevel(str, Enum):
    """Data quality level indicators."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class VersionedSchema(BaseModel, ABC):
    """
    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    """
    
    # Metadata fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    schema_version: SchemaVersion = Field(
        default=SchemaVersion.V1_0_0,
        description="Schema version for this record"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when record was created"
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when record was last updated"
    )
    data_quality: DataQualityLevel = Field(
        default=DataQualityLevel.UNKNOWN,
        description="Data quality assessment level"
    )
    
    model_config = {
        "use_enum_values": True,
        "validate_assignment": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat() if v else None,
            uuid.UUID: lambda v: str(v)
        },
        "extra": "forbid",
        "str_strip_whitespace": True
    }
        
    @abstractmethod
    def get_schema_name(self) -> str:
        """Return the name of this schema type."""
        pass
    
    @abstractmethod
    def validate_data_integrity(self) -> List[str]:
        """
        Validate data integrity and return list of validation errors.
        
        Returns:
            List of validation error messages, empty if valid
        """
        pass
    
    def can_migrate_to(self, target_version: SchemaVersion) -> bool:
        """Check if migration to target version is possible."""
        return self.schema_version.is_compatible_with(target_version)
    
    @classmethod
    def from_previous_version(cls: Type[T], old_data: Dict[str, Any], 
                            old_version: SchemaVersion) -> T:
        """
        Create instance from previous version data.
        
        This method should be overridden by subclasses to handle specific migrations.
        """
        # Default implementation - direct conversion
        return cls(**old_data)
    
    def to_dict_versioned(self) -> Dict[str, Any]:
        """Export to dictionary with version information."""
        data = self.dict()
        data['_schema_name'] = self.get_schema_name()
        data['_export_timestamp'] = datetime.utcnow().isoformat()
        return data
    
    @model_validator(mode='before')
    def set_updated_at(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically set updated_at timestamp."""
        if isinstance(values, dict):
            if values.get('updated_at') is None and 'id' in values:
                # This is an update operation
                values['updated_at'] = datetime.utcnow()
        return values


class GeographicBoundary(BaseModel):
    """Base model for geographic boundary data."""
    
    boundary_id: str = Field(..., description="Unique boundary identifier")
    boundary_type: str = Field(..., description="Type of boundary (SA2, SA3, etc)")
    name: str = Field(..., description="Human-readable boundary name")
    state: str = Field(..., description="State or territory code")
    
    # Geometric properties
    area_sq_km: Optional[float] = Field(None, ge=0, description="Area in square kilometers")
    perimeter_km: Optional[float] = Field(None, ge=0, description="Perimeter in kilometers")
    
    # Centroid coordinates
    centroid_lat: Optional[float] = Field(None, ge=-90, le=90, description="Centroid latitude")
    centroid_lon: Optional[float] = Field(None, ge=-180, le=180, description="Centroid longitude")
    
    # GeoJSON geometry (stored as dict for flexibility)
    geometry: Optional[Dict[str, Any]] = Field(None, description="GeoJSON geometry object")
    
    @field_validator('state')
    @classmethod
    def validate_state_code(cls, v: str) -> str:
        """Validate Australian state/territory codes."""
        valid_states = {'NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'}
        if v.upper() not in valid_states:
            raise ValueError(f"Invalid state code: {v}. Must be one of {valid_states}")
        return v.upper()
    
    @field_validator('geometry')
    @classmethod
    def validate_geometry(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Basic validation of GeoJSON geometry."""
        if v is None:
            return v
            
        if 'type' not in v:
            raise ValueError("Geometry must have a 'type' field")
            
        valid_types = {'Point', 'MultiPoint', 'LineString', 'MultiLineString', 
                      'Polygon', 'MultiPolygon', 'GeometryCollection'}
        if v['type'] not in valid_types:
            raise ValueError(f"Invalid geometry type: {v['type']}")
            
        if 'coordinates' not in v and v['type'] != 'GeometryCollection':
            raise ValueError("Geometry must have 'coordinates' field")
            
        return v


class TemporalData(BaseModel):
    """Base model for time-series data."""
    
    reference_date: datetime = Field(..., description="Reference date for the data")
    period_type: str = Field(..., description="Type of time period (annual, quarterly, etc)")
    period_start: datetime = Field(..., description="Start of the data period")
    period_end: datetime = Field(..., description="End of the data period")
    
    @model_validator(mode='after')
    def validate_period_order(self) -> 'TemporalData':
        """Ensure period_end is after period_start."""
        if self.period_end < self.period_start:
            raise ValueError("period_end must be after period_start")
        return self


class DataSource(BaseModel):
    """Information about data source and provenance."""
    
    source_name: str = Field(..., description="Name of data source")
    source_url: Optional[str] = Field(None, description="URL of data source")
    source_date: datetime = Field(..., description="Date when data was sourced")
    source_version: Optional[str] = Field(None, description="Version of source data")
    attribution: str = Field(..., description="Required attribution text")
    license: Optional[str] = Field(None, description="Data license information")
    
    @field_validator('source_url')
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        """Basic URL validation."""
        if v and not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError("URL must start with http:// or https://")
        return v


class MigrationRecord(BaseModel):
    """Record of schema migration operations."""
    
    migration_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_version: SchemaVersion = Field(..., description="Source schema version")
    to_version: SchemaVersion = Field(..., description="Target schema version")
    migration_date: datetime = Field(default_factory=datetime.utcnow)
    record_count: int = Field(..., ge=0, description="Number of records migrated")
    success: bool = Field(..., description="Whether migration was successful")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    duration_seconds: Optional[float] = Field(None, ge=0, description="Migration duration")
    
    model_config = {
        "use_enum_values": True,
        "validate_assignment": True
    }


# Utility functions for schema operations

def validate_schema_compatibility(schema_class: Type[VersionedSchema], 
                                data: Dict[str, Any]) -> bool:
    """
    Check if data is compatible with a schema class.
    
    Args:
        schema_class: The schema class to validate against
        data: Data dictionary to validate
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        schema_class(**data)
        return True
    except ValidationError:
        return False


def get_schema_version(data: Dict[str, Any]) -> Optional[SchemaVersion]:
    """Extract schema version from data dictionary."""
    version_str = data.get('schema_version')
    if version_str:
        try:
            return SchemaVersion.from_string(version_str)
        except ValueError:
            return None
    return None


def upgrade_schema(data: Dict[str, Any], 
                  schema_class: Type[VersionedSchema],
                  target_version: SchemaVersion) -> Dict[str, Any]:
    """
    Upgrade data to target schema version.
    
    Args:
        data: Input data dictionary
        schema_class: Target schema class
        target_version: Desired schema version
        
    Returns:
        Upgraded data dictionary
        
    Raises:
        ValueError: If upgrade is not possible
    """
    current_version = get_schema_version(data)
    if not current_version:
        raise ValueError("Cannot determine current schema version")
        
    if current_version == target_version:
        return data
        
    # Create instance using migration method
    instance = schema_class.from_previous_version(data, current_version)
    instance.schema_version = target_version
    
    return instance.model_dump()


# Enhanced versioning utilities with thread safety

class SchemaVersionManager:
    """Thread-safe schema version management utilities."""
    
    _lock = threading.Lock()
    _version_cache: ClassVar[Dict[str, SchemaVersion]] = {}
    
    @classmethod
    def get_latest_version(cls, schema_name: str) -> SchemaVersion:
        """Get the latest version for a schema type."""
        with cls._lock:
            if schema_name not in cls._version_cache:
                # Default to latest available version
                cls._version_cache[schema_name] = SchemaVersion.V2_0_0
            return cls._version_cache[schema_name]
    
    @classmethod
    def register_schema_version(cls, schema_name: str, version: SchemaVersion) -> None:
        """Register a schema version."""
        with cls._lock:
            cls._version_cache[schema_name] = version
    
    @classmethod
    def is_version_supported(cls, schema_name: str, version: SchemaVersion) -> bool:
        """Check if a schema version is supported."""
        # All current versions are supported
        return version in SchemaVersion
    
    @classmethod
    def get_migration_path(cls, from_version: SchemaVersion, 
                          to_version: SchemaVersion) -> List[SchemaVersion]:
        """Get the migration path between two versions."""
        all_versions = list(SchemaVersion)
        
        try:
            from_idx = all_versions.index(from_version)
            to_idx = all_versions.index(to_version)
        except ValueError:
            raise ValueError("Invalid schema version")
            
        if from_idx == to_idx:
            return []
        elif from_idx < to_idx:
            # Forward migration
            return all_versions[from_idx + 1:to_idx + 1]
        else:
            # Backward migration (not typically supported)
            raise ValueError("Backward migration not supported")


class ValidationMetrics:
    """Collect and track validation performance metrics."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.total_validations = 0
            self.successful_validations = 0
            self.failed_validations = 0
            self.validation_times = []
            self.error_types = {}
    
    def record_validation(self, success: bool, duration: float, 
                         error_type: Optional[str] = None) -> None:
        """Record a validation result."""
        with self._lock:
            self.total_validations += 1
            self.validation_times.append(duration)
            
            if success:
                self.successful_validations += 1
            else:
                self.failed_validations += 1
                if error_type:
                    self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        with self._lock:
            avg_time = sum(self.validation_times) / len(self.validation_times) if self.validation_times else 0
            success_rate = (self.successful_validations / self.total_validations * 100) if self.total_validations > 0 else 0
            
            return {
                'total_validations': self.total_validations,
                'success_rate': success_rate,
                'average_validation_time': avg_time,
                'most_common_errors': sorted(self.error_types.items(), 
                                           key=lambda x: x[1], reverse=True)[:5]
            }


class CompatibilityChecker:
    """Check compatibility between schema versions."""
    
    @staticmethod
    def check_backward_compatibility(old_schema: Type[VersionedSchema],
                                   new_schema: Type[VersionedSchema]) -> List[str]:
        """
        Check backward compatibility between two schema versions.
        
        Returns list of compatibility issues.
        """
        issues = []
        
        old_fields = set(old_schema.__fields__.keys())
        new_fields = set(new_schema.__fields__.keys())
        
        # Check for removed required fields
        removed_fields = old_fields - new_fields
        for field in removed_fields:
            if old_schema.__fields__[field].required:
                issues.append(f"Required field '{field}' was removed")
        
        # Check for type changes in existing fields
        common_fields = old_fields & new_fields
        for field in common_fields:
            old_type = old_schema.__fields__[field].type_
            new_type = new_schema.__fields__[field].type_
            if old_type != new_type:
                issues.append(f"Field '{field}' type changed from {old_type} to {new_type}")
        
        return issues
    
    @staticmethod
    def check_forward_compatibility(old_version: SchemaVersion,
                                  new_version: SchemaVersion) -> bool:
        """Check if old version can be migrated to new version."""
        # Check major version compatibility
        old_ver = semver.VersionInfo.parse(old_version.value)
        new_ver = semver.VersionInfo.parse(new_version.value)
        
        # Major version changes may break compatibility
        if new_ver.major > old_ver.major:
            return False
        
        return True


# Global schema registry for easy access
_global_schema_registry: Optional['SchemaRegistry'] = None


def get_global_schema_registry() -> 'SchemaRegistry':
    """Get or create the global schema registry."""
    global _global_schema_registry
    if _global_schema_registry is None:
        from .schema_manager import SchemaRegistry  # Avoid circular import
        _global_schema_registry = SchemaRegistry()
    return _global_schema_registry