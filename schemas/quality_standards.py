"""
Data quality standards and compliance requirements for AHGD.

This module defines Australian health data standards compliance requirements,
data completeness standards, statistical validation thresholds, and
geographic validation requirements.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator

from .base_schema import VersionedSchema


class ComplianceStandard(str, Enum):
    """Australian health data compliance standards."""
    AIHW = "aihw"  # Australian Institute of Health and Welfare
    ABS = "abs"    # Australian Bureau of Statistics
    MEDICARE = "medicare"  # Medicare data standards
    PBS = "pbs"    # Pharmaceutical Benefits Scheme
    NHDD = "nhdd"  # National Health Data Dictionary
    METeOR = "meteor"  # Metadata Online Registry
    SNOMED_CT_AU = "snomed_ct_au"  # SNOMED CT Australian edition
    ICD_10_AM = "icd_10_am"  # ICD-10-AM Australian modification
    ACHI = "achi"  # Australian Classification of Health Interventions
    ARIA = "aria"  # Accessibility/Remoteness Index of Australia


class QualityLevel(str, Enum):
    """Data quality requirement levels."""
    CRITICAL = "critical"    # Must meet 100% compliance
    HIGH = "high"           # Must meet ≥95% compliance
    MEDIUM = "medium"       # Must meet ≥90% compliance
    LOW = "low"             # Must meet ≥80% compliance
    OPTIONAL = "optional"   # No minimum requirement


class ValidationSeverity(str, Enum):
    """Validation error severity levels."""
    FATAL = "fatal"         # Prevents data processing
    ERROR = "error"         # Significant quality issue
    WARNING = "warning"     # Minor quality concern
    INFO = "info"          # Informational message


class AustralianHealthDataStandard(VersionedSchema):
    """
    Australian health data standards compliance specification.
    
    Defines compliance requirements for Australian health data standards
    including AIHW, ABS, Medicare, and other regulatory frameworks.
    """
    
    # === STANDARD IDENTIFICATION ===
    standard_name: ComplianceStandard = Field(..., description="Name of compliance standard")
    standard_version: str = Field(..., description="Version of the standard")
    standard_url: Optional[str] = Field(None, description="URL to standard documentation")
    effective_date: datetime = Field(..., description="When standard becomes effective")
    
    # === SCOPE ===
    applicable_data_types: List[str] = Field(
        ...,
        description="Data types this standard applies to"
    )
    mandatory_fields: List[str] = Field(
        ...,
        description="Fields required by this standard"
    )
    optional_fields: List[str] = Field(
        default_factory=list,
        description="Fields recommended but not required"
    )
    
    # === FIELD SPECIFICATIONS ===
    field_requirements: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Detailed requirements for each field"
    )
    
    # === VALIDATION RULES ===
    format_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Format validation rules"
    )
    value_constraints: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Value range and constraint rules"
    )
    business_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Business logic validation rules"
    )
    
    # === QUALITY THRESHOLDS ===
    minimum_completeness: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Minimum data completeness percentage required"
    )
    accuracy_threshold: float = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Minimum accuracy percentage required"
    )
    
    # === REPORTING REQUIREMENTS ===
    reporting_frequency: str = Field(
        ...,
        description="Required reporting frequency"
    )
    audit_requirements: List[str] = Field(
        default_factory=list,
        description="Audit and compliance reporting requirements"
    )
    
    # === PENALTIES AND ENFORCEMENT ===
    non_compliance_penalties: List[str] = Field(
        default_factory=list,
        description="Penalties for non-compliance"
    )
    grace_period_days: Optional[int] = Field(
        None,
        ge=0,
        description="Grace period for compliance (days)"
    )
    
    @field_validator('reporting_frequency')
    @classmethod
    def validate_reporting_frequency(cls, v: str) -> str:
        """Validate reporting frequency."""
        valid_frequencies = {
            'real-time', 'daily', 'weekly', 'monthly', 
            'quarterly', 'annually', 'on-demand'
        }
        if v.lower() not in valid_frequencies:
            raise ValueError(f"Invalid reporting frequency: {v}")
        return v.lower()
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "AustralianHealthDataStandard"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate health data standard definition."""
        errors = []
        
        # Check field requirements consistency
        mandatory_set = set(self.mandatory_fields)
        optional_set = set(self.optional_fields)
        overlap = mandatory_set & optional_set
        if overlap:
            errors.append(f"Fields in both mandatory and optional lists: {overlap}")
        
        # Validate field requirements
        for field_name, requirements in self.field_requirements.items():
            if 'data_type' not in requirements:
                errors.append(f"Missing data_type for field {field_name}")
            if 'quality_level' not in requirements:
                errors.append(f"Missing quality_level for field {field_name}")
        
        return errors


class DataCompletenessRequirement(VersionedSchema):
    """
    Data completeness requirements by field and context.
    
    Defines specific completeness requirements for different
    data fields based on their importance and usage context.
    """
    
    # === REQUIREMENT IDENTIFICATION ===
    requirement_name: str = Field(..., description="Name of completeness requirement")
    data_domain: str = Field(..., description="Data domain (health, demographic, geographic)")
    context: str = Field(..., description="Usage context (analysis, reporting, operational)")
    
    # === FIELD REQUIREMENTS ===
    field_completeness_requirements: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Completeness requirements by field"
    )
    
    # === CONDITIONAL REQUIREMENTS ===
    conditional_requirements: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Completeness requirements based on conditions"
    )
    
    # === BUSINESS RULES ===
    business_critical_fields: List[str] = Field(
        ...,
        description="Fields critical for business operations"
    )
    analysis_required_fields: List[str] = Field(
        default_factory=list,
        description="Fields required for statistical analysis"
    )
    regulatory_required_fields: List[str] = Field(
        default_factory=list,
        description="Fields required for regulatory compliance"
    )
    
    # === QUALITY LEVELS ===
    critical_completeness_threshold: float = Field(
        default=100.0,
        ge=95.0,
        le=100.0,
        description="Completeness threshold for critical fields (%)"
    )
    high_completeness_threshold: float = Field(
        default=95.0,
        ge=90.0,
        le=100.0,
        description="Completeness threshold for high-priority fields (%)"
    )
    medium_completeness_threshold: float = Field(
        default=90.0,
        ge=80.0,
        le=100.0,
        description="Completeness threshold for medium-priority fields (%)"
    )
    low_completeness_threshold: float = Field(
        default=80.0,
        ge=50.0,
        le=100.0,
        description="Completeness threshold for low-priority fields (%)"
    )
    
    # === EXCEPTIONS ===
    allowed_exceptions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Allowed exceptions to completeness requirements"
    )
    seasonal_adjustments: Dict[str, float] = Field(
        default_factory=dict,
        description="Seasonal adjustments to completeness thresholds"
    )
    
    @field_validator('data_domain')
    @classmethod
    def validate_data_domain(cls, v: str) -> str:
        """Validate data domain."""
        valid_domains = {
            'health', 'demographic', 'geographic', 'socioeconomic',
            'environmental', 'administrative', 'clinical'
        }
        if v.lower() not in valid_domains:
            raise ValueError(f"Invalid data domain: {v}")
        return v.lower()
    
    def get_field_completeness_requirement(self, field_name: str) -> Optional[float]:
        """Get completeness requirement for a specific field."""
        if field_name in self.field_completeness_requirements:
            return self.field_completeness_requirements[field_name].get('threshold')
        
        # Determine based on field criticality
        if field_name in self.business_critical_fields:
            return self.critical_completeness_threshold
        elif field_name in self.regulatory_required_fields:
            return self.high_completeness_threshold
        elif field_name in self.analysis_required_fields:
            return self.medium_completeness_threshold
        else:
            return self.low_completeness_threshold
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "DataCompletenessRequirement"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate completeness requirement specification."""
        errors = []
        
        # Check threshold ordering
        thresholds = [
            self.critical_completeness_threshold,
            self.high_completeness_threshold,
            self.medium_completeness_threshold,
            self.low_completeness_threshold
        ]
        
        if thresholds != sorted(thresholds, reverse=True):
            errors.append("Completeness thresholds should be in descending order")
        
        # Validate field requirements
        for field_name, requirements in self.field_completeness_requirements.items():
            if 'threshold' not in requirements:
                errors.append(f"Missing threshold for field {field_name}")
            elif not (0 <= requirements['threshold'] <= 100):
                errors.append(f"Invalid threshold for field {field_name}")
        
        return errors


class StatisticalValidationThreshold(VersionedSchema):
    """
    Statistical validation thresholds and rules.
    
    Defines statistical validation requirements including
    outlier detection, distribution checks, and relationship validation.
    """
    
    # === THRESHOLD IDENTIFICATION ===
    threshold_name: str = Field(..., description="Name of statistical threshold")
    statistical_method: str = Field(..., description="Statistical method or test")
    applicable_fields: List[str] = Field(..., description="Fields this threshold applies to")
    
    # === OUTLIER DETECTION ===
    outlier_detection_method: str = Field(
        default="iqr",
        description="Outlier detection method (iqr, zscore, modified_zscore)"
    )
    outlier_threshold: float = Field(
        default=3.0,
        gt=0,
        description="Outlier detection threshold"
    )
    outlier_action: str = Field(
        default="flag",
        description="Action for outliers (flag, exclude, investigate)"
    )
    
    # === DISTRIBUTION VALIDATION ===
    expected_distribution: Optional[str] = Field(
        None,
        description="Expected statistical distribution (normal, poisson, etc.)"
    )
    distribution_test: Optional[str] = Field(
        None,
        description="Statistical test for distribution (shapiro, kolmogorov)"
    )
    distribution_p_value_threshold: float = Field(
        default=0.05,
        gt=0,
        le=1,
        description="P-value threshold for distribution tests"
    )
    
    # === RANGE VALIDATION ===
    minimum_value: Optional[float] = Field(None, description="Minimum allowed value")
    maximum_value: Optional[float] = Field(None, description="Maximum allowed value")
    expected_mean_range: Optional[Dict[str, float]] = Field(
        None,
        description="Expected range for mean values"
    )
    expected_std_range: Optional[Dict[str, float]] = Field(
        None,
        description="Expected range for standard deviation"
    )
    
    # === RELATIONSHIP VALIDATION ===
    correlation_checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Correlation validation rules"
    )
    ratio_checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ratio validation rules"
    )
    trend_checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Trend validation rules"
    )
    
    # === TEMPORAL VALIDATION ===
    temporal_consistency_checks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Temporal consistency validation rules"
    )
    seasonality_checks: bool = Field(
        default=False,
        description="Whether to perform seasonality validation"
    )
    
    # === SAMPLE SIZE REQUIREMENTS ===
    minimum_sample_size: Optional[int] = Field(
        None,
        ge=1,
        description="Minimum sample size for valid statistics"
    )
    confidence_level: float = Field(
        default=95.0,
        ge=90.0,
        le=99.9,
        description="Required confidence level (%)"
    )
    margin_of_error: float = Field(
        default=5.0,
        gt=0,
        le=50,
        description="Acceptable margin of error (%)"
    )
    
    # === QUALITY ASSESSMENT ===
    severity_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Severity thresholds for different validation failures"
    )
    
    @field_validator('outlier_detection_method')
    @classmethod
    def validate_outlier_method(cls, v: str) -> str:
        """Validate outlier detection method."""
        valid_methods = {'iqr', 'zscore', 'modified_zscore', 'isolation_forest', 'lof'}
        if v.lower() not in valid_methods:
            raise ValueError(f"Invalid outlier detection method: {v}")
        return v.lower()
    
    @field_validator('outlier_action')
    @classmethod
    def validate_outlier_action(cls, v: str) -> str:
        """Validate outlier action."""
        valid_actions = {'flag', 'exclude', 'investigate', 'transform', 'cap'}
        if v.lower() not in valid_actions:
            raise ValueError(f"Invalid outlier action: {v}")
        return v.lower()
    
    def is_outlier(self, value: float, dataset_stats: Dict[str, float]) -> bool:
        """
        Check if a value is an outlier based on configured method.
        
        Args:
            value: Value to check
            dataset_stats: Dataset statistics (mean, std, q1, q3, etc.)
            
        Returns:
            True if value is an outlier
        """
        if self.outlier_detection_method == 'iqr':
            q1 = dataset_stats.get('q1', 0)
            q3 = dataset_stats.get('q3', 0)
            iqr = q3 - q1
            lower_bound = q1 - (self.outlier_threshold * iqr)
            upper_bound = q3 + (self.outlier_threshold * iqr)
            return value < lower_bound or value > upper_bound
            
        elif self.outlier_detection_method == 'zscore':
            mean = dataset_stats.get('mean', 0)
            std = dataset_stats.get('std', 1)
            zscore = abs((value - mean) / std) if std > 0 else 0
            return zscore > self.outlier_threshold
            
        return False
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "StatisticalValidationThreshold"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate statistical threshold specification."""
        errors = []
        
        # Check range consistency
        if (self.minimum_value is not None and 
            self.maximum_value is not None and 
            self.minimum_value >= self.maximum_value):
            errors.append("Minimum value must be less than maximum value")
        
        # Validate correlation checks
        for corr_check in self.correlation_checks:
            if 'field1' not in corr_check or 'field2' not in corr_check:
                errors.append("Correlation check missing required fields")
            if 'expected_correlation' in corr_check:
                corr = corr_check['expected_correlation']
                if not (-1 <= corr <= 1):
                    errors.append("Correlation value must be between -1 and 1")
        
        return errors


class GeographicValidationRequirement(VersionedSchema):
    """
    Geographic validation requirements and spatial quality standards.
    
    Defines validation rules specific to geographic and spatial data
    including coordinate validation, spatial relationships, and topology.
    """
    
    # === REQUIREMENT IDENTIFICATION ===
    requirement_name: str = Field(..., description="Name of geographic validation requirement")
    geographic_scope: str = Field(..., description="Geographic scope (australia, state, sa2)")
    coordinate_system: str = Field(
        default="GDA2020",
        description="Required coordinate reference system"
    )
    
    # === COORDINATE VALIDATION ===
    coordinate_precision: int = Field(
        default=6,
        ge=1,
        le=15,
        description="Required decimal places for coordinates"
    )
    coordinate_bounds: Dict[str, Dict[str, float]] = Field(
        ...,
        description="Valid coordinate bounds by region"
    )
    
    # === SPATIAL ACCURACY ===
    positional_accuracy_metres: float = Field(
        default=100.0,
        gt=0,
        description="Required positional accuracy in metres"
    )
    scale_accuracy: Optional[str] = Field(
        None,
        description="Required scale accuracy (e.g., 1:10000)"
    )
    
    # === GEOMETRY VALIDATION ===
    geometry_validation_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Geometry validation rules"
    )
    topology_validation: bool = Field(
        default=True,
        description="Whether to validate topology"
    )
    self_intersection_tolerance: float = Field(
        default=0.001,
        ge=0,
        description="Tolerance for self-intersection detection"
    )
    
    # === BOUNDARY VALIDATION ===
    boundary_completeness_threshold: float = Field(
        default=100.0,
        ge=95.0,
        le=100.0,
        description="Required boundary completeness (%)"
    )
    administrative_boundary_compliance: bool = Field(
        default=True,
        description="Must comply with official administrative boundaries"
    )
    
    # === SPATIAL RELATIONSHIPS ===
    containment_validation: bool = Field(
        default=True,
        description="Validate spatial containment relationships"
    )
    adjacency_validation: bool = Field(
        default=True,
        description="Validate spatial adjacency relationships"
    )
    overlap_tolerance_metres: float = Field(
        default=1.0,
        ge=0,
        description="Tolerance for boundary overlaps (metres)"
    )
    gap_tolerance_metres: float = Field(
        default=1.0,
        ge=0,
        description="Tolerance for boundary gaps (metres)"
    )
    
    # === AREA AND DISTANCE VALIDATION ===
    minimum_area_square_metres: Optional[float] = Field(
        None,
        gt=0,
        description="Minimum valid area in square metres"
    )
    maximum_area_square_metres: Optional[float] = Field(
        None,
        gt=0,
        description="Maximum valid area in square metres"
    )
    area_calculation_method: str = Field(
        default="spherical",
        description="Method for area calculation (spherical, planar)"
    )
    
    # === COORDINATE SYSTEM VALIDATION ===
    required_projections: List[str] = Field(
        default_factory=list,
        description="Required map projections to support"
    )
    transformation_accuracy: float = Field(
        default=1.0,
        gt=0,
        description="Required transformation accuracy (metres)"
    )
    
    # === DATA QUALITY ===
    geometric_quality_indicators: List[str] = Field(
        default_factory=list,
        description="Required geometric quality indicators"
    )
    completeness_assessment_method: str = Field(
        default="boundary_coverage",
        description="Method for assessing spatial completeness"
    )
    
    # === TEMPORAL VALIDATION ===
    temporal_consistency_validation: bool = Field(
        default=True,
        description="Validate temporal consistency of boundaries"
    )
    change_detection_threshold: float = Field(
        default=5.0,
        gt=0,
        description="Threshold for detecting significant boundary changes (%)"
    )
    
    @field_validator('coordinate_system')
    @classmethod
    def validate_coordinate_system(cls, v: str) -> str:
        """Validate coordinate reference system."""
        valid_systems = {
            'gda2020', 'gda94', 'wgs84', 'mga94', 'mga2020',
            'epsg:4326', 'epsg:3857', 'epsg:7844'
        }
        if v.lower() not in valid_systems:
            raise ValueError(f"Invalid coordinate system: {v}")
        return v.upper()
    
    @field_validator('area_calculation_method')
    @classmethod
    def validate_area_method(cls, v: str) -> str:
        """Validate area calculation method."""
        valid_methods = {'spherical', 'planar', 'ellipsoidal'}
        if v.lower() not in valid_methods:
            raise ValueError(f"Invalid area calculation method: {v}")
        return v.lower()
    
    def validate_coordinates(self, latitude: float, longitude: float, region: str = 'australia') -> List[str]:
        """
        Validate coordinate values against bounds.
        
        Args:
            latitude: Latitude value
            longitude: Longitude value
            region: Region to validate against
            
        Returns:
            List of validation errors
        """
        errors = []
        
        if region in self.coordinate_bounds:
            bounds = self.coordinate_bounds[region]
            
            if 'min_lat' in bounds and latitude < bounds['min_lat']:
                errors.append(f"Latitude {latitude} below minimum {bounds['min_lat']}")
            if 'max_lat' in bounds and latitude > bounds['max_lat']:
                errors.append(f"Latitude {latitude} above maximum {bounds['max_lat']}")
            if 'min_lon' in bounds and longitude < bounds['min_lon']:
                errors.append(f"Longitude {longitude} below minimum {bounds['min_lon']}")
            if 'max_lon' in bounds and longitude > bounds['max_lon']:
                errors.append(f"Longitude {longitude} above maximum {bounds['max_lon']}")
        
        return errors
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "GeographicValidationRequirement"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate geographic requirement specification."""
        errors = []
        
        # Check area bounds consistency
        if (self.minimum_area_square_metres is not None and 
            self.maximum_area_square_metres is not None and 
            self.minimum_area_square_metres >= self.maximum_area_square_metres):
            errors.append("Minimum area must be less than maximum area")
        
        # Validate coordinate bounds
        for region, bounds in self.coordinate_bounds.items():
            if 'min_lat' in bounds and 'max_lat' in bounds:
                if bounds['min_lat'] >= bounds['max_lat']:
                    errors.append(f"Invalid latitude bounds for region {region}")
            if 'min_lon' in bounds and 'max_lon' in bounds:
                if bounds['min_lon'] >= bounds['max_lon']:
                    errors.append(f"Invalid longitude bounds for region {region}")
        
        return errors


class QualityStandardsRegistry(VersionedSchema):
    """
    Registry of all quality standards and requirements.
    
    Central registry managing all quality standards, thresholds,
    and validation requirements for the AHGD project.
    """
    
    # === REGISTRY IDENTIFICATION ===
    registry_version: str = Field(..., description="Version of quality standards registry")
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="When registry was last updated"
    )
    
    # === STANDARDS INVENTORY ===
    compliance_standards: List[str] = Field(
        ...,
        description="IDs of registered compliance standards"
    )
    completeness_requirements: List[str] = Field(
        ...,
        description="IDs of registered completeness requirements"
    )
    statistical_thresholds: List[str] = Field(
        ...,
        description="IDs of registered statistical thresholds"
    )
    geographic_requirements: List[str] = Field(
        ...,
        description="IDs of registered geographic requirements"
    )
    
    # === GLOBAL SETTINGS ===
    default_quality_level: QualityLevel = Field(
        default=QualityLevel.HIGH,
        description="Default quality level when not specified"
    )
    validation_severity_mapping: Dict[str, ValidationSeverity] = Field(
        default_factory=dict,
        description="Mapping of validation types to severity levels"
    )
    
    # === REPORTING ===
    quality_reporting_schedule: str = Field(
        default="monthly",
        description="Quality reporting schedule"
    )
    stakeholder_notifications: List[str] = Field(
        default_factory=list,
        description="Stakeholders to notify of quality issues"
    )
    
    # === EXEMPTIONS ===
    registered_exemptions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Registered exemptions from quality standards"
    )
    temporary_waivers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Temporary waivers from specific requirements"
    )
    
    def get_applicable_standards(self, data_type: str, context: str) -> List[str]:
        """
        Get applicable quality standards for specific data type and context.
        
        Args:
            data_type: Type of data (health, demographic, etc.)
            context: Usage context (analysis, reporting, etc.)
            
        Returns:
            List of applicable standard IDs
        """
        # Implementation would query registered standards
        # This is a placeholder for the actual logic
        return self.compliance_standards
    
    def get_schema_name(self) -> str:
        """Return the schema name."""
        return "QualityStandardsRegistry"
    
    def validate_data_integrity(self) -> List[str]:
        """Validate quality standards registry."""
        errors = []
        
        # Check for duplicate registrations
        all_standards = (
            self.compliance_standards + 
            self.completeness_requirements + 
            self.statistical_thresholds + 
            self.geographic_requirements
        )
        
        if len(all_standards) != len(set(all_standards)):
            errors.append("Duplicate standard IDs found in registry")
        
        # Validate exemptions
        for exemption in self.registered_exemptions:
            if 'standard_id' not in exemption or 'reason' not in exemption:
                errors.append("Exemption missing required fields")
        
        return errors


# Utility functions for quality standards

def get_default_australian_health_standards() -> List[Dict[str, Any]]:
    """Get default Australian health data standards configuration."""
    return [
        {
            "standard_name": "aihw",
            "standard_version": "2023.1",
            "minimum_completeness": 95.0,
            "accuracy_threshold": 98.0,
            "mandatory_fields": [
                "person_id", "date_of_birth", "sex", "indigenous_status",
                "postcode", "date_of_service"
            ]
        },
        {
            "standard_name": "abs",
            "standard_version": "2021",
            "minimum_completeness": 100.0,
            "accuracy_threshold": 99.5,
            "mandatory_fields": [
                "sa2_code", "sa3_code", "sa4_code", "state_code"
            ]
        },
        {
            "standard_name": "medicare",
            "standard_version": "2023",
            "minimum_completeness": 99.0,
            "accuracy_threshold": 99.9,
            "mandatory_fields": [
                "medicare_number", "service_date", "item_number",
                "provider_number", "benefit_paid"
            ]
        }
    ]


def get_default_coordinate_bounds() -> Dict[str, Dict[str, float]]:
    """Get default coordinate bounds for Australian regions."""
    return {
        "australia": {
            "min_lat": -44.0,
            "max_lat": -10.0,
            "min_lon": 112.0,
            "max_lon": 154.0
        },
        "tasmania": {
            "min_lat": -43.7,
            "max_lat": -39.2,
            "min_lon": 143.8,
            "max_lon": 148.5
        },
        "mainland": {
            "min_lat": -39.2,
            "max_lat": -10.0,
            "min_lon": 112.0,
            "max_lon": 154.0
        }
    }


def validate_against_standards(
    data: Dict[str, Any],
    standards: List[AustralianHealthDataStandard]
) -> Dict[str, List[str]]:
    """
    Validate data against multiple quality standards.
    
    Args:
        data: Data to validate
        standards: List of standards to validate against
        
    Returns:
        Dictionary mapping standard names to validation errors
    """
    validation_results = {}
    
    for standard in standards:
        errors = []
        
        # Check mandatory fields
        for field in standard.mandatory_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing mandatory field: {field}")
        
        # Additional validation logic would go here
        
        validation_results[standard.standard_name.value] = errors
    
    return validation_results