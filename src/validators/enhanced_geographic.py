"""
Enhanced Geographic Validation for Australian SA2 Data

This module provides comprehensive enhanced geographic validation specifically
designed for Australian Statistical Area Level 2 (SA2) data. It extends the
base geographic validator with advanced spatial operations including boundary
topology validation, coordinate reference system validation, spatial hierarchy
validation, and geographic consistency checks.
"""

import logging
import re
import math
import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Geographic libraries for spatial operations
try:
    import geopandas as gpd
    import shapely
    from shapely.geometry import Point, Polygon, MultiPolygon
    from shapely.validation import explain_validity
    from shapely.ops import unary_union
    import pyproj
    from pyproj import CRS, Transformer
    SPATIAL_LIBRARIES_AVAILABLE = True
except ImportError:
    SPATIAL_LIBRARIES_AVAILABLE = False

from ..utils.interfaces import (
    DataBatch,
    DataRecord,
    ValidationResult,
    ValidationSeverity,
    GeographicValidationError
)
from .geographic_validator import GeographicValidator


@dataclass
class SA2CoverageResult:
    """Result of SA2 coverage validation."""
    total_sa2_codes: int
    missing_sa2_codes: List[str] = field(default_factory=list)
    invalid_format_codes: List[str] = field(default_factory=list)
    duplicate_codes: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0


@dataclass
class BoundaryTopologyResult:
    """Result of boundary topology validation."""
    geometry_id: str
    is_valid_topology: bool
    has_gaps: bool = False
    has_overlaps: bool = False
    is_closed_polygon: bool = True
    has_self_intersections: bool = False
    topology_errors: List[str] = field(default_factory=list)


@dataclass
class CRSValidationResult:
    """Result of coordinate reference system validation."""
    coordinate_system: str
    is_valid_crs: bool
    coordinates_in_bounds: bool
    precision_adequate: bool
    transformation_accuracy: float
    crs_errors: List[str] = field(default_factory=list)


@dataclass
class SpatialHierarchyResult:
    """Result of spatial hierarchy validation."""
    sa2_code: str
    sa3_code: Optional[str] = None
    sa4_code: Optional[str] = None
    state_code: Optional[str] = None
    hierarchy_valid: bool = True
    containment_valid: bool = True
    hierarchy_errors: List[str] = field(default_factory=list)


@dataclass
class GeographicConsistencyResult:
    """Result of geographic consistency validation."""
    area_calculation_valid: bool = True
    population_density_valid: bool = True
    centroid_valid: bool = True
    coastal_classification_valid: bool = True
    consistency_errors: List[str] = field(default_factory=list)


class EnhancedGeographicValidator(GeographicValidator):
    """
    Enhanced Geographic Validator for Australian SA2 Data.
    
    This validator provides comprehensive spatial validation for Australian
    Statistical Area Level 2 (SA2) data, including advanced topology validation,
    coordinate reference system validation, spatial hierarchy validation,
    and geographic consistency checks.
    """
    
    # Official SA2 count for Australia (as of 2021 ASGS)
    OFFICIAL_SA2_COUNT = 2473
    
    # Australian CRS definitions
    AUSTRALIAN_CRS = {
        'GDA2020_MGA55': 7855,  # Primary CRS for most of Australia
        'GDA2020_MGA54': 7854,  # Western regions
        'GDA2020_MGA56': 7856,  # Eastern regions
        'GDA2020': 7844,        # Geographic CRS
        'GDA94': 4283           # Legacy CRS
    }
    
    def __init__(
        self,
        validator_id: str = "enhanced_geographic_validator",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the enhanced geographic validator.
        
        Args:
            validator_id: Unique identifier for this validator
            config: Configuration dictionary containing enhanced geographic rules
            logger: Optional logger instance
        """
        super().__init__(validator_id, config, logger)
        
        if not SPATIAL_LIBRARIES_AVAILABLE:
            self.logger.warning(
                "Spatial libraries (geopandas, shapely, pyproj) not available. "
                "Enhanced geographic validation will be limited."
            )
        
        # Enhanced validation configuration
        self.enhanced_config = self.config.get('enhanced_geographic', {})
        
        # SA2 coverage configuration
        self.sa2_coverage_config = self.enhanced_config.get('sa2_coverage', {})
        self.official_sa2_codes_file = self.sa2_coverage_config.get('official_sa2_codes_file')
        self.official_sa2_codes = self._load_official_sa2_codes()
        
        # Boundary topology configuration
        self.topology_config = self.enhanced_config.get('boundary_topology', {})
        self.topology_tolerance = self.topology_config.get('tolerance_metres', 100)
        
        # CRS validation configuration
        self.crs_config = self.enhanced_config.get('crs_validation', {})
        self.target_crs = self.crs_config.get('target_crs', 7855)  # GDA2020 MGA Zone 55
        self.coordinate_precision = self.crs_config.get('coordinate_precision', 6)
        
        # Spatial hierarchy configuration
        self.hierarchy_config = self.enhanced_config.get('spatial_hierarchy', {})
        
        # Geographic consistency configuration
        self.consistency_config = self.enhanced_config.get('geographic_consistency', {})
        
        # Performance tracking
        self.enhanced_statistics = {
            'sa2_coverage_checks': 0,
            'boundary_topology_checks': 0,
            'crs_validation_checks': 0,
            'spatial_hierarchy_checks': 0,
            'geographic_consistency_checks': 0,
            'spatial_operations_time': 0.0
        }
    
    def validate(self, data: DataBatch) -> List[ValidationResult]:
        """
        Perform enhanced geographic validation.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Enhanced geographic validation results
        """
        if not data:
            return [ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="enhanced_geographic_empty_data",
                message="Cannot perform enhanced geographic validation on empty dataset"
            )]
        
        results = []
        start_time = datetime.now()
        
        try:
            # Perform base geographic validation first
            base_results = super().validate(data)
            results.extend(base_results)
            
            # Enhanced SA2 Coverage Validation
            sa2_coverage_results = self.validate_sa2_coverage(data)
            results.extend(sa2_coverage_results)
            
            # Boundary Topology Validation
            topology_results = self.validate_boundary_topology(data)
            results.extend(topology_results)
            
            # Coordinate Reference System Validation
            crs_results = self.validate_coordinate_reference_system(data)
            results.extend(crs_results)
            
            # Spatial Hierarchy Validation
            hierarchy_results = self.validate_spatial_hierarchy(data)
            results.extend(hierarchy_results)
            
            # Geographic Consistency Checks
            consistency_results = self.validate_geographic_consistency(data)
            results.extend(consistency_results)
            
            # Update performance statistics
            duration = (datetime.now() - start_time).total_seconds()
            self.enhanced_statistics['spatial_operations_time'] += duration
            
            self.logger.info(
                f"Enhanced geographic validation completed in {duration:.2f}s: "
                f"{len(results)} issues found across {len(data)} records"
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced geographic validation failed: {e}")
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="enhanced_geographic_validation_error",
                message=f"Enhanced geographic validation failed: {str(e)}"
            ))
        
        return results
    
    def validate_sa2_coverage(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate complete coverage of all 2,473 official SA2 areas in Australia.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: SA2 coverage validation results
        """
        results = []
        self.enhanced_statistics['sa2_coverage_checks'] += 1
        
        try:
            # Extract SA2 codes from data
            sa2_codes_in_data = set()
            invalid_format_codes = []
            
            for record_idx, record in enumerate(data):
                sa2_code = record.get('sa2_code')
                if sa2_code:
                    sa2_str = str(sa2_code).strip()
                    
                    # Validate SA2 code format (11-digit format: SSSAASSSSS)
                    if self._validate_sa2_code_format_enhanced(sa2_str):
                        sa2_codes_in_data.add(sa2_str)
                    else:
                        invalid_format_codes.append(sa2_str)
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            rule_id="sa2_invalid_format_enhanced",
                            message=f"Invalid SA2 code format: {sa2_str} (expected 11-digit format: SSSAASSSSS)",
                            details={
                                'sa2_code': sa2_str,
                                'expected_format': 'SSSAASSSSS (11 digits)',
                                'pattern': r'^[1-8]\d{2}[0-9A-Z]{2}\d{5}$'
                            },
                            affected_records=[record_idx]
                        ))
            
            # Check for missing SA2 codes
            if self.official_sa2_codes:
                missing_sa2_codes = self.official_sa2_codes - sa2_codes_in_data
                coverage_percentage = (len(sa2_codes_in_data) / len(self.official_sa2_codes)) * 100
                
                if missing_sa2_codes:
                    severity = ValidationSeverity.ERROR if len(missing_sa2_codes) > 50 else ValidationSeverity.WARNING
                    results.append(ValidationResult(
                        is_valid=len(missing_sa2_codes) == 0,
                        severity=severity,
                        rule_id="sa2_coverage_incomplete",
                        message=f"Missing {len(missing_sa2_codes)} SA2 codes from dataset. Coverage: {coverage_percentage:.1f}%",
                        details={
                            'total_official_sa2_codes': len(self.official_sa2_codes),
                            'sa2_codes_in_data': len(sa2_codes_in_data),
                            'missing_sa2_count': len(missing_sa2_codes),
                            'missing_sa2_codes': sorted(list(missing_sa2_codes))[:10],  # First 10 for brevity
                            'coverage_percentage': coverage_percentage
                        }
                    ))
                else:
                    results.append(ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.INFO,
                        rule_id="sa2_coverage_complete",
                        message=f"Complete SA2 coverage achieved: {len(sa2_codes_in_data)} codes (100.0%)",
                        details={
                            'sa2_codes_found': len(sa2_codes_in_data),
                            'coverage_percentage': 100.0
                        }
                    ))
            
            # Check for duplicate SA2 codes
            sa2_code_counts = {}
            for record_idx, record in enumerate(data):
                sa2_code = record.get('sa2_code')
                if sa2_code:
                    sa2_str = str(sa2_code).strip()
                    if sa2_str in sa2_code_counts:
                        sa2_code_counts[sa2_str].append(record_idx)
                    else:
                        sa2_code_counts[sa2_str] = [record_idx]
            
            for sa2_code, record_indices in sa2_code_counts.items():
                if len(record_indices) > 1:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id="sa2_duplicate_codes",
                        message=f"Duplicate SA2 code found: {sa2_code} appears {len(record_indices)} times",
                        details={
                            'sa2_code': sa2_code,
                            'duplicate_count': len(record_indices),
                            'record_indices': record_indices
                        },
                        affected_records=record_indices
                    ))
            
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="sa2_coverage_validation_error",
                message=f"SA2 coverage validation failed: {str(e)}"
            ))
        
        return results
    
    def validate_boundary_topology(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate boundary topology including gaps, overlaps, and polygon validity.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Boundary topology validation results
        """
        results = []
        self.enhanced_statistics['boundary_topology_checks'] += 1
        
        if not SPATIAL_LIBRARIES_AVAILABLE:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                rule_id="boundary_topology_libraries_unavailable",
                message="Spatial libraries unavailable for boundary topology validation"
            ))
            return results
        
        try:
            # Extract geometric data
            geometries = []
            for record_idx, record in enumerate(data):
                # Check if record has geometry data
                if self._has_geometry_data(record):
                    geometry_result = self._validate_single_boundary_topology(record, record_idx)
                    if geometry_result:
                        results.append(geometry_result)
                        
                # Store geometry for adjacency checking
                if self._extract_geometry_from_record(record):
                    geometries.append((record_idx, self._extract_geometry_from_record(record)))
            
            # Check for boundary gaps and overlaps between adjacent SA2 areas
            if len(geometries) > 1:
                adjacency_results = self._validate_boundary_adjacency(geometries)
                results.extend(adjacency_results)
            
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="boundary_topology_validation_error",
                message=f"Boundary topology validation failed: {str(e)}"
            ))
        
        return results
    
    def validate_coordinate_reference_system(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate coordinate reference system (CRS) compliance with EPSG:7855 (GDA2020 MGA Zone 55).
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: CRS validation results
        """
        results = []
        self.enhanced_statistics['crs_validation_checks'] += 1
        
        if not SPATIAL_LIBRARIES_AVAILABLE:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                rule_id="crs_validation_libraries_unavailable",
                message="Spatial libraries unavailable for CRS validation"
            ))
            return results
        
        try:
            # Create CRS transformers
            target_crs = CRS.from_epsg(self.target_crs)
            wgs84_crs = CRS.from_epsg(4326)  # WGS84 for geographic coordinates
            
            for record_idx, record in enumerate(data):
                lat = record.get('latitude') or record.get('lat')
                lon = record.get('longitude') or record.get('lon')
                
                if lat is not None and lon is not None:
                    crs_result = self._validate_coordinate_crs(lat, lon, record_idx, target_crs, wgs84_crs)
                    if crs_result:
                        results.append(crs_result)
                
                # Validate coordinate bounds for Australian continent
                bounds_result = self._validate_australian_territorial_bounds(lat, lon, record_idx)
                if bounds_result:
                    results.append(bounds_result)
                
                # Validate coordinate precision and accuracy
                precision_result = self._validate_coordinate_precision_enhanced(lat, lon, record_idx)
                if precision_result:
                    results.append(precision_result)
            
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="crs_validation_error",
                message=f"CRS validation failed: {str(e)}"
            ))
        
        return results
    
    def validate_spatial_hierarchy(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate spatial hierarchy consistency (SA2 -> SA3 -> SA4 -> State).
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Spatial hierarchy validation results
        """
        results = []
        self.enhanced_statistics['spatial_hierarchy_checks'] += 1
        
        try:
            for record_idx, record in enumerate(data):
                sa2_code = record.get('sa2_code')
                sa3_code = record.get('sa3_code')
                sa4_code = record.get('sa4_code')
                state_code = record.get('state_code')
                
                # Enhanced SA2 -> SA3 validation
                if sa2_code and sa3_code:
                    hierarchy_result = self._validate_enhanced_sa2_sa3_hierarchy(sa2_code, sa3_code, record_idx)
                    if hierarchy_result:
                        results.append(hierarchy_result)
                
                # Enhanced SA3 -> SA4 validation
                if sa3_code and sa4_code:
                    hierarchy_result = self._validate_enhanced_sa3_sa4_hierarchy(sa3_code, sa4_code, record_idx)
                    if hierarchy_result:
                        results.append(hierarchy_result)
                
                # Enhanced SA4 -> State validation
                if sa4_code and state_code:
                    hierarchy_result = self._validate_enhanced_sa4_state_hierarchy(sa4_code, state_code, record_idx)
                    if hierarchy_result:
                        results.append(hierarchy_result)
                
                # Validate hierarchical consistency across all levels
                consistency_result = self._validate_complete_hierarchy_consistency(
                    sa2_code, sa3_code, sa4_code, state_code, record_idx
                )
                if consistency_result:
                    results.append(consistency_result)
            
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="spatial_hierarchy_validation_error",
                message=f"Spatial hierarchy validation failed: {str(e)}"
            ))
        
        return results
    
    def validate_geographic_consistency(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate geographic consistency including area calculations, population density,
        and coastal classifications.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Geographic consistency validation results
        """
        results = []
        self.enhanced_statistics['geographic_consistency_checks'] += 1
        
        try:
            for record_idx, record in enumerate(data):
                # Validate area calculations
                area_result = self._validate_area_calculations(record, record_idx)
                if area_result:
                    results.append(area_result)
                
                # Validate population density calculations
                density_result = self._validate_population_density(record, record_idx)
                if density_result:
                    results.append(density_result)
                
                # Validate centroid coordinates are within boundaries
                centroid_result = self._validate_centroid_within_boundaries(record, record_idx)
                if centroid_result:
                    results.append(centroid_result)
                
                # Validate coastal vs inland classification accuracy
                coastal_result = self._validate_coastal_classification(record, record_idx)
                if coastal_result:
                    results.append(coastal_result)
            
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="geographic_consistency_validation_error",
                message=f"Geographic consistency validation failed: {str(e)}"
            ))
        
        return results
    
    def get_validation_rules(self) -> List[str]:
        """
        Get the list of enhanced validation rules supported by this validator.
        
        Returns:
            List[str]: List of enhanced validation rule identifiers
        """
        base_rules = super().get_validation_rules()
        enhanced_rules = [
            "enhanced_sa2_coverage_validation",
            "boundary_topology_validation", 
            "coordinate_reference_system_validation",
            "spatial_hierarchy_validation",
            "geographic_consistency_validation"
        ]
        return base_rules + enhanced_rules
    
    # Private helper methods
    
    def _load_official_sa2_codes(self) -> Set[str]:
        """
        Load official SA2 codes from file or generate expected codes.
        
        Returns:
            Set[str]: Set of official SA2 codes
        """
        if self.official_sa2_codes_file and Path(self.official_sa2_codes_file).exists():
            try:
                with open(self.official_sa2_codes_file, 'r') as f:
                    codes = {line.strip() for line in f if line.strip()}
                self.logger.info(f"Loaded {len(codes)} official SA2 codes from {self.official_sa2_codes_file}")
                return codes
            except Exception as e:
                self.logger.warning(f"Failed to load SA2 codes from file: {e}")
        
        # Generate expected SA2 codes based on state prefixes and known ranges
        # This is a simplified version - in production, use the official ABS file
        expected_codes = set()
        state_prefixes = ['1', '2', '3', '4', '5', '6', '7', '8']  # NSW, VIC, QLD, SA, WA, TAS, NT, ACT
        
        for prefix in state_prefixes:
            # Generate plausible SA2 codes for each state
            # This is a placeholder - replace with actual ABS SA2 code list
            for i in range(10001, 19999):  # Example range
                if len(expected_codes) >= self.OFFICIAL_SA2_COUNT:
                    break
                code = f"{prefix}{i:08d}"
                expected_codes.add(code)
        
        self.logger.warning(
            f"Using generated SA2 codes ({len(expected_codes)} codes). "
            "For production use, load official ABS SA2 code list."
        )
        return expected_codes
    
    def _validate_sa2_code_format_enhanced(self, sa2_code: str) -> bool:
        """
        Validate enhanced SA2 code format (11-digit format: SSSAASSSSS).
        
        Args:
            sa2_code: SA2 code to validate
            
        Returns:
            bool: True if format is valid
        """
        # Enhanced pattern for 11-digit SA2 codes
        pattern = r'^[1-8]\d{2}[0-9A-Z]{2}\d{6}$'
        return bool(re.match(pattern, sa2_code))
    
    def _has_geometry_data(self, record: DataRecord) -> bool:
        """
        Check if record contains geometry data.
        
        Args:
            record: Data record to check
            
        Returns:
            bool: True if geometry data is present
        """
        geometry_fields = ['geometry', 'wkt', 'geojson', 'boundary_coordinates']
        return any(field in record for field in geometry_fields)
    
    def _extract_geometry_from_record(self, record: DataRecord) -> Optional[shapely.geometry.base.BaseGeometry]:
        """
        Extract geometry from a data record.
        
        Args:
            record: Data record containing geometry data
            
        Returns:
            Optional[BaseGeometry]: Extracted geometry or None
        """
        if not SPATIAL_LIBRARIES_AVAILABLE:
            return None
        
        try:
            # Try different geometry formats
            if 'geometry' in record and record['geometry']:
                return shapely.wkt.loads(str(record['geometry']))
            elif 'wkt' in record and record['wkt']:
                return shapely.wkt.loads(str(record['wkt']))
            elif 'geojson' in record and record['geojson']:
                return shapely.geometry.shape(record['geojson'])
            elif 'boundary_coordinates' in record and record['boundary_coordinates']:
                # Assume it's a list of coordinate pairs forming a polygon
                coords = record['boundary_coordinates']
                if isinstance(coords, list) and len(coords) >= 3:
                    return Polygon(coords)
        except Exception as e:
            self.logger.debug(f"Failed to extract geometry from record: {e}")
        
        return None
    
    def _validate_single_boundary_topology(self, record: DataRecord, record_idx: int) -> Optional[ValidationResult]:
        """
        Validate topology of a single boundary geometry.
        
        Args:
            record: Data record with geometry
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Validation result if issues found
        """
        geometry = self._extract_geometry_from_record(record)
        if not geometry:
            return None
        
        try:
            # Check if geometry is valid
            if not geometry.is_valid:
                validity_reason = explain_validity(geometry)
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="boundary_invalid_topology",
                    message=f"Invalid boundary topology: {validity_reason}",
                    details={
                        'geometry_type': geometry.geom_type,
                        'validity_reason': validity_reason,
                        'sa2_code': record.get('sa2_code')
                    },
                    affected_records=[record_idx]
                )
            
            # Check for self-intersections
            if hasattr(geometry, 'exterior') and geometry.exterior.is_ring:
                # For polygons, check if exterior ring is simple (no self-intersections)
                if not geometry.exterior.is_simple:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id="boundary_self_intersection",
                        message="Boundary has self-intersections",
                        details={
                            'geometry_type': geometry.geom_type,
                            'sa2_code': record.get('sa2_code')
                        },
                        affected_records=[record_idx]
                    )
            
            # Check minimum area threshold
            min_area = self.topology_config.get('minimum_area_threshold', 0.0001)
            if geometry.area < min_area:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="boundary_minimum_area",
                    message=f"Boundary area {geometry.area:.6f} sq degrees below minimum threshold {min_area}",
                    details={
                        'area': geometry.area,
                        'minimum_threshold': min_area,
                        'sa2_code': record.get('sa2_code')
                    },
                    affected_records=[record_idx]
                )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="boundary_topology_check_error",
                message=f"Boundary topology check failed: {str(e)}",
                affected_records=[record_idx]
            )
        
        return None
    
    def _validate_boundary_adjacency(self, geometries: List[Tuple[int, Any]]) -> List[ValidationResult]:
        """
        Validate boundary adjacency for gap and overlap detection.
        
        Args:
            geometries: List of (record_idx, geometry) tuples
            
        Returns:
            List[ValidationResult]: Adjacency validation results
        """
        results = []
        tolerance = self.topology_tolerance / 111000  # Convert metres to degrees (approximate)
        
        try:
            # Sample geometries if too many (performance consideration)
            if len(geometries) > 100:
                import random
                geometries = random.sample(geometries, 100)
            
            # Check for overlaps and gaps between nearby geometries
            for i in range(len(geometries)):
                for j in range(i + 1, len(geometries)):
                    idx1, geom1 = geometries[i]
                    idx2, geom2 = geometries[j]
                    
                    if geom1 and geom2:
                        # Check if geometries are adjacent (share a boundary)
                        if geom1.touches(geom2):
                            # Check for overlaps
                            if geom1.overlaps(geom2):
                                overlap_area = geom1.intersection(geom2).area
                                results.append(ValidationResult(
                                    is_valid=False,
                                    severity=ValidationSeverity.ERROR,
                                    rule_id="boundary_overlap_detected",
                                    message=f"Boundary overlap detected between adjacent areas (overlap area: {overlap_area:.6f})",
                                    details={
                                        'overlap_area': overlap_area,
                                        'tolerance': tolerance
                                    },
                                    affected_records=[idx1, idx2]
                                ))
                        
                        # Check for gaps (distance between geometries)
                        distance = geom1.distance(geom2)
                        if 0 < distance < tolerance:
                            results.append(ValidationResult(
                                is_valid=False,
                                severity=ValidationSeverity.WARNING,
                                rule_id="boundary_gap_detected",
                                message=f"Small gap detected between boundaries (distance: {distance:.6f} degrees)",
                                details={
                                    'gap_distance': distance,
                                    'tolerance': tolerance
                                },
                                affected_records=[idx1, idx2]
                            ))
        
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="boundary_adjacency_check_error",
                message=f"Boundary adjacency check failed: {str(e)}"
            ))
        
        return results
    
    def _validate_coordinate_crs(
        self, 
        lat: Any, 
        lon: Any, 
        record_idx: int, 
        target_crs: Any, 
        source_crs: Any
    ) -> Optional[ValidationResult]:
        """
        Validate coordinate reference system transformation.
        
        Args:
            lat: Latitude value
            lon: Longitude value
            record_idx: Record index
            target_crs: Target CRS object
            source_crs: Source CRS object
            
        Returns:
            Optional[ValidationResult]: CRS validation result
        """
        try:
            lat_val = float(lat)
            lon_val = float(lon)
            
            # Create transformer
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            
            # Transform coordinates
            x, y = transformer.transform(lon_val, lat_val)
            
            # Check if transformed coordinates are reasonable for Australian extent
            # GDA2020 MGA Zone 55 bounds (approximate)
            mga55_bounds = {
                'x_min': 140000,   # Western bound
                'x_max': 880000,   # Eastern bound  
                'y_min': 5160000,  # Southern bound
                'y_max': 8900000   # Northern bound
            }
            
            if not (mga55_bounds['x_min'] <= x <= mga55_bounds['x_max'] and 
                    mga55_bounds['y_min'] <= y <= mga55_bounds['y_max']):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="crs_coordinates_out_of_bounds",
                    message=f"Transformed coordinates ({x:.0f}, {y:.0f}) outside expected Australian bounds",
                    details={
                        'original_lat': lat_val,
                        'original_lon': lon_val,
                        'transformed_x': x,
                        'transformed_y': y,
                        'target_crs': self.target_crs,
                        'expected_bounds': mga55_bounds
                    },
                    affected_records=[record_idx]
                )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="crs_transformation_error",
                message=f"CRS transformation failed: {str(e)}",
                details={'latitude': lat, 'longitude': lon},
                affected_records=[record_idx]
            )
        
        return None
    
    def _validate_australian_territorial_bounds(self, lat: Any, lon: Any, record_idx: int) -> Optional[ValidationResult]:
        """
        Validate coordinates are within Australian territorial bounds.
        
        Args:
            lat: Latitude value
            lon: Longitude value  
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Territorial bounds validation result
        """
        try:
            lat_val = float(lat)
            lon_val = float(lon)
            
            # Extended Australian territorial bounds (including external territories)
            territorial_bounds = {
                'lat_min': -54.777,  # Macquarie Island
                'lat_max': -9.142,   # Boigu Island, Torres Strait
                'lon_min': 72.246,   # Heard Island  
                'lon_max': 167.998   # Norfolk Island
            }
            
            if not (territorial_bounds['lat_min'] <= lat_val <= territorial_bounds['lat_max'] and
                    territorial_bounds['lon_min'] <= lon_val <= territorial_bounds['lon_max']):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="coordinates_outside_australian_territory",
                    message=f"Coordinates ({lat_val}, {lon_val}) outside Australian territorial bounds",
                    details={
                        'latitude': lat_val,
                        'longitude': lon_val,
                        'territorial_bounds': territorial_bounds
                    },
                    affected_records=[record_idx]
                )
            
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="coordinates_invalid_format_crs",
                message=f"Invalid coordinate format for CRS validation: lat={lat}, lon={lon}",
                affected_records=[record_idx]
            )
        
        return None
    
    def _validate_coordinate_precision_enhanced(self, lat: Any, lon: Any, record_idx: int) -> Optional[ValidationResult]:
        """
        Validate enhanced coordinate precision requirements.
        
        Args:
            lat: Latitude value
            lon: Longitude value
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Precision validation result
        """
        try:
            lat_str = str(lat)
            lon_str = str(lon)
            
            # Check decimal places (precision)
            lat_decimals = len(lat_str.split('.')[-1]) if '.' in lat_str else 0
            lon_decimals = len(lon_str.split('.')[-1]) if '.' in lon_str else 0
            
            min_precision = self.coordinate_precision
            
            if lat_decimals < min_precision or lon_decimals < min_precision:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="coordinate_precision_insufficient",
                    message=f"Coordinate precision insufficient: lat={lat_decimals} decimals, lon={lon_decimals} decimals (minimum: {min_precision})",
                    details={
                        'latitude': lat,
                        'longitude': lon,
                        'lat_precision': lat_decimals,
                        'lon_precision': lon_decimals,
                        'minimum_precision': min_precision,
                        'accuracy_metres': 10 ** (6 - min(lat_decimals, lon_decimals)) * 111  # Approximate
                    },
                    affected_records=[record_idx]
                )
        
        except Exception:
            pass  # Precision check is non-critical
        
        return None
    
    def _validate_enhanced_sa2_sa3_hierarchy(self, sa2_code: str, sa3_code: str, record_idx: int) -> Optional[ValidationResult]:
        """
        Enhanced validation of SA2 to SA3 hierarchy.
        
        Args:
            sa2_code: SA2 code
            sa3_code: SA3 code
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Hierarchy validation result
        """
        sa2_str = str(sa2_code).strip()
        sa3_str = str(sa3_code).strip()
        
        # Enhanced hierarchy check for 11-digit SA2 codes
        if len(sa2_str) == 11 and len(sa3_str) >= 5:
            # SA3 code should match the first 5 characters of SA2 code
            expected_sa3_prefix = sa2_str[:5]
            actual_sa3_prefix = sa3_str[:5]
            
            if expected_sa3_prefix != actual_sa3_prefix:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="enhanced_sa2_sa3_hierarchy_mismatch",
                    message=f"Enhanced SA2 {sa2_code} does not belong to SA3 {sa3_code} (hierarchy mismatch)",
                    details={
                        'sa2_code': sa2_code,
                        'sa3_code': sa3_code,
                        'expected_sa3_prefix': expected_sa3_prefix,
                        'actual_sa3_prefix': actual_sa3_prefix,
                        'hierarchy_rule': 'SA3 code should match first 5 characters of SA2 code'
                    },
                    affected_records=[record_idx]
                )
        
        return None
    
    def _validate_enhanced_sa3_sa4_hierarchy(self, sa3_code: str, sa4_code: str, record_idx: int) -> Optional[ValidationResult]:
        """
        Enhanced validation of SA3 to SA4 hierarchy.
        
        Args:
            sa3_code: SA3 code
            sa4_code: SA4 code
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Hierarchy validation result
        """
        sa3_str = str(sa3_code).strip()
        sa4_str = str(sa4_code).strip()
        
        # SA4 code should match the first 3 characters of SA3 code
        if len(sa3_str) >= 3 and len(sa4_str) >= 3:
            expected_sa4_prefix = sa3_str[:3]
            actual_sa4_prefix = sa4_str[:3]
            
            if expected_sa4_prefix != actual_sa4_prefix:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="enhanced_sa3_sa4_hierarchy_mismatch",
                    message=f"Enhanced SA3 {sa3_code} does not belong to SA4 {sa4_code} (hierarchy mismatch)",
                    details={
                        'sa3_code': sa3_code,
                        'sa4_code': sa4_code,
                        'expected_sa4_prefix': expected_sa4_prefix,
                        'actual_sa4_prefix': actual_sa4_prefix,
                        'hierarchy_rule': 'SA4 code should match first 3 characters of SA3 code'
                    },
                    affected_records=[record_idx]
                )
        
        return None
    
    def _validate_enhanced_sa4_state_hierarchy(self, sa4_code: str, state_code: str, record_idx: int) -> Optional[ValidationResult]:
        """
        Enhanced validation of SA4 to state hierarchy.
        
        Args:
            sa4_code: SA4 code
            state_code: State code
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Hierarchy validation result
        """
        sa4_str = str(sa4_code).strip()
        state_str = str(state_code).strip()
        
        # State code should match the first digit of SA4 code
        if len(sa4_str) >= 1:
            expected_state = sa4_str[0]
            
            if expected_state != state_str:
                state_name = self.state_sa2_mapping.get(expected_state, 'Unknown')
                actual_state_name = self.state_sa2_mapping.get(state_str, 'Unknown')
                
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="enhanced_sa4_state_hierarchy_mismatch",
                    message=f"Enhanced SA4 {sa4_code} indicates {state_name} but state code indicates {actual_state_name}",
                    details={
                        'sa4_code': sa4_code,
                        'state_code': state_code,
                        'expected_state': expected_state,
                        'expected_state_name': state_name,
                        'actual_state_name': actual_state_name,
                        'hierarchy_rule': 'State code should match first digit of SA4 code'
                    },
                    affected_records=[record_idx]
                )
        
        return None
    
    def _validate_complete_hierarchy_consistency(
        self, 
        sa2_code: Optional[str], 
        sa3_code: Optional[str], 
        sa4_code: Optional[str], 
        state_code: Optional[str], 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """
        Validate complete hierarchy consistency across all levels.
        
        Args:
            sa2_code: SA2 code
            sa3_code: SA3 code
            sa4_code: SA4 code
            state_code: State code
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Complete hierarchy validation result
        """
        if not all([sa2_code, sa3_code, sa4_code, state_code]):
            return None  # Skip if any level is missing
        
        try:
            sa2_str = str(sa2_code).strip()
            sa3_str = str(sa3_code).strip()
            sa4_str = str(sa4_code).strip()
            state_str = str(state_code).strip()
            
            errors = []
            
            # Check full hierarchy chain
            if len(sa2_str) >= 1 and sa2_str[0] != state_str:
                errors.append(f"SA2 state prefix ({sa2_str[0]}) doesn't match state code ({state_str})")
            
            if len(sa3_str) >= 1 and sa3_str[0] != state_str:
                errors.append(f"SA3 state prefix ({sa3_str[0]}) doesn't match state code ({state_str})")
            
            if len(sa4_str) >= 1 and sa4_str[0] != state_str:
                errors.append(f"SA4 state prefix ({sa4_str[0]}) doesn't match state code ({state_str})")
            
            if errors:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="complete_hierarchy_consistency_error",
                    message=f"Complete hierarchy consistency errors: {'; '.join(errors)}",
                    details={
                        'sa2_code': sa2_code,
                        'sa3_code': sa3_code,
                        'sa4_code': sa4_code,
                        'state_code': state_code,
                        'consistency_errors': errors
                    },
                    affected_records=[record_idx]
                )
        
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="hierarchy_consistency_check_error",
                message=f"Hierarchy consistency check failed: {str(e)}",
                affected_records=[record_idx]
            )
        
        return None
    
    def _validate_area_calculations(self, record: DataRecord, record_idx: int) -> Optional[ValidationResult]:
        """
        Validate area calculations for geographic consistency.
        
        Args:
            record: Data record
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Area validation result
        """
        area_sqkm = record.get('geographic_area_sqkm') or record.get('area_sqkm')
        if area_sqkm is None:
            return None
        
        try:
            area_val = float(area_sqkm)
            
            # Validate area is within reasonable bounds for Australian SA2s
            min_area = 0.001   # 0.001 sq km (very small urban areas)
            max_area = 100000  # 100,000 sq km (very large remote areas)
            
            if area_val < min_area:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="area_below_minimum_threshold",
                    message=f"Area {area_val} sq km below minimum threshold {min_area} sq km",
                    details={
                        'area_sqkm': area_val,
                        'minimum_threshold': min_area,
                        'sa2_code': record.get('sa2_code')
                    },
                    affected_records=[record_idx]
                )
            
            if area_val > max_area:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="area_above_maximum_threshold",
                    message=f"Area {area_val} sq km above maximum threshold {max_area} sq km",
                    details={
                        'area_sqkm': area_val,
                        'maximum_threshold': max_area,
                        'sa2_code': record.get('sa2_code')
                    },
                    affected_records=[record_idx]
                )
        
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="area_invalid_format",
                message=f"Invalid area format: {area_sqkm}",
                details={'area_value': area_sqkm},
                affected_records=[record_idx]
            )
        
        return None
    
    def _validate_population_density(self, record: DataRecord, record_idx: int) -> Optional[ValidationResult]:
        """
        Validate population density calculations.
        
        Args:
            record: Data record
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Population density validation result
        """
        population = record.get('total_population') or record.get('usual_resident_population')
        area_sqkm = record.get('geographic_area_sqkm') or record.get('area_sqkm')
        
        if population is None or area_sqkm is None:
            return None
        
        try:
            pop_val = float(population)
            area_val = float(area_sqkm)
            
            if area_val <= 0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="population_density_zero_area",
                    message="Cannot calculate population density: area is zero or negative",
                    details={
                        'population': pop_val,
                        'area_sqkm': area_val,
                        'sa2_code': record.get('sa2_code')
                    },
                    affected_records=[record_idx]
                )
            
            density = pop_val / area_val
            
            # Validate density is within reasonable bounds
            max_density = 50000  # 50,000 people per sq km (very dense urban)
            min_density = 0      # 0 people per sq km (uninhabited areas)
            
            if density > max_density:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="population_density_extremely_high",
                    message=f"Population density {density:.1f} people/sq km exceeds maximum expected {max_density}",
                    details={
                        'population': pop_val,
                        'area_sqkm': area_val,
                        'density_per_sqkm': density,
                        'maximum_expected': max_density,
                        'sa2_code': record.get('sa2_code')
                    },
                    affected_records=[record_idx]
                )
        
        except (ValueError, TypeError, ZeroDivisionError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="population_density_calculation_error",
                message="Population density calculation failed",
                details={
                    'population': population,
                    'area_sqkm': area_sqkm
                },
                affected_records=[record_idx]
            )
        
        return None
    
    def _validate_centroid_within_boundaries(self, record: DataRecord, record_idx: int) -> Optional[ValidationResult]:
        """
        Validate centroid coordinates are within the area's boundaries.
        
        Args:
            record: Data record
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Centroid validation result
        """
        lat = record.get('latitude') or record.get('lat') or record.get('centroid_lat')
        lon = record.get('longitude') or record.get('lon') or record.get('centroid_lon')
        
        if lat is None or lon is None:
            return None
        
        # This would require boundary geometry to properly validate
        # For now, do basic coordinate validation
        try:
            lat_val = float(lat)
            lon_val = float(lon)
            
            # Basic check: ensure centroid is in Australia
            if not (-54.777 <= lat_val <= -9.142 and 72.246 <= lon_val <= 167.998):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="centroid_outside_australia",
                    message=f"Centroid coordinates ({lat_val}, {lon_val}) are outside Australian bounds",
                    details={
                        'centroid_lat': lat_val,
                        'centroid_lon': lon_val,
                        'sa2_code': record.get('sa2_code')
                    },
                    affected_records=[record_idx]
                )
        
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="centroid_invalid_coordinates",
                message=f"Invalid centroid coordinates: lat={lat}, lon={lon}",
                affected_records=[record_idx]
            )
        
        return None
    
    def _validate_coastal_classification(self, record: DataRecord, record_idx: int) -> Optional[ValidationResult]:
        """
        Validate coastal vs inland classification accuracy.
        
        Args:
            record: Data record
            record_idx: Record index
            
        Returns:
            Optional[ValidationResult]: Coastal classification validation result
        """
        is_coastal = record.get('is_coastal') or record.get('coastal_area')
        lat = record.get('latitude') or record.get('lat')
        lon = record.get('longitude') or record.get('lon')
        
        if is_coastal is None or lat is None or lon is None:
            return None
        
        try:
            lat_val = float(lat)
            lon_val = float(lon)
            
            # Simple heuristic: areas within certain distance of coast
            # This is a placeholder - in production, use proper coastal boundary data
            coastal_threshold_degrees = 0.5  # Approximately 50km
            
            # Australia's approximate coastal bounds
            is_near_coast = (
                abs(lat_val - (-43.7)) < coastal_threshold_degrees or  # Near southern coast
                abs(lat_val - (-10.7)) < coastal_threshold_degrees or  # Near northern coast
                abs(lon_val - 113.2) < coastal_threshold_degrees or   # Near western coast
                abs(lon_val - 153.6) < coastal_threshold_degrees      # Near eastern coast
            )
            
            # Check for inconsistency
            if bool(is_coastal) != is_near_coast:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.INFO,
                    rule_id="coastal_classification_inconsistent",
                    message=f"Coastal classification may be inconsistent with location",
                    details={
                        'is_coastal': is_coastal,
                        'estimated_near_coast': is_near_coast,
                        'latitude': lat_val,
                        'longitude': lon_val,
                        'sa2_code': record.get('sa2_code')
                    },
                    affected_records=[record_idx]
                )
        
        except (ValueError, TypeError):
            pass  # Non-critical validation
        
        return None
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """
        Get enhanced geographic validation statistics.
        
        Returns:
            Dict[str, Any]: Enhanced validation statistics
        """
        base_stats = self.get_geographic_statistics()
        base_stats.update(self.enhanced_statistics)
        return base_stats