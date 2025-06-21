"""
Geographic Validation Framework

This module provides comprehensive geographic and spatial validation for
Australian health geography datasets, including SA2 boundary validation,
coordinate system validation, topology checking, coverage validation,
and geographic relationship validation.
"""

import logging
import re
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

from ..utils.interfaces import (
    DataBatch, 
    DataRecord, 
    ValidationResult, 
    ValidationSeverity,
    GeographicValidationError
)
from .base import BaseValidator


@dataclass
class CoordinateValidationResult:
    """Result of coordinate validation."""
    is_valid: bool
    latitude: Optional[float]
    longitude: Optional[float]
    error_message: Optional[str] = None
    coordinate_system: Optional[str] = None


@dataclass
class BoundaryValidationResult:
    """Result of boundary validation."""
    sa2_code: str
    is_within_bounds: bool
    state_code: Optional[str] = None
    boundary_violation: Optional[str] = None


@dataclass
class TopologyValidationResult:
    """Result of topology validation."""
    geometry_id: str
    is_valid: bool
    topology_errors: List[str] = field(default_factory=list)
    area_sqkm: Optional[float] = None


@dataclass
class CoverageValidationResult:
    """Result of coverage validation."""
    coverage_type: str
    expected_count: int
    actual_count: int
    missing_items: List[str] = field(default_factory=list)
    coverage_percentage: float = 0.0


class GeographicValidator(BaseValidator):
    """
    Comprehensive geographic validation framework for Australian health geography data.
    
    This validator provides spatial validation including coordinate system validation,
    SA2 boundary validation, topology checking, coverage validation, and geographic
    relationship validation specific to Australian geography.
    """
    
    def __init__(
        self,
        validator_id: str = "geographic_validator",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the geographic validator.
        
        Args:
            validator_id: Unique identifier for this validator
            config: Configuration dictionary containing geographic rules
            logger: Optional logger instance
        """
        super().__init__(validator_id, config or {}, logger)
        
        # Load geographic validation configuration
        self.geographic_config = self.config.get('geographic_rules', {})
        
        # Coordinate system configuration
        self.coordinate_systems = self.geographic_config.get('coordinate_systems', {})
        
        # Geographic boundaries
        self.geographic_boundaries = self.geographic_config.get('geographic_boundaries', {})
        
        # SA2 validation configuration
        self.sa2_config = self.geographic_config.get('sa2_validation', {})
        
        # Postcode validation configuration
        self.postcode_config = self.geographic_config.get('postcode_validation', {})
        
        # Topology validation configuration
        self.topology_config = self.geographic_config.get('topology_validation', {})
        
        # Coverage validation configuration
        self.coverage_config = self.geographic_config.get('coverage_validation', {})
        
        # Spatial relationships configuration
        self.spatial_relationships = self.geographic_config.get('spatial_relationships', {})
        
        # Reference datasets
        self.reference_datasets = self.geographic_config.get('reference_datasets', {})
        
        # Australian state/territory bounds
        self.state_bounds = self.geographic_boundaries.get('state_territory_bounds', {})
        
        # Continental Australia bounds
        self.australia_bounds = self.geographic_boundaries.get('continental_australia', {})
        
        # State/Territory to SA2 prefix mapping
        self.state_sa2_mapping = {
            "1": "NSW", "2": "VIC", "3": "QLD", "4": "SA",
            "5": "WA", "6": "TAS", "7": "NT", "8": "ACT"
        }
        
        # Postcode state ranges
        self.postcode_state_ranges = self.postcode_config.get('state_postcode_ranges', {})
        
        # Validation statistics
        self._geographic_statistics = {
            'coordinates_validated': 0,
            'sa2_codes_validated': 0,
            'postcodes_validated': 0,
            'boundary_checks': 0,
            'topology_checks': 0,
            'coverage_checks': 0
        }
        
    def validate(self, data: DataBatch) -> List[ValidationResult]:
        """
        Perform comprehensive geographic validation.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Geographic validation results
        """
        if not data:
            return [ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="geographic_empty_data",
                message="Cannot perform geographic validation on empty dataset"
            )]
        
        results = []
        start_time = datetime.now()
        
        try:
            # Coordinate validation
            coordinate_results = self.validate_coordinates(data)
            results.extend(coordinate_results)
            
            # SA2 boundary validation
            sa2_results = self.validate_sa2_boundaries(data)
            results.extend(sa2_results)
            
            # Postcode validation
            postcode_results = self.validate_postcodes(data)
            results.extend(postcode_results)
            
            # Topology validation
            topology_results = self.validate_topology(data)
            results.extend(topology_results)
            
            # Coverage validation
            coverage_results = self.validate_coverage(data)
            results.extend(coverage_results)
            
            # Spatial relationship validation
            spatial_results = self.validate_spatial_relationships(data)
            results.extend(spatial_results)
            
            # Geographic hierarchy validation
            hierarchy_results = self.validate_geographic_hierarchy(data)
            results.extend(hierarchy_results)
            
            # Update statistics
            self._update_geographic_statistics(results, len(data))
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Geographic validation completed in {duration:.2f}s: "
                f"{len(results)} geographic issues found across {len(data)} records"
            )
            
        except Exception as e:
            self.logger.error(f"Geographic validation failed: {e}")
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="geographic_validation_error",
                message=f"Geographic validation failed: {str(e)}"
            ))
        
        return results
    
    def validate_coordinates(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate geographic coordinates.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Coordinate validation results
        """
        results = []
        coordinate_validation_config = self.geographic_config.get('coordinate_validation', {})
        
        for record_idx, record in enumerate(data):
            # Check for latitude and longitude fields
            lat = record.get('latitude') or record.get('lat') or record.get('y')
            lon = record.get('longitude') or record.get('lon') or record.get('lng') or record.get('x')
            
            if lat is not None and lon is not None:
                coord_result = self._validate_coordinate_pair(lat, lon, record_idx)
                if coord_result:
                    results.append(coord_result)
                
                # Validate coordinates are within Australian bounds
                aus_bounds_result = self._validate_australian_bounds(lat, lon, record_idx)
                if aus_bounds_result:
                    results.append(aus_bounds_result)
                
                # Validate coordinate precision
                precision_result = self._validate_coordinate_precision(lat, lon, record_idx)
                if precision_result:
                    results.append(precision_result)
                
                # Check for "Null Island" (0,0) coordinates
                null_island_result = self._check_null_island(lat, lon, record_idx)
                if null_island_result:
                    results.append(null_island_result)
                
                self._geographic_statistics['coordinates_validated'] += 1
        
        # Check for duplicate coordinates
        duplicate_results = self._check_duplicate_coordinates(data)
        results.extend(duplicate_results)
        
        return results
    
    def validate_sa2_boundaries(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate SA2 boundary codes and relationships.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: SA2 boundary validation results
        """
        results = []
        
        for record_idx, record in enumerate(data):
            sa2_code = record.get('sa2_code')
            if sa2_code:
                # Validate SA2 code format
                format_result = self._validate_sa2_code_format(sa2_code, record_idx)
                if format_result:
                    results.append(format_result)
                
                # Validate SA2 state prefix
                state_code = record.get('state_code')
                if state_code:
                    state_result = self._validate_sa2_state_consistency(sa2_code, state_code, record_idx)
                    if state_result:
                        results.append(state_result)
                
                # Validate SA2 with coordinates if available
                lat = record.get('latitude') or record.get('lat')
                lon = record.get('longitude') or record.get('lon')
                if lat is not None and lon is not None:
                    coord_sa2_result = self._validate_sa2_coordinate_consistency(
                        sa2_code, lat, lon, record_idx
                    )
                    if coord_sa2_result:
                        results.append(coord_sa2_result)
                
                self._geographic_statistics['sa2_codes_validated'] += 1
        
        return results
    
    def validate_postcodes(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate Australian postcodes.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Postcode validation results
        """
        results = []
        
        for record_idx, record in enumerate(data):
            postcode = record.get('postcode') or record.get('postal_code')
            if postcode:
                # Validate postcode format
                format_result = self._validate_postcode_format(postcode, record_idx)
                if format_result:
                    results.append(format_result)
                
                # Validate postcode state consistency
                state_code = record.get('state_code')
                if state_code:
                    state_result = self._validate_postcode_state_consistency(
                        postcode, state_code, record_idx
                    )
                    if state_result:
                        results.append(state_result)
                
                self._geographic_statistics['postcodes_validated'] += 1
        
        return results
    
    def validate_topology(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate geometric topology.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Topology validation results
        """
        results = []
        
        # Check for minimum area thresholds
        min_area_threshold = self.topology_config.get('geometry_checks', {}).get('minimum_area_threshold', 0.0001)
        
        for record_idx, record in enumerate(data):
            area = record.get('geographic_area_sqkm') or record.get('area_sqkm')
            if area is not None:
                try:
                    area_val = float(area)
                    if area_val < min_area_threshold:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            rule_id="topology_minimum_area",
                            message=f"Geographic area {area_val} sq km below minimum threshold {min_area_threshold}",
                            details={
                                'area_sqkm': area_val,
                                'minimum_threshold': min_area_threshold
                            },
                            affected_records=[record_idx]
                        ))
                    
                    # Check for unreasonably large areas
                    max_area_threshold = 100000  # 100,000 sq km
                    if area_val > max_area_threshold:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            rule_id="topology_maximum_area",
                            message=f"Geographic area {area_val} sq km exceeds reasonable maximum {max_area_threshold}",
                            details={
                                'area_sqkm': area_val,
                                'maximum_threshold': max_area_threshold
                            },
                            affected_records=[record_idx]
                        ))
                    
                    self._geographic_statistics['topology_checks'] += 1
                    
                except (ValueError, TypeError):
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id="topology_invalid_area",
                        message=f"Invalid area value: {area}",
                        details={'area_value': area},
                        affected_records=[record_idx]
                    ))
        
        return results
    
    def validate_coverage(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate geographic coverage completeness.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Coverage validation results
        """
        results = []
        
        # Check SA2 coverage
        sa2_coverage_result = self._validate_sa2_coverage(data)
        if sa2_coverage_result:
            results.append(sa2_coverage_result)
        
        # Check state coverage
        state_coverage_result = self._validate_state_coverage(data)
        if state_coverage_result:
            results.append(state_coverage_result)
        
        # Check population coverage vs ABS totals
        population_coverage_result = self._validate_population_coverage(data)
        if population_coverage_result:
            results.append(population_coverage_result)
        
        self._geographic_statistics['coverage_checks'] += 1
        
        return results
    
    def validate_spatial_relationships(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate spatial relationships between geographic entities.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Spatial relationship validation results
        """
        results = []
        
        # Validate SA2-SA3 relationships
        sa2_sa3_results = self._validate_sa2_sa3_relationships(data)
        results.extend(sa2_sa3_results)
        
        # Validate distances between centroids
        distance_results = self._validate_centroid_distances(data)
        results.extend(distance_results)
        
        return results
    
    def validate_geographic_hierarchy(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate geographic hierarchy consistency.
        
        Args:
            data: Batch of data records
            
        Returns:
            List[ValidationResult]: Geographic hierarchy validation results
        """
        results = []
        
        for record_idx, record in enumerate(data):
            sa2_code = record.get('sa2_code')
            sa3_code = record.get('sa3_code')
            sa4_code = record.get('sa4_code')
            state_code = record.get('state_code')
            
            # Validate SA2 -> SA3 hierarchy
            if sa2_code and sa3_code:
                hierarchy_result = self._validate_sa2_sa3_hierarchy(sa2_code, sa3_code, record_idx)
                if hierarchy_result:
                    results.append(hierarchy_result)
            
            # Validate SA3 -> SA4 hierarchy
            if sa3_code and sa4_code:
                hierarchy_result = self._validate_sa3_sa4_hierarchy(sa3_code, sa4_code, record_idx)
                if hierarchy_result:
                    results.append(hierarchy_result)
            
            # Validate SA4 -> State hierarchy
            if sa4_code and state_code:
                hierarchy_result = self._validate_sa4_state_hierarchy(sa4_code, state_code, record_idx)
                if hierarchy_result:
                    results.append(hierarchy_result)
        
        return results
    
    def get_validation_rules(self) -> List[str]:
        """
        Get the list of validation rules supported by this validator.
        
        Returns:
            List[str]: List of validation rule identifiers
        """
        return [
            "coordinate_validation",
            "sa2_boundary_validation",
            "postcode_validation",
            "topology_validation",
            "coverage_validation",
            "spatial_relationship_validation",
            "geographic_hierarchy_validation"
        ]
    
    # Private validation methods
    
    def _validate_coordinate_pair(self, lat: Any, lon: Any, record_idx: int) -> Optional[ValidationResult]:
        """Validate a latitude/longitude coordinate pair."""
        try:
            lat_val = float(lat)
            lon_val = float(lon)
            
            # Validate latitude range
            if not (-90 <= lat_val <= 90):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="coordinate_invalid_latitude",
                    message=f"Invalid latitude: {lat_val} (must be between -90 and 90)",
                    details={'latitude': lat_val, 'longitude': lon_val},
                    affected_records=[record_idx]
                )
            
            # Validate longitude range
            if not (-180 <= lon_val <= 180):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="coordinate_invalid_longitude",
                    message=f"Invalid longitude: {lon_val} (must be between -180 and 180)",
                    details={'latitude': lat_val, 'longitude': lon_val},
                    affected_records=[record_idx]
                )
                
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="coordinate_invalid_format",
                message=f"Invalid coordinate format: lat={lat}, lon={lon}",
                details={'latitude': lat, 'longitude': lon},
                affected_records=[record_idx]
            )
        
        return None
    
    def _validate_australian_bounds(self, lat: Any, lon: Any, record_idx: int) -> Optional[ValidationResult]:
        """Validate coordinates are within Australian bounds."""
        try:
            lat_val = float(lat)
            lon_val = float(lon)
            
            # Get Australia bounds
            aus_bounds = self.australia_bounds
            lat_bounds = aus_bounds.get('latitude_bounds', {})
            lon_bounds = aus_bounds.get('longitude_bounds', {})
            
            # Check latitude bounds
            if lat_bounds and not (lat_bounds.get('min', -90) <= lat_val <= lat_bounds.get('max', 90)):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="coordinate_outside_australia",
                    message=f"Coordinates ({lat_val}, {lon_val}) appear to be outside Australia",
                    details={
                        'latitude': lat_val,
                        'longitude': lon_val,
                        'australia_lat_bounds': lat_bounds,
                        'australia_lon_bounds': lon_bounds
                    },
                    affected_records=[record_idx]
                )
            
            # Check longitude bounds
            if lon_bounds and not (lon_bounds.get('min', -180) <= lon_val <= lon_bounds.get('max', 180)):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="coordinate_outside_australia",
                    message=f"Coordinates ({lat_val}, {lon_val}) appear to be outside Australia",
                    details={
                        'latitude': lat_val,
                        'longitude': lon_val,
                        'australia_lat_bounds': lat_bounds,
                        'australia_lon_bounds': lon_bounds
                    },
                    affected_records=[record_idx]
                )
                
        except (ValueError, TypeError):
            pass  # Already handled in coordinate pair validation
        
        return None
    
    def _validate_coordinate_precision(self, lat: Any, lon: Any, record_idx: int) -> Optional[ValidationResult]:
        """Validate coordinate precision."""
        try:
            lat_str = str(lat)
            lon_str = str(lon)
            
            # Check decimal places (precision)
            lat_decimals = len(lat_str.split('.')[-1]) if '.' in lat_str else 0
            lon_decimals = len(lon_str.split('.')[-1]) if '.' in lon_str else 0
            
            min_precision = 4  # At least 4 decimal places for reasonable precision
            
            if lat_decimals < min_precision or lon_decimals < min_precision:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.INFO,
                    rule_id="coordinate_low_precision",
                    message=f"Low coordinate precision: lat={lat_decimals} decimals, lon={lon_decimals} decimals",
                    details={
                        'latitude': lat,
                        'longitude': lon,
                        'lat_precision': lat_decimals,
                        'lon_precision': lon_decimals,
                        'minimum_precision': min_precision
                    },
                    affected_records=[record_idx]
                )
                
        except Exception:
            pass  # Precision check is informational only
        
        return None
    
    def _check_null_island(self, lat: Any, lon: Any, record_idx: int) -> Optional[ValidationResult]:
        """Check for 'Null Island' (0,0) coordinates."""
        try:
            lat_val = float(lat)
            lon_val = float(lon)
            
            if lat_val == 0.0 and lon_val == 0.0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="coordinate_null_island",
                    message="Coordinates are at 'Null Island' (0,0) - likely data error",
                    details={'latitude': lat_val, 'longitude': lon_val},
                    affected_records=[record_idx]
                )
                
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _check_duplicate_coordinates(self, data: DataBatch) -> List[ValidationResult]:
        """Check for duplicate coordinate pairs."""
        results = []
        
        coordinate_counts = {}
        coordinate_indices = {}
        
        for idx, record in enumerate(data):
            lat = record.get('latitude') or record.get('lat')
            lon = record.get('longitude') or record.get('lon')
            
            if lat is not None and lon is not None:
                try:
                    lat_val = float(lat)
                    lon_val = float(lon)
                    
                    # Round to avoid floating point precision issues
                    coord_key = (round(lat_val, 6), round(lon_val, 6))
                    
                    if coord_key in coordinate_counts:
                        coordinate_counts[coord_key] += 1
                        coordinate_indices[coord_key].append(idx)
                    else:
                        coordinate_counts[coord_key] = 1
                        coordinate_indices[coord_key] = [idx]
                        
                except (ValueError, TypeError):
                    continue
        
        # Report duplicates
        tolerance_metres = self.geographic_config.get('coordinate_validation', {}).get('duplicate_coordinates', {}).get('tolerance_metres', 10)
        
        for coord_key, count in coordinate_counts.items():
            if count > 1:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="coordinate_duplicates",
                    message=f"Duplicate coordinates found: {coord_key} appears {count} times",
                    details={
                        'coordinates': coord_key,
                        'duplicate_count': count,
                        'tolerance_metres': tolerance_metres
                    },
                    affected_records=coordinate_indices[coord_key]
                ))
        
        return results
    
    def _validate_sa2_code_format(self, sa2_code: Any, record_idx: int) -> Optional[ValidationResult]:
        """Validate SA2 code format."""
        pattern = self.sa2_config.get('code_format', {}).get('pattern', r"^[1-8][0-9]{8}$")
        sa2_str = str(sa2_code).strip()
        
        if not re.match(pattern, sa2_str):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="sa2_code_format",
                message=f"Invalid SA2 code format: {sa2_code}",
                details={
                    'sa2_code': sa2_code,
                    'expected_pattern': pattern
                },
                affected_records=[record_idx]
            )
        
        return None
    
    def _validate_sa2_state_consistency(self, sa2_code: str, state_code: str, record_idx: int) -> Optional[ValidationResult]:
        """Validate SA2 code state prefix consistency."""
        sa2_str = str(sa2_code).strip()
        state_str = str(state_code).strip()
        
        if len(sa2_str) >= 1:
            sa2_state_prefix = sa2_str[0]
            expected_state = self.state_sa2_mapping.get(sa2_state_prefix)
            actual_state = self.state_sa2_mapping.get(state_str)
            
            if sa2_state_prefix != state_str:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="sa2_state_consistency",
                    message=f"SA2 code {sa2_code} prefix indicates {expected_state} but state code indicates {actual_state}",
                    details={
                        'sa2_code': sa2_code,
                        'state_code': state_code,
                        'sa2_state_prefix': sa2_state_prefix,
                        'expected_state': expected_state,
                        'actual_state': actual_state
                    },
                    affected_records=[record_idx]
                )
        
        return None
    
    def _validate_sa2_coordinate_consistency(
        self, 
        sa2_code: str, 
        lat: Any, 
        lon: Any, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate SA2 code and coordinate consistency (placeholder)."""
        # This would require spatial data to properly validate
        # For now, just check if coordinates are roughly in the right state
        
        try:
            lat_val = float(lat)
            lon_val = float(lon)
            sa2_str = str(sa2_code).strip()
            
            if len(sa2_str) >= 1:
                state_prefix = sa2_str[0]
                expected_state = self.state_sa2_mapping.get(state_prefix)
                
                # Basic bounds checking for major states
                state_bounds = self.state_bounds.get(expected_state, {})
                if state_bounds:
                    lat_bounds = state_bounds.get('latitude_bounds', {})
                    lon_bounds = state_bounds.get('longitude_bounds', {})
                    
                    lat_in_bounds = (
                        lat_bounds.get('min', -90) <= lat_val <= lat_bounds.get('max', 90)
                    )
                    lon_in_bounds = (
                        lon_bounds.get('min', -180) <= lon_val <= lon_bounds.get('max', 180)
                    )
                    
                    if not (lat_in_bounds and lon_in_bounds):
                        return ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            rule_id="sa2_coordinate_consistency",
                            message=f"Coordinates ({lat_val}, {lon_val}) may not be consistent with SA2 {sa2_code} in {expected_state}",
                            details={
                                'sa2_code': sa2_code,
                                'latitude': lat_val,
                                'longitude': lon_val,
                                'expected_state': expected_state,
                                'state_bounds': state_bounds
                            },
                            affected_records=[record_idx]
                        )
                        
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _validate_postcode_format(self, postcode: Any, record_idx: int) -> Optional[ValidationResult]:
        """Validate postcode format."""
        pattern = self.postcode_config.get('format_rules', {}).get('pattern', r"^[0-9]{4}$")
        postcode_str = str(postcode).strip()
        
        if not re.match(pattern, postcode_str):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="postcode_format",
                message=f"Invalid postcode format: {postcode}",
                details={
                    'postcode': postcode,
                    'expected_pattern': pattern
                },
                affected_records=[record_idx]
            )
        
        return None
    
    def _validate_postcode_state_consistency(
        self, 
        postcode: Any, 
        state_code: str, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate postcode state consistency."""
        try:
            postcode_num = int(str(postcode).strip())
            state_str = str(state_code).strip()
            state_name = self.state_sa2_mapping.get(state_str)
            
            if state_name and state_name in self.postcode_state_ranges:
                state_range = self.postcode_state_ranges[state_name]
                min_postcode = state_range.get('min', 0)
                max_postcode = state_range.get('max', 9999)
                additional_postcodes = state_range.get('additional', [])
                
                is_in_range = (min_postcode <= postcode_num <= max_postcode)
                is_additional = postcode_num in additional_postcodes
                
                if not (is_in_range or is_additional):
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        rule_id="postcode_state_consistency",
                        message=f"Postcode {postcode} may not belong to {state_name}",
                        details={
                            'postcode': postcode_num,
                            'state_code': state_code,
                            'state_name': state_name,
                            'expected_range': f"{min_postcode}-{max_postcode}",
                            'additional_postcodes': additional_postcodes
                        },
                        affected_records=[record_idx]
                    )
                    
        except (ValueError, TypeError):
            pass
        
        return None
    
    def _validate_sa2_coverage(self, data: DataBatch) -> Optional[ValidationResult]:
        """Validate SA2 coverage completeness."""
        coverage_config = self.coverage_config.get('completeness_checks', {}).get('sa2_coverage', {})
        
        if not coverage_config:
            return None
        
        expected_sa2_count = coverage_config.get('expected_sa2_count', 2310)
        tolerance = coverage_config.get('tolerance', 0.05)
        
        # Count unique SA2 codes
        sa2_codes = set()
        for record in data:
            sa2_code = record.get('sa2_code')
            if sa2_code:
                sa2_codes.add(str(sa2_code).strip())
        
        actual_count = len(sa2_codes)
        coverage_ratio = actual_count / expected_sa2_count if expected_sa2_count > 0 else 0
        
        if abs(coverage_ratio - 1.0) > tolerance:
            severity = ValidationSeverity.ERROR if abs(coverage_ratio - 1.0) > 0.2 else ValidationSeverity.WARNING
            
            return ValidationResult(
                is_valid=abs(coverage_ratio - 1.0) <= tolerance,
                severity=severity,
                rule_id="sa2_coverage",
                message=f"SA2 coverage: {actual_count} found, expected ~{expected_sa2_count} ({coverage_ratio:.1%})",
                details={
                    'actual_sa2_count': actual_count,
                    'expected_sa2_count': expected_sa2_count,
                    'coverage_ratio': coverage_ratio,
                    'tolerance': tolerance
                }
            )
        
        return None
    
    def _validate_state_coverage(self, data: DataBatch) -> Optional[ValidationResult]:
        """Validate state/territory coverage."""
        coverage_config = self.coverage_config.get('completeness_checks', {}).get('state_coverage', {})
        
        if not coverage_config:
            return None
        
        required_states = set(coverage_config.get('required_states', []))
        
        # Find states present in data
        states_found = set()
        for record in data:
            state_code = record.get('state_code')
            if state_code:
                state_name = self.state_sa2_mapping.get(str(state_code).strip())
                if state_name:
                    states_found.add(state_name)
        
        missing_states = required_states - states_found
        
        if missing_states:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="state_coverage",
                message=f"Missing states/territories: {sorted(missing_states)}",
                details={
                    'required_states': sorted(required_states),
                    'states_found': sorted(states_found),
                    'missing_states': sorted(missing_states)
                }
            )
        
        return None
    
    def _validate_population_coverage(self, data: DataBatch) -> Optional[ValidationResult]:
        """Validate population coverage against ABS totals."""
        coverage_config = self.coverage_config.get('population_coverage', {})
        
        if not coverage_config:
            return None
        
        reference_total = coverage_config.get('reference_total_population', 25700000)
        tolerance = coverage_config.get('tolerance', 0.10)
        
        # Sum population from data
        total_population = 0
        population_records = 0
        
        for record in data:
            population = record.get('total_population') or record.get('usual_resident_population')
            if population is not None:
                try:
                    total_population += float(population)
                    population_records += 1
                except (ValueError, TypeError):
                    continue
        
        if population_records > 0:
            coverage_ratio = total_population / reference_total
            
            if abs(coverage_ratio - 1.0) > tolerance:
                severity = ValidationSeverity.WARNING
                
                return ValidationResult(
                    is_valid=abs(coverage_ratio - 1.0) <= tolerance,
                    severity=severity,
                    rule_id="population_coverage",
                    message=f"Population coverage: {total_population:,.0f} found, expected ~{reference_total:,.0f} ({coverage_ratio:.1%})",
                    details={
                        'total_population': total_population,
                        'reference_population': reference_total,
                        'coverage_ratio': coverage_ratio,
                        'tolerance': tolerance,
                        'records_with_population': population_records
                    }
                )
        
        return None
    
    def _validate_sa2_sa3_relationships(self, data: DataBatch) -> List[ValidationResult]:
        """Validate SA2 to SA3 hierarchical relationships."""
        results = []
        
        for record_idx, record in enumerate(data):
            sa2_code = record.get('sa2_code')
            sa3_code = record.get('sa3_code')
            
            if sa2_code and sa3_code:
                hierarchy_result = self._validate_sa2_sa3_hierarchy(sa2_code, sa3_code, record_idx)
                if hierarchy_result:
                    results.append(hierarchy_result)
        
        return results
    
    def _validate_centroid_distances(self, data: DataBatch) -> List[ValidationResult]:
        """Validate distances between SA2 centroids."""
        results = []
        
        distance_config = self.geographic_config.get('spatial_calculations', {}).get('distance_validation', {})
        
        if not distance_config:
            return results
        
        min_distance_km = distance_config.get('min_distance_km', 0.1)
        max_distance_km = distance_config.get('max_distance_km', 4000)
        
        # Extract coordinates
        coordinates = []
        for idx, record in enumerate(data):
            lat = record.get('latitude') or record.get('lat')
            lon = record.get('longitude') or record.get('lon')
            sa2_code = record.get('sa2_code')
            
            if lat is not None and lon is not None and sa2_code:
                try:
                    coordinates.append((idx, float(lat), float(lon), str(sa2_code)))
                except (ValueError, TypeError):
                    continue
        
        # Check distances between nearby points (sample for performance)
        if len(coordinates) > 100:
            import random
            coordinates = random.sample(coordinates, 100)
        
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                idx1, lat1, lon1, sa2_1 = coordinates[i]
                idx2, lat2, lon2, sa2_2 = coordinates[j]
                
                distance_km = self._calculate_distance_km(lat1, lon1, lat2, lon2)
                
                if distance_km < min_distance_km:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.INFO,
                        rule_id="centroid_distance_minimum",
                        message=f"Very close SA2 centroids: {sa2_1} and {sa2_2} are {distance_km:.3f} km apart",
                        details={
                            'sa2_code_1': sa2_1,
                            'sa2_code_2': sa2_2,
                            'distance_km': distance_km,
                            'minimum_expected': min_distance_km
                        },
                        affected_records=[idx1, idx2]
                    ))
                
                if distance_km > max_distance_km:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.INFO,
                        rule_id="centroid_distance_maximum",
                        message=f"Very distant SA2 centroids: {sa2_1} and {sa2_2} are {distance_km:.1f} km apart",
                        details={
                            'sa2_code_1': sa2_1,
                            'sa2_code_2': sa2_2,
                            'distance_km': distance_km,
                            'maximum_expected': max_distance_km
                        },
                        affected_records=[idx1, idx2]
                    ))
        
        return results
    
    def _validate_sa2_sa3_hierarchy(self, sa2_code: str, sa3_code: str, record_idx: int) -> Optional[ValidationResult]:
        """Validate SA2 to SA3 hierarchy."""
        sa2_str = str(sa2_code).strip()
        sa3_str = str(sa3_code).strip()
        
        # SA3 code should be the first 5 digits of SA2 code
        if len(sa2_str) >= 5 and len(sa3_str) >= 5:
            expected_sa3_prefix = sa2_str[:5]
            actual_sa3_prefix = sa3_str[:5]
            
            if expected_sa3_prefix != actual_sa3_prefix:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="sa2_sa3_hierarchy",
                    message=f"SA2 {sa2_code} does not belong to SA3 {sa3_code}",
                    details={
                        'sa2_code': sa2_code,
                        'sa3_code': sa3_code,
                        'expected_sa3_prefix': expected_sa3_prefix,
                        'actual_sa3_prefix': actual_sa3_prefix
                    },
                    affected_records=[record_idx]
                )
        
        return None
    
    def _validate_sa3_sa4_hierarchy(self, sa3_code: str, sa4_code: str, record_idx: int) -> Optional[ValidationResult]:
        """Validate SA3 to SA4 hierarchy."""
        sa3_str = str(sa3_code).strip()
        sa4_str = str(sa4_code).strip()
        
        # SA4 code should be the first 3 digits of SA3 code
        if len(sa3_str) >= 3 and len(sa4_str) >= 3:
            expected_sa4_prefix = sa3_str[:3]
            actual_sa4_prefix = sa4_str[:3]
            
            if expected_sa4_prefix != actual_sa4_prefix:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="sa3_sa4_hierarchy",
                    message=f"SA3 {sa3_code} does not belong to SA4 {sa4_code}",
                    details={
                        'sa3_code': sa3_code,
                        'sa4_code': sa4_code,
                        'expected_sa4_prefix': expected_sa4_prefix,
                        'actual_sa4_prefix': actual_sa4_prefix
                    },
                    affected_records=[record_idx]
                )
        
        return None
    
    def _validate_sa4_state_hierarchy(self, sa4_code: str, state_code: str, record_idx: int) -> Optional[ValidationResult]:
        """Validate SA4 to state hierarchy."""
        sa4_str = str(sa4_code).strip()
        state_str = str(state_code).strip()
        
        # State code should match the first digit of SA4 code
        if len(sa4_str) >= 1:
            expected_state = sa4_str[0]
            
            if expected_state != state_str:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="sa4_state_hierarchy",
                    message=f"SA4 {sa4_code} does not belong to state {state_code}",
                    details={
                        'sa4_code': sa4_code,
                        'state_code': state_code,
                        'expected_state': expected_state
                    },
                    affected_records=[record_idx]
                )
        
        return None
    
    def _calculate_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula."""
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in kilometres
        r = 6371
        
        return c * r
    
    def _update_geographic_statistics(self, results: List[ValidationResult], record_count: int) -> None:
        """Update geographic validation statistics."""
        self._geographic_statistics['total_validations'] = self._geographic_statistics.get('total_validations', 0) + 1
        self._geographic_statistics['total_records_validated'] = self._geographic_statistics.get('total_records_validated', 0) + record_count
        
        for result in results:
            if result.severity == ValidationSeverity.ERROR:
                self._geographic_statistics['geographic_errors'] = self._geographic_statistics.get('geographic_errors', 0) + 1
            elif result.severity == ValidationSeverity.WARNING:
                self._geographic_statistics['geographic_warnings'] = self._geographic_statistics.get('geographic_warnings', 0) + 1
            else:
                self._geographic_statistics['geographic_info'] = self._geographic_statistics.get('geographic_info', 0) + 1
    
    def get_geographic_statistics(self) -> Dict[str, Any]:
        """Get geographic validation statistics."""
        return dict(self._geographic_statistics)