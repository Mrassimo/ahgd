"""
Boundary processing transformer for the AHGD ETL pipeline.

This module provides comprehensive boundary processing capabilities for
Australian Bureau of Statistics Statistical Areas (SA1, SA2, SA3, SA4),
including topology validation, geometry simplification, and spatial indexing.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from .base import BaseTransformer
from ..utils.interfaces import (
    DataBatch,
    DataRecord,
    GeographicValidationError,
    TransformationError,
    ValidationResult,
    ValidationSeverity,
)


@dataclass
class GeometryInfo:
    """Information about a geometry object."""
    geometry_type: str  # Point, LineString, Polygon, MultiPolygon
    coordinate_count: int
    bbox: Tuple[float, float, float, float]  # min_lon, min_lat, max_lon, max_lat
    area_sqkm: Optional[float] = None
    perimeter_km: Optional[float] = None
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BoundaryRecord:
    """Represents a statistical area boundary."""
    area_code: str
    area_type: str  # SA1, SA2, SA3, SA4
    area_name: str
    state_code: str
    geometry: Dict[str, Any]  # GeoJSON geometry
    geometry_info: GeometryInfo
    parent_areas: Dict[str, str] = field(default_factory=dict)  # area_type -> area_code
    child_areas: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class TopologyValidator:
    """
    Validator for ensuring geographic topology is correct.
    
    Validates that statistical area boundaries maintain proper hierarchical
    relationships and geometric integrity.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the topology validator.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation settings
        self.check_hierarchy = config.get('check_hierarchy', True)
        self.check_topology = config.get('check_topology', True)
        self.tolerance_metres = config.get('tolerance_metres', 1.0)
        
        # Statistical area hierarchy
        self.hierarchy = ['SA1', 'SA2', 'SA3', 'SA4', 'STATE']
        
        # Statistics
        self._validation_stats = {
            'boundaries_validated': 0,
            'topology_errors': 0,
            'hierarchy_errors': 0,
            'geometry_errors': 0
        }
    
    def validate_boundary(self, boundary: BoundaryRecord) -> List[ValidationResult]:
        """
        Validate a single boundary record.
        
        Args:
            boundary: Boundary record to validate
            
        Returns:
            List[ValidationResult]: Validation results
        """
        validation_results = []
        self._validation_stats['boundaries_validated'] += 1
        
        # Validate geometry
        geometry_results = self._validate_geometry(boundary)
        validation_results.extend(geometry_results)
        
        # Validate hierarchy if enabled
        if self.check_hierarchy:
            hierarchy_results = self._validate_hierarchy(boundary)
            validation_results.extend(hierarchy_results)
        
        # Validate topology if enabled
        if self.check_topology:
            topology_results = self._validate_topology(boundary)
            validation_results.extend(topology_results)
        
        return validation_results
    
    def validate_boundary_collection(self, boundaries: List[BoundaryRecord]) -> List[ValidationResult]:
        """
        Validate a collection of boundaries for consistency.
        
        Args:
            boundaries: List of boundary records
            
        Returns:
            List[ValidationResult]: Validation results
        """
        validation_results = []
        
        # Group boundaries by type
        boundary_groups = {}
        for boundary in boundaries:
            area_type = boundary.area_type
            if area_type not in boundary_groups:
                boundary_groups[area_type] = []
            boundary_groups[area_type].append(boundary)
        
        # Validate each group
        for area_type, group_boundaries in boundary_groups.items():
            group_results = self._validate_boundary_group(area_type, group_boundaries)
            validation_results.extend(group_results)
        
        # Cross-hierarchy validation
        cross_hierarchy_results = self._validate_cross_hierarchy(boundary_groups)
        validation_results.extend(cross_hierarchy_results)
        
        return validation_results
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self._validation_stats.copy()
    
    def _validate_geometry(self, boundary: BoundaryRecord) -> List[ValidationResult]:
        """Validate geometry properties."""
        results = []
        geometry = boundary.geometry
        
        # Check geometry type
        if 'type' not in geometry:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="missing_geometry_type",
                message="Geometry missing 'type' property"
            ))
            self._validation_stats['geometry_errors'] += 1
            return results
        
        geom_type = geometry['type']
        
        # Validate based on area type
        expected_types = {
            'SA1': ['Polygon', 'MultiPolygon'],
            'SA2': ['Polygon', 'MultiPolygon'],
            'SA3': ['Polygon', 'MultiPolygon'],
            'SA4': ['Polygon', 'MultiPolygon']
        }
        
        if boundary.area_type in expected_types:
            if geom_type not in expected_types[boundary.area_type]:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="invalid_geometry_type",
                    message=f"{boundary.area_type} must have {expected_types[boundary.area_type]} geometry, got {geom_type}"
                ))
                self._validation_stats['geometry_errors'] += 1
        
        # Validate coordinates
        if 'coordinates' not in geometry:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="missing_coordinates",
                message="Geometry missing coordinates"
            ))
            self._validation_stats['geometry_errors'] += 1
        else:
            coord_results = self._validate_coordinates(geometry['coordinates'], geom_type)
            results.extend(coord_results)
        
        return results
    
    def _validate_coordinates(self, coordinates: List, geometry_type: str) -> List[ValidationResult]:
        """Validate coordinate arrays."""
        results = []
        
        try:
            if geometry_type == 'Polygon':
                # Polygon should have at least one ring
                if not coordinates or len(coordinates) == 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id="empty_polygon",
                        message="Polygon has no coordinate rings"
                    ))
                    return results
                
                # Validate each ring
                for i, ring in enumerate(coordinates):
                    ring_results = self._validate_ring(ring, f"ring_{i}")
                    results.extend(ring_results)
                    
            elif geometry_type == 'MultiPolygon':
                # MultiPolygon should have at least one polygon
                if not coordinates or len(coordinates) == 0:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id="empty_multipolygon",
                        message="MultiPolygon has no polygons"
                    ))
                    return results
                
                # Validate each polygon
                for i, polygon in enumerate(coordinates):
                    for j, ring in enumerate(polygon):
                        ring_results = self._validate_ring(ring, f"polygon_{i}_ring_{j}")
                        results.extend(ring_results)
        
        except Exception as e:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="coordinate_validation_error",
                message=f"Error validating coordinates: {e}"
            ))
            self._validation_stats['geometry_errors'] += 1
        
        return results
    
    def _validate_ring(self, ring: List, ring_id: str) -> List[ValidationResult]:
        """Validate a coordinate ring."""
        results = []
        
        # Ring must have at least 4 points
        if len(ring) < 4:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="insufficient_ring_points",
                message=f"Ring {ring_id} has {len(ring)} points, minimum 4 required"
            ))
            self._validation_stats['geometry_errors'] += 1
            return results
        
        # Ring must be closed (first and last points must be the same)
        if ring[0] != ring[-1]:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="unclosed_ring",
                message=f"Ring {ring_id} is not closed"
            ))
            self._validation_stats['geometry_errors'] += 1
        
        # Validate individual coordinates
        for i, coord in enumerate(ring):
            if not isinstance(coord, list) or len(coord) < 2:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="invalid_coordinate",
                    message=f"Invalid coordinate at {ring_id}[{i}]: {coord}"
                ))
                self._validation_stats['geometry_errors'] += 1
                continue
            
            lon, lat = coord[0], coord[1]
            
            # Validate coordinate ranges
            if not (-180 <= lon <= 180):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="longitude_out_of_range",
                    message=f"Longitude {lon} out of range at {ring_id}[{i}]"
                ))
                self._validation_stats['geometry_errors'] += 1
            
            if not (-90 <= lat <= 90):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="latitude_out_of_range",
                    message=f"Latitude {lat} out of range at {ring_id}[{i}]"
                ))
                self._validation_stats['geometry_errors'] += 1
            
            # Australian bounds check
            if not (110 <= lon <= 155 and -45 <= lat <= -9):
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="outside_australian_bounds",
                    message=f"Coordinate {lon}, {lat} outside Australian bounds at {ring_id}[{i}]"
                ))
        
        return results
    
    def _validate_hierarchy(self, boundary: BoundaryRecord) -> List[ValidationResult]:
        """Validate statistical area hierarchy."""
        results = []
        
        # Check that boundary has appropriate parent relationships
        area_type = boundary.area_type
        
        # Define expected parent relationships
        expected_parents = {
            'SA1': ['SA2'],
            'SA2': ['SA3'],
            'SA3': ['SA4'],
            'SA4': ['STATE']
        }
        
        if area_type in expected_parents:
            for parent_type in expected_parents[area_type]:
                if parent_type not in boundary.parent_areas:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        rule_id="missing_parent_relationship",
                        message=f"{area_type} {boundary.area_code} missing {parent_type} parent"
                    ))
                    self._validation_stats['hierarchy_errors'] += 1
        
        # Validate area code format
        code_results = self._validate_area_code_format(boundary.area_code, area_type)
        results.extend(code_results)
        
        return results
    
    def _validate_area_code_format(self, area_code: str, area_type: str) -> List[ValidationResult]:
        """Validate area code format."""
        results = []
        
        # Expected code lengths
        expected_lengths = {
            'SA1': 11,  # e.g., 10102100101
            'SA2': 9,   # e.g., 101021007
            'SA3': 5,   # e.g., 10102
            'SA4': 3    # e.g., 101
        }
        
        if area_type in expected_lengths:
            expected_length = expected_lengths[area_type]
            
            if len(area_code) != expected_length:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="invalid_area_code_length",
                    message=f"{area_type} code {area_code} has length {len(area_code)}, expected {expected_length}"
                ))
                self._validation_stats['hierarchy_errors'] += 1
            
            if not area_code.isdigit():
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="non_numeric_area_code",
                    message=f"{area_type} code {area_code} contains non-numeric characters"
                ))
                self._validation_stats['hierarchy_errors'] += 1
        
        return results
    
    def _validate_topology(self, boundary: BoundaryRecord) -> List[ValidationResult]:
        """Validate topological properties."""
        results = []
        
        # Basic topology checks would go here
        # For now, we'll do basic checks on geometry info
        
        if boundary.geometry_info:
            geom_info = boundary.geometry_info
            
            # Check if area is reasonable for area type
            if geom_info.area_sqkm:
                area_ranges = {
                    'SA1': (0.1, 50),     # SA1s are typically small
                    'SA2': (1, 500),      # SA2s are medium sized
                    'SA3': (10, 5000),    # SA3s are larger
                    'SA4': (100, 50000)   # SA4s can be very large
                }
                
                if boundary.area_type in area_ranges:
                    min_area, max_area = area_ranges[boundary.area_type]
                    
                    if geom_info.area_sqkm < min_area:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            rule_id="area_too_small",
                            message=f"{boundary.area_type} {boundary.area_code} area {geom_info.area_sqkm} km² is unusually small"
                        ))
                    
                    if geom_info.area_sqkm > max_area:
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            rule_id="area_too_large",
                            message=f"{boundary.area_type} {boundary.area_code} area {geom_info.area_sqkm} km² is unusually large"
                        ))
        
        return results
    
    def _validate_boundary_group(self, area_type: str, boundaries: List[BoundaryRecord]) -> List[ValidationResult]:
        """Validate a group of boundaries of the same type."""
        results = []
        
        # Check for duplicate area codes
        area_codes = [b.area_code for b in boundaries]
        duplicates = set([code for code in area_codes if area_codes.count(code) > 1])
        
        if duplicates:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="duplicate_area_codes",
                message=f"Duplicate {area_type} codes found: {list(duplicates)}"
            ))
            self._validation_stats['topology_errors'] += 1
        
        return results
    
    def _validate_cross_hierarchy(self, boundary_groups: Dict[str, List[BoundaryRecord]]) -> List[ValidationResult]:
        """Validate relationships across hierarchy levels."""
        results = []
        
        # This would implement more complex cross-hierarchy validation
        # For now, just check that we have reasonable numbers for each level
        
        expected_counts = {
            'SA1': (57000, 58000),    # Approximately 57,523 SA1s
            'SA2': (2400, 2500),      # Approximately 2,473 SA2s
            'SA3': (300, 400),        # Approximately 351 SA3s
            'SA4': (80, 120)          # Approximately 108 SA4s
        }
        
        for area_type, boundaries in boundary_groups.items():
            if area_type in expected_counts:
                min_count, max_count = expected_counts[area_type]
                actual_count = len(boundaries)
                
                if actual_count < min_count or actual_count > max_count:
                    results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        rule_id="unexpected_boundary_count",
                        message=f"Unexpected {area_type} count: {actual_count} (expected {min_count}-{max_count})"
                    ))
        
        return results


class GeometrySimplifier:
    """
    Simplifier for optimising boundaries for web delivery.
    
    Reduces the complexity of boundary geometries while maintaining
    visual fidelity and topological integrity.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the geometry simplifier.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Simplification settings
        self.tolerance_degrees = config.get('tolerance_degrees', 0.0001)  # ~10 metres
        self.preserve_topology = config.get('preserve_topology', True)
        self.min_area_threshold = config.get('min_area_threshold', 0.1)  # km²
        
        # Output levels
        self.simplification_levels = config.get('simplification_levels', {
            'high_detail': 0.00001,    # ~1 metre
            'medium_detail': 0.0001,   # ~10 metres  
            'low_detail': 0.001,       # ~100 metres
            'overview': 0.01           # ~1 kilometre
        })
        
        # Statistics
        self._simplification_stats = {
            'boundaries_processed': 0,
            'total_points_before': 0,
            'total_points_after': 0,
            'compression_ratio': 0.0
        }
    
    def simplify_boundary(self, boundary: BoundaryRecord, detail_level: str = 'medium_detail') -> BoundaryRecord:
        """
        Simplify a boundary geometry.
        
        Args:
            boundary: Boundary record to simplify
            detail_level: Level of detail to maintain
            
        Returns:
            BoundaryRecord: Simplified boundary
        """
        if detail_level not in self.simplification_levels:
            raise ValueError(f"Unknown detail level: {detail_level}")
        
        tolerance = self.simplification_levels[detail_level]
        
        self.logger.debug(f"Simplifying {boundary.area_type} {boundary.area_code} with tolerance {tolerance}")
        
        # Count original points
        original_points = self._count_geometry_points(boundary.geometry)
        self._simplification_stats['total_points_before'] += original_points
        
        # Simplify geometry
        simplified_geometry = self._simplify_geometry(boundary.geometry, tolerance)
        
        # Count simplified points
        simplified_points = self._count_geometry_points(simplified_geometry)
        self._simplification_stats['total_points_after'] += simplified_points
        
        # Update statistics
        self._simplification_stats['boundaries_processed'] += 1
        
        # Calculate compression ratio
        if self._simplification_stats['total_points_before'] > 0:
            self._simplification_stats['compression_ratio'] = (
                self._simplification_stats['total_points_after'] / 
                self._simplification_stats['total_points_before']
            )
        
        # Create simplified boundary
        simplified_boundary = BoundaryRecord(
            area_code=boundary.area_code,
            area_type=boundary.area_type,
            area_name=boundary.area_name,
            state_code=boundary.state_code,
            geometry=simplified_geometry,
            geometry_info=self._calculate_geometry_info(simplified_geometry),
            parent_areas=boundary.parent_areas.copy(),
            child_areas=boundary.child_areas.copy()
        )
        
        self.logger.debug(
            f"Simplified {boundary.area_code}: {original_points} → {simplified_points} points "
            f"({simplified_points/original_points*100:.1f}% retained)"
        )
        
        return simplified_boundary
    
    def simplify_boundary_collection(
        self, 
        boundaries: List[BoundaryRecord], 
        detail_level: str = 'medium_detail'
    ) -> List[BoundaryRecord]:
        """
        Simplify a collection of boundaries.
        
        Args:
            boundaries: List of boundary records
            detail_level: Level of detail to maintain
            
        Returns:
            List[BoundaryRecord]: Simplified boundaries
        """
        simplified_boundaries = []
        
        for boundary in boundaries:
            try:
                simplified_boundary = self.simplify_boundary(boundary, detail_level)
                simplified_boundaries.append(simplified_boundary)
            except Exception as e:
                self.logger.error(f"Failed to simplify boundary {boundary.area_code}: {e}")
                # Include original boundary if simplification fails
                simplified_boundaries.append(boundary)
        
        return simplified_boundaries
    
    def get_simplification_statistics(self) -> Dict[str, Any]:
        """Get simplification statistics."""
        return self._simplification_stats.copy()
    
    def _simplify_geometry(self, geometry: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
        """Simplify a GeoJSON geometry."""
        geometry_type = geometry['type']
        
        if geometry_type == 'Polygon':
            return self._simplify_polygon(geometry, tolerance)
        elif geometry_type == 'MultiPolygon':
            return self._simplify_multipolygon(geometry, tolerance)
        else:
            # Return unchanged for unsupported types
            return geometry
    
    def _simplify_polygon(self, polygon: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
        """Simplify a polygon geometry."""
        simplified_rings = []
        
        for ring in polygon['coordinates']:
            simplified_ring = self._simplify_ring(ring, tolerance)
            if len(simplified_ring) >= 4:  # Must have at least 4 points
                simplified_rings.append(simplified_ring)
        
        return {
            'type': 'Polygon',
            'coordinates': simplified_rings
        }
    
    def _simplify_multipolygon(self, multipolygon: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
        """Simplify a multipolygon geometry."""
        simplified_polygons = []
        
        for polygon_coords in multipolygon['coordinates']:
            simplified_rings = []
            
            for ring in polygon_coords:
                simplified_ring = self._simplify_ring(ring, tolerance)
                if len(simplified_ring) >= 4:
                    simplified_rings.append(simplified_ring)
            
            if simplified_rings:
                simplified_polygons.append(simplified_rings)
        
        return {
            'type': 'MultiPolygon',
            'coordinates': simplified_polygons
        }
    
    def _simplify_ring(self, ring: List[List[float]], tolerance: float) -> List[List[float]]:
        """
        Simplify a coordinate ring using Douglas-Peucker algorithm.
        
        Args:
            ring: List of [lon, lat] coordinates
            tolerance: Simplification tolerance in degrees
            
        Returns:
            List[List[float]]: Simplified ring
        """
        if len(ring) <= 4:
            return ring
        
        # Simple implementation of Douglas-Peucker algorithm
        # For production use, would use a more sophisticated implementation
        
        simplified = self._douglas_peucker(ring[:-1], tolerance)  # Exclude last point (duplicate of first)
        simplified.append(simplified[0])  # Close the ring
        
        return simplified
    
    def _douglas_peucker(self, points: List[List[float]], tolerance: float) -> List[List[float]]:
        """Douglas-Peucker line simplification algorithm."""
        if len(points) <= 2:
            return points
        
        # Find the point with maximum distance from line between first and last points
        max_distance = 0
        max_index = 0
        
        start = points[0]
        end = points[-1]
        
        for i in range(1, len(points) - 1):
            distance = self._point_line_distance(points[i], start, end)
            if distance > max_distance:
                max_distance = distance
                max_index = i
        
        # If max distance is greater than tolerance, recursively simplify
        if max_distance > tolerance:
            # Recursive call for both segments
            left_segment = self._douglas_peucker(points[:max_index + 1], tolerance)
            right_segment = self._douglas_peucker(points[max_index:], tolerance)
            
            # Combine results (remove duplicate point at junction)
            return left_segment[:-1] + right_segment
        else:
            # All points can be removed except endpoints
            return [start, end]
    
    def _point_line_distance(self, point: List[float], line_start: List[float], line_end: List[float]) -> float:
        """Calculate perpendicular distance from point to line segment."""
        # Simple Euclidean distance calculation
        # For production use, would implement proper geodesic distance
        
        px, py = point[0], point[1]
        x1, y1 = line_start[0], line_start[1]
        x2, y2 = line_end[0], line_end[1]
        
        # Vector from line_start to line_end
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Line start and end are the same point
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
        
        # Parameter t for projection of point onto line
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        
        # Clamp t to [0, 1] to stay within line segment
        t = max(0, min(1, t))
        
        # Point on line closest to input point
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from input point to closest point on line
        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5
    
    def _count_geometry_points(self, geometry: Dict[str, Any]) -> int:
        """Count total number of coordinate points in geometry."""
        geometry_type = geometry['type']
        
        if geometry_type == 'Polygon':
            return sum(len(ring) for ring in geometry['coordinates'])
        elif geometry_type == 'MultiPolygon':
            return sum(
                sum(len(ring) for ring in polygon) 
                for polygon in geometry['coordinates']
            )
        else:
            return 0
    
    def _calculate_geometry_info(self, geometry: Dict[str, Any]) -> GeometryInfo:
        """Calculate basic geometry information."""
        coords = geometry['coordinates']
        geometry_type = geometry['type']
        
        # Count coordinates
        coord_count = self._count_geometry_points(geometry)
        
        # Calculate bounding box
        all_coords = []
        if geometry_type == 'Polygon':
            for ring in coords:
                all_coords.extend(ring)
        elif geometry_type == 'MultiPolygon':
            for polygon in coords:
                for ring in polygon:
                    all_coords.extend(ring)
        
        if all_coords:
            lons = [coord[0] for coord in all_coords]
            lats = [coord[1] for coord in all_coords]
            bbox = (min(lons), min(lats), max(lons), max(lats))
        else:
            bbox = (0, 0, 0, 0)
        
        return GeometryInfo(
            geometry_type=geometry_type,
            coordinate_count=coord_count,
            bbox=bbox,
            is_valid=True
        )


class SpatialIndexBuilder:
    """
    Builder for spatial indices to enable fast lookups.
    
    Creates spatial indices for statistical area boundaries to enable
    efficient point-in-polygon queries and spatial joins.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the spatial index builder.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Index settings
        self.index_type = config.get('index_type', 'rtree')  # rtree, quadtree, grid
        self.max_depth = config.get('max_depth', 10)
        self.max_items_per_node = config.get('max_items_per_node', 50)
        
        # Spatial indices by area type
        self._indices: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._index_stats = {
            'boundaries_indexed': 0,
            'index_build_time_seconds': 0,
            'index_memory_usage_mb': 0
        }
    
    def build_spatial_index(self, boundaries: List[BoundaryRecord]) -> Dict[str, Any]:
        """
        Build spatial index for boundary collection.
        
        Args:
            boundaries: List of boundary records
            
        Returns:
            Dict[str, Any]: Spatial index information
        """
        start_time = datetime.now()
        
        self.logger.info(f"Building spatial index for {len(boundaries)} boundaries")
        
        # Group boundaries by type
        boundary_groups = {}
        for boundary in boundaries:
            area_type = boundary.area_type
            if area_type not in boundary_groups:
                boundary_groups[area_type] = []
            boundary_groups[area_type].append(boundary)
        
        # Build index for each group
        for area_type, group_boundaries in boundary_groups.items():
            index_info = self._build_type_index(area_type, group_boundaries)
            self._indices[area_type] = index_info
        
        # Update statistics
        build_time = (datetime.now() - start_time).total_seconds()
        self._index_stats['boundaries_indexed'] = len(boundaries)
        self._index_stats['index_build_time_seconds'] = build_time
        
        self.logger.info(f"Spatial index built in {build_time:.2f} seconds")
        
        return {
            'total_boundaries': len(boundaries),
            'boundary_groups': {k: len(v) for k, v in boundary_groups.items()},
            'build_time_seconds': build_time,
            'index_types': list(self._indices.keys())
        }
    
    def query_point(self, longitude: float, latitude: float, area_type: str) -> List[str]:
        """
        Query spatial index for point intersection.
        
        Args:
            longitude: Point longitude
            latitude: Point latitude
            area_type: Statistical area type to query
            
        Returns:
            List[str]: List of area codes containing the point
        """
        if area_type not in self._indices:
            return []
        
        index_info = self._indices[area_type]
        bbox_index = index_info.get('bbox_index', {})
        
        # Simple bounding box intersection check
        candidates = []
        for area_code, bbox in bbox_index.items():
            min_lon, min_lat, max_lon, max_lat = bbox
            if min_lon <= longitude <= max_lon and min_lat <= latitude <= max_lat:
                candidates.append(area_code)
        
        # For production, would implement proper point-in-polygon test
        return candidates
    
    def query_bbox(
        self, 
        min_lon: float, 
        min_lat: float, 
        max_lon: float, 
        max_lat: float, 
        area_type: str
    ) -> List[str]:
        """
        Query spatial index for bounding box intersection.
        
        Args:
            min_lon: Minimum longitude
            min_lat: Minimum latitude
            max_lon: Maximum longitude
            max_lat: Maximum latitude
            area_type: Statistical area type to query
            
        Returns:
            List[str]: List of area codes intersecting the bounding box
        """
        if area_type not in self._indices:
            return []
        
        index_info = self._indices[area_type]
        bbox_index = index_info.get('bbox_index', {})
        
        # Bounding box intersection test
        intersecting = []
        for area_code, bbox in bbox_index.items():
            bbox_min_lon, bbox_min_lat, bbox_max_lon, bbox_max_lat = bbox
            
            # Check for intersection
            if not (max_lon < bbox_min_lon or min_lon > bbox_max_lon or 
                    max_lat < bbox_min_lat or min_lat > bbox_max_lat):
                intersecting.append(area_code)
        
        return intersecting
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """Get spatial index statistics."""
        stats = self._index_stats.copy()
        stats['index_types'] = list(self._indices.keys())
        
        # Add per-type statistics
        for area_type, index_info in self._indices.items():
            stats[f'{area_type}_count'] = index_info.get('boundary_count', 0)
        
        return stats
    
    def _build_type_index(self, area_type: str, boundaries: List[BoundaryRecord]) -> Dict[str, Any]:
        """Build spatial index for a specific area type."""
        self.logger.debug(f"Building index for {area_type}: {len(boundaries)} boundaries")
        
        # Simple bounding box index
        bbox_index = {}
        area_index = {}
        
        for boundary in boundaries:
            bbox = boundary.geometry_info.bbox
            bbox_index[boundary.area_code] = bbox
            area_index[boundary.area_code] = {
                'area_name': boundary.area_name,
                'state_code': boundary.state_code,
                'area_sqkm': boundary.geometry_info.area_sqkm,
                'parent_areas': boundary.parent_areas.copy()
            }
        
        return {
            'area_type': area_type,
            'boundary_count': len(boundaries),
            'bbox_index': bbox_index,
            'area_index': area_index,
            'index_type': self.index_type,
            'created_at': datetime.now()
        }


class BoundaryProcessor(BaseTransformer):
    """
    Main boundary processor for SA1, SA2, SA3, SA4 boundary files.
    
    Processes Australian Bureau of Statistics boundary files, validates
    topology, simplifies geometries, and builds spatial indices.
    """
    
    def __init__(
        self,
        transformer_id: str = "boundary_processor",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the boundary processor.
        
        Args:
            transformer_id: Unique identifier for this transformer
            config: Configuration dictionary
            logger: Optional logger instance
        """
        config = config or {}
        super().__init__(transformer_id, config, logger)
        
        # Component processors
        self.topology_validator = TopologyValidator(config, logger)
        self.geometry_simplifier = GeometrySimplifier(config, logger)
        self.spatial_indexer = SpatialIndexBuilder(config, logger)
        
        # Configuration
        self.enable_validation = config.get('enable_validation', True)
        self.enable_simplification = config.get('enable_simplification', True)
        self.enable_spatial_indexing = config.get('enable_spatial_indexing', True)
        self.simplification_level = config.get('simplification_level', 'medium_detail')
        
        # Input/output columns
        self.area_code_column = config.get('area_code_column', 'area_code')
        self.area_type_column = config.get('area_type_column', 'area_type')
        self.area_name_column = config.get('area_name_column', 'area_name')
        self.state_code_column = config.get('state_code_column', 'state_code')
        self.geometry_column = config.get('geometry_column', 'geometry')
        
        # Processing statistics
        self._processing_stats = {
            'boundaries_processed': 0,
            'validation_errors': 0,
            'simplification_ratio': 0.0,
            'spatial_index_built': False,
            'processing_time_seconds': 0
        }
    
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Process boundary data.
        
        Args:
            data: Batch of boundary data records
            **kwargs: Additional processing parameters
            
        Returns:
            DataBatch: Processed boundary data
            
        Raises:
            TransformationError: If processing fails
        """
        if not data:
            return data
        
        start_time = datetime.now()
        self.logger.info(f"Processing {len(data)} boundary records")
        
        # Convert to boundary records
        boundary_records = self._convert_to_boundary_records(data)
        
        # Validate boundaries if enabled
        if self.enable_validation:
            validation_results = self._validate_boundaries(boundary_records)
            self._processing_stats['validation_errors'] = len([
                r for r in validation_results if not r.is_valid and r.severity == ValidationSeverity.ERROR
            ])
        
        # Simplify geometries if enabled
        if self.enable_simplification:
            boundary_records = self._simplify_boundaries(boundary_records)
            simplification_stats = self.geometry_simplifier.get_simplification_statistics()
            self._processing_stats['simplification_ratio'] = simplification_stats.get('compression_ratio', 0.0)
        
        # Build spatial index if enabled
        if self.enable_spatial_indexing:
            index_info = self.spatial_indexer.build_spatial_index(boundary_records)
            self._processing_stats['spatial_index_built'] = True
        
        # Convert back to data records
        processed_data = self._convert_from_boundary_records(boundary_records)
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self._processing_stats['boundaries_processed'] = len(data)
        self._processing_stats['processing_time_seconds'] = processing_time
        
        self.logger.info(f"Boundary processing completed in {processing_time:.2f} seconds")
        
        return processed_data
    
    def get_schema(self) -> Dict[str, str]:
        """
        Get the expected output schema.
        
        Returns:
            Dict[str, str]: Schema definition
        """
        schema = {
            self.area_code_column: "string",
            self.area_type_column: "string",
            self.area_name_column: "string",
            self.state_code_column: "string",
            self.geometry_column: "object",  # GeoJSON geometry
            "geometry_info": "object",
            "validation_status": "string",
        }
        
        if self.enable_simplification:
            schema.update({
                "original_coordinate_count": "integer",
                "simplified_coordinate_count": "integer",
                "simplification_level": "string",
            })
        
        if self.enable_spatial_indexing:
            schema.update({
                "spatial_index_key": "string",
                "bounding_box": "object",
            })
        
        return schema
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics."""
        stats = self._processing_stats.copy()
        
        # Add component statistics
        if self.enable_validation:
            stats['validation_stats'] = self.topology_validator.get_validation_statistics()
        
        if self.enable_simplification:
            stats['simplification_stats'] = self.geometry_simplifier.get_simplification_statistics()
        
        if self.enable_spatial_indexing:
            stats['spatial_index_stats'] = self.spatial_indexer.get_index_statistics()
        
        return stats
    
    def _convert_to_boundary_records(self, data: DataBatch) -> List[BoundaryRecord]:
        """Convert data records to boundary records."""
        boundary_records = []
        
        for record in data:
            try:
                # Extract required fields
                area_code = str(record.get(self.area_code_column, ''))
                area_type = str(record.get(self.area_type_column, ''))
                area_name = str(record.get(self.area_name_column, ''))
                state_code = str(record.get(self.state_code_column, ''))
                geometry = record.get(self.geometry_column, {})
                
                if not area_code or not area_type:
                    self.logger.warning(f"Skipping record with missing area_code or area_type")
                    continue
                
                # Calculate geometry info
                geometry_info = self._calculate_geometry_info(geometry)
                
                # Extract parent relationships if available
                parent_areas = {}
                for parent_type in ['SA2', 'SA3', 'SA4', 'STATE']:
                    parent_key = f'{parent_type.lower()}_code'
                    if parent_key in record:
                        parent_areas[parent_type] = str(record[parent_key])
                
                boundary_record = BoundaryRecord(
                    area_code=area_code,
                    area_type=area_type,
                    area_name=area_name,
                    state_code=state_code,
                    geometry=geometry,
                    geometry_info=geometry_info,
                    parent_areas=parent_areas
                )
                
                boundary_records.append(boundary_record)
                
            except Exception as e:
                self.logger.error(f"Failed to convert record to boundary record: {e}")
                continue
        
        return boundary_records
    
    def _validate_boundaries(self, boundaries: List[BoundaryRecord]) -> List[ValidationResult]:
        """Validate boundary records."""
        all_validation_results = []
        
        # Validate individual boundaries
        for boundary in boundaries:
            results = self.topology_validator.validate_boundary(boundary)
            all_validation_results.extend(results)
        
        # Validate collection
        collection_results = self.topology_validator.validate_boundary_collection(boundaries)
        all_validation_results.extend(collection_results)
        
        # Log validation summary
        error_count = len([r for r in all_validation_results if not r.is_valid and r.severity == ValidationSeverity.ERROR])
        warning_count = len([r for r in all_validation_results if r.severity == ValidationSeverity.WARNING])
        
        self.logger.info(f"Validation completed: {error_count} errors, {warning_count} warnings")
        
        return all_validation_results
    
    def _simplify_boundaries(self, boundaries: List[BoundaryRecord]) -> List[BoundaryRecord]:
        """Simplify boundary geometries."""
        return self.geometry_simplifier.simplify_boundary_collection(boundaries, self.simplification_level)
    
    def _convert_from_boundary_records(self, boundaries: List[BoundaryRecord]) -> DataBatch:
        """Convert boundary records back to data records."""
        data_records = []
        
        for boundary in boundaries:
            record = {
                self.area_code_column: boundary.area_code,
                self.area_type_column: boundary.area_type,
                self.area_name_column: boundary.area_name,
                self.state_code_column: boundary.state_code,
                self.geometry_column: boundary.geometry,
                "geometry_info": boundary.geometry_info.to_dict() if hasattr(boundary.geometry_info, 'to_dict') else boundary.geometry_info.__dict__,
                "validation_status": "validated" if self.enable_validation else "not_validated",
            }
            
            # Add parent area relationships
            for parent_type, parent_code in boundary.parent_areas.items():
                record[f'{parent_type.lower()}_code'] = parent_code
            
            # Add processing metadata
            if self.enable_simplification:
                record.update({
                    "simplification_level": self.simplification_level,
                })
            
            data_records.append(record)
        
        return data_records
    
    def _calculate_geometry_info(self, geometry: Dict[str, Any]) -> GeometryInfo:
        """Calculate geometry information."""
        if not geometry or 'type' not in geometry:
            return GeometryInfo(
                geometry_type="unknown",
                coordinate_count=0,
                bbox=(0, 0, 0, 0),
                is_valid=False,
                validation_errors=["Missing or invalid geometry"]
            )
        
        try:
            geometry_type = geometry['type']
            
            # Count coordinates
            coord_count = self._count_geometry_points(geometry)
            
            # Calculate bounding box
            bbox = self._calculate_bbox(geometry)
            
            # Calculate area (simplified)
            area_sqkm = self._calculate_area(geometry)
            
            return GeometryInfo(
                geometry_type=geometry_type,
                coordinate_count=coord_count,
                bbox=bbox,
                area_sqkm=area_sqkm,
                is_valid=True
            )
            
        except Exception as e:
            return GeometryInfo(
                geometry_type="unknown",
                coordinate_count=0,
                bbox=(0, 0, 0, 0),
                is_valid=False,
                validation_errors=[str(e)]
            )
    
    def _count_geometry_points(self, geometry: Dict[str, Any]) -> int:
        """Count coordinate points in geometry."""
        geometry_type = geometry['type']
        
        if geometry_type == 'Polygon':
            return sum(len(ring) for ring in geometry['coordinates'])
        elif geometry_type == 'MultiPolygon':
            return sum(
                sum(len(ring) for ring in polygon) 
                for polygon in geometry['coordinates']
            )
        else:
            return 0
    
    def _calculate_bbox(self, geometry: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """Calculate bounding box of geometry."""
        all_coords = []
        
        if geometry['type'] == 'Polygon':
            for ring in geometry['coordinates']:
                all_coords.extend(ring)
        elif geometry['type'] == 'MultiPolygon':
            for polygon in geometry['coordinates']:
                for ring in polygon:
                    all_coords.extend(ring)
        
        if all_coords:
            lons = [coord[0] for coord in all_coords]
            lats = [coord[1] for coord in all_coords]
            return (min(lons), min(lats), max(lons), max(lats))
        else:
            return (0, 0, 0, 0)
    
    def _calculate_area(self, geometry: Dict[str, Any]) -> Optional[float]:
        """Calculate approximate area in square kilometres."""
        # This is a very simplified area calculation
        # For production use, would implement proper geodesic area calculation
        
        bbox = self._calculate_bbox(geometry)
        if bbox == (0, 0, 0, 0):
            return None
        
        min_lon, min_lat, max_lon, max_lat = bbox
        
        # Rough area calculation using bounding box
        # Convert degrees to kilometres (very approximate)
        lon_km = (max_lon - min_lon) * 111.32 * np.cos(np.radians((min_lat + max_lat) / 2))
        lat_km = (max_lat - min_lat) * 110.54
        
        return lon_km * lat_km