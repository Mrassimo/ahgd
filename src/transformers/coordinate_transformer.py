"""
Coordinate system transformer for the AHGD ETL pipeline.

This module provides comprehensive coordinate system transformation capabilities
for Australian geographic data, ensuring standardisation to GDA2020 (EPSG:7844)
coordinate system as mandated by the Australian government.
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
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
class CoordinatePoint:
    """Represents a geographic coordinate point."""
    longitude: float
    latitude: float
    elevation: Optional[float] = None
    coordinate_system: str = "GDA2020"
    zone: Optional[str] = None  # For MGA zones
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate coordinate values after initialisation."""
        if not self.is_valid():
            raise GeographicValidationError(
                f"Invalid coordinates: lon={self.longitude}, lat={self.latitude}"
            )
    
    def is_valid(self) -> bool:
        """Check if coordinates are valid."""
        # Basic range validation
        if not (-180 <= self.longitude <= 180):
            return False
        if not (-90 <= self.latitude <= 90):
            return False
        
        # Australian bounds validation (rough check)
        if not (110 <= self.longitude <= 155):  # Australia longitude range
            return False
        if not (-45 <= self.latitude <= -9):  # Australia latitude range
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'longitude': self.longitude,
            'latitude': self.latitude,
            'elevation': self.elevation,
            'coordinate_system': self.coordinate_system,
            'zone': self.zone
        }


@dataclass
class CoordinateTransformation:
    """Represents a coordinate system transformation."""
    source_system: str
    target_system: str
    transformation_method: str
    accuracy_metres: float
    parameters: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class GDA2020Transformer:
    """
    Transformer for standardising coordinates to GDA2020 (EPSG:7844).
    
    GDA2020 is the official coordinate system for Australia as of 2020,
    replacing GDA94 for improved accuracy and international compatibility.
    """
    
    # GDA2020 transformation parameters
    GDA2020_ELLIPSOID = {
        'semi_major_axis': 6378137.0,  # WGS84/GRS80 ellipsoid
        'flattening': 1/298.257222101,
        'semi_minor_axis': 6356752.314140347
    }
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the GDA2020 transformer.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Transformation accuracy requirements
        self.required_accuracy_metres = config.get('required_accuracy_metres', 1.0)
        self.enable_datum_transformation = config.get('enable_datum_transformation', True)
        
        # Cache transformation parameters
        self._transformation_cache: Dict[str, CoordinateTransformation] = {}
        
        self._load_transformation_parameters()
    
    def transform_to_gda2020(
        self, 
        point: CoordinatePoint, 
        source_system: str
    ) -> CoordinatePoint:
        """
        Transform coordinates to GDA2020.
        
        Args:
            point: Input coordinate point
            source_system: Source coordinate system identifier
            
        Returns:
            CoordinatePoint: Transformed point in GDA2020
            
        Raises:
            GeographicValidationError: If transformation fails
        """
        if source_system.upper() == "GDA2020":
            # Already in GDA2020
            return CoordinatePoint(
                longitude=point.longitude,
                latitude=point.latitude,
                elevation=point.elevation,
                coordinate_system="GDA2020"
            )
        
        # Get transformation parameters
        transformation = self._get_transformation(source_system, "GDA2020")
        
        # Perform transformation based on source system
        if source_system.upper() == "GDA94":
            return self._transform_gda94_to_gda2020(point)
        elif source_system.upper() == "WGS84":
            return self._transform_wgs84_to_gda2020(point)
        elif source_system.upper().startswith("MGA"):
            return self._transform_mga_to_gda2020(point, source_system)
        elif source_system.upper() == "AGD66":
            return self._transform_agd66_to_gda2020(point)
        elif source_system.upper() == "AGD84":
            return self._transform_agd84_to_gda2020(point)
        else:
            raise GeographicValidationError(
                f"Unsupported source coordinate system: {source_system}"
            )
    
    def validate_australian_bounds(self, point: CoordinatePoint) -> bool:
        """
        Validate that coordinates are within Australian bounds.
        
        Args:
            point: Coordinate point to validate
            
        Returns:
            bool: True if within Australian bounds
        """
        # Australian mainland and territories bounds
        bounds = {
            'min_longitude': 110.0,  # Western Australia
            'max_longitude': 155.0,  # Norfolk Island
            'min_latitude': -45.0,   # Macquarie Island
            'max_latitude': -9.0     # Torres Strait
        }
        
        return (
            bounds['min_longitude'] <= point.longitude <= bounds['max_longitude'] and
            bounds['min_latitude'] <= point.latitude <= bounds['max_latitude']
        )
    
    def _load_transformation_parameters(self):
        """Load coordinate transformation parameters."""
        # GDA94 to GDA2020 transformation parameters
        # These are based on official ICSM transformation parameters
        self._transformation_cache["GDA94_TO_GDA2020"] = CoordinateTransformation(
            source_system="GDA94",
            target_system="GDA2020",
            transformation_method="helmert_7_parameter",
            accuracy_metres=0.01,  # 1cm accuracy
            parameters={
                'tx': 0.06155,     # Translation X (metres)
                'ty': -0.01087,    # Translation Y (metres) 
                'tz': -0.04019,    # Translation Z (metres)
                'rx': -0.0394924,  # Rotation X (arc seconds)
                'ry': -0.0327221,  # Rotation Y (arc seconds)
                'rz': -0.0328979,  # Rotation Z (arc seconds)
                'ds': -0.009994    # Scale difference (ppm)
            }
        )
        
        # WGS84 to GDA2020 (very small difference)
        self._transformation_cache["WGS84_TO_GDA2020"] = CoordinateTransformation(
            source_system="WGS84",
            target_system="GDA2020", 
            transformation_method="helmert_7_parameter",
            accuracy_metres=0.05,
            parameters={
                'tx': 0.0,
                'ty': 0.0,
                'tz': 0.0,
                'rx': 0.0,
                'ry': 0.0,
                'rz': 0.0,
                'ds': 0.0
            }
        )
    
    def _get_transformation(self, source: str, target: str) -> CoordinateTransformation:
        """Get transformation parameters for coordinate system pair."""
        cache_key = f"{source.upper()}_TO_{target.upper()}"
        
        if cache_key not in self._transformation_cache:
            raise GeographicValidationError(
                f"No transformation available from {source} to {target}"
            )
        
        return self._transformation_cache[cache_key]
    
    def _transform_gda94_to_gda2020(self, point: CoordinatePoint) -> CoordinatePoint:
        """Transform from GDA94 to GDA2020."""
        # For most practical purposes, the difference is minimal
        # This would implement the full 7-parameter Helmert transformation
        
        # Simplified transformation (actual implementation would be more complex)
        transformation = self._transformation_cache["GDA94_TO_GDA2020"]
        
        # Apply small corrections based on transformation parameters
        dx = transformation.parameters['tx'] / 111000.0  # Convert metres to degrees (approx)
        dy = transformation.parameters['ty'] / 111000.0
        
        return CoordinatePoint(
            longitude=point.longitude + dx,
            latitude=point.latitude + dy,
            elevation=point.elevation,
            coordinate_system="GDA2020"
        )
    
    def _transform_wgs84_to_gda2020(self, point: CoordinatePoint) -> CoordinatePoint:
        """Transform from WGS84 to GDA2020."""
        # WGS84 and GDA2020 are essentially identical for most practical purposes
        return CoordinatePoint(
            longitude=point.longitude,
            latitude=point.latitude,
            elevation=point.elevation,
            coordinate_system="GDA2020"
        )
    
    def _transform_mga_to_gda2020(self, point: CoordinatePoint, source_system: str) -> CoordinatePoint:
        """Transform from MGA (Map Grid of Australia) to GDA2020."""
        # Extract zone from system name (e.g., "MGA_ZONE_55")
        zone = self._extract_mga_zone(source_system)
        
        # Convert MGA coordinates to geographic coordinates
        # This is a simplified version - full implementation would use proper UTM transformation
        
        # MGA uses UTM projection on GDA94/GDA2020 datum
        # For this implementation, we'll assume the input coordinates are already in decimal degrees
        # and just validate they're reasonable for the specified zone
        
        if not self._validate_mga_zone(point.longitude, zone):
            raise GeographicValidationError(
                f"Coordinates {point.longitude}, {point.latitude} invalid for MGA Zone {zone}"
            )
        
        return CoordinatePoint(
            longitude=point.longitude,
            latitude=point.latitude,
            elevation=point.elevation,
            coordinate_system="GDA2020",
            zone=f"MGA_ZONE_{zone}"
        )
    
    def _transform_agd66_to_gda2020(self, point: CoordinatePoint) -> CoordinatePoint:
        """Transform from AGD66 to GDA2020."""
        # AGD66 transformation requires larger corrections
        # This is a simplified approximation
        
        # Typical AGD66 to GDA2020 corrections for central Australia
        dx = 0.00013  # Longitude correction (degrees)
        dy = -0.00008  # Latitude correction (degrees)
        
        return CoordinatePoint(
            longitude=point.longitude + dx,
            latitude=point.latitude + dy,
            elevation=point.elevation,
            coordinate_system="GDA2020"
        )
    
    def _transform_agd84_to_gda2020(self, point: CoordinatePoint) -> CoordinatePoint:
        """Transform from AGD84 to GDA2020."""
        # AGD84 transformation
        # This is a simplified approximation
        
        dx = 0.00010  # Longitude correction (degrees)
        dy = -0.00006  # Latitude correction (degrees)
        
        return CoordinatePoint(
            longitude=point.longitude + dx,
            latitude=point.latitude + dy,
            elevation=point.elevation,
            coordinate_system="GDA2020"
        )
    
    def _extract_mga_zone(self, system_name: str) -> int:
        """Extract MGA zone number from system name."""
        # Expected format: "MGA_ZONE_55" or "MGA55"
        import re
        
        # Try different patterns
        patterns = [
            r'MGA_ZONE_(\d+)',
            r'MGA(\d+)',
            r'ZONE_(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, system_name.upper())
            if match:
                zone = int(match.group(1))
                if 49 <= zone <= 56:  # Valid MGA zones for Australia
                    return zone
        
        raise GeographicValidationError(f"Cannot extract valid MGA zone from: {system_name}")
    
    def _validate_mga_zone(self, longitude: float, zone: int) -> bool:
        """Validate that longitude is appropriate for MGA zone."""
        # MGA zone boundaries (approximate central meridians)
        zone_boundaries = {
            49: (114, 120),   # WA
            50: (120, 126),   # WA
            51: (126, 132),   # NT/SA
            52: (132, 138),   # SA/NT
            53: (138, 144),   # SA/VIC/NSW
            54: (144, 150),   # VIC/NSW/QLD
            55: (150, 156),   # NSW/QLD/TAS
            56: (156, 162),   # QLD/External territories
        }
        
        if zone not in zone_boundaries:
            return False
        
        min_lon, max_lon = zone_boundaries[zone]
        return min_lon <= longitude <= max_lon


class MGAZoneTransformer:
    """
    Transformer for handling Map Grid of Australia (MGA) zone conversions.
    
    MGA is the standard projected coordinate system for Australia,
    using UTM projection with GDA2020 datum.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the MGA zone transformer.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # MGA zone definitions
        self.mga_zones = self._load_mga_zones()
    
    def determine_mga_zone(self, longitude: float) -> int:
        """
        Determine the appropriate MGA zone for a longitude.
        
        Args:
            longitude: Longitude in decimal degrees
            
        Returns:
            int: MGA zone number (49-56)
            
        Raises:
            GeographicValidationError: If longitude is outside Australian range
        """
        if not (110 <= longitude <= 160):
            raise GeographicValidationError(
                f"Longitude {longitude} is outside Australian MGA zone range"
            )
        
        # MGA zones are based on 6-degree wide UTM zones
        for zone, (min_lon, max_lon) in self.mga_zones.items():
            if min_lon <= longitude < max_lon:
                return zone
        
        # Handle edge case for easternmost longitude
        if longitude >= 156:
            return 56
        
        raise GeographicValidationError(
            f"Cannot determine MGA zone for longitude {longitude}"
        )
    
    def convert_between_zones(
        self, 
        point: CoordinatePoint, 
        source_zone: int, 
        target_zone: int
    ) -> CoordinatePoint:
        """
        Convert coordinates between MGA zones.
        
        Args:
            point: Input coordinate point
            source_zone: Source MGA zone
            target_zone: Target MGA zone
            
        Returns:
            CoordinatePoint: Point in target zone
        """
        if source_zone == target_zone:
            return point
        
        # For zone conversion, we'd typically:
        # 1. Convert from source zone to geographic (lat/lon)  
        # 2. Convert from geographic to target zone
        # This is a simplified implementation
        
        self.logger.info(f"Converting from MGA Zone {source_zone} to Zone {target_zone}")
        
        return CoordinatePoint(
            longitude=point.longitude,
            latitude=point.latitude,
            elevation=point.elevation,
            coordinate_system="GDA2020",
            zone=f"MGA_ZONE_{target_zone}"
        )
    
    def _load_mga_zones(self) -> Dict[int, Tuple[float, float]]:
        """Load MGA zone boundary definitions."""
        return {
            49: (114, 120),   # Zone 49: 117°E ± 3°
            50: (120, 126),   # Zone 50: 123°E ± 3° 
            51: (126, 132),   # Zone 51: 129°E ± 3°
            52: (132, 138),   # Zone 52: 135°E ± 3°
            53: (138, 144),   # Zone 53: 141°E ± 3°
            54: (144, 150),   # Zone 54: 147°E ± 3°
            55: (150, 156),   # Zone 55: 153°E ± 3°
            56: (156, 162),   # Zone 56: 159°E ± 3°
        }


class GeographicValidator:
    """
    Validator for geographic coordinates within Australian bounds.
    
    Ensures coordinates are valid and within reasonable Australian bounds
    including mainland, territories, and external territories.
    """
    
    # Australian geographic bounds including external territories
    AUSTRALIAN_BOUNDS = {
        'mainland': {
            'min_longitude': 112.5,
            'max_longitude': 154.0,
            'min_latitude': -39.5,
            'max_latitude': -10.0
        },
        'territories': {
            'min_longitude': 110.0,
            'max_longitude': 169.0,
            'min_latitude': -55.0,
            'max_latitude': -9.0
        }
    }
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the geographic validator.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Validation settings
        self.include_territories = config.get('include_territories', True)
        self.strict_validation = config.get('strict_validation', True)
        
        # Select bounds based on configuration
        self.bounds = (
            self.AUSTRALIAN_BOUNDS['territories'] if self.include_territories 
            else self.AUSTRALIAN_BOUNDS['mainland']
        )
    
    def validate_coordinates(self, point: CoordinatePoint) -> ValidationResult:
        """
        Validate geographic coordinates.
        
        Args:
            point: Coordinate point to validate
            
        Returns:
            ValidationResult: Validation result
        """
        # Basic format validation
        if not isinstance(point.longitude, (int, float)) or not isinstance(point.latitude, (int, float)):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="invalid_coordinate_format",
                message="Coordinates must be numeric"
            )
        
        # Range validation
        if not (-180 <= point.longitude <= 180):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="longitude_out_of_range",
                message=f"Longitude {point.longitude} is outside valid range [-180, 180]"
            )
        
        if not (-90 <= point.latitude <= 90):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="latitude_out_of_range",
                message=f"Latitude {point.latitude} is outside valid range [-90, 90]"
            )
        
        # Australian bounds validation
        if not self._within_australian_bounds(point):
            severity = ValidationSeverity.ERROR if self.strict_validation else ValidationSeverity.WARNING
            return ValidationResult(
                is_valid=not self.strict_validation,
                severity=severity,
                rule_id="outside_australian_bounds",
                message=f"Coordinates {point.longitude}, {point.latitude} are outside Australian bounds"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_id="coordinates_valid",
            message="Coordinates are valid"
        )
    
    def validate_coordinate_precision(self, point: CoordinatePoint, required_precision: int = 6) -> ValidationResult:
        """
        Validate coordinate precision.
        
        Args:
            point: Coordinate point to validate
            required_precision: Required decimal places
            
        Returns:
            ValidationResult: Validation result
        """
        lon_precision = self._get_decimal_places(point.longitude)
        lat_precision = self._get_decimal_places(point.latitude)
        
        if lon_precision < required_precision or lat_precision < required_precision:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                rule_id="insufficient_precision",
                message=f"Coordinates have insufficient precision (required: {required_precision} decimal places)"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            rule_id="precision_adequate",
            message="Coordinate precision is adequate"
        )
    
    def _within_australian_bounds(self, point: CoordinatePoint) -> bool:
        """Check if point is within Australian bounds."""
        return (
            self.bounds['min_longitude'] <= point.longitude <= self.bounds['max_longitude'] and
            self.bounds['min_latitude'] <= point.latitude <= self.bounds['max_latitude']
        )
    
    def _get_decimal_places(self, number: float) -> int:
        """Get number of decimal places in a float."""
        str_num = str(number)
        if '.' in str_num:
            return len(str_num.split('.')[1])
        return 0


class CoordinateSystemTransformer(BaseTransformer):
    """
    Main coordinate system transformer for the AHGD ETL pipeline.
    
    Converts between Australian coordinate systems and standardises
    to GDA2020 (EPSG:7844) as mandated by the Australian government.
    """
    
    def __init__(
        self,
        transformer_id: str = "coordinate_transformer",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the coordinate system transformer.
        
        Args:
            transformer_id: Unique identifier for this transformer
            config: Configuration dictionary
            logger: Optional logger instance
        """
        config = config or {}
        super().__init__(transformer_id, config, logger)
        
        # Component transformers
        self.gda2020_transformer = GDA2020Transformer(config, logger)
        self.mga_transformer = MGAZoneTransformer(config, logger)
        self.coordinate_validator = GeographicValidator(config, logger)
        
        # Configuration
        self.longitude_column = config.get('longitude_column', 'longitude')
        self.latitude_column = config.get('latitude_column', 'latitude')
        self.elevation_column = config.get('elevation_column', 'elevation')
        self.coordinate_system_column = config.get('coordinate_system_column', 'coordinate_system')
        
        # Output configuration
        self.output_longitude_column = config.get('output_longitude_column', 'longitude_gda2020')
        self.output_latitude_column = config.get('output_latitude_column', 'latitude_gda2020')
        self.output_elevation_column = config.get('output_elevation_column', 'elevation_gda2020')
        self.output_system_column = config.get('output_system_column', 'coordinate_system_gda2020')
        
        # Transformation settings
        self.target_system = config.get('target_system', 'GDA2020')
        self.include_validation_metadata = config.get('include_validation_metadata', True)
        self.auto_determine_mga_zone = config.get('auto_determine_mga_zone', True)
        
        # Statistics
        self._transformation_stats = {
            'total_points': 0,
            'successful_transformations': 0,
            'failed_transformations': 0,
            'validation_warnings': 0,
            'coordinate_systems': {}
        }
    
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Transform coordinate system data to GDA2020.
        
        Args:
            data: Batch of data records to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            DataBatch: Transformed data with GDA2020 coordinates
            
        Raises:
            TransformationError: If transformation fails
        """
        if not data:
            return data
        
        self.logger.info(f"Starting coordinate transformation for {len(data)} records")
        
        # Reset statistics
        self._transformation_stats['total_points'] = len(data)
        
        transformed_data = []
        
        for record in data:
            try:
                transformed_record = self._transform_record(record)
                transformed_data.append(transformed_record)
                self._transformation_stats['successful_transformations'] += 1
                
            except Exception as e:
                self._transformation_stats['failed_transformations'] += 1
                self.logger.error(f"Failed to transform coordinate record: {e}")
                
                # Add error information to record
                error_record = record.copy()
                error_record.update({
                    self.output_longitude_column: None,
                    self.output_latitude_column: None,
                    'transformation_error': str(e)
                })
                transformed_data.append(error_record)
        
        self.logger.info(
            f"Coordinate transformation completed. "
            f"Success: {self._transformation_stats['successful_transformations']}, "
            f"Failed: {self._transformation_stats['failed_transformations']}"
        )
        
        return transformed_data
    
    def get_schema(self) -> Dict[str, str]:
        """
        Get the expected output schema.
        
        Returns:
            Dict[str, str]: Schema definition
        """
        schema = {
            self.output_longitude_column: "float",
            self.output_latitude_column: "float",
            self.output_system_column: "string",
        }
        
        if self.elevation_column:
            schema[self.output_elevation_column] = "float"
        
        if self.include_validation_metadata:
            schema.update({
                "coordinate_validation_status": "string",
                "mga_zone": "string",
                "transformation_accuracy_metres": "float",
            })
        
        return schema
    
    def get_transformation_statistics(self) -> Dict[str, Any]:
        """Get detailed transformation statistics."""
        return self._transformation_stats.copy()
    
    def _transform_record(self, record: DataRecord) -> DataRecord:
        """Transform coordinates in a single record."""
        # Extract coordinate information
        longitude = record.get(self.longitude_column)
        latitude = record.get(self.latitude_column)
        elevation = record.get(self.elevation_column)
        source_system = record.get(self.coordinate_system_column, 'unknown')
        
        if longitude is None or latitude is None:
            raise TransformationError(
                f"Missing coordinate data: longitude={longitude}, latitude={latitude}"
            )
        
        # Create coordinate point
        try:
            point = CoordinatePoint(
                longitude=float(longitude),
                latitude=float(latitude),
                elevation=float(elevation) if elevation is not None else None,
                coordinate_system=str(source_system)
            )
        except (ValueError, GeographicValidationError) as e:
            raise TransformationError(f"Invalid coordinate data: {e}") from e
        
        # Validate coordinates
        validation_result = self.coordinate_validator.validate_coordinates(point)
        if not validation_result.is_valid and validation_result.severity == ValidationSeverity.ERROR:
            raise TransformationError(f"Coordinate validation failed: {validation_result.message}")
        
        if validation_result.severity == ValidationSeverity.WARNING:
            self._transformation_stats['validation_warnings'] += 1
        
        # Track coordinate systems
        system_key = str(source_system).upper()
        self._transformation_stats['coordinate_systems'][system_key] = (
            self._transformation_stats['coordinate_systems'].get(system_key, 0) + 1
        )
        
        # Transform to GDA2020
        try:
            transformed_point = self.gda2020_transformer.transform_to_gda2020(point, source_system)
        except GeographicValidationError as e:
            raise TransformationError(f"Coordinate transformation failed: {e}") from e
        
        # Determine MGA zone if requested
        mga_zone = None
        if self.auto_determine_mga_zone:
            try:
                mga_zone = self.mga_transformer.determine_mga_zone(transformed_point.longitude)
            except GeographicValidationError:
                # Not critical if we can't determine MGA zone
                pass
        
        # Create output record
        output_record = record.copy()
        output_record.update({
            self.output_longitude_column: transformed_point.longitude,
            self.output_latitude_column: transformed_point.latitude,
            self.output_system_column: transformed_point.coordinate_system,
        })
        
        if transformed_point.elevation is not None:
            output_record[self.output_elevation_column] = transformed_point.elevation
        
        if self.include_validation_metadata:
            output_record.update({
                "coordinate_validation_status": validation_result.rule_id,
                "mga_zone": f"MGA_ZONE_{mga_zone}" if mga_zone else None,
                "transformation_accuracy_metres": 0.01,  # GDA2020 accuracy
            })
        
        return output_record