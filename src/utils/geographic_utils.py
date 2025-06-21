"""
Geographic utilities for the AHGD ETL pipeline.

This module provides Australian geographic constants, validation utilities,
and helper functions for working with Australian statistical areas and
coordinate systems.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
from datetime import datetime


@dataclass
class AustralianGeographicConstants:
    """
    Constants for Australian geographic system.
    
    Contains official definitions and boundaries for Australian states,
    territories, coordinate systems, and statistical areas.
    """
    
    # Australian states and territories
    STATES = {
        '1': {'name': 'New South Wales', 'abbreviation': 'NSW', 'capital': 'Sydney'},
        '2': {'name': 'Victoria', 'abbreviation': 'VIC', 'capital': 'Melbourne'},
        '3': {'name': 'Queensland', 'abbreviation': 'QLD', 'capital': 'Brisbane'},
        '4': {'name': 'South Australia', 'abbreviation': 'SA', 'capital': 'Adelaide'},
        '5': {'name': 'Western Australia', 'abbreviation': 'WA', 'capital': 'Perth'},
        '6': {'name': 'Tasmania', 'abbreviation': 'TAS', 'capital': 'Hobart'},
        '7': {'name': 'Northern Territory', 'abbreviation': 'NT', 'capital': 'Darwin'},
        '8': {'name': 'Australian Capital Territory', 'abbreviation': 'ACT', 'capital': 'Canberra'},
        '9': {'name': 'Other Territories', 'abbreviation': 'OT', 'capital': None}
    }
    
    # Geographic bounds for Australia and territories
    GEOGRAPHIC_BOUNDS = {
        'mainland': {
            'min_longitude': 112.5,
            'max_longitude': 154.0,
            'min_latitude': -39.5,
            'max_latitude': -10.0
        },
        'all_territories': {
            'min_longitude': 96.0,   # Cocos Islands
            'max_longitude': 169.0,  # Norfolk Island
            'min_latitude': -55.0,   # Macquarie Island
            'max_latitude': -9.0     # Torres Strait
        },
        'by_state': {
            'NSW': {'min_longitude': 140.999, 'max_longitude': 153.639, 'min_latitude': -37.505, 'max_latitude': -28.157},
            'VIC': {'min_longitude': 140.961, 'max_longitude': 149.976, 'min_latitude': -39.200, 'max_latitude': -33.980},
            'QLD': {'min_longitude': 137.994, 'max_longitude': 153.552, 'min_latitude': -29.178, 'max_latitude': -9.142},
            'SA': {'min_longitude': 129.002, 'max_longitude': 141.003, 'min_latitude': -38.062, 'max_latitude': -25.996},
            'WA': {'min_longitude': 112.921, 'max_longitude': 129.002, 'min_latitude': -35.134, 'max_latitude': -13.689},
            'TAS': {'min_longitude': 143.816, 'max_longitude': 148.477, 'min_latitude': -43.648, 'max_latitude': -39.573},
            'NT': {'min_longitude': 129.002, 'max_longitude': 138.001, 'min_latitude': -26.000, 'max_latitude': -10.962},
            'ACT': {'min_longitude': 148.760, 'max_longitude': 149.399, 'min_latitude': -35.921, 'max_latitude': -35.124}
        }
    }
    
    # Australian postcode ranges by state
    POSTCODE_RANGES = {
        'NSW': [(1000, 1999), (2000, 2599), (2619, 2899), (2921, 2999)],
        'ACT': [(200, 299), (2600, 2618), (2900, 2920)],
        'VIC': [(3000, 3999), (8000, 8999)],
        'QLD': [(4000, 4999), (9000, 9999)],
        'SA': [(5000, 5999)],
        'WA': [(6000, 6999)],
        'TAS': [(7000, 7999)],
        'NT': [(800, 999)]
    }
    
    # MGA (Map Grid of Australia) zones
    MGA_ZONES = {
        49: {'central_meridian': 117, 'states': ['WA'], 'bounds': (114, 120)},
        50: {'central_meridian': 123, 'states': ['WA'], 'bounds': (120, 126)},
        51: {'central_meridian': 129, 'states': ['WA', 'NT', 'SA'], 'bounds': (126, 132)},
        52: {'central_meridian': 135, 'states': ['NT', 'SA'], 'bounds': (132, 138)},
        53: {'central_meridian': 141, 'states': ['SA', 'VIC', 'NSW'], 'bounds': (138, 144)},
        54: {'central_meridian': 147, 'states': ['VIC', 'NSW', 'QLD', 'TAS'], 'bounds': (144, 150)},
        55: {'central_meridian': 153, 'states': ['NSW', 'QLD', 'TAS'], 'bounds': (150, 156)},
        56: {'central_meridian': 159, 'states': ['QLD', 'OT'], 'bounds': (156, 162)}
    }
    
    # Coordinate systems used in Australia
    COORDINATE_SYSTEMS = {
        'GDA2020': {
            'epsg_code': 7844,
            'datum': 'Geocentric Datum of Australia 2020',
            'ellipsoid': 'GRS 1980',
            'official_since': '2020-01-01',
            'authority': 'ICSM'
        },
        'GDA94': {
            'epsg_code': 4283,
            'datum': 'Geocentric Datum of Australia 1994',
            'ellipsoid': 'GRS 1980',
            'official_until': '2020-01-01',
            'authority': 'ICSM'
        },
        'AGD66': {
            'epsg_code': 4202,
            'datum': 'Australian Geodetic Datum 1966',
            'ellipsoid': 'Australian National Spheroid',
            'status': 'legacy',
            'authority': 'ICSM'
        },
        'AGD84': {
            'epsg_code': 4203,
            'datum': 'Australian Geodetic Datum 1984',
            'ellipsoid': 'Australian National Spheroid',
            'status': 'legacy',
            'authority': 'ICSM'
        }
    }
    
    # Statistical area hierarchy
    STATISTICAL_AREA_HIERARCHY = {
        'SA1': {
            'level': 1,
            'typical_count': 57523,
            'min_population': 200,
            'max_population': 800,
            'parent': None,
            'children': None,
            'code_format': r'^\d{11}$'
        },
        'SA2': {
            'level': 2,
            'typical_count': 2473,
            'min_population': 3000,
            'max_population': 25000,
            'parent': 'SA1',
            'children': 'SA1',
            'code_format': r'^\d{9}$'
        },
        'SA3': {
            'level': 3,
            'typical_count': 351,
            'min_population': 30000,
            'max_population': 130000,
            'parent': 'SA2',
            'children': 'SA2',
            'code_format': r'^\d{5}$'
        },
        'SA4': {
            'level': 4,
            'typical_count': 108,
            'min_population': 100000,
            'max_population': 500000,
            'parent': 'SA3',
            'children': 'SA3',
            'code_format': r'^\d{3}$'
        },
        'STATE': {
            'level': 5,
            'typical_count': 9,
            'parent': 'SA4',
            'children': 'SA4',
            'code_format': r'^\d{1}$'
        }
    }


class SA2HierarchyValidator:
    """
    Validator for SA1→SA2→SA3→SA4→State containment hierarchy.
    
    Ensures that statistical areas maintain proper hierarchical relationships
    as defined by the Australian Bureau of Statistics.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialise the hierarchy validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.constants = AustralianGeographicConstants()
        
        # Validation statistics
        self._validation_stats = {
            'validations_performed': 0,
            'hierarchy_violations': 0,
            'code_format_errors': 0,
            'containment_errors': 0
        }
    
    def validate_hierarchy_code(self, area_code: str, area_type: str) -> Tuple[bool, List[str]]:
        """
        Validate that an area code follows proper hierarchical format.
        
        Args:
            area_code: Statistical area code
            area_type: Type of statistical area (SA1, SA2, SA3, SA4, STATE)
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        self._validation_stats['validations_performed'] += 1
        errors = []
        
        if area_type not in self.constants.STATISTICAL_AREA_HIERARCHY:
            errors.append(f"Unknown area type: {area_type}")
            return False, errors
        
        hierarchy_info = self.constants.STATISTICAL_AREA_HIERARCHY[area_type]
        
        # Validate code format
        pattern = hierarchy_info['code_format']
        if not re.match(pattern, area_code):
            errors.append(f"{area_type} code {area_code} does not match expected format {pattern}")
            self._validation_stats['code_format_errors'] += 1
        
        # Validate hierarchical containment through code structure
        hierarchy_errors = self._validate_code_hierarchy(area_code, area_type)
        errors.extend(hierarchy_errors)
        
        if hierarchy_errors:
            self._validation_stats['hierarchy_violations'] += 1
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_containment(
        self, 
        child_code: str, 
        child_type: str, 
        parent_code: str, 
        parent_type: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a child area is properly contained within a parent area.
        
        Args:
            child_code: Child area code
            child_type: Child area type
            parent_code: Parent area code
            parent_type: Parent area type
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check hierarchy levels
        child_level = self.constants.STATISTICAL_AREA_HIERARCHY.get(child_type, {}).get('level', 0)
        parent_level = self.constants.STATISTICAL_AREA_HIERARCHY.get(parent_type, {}).get('level', 0)
        
        if child_level >= parent_level:
            errors.append(f"{child_type} (level {child_level}) cannot be contained in {parent_type} (level {parent_level})")
            self._validation_stats['containment_errors'] += 1
            return False, errors
        
        # Validate containment through code prefixes
        containment_valid = self._check_code_containment(child_code, child_type, parent_code, parent_type)
        
        if not containment_valid:
            errors.append(f"{child_type} {child_code} is not contained within {parent_type} {parent_code}")
            self._validation_stats['containment_errors'] += 1
        
        return containment_valid, errors
    
    def extract_parent_codes(self, area_code: str, area_type: str) -> Dict[str, str]:
        """
        Extract all parent area codes from a given area code.
        
        Args:
            area_code: Statistical area code
            area_type: Type of statistical area
            
        Returns:
            Dict[str, str]: Dictionary mapping parent types to parent codes
        """
        parent_codes = {}
        
        if area_type == 'SA1':
            # SA1: 10102100101 -> SA2: 101021001, SA3: 10102, SA4: 101, STATE: 1
            if len(area_code) == 11:
                parent_codes['SA2'] = area_code[:9]
                parent_codes['SA3'] = area_code[:5]
                parent_codes['SA4'] = area_code[:3]
                parent_codes['STATE'] = area_code[:1]
        
        elif area_type == 'SA2':
            # SA2: 101021001 -> SA3: 10102, SA4: 101, STATE: 1
            if len(area_code) == 9:
                parent_codes['SA3'] = area_code[:5]
                parent_codes['SA4'] = area_code[:3]
                parent_codes['STATE'] = area_code[:1]
        
        elif area_type == 'SA3':
            # SA3: 10102 -> SA4: 101, STATE: 1
            if len(area_code) == 5:
                parent_codes['SA4'] = area_code[:3]
                parent_codes['STATE'] = area_code[:1]
        
        elif area_type == 'SA4':
            # SA4: 101 -> STATE: 1
            if len(area_code) == 3:
                parent_codes['STATE'] = area_code[:1]
        
        return parent_codes
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self._validation_stats.copy()
    
    def _validate_code_hierarchy(self, area_code: str, area_type: str) -> List[str]:
        """Validate hierarchical structure within the code itself."""
        errors = []
        
        # Extract expected parent codes
        parent_codes = self.extract_parent_codes(area_code, area_type)
        
        # Validate each parent code format
        for parent_type, parent_code in parent_codes.items():
            parent_hierarchy = self.constants.STATISTICAL_AREA_HIERARCHY.get(parent_type, {})
            parent_pattern = parent_hierarchy.get('code_format', '')
            
            if parent_pattern and not re.match(parent_pattern, parent_code):
                errors.append(
                    f"Extracted {parent_type} code {parent_code} from {area_type} {area_code} "
                    f"does not match expected format {parent_pattern}"
                )
        
        # Validate state code is valid
        if 'STATE' in parent_codes:
            state_code = parent_codes['STATE']
            if state_code not in self.constants.STATES:
                errors.append(f"Invalid state code {state_code} extracted from {area_type} {area_code}")
        
        return errors
    
    def _check_code_containment(
        self, 
        child_code: str, 
        child_type: str, 
        parent_code: str, 
        parent_type: str
    ) -> bool:
        """Check if child code is properly contained within parent based on code structure."""
        
        # For Australian statistical areas, containment is determined by code prefixes
        if child_type == 'SA1' and parent_type == 'SA2':
            return len(child_code) == 11 and child_code.startswith(parent_code) and len(parent_code) == 9
        
        elif child_type == 'SA1' and parent_type == 'SA3':
            return len(child_code) == 11 and child_code.startswith(parent_code) and len(parent_code) == 5
        
        elif child_type == 'SA1' and parent_type == 'SA4':
            return len(child_code) == 11 and child_code.startswith(parent_code) and len(parent_code) == 3
        
        elif child_type == 'SA1' and parent_type == 'STATE':
            return len(child_code) == 11 and child_code.startswith(parent_code) and len(parent_code) == 1
        
        elif child_type == 'SA2' and parent_type == 'SA3':
            return len(child_code) == 9 and child_code.startswith(parent_code) and len(parent_code) == 5
        
        elif child_type == 'SA2' and parent_type == 'SA4':
            return len(child_code) == 9 and child_code.startswith(parent_code) and len(parent_code) == 3
        
        elif child_type == 'SA2' and parent_type == 'STATE':
            return len(child_code) == 9 and child_code.startswith(parent_code) and len(parent_code) == 1
        
        elif child_type == 'SA3' and parent_type == 'SA4':
            return len(child_code) == 5 and child_code.startswith(parent_code) and len(parent_code) == 3
        
        elif child_type == 'SA3' and parent_type == 'STATE':
            return len(child_code) == 5 and child_code.startswith(parent_code) and len(parent_code) == 1
        
        elif child_type == 'SA4' and parent_type == 'STATE':
            return len(child_code) == 3 and child_code.startswith(parent_code) and len(parent_code) == 1
        
        return False


class PostcodeValidator:
    """
    Validator for Australian postcodes by state.
    
    Validates postcode formats and ensures they belong to the correct
    Australian state or territory.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialise the postcode validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.constants = AustralianGeographicConstants()
        
        # Build postcode lookup
        self._postcode_to_state = self._build_postcode_lookup()
        
        # Validation statistics
        self._validation_stats = {
            'postcodes_validated': 0,
            'invalid_format': 0,
            'unknown_postcodes': 0,
            'state_mismatches': 0
        }
    
    def validate_postcode(self, postcode: Union[str, int], expected_state: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Validate an Australian postcode.
        
        Args:
            postcode: Postcode to validate
            expected_state: Expected state code (optional)
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        self._validation_stats['postcodes_validated'] += 1
        errors = []
        
        # Normalise postcode to string
        postcode_str = str(postcode).zfill(4)
        
        # Validate format
        if not self._validate_postcode_format(postcode_str):
            errors.append(f"Invalid postcode format: {postcode}")
            self._validation_stats['invalid_format'] += 1
            return False, errors
        
        # Check if postcode exists in any state
        postcode_int = int(postcode_str)
        actual_state = self._postcode_to_state.get(postcode_int)
        
        if actual_state is None:
            errors.append(f"Unknown Australian postcode: {postcode_str}")
            self._validation_stats['unknown_postcodes'] += 1
            return False, errors
        
        # Validate against expected state if provided
        if expected_state and expected_state != actual_state:
            errors.append(f"Postcode {postcode_str} belongs to {actual_state}, not {expected_state}")
            self._validation_stats['state_mismatches'] += 1
            return False, errors
        
        return True, []
    
    def get_postcode_state(self, postcode: Union[str, int]) -> Optional[str]:
        """
        Get the state for a given postcode.
        
        Args:
            postcode: Postcode to lookup
            
        Returns:
            Optional[str]: State abbreviation or None if not found
        """
        postcode_str = str(postcode).zfill(4)
        
        if not self._validate_postcode_format(postcode_str):
            return None
        
        postcode_int = int(postcode_str)
        return self._postcode_to_state.get(postcode_int)
    
    def validate_postcode_for_state(self, postcode: Union[str, int], state: str) -> bool:
        """
        Check if a postcode is valid for a specific state.
        
        Args:
            postcode: Postcode to validate
            state: State abbreviation
            
        Returns:
            bool: True if postcode is valid for the state
        """
        is_valid, _ = self.validate_postcode(postcode, state)
        return is_valid
    
    def get_state_postcode_ranges(self, state: str) -> List[Tuple[int, int]]:
        """
        Get postcode ranges for a specific state.
        
        Args:
            state: State abbreviation
            
        Returns:
            List[Tuple[int, int]]: List of (min_postcode, max_postcode) ranges
        """
        return self.constants.POSTCODE_RANGES.get(state, [])
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self._validation_stats.copy()
    
    def _validate_postcode_format(self, postcode: str) -> bool:
        """Validate postcode format (4 digits)."""
        return len(postcode) == 4 and postcode.isdigit()
    
    def _build_postcode_lookup(self) -> Dict[int, str]:
        """Build lookup table from postcode to state."""
        lookup = {}
        
        for state, ranges in self.constants.POSTCODE_RANGES.items():
            for min_code, max_code in ranges:
                for postcode in range(min_code, max_code + 1):
                    lookup[postcode] = state
        
        return lookup


class DistanceCalculator:
    """
    Calculator for distances using Australian map projections.
    
    Provides accurate distance calculations between geographic coordinates
    using appropriate projections for the Australian continent.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialise the distance calculator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.constants = AustralianGeographicConstants()
        
        # Earth radius in kilometres (WGS84/GRS80)
        self.EARTH_RADIUS_KM = 6378.137
        
        # Calculation statistics
        self._calculation_stats = {
            'distances_calculated': 0,
            'great_circle_calculations': 0,
            'haversine_calculations': 0,
            'projected_calculations': 0
        }
    
    def calculate_great_circle_distance(
        self, 
        lon1: float, 
        lat1: float, 
        lon2: float, 
        lat2: float
    ) -> float:
        """
        Calculate great circle distance between two points.
        
        Uses the haversine formula for accurate distance calculation
        on the Earth's surface.
        
        Args:
            lon1: Longitude of first point (decimal degrees)
            lat1: Latitude of first point (decimal degrees)
            lon2: Longitude of second point (decimal degrees)
            lat2: Latitude of second point (decimal degrees)
            
        Returns:
            float: Distance in kilometres
        """
        self._calculation_stats['distances_calculated'] += 1
        self._calculation_stats['great_circle_calculations'] += 1
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        # Haversine formula
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance_km = self.EARTH_RADIUS_KM * c
        
        return distance_km
    
    def calculate_haversine_distance(
        self, 
        lon1: float, 
        lat1: float, 
        lon2: float, 
        lat2: float
    ) -> float:
        """
        Calculate distance using haversine formula.
        
        Alternative implementation of great circle distance calculation.
        
        Args:
            lon1: Longitude of first point (decimal degrees)
            lat1: Latitude of first point (decimal degrees)
            lon2: Longitude of second point (decimal degrees)
            lat2: Latitude of second point (decimal degrees)
            
        Returns:
            float: Distance in kilometres
        """
        self._calculation_stats['distances_calculated'] += 1
        self._calculation_stats['haversine_calculations'] += 1
        
        # Convert to radians
        lon1_rad = math.radians(lon1)
        lat1_rad = math.radians(lat1)
        lon2_rad = math.radians(lon2)
        lat2_rad = math.radians(lat2)
        
        # Haversine calculation
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon / 2) ** 2)
        
        c = 2 * math.asin(math.sqrt(a))
        
        distance_km = self.EARTH_RADIUS_KM * c
        
        return distance_km
    
    def calculate_euclidean_distance(
        self, 
        x1: float, 
        y1: float, 
        x2: float, 
        y2: float,
        units: str = 'degrees'
    ) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Suitable for projected coordinates or small distances.
        
        Args:
            x1: X coordinate of first point
            y1: Y coordinate of first point
            x2: X coordinate of second point
            y2: Y coordinate of second point
            units: Units of input coordinates ('degrees', 'metres', 'kilometres')
            
        Returns:
            float: Distance in kilometres
        """
        self._calculation_stats['distances_calculated'] += 1
        self._calculation_stats['projected_calculations'] += 1
        
        # Calculate Euclidean distance
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx ** 2 + dy ** 2)
        
        # Convert to kilometres based on input units
        if units == 'degrees':
            # Rough conversion for degrees to kilometres (Australia)
            # More accurate near the centre of Australia
            avg_lat = (y1 + y2) / 2
            km_per_degree_lon = 111.32 * math.cos(math.radians(avg_lat))
            km_per_degree_lat = 110.54
            
            dx_km = dx * km_per_degree_lon
            dy_km = dy * km_per_degree_lat
            distance = math.sqrt(dx_km ** 2 + dy_km ** 2)
            
        elif units == 'metres':
            distance = distance / 1000.0
        elif units == 'kilometres':
            pass  # Already in kilometres
        else:
            raise ValueError(f"Unsupported units: {units}")
        
        return distance
    
    def calculate_bearing(
        self, 
        lon1: float, 
        lat1: float, 
        lon2: float, 
        lat2: float
    ) -> float:
        """
        Calculate bearing from point 1 to point 2.
        
        Args:
            lon1: Longitude of first point (decimal degrees)
            lat1: Latitude of first point (decimal degrees)
            lon2: Longitude of second point (decimal degrees)
            lat2: Latitude of second point (decimal degrees)
            
        Returns:
            float: Bearing in degrees (0-360)
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        # Calculate bearing
        y = math.sin(delta_lon) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalise to 0-360 degrees
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg
    
    def is_within_distance(
        self, 
        lon1: float, 
        lat1: float, 
        lon2: float, 
        lat2: float, 
        max_distance_km: float
    ) -> bool:
        """
        Check if two points are within a specified distance.
        
        Args:
            lon1: Longitude of first point (decimal degrees)
            lat1: Latitude of first point (decimal degrees)
            lon2: Longitude of second point (decimal degrees)
            lat2: Latitude of second point (decimal degrees)
            max_distance_km: Maximum distance in kilometres
            
        Returns:
            bool: True if points are within the specified distance
        """
        distance = self.calculate_great_circle_distance(lon1, lat1, lon2, lat2)
        return distance <= max_distance_km
    
    def find_mga_zone_for_coordinates(self, longitude: float, latitude: float) -> Optional[int]:
        """
        Find the appropriate MGA zone for given coordinates.
        
        Args:
            longitude: Longitude in decimal degrees
            latitude: Latitude in decimal degrees
            
        Returns:
            Optional[int]: MGA zone number or None if outside Australian range
        """
        # Check if coordinates are within Australian bounds
        bounds = self.constants.GEOGRAPHIC_BOUNDS['all_territories']
        
        if not (bounds['min_longitude'] <= longitude <= bounds['max_longitude'] and
                bounds['min_latitude'] <= latitude <= bounds['max_latitude']):
            return None
        
        # Find appropriate MGA zone based on longitude
        for zone, info in self.constants.MGA_ZONES.items():
            min_lon, max_lon = info['bounds']
            if min_lon <= longitude < max_lon:
                return zone
        
        # Handle edge case for easternmost coordinates
        if longitude >= 156:
            return 56
        
        return None
    
    def get_calculation_statistics(self) -> Dict[str, Any]:
        """Get calculation statistics."""
        return self._calculation_stats.copy()


def validate_australian_coordinates(longitude: float, latitude: float, include_territories: bool = True) -> Tuple[bool, List[str]]:
    """
    Validate that coordinates are within Australian bounds.
    
    Args:
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees
        include_territories: Whether to include external territories
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    errors = []
    constants = AustralianGeographicConstants()
    
    # Select appropriate bounds
    bounds_key = 'all_territories' if include_territories else 'mainland'
    bounds = constants.GEOGRAPHIC_BOUNDS[bounds_key]
    
    # Validate longitude
    if not (bounds['min_longitude'] <= longitude <= bounds['max_longitude']):
        errors.append(
            f"Longitude {longitude} outside Australian bounds "
            f"[{bounds['min_longitude']}, {bounds['max_longitude']}]"
        )
    
    # Validate latitude
    if not (bounds['min_latitude'] <= latitude <= bounds['max_latitude']):
        errors.append(
            f"Latitude {latitude} outside Australian bounds "
            f"[{bounds['min_latitude']}, {bounds['max_latitude']}]"
        )
    
    return len(errors) == 0, errors


def determine_state_from_coordinates(longitude: float, latitude: float) -> Optional[str]:
    """
    Determine the likely state from coordinates.
    
    Args:
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees
        
    Returns:
        Optional[str]: State abbreviation or None if cannot be determined
    """
    constants = AustralianGeographicConstants()
    
    # Check each state's bounds
    for state_code, state_info in constants.STATES.items():
        state_abbrev = state_info['abbreviation']
        
        if state_abbrev in constants.GEOGRAPHIC_BOUNDS['by_state']:
            bounds = constants.GEOGRAPHIC_BOUNDS['by_state'][state_abbrev]
            
            if (bounds['min_longitude'] <= longitude <= bounds['max_longitude'] and
                bounds['min_latitude'] <= latitude <= bounds['max_latitude']):
                return state_abbrev
    
    return None


def get_australian_time_zone(longitude: float, latitude: float) -> str:
    """
    Get the likely time zone for Australian coordinates.
    
    Args:
        longitude: Longitude in decimal degrees
        latitude: Latitude in decimal degrees
        
    Returns:
        str: Time zone identifier
    """
    state = determine_state_from_coordinates(longitude, latitude)
    
    # Australian time zones by state (simplified)
    time_zones = {
        'WA': 'Australia/Perth',          # UTC+8
        'SA': 'Australia/Adelaide',       # UTC+9:30
        'NT': 'Australia/Darwin',         # UTC+9:30
        'QLD': 'Australia/Brisbane',      # UTC+10
        'NSW': 'Australia/Sydney',        # UTC+10/11 (DST)
        'VIC': 'Australia/Melbourne',     # UTC+10/11 (DST)
        'TAS': 'Australia/Hobart',        # UTC+10/11 (DST)
        'ACT': 'Australia/Canberra',      # UTC+10/11 (DST)
    }
    
    return time_zones.get(state, 'Australia/Sydney')  # Default to Sydney