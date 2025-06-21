"""
Geographic data integration pipeline for AHGD project.

This module implements the geographic data integration workflow that combines
boundary data, coordinate systems, administrative hierarchies, and spatial
relationships into comprehensive geographic profiles.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import json

from ..base_pipeline import BasePipeline
from ..stage import Stage
from ...utils.integration_rules import DataIntegrationRules, ConflictResolver
from ...utils.interfaces import DataBatch, PipelineError
from ...utils.logging import get_logger, track_lineage
from ...schemas.base_schema import GeographicBoundary


class GeographicDataValidationStage(Stage):
    """Validates geographic data before integration."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("geographic_data_validation", config)
        self.coordinate_bounds = config.get('coordinate_bounds', {
            'australia': {
                'min_lat': -44.0,
                'max_lat': -10.0,
                'min_lon': 112.0,
                'max_lon': 154.0
            }
        })
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Validate geographic data inputs."""
        validated_data = []
        validation_errors = []
        
        for record in data:
            sa2_code = record.get('sa2_code')
            
            # Validate SA2 code format
            if not self._validate_sa2_code(sa2_code):
                validation_errors.append({
                    'sa2_code': sa2_code,
                    'error': 'invalid_sa2_code_format'
                })
                continue
            
            # Validate geographic hierarchy
            hierarchy_validation = self._validate_geographic_hierarchy(record)
            if hierarchy_validation['errors']:
                validation_errors.extend([
                    {'sa2_code': sa2_code, **error} for error in hierarchy_validation['errors']
                ])
            
            # Validate coordinates
            coordinate_validation = self._validate_coordinates(record)
            if coordinate_validation['errors']:
                validation_errors.extend([
                    {'sa2_code': sa2_code, **error} for error in coordinate_validation['errors']
                ])
            
            # Validate area calculations
            area_validation = self._validate_area_calculations(record)
            if area_validation['errors']:
                validation_errors.extend([
                    {'sa2_code': sa2_code, **error} for error in area_validation['errors']
                ])
            
            # Only include records that pass basic validation
            if not hierarchy_validation['errors']:
                validated_data.append(record)
        
        # Log validation results
        if validation_errors:
            self.logger.warning(f"Geographic validation found {len(validation_errors)} errors")
            for error in validation_errors[:3]:
                self.logger.warning(f"Validation error: {error}")
        
        self.logger.info(f"Geographic validation: {len(validated_data)} records validated")
        
        return validated_data
    
    def _validate_sa2_code(self, sa2_code: str) -> bool:
        """Validate SA2 code format."""
        if not sa2_code or not isinstance(sa2_code, str):
            return False
        
        # SA2 codes should be 9 digits
        if len(sa2_code) != 9 or not sa2_code.isdigit():
            return False
        
        # First digit should be valid state code (1-8)
        state_digit = int(sa2_code[0])
        if not 1 <= state_digit <= 8:
            return False
        
        return True
    
    def _validate_geographic_hierarchy(self, record: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Validate geographic hierarchy consistency."""
        errors = []
        
        sa2_code = record.get('sa2_code')
        geographic_hierarchy = record.get('geographic_hierarchy', {})
        
        if not geographic_hierarchy:
            errors.append({'error': 'missing_geographic_hierarchy'})
            return {'errors': errors}
        
        # Validate SA3 code
        sa3_code = geographic_hierarchy.get('sa3_code')
        if sa3_code:
            if len(sa3_code) != 5 or not sa3_code.isdigit():
                errors.append({
                    'error': 'invalid_sa3_code',
                    'sa3_code': sa3_code
                })
            elif sa2_code and not sa2_code.startswith(sa3_code):
                errors.append({
                    'error': 'sa2_sa3_hierarchy_mismatch',
                    'sa2_code': sa2_code,
                    'sa3_code': sa3_code
                })
        
        # Validate SA4 code
        sa4_code = geographic_hierarchy.get('sa4_code')
        if sa4_code:
            if len(sa4_code) != 3 or not sa4_code.isdigit():
                errors.append({
                    'error': 'invalid_sa4_code',
                    'sa4_code': sa4_code
                })
            elif sa3_code and not sa3_code.startswith(sa4_code):
                errors.append({
                    'error': 'sa3_sa4_hierarchy_mismatch',
                    'sa3_code': sa3_code,
                    'sa4_code': sa4_code
                })
        
        # Validate state code consistency
        state_code = geographic_hierarchy.get('state_code')
        if state_code and sa2_code:
            sa2_state_digit = sa2_code[0]
            state_mapping = {
                '1': 'NSW', '2': 'VIC', '3': 'QLD', '4': 'SA',
                '5': 'WA', '6': 'TAS', '7': 'NT', '8': 'ACT'
            }
            expected_state = state_mapping.get(sa2_state_digit)
            
            if expected_state and state_code != expected_state:
                errors.append({
                    'error': 'state_code_mismatch',
                    'expected_state': expected_state,
                    'actual_state': state_code
                })
        
        return {'errors': errors}
    
    def _validate_coordinates(self, record: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Validate coordinate data."""
        errors = []
        
        boundary_data = record.get('boundary_data', {})
        
        if isinstance(boundary_data, dict):
            centroid_lat = boundary_data.get('centroid_latitude')
            centroid_lon = boundary_data.get('centroid_longitude')
            
            # Validate centroid coordinates
            if centroid_lat is not None:
                if not isinstance(centroid_lat, (int, float)):
                    errors.append({
                        'error': 'invalid_latitude_type',
                        'latitude': centroid_lat
                    })
                elif not (-44.0 <= centroid_lat <= -10.0):
                    errors.append({
                        'error': 'latitude_out_of_bounds',
                        'latitude': centroid_lat
                    })
            
            if centroid_lon is not None:
                if not isinstance(centroid_lon, (int, float)):
                    errors.append({
                        'error': 'invalid_longitude_type',
                        'longitude': centroid_lon
                    })
                elif not (112.0 <= centroid_lon <= 154.0):
                    errors.append({
                        'error': 'longitude_out_of_bounds',
                        'longitude': centroid_lon
                    })
        
        return {'errors': errors}
    
    def _validate_area_calculations(self, record: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """Validate area and distance calculations."""
        errors = []
        
        boundary_data = record.get('boundary_data', {})
        
        if isinstance(boundary_data, dict):
            area_sq_km = boundary_data.get('area_sq_km')
            
            if area_sq_km is not None:
                if not isinstance(area_sq_km, (int, float)):
                    errors.append({
                        'error': 'invalid_area_type',
                        'area': area_sq_km
                    })
                elif area_sq_km <= 0:
                    errors.append({
                        'error': 'negative_or_zero_area',
                        'area': area_sq_km
                    })
                elif area_sq_km > 50000:  # Very large for SA2
                    errors.append({
                        'error': 'unrealistic_large_area',
                        'area': area_sq_km
                    })
        
        return {'errors': errors}


class BoundaryStandardisationStage(Stage):
    """Standardises boundary data from different sources."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("boundary_standardisation", config)
        self.target_coordinate_system = config.get('target_coordinate_system', 'GDA2020')
        self.simplification_tolerance = config.get('simplification_tolerance', 100)  # metres
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Standardise boundary data."""
        standardised_data = []
        
        for record in data:
            standardised_record = record.copy()
            
            # Standardise coordinate systems
            standardised_record = self._standardise_coordinate_system(standardised_record)
            
            # Standardise geometry format
            standardised_record = self._standardise_geometry_format(standardised_record)
            
            # Calculate derived geometric properties
            standardised_record = self._calculate_geometric_properties(standardised_record)
            
            standardised_data.append(standardised_record)
        
        self.logger.info(f"Standardised {len(standardised_data)} geographic records")
        return standardised_data
    
    def _standardise_coordinate_system(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure consistent coordinate reference system."""
        boundary_data = record.get('boundary_data', {})
        
        if isinstance(boundary_data, dict):
            # Set coordinate system if not specified
            if 'coordinate_system' not in boundary_data:
                boundary_data['coordinate_system'] = self.target_coordinate_system
            
            # Record transformation if needed
            current_crs = boundary_data.get('coordinate_system')
            if current_crs and current_crs != self.target_coordinate_system:
                boundary_data['coordinate_system_original'] = current_crs
                boundary_data['coordinate_system'] = self.target_coordinate_system
                boundary_data['coordinate_transformation_applied'] = True
        
        return record
    
    def _standardise_geometry_format(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Standardise geometry data format."""
        boundary_data = record.get('boundary_data', {})
        
        if isinstance(boundary_data, dict):
            geometry = boundary_data.get('geometry')
            
            if geometry:
                # Ensure geometry is in standard GeoJSON format
                if isinstance(geometry, str):
                    try:
                        # Try to parse as JSON
                        geometry_dict = json.loads(geometry)
                        boundary_data['geometry'] = geometry_dict
                    except json.JSONDecodeError:
                        # If not JSON, store as WKT reference
                        boundary_data['geometry_wkt'] = geometry
                        boundary_data['geometry'] = None
                
                # Validate GeoJSON structure
                elif isinstance(geometry, dict):
                    if 'type' not in geometry or 'coordinates' not in geometry:
                        self.logger.warning(f"Invalid geometry format for SA2 {record.get('sa2_code')}")
                        boundary_data['geometry_invalid'] = True
        
        return record
    
    def _calculate_geometric_properties(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived geometric properties."""
        boundary_data = record.get('boundary_data', {})
        
        if isinstance(boundary_data, dict):
            area_sq_km = boundary_data.get('area_sq_km', 0)
            
            # Calculate population density if population data is available
            total_population = record.get('total_population', 0)
            if area_sq_km > 0 and total_population >= 0:
                population_density = total_population / area_sq_km
                record['population_density_per_sq_km'] = population_density
            
            # Calculate compactness ratio if geometry is available
            geometry = boundary_data.get('geometry')
            if geometry and area_sq_km > 0:
                # Simplified compactness calculation
                # (would be more sophisticated with actual geometry processing)
                boundary_data['compactness_estimated'] = True
        
        return record


class SpatialRelationshipsStage(Stage):
    """Establishes spatial relationships between geographic areas."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("spatial_relationships", config)
        self.adjacency_tolerance = config.get('adjacency_tolerance', 10)  # metres
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Establish spatial relationships."""
        enhanced_data = []
        
        # Create SA2 lookup for relationship building
        sa2_lookup = {record.get('sa2_code'): record for record in data}
        
        for record in data:
            enhanced_record = record.copy()
            
            # Add hierarchical relationships
            enhanced_record = self._add_hierarchical_relationships(enhanced_record, sa2_lookup)
            
            # Add spatial classification
            enhanced_record = self._add_spatial_classifications(enhanced_record)
            
            # Add accessibility metrics
            enhanced_record = self._add_accessibility_metrics(enhanced_record)
            
            enhanced_data.append(enhanced_record)
        
        self.logger.info(f"Enhanced {len(enhanced_data)} records with spatial relationships")
        return enhanced_data
    
    def _add_hierarchical_relationships(
        self, 
        record: Dict[str, Any], 
        sa2_lookup: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add hierarchical geographic relationships."""
        geographic_hierarchy = record.get('geographic_hierarchy', {})
        
        # Add parent-child relationships
        sa3_code = geographic_hierarchy.get('sa3_code')
        if sa3_code:
            # Find other SA2s in the same SA3
            same_sa3_areas = []
            for other_sa2_code, other_record in sa2_lookup.items():
                other_hierarchy = other_record.get('geographic_hierarchy', {})
                if other_hierarchy.get('sa3_code') == sa3_code and other_sa2_code != record.get('sa2_code'):
                    same_sa3_areas.append(other_sa2_code)
            
            record['sa3_sibling_areas'] = same_sa3_areas
            record['sa3_sibling_count'] = len(same_sa3_areas)
        
        # Add state-level context
        state_code = geographic_hierarchy.get('state_code')
        if state_code:
            record['state_context'] = {
                'state_code': state_code,
                'is_capital_city_statistical_area': self._is_capital_city_area(record),
                'remoteness_context': self._get_remoteness_context(record)
            }
        
        return record
    
    def _add_spatial_classifications(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add spatial classification information."""
        # Urban/Rural classification based on population density
        population_density = record.get('population_density_per_sq_km', 0)
        
        if population_density >= 400:
            urbanisation = "major_urban"
        elif population_density >= 100:
            urbanisation = "other_urban"
        elif population_density >= 10:
            urbanisation = "bounded_locality"
        else:
            urbanisation = "rural_balance"
        
        record['urbanisation'] = urbanisation
        
        # Remoteness classification (simplified)
        # In reality, this would use proper ARIA calculations
        if population_density >= 400:
            remoteness_category = "Major Cities"
        elif population_density >= 100:
            remoteness_category = "Inner Regional"
        elif population_density >= 10:
            remoteness_category = "Outer Regional"
        else:
            remoteness_category = "Remote"
        
        record['remoteness_category'] = remoteness_category
        
        return record
    
    def _add_accessibility_metrics(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add accessibility and distance metrics."""
        boundary_data = record.get('boundary_data', {})
        
        if isinstance(boundary_data, dict):
            centroid_lat = boundary_data.get('centroid_latitude')
            centroid_lon = boundary_data.get('centroid_longitude')
            
            if centroid_lat and centroid_lon:
                # Calculate distances to major centres (simplified)
                distances = self._calculate_distances_to_major_centres(centroid_lat, centroid_lon)
                record['distances_to_major_centres'] = distances
                
                # Accessibility score based on distances
                accessibility_score = self._calculate_accessibility_score(distances)
                record['accessibility_score'] = accessibility_score
        
        return record
    
    def _is_capital_city_area(self, record: Dict[str, Any]) -> bool:
        """Determine if area is part of a capital city statistical area."""
        # Simplified implementation - would use actual CCSA boundaries
        sa4_code = record.get('geographic_hierarchy', {}).get('sa4_code')
        if not sa4_code:
            return False
        
        # Major capital city SA4 codes (simplified)
        capital_city_sa4s = [
            '101', '102', '103', '104', '105',  # Sydney
            '201', '202', '203', '204', '205',  # Melbourne
            '301', '302', '303',                # Brisbane
            '401', '402',                       # Adelaide
            '501', '502',                       # Perth
            '701',                              # Darwin
            '801'                               # Canberra
        ]
        
        return sa4_code in capital_city_sa4s
    
    def _get_remoteness_context(self, record: Dict[str, Any]) -> str:
        """Get remoteness context for the area."""
        remoteness_category = record.get('remoteness_category', 'Unknown')
        
        context_mapping = {
            'Major Cities': 'highly_accessible',
            'Inner Regional': 'accessible',
            'Outer Regional': 'moderately_accessible',
            'Remote': 'remote',
            'Very Remote': 'very_remote'
        }
        
        return context_mapping.get(remoteness_category, 'unknown')
    
    def _calculate_distances_to_major_centres(self, lat: float, lon: float) -> Dict[str, float]:
        """Calculate distances to major urban centres."""
        # Major city coordinates (simplified)
        major_centres = {
            'Sydney': (-33.8688, 151.2093),
            'Melbourne': (-37.8136, 144.9631),
            'Brisbane': (-27.4698, 153.0251),
            'Perth': (-31.9505, 115.8605),
            'Adelaide': (-34.9285, 138.6007),
            'Canberra': (-35.2809, 149.1300),
            'Darwin': (-12.4634, 130.8456),
            'Hobart': (-42.8821, 147.3272)
        }
        
        distances = {}
        for city, (city_lat, city_lon) in major_centres.items():
            # Simplified distance calculation (Euclidean)
            # In practice, would use great circle distance
            distance = ((lat - city_lat) ** 2 + (lon - city_lon) ** 2) ** 0.5 * 111  # Approx km per degree
            distances[f"distance_to_{city.lower()}_km"] = round(distance, 1)
        
        return distances
    
    def _calculate_accessibility_score(self, distances: Dict[str, float]) -> float:
        """Calculate overall accessibility score based on distances."""
        if not distances:
            return 0.0
        
        # Find minimum distance to any major centre
        min_distance = min(distances.values())
        
        # Calculate accessibility score (inverse of distance, normalised)
        if min_distance <= 50:
            return 100.0
        elif min_distance <= 200:
            return 100.0 - ((min_distance - 50) / 150) * 50  # Scale from 100 to 50
        else:
            return max(0.0, 50.0 - ((min_distance - 200) / 500) * 50)  # Scale from 50 to 0


class GeographicQualityAssessmentStage(Stage):
    """Assesses geographic data quality and completeness."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("geographic_quality_assessment", config)
        self.quality_thresholds = config.get('quality_thresholds', {})
        
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Assess geographic data quality."""
        quality_assessed_data = []
        
        for record in data:
            assessed_record = record.copy()
            
            # Calculate geographic completeness
            completeness_scores = self._calculate_geographic_completeness(record)
            assessed_record['geographic_completeness_scores'] = completeness_scores
            
            # Calculate spatial accuracy
            accuracy_scores = self._assess_spatial_accuracy(record)
            assessed_record['geographic_accuracy_scores'] = accuracy_scores
            
            # Overall geographic quality
            overall_quality = self._calculate_overall_geographic_quality(
                completeness_scores, accuracy_scores
            )
            assessed_record['geographic_overall_quality'] = overall_quality
            
            # Quality flags
            quality_flags = self._generate_geographic_quality_flags(record)
            assessed_record['geographic_quality_flags'] = quality_flags
            
            quality_assessed_data.append(assessed_record)
        
        self.logger.info(f"Assessed geographic quality for {len(quality_assessed_data)} records")
        return quality_assessed_data
    
    def _calculate_geographic_completeness(self, record: Dict[str, Any]) -> Dict[str, float]:
        """Calculate geographic data completeness scores."""
        completeness_scores = {}
        
        # Hierarchy completeness
        geographic_hierarchy = record.get('geographic_hierarchy', {})
        expected_hierarchy_fields = ['sa3_code', 'sa4_code', 'state_code']
        hierarchy_available = sum(
            1 for field in expected_hierarchy_fields 
            if geographic_hierarchy.get(field) is not None
        )
        completeness_scores['hierarchy'] = hierarchy_available / len(expected_hierarchy_fields)
        
        # Boundary data completeness
        boundary_data = record.get('boundary_data', {})
        expected_boundary_fields = ['centroid_latitude', 'centroid_longitude', 'area_sq_km']
        boundary_available = sum(
            1 for field in expected_boundary_fields
            if boundary_data.get(field) is not None
        )
        completeness_scores['boundary'] = boundary_available / len(expected_boundary_fields)
        
        # Geometry completeness
        geometry = boundary_data.get('geometry')
        if geometry and isinstance(geometry, dict) and 'coordinates' in geometry:
            completeness_scores['geometry'] = 1.0
        else:
            completeness_scores['geometry'] = 0.0
        
        # Classification completeness
        classification_fields = ['urbanisation', 'remoteness_category']
        classification_available = sum(
            1 for field in classification_fields
            if record.get(field) is not None
        )
        completeness_scores['classification'] = classification_available / len(classification_fields)
        
        return completeness_scores
    
    def _assess_spatial_accuracy(self, record: Dict[str, Any]) -> Dict[str, float]:
        """Assess spatial data accuracy."""
        accuracy_scores = {}
        
        # Coordinate accuracy (based on precision and bounds)
        boundary_data = record.get('boundary_data', {})
        centroid_lat = boundary_data.get('centroid_latitude')
        centroid_lon = boundary_data.get('centroid_longitude')
        
        if centroid_lat is not None and centroid_lon is not None:
            # Check coordinate precision (more decimal places = higher accuracy)
            lat_precision = len(str(centroid_lat).split('.')[-1]) if '.' in str(centroid_lat) else 0
            lon_precision = len(str(centroid_lon).split('.')[-1]) if '.' in str(centroid_lon) else 0
            
            # Score based on precision (6 decimal places = ~1m accuracy)
            coordinate_accuracy = min(1.0, (lat_precision + lon_precision) / 12)
            accuracy_scores['coordinates'] = coordinate_accuracy
        else:
            accuracy_scores['coordinates'] = 0.0
        
        # Area accuracy (based on reasonableness for SA2)
        area_sq_km = boundary_data.get('area_sq_km')
        if area_sq_km is not None:
            # SA2 areas typically range from 0.1 to 10,000 sq km
            if 0.1 <= area_sq_km <= 10000:
                accuracy_scores['area'] = 1.0
            else:
                accuracy_scores['area'] = 0.5  # Questionable but not impossible
        else:
            accuracy_scores['area'] = 0.0
        
        # Hierarchy consistency accuracy
        hierarchy_consistency = self._check_hierarchy_consistency(record)
        accuracy_scores['hierarchy_consistency'] = hierarchy_consistency
        
        return accuracy_scores
    
    def _check_hierarchy_consistency(self, record: Dict[str, Any]) -> float:
        """Check consistency of geographic hierarchy."""
        sa2_code = record.get('sa2_code')
        geographic_hierarchy = record.get('geographic_hierarchy', {})
        
        if not sa2_code or not geographic_hierarchy:
            return 0.0
        
        consistency_checks = []
        
        # SA2-SA3 consistency
        sa3_code = geographic_hierarchy.get('sa3_code')
        if sa3_code:
            if sa2_code.startswith(sa3_code):
                consistency_checks.append(1.0)
            else:
                consistency_checks.append(0.0)
        
        # SA3-SA4 consistency
        sa4_code = geographic_hierarchy.get('sa4_code')
        if sa3_code and sa4_code:
            if sa3_code.startswith(sa4_code):
                consistency_checks.append(1.0)
            else:
                consistency_checks.append(0.0)
        
        # State consistency
        state_code = geographic_hierarchy.get('state_code')
        if state_code:
            state_digit = sa2_code[0]
            state_mapping = {
                '1': 'NSW', '2': 'VIC', '3': 'QLD', '4': 'SA',
                '5': 'WA', '6': 'TAS', '7': 'NT', '8': 'ACT'
            }
            expected_state = state_mapping.get(state_digit)
            
            if expected_state == state_code:
                consistency_checks.append(1.0)
            else:
                consistency_checks.append(0.0)
        
        return sum(consistency_checks) / len(consistency_checks) if consistency_checks else 0.0
    
    def _calculate_overall_geographic_quality(
        self, 
        completeness_scores: Dict[str, float], 
        accuracy_scores: Dict[str, float]
    ) -> float:
        """Calculate overall geographic quality score."""
        # Weight completeness and accuracy equally
        completeness_avg = sum(completeness_scores.values()) / len(completeness_scores)
        accuracy_avg = sum(accuracy_scores.values()) / len(accuracy_scores)
        
        return (completeness_avg + accuracy_avg) / 2
    
    def _generate_geographic_quality_flags(self, record: Dict[str, Any]) -> List[str]:
        """Generate quality flags for geographic data."""
        flags = []
        
        # High accuracy coordinates
        boundary_data = record.get('boundary_data', {})
        centroid_lat = boundary_data.get('centroid_latitude')
        centroid_lon = boundary_data.get('centroid_longitude')
        
        if centroid_lat and centroid_lon:
            lat_precision = len(str(centroid_lat).split('.')[-1]) if '.' in str(centroid_lat) else 0
            if lat_precision >= 6:
                flags.append('high_precision_coordinates')
        
        # Complete hierarchy
        geographic_hierarchy = record.get('geographic_hierarchy', {})
        if all(geographic_hierarchy.get(field) for field in ['sa3_code', 'sa4_code', 'state_code']):
            flags.append('complete_hierarchy')
        
        # Valid geometry
        geometry = boundary_data.get('geometry')
        if geometry and isinstance(geometry, dict) and 'coordinates' in geometry:
            flags.append('valid_geometry')
        
        # Consistent classifications
        if record.get('urbanisation') and record.get('remoteness_category'):
            flags.append('complete_classifications')
        
        return flags


class GeographicIntegrationPipeline(BasePipeline):
    """
    Complete geographic data integration pipeline.
    
    Orchestrates the end-to-end geographic data integration process from
    validation through quality assessment.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the geographic integration pipeline.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger instance
        """
        super().__init__("geographic_integration_pipeline", config, logger)
        
        # Create pipeline stages
        self.stages = [
            GeographicDataValidationStage(config.get('validation', {})),
            BoundaryStandardisationStage(config.get('standardisation', {})),
            SpatialRelationshipsStage(config.get('spatial_relationships', {})),
            GeographicQualityAssessmentStage(config.get('quality_assessment', {}))
        ]
        
        # Integration tracking
        self.integration_metrics = {
            'records_processed': 0,
            'validation_errors': 0,
            'coordinate_accuracy': [],
            'hierarchy_completeness': []
        }
    
    def execute(self, data: DataBatch, **kwargs) -> DataBatch:
        """Execute the complete geographic integration pipeline."""
        self.logger.info("Starting geographic data integration pipeline")
        
        # Track lineage
        track_lineage(
            input_data="raw_geographic_data",
            output_data="integrated_geographic_data",
            transformation="geographic_integration_pipeline"
        )
        
        # Execute pipeline stages
        current_data = data
        
        for stage in self.stages:
            try:
                self.logger.info(f"Executing stage: {stage.name}")
                current_data = stage.execute(current_data, **kwargs)
                self.logger.info(f"Stage {stage.name} completed: {len(current_data)} records")
                
            except Exception as e:
                self.logger.error(f"Stage {stage.name} failed: {e}")
                raise PipelineError(f"Geographic integration pipeline failed at stage {stage.name}: {e}")
        
        # Update metrics
        self.integration_metrics['records_processed'] = len(current_data)
        
        # Calculate quality metrics
        coordinate_accuracy = []
        hierarchy_completeness = []
        
        for record in current_data:
            accuracy_scores = record.get('geographic_accuracy_scores', {})
            if 'coordinates' in accuracy_scores:
                coordinate_accuracy.append(accuracy_scores['coordinates'])
            
            completeness_scores = record.get('geographic_completeness_scores', {})
            if 'hierarchy' in completeness_scores:
                hierarchy_completeness.append(completeness_scores['hierarchy'])
        
        if coordinate_accuracy:
            avg_coord_accuracy = sum(coordinate_accuracy) / len(coordinate_accuracy)
            self.integration_metrics['coordinate_accuracy'].append(avg_coord_accuracy)
            self.logger.info(f"Average coordinate accuracy: {avg_coord_accuracy:.2f}")
        
        if hierarchy_completeness:
            avg_hierarchy_completeness = sum(hierarchy_completeness) / len(hierarchy_completeness)
            self.integration_metrics['hierarchy_completeness'].append(avg_hierarchy_completeness)
            self.logger.info(f"Average hierarchy completeness: {avg_hierarchy_completeness:.2f}")
        
        self.logger.info(f"Geographic integration pipeline completed: {len(current_data)} records integrated")
        
        return current_data
    
    def get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration pipeline metrics."""
        return self.integration_metrics.copy()
    
    def validate_pipeline_config(self) -> List[str]:
        """Validate pipeline configuration."""
        errors = []
        
        # Check required configuration sections
        required_sections = ['validation', 'standardisation', 'spatial_relationships', 'quality_assessment']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required configuration section: {section}")
        
        return errors