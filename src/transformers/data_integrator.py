"""
Data integration classes for AHGD project.

This module implements comprehensive data integration functionality to create
MasterHealthRecord instances by combining multiple data sources at SA2 level
with data quality assessment and conflict resolution.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from .base import BaseTransformer, MissingValueStrategy
from schemas.integrated_schema import (
    MasterHealthRecord, 
    SA2HealthProfile,
    DataIntegrationLevel,
    UrbanRuralClassification
)
from schemas.base_schema import GeographicBoundary, TemporalData, DataSource
from schemas.seifa_schema import SEIFAIndexType
from ..utils.interfaces import DataBatch, TransformationError
from ..utils.logging import get_logger


@dataclass
class DataSourceRecord:
    """Container for data from a specific source with quality metadata."""
    
    source_name: str
    source_priority: int
    data: Dict[str, Any]
    quality_score: float
    last_updated: datetime
    coverage_score: float = 1.0  # 0-1 indicating data coverage completeness
    reliability_score: float = 1.0  # 0-1 indicating data reliability
    
    @property
    def overall_quality(self) -> float:
        """Calculate overall quality score combining all quality metrics."""
        return (self.quality_score * 0.4 + 
                self.coverage_score * 0.3 + 
                self.reliability_score * 0.3)


@dataclass
class IntegrationDecision:
    """Record of a data integration decision for audit trail."""
    
    field_name: str
    selected_source: str
    selected_value: Any
    alternative_sources: Dict[str, Any]
    decision_reason: str
    confidence_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MasterDataIntegrator(BaseTransformer):
    """
    Main class that creates MasterHealthRecord instances from multiple data sources.
    
    Combines data from census, health indicators, SEIFA, geographic boundaries,
    and other sources into comprehensive health records with full audit trail.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the master data integrator.
        
        Args:
            config: Configuration including data source priorities and rules
            logger: Optional logger instance
        """
        super().__init__("master_data_integrator", config, logger)
        
        self.logger = get_logger(__name__) if logger is None else logger
        
        # Data source configuration
        self.source_priorities = config.get('source_priorities', {})
        self.quality_thresholds = config.get('quality_thresholds', {})
        self.integration_rules = config.get('integration_rules', {})
        
        # Integration components
        self.sa2_aggregator = SA2DataAggregator(config.get('aggregation', {}), logger)
        self.indicator_calculator = HealthIndicatorCalculator(config.get('indicators', {}), logger)
        self.demographic_builder = DemographicProfileBuilder(config.get('demographics', {}), logger)
        self.quality_calculator = QualityScoreCalculator(config.get('quality', {}), logger)
        
        # Integration tracking
        self.integration_decisions: List[IntegrationDecision] = []
        self.data_conflicts: List[Dict[str, Any]] = []
        
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Transform input data into MasterHealthRecord instances.
        
        Args:
            data: Batch of raw data records grouped by SA2
            **kwargs: Additional parameters including SA2 codes
            
        Returns:
            DataBatch: List of MasterHealthRecord instances
        """
        try:
            master_records = []
            
            for record_data in data:
                sa2_code = record_data.get('sa2_code')
                if not sa2_code:
                    self.logger.warning("Skipping record without SA2 code")
                    continue
                
                # Create master health record for this SA2
                master_record = self.create_master_health_record(sa2_code, record_data)
                if master_record:
                    master_records.append(master_record.__dict__)
            
            self.logger.info(f"Created {len(master_records)} master health records")
            return master_records
            
        except Exception as e:
            self.logger.error(f"Master data integration failed: {e}")
            raise TransformationError(f"Master data integration failed: {e}") from e
    
    def create_master_health_record(
        self, 
        sa2_code: str, 
        source_data: Dict[str, Any]
    ) -> Optional[MasterHealthRecord]:
        """
        Create a complete MasterHealthRecord for a specific SA2.
        
        Args:
            sa2_code: 9-digit SA2 code
            source_data: Raw data from multiple sources for this SA2
            
        Returns:
            MasterHealthRecord: Complete integrated health record
        """
        try:
            self.logger.debug(f"Creating master health record for SA2: {sa2_code}")
            
            # Extract and organise data by source
            source_records = self._organise_source_data(sa2_code, source_data)
            
            # Build geographic components
            geographic_data = self._integrate_geographic_data(sa2_code, source_records)
            
            # Build demographic profile
            demographic_data = self.demographic_builder.build_profile(sa2_code, source_records)
            
            # Calculate health indicators
            health_data = self.indicator_calculator.calculate_health_indicators(sa2_code, source_records)
            
            # Calculate quality metrics
            quality_data = self.quality_calculator.calculate_quality_scores(source_records)
            
            # Create the master record
            master_record = MasterHealthRecord(
                # Primary identification
                sa2_code=sa2_code,
                sa2_name=self._select_best_value('sa2_name', source_records),
                
                # Geographic dimensions
                geographic_hierarchy=geographic_data['hierarchy'],
                boundary_data=geographic_data['boundary'],
                urbanisation=self._determine_urbanisation(geographic_data),
                remoteness_category=geographic_data.get('remoteness_category', 'Unknown'),
                
                # Demographic profile
                demographic_profile=demographic_data['profile'],
                total_population=demographic_data['total_population'],
                population_density_per_sq_km=demographic_data['population_density'],
                median_age=demographic_data.get('median_age'),
                
                # Socioeconomic indicators
                seifa_scores=self._extract_seifa_scores(source_records),
                seifa_deciles=self._extract_seifa_deciles(source_records),
                disadvantage_category=self._determine_disadvantage_category(source_records),
                median_household_income=health_data.get('median_household_income'),
                unemployment_rate=health_data.get('unemployment_rate'),
                
                # Health indicators summary
                health_outcomes_summary=health_data['outcomes_summary'],
                life_expectancy=health_data.get('life_expectancy'),
                self_assessed_health=health_data.get('self_assessed_health'),
                
                # Mortality indicators
                mortality_indicators=health_data.get('mortality_indicators', {}),
                avoidable_mortality_rate=health_data.get('avoidable_mortality_rate'),
                infant_mortality_rate=health_data.get('infant_mortality_rate'),
                
                # Disease prevalence
                chronic_disease_prevalence=health_data.get('chronic_disease_prevalence', {}),
                
                # Mental health
                mental_health_indicators=health_data.get('mental_health_indicators', {}),
                psychological_distress_high=health_data.get('psychological_distress_high'),
                
                # Healthcare access and utilisation
                healthcare_access=health_data.get('healthcare_access', {}),
                gp_services_per_1000=health_data.get('gp_services_per_1000'),
                specialist_services_per_1000=health_data.get('specialist_services_per_1000'),
                bulk_billing_rate=health_data.get('bulk_billing_rate'),
                emergency_dept_presentations_per_1000=health_data.get('emergency_dept_presentations_per_1000'),
                
                # Pharmaceutical utilisation
                pharmaceutical_utilisation=health_data.get('pharmaceutical_utilisation', {}),
                
                # Risk factors
                risk_factors=health_data.get('risk_factors', {}),
                smoking_prevalence=health_data.get('smoking_prevalence'),
                obesity_prevalence=health_data.get('obesity_prevalence'),
                physical_inactivity_prevalence=health_data.get('physical_inactivity_prevalence'),
                harmful_alcohol_use_prevalence=health_data.get('harmful_alcohol_use_prevalence'),
                
                # Environmental factors
                environmental_indicators=health_data.get('environmental_indicators', {}),
                air_quality_index=health_data.get('air_quality_index'),
                green_space_access=health_data.get('green_space_access'),
                
                # Data integration metadata
                integration_level=self._determine_integration_level(source_records),
                data_completeness_score=quality_data['completeness_score'],
                integration_timestamp=datetime.utcnow(),
                source_datasets=self._extract_source_datasets(source_records),
                missing_indicators=quality_data['missing_indicators'],
                
                # Derived indicators
                composite_health_index=self._calculate_composite_health_index(health_data),
                health_inequality_index=self._calculate_health_inequality_index(health_data, demographic_data),
                
                # Schema version
                schema_version="2.0.0"
            )
            
            # Record integration decisions for audit
            self._record_integration_audit(sa2_code, source_records, master_record)
            
            return master_record
            
        except Exception as e:
            self.logger.error(f"Failed to create master health record for {sa2_code}: {e}")
            return None
    
    def get_schema(self) -> Dict[str, str]:
        """Get the output schema definition."""
        return {
            'sa2_code': 'string',
            'sa2_name': 'string',
            'total_population': 'integer',
            'life_expectancy': 'float',
            'seifa_irsad_score': 'integer',
            'data_completeness_score': 'float',
            'integration_timestamp': 'datetime'
        }
    
    def _organise_source_data(
        self, 
        sa2_code: str, 
        raw_data: Dict[str, Any]
    ) -> List[DataSourceRecord]:
        """
        Organise raw data into structured source records with quality metrics.
        
        Args:
            sa2_code: SA2 identifier
            raw_data: Raw data dictionary from multiple sources
            
        Returns:
            List[DataSourceRecord]: Organised source data with quality metrics
        """
        source_records = []
        
        # Expected source types and their priorities
        source_types = {
            'census': 1,
            'seifa': 2,
            'health_indicators': 3,
            'geographic_boundaries': 4,
            'medicare_pbs': 5,
            'environmental': 6
        }
        
        for source_type, priority in source_types.items():
            if source_type in raw_data:
                source_data = raw_data[source_type]
                
                # Calculate quality scores
                quality_score = self._assess_data_quality(source_data, source_type)
                coverage_score = self._assess_coverage(source_data, source_type)
                reliability_score = self.source_priorities.get(source_type, {}).get('reliability', 1.0)
                
                source_record = DataSourceRecord(
                    source_name=source_type,
                    source_priority=priority,
                    data=source_data,
                    quality_score=quality_score,
                    last_updated=source_data.get('last_updated', datetime.utcnow()),
                    coverage_score=coverage_score,
                    reliability_score=reliability_score
                )
                
                source_records.append(source_record)
        
        # Sort by overall quality (highest first)
        source_records.sort(key=lambda x: x.overall_quality, reverse=True)
        
        return source_records
    
    def _integrate_geographic_data(
        self, 
        sa2_code: str, 
        source_records: List[DataSourceRecord]
    ) -> Dict[str, Any]:
        """
        Integrate geographic data from multiple sources.
        
        Args:
            sa2_code: SA2 identifier
            source_records: List of source data records
            
        Returns:
            Dict containing integrated geographic data
        """
        geographic_data = {}
        
        # Extract geographic hierarchy
        hierarchy = {}
        for record in source_records:
            if 'geographic' in record.source_name or 'census' in record.source_name:
                data = record.data
                if 'sa3_code' in data:
                    hierarchy['sa3_code'] = data['sa3_code']
                if 'sa4_code' in data:
                    hierarchy['sa4_code'] = data['sa4_code']
                if 'state_code' in data:
                    hierarchy['state_code'] = data['state_code']
                if hierarchy:  # Found geographic data
                    break
        
        geographic_data['hierarchy'] = hierarchy
        
        # Create boundary data
        boundary_info = self._select_best_geographic_boundary(source_records)
        geographic_data['boundary'] = GeographicBoundary(
            geometry=boundary_info.get('geometry', {}),
            centroid_latitude=boundary_info.get('centroid_lat', 0.0),
            centroid_longitude=boundary_info.get('centroid_lon', 0.0),
            area_sq_km=boundary_info.get('area_sqkm', 0.0),
            boundary_source=boundary_info.get('source', 'unknown'),
            last_updated=boundary_info.get('last_updated', datetime.utcnow())
        )
        
        # Add remoteness category
        geographic_data['remoteness_category'] = self._select_best_value('remoteness_category', source_records)
        
        return geographic_data
    
    def _select_best_value(self, field_name: str, source_records: List[DataSourceRecord]) -> Any:
        """
        Select the best value for a field from multiple sources using quality-based selection.
        
        Args:
            field_name: Name of the field to select
            source_records: List of source data records
            
        Returns:
            Best value for the field, or None if not found
        """
        candidates = []
        
        for record in source_records:
            if field_name in record.data and record.data[field_name] is not None:
                candidates.append((record.data[field_name], record.overall_quality, record.source_name))
        
        if not candidates:
            return None
        
        # Sort by quality score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        selected_value = candidates[0][0]
        selected_source = candidates[0][2]
        
        # Record decision for audit
        alternatives = {source: value for value, _, source in candidates[1:]}
        decision = IntegrationDecision(
            field_name=field_name,
            selected_source=selected_source,
            selected_value=selected_value,
            alternative_sources=alternatives,
            decision_reason="highest_quality_score",
            confidence_score=candidates[0][1]
        )
        self.integration_decisions.append(decision)
        
        return selected_value
    
    def _extract_seifa_scores(self, source_records: List[DataSourceRecord]) -> Dict[SEIFAIndexType, float]:
        """Extract SEIFA scores from source data."""
        seifa_scores = {}
        
        for record in source_records:
            if 'seifa' in record.source_name:
                data = record.data
                for index_type in SEIFAIndexType:
                    score_field = f"seifa_{index_type.value}_score"
                    if score_field in data:
                        seifa_scores[index_type] = float(data[score_field])
        
        return seifa_scores
    
    def _extract_seifa_deciles(self, source_records: List[DataSourceRecord]) -> Dict[SEIFAIndexType, int]:
        """Extract SEIFA deciles from source data."""
        seifa_deciles = {}
        
        for record in source_records:
            if 'seifa' in record.source_name:
                data = record.data
                for index_type in SEIFAIndexType:
                    decile_field = f"seifa_{index_type.value}_decile"
                    if decile_field in data:
                        seifa_deciles[index_type] = int(data[decile_field])
        
        return seifa_deciles
    
    def _determine_urbanisation(self, geographic_data: Dict[str, Any]) -> UrbanRuralClassification:
        """Determine urbanisation classification from geographic data."""
        # This would use actual classification logic based on population density,
        # distance to urban centres, etc. For now, return a default
        return UrbanRuralClassification.MAJOR_URBAN
    
    def _determine_disadvantage_category(self, source_records: List[DataSourceRecord]) -> str:
        """Determine overall disadvantage category from SEIFA scores."""
        # Extract IRSD decile (most commonly used for disadvantage)
        irsd_decile = None
        for record in source_records:
            if 'seifa' in record.source_name and 'seifa_irsd_decile' in record.data:
                irsd_decile = record.data['seifa_irsd_decile']
                break
        
        if irsd_decile is None:
            return "Unknown"
        
        if irsd_decile <= 3:
            return "Most Disadvantaged"
        elif irsd_decile <= 5:
            return "Disadvantaged"
        elif irsd_decile <= 7:
            return "Average"
        elif irsd_decile <= 9:
            return "Advantaged"
        else:
            return "Most Advantaged"
    
    def _determine_integration_level(self, source_records: List[DataSourceRecord]) -> DataIntegrationLevel:
        """Determine the level of data integration achieved."""
        source_count = len(source_records)
        
        if source_count >= 5:
            return DataIntegrationLevel.COMPREHENSIVE
        elif source_count >= 3:
            return DataIntegrationLevel.STANDARD
        elif source_count >= 2:
            return DataIntegrationLevel.MINIMAL
        else:
            return DataIntegrationLevel.MINIMAL
    
    def _extract_source_datasets(self, source_records: List[DataSourceRecord]) -> List[str]:
        """Extract list of source dataset names."""
        return [record.source_name for record in source_records]
    
    def _calculate_composite_health_index(self, health_data: Dict[str, Any]) -> Optional[float]:
        """Calculate a composite health index score from multiple indicators."""
        # This would implement a weighted scoring algorithm
        # For now, return a placeholder based on life expectancy
        life_expectancy = health_data.get('life_expectancy')
        if life_expectancy:
            # Simple scoring: normalize against Australian average of ~83 years
            return min(100.0, (float(life_expectancy) / 83.0) * 100.0)
        return None
    
    def _calculate_health_inequality_index(
        self, 
        health_data: Dict[str, Any], 
        demographic_data: Dict[str, Any]
    ) -> Optional[float]:
        """Calculate health inequality index relative to national average."""
        # This would implement inequality measurement
        # For now, return a placeholder
        return 0.25  # Placeholder value
    
    def _select_best_geographic_boundary(self, source_records: List[DataSourceRecord]) -> Dict[str, Any]:
        """Select the best geographic boundary data from available sources."""
        for record in source_records:
            if 'geographic' in record.source_name and 'geometry' in record.data:
                return record.data
        
        # Fallback to any source with geographic data
        for record in source_records:
            if any(field in record.data for field in ['centroid_lat', 'centroid_lon', 'area_sqkm']):
                return record.data
        
        return {}
    
    def _assess_data_quality(self, data: Dict[str, Any], source_type: str) -> float:
        """Assess the quality of data from a specific source."""
        # Implement quality assessment logic
        # For now, return based on completeness
        if not data:
            return 0.0
        
        # Count non-null values
        non_null_count = sum(1 for value in data.values() if value is not None)
        total_count = len(data)
        
        return non_null_count / total_count if total_count > 0 else 0.0
    
    def _assess_coverage(self, data: Dict[str, Any], source_type: str) -> float:
        """Assess the coverage completeness of data from a source."""
        # Implement coverage assessment logic
        # For now, return a fixed score based on source type
        coverage_scores = {
            'census': 0.95,
            'seifa': 0.98,
            'health_indicators': 0.85,
            'geographic_boundaries': 0.99,
            'medicare_pbs': 0.80,
            'environmental': 0.70
        }
        return coverage_scores.get(source_type, 0.5)
    
    def _record_integration_audit(
        self, 
        sa2_code: str, 
        source_records: List[DataSourceRecord], 
        master_record: MasterHealthRecord
    ) -> None:
        """Record integration decisions for audit trail."""
        audit_record = {
            'sa2_code': sa2_code,
            'integration_timestamp': datetime.utcnow(),
            'sources_used': [r.source_name for r in source_records],
            'integration_decisions': len(self.integration_decisions),
            'data_conflicts': len(self.data_conflicts),
            'final_completeness': master_record.data_completeness_score
        }
        
        self.logger.debug(f"Integration audit recorded for {sa2_code}: {audit_record}")


class SA2DataAggregator:
    """Aggregates multiple data sources at SA2 level with statistical validation."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialise the SA2 data aggregator."""
        self.config = config
        self.logger = get_logger(__name__) if logger is None else logger
        self.aggregation_methods = config.get('methods', {})
    
    def aggregate_health_data(self, sa2_code: str, source_data: List[DataSourceRecord]) -> Dict[str, Any]:
        """
        Aggregate health data from multiple sources for a specific SA2.
        
        Args:
            sa2_code: SA2 identifier
            source_data: List of source data records
            
        Returns:
            Dictionary of aggregated health indicators
        """
        aggregated_data = {}
        
        # Group data by indicator type
        health_indicators = {}
        for record in source_data:
            if 'health' in record.source_name:
                health_indicators.update(record.data)
        
        # Apply aggregation methods
        for indicator, values in health_indicators.items():
            if isinstance(values, list):
                # Multiple values - apply statistical aggregation
                method = self.aggregation_methods.get(indicator, 'mean')
                aggregated_data[indicator] = self._apply_aggregation_method(values, method)
            else:
                aggregated_data[indicator] = values
        
        return aggregated_data
    
    def _apply_aggregation_method(self, values: List[float], method: str) -> float:
        """Apply statistical aggregation method to a list of values."""
        if not values:
            return 0.0
        
        if method == 'mean':
            return np.mean(values)
        elif method == 'median':
            return np.median(values)
        elif method == 'weighted_mean':
            # Would implement weighted averaging based on data quality
            return np.mean(values)
        elif method == 'max':
            return np.max(values)
        elif method == 'min':
            return np.min(values)
        else:
            return np.mean(values)  # Default to mean


class HealthIndicatorCalculator:
    """Calculates derived health indicators and standardised rates."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialise the health indicator calculator."""
        self.config = config
        self.logger = get_logger(__name__) if logger is None else logger
        self.standard_populations = config.get('standard_populations', {})
    
    def calculate_health_indicators(
        self, 
        sa2_code: str, 
        source_data: List[DataSourceRecord]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive health indicators for an SA2.
        
        Args:
            sa2_code: SA2 identifier
            source_data: List of source data records
            
        Returns:
            Dictionary of calculated health indicators
        """
        indicators = {}
        
        # Extract raw health data
        health_data = self._extract_health_data(source_data)
        
        # Calculate standardised rates
        indicators['outcomes_summary'] = self._calculate_outcomes_summary(health_data)
        
        # Calculate mortality indicators
        indicators['mortality_indicators'] = self._calculate_mortality_indicators(health_data)
        
        # Calculate morbidity indicators
        indicators['chronic_disease_prevalence'] = self._calculate_disease_prevalence(health_data)
        
        # Calculate healthcare access indicators
        indicators['healthcare_access'] = self._calculate_healthcare_access(health_data)
        
        # Extract specific indicators
        indicators.update(self._extract_specific_indicators(health_data))
        
        return indicators
    
    def _extract_health_data(self, source_data: List[DataSourceRecord]) -> Dict[str, Any]:
        """Extract health data from source records."""
        health_data = {}
        for record in source_data:
            if 'health' in record.source_name or 'medicare' in record.source_name:
                health_data.update(record.data)
        return health_data
    
    def _calculate_outcomes_summary(self, health_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate summary of key health outcomes."""
        summary = {}
        
        # Extract key indicators
        if 'life_expectancy' in health_data:
            summary['life_expectancy_score'] = min(100.0, health_data['life_expectancy'] / 85.0 * 100)
        
        if 'infant_mortality_rate' in health_data:
            # Lower is better for mortality (invert score)
            summary['infant_mortality_score'] = max(0.0, 100.0 - health_data['infant_mortality_rate'] * 10)
        
        return summary
    
    def _calculate_mortality_indicators(self, health_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate age-standardised mortality indicators."""
        mortality = {}
        
        # Extract mortality data
        for key, value in health_data.items():
            if 'mortality' in key and isinstance(value, (int, float)):
                mortality[key] = float(value)
        
        return mortality
    
    def _calculate_disease_prevalence(self, health_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate chronic disease prevalence rates."""
        prevalence = {}
        
        # Extract prevalence data
        for key, value in health_data.items():
            if 'prevalence' in key and isinstance(value, (int, float)):
                prevalence[key] = float(value)
        
        return prevalence
    
    def _calculate_healthcare_access(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate healthcare access metrics."""
        access = {}
        
        # Extract access-related indicators
        for key, value in health_data.items():
            if any(term in key for term in ['services', 'access', 'distance', 'availability']):
                access[key] = value
        
        return access
    
    def _extract_specific_indicators(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract specific health indicators by name."""
        indicators = {}
        
        # Direct mapping of specific indicators
        indicator_mappings = {
            'life_expectancy': 'life_expectancy',
            'gp_services_per_1000': 'gp_services_per_1000',
            'specialist_services_per_1000': 'specialist_services_per_1000',
            'bulk_billing_rate': 'bulk_billing_rate',
            'smoking_prevalence': 'smoking_prevalence',
            'obesity_prevalence': 'obesity_prevalence'
        }
        
        for source_key, target_key in indicator_mappings.items():
            if source_key in health_data:
                indicators[target_key] = health_data[source_key]
        
        return indicators


class DemographicProfileBuilder:
    """Builds complete demographic profiles from census and related data."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialise the demographic profile builder."""
        self.config = config
        self.logger = get_logger(__name__) if logger is None else logger
    
    def build_profile(
        self, 
        sa2_code: str, 
        source_data: List[DataSourceRecord]
    ) -> Dict[str, Any]:
        """
        Build comprehensive demographic profile for an SA2.
        
        Args:
            sa2_code: SA2 identifier
            source_data: List of source data records
            
        Returns:
            Dictionary containing demographic profile data
        """
        profile_data = {}
        
        # Extract census data
        census_data = self._extract_census_data(source_data)
        
        # Build age-sex profile
        profile_data['profile'] = self._build_age_sex_profile(census_data)
        
        # Extract population totals
        profile_data['total_population'] = census_data.get('total_population', 0)
        
        # Calculate population density
        area_sqkm = self._get_area_from_sources(source_data)
        if area_sqkm and area_sqkm > 0:
            profile_data['population_density'] = profile_data['total_population'] / area_sqkm
        else:
            profile_data['population_density'] = 0.0
        
        # Extract demographic indicators
        profile_data['median_age'] = census_data.get('median_age')
        
        return profile_data
    
    def _extract_census_data(self, source_data: List[DataSourceRecord]) -> Dict[str, Any]:
        """Extract census data from source records."""
        census_data = {}
        for record in source_data:
            if 'census' in record.source_name:
                census_data.update(record.data)
        return census_data
    
    def _build_age_sex_profile(self, census_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build age-sex demographic profile."""
        profile = {
            'age_groups': {},
            'sex_distribution': {}
        }
        
        # Extract age group data
        for key, value in census_data.items():
            if 'age_' in key and isinstance(value, (int, float)):
                profile['age_groups'][key] = value
        
        # Extract sex distribution
        for key, value in census_data.items():
            if key in ['male_population', 'female_population'] and isinstance(value, (int, float)):
                profile['sex_distribution'][key] = value
        
        return profile
    
    def _get_area_from_sources(self, source_data: List[DataSourceRecord]) -> Optional[float]:
        """Get area information from geographic sources."""
        for record in source_data:
            if 'area_sqkm' in record.data:
                return float(record.data['area_sqkm'])
        return None


class QualityScoreCalculator:
    """Calculates data quality scores for integrated records."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialise the quality score calculator."""
        self.config = config
        self.logger = get_logger(__name__) if logger is None else logger
        self.quality_weights = config.get('weights', {})
    
    def calculate_quality_scores(self, source_data: List[DataSourceRecord]) -> Dict[str, Any]:
        """
        Calculate comprehensive data quality scores.
        
        Args:
            source_data: List of source data records
            
        Returns:
            Dictionary containing quality metrics
        """
        quality_data = {}
        
        # Calculate overall completeness
        quality_data['completeness_score'] = self._calculate_completeness_score(source_data)
        
        # Identify missing indicators
        quality_data['missing_indicators'] = self._identify_missing_indicators(source_data)
        
        # Calculate source quality weights
        quality_data['source_weights'] = self._calculate_source_weights(source_data)
        
        return quality_data
    
    def _calculate_completeness_score(self, source_data: List[DataSourceRecord]) -> float:
        """Calculate overall data completeness score."""
        if not source_data:
            return 0.0
        
        total_fields = 0
        complete_fields = 0
        
        for record in source_data:
            for key, value in record.data.items():
                total_fields += 1
                if value is not None and value != '':
                    complete_fields += 1
        
        return (complete_fields / total_fields * 100.0) if total_fields > 0 else 0.0
    
    def _identify_missing_indicators(self, source_data: List[DataSourceRecord]) -> List[str]:
        """Identify which indicators are missing from the integrated data."""
        missing = []
        
        # Define expected indicators
        expected_indicators = [
            'life_expectancy', 'infant_mortality_rate', 'gp_services_per_1000',
            'seifa_irsd_score', 'total_population', 'median_age'
        ]
        
        # Check which are missing
        available_fields = set()
        for record in source_data:
            available_fields.update(record.data.keys())
        
        for indicator in expected_indicators:
            if indicator not in available_fields:
                missing.append(indicator)
        
        return missing
    
    def _calculate_source_weights(self, source_data: List[DataSourceRecord]) -> Dict[str, float]:
        """Calculate quality-based weights for each source."""
        weights = {}
        
        for record in source_data:
            weights[record.source_name] = record.overall_quality
        
        return weights