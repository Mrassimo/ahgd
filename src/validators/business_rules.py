"""
Australian Health Data Business Rules Validator

This module provides comprehensive business rule validation for Australian health
geography datasets, including ABS, AIHW, Medicare, PBS, SEIFA, and Census data
validation rules specific to Australian health and demographic data standards.
"""

import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from ..utils.interfaces import (
    DataBatch, 
    DataRecord, 
    ValidationResult, 
    ValidationSeverity,
    ValidationError
)
from .base import BaseValidator


@dataclass
class BusinessRule:
    """Business rule definition for Australian health data."""
    rule_id: str
    rule_category: str
    description: str
    validation_type: str
    severity: ValidationSeverity
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    
@dataclass 
class ReferenceDataSet:
    """Reference dataset for validation."""
    name: str
    source: str
    refresh_frequency: str
    validation_scope: List[str]
    data: Optional[Set[str]] = None
    last_updated: Optional[datetime] = None


class AustralianHealthBusinessRulesValidator(BaseValidator):
    """
    Comprehensive business rules validator for Australian health geography data.
    
    This validator implements domain-specific validation rules for Australian
    health and demographic datasets including ABS statistical areas, AIHW health
    data, Medicare utilisation, PBS prescriptions, SEIFA indexes, and Census data.
    """
    
    def __init__(
        self,
        validator_id: str = "australian_health_business_rules",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the Australian health business rules validator.
        
        Args:
            validator_id: Unique identifier for this validator
            config: Configuration dictionary containing business rules
            logger: Optional logger instance
        """
        super().__init__(validator_id, config or {}, logger)
        
        # Load business rules configuration
        self.business_rules_config = self.config.get('business_rules', {})
        
        # Load different rule categories
        self.abs_rules = self.business_rules_config.get('abs_rules', {})
        self.aihw_rules = self.business_rules_config.get('aihw_rules', {})
        self.medicare_pbs_rules = self.business_rules_config.get('medicare_pbs_rules', {})
        self.seifa_rules = self.business_rules_config.get('seifa_rules', {})
        self.census_rules = self.business_rules_config.get('census_rules', {})
        self.geographic_rules = self.business_rules_config.get('geographic_consistency', {})
        self.temporal_rules = self.business_rules_config.get('temporal_consistency', {})
        self.cross_dataset_rules = self.business_rules_config.get('cross_dataset_validation', {})
        
        # Reference datasets
        self.reference_data = self._load_reference_datasets()
        
        # State/Territory mappings
        self.state_territory_mapping = {
            "1": "NSW", "2": "VIC", "3": "QLD", "4": "SA",
            "5": "WA", "6": "TAS", "7": "NT", "8": "ACT"
        }
        
        # Load business rules
        self.business_rules = self._load_business_rules()
        
        # Validation statistics
        self._rule_statistics = {}
        
    def validate(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate data against Australian health business rules.
        
        Args:
            data: Batch of data records to validate
            
        Returns:
            List[ValidationResult]: Business rule validation results
        """
        if not data:
            return [ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="business_rules_empty_data",
                message="Cannot validate business rules on empty dataset"
            )]
        
        results = []
        start_time = datetime.now()
        
        try:
            # ABS Statistical Areas validation
            abs_results = self._validate_abs_rules(data)
            results.extend(abs_results)
            
            # AIHW Health data validation
            aihw_results = self._validate_aihw_rules(data)
            results.extend(aihw_results)
            
            # Medicare and PBS validation
            medicare_pbs_results = self._validate_medicare_pbs_rules(data)
            results.extend(medicare_pbs_results)
            
            # SEIFA validation
            seifa_results = self._validate_seifa_rules(data)
            results.extend(seifa_results)
            
            # Census data validation
            census_results = self._validate_census_rules(data)
            results.extend(census_results)
            
            # Geographic consistency validation
            geographic_results = self._validate_geographic_consistency(data)
            results.extend(geographic_results)
            
            # Temporal consistency validation
            temporal_results = self._validate_temporal_consistency(data)
            results.extend(temporal_results)
            
            # Cross-dataset validation
            cross_dataset_results = self._validate_cross_dataset_rules(data)
            results.extend(cross_dataset_results)
            
            # Custom validators
            custom_results = self._apply_custom_validators(data)
            results.extend(custom_results)
            
            # Update statistics
            self._update_rule_statistics(results)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"Business rules validation completed in {duration:.2f}s: "
                f"{len(results)} rule violations found across {len(data)} records"
            )
            
        except Exception as e:
            self.logger.error(f"Business rules validation failed: {e}")
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="business_rules_validation_error",
                message=f"Business rules validation failed: {str(e)}"
            ))
        
        return results
    
    def get_validation_rules(self) -> List[str]:
        """
        Get the list of validation rules supported by this validator.
        
        Returns:
            List[str]: List of validation rule identifiers
        """
        return [rule.rule_id for rule in self.business_rules]
    
    def _load_reference_datasets(self) -> Dict[str, ReferenceDataSet]:
        """Load reference datasets for validation."""
        reference_config = self.business_rules_config.get('reference_data', {})
        reference_datasets = {}
        
        for dataset_name, dataset_config in reference_config.items():
            reference_datasets[dataset_name] = ReferenceDataSet(
                name=dataset_name,
                source=dataset_config.get('source', ''),
                refresh_frequency=dataset_config.get('refresh_frequency', ''),
                validation_scope=dataset_config.get('validation_scope', [])
            )
        
        return reference_datasets
    
    def _load_business_rules(self) -> List[BusinessRule]:
        """Load business rules from configuration."""
        rules = []
        
        # Load ABS rules
        rules.extend(self._load_abs_business_rules())
        
        # Load AIHW rules  
        rules.extend(self._load_aihw_business_rules())
        
        # Load Medicare/PBS rules
        rules.extend(self._load_medicare_pbs_business_rules())
        
        # Load SEIFA rules
        rules.extend(self._load_seifa_business_rules())
        
        # Load Census rules
        rules.extend(self._load_census_business_rules())
        
        # Load geographic consistency rules
        rules.extend(self._load_geographic_business_rules())
        
        # Load temporal consistency rules
        rules.extend(self._load_temporal_business_rules())
        
        return rules
    
    def _load_abs_business_rules(self) -> List[BusinessRule]:
        """Load Australian Bureau of Statistics business rules."""
        rules = []
        
        # SA2 validation rules
        sa2_rules = self.abs_rules.get('statistical_areas', {}).get('sa2_rules', [])
        for rule_config in sa2_rules:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='abs_sa2',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        # State/Territory mapping rules
        state_mapping = self.abs_rules.get('statistical_areas', {}).get('state_territory_mapping', [])
        for rule_config in state_mapping:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='abs_state_mapping',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        return rules
    
    def _load_aihw_business_rules(self) -> List[BusinessRule]:
        """Load Australian Institute of Health and Welfare business rules."""
        rules = []
        
        # Mortality rates rules
        mortality_rules = self.aihw_rules.get('health_data_constraints', {}).get('mortality_rates', [])
        for rule_config in mortality_rules:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='aihw_mortality',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        # Hospitalisation rules
        hospital_rules = self.aihw_rules.get('health_data_constraints', {}).get('hospitalisation_data', [])
        for rule_config in hospital_rules:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='aihw_hospitalisation',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        return rules
    
    def _load_medicare_pbs_business_rules(self) -> List[BusinessRule]:
        """Load Medicare and PBS business rules."""
        rules = []
        
        # Medicare utilisation rules
        medicare_rules = self.medicare_pbs_rules.get('medicare_utilisation', [])
        for rule_config in medicare_rules:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='medicare',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        # PBS utilisation rules
        pbs_rules = self.medicare_pbs_rules.get('pbs_utilisation', [])
        for rule_config in pbs_rules:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='pbs',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        return rules
    
    def _load_seifa_business_rules(self) -> List[BusinessRule]:
        """Load SEIFA business rules."""
        rules = []
        
        # SEIFA index validation
        seifa_validation = self.seifa_rules.get('index_validation', [])
        for rule_config in seifa_validation:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='seifa',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        return rules
    
    def _load_census_business_rules(self) -> List[BusinessRule]:
        """Load Census data business rules."""
        rules = []
        
        # Demographic data rules
        demographic_rules = self.census_rules.get('demographic_data', [])
        for rule_config in demographic_rules:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='census_demographic',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        # Income data rules
        income_rules = self.census_rules.get('income_data', [])
        for rule_config in income_rules:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='census_income',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        return rules
    
    def _load_geographic_business_rules(self) -> List[BusinessRule]:
        """Load geographic consistency business rules."""
        rules = []
        
        # Spatial relationships
        spatial_rules = self.geographic_rules.get('spatial_relationships', [])
        for rule_config in spatial_rules:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='geographic',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        return rules
    
    def _load_temporal_business_rules(self) -> List[BusinessRule]:
        """Load temporal consistency business rules."""
        rules = []
        
        # Time series validation
        temporal_validation = self.temporal_rules.get('time_series_validation', [])
        for rule_config in temporal_validation:
            rules.append(BusinessRule(
                rule_id=rule_config['rule_id'],
                rule_category='temporal',
                description=rule_config['description'],
                validation_type=rule_config['validation_type'],
                severity=ValidationSeverity(rule_config['severity']),
                parameters=rule_config
            ))
        
        return rules
    
    def _validate_abs_rules(self, data: DataBatch) -> List[ValidationResult]:
        """Validate ABS statistical areas rules."""
        results = []
        
        for record_idx, record in enumerate(data):
            # SA2 code format validation
            sa2_code = record.get('sa2_code')
            if sa2_code:
                sa2_result = self._validate_sa2_code_format(sa2_code, record_idx)
                if sa2_result:
                    results.append(sa2_result)
                
                # SA2 population range validation
                population = record.get('usual_resident_population') or record.get('total_population')
                if population:
                    pop_result = self._validate_sa2_population_range(population, sa2_code, record_idx)
                    if pop_result:
                        results.append(pop_result)
                
                # SA2 area range validation
                area = record.get('geographic_area_sqkm')
                if area:
                    area_result = self._validate_sa2_area_range(area, sa2_code, record_idx)
                    if area_result:
                        results.append(area_result)
                
                # SA2 state prefix validation
                state_code = record.get('state_code')
                if state_code:
                    state_result = self._validate_sa2_state_prefix(sa2_code, state_code, record_idx)
                    if state_result:
                        results.append(state_result)
        
        return results
    
    def _validate_aihw_rules(self, data: DataBatch) -> List[ValidationResult]:
        """Validate AIHW health data rules."""
        results = []
        
        for record_idx, record in enumerate(data):
            # Mortality rate validation
            mortality_rate = record.get('asr_mortality_rate')
            if mortality_rate is not None:
                mortality_result = self._validate_mortality_rate_range(mortality_rate, record_idx)
                if mortality_result:
                    results.append(mortality_result)
            
            # Mortality confidence intervals validation
            mortality_lower = record.get('asr_mortality_lower_ci')
            mortality_upper = record.get('asr_mortality_upper_ci')
            if mortality_rate and mortality_lower and mortality_upper:
                ci_result = self._validate_mortality_confidence_intervals(
                    mortality_rate, mortality_lower, mortality_upper, record_idx
                )
                if ci_result:
                    results.append(ci_result)
            
            # Hospital separation rates validation
            hospital_rate = record.get('hospital_separations_per_1000')
            if hospital_rate is not None:
                hospital_result = self._validate_hospital_separation_rates(hospital_rate, record_idx)
                if hospital_result:
                    results.append(hospital_result)
        
        return results
    
    def _validate_medicare_pbs_rules(self, data: DataBatch) -> List[ValidationResult]:
        """Validate Medicare and PBS rules."""
        results = []
        
        for record_idx, record in enumerate(data):
            # Medicare services per capita validation
            medicare_services = record.get('medicare_services_per_capita')
            if medicare_services is not None:
                medicare_result = self._validate_medicare_services_per_capita(medicare_services, record_idx)
                if medicare_result:
                    results.append(medicare_result)
            
            # Medicare benefits paid validation
            medicare_benefits = record.get('medicare_benefits_paid_per_capita')
            if medicare_benefits is not None:
                benefits_result = self._validate_medicare_benefits_paid(medicare_benefits, record_idx)
                if benefits_result:
                    results.append(benefits_result)
            
            # PBS prescriptions per capita validation
            pbs_prescriptions = record.get('pbs_prescriptions_per_capita')
            if pbs_prescriptions is not None:
                pbs_result = self._validate_pbs_prescriptions_per_capita(pbs_prescriptions, record_idx)
                if pbs_result:
                    results.append(pbs_result)
            
            # PBS cost per capita validation
            pbs_cost = record.get('pbs_cost_per_capita')
            if pbs_cost is not None:
                cost_result = self._validate_pbs_cost_per_capita(pbs_cost, record_idx)
                if cost_result:
                    results.append(cost_result)
        
        return results
    
    def _validate_seifa_rules(self, data: DataBatch) -> List[ValidationResult]:
        """Validate SEIFA index rules."""
        results = []
        
        seifa_score_columns = [
            'seifa_irsad_score', 'seifa_irsed_score', 
            'seifa_ier_score', 'seifa_ieo_score'
        ]
        
        seifa_decile_columns = [
            'seifa_irsad_decile', 'seifa_irsed_decile',
            'seifa_ier_decile', 'seifa_ieo_decile'
        ]
        
        for record_idx, record in enumerate(data):
            # SEIFA score validation
            for score_column in seifa_score_columns:
                score = record.get(score_column)
                if score is not None:
                    score_result = self._validate_seifa_score_range(score, score_column, record_idx)
                    if score_result:
                        results.append(score_result)
            
            # SEIFA decile validation
            for decile_column in seifa_decile_columns:
                decile = record.get(decile_column)
                if decile is not None:
                    decile_result = self._validate_seifa_decile_range(decile, decile_column, record_idx)
                    if decile_result:
                        results.append(decile_result)
        
        return results
    
    def _validate_census_rules(self, data: DataBatch) -> List[ValidationResult]:
        """Validate Census data rules."""
        results = []
        
        for record_idx, record in enumerate(data):
            # Age group totals validation
            age_result = self._validate_age_group_totals(record, record_idx)
            if age_result:
                results.append(age_result)
            
            # Sex distribution validation
            sex_result = self._validate_sex_distribution(record, record_idx)
            if sex_result:
                results.append(sex_result)
            
            # Income quartiles validation
            income_result = self._validate_income_quartiles(record, record_idx)
            if income_result:
                results.append(income_result)
        
        return results
    
    def _validate_geographic_consistency(self, data: DataBatch) -> List[ValidationResult]:
        """Validate geographic consistency rules."""
        results = []
        
        for record_idx, record in enumerate(data):
            # SA2 within SA3 validation
            sa2_sa3_result = self._validate_sa2_within_sa3(record, record_idx)
            if sa2_sa3_result:
                results.append(sa2_sa3_result)
            
            # Coordinate SA2 match validation
            coord_result = self._validate_coordinate_sa2_match(record, record_idx)
            if coord_result:
                results.append(coord_result)
        
        return results
    
    def _validate_temporal_consistency(self, data: DataBatch) -> List[ValidationResult]:
        """Validate temporal consistency rules."""
        results = []
        
        # Group data by SA2 for time series analysis
        sa2_groups = {}
        for record_idx, record in enumerate(data):
            sa2_code = record.get('sa2_code')
            if sa2_code:
                if sa2_code not in sa2_groups:
                    sa2_groups[sa2_code] = []
                sa2_groups[sa2_code].append((record_idx, record))
        
        # Validate population growth for each SA2
        for sa2_code, records in sa2_groups.items():
            if len(records) > 1:
                growth_result = self._validate_population_growth_reasonable(records)
                if growth_result:
                    results.append(growth_result)
        
        # Data year consistency
        year_result = self._validate_data_year_consistency(data)
        if year_result:
            results.append(year_result)
        
        return results
    
    def _validate_cross_dataset_rules(self, data: DataBatch) -> List[ValidationResult]:
        """Validate cross-dataset consistency rules."""
        results = []
        
        # This would typically involve comparing against external reference datasets
        # For now, implement basic cross-field validation
        
        for record_idx, record in enumerate(data):
            # Health rates demographic plausibility
            plausibility_result = self._validate_health_rates_demographic_plausibility(record, record_idx)
            if plausibility_result:
                results.append(plausibility_result)
        
        return results
    
    def _apply_custom_validators(self, data: DataBatch) -> List[ValidationResult]:
        """Apply custom validators specific to Australian health data."""
        results = []
        
        # Indigenous population validator
        indigenous_results = self._validate_indigenous_population_percentage(data)
        results.extend(indigenous_results)
        
        # Remote area validator
        remote_results = self._apply_remote_area_rules(data)
        results.extend(remote_results)
        
        return results
    
    # Individual validation methods
    
    def _validate_sa2_code_format(self, sa2_code: Any, record_idx: int) -> Optional[ValidationResult]:
        """Validate SA2 code format."""
        pattern = r"^[1-8][0-9]{8}$"
        sa2_str = str(sa2_code).strip()
        
        if not re.match(pattern, sa2_str):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="sa2_code_format",
                message=f"Invalid SA2 code format: {sa2_code}",
                details={'sa2_code': sa2_code, 'expected_pattern': pattern},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_sa2_population_range(
        self, 
        population: Any, 
        sa2_code: str, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate SA2 population range."""
        try:
            pop_value = float(population)
            if not (100 <= pop_value <= 50000):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="sa2_population_range",
                    message=f"SA2 population {pop_value} outside typical range (100-50,000)",
                    details={'sa2_code': sa2_code, 'population': pop_value},
                    affected_records=[record_idx]
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="sa2_population_range",
                message=f"Invalid population value: {population}",
                details={'sa2_code': sa2_code, 'population': population},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_sa2_area_range(
        self, 
        area: Any, 
        sa2_code: str, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate SA2 geographic area range."""
        try:
            area_value = float(area)
            if not (0.1 <= area_value <= 50000):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="sa2_area_range",
                    message=f"SA2 area {area_value} sq km outside reasonable range (0.1-50,000)",
                    details={'sa2_code': sa2_code, 'area_sqkm': area_value},
                    affected_records=[record_idx]
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="sa2_area_range",
                message=f"Invalid area value: {area}",
                details={'sa2_code': sa2_code, 'area': area},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_sa2_state_prefix(
        self, 
        sa2_code: str, 
        state_code: str, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate SA2 code state prefix matches state code."""
        sa2_str = str(sa2_code).strip()
        state_str = str(state_code).strip()
        
        if len(sa2_str) >= 1 and sa2_str[0] != state_str:
            expected_state = self.state_territory_mapping.get(sa2_str[0], 'Unknown')
            actual_state = self.state_territory_mapping.get(state_str, 'Unknown')
            
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="sa2_state_prefix",
                message=f"SA2 code {sa2_code} prefix indicates {expected_state} but state code indicates {actual_state}",
                details={
                    'sa2_code': sa2_code,
                    'state_code': state_code,
                    'expected_state': expected_state,
                    'actual_state': actual_state
                },
                affected_records=[record_idx]
            )
        return None
    
    def _validate_mortality_rate_range(
        self, 
        mortality_rate: Any, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate mortality rate range."""
        try:
            rate_value = float(mortality_rate)
            if not (0.0 <= rate_value <= 50.0):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="mortality_rate_range",
                    message=f"Mortality rate {rate_value} outside reasonable range (0-50 per 1,000)",
                    details={'mortality_rate': rate_value},
                    affected_records=[record_idx]
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="mortality_rate_range",
                message=f"Invalid mortality rate value: {mortality_rate}",
                details={'mortality_rate': mortality_rate},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_mortality_confidence_intervals(
        self,
        point_estimate: float,
        lower_ci: float,
        upper_ci: float,
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate mortality rate confidence intervals."""
        if not (lower_ci <= point_estimate <= upper_ci):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="mortality_confidence_intervals",
                message=f"Mortality rate {point_estimate} not within confidence interval [{lower_ci}, {upper_ci}]",
                details={
                    'point_estimate': point_estimate,
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci
                },
                affected_records=[record_idx]
            )
        return None
    
    def _validate_hospital_separation_rates(
        self, 
        separation_rate: Any, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate hospital separation rates."""
        try:
            rate_value = float(separation_rate)
            if not (0.0 <= rate_value <= 2000.0):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="hospital_separation_rates",
                    message=f"Hospital separation rate {rate_value} outside reasonable range (0-2,000 per 1,000)",
                    details={'separation_rate': rate_value},
                    affected_records=[record_idx]
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="hospital_separation_rates",
                message=f"Invalid hospital separation rate: {separation_rate}",
                details={'separation_rate': separation_rate},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_medicare_services_per_capita(
        self, 
        services_per_capita: Any, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate Medicare services per capita."""
        try:
            services_value = float(services_per_capita)
            if not (0.0 <= services_value <= 100.0):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="medicare_services_per_capita",
                    message=f"Medicare services per capita {services_value} outside reasonable range (0-100)",
                    details={'services_per_capita': services_value},
                    affected_records=[record_idx]
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="medicare_services_per_capita",
                message=f"Invalid Medicare services per capita: {services_per_capita}",
                details={'services_per_capita': services_per_capita},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_medicare_benefits_paid(
        self, 
        benefits_paid: Any, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate Medicare benefits paid per capita."""
        try:
            benefits_value = float(benefits_paid)
            if not (0.0 <= benefits_value <= 10000.0):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="medicare_benefits_paid",
                    message=f"Medicare benefits paid {benefits_value} AUD outside reasonable range (0-10,000)",
                    details={'benefits_paid': benefits_value},
                    affected_records=[record_idx]
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="medicare_benefits_paid",
                message=f"Invalid Medicare benefits paid: {benefits_paid}",
                details={'benefits_paid': benefits_paid},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_pbs_prescriptions_per_capita(
        self, 
        prescriptions_per_capita: Any, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate PBS prescriptions per capita."""
        try:
            prescriptions_value = float(prescriptions_per_capita)
            if not (0.0 <= prescriptions_value <= 50.0):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="pbs_prescriptions_per_capita",
                    message=f"PBS prescriptions per capita {prescriptions_value} outside reasonable range (0-50)",
                    details={'prescriptions_per_capita': prescriptions_value},
                    affected_records=[record_idx]
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="pbs_prescriptions_per_capita",
                message=f"Invalid PBS prescriptions per capita: {prescriptions_per_capita}",
                details={'prescriptions_per_capita': prescriptions_per_capita},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_pbs_cost_per_capita(
        self, 
        cost_per_capita: Any, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate PBS cost per capita."""
        try:
            cost_value = float(cost_per_capita)
            if not (0.0 <= cost_value <= 5000.0):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    rule_id="pbs_cost_per_capita",
                    message=f"PBS cost per capita {cost_value} AUD outside reasonable range (0-5,000)",
                    details={'cost_per_capita': cost_value},
                    affected_records=[record_idx]
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="pbs_cost_per_capita",
                message=f"Invalid PBS cost per capita: {cost_per_capita}",
                details={'cost_per_capita': cost_per_capita},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_seifa_score_range(
        self, 
        score: Any, 
        column: str, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate SEIFA score range."""
        try:
            score_value = float(score)
            if not (400 <= score_value <= 1200):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="seifa_score_range",
                    message=f"SEIFA score {score_value} in {column} outside valid range (400-1200)",
                    details={'column': column, 'score': score_value},
                    affected_records=[record_idx]
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="seifa_score_range",
                message=f"Invalid SEIFA score in {column}: {score}",
                details={'column': column, 'score': score},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_seifa_decile_range(
        self, 
        decile: Any, 
        column: str, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate SEIFA decile range."""
        try:
            decile_value = int(decile)
            if not (1 <= decile_value <= 10):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="seifa_decile_range",
                    message=f"SEIFA decile {decile_value} in {column} outside valid range (1-10)",
                    details={'column': column, 'decile': decile_value},
                    affected_records=[record_idx]
                )
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="seifa_decile_range",
                message=f"Invalid SEIFA decile in {column}: {decile}",
                details={'column': column, 'decile': decile},
                affected_records=[record_idx]
            )
        return None
    
    def _validate_age_group_totals(self, record: DataRecord, record_idx: int) -> Optional[ValidationResult]:
        """Validate that age group populations sum to total population."""
        total_population = record.get('total_population')
        if not total_population:
            return None
        
        age_columns = [
            'age_0_4_years', 'age_5_14_years', 'age_15_24_years', 'age_25_34_years',
            'age_35_44_years', 'age_45_54_years', 'age_55_64_years', 'age_65_74_years',
            'age_75_84_years', 'age_85_years_over'
        ]
        
        age_sum = 0
        missing_age_data = []
        
        for age_col in age_columns:
            age_value = record.get(age_col)
            if age_value is not None:
                try:
                    age_sum += float(age_value)
                except (ValueError, TypeError):
                    missing_age_data.append(age_col)
            else:
                missing_age_data.append(age_col)
        
        if not missing_age_data:  # Only validate if we have complete age data
            try:
                total_pop = float(total_population)
                tolerance = 0.02  # 2% tolerance
                difference = abs(age_sum - total_pop) / total_pop if total_pop > 0 else 1
                
                if difference > tolerance:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        rule_id="age_group_totals",
                        message=f"Age group sum {age_sum} differs from total population {total_pop} by {difference:.1%}",
                        details={
                            'total_population': total_pop,
                            'age_group_sum': age_sum,
                            'difference_percentage': difference,
                            'tolerance': tolerance
                        },
                        affected_records=[record_idx]
                    )
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _validate_sex_distribution(self, record: DataRecord, record_idx: int) -> Optional[ValidationResult]:
        """Validate that male and female populations sum to total population."""
        total_population = record.get('total_population')
        male_population = record.get('male_population')
        female_population = record.get('female_population')
        
        if total_population and male_population is not None and female_population is not None:
            try:
                total_pop = float(total_population)
                male_pop = float(male_population)
                female_pop = float(female_population)
                
                sex_sum = male_pop + female_pop
                tolerance = 0.01  # 1% tolerance
                difference = abs(sex_sum - total_pop) / total_pop if total_pop > 0 else 1
                
                if difference > tolerance:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        rule_id="sex_distribution",
                        message=f"Male + female population {sex_sum} differs from total {total_pop} by {difference:.1%}",
                        details={
                            'total_population': total_pop,
                            'male_population': male_pop,
                            'female_population': female_pop,
                            'sex_sum': sex_sum,
                            'difference_percentage': difference
                        },
                        affected_records=[record_idx]
                    )
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _validate_income_quartiles(self, record: DataRecord, record_idx: int) -> Optional[ValidationResult]:
        """Validate that income quartiles are in correct order."""
        q1 = record.get('income_first_quartile')
        median = record.get('median_household_income_weekly')
        q3 = record.get('income_third_quartile')
        
        if q1 is not None and median is not None and q3 is not None:
            try:
                q1_val = float(q1)
                median_val = float(median)
                q3_val = float(q3)
                
                if not (q1_val <= median_val <= q3_val):
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id="income_quartiles",
                        message=f"Income quartiles not in correct order: Q1={q1_val}, Median={median_val}, Q3={q3_val}",
                        details={
                            'q1': q1_val,
                            'median': median_val,
                            'q3': q3_val
                        },
                        affected_records=[record_idx]
                    )
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _validate_sa2_within_sa3(self, record: DataRecord, record_idx: int) -> Optional[ValidationResult]:
        """Validate SA2 is within correct SA3."""
        sa2_code = record.get('sa2_code')
        sa3_code = record.get('sa3_code')
        
        if sa2_code and sa3_code:
            sa2_str = str(sa2_code).strip()
            sa3_str = str(sa3_code).strip()
            
            # SA3 code should be the first 5 digits of SA2 code
            if len(sa2_str) >= 5 and len(sa3_str) >= 5:
                expected_sa3 = sa2_str[:5]
                if sa3_str[:5] != expected_sa3:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        rule_id="sa2_within_sa3",
                        message=f"SA2 {sa2_code} does not belong to SA3 {sa3_code}",
                        details={
                            'sa2_code': sa2_code,
                            'sa3_code': sa3_code,
                            'expected_sa3_prefix': expected_sa3
                        },
                        affected_records=[record_idx]
                    )
        
        return None
    
    def _validate_coordinate_sa2_match(self, record: DataRecord, record_idx: int) -> Optional[ValidationResult]:
        """Validate coordinates are within SA2 boundary (placeholder)."""
        # This would require spatial data and is beyond current scope
        # Return None for now
        return None
    
    def _validate_population_growth_reasonable(
        self, 
        records: List[Tuple[int, DataRecord]]
    ) -> Optional[ValidationResult]:
        """Validate reasonable population growth patterns."""
        if len(records) < 2:
            return None
        
        # Sort by year
        sorted_records = sorted(records, key=lambda x: x[1].get('data_year', 0))
        
        for i in range(1, len(sorted_records)):
            prev_idx, prev_record = sorted_records[i-1]
            curr_idx, curr_record = sorted_records[i]
            
            prev_pop = prev_record.get('total_population')
            curr_pop = curr_record.get('total_population')
            prev_year = prev_record.get('data_year')
            curr_year = curr_record.get('data_year')
            
            if all(x is not None for x in [prev_pop, curr_pop, prev_year, curr_year]):
                try:
                    prev_pop_val = float(prev_pop)
                    curr_pop_val = float(curr_pop)
                    year_diff = int(curr_year) - int(prev_year)
                    
                    if year_diff > 0 and prev_pop_val > 0:
                        annual_change = ((curr_pop_val - prev_pop_val) / prev_pop_val) / year_diff
                        
                        if abs(annual_change) > 0.15:  # 15% annual change threshold
                            return ValidationResult(
                                is_valid=False,
                                severity=ValidationSeverity.WARNING,
                                rule_id="population_growth_reasonable",
                                message=f"Unreasonable population change: {annual_change:.1%} annually",
                                details={
                                    'sa2_code': curr_record.get('sa2_code'),
                                    'previous_population': prev_pop_val,
                                    'current_population': curr_pop_val,
                                    'previous_year': prev_year,
                                    'current_year': curr_year,
                                    'annual_change': annual_change
                                },
                                affected_records=[prev_idx, curr_idx]
                            )
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def _validate_data_year_consistency(self, data: DataBatch) -> Optional[ValidationResult]:
        """Validate all data in batch is from the same year."""
        years = set()
        for record in data:
            year = record.get('data_year')
            if year is not None:
                years.add(str(year))
        
        if len(years) > 1:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                rule_id="data_year_consistency",
                message=f"Multiple data years found in batch: {sorted(years)}",
                details={'years_found': sorted(years)}
            )
        
        return None
    
    def _validate_health_rates_demographic_plausibility(
        self, 
        record: DataRecord, 
        record_idx: int
    ) -> Optional[ValidationResult]:
        """Validate health rates are plausible given demographics."""
        # Basic plausibility check - older areas should generally have higher mortality
        median_age = record.get('median_age')
        mortality_rate = record.get('asr_mortality_rate')
        
        if median_age is not None and mortality_rate is not None:
            try:
                age_val = float(median_age)
                mortality_val = float(mortality_rate)
                
                # Very basic check - areas with very low median age shouldn't have very high mortality
                if age_val < 30 and mortality_val > 30:
                    return ValidationResult(
                        is_valid=True,  # This is informational, not an error
                        severity=ValidationSeverity.INFO,
                        rule_id="health_rates_demographic_plausibility",
                        message=f"High mortality rate {mortality_val} in young area (median age {age_val})",
                        details={
                            'median_age': age_val,
                            'mortality_rate': mortality_val
                        },
                        affected_records=[record_idx]
                    )
            except (ValueError, TypeError):
                pass
        
        return None
    
    def _validate_indigenous_population_percentage(self, data: DataBatch) -> List[ValidationResult]:
        """Validate indigenous population percentages."""
        results = []
        
        for record_idx, record in enumerate(data):
            indigenous_pct = record.get('indigenous_population_percentage')
            if indigenous_pct is not None:
                try:
                    pct_val = float(indigenous_pct)
                    if pct_val > 95.0:  # 95% threshold from config
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.WARNING,
                            rule_id="indigenous_population_validator",
                            message=f"Indigenous population percentage {pct_val}% seems unusually high",
                            details={'indigenous_percentage': pct_val},
                            affected_records=[record_idx]
                        ))
                except (ValueError, TypeError):
                    pass
        
        return results
    
    def _apply_remote_area_rules(self, data: DataBatch) -> List[ValidationResult]:
        """Apply different validation rules for remote areas."""
        results = []
        
        for record_idx, record in enumerate(data):
            # Calculate population density to identify remote areas
            population = record.get('total_population')
            area = record.get('geographic_area_sqkm')
            
            if population is not None and area is not None:
                try:
                    pop_val = float(population)
                    area_val = float(area)
                    
                    if area_val > 0:
                        density = pop_val / area_val
                        
                        # Remote area threshold: 0.5 people per sq km
                        if density < 0.5:
                            # Apply more lenient rules for remote areas
                            # For example, allow wider ranges for health service utilisation
                            medicare_services = record.get('medicare_services_per_capita')
                            if medicare_services is not None:
                                try:
                                    services_val = float(medicare_services)
                                    # More lenient range for remote areas (0-150 instead of 0-100)
                                    if services_val > 150.0:
                                        results.append(ValidationResult(
                                            is_valid=False,
                                            severity=ValidationSeverity.INFO,
                                            rule_id="remote_area_validator",
                                            message=f"High Medicare services per capita {services_val} in remote area",
                                            details={
                                                'medicare_services_per_capita': services_val,
                                                'population_density': density,
                                                'area_classification': 'remote'
                                            },
                                            affected_records=[record_idx]
                                        ))
                                except (ValueError, TypeError):
                                    pass
                except (ValueError, TypeError):
                    pass
        
        return results
    
    def _update_rule_statistics(self, results: List[ValidationResult]) -> None:
        """Update business rule validation statistics."""
        for result in results:
            rule_id = result.rule_id
            
            if rule_id not in self._rule_statistics:
                self._rule_statistics[rule_id] = {
                    'total_violations': 0,
                    'error_count': 0,
                    'warning_count': 0,
                    'info_count': 0
                }
            
            self._rule_statistics[rule_id]['total_violations'] += 1
            
            if result.severity == ValidationSeverity.ERROR:
                self._rule_statistics[rule_id]['error_count'] += 1
            elif result.severity == ValidationSeverity.WARNING:
                self._rule_statistics[rule_id]['warning_count'] += 1
            else:
                self._rule_statistics[rule_id]['info_count'] += 1
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get business rule validation statistics."""
        return dict(self._rule_statistics)