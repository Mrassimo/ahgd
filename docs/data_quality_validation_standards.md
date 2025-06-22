# Data Quality Validation Standards - AHGD Phase 5

**Document Version:** 1.0  
**Standards Date:** 22 June 2025  
**Applicable Scope:** Australian Health and Geographic Data (AHGD) Pipeline  
**Compliance Framework:** Australian Government Data Standards  
**Review Cycle:** Quarterly  

## Executive Summary

This document defines comprehensive data quality validation standards for the AHGD pipeline Phase 5 deployment. It establishes minimum quality thresholds, validation criteria, and monitoring requirements to ensure data meets Australian government standards and international best practices.

**Quality Framework Overview:**
- **Target Completeness:** >90% for critical fields, >95% for core datasets
- **Geographic Standards:** Full compliance with ASGS 2021 and GDA2020
- **Health Standards:** Alignment with AIHW and METEOR requirements
- **Statistical Accuracy:** >95% accuracy for derived indicators
- **Temporal Consistency:** <7 days variance for aligned datasets

## 1. Data Completeness Standards

### 1.1 Completeness Thresholds by Data Domain

#### Critical Fields (100% Completeness Required)
```yaml
critical_fields:
  geographic_identifiers:
    - sa2_code_2021
    - sa2_name
    - state_code
    - geographic_hierarchy
  
  administrative_identifiers:
    - data_source_id
    - reference_year
    - extraction_timestamp
    - data_version
  
  quality_indicators:
    - quality_score
    - completeness_percentage
    - data_lineage_id
```

#### Core Fields (≥95% Completeness Required)
```yaml
core_fields:
  demographic_data:
    - total_population
    - population_by_age_group
    - median_age
    - sex_distribution
  
  health_indicators:
    - mortality_rate_age_standardised
    - healthcare_utilisation_rate
    - gp_services_per_capita
  
  socioeconomic_data:
    - seifa_irsd_score
    - median_household_income
    - education_attainment
```

#### Standard Fields (≥90% Completeness Required)
```yaml
standard_fields:
  environmental_data:
    - climate_indicators
    - air_quality_measurements
    - uv_exposure_index
  
  healthcare_access:
    - distance_to_nearest_hospital
    - bulk_billing_rate
    - specialist_services_availability
```

#### Optional Fields (≥80% Completeness Required)
```yaml
optional_fields:
  supplementary_indicators:
    - derived_health_metrics
    - composite_indices
    - experimental_measures
```

### 1.2 Completeness Calculation Methods

#### Standard Completeness Formula
```python
completeness_percentage = (
    (total_records - null_records - invalid_records) / total_records
) * 100
```

#### Null Value Patterns
```yaml
null_patterns:
  explicit_nulls: ["NULL", "null", "None", ""]
  missing_indicators: ["N/A", "Not Available", "Unknown", "Not Stated"]
  invalid_placeholders: ["999999", "-9", "MISSING", "TBD"]
```

#### Quality Weighting by Geographic Coverage
```yaml
geographic_weighting:
  sa2_coverage_requirement: 100%  # All 2,473 SA2 areas
  population_coverage_target: 99%  # By population weight
  geographic_distribution: "Ensure coverage across all states/territories"
```

## 2. Australian Geographic Standards Compliance

### 2.1 Australian Statistical Geography Standard (ASGS) 2021

#### Mandatory Compliance Requirements
```yaml
asgs_2021_compliance:
  edition: "ASGS Edition 3 (2021)"
  effective_date: "2021-07-01"
  end_date: "2026-06-30"
  
  geographic_levels:
    sa2:
      total_areas: 2473
      code_format: "^[0-9]{11}$"
      name_format: "[A-Za-z0-9\\s\\-\\(\\)]+"
      hierarchy_validation: true
    
    sa3:
      total_areas: 358
      code_format: "^[0-9]{5}$"
      containment_validation: true
    
    sa4:
      total_areas: 107
      code_format: "^[0-9]{3}$"
      containment_validation: true
    
    state_territory:
      total_areas: 9
      code_format: "^[1-9]$"
      official_names: [
        "New South Wales", "Victoria", "Queensland", 
        "South Australia", "Western Australia", "Tasmania",
        "Northern Territory", "Australian Capital Territory",
        "Other Territories"
      ]
```

#### Geographic Boundary Validation
```yaml
boundary_validation:
  coordinate_system: "GDA2020"
  projection: "EPSG:7844"
  precision_decimal_places: 6
  
  topology_checks:
    - no_self_intersections
    - valid_polygon_rings
    - closed_boundaries
    - no_duplicate_vertices
  
  spatial_relationships:
    - sa2_within_sa3: true
    - sa3_within_sa4: true
    - sa4_within_state: true
    - complete_coverage: true
    - no_overlaps: true
    - gap_tolerance_metres: 1.0
```

### 2.2 Geocentric Datum of Australia 2020 (GDA2020)

#### Coordinate System Requirements
```yaml
gda2020_compliance:
  datum: "GDA2020"
  epsg_codes:
    geographic: "EPSG:7844"  # GDA2020 geographic 2D
    utm_zones:
      - "EPSG:7845"  # GDA2020 / MGA zone 49
      - "EPSG:7846"  # GDA2020 / MGA zone 50
      - "EPSG:7847"  # GDA2020 / MGA zone 51
      - "EPSG:7848"  # GDA2020 / MGA zone 52
      - "EPSG:7849"  # GDA2020 / MGA zone 53
      - "EPSG:7850"  # GDA2020 / MGA zone 54
      - "EPSG:7851"  # GDA2020 / MGA zone 55
      - "EPSG:7852"  # GDA2020 / MGA zone 56
  
  coordinate_bounds:
    australia:
      latitude_min: -44.0
      latitude_max: -10.0
      longitude_min: 112.0
      longitude_max: 154.0
    
    mainland:
      latitude_min: -39.2
      latitude_max: -10.0
      longitude_min: 112.0
      longitude_max: 154.0
    
    tasmania:
      latitude_min: -43.7
      latitude_max: -39.2
      longitude_min: 143.8
      longitude_max: 148.5
```

#### Coordinate Transformation Accuracy
```yaml
transformation_requirements:
  positional_accuracy_metres: 1.0
  transformation_method: "7-parameter Helmert"
  validation_checkpoints: "Australian Fiducial Network"
  quality_assessment: "RMSE calculation required"
```

## 3. Health Indicator Quality Standards

### 3.1 AIHW Data Quality Framework Compliance

#### Data Quality Dimensions
```yaml
aihw_quality_dimensions:
  accuracy:
    definition: "Closeness of data to true or accepted values"
    measurement: "Comparison with authoritative sources"
    target_threshold: 95%
    validation_methods:
      - cross_validation_with_source_systems
      - expert_review_of_outliers
      - consistency_checks_across_time_periods
  
  completeness:
    definition: "Extent to which data are not missing"
    measurement: "Percentage of expected data present"
    target_threshold: 90%
    critical_fields_threshold: 95%
  
  consistency:
    definition: "Absence of contradictions across datasets"
    measurement: "Concordance between related data elements"
    target_threshold: 95%
    validation_rules:
      - demographic_totals_consistency
      - geographic_aggregation_consistency
      - temporal_trend_consistency
  
  timeliness:
    definition: "Currency and availability of data"
    measurement: "Time between data collection and availability"
    target_threshold: "Within 90 days of reference period"
    data_freshness_requirements:
      - mortality_data: "Annual, within 12 months"
      - health_indicators: "Biennial, within 18 months"
      - hospitalisation_data: "Annual, within 12 months"
```

#### Health Indicator Validation Rules
```yaml
health_indicator_validation:
  mortality_rates:
    rate_calculation_method: "Age-standardised rates per 100,000"
    standard_population: "Australian Standard Population 2001"
    confidence_intervals: "95% CI required for rates"
    minimum_count_threshold: 5
    suppression_rules:
      - suppress_if_count_less_than_5
      - apply_complementary_suppression
      - use_rate_suppression_for_small_populations
  
  healthcare_utilisation:
    rate_calculation_method: "Per capita or per 1000 population"
    population_denominator: "ABS Estimated Resident Population"
    temporal_alignment: "Calendar year or financial year"
    bulk_billing_calculations:
      - percentage_of_services_bulk_billed
      - exclude_non_bulk_billable_items
      - apply_privacy_protection_thresholds
  
  disease_prevalence:
    prevalence_calculation: "Age-standardised prevalence rates"
    case_definition: "WHO/AIHW standard definitions"
    data_sources: "National health surveys, administrative data"
    confidence_intervals: "95% CI for all prevalence estimates"
```

### 3.2 METEOR Metadata Standards

#### Metadata Requirements
```yaml
meteor_compliance:
  data_element_definitions:
    standard_source: "METEOR (Metadata Online Registry)"
    version_control: "Use current METEOR version"
    change_management: "Track METEOR updates quarterly"
  
  classification_standards:
    icd_10_am: "ICD-10-AM Australian Modification"
    achi: "Australian Classification of Health Interventions"
    snomed_ct_au: "SNOMED CT Australian Edition"
    icpc_2_plus: "International Classification of Primary Care"
  
  data_quality_statements:
    required_elements:
      - institutional_environment
      - data_collection_framework
      - accuracy_assessment
      - coherence_assessment
      - interpretability_guidance
      - accessibility_information
```

## 4. Statistical Accuracy Requirements

### 4.1 Derived Indicator Validation

#### Calculation Accuracy Standards
```yaml
derived_indicators:
  health_service_accessibility:
    calculation_method: "2SFCA (Two-Step Floating Catchment Area)"
    accuracy_threshold: 95%
    validation_approach:
      - compare_with_known_benchmarks
      - sensitivity_analysis_of_parameters
      - expert_validation_of_results
  
  socioeconomic_health_correlation:
    correlation_methods: ["Pearson", "Spearman", "Kendall"]
    significance_threshold: "p < 0.05"
    effect_size_reporting: "Cohen's conventions"
    confounding_adjustment: "Control for age, sex, remoteness"
  
  composite_health_indices:
    weighting_methodology: "Expert consensus or factor analysis"
    normalisation_method: "Z-score or min-max scaling"
    validation_requirements:
      - internal_consistency_analysis
      - construct_validity_assessment
      - convergent_validity_testing
```

#### Statistical Methods Validation
```yaml
statistical_validation:
  outlier_detection:
    methods: ["IQR method", "Z-score", "Modified Z-score", "Isolation Forest"]
    threshold_iqr: 1.5
    threshold_zscore: 3.0
    threshold_modified_zscore: 3.5
    action: "Flag and investigate, do not automatically exclude"
  
  missing_data_handling:
    acceptable_methods:
      - listwise_deletion: "For <5% missing"
      - multiple_imputation: "For 5-20% missing"
      - sensitivity_analysis: "Always required for >10% missing"
    prohibited_methods:
      - single_imputation_with_mean
      - last_observation_carried_forward
      - simple_interpolation_without_validation
  
  confidence_intervals:
    required_for: "All rates, proportions, and means"
    confidence_level: 95%
    calculation_method: "Bootstrap or analytical methods"
    reporting_format: "Point estimate (95% CI: lower, upper)"
```

## 5. Temporal Consistency Standards

### 5.1 Data Alignment Requirements

#### Temporal Synchronisation
```yaml
temporal_alignment:
  reference_periods:
    census_data: "2021-08-10"  # Census night
    health_data: "Calendar year 2021"
    medicare_data: "Calendar year 2021"
    climate_data: "Calendar year 2021"
    
  alignment_tolerance:
    maximum_variance_days: 7
    preferred_variance_days: 0
    
  temporal_interpolation:
    methods: ["Linear interpolation", "Seasonal adjustment"]
    validation_required: true
    maximum_gap_months: 6
```

#### Time Series Validation
```yaml
time_series_validation:
  trend_analysis:
    minimum_time_points: 3
    trend_significance_test: "Mann-Kendall test"
    seasonal_decomposition: "Where applicable"
    
  change_detection:
    sudden_change_threshold: "2 standard deviations"
    investigation_required_if: "Change >20% year-on-year"
    
  forecast_validation:
    holdout_period: "Most recent 12 months"
    accuracy_measures: ["MAPE", "RMSE", "MAE"]
    acceptable_mape: "<10%"
```

## 6. Data Integration Quality Standards

### 6.1 Cross-Dataset Consistency

#### Record Linkage Validation
```yaml
record_linkage:
  geographic_linkage:
    primary_key: "sa2_code_2021"
    linkage_success_rate: ">99%"
    unmatched_records_action: "Investigate and document"
    
  temporal_linkage:
    alignment_method: "Reference year matching"
    acceptable_misalignment: "±30 days"
    interpolation_rules: "Linear for continuous variables"
    
  validation_checks:
    - population_consistency_across_sources
    - geographic_boundary_alignment
    - temporal_period_consistency
    - data_source_version_compatibility
```

#### Integration Quality Metrics
```yaml
integration_quality:
  data_fusion_accuracy:
    measurement: "Cross-validation against known ground truth"
    threshold: ">95% agreement"
    
  schema_compatibility:
    field_mapping_accuracy: "100% for core fields"
    data_type_consistency: "No type conversion errors"
    enumeration_standardisation: "Consistent coding schemes"
    
  referential_integrity:
    foreign_key_validation: "100% valid references"
    lookup_table_completeness: "100% coverage"
    relationship_consistency: "No orphaned records"
```

## 7. Quality Monitoring and Reporting

### 7.1 Automated Quality Checks

#### Real-Time Monitoring
```yaml
automated_monitoring:
  quality_gates:
    extraction_quality_gate:
      completeness_threshold: 95%
      schema_compliance: 100%
      data_freshness_hours: 24
      
    transformation_quality_gate:
      accuracy_threshold: 95%
      consistency_checks: 100%
      geometric_validation: 99%
      
    integration_quality_gate:
      linkage_success_rate: 99%
      cross_dataset_consistency: 95%
      derived_indicator_accuracy: 95%
      
    loading_quality_gate:
      export_completeness: 100%
      format_compliance: 100%
      metadata_generation: 100%
```

#### Quality Dashboards
```yaml
quality_dashboards:
  real_time_metrics:
    - current_pipeline_status
    - quality_gate_pass_rates
    - error_counts_by_category
    - data_freshness_indicators
    
  trend_analysis:
    - quality_score_trends
    - completeness_trends_by_domain
    - accuracy_trends_over_time
    - user_feedback_analysis
    
  exception_reporting:
    - quality_gate_failures
    - unusual_patterns_detected
    - data_source_unavailability
    - performance_degradation_alerts
```

### 7.2 Quality Assurance Reporting

#### Regular Quality Reports
```yaml
reporting_schedule:
  daily_reports:
    - pipeline_execution_summary
    - quality_gate_status
    - error_log_summary
    
  weekly_reports:
    - quality_trends_analysis
    - completeness_assessment
    - user_feedback_summary
    
  monthly_reports:
    - comprehensive_quality_assessment
    - compliance_status_report
    - recommendations_for_improvement
    
  quarterly_reports:
    - annual_quality_review
    - standards_compliance_audit
    - strategic_quality_planning
```

## 8. Quality Standards Implementation

### 8.1 Technical Implementation

#### Validation Pipeline Architecture
```yaml
validation_architecture:
  stage_1_extraction:
    - source_availability_checks
    - schema_validation
    - completeness_assessment
    - data_freshness_validation
    
  stage_2_transformation:
    - geographic_standardisation_validation
    - data_type_consistency_checks
    - business_rules_validation
    - derived_field_calculation_verification
    
  stage_3_integration:
    - cross_dataset_consistency_validation
    - temporal_alignment_verification
    - statistical_accuracy_assessment
    - relationship_integrity_checks
    
  stage_4_loading:
    - export_format_validation
    - metadata_completeness_checks
    - access_control_verification
    - documentation_generation_validation
```

#### Quality Control Automation
```yaml
automation_framework:
  validation_scripts:
    language: "Python with Pydantic v2"
    framework: "Built on schemas/quality_standards.py"
    execution: "Integrated with DVC pipeline"
    
  quality_metrics_collection:
    storage: "Time-series database"
    aggregation: "Real-time and batch processing"
    alerting: "Threshold-based notifications"
    
  reporting_automation:
    generation: "Automated report generation"
    distribution: "Email and dashboard publishing"
    archival: "Quality metrics historical storage"
```

## 9. Compliance Verification

### 9.1 Audit Requirements

#### Internal Audits
```yaml
internal_audits:
  frequency: "Monthly for critical components"
  scope: "All quality standards and procedures"
  documentation: "Comprehensive audit trails"
  
  audit_checklist:
    - completeness_thresholds_met
    - geographic_standards_compliance
    - health_standards_alignment
    - statistical_accuracy_verification
    - temporal_consistency_validation
    - integration_quality_assessment
```

#### External Validation
```yaml
external_validation:
  frequency: "Annual"
  scope: "Independent quality assessment"
  validators: "Subject matter experts"
  
  validation_areas:
    - methodology_review
    - results_verification
    - compliance_assessment
    - recommendations_for_improvement
```

### 9.2 Continuous Improvement

#### Quality Enhancement Process
```yaml
continuous_improvement:
  feedback_mechanisms:
    - user_feedback_collection
    - expert_advisory_panel
    - data_custodian_consultation
    - international_best_practice_review
    
  enhancement_priorities:
    - accuracy_improvements
    - completeness_optimisation
    - timeliness_enhancements
    - usability_improvements
    
  implementation_cycle:
    - quarterly_enhancement_planning
    - agile_development_approach
    - comprehensive_testing_requirements
    - staged_deployment_strategy
```

## 10. Success Criteria and Targets

### 10.1 Phase 5 Quality Targets

#### Minimum Acceptable Standards
```yaml
minimum_standards:
  overall_completeness: 90%
  critical_field_completeness: 100%
  geographic_coverage: 100%  # All SA2 areas
  health_indicator_accuracy: 95%
  integration_success_rate: 99%
  user_satisfaction_score: 4.0/5.0
```

#### Excellence Targets
```yaml
excellence_targets:
  overall_completeness: 95%
  statistical_accuracy: 98%
  data_freshness: "Within 30 days"
  response_time_p95: "<2 seconds"
  error_rate: "<0.1%"
  user_satisfaction_score: 4.5/5.0
```

### 10.2 Success Measurement

#### Key Performance Indicators
```yaml
quality_kpis:
  data_quality_score:
    calculation: "Weighted average of all quality dimensions"
    target: ">95%"
    
  compliance_rate:
    measurement: "Percentage of standards met"
    target: "100%"
    
  user_adoption_rate:
    measurement: "Active users and download metrics"
    target: "Growing trend"
    
  feedback_quality:
    measurement: "User satisfaction and expert validation"
    target: "Positive feedback >90%"
```

---

**Document Control:**
- **Document Owner:** Data Quality Team
- **Technical Implementation:** ETL Engineering Team  
- **Subject Matter Review:** Health Data Specialists
- **Approval Authority:** Data Governance Committee
- **Next Review Date:** 22 September 2025