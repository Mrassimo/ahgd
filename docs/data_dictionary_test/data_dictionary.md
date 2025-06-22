# AHGD Data Dictionary

Comprehensive data dictionary for Australian Health Geography Data schemas.

Generated: 2025-06-22 09:25:11

## Table of Contents

- [Demographics](#demographics)
  - [CensusDemographics](#censusdemographics)
- [Geographic](#geographic)
  - [GeographicValidationRequirement](#geographicvalidationrequirement)
  - [GeographicBoundary](#geographicboundary)
  - [GeographicBoundary](#geographicboundary)
  - [SA2BoundaryRelationship](#sa2boundaryrelationship)
  - [GeographicBoundary](#geographicboundary)
  - [GeographicHealthMapping](#geographichealthmapping)
  - [GeographicBoundary](#geographicboundary)
- [Health](#health)
  - [EnvironmentalHealthIndex](#environmentalhealthindex)
  - [AustralianHealthDataStandard](#australianhealthdatastandard)
  - [MortalityRecord](#mortalityrecord)
  - [MortalityStatistics](#mortalitystatistics)
  - [MortalityTrend](#mortalitytrend)
  - [HealthIndicatorSummary](#healthindicatorsummary)
  - [MasterHealthRecord](#masterhealthrecord)
  - [SA2HealthProfile](#sa2healthprofile)
  - [HealthDataAggregate](#healthdataaggregate)
  - [HealthIndicator](#healthindicator)
  - [HealthcareUtilisation](#healthcareutilisation)
  - [MentalHealthIndicator](#mentalhealthindicator)
  - [MortalityData](#mortalitydata)
- [Other](#other)
  - [ClimateStatistics](#climatestatistics)
  - [DataSource](#datasource)
  - [TemporalData](#temporaldata)
  - [VersionedSchema](#versionedschema)
  - [WeatherObservation](#weatherobservation)
  - [DataCompletenessRequirement](#datacompletenessrequirement)
  - [QualityStandardsRegistry](#qualitystandardsregistry)
  - [StatisticalValidationThreshold](#statisticalvalidationthreshold)
  - [VersionedSchema](#versionedschema)
  - [CensusEducation](#censuseducation)
  - [CensusEmployment](#censusemployment)
  - [CensusHousing](#censushousing)
  - [DataSource](#datasource)
  - [TemporalData](#temporaldata)
  - [VersionedSchema](#versionedschema)
  - [DataSource](#datasource)
  - [TemporalData](#temporaldata)
  - [VersionedSchema](#versionedschema)
  - [DataSource](#datasource)
  - [VersionedSchema](#versionedschema)
  - [APIResponseSchema](#apiresponseschema)
  - [DataQualityReport](#dataqualityreport)
  - [DataSource](#datasource)
  - [DataWarehouseTable](#datawarehousetable)
  - [ExportSpecification](#exportspecification)
  - [VersionedSchema](#versionedschema)
  - [WebPlatformDataStructure](#webplatformdatastructure)
  - [DataSource](#datasource)
  - [SA2Coordinates](#sa2coordinates)
  - [SA2GeometryValidation](#sa2geometryvalidation)
  - [VersionedSchema](#versionedschema)
  - [DataSource](#datasource)
  - [TemporalData](#temporaldata)
  - [VersionedSchema](#versionedschema)
  - [BaseSettings](#basesettings)
  - [DataSource](#datasource)
  - [MigrationRecord](#migrationrecord)
  - [TemporalData](#temporaldata)
  - [VersionedSchema](#versionedschema)
  - [DataSource](#datasource)
  - [DiseasePrevalence](#diseaseprevalence)
  - [RiskFactorData](#riskfactordata)
  - [TemporalData](#temporaldata)
  - [VersionedSchema](#versionedschema)
- [Socio-Economic](#socio-economic)
  - [SEIFAAggregate](#seifaaggregate)
  - [SEIFAComparison](#seifacomparison)
  - [SEIFAComponent](#seifacomponent)
  - [SEIFAScore](#seifascore)

## Demographics

### CensusDemographics

Schema for basic census demographic data (Table G01).

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level (SA1, SA2, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_name | str | Geographic area name |  | Example text, Sample value, Data entry | Multiple sources |
| state_territory | str | State or territory |  | NSW, VIC, QLD | Multiple sources |
| census_year | int | Census year |  | 123, 456, 789 | Australian Bureau of Statistics (ABS) |
| table_code | str | Census table code |  | Example text, Sample value, Data entry | Multiple sources |
| table_name | str | Table name |  | Example text, Sample value, Data entry | Multiple sources |
| total_population | int | Total usual resident population |  | 5432, 12876, 3241 | Multiple sources |
| males | int | Number of males |  | 123, 456, 789 | Multiple sources |
| females | int | Number of females |  | 123, 456, 789 | Multiple sources |
| age_0_4 | int | Population aged 0-4 years |  | 123, 456, 789 | Multiple sources |
| age_5_9 | int | Population aged 5-9 years |  | 123, 456, 789 | Multiple sources |
| age_10_14 | int | Population aged 10-14 years |  | 123, 456, 789 | Multiple sources |
| age_15_19 | int | Population aged 15-19 years |  | 123, 456, 789 | Multiple sources |
| age_20_24 | int | Population aged 20-24 years |  | 123, 456, 789 | Multiple sources |
| age_25_29 | int | Population aged 25-29 years |  | 123, 456, 789 | Multiple sources |
| age_30_34 | int | Population aged 30-34 years |  | 123, 456, 789 | Multiple sources |
| age_35_39 | int | Population aged 35-39 years |  | 123, 456, 789 | Multiple sources |
| age_40_44 | int | Population aged 40-44 years |  | 123, 456, 789 | Multiple sources |
| age_45_49 | int | Population aged 45-49 years |  | 123, 456, 789 | Multiple sources |
| age_50_54 | int | Population aged 50-54 years |  | 123, 456, 789 | Multiple sources |
| age_55_59 | int | Population aged 55-59 years |  | 123, 456, 789 | Multiple sources |
| age_60_64 | int | Population aged 60-64 years |  | 123, 456, 789 | Multiple sources |
| age_65_69 | int | Population aged 65-69 years |  | 123, 456, 789 | Multiple sources |
| age_70_74 | int | Population aged 70-74 years |  | 123, 456, 789 | Multiple sources |
| age_75_79 | int | Population aged 75-79 years |  | 123, 456, 789 | Multiple sources |
| age_80_84 | int | Population aged 80-84 years |  | 123, 456, 789 | Multiple sources |
| age_85_plus | int | Population aged 85+ years |  | 123, 456, 789 | Multiple sources |
| indigenous | int | Aboriginal and Torres Strait Islander population |  | 123, 456, 789 | Multiple sources |
| non_indigenous | int | Non-Indigenous population |  | 123, 456, 789 | Multiple sources |
| indigenous_not_stated | int | Indigenous status not stated |  | NSW, VIC, QLD | Multiple sources |
| total_private_dwellings | int | Total private dwellings |  | 123, 456, 789 | Multiple sources |
| occupied_private_dwellings | int | Occupied private dwellings |  | 123, 456, 789 | Multiple sources |
| unoccupied_private_dwellings | int | Unoccupied private dwellings |  | 123, 456, 789 | Multiple sources |
| total_families | int | Total number of families |  | 123, 456, 789 | Multiple sources |
| data_source | DataSource | Source of census data |  |  | Multiple sources |

## Geographic

### GeographicValidationRequirement


    Geographic validation requirements and spatial quality standards.
    
    Defines validation rules specific to geographic and spatial data
    including coordinate validation, spatial relationships, and topology.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| requirement_name | str | Name of geographic validation requirement |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_scope | str | Geographic scope (australia, state, sa2) |  | Example text, Sample value, Data entry | Multiple sources |
| coordinate_system | str | Required coordinate reference system |  | Example text, Sample value, Data entry | Multiple sources |
| coordinate_precision | int | Required decimal places for coordinates |  | 123, 456, 789 | Multiple sources |
| coordinate_bounds | Dict[str, Dict[str, float]] | Valid coordinate bounds by region |  |  | Multiple sources |
| positional_accuracy_metres | float | Required positional accuracy in metres |  | 42.5, 67.8, 91.2 | Multiple sources |
| scale_accuracy | Optional[str] | Required scale accuracy (e.g., 1:10000) |  |  | Multiple sources |
| geometry_validation_rules | List[Dict[str, Any]] | Geometry validation rules |  |  | Multiple sources |
| topology_validation | bool | Whether to validate topology |  | true, false | Multiple sources |
| self_intersection_tolerance | float | Tolerance for self-intersection detection |  | 42.5, 67.8, 91.2 | Multiple sources |
| boundary_completeness_threshold | float | Required boundary completeness (%) |  | 42.5, 67.8, 91.2 | Multiple sources |
| administrative_boundary_compliance | bool | Must comply with official administrative boundaries |  | true, false | Multiple sources |
| containment_validation | bool | Validate spatial containment relationships |  | true, false | Multiple sources |
| adjacency_validation | bool | Validate spatial adjacency relationships |  | true, false | Multiple sources |
| overlap_tolerance_metres | float | Tolerance for boundary overlaps (metres) |  | 42.5, 67.8, 91.2 | Multiple sources |
| gap_tolerance_metres | float | Tolerance for boundary gaps (metres) |  | 42.5, 67.8, 91.2 | Multiple sources |
| minimum_area_square_metres | Optional[float] | Minimum valid area in square metres |  |  | Multiple sources |
| maximum_area_square_metres | Optional[float] | Maximum valid area in square metres |  |  | Multiple sources |
| area_calculation_method | str | Method for area calculation (spherical, planar) |  | Example text, Sample value, Data entry | Multiple sources |
| required_projections | List[str] | Required map projections to support |  |  | Multiple sources |
| transformation_accuracy | float | Required transformation accuracy (metres) |  | 42.5, 67.8, 91.2 | Multiple sources |
| geometric_quality_indicators | List[str] | Required geometric quality indicators |  |  | Multiple sources |
| completeness_assessment_method | str | Method for assessing spatial completeness |  | Example text, Sample value, Data entry | Multiple sources |
| temporal_consistency_validation | bool | Validate temporal consistency of boundaries |  | true, false | Multiple sources |
| change_detection_threshold | float | Threshold for detecting significant boundary changes (%) |  | 42.5, 67.8, 91.2 | Multiple sources |

### GeographicBoundary

Base model for geographic boundary data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| boundary_id | str | Unique boundary identifier |  | Example text, Sample value, Data entry | Multiple sources |
| boundary_type | str | Type of boundary (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| name | str | Human-readable boundary name |  | Example text, Sample value, Data entry | Multiple sources |
| state | str | State or territory code |  | NSW, VIC, QLD | Multiple sources |
| area_sq_km | Optional[float] | Area in square kilometers |  |  | Multiple sources |
| perimeter_km | Optional[float] | Perimeter in kilometers |  |  | Multiple sources |
| centroid_lat | Optional[float] | Centroid latitude |  |  | Multiple sources |
| centroid_lon | Optional[float] | Centroid longitude |  |  | Multiple sources |
| geometry | Optional[Dict[str, Any]] | GeoJSON geometry object |  |  | Multiple sources |

### GeographicBoundary

Base model for geographic boundary data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| boundary_id | str | Unique boundary identifier |  | Example text, Sample value, Data entry | Multiple sources |
| boundary_type | str | Type of boundary (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| name | str | Human-readable boundary name |  | Example text, Sample value, Data entry | Multiple sources |
| state | str | State or territory code |  | NSW, VIC, QLD | Multiple sources |
| area_sq_km | Optional[float] | Area in square kilometers |  |  | Multiple sources |
| perimeter_km | Optional[float] | Perimeter in kilometers |  |  | Multiple sources |
| centroid_lat | Optional[float] | Centroid latitude |  |  | Multiple sources |
| centroid_lon | Optional[float] | Centroid longitude |  |  | Multiple sources |
| geometry | Optional[Dict[str, Any]] | GeoJSON geometry object |  |  | Multiple sources |

### SA2BoundaryRelationship

Schema for SA2 spatial relationships and adjacency.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| sa2_code | str | Primary SA2 code (Statistical Area Level 2 code (ASGS 2021)) |  | 101011007, 201011021, 301011001 | Australian Bureau of Statistics (ABS) |
| adjacent_sa2s | List[Dict[str, Any]] | List of adjacent SA2s with shared boundary info |  |  | Australian Bureau of Statistics (ABS) |
| contains_sa1s | List[str] | List of SA1 codes contained within this SA2 |  |  | Multiple sources |
| nearest_coast_km | Optional[float] | Distance to nearest coastline in km |  |  | Multiple sources |
| nearest_capital_km | Optional[float] | Distance to nearest capital city in km |  |  | Multiple sources |
| remoteness_category | Optional[str] | ABS remoteness structure category (ARIA (Accessibility/Remoteness Index of Australia)) |  |  | Multiple sources |

### GeographicBoundary

Base model for geographic boundary data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| boundary_id | str | Unique boundary identifier |  | Example text, Sample value, Data entry | Multiple sources |
| boundary_type | str | Type of boundary (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| name | str | Human-readable boundary name |  | Example text, Sample value, Data entry | Multiple sources |
| state | str | State or territory code |  | NSW, VIC, QLD | Multiple sources |
| area_sq_km | Optional[float] | Area in square kilometers |  |  | Multiple sources |
| perimeter_km | Optional[float] | Perimeter in kilometers |  |  | Multiple sources |
| centroid_lat | Optional[float] | Centroid latitude |  |  | Multiple sources |
| centroid_lon | Optional[float] | Centroid longitude |  |  | Multiple sources |
| geometry | Optional[Dict[str, Any]] | GeoJSON geometry object |  |  | Multiple sources |

### GeographicHealthMapping


    Geographic relationships and health data linkages.
    
    Defines spatial relationships and geographic patterns in health data
    for spatial analysis and mapping applications.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| primary_area_id | str | Primary geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| primary_area_type | str | Primary area type (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| primary_area_name | str | Primary area name |  | Example text, Sample value, Data entry | Multiple sources |
| containing_areas | Dict[str, str] | Higher-level areas containing this area |  |  | Multiple sources |
| contained_areas | List[str] | Lower-level areas contained within this area |  |  | Multiple sources |
| adjacent_areas | List[Dict[str, Any]] | Adjacent areas with relationship metrics |  |  | Multiple sources |
| centroid_coordinates | Dict[str, float] | Centroid latitude and longitude |  |  | Multiple sources |
| area_square_km | float | Area in square kilometres |  | 42.5, 67.8, 91.2 | Multiple sources |
| perimeter_km | float | Perimeter in kilometres |  | 42.5, 67.8, 91.2 | Multiple sources |
| compactness_ratio | Optional[float] | Shape compactness ratio |  |  | Multiple sources |
| distance_to_services | Dict[str, float] | Distance to key health services (km) |  |  | Multiple sources |
| travel_time_to_services | Dict[str, float] | Travel time to key services (minutes) |  |  | Multiple sources |
| service_catchment_populations | Dict[str, int] | Population within service catchments |  | 5432, 12876, 3241 | Multiple sources |
| health_services_within_area | Dict[str, int] | Count of health services within area |  |  | Australian Institute of Health and Welfare (AIHW) |
| health_workforce_density | Dict[str, float] | Health workforce per 1,000 population |  |  | Australian Institute of Health and Welfare (AIHW) |
| environmental_exposures | Dict[str, float] | Environmental health exposure metrics |  |  | Multiple sources |
| green_space_percentage | Optional[float] | Green space coverage % |  |  | Multiple sources |
| air_quality_metrics | Dict[str, float] | Air quality measurements |  |  | Multiple sources |
| spatial_health_clusters | List[Dict[str, Any]] | Identified spatial health clusters |  |  | Australian Institute of Health and Welfare (AIHW) |
| health_hotspots | List[str] | Health condition hotspot classifications |  |  | Australian Institute of Health and Welfare (AIHW) |
| health_coldspots | List[str] | Areas with better than expected health |  |  | Australian Institute of Health and Welfare (AIHW) |
| remoteness_area | str | ABS Remoteness Area classification |  | Example text, Sample value, Data entry | Multiple sources |
| accessibility_category | str | Health service accessibility classification |  | Example text, Sample value, Data entry | Multiple sources |
| transport_access_score | Optional[float] | Public transport access score |  |  | Multiple sources |
| spatial_autocorrelation | Dict[str, float] | Spatial autocorrelation measures for health indicators |  |  | Multiple sources |
| spillover_effects | Dict[str, float] | Spillover effects from neighbouring areas |  |  | Multiple sources |
| coordinate_system | str | Coordinate reference system |  | Example text, Sample value, Data entry | Multiple sources |
| geometry_source | str | Source of geometric data |  | Example text, Sample value, Data entry | Multiple sources |
| geometry_date | datetime | Date of geometry data |  |  | Multiple sources |
| simplification_tolerance | Optional[float] | Geometry simplification tolerance (metres) |  |  | Multiple sources |

### GeographicBoundary

Base model for geographic boundary data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| boundary_id | str | Unique boundary identifier |  | Example text, Sample value, Data entry | Multiple sources |
| boundary_type | str | Type of boundary (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| name | str | Human-readable boundary name |  | Example text, Sample value, Data entry | Multiple sources |
| state | str | State or territory code |  | NSW, VIC, QLD | Multiple sources |
| area_sq_km | Optional[float] | Area in square kilometers |  |  | Multiple sources |
| perimeter_km | Optional[float] | Perimeter in kilometers |  |  | Multiple sources |
| centroid_lat | Optional[float] | Centroid latitude |  |  | Multiple sources |
| centroid_lon | Optional[float] | Centroid longitude |  |  | Multiple sources |
| geometry | Optional[Dict[str, Any]] | GeoJSON geometry object |  |  | Multiple sources |

## Health

### EnvironmentalHealthIndex

Schema for environmental health risk indices.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level |  | Example text, Sample value, Data entry | Multiple sources |
| assessment_date | date | Date of assessment |  |  | Multiple sources |
| heat_index_mean | Optional[float] | Mean heat index |  |  | Multiple sources |
| heat_index_max | Optional[float] | Maximum heat index |  |  | Multiple sources |
| extreme_heat_days | Optional[int] | Days with extreme heat index |  |  | Multiple sources |
| atmospheric_stability_index | Optional[float] | Atmospheric stability index |  |  | Multiple sources |
| ventilation_coefficient | Optional[float] | Atmospheric ventilation coefficient |  |  | Multiple sources |
| dust_storm_risk_days | Optional[int] | Days with dust storm risk |  |  | Multiple sources |
| uv_index_mean | Optional[float] | Mean UV index |  |  | Multiple sources |
| uv_index_max | Optional[float] | Maximum UV index |  |  | Multiple sources |
| extreme_uv_days | Optional[int] | Days with extreme UV |  |  | Multiple sources |
| discomfort_index_mean | Optional[float] | Mean discomfort index |  |  | Multiple sources |
| very_humid_days | Optional[int] | Days with very high humidity |  |  | Multiple sources |
| heatwave_events | Optional[int] | Number of heatwave events |  |  | Multiple sources |
| severe_storm_days | Optional[int] | Days with severe storms |  |  | Multiple sources |
| drought_index | Optional[float] | Drought severity index |  |  | Multiple sources |
| flood_risk_days | Optional[int] | Days with flood risk |  |  | Multiple sources |
| overall_health_risk_score | float | Overall environmental health risk score |  | 42.5, 67.8, 91.2 | Australian Institute of Health and Welfare (AIHW) |
| heat_stress_risk | float | Heat stress risk score |  | 42.5, 67.8, 91.2 | Multiple sources |
| respiratory_risk | float | Respiratory health risk score |  | 42.5, 67.8, 91.2 | Multiple sources |
| elderly_risk_multiplier | float | Risk multiplier for elderly |  | 42.5, 67.8, 91.2 | Multiple sources |
| children_risk_multiplier | float | Risk multiplier for children |  | 42.5, 67.8, 91.2 | Multiple sources |
| outdoor_worker_risk | float | Risk score for outdoor workers |  | 42.5, 67.8, 91.2 | Multiple sources |
| data_source | DataSource | Source of environmental health data |  |  | Multiple sources |

### AustralianHealthDataStandard


    Australian health data standards compliance specification.
    
    Defines compliance requirements for Australian health data standards
    including AIHW, ABS, Medicare, and other regulatory frameworks.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| standard_name | ComplianceStandard | Name of compliance standard |  |  | Multiple sources |
| standard_version | str | Version of the standard |  | Example text, Sample value, Data entry | Multiple sources |
| standard_url | Optional[str] | URL to standard documentation |  |  | Multiple sources |
| effective_date | datetime | When standard becomes effective |  |  | Multiple sources |
| applicable_data_types | List[str] | Data types this standard applies to |  |  | Multiple sources |
| mandatory_fields | List[str] | Fields required by this standard |  |  | Multiple sources |
| optional_fields | List[str] | Fields recommended but not required |  |  | Multiple sources |
| field_requirements | Dict[str, Dict[str, Any]] | Detailed requirements for each field |  |  | Multiple sources |
| format_rules | List[Dict[str, Any]] | Format validation rules |  |  | Multiple sources |
| value_constraints | List[Dict[str, Any]] | Value range and constraint rules |  |  | Multiple sources |
| business_rules | List[Dict[str, Any]] | Business logic validation rules |  |  | Multiple sources |
| minimum_completeness | float | Minimum data completeness percentage required |  | 42.5, 67.8, 91.2 | Multiple sources |
| accuracy_threshold | float | Minimum accuracy percentage required |  | 42.5, 67.8, 91.2 | Multiple sources |
| reporting_frequency | str | Required reporting frequency |  | Example text, Sample value, Data entry | Multiple sources |
| audit_requirements | List[str] | Audit and compliance reporting requirements |  |  | Multiple sources |
| non_compliance_penalties | List[str] | Penalties for non-compliance |  |  | Multiple sources |
| grace_period_days | Optional[int] | Grace period for compliance (days) |  |  | Multiple sources |

### MortalityRecord

Schema for individual mortality records.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| record_id | str | Unique mortality record identifier |  | Example text, Sample value, Data entry | Multiple sources |
| registration_year | int | Year of death registration |  | 123, 456, 789 | Multiple sources |
| registration_state | str | State of death registration |  | NSW, VIC, QLD | Multiple sources |
| usual_residence_sa2 | Optional[str] | SA2 of usual residence |  |  | Australian Bureau of Statistics (ABS) |
| usual_residence_sa3 | Optional[str] | SA3 of usual residence |  |  | Australian Bureau of Statistics (ABS) |
| usual_residence_sa4 | Optional[str] | SA4 of usual residence |  |  | Australian Bureau of Statistics (ABS) |
| usual_residence_state | Optional[str] | State of usual residence |  | NSW, VIC, QLD | Multiple sources |
| place_of_death_sa2 | Optional[str] | SA2 where death occurred |  |  | Australian Bureau of Statistics (ABS) |
| place_of_death_type | PlaceOfDeathType | Type of place where death occurred |  |  | Multiple sources |
| age_at_death | int | Age at death in years |  | 123, 456, 789 | Multiple sources |
| sex | str | Sex of deceased |  | Example text, Sample value, Data entry | Multiple sources |
| indigenous_status | Optional[str] | Indigenous status |  |  | Multiple sources |
| country_of_birth | Optional[str] | Country of birth |  |  | Multiple sources |
| underlying_cause_icd10 | str | Underlying cause of death (ICD-10) |  | Example text, Sample value, Data entry | Multiple sources |
| underlying_cause_description | str | Description of underlying cause |  | Example text, Sample value, Data entry | Multiple sources |
| immediate_cause_icd10 | Optional[str] | Immediate cause of death (ICD-10) |  |  | Multiple sources |
| contributing_causes | List[str] | Contributing causes (ICD-10) |  |  | Multiple sources |
| death_date | date | Date of death |  |  | Multiple sources |
| registration_type | DeathRegistrationType | Type of death certification |  |  | Multiple sources |
| autopsy_performed | Optional[bool] | Whether autopsy was performed |  |  | Multiple sources |
| is_external_cause | bool | Whether death was due to external causes |  | true, false | Multiple sources |
| is_injury_death | bool | Whether death was injury-related |  | true, false | Multiple sources |
| is_suicide | bool | Whether death was suicide |  | true, false | Multiple sources |
| is_accident | bool | Whether death was accidental |  | true, false | Multiple sources |
| is_assault | bool | Whether death was due to assault |  | true, false | Multiple sources |
| years_of_life_lost | Optional[float] | Years of potential life lost |  |  | Multiple sources |
| age_standardised_flag | bool | Whether record is included in age-standardised rates |  | true, false | Multiple sources |
| data_source | DataSource | Source of mortality data |  |  | Multiple sources |

### MortalityStatistics

Schema for aggregated mortality statistics.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_name | str | Geographic area name |  | Example text, Sample value, Data entry | Multiple sources |
| reference_year | int | Reference year |  | 123, 456, 789 | Multiple sources |
| population_base | int | Population base for rate calculation |  | 5432, 12876, 3241 | Multiple sources |
| total_deaths | int | Total number of deaths |  | 123, 456, 789 | Multiple sources |
| male_deaths | int | Deaths among males |  | 123, 456, 789 | Multiple sources |
| female_deaths | int | Deaths among females |  | 123, 456, 789 | Multiple sources |
| deaths_0_4 | int | Deaths aged 0-4 years |  | 123, 456, 789 | Multiple sources |
| deaths_5_14 | int | Deaths aged 5-14 years |  | 123, 456, 789 | Multiple sources |
| deaths_15_24 | int | Deaths aged 15-24 years |  | 123, 456, 789 | Multiple sources |
| deaths_25_34 | int | Deaths aged 25-34 years |  | 123, 456, 789 | Multiple sources |
| deaths_35_44 | int | Deaths aged 35-44 years |  | 123, 456, 789 | Multiple sources |
| deaths_45_54 | int | Deaths aged 45-54 years |  | 123, 456, 789 | Multiple sources |
| deaths_55_64 | int | Deaths aged 55-64 years |  | 123, 456, 789 | Multiple sources |
| deaths_65_74 | int | Deaths aged 65-74 years |  | 123, 456, 789 | Multiple sources |
| deaths_75_84 | int | Deaths aged 75-84 years |  | 123, 456, 789 | Multiple sources |
| deaths_85_plus | int | Deaths aged 85+ years |  | 123, 456, 789 | Multiple sources |
| cardiovascular_deaths | int | Deaths from cardiovascular disease |  | 123, 456, 789 | Multiple sources |
| cancer_deaths | int | Deaths from cancer |  | 123, 456, 789 | Multiple sources |
| respiratory_deaths | int | Deaths from respiratory disease |  | 123, 456, 789 | Multiple sources |
| external_deaths | int | Deaths from external causes |  | 123, 456, 789 | Multiple sources |
| dementia_deaths | int | Deaths from dementia |  | 123, 456, 789 | Multiple sources |
| diabetes_deaths | int | Deaths from diabetes |  | 123, 456, 789 | Multiple sources |
| kidney_disease_deaths | int | Deaths from kidney disease |  | 123, 456, 789 | Multiple sources |
| suicide_deaths | int | Deaths from suicide |  | 123, 456, 789 | Multiple sources |
| crude_death_rate | float | Crude death rate per 100,000 |  | 15.2, 8.7, 23.1 | Multiple sources |
| age_standardised_rate | Optional[float] | Age-standardised death rate |  |  | Multiple sources |
| infant_mortality_rate | Optional[float] | Infant mortality rate per 1,000 births |  |  | Australian Institute of Health and Welfare (AIHW) |
| life_expectancy_male | Optional[float] | Male life expectancy |  | 82.1, 79.8, 84.3 | Australian Institute of Health and Welfare (AIHW) |
| life_expectancy_female | Optional[float] | Female life expectancy |  | 82.1, 79.8, 84.3 | Australian Institute of Health and Welfare (AIHW) |
| total_yll | Optional[float] | Total years of life lost |  |  | Multiple sources |
| yll_rate | Optional[float] | YLL rate per 100,000 |  |  | Multiple sources |
| completeness_score | float | Data completeness percentage |  | 42.5, 67.8, 91.2 | Multiple sources |
| timeliness_score | Optional[float] | Data timeliness score |  |  | Multiple sources |
| data_source | DataSource | Source of mortality statistics |  |  | Multiple sources |

### MortalityTrend

Schema for mortality trend analysis over time.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level |  | Example text, Sample value, Data entry | Multiple sources |
| cause_of_death | str | Cause of death (ICD-10 or category) |  | Example text, Sample value, Data entry | Multiple sources |
| start_year | int | Start year of trend |  | 123, 456, 789 | Multiple sources |
| end_year | int | End year of trend |  | 123, 456, 789 | Multiple sources |
| data_points | List[Dict[str, Any]] | Yearly data points |  |  | Multiple sources |
| trend_direction | str | Overall trend direction (increasing/decreasing/stable) |  | Example text, Sample value, Data entry | Multiple sources |
| annual_change_rate | float | Average annual change rate (%) |  | 15.2, 8.7, 23.1 | Multiple sources |
| r_squared | Optional[float] | R-squared value for trend line |  |  | Multiple sources |
| statistical_significance | Optional[bool] | Whether trend is statistically significant |  |  | Multiple sources |
| has_breakpoints | bool | Whether trend has significant breakpoints |  | true, false | Multiple sources |
| breakpoint_years | List[int] | Years where trend changes |  |  | Multiple sources |

### HealthIndicatorSummary


    Standardised health indicator aggregations for analysis and reporting.
    
    Provides summary statistics and aggregated indicators suitable for
    comparative analysis across areas or time periods.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_level | str | Geographic aggregation level |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_name | str | Geographic area name |  | Example text, Sample value, Data entry | Multiple sources |
| reporting_period | TemporalData | Reporting period |  |  | Multiple sources |
| population_covered | int | Total population covered |  | 5432, 12876, 3241 | Multiple sources |
| population_density | float | Population density per sq km |  | 5432, 12876, 3241 | Multiple sources |
| health_outcome_score | Optional[float] | Composite health outcome score (0-100) |  |  | Australian Institute of Health and Welfare (AIHW) |
| mortality_indicators | Dict[str, float] | Key mortality indicators (age-standardised rates) |  |  | Australian Institute of Health and Welfare (AIHW) |
| avoidable_deaths_rate | Optional[float] | Avoidable deaths rate per 100,000 |  |  | Multiple sources |
| chronic_disease_burden | Dict[str, float] | Chronic disease prevalence summary |  |  | Multiple sources |
| disability_adjusted_life_years | Optional[float] | Disability-adjusted life years per 1,000 |  |  | Multiple sources |
| mental_health_indicators | Dict[str, float] | Mental health indicator summary |  |  | Australian Institute of Health and Welfare (AIHW) |
| healthcare_access_score | Optional[float] | Healthcare access composite score |  |  | Australian Institute of Health and Welfare (AIHW) |
| healthcare_utilisation_indicators | Dict[str, float] | Healthcare utilisation summary |  |  | Australian Institute of Health and Welfare (AIHW) |
| healthcare_quality_indicators | Dict[str, float] | Healthcare quality measures |  |  | Australian Institute of Health and Welfare (AIHW) |
| prevention_indicators | Dict[str, float] | Prevention and screening indicators |  |  | Multiple sources |
| immunisation_coverage | Optional[float] | Overall immunisation coverage % |  |  | Multiple sources |
| risk_factor_burden | Dict[str, float] | Risk factor prevalence summary |  |  | Multiple sources |
| modifiable_risk_score | Optional[float] | Modifiable risk factor burden score |  |  | Multiple sources |
| health_equity_indicators | Dict[str, float] | Health equity and disparity measures |  |  | Australian Institute of Health and Welfare (AIHW) |
| socioeconomic_health_gradient | Optional[float] | Health gradient across socioeconomic groups |  |  | Australian Institute of Health and Welfare (AIHW) |
| national_comparison | Dict[str, float] | Comparison to national averages (ratio) |  |  | Multiple sources |
| state_comparison | Dict[str, float] | Comparison to state averages (ratio) |  | NSW, VIC, QLD | Multiple sources |
| peer_group_comparison | Dict[str, float] | Comparison to similar areas |  |  | Multiple sources |
| trend_indicators | Dict[str, float] | Trend changes (annual % change) |  |  | Multiple sources |
| improvement_indicators | List[str] | Indicators showing improvement |  |  | Multiple sources |
| decline_indicators | List[str] | Indicators showing decline |  |  | Multiple sources |
| indicator_completeness | Dict[str, float] | Completeness % for each indicator category |  |  | Multiple sources |
| overall_completeness | float | Overall indicator completeness % |  | 42.5, 67.8, 91.2 | Multiple sources |
| quality_score | float | Overall data quality score |  | 42.5, 67.8, 91.2 | Multiple sources |

### MasterHealthRecord


    Master integrated health record combining all data sources.
    
    This is the primary target schema representing a complete health and
    demographic profile for a Statistical Area Level 2 (SA2).
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| sa2_code | str | 9-digit Statistical Area Level 2 code (primary key) (Statistical Area Level 2 code (ASGS 2021)) |  | 101011007, 201011021, 301011001 | Australian Bureau of Statistics (ABS) |
| sa2_name | str | Official SA2 name |  | Example text, Sample value, Data entry | Australian Bureau of Statistics (ABS) |
| geographic_hierarchy | Dict[str, str] | Complete geographic hierarchy (SA1s, SA3, SA4, State, Postcode) |  |  | Multiple sources |
| boundary_data | GeographicBoundary | Complete boundary geometry and metrics |  |  | Multiple sources |
| urbanisation | UrbanRuralClassification | Urban/rural classification |  |  | Multiple sources |
| remoteness_category | str | ABS Remoteness Area classification (ARIA (Accessibility/Remoteness Index of Australia)) |  | Example text, Sample value, Data entry | Multiple sources |
| demographic_profile | Dict[str, Any] | Complete demographic breakdown |  |  | Multiple sources |
| total_population | int | Total usual resident population |  | 5432, 12876, 3241 | Multiple sources |
| population_density_per_sq_km | float | Population density per square kilometre |  | 5432, 12876, 3241 | Multiple sources |
| median_age | Optional[float] | Median age of residents |  |  | Multiple sources |
| seifa_scores | Dict[SEIFAIndexType, float] | All SEIFA index scores (IRSD, IRSAD, IER, IEO) (Socio-Economic Indexes for Areas) |  | 1156, 987, 1034 | Australian Bureau of Statistics (ABS) |
| seifa_deciles | Dict[SEIFAIndexType, int] | National deciles for all SEIFA indexes |  | 1156, 987, 1034 | Australian Bureau of Statistics (ABS) |
| disadvantage_category | str | Overall disadvantage classification |  | Example text, Sample value, Data entry | Multiple sources |
| median_household_income | Optional[float] | Median weekly household income |  |  | Multiple sources |
| unemployment_rate | Optional[float] | Unemployment rate percentage |  |  | Multiple sources |
| health_outcomes_summary | Dict[str, float] | Summary of key health outcome indicators |  |  | Australian Institute of Health and Welfare (AIHW) |
| life_expectancy | Optional[Dict[str, float]] | Life expectancy by sex (years) (Life expectancy at birth) |  | 82.1, 79.8, 84.3 | Australian Institute of Health and Welfare (AIHW) |
| self_assessed_health | Optional[Dict[HealthOutcome, float]] | Distribution of self-assessed health outcomes (%) |  |  | Australian Institute of Health and Welfare (AIHW) |
| mortality_indicators | Dict[str, float] | Age-standardised mortality rates by major causes |  |  | Australian Institute of Health and Welfare (AIHW) |
| avoidable_mortality_rate | Optional[float] | Avoidable mortality rate per 100,000 |  |  | Australian Institute of Health and Welfare (AIHW) |
| infant_mortality_rate | Optional[float] | Infant mortality rate per 1,000 live births |  |  | Australian Institute of Health and Welfare (AIHW) |
| chronic_disease_prevalence | Dict[str, float] | Prevalence rates for major chronic diseases (%) |  |  | Multiple sources |
| mental_health_indicators | Dict[str, float] | Mental health prevalence and service utilisation rates |  |  | Australian Institute of Health and Welfare (AIHW) |
| psychological_distress_high | Optional[float] | High psychological distress prevalence (%) |  |  | Multiple sources |
| healthcare_access | Dict[str, Any] | Healthcare access and availability metrics |  |  | Australian Institute of Health and Welfare (AIHW) |
| gp_services_per_1000 | Optional[float] | GP services per 1,000 population |  |  | Multiple sources |
| specialist_services_per_1000 | Optional[float] | Specialist services per 1,000 population |  |  | Multiple sources |
| bulk_billing_rate | Optional[float] | Bulk billing rate percentage |  |  | Multiple sources |
| emergency_dept_presentations_per_1000 | Optional[float] | Emergency department presentations per 1,000 population |  |  | Multiple sources |
| pharmaceutical_utilisation | Dict[str, float] | PBS pharmaceutical utilisation rates |  |  | Multiple sources |
| risk_factors | Dict[str, float] | Prevalence of major modifiable risk factors (%) |  |  | Multiple sources |
| smoking_prevalence | Optional[float] | Current smoking prevalence (%) |  |  | Multiple sources |
| obesity_prevalence | Optional[float] | Obesity prevalence (%) |  |  | Multiple sources |
| physical_inactivity_prevalence | Optional[float] | Physical inactivity prevalence (%) |  |  | Multiple sources |
| harmful_alcohol_use_prevalence | Optional[float] | Harmful alcohol use prevalence (%) |  |  | Multiple sources |
| environmental_indicators | Dict[str, Any] | Environmental health indicators |  |  | Multiple sources |
| air_quality_index | Optional[float] | Average air quality index |  |  | Multiple sources |
| green_space_access | Optional[float] | Percentage with access to green space |  |  | Multiple sources |
| integration_level | DataIntegrationLevel | Level of data integration achieved |  |  | Multiple sources |
| data_completeness_score | float | Overall data completeness percentage |  | 42.5, 67.8, 91.2 | Multiple sources |
| integration_timestamp | datetime | When this integrated record was created |  |  | Multiple sources |
| source_datasets | List[str] | List of source datasets included in integration |  |  | Multiple sources |
| missing_indicators | List[str] | List of indicators that could not be integrated |  |  | Multiple sources |
| composite_health_index | Optional[float] | Composite health index score (0-100) |  |  | Australian Institute of Health and Welfare (AIHW) |
| health_inequality_index | Optional[float] | Health inequality index relative to national average |  |  | Australian Institute of Health and Welfare (AIHW) |

### SA2HealthProfile


    Complete SA2-level health and demographic profile.
    
    A focused view of health indicators and outcomes for a single SA2,
    suitable for health analysis and reporting.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| sa2_code | str | SA2 code (Statistical Area Level 2 code (ASGS 2021)) |  | 101011007, 201011021, 301011001 | Australian Bureau of Statistics (ABS) |
| sa2_name | str | SA2 name |  | Example text, Sample value, Data entry | Australian Bureau of Statistics (ABS) |
| reference_period | TemporalData | Data reference period |  |  | Multiple sources |
| total_population | int | Total population |  | 5432, 12876, 3241 | Multiple sources |
| population_by_age_sex | Dict[str, Dict[str, int]] | Population breakdown by age group and sex |  | 5432, 12876, 3241 | Multiple sources |
| indigenous_population_percentage | Optional[float] | Aboriginal and Torres Strait Islander population % |  | 5432, 12876, 3241 | Multiple sources |
| seifa_disadvantage_decile | int | SEIFA disadvantage decile (1=most disadvantaged) |  | 1156, 987, 1034 | Australian Bureau of Statistics (ABS) |
| socioeconomic_category | str | Socioeconomic classification (Most Disadvantaged to Least Disadvantaged) |  | Example text, Sample value, Data entry | Multiple sources |
| life_expectancy_male | Optional[float] | Male life expectancy (years) |  | 82.1, 79.8, 84.3 | Australian Institute of Health and Welfare (AIHW) |
| life_expectancy_female | Optional[float] | Female life expectancy (years) |  | 82.1, 79.8, 84.3 | Australian Institute of Health and Welfare (AIHW) |
| excellent_very_good_health_percentage | Optional[float] | % reporting excellent/very good health |  |  | Australian Institute of Health and Welfare (AIHW) |
| all_cause_mortality_rate | Optional[float] | All-cause age-standardised mortality rate |  |  | Australian Institute of Health and Welfare (AIHW) |
| cardiovascular_mortality_rate | Optional[float] | Cardiovascular disease mortality rate |  |  | Australian Institute of Health and Welfare (AIHW) |
| cancer_mortality_rate | Optional[float] | Cancer mortality rate |  |  | Australian Institute of Health and Welfare (AIHW) |
| respiratory_mortality_rate | Optional[float] | Respiratory disease mortality rate |  |  | Australian Institute of Health and Welfare (AIHW) |
| diabetes_mortality_rate | Optional[float] | Diabetes mortality rate |  |  | Australian Institute of Health and Welfare (AIHW) |
| suicide_mortality_rate | Optional[float] | Suicide mortality rate |  |  | Australian Institute of Health and Welfare (AIHW) |
| diabetes_prevalence | Optional[float] | Diabetes prevalence % |  |  | Multiple sources |
| hypertension_prevalence | Optional[float] | Hypertension prevalence % |  |  | Multiple sources |
| heart_disease_prevalence | Optional[float] | Heart disease prevalence % |  |  | Multiple sources |
| asthma_prevalence | Optional[float] | Asthma prevalence % |  |  | Multiple sources |
| copd_prevalence | Optional[float] | COPD prevalence % |  |  | Multiple sources |
| mental_health_condition_prevalence | Optional[float] | Mental health condition prevalence % |  |  | Australian Institute of Health and Welfare (AIHW) |
| gp_visits_per_capita | Optional[float] | GP visits per capita per year |  |  | Multiple sources |
| specialist_visits_per_capita | Optional[float] | Specialist visits per capita per year |  |  | Multiple sources |
| hospital_admissions_per_1000 | Optional[float] | Hospital admissions per 1,000 population |  |  | Multiple sources |
| emergency_presentations_per_1000 | Optional[float] | Emergency department presentations per 1,000 |  |  | Multiple sources |
| current_smoking_percentage | Optional[float] | Current smoking prevalence % |  |  | Multiple sources |
| obesity_percentage | Optional[float] | Obesity prevalence % |  |  | Multiple sources |
| overweight_obesity_percentage | Optional[float] | Overweight and obesity prevalence % |  |  | Multiple sources |
| physical_inactivity_percentage | Optional[float] | Physical inactivity prevalence % |  |  | Multiple sources |
| risky_alcohol_consumption_percentage | Optional[float] | Risky alcohol consumption prevalence % |  |  | Multiple sources |
| high_psychological_distress_percentage | Optional[float] | High psychological distress prevalence % |  |  | Multiple sources |
| bulk_billing_percentage | Optional[float] | Bulk billing rate % |  |  | Multiple sources |
| distance_to_nearest_hospital_km | Optional[float] | Distance to nearest hospital (km) |  |  | Multiple sources |
| gp_workforce_per_1000 | Optional[float] | GP workforce per 1,000 population |  |  | Multiple sources |
| infant_mortality_rate | Optional[float] | Infant mortality rate per 1,000 live births |  |  | Australian Institute of Health and Welfare (AIHW) |
| low_birth_weight_percentage | Optional[float] | Low birth weight percentage |  |  | Multiple sources |
| profile_completeness_score | float | Health profile data completeness % |  | 42.5, 67.8, 91.2 | Multiple sources |
| data_quality_flags | List[str] | Any data quality issues identified |  |  | Multiple sources |

### HealthDataAggregate

Schema for aggregated health data across multiple indicators.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level |  | Example text, Sample value, Data entry | Multiple sources |
| reporting_period | TemporalData | Reporting period |  |  | Multiple sources |
| total_indicators | int | Total number of indicators |  | 123, 456, 789 | Multiple sources |
| indicators_by_type | Dict[str, int] | Count of indicators by type |  |  | Multiple sources |
| population_health_score | Optional[float] | Overall population health score |  | 5432, 12876, 3241 | Australian Institute of Health and Welfare (AIHW) |
| top_mortality_causes | List[Dict[str, Any]] | Top causes of mortality |  |  | Australian Institute of Health and Welfare (AIHW) |
| top_morbidity_conditions | List[Dict[str, Any]] | Top morbidity conditions |  |  | Multiple sources |
| key_risk_factors | List[Dict[str, Any]] | Key risk factors |  |  | Multiple sources |
| data_completeness_score | float | Data completeness percentage |  | 42.5, 67.8, 91.2 | Multiple sources |
| missing_indicators | List[str] | List of missing key indicators |  |  | Multiple sources |

### HealthIndicator

Base schema for health indicator data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level (SA2, SA3, SA4, STATE) |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_name | str | Name of health indicator |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_code | str | Unique indicator code |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_type | HealthIndicatorType | Type of health indicator |  |  | Multiple sources |
| value | float | Indicator value |  | 42.5, 67.8, 91.2 | Multiple sources |
| unit | str | Unit of measurement |  | Example text, Sample value, Data entry | Multiple sources |
| confidence_interval_lower | Optional[float] | Lower CI bound |  |  | Multiple sources |
| confidence_interval_upper | Optional[float] | Upper CI bound |  |  | Multiple sources |
| standard_error | Optional[float] | Standard error |  |  | Multiple sources |
| sample_size | Optional[int] | Sample size if applicable |  |  | Multiple sources |
| age_group | AgeGroupType | Age group for this indicator |  |  | Multiple sources |
| sex | Optional[str] | Sex (Male/Female/Persons) |  |  | Multiple sources |
| suppressed | bool | Whether value is suppressed for privacy |  | true, false | Multiple sources |
| reliability | Optional[str] | Statistical reliability rating |  |  | Multiple sources |
| data_source | DataSource | Source of the health data |  |  | Multiple sources |

### HealthcareUtilisation

Schema for healthcare utilisation data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level (SA2, SA3, SA4, STATE) |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_name | str | Name of health indicator |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_code | str | Unique indicator code |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_type | HealthIndicatorType | Type of health indicator |  |  | Multiple sources |
| value | float | Indicator value |  | 42.5, 67.8, 91.2 | Multiple sources |
| unit | str | Unit of measurement |  | Example text, Sample value, Data entry | Multiple sources |
| confidence_interval_lower | Optional[float] | Lower CI bound |  |  | Multiple sources |
| confidence_interval_upper | Optional[float] | Upper CI bound |  |  | Multiple sources |
| standard_error | Optional[float] | Standard error |  |  | Multiple sources |
| sample_size | Optional[int] | Sample size if applicable |  |  | Multiple sources |
| age_group | AgeGroupType | Age group for this indicator |  |  | Multiple sources |
| sex | Optional[str] | Sex (Male/Female/Persons) |  |  | Multiple sources |
| suppressed | bool | Whether value is suppressed for privacy |  | true, false | Multiple sources |
| reliability | Optional[str] | Statistical reliability rating |  |  | Multiple sources |
| data_source | DataSource | Source of the health data |  |  | Multiple sources |
| service_type | str | Type of healthcare service |  | Example text, Sample value, Data entry | Multiple sources |
| service_category | str | Category of service |  | Example text, Sample value, Data entry | Multiple sources |
| visits_count | Optional[int] | Number of visits/services |  |  | Multiple sources |
| utilisation_rate | float | Utilisation rate |  | 15.2, 8.7, 23.1 | Multiple sources |
| total_cost | Optional[float] | Total cost if available |  |  | Multiple sources |
| average_cost_per_service | Optional[float] | Average cost |  |  | Multiple sources |
| bulk_billed_percentage | Optional[float] | Percentage bulk billed |  |  | Multiple sources |
| provider_type | Optional[str] | Type of healthcare provider |  |  | Multiple sources |
| average_wait_days | Optional[float] | Average wait time |  |  | Multiple sources |

### MentalHealthIndicator

Schema for mental health specific indicators.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level (SA2, SA3, SA4, STATE) |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_name | str | Name of health indicator |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_code | str | Unique indicator code |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_type | HealthIndicatorType | Type of health indicator |  |  | Multiple sources |
| value | float | Indicator value |  | 42.5, 67.8, 91.2 | Multiple sources |
| unit | str | Unit of measurement |  | Example text, Sample value, Data entry | Multiple sources |
| confidence_interval_lower | Optional[float] | Lower CI bound |  |  | Multiple sources |
| confidence_interval_upper | Optional[float] | Upper CI bound |  |  | Multiple sources |
| standard_error | Optional[float] | Standard error |  |  | Multiple sources |
| sample_size | Optional[int] | Sample size if applicable |  |  | Multiple sources |
| age_group | AgeGroupType | Age group for this indicator |  |  | Multiple sources |
| sex | Optional[str] | Sex (Male/Female/Persons) |  |  | Multiple sources |
| suppressed | bool | Whether value is suppressed for privacy |  | true, false | Multiple sources |
| reliability | Optional[str] | Statistical reliability rating |  |  | Multiple sources |
| data_source | DataSource | Source of the health data |  |  | Multiple sources |
| condition_name | str | Mental health condition name |  | Example text, Sample value, Data entry | Multiple sources |
| condition_category | str | Category of mental health condition |  | Example text, Sample value, Data entry | Multiple sources |
| severity_distribution | Optional[Dict[str, float]] | Distribution across severity levels |  |  | Multiple sources |
| functional_impact_score | Optional[float] | Functional impact score |  |  | Multiple sources |
| treatment_rate | Optional[float] | % receiving treatment |  |  | Multiple sources |
| medication_rate | Optional[float] | % on medication |  |  | Multiple sources |
| therapy_rate | Optional[float] | % receiving therapy |  |  | Multiple sources |
| recovery_rate | Optional[float] | Recovery rate % |  |  | Multiple sources |
| relapse_rate | Optional[float] | Relapse rate % |  |  | Multiple sources |

### MortalityData

Schema for mortality-specific data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level (SA2, SA3, SA4, STATE) |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_name | str | Name of health indicator |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_code | str | Unique indicator code |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_type | HealthIndicatorType | Type of health indicator |  |  | Multiple sources |
| value | float | Indicator value |  | 42.5, 67.8, 91.2 | Multiple sources |
| unit | str | Unit of measurement |  | Example text, Sample value, Data entry | Multiple sources |
| confidence_interval_lower | Optional[float] | Lower CI bound |  |  | Multiple sources |
| confidence_interval_upper | Optional[float] | Upper CI bound |  |  | Multiple sources |
| standard_error | Optional[float] | Standard error |  |  | Multiple sources |
| sample_size | Optional[int] | Sample size if applicable |  |  | Multiple sources |
| age_group | AgeGroupType | Age group for this indicator |  |  | Multiple sources |
| sex | Optional[str] | Sex (Male/Female/Persons) |  |  | Multiple sources |
| suppressed | bool | Whether value is suppressed for privacy |  | true, false | Multiple sources |
| reliability | Optional[str] | Statistical reliability rating |  |  | Multiple sources |
| data_source | DataSource | Source of the health data |  |  | Multiple sources |
| cause_of_death | str | ICD-10 cause of death category |  | Example text, Sample value, Data entry | Multiple sources |
| icd10_code | Optional[str] | Specific ICD-10 code |  |  | Multiple sources |
| deaths_count | Optional[int] | Number of deaths |  |  | Multiple sources |
| years_of_life_lost | Optional[float] | Years of life lost |  |  | Multiple sources |
| age_standardised_rate | Optional[float] | Age-standardised rate |  |  | Multiple sources |
| crude_rate | Optional[float] | Crude mortality rate |  |  | Multiple sources |
| is_premature | bool | Whether classified as premature death |  | true, false | Multiple sources |
| preventable | bool | Whether death is preventable |  | true, false | Multiple sources |

## Other

### ClimateStatistics

Schema for aggregated climate statistics.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_name | str | Geographic area name |  | Example text, Sample value, Data entry | Multiple sources |
| reference_period | str | Reference period (e.g., '1991-2020') |  | Example text, Sample value, Data entry | Multiple sources |
| statistic_period | str | Period for statistics (monthly/seasonal/annual) |  | Example text, Sample value, Data entry | Multiple sources |
| month | Optional[int] | Month (if monthly statistics) |  |  | Multiple sources |
| season | Optional[str] | Season (if seasonal statistics) |  |  | Multiple sources |
| mean_max_temperature | Optional[float] | Mean maximum temperature |  |  | Multiple sources |
| mean_min_temperature | Optional[float] | Mean minimum temperature |  |  | Multiple sources |
| mean_temperature | Optional[float] | Mean temperature |  |  | Multiple sources |
| highest_temperature | Optional[float] | Highest temperature on record |  |  | Multiple sources |
| lowest_temperature | Optional[float] | Lowest temperature on record |  |  | Multiple sources |
| days_over_35c | Optional[float] | Average days over 35°C |  |  | Multiple sources |
| days_over_40c | Optional[float] | Average days over 40°C |  |  | Multiple sources |
| days_under_0c | Optional[float] | Average days under 0°C |  |  | Multiple sources |
| days_under_minus5c | Optional[float] | Average days under -5°C |  |  | Multiple sources |
| mean_rainfall | Optional[float] | Mean rainfall |  |  | Bureau of Meteorology (BOM) |
| median_rainfall | Optional[float] | Median rainfall |  |  | Bureau of Meteorology (BOM) |
| rainfall_decile_1 | Optional[float] | 10th percentile rainfall |  |  | Bureau of Meteorology (BOM) |
| rainfall_decile_9 | Optional[float] | 90th percentile rainfall |  |  | Bureau of Meteorology (BOM) |
| highest_daily_rainfall | Optional[float] | Highest daily rainfall |  |  | Bureau of Meteorology (BOM) |
| mean_rain_days | Optional[float] | Average number of rain days |  |  | Multiple sources |
| days_over_1mm | Optional[float] | Days with >1mm rain |  |  | Multiple sources |
| days_over_10mm | Optional[float] | Days with >10mm rain |  |  | Multiple sources |
| days_over_25mm | Optional[float] | Days with >25mm rain |  |  | Multiple sources |
| mean_humidity_9am | Optional[float] | Mean 9am humidity |  |  | Multiple sources |
| mean_humidity_3pm | Optional[float] | Mean 3pm humidity |  |  | Multiple sources |
| mean_pressure | Optional[float] | Mean sea level pressure |  |  | Multiple sources |
| mean_wind_speed | Optional[float] | Mean wind speed |  |  | Multiple sources |
| prevailing_wind_direction | Optional[str] | Prevailing wind direction |  |  | Multiple sources |
| highest_gust | Optional[float] | Highest wind gust recorded |  |  | Multiple sources |
| mean_sunshine_hours | Optional[float] | Mean daily sunshine hours |  |  | Multiple sources |
| mean_solar_radiation | Optional[float] | Mean solar radiation |  |  | Multiple sources |
| mean_evaporation | Optional[float] | Mean daily evaporation |  |  | Multiple sources |
| heat_wave_days | Optional[float] | Average heat wave days |  |  | Multiple sources |
| frost_days | Optional[float] | Average frost days |  |  | Multiple sources |
| growing_degree_days | Optional[float] | Growing degree days |  |  | Multiple sources |
| temperature_completeness | float | Temperature data completeness % |  | 42.5, 67.8, 91.2 | Multiple sources |
| rainfall_completeness | float | Rainfall data completeness % |  | 42.5, 67.8, 91.2 | Bureau of Meteorology (BOM) |
| overall_completeness | float | Overall data completeness % |  | 42.5, 67.8, 91.2 | Multiple sources |
| data_source | DataSource | Source of climate statistics |  |  | Multiple sources |

### DataSource

Information about data source and provenance.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| source_name | str | Name of data source |  | Example text, Sample value, Data entry | Multiple sources |
| source_url | Optional[str] | URL of data source |  |  | Multiple sources |
| source_date | datetime | Date when data was sourced |  |  | Multiple sources |
| source_version | Optional[str] | Version of source data |  |  | Multiple sources |
| attribution | str | Required attribution text |  | Example text, Sample value, Data entry | Multiple sources |
| license | Optional[str] | Data license information |  |  | Multiple sources |

### TemporalData

Base model for time-series data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |

### VersionedSchema


    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |

### WeatherObservation

Schema for individual weather observations.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| station_id | str | BOM station identifier |  | Example text, Sample value, Data entry | Multiple sources |
| station_name | str | Weather station name |  | Example text, Sample value, Data entry | Multiple sources |
| station_type | WeatherStationType | Type of weather station |  |  | Multiple sources |
| latitude | float | Station latitude |  | 42.5, 67.8, 91.2 | Multiple sources |
| longitude | float | Station longitude |  | 42.5, 67.8, 91.2 | Multiple sources |
| elevation | float | Station elevation (metres above sea level) |  | 42.5, 67.8, 91.2 | Multiple sources |
| observation_date | date | Date of observation |  |  | Multiple sources |
| observation_time | Optional[str] | Time of observation (24hr format) |  |  | Multiple sources |
| max_temperature | Optional[float] | Maximum temperature |  |  | Multiple sources |
| min_temperature | Optional[float] | Minimum temperature |  |  | Multiple sources |
| temperature_9am | Optional[float] | Temperature at 9am |  |  | Multiple sources |
| temperature_3pm | Optional[float] | Temperature at 3pm |  |  | Multiple sources |
| rainfall_24hr | Optional[float] | 24-hour rainfall |  |  | Bureau of Meteorology (BOM) |
| rainfall_days | Optional[int] | Number of rain days in period |  |  | Bureau of Meteorology (BOM) |
| humidity_9am | Optional[float] | Relative humidity at 9am |  |  | Multiple sources |
| humidity_3pm | Optional[float] | Relative humidity at 3pm |  |  | Multiple sources |
| wind_speed_9am | Optional[float] | Wind speed at 9am (km/h) |  |  | Multiple sources |
| wind_speed_3pm | Optional[float] | Wind speed at 3pm (km/h) |  |  | Multiple sources |
| wind_direction_9am | Optional[int] | Wind direction at 9am (degrees) |  |  | Multiple sources |
| wind_direction_3pm | Optional[int] | Wind direction at 3pm (degrees) |  |  | Multiple sources |
| wind_gust_speed | Optional[float] | Maximum wind gust speed (km/h) |  |  | Multiple sources |
| wind_gust_direction | Optional[int] | Wind gust direction (degrees) |  |  | Multiple sources |
| pressure_msl_9am | Optional[float] | MSL pressure at 9am |  |  | Multiple sources |
| pressure_msl_3pm | Optional[float] | MSL pressure at 3pm |  |  | Multiple sources |
| cloud_cover_9am | Optional[int] | Cloud cover at 9am (octas) |  |  | Multiple sources |
| cloud_cover_3pm | Optional[int] | Cloud cover at 3pm (octas) |  |  | Multiple sources |
| visibility_9am | Optional[float] | Visibility at 9am (km) |  |  | Multiple sources |
| visibility_3pm | Optional[float] | Visibility at 3pm (km) |  |  | Multiple sources |
| sunshine_hours | Optional[float] | Daily sunshine hours |  |  | Multiple sources |
| solar_radiation | Optional[float] | Solar radiation (MJ/m²) |  |  | Multiple sources |
| evaporation | Optional[float] | Class A pan evaporation (mm) |  |  | Multiple sources |
| max_temp_quality | ObservationQuality |  |  |  | Multiple sources |
| min_temp_quality | ObservationQuality |  |  |  | Multiple sources |
| rainfall_quality | ObservationQuality |  |  |  | Bureau of Meteorology (BOM) |
| data_source | DataSource | Source of weather observation |  |  | Multiple sources |

### DataCompletenessRequirement


    Data completeness requirements by field and context.
    
    Defines specific completeness requirements for different
    data fields based on their importance and usage context.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| requirement_name | str | Name of completeness requirement |  | Example text, Sample value, Data entry | Multiple sources |
| data_domain | str | Data domain (health, demographic, geographic) |  | Example text, Sample value, Data entry | Multiple sources |
| context | str | Usage context (analysis, reporting, operational) |  | Example text, Sample value, Data entry | Multiple sources |
| field_completeness_requirements | Dict[str, Dict[str, Any]] | Completeness requirements by field |  |  | Multiple sources |
| conditional_requirements | List[Dict[str, Any]] | Completeness requirements based on conditions |  |  | Multiple sources |
| business_critical_fields | List[str] | Fields critical for business operations |  |  | Multiple sources |
| analysis_required_fields | List[str] | Fields required for statistical analysis |  |  | Multiple sources |
| regulatory_required_fields | List[str] | Fields required for regulatory compliance |  |  | Multiple sources |
| critical_completeness_threshold | float | Completeness threshold for critical fields (%) |  | 42.5, 67.8, 91.2 | Multiple sources |
| high_completeness_threshold | float | Completeness threshold for high-priority fields (%) |  | 42.5, 67.8, 91.2 | Multiple sources |
| medium_completeness_threshold | float | Completeness threshold for medium-priority fields (%) |  | 42.5, 67.8, 91.2 | Multiple sources |
| low_completeness_threshold | float | Completeness threshold for low-priority fields (%) |  | 42.5, 67.8, 91.2 | Multiple sources |
| allowed_exceptions | List[Dict[str, Any]] | Allowed exceptions to completeness requirements |  |  | Multiple sources |
| seasonal_adjustments | Dict[str, float] | Seasonal adjustments to completeness thresholds |  |  | Multiple sources |

### QualityStandardsRegistry


    Registry of all quality standards and requirements.
    
    Central registry managing all quality standards, thresholds,
    and validation requirements for the AHGD project.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| registry_version | str | Version of quality standards registry |  | Example text, Sample value, Data entry | Multiple sources |
| last_updated | datetime | When registry was last updated |  |  | Multiple sources |
| compliance_standards | List[str] | IDs of registered compliance standards |  |  | Multiple sources |
| completeness_requirements | List[str] | IDs of registered completeness requirements |  |  | Multiple sources |
| statistical_thresholds | List[str] | IDs of registered statistical thresholds |  |  | Multiple sources |
| geographic_requirements | List[str] | IDs of registered geographic requirements |  |  | Multiple sources |
| default_quality_level | QualityLevel | Default quality level when not specified |  |  | Multiple sources |
| validation_severity_mapping | Dict[str, ValidationSeverity] | Mapping of validation types to severity levels |  |  | Multiple sources |
| quality_reporting_schedule | str | Quality reporting schedule |  | Example text, Sample value, Data entry | Multiple sources |
| stakeholder_notifications | List[str] | Stakeholders to notify of quality issues |  |  | Multiple sources |
| registered_exemptions | List[Dict[str, Any]] | Registered exemptions from quality standards |  |  | Multiple sources |
| temporary_waivers | List[Dict[str, Any]] | Temporary waivers from specific requirements |  |  | Multiple sources |

### StatisticalValidationThreshold


    Statistical validation thresholds and rules.
    
    Defines statistical validation requirements including
    outlier detection, distribution checks, and relationship validation.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| threshold_name | str | Name of statistical threshold |  | Example text, Sample value, Data entry | Multiple sources |
| statistical_method | str | Statistical method or test |  | Example text, Sample value, Data entry | Multiple sources |
| applicable_fields | List[str] | Fields this threshold applies to |  |  | Multiple sources |
| outlier_detection_method | str | Outlier detection method (iqr, zscore, modified_zscore) |  | Example text, Sample value, Data entry | Multiple sources |
| outlier_threshold | float | Outlier detection threshold |  | 42.5, 67.8, 91.2 | Multiple sources |
| outlier_action | str | Action for outliers (flag, exclude, investigate) |  | Example text, Sample value, Data entry | Multiple sources |
| expected_distribution | Optional[str] | Expected statistical distribution (normal, poisson, etc.) |  |  | Multiple sources |
| distribution_test | Optional[str] | Statistical test for distribution (shapiro, kolmogorov) |  |  | Multiple sources |
| distribution_p_value_threshold | float | P-value threshold for distribution tests |  | 42.5, 67.8, 91.2 | Multiple sources |
| minimum_value | Optional[float] | Minimum allowed value |  |  | Multiple sources |
| maximum_value | Optional[float] | Maximum allowed value |  |  | Multiple sources |
| expected_mean_range | Optional[Dict[str, float]] | Expected range for mean values |  |  | Multiple sources |
| expected_std_range | Optional[Dict[str, float]] | Expected range for standard deviation |  |  | Multiple sources |
| correlation_checks | List[Dict[str, Any]] | Correlation validation rules |  |  | Multiple sources |
| ratio_checks | List[Dict[str, Any]] | Ratio validation rules |  |  | Multiple sources |
| trend_checks | List[Dict[str, Any]] | Trend validation rules |  |  | Multiple sources |
| temporal_consistency_checks | List[Dict[str, Any]] | Temporal consistency validation rules |  |  | Multiple sources |
| seasonality_checks | bool | Whether to perform seasonality validation |  | true, false | Multiple sources |
| minimum_sample_size | Optional[int] | Minimum sample size for valid statistics |  |  | Multiple sources |
| confidence_level | float | Required confidence level (%) |  | 42.5, 67.8, 91.2 | Multiple sources |
| margin_of_error | float | Acceptable margin of error (%) |  | 42.5, 67.8, 91.2 | Multiple sources |
| severity_thresholds | Dict[str, float] | Severity thresholds for different validation failures |  |  | Multiple sources |

### VersionedSchema


    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |

### CensusEducation

Schema for census education data (Tables G16, G18).

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level |  | Example text, Sample value, Data entry | Multiple sources |
| census_year | int | Census year |  | 123, 456, 789 | Australian Bureau of Statistics (ABS) |
| education_pop_base | int | Population base for education data |  | 123, 456, 789 | Multiple sources |
| year_12_or_equivalent | int | Completed Year 12 or equivalent |  | 123, 456, 789 | Multiple sources |
| year_11_or_equivalent | int | Completed Year 11 or equivalent |  | 123, 456, 789 | Multiple sources |
| year_10_or_equivalent | int | Completed Year 10 or equivalent |  | 123, 456, 789 | Multiple sources |
| year_9_or_equivalent | int | Completed Year 9 or equivalent |  | 123, 456, 789 | Multiple sources |
| year_8_or_below | int | Year 8 or below |  | 123, 456, 789 | Multiple sources |
| did_not_go_to_school | int | Did not go to school |  | 123, 456, 789 | Multiple sources |
| schooling_not_stated | int | Schooling level not stated |  | NSW, VIC, QLD | Multiple sources |
| postgraduate_degree | int | Postgraduate degree level |  | 123, 456, 789 | Multiple sources |
| graduate_diploma | int | Graduate diploma and graduate certificate |  | 123, 456, 789 | Multiple sources |
| bachelor_degree | int | Bachelor degree level |  | 123, 456, 789 | Multiple sources |
| advanced_diploma | int | Advanced diploma and diploma level |  | 123, 456, 789 | Multiple sources |
| certificate_iii_iv | int | Certificate III & IV level |  | 123, 456, 789 | Multiple sources |
| certificate_i_ii | int | Certificate I & II level |  | 123, 456, 789 | Multiple sources |
| no_qualification | int | No non-school qualification |  | 123, 456, 789 | Multiple sources |
| qualification_not_stated | int | Qualification not stated |  | NSW, VIC, QLD | Multiple sources |
| natural_physical_sciences | int | Natural and physical sciences |  | 123, 456, 789 | Multiple sources |
| information_technology | int | Information technology |  | 123, 456, 789 | Multiple sources |
| engineering | int | Engineering and related technologies |  | 123, 456, 789 | Multiple sources |
| architecture_building | int | Architecture and building |  | 123, 456, 789 | Multiple sources |
| agriculture | int | Agriculture, environmental and related |  | 123, 456, 789 | Multiple sources |
| health | int | Health |  | 123, 456, 789 | Australian Institute of Health and Welfare (AIHW) |
| education | int | Education |  | 123, 456, 789 | Multiple sources |
| management_commerce | int | Management and commerce |  | 123, 456, 789 | Multiple sources |
| society_culture | int | Society and culture |  | 123, 456, 789 | Multiple sources |
| creative_arts | int | Creative arts |  | 123, 456, 789 | Multiple sources |
| food_hospitality | int | Food, hospitality and personal services |  | 123, 456, 789 | Multiple sources |
| mixed_field | int | Mixed field programmes |  | 123, 456, 789 | Multiple sources |
| field_not_stated | int | Field of study not stated |  | NSW, VIC, QLD | Multiple sources |
| data_source | DataSource | Source of education data |  |  | Multiple sources |

### CensusEmployment

Schema for census employment data (Tables G17, G43-G51).

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level |  | Example text, Sample value, Data entry | Multiple sources |
| census_year | int | Census year |  | 123, 456, 789 | Australian Bureau of Statistics (ABS) |
| labour_force_pop | int | Population aged 15+ for labour force |  | 123, 456, 789 | Multiple sources |
| employed_full_time | int | Employed full-time |  | 123, 456, 789 | Multiple sources |
| employed_part_time | int | Employed part-time |  | 123, 456, 789 | Multiple sources |
| unemployed | int | Unemployed |  | 123, 456, 789 | Multiple sources |
| not_in_labour_force | int | Not in labour force |  | 123, 456, 789 | Multiple sources |
| labour_force_not_stated | int | Labour force status not stated |  | NSW, VIC, QLD | Multiple sources |
| agriculture_forestry_fishing | int | Agriculture, forestry and fishing |  | 123, 456, 789 | Multiple sources |
| mining | int | Mining |  | 123, 456, 789 | Multiple sources |
| manufacturing | int | Manufacturing |  | 123, 456, 789 | Multiple sources |
| electricity_gas_water | int | Electricity, gas, water and waste services |  | 123, 456, 789 | Multiple sources |
| construction | int | Construction |  | 123, 456, 789 | Multiple sources |
| wholesale_trade | int | Wholesale trade |  | 123, 456, 789 | Multiple sources |
| retail_trade | int | Retail trade |  | 123, 456, 789 | Multiple sources |
| accommodation_food | int | Accommodation and food services |  | 123, 456, 789 | Multiple sources |
| transport_postal | int | Transport, postal and warehousing |  | 123, 456, 789 | Multiple sources |
| information_media | int | Information media and telecommunications |  | 123, 456, 789 | Multiple sources |
| financial_insurance | int | Financial and insurance services |  | 123, 456, 789 | Multiple sources |
| rental_real_estate | int | Rental, hiring and real estate services |  | NSW, VIC, QLD | Multiple sources |
| professional_services | int | Professional, scientific and technical services |  | 123, 456, 789 | Multiple sources |
| administrative_support | int | Administrative and support services |  | 123, 456, 789 | Multiple sources |
| public_administration | int | Public administration and safety |  | 123, 456, 789 | Multiple sources |
| education_training | int | Education and training |  | 123, 456, 789 | Multiple sources |
| health_social_assistance | int | Health care and social assistance |  | 123, 456, 789 | Australian Institute of Health and Welfare (AIHW) |
| arts_recreation | int | Arts and recreation services |  | 123, 456, 789 | Multiple sources |
| other_services | int | Other services |  | 123, 456, 789 | Multiple sources |
| industry_not_stated | int | Industry not stated |  | NSW, VIC, QLD | Multiple sources |
| managers | int | Managers |  | 123, 456, 789 | Multiple sources |
| professionals | int | Professionals |  | 123, 456, 789 | Multiple sources |
| technicians_trades | int | Technicians and trades workers |  | 123, 456, 789 | Multiple sources |
| community_personal_service | int | Community and personal service workers |  | 123, 456, 789 | Multiple sources |
| clerical_administrative | int | Clerical and administrative workers |  | 123, 456, 789 | Multiple sources |
| sales_workers | int | Sales workers |  | 123, 456, 789 | Multiple sources |
| machinery_operators | int | Machinery operators and drivers |  | 123, 456, 789 | Multiple sources |
| labourers | int | Labourers |  | 123, 456, 789 | Multiple sources |
| occupation_not_stated | int | Occupation not stated |  | NSW, VIC, QLD | Multiple sources |
| data_source | DataSource | Source of employment data |  |  | Multiple sources |

### CensusHousing

Schema for census housing and dwelling data (Tables G31-G42).

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level |  | Example text, Sample value, Data entry | Multiple sources |
| census_year | int | Census year |  | 123, 456, 789 | Australian Bureau of Statistics (ABS) |
| separate_house | int | Separate house |  | 123, 456, 789 | Multiple sources |
| semi_detached | int | Semi-detached, row or terrace house |  | 123, 456, 789 | Multiple sources |
| flat_apartment | int | Flat or apartment |  | 123, 456, 789 | Multiple sources |
| other_dwelling | int | Other dwelling |  | 123, 456, 789 | Multiple sources |
| dwelling_structure_not_stated | int | Dwelling structure not stated |  | NSW, VIC, QLD | Multiple sources |
| owned_outright | int | Owned outright |  | 123, 456, 789 | Multiple sources |
| owned_with_mortgage | int | Owned with a mortgage |  | 123, 456, 789 | Multiple sources |
| rented | int | Rented |  | 123, 456, 789 | Multiple sources |
| other_tenure | int | Other tenure type |  | 123, 456, 789 | Multiple sources |
| tenure_not_stated | int | Tenure type not stated |  | NSW, VIC, QLD | Multiple sources |
| state_territory_housing | int | State or territory housing authority |  | NSW, VIC, QLD | Multiple sources |
| private_landlord | int | Private landlord |  | 123, 456, 789 | Multiple sources |
| real_estate_agent | int | Real estate agent |  | NSW, VIC, QLD | Multiple sources |
| other_landlord | int | Other landlord type |  | 123, 456, 789 | Multiple sources |
| landlord_not_stated | int | Landlord type not stated |  | NSW, VIC, QLD | Multiple sources |
| no_bedrooms | int | No bedrooms (bed-sitters etc) |  | 123, 456, 789 | Multiple sources |
| one_bedroom | int | 1 bedroom |  | 123, 456, 789 | Multiple sources |
| two_bedrooms | int | 2 bedrooms |  | 123, 456, 789 | Multiple sources |
| three_bedrooms | int | 3 bedrooms |  | 123, 456, 789 | Multiple sources |
| four_bedrooms | int | 4 bedrooms |  | 123, 456, 789 | Multiple sources |
| five_plus_bedrooms | int | 5 or more bedrooms |  | 123, 456, 789 | Multiple sources |
| bedrooms_not_stated | int | Number of bedrooms not stated |  | NSW, VIC, QLD | Multiple sources |
| median_mortgage_monthly | Optional[int] | Median monthly mortgage payment |  |  | Multiple sources |
| median_rent_weekly | Optional[int] | Median weekly rent |  |  | Multiple sources |
| internet_connection | int | Dwellings with internet connection |  | 123, 456, 789 | Multiple sources |
| no_internet | int | Dwellings without internet |  | 123, 456, 789 | Multiple sources |
| internet_not_stated | int | Internet connection not stated |  | NSW, VIC, QLD | Multiple sources |
| no_motor_vehicles | int | No motor vehicles |  | 123, 456, 789 | Multiple sources |
| one_motor_vehicle | int | 1 motor vehicle |  | 123, 456, 789 | Multiple sources |
| two_motor_vehicles | int | 2 motor vehicles |  | 123, 456, 789 | Multiple sources |
| three_plus_vehicles | int | 3 or more motor vehicles |  | 123, 456, 789 | Multiple sources |
| vehicles_not_stated | int | Number of vehicles not stated |  | NSW, VIC, QLD | Multiple sources |
| data_source | DataSource | Source of housing data |  |  | Multiple sources |

### DataSource

Information about data source and provenance.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| source_name | str | Name of data source |  | Example text, Sample value, Data entry | Multiple sources |
| source_url | Optional[str] | URL of data source |  |  | Multiple sources |
| source_date | datetime | Date when data was sourced |  |  | Multiple sources |
| source_version | Optional[str] | Version of source data |  |  | Multiple sources |
| attribution | str | Required attribution text |  | Example text, Sample value, Data entry | Multiple sources |
| license | Optional[str] | Data license information |  |  | Multiple sources |

### TemporalData

Base model for time-series data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |

### VersionedSchema


    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |

### DataSource

Information about data source and provenance.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| source_name | str | Name of data source |  | Example text, Sample value, Data entry | Multiple sources |
| source_url | Optional[str] | URL of data source |  |  | Multiple sources |
| source_date | datetime | Date when data was sourced |  |  | Multiple sources |
| source_version | Optional[str] | Version of source data |  |  | Multiple sources |
| attribution | str | Required attribution text |  | Example text, Sample value, Data entry | Multiple sources |
| license | Optional[str] | Data license information |  |  | Multiple sources |

### TemporalData

Base model for time-series data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |

### VersionedSchema


    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |

### DataSource

Information about data source and provenance.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| source_name | str | Name of data source |  | Example text, Sample value, Data entry | Multiple sources |
| source_url | Optional[str] | URL of data source |  |  | Multiple sources |
| source_date | datetime | Date when data was sourced |  |  | Multiple sources |
| source_version | Optional[str] | Version of source data |  |  | Multiple sources |
| attribution | str | Required attribution text |  | Example text, Sample value, Data entry | Multiple sources |
| license | Optional[str] | Data license information |  |  | Multiple sources |

### VersionedSchema


    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |

### APIResponseSchema


    Schema for API response structures.
    
    Defines standardised API response formats for different
    types of health data queries and endpoints.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| api_version | APIVersion | API version |  |  | Multiple sources |
| response_id | str | Unique response identifier |  | Example text, Sample value, Data entry | Multiple sources |
| timestamp | datetime | Response generation timestamp |  |  | Multiple sources |
| endpoint | str | API endpoint called |  | Example text, Sample value, Data entry | Multiple sources |
| method | str | HTTP method used |  | Example text, Sample value, Data entry | Multiple sources |
| query_parameters | Dict[str, Any] | Query parameters provided |  |  | Multiple sources |
| status | str | Response status (success, error, partial) |  | Example text, Sample value, Data entry | Multiple sources |
| status_code | int | HTTP status code |  | 123, 456, 789 | Multiple sources |
| message | Optional[str] | Status message |  |  | Multiple sources |
| data | Union[Dict[str, Any], List[Dict[str, Any]], NoneType] | Response data payload |  |  | Multiple sources |
| pagination | Optional[Dict[str, Any]] | Pagination information for large result sets |  |  | Multiple sources |
| metadata | Dict[str, Any] | Additional response metadata |  |  | Multiple sources |
| links | Dict[str, str] | Related API endpoints and resources |  |  | Multiple sources |
| errors | List[Dict[str, Any]] | Error details if status is error |  |  | Multiple sources |
| warnings | List[str] | Warning messages |  |  | Multiple sources |

### DataQualityReport


    Data quality report for exported datasets.
    
    Provides comprehensive quality assessment and metrics
    for exported data to ensure fitness for purpose.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| report_id | str | Unique report identifier |  | Example text, Sample value, Data entry | Multiple sources |
| dataset_name | str | Name of the assessed dataset |  | Example text, Sample value, Data entry | Multiple sources |
| assessment_date | datetime | Date of quality assessment |  |  | Multiple sources |
| assessment_scope | str | Scope of assessment (full, sample, targeted) |  | Example text, Sample value, Data entry | Multiple sources |
| records_assessed | int | Number of records assessed |  | 123, 456, 789 | Multiple sources |
| columns_assessed | int | Number of columns assessed |  | 123, 456, 789 | Multiple sources |
| overall_completeness | float | Overall data completeness percentage |  | 42.5, 67.8, 91.2 | Multiple sources |
| column_completeness | Dict[str, float] | Completeness percentage by column |  |  | Multiple sources |
| critical_field_completeness | Dict[str, float] | Completeness for business-critical fields |  |  | Multiple sources |
| accuracy_score | float | Overall accuracy score |  | 42.5, 67.8, 91.2 | Multiple sources |
| validation_rules_passed | int | Number of validation rules passed |  | 123, 456, 789 | Multiple sources |
| validation_rules_failed | int | Number of validation rules failed |  | 123, 456, 789 | Multiple sources |
| accuracy_issues | List[Dict[str, Any]] | Identified accuracy issues |  |  | Multiple sources |
| consistency_score | float | Data consistency score |  | 42.5, 67.8, 91.2 | Multiple sources |
| duplicate_records | int | Number of duplicate records found |  | 123, 456, 789 | Multiple sources |
| inconsistent_formats | Dict[str, int] | Format inconsistencies by column |  |  | Multiple sources |
| referential_integrity_issues | int | Referential integrity violations |  | 123, 456, 789 | Multiple sources |
| validity_score | float | Data validity score |  | 42.5, 67.8, 91.2 | Multiple sources |
| invalid_values | Dict[str, int] | Invalid values count by column |  |  | Multiple sources |
| range_violations | Dict[str, int] | Range constraint violations by column |  |  | Multiple sources |
| format_violations | Dict[str, int] | Format constraint violations by column |  |  | Multiple sources |
| timeliness_score | float | Data timeliness score |  | 42.5, 67.8, 91.2 | Multiple sources |
| data_freshness_days | float | Average data age in days |  | 42.5, 67.8, 91.2 | Multiple sources |
| outdated_records | int | Number of outdated records |  | 123, 456, 789 | Multiple sources |
| overall_quality_score | float | Overall data quality score |  | 42.5, 67.8, 91.2 | Multiple sources |
| quality_grade | str | Quality grade (A, B, C, D, F) |  | Example text, Sample value, Data entry | Multiple sources |
| fitness_for_purpose | str | Fitness assessment (excellent, good, adequate, poor) |  | Example text, Sample value, Data entry | Multiple sources |
| improvement_recommendations | List[str] | Recommendations for quality improvement |  |  | Multiple sources |
| priority_issues | List[str] | High-priority issues requiring attention |  |  | Multiple sources |
| compliance_standards | List[str] | Compliance standards assessed against |  |  | Multiple sources |
| compliance_status | Dict[str, str] | Compliance status by standard |  |  | Multiple sources |

### DataSource

Information about data source and provenance.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| source_name | str | Name of data source |  | Example text, Sample value, Data entry | Multiple sources |
| source_url | Optional[str] | URL of data source |  |  | Multiple sources |
| source_date | datetime | Date when data was sourced |  |  | Multiple sources |
| source_version | Optional[str] | Version of source data |  |  | Multiple sources |
| attribution | str | Required attribution text |  | Example text, Sample value, Data entry | Multiple sources |
| license | Optional[str] | Data license information |  |  | Multiple sources |

### DataWarehouseTable


    Schema for data warehouse table definitions.
    
    Defines the structure and properties of tables in the target
    data warehouse for analytics and reporting.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| table_name | str | Table name in data warehouse |  | Example text, Sample value, Data entry | Multiple sources |
| schema_name | str | Database schema name |  | Example text, Sample value, Data entry | Multiple sources |
| table_type | str | Table type (fact, dimension, aggregate) |  | Example text, Sample value, Data entry | Multiple sources |
| columns | List[Dict[str, Any]] | Column definitions with data types and constraints |  |  | Multiple sources |
| primary_keys | List[str] | Primary key column names |  |  | Multiple sources |
| foreign_keys | List[Dict[str, Any]] | Foreign key relationships |  |  | Multiple sources |
| indexes | List[Dict[str, Any]] | Index definitions for performance |  |  | Multiple sources |
| partition_strategy | Optional[str] | Partitioning strategy (date, range, hash) |  |  | Multiple sources |
| partition_columns | List[str] | Columns used for partitioning |  |  | Multiple sources |
| retention_period_days | Optional[int] | Data retention period in days |  |  | Multiple sources |
| archival_strategy | Optional[str] | Data archival strategy |  |  | Multiple sources |
| data_lineage | List[str] | Source tables/systems feeding this table |  |  | Multiple sources |
| refresh_frequency | str | Data refresh frequency (daily, weekly, monthly) |  | Example text, Sample value, Data entry | Multiple sources |
| quality_checks | List[str] | Data quality checks applied |  |  | Multiple sources |
| access_level | str | Access level (public, internal, restricted) |  | Example text, Sample value, Data entry | Multiple sources |
| authorized_users | List[str] | Authorized user groups |  |  | Multiple sources |

### ExportSpecification


    Specification for data export operations.
    
    Defines how data should be exported including format,
    compression, filtering, and destination details.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| export_name | str | Name of the export |  | Example text, Sample value, Data entry | Multiple sources |
| export_description | str | Description of export purpose |  | Example text, Sample value, Data entry | Multiple sources |
| export_type | str | Type of export (full, incremental, filtered) |  | Example text, Sample value, Data entry | Multiple sources |
| source_tables | List[str] | Source tables/views to export |  |  | Multiple sources |
| join_conditions | List[str] | SQL join conditions if multiple tables |  |  | Multiple sources |
| filter_conditions | List[str] | Filter conditions to apply |  |  | Multiple sources |
| output_format | ExportFormat | Output format |  |  | Multiple sources |
| compression | CompressionType | Compression type |  |  | Multiple sources |
| encoding | str | Character encoding |  | Example text, Sample value, Data entry | Multiple sources |
| file_naming_pattern | str | File naming pattern with placeholders |  | Example text, Sample value, Data entry | Multiple sources |
| max_file_size_mb | Optional[int] | Maximum file size before splitting |  |  | Multiple sources |
| include_headers | bool | Include column headers (for CSV/Excel) |  | true, false | Multiple sources |
| included_columns | List[str] | Specific columns to include (empty = all) |  |  | Multiple sources |
| excluded_columns | List[str] | Columns to exclude |  |  | Multiple sources |
| column_mappings | Dict[str, str] | Column name mappings (internal -> export) |  |  | Multiple sources |
| date_format | str | Date format pattern |  | Example text, Sample value, Data entry | Multiple sources |
| decimal_places | Optional[int] | Decimal places for numeric fields |  |  | Multiple sources |
| null_value_representation | str | How to represent null values |  | Example text, Sample value, Data entry | Multiple sources |
| destination_type | str | Destination type (local, s3, azure, http) |  | Example text, Sample value, Data entry | Multiple sources |
| destination_path | str | Destination path or URL |  | Example text, Sample value, Data entry | Multiple sources |
| destination_credentials | Optional[str] | Credentials reference (not the actual credentials) |  |  | Multiple sources |
| schedule_expression | Optional[str] | Cron expression for scheduled exports |  |  | Multiple sources |
| timezone | str | Timezone for scheduling |  | Example text, Sample value, Data entry | Multiple sources |
| row_count_validation | bool | Validate row count against source |  | true, false | Multiple sources |
| data_validation_rules | List[str] | Data validation rules to apply |  |  | Multiple sources |

### VersionedSchema


    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |

### WebPlatformDataStructure


    Data structure for web platform consumption.
    
    Defines optimised data structures for web dashboard
    and interactive visualisation components.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| content_type | str | Type of content (map, chart, table, summary) |  | Example text, Sample value, Data entry | Multiple sources |
| content_id | str | Unique content identifier |  | Example text, Sample value, Data entry | Multiple sources |
| content_title | str | Display title |  | Example text, Sample value, Data entry | Multiple sources |
| content_description | str | Content description |  | Example text, Sample value, Data entry | Multiple sources |
| data_structure | str | Data structure type (geojson, timeseries, matrix, hierarchical) |  | Example text, Sample value, Data entry | Multiple sources |
| data_payload | Dict[str, Any] | Optimised data payload for web consumption |  |  | Multiple sources |
| chart_config | Optional[Dict[str, Any]] | Chart.js or similar visualisation configuration |  |  | Multiple sources |
| map_config | Optional[Dict[str, Any]] | Leaflet or similar map configuration |  |  | Multiple sources |
| table_config | Optional[Dict[str, Any]] | Table display configuration |  |  | Multiple sources |
| interactive_elements | List[str] | Available interactive features |  |  | Multiple sources |
| filter_options | List[Dict[str, Any]] | Available filter controls |  |  | Multiple sources |
| drill_down_paths | List[str] | Available drill-down navigation paths |  |  | Multiple sources |
| cache_duration_seconds | int | Recommended cache duration |  | 123, 456, 789 | Multiple sources |
| lazy_loading | bool | Whether content supports lazy loading |  | true, false | Multiple sources |
| compression_applied | bool | Whether data is compressed |  | true, false | Multiple sources |
| responsive_breakpoints | Dict[str, Dict[str, Any]] | Responsive design configurations |  |  | Multiple sources |
| mobile_optimised | bool | Whether optimised for mobile devices |  | true, false | Multiple sources |
| accessibility_features | List[str] | Implemented accessibility features |  |  | Multiple sources |
| alt_text_descriptions | Dict[str, str] | Alternative text descriptions for screen readers |  |  | Multiple sources |
| data_last_updated | datetime | When underlying data was last updated |  |  | Multiple sources |
| content_generated | datetime | When this content structure was generated |  |  | Multiple sources |
| refresh_trigger | Optional[str] | What triggers content refresh |  |  | Multiple sources |

### DataSource

Information about data source and provenance.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| source_name | str | Name of data source |  | Example text, Sample value, Data entry | Multiple sources |
| source_url | Optional[str] | URL of data source |  |  | Multiple sources |
| source_date | datetime | Date when data was sourced |  |  | Multiple sources |
| source_version | Optional[str] | Version of source data |  |  | Multiple sources |
| attribution | str | Required attribution text |  | Example text, Sample value, Data entry | Multiple sources |
| license | Optional[str] | Data license information |  |  | Multiple sources |

### SA2Coordinates

Schema for SA2 coordinate data with validation.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| sa2_code | str | 9-digit SA2 code (Statistical Area Level 2 code (ASGS 2021)) |  | 101011007, 201011021, 301011001 | Australian Bureau of Statistics (ABS) |
| sa2_name | str | SA2 name |  | Example text, Sample value, Data entry | Australian Bureau of Statistics (ABS) |
| boundary_data | GeographicBoundary | Geographic boundary information |  |  | Multiple sources |
| population | Optional[int] | Population count |  | 5432, 12876, 3241 | Multiple sources |
| dwellings | Optional[int] | Number of dwellings |  |  | Multiple sources |
| neighbours | List[str] | List of neighbouring SA2 codes |  |  | Multiple sources |
| sa3_code | str | Parent SA3 code |  | Example text, Sample value, Data entry | Australian Bureau of Statistics (ABS) |
| sa4_code | str | Parent SA4 code |  | Example text, Sample value, Data entry | Australian Bureau of Statistics (ABS) |
| state_code | str | State/territory code |  | NSW, VIC, QLD | Multiple sources |
| data_source | DataSource | Source of the SA2 data |  |  | Multiple sources |

### SA2GeometryValidation

Extended schema for detailed SA2 geometry validation.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| sa2_code | str | 9-digit SA2 code (Statistical Area Level 2 code (ASGS 2021)) |  | 101011007, 201011021, 301011001 | Australian Bureau of Statistics (ABS) |
| is_valid_geometry | bool | Whether geometry is valid |  | true, false | Multiple sources |
| geometry_errors | List[str] | List of geometry validation errors |  |  | Multiple sources |
| is_simple | bool | Whether geometry is simple (no self-intersections) |  | true, false | Multiple sources |
| is_closed | bool | Whether all rings are properly closed |  | true, false | Multiple sources |
| has_holes | bool | Whether polygon has interior holes |  | true, false | Multiple sources |
| compactness_ratio | Optional[float] | Polsby-Popper compactness ratio |  |  | Multiple sources |
| coordinate_precision | int | Decimal places in coordinates |  | 123, 456, 789 | Multiple sources |

### VersionedSchema


    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |

### DataSource

Information about data source and provenance.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| source_name | str | Name of data source |  | Example text, Sample value, Data entry | Multiple sources |
| source_url | Optional[str] | URL of data source |  |  | Multiple sources |
| source_date | datetime | Date when data was sourced |  |  | Multiple sources |
| source_version | Optional[str] | Version of source data |  |  | Multiple sources |
| attribution | str | Required attribution text |  | Example text, Sample value, Data entry | Multiple sources |
| license | Optional[str] | Data license information |  |  | Multiple sources |

### TemporalData

Base model for time-series data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |

### VersionedSchema


    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |

### BaseSettings


    Base class for settings, allowing values to be overridden by environment variables.

    This is useful in production for secrets you do not wish to save in code, it plays nicely with docker(-compose),
    Heroku and any 12 factor app design.

    All the below attributes can be set via `model_config`.

    Args:
        _case_sensitive: Whether environment and CLI variable names should be read with case-sensitivity.
            Defaults to `None`.
        _nested_model_default_partial_update: Whether to allow partial updates on nested model default object fields.
            Defaults to `False`.
        _env_prefix: Prefix for all environment variables. Defaults to `None`.
        _env_file: The env file(s) to load settings values from. Defaults to `Path('')`, which
            means that the value from `model_config['env_file']` should be used. You can also pass
            `None` to indicate that environment variables should not be loaded from an env file.
        _env_file_encoding: The env file encoding, e.g. `'latin-1'`. Defaults to `None`.
        _env_ignore_empty: Ignore environment variables where the value is an empty string. Default to `False`.
        _env_nested_delimiter: The nested env values delimiter. Defaults to `None`.
        _env_nested_max_split: The nested env values maximum nesting. Defaults to `None`, which means no limit.
        _env_parse_none_str: The env string value that should be parsed (e.g. "null", "void", "None", etc.)
            into `None` type(None). Defaults to `None` type(None), which means no parsing should occur.
        _env_parse_enums: Parse enum field names to values. Defaults to `None.`, which means no parsing should occur.
        _cli_prog_name: The CLI program name to display in help text. Defaults to `None` if _cli_parse_args is `None`.
            Otherwise, defaults to sys.argv[0].
        _cli_parse_args: The list of CLI arguments to parse. Defaults to None.
            If set to `True`, defaults to sys.argv[1:].
        _cli_settings_source: Override the default CLI settings source with a user defined instance. Defaults to None.
        _cli_parse_none_str: The CLI string value that should be parsed (e.g. "null", "void", "None", etc.) into
            `None` type(None). Defaults to _env_parse_none_str value if set. Otherwise, defaults to "null" if
            _cli_avoid_json is `False`, and "None" if _cli_avoid_json is `True`.
        _cli_hide_none_type: Hide `None` values in CLI help text. Defaults to `False`.
        _cli_avoid_json: Avoid complex JSON objects in CLI help text. Defaults to `False`.
        _cli_enforce_required: Enforce required fields at the CLI. Defaults to `False`.
        _cli_use_class_docs_for_groups: Use class docstrings in CLI group help text instead of field descriptions.
            Defaults to `False`.
        _cli_exit_on_error: Determines whether or not the internal parser exits with error info when an error occurs.
            Defaults to `True`.
        _cli_prefix: The root parser command line arguments prefix. Defaults to "".
        _cli_flag_prefix_char: The flag prefix character to use for CLI optional arguments. Defaults to '-'.
        _cli_implicit_flags: Whether `bool` fields should be implicitly converted into CLI boolean flags.
            (e.g. --flag, --no-flag). Defaults to `False`.
        _cli_ignore_unknown_args: Whether to ignore unknown CLI args and parse only known ones. Defaults to `False`.
        _cli_kebab_case: CLI args use kebab case. Defaults to `False`.
        _cli_shortcuts: Mapping of target field name to alias names. Defaults to `None`.
        _secrets_dir: The secret files directory or a sequence of directories. Defaults to `None`.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|

### DataSource

Information about data source and provenance.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| source_name | str | Name of data source |  | Example text, Sample value, Data entry | Multiple sources |
| source_url | Optional[str] | URL of data source |  |  | Multiple sources |
| source_date | datetime | Date when data was sourced |  |  | Multiple sources |
| source_version | Optional[str] | Version of source data |  |  | Multiple sources |
| attribution | str | Required attribution text |  | Example text, Sample value, Data entry | Multiple sources |
| license | Optional[str] | Data license information |  |  | Multiple sources |

### MigrationRecord

Record of schema migration operations.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| migration_id | str |  |  | Example text, Sample value, Data entry | Multiple sources |
| from_version | SchemaVersion | Source schema version |  |  | Multiple sources |
| to_version | SchemaVersion | Target schema version |  |  | Multiple sources |
| migration_date | datetime |  |  |  | Multiple sources |
| record_count | int | Number of records migrated |  | 123, 456, 789 | Multiple sources |
| success | bool | Whether migration was successful |  | true, false | Multiple sources |
| errors | List[str] | Any errors encountered |  |  | Multiple sources |
| duration_seconds | Optional[float] | Migration duration |  |  | Multiple sources |

### TemporalData

Base model for time-series data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |

### VersionedSchema


    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |

### DataSource

Information about data source and provenance.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| source_name | str | Name of data source |  | Example text, Sample value, Data entry | Multiple sources |
| source_url | Optional[str] | URL of data source |  |  | Multiple sources |
| source_date | datetime | Date when data was sourced |  |  | Multiple sources |
| source_version | Optional[str] | Version of source data |  |  | Multiple sources |
| attribution | str | Required attribution text |  | Example text, Sample value, Data entry | Multiple sources |
| license | Optional[str] | Data license information |  |  | Multiple sources |

### DiseasePrevalence

Schema for disease prevalence data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level (SA2, SA3, SA4, STATE) |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_name | str | Name of health indicator |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_code | str | Unique indicator code |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_type | HealthIndicatorType | Type of health indicator |  |  | Multiple sources |
| value | float | Indicator value |  | 42.5, 67.8, 91.2 | Multiple sources |
| unit | str | Unit of measurement |  | Example text, Sample value, Data entry | Multiple sources |
| confidence_interval_lower | Optional[float] | Lower CI bound |  |  | Multiple sources |
| confidence_interval_upper | Optional[float] | Upper CI bound |  |  | Multiple sources |
| standard_error | Optional[float] | Standard error |  |  | Multiple sources |
| sample_size | Optional[int] | Sample size if applicable |  |  | Multiple sources |
| age_group | AgeGroupType | Age group for this indicator |  |  | Multiple sources |
| sex | Optional[str] | Sex (Male/Female/Persons) |  |  | Multiple sources |
| suppressed | bool | Whether value is suppressed for privacy |  | true, false | Multiple sources |
| reliability | Optional[str] | Statistical reliability rating |  |  | Multiple sources |
| data_source | DataSource | Source of the health data |  |  | Multiple sources |
| disease_name | str | Name of disease or condition |  | Example text, Sample value, Data entry | Multiple sources |
| disease_category | str | Disease category |  | Example text, Sample value, Data entry | Multiple sources |
| icd10_codes | List[str] | Related ICD-10 codes |  |  | Multiple sources |
| prevalence_count | Optional[int] | Number of cases |  |  | Multiple sources |
| prevalence_rate | float | Prevalence rate % |  | 15.2, 8.7, 23.1 | Multiple sources |
| severity_level | Optional[str] | Severity classification |  |  | Multiple sources |
| hospitalisations | Optional[int] | Related hospitalisations |  |  | Multiple sources |
| common_comorbidities | List[str] | Common co-occurring conditions |  |  | Multiple sources |

### RiskFactorData

Schema for health risk factor data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier (SA2, SA3, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level (SA2, SA3, SA4, STATE) |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_name | str | Name of health indicator |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_code | str | Unique indicator code |  | Example text, Sample value, Data entry | Multiple sources |
| indicator_type | HealthIndicatorType | Type of health indicator |  |  | Multiple sources |
| value | float | Indicator value |  | 42.5, 67.8, 91.2 | Multiple sources |
| unit | str | Unit of measurement |  | Example text, Sample value, Data entry | Multiple sources |
| confidence_interval_lower | Optional[float] | Lower CI bound |  |  | Multiple sources |
| confidence_interval_upper | Optional[float] | Upper CI bound |  |  | Multiple sources |
| standard_error | Optional[float] | Standard error |  |  | Multiple sources |
| sample_size | Optional[int] | Sample size if applicable |  |  | Multiple sources |
| age_group | AgeGroupType | Age group for this indicator |  |  | Multiple sources |
| sex | Optional[str] | Sex (Male/Female/Persons) |  |  | Multiple sources |
| suppressed | bool | Whether value is suppressed for privacy |  | true, false | Multiple sources |
| reliability | Optional[str] | Statistical reliability rating |  |  | Multiple sources |
| data_source | DataSource | Source of the health data |  |  | Multiple sources |
| risk_factor_name | str | Name of risk factor |  | Example text, Sample value, Data entry | Multiple sources |
| risk_category | str | Category of risk factor |  | Example text, Sample value, Data entry | Multiple sources |
| exposed_percentage | float | % of population exposed |  | 15.2, 8.7, 23.1 | Multiple sources |
| high_risk_percentage | Optional[float] | % at high risk |  |  | Multiple sources |
| attributable_burden | Optional[float] | Disease burden attributable to risk factor |  |  | Multiple sources |
| relative_risk | Optional[float] | Relative risk ratio |  |  | Multiple sources |
| modifiable | bool | Whether risk factor is modifiable |  | true, false | Multiple sources |
| intervention_available | bool | Whether interventions exist |  | true, false | Multiple sources |

### TemporalData

Base model for time-series data.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| reference_date | datetime | Reference date for the data |  |  | Multiple sources |
| period_type | str | Type of time period (annual, quarterly, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| period_start | datetime | Start of the data period |  |  | Multiple sources |
| period_end | datetime | End of the data period |  |  | Multiple sources |

### VersionedSchema


    Base class for all versioned schemas in AHGD.
    
    Provides common fields and versioning capabilities for all data models.
    

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |

## Socio-Economic

### SEIFAAggregate

Schema for aggregated SEIFA statistics.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| aggregation_level | str | Level of aggregation (state, national) |  | Example text, Sample value, Data entry | Multiple sources |
| aggregation_id | str | Identifier for aggregation area |  | Example text, Sample value, Data entry | Multiple sources |
| aggregation_name | str | Name of aggregation area |  | Example text, Sample value, Data entry | Multiple sources |
| index_type | SEIFAIndexType | SEIFA index type |  |  | Multiple sources |
| reference_year | int | Census year |  | 123, 456, 789 | Multiple sources |
| area_count | int | Number of areas included |  | 123, 456, 789 | Multiple sources |
| population_total | int | Total population covered |  | 5432, 12876, 3241 | Multiple sources |
| mean_score | float | Mean SEIFA score |  | 42.5, 67.8, 91.2 | Multiple sources |
| median_score | float | Median SEIFA score |  | 42.5, 67.8, 91.2 | Multiple sources |
| std_dev_score | float | Standard deviation of scores |  | 42.5, 67.8, 91.2 | Multiple sources |
| min_score | float | Minimum score |  | 42.5, 67.8, 91.2 | Multiple sources |
| max_score | float | Maximum score |  | 42.5, 67.8, 91.2 | Multiple sources |
| decile_distribution | Dict[int, int] | Count of areas in each decile |  |  | Multiple sources |
| quintile_distribution | Dict[int, int] | Count of areas in each quintile |  |  | Multiple sources |
| gini_coefficient | Optional[float] | Gini coefficient of inequality |  |  | Multiple sources |
| percentile_ratio_90_10 | Optional[float] | 90th/10th percentile ratio |  |  | Multiple sources |

### SEIFAComparison

Schema for SEIFA comparisons between areas or time periods.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| comparison_type | str | Type of comparison (temporal, spatial) |  | Example text, Sample value, Data entry | Multiple sources |
| area_1_id | str | First area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| area_1_name | str | First area name |  | Example text, Sample value, Data entry | Multiple sources |
| area_2_id | str | Second area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| area_2_name | str | Second area name |  | Example text, Sample value, Data entry | Multiple sources |
| index_type | SEIFAIndexType | SEIFA index type |  |  | Multiple sources |
| area_1_score | float | First area score |  | 42.5, 67.8, 91.2 | Multiple sources |
| area_2_score | float | Second area score |  | 42.5, 67.8, 91.2 | Multiple sources |
| score_difference | float | Score difference (area_2 - area_1) |  | 42.5, 67.8, 91.2 | Multiple sources |
| area_1_decile | int | First area decile |  | 123, 456, 789 | Multiple sources |
| area_2_decile | int | Second area decile |  | 123, 456, 789 | Multiple sources |
| decile_change | int | Decile change |  | 123, 456, 789 | Multiple sources |
| significant_change | Optional[bool] | Whether change is statistically significant |  |  | Multiple sources |
| confidence_level | Optional[float] | Confidence level % |  |  | Multiple sources |
| time_period_1 | Optional[int] | First time period (year) |  |  | Multiple sources |
| time_period_2 | Optional[int] | Second time period (year) |  |  | Multiple sources |

### SEIFAComponent

Schema for SEIFA index component variables.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier |  | Example text, Sample value, Data entry | Multiple sources |
| index_type | SEIFAIndexType | SEIFA index type |  |  | Multiple sources |
| reference_year | int | Census year |  | 123, 456, 789 | Multiple sources |
| variable_name | str | Name of component variable |  | Example text, Sample value, Data entry | Multiple sources |
| variable_code | str | Census variable code |  | Example text, Sample value, Data entry | Multiple sources |
| variable_description | str | Description of what variable measures |  | Example text, Sample value, Data entry | Multiple sources |
| raw_value | float | Raw variable value |  | 42.5, 67.8, 91.2 | Multiple sources |
| standardised_value | float | Standardised value |  | 42.5, 67.8, 91.2 | Multiple sources |
| weight | float | Weight in index calculation |  | 42.5, 67.8, 91.2 | Multiple sources |
| contribution | float | Contribution to final score |  | 42.5, 67.8, 91.2 | Multiple sources |
| positive_indicator | bool | Whether higher values indicate advantage |  | true, false | Multiple sources |

### SEIFAScore

Schema for SEIFA index scores and rankings.

| Field | Type | Description | Constraints | Examples | Source |
|-------|------|-------------|-------------|----------|--------|
| id | str | Unique identifier |  | Example text, Sample value, Data entry | Multiple sources |
| schema_version | SchemaVersion | Schema version for this record |  |  | Multiple sources |
| created_at | datetime | Timestamp when record was created |  |  | Multiple sources |
| updated_at | Optional[datetime] | Timestamp when record was last updated |  |  | Multiple sources |
| data_quality | DataQualityLevel | Data quality assessment level |  |  | Multiple sources |
| geographic_id | str | Geographic area identifier (SA1, SA2, etc) |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_level | str | Geographic level |  | Example text, Sample value, Data entry | Multiple sources |
| geographic_name | str | Geographic area name |  | Example text, Sample value, Data entry | Multiple sources |
| index_type | SEIFAIndexType | Type of SEIFA index |  |  | Multiple sources |
| reference_year | int | Census year for SEIFA data |  | 123, 456, 789 | Multiple sources |
| score | float | SEIFA score (standardised to mean 1000) |  | 42.5, 67.8, 91.2 | Multiple sources |
| national_rank | int | National rank (1 = most disadvantaged) |  | 123, 456, 789 | Multiple sources |
| national_decile | int | National decile |  | 123, 456, 789 | Multiple sources |
| national_quintile | int | National quintile |  | 123, 456, 789 | Multiple sources |
| national_percentile | float | National percentile |  | 42.5, 67.8, 91.2 | Multiple sources |
| state_rank | int | State rank |  | NSW, VIC, QLD | Multiple sources |
| state_decile | int | State decile |  | NSW, VIC, QLD | Multiple sources |
| state_code | str | State/territory code |  | NSW, VIC, QLD | Multiple sources |
| usual_resident_population | Optional[int] | Usual resident population |  | 5432, 12876, 3241 | Multiple sources |
| score_reliability | Optional[str] | Score reliability indicator |  |  | Multiple sources |
| excluded | bool | Whether area is excluded from rankings |  | true, false | Multiple sources |
| exclusion_reason | Optional[str] | Reason for exclusion if applicable |  |  | Multiple sources |
| data_source | DataSource | Source of SEIFA data |  |  | Multiple sources |

