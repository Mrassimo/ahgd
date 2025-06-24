"""
Test suite for census schema validation, focusing on CensusSEIFA and IntegratedCensusData schemas.

This module tests Pydantic schema validation for Australian Census data schemas,
with comprehensive coverage of the new CensusSEIFA schema for SEIFA index validation
and the IntegratedCensusData schema for unified census data integration.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from schemas.census_schema import CensusSEIFA, IntegratedCensusData
from schemas.base_schema import DataSource, DataQualityLevel


class TestCensusSEIFASchema:
    """Test suite for CensusSEIFA schema validation."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Valid base data for SEIFA schema testing
        current_time = datetime.now()
        self.valid_base_data = {
            "geographic_id": "101021007",
            "geographic_level": "SA2",
            "geographic_name": "Pyrmont - Ultimo",
            "state_territory": "NSW",
            "census_year": 2021,
            "irsd_score": 956,
            "irsd_rank": 15234,
            "irsd_decile": 7,
            
            # TemporalData required fields
            "reference_date": current_time,
            "period_type": "annual",
            "period_start": datetime(2021, 1, 1),
            "period_end": datetime(2021, 12, 31),
            
            # DataSource with all required fields
            "data_source": {
                "source_name": "ABS SEIFA 2021",
                "source_url": "https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia",
                "source_date": current_time,
                "attribution": "Australian Bureau of Statistics (ABS), Socio-Economic Indexes for Areas (SEIFA) 2021",
                "source_version": "2021.0.0"
            },
            
            # VersionedSchema fields
            "schema_version": "2.0.0"
        }
    
    def test_census_seifa_schema_validation_with_valid_data(self):
        """Test CensusSEIFA schema validation with valid data."""
        # Act: Create SEIFA instance with valid data
        seifa = CensusSEIFA(**self.valid_base_data)
        
        # Assert: Verify instance was created correctly
        assert seifa is not None
        assert seifa.geographic_id == "101021007"
        assert seifa.geographic_level == "SA2"
        assert seifa.state_territory == "NSW"
        assert seifa.census_year == 2021
        assert seifa.irsd_score == 956
        assert seifa.irsd_decile == 7
        assert seifa.get_schema_name() == "CensusSEIFA"
    
    def test_census_seifa_with_all_indices(self):
        """Test CensusSEIFA schema with all four SEIFA indices."""
        # Arrange: Add all SEIFA indices
        data = self.valid_base_data.copy()
        data.update({
            "irsad_score": 1025,
            "irsad_rank": 8945,
            "irsad_decile": 8,
            "ier_score": 1134,
            "ier_rank": 6234,
            "ier_decile": 9,
            "ieo_score": 987,
            "ieo_rank": 12456,
            "ieo_decile": 6
        })
        
        # Act: Create SEIFA instance
        seifa = CensusSEIFA(**data)
        
        # Assert: Verify all indices are set
        assert seifa.irsad_score == 1025
        assert seifa.irsd_score == 956
        assert seifa.ier_score == 1134
        assert seifa.ieo_score == 987
        assert all([seifa.irsad_decile, seifa.irsd_decile, seifa.ier_decile, seifa.ieo_decile])
    
    def test_census_seifa_state_rankings_and_deciles(self):
        """Test state-level rankings and deciles validation."""
        # Arrange: Add state-level data
        data = self.valid_base_data.copy()
        data.update({
            "irsad_state_rank": 234,
            "irsd_state_rank": 456,
            "irsad_state_decile": 6,
            "irsd_state_decile": 4
        })
        
        # Act: Create SEIFA instance
        seifa = CensusSEIFA(**data)
        
        # Assert: Verify state-level fields
        assert seifa.irsad_state_rank == 234
        assert seifa.irsd_state_rank == 456
        assert seifa.irsad_state_decile == 6
        assert seifa.irsd_state_decile == 4
    
    def test_census_seifa_validation_error_invalid_decile(self):
        """Test that invalid decile values raise ValidationError."""
        # Arrange: Create data with invalid decile (> 10)
        data = self.valid_base_data.copy()
        data["irsd_decile"] = 11
        
        # Act & Assert: Expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            CensusSEIFA(**data)
        
        assert "irsd_decile" in str(exc_info.value)
    
    def test_census_seifa_validation_error_invalid_year(self):
        """Test that invalid census year raises ValidationError."""
        # Arrange: Create data with invalid year
        data = self.valid_base_data.copy()
        data["census_year"] = 2022  # Invalid SEIFA year
        
        # Act & Assert: Expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            CensusSEIFA(**data)
        
        assert "Invalid SEIFA year" in str(exc_info.value)
    
    def test_census_seifa_validation_error_invalid_state(self):
        """Test that invalid state/territory raises ValidationError."""
        # Arrange: Create data with invalid state
        data = self.valid_base_data.copy()
        data["state_territory"] = "XX"  # Invalid state code
        
        # Act & Assert: Expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            CensusSEIFA(**data)
        
        assert "Invalid state/territory" in str(exc_info.value)
    
    def test_census_seifa_validation_error_no_indices(self):
        """Test that missing all SEIFA indices raises ValidationError."""
        # Arrange: Create data without any SEIFA scores
        data = self.valid_base_data.copy()
        del data["irsd_score"]  # Remove the only SEIFA score
        
        # Act & Assert: Expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            CensusSEIFA(**data)
        
        assert "At least one SEIFA index score must be provided" in str(exc_info.value)
    
    def test_census_seifa_validation_error_score_decile_inconsistency(self):
        """Test validation of score-decile consistency."""
        # Arrange: Create data with inconsistent score and decile
        data = self.valid_base_data.copy()
        data.update({
            "irsd_score": 700,  # Low score (disadvantaged)
            "irsd_decile": 9    # High decile (advantaged) - inconsistent
        })
        
        # Act & Assert: Expect ValidationError
        with pytest.raises(ValidationError) as exc_info:
            CensusSEIFA(**data)
        
        assert "IRSD score and decile appear inconsistent" in str(exc_info.value)
    
    def test_census_seifa_score_range_validation(self):
        """Test SEIFA score range validation."""
        # Test minimum valid score
        data = self.valid_base_data.copy()
        data["irsd_score"] = 1
        seifa = CensusSEIFA(**data)
        assert seifa.irsd_score == 1
        
        # Test maximum valid score
        data["irsd_score"] = 2000
        seifa = CensusSEIFA(**data)
        assert seifa.irsd_score == 2000
        
        # Test invalid score (too high)
        data["irsd_score"] = 2001
        with pytest.raises(ValidationError):
            CensusSEIFA(**data)
    
    def test_census_seifa_disadvantage_severity_validation(self):
        """Test disadvantage severity category validation."""
        # Arrange: Test valid categories
        valid_categories = ["very_high", "high", "moderate", "low", "very_low"]
        
        for category in valid_categories:
            data = self.valid_base_data.copy()
            data["disadvantage_severity"] = category
            
            # Act: Create SEIFA instance
            seifa = CensusSEIFA(**data)
            
            # Assert: Verify category is normalized to lowercase
            assert seifa.disadvantage_severity == category.lower()
        
        # Test invalid category
        data = self.valid_base_data.copy()
        data["disadvantage_severity"] = "invalid_category"
        
        with pytest.raises(ValidationError) as exc_info:
            CensusSEIFA(**data)
        
        assert "Invalid disadvantage severity" in str(exc_info.value)
    
    def test_census_seifa_get_overall_disadvantage_level(self):
        """Test calculation of overall disadvantage level."""
        # Test very high disadvantage (decile 1-2)
        data = self.valid_base_data.copy()
        data["irsd_decile"] = 1
        seifa = CensusSEIFA(**data)
        assert seifa.get_overall_disadvantage_level() == "very_high"
        
        # Test high disadvantage (decile 3-4)
        data["irsd_decile"] = 4
        seifa = CensusSEIFA(**data)
        assert seifa.get_overall_disadvantage_level() == "high"
        
        # Test moderate disadvantage (decile 5-6)
        data["irsd_decile"] = 6
        seifa = CensusSEIFA(**data)
        assert seifa.get_overall_disadvantage_level() == "moderate"
        
        # Test low disadvantage (decile 7-8)
        data["irsd_decile"] = 8
        seifa = CensusSEIFA(**data)
        assert seifa.get_overall_disadvantage_level() == "low"
        
        # Test very low disadvantage (decile 9-10)
        data["irsd_decile"] = 10
        seifa = CensusSEIFA(**data)
        assert seifa.get_overall_disadvantage_level() == "very_low"
        
        # Test None when no decile
        data["irsd_decile"] = None
        seifa = CensusSEIFA(**data)
        assert seifa.get_overall_disadvantage_level() is None
    
    def test_census_seifa_data_integrity_validation(self):
        """Test data integrity validation method."""
        # Test scores within typical range (no errors)
        data = self.valid_base_data.copy()
        data["irsd_score"] = 950  # Typical range
        seifa = CensusSEIFA(**data)
        errors = seifa.validate_data_integrity()
        assert len(errors) == 0
        
        # Test score outside typical range
        data["irsd_score"] = 500  # Below typical range
        seifa = CensusSEIFA(**data)
        errors = seifa.validate_data_integrity()
        assert len(errors) > 0
        assert "outside typical range" in errors[0]
        
        # Test unusually high rank
        data = self.valid_base_data.copy()
        data["irsd_rank"] = 60000  # Unusually high
        seifa = CensusSEIFA(**data)
        errors = seifa.validate_data_integrity()
        assert len(errors) > 0
        assert "appears unusually high" in errors[0]
    
    def test_census_seifa_optional_fields(self):
        """Test that optional fields can be None."""
        # Arrange: Minimal required data
        data = {
            "geographic_id": "101021007",
            "geographic_level": "SA2",
            "geographic_name": "Pyrmont - Ultimo",
            "state_territory": "NSW",
            "census_year": 2021,
            "irsd_score": 956,  # At least one score required
            "data_source": self.valid_base_data["data_source"],
            "schema_version": "2.0.0",
            "last_updated": datetime.now()
        }
        
        # Act: Create SEIFA instance
        seifa = CensusSEIFA(**data)
        
        # Assert: Verify optional fields are None
        assert seifa.irsad_score is None
        assert seifa.ier_score is None
        assert seifa.ieo_score is None
        assert seifa.overall_advantage_score is None
        assert seifa.population_base is None
    
    def test_census_seifa_state_territory_normalization(self):
        """Test state/territory code normalization to uppercase."""
        # Arrange: Lowercase state code
        data = self.valid_base_data.copy()
        data["state_territory"] = "nsw"
        
        # Act: Create SEIFA instance
        seifa = CensusSEIFA(**data)
        
        # Assert: Verify normalization to uppercase
        assert seifa.state_territory == "NSW"
    
    def test_census_seifa_composite_indicators(self):
        """Test composite indicators and derived fields."""
        # Arrange: Add composite indicators
        data = self.valid_base_data.copy()
        data.update({
            "overall_advantage_score": 75.5,
            "disadvantage_severity": "moderate",
            "population_base": 12456
        })
        
        # Act: Create SEIFA instance
        seifa = CensusSEIFA(**data)
        
        # Assert: Verify composite fields
        assert seifa.overall_advantage_score == 75.5
        assert seifa.disadvantage_severity == "moderate"
        assert seifa.population_base == 12456


class TestIntegratedCensusDataSchema:
    """Test suite for IntegratedCensusData master schema validation."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Valid base data for integrated census schema testing
        current_time = datetime.now()
        self.valid_integrated_data = {
            # Universal identifiers
            "geographic_id": "101021007",
            "geographic_level": "SA2",
            "geographic_name": "Pyrmont - Ultimo",
            "state_territory": "NSW",
            "census_year": 2021,
            
            # Population metrics
            "total_population": 12500,
            "working_age_population": 9500,
            "population_consistency_flag": True,
            
            # Demographics indicators (optional)
            "median_age": 35.5,
            "sex_ratio": 98.5,
            "indigenous_percentage": 2.5,
            
            # Housing indicators (optional)
            "home_ownership_rate": 45.2,
            "median_rent_weekly": 550,
            "housing_stress_rate": 28.5,
            
            # Employment indicators (optional)
            "unemployment_rate": 4.5,
            "participation_rate": 72.8,
            "professional_employment_rate": 42.3,
            
            # Education indicators (optional)
            "university_qualification_rate": 38.5,
            "year12_completion_rate": 78.2,
            
            # SEIFA indicators (optional)
            "irsad_score": 1045,
            "irsd_score": 956,
            "irsad_decile": 8,
            "irsd_decile": 7,
            
            # Derived insights (required)
            "socioeconomic_profile": "medium-high",
            "livability_index": 75.5,
            "economic_opportunity_score": 68.2,
            
            # Data quality metrics (required)
            "data_completeness_score": 0.85,
            "demographics_completeness": 0.95,
            "housing_completeness": 0.90,
            "employment_completeness": 0.85,
            "education_completeness": 0.80,
            "seifa_completeness": 0.75,
            "temporal_quality_flag": True,
            "consistency_score": 0.92,
            
            # Integration metadata
            "source_datasets": ["demographics", "housing", "employment", "education", "seifa"],
            "integration_timestamp": current_time,
            
            # TemporalData required fields
            "reference_date": current_time,
            "period_type": "annual",
            "period_start": datetime(2021, 1, 1),
            "period_end": datetime(2021, 12, 31),
            
            # DataSource with all required fields
            "data_source": {
                "source_name": "ABS Census 2021 Integrated Dataset",
                "source_url": "https://www.abs.gov.au/census",
                "source_date": current_time,
                "attribution": "Australian Bureau of Statistics (ABS), Census 2021",
                "source_version": "2021.0.0"
            },
            
            # VersionedSchema fields
            "schema_version": "2.0.0"
        }
    
    def test_integrated_census_data_validation_with_valid_data(self):
        """Test IntegratedCensusData schema validation with valid data."""
        # Act: Create integrated census instance with valid data
        integrated = IntegratedCensusData(**self.valid_integrated_data)
        
        # Assert: Verify instance was created correctly
        assert integrated is not None
        assert integrated.geographic_id == "101021007"
        assert integrated.total_population == 12500
        assert integrated.working_age_population == 9500
        assert integrated.socioeconomic_profile == "medium-high"
        assert integrated.data_completeness_score == 0.85
        assert integrated.get_schema_name() == "IntegratedCensusData"
    
    def test_integrated_census_data_quality_assessment(self):
        """Test data quality assessment method."""
        # Test high quality assessment - update domain scores to match
        high_quality_data = self.valid_integrated_data.copy()
        high_quality_data["data_completeness_score"] = 0.95
        high_quality_data["consistency_score"] = 0.93
        high_quality_data["demographics_completeness"] = 0.95
        high_quality_data["housing_completeness"] = 0.95
        high_quality_data["employment_completeness"] = 0.95
        high_quality_data["education_completeness"] = 0.95
        high_quality_data["seifa_completeness"] = 0.95
        integrated = IntegratedCensusData(**high_quality_data)
        assert integrated.get_data_quality_assessment() == "high"
        
        # Test medium quality assessment
        medium_quality_data = self.valid_integrated_data.copy()
        medium_quality_data["data_completeness_score"] = 0.75
        medium_quality_data["consistency_score"] = 0.72
        medium_quality_data["demographics_completeness"] = 0.75
        medium_quality_data["housing_completeness"] = 0.75
        medium_quality_data["employment_completeness"] = 0.75
        medium_quality_data["education_completeness"] = 0.75
        medium_quality_data["seifa_completeness"] = 0.75
        integrated = IntegratedCensusData(**medium_quality_data)
        assert integrated.get_data_quality_assessment() == "medium"
        
        # Test low quality assessment
        low_quality_data = self.valid_integrated_data.copy()
        low_quality_data["data_completeness_score"] = 0.45
        low_quality_data["consistency_score"] = 0.60
        low_quality_data["demographics_completeness"] = 0.45
        low_quality_data["housing_completeness"] = 0.45
        low_quality_data["employment_completeness"] = 0.45
        low_quality_data["education_completeness"] = 0.45
        low_quality_data["seifa_completeness"] = 0.45
        integrated = IntegratedCensusData(**low_quality_data)
        assert integrated.get_data_quality_assessment() == "low"
    
    def test_integrated_census_population_consistency_validation(self):
        """Test population consistency validation."""
        # Test invalid case: working age > total population
        invalid_data = self.valid_integrated_data.copy()
        invalid_data["total_population"] = 5000
        invalid_data["working_age_population"] = 6000  # Invalid - exceeds total
        
        with pytest.raises(ValidationError) as exc_info:
            IntegratedCensusData(**invalid_data)
        
        assert "Working age population cannot exceed total population" in str(exc_info.value)
    
    def test_integrated_census_percentage_fields_validation(self):
        """Test percentage field validation."""
        # Test invalid percentage > 100
        invalid_data = self.valid_integrated_data.copy()
        invalid_data["unemployment_rate"] = 105.5  # Invalid
        
        with pytest.raises(ValidationError) as exc_info:
            IntegratedCensusData(**invalid_data)
        
        assert "less than or equal to 100" in str(exc_info.value)
        
        # Test invalid percentage < 0
        invalid_data = self.valid_integrated_data.copy()
        invalid_data["home_ownership_rate"] = -5.0  # Invalid
        
        with pytest.raises(ValidationError) as exc_info:
            IntegratedCensusData(**invalid_data)
        
        assert "ge=0" in str(exc_info.value) or "greater than or equal to 0" in str(exc_info.value)
    
    def test_integrated_census_completeness_score_consistency(self):
        """Test completeness score consistency validation."""
        # Test inconsistent completeness scores
        invalid_data = self.valid_integrated_data.copy()
        invalid_data["data_completeness_score"] = 0.50  # Doesn't match domain averages
        
        with pytest.raises(ValidationError) as exc_info:
            IntegratedCensusData(**invalid_data)
        
        assert "Data completeness score inconsistent with domain scores" in str(exc_info.value)
    
    def test_integrated_census_socioeconomic_profile_validation(self):
        """Test socioeconomic profile validation."""
        # Test valid profiles
        valid_profiles = ["high", "medium-high", "medium-low", "low"]
        
        for profile in valid_profiles:
            data = self.valid_integrated_data.copy()
            data["socioeconomic_profile"] = profile
            integrated = IntegratedCensusData(**data)
            assert integrated.socioeconomic_profile == profile.lower()
        
        # Test invalid profile
        invalid_data = self.valid_integrated_data.copy()
        invalid_data["socioeconomic_profile"] = "very-high"  # Invalid
        
        with pytest.raises(ValidationError) as exc_info:
            IntegratedCensusData(**invalid_data)
        
        assert "Invalid socioeconomic profile" in str(exc_info.value)
    
    def test_integrated_census_seifa_advantage_category_validation(self):
        """Test SEIFA advantage category validation."""
        # Test valid categories
        valid_categories = ["very_high", "high", "moderate", "low", "very_low"]
        
        for category in valid_categories:
            data = self.valid_integrated_data.copy()
            data["seifa_advantage_category"] = category
            integrated = IntegratedCensusData(**data)
            assert integrated.seifa_advantage_category == category.lower()
        
        # Test None is allowed
        data = self.valid_integrated_data.copy()
        data["seifa_advantage_category"] = None
        integrated = IntegratedCensusData(**data)
        assert integrated.seifa_advantage_category is None
    
    def test_integrated_census_data_integrity_validation(self):
        """Test data integrity validation method."""
        # Test valid data - no errors
        integrated = IntegratedCensusData(**self.valid_integrated_data)
        errors = integrated.validate_data_integrity()
        assert len(errors) == 0
        
        # Test zero population error
        data = self.valid_integrated_data.copy()
        data["total_population"] = 0
        integrated = IntegratedCensusData(**data)
        errors = integrated.validate_data_integrity()
        assert "Total population is zero" in errors
        
        # Test low completeness error
        data = self.valid_integrated_data.copy()
        data["data_completeness_score"] = 0.45
        data["demographics_completeness"] = 0.45
        data["housing_completeness"] = 0.45
        data["employment_completeness"] = 0.45
        data["education_completeness"] = 0.45
        data["seifa_completeness"] = 0.45
        integrated = IntegratedCensusData(**data)
        errors = integrated.validate_data_integrity()
        assert any("Data completeness below threshold" in e for e in errors)
        
        # Test temporal quality error
        data = self.valid_integrated_data.copy()
        data["temporal_quality_flag"] = False
        integrated = IntegratedCensusData(**data)
        errors = integrated.validate_data_integrity()
        assert "Temporal alignment issues detected" in errors
    
    def test_integrated_census_minimal_required_fields(self):
        """Test creation with only required fields."""
        # Minimal data with only required fields
        minimal_data = {
            "geographic_id": "101021007",
            "geographic_level": "SA2",
            "geographic_name": "Pyrmont - Ultimo",
            "state_territory": "NSW",
            "census_year": 2021,
            "total_population": 10000,
            "socioeconomic_profile": "medium-low",
            "data_completeness_score": 0.70,
            "temporal_quality_flag": True,
            "consistency_score": 0.75,
            "source_datasets": ["demographics"],
            
            # Domain completeness scores (required for validation)
            "demographics_completeness": 0.70,
            "housing_completeness": 0.70,
            "employment_completeness": 0.70,
            "education_completeness": 0.70,
            "seifa_completeness": 0.70,
            
            # TemporalData required fields
            "reference_date": datetime.now(),
            "period_type": "annual",
            "period_start": datetime(2021, 1, 1),
            "period_end": datetime(2021, 12, 31),
            
            # DataSource required fields
            "data_source": {
                "source_name": "ABS Census 2021",
                "source_date": datetime.now(),
                "attribution": "Australian Bureau of Statistics"
            },
            
            "schema_version": "2.0.0"
        }
        
        # Act: Create instance with minimal data
        integrated = IntegratedCensusData(**minimal_data)
        
        # Assert: Verify creation succeeded
        assert integrated is not None
        assert integrated.working_age_population is None
        assert integrated.median_age is None
        assert integrated.unemployment_rate is None
        assert integrated.livability_index is None
    
    def test_integrated_census_full_domain_integration(self):
        """Test integration with all domain fields populated."""
        # Create comprehensive data with all domains
        full_data = self.valid_integrated_data.copy()
        
        # Add additional fields
        full_data.update({
            "age_dependency_ratio": 42.5,
            "median_mortgage_monthly": 2400,
            "average_household_size": 2.4,
            "median_personal_income": 65000,
            "industry_diversity_index": 0.78,
            "vocational_qualification_rate": 25.6,
            "seifa_advantage_category": "high",
            "social_cohesion_index": 72.3,
            "housing_market_pressure": 68.9,
            "integration_sk": 50001
        })
        
        # Act: Create fully populated instance
        integrated = IntegratedCensusData(**full_data)
        
        # Assert: Verify all fields are populated
        assert integrated.age_dependency_ratio == 42.5
        assert integrated.median_mortgage_monthly == 2400
        assert integrated.industry_diversity_index == 0.78
        assert integrated.social_cohesion_index == 72.3
        assert integrated.integration_sk == 50001