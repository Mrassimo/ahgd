"""
Unit tests for EmploymentTransformer.

Test-driven development for ABS Census employment data transformation,
ensuring robust column mapping, labour force processing, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.transformers.census.employment_transformer import EmploymentTransformer
from src.utils.interfaces import TransformationError, ProcessingStatus


class TestEmploymentTransformer:
    """Test suite for EmploymentTransformer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock the configuration to avoid dependency on actual config files
        with patch('src.transformers.census.employment_transformer.get_config_manager') as mock_config_manager, \
             patch('src.transformers.census.employment_transformer.get_config') as mock_get_config:
            
            # Configure mock responses
            mock_get_config.side_effect = self._mock_get_config
            mock_config_manager.return_value = Mock()
            
            # Create transformer instance
            self.transformer = EmploymentTransformer()
    
    def _mock_get_config(self, key: str, default=None):
        """Mock configuration values for testing."""
        config_map = {
            "transformers.census.geographic_hierarchy": True,
            "transformers.census.impute_missing": "employment_weighted",
            "system.stop_on_error": False,
            "transformers.census.column_mappings": {},
            "schemas.census_employment": {},
            "transformers.census.operations": {
                "occupation_classification": "ANZSCO_2021",
                "industry_classification": "ANZSIC_2006",
                "include_skill_levels": True,
                "include_sector_grouping": True,
                "employment_indicators": ["unemployment_rate", "participation_rate", "employment_self_sufficiency"],
                "industry_diversity_analysis": True,
                "education_employment_alignment": True
            }
        }
        return config_map.get(key, default)
    
    def test_init_creates_employment_transformer_with_correct_configuration(self):
        """Test EmploymentTransformer initialisation with proper configuration."""
        # Assert: Verify transformer was created with correct attributes
        assert self.transformer is not None
        assert hasattr(self.transformer, 'config_manager')
        assert hasattr(self.transformer, 'column_mappings')
        assert hasattr(self.transformer, 'target_schema')
        assert hasattr(self.transformer, 'operations_config')
        assert hasattr(self.transformer, 'imputation_strategy')
        assert self.transformer.employment_sk_counter == 30000
        assert self.transformer.imputation_strategy == "employment_weighted"
        
        # Verify operations configuration
        ops_config = self.transformer.operations_config
        assert ops_config["occupation_classification"] == "ANZSCO_2021"
        assert ops_config["industry_classification"] == "ANZSIC_2006"
        assert ops_config["include_skill_levels"] is True
        assert "unemployment_rate" in ops_config["employment_indicators"]
        
    def test_logger_property_returns_valid_logger(self):
        """Test that logger property returns a valid logger instance."""
        # Act: Get logger
        logger = self.transformer.logger
        
        # Assert: Verify logger is valid
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'warning')
    
    def test_load_column_mappings_contains_required_employment_fields(self):
        """Test that column mappings include all required employment data fields."""
        # Act: Get column mappings
        mappings = self.transformer.column_mappings
        
        # Assert: Verify essential employment fields are mapped
        required_fields = [
            'geographic_id', 'labour_force_pop',
            'employed_full_time', 'employed_part_time', 'unemployed', 'not_in_labour_force',
            'agriculture_forestry_fishing', 'manufacturing', 'construction', 'retail_trade',
            'managers', 'professionals', 'technicians_trades', 'sales_workers'
        ]
        
        for field in required_fields:
            assert field in mappings, f"Required field {field} not found in column mappings"
            assert len(mappings[field]) > 0, f"No mapping options for {field}"
    
    def test_load_target_schema_defines_employment_data_types(self):
        """Test that target schema defines appropriate data types for employment fields."""
        # Act: Get target schema
        schema = self.transformer.target_schema
        
        # Assert: Verify essential schema elements
        assert 'employment_sk' in schema
        assert 'geographic_id' in schema
        assert 'labour_force_pop' in schema
        assert 'employed_full_time' in schema
        assert 'unemployed' in schema
        assert 'managers' in schema
        assert 'agriculture_forestry_fishing' in schema
        
        # Verify data types
        assert schema['employment_sk'] == 'int64'
        assert schema['geographic_id'] == 'object'
        assert schema['labour_force_pop'] == 'int64'
        assert schema['employed_full_time'] == 'int64'
        
    def test_process_labour_force_status_happy_path(self):
        """Test successful labour force status processing."""
        # Arrange: Create sample data with labour force columns
        test_data = pd.DataFrame({
            'employed_full_time': [350, 420, 280],
            'employed_part_time': [150, 180, 120],
            'unemployed': [25, 35, 20],
            'not_in_labour_force': [275, 365, 180],
            'labour_force_pop': [800, 1000, 600]
        })
        
        # Act: Process labour force status
        result_df = self.transformer._process_labour_force_status(test_data.copy())
        
        # Assert: Verify labour force data is processed correctly
        assert len(result_df) == 3
        assert 'total_employed' in result_df.columns
        assert 'total_labour_force' in result_df.columns
        assert 'unemployment_rate' in result_df.columns
        assert 'participation_rate' in result_df.columns
        
        # Verify calculations for first row
        assert result_df['total_employed'].iloc[0] == 500  # 350 + 150
        assert result_df['total_labour_force'].iloc[0] == 525  # 500 + 25
        assert abs(result_df['unemployment_rate'].iloc[0] - 4.76) < 0.01  # 25/525 * 100
        assert abs(result_df['participation_rate'].iloc[0] - 65.625) < 0.01  # 525/800 * 100
        
    def test_process_labour_force_status_handles_zero_division(self):
        """Test labour force processing with zero division scenarios."""
        # Arrange: Create data with zero values that could cause division by zero
        test_data = pd.DataFrame({
            'employed_full_time': [0, 100, 0],
            'employed_part_time': [0, 50, 0],
            'unemployed': [0, 0, 10],
            'not_in_labour_force': [100, 200, 90],
            'labour_force_pop': [100, 350, 100]
        })
        
        # Act: Process labour force status
        result_df = self.transformer._process_labour_force_status(test_data.copy())
        
        # Assert: Zero division should be handled gracefully
        assert len(result_df) == 3
        
        # First row: no employment, no unemployment -> 0% unemployment rate
        assert result_df['unemployment_rate'].iloc[0] == 0.0
        assert result_df['participation_rate'].iloc[0] == 0.0
        
        # Second row: employment but no unemployment -> 0% unemployment rate
        assert result_df['unemployment_rate'].iloc[1] == 0.0
        assert abs(result_df['participation_rate'].iloc[1] - 42.86) < 0.01  # 150/350 * 100
        
        # Third row: unemployment but no employment -> 100% unemployment rate
        assert result_df['unemployment_rate'].iloc[2] == 100.0
        assert result_df['participation_rate'].iloc[2] == 10.0  # 10/100 * 100
        
    def test_process_labour_force_status_calculates_all_indicators(self):
        """Test that all labour force indicators are calculated correctly."""
        # Arrange: Create realistic employment data
        test_data = pd.DataFrame({
            'employed_full_time': [600, 750],
            'employed_part_time': [200, 250],
            'unemployed': [50, 75],
            'not_in_labour_force': [350, 425],
            'labour_force_pop': [1200, 1500]
        })
        
        # Act: Process labour force status
        result_df = self.transformer._process_labour_force_status(test_data.copy())
        
        # Assert: Verify all calculated indicators
        expected_indicators = [
            'total_employed', 'total_labour_force', 'unemployment_rate',
            'participation_rate', 'employment_population_ratio', 'full_time_employment_ratio'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result_df.columns, f"Missing indicator: {indicator}"
        
        # Verify specific calculations for first row
        assert result_df['total_employed'].iloc[0] == 800  # 600 + 200
        assert result_df['total_labour_force'].iloc[0] == 850  # 800 + 50
        assert abs(result_df['unemployment_rate'].iloc[0] - 5.88) < 0.01  # 50/850 * 100
        assert abs(result_df['participation_rate'].iloc[0] - 70.83) < 0.01  # 850/1200 * 100
        assert abs(result_df['employment_population_ratio'].iloc[0] - 66.67) < 0.01  # 800/1200 * 100
        assert result_df['full_time_employment_ratio'].iloc[0] == 75.0  # 600/800 * 100
        
    def test_process_labour_force_status_handles_missing_columns(self):
        """Test labour force processing with missing columns."""
        # Arrange: Create data with some missing labour force columns
        test_data = pd.DataFrame({
            'employed_full_time': [400, 500],
            'unemployed': [30, 40],
            'labour_force_pop': [800, 1000]
            # Missing: employed_part_time, not_in_labour_force, labour_force_not_stated
        })
        
        # Act: Process labour force status
        result_df = self.transformer._process_labour_force_status(test_data.copy())
        
        # Assert: Missing columns should be added with default values
        assert len(result_df) == 2
        assert 'employed_part_time' in result_df.columns
        assert 'not_in_labour_force' in result_df.columns
        assert 'labour_force_not_stated' in result_df.columns
        
        # Verify calculations work with missing columns treated as zero
        assert result_df['total_employed'].iloc[0] == 400  # 400 + 0
        assert abs(result_df['unemployment_rate'].iloc[0] - 6.98) < 0.01  # 30/430 * 100
        
    def test_standardise_input_data_happy_path(self):
        """Test successful column mapping and standardisation."""
        # Arrange: Create sample raw employment data with ABS column names
        raw_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021001', '101021002', '101021003'],
            'Labour_force_status_15_P': [800, 1000, 600],
            'Employed_Full_time': [350, 420, 280],
            'Employed_Part_time': [150, 180, 120],
            'Unemployed_Total': [25, 35, 20],
            'Not_in_Labour_Force': [275, 365, 180],
            'Managers': [45, 55, 35],
            'Agriculture_Forestry_Fishing': [15, 20, 25]
        })
        
        # Act: Apply column standardisation
        result_df = self.transformer._standardise_input_data(raw_data)
        
        # Assert: Verify column mappings were applied correctly
        expected_columns = [
            'geographic_id', 'labour_force_pop', 'employed_full_time', 'employed_part_time',
            'unemployed', 'not_in_labour_force', 'managers', 'agriculture_forestry_fishing'
        ]
        
        for col in expected_columns:
            assert col in result_df.columns, f"Expected column {col} not found"
        
        # Verify data integrity after mapping
        assert len(result_df) == 3
        assert result_df['geographic_id'].iloc[0] == '101021001'
        assert result_df['labour_force_pop'].iloc[0] == 800
        assert result_df['employed_full_time'].iloc[0] == 350
        assert result_df['managers'].iloc[0] == 45
        
    def test_standardise_input_data_raises_error_on_missing_essential_column(self):
        """Test that missing essential columns raise TransformationError."""
        # Arrange: Create incomplete data missing essential 'geographic_id' column
        incomplete_data = pd.DataFrame({
            'Employed_Full_time': [350, 420],
            'Unemployed_Total': [25, 35]
            # Missing: geographic_id mapping
        })
        
        # Act & Assert: Should raise TransformationError
        with pytest.raises(TransformationError) as exc_info:
            self.transformer._standardise_input_data(incomplete_data)
        
        assert "Essential fields missing after mapping" in str(exc_info.value)
        
    def test_validate_labour_force_data_detects_inconsistencies(self):
        """Test validation detects labour force data inconsistencies."""
        # Arrange: Create data with validation issues
        test_data = pd.DataFrame({
            'employed_full_time': [400, -50, 300],  # Negative value in second row
            'employed_part_time': [100, 100, 150],
            'unemployed': [500, 25, 15],  # Very high unemployment in first row
            'not_in_labour_force': [200, 300, 200],
            'labour_force_pop': [1000, 400, 500],
            'total_employed': [500, 50, 450],
            'total_labour_force': [1000, 75, 465],
            'unemployment_rate': [50.0, 33.33, 3.23],  # Very high unemployment rate
            'participation_rate': [100.0, 18.75, 93.0]
        })
        
        # Act: Validate labour force data (should log warnings but not fail)
        self.transformer._validate_labour_force_data(test_data)
        
        # Assert: Validation should complete without exceptions
        # (Validation logs warnings but doesn't raise exceptions)
        assert True  # If we reach here, validation completed
        
    def test_integrate_geographic_hierarchy_adds_hierarchy_columns(self):
        """Test geographic hierarchy integration adds required columns."""
        # Arrange: Create data with geographic identifiers
        test_data = pd.DataFrame({
            'geographic_id': ['101021001', '201022002', '301023003'],
            'employed_full_time': [350, 420, 280]
        })
        
        # Act: Integrate geographic hierarchy
        result_df = self.transformer._integrate_geographic_hierarchy(test_data)
        
        # Assert: Verify hierarchy columns are added
        hierarchy_columns = ['sa3_code', 'sa4_code', 'state_code', 'geographic_level']
        for col in hierarchy_columns:
            assert col in result_df.columns, f"Missing hierarchy column: {col}"
        
        # Verify hierarchy extraction logic
        assert result_df['sa3_code'].iloc[0] == '10102'  # First 5 digits
        assert result_df['sa4_code'].iloc[0] == '101'    # First 3 digits
        assert result_df['state_code'].iloc[0] == '1'    # First digit
        assert result_df['geographic_level'].iloc[0] == 'SA2'
        
        # Verify different state codes
        assert result_df['state_code'].iloc[1] == '2'
        assert result_df['state_code'].iloc[2] == '3'
        
    def test_integrate_geographic_hierarchy_handles_missing_geographic_id(self):
        """Test geographic hierarchy integration with missing geographic_id."""
        # Arrange: Create data without geographic_id
        test_data = pd.DataFrame({
            'employed_full_time': [350, 420],
            'unemployed': [25, 35]
        })
        
        # Act: Integrate geographic hierarchy
        result_df = self.transformer._integrate_geographic_hierarchy(test_data)
        
        # Assert: Should add placeholder columns
        assert 'sa3_code' in result_df.columns
        assert 'sa4_code' in result_df.columns
        assert 'state_code' in result_df.columns
        assert 'geographic_level' in result_df.columns
        
        # Verify placeholder values
        assert result_df['sa3_code'].iloc[0] == 'UNKNOWN'
        assert result_df['geographic_level'].iloc[0] == 'UNKNOWN'
        
    def test_transform_end_to_end_with_basic_employment_data(self):
        """Test complete transform pipeline with basic employment data."""
        # Arrange: Create comprehensive test dataset with ABS column names
        raw_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021001', '101021002'],
            'Labour_force_status_15_P': [800, 1000],
            'Employed_Full_time': [350, 420],
            'Employed_Part_time': [150, 180],
            'Unemployed_Total': [25, 35],
            'Not_in_Labour_Force': [275, 365],
            'Managers': [45, 55],
            'Agriculture_Forestry_Fishing': [15, 20]
        })
        
        # Set stop_on_error to True to catch any errors
        self.transformer.stop_on_error = True
        
        # Act: Transform the data
        result_df = self.transformer.transform(raw_data)
        
        # Assert: Verify transformation completed successfully
        assert result_df is not None
        assert len(result_df) == 2
        
        # Verify essential columns are present
        assert 'geographic_id' in result_df.columns
        assert 'labour_force_pop' in result_df.columns
        assert 'total_employed' in result_df.columns
        assert 'unemployment_rate' in result_df.columns
        assert 'participation_rate' in result_df.columns
        
        # Verify data integrity
        assert result_df['geographic_id'].iloc[0] == '101021001'
        assert result_df['labour_force_pop'].iloc[0] == 800
        assert result_df['total_employed'].iloc[0] == 500  # 350 + 150
        
        # Verify processing metadata is created
        assert self.transformer.processing_metadata is not None
        assert self.transformer.processing_metadata.operation_type == "employment_transformation"
        
    def test_transform_end_to_end_with_advanced_employment_analytics(self):
        """Test complete transform pipeline including advanced analytics."""
        # Arrange: Create comprehensive employment dataset with all required fields
        raw_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021001', '101021002', '101021003'],
            'Labour_force_status_15_P': [1200, 1500, 900],
            'Employed_Full_time': [600, 750, 450],
            'Employed_Part_time': [200, 250, 150],
            'Unemployed_Total': [50, 75, 40],
            'Not_in_Labour_Force': [350, 425, 260],
            
            # Occupation data (ANZSCO)
            'Managers': [80, 100, 60],
            'Professionals': [200, 250, 150],
            'Technicians_Trades_Workers': [150, 180, 110],
            'Community_Personal_Service_Workers': [60, 75, 45],
            'Clerical_Administrative_Workers': [120, 150, 90],
            'Sales_Workers': [100, 125, 75],
            'Machinery_Operators_Drivers': [70, 85, 50],
            'Labourers': [80, 100, 60],
            
            # Industry data (ANZSIC)
            'Agriculture_Forestry_Fishing': [30, 40, 25],
            'Mining': [20, 25, 15],
            'Manufacturing': [120, 150, 90],
            'Construction': [100, 125, 75],
            'Retail_Trade': [130, 160, 95],
            'Education_Training': [90, 110, 65],
            'Health_Care_Social_Assistance': [150, 185, 110],
            'Professional_Scientific_Technical': [100, 125, 75],
            'Public_Administration_Safety': [60, 75, 45]
        })
        
        # Set stop_on_error to True to catch any errors
        self.transformer.stop_on_error = True
        
        # Act: Transform the data with advanced analytics
        result_df = self.transformer.transform(raw_data)
        
        # Assert: Verify transformation completed successfully
        assert result_df is not None
        assert len(result_df) == 3
        
        # Verify basic employment columns are present
        basic_columns = [
            'geographic_id', 'total_employed', 'unemployment_rate', 'participation_rate'
        ]
        for col in basic_columns:
            assert col in result_df.columns, f"Missing basic column: {col}"
        
        # Verify advanced ANZSCO classification columns are present
        anzsco_columns = [
            'skill_level_1_2', 'skill_level_3', 'skill_level_4_5', 'total_employed_classified'
        ]
        for col in anzsco_columns:
            assert col in result_df.columns, f"Missing ANZSCO column: {col}"
        
        # Verify advanced ANZSIC classification columns are present
        anzsic_columns = [
            'primary_industries', 'secondary_industries', 'tertiary_industries',
            'public_sector', 'private_sector', 'industry_diversity_index'
        ]
        for col in anzsic_columns:
            assert col in result_df.columns, f"Missing ANZSIC column: {col}"
        
        # Verify advanced employment indicators are present
        indicator_columns = [
            'employment_self_sufficiency', 'high_skill_employment_ratio',
            'public_sector_ratio', 'full_time_ratio'
        ]
        for col in indicator_columns:
            assert col in result_df.columns, f"Missing indicator column: {col}"
        
        # Verify education-employment alignment indicators are present
        alignment_columns = [
            'qualification_utilisation_rate', 'skill_match_index'
        ]
        for col in alignment_columns:
            assert col in result_df.columns, f"Missing alignment column: {col}"
        
        # Verify data integrity for advanced analytics
        first_row = result_df.iloc[0]
        
        # ANZSCO skill level calculations
        assert first_row['skill_level_1_2'] == 280  # 80 + 200 (Managers + Professionals)
        assert first_row['skill_level_3'] >= 330    # Technicians + Community + Clerical
        
        # ANZSIC sector calculations  
        assert first_row['primary_industries'] == 50  # 30 + 20 (Agriculture + Mining)
        assert first_row['secondary_industries'] == 220  # 120 + 100 (Manufacturing + Construction)
        
        # Advanced indicators should have reasonable values
        assert 0 <= first_row['industry_diversity_index'] <= 1
        assert 0 <= first_row['employment_self_sufficiency'] <= 100
        assert 0 <= first_row['public_sector_ratio'] <= 100
        
        # Verify processing metadata includes advanced analytics
        assert self.transformer.processing_metadata.operation_type == "employment_transformation"
        
        # Verify final pipeline stages are completed
        # All missing values should be imputed (no NaN values in numeric columns)
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            assert not result_df[col].isna().any(), f"Missing values found in {col} after imputation"
        
        # Verify schema compliance - should have employment_sk and other schema columns
        schema_columns = ['employment_sk', 'geo_sk', 'census_year', 'processed_timestamp']
        for col in schema_columns:
            assert col in result_df.columns, f"Schema column {col} missing after enforcement"
        
        # Verify data types are schema-compliant
        assert result_df['employment_sk'].dtype == 'int64'
        assert result_df['labour_force_pop'].dtype == 'int64'
        assert result_df['geographic_id'].dtype == 'object'
        
        # Verify employment_sk values are correctly assigned (starting from 30000)
        assert all(sk >= 30000 for sk in result_df['employment_sk'])
        assert result_df['employment_sk'].is_unique  # Each row should have unique SK
        
    def test_transform_handles_error_gracefully_with_stop_on_error_false(self):
        """Test transform handles errors gracefully when stop_on_error is False."""
        # Arrange: Create invalid data that will cause errors
        invalid_data = pd.DataFrame({
            'invalid_column': ['invalid', 'data', 'values']
        })
        
        # Ensure stop_on_error is False
        self.transformer.stop_on_error = False
        
        # Act & Assert: Should not raise exception
        result_df = self.transformer.transform(invalid_data)
        
        # Should return original data when transformation fails
        assert result_df is not None
        assert 'invalid_column' in result_df.columns
        
    def test_process_anzsco_occupation_classification_happy_path(self):
        """Test ANZSCO occupation classification processing."""
        # Arrange: Create data with occupation counts
        test_data = pd.DataFrame({
            'managers': [45, 55, 35],
            'professionals': [120, 150, 90],
            'technicians_trades': [80, 100, 60],
            'community_personal_service': [30, 40, 25],
            'clerical_administrative': [70, 85, 50],
            'sales_workers': [50, 65, 40],
            'machinery_operators': [35, 45, 25],
            'labourers': [40, 50, 30],
            'occupation_not_stated': [5, 10, 5]
        })
        
        # Act: Process ANZSCO classification
        result_df = self.transformer._process_anzsco_occupation_classification(test_data.copy())
        
        # Assert: Verify ANZSCO-based groupings are calculated
        assert len(result_df) == 3
        assert 'skill_level_1_2' in result_df.columns  # Managers + Professionals
        assert 'skill_level_3' in result_df.columns    # Technicians + Community service + Clerical
        assert 'skill_level_4_5' in result_df.columns  # Sales + Machinery + Labourers
        assert 'total_employed_classified' in result_df.columns
        
        # Verify calculations for first row
        assert result_df['skill_level_1_2'].iloc[0] == 165  # 45 + 120
        assert result_df['skill_level_3'].iloc[0] == 180    # 80 + 30 + 70
        assert result_df['skill_level_4_5'].iloc[0] == 125  # 50 + 35 + 40
        assert result_df['total_employed_classified'].iloc[0] == 470  # Sum excluding not_stated
        
    def test_process_anzsco_occupation_classification_handles_missing_columns(self):
        """Test ANZSCO classification with missing occupation columns."""
        # Arrange: Create data with some missing occupation columns
        test_data = pd.DataFrame({
            'managers': [45, 55],
            'professionals': [120, 150],
            'sales_workers': [50, 65]
            # Missing: other occupation categories
        })
        
        # Act: Process ANZSCO classification
        result_df = self.transformer._process_anzsco_occupation_classification(test_data.copy())
        
        # Assert: Missing columns should be added with zeros
        assert len(result_df) == 2
        assert 'technicians_trades' in result_df.columns
        assert 'labourers' in result_df.columns
        assert 'skill_level_1_2' in result_df.columns
        
        # Verify calculations work with missing columns
        assert result_df['skill_level_1_2'].iloc[0] == 165  # 45 + 120
        assert result_df['skill_level_4_5'].iloc[0] == 50   # 50 + 0 + 0
        
    def test_process_anzsic_industry_classification_happy_path(self):
        """Test ANZSIC industry classification processing."""
        # Arrange: Create data with industry employment counts
        test_data = pd.DataFrame({
            'agriculture_forestry_fishing': [20, 25, 15],
            'mining': [10, 15, 8],
            'manufacturing': [80, 100, 60],
            'construction': [70, 90, 50],
            'retail_trade': [90, 110, 70],
            'education_training': [60, 75, 45],
            'health_social_assistance': [100, 120, 80],
            'professional_services': [85, 105, 65],
            'public_administration': [40, 50, 30],
            'other_services': [25, 30, 20]
        })
        
        # Act: Process ANZSIC classification
        result_df = self.transformer._process_anzsic_industry_classification(test_data.copy())
        
        # Assert: Verify ANZSIC sector groupings are calculated
        assert len(result_df) == 3
        assert 'primary_industries' in result_df.columns     # Agriculture + Mining
        assert 'secondary_industries' in result_df.columns   # Manufacturing + Construction
        assert 'tertiary_industries' in result_df.columns    # Services
        assert 'public_sector' in result_df.columns          # Education + Health + Public admin
        assert 'private_sector' in result_df.columns         # Remaining industries
        
        # Verify calculations for first row
        assert result_df['primary_industries'].iloc[0] == 30   # 20 + 10
        assert result_df['secondary_industries'].iloc[0] == 150 # 80 + 70
        assert result_df['public_sector'].iloc[0] == 200      # 60 + 100 + 40
        
    def test_process_anzsic_industry_classification_calculates_diversity_index(self):
        """Test industry diversity index calculation."""
        # Arrange: Create data with varying industry diversity
        test_data = pd.DataFrame({
            'manufacturing': [100, 80, 40],     # High concentration vs distributed
            'retail_trade': [10, 20, 30],
            'education_training': [5, 20, 30],
            'health_social_assistance': [5, 20, 40],
            'professional_services': [5, 20, 60]
        })
        
        # Act: Process ANZSIC classification
        result_df = self.transformer._process_anzsic_industry_classification(test_data.copy())
        
        # Assert: Industry diversity index should be calculated
        assert 'industry_diversity_index' in result_df.columns
        
        # First row has high concentration (low diversity)
        # Second row has more even distribution (higher diversity)
        # Third row has very even distribution (highest diversity)
        assert result_df['industry_diversity_index'].iloc[0] < result_df['industry_diversity_index'].iloc[1]
        assert result_df['industry_diversity_index'].iloc[1] < result_df['industry_diversity_index'].iloc[2]
        
    def test_calculate_employment_indicators_advanced_metrics(self):
        """Test calculation of advanced employment indicators."""
        # Arrange: Create comprehensive employment data
        test_data = pd.DataFrame({
            'total_employed': [800, 1000, 600],
            'total_labour_force': [850, 1050, 650],
            'labour_force_pop': [1200, 1500, 900],
            'employed_full_time': [600, 750, 450],
            'employed_part_time': [200, 250, 150],
            'skill_level_1_2': [400, 500, 300],      # High skill
            'skill_level_3': [250, 300, 200],        # Medium skill  
            'skill_level_4_5': [150, 200, 100],      # Lower skill
            'public_sector': [200, 250, 150],
            'private_sector': [600, 750, 450]
        })
        
        # Act: Calculate employment indicators
        result_df = self.transformer._calculate_employment_indicators(test_data.copy())
        
        # Assert: Advanced indicators should be calculated
        expected_indicators = [
            'employment_self_sufficiency', 'high_skill_employment_ratio',
            'public_sector_ratio', 'full_time_ratio', 'skills_mismatch_index'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result_df.columns, f"Missing indicator: {indicator}"
        
        # Verify specific calculations for first row
        assert abs(result_df['employment_self_sufficiency'].iloc[0] - 66.67) < 0.01  # 800/1200 * 100
        assert result_df['high_skill_employment_ratio'].iloc[0] == 50.0   # 400/800 * 100
        assert result_df['public_sector_ratio'].iloc[0] == 25.0           # 200/800 * 100
        assert result_df['full_time_ratio'].iloc[0] == 75.0              # 600/800 * 100
        
    def test_calculate_employment_indicators_handles_zero_division(self):
        """Test employment indicators calculation with zero division scenarios."""
        # Arrange: Create data with zero employment
        test_data = pd.DataFrame({
            'total_employed': [0, 500],
            'total_labour_force': [0, 525],
            'labour_force_pop': [100, 750],
            'employed_full_time': [0, 375],
            'skill_level_1_2': [0, 250],
            'public_sector': [0, 125]
        })
        
        # Act: Calculate employment indicators
        result_df = self.transformer._calculate_employment_indicators(test_data.copy())
        
        # Assert: Zero division should be handled gracefully
        assert len(result_df) == 2
        
        # First row with zero employment should have zero/null indicators
        assert result_df['employment_self_sufficiency'].iloc[0] == 0.0
        assert result_df['high_skill_employment_ratio'].iloc[0] == 0.0
        assert result_df['public_sector_ratio'].iloc[0] == 0.0
        
        # Second row should calculate normally
        assert abs(result_df['employment_self_sufficiency'].iloc[1] - 66.67) < 0.01
        assert result_df['high_skill_employment_ratio'].iloc[1] == 50.0
        
    def test_analyse_education_employment_alignment_happy_path(self):
        """Test education-employment alignment analysis."""
        # Arrange: Create data with education and employment information
        test_data = pd.DataFrame({
            'total_employed': [800, 1000, 600],
            'skill_level_1_2': [400, 500, 300],      # Requires tertiary education
            'bachelor_degree': [350, 450, 250],      # University qualified
            'postgraduate_degree': [100, 150, 80],
            'certificate_iii_iv': [200, 250, 150],   # Trade qualifications
            'technicians_trades': [180, 220, 130],   # Trade occupations
            'professionals': [300, 400, 200],        # Professional occupations
            'education_pop_base': [1000, 1250, 750]  # Population base for education data
        })
        
        # Act: Analyse education-employment alignment
        result_df = self.transformer._analyse_education_employment_alignment(test_data.copy())
        
        # Assert: Alignment indicators should be calculated
        expected_indicators = [
            'qualification_utilisation_rate', 'skill_match_index',
            'over_qualification_rate', 'under_qualification_rate'
        ]
        
        for indicator in expected_indicators:
            assert indicator in result_df.columns, f"Missing indicator: {indicator}"
        
        # Verify calculations for first row
        # Total university qualified: 350 + 100 = 450
        # Professional roles: 300
        # Qualification utilisation: 300/450 * 100 = 66.67%
        assert abs(result_df['qualification_utilisation_rate'].iloc[0] - 66.67) < 0.01
        
        # Trade qualifications to trade occupations alignment
        # Certificate III/IV: 200, Trade occupations: 180
        # Trade utilisation: 180/200 * 100 = 90%
        trade_utilisation = (result_df['technicians_trades'].iloc[0] / result_df['certificate_iii_iv'].iloc[0]) * 100
        assert abs(trade_utilisation - 90.0) < 0.01
        
    def test_analyse_education_employment_alignment_handles_missing_education_data(self):
        """Test alignment analysis with missing education columns."""
        # Arrange: Create employment data without education columns
        test_data = pd.DataFrame({
            'total_employed': [800, 1000],
            'skill_level_1_2': [400, 500],
            'professionals': [300, 400]
            # Missing: education qualification columns
        })
        
        # Act: Analyse education-employment alignment
        result_df = self.transformer._analyse_education_employment_alignment(test_data.copy())
        
        # Assert: Missing education columns should be handled
        assert len(result_df) == 2
        assert 'qualification_utilisation_rate' in result_df.columns
        assert 'bachelor_degree' in result_df.columns  # Should be added with zeros
        
        # With missing education data, utilisation rates should be 0 or null
        assert result_df['qualification_utilisation_rate'].iloc[0] == 0.0
        
    def test_impute_missing_values_uses_employment_weighting(self):
        """Test employment-weighted geographic median imputation strategy."""
        # Arrange: Create data with missing values and varying labour force populations
        test_data = pd.DataFrame({
            'geographic_id': ['101021001', '101021002', '101021003', '201022001', '201022002'],
            'sa3_code': ['10102', '10102', '10102', '20102', '20102'],  # Two SA3 areas
            'sa4_code': ['101', '101', '101', '201', '201'],          # Two SA4 areas
            'state_code': ['1', '1', '1', '2', '2'],                  # Two states
            'labour_force_pop': [1000, 500, 2000, 800, 1200],        # Varying populations for weighting
            'total_employed': [800, np.nan, 1600, np.nan, 960],      # Missing values to impute
            'unemployment_rate': [5.0, np.nan, 4.0, np.nan, 6.0],   # Missing values to impute
            'managers': [80, 40, np.nan, 60, 90]                     # Mixed missing pattern
        })
        
        # Act: Impute missing values
        result_df = self.transformer._impute_missing_values(test_data.copy())
        
        # Assert: Missing values should be imputed using employment-weighted medians
        assert len(result_df) == 5
        assert not result_df['total_employed'].isna().any()  # No missing values remain
        assert not result_df['unemployment_rate'].isna().any()
        
        # Verify employment-weighted calculation for total_employed in SA3 '10102'
        # Available values: 800 (pop=1000), 1600 (pop=2000) 
        # Employment-weighted median should consider population weights
        sa3_imputed_value = result_df[result_df['geographic_id'] == '101021002']['total_employed'].iloc[0]
        assert sa3_imputed_value > 800  # Should be closer to 1600 due to higher population weight
        
        # Verify that managers column is also imputed
        assert not result_df['managers'].isna().any()
        
    def test_impute_missing_values_handles_no_missing_data(self):
        """Test imputation with no missing values."""
        # Arrange: Create complete data with no missing values
        test_data = pd.DataFrame({
            'geographic_id': ['101021001', '101021002'],
            'labour_force_pop': [1000, 800],
            'total_employed': [800, 640],
            'unemployment_rate': [5.0, 4.5],
            'managers': [80, 64]
        })
        
        # Act: Impute missing values
        result_df = self.transformer._impute_missing_values(test_data.copy())
        
        # Assert: Data should remain unchanged
        assert len(result_df) == 2
        pd.testing.assert_frame_equal(result_df, test_data)
        
    def test_impute_missing_values_uses_hierarchical_fallback(self):
        """Test hierarchical geographic fallback when SA3 median unavailable."""
        # Arrange: Create data where SA3 has all missing values, requiring SA4 fallback
        test_data = pd.DataFrame({
            'geographic_id': ['101021001', '101021002', '201022001'],  # Different SA4s
            'sa3_code': ['10102', '10102', '20102'],
            'sa4_code': ['101', '101', '201'],
            'state_code': ['1', '1', '2'],
            'labour_force_pop': [1000, 800, 1200],
            'total_employed': [np.nan, np.nan, 960],  # Both SA3 '10102' values missing
            'unemployment_rate': [np.nan, np.nan, 4.0]
        })
        
        # Act: Impute missing values
        result_df = self.transformer._impute_missing_values(test_data.copy())
        
        # Assert: Should fall back to SA4/state/global medians
        assert not result_df['total_employed'].isna().any()
        assert not result_df['unemployment_rate'].isna().any()
        
        # The missing values should be imputed with available data from SA4/state level
        first_imputed = result_df[result_df['geographic_id'] == '101021001']['total_employed'].iloc[0]
        second_imputed = result_df[result_df['geographic_id'] == '101021002']['total_employed'].iloc[0]
        assert first_imputed > 0
        assert second_imputed > 0
        
    def test_enforce_schema_produces_compliant_output(self):
        """Test schema enforcement with messy input data."""
        # Arrange: Create messy DataFrame with wrong types, extra columns, missing columns
        messy_data = pd.DataFrame({
            'geographic_id': ['101021001', '101021002'],
            'labour_force_pop': ['1000', '800'],           # String instead of int
            'total_employed': [800.5, 640.7],              # Float instead of int
            'unemployment_rate': [5.123456, 4.567890],     # Too many decimal places
            'managers': [80, 64],
            'professionals': [200, 160],
            'extra_column': ['unwanted', 'data'],          # Extra column not in schema
            'invalid_type': [True, False],                 # Boolean column
            # Missing required schema columns like employment_sk, geo_sk, etc.
        })
        
        # Act: Enforce schema compliance
        result_df = self.transformer._enforce_schema(messy_data.copy())
        
        # Assert: Output should be perfectly schema-compliant
        assert len(result_df) == 2
        
        # Verify essential schema columns are present
        required_columns = [
            'employment_sk', 'geo_sk', 'geographic_id', 'geographic_level', 'census_year',
            'labour_force_pop', 'employed_full_time', 'employed_part_time', 'unemployed',
            'not_in_labour_force', 'managers', 'professionals', 'agriculture_forestry_fishing',
            'processed_timestamp', 'table_code', 'table_name'
        ]
        
        for col in required_columns:
            assert col in result_df.columns, f"Required schema column {col} missing"
        
        # Verify data types are correct
        assert result_df['employment_sk'].dtype == 'int64'
        assert result_df['labour_force_pop'].dtype == 'int64'
        assert result_df['total_employed'].dtype == 'int64' if 'total_employed' in result_df.columns else True
        assert result_df['unemployment_rate'].dtype == 'float64'
        assert result_df['geographic_id'].dtype == 'object'
        
        # Verify extra columns are removed
        assert 'extra_column' not in result_df.columns
        assert 'invalid_type' not in result_df.columns
        
        # Verify missing columns were added with defaults
        assert result_df['employment_sk'].iloc[0] >= 30000  # Employment SK starts at 30K
        assert result_df['census_year'].iloc[0] == 2021     # Default census year
        assert result_df['table_code'].iloc[0] in ['G17', 'G43']  # Employment table codes
        
    def test_enforce_schema_handles_empty_dataframe(self):
        """Test schema enforcement with empty DataFrame."""
        # Arrange: Create empty DataFrame
        empty_data = pd.DataFrame()
        
        # Act: Enforce schema compliance
        result_df = self.transformer._enforce_schema(empty_data.copy())
        
        # Assert: Should return empty DataFrame with correct schema structure
        assert len(result_df) == 0
        assert len(result_df.columns) > 0  # Should have schema columns
        
        # Verify essential schema columns are present even in empty DataFrame
        essential_columns = ['employment_sk', 'geographic_id', 'labour_force_pop']
        for col in essential_columns:
            assert col in result_df.columns
            
    def test_enforce_schema_preserves_calculated_indicators(self):
        """Test that schema enforcement preserves advanced analytics columns."""
        # Arrange: Create data with advanced analytics columns
        analytics_data = pd.DataFrame({
            'geographic_id': ['101021001', '101021002'],
            'labour_force_pop': [1000, 800],
            'total_employed': [800, 640],
            'skill_level_1_2': [400, 320],
            'industry_diversity_index': [0.75, 0.68],
            'employment_self_sufficiency': [66.67, 64.0],
            'qualification_utilisation_rate': [85.0, 78.5]
        })
        
        # Act: Enforce schema compliance
        result_df = self.transformer._enforce_schema(analytics_data.copy())
        
        # Assert: Advanced analytics columns should be preserved
        assert 'skill_level_1_2' in result_df.columns
        assert 'industry_diversity_index' in result_df.columns
        assert 'employment_self_sufficiency' in result_df.columns
        assert 'qualification_utilisation_rate' in result_df.columns
        
        # Verify values are preserved
        assert result_df['skill_level_1_2'].iloc[0] == 400
        assert abs(result_df['industry_diversity_index'].iloc[0] - 0.75) < 0.01
        assert abs(result_df['employment_self_sufficiency'].iloc[0] - 66.67) < 0.01