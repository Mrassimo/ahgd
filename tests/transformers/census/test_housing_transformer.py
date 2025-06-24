"""
Unit tests for HousingTransformer.

Test-driven development for ABS Census housing data transformation,
ensuring robust column mapping, dwelling processing, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.transformers.census.housing_transformer import HousingTransformer
from src.utils.interfaces import TransformationError, ProcessingStatus


class TestHousingTransformer:
    """Test suite for HousingTransformer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock the configuration to avoid dependency on actual config files
        with patch('src.transformers.census.housing_transformer.get_config_manager') as mock_config_manager, \
             patch('src.transformers.census.housing_transformer.get_config') as mock_get_config:
            
            # Configure mock responses
            mock_get_config.side_effect = self._mock_get_config
            mock_config_manager.return_value = Mock()
            
            # Create transformer instance
            self.transformer = HousingTransformer()
    
    def _mock_get_config(self, key: str, default=None):
        """Mock configuration values for testing."""
        config_map = {
            "transformers.census.geographic_hierarchy": True,
            "transformers.census.impute_missing": "category_mode",
            "system.stop_on_error": False,
            "transformers.census.column_mappings": {},
            "schemas.census_housing": {},
            "transformers.census.operations": {
                "dwelling_structure_system": "abs_standard",
                "tenure_classification": "abs_standard",
                "include_landlord_types": True,
                "bedroom_categories": "standard_groups",
                "housing_ratios": ["home_ownership_rate", "rental_rate"],
                "housing_indicators": ["dwelling_diversity_index", "internet_penetration_rate"]
            }
        }
        return config_map.get(key, default)
    
    def test_init_creates_housing_transformer_with_correct_configuration(self):
        """Test HousingTransformer initialisation with proper configuration."""
        # Assert: Verify transformer was created with correct attributes
        assert self.transformer is not None
        assert hasattr(self.transformer, 'config_manager')
        assert hasattr(self.transformer, 'column_mappings')
        assert hasattr(self.transformer, 'target_schema')
        assert hasattr(self.transformer, 'operations_config')
        assert hasattr(self.transformer, 'imputation_strategy')
        assert self.transformer.housing_sk_counter == 20000
        assert self.transformer.imputation_strategy == "category_mode"
        
        # Verify operations configuration
        ops_config = self.transformer.operations_config
        assert ops_config["dwelling_structure_system"] == "abs_standard"
        assert ops_config["tenure_classification"] == "abs_standard"
        assert ops_config["include_landlord_types"] is True
        assert "home_ownership_rate" in ops_config["housing_ratios"]
        
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
    
    def test_load_column_mappings_contains_required_housing_fields(self):
        """Test that column mappings include all required housing data fields."""
        # Act: Get column mappings
        mappings = self.transformer.column_mappings
        
        # Assert: Verify essential housing fields are mapped
        required_fields = [
            'geographic_id', 'geographic_name', 'state_territory',
            'separate_house', 'semi_detached', 'flat_apartment', 'other_dwelling',
            'owned_outright', 'owned_with_mortgage', 'rented', 'other_tenure',
            'no_bedrooms', 'one_bedroom', 'two_bedrooms', 'three_bedrooms',
            'internet_connection', 'no_internet',
            'no_motor_vehicles', 'one_motor_vehicle', 'two_motor_vehicles',
            'median_mortgage_monthly', 'median_rent_weekly'
        ]
        
        for field in required_fields:
            assert field in mappings, f"Required field {field} not found in column mappings"
            assert len(mappings[field]) > 0, f"No mapping options for {field}"
    
    def test_load_target_schema_defines_housing_data_types(self):
        """Test that target schema defines appropriate data types for housing fields."""
        # Act: Get target schema
        schema = self.transformer.target_schema
        
        # Assert: Verify essential schema elements
        assert 'housing_sk' in schema
        assert 'geographic_id' in schema
        assert 'separate_house' in schema
        assert 'owned_outright' in schema
        assert 'internet_connection' in schema
        assert 'median_mortgage_monthly' in schema
        
        # Verify data types
        assert schema['housing_sk'] == 'int64'
        assert schema['geographic_id'] == 'object'
        assert schema['separate_house'] == 'int64'
        assert schema['median_mortgage_monthly'] == 'Int64'  # Nullable
    
    def test_process_dwelling_structure_happy_path(self):
        """Test successful dwelling structure processing."""
        # Arrange: Create sample data with dwelling structure columns
        test_data = pd.DataFrame({
            'separate_house': [150, 200, 100],
            'semi_detached': [25, 30, 15],
            'flat_apartment': [75, 50, 120],
            'other_dwelling': [5, 8, 3],
            'dwelling_structure_not_stated': [2, 1, 1]
        })
        
        # Act: Process dwelling structure
        result_df = self.transformer._process_dwelling_structure(test_data.copy())
        
        # Assert: Verify structure is preserved and processed correctly
        assert len(result_df) == 3
        assert 'separate_house' in result_df.columns
        assert 'semi_detached' in result_df.columns
        assert 'flat_apartment' in result_df.columns
        assert 'other_dwelling' in result_df.columns
        
        # Verify data integrity is maintained
        assert result_df['separate_house'].iloc[0] == 150
        assert result_df['flat_apartment'].iloc[2] == 120
    
    def test_process_tenure_type_happy_path(self):
        """Test successful tenure type processing."""
        # Arrange: Create sample data with tenure type columns
        test_data = pd.DataFrame({
            'owned_outright': [85, 95, 78],
            'owned_with_mortgage': [120, 110, 145],
            'rented': [45, 55, 67],
            'other_tenure': [5, 8, 3],
            'tenure_not_stated': [2, 1, 2]
        })
        
        # Act: Process tenure type
        result_df = self.transformer._process_tenure_type(test_data.copy())
        
        # Assert: Verify tenure data is processed correctly
        assert len(result_df) == 3
        assert 'owned_outright' in result_df.columns
        assert 'owned_with_mortgage' in result_df.columns
        assert 'rented' in result_df.columns
        
        # Verify data integrity
        assert result_df['owned_outright'].iloc[0] == 85
        assert result_df['rented'].iloc[2] == 67
    
    def test_process_internet_connection_happy_path(self):
        """Test successful internet connection processing."""
        # Arrange: Create sample data with internet connection columns
        test_data = pd.DataFrame({
            'internet_connection': [180, 190, 165],
            'no_internet': [25, 15, 30],
            'internet_not_stated': [3, 2, 4]
        })
        
        # Act: Process internet connection
        result_df = self.transformer._process_internet_connection(test_data.copy())
        
        # Assert: Verify internet data is processed correctly
        assert len(result_df) == 3
        assert 'internet_connection' in result_df.columns
        assert 'no_internet' in result_df.columns
        
        # Verify data integrity
        assert result_df['internet_connection'].iloc[0] == 180
        assert result_df['no_internet'].iloc[2] == 30
    
    def test_process_vehicle_data_happy_path(self):
        """Test successful vehicle data processing."""
        # Arrange: Create sample data with vehicle columns
        test_data = pd.DataFrame({
            'no_motor_vehicles': [35, 45, 28],
            'one_motor_vehicle': [85, 75, 92],
            'two_motor_vehicles': [95, 105, 88],
            'three_plus_vehicles': [25, 30, 22],
            'vehicles_not_stated': [3, 2, 1]
        })
        
        # Act: Process vehicle data
        result_df = self.transformer._process_vehicle_data(test_data.copy())
        
        # Assert: Verify vehicle data is processed correctly
        assert len(result_df) == 3
        assert 'no_motor_vehicles' in result_df.columns
        assert 'one_motor_vehicle' in result_df.columns
        assert 'two_motor_vehicles' in result_df.columns
        assert 'three_plus_vehicles' in result_df.columns
        
        # Verify data integrity
        assert result_df['no_motor_vehicles'].iloc[0] == 35
        assert result_df['two_motor_vehicles'].iloc[2] == 88
    
    def test_process_mortgage_and_rent_payments_happy_path(self):
        """Test successful mortgage and rent payment processing."""
        # Arrange: Create sample data with payment columns
        test_data = pd.DataFrame({
            'median_mortgage_monthly': [2800, 3200, 2500],
            'median_rent_weekly': [450, 520, 380]
        })
        
        # Act: Process payment data
        result_df = self.transformer._process_mortgage_and_rent_payments(test_data.copy())
        
        # Assert: Verify payment data is processed correctly
        assert len(result_df) == 3
        assert 'median_mortgage_monthly' in result_df.columns
        assert 'median_rent_weekly' in result_df.columns
        
        # Verify data integrity
        assert result_df['median_mortgage_monthly'].iloc[0] == 2800
        assert result_df['median_rent_weekly'].iloc[2] == 380
    
    def test_process_dwelling_structure_handles_missing_columns(self):
        """Test dwelling structure processing with missing columns."""
        # Arrange: Create data with some missing dwelling structure columns
        test_data = pd.DataFrame({
            'separate_house': [150, 200, 100],
            'flat_apartment': [75, 50, 120],
            # Missing semi_detached, other_dwelling
        })
        
        # Act: Process dwelling structure
        result_df = self.transformer._process_dwelling_structure(test_data.copy())
        
        # Assert: Should handle missing columns gracefully
        assert len(result_df) == 3
        assert 'separate_house' in result_df.columns
        assert 'flat_apartment' in result_df.columns
    
    def test_process_tenure_type_handles_zero_values(self):
        """Test tenure type processing with zero values."""
        # Arrange: Create data with zero values
        test_data = pd.DataFrame({
            'owned_outright': [0, 95, 78],
            'owned_with_mortgage': [120, 0, 145],
            'rented': [45, 55, 0],
            'other_tenure': [0, 0, 0]
        })
        
        # Act: Process tenure type
        result_df = self.transformer._process_tenure_type(test_data.copy())
        
        # Assert: Should handle zero values correctly
        assert len(result_df) == 3
        assert result_df['owned_outright'].iloc[0] == 0
        assert result_df['owned_with_mortgage'].iloc[1] == 0
        assert result_df['rented'].iloc[2] == 0
    
    def test_impute_missing_values_uses_category_mode(self):
        """Test missing value imputation using category mode strategy."""
        # Arrange: Create data with missing values
        test_data = pd.DataFrame({
            'sa3_code': ['301', '301', '302', '302'],
            'state_territory': ['VIC', 'VIC', 'VIC', 'VIC'],
            'separate_house': [150, np.nan, 200, 180],
            'owned_outright': [85, 90, np.nan, 88],
            'internet_connection': [180, np.nan, 175, np.nan]
        })
        
        # Act: Impute missing values
        result_df = self.transformer._impute_missing_values(test_data.copy())
        
        # Assert: Missing values should be imputed
        assert not result_df['separate_house'].isna().any()
        assert not result_df['owned_outright'].isna().any()
        assert not result_df['internet_connection'].isna().any()
        
        # For geographic group 301, separate_house should be mode (150)
        group_301_rows = result_df[result_df['sa3_code'] == '301']
        assert group_301_rows['separate_house'].iloc[1] == 150  # Imputed value
    
    def test_enforce_schema_adds_missing_columns_with_defaults(self):
        """Test schema enforcement adds missing columns with appropriate defaults."""
        # Arrange: Create minimal data missing some schema columns
        test_data = pd.DataFrame({
            'geographic_id': ['101021001', '101021002'],
            'separate_house': [150, 200],
            'owned_outright': [85, 95]
        })
        
        # Act: Enforce schema
        result_df = self.transformer._enforce_schema(test_data.copy())
        
        # Assert: Missing columns should be added with defaults
        expected_columns = [
            'housing_sk', 'geo_sk', 'geographic_id', 'geographic_level',
            'geographic_name', 'state_territory', 'census_year',
            'separate_house', 'owned_outright', 'processed_timestamp'
        ]
        
        for col in expected_columns:
            assert col in result_df.columns, f"Expected column {col} not found after schema enforcement"
        
        # Verify surrogate keys are generated
        assert result_df['housing_sk'].dtype == 'int64'
        assert result_df['housing_sk'].iloc[0] >= 20000  # Starting counter
    
    def test_transform_end_to_end_integration(self):
        """Test complete transform pipeline integration."""
        # Arrange: Create comprehensive test dataset with ABS column names
        raw_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021001', '101021002', '101021003'],
            'SA2_NAME_2021': ['Sydney - Haymarket', 'Sydney - CBD', 'Melbourne - CBD'],
            'STATE_CODE_2021': ['1', '1', '2'],
            
            # Dwelling structure
            'Separate_house': [150, 180, 120],
            'Semi_detached_row_terrace': [25, 30, 35],
            'Flat_unit_apartment': [75, 85, 95],
            'Other_dwelling': [5, 3, 8],
            
            # Tenure type
            'Owned_outright': [85, 95, 78],
            'Owned_with_mortgage': [120, 125, 135],
            'Rented': [45, 55, 67],
            'Other_tenure_type': [5, 8, 6],
            
            # Internet and vehicles
            'Internet_connected': [180, 195, 175],
            'No_internet_connection': [25, 15, 30],
            'No_vehicles': [35, 40, 45],
            'One_vehicle': [85, 90, 80],
            'Two_vehicles': [95, 100, 85],
            
            # Housing costs
            'Median_mortgage_monthly': [2800, 3200, 2500],
            'Median_rent_weekly': [450, 520, 380]
        })
        
        # Set stop_on_error to True to see any actual errors
        self.transformer.stop_on_error = True
        
        # Act: Transform the data
        result_df = self.transformer.transform(raw_data)
        
        # Assert: Verify transformation completed successfully
        assert result_df is not None
        assert len(result_df) == 3
        
        # Verify essential columns are present and properly mapped
        assert 'geographic_id' in result_df.columns
        assert 'separate_house' in result_df.columns
        assert 'owned_outright' in result_df.columns
        assert 'internet_connection' in result_df.columns
        
        # Verify data integrity after transformation
        assert result_df['geographic_id'].iloc[0] == '101021001'
        assert result_df['separate_house'].iloc[0] == 150
        assert result_df['owned_outright'].iloc[0] == 85
        
        # Verify processing metadata is created
        assert self.transformer.processing_metadata is not None
        assert self.transformer.processing_metadata.operation_type == "housing_transformation"
    
    def test_transform_handles_transformation_error_with_stop_on_error_false(self):
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
    
    def test_transform_raises_transformation_error_with_stop_on_error_true(self):
        """Test transform raises TransformationError when stop_on_error is True."""
        # Arrange: Create invalid data that will cause errors
        invalid_data = pd.DataFrame({
            'invalid_column': ['invalid', 'data', 'values']
        })
        
        # Set stop_on_error to True
        self.transformer.stop_on_error = True
        
        # Act & Assert: Should raise TransformationError
        with pytest.raises(TransformationError):
            self.transformer.transform(invalid_data)