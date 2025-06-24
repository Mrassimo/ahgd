"""
Unit tests for DemographicTransformer.

Test-driven development for ABS Census demographic data transformation,
ensuring robust column mapping, validation, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.transformers.census.demographic_transformer import DemographicTransformer
from src.utils.interfaces import TransformationError, ProcessingStatus


class TestDemographicTransformer:
    """Test suite for DemographicTransformer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock the configuration to avoid dependency on actual config files
        with patch('src.transformers.census.demographic_transformer.get_config_manager') as mock_config_manager, \
             patch('src.transformers.census.demographic_transformer.get_config') as mock_get_config:
            
            # Configure mock responses
            mock_get_config.side_effect = self._mock_get_config
            mock_config_manager.return_value = Mock()
            
            # Create transformer instance
            self.transformer = DemographicTransformer()
    
    def _mock_get_config(self, key: str, default=None):
        """Mock configuration values for testing."""
        config_map = {
            "transformers.census.geographic_hierarchy": True,
            "transformers.census.impute_missing": "geographic_median",
            "system.stop_on_error": False,
            "transformers.census.column_mappings": {},
            "schemas.census_demographics": {},
            "transformers.census.operations": {
                "age_group_system": "5_year_groups",
                "include_broad_groups": True,
                "ratios": ["dependency_ratio", "sex_ratio"],
                "indicators": ["population_density"]
            }
        }
        return config_map.get(key, default)
    
    def test_standardise_input_data_happy_path(self):
        """Test successful column mapping and standardisation."""
        # Arrange: Create sample raw census data with ABS column names
        raw_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021001', '101021002', '101021003'],
            'SA2_NAME_2021': ['Sydney - Haymarket', 'Sydney - CBD', 'Melbourne - CBD'],
            'STATE_CODE_2021': ['1', '1', '2'],
            'Tot_P_P': [2847, 1956, 2134],
            'Male_P': [1425, 978, 1067],
            'Female_P': [1422, 978, 1067],
            'Age_0_4_yr_M': [45, 32, 38],
            'Age_0_4_yr_F': [43, 30, 36],
            'Indigenous_P': [28, 19, 21],
            'Non_Indigenous_P': [2785, 1911, 2089],
            'Indigenous_NS_P': [34, 26, 24],
            'Total_dwell_P': [1234, 867, 945],
            'OPD_P': [1156, 823, 901],
            'UPD_P': [78, 44, 44],
            'Total_families': [678, 445, 512]
        })
        
        # Act: Apply column standardisation
        result_df = self.transformer._standardise_input_data(raw_data)
        
        # Assert: Verify column mappings were applied correctly
        expected_columns = [
            'geographic_id', 'geographic_name', 'state_territory',
            'total_population', 'males', 'females',
            'age_0_4_male', 'age_0_4_female',
            'indigenous', 'non_indigenous', 'indigenous_not_stated',
            'total_private_dwellings', 'occupied_private_dwellings',
            'unoccupied_private_dwellings', 'total_families'
        ]
        
        for col in expected_columns:
            assert col in result_df.columns, f"Expected column {col} not found"
        
        # Verify data integrity after mapping
        assert len(result_df) == 3, "Row count should be preserved"
        assert result_df['geographic_id'].iloc[0] == '101021001', "Geographic ID mapping failed"
        assert result_df['total_population'].iloc[0] == 2847, "Population mapping failed"
        assert result_df['males'].iloc[0] == 1425, "Male population mapping failed"
        assert result_df['females'].iloc[0] == 1422, "Female population mapping failed"
        
        # Verify age group mappings
        assert result_df['age_0_4_male'].iloc[0] == 45, "Age group male mapping failed"
        assert result_df['age_0_4_female'].iloc[0] == 43, "Age group female mapping failed"
    
    def test_standardise_input_data_raises_error_on_missing_required_column(self):
        """Test that missing required columns raise TransformationError."""
        # Arrange: Create incomplete data missing required 'total_population' column
        incomplete_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021001', '101021002'],
            'SA2_NAME_2021': ['Sydney - Haymarket', 'Sydney - CBD'],
            'Male_P': [1425, 978],
            'Female_P': [1422, 978],
            # Missing 'Tot_P_P' (total_population) - this is required
        })
        
        # Act & Assert: Verify TransformationError is raised
        with pytest.raises(TransformationError) as exc_info:
            self.transformer._standardise_input_data(incomplete_data)
        
        # Verify error message mentions missing required columns
        error_message = str(exc_info.value)
        assert "Missing required columns" in error_message
        assert "total_population" in error_message
    
    def test_standardise_input_data_handles_alternative_column_names(self):
        """Test handling of alternative ABS column naming patterns."""
        # Arrange: Create data with older ABS column naming convention
        raw_data_old_format = pd.DataFrame({
            'SA2_MAIN21': ['101021001', '101021002'],  # Alternative SA2 code format
            'SA2_NAME21': ['Sydney Area', 'Melbourne Area'],  # Alternative name format
            'Total_Persons': [2500, 1800],  # Alternative total population column
            'Males': [1250, 900],  # Alternative males column
            'Females': [1250, 900],  # Alternative females column
        })
        
        # Act: Apply standardisation
        result_df = self.transformer._standardise_input_data(raw_data_old_format)
        
        # Assert: Verify alternative column names were mapped correctly
        assert 'geographic_id' in result_df.columns
        assert 'geographic_name' in result_df.columns
        assert 'total_population' in result_df.columns
        assert 'males' in result_df.columns
        assert 'females' in result_df.columns
        
        # Verify data values were preserved
        assert result_df['geographic_id'].iloc[0] == '101021001'
        assert result_df['total_population'].iloc[0] == 2500
        assert result_df['males'].iloc[0] == 1250
        assert result_df['females'].iloc[0] == 1250
    
    def test_find_matching_column_returns_first_match(self):
        """Test priority-based column matching logic."""
        # Arrange: Create DataFrame with multiple potential matches
        test_df = pd.DataFrame({
            'SA2_CODE': ['test1'],  # Lower priority match
            'SA2_CODE_2021': ['test2'],  # Higher priority match (should be selected)
            'other_column': ['test3']
        })
        
        candidates = ['SA2_CODE_2021', 'SA2_MAIN21', 'SA2_CODE']  # Priority order
        
        # Act: Find matching column
        result = self.transformer._find_matching_column(test_df, candidates)
        
        # Assert: Should return the highest priority match
        assert result == 'SA2_CODE_2021', "Should return first matching column in priority order"
    
    def test_find_matching_column_returns_none_when_no_match(self):
        """Test that no match returns None."""
        # Arrange: Create DataFrame without any matching columns
        test_df = pd.DataFrame({
            'unrelated_column1': ['test1'],
            'unrelated_column2': ['test2']
        })
        
        candidates = ['SA2_CODE_2021', 'SA2_MAIN21', 'SA2_CODE']
        
        # Act: Find matching column
        result = self.transformer._find_matching_column(test_df, candidates)
        
        # Assert: Should return None when no match found
        assert result is None, "Should return None when no matching columns found"
    
    def test_transform_method_integration(self):
        """Test the main transform method end-to-end."""
        # Arrange: Create comprehensive test data
        raw_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021001', '101021002'],
            'SA2_NAME_2021': ['Sydney - Haymarket', 'Sydney - CBD'],
            'STATE_CODE_2021': ['1', '1'],
            'Tot_P_P': [2847, 1956],
            'Male_P': [1425, 978],
            'Female_P': [1422, 978]
        })
        
        # Act: Run full transformation
        result_df = self.transformer.transform(raw_data)
        
        # Assert: Verify transformation completed successfully
        assert len(result_df) == 2, "Should preserve row count"
        assert 'geographic_id' in result_df.columns, "Should have standardised columns"
        assert 'total_population' in result_df.columns, "Should have standardised columns"
        
        # Verify processing metadata was created
        assert self.transformer.processing_metadata is not None
        assert self.transformer.processing_metadata.records_processed == 2
    
    @patch('src.transformers.census.demographic_transformer.get_logger')
    def test_logger_property_thread_safe(self, mock_get_logger):
        """Test that logger property creates new instances for thread safety."""
        # Arrange
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # Act: Access logger property multiple times
        logger1 = self.transformer.logger
        logger2 = self.transformer.logger
        
        # Assert: Should call get_logger each time for thread safety
        assert mock_get_logger.call_count >= 2, "Should create new logger instances"
        mock_get_logger.assert_called_with(self.transformer._logger_name)
    
    def test_integrate_geographic_hierarchy_successful_join(self):
        """Test successful geographic hierarchy integration."""
        # Arrange: Create data with matching geographic codes
        input_data = pd.DataFrame({
            'geographic_id': ['101021001', '101021002', '999999999'],  # Last one will be unmatched
            'total_population': [2847, 1956, 1000]
        })
        
        # Act: Integrate geographic hierarchy
        result_df = self.transformer._integrate_geographic_hierarchy(input_data)
        
        # Assert: Verify successful integration
        assert 'geo_sk' in result_df.columns, "Should add geo_sk column"
        assert 'sa3_code' in result_df.columns, "Should add sa3_code column"
        assert 'sa4_name' in result_df.columns, "Should add sa4_name column"
        
        # Verify matched records have valid geo_sk
        matched_records = result_df[result_df['geographic_id'].isin(['101021001', '101021002'])]
        assert all(matched_records['geo_sk'] > 0), "Matched records should have positive geo_sk"
        
        # Verify unmatched records have unknown geo_sk
        unmatched_records = result_df[result_df['geographic_id'] == '999999999']
        assert all(unmatched_records['geo_sk'] == -99), "Unmatched records should have unknown geo_sk (-99)"
    
    def test_standardise_age_groups_creates_all_18_groups(self):
        """Test that age group standardisation creates all 18 required age groups."""
        # Arrange: Create data with all age group columns
        input_data = pd.DataFrame({
            'total_population': [2000, 1500],
            'age_0_4_male': [50, 40], 'age_0_4_female': [48, 38],
            'age_5_9_male': [55, 45], 'age_5_9_female': [53, 43],
            'age_10_14_male': [60, 50], 'age_10_14_female': [58, 48],
            'age_15_19_male': [65, 55], 'age_15_19_female': [63, 53],
            'age_20_24_male': [70, 60], 'age_20_24_female': [68, 58],
            'age_25_29_male': [75, 65], 'age_25_29_female': [73, 63],
            'age_30_34_male': [80, 70], 'age_30_34_female': [78, 68],
            'age_35_39_male': [85, 75], 'age_35_39_female': [83, 73],
            'age_40_44_male': [90, 80], 'age_40_44_female': [88, 78],
            'age_45_49_male': [95, 85], 'age_45_49_female': [93, 83],
            'age_50_54_male': [100, 90], 'age_50_54_female': [98, 88],
            'age_55_59_male': [105, 95], 'age_55_59_female': [103, 93],
            'age_60_64_male': [110, 100], 'age_60_64_female': [108, 98],
            'age_65_69_male': [115, 105], 'age_65_69_female': [113, 103],
            'age_70_74_male': [120, 110], 'age_70_74_female': [118, 108],
            'age_75_79_male': [125, 115], 'age_75_79_female': [123, 113],
            'age_80_84_male': [130, 120], 'age_80_84_female': [128, 118],
            'age_85_plus_male': [135, 125], 'age_85_plus_female': [133, 123]
        })
        
        # Act: Standardise age groups
        result_df = self.transformer._standardise_age_groups(input_data)
        
        # Assert: Verify all 18 age groups are created
        expected_age_groups = [
            'age_0_4', 'age_5_9', 'age_10_14', 'age_15_19', 'age_20_24',
            'age_25_29', 'age_30_34', 'age_35_39', 'age_40_44', 'age_45_49',
            'age_50_54', 'age_55_59', 'age_60_64', 'age_65_69', 'age_70_74',
            'age_75_79', 'age_80_84', 'age_85_plus'
        ]
        
        for age_group in expected_age_groups:
            assert age_group in result_df.columns, f"Missing age group: {age_group}"
        
        # Verify aggregation is correct (male + female = total)
        assert result_df['age_0_4'].iloc[0] == 98, "Age 0-4 aggregation incorrect (50+48=98)"
        assert result_df['age_5_9'].iloc[0] == 108, "Age 5-9 aggregation incorrect (55+53=108)"
        
        # Verify data types are integers
        for age_group in expected_age_groups:
            assert result_df[age_group].dtype == 'int64', f"Age group {age_group} should be integer type"
    
    def test_standardise_age_groups_handles_missing_columns(self):
        """Test age group standardisation when some age columns are missing."""
        # Arrange: Create data with only some age group columns
        input_data = pd.DataFrame({
            'total_population': [1000],
            'age_0_4_male': [25], 'age_0_4_female': [23],
            'age_5_9_male': [30], 'age_5_9_female': [28],
            # Missing other age groups
        })
        
        # Act: Standardise age groups
        result_df = self.transformer._standardise_age_groups(input_data)
        
        # Assert: Verify available age groups are processed
        assert result_df['age_0_4'].iloc[0] == 48, "Available age groups should be aggregated"
        assert result_df['age_5_9'].iloc[0] == 58, "Available age groups should be aggregated"
        
        # Verify missing age groups are set to 0
        assert result_df['age_10_14'].iloc[0] == 0, "Missing age groups should default to 0"
        assert result_df['age_15_19'].iloc[0] == 0, "Missing age groups should default to 0"
    
    def test_process_indigenous_status_validates_consistency(self):
        """Test Indigenous status processing and validation."""
        # Arrange: Create data with Indigenous status columns
        input_data = pd.DataFrame({
            'total_population': [1000, 2000],
            'indigenous': [50, 100],
            'non_indigenous': [930, 1850],
            'indigenous_not_stated': [20, 50]
        })
        
        # Act: Process Indigenous status
        result_df = self.transformer._process_indigenous_status(input_data)
        
        # Assert: Verify data types and values
        assert result_df['indigenous'].dtype == 'int64', "Indigenous column should be integer"
        assert result_df['non_indigenous'].dtype == 'int64', "Non-indigenous column should be integer"
        assert result_df['indigenous_not_stated'].dtype == 'int64', "Indigenous not stated should be integer"
        
        # Verify totals are reasonable (within 5 of total population)
        indigenous_total = result_df['indigenous'] + result_df['non_indigenous'] + result_df['indigenous_not_stated']
        discrepancy = abs(indigenous_total - result_df['total_population'])
        assert all(discrepancy <= 5), "Indigenous status totals should be consistent with population"
    
    def test_process_dwelling_data_validates_occupancy(self):
        """Test dwelling data processing and occupancy validation."""
        # Arrange: Create data with dwelling columns
        input_data = pd.DataFrame({
            'total_private_dwellings': [500, 750],
            'occupied_private_dwellings': [450, 700],
            'unoccupied_private_dwellings': [50, 50],
            'total_families': [380, 620]
        })
        
        # Act: Process dwelling data
        result_df = self.transformer._process_dwelling_data(input_data)
        
        # Assert: Verify data types
        dwelling_columns = ['total_private_dwellings', 'occupied_private_dwellings', 
                           'unoccupied_private_dwellings', 'total_families']
        for col in dwelling_columns:
            assert result_df[col].dtype == 'int64', f"Dwelling column {col} should be integer"
        
        # Verify occupancy consistency (occupied + unoccupied = total)
        occupancy_total = result_df['occupied_private_dwellings'] + result_df['unoccupied_private_dwellings']
        discrepancy = abs(occupancy_total - result_df['total_private_dwellings'])
        assert all(discrepancy <= 2), "Dwelling occupancy should be consistent"
    
    def test_transform_end_to_end_integration(self):
        """Test complete transformation pipeline end-to-end."""
        # Arrange: Create comprehensive raw census data
        raw_data = pd.DataFrame({
            # Geographic identification
            'SA2_CODE_2021': ['101021001', '101021002'],
            'SA2_NAME_2021': ['Sydney - Haymarket', 'Sydney - CBD'],
            'STATE_CODE_2021': ['1', '1'],
            
            # Basic demographics
            'Tot_P_P': [2847, 1956],
            'Male_P': [1425, 978],
            'Female_P': [1422, 978],
            
            # Age groups (sample - not all 18 groups for brevity)
            'Age_0_4_yr_M': [45, 32], 'Age_0_4_yr_F': [43, 30],
            'Age_5_9_yr_M': [55, 42], 'Age_5_9_yr_F': [53, 40],
            'Age_10_14_yr_M': [65, 52], 'Age_10_14_yr_F': [63, 50],
            
            # Indigenous status
            'Indigenous_P': [28, 19],
            'Non_Indigenous_P': [2785, 1911],
            'Indigenous_NS_P': [34, 26],
            
            # Dwelling data
            'Total_dwell_P': [1234, 867],
            'OPD_P': [1156, 823],
            'UPD_P': [78, 44],
            'Total_families': [678, 445]
        })
        
        # Act: Run complete transformation
        result_df = self.transformer.transform(raw_data)
        
        # Assert: Verify comprehensive transformation results
        assert len(result_df) == 2, "Should preserve record count"
        
        # Verify core demographic fields
        expected_core_fields = [
            'geographic_id', 'total_population', 'males', 'females',
            'indigenous', 'non_indigenous', 'indigenous_not_stated',
            'total_private_dwellings', 'occupied_private_dwellings', 
            'unoccupied_private_dwellings', 'total_families'
        ]
        
        for field in expected_core_fields:
            assert field in result_df.columns, f"Missing core field: {field}"
        
        # Verify geographic integration occurred (geo_sk should be present)
        assert 'geo_sk' in result_df.columns, "Geographic integration should add geo_sk"
        assert 'sa3_code' in result_df.columns, "Geographic integration should add hierarchy"
        
        # Verify age groups were created
        age_group_samples = ['age_0_4', 'age_5_9', 'age_10_14']
        for age_group in age_group_samples:
            assert age_group in result_df.columns, f"Missing age group: {age_group}"
            assert result_df[age_group].iloc[0] > 0, f"Age group {age_group} should have data"
        
        # Verify ETL metadata was added
        metadata_fields = ['census_year', 'table_code', 'table_name', 'etl_processed_at']
        for field in metadata_fields:
            assert field in result_df.columns, f"Missing metadata field: {field}"
        
        # Verify data integrity
        assert result_df['census_year'].iloc[0] == 2021, "Census year should be 2021"
        assert result_df['table_code'].iloc[0] == 'G01', "Table code should be G01"
        
        # Verify processing metadata was updated
        assert self.transformer.processing_metadata is not None
        assert self.transformer.processing_metadata.records_processed == 2
        assert self.transformer.processing_metadata.status == ProcessingStatus.COMPLETED
    
    def test_calculate_demographic_ratios(self):
        """Test calculation of demographic ratios (dependency, sex, child, elderly)."""
        # Arrange: Create data with age groups and population data
        input_data = pd.DataFrame({
            'total_population': [10000, 5000],
            'males': [5100, 2600],
            'females': [4900, 2400],
            # Working age population (15-64)
            'age_15_19': [800, 400],
            'age_20_24': [900, 450],
            'age_25_29': [1000, 500],
            'age_30_34': [1100, 550],
            'age_35_39': [1000, 500],
            'age_40_44': [900, 450],
            'age_45_49': [800, 400],
            'age_50_54': [700, 350],
            'age_55_59': [600, 300],
            'age_60_64': [500, 250],
            # Young dependents (0-14)
            'age_0_4': [600, 300],
            'age_5_9': [550, 275],
            'age_10_14': [500, 250],
            # Elderly dependents (65+)
            'age_65_69': [400, 200],
            'age_70_74': [350, 175],
            'age_75_79': [250, 125],
            'age_80_84': [150, 75],
            'age_85_plus': [100, 50]
        })
        
        # Act: Calculate demographic ratios
        result_df = self.transformer._calculate_demographic_ratios(input_data)
        
        # Assert: Verify dependency ratio calculation
        # Working age = sum(age_15_19 to age_60_64) = 8300 for row 0, 4150 for row 1
        # Young dependents = sum(age_0_4 to age_10_14) = 1650 for row 0, 825 for row 1
        # Elderly dependents = sum(age_65_69 to age_85_plus) = 1250 for row 0, 625 for row 1
        # Total dependents = 1650 + 1250 = 2900 for row 0, 1450 for row 1
        # Dependency ratio = (2900 / 8300) * 100 = 34.94 for row 0
        expected_dependency_ratio_0 = (2900 / 8300) * 100
        expected_dependency_ratio_1 = (1450 / 4150) * 100
        
        assert 'dependency_ratio' in result_df.columns, "Should have dependency ratio column"
        assert abs(result_df['dependency_ratio'].iloc[0] - expected_dependency_ratio_0) < 0.1, \
            f"Dependency ratio calculation incorrect for row 0"
        assert abs(result_df['dependency_ratio'].iloc[1] - expected_dependency_ratio_1) < 0.1, \
            f"Dependency ratio calculation incorrect for row 1"
        
        # Assert: Verify sex ratio calculation (males per 100 females)
        expected_sex_ratio_0 = (5100 / 4900) * 100  # 104.08
        expected_sex_ratio_1 = (2600 / 2400) * 100  # 108.33
        
        assert 'sex_ratio' in result_df.columns, "Should have sex ratio column"
        assert abs(result_df['sex_ratio'].iloc[0] - expected_sex_ratio_0) < 0.1, \
            "Sex ratio calculation incorrect for row 0"
        assert abs(result_df['sex_ratio'].iloc[1] - expected_sex_ratio_1) < 0.1, \
            "Sex ratio calculation incorrect for row 1"
        
        # Assert: Verify child ratio (0-14 per 100 working age)
        expected_child_ratio_0 = (1650 / 8300) * 100  # 19.88
        expected_child_ratio_1 = (825 / 4150) * 100    # 19.88
        
        assert 'child_ratio' in result_df.columns, "Should have child ratio column"
        assert abs(result_df['child_ratio'].iloc[0] - expected_child_ratio_0) < 0.1, \
            "Child ratio calculation incorrect for row 0"
        
        # Assert: Verify elderly ratio (65+ per 100 working age)
        expected_elderly_ratio_0 = (1250 / 8300) * 100  # 15.06
        expected_elderly_ratio_1 = (625 / 4150) * 100    # 15.06
        
        assert 'elderly_ratio' in result_df.columns, "Should have elderly ratio column"
        assert abs(result_df['elderly_ratio'].iloc[0] - expected_elderly_ratio_0) < 0.1, \
            "Elderly ratio calculation incorrect for row 0"
        
        # Verify data types
        ratio_columns = ['dependency_ratio', 'sex_ratio', 'child_ratio', 'elderly_ratio']
        for col in ratio_columns:
            assert result_df[col].dtype == 'float64', f"{col} should be float type"
        
        # Verify no negative ratios
        for col in ratio_columns:
            assert all(result_df[col] >= 0), f"{col} should not have negative values"
    
    def test_derive_demographic_indicators(self):
        """Test derivation of demographic indicators (density, median age deviation, diversity index)."""
        # Arrange: Create data with required fields including area
        input_data = pd.DataFrame({
            'total_population': [10000, 5000, 2000],
            'males': [5100, 2600, 1100],
            'females': [4900, 2400, 900],
            'area_sq_km': [2.5, 10.0, 50.0],  # Small urban, medium suburban, large rural
            'indigenous': [500, 200, 50],
            'non_indigenous': [9300, 4700, 1900],
            'indigenous_not_stated': [200, 100, 50],
            # Age groups for median age calculation
            'age_0_4': [600, 300, 120],
            'age_5_9': [550, 275, 110],
            'age_10_14': [500, 250, 100],
            'age_15_19': [800, 400, 160],
            'age_20_24': [900, 450, 180],
            'age_25_29': [1000, 500, 200],
            'age_30_34': [1100, 550, 220],
            'age_35_39': [1000, 500, 200],
            'age_40_44': [900, 450, 180],
            'age_45_49': [800, 400, 160],
            'age_50_54': [700, 350, 140],
            'age_55_59': [600, 300, 120],
            'age_60_64': [500, 250, 100],
            'age_65_69': [400, 200, 80],
            'age_70_74': [350, 175, 70],
            'age_75_79': [250, 125, 50],
            'age_80_84': [150, 75, 30],
            'age_85_plus': [100, 50, 20]
        })
        
        # Act: Derive demographic indicators
        result_df = self.transformer._derive_demographic_indicators(input_data)
        
        # Assert: Verify population density calculation
        expected_density_0 = 10000 / 2.5  # 4000 people per sq km (urban)
        expected_density_1 = 5000 / 10.0  # 500 people per sq km (suburban)
        expected_density_2 = 2000 / 50.0  # 40 people per sq km (rural)
        
        assert 'population_density' in result_df.columns, "Should have population density column"
        assert abs(result_df['population_density'].iloc[0] - expected_density_0) < 0.1, \
            "Population density calculation incorrect for urban area"
        assert abs(result_df['population_density'].iloc[1] - expected_density_1) < 0.1, \
            "Population density calculation incorrect for suburban area"
        assert abs(result_df['population_density'].iloc[2] - expected_density_2) < 0.1, \
            "Population density calculation incorrect for rural area"
        
        # Assert: Verify median age deviation is calculated
        assert 'median_age_deviation' in result_df.columns, "Should have median age deviation column"
        assert result_df['median_age_deviation'].dtype == 'float64', "Median age deviation should be float"
        # The actual median age calculation is complex, so we'll just verify it's a reasonable value
        assert all(result_df['median_age_deviation'].between(-20, 20)), \
            "Median age deviation should be within reasonable range"
        
        # Assert: Verify diversity index calculation
        # Diversity index = 1 - sum((group_i / total)^2) for each group
        # For row 0: indigenous=500, non_indigenous=9300, not_stated=200, total=10000
        # Index = 1 - ((500/10000)^2 + (9300/10000)^2 + (200/10000)^2)
        # Index = 1 - (0.0025 + 0.8649 + 0.0004) = 1 - 0.8678 = 0.1322
        total_0 = 500 + 9300 + 200
        expected_diversity_0 = 1 - ((500/total_0)**2 + (9300/total_0)**2 + (200/total_0)**2)
        
        assert 'diversity_index' in result_df.columns, "Should have diversity index column"
        assert abs(result_df['diversity_index'].iloc[0] - expected_diversity_0) < 0.01, \
            "Diversity index calculation incorrect"
        assert all(result_df['diversity_index'].between(0, 1)), \
            "Diversity index should be between 0 and 1"
        
        # Verify data types
        indicator_columns = ['population_density', 'median_age_deviation', 'diversity_index']
        for col in indicator_columns:
            assert result_df[col].dtype == 'float64', f"{col} should be float type"
        
        # Verify handling of edge cases (zero area should not cause division error)
        input_with_zero_area = input_data.copy()
        input_with_zero_area.loc[0, 'area_sq_km'] = 0
        result_edge = self.transformer._derive_demographic_indicators(input_with_zero_area)
        assert result_edge['population_density'].iloc[0] == 0, \
            "Should handle zero area gracefully"
    
    def test_impute_missing_values(self):
        """Test hierarchical geographic median imputation for missing values."""
        # Arrange: Create data with nulls and geographic hierarchy
        input_data = pd.DataFrame({
            'geographic_id': ['101021001', '101021002', '101021003', '102031001', '102031002', '201041001'],
            'sa3_code': ['10102', '10102', '10102', '10203', '10203', '20104'],  # Two SA3s
            'sa4_code': ['101', '101', '101', '102', '102', '201'],  # Two SA4s
            'state_territory': ['NSW', 'NSW', 'NSW', 'NSW', 'NSW', 'VIC'],  # Two states
            'total_population': [1000, np.nan, 1200, 2000, np.nan, 3000],  # Missing values
            'males': [500, 550, np.nan, 1000, 1100, np.nan],
            'females': [500, np.nan, 600, 1000, np.nan, 1500],
            'age_0_4': [100, 110, np.nan, np.nan, np.nan, 300],  # All NSW SA3 10203 missing
            'indigenous': [50, np.nan, 60, 100, 110, np.nan],
            'dependency_ratio': [35.5, np.nan, 36.5, np.nan, np.nan, 40.0],  # Multiple missing
            'population_density': [np.nan, np.nan, np.nan, 500.0, 550.0, np.nan]  # All SA3 10102 missing
        })
        
        # Act: Impute missing values using hierarchical geographic median
        result_df = self.transformer._impute_missing_values(input_data)
        
        # Assert: Verify SA3-level median imputation
        # For SA3 10102: total_population median = median([1000, 1200]) = 1100
        assert result_df.loc[1, 'total_population'] == 1100, \
            "Should impute with SA3 median for total_population"
        
        # For SA3 10102: males median = median([500, 550]) = 525
        assert result_df.loc[2, 'males'] == 525, \
            "Should impute with SA3 median for males"
        
        # Assert: Verify SA4-level median imputation when SA3 has no data
        # For age_0_4 in SA3 10203 (all null), should use SA4 101 median = median([100, 110]) = 105
        assert result_df.loc[3, 'age_0_4'] == 105, \
            "Should fall back to SA4 median when SA3 has no data"
        assert result_df.loc[4, 'age_0_4'] == 105, \
            "Should fall back to SA4 median when SA3 has no data"
        
        # Assert: Verify State-level median imputation when SA4 has no data
        # For population_density in SA3 10102 (all null) and SA4 101 (all null), 
        # should use NSW state median = median([500, 550]) = 525
        assert result_df.loc[0, 'population_density'] == 525, \
            "Should fall back to state median when SA3 and SA4 have no data"
        assert result_df.loc[1, 'population_density'] == 525, \
            "Should fall back to state median when SA3 and SA4 have no data"
        assert result_df.loc[2, 'population_density'] == 525, \
            "Should fall back to state median when SA3 and SA4 have no data"
        
        # Assert: Verify global median imputation when all geographic levels have no data
        # For males in VIC (only one record, which is null)
        # Should use global median = median([500, 550, 1000, 1100]) = 775
        assert result_df.loc[5, 'males'] == 775, \
            "Should fall back to global median when all geographic levels have no data"
        
        # Verify no nulls remain in numeric columns
        numeric_columns = ['total_population', 'males', 'females', 'age_0_4', 
                          'indigenous', 'dependency_ratio', 'population_density']
        for col in numeric_columns:
            assert result_df[col].notna().all(), f"Column {col} should have no nulls after imputation"
        
        # Verify data types are preserved
        for col in numeric_columns:
            assert result_df[col].dtype in ['int64', 'float64'], \
                f"Column {col} should maintain numeric type after imputation"
    
    def test_enforce_schema(self):
        """Test final schema enforcement for data types and column ordering."""
        # Arrange: Create data with mixed types and missing columns
        input_data = pd.DataFrame({
            # Core required columns (but in wrong order)
            'total_population': [1000.0, 2000.0],  # Float instead of int
            'geographic_id': ['101021001', '101021002'],
            'males': ['500', '1000'],  # String instead of int
            'females': [500.5, 1000.5],  # Float that should be int
            # Missing required columns that should be added
            # Missing: demographic_sk, geo_sk, census_year, table_code, table_name
            # Missing: many age groups, ratios, indicators
            # Extra column that's not in schema
            'extra_column': ['should', 'be_removed'],
            # Some age groups (partial)
            'age_0_4': [100, 200],
            'age_5_9': [110, 210],
            # Some other fields
            'indigenous': [50.0, 100.0],
            'dependency_ratio': [35.5, 38.2],
            'population_density': [1000, 2000],
            'etl_processed_at': ['2024-01-15', '2024-01-15']  # String instead of timestamp
        })
        
        # Act: Enforce schema
        result_df = self.transformer._enforce_schema(input_data)
        
        # Assert: Verify correct data types are enforced
        # Integer columns
        int_columns = ['demographic_sk', 'geo_sk', 'census_year', 'total_population', 
                      'males', 'females', 'indigenous', 'non_indigenous', 
                      'indigenous_not_stated', 'total_private_dwellings',
                      'occupied_private_dwellings', 'unoccupied_private_dwellings',
                      'total_families'] + [f'age_{g}' for g in ['0_4', '5_9', '10_14', '15_19', 
                                                                 '20_24', '25_29', '30_34', '35_39',
                                                                 '40_44', '45_49', '50_54', '55_59',
                                                                 '60_64', '65_69', '70_74', '75_79',
                                                                 '80_84', '85_plus']]
        
        for col in int_columns:
            if col in result_df.columns:
                assert result_df[col].dtype == 'int64', f"{col} should be int64 type"
        
        # Float columns
        float_columns = ['dependency_ratio', 'sex_ratio', 'child_ratio', 'elderly_ratio',
                        'population_density', 'median_age_deviation', 'diversity_index']
        
        for col in float_columns:
            if col in result_df.columns:
                assert result_df[col].dtype == 'float64', f"{col} should be float64 type"
        
        # String columns
        string_columns = ['geographic_id', 'geographic_name', 'state_territory', 
                         'sa3_code', 'sa3_name', 'sa4_code', 'sa4_name', 
                         'gcc_code', 'gcc_name', 'table_code', 'table_name']
        
        for col in string_columns:
            if col in result_df.columns:
                assert result_df[col].dtype == 'object', f"{col} should be string (object) type"
        
        # Timestamp column
        assert pd.api.types.is_datetime64_any_dtype(result_df['etl_processed_at']), \
            "etl_processed_at should be timestamp type"
        
        # Assert: Verify missing columns are added with defaults
        assert 'demographic_sk' in result_df.columns, "Should add demographic_sk column"
        assert 'geo_sk' in result_df.columns, "Should add geo_sk column"
        assert 'census_year' in result_df.columns, "Should add census_year column"
        assert result_df['census_year'].iloc[0] == 2021, "Census year should default to 2021"
        assert result_df['table_code'].iloc[0] == 'G01', "Table code should default to G01"
        
        # Assert: Verify all 18 age groups are present
        expected_age_groups = [
            'age_0_4', 'age_5_9', 'age_10_14', 'age_15_19', 'age_20_24',
            'age_25_29', 'age_30_34', 'age_35_39', 'age_40_44', 'age_45_49',
            'age_50_54', 'age_55_59', 'age_60_64', 'age_65_69', 'age_70_74',
            'age_75_79', 'age_80_84', 'age_85_plus'
        ]
        for age_group in expected_age_groups:
            assert age_group in result_df.columns, f"Missing age group: {age_group}"
        
        # Assert: Verify column ordering matches schema
        # The exact order should match CensusDemographics schema
        expected_column_order = [
            # Keys
            'demographic_sk', 'geo_sk',
            # Geographic identifiers
            'geographic_id', 'geographic_name', 'state_territory',
            'sa3_code', 'sa3_name', 'sa4_code', 'sa4_name',
            'gcc_code', 'gcc_name',
            # Demographics
            'total_population', 'males', 'females',
            # Age groups (18 columns)
        ] + expected_age_groups + [
            # Indigenous status
            'indigenous', 'non_indigenous', 'indigenous_not_stated',
            # Dwellings
            'total_private_dwellings', 'occupied_private_dwellings',
            'unoccupied_private_dwellings', 'total_families',
            # Ratios
            'dependency_ratio', 'sex_ratio', 'child_ratio', 'elderly_ratio',
            # Indicators
            'population_density', 'median_age_deviation', 'diversity_index',
            # Metadata
            'census_year', 'table_code', 'table_name', 'etl_processed_at'
        ]
        
        # Check first 10 columns are in correct order
        for i, col in enumerate(expected_column_order[:10]):
            if col in result_df.columns:
                actual_col = result_df.columns[i]
                assert actual_col == col, f"Column order mismatch at position {i}: expected {col}, got {actual_col}"
        
        # Assert: Verify extra columns are removed
        assert 'extra_column' not in result_df.columns, "Extra columns should be removed"
        
        # Verify data conversions
        assert result_df['males'].iloc[0] == 500, "String '500' should convert to int 500"
        assert result_df['females'].iloc[0] == 500, "Float 500.5 should round to int 500"