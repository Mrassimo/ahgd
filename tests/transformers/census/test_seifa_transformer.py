"""
Unit tests for SEIFATransformer.

Test-driven development for ABS SEIFA data transformation,
ensuring robust score processing, ranking generation, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.transformers.census.seifa_transformer import SEIFATransformer
from src.utils.interfaces import TransformationError, ProcessingStatus


class TestSEIFATransformer:
    """Test suite for SEIFATransformer class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Mock the configuration to avoid dependency on actual config files
        with patch('src.transformers.census.seifa_transformer.get_config_manager') as mock_config_manager, \
             patch('src.transformers.census.seifa_transformer.get_config') as mock_get_config:
            
            # Configure mock responses
            mock_get_config.side_effect = self._mock_get_config
            mock_config_manager.return_value = Mock()
            
            # Create transformer instance
            self.transformer = SEIFATransformer()
    
    def _mock_get_config(self, key: str, default=None):
        """Mock configuration values for testing."""
        config_map = {
            "transformers.census.geographic_hierarchy": True,
            "transformers.census.impute_missing": "score_median",
            "system.stop_on_error": False,
            "transformers.census.column_mappings": {},
            "schemas.census_seifa": {
                "required_indices": ["irsd_score"],
                "optional_indices": ["irsad_score", "ier_score", "ieo_score"],
                "score_range": {"min": 1, "max": 2000},
                "typical_range": {"min": 600, "max": 1400}
            },
            "transformers.census.operations": {
                "normalisation": "z_score",
                "reference_population": "australia",
                "weights": {
                    "economic": 0.4,
                    "education": 0.3,
                    "housing": 0.2,
                    "accessibility": 0.1
                },
                "ranking_levels": ["national", "state", "regional"],
                "percentile_groups": [10, 25, 50, 75, 90],
                "composite_indices": True,
                "include_state_rankings": True
            }
        }
        return config_map.get(key, default)
    
    def test_init_creates_seifa_transformer_with_correct_configuration(self):
        """Test SEIFATransformer initialisation with proper configuration."""
        # Assert: Verify transformer was created with correct attributes
        assert self.transformer is not None
        assert hasattr(self.transformer, 'config_manager')
        assert hasattr(self.transformer, 'column_mappings')
        assert hasattr(self.transformer, 'target_schema')
        assert hasattr(self.transformer, 'operations_config')
        assert hasattr(self.transformer, 'imputation_strategy')
        assert self.transformer.seifa_sk_counter == 40000
        assert self.transformer.imputation_strategy == "score_median"
        
        # Verify operations configuration
        ops_config = self.transformer.operations_config
        assert ops_config["normalisation"] == "z_score"
        assert ops_config["reference_population"] == "australia"
        assert ops_config["composite_indices"] is True
        assert "economic" in ops_config["weights"]
    
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
    
    def test_load_column_mappings_contains_required_seifa_fields(self):
        """Test that column mappings include all required SEIFA data fields."""
        # Act: Get column mappings
        mappings = self.transformer.column_mappings
        
        # Assert: Verify essential SEIFA fields are mapped
        required_fields = [
            'geographic_id', 'irsad_score', 'irsd_score', 'ier_score', 'ieo_score',
            'irsad_rank', 'irsd_rank', 'ier_rank', 'ieo_rank',
            'irsad_decile', 'irsd_decile', 'ier_decile', 'ieo_decile'
        ]
        
        for field in required_fields:
            assert field in mappings, f"Required field {field} not found in column mappings"
            assert len(mappings[field]) > 0, f"No mapping options for {field}"
    
    def test_validate_input_data_with_valid_seifa_data(self):
        """Test input validation with valid SEIFA data."""
        # Arrange: Create valid input data
        input_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021007', '101021008', '101021009'],
            'SA2_NAME_2021': ['Pyrmont - Ultimo', 'Surry Hills', 'Woolloomooloo'],
            'IRSD_Score': [956, 834, 1123],
            'IRSAD_Score': [1045, 901, 1200]
        })
        
        # Act: Validate input data
        result = self.transformer._validate_input_data(input_data)
        
        # Assert: Verify validation passes
        assert len(result) == 3
        assert result.equals(input_data)
    
    def test_validate_input_data_raises_error_for_empty_data(self):
        """Test that empty input data raises TransformationError."""
        # Arrange: Create empty DataFrame
        input_data = pd.DataFrame()
        
        # Act & Assert: Expect TransformationError
        with pytest.raises(TransformationError) as exc_info:
            self.transformer._validate_input_data(input_data)
        
        assert "Input SEIFA data is empty" in str(exc_info.value)
    
    def test_validate_input_data_raises_error_for_missing_geographic_id(self):
        """Test validation fails when no geographic identifier is present."""
        # Arrange: Create data without geographic identifiers
        input_data = pd.DataFrame({
            'IRSD_Score': [956, 834, 1123],
            'some_other_field': ['A', 'B', 'C']
        })
        
        # Act & Assert: Expect TransformationError
        with pytest.raises(TransformationError) as exc_info:
            self.transformer._validate_input_data(input_data)
        
        assert "No geographic identifier found" in str(exc_info.value)
    
    def test_validate_input_data_raises_error_for_missing_seifa_scores(self):
        """Test validation fails when no SEIFA scores are present."""
        # Arrange: Create data without SEIFA scores
        input_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021007', '101021008'],
            'SA2_NAME_2021': ['Pyrmont - Ultimo', 'Surry Hills'],
            'Population': [5000, 6000]
        })
        
        # Act & Assert: Expect TransformationError
        with pytest.raises(TransformationError) as exc_info:
            self.transformer._validate_input_data(input_data)
        
        assert "No SEIFA index scores found" in str(exc_info.value)
    
    def test_map_columns_maps_abs_format_to_standard_names(self):
        """Test column mapping from ABS format to standard names."""
        # Arrange: Create data with ABS column names
        input_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021007', '101021008'],
            'SA2_NAME_2021': ['Pyrmont - Ultimo', 'Surry Hills'],
            'IRSD_Score': [956, 834],
            'IRSAD_Score': [1045, 901],
            'IRSD_Rank': [15234, 20456]
        })
        
        # Act: Map columns
        result = self.transformer._map_columns(input_data)
        
        # Assert: Verify columns are mapped to standard names
        assert 'geographic_id' in result.columns
        assert 'geographic_name' in result.columns
        assert 'irsd_score' in result.columns
        assert 'irsad_score' in result.columns
        assert 'irsd_rank' in result.columns
    
    def test_standardise_geographic_data_sets_correct_defaults(self):
        """Test geographic data standardisation sets correct defaults."""
        # Arrange: Create data with mapped columns
        input_data = pd.DataFrame({
            'geographic_id': ['101021007', '101021008', '102031015'],
            'geographic_name': ['Pyrmont - Ultimo', 'Surry Hills', 'Melbourne CBD'],
            'state_territory': ['nsw', 'NSW', 'vic'],
            'irsd_score': [956, 834, 1123]
        })
        
        # Act: Standardise geographic data
        result = self.transformer._standardise_geographic_data(input_data)
        
        # Assert: Verify standardisation
        assert all(result['state_territory'].isin(['NSW', 'VIC']))  # Uppercase normalisation
        assert 'geographic_level' in result.columns
        assert all(result['geographic_level'] == 'SA2')  # Inferred from ID length
        assert 'census_year' in result.columns
        assert all(result['census_year'] == 2021)  # Default year
    
    def test_process_seifa_scores_converts_to_numeric_and_validates_ranges(self):
        """Test SEIFA score processing converts to numeric and validates ranges."""
        # Arrange: Create data with string scores and edge cases
        input_data = pd.DataFrame({
            'geographic_id': ['101021007', '101021008', '101021009'],
            'irsd_score': ['956', '834', 'invalid'],
            'irsad_score': [1045, 50, 1500]  # Include out-of-range value
        })
        
        # Act: Process SEIFA scores
        result = self.transformer._process_seifa_scores(input_data)
        
        # Assert: Verify score processing
        assert result['irsd_score'].dtype in ['int64', 'Int64', 'float64']
        assert result['irsad_score'].dtype in ['int64', 'Int64', 'float64']
        assert pd.isna(result.loc[2, 'irsd_score'])  # Invalid value becomes NaN
        assert result.loc[0, 'irsd_score'] == 956
        assert result.loc[1, 'irsd_score'] == 834
    
    def test_generate_rankings_and_deciles_creates_correct_rankings(self):
        """Test ranking and decile generation creates correct values."""
        # Arrange: Create data with known scores for predictable rankings
        input_data = pd.DataFrame({
            'geographic_id': ['101021007', '101021008', '101021009', '101021010'],
            'irsd_score': [600, 800, 1000, 1200],  # Low to high scores
            'state_territory': ['NSW', 'NSW', 'VIC', 'VIC']
        })
        
        # Act: Generate rankings and deciles
        result = self.transformer._generate_rankings_and_deciles(input_data)
        
        # Assert: Verify rankings (lower score = higher rank number for disadvantage)
        assert 'irsd_rank' in result.columns
        assert 'irsd_decile' in result.columns
        
        # Lowest score (600) should have rank 1 (most disadvantaged)
        assert result.loc[0, 'irsd_rank'] == 1
        # Highest score (1200) should have highest rank
        assert result.loc[3, 'irsd_rank'] == 4
        
        # Verify deciles exist and are in valid range
        assert all(result['irsd_decile'] >= 1)
        assert all(result['irsd_decile'] <= 10)
    
    def test_generate_state_rankings_creates_state_level_rankings(self):
        """Test state-level ranking generation."""
        # Arrange: Create data with multiple states
        input_data = pd.DataFrame({
            'geographic_id': ['101021007', '101021008', '102031009', '102031010'],
            'irsd_score': [600, 800, 700, 900],
            'irsad_score': [650, 850, 750, 950],
            'state_territory': ['NSW', 'NSW', 'VIC', 'VIC']
        })
        
        # Act: Generate state rankings
        self.transformer._generate_state_rankings(input_data)
        
        # Assert: Verify state rankings exist
        assert 'irsd_state_rank' in input_data.columns
        assert 'irsad_state_rank' in input_data.columns
        assert 'irsd_state_decile' in input_data.columns
        assert 'irsad_state_decile' in input_data.columns
        
        # Verify state rankings are within state groups
        nsw_data = input_data[input_data['state_territory'] == 'NSW']
        assert nsw_data.loc[0, 'irsd_state_rank'] == 1  # Lowest score in NSW
        assert nsw_data.loc[1, 'irsd_state_rank'] == 2  # Higher score in NSW
    
    def test_create_composite_indices_calculates_weighted_scores(self):
        """Test composite index creation with weighted scores."""
        # Arrange: Create data with multiple SEIFA indices
        input_data = pd.DataFrame({
            'geographic_id': ['101021007', '101021008'],
            'irsad_score': [1000, 800],
            'ier_score': [1100, 900],
            'ieo_score': [950, 850],
            'irsd_decile': [7, 4]
        })
        
        # Act: Create composite indices
        result = self.transformer._create_composite_indices(input_data)
        
        # Assert: Verify composite indices
        assert 'overall_advantage_score' in result.columns
        assert 'disadvantage_severity' in result.columns
        
        # Verify disadvantage severity mapping
        assert result.loc[0, 'disadvantage_severity'] == 'low'  # Decile 7
        assert result.loc[1, 'disadvantage_severity'] == 'high'  # Decile 4
        
        # Verify overall advantage score is numeric
        assert pd.api.types.is_numeric_dtype(result['overall_advantage_score'])
    
    def test_integrate_geographic_hierarchy_adds_metadata_fields(self):
        """Test geographic hierarchy integration adds required metadata."""
        # Arrange: Create processed data
        input_data = pd.DataFrame({
            'geographic_id': ['101021007', '101021008'],
            'irsd_score': [956, 834],
            'irsd_decile': [7, 4]
        })
        
        # Act: Integrate geographic hierarchy
        result = self.transformer._integrate_geographic_hierarchy(input_data)
        
        # Assert: Verify metadata fields added
        metadata_fields = [
            'seifa_sk', 'data_source_name', 'data_source_url',
            'extraction_date', 'quality_level', 'source_version',
            'schema_version', 'last_updated'
        ]
        
        for field in metadata_fields:
            assert field in result.columns, f"Metadata field {field} not added"
        
        # Verify surrogate keys are unique and sequential
        assert result['seifa_sk'].iloc[0] == 40000
        assert result['seifa_sk'].iloc[1] == 40001
        
        # Verify data source information
        assert all(result['data_source_name'] == 'ABS SEIFA 2021')
        assert all(result['quality_level'] == 'HIGH')
    
    def test_impute_missing_values_with_score_median_strategy(self):
        """Test missing value imputation using score median strategy."""
        # Arrange: Create data with missing values
        input_data = pd.DataFrame({
            'geographic_id': ['101021007', '101021008', '101021009'],
            'irsd_score': [956, np.nan, 834],
            'irsad_score': [1045, 901, np.nan]
        })
        
        # Act: Impute missing values
        result = self.transformer._impute_missing_values(input_data)
        
        # Assert: Verify missing values are imputed
        assert not result['irsd_score'].isna().any()
        assert not result['irsad_score'].isna().any()
        
        # Verify imputation used median values
        expected_irsd_median = (956 + 834) / 2  # 895
        expected_irsad_median = (1045 + 901) / 2  # 973
        
        assert result.loc[1, 'irsd_score'] == expected_irsd_median
        assert result.loc[2, 'irsad_score'] == expected_irsad_median
    
    def test_enforce_schema_ensures_correct_data_types_and_structure(self):
        """Test schema enforcement ensures correct data types and structure."""
        # Arrange: Create data with mixed types
        input_data = pd.DataFrame({
            'geographic_id': [101021007, 101021008],  # Numeric (should be string)
            'geographic_level': ['SA2', 'SA2'],
            'irsd_score': [956.0, 834.0],  # Float (should be integer)
            'irsd_decile': [7.0, 4.0],  # Float (should be integer)
            'extra_column': ['A', 'B'],  # Should be removed
            'census_year': [2021, 2021]
        })
        
        # Act: Enforce schema
        result = self.transformer._enforce_schema(input_data)
        
        # Assert: Verify schema compliance
        assert result['geographic_id'].dtype == object  # String type
        assert result['irsd_score'].dtype in ['int64', 'Int64']  # Integer type
        assert result['irsd_decile'].dtype in ['int64', 'Int64']  # Integer type
        
        # Verify extra columns are removed
        assert 'extra_column' not in result.columns
        
        # Verify required fields exist
        required_fields = ['geographic_id', 'geographic_level', 'census_year']
        for field in required_fields:
            assert field in result.columns
    
    def test_transform_end_to_end_with_valid_seifa_data(self):
        """Test complete end-to-end transformation with valid SEIFA data."""
        # Arrange: Create comprehensive test data
        input_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021007', '101021008', '102031009', '102031010'],
            'SA2_NAME_2021': ['Pyrmont - Ultimo', 'Surry Hills', 'Melbourne CBD', 'Southbank'],
            'STATE_CODE': ['NSW', 'NSW', 'VIC', 'VIC'],
            'IRSD_Score': [956, 834, 1123, 901],
            'IRSAD_Score': [1045, 901, 1200, 950],
            'IER_Score': [1134, 987, 1300, 1050],
            'Population': [8500, 12000, 15000, 9500]
        })
        
        # Act: Perform complete transformation
        result = self.transformer.transform(input_data)
        
        # Assert: Verify transformation success
        assert len(result) == 4
        assert 'geographic_id' in result.columns
        assert 'irsd_score' in result.columns
        assert 'irsd_decile' in result.columns
        assert 'overall_advantage_score' in result.columns
        
        # Verify processing metadata was created
        metadata = self.transformer.get_processing_metadata()
        assert metadata is not None
        assert metadata.status == ProcessingStatus.COMPLETED
        assert metadata.records_processed == 4
        assert metadata.records_failed == 0
    
    def test_transform_handles_transformation_error_gracefully(self):
        """Test transformation handles errors gracefully when stop_on_error is False."""
        # Arrange: Create invalid data that will cause errors
        input_data = pd.DataFrame({
            'invalid_column': ['A', 'B', 'C']
        })
        
        # Act: Transform with error handling
        result = self.transformer.transform(input_data)
        
        # Assert: Verify graceful error handling
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0  # Empty DataFrame returned
        
        # Verify error was logged in metadata
        metadata = self.transformer.get_processing_metadata()
        assert metadata is not None
        assert metadata.status == ProcessingStatus.FAILED
        assert metadata.records_failed >= 1
    
    def test_transform_raises_error_when_stop_on_error_is_true(self):
        """Test transformation raises TransformationError when stop_on_error is True."""
        # Arrange: Configure to stop on error
        self.transformer.stop_on_error = True
        
        # Create invalid data
        input_data = pd.DataFrame({
            'invalid_column': ['A', 'B', 'C']
        })
        
        # Act & Assert: Expect TransformationError
        with pytest.raises(TransformationError):
            self.transformer.transform(input_data)
    
    def test_create_empty_schema_dataframe_returns_correct_structure(self):
        """Test creation of empty schema DataFrame with correct structure."""
        # Act: Create empty schema DataFrame
        result = self.transformer._create_empty_schema_dataframe()
        
        # Assert: Verify structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        
        # Verify essential columns exist
        expected_columns = [
            'geographic_id', 'geographic_level', 'census_year',
            'irsd_score', 'irsad_score', 'ier_score', 'ieo_score',
            'irsd_decile', 'irsad_decile'
        ]
        
        for col in expected_columns:
            assert col in result.columns
    
    def test_seifa_transformer_processes_multiple_indices_correctly(self):
        """Test transformer correctly processes all four SEIFA indices."""
        # Arrange: Create data with all SEIFA indices
        input_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021007', '101021008'],
            'IRSD_Score': [956, 834],
            'IRSAD_Score': [1045, 901],
            'IER_Score': [1134, 987],
            'IEO_Score': [1089, 923],
            'STATE_CODE': ['NSW', 'NSW']
        })
        
        # Act: Transform data
        result = self.transformer.transform(input_data)
        
        # Assert: Verify all indices are processed
        seifa_indices = ['irsd_score', 'irsad_score', 'ier_score', 'ieo_score']
        seifa_deciles = ['irsd_decile', 'irsad_decile', 'ier_decile', 'ieo_decile']
        
        for index in seifa_indices:
            assert index in result.columns
            assert not result[index].isna().any()
        
        for decile in seifa_deciles:
            assert decile in result.columns
            # Check deciles are in valid range where not null
            valid_deciles = result[decile].dropna()
            if len(valid_deciles) > 0:
                assert all(valid_deciles >= 1)
                assert all(valid_deciles <= 10)
    
    def test_seifa_transformer_geographic_concordance_integration(self):
        """Test SEIFA transformer geographic concordance features."""
        # Arrange: Create data with geographic hierarchy
        input_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021007', '102031009', '301041012'],
            'SA2_NAME_2021': ['Pyrmont - Ultimo', 'Melbourne CBD', 'Brisbane CBD'],
            'STATE_CODE': ['NSW', 'VIC', 'QLD'],
            'IRSD_Score': [956, 1123, 890]
        })
        
        # Act: Transform with geographic integration
        result = self.transformer.transform(input_data)
        
        # Assert: Verify geographic integration
        assert 'geographic_level' in result.columns
        assert all(result['geographic_level'] == 'SA2')
        
        # Verify state normalisation
        assert set(result['state_territory'].unique()) == {'NSW', 'VIC', 'QLD'}
        
        # Verify surrogate keys are assigned
        assert 'seifa_sk' in result.columns
        assert result['seifa_sk'].nunique() == len(result)  # All unique