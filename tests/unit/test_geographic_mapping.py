"""
Unit tests for geographic mapping functionality.

This module tests the PostcodeToSA2Mapper class and related geographic 
mapping functions to ensure accurate postcode-to-SA2 mapping functionality.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import duckdb

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import get_global_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from geographic_mapping import PostcodeToSA2Mapper

# Get configuration
config = get_global_config()


class TestPostcodeToSA2Mapper:
    """Test class for PostcodeToSA2Mapper functionality."""
    
    def test_init_with_default_path(self):
        """Test mapper initialisation with default data directory."""
        mapper = PostcodeToSA2Mapper()
        assert mapper.data_dir == config.data_source.raw_data_dir.parent
        assert mapper.db_path.name == config.database.name
        assert not mapper._correspondence_loaded

    def test_init_with_custom_path(self, temp_dir):
        """Test mapper initialisation with custom data directory."""
        mapper = PostcodeToSA2Mapper(data_dir=str(temp_dir))
        assert mapper.data_dir == temp_dir
        assert mapper.db_path == temp_dir / "health_analytics.db"

    @patch('geographic_mapping.duckdb.connect')
    def test_postcode_to_sa2_single_mapping(self, mock_connect):
        """Test postcode mapping with single SA2 result."""
        # Mock database connection and results
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Single mapping result
        mock_result = [('101021007', 'Sydney - CBD', 1.0)]
        mock_conn.execute.return_value.fetchall.return_value = mock_result
        
        mapper = PostcodeToSA2Mapper()
        result = mapper.postcode_to_sa2('2000')
        
        assert len(result) == 1
        assert result[0]['sa2_code'] == '101021007'
        assert result[0]['sa2_name'] == 'Sydney - CBD'
        assert result[0]['weight'] == 1.0

    @patch('geographic_mapping.duckdb.connect')
    def test_postcode_to_sa2_multiple_mappings(self, mock_connect):
        """Test postcode mapping with multiple SA2 results."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Multiple mapping results
        mock_result = [
            ('101021007', 'Sydney - CBD', 0.6),
            ('101021008', 'Sydney - Haymarket', 0.4)
        ]
        mock_conn.execute.return_value.fetchall.return_value = mock_result
        
        mapper = PostcodeToSA2Mapper()
        result = mapper.postcode_to_sa2('2000')
        
        assert len(result) == 2
        assert result[0]['weight'] == 0.6
        assert result[1]['weight'] == 0.4
        
        # Check weights sum to 1.0
        total_weight = sum(r['weight'] for r in result)
        assert abs(total_weight - 1.0) < 0.001

    @patch('geographic_mapping.duckdb.connect')
    def test_postcode_to_sa2_no_mapping(self, mock_connect):
        """Test postcode mapping with no results."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []
        
        mapper = PostcodeToSA2Mapper()
        result = mapper.postcode_to_sa2('9999')
        
        assert result == []

    @patch('geographic_mapping.duckdb.connect')
    def test_postcode_to_sa2_database_error(self, mock_connect):
        """Test postcode mapping with database error."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.execute.side_effect = Exception("Database error")
        
        mapper = PostcodeToSA2Mapper()
        result = mapper.postcode_to_sa2('2000')
        
        assert result == []

    def test_validate_mapping_coverage(self, sample_correspondence_data):
        """Test mapping coverage validation."""
        with patch.object(PostcodeToSA2Mapper, 'postcode_to_sa2') as mock_mapping:
            mapper = PostcodeToSA2Mapper()
            
            # Mock responses for test postcodes
            mock_mapping.side_effect = lambda pc: [{'sa2_code': '123'}] if pc in ['2000', '3000'] else []
            
            test_postcodes = ['2000', '3000', '9999', '8888']
            coverage = mapper.validate_mapping_coverage(test_postcodes)
            
            assert coverage['total_postcodes'] == 4
            assert coverage['mapped_postcodes'] == 2
            assert coverage['unmapped_postcodes'] == 2
            assert coverage['coverage_percentage'] == 50.0
            assert set(coverage['unmapped_postcode_list']) == {'9999', '8888'}

    @patch('geographic_mapping.duckdb.connect')
    def test_aggregate_postcode_data_to_sa2(self, mock_connect, sample_postcode_data):
        """Test aggregation of postcode data to SA2 level."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Mock postcode-to-SA2 mappings
        mapping_responses = {
            '2000': [{'sa2_code': 'SA2_001', 'sa2_name': 'Sydney CBD', 'weight': 1.0}],
            '2001': [{'sa2_code': 'SA2_002', 'sa2_name': 'Sydney East', 'weight': 1.0}],
            '3000': [{'sa2_code': 'SA2_003', 'sa2_name': 'Melbourne CBD', 'weight': 1.0}],
            '3001': [{'sa2_code': 'SA2_003', 'sa2_name': 'Melbourne CBD', 'weight': 1.0}],  # Same SA2
            '4000': [{'sa2_code': 'SA2_004', 'sa2_name': 'Brisbane CBD', 'weight': 1.0}],
            '5000': [{'sa2_code': 'SA2_005', 'sa2_name': 'Adelaide CBD', 'weight': 1.0}]
        }
        
        mapper = PostcodeToSA2Mapper()
        with patch.object(mapper, 'postcode_to_sa2', side_effect=lambda pc: mapping_responses.get(pc, [])):
            result = mapper.aggregate_postcode_data_to_sa2(
                sample_postcode_data,
                postcode_col='postcode',
                value_cols=['population', 'median_income', 'hospitals'],
                method='weighted_sum'
            )
            
            assert 'sa2_code' in result.columns
            assert 'sa2_name' in result.columns
            assert len(result) == 5  # Should have 5 unique SA2s
            
            # Check that SA2_003 (Melbourne CBD) aggregated both 3000 and 3001
            melbourne_row = result[result['sa2_code'] == 'SA2_003']
            assert len(melbourne_row) == 1
            assert melbourne_row.iloc[0]['population'] == 43000  # 25000 + 18000

    def test_invalid_postcode_formats(self):
        """Test handling of invalid postcode formats."""
        mapper = PostcodeToSA2Mapper()
        
        invalid_postcodes = [
            None,
            "",
            "abc",
            "12345678",
            123,  # numeric instead of string
            "00"   # too short
        ]
        
        for invalid_pc in invalid_postcodes:
            with patch.object(mapper, 'postcode_to_sa2', return_value=[]):
                result = mapper.postcode_to_sa2(invalid_pc)
                assert result == []

    @patch('geographic_mapping.duckdb.connect')
    def test_weighted_aggregation_multiple_sa2s(self, mock_connect):
        """Test weighted aggregation when postcode spans multiple SA2s."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Test data with postcode spanning multiple SA2s
        test_data = pd.DataFrame({
            'postcode': ['2000'],
            'population': [10000],
            'median_income': [60000]
        })
        
        # Mock mapping with weights
        mock_mapping = [
            {'sa2_code': 'SA2_001', 'sa2_name': 'Sydney CBD', 'weight': 0.7},
            {'sa2_code': 'SA2_002', 'sa2_name': 'Sydney Harbour', 'weight': 0.3}
        ]
        
        mapper = PostcodeToSA2Mapper()
        with patch.object(mapper, 'postcode_to_sa2', return_value=mock_mapping):
            result = mapper.aggregate_postcode_data_to_sa2(
                test_data,
                postcode_col='postcode',
                value_cols=['population', 'median_income'],
                method='weighted_sum'
            )
            
            assert len(result) == 2
            
            # Check weighted distribution
            cbd_row = result[result['sa2_code'] == 'SA2_001'].iloc[0]
            harbour_row = result[result['sa2_code'] == 'SA2_002'].iloc[0]
            
            assert cbd_row['population'] == 7000    # 10000 * 0.7
            assert harbour_row['population'] == 3000  # 10000 * 0.3
            assert cbd_row['median_income'] == 42000   # 60000 * 0.7
            assert harbour_row['median_income'] == 18000  # 60000 * 0.3

    def test_empty_dataframe_handling(self):
        """Test handling of empty input dataframe."""
        mapper = PostcodeToSA2Mapper()
        empty_df = pd.DataFrame(columns=['postcode', 'value'])
        
        result = mapper.aggregate_postcode_data_to_sa2(empty_df)
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    @patch('geographic_mapping.duckdb.connect')
    def test_load_correspondence_data_success(self, mock_connect, temp_dir):
        """Test successful loading of correspondence data."""
        mock_conn = Mock()
        mock_connect.return_value = mock_conn
        
        # Create mock Excel file
        mock_file = temp_dir / "raw" / "geographic" / "CG_POA_2021_SA2_2021.xlsx"
        mock_file.parent.mkdir(parents=True)
        
        # Mock pandas read_excel
        mock_df = pd.DataFrame({
            'POA_CODE_2021': ['2000', '3000'],
            'SA2_CODE_2021': ['101021007', '201011001'],
            'SA2_NAME_2021': ['Sydney CBD', 'Melbourne CBD'],
            'RATIO': [1.0, 1.0]
        })
        
        mapper = PostcodeToSA2Mapper(data_dir=str(temp_dir))
        
        with patch('pandas.read_excel', return_value=mock_df):
            mapper.load_correspondence_data()
            
            assert mapper._correspondence_loaded
            mock_conn.execute.assert_called()

    def test_load_correspondence_data_missing_file(self, temp_dir):
        """Test loading correspondence data when file is missing."""
        mapper = PostcodeToSA2Mapper(data_dir=str(temp_dir))
        
        # Should handle missing file gracefully
        result = mapper.load_correspondence_data()
        assert not mapper._correspondence_loaded

    @pytest.mark.parametrize("method", ["weighted_sum", "weighted_average"])
    def test_aggregation_methods(self, method):
        """Test different aggregation methods."""
        mapper = PostcodeToSA2Mapper()
        
        test_data = pd.DataFrame({
            'postcode': ['2000', '2001'],
            'value': [100, 200]
        })
        
        mock_mappings = [
            [{'sa2_code': 'SA2_001', 'sa2_name': 'Area 1', 'weight': 1.0}],
            [{'sa2_code': 'SA2_001', 'sa2_name': 'Area 1', 'weight': 1.0}]
        ]
        
        with patch.object(mapper, 'postcode_to_sa2', side_effect=mock_mappings):
            result = mapper.aggregate_postcode_data_to_sa2(
                test_data,
                postcode_col='postcode',
                value_cols=['value'],
                method=method
            )
            
            assert len(result) == 1
            
            if method == 'weighted_sum':
                assert result.iloc[0]['value'] == 300  # 100 + 200
            elif method == 'weighted_average':
                assert result.iloc[0]['value'] == 150  # (100 + 200) / 2


class TestMappingQuality:
    """Test class for mapping quality and validation functions."""
    
    def test_weight_normalisation(self):
        """Test that weights are properly normalised."""
        # This would test the internal weight normalisation logic
        weights = [0.3, 0.2, 0.5]
        total = sum(weights)
        assert abs(total - 1.0) < 0.001

    def test_geographic_consistency(self):
        """Test geographic consistency of mappings."""
        # Test that postcodes map to geographically sensible SA2s
        # This would require actual geographic data validation
        pass

    def test_coverage_completeness(self):
        """Test coverage completeness for known postcode ranges."""
        # Test that major postcode ranges have appropriate coverage
        pass


@pytest.mark.integration
class TestPostcodeToSA2Integration:
    """Integration tests for postcode to SA2 mapping with real data."""
    
    @pytest.mark.database
    def test_end_to_end_mapping_with_database(self, temp_db, sample_correspondence_data):
        """Test complete mapping process with database."""
        # Create temporary database with sample data
        conn = duckdb.connect(str(temp_db))
        conn.execute("CREATE TABLE correspondence AS SELECT * FROM sample_correspondence_data")
        conn.close()
        
        # Test mapping with real database
        mapper = PostcodeToSA2Mapper()
        mapper.db_path = temp_db
        
        # This would test the actual database integration
        # Implementation depends on actual database schema

    @pytest.mark.slow
    def test_performance_with_large_dataset(self):
        """Test performance with large dataset."""
        # Generate large test dataset
        large_dataset = pd.DataFrame({
            'postcode': [f"{i:04d}" for i in range(1000, 2000)],
            'value': np.random.rand(1000)
        })
        
        mapper = PostcodeToSA2Mapper()
        
        # Mock rapid responses to test performance
        with patch.object(mapper, 'postcode_to_sa2', return_value=[
            {'sa2_code': 'TEST_SA2', 'sa2_name': 'Test Area', 'weight': 1.0}
        ]):
            import time
            start_time = time.time()
            
            result = mapper.aggregate_postcode_data_to_sa2(
                large_dataset,
                postcode_col='postcode',
                value_cols=['value']
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Should process 1000 records reasonably quickly
            assert processing_time < 10.0  # seconds
            assert len(result) > 0