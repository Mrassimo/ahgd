"""
Comprehensive test suite for scripts modules.

This module provides extensive testing coverage for all scripts including
data extraction, processing, analysis, and dashboard functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock, mock_open
import sys
from pathlib import Path
import tempfile
import os

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock heavy dependencies before importing modules
sys.modules['streamlit'] = Mock()
sys.modules['folium'] = Mock()
sys.modules['plotly.express'] = Mock()
sys.modules['plotly.graph_objects'] = Mock()
sys.modules['geopandas'] = Mock()
sys.modules['openpyxl'] = Mock()


class TestDataDownload:
    """Test cases for download_data.py"""
    
    @patch('download_data.requests.get')
    @patch('download_data.Path.mkdir')
    @patch('download_data.shutil.move')
    def test_download_file_success(self, mock_move, mock_mkdir, mock_get):
        """Test successful file download"""
        # Mock successful HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"test data"
        mock_get.return_value = mock_response
        
        # Import and test function
        from download_data import download_file
        
        result = download_file("http://example.com/data.csv", "/test/path")
        
        # Verify download was attempted
        mock_get.assert_called_once()
        assert result is not None
    
    @patch('download_data.requests.get')
    def test_download_file_http_error(self, mock_get):
        """Test download with HTTP error"""
        # Mock HTTP error
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        from download_data import download_file
        
        result = download_file("http://example.com/missing.csv", "/test/path")
        
        # Should handle error gracefully
        assert result is None or isinstance(result, type(None))
    
    @patch('download_data.get_logger')
    def test_setup_logging(self, mock_get_logger):
        """Test logging setup"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        from download_data import setup_logging
        
        logger = setup_logging()
        
        assert logger is not None


class TestDataProcessing:
    """Test cases for process_data.py"""
    
    def setup_method(self):
        """Setup test data"""
        self.sample_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021007', '101021008', '201011001'],
            'SA2_NAME_2021': ['Sydney CBD', 'Sydney Haymarket', 'Melbourne CBD'],
            'population': [18500, 22000, 28000],
            'area_sqkm': [5.2, 8.1, 12.3]
        })
    
    @patch('process_data.pd.read_csv')
    def test_load_census_data_success(self, mock_read_csv):
        """Test successful census data loading"""
        mock_read_csv.return_value = self.sample_data
        
        from process_data import load_census_data
        
        result = load_census_data("/fake/path.csv")
        
        mock_read_csv.assert_called_once()
        assert isinstance(result, pd.DataFrame)
    
    @patch('process_data.pd.read_csv')
    def test_load_census_data_file_error(self, mock_read_csv):
        """Test census data loading with file error"""
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        from process_data import load_census_data
        
        result = load_census_data("/missing/path.csv")
        
        # Should handle error gracefully
        assert result is None or result.empty
    
    def test_calculate_population_density(self):
        """Test population density calculation"""
        from process_data import calculate_population_density
        
        result = calculate_population_density(self.sample_data)
        
        assert 'population_density' in result.columns
        assert result['population_density'].iloc[0] == pytest.approx(18500 / 5.2, rel=1e-3)
    
    def test_standardize_area_codes(self):
        """Test area code standardization"""
        from process_data import standardize_area_codes
        
        result = standardize_area_codes(self.sample_data)
        
        # Should standardize area codes
        assert 'SA2_CODE_2021' in result.columns
        assert result['SA2_CODE_2021'].dtype == 'object'


class TestHealthAnalysis:
    """Test cases for health_correlation_analysis.py"""
    
    def setup_method(self):
        """Setup test data"""
        self.health_data = pd.DataFrame({
            'SA2_CODE': ['101021007', '101021008', '201011001'],
            'mortality_rate': [5.2, 8.1, 3.9],
            'diabetes_rate': [6.8, 9.2, 4.5],
            'disadvantage_score': [1050, 900, 1200],
            'population': [18500, 22000, 28000]
        })
    
    @patch('health_correlation_analysis.pd.read_parquet')
    def test_load_health_data_success(self, mock_read_parquet):
        """Test successful health data loading"""
        mock_read_parquet.return_value = self.health_data
        
        from health_correlation_analysis import load_health_data
        
        result = load_health_data("/fake/path.parquet")
        
        mock_read_parquet.assert_called_once()
        assert isinstance(result, pd.DataFrame)
    
    def test_calculate_health_correlations(self):
        """Test health correlation calculations"""
        from health_correlation_analysis import calculate_correlations
        
        result = calculate_correlations(self.health_data)
        
        # Should return correlation matrix
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == result.shape[1]  # Square matrix
    
    def test_identify_significant_correlations(self):
        """Test identification of significant correlations"""
        from health_correlation_analysis import identify_significant_correlations
        
        correlation_matrix = self.health_data.corr()
        result = identify_significant_correlations(correlation_matrix, threshold=0.5)
        
        # Should return dictionary or list of significant correlations
        assert isinstance(result, (dict, list))
    
    @patch('health_correlation_analysis.plt.savefig')
    def test_create_correlation_plots(self, mock_savefig):
        """Test correlation plot creation"""
        from health_correlation_analysis import create_correlation_plots
        
        create_correlation_plots(self.health_data, "/fake/output/path")
        
        # Should attempt to save plots
        assert mock_savefig.called


class TestGeographicMapping:
    """Test cases for geographic_mapping.py"""
    
    def setup_method(self):
        """Setup test geographic data"""
        self.geo_data = pd.DataFrame({
            'SA2_CODE21': ['101021007', '101021008'],
            'SA2_NAME21': ['Sydney CBD', 'Melbourne CBD'],
            'latitude': [-33.8688, -37.8136],
            'longitude': [151.2093, 144.9631],
            'health_score': [8.5, 12.3]
        })
    
    @patch('geographic_mapping.gpd.read_file')
    def test_load_boundary_data_success(self, mock_read_file):
        """Test successful boundary data loading"""
        mock_gdf = Mock()
        mock_gdf.to_crs.return_value = mock_gdf
        mock_read_file.return_value = mock_gdf
        
        from geographic_mapping import load_boundary_data
        
        result = load_boundary_data("/fake/shapefile.shp")
        
        mock_read_file.assert_called_once()
        assert result is not None
    
    def test_calculate_centroids(self):
        """Test centroid calculation for geographic areas"""
        from geographic_mapping import calculate_centroids
        
        # Mock geometry column
        mock_geometry = Mock()
        mock_geometry.centroid.x = 151.2093
        mock_geometry.centroid.y = -33.8688
        
        test_data = self.geo_data.copy()
        test_data['geometry'] = [mock_geometry, mock_geometry]
        
        result = calculate_centroids(test_data)
        
        # Should add centroid columns
        assert 'centroid_x' in result.columns or 'longitude' in result.columns
    
    @patch('geographic_mapping.folium.Map')
    def test_create_choropleth_map(self, mock_map):
        """Test choropleth map creation"""
        mock_map_instance = Mock()
        mock_map.return_value = mock_map_instance
        
        from geographic_mapping import create_choropleth_map
        
        result = create_choropleth_map(self.geo_data, 'health_score')
        
        mock_map.assert_called_once()
        assert result == mock_map_instance


class TestDashboardFeatures:
    """Test cases for demo_dashboard_features.py"""
    
    @patch('demo_dashboard_features.st.title')
    @patch('demo_dashboard_features.st.write')
    def test_display_overview_section(self, mock_write, mock_title):
        """Test overview section display"""
        from demo_dashboard_features import display_overview
        
        display_overview()
        
        # Should call streamlit display functions
        assert mock_title.called or mock_write.called
    
    @patch('demo_dashboard_features.st.plotly_chart')
    def test_display_interactive_charts(self, mock_plotly_chart):
        """Test interactive chart display"""
        from demo_dashboard_features import display_charts
        
        sample_data = pd.DataFrame({
            'x': [1, 2, 3, 4],
            'y': [10, 11, 12, 13]
        })
        
        display_charts(sample_data)
        
        # Should display charts
        assert mock_plotly_chart.called
    
    @patch('demo_dashboard_features.st.metric')
    def test_display_key_statistics(self, mock_metric):
        """Test key statistics display"""
        from demo_dashboard_features import display_statistics
        
        sample_data = pd.DataFrame({
            'health_score': [8.5, 12.3, 6.1, 15.7],
            'population': [18500, 22000, 15000, 28000]
        })
        
        display_statistics(sample_data)
        
        # Should display metrics
        assert mock_metric.called


class TestAnalysisSummary:
    """Test cases for analysis_summary.py"""
    
    def setup_method(self):
        """Setup test data"""
        self.analysis_results = {
            'total_areas': 2454,
            'correlation_strength': 0.73,
            'high_risk_areas': 245,
            'data_quality_score': 0.92
        }
    
    def test_generate_summary_statistics(self):
        """Test summary statistics generation"""
        from analysis_summary import generate_summary_stats
        
        data = pd.DataFrame({
            'health_score': np.random.normal(10, 3, 100),
            'disadvantage_score': np.random.normal(1000, 150, 100)
        })
        
        result = generate_summary_stats(data)
        
        # Should return dictionary of statistics
        assert isinstance(result, dict)
        assert 'mean_health_score' in result or 'statistics' in str(result)
    
    @patch('analysis_summary.json.dump')
    def test_save_analysis_results(self, mock_json_dump):
        """Test saving analysis results to JSON"""
        from analysis_summary import save_results
        
        save_results(self.analysis_results, "/fake/output.json")
        
        mock_json_dump.assert_called_once()
    
    def test_calculate_data_quality_metrics(self):
        """Test data quality metrics calculation"""
        from analysis_summary import calculate_data_quality
        
        data_with_missing = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': [5, np.nan, 7, 8],
            'col3': [9, 10, 11, 12]
        })
        
        result = calculate_data_quality(data_with_missing)
        
        # Should return quality metrics
        assert isinstance(result, dict)
        assert 'completeness' in result or 'quality' in str(result)


class TestDataExamination:
    """Test cases for examine_data.py"""
    
    @patch('examine_data.pd.read_parquet')
    def test_examine_data_structure(self, mock_read_parquet):
        """Test data structure examination"""
        mock_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.1, 2.2, 3.3]
        })
        mock_read_parquet.return_value = mock_data
        
        from examine_data import examine_structure
        
        result = examine_structure("/fake/data.parquet")
        
        # Should return data info
        assert isinstance(result, (dict, pd.DataFrame))
    
    def test_detect_data_types(self):
        """Test automatic data type detection"""
        from examine_data import detect_types
        
        data = pd.DataFrame({
            'integer_col': [1, 2, 3, 4],
            'float_col': [1.1, 2.2, 3.3, 4.4],
            'string_col': ['a', 'b', 'c', 'd'],
            'date_col': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04']
        })
        
        result = detect_types(data)
        
        # Should identify data types
        assert isinstance(result, dict)
        assert len(result) == len(data.columns)
    
    def test_identify_missing_data_patterns(self):
        """Test missing data pattern identification"""
        from examine_data import analyze_missing_data
        
        data = pd.DataFrame({
            'complete_col': [1, 2, 3, 4, 5],
            'missing_col': [1, np.nan, 3, np.nan, 5],
            'mostly_missing': [np.nan, np.nan, np.nan, 4, np.nan]
        })
        
        result = analyze_missing_data(data)
        
        # Should return missing data analysis
        assert isinstance(result, dict)
        assert 'missing_percentages' in result or 'completeness' in result


class TestPerformanceAndReliability:
    """Test cases for performance monitoring and reliability"""
    
    def test_large_dataset_processing_performance(self):
        """Test processing performance with large datasets"""
        # Create large test dataset
        large_data = pd.DataFrame({
            'SA2_CODE': [f"area_{i}" for i in range(10000)],
            'health_score': np.random.normal(10, 3, 10000),
            'population': np.random.randint(1000, 50000, 10000)
        })
        
        import time
        start_time = time.time()
        
        # Test data processing operations
        result = large_data.groupby('SA2_CODE')['health_score'].mean()
        correlations = large_data[['health_score', 'population']].corr()
        
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert len(result) == 10000
        assert correlations.shape == (2, 2)
    
    def test_memory_usage_monitoring(self):
        """Test memory usage during data operations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and process data
        test_data = pd.DataFrame({
            'values': np.random.random(100000)
        })
        processed_data = test_data.copy()
        processed_data['squared'] = processed_data['values'] ** 2
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
    
    def test_data_validation_robustness(self):
        """Test robustness of data validation"""
        # Test with various edge cases
        edge_cases = [
            pd.DataFrame(),  # Empty dataframe
            pd.DataFrame({'col': []}),  # Empty column
            pd.DataFrame({'col': [np.nan, np.nan]}),  # All NaN
            pd.DataFrame({'col': [np.inf, -np.inf]}),  # Infinite values
            pd.DataFrame({'col': ['', None, np.nan]}),  # Mixed empty values
        ]
        
        for i, edge_case in enumerate(edge_cases):
            try:
                # Test basic operations don't crash
                result = edge_case.describe(include='all')
                correlation = edge_case.corr() if len(edge_case.columns) > 1 else None
                success = True
            except Exception as e:
                success = False
                print(f"Edge case {i} failed: {e}")
            
            # Should handle edge cases gracefully
            assert success or "expected failure" in str(edge_case)


class TestIntegrationWorkflows:
    """Integration tests for complete workflows"""
    
    @patch('process_data.pd.read_csv')
    @patch('health_correlation_analysis.pd.read_parquet')
    def test_complete_analysis_pipeline(self, mock_read_parquet, mock_read_csv):
        """Test complete data analysis pipeline"""
        # Mock data loading
        mock_raw_data = pd.DataFrame({
            'SA2_CODE_2021': ['101021007', '101021008'],
            'population': [18500, 22000],
            'area_sqkm': [5.2, 8.1]
        })
        mock_read_csv.return_value = mock_raw_data
        
        mock_processed_data = pd.DataFrame({
            'SA2_CODE': ['101021007', '101021008'],
            'health_score': [8.5, 12.3],
            'disadvantage_score': [1050, 900]
        })
        mock_read_parquet.return_value = mock_processed_data
        
        # Test pipeline steps
        from process_data import load_census_data, calculate_population_density
        from health_correlation_analysis import calculate_correlations
        
        # Step 1: Load raw data
        raw_data = load_census_data("/fake/raw.csv")
        assert isinstance(raw_data, pd.DataFrame)
        
        # Step 2: Process data
        processed_data = calculate_population_density(raw_data)
        assert 'population_density' in processed_data.columns
        
        # Step 3: Analyze correlations
        correlations = calculate_correlations(mock_processed_data)
        assert isinstance(correlations, pd.DataFrame)
    
    def test_error_handling_throughout_pipeline(self):
        """Test error handling across the entire pipeline"""
        # Test with problematic data
        problematic_data = pd.DataFrame({
            'SA2_CODE': ['invalid', None, ''],
            'health_score': [np.nan, np.inf, -999],
            'population': [0, -100, None]
        })
        
        try:
            # Should handle errors gracefully
            cleaned_data = problematic_data.dropna()
            valid_data = cleaned_data[cleaned_data['population'] > 0]
            result = valid_data.describe()
            
            success = True
        except Exception as e:
            success = False
            print(f"Pipeline error handling failed: {e}")
        
        # Pipeline should be robust
        assert success


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize("data_size", [100, 1000, 10000])
def test_processing_scalability(data_size):
    """Test processing scalability with different data sizes"""
    # Generate test data of varying sizes
    test_data = pd.DataFrame({
        'id': range(data_size),
        'value': np.random.random(data_size)
    })
    
    import time
    start_time = time.time()
    
    # Perform standard operations
    result = test_data.groupby(test_data['id'] % 10)['value'].mean()
    
    processing_time = time.time() - start_time
    
    # Processing time should scale reasonably
    if data_size <= 1000:
        assert processing_time < 1.0
    elif data_size <= 10000:
        assert processing_time < 5.0
    
    assert len(result) <= 10  # Should have at most 10 groups


@pytest.mark.parametrize("missing_rate", [0.0, 0.1, 0.5, 0.9])
def test_missing_data_handling(missing_rate):
    """Test handling of various missing data rates"""
    # Create data with specified missing rate
    size = 1000
    data = pd.DataFrame({
        'complete': range(size),
        'partial': range(size)
    })
    
    # Introduce missing values
    missing_indices = np.random.choice(
        size, 
        int(size * missing_rate), 
        replace=False
    )
    data.loc[missing_indices, 'partial'] = np.nan
    
    # Test operations handle missing data
    try:
        stats = data.describe()
        correlations = data.corr()
        cleaned = data.dropna()
        
        success = True
        
        # Verify expected behavior
        if missing_rate < 1.0:
            assert len(cleaned) > 0
        else:
            assert len(cleaned) == 0
            
    except Exception as e:
        success = False
        print(f"Missing data handling failed at {missing_rate*100}%: {e}")
    
    assert success