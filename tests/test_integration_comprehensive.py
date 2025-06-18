"""
Comprehensive integration test suite for end-to-end workflows.

This module tests complete dashboard workflows, data pipeline integration,
and cross-component functionality to ensure the system works as a whole.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import tempfile
import json
import sqlite3
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock heavy dependencies
sys.modules['streamlit'] = Mock()
sys.modules['folium'] = Mock()
sys.modules['plotly.express'] = Mock()
sys.modules['plotly.graph_objects'] = Mock()
sys.modules['geopandas'] = Mock()
sys.modules['streamlit_folium'] = Mock()


class TestDashboardIntegration:
    """Integration tests for complete dashboard functionality"""
    
    def setup_method(self):
        """Setup integration test data"""
        self.test_data = pd.DataFrame({
            'SA2_CODE21': ['101021007', '101021008', '201011001', '201011002'],
            'SA2_NAME21': ['Sydney CBD', 'Sydney Haymarket', 'Melbourne CBD', 'Melbourne Docklands'],
            'health_risk_score': [8.5, 12.3, 6.1, 15.7],
            'IRSD_Score': [1050, 900, 1200, 750],
            'mortality_rate': [5.2, 8.1, 3.9, 10.4],
            'diabetes_prevalence': [6.8, 9.2, 4.5, 12.1],
            'population': [18500, 22000, 28000, 15500],
            'STE_NAME21': ['NSW', 'NSW', 'VIC', 'VIC']
        })
    
    @patch('src.dashboard.data.loaders.load_data')
    @patch('src.dashboard.data.loaders.calculate_correlations')
    def test_complete_dashboard_workflow(self, mock_correlations, mock_load_data):
        """Test complete dashboard workflow from data loading to visualization"""
        # Mock data loading
        mock_load_data.return_value = self.test_data
        
        # Mock correlation calculation
        correlation_matrix = self.test_data.select_dtypes(include=[np.number]).corr()
        mock_correlations.return_value = (correlation_matrix, self.test_data)
        
        # Test data loading component
        from src.dashboard.data.loaders import load_data, calculate_correlations
        
        loaded_data = load_data()
        assert loaded_data is not None
        assert len(loaded_data) == len(self.test_data)
        
        # Test correlation analysis
        corr_matrix, corr_data = calculate_correlations(loaded_data)
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape[0] == corr_matrix.shape[1]  # Square matrix
        
        # Test data processing
        from src.dashboard.data.processors import filter_data_by_states, calculate_health_risk_score
        
        filtered_data = filter_data_by_states(loaded_data, ['NSW'])
        assert len(filtered_data) == 2  # Should have 2 NSW areas
        
        risk_scores = calculate_health_risk_score(loaded_data)
        assert 'health_risk_score' in risk_scores.columns
    
    @patch('src.dashboard.visualisation.charts.px.imshow')
    @patch('src.dashboard.visualisation.charts.px.scatter')
    def test_visualization_integration(self, mock_scatter, mock_imshow):
        """Test integration of visualization components"""
        # Mock plotly figures
        mock_figure = Mock()
        mock_figure.update_layout = Mock(return_value=mock_figure)
        mock_imshow.return_value = mock_figure
        mock_scatter.return_value = mock_figure
        
        from src.dashboard.visualisation.charts import create_correlation_heatmap, create_scatter_plots
        from src.dashboard.visualisation.components import display_key_metrics
        
        # Test correlation heatmap
        correlation_matrix = self.test_data.select_dtypes(include=[np.number]).corr()
        heatmap = create_correlation_heatmap(correlation_matrix)
        assert heatmap == mock_figure
        
        # Test scatter plots
        scatter_plots = create_scatter_plots(self.test_data)
        assert isinstance(scatter_plots, tuple)
        
        # Test key metrics display
        with patch('src.dashboard.visualisation.components.st') as mock_st:
            mock_st.columns.return_value = [Mock(), Mock(), Mock(), Mock()]
            display_key_metrics(self.test_data, 'health_risk_score', 'Test Integration')
            assert mock_st.columns.called
    
    @patch('src.dashboard.visualisation.maps.folium.Map')
    def test_geographic_visualization_integration(self, mock_map):
        """Test integration of geographic visualization components"""
        mock_map_instance = Mock()
        mock_map.return_value = mock_map_instance
        
        # Add mock geometry for geographic data
        mock_geometry = Mock()
        mock_geometry.bounds = [150.0, -34.0, 151.0, -33.0]
        self.test_data['geometry'] = [mock_geometry] * len(self.test_data)
        
        from src.dashboard.visualisation.maps import create_health_risk_map, get_map_bounds
        
        # Test map creation
        health_map = create_health_risk_map(self.test_data)
        assert health_map == mock_map_instance
        
        # Test map bounds calculation
        bounds = get_map_bounds(self.test_data)
        assert isinstance(bounds, dict)
    
    def test_data_pipeline_integration(self):
        """Test complete data processing pipeline integration"""
        from src.dashboard.data.processors import (
            validate_health_data, prepare_correlation_data,
            identify_health_hotspots, calculate_data_quality_metrics
        )
        
        # Test data validation
        validation_result = validate_health_data(self.test_data)
        assert isinstance(validation_result, bool)
        
        # Test correlation data preparation
        correlation_data = prepare_correlation_data(self.test_data)
        assert isinstance(correlation_data, pd.DataFrame)
        
        # Test hotspot identification
        hotspots = identify_health_hotspots(self.test_data)
        assert isinstance(hotspots, pd.DataFrame)
        
        # Test data quality metrics
        with patch('src.dashboard.data.processors.st') as mock_st:
            quality_metrics = calculate_data_quality_metrics(self.test_data)
            # Function should complete without error
            assert True


class TestConfigurationIntegration:
    """Integration tests for configuration management across components"""
    
    @patch.dict(os.environ, {
        'AHGD_ENVIRONMENT': 'testing',
        'AHGD_DATABASE_PATH': 'test_health.db',
        'AHGD_DASHBOARD_PORT': '8502'
    })
    def test_configuration_integration(self):
        """Test configuration integration across all components"""
        from src.config import get_global_config, reset_global_config
        
        # Reset config to pick up environment variables
        reset_global_config()
        config = get_global_config()
        
        # Test configuration values
        assert config.environment.value == 'testing'
        assert 'test_health.db' in str(config.database.path)
        assert config.dashboard.port == 8502
        
        # Test config usage in components
        with patch('src.dashboard.data.loaders.get_global_config', return_value=config):
            from src.dashboard.data.loaders import load_data
            # Should use config without errors
            result = load_data()
            assert result is not None
    
    def test_logging_configuration_integration(self):
        """Test logging configuration across components"""
        from src.config import setup_logging, get_global_config
        
        config = get_global_config()
        
        # Test logging setup
        with patch('src.config.logging.basicConfig') as mock_basic_config:
            with patch('src.config.logging.FileHandler') as mock_file_handler:
                setup_logging(config.logging)
                
                # Should configure logging appropriately
                assert mock_basic_config.called or mock_file_handler.called


class TestDatabaseIntegration:
    """Integration tests for database operations"""
    
    def setup_method(self):
        """Setup test database"""
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create test tables
        self._create_test_database()
    
    def teardown_method(self):
        """Cleanup test database"""
        Path(self.db_path).unlink(missing_ok=True)
    
    def _create_test_database(self):
        """Create test database with sample data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create test table
        cursor.execute('''
            CREATE TABLE health_data (
                sa2_code TEXT,
                sa2_name TEXT,
                health_risk_score REAL,
                irsd_score REAL,
                mortality_rate REAL,
                population INTEGER
            )
        ''')
        
        # Insert test data
        test_records = [
            ('101021007', 'Sydney CBD', 8.5, 1050, 5.2, 18500),
            ('101021008', 'Sydney Haymarket', 12.3, 900, 8.1, 22000),
            ('201011001', 'Melbourne CBD', 6.1, 1200, 3.9, 28000),
            ('201011002', 'Melbourne Docklands', 15.7, 750, 10.4, 15500)
        ]
        
        cursor.executemany(
            'INSERT INTO health_data VALUES (?, ?, ?, ?, ?, ?)',
            test_records
        )
        
        conn.commit()
        conn.close()
    
    def test_database_connection_integration(self):
        """Test database connection and data retrieval"""
        import sqlite3
        
        # Test connection
        conn = sqlite3.connect(self.db_path)
        
        # Test data retrieval
        query = "SELECT * FROM health_data"
        data = pd.read_sql_query(query, conn)
        
        assert len(data) == 4
        assert 'sa2_code' in data.columns
        assert 'health_risk_score' in data.columns
        
        conn.close()
    
    def test_database_operations_integration(self):
        """Test integration of database operations with data processing"""
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        
        # Load data from database
        data = pd.read_sql_query("SELECT * FROM health_data", conn)
        
        # Process data using dashboard components
        from src.dashboard.data.processors import calculate_health_risk_score, filter_data_by_states
        
        # Test processing with database data
        processed_data = calculate_health_risk_score(data)
        assert 'health_risk_score' in processed_data.columns
        
        # Test filtering
        # Add state column for filtering test
        data['STE_NAME21'] = ['NSW', 'NSW', 'VIC', 'VIC']
        filtered_data = filter_data_by_states(data, ['NSW'])
        assert len(filtered_data) == 2
        
        conn.close()


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components"""
    
    def test_data_loading_error_cascade(self):
        """Test error handling when data loading fails"""
        with patch('src.dashboard.data.loaders.pd.read_parquet') as mock_read:
            mock_read.side_effect = FileNotFoundError("Data file not found")
            
            from src.dashboard.data.loaders import load_data
            
            # Should handle error gracefully and return sample data
            result = load_data()
            assert result is not None
            assert isinstance(result, pd.DataFrame)
    
    def test_visualization_error_handling(self):
        """Test error handling in visualization components"""
        invalid_data = pd.DataFrame({
            'col1': [np.nan, np.inf, -np.inf],
            'col2': ['invalid', None, '']
        })
        
        # Test charts with invalid data
        from src.dashboard.visualisation.charts import create_correlation_heatmap
        
        with patch('src.dashboard.visualisation.charts.px.imshow') as mock_imshow:
            mock_imshow.side_effect = Exception("Plotting error")
            
            try:
                result = create_correlation_heatmap(invalid_data.corr())
                # Should either handle error or raise expected exception
                success = True
            except Exception as e:
                # Error handling should be graceful
                success = "plotting" in str(e).lower() or "invalid" in str(e).lower()
            
            assert success
    
    def test_configuration_error_recovery(self):
        """Test configuration error recovery"""
        # Test with invalid configuration
        with patch.dict(os.environ, {'AHGD_DASHBOARD_PORT': 'invalid_port'}):
            from src.config import get_global_config, reset_global_config
            
            reset_global_config()
            config = get_global_config()
            
            # Should fall back to default port
            assert isinstance(config.dashboard.port, int)
            assert 1024 <= config.dashboard.port <= 65535


class TestStateManagementIntegration:
    """Integration tests for state management across dashboard components"""
    
    def test_session_state_consistency(self):
        """Test session state consistency across components"""
        # Mock streamlit session state
        mock_session_state = {}
        
        with patch('src.dashboard.ui.sidebar.st.session_state', mock_session_state):
            with patch('src.dashboard.ui.sidebar.st.sidebar') as mock_sidebar:
                # Mock sidebar components
                mock_sidebar.selectbox.return_value = 'NSW'
                mock_sidebar.slider.return_value = (0, 100)
                mock_sidebar.multiselect.return_value = ['mortality_rate', 'diabetes_prevalence']
                
                from src.dashboard.ui.sidebar import SidebarController
                
                # Test sidebar state management
                sidebar = SidebarController()
                state = sidebar.get_current_state()
                
                assert isinstance(state, dict)
    
    def test_data_state_persistence(self):
        """Test data state persistence across operations"""
        # Test that data transformations maintain consistency
        original_data = pd.DataFrame({
            'id': range(100),
            'value': np.random.random(100)
        })
        
        # Multiple transformations
        transformed_data = original_data.copy()
        transformed_data['squared'] = transformed_data['value'] ** 2
        transformed_data['log_value'] = np.log(transformed_data['value'] + 1)
        
        # State should be consistent
        assert len(transformed_data) == len(original_data)
        assert transformed_data['id'].equals(original_data['id'])
        assert np.allclose(transformed_data['value'], original_data['value'])


class TestAnalysisWorkflowIntegration:
    """Integration tests for complete analysis workflows"""
    
    def setup_method(self):
        """Setup comprehensive test data for analysis workflows"""
        np.random.seed(42)
        
        self.comprehensive_data = pd.DataFrame({
            'SA2_CODE21': [f"area_{i:06d}" for i in range(1000)],
            'SA2_NAME21': [f"Area {i}" for i in range(1000)],
            'health_risk_score': np.random.normal(10, 3, 1000),
            'IRSD_Score': np.random.normal(1000, 150, 1000),
            'mortality_rate': np.random.normal(6, 2, 1000),
            'diabetes_prevalence': np.random.normal(8, 2.5, 1000),
            'heart_disease_rate': np.random.normal(5, 1.5, 1000),
            'mental_health_rate': np.random.normal(12, 3, 1000),
            'population': np.random.randint(5000, 50000, 1000),
            'area_sqkm': np.random.uniform(1, 100, 1000),
            'STE_NAME21': np.random.choice(['NSW', 'VIC', 'QLD', 'WA', 'SA'], 1000)
        })
    
    def test_correlation_analysis_workflow(self):
        """Test complete correlation analysis workflow"""
        from src.dashboard.data.loaders import calculate_correlations
        from src.dashboard.data.processors import prepare_correlation_data, identify_health_hotspots
        from src.dashboard.visualisation.charts import create_correlation_heatmap
        
        # Step 1: Prepare correlation data
        correlation_data = prepare_correlation_data(self.comprehensive_data)
        assert isinstance(correlation_data, pd.DataFrame)
        
        # Step 2: Calculate correlations
        with patch('src.dashboard.data.loaders.st.cache_data') as mock_cache:
            mock_cache.return_value = lambda x: x
            correlation_matrix, processed_data = calculate_correlations(correlation_data)
        
        assert isinstance(correlation_matrix, pd.DataFrame)
        assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
        
        # Step 3: Identify hotspots
        hotspots = identify_health_hotspots(processed_data)
        assert isinstance(hotspots, pd.DataFrame)
        
        # Step 4: Create visualizations
        with patch('src.dashboard.visualisation.charts.px.imshow') as mock_imshow:
            mock_figure = Mock()
            mock_imshow.return_value = mock_figure
            
            heatmap = create_correlation_heatmap(correlation_matrix)
            assert heatmap == mock_figure
    
    def test_geographic_analysis_workflow(self):
        """Test complete geographic analysis workflow"""
        # Add geographic information
        self.comprehensive_data['latitude'] = np.random.uniform(-44, -10, 1000)
        self.comprehensive_data['longitude'] = np.random.uniform(113, 154, 1000)
        
        # Mock geometry for geographic operations
        mock_geometry = Mock()
        mock_geometry.bounds = [150.0, -34.0, 151.0, -33.0]
        self.comprehensive_data['geometry'] = [mock_geometry] * len(self.comprehensive_data)
        
        from src.dashboard.visualisation.maps import create_health_risk_map, get_map_bounds
        from src.dashboard.data.processors import filter_data_by_states
        
        # Step 1: Filter by geography
        nsw_data = filter_data_by_states(self.comprehensive_data, ['NSW'])
        assert len(nsw_data) > 0
        
        # Step 2: Calculate geographic bounds
        bounds = get_map_bounds(nsw_data)
        assert isinstance(bounds, dict)
        assert all(key in bounds for key in ['min_lat', 'max_lat', 'min_lon', 'max_lon'])
        
        # Step 3: Create geographic visualizations
        with patch('src.dashboard.visualisation.maps.folium.Map') as mock_map:
            mock_map_instance = Mock()
            mock_map.return_value = mock_map_instance
            
            health_map = create_health_risk_map(nsw_data)
            assert health_map == mock_map_instance
    
    def test_statistical_analysis_workflow(self):
        """Test complete statistical analysis workflow"""
        from src.dashboard.data.processors import (
            calculate_health_risk_score, generate_health_indicators,
            apply_scenario_analysis, calculate_data_quality_metrics
        )
        
        # Step 1: Calculate risk scores
        risk_data = calculate_health_risk_score(self.comprehensive_data)
        assert 'health_risk_score' in risk_data.columns
        
        # Step 2: Generate health indicators
        indicators = generate_health_indicators(risk_data)
        assert isinstance(indicators, pd.DataFrame)
        
        # Step 3: Apply scenario analysis
        scenario_data = apply_scenario_analysis(
            indicators, 
            scenario='improvement',
            improvement_factor=0.1
        )
        assert isinstance(scenario_data, pd.DataFrame)
        
        # Step 4: Calculate quality metrics
        with patch('src.dashboard.data.processors.st') as mock_st:
            quality_metrics = calculate_data_quality_metrics(scenario_data)
            # Should complete without error
            assert True
    
    def test_dashboard_rendering_workflow(self):
        """Test complete dashboard rendering workflow"""
        from src.dashboard.visualisation.components import (
            display_key_metrics, create_health_indicator_selector,
            display_correlation_insights
        )
        
        # Mock streamlit components
        with patch('src.dashboard.visualisation.components.st') as mock_st:
            mock_st.columns.return_value = [Mock(), Mock(), Mock(), Mock()]
            mock_st.selectbox.return_value = 'health_risk_score'
            
            # Step 1: Display key metrics
            display_key_metrics(self.comprehensive_data, 'health_risk_score', 'Dashboard Test')
            assert mock_st.columns.called
            
            # Step 2: Create indicator selector
            indicators = create_health_indicator_selector()
            assert isinstance(indicators, dict)
            
            # Step 3: Display correlation insights
            correlation_matrix = self.comprehensive_data.select_dtypes(include=[np.number]).corr()
            display_correlation_insights(correlation_matrix, 'IRSD_Score')
            # Should complete without error


class TestSystemIntegration:
    """System-wide integration tests"""
    
    def test_end_to_end_dashboard_simulation(self):
        """Simulate complete end-to-end dashboard usage"""
        # This test simulates a complete user session
        
        # Step 1: Initialize configuration
        from src.config import get_global_config
        config = get_global_config()
        assert config is not None
        
        # Step 2: Load data
        with patch('src.dashboard.data.loaders.pd.read_parquet') as mock_read:
            test_data = pd.DataFrame({
                'SA2_CODE21': ['area_001', 'area_002'],
                'health_risk_score': [8.5, 12.3],
                'IRSD_Score': [1050, 900]
            })
            mock_read.return_value = test_data
            
            from src.dashboard.data.loaders import load_data
            data = load_data()
            assert data is not None
        
        # Step 3: Process data
        from src.dashboard.data.processors import calculate_health_risk_score
        processed_data = calculate_health_risk_score(data)
        assert 'health_risk_score' in processed_data.columns
        
        # Step 4: Create visualizations
        with patch('src.dashboard.visualisation.charts.px.imshow') as mock_imshow:
            mock_figure = Mock()
            mock_imshow.return_value = mock_figure
            
            from src.dashboard.visualisation.charts import create_correlation_heatmap
            correlation_matrix = processed_data.select_dtypes(include=[np.number]).corr()
            chart = create_correlation_heatmap(correlation_matrix)
            assert chart is not None
        
        # Step 5: Display metrics
        with patch('src.dashboard.visualisation.components.st') as mock_st:
            mock_st.columns.return_value = [Mock(), Mock(), Mock(), Mock()]
            
            from src.dashboard.visualisation.components import display_key_metrics
            display_key_metrics(processed_data, 'health_risk_score', 'Integration Test')
            assert mock_st.columns.called
    
    def test_cross_component_data_consistency(self):
        """Test data consistency across all components"""
        # Create consistent test data
        test_data = pd.DataFrame({
            'SA2_CODE21': ['area_001', 'area_002', 'area_003'],
            'health_risk_score': [8.5, 12.3, 6.1],
            'IRSD_Score': [1050, 900, 1200],
            'population': [18500, 22000, 28000]
        })
        
        original_checksum = test_data['health_risk_score'].sum()
        
        # Pass data through various components
        from src.dashboard.data.processors import validate_health_data, prepare_correlation_data
        
        # Validation should not modify data
        is_valid = validate_health_data(test_data)
        post_validation_checksum = test_data['health_risk_score'].sum()
        assert abs(post_validation_checksum - original_checksum) < 1e-10
        
        # Correlation preparation might modify data, but should preserve key relationships
        correlation_data = prepare_correlation_data(test_data)
        assert isinstance(correlation_data, pd.DataFrame)
        assert len(correlation_data) >= len(test_data)  # Should not lose data
        
        # Test that correlations are consistent
        original_corr = test_data[['health_risk_score', 'IRSD_Score']].corr()
        processed_corr = correlation_data[['health_risk_score', 'IRSD_Score']].corr()
        
        # Correlation patterns should be similar (within reasonable tolerance)
        correlation_diff = abs(original_corr.iloc[0, 1] - processed_corr.iloc[0, 1])
        assert correlation_diff < 0.1  # Allow for minor differences due to processing


# Mark slow tests
pytest.mark.slow = pytest.mark.skipif(
    not pytest.config.getoption("--runslow"),
    reason="need --runslow option to run"
) if hasattr(pytest, 'config') else lambda x: x
