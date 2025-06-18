"""
Integration tests for dashboard components.

This module tests Streamlit dashboard functionality, component integration,
and end-to-end dashboard workflows.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import json

# Add the project paths to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from tests.fixtures.sample_data import (
    get_sample_health_data, get_sample_seifa_data,
    get_sample_correspondence_data, get_sample_demographic_data,
    get_sample_geographic_boundaries
)


@pytest.mark.integration
class TestDashboardDataLoading:
    """Test dashboard data loading and processing."""
    
    @patch('streamlit.cache_data')
    def test_health_data_loading_with_cache(self, mock_cache, temp_dir):
        """Test health data loading with Streamlit caching."""
        # Mock the cache decorator to just return the function
        mock_cache.side_effect = lambda func: func
        
        # Create sample data file
        sample_data = get_sample_health_data()
        data_file = temp_dir / "health_data.csv"
        sample_data.to_csv(data_file, index=False)
        
        # Mock data loading function
        def load_health_data(file_path):
            return pd.read_csv(file_path)
        
        # Test loading
        loaded_data = load_health_data(data_file)
        
        assert len(loaded_data) == len(sample_data)
        assert list(loaded_data.columns) == list(sample_data.columns)
        assert loaded_data['mortality_rate'].dtype == float
    
    def test_geographic_data_loading(self, temp_dir):
        """Test loading geographic boundary data."""
        # Create sample geographic data
        geo_data = get_sample_geographic_boundaries()
        geo_file = temp_dir / "boundaries.json"
        
        with open(geo_file, 'w') as f:
            json.dump(geo_data, f)
        
        # Mock loading function
        def load_geographic_data(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
        
        loaded_geo = load_geographic_data(geo_file)
        
        assert loaded_geo['type'] == 'FeatureCollection'
        assert len(loaded_geo['features']) > 0
        assert 'geometry' in loaded_geo['features'][0]
        assert 'properties' in loaded_geo['features'][0]
    
    def test_data_validation_pipeline(self):
        """Test data validation in dashboard pipeline."""
        # Test with valid data
        valid_data = get_sample_health_data()
        
        def validate_health_data(df):
            issues = []
            
            # Check required columns
            required_cols = ['sa2_code', 'year', 'mortality_rate', 'population']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                issues.append(f"Missing columns: {missing_cols}")
            
            # Check for null values in critical columns
            for col in required_cols:
                if col in df.columns and df[col].isnull().any():
                    issues.append(f"Null values found in {col}")
            
            # Check data ranges
            if 'mortality_rate' in df.columns:
                invalid_rates = df[(df['mortality_rate'] < 0) | (df['mortality_rate'] > 100)]
                if len(invalid_rates) > 0:
                    issues.append(f"Invalid mortality rates: {len(invalid_rates)} records")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'record_count': len(df)
            }
        
        validation_result = validate_health_data(valid_data)
        assert validation_result['valid'] is True
        assert len(validation_result['issues']) == 0
        assert validation_result['record_count'] > 0
        
        # Test with invalid data
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'mortality_rate'] = -5.0  # Invalid rate
        invalid_data.loc[1, 'sa2_code'] = None  # Null value
        
        invalid_result = validate_health_data(invalid_data)
        assert invalid_result['valid'] is False
        assert len(invalid_result['issues']) > 0


@pytest.mark.integration
class TestDashboardVisualisations:
    """Test dashboard visualisation components."""
    
    def test_choropleth_map_data_preparation(self):
        """Test data preparation for choropleth maps."""
        health_data = get_sample_health_data()
        seifa_data = get_sample_seifa_data()
        
        def prepare_map_data(health_df, seifa_df, metric='mortality_rate'):
            # Merge health and seifa data
            merged = health_df.merge(
                seifa_df[['sa2_code_2021', 'sa2_name_2021', 'irsad_decile']],
                left_on='sa2_code',
                right_on='sa2_code_2021',
                how='inner'
            )
            
            # Prepare for mapping
            map_data = merged[['sa2_code', 'sa2_name_2021', metric, 'irsad_decile']].copy()
            map_data = map_data.rename(columns={
                'sa2_name_2021': 'area_name',
                metric: 'map_value'
            })
            
            return map_data
        
        map_ready_data = prepare_map_data(health_data, seifa_data)
        
        assert len(map_ready_data) > 0
        assert 'sa2_code' in map_ready_data.columns
        assert 'area_name' in map_ready_data.columns
        assert 'map_value' in map_ready_data.columns
        assert map_ready_data['map_value'].dtype in [float, int]
    
    @patch('altair.Chart')
    def test_scatter_plot_creation(self, mock_chart):
        """Test creation of scatter plot visualisations."""
        # Mock Altair Chart
        mock_chart_instance = Mock()
        mock_chart.return_value = mock_chart_instance
        mock_chart_instance.mark_circle.return_value = mock_chart_instance
        mock_chart_instance.encode.return_value = mock_chart_instance
        mock_chart_instance.properties.return_value = mock_chart_instance
        
        health_data = get_sample_health_data()
        seifa_data = get_sample_seifa_data()
        
        # Merge data for correlation analysis
        correlation_data = health_data.merge(
            seifa_data[['sa2_code_2021', 'irsad_score']],
            left_on='sa2_code',
            right_on='sa2_code_2021'
        )
        
        def create_scatter_plot(data, x_col, y_col, title):
            import altair as alt
            
            chart = alt.Chart(data).mark_circle(size=60).encode(
                x=alt.X(x_col, title=x_col.replace('_', ' ').title()),
                y=alt.Y(y_col, title=y_col.replace('_', ' ').title()),
                tooltip=['sa2_code', x_col, y_col]
            ).properties(
                width=600,
                height=400,
                title=title
            )
            
            return chart
        
        # Test chart creation
        chart = create_scatter_plot(
            correlation_data,
            'irsad_score',
            'mortality_rate',
            'Mortality Rate vs SEIFA Score'
        )
        
        # Verify chart was created (mocked)
        mock_chart.assert_called_once()
    
    def test_summary_statistics_calculation(self):
        """Test calculation of summary statistics for dashboard."""
        health_data = get_sample_health_data()
        
        def calculate_summary_stats(df, metric_col):
            stats = {
                'count': len(df),
                'mean': df[metric_col].mean(),
                'median': df[metric_col].median(),
                'std': df[metric_col].std(),
                'min': df[metric_col].min(),
                'max': df[metric_col].max(),
                'q25': df[metric_col].quantile(0.25),
                'q75': df[metric_col].quantile(0.75)
            }
            
            # Calculate additional insights
            stats['range'] = stats['max'] - stats['min']
            stats['cv'] = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
            
            return stats
        
        mortality_stats = calculate_summary_stats(health_data, 'mortality_rate')
        
        assert mortality_stats['count'] == len(health_data)
        assert mortality_stats['mean'] > 0
        assert mortality_stats['min'] <= mortality_stats['median'] <= mortality_stats['max']
        assert mortality_stats['q25'] <= mortality_stats['median'] <= mortality_stats['q75']
        assert mortality_stats['std'] >= 0
        assert mortality_stats['range'] >= 0


@pytest.mark.integration
class TestDashboardInteractivity:
    """Test dashboard interactive components."""
    
    def test_filter_functionality(self):
        """Test data filtering functionality."""
        health_data = get_sample_health_data()
        
        def apply_filters(df, filters):
            filtered_df = df.copy()
            
            # Apply year filter
            if 'year' in filters and filters['year']:
                filtered_df = filtered_df[filtered_df['year'].isin(filters['year'])]
            
            # Apply SA2 filter
            if 'sa2_codes' in filters and filters['sa2_codes']:
                filtered_df = filtered_df[filtered_df['sa2_code'].isin(filters['sa2_codes'])]
            
            # Apply metric range filter
            if 'mortality_range' in filters and filters['mortality_range']:
                min_val, max_val = filters['mortality_range']
                filtered_df = filtered_df[
                    (filtered_df['mortality_rate'] >= min_val) &
                    (filtered_df['mortality_rate'] <= max_val)
                ]
            
            return filtered_df
        
        # Test with no filters
        no_filter_result = apply_filters(health_data, {})
        assert len(no_filter_result) == len(health_data)
        
        # Test with year filter
        year_filter = {'year': [2021]}
        year_filtered = apply_filters(health_data, year_filter)
        assert all(year_filtered['year'] == 2021)
        
        # Test with SA2 filter
        sa2_filter = {'sa2_codes': ['101021007', '201011001']}
        sa2_filtered = apply_filters(health_data, sa2_filter)
        assert all(sa2_filtered['sa2_code'].isin(['101021007', '201011001']))
        
        # Test with range filter
        range_filter = {'mortality_range': (0, 5)}
        range_filtered = apply_filters(health_data, range_filter)
        assert all((range_filtered['mortality_rate'] >= 0) & (range_filtered['mortality_rate'] <= 5))
    
    def test_dynamic_metric_selection(self):
        """Test dynamic metric selection functionality."""
        health_data = get_sample_health_data()
        
        available_metrics = {
            'mortality_rate': 'Mortality Rate (per 1,000)',
            'chronic_disease_rate': 'Chronic Disease Rate (%)',
            'mental_health_rate': 'Mental Health Rate (%)',
            'diabetes_rate': 'Diabetes Rate (%)',
            'heart_disease_rate': 'Heart Disease Rate (%)'
        }
        
        def get_metric_data(df, selected_metric):
            if selected_metric not in df.columns:
                raise ValueError(f"Metric {selected_metric} not found in data")
            
            metric_data = df[['sa2_code', selected_metric]].copy()
            metric_data = metric_data.rename(columns={selected_metric: 'value'})
            
            return metric_data
        
        # Test each available metric
        for metric_key in available_metrics.keys():
            metric_data = get_metric_data(health_data, metric_key)
            assert len(metric_data) == len(health_data)
            assert 'sa2_code' in metric_data.columns
            assert 'value' in metric_data.columns
            assert metric_data['value'].dtype in [float, int]
        
        # Test invalid metric
        with pytest.raises(ValueError):
            get_metric_data(health_data, 'invalid_metric')
    
    def test_correlation_analysis_component(self):
        """Test correlation analysis component."""
        health_data = get_sample_health_data()
        seifa_data = get_sample_seifa_data()
        
        def calculate_correlations(health_df, seifa_df):
            # Merge datasets
            merged = health_df.merge(
                seifa_df[['sa2_code_2021', 'irsad_score', 'irsad_decile']],
                left_on='sa2_code',
                right_on='sa2_code_2021'
            )
            
            # Calculate correlations
            health_metrics = ['mortality_rate', 'chronic_disease_rate', 'mental_health_rate']
            seifa_metrics = ['irsad_score', 'irsad_decile']
            
            correlations = {}
            for health_metric in health_metrics:
                correlations[health_metric] = {}
                for seifa_metric in seifa_metrics:
                    corr = merged[health_metric].corr(merged[seifa_metric])
                    correlations[health_metric][seifa_metric] = corr
            
            return correlations
        
        correlations = calculate_correlations(health_data, seifa_data)
        
        # Verify correlation structure
        assert 'mortality_rate' in correlations
        assert 'irsad_score' in correlations['mortality_rate']
        
        # Check correlation values are valid
        for health_metric in correlations:
            for seifa_metric in correlations[health_metric]:
                corr_value = correlations[health_metric][seifa_metric]
                if not pd.isna(corr_value):
                    assert -1 <= corr_value <= 1


@pytest.mark.integration
class TestDashboardLayout:
    """Test dashboard layout and structure."""
    
    @patch('streamlit.sidebar')
    @patch('streamlit.columns')
    def test_sidebar_layout(self, mock_columns, mock_sidebar):
        """Test sidebar layout configuration."""
        # Mock Streamlit components
        mock_sidebar_obj = Mock()
        mock_sidebar.return_value = mock_sidebar_obj
        
        def create_sidebar_controls():
            import streamlit as st
            
            with st.sidebar:
                # Year selection
                years = st.multiselect(
                    "Select Years",
                    options=[2018, 2019, 2020, 2021],
                    default=[2021]
                )
                
                # Metric selection
                metric = st.selectbox(
                    "Select Health Metric",
                    options=['mortality_rate', 'chronic_disease_rate'],
                    index=0
                )
                
                # Filter options
                show_filters = st.checkbox("Show Advanced Filters", False)
                
            return {
                'years': years,
                'metric': metric,
                'show_filters': show_filters
            }
        
        # Test sidebar creation (mocked)
        controls = create_sidebar_controls()
        
        # Verify sidebar was accessed
        mock_sidebar.assert_called()
    
    @patch('streamlit.container')
    def test_main_content_layout(self, mock_container):
        """Test main content area layout."""
        mock_container_obj = Mock()
        mock_container.return_value = mock_container_obj
        
        def create_main_layout():
            import streamlit as st
            
            # Header section
            header_container = st.container()
            with header_container:
                st.title("Australian Health Analytics Dashboard")
                st.markdown("**Data-driven insights into health outcomes across Australia**")
            
            # Metrics overview
            metrics_container = st.container()
            with metrics_container:
                col1, col2, col3, col4 = st.columns(4)
                # Metrics would be displayed here
            
            # Main visualisation
            viz_container = st.container()
            with viz_container:
                st.subheader("Geographic Distribution")
                # Map or chart would be displayed here
            
            # Analysis section
            analysis_container = st.container()
            with analysis_container:
                st.subheader("Correlation Analysis")
                # Analysis results would be displayed here
            
            return {
                'header': header_container,
                'metrics': metrics_container,
                'visualisation': viz_container,
                'analysis': analysis_container
            }
        
        # Test layout creation (mocked)
        layout = create_main_layout()
        
        # Verify containers were created
        assert mock_container.call_count >= 4


@pytest.mark.integration
class TestDashboardDataFlow:
    """Test end-to-end data flow in dashboard."""
    
    def test_complete_data_pipeline(self, temp_dir):
        """Test complete data pipeline from loading to visualisation."""
        # Step 1: Create sample data files
        health_data = get_sample_health_data()
        seifa_data = get_sample_seifa_data()
        correspondence_data = get_sample_correspondence_data()
        
        health_file = temp_dir / "health.csv"
        seifa_file = temp_dir / "seifa.csv"
        correspondence_file = temp_dir / "correspondence.csv"
        
        health_data.to_csv(health_file, index=False)
        seifa_data.to_csv(seifa_file, index=False)
        correspondence_data.to_csv(correspondence_file, index=False)
        
        # Step 2: Simulate data loading pipeline
        def load_and_process_data(data_dir):
            # Load data
            health_df = pd.read_csv(data_dir / "health.csv")
            seifa_df = pd.read_csv(data_dir / "seifa.csv")
            correspondence_df = pd.read_csv(data_dir / "correspondence.csv")
            
            # Process and merge
            processed_data = health_df.merge(
                seifa_df[['sa2_code_2021', 'sa2_name_2021', 'irsad_score', 'irsad_decile']],
                left_on='sa2_code',
                right_on='sa2_code_2021',
                how='left'
            )
            
            # Add postcode mapping information
            postcode_summary = correspondence_df.groupby('SA2_CODE_2021').agg({
                'POA_CODE_2021': lambda x: ', '.join(x.unique()[:3]),  # First 3 postcodes
                'RATIO': 'sum'
            }).reset_index()
            
            final_data = processed_data.merge(
                postcode_summary,
                left_on='sa2_code',
                right_on='SA2_CODE_2021',
                how='left'
            )
            
            return final_data
        
        # Step 3: Execute pipeline
        result_data = load_and_process_data(temp_dir)
        
        # Step 4: Verify pipeline results
        assert len(result_data) > 0
        assert 'mortality_rate' in result_data.columns  # From health data
        assert 'irsad_score' in result_data.columns     # From seifa data
        assert 'POA_CODE_2021' in result_data.columns   # From correspondence data
        
        # Check data quality
        assert result_data['mortality_rate'].notna().all()
        assert result_data['irsad_score'].notna().sum() > 0  # Some should match
    
    def test_error_handling_in_pipeline(self, temp_dir):
        """Test error handling in data pipeline."""
        def robust_data_loader(data_dir):
            errors = []
            loaded_data = {}
            
            # Try to load each dataset
            datasets = {
                'health': 'health.csv',
                'seifa': 'seifa.csv',
                'correspondence': 'correspondence.csv'
            }
            
            for dataset_name, filename in datasets.items():
                try:
                    file_path = data_dir / filename
                    if file_path.exists():
                        loaded_data[dataset_name] = pd.read_csv(file_path)
                    else:
                        errors.append(f"File not found: {filename}")
                except Exception as e:
                    errors.append(f"Error loading {filename}: {str(e)}")
            
            return loaded_data, errors
        
        # Test with missing files
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        data, errors = robust_data_loader(empty_dir)
        
        assert len(errors) > 0  # Should have errors for missing files
        assert len(data) == 0   # Should have no loaded data
        
        # Test with some files present
        health_data = get_sample_health_data()
        health_file = empty_dir / "health.csv"
        health_data.to_csv(health_file, index=False)
        
        partial_data, partial_errors = robust_data_loader(empty_dir)
        
        assert 'health' in partial_data
        assert len(partial_errors) == 2  # Still missing 2 files


@pytest.mark.integration
@pytest.mark.slow
class TestDashboardPerformance:
    """Test dashboard performance characteristics."""
    
    def test_large_dataset_handling(self, temp_dir):
        """Test dashboard performance with larger datasets."""
        import time
        
        # Generate larger dataset
        large_health_data = pd.concat([get_sample_health_data() for _ in range(100)])
        large_health_data['sa2_code'] = [f"SA2_{i//10}" for i in range(len(large_health_data))]
        
        # Save to file
        large_file = temp_dir / "large_health.csv"
        large_health_data.to_csv(large_file, index=False)
        
        # Test loading performance
        start_time = time.time()
        loaded_data = pd.read_csv(large_file)
        load_time = time.time() - start_time
        
        # Test processing performance
        start_time = time.time()
        summary_stats = loaded_data.groupby('sa2_code').agg({
            'mortality_rate': ['mean', 'std', 'count'],
            'population': 'sum'
        })
        process_time = time.time() - start_time
        
        # Performance expectations
        assert load_time < 2.0      # Should load quickly
        assert process_time < 1.0   # Should process quickly
        assert len(summary_stats) > 0
    
    def test_memory_usage_monitoring(self):
        """Test memory usage patterns in dashboard operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform memory-intensive operations
        large_datasets = []
        for i in range(10):
            data = get_sample_health_data()
            # Expand dataset
            expanded = pd.concat([data for _ in range(100)])
            large_datasets.append(expanded)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Clean up
        del large_datasets
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = peak_memory - initial_memory
        memory_recovered = peak_memory - final_memory
        
        # Memory should increase during processing but not excessively
        assert memory_increase < 500  # Should not use more than 500MB
        assert memory_recovered > 0   # Some memory should be recovered


@pytest.mark.integration
class TestDashboardConfiguration:
    """Test dashboard configuration and settings."""
    
    def test_configuration_loading(self, sample_config, temp_dir):
        """Test loading dashboard configuration."""
        # Create config file
        config_file = temp_dir / "dashboard_config.json"
        with open(config_file, 'w') as f:
            json.dump(sample_config, f)
        
        def load_dashboard_config(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Apply defaults for missing values
            defaults = {
                'dashboard': {
                    'host': 'localhost',
                    'port': 8501,
                    'debug': False,
                    'page_title': 'Health Analytics Dashboard'
                }
            }
            
            # Merge with defaults
            for section, values in defaults.items():
                if section not in config:
                    config[section] = {}
                for key, default_value in values.items():
                    if key not in config[section]:
                        config[section][key] = default_value
            
            return config
        
        loaded_config = load_dashboard_config(config_file)
        
        assert 'dashboard' in loaded_config
        assert loaded_config['dashboard']['host'] is not None
        assert loaded_config['dashboard']['port'] > 0
    
    def test_theme_and_styling_configuration(self):
        """Test theme and styling configuration."""
        def get_dashboard_theme():
            return {
                'primary_color': '#1f77b4',
                'background_color': '#ffffff',
                'secondary_background_color': '#f0f2f6',
                'text_color': '#262730',
                'font': 'sans serif',
                'base': 'light'
            }
        
        theme = get_dashboard_theme()
        
        assert 'primary_color' in theme
        assert theme['primary_color'].startswith('#')  # Should be hex color
        assert theme['base'] in ['light', 'dark']
        assert len(theme['primary_color']) == 7  # Hex color format
