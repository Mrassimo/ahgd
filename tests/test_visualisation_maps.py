"""
Tests for maps visualization module

This module tests the geographic visualization functions to ensure they
create proper Folium maps with correct styling and interactive features.

Author: Portfolio Demonstration
Date: June 2025
"""

import pytest
import pandas as pd
import numpy as np
import folium
import geopandas as gpd
from shapely.geometry import Point, Polygon
from unittest.mock import patch, MagicMock

from src.dashboard.visualisation.maps import (
    create_health_risk_map,
    get_map_bounds,
    create_simple_point_map,
    _create_tooltip_text
)


@pytest.fixture
def sample_geodata():
    """Create sample geospatial data for testing"""
    # Create simple polygon geometries
    geometries = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)])
    ]
    
    data = {
        'SA2_CODE21': ['101', '102', '103'],
        'SA2_NAME21': ['Area A', 'Area B', 'Area C'],
        'STATE_NAME21': ['NSW', 'NSW', 'VIC'],
        'IRSD_Score': [950.0, 1050.0, 900.0],
        'IRSD_Decile_Australia': [3.0, 7.0, 2.0],
        'health_risk_score': [8.5, 5.2, 9.1],
        'mortality_rate': [12.3, 8.7, 15.2],
        'diabetes_prevalence': [8.2, 5.4, 10.1],
        'geometry': geometries
    }
    
    gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
    return gdf


@pytest.fixture
def sample_point_data():
    """Create sample point data for testing"""
    return pd.DataFrame({
        'name': ['Point A', 'Point B', 'Point C'],
        'latitude': [-33.8688, -37.8136, -27.4698],
        'longitude': [151.2093, 144.9631, 153.0251],
        'value': [10.5, 15.2, 8.9]
    })


class TestCreateHealthRiskMap:
    """Test suite for create_health_risk_map function"""
    
    def test_creates_folium_map(self, sample_geodata):
        """Test that function returns a Folium map object"""
        result = create_health_risk_map(sample_geodata)
        
        assert isinstance(result, folium.Map)
        assert hasattr(result, '_children')
    
    def test_handles_empty_data(self):
        """Test handling of empty datasets"""
        empty_data = gpd.GeoDataFrame()
        
        with patch('streamlit.warning') as mock_warning:
            result = create_health_risk_map(empty_data)
            
        assert result is None
        mock_warning.assert_called_once()
    
    def test_handles_missing_geometry(self, sample_geodata):
        """Test handling of data without geometry column"""
        data_no_geometry = sample_geodata.drop(columns=['geometry'])
        
        with patch('streamlit.warning') as mock_warning:
            result = create_health_risk_map(data_no_geometry)
            
        assert result is None
        mock_warning.assert_called_once()
    
    def test_handles_missing_indicator(self, sample_geodata):
        """Test handling of missing health indicator"""
        with patch('streamlit.warning') as mock_warning:
            result = create_health_risk_map(sample_geodata, indicator='missing_column')
            
        assert result is None
        mock_warning.assert_called_once()
    
    def test_custom_indicator(self, sample_geodata):
        """Test using custom health indicator"""
        result = create_health_risk_map(sample_geodata, indicator='mortality_rate')
        
        assert isinstance(result, folium.Map)
        # Check that choropleth layer was added
        assert len([child for child in result._children.values() 
                   if hasattr(child, '_template_name') and 
                   'choropleth' in child._template_name.lower()]) > 0
    
    def test_map_centering(self, sample_geodata):
        """Test that map is properly centered on data"""
        result = create_health_risk_map(sample_geodata)
        
        # Get map location
        map_center = result.location
        
        # Calculate expected center from bounds
        bounds = sample_geodata.total_bounds
        expected_lat = (bounds[1] + bounds[3]) / 2
        expected_lon = (bounds[0] + bounds[2]) / 2
        
        assert abs(map_center[0] - expected_lat) < 0.1
        assert abs(map_center[1] - expected_lon) < 0.1
    
    def test_markers_added(self, sample_geodata):
        """Test that markers are added for valid data points"""
        result = create_health_risk_map(sample_geodata)
        
        # Count markers in map children
        markers = [child for child in result._children.values() 
                  if hasattr(child, '_template_name') and 
                  'marker' in child._template_name.lower()]
        
        # Should have one marker per valid data row
        expected_markers = len(sample_geodata.dropna(subset=['health_risk_score']))
        assert len(markers) == expected_markers


class TestGetMapBounds:
    """Test suite for get_map_bounds function"""
    
    def test_calculates_bounds_correctly(self, sample_geodata):
        """Test correct calculation of geographic bounds"""
        bounds = get_map_bounds(sample_geodata)
        
        expected_bounds = sample_geodata.total_bounds
        
        assert bounds['min_lon'] == expected_bounds[0]
        assert bounds['min_lat'] == expected_bounds[1]
        assert bounds['max_lon'] == expected_bounds[2]
        assert bounds['max_lat'] == expected_bounds[3]
        
        # Test center calculations
        expected_center_lat = (expected_bounds[1] + expected_bounds[3]) / 2
        expected_center_lon = (expected_bounds[0] + expected_bounds[2]) / 2
        
        assert bounds['center_lat'] == expected_center_lat
        assert bounds['center_lon'] == expected_center_lon
    
    def test_raises_error_without_geometry(self):
        """Test error handling for data without geometry"""
        data_no_geometry = pd.DataFrame({'test': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Data must contain 'geometry' column"):
            get_map_bounds(data_no_geometry)


class TestCreateSimplePointMap:
    """Test suite for create_simple_point_map function"""
    
    def test_creates_point_map(self, sample_point_data):
        """Test creation of simple point map"""
        result = create_simple_point_map(
            sample_point_data, 
            lat_col='latitude', 
            lon_col='longitude',
            popup_col='name'
        )
        
        assert isinstance(result, folium.Map)
    
    def test_map_centered_on_data(self, sample_point_data):
        """Test that map is centered on point data"""
        result = create_simple_point_map(
            sample_point_data, 
            lat_col='latitude', 
            lon_col='longitude'
        )
        
        expected_lat = sample_point_data['latitude'].mean()
        expected_lon = sample_point_data['longitude'].mean()
        
        map_center = result.location
        assert abs(map_center[0] - expected_lat) < 0.1
        assert abs(map_center[1] - expected_lon) < 0.1
    
    def test_markers_added_for_each_point(self, sample_point_data):
        """Test that markers are added for each data point"""
        result = create_simple_point_map(
            sample_point_data, 
            lat_col='latitude', 
            lon_col='longitude',
            popup_col='name'
        )
        
        # Count markers
        markers = [child for child in result._children.values() 
                  if hasattr(child, '_template_name') and 
                  'marker' in child._template_name.lower()]
        
        assert len(markers) == len(sample_point_data)


class TestCreateTooltipText:
    """Test suite for _create_tooltip_text function"""
    
    def test_creates_formatted_tooltip(self, sample_geodata):
        """Test creation of formatted tooltip text"""
        row = sample_geodata.iloc[0]
        indicator = 'health_risk_score'
        
        tooltip = _create_tooltip_text(row, indicator)
        
        assert row['SA2_NAME21'] in tooltip
        assert row['STATE_NAME21'] in tooltip
        assert str(row['IRSD_Score']) in tooltip
        assert str(row['IRSD_Decile_Australia']) in tooltip
        assert str(row[indicator]) in tooltip
    
    def test_handles_different_indicators(self, sample_geodata):
        """Test tooltip creation with different indicators"""
        row = sample_geodata.iloc[0]
        indicator = 'mortality_rate'
        
        tooltip = _create_tooltip_text(row, indicator)
        
        assert 'Mortality Rate' in tooltip
        assert str(row[indicator]) in tooltip