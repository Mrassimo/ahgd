"""
Pytest configuration and shared fixtures for AHGD tests.

This module provides shared test fixtures, configuration, and utilities
for the Australian Health Geography Data Analytics test suite.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import pandas as pd
import duckdb
import sys
import os

# Add the scripts directory and src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Test configuration
TEST_DATA_DIR = Path(__file__).parent / "fixtures"
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Provide path to project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create and cleanup temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_db() -> Generator[Path, None, None]:
    """Create a temporary DuckDB database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        temp_db_path = Path(temp_file.name)
    
    try:
        yield temp_db_path
    finally:
        if temp_db_path.exists():
            temp_db_path.unlink()


@pytest.fixture
def sample_postcode_data() -> pd.DataFrame:
    """Create sample postcode-level data for testing."""
    return pd.DataFrame({
        'postcode': ['2000', '2001', '3000', '3001', '4000', '5000'],
        'population': [15000, 12000, 25000, 18000, 20000, 16000],
        'median_income': [65000, 58000, 72000, 55000, 62000, 59000],
        'hospitals': [2, 1, 3, 2, 2, 1],
        'area_sqkm': [5.2, 8.1, 12.3, 6.7, 15.4, 9.8]
    })


@pytest.fixture
def sample_sa2_data() -> pd.DataFrame:
    """Create sample SA2-level data for testing."""
    return pd.DataFrame({
        'sa2_main16': ['101021007', '101021008', '201011001', '201011002'],
        'sa2_name16': ['Sydney - CBD', 'Sydney - Haymarket', 'Melbourne - CBD', 'Melbourne - Docklands'],
        'population': [18500, 22000, 28000, 15500],
        'disadvantage_score': [1050, 980, 1120, 1200],
        'health_outcome': ['Good', 'Fair', 'Excellent', 'Good']
    })


@pytest.fixture
def sample_seifa_data() -> pd.DataFrame:
    """Create sample SEIFA disadvantage data for testing."""
    return pd.DataFrame({
        'sa2_code_2021': ['101021007', '101021008', '201011001', '201011002'],
        'sa2_name_2021': ['Sydney - CBD', 'Sydney - Haymarket', 'Melbourne - CBD', 'Melbourne - Docklands'],
        'irsad_score': [1050.5, 980.2, 1120.8, 1200.1],
        'irsad_decile': [7, 5, 8, 9],
        'irsd_score': [1040.2, 975.8, 1115.3, 1195.7],
        'irsd_decile': [7, 5, 8, 9]
    })


@pytest.fixture
def sample_health_data() -> pd.DataFrame:
    """Create sample health outcome data for testing."""
    return pd.DataFrame({
        'sa2_code': ['101021007', '101021008', '201011001', '201011002'],
        'year': [2021, 2021, 2021, 2021],
        'mortality_rate': [5.2, 6.8, 4.1, 3.9],
        'chronic_disease_rate': [15.2, 18.4, 12.8, 11.5],
        'mental_health_rate': [8.9, 12.1, 7.6, 6.8],
        'population': [18500, 22000, 28000, 15500]
    })


@pytest.fixture
def sample_correspondence_data() -> pd.DataFrame:
    """Create sample postcode-SA2 correspondence data for testing."""
    return pd.DataFrame({
        'POA_CODE_2021': ['2000', '2000', '2001', '3000', '3001', '4000', '5000'],
        'SA2_CODE_2021': ['101021007', '101021008', '101021009', '201011001', '201011002', '301011003', '401011004'],
        'SA2_NAME_2021': ['Sydney - CBD', 'Sydney - Haymarket', 'Sydney - The Rocks', 
                         'Melbourne - CBD', 'Melbourne - Docklands', 'Brisbane - CBD', 'Adelaide - CBD'],
        'RATIO': [0.6, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0]
    })


@pytest.fixture
def mock_database_connection(temp_db):
    """Create a mock database connection with sample data."""
    conn = duckdb.connect(str(temp_db))
    
    # Create sample tables
    sample_data = {
        'correspondence': pd.DataFrame({
            'POA_CODE_2021': ['2000', '2000', '2001', '3000'],
            'SA2_CODE_2021': ['101021007', '101021008', '101021009', '201011001'],
            'SA2_NAME_2021': ['Sydney - CBD', 'Sydney - Haymarket', 'Sydney - The Rocks', 'Melbourne - CBD'],
            'RATIO': [0.6, 0.4, 1.0, 1.0]
        })
    }
    
    for table_name, df in sample_data.items():
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    
    yield conn
    conn.close()


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for testing."""
    return {
        'data_dir': '/test/data',
        'database': {
            'path': 'test_health_analytics.db',
            'timeout': 30
        },
        'processing': {
            'chunk_size': 1000,
            'max_workers': 2
        },
        'analysis': {
            'correlation_threshold': 0.5,
            'significance_level': 0.05
        },
        'visualisation': {
            'map_style': 'OpenStreetMap',
            'colour_scheme': 'viridis'
        }
    }


# Test markers for organizing tests
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that test individual functions"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that test component interaction"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that may take longer to run"
    )
    config.addinivalue_line(
        "markers", "database: Tests that require database access"
    )
    config.addinivalue_line(
        "markers", "network: Tests that require network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add database marker to tests that use database fixtures
        if any(fixture in item.fixturenames for fixture in ['temp_db', 'mock_database_connection']):
            item.add_marker(pytest.mark.database)
