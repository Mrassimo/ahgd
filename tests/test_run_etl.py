#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for the main ETL execution script run_etl.py."""

import pytest
from unittest.mock import patch, MagicMock
import argparse
from pathlib import Path
import sys
import polars as pl

# Adjust path to import from root directory
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Import directly from the root script
try:
    import run_etl
except ImportError as e:
    pytest.fail(f"Failed to import run_etl.py. Ensure it is in the project root and path is correct. Error: {e}")

# --- Mock Fixtures ----

@pytest.fixture
def mock_paths(tmp_path, monkeypatch):
    """Creates mock paths and sets BASE_DIR environment variable."""
    paths = {
        'BASE_DIR': tmp_path,
        'OUTPUT_DIR': tmp_path / 'output',
        'DATA_DIR': tmp_path / 'data',
        'RAW_DATA_DIR': tmp_path / 'data' / 'raw',
        'TEMP_DIR': tmp_path / 'data' / 'raw' / 'temp',
        'LOG_DIR': tmp_path / 'logs'
    }
    # Create directories
    for path in paths.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    # Set BASE_DIR environment variable for config loading
    monkeypatch.setenv('BASE_DIR', str(tmp_path))
    # Also patch config.PATHS if run_etl imports it early
    try:
        with patch('run_etl.config.PATHS', paths):
             yield paths
    except AttributeError:
         yield paths

@pytest.fixture
def mock_args():
    """Fixture for mock command line arguments."""
    args = argparse.Namespace(
        steps=["all"],
        log_level="INFO",
        force_download=False,
        config_file=None
    )
    return args

@pytest.fixture
def mock_dependencies(mock_paths): # Ensure paths are set up
    """Mock dependencies called within run_etl.main.
    
    Mocks all external dependencies to isolate tests:
    - Directory setup, logging
    - ETL components (geo, time, census, dimensions, validation)
    - CLI args parsing
    
    Note: All census-related functionality should go through mock_process_census
    as we don't directly import from etl_logic.census.
    """
    with patch('run_etl.config.initialise_directories') as mock_init_dirs, \
         patch('run_etl.utils.setup_logging') as mock_setup_logging, \
         patch('run_etl.geography.process_geographic_data') as mock_process_geo, \
         patch('run_etl.time_dimension.create_time_dimension') as mock_create_time, \
         patch('run_etl.census.process_all_census_tables') as mock_process_census, \
         patch('run_etl.dimensions.generate_all_dimensions') as mock_generate_dims, \
         patch('run_etl.validation.run_all_data_quality_checks') as mock_validate, \
         patch('run_etl.argparse.ArgumentParser.parse_args') as mock_parse_args:

        yield {
            "init_dirs": mock_init_dirs,
            "setup_logging": mock_setup_logging,
            "process_geo": mock_process_geo,
            "create_time": mock_create_time,
            "process_census": mock_process_census,
            "generate_dims": mock_generate_dims,
            "validate": mock_validate,
            "parse_args": mock_parse_args,
        }

# --- Tests using Mocks ---

def test_run_etl_main_all_steps(mock_dependencies, mock_args):
    """Test the main execution flow with 'all' steps."""
    mock_dependencies["parse_args"].return_value = mock_args
    mock_dependencies["process_geo"].return_value = True
    mock_dependencies["create_time"].return_value = (True, 202101)
    mock_dependencies["process_census"].return_value = True
    mock_dependencies["generate_dims"].return_value = True
    mock_dependencies["validate"].return_value = True

    try:
        run_etl.main()
    except SystemExit as e:
        assert e.code == 0

    mock_dependencies["parse_args"].assert_called_once()
    mock_dependencies["init_dirs"].assert_called_once()
    mock_dependencies["setup_logging"].assert_called_once()
    mock_dependencies["process_geo"].assert_called_once()
    mock_dependencies["create_time"].assert_called_once()
    mock_dependencies["process_census"].assert_called_once()
    mock_dependencies["generate_dims"].assert_called_once()
    mock_dependencies["validate"].assert_called_once()

def test_run_etl_main_specific_steps(mock_dependencies, mock_args):
    """Test running specific steps, e.g., 'extract' and 'transform_geo'."""
    mock_args.steps = ["extract", "transform_geo"]
    mock_dependencies["parse_args"].return_value = mock_args
    mock_dependencies["process_geo"].return_value = True

    try:
        run_etl.main()
    except SystemExit as e:
        assert e.code == 0

    mock_dependencies["parse_args"].assert_called_once()
    mock_dependencies["init_dirs"].assert_called_once()
    mock_dependencies["setup_logging"].assert_called_once()
    mock_dependencies["process_geo"].assert_called_once()
    mock_dependencies["create_time"].assert_not_called()
    mock_dependencies["process_census"].assert_not_called()
    mock_dependencies["generate_dims"].assert_not_called()
    mock_dependencies["validate"].assert_not_called()

def test_run_etl_main_validation_failure(mock_dependencies, mock_args):
    """Test that overall failure is reported if validation fails."""
    mock_dependencies["parse_args"].return_value = mock_args
    mock_dependencies["process_geo"].return_value = True
    mock_dependencies["create_time"].return_value = (True, 202101)
    mock_dependencies["process_census"].return_value = True
    mock_dependencies["generate_dims"].return_value = True
    mock_dependencies["validate"].return_value = False # Simulate failure

    with pytest.raises(SystemExit) as exc_info:
        run_etl.main()

    mock_dependencies["validate"].assert_called_once()
    assert exc_info.value.code != 0

# Add more tests as needed 