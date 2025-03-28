"""Configuration settings for the AHGD ETL pipeline.

This module contains all configuration constants, paths, and data source URLs
used throughout the ETL process.
"""

from pathlib import Path
from typing import Dict, List

# Note: BASE_DIR will be set dynamically in the Colab notebook
# These paths are relative to BASE_DIR
RELATIVE_PATHS = {
    'DATA_DIR': 'data',
    'RAW_DATA_DIR': 'data/raw',
    'OUTPUT_DIR': 'output',
    'TEMP_DIR': 'data/raw/temp',
    'LOG_DIR': 'logs',
    'GEOGRAPHIC_DIR': 'data/raw/geographic',
    'CENSUS_DIR': 'data/raw/census',
    'TEMP_ZIP_DIR': 'data/raw/temp/zips',
    'TEMP_EXTRACT_DIR': 'data/raw/temp/extract'
}

# Data source URLs
DATA_URLS = {
    # ASGS Main Structures (Using GDA2020 Shapefiles)
    'SA1_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA1_2021_AUST_GDA2020_SHP.zip',
    'SA2_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_GDA2020_SHP.zip',
    'SA3_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA3_2021_AUST_GDA2020_SHP.zip',
    'SA4_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA4_2021_AUST_GDA2020_SHP.zip',
    'STE_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/STE_2021_AUST_GDA2020_SHP.zip',
    'POA_2021_AUST_GDA2020': 'https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/POA_2021_AUST_GDA2020_SHP.zip',
    'CENSUS_GCP_AUS_2021': 'https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_ALL_for_AUS.zip'
}

# Geographic levels to process
GEO_LEVELS_SHP_PROCESS = {
    'SA1': 'SA1_2021_AUST_GDA2020',
    'SA2': 'SA2_2021_AUST_GDA2020',
    'SA3': 'SA3_2021_AUST_GDA2020',
    'SA4': 'SA4_2021_AUST_GDA2020',
    'STATE': 'STE_2021_AUST_GDA2020'
}

# Geographic levels for Census data processing
GEO_LEVELS_CENSUS_PROCESS = ['SA1', 'SA2']

# Census table patterns
CENSUS_TABLE_PATTERNS = {
    "G01": r"2021\s*Census_G01[_\s].*?(" + "|".join(GEO_LEVELS_CENSUS_PROCESS) + r")\.csv$"
}

def get_required_geo_zips() -> Dict[str, str]:
    """Generate required ZIP URLs based on GEO_LEVELS_SHP_PROCESS.
    
    Returns:
        Dict[str, str]: Dictionary mapping zip filenames to their download URLs.
    """
    required_zips = {}
    for level, prefix in GEO_LEVELS_SHP_PROCESS.items():
        if prefix in DATA_URLS:
            zip_filename = f"{prefix}_SHP.zip"
            required_zips[zip_filename] = DATA_URLS[prefix]
    return required_zips

def get_required_census_zips() -> Dict[str, str]:
    """Get required Census data pack ZIP URLs.
    
    Returns:
        Dict[str, str]: Dictionary mapping zip filenames to their download URLs.
    """
    required_zips = {}
    census_key = 'CENSUS_GCP_AUS_2021'
    if census_key in DATA_URLS:
        url = DATA_URLS[census_key]
        # Extract filename from URL or use last part of path
        filename = url.split('/')[-1]
        required_zips[filename] = url
    return required_zips

def get_paths(base_dir: Path) -> Dict[str, Path]:
    """Generate absolute paths based on BASE_DIR.
    
    Args:
        base_dir (Path): The base directory path.
        
    Returns:
        Dict[str, Path]: Dictionary of absolute paths.
    """
    return {
        name: base_dir / path
        for name, path in RELATIVE_PATHS.items()
    } 