"""Configuration settings for the AHGD ETL pipeline.

This module contains all configuration constants, paths, and data source URLs
used throughout the ETL process. It automatically determines the base directory
from the BASE_DIR environment variable (or .env file), defaulting to the
current working directory.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import polars as pl
import logging

# Load environment variables from .env file, if present
load_dotenv()

# Set up logger
logger = logging.getLogger('ahgd_etl')

# Determine BASE_DIR: Use environment variable or default to current dir '.'
# Ensure it's an absolute path
BASE_DIR = Path(os.getenv('BASE_DIR', '.')).resolve()

# Import URLs after potentially setting BASE_DIR if direct_urls needs it (though it doesn't currently)
# It's generally safer to define BASE_DIR early.
try:
    # Assuming direct_urls.py is in the project root or accessible via PYTHONPATH
    # If direct_urls is in the same directory (etl_logic), use: from .direct_urls import ...
    # Adjust the import based on your project structure if needed.
    # Given the original import, assuming direct_urls is at the root:
    import sys
    # Add project root to path if BASE_DIR is not the current working directory
    # This helps ensure imports work consistently
    if str(BASE_DIR) not in sys.path:
         # Check if parent is already there, avoid adding duplicates
        if str(BASE_DIR.parent) not in sys.path:
             sys.path.insert(0, str(BASE_DIR.parent)) # Add parent for package imports
        # Add BASE_DIR itself if direct_urls is directly under it
        # sys.path.insert(0, str(BASE_DIR))

    from scripts.direct_urls import ASGS2021_URLS, CENSUS_URLS
except ImportError as e:
    print(f"Error importing direct_urls: {e}")
    print("Ensure direct_urls.py is in the project root or accessible via PYTHONPATH.")
    # Provide default empty dicts to avoid crashing downstream code if import fails
    ASGS2021_URLS = {}
    CENSUS_URLS = {}


# Relative paths from BASE_DIR
RELATIVE_PATHS = {
    'DATA_DIR': 'data',
    'RAW_DATA_DIR': 'data/raw',
    'OUTPUT_DIR': 'output',
    'TEMP_DIR': 'data/raw/temp', # Changed from data/temp for consistency
    'LOG_DIR': 'logs',
    'GEOGRAPHIC_DIR': 'data/raw/geographic',
    'CENSUS_DIR': 'data/raw/census',
    'TEMP_ZIP_DIR': 'data/raw/temp/zips', # Kept under temp
    'TEMP_EXTRACT_DIR': 'data/raw/temp/extract' # Kept under temp
}

# --- Absolute Paths ---
# Calculate absolute paths immediately using the determined BASE_DIR
PATHS: Dict[str, Path] = {
    name: BASE_DIR / path_str
    for name, path_str in RELATIVE_PATHS.items()
}

# --- Data Source URLs ---
DATA_URLS = {
    # ASGS Main Structures (Using GDA2020 Shapefiles)
    'SA1_2021_AUST_GDA2020': ASGS2021_URLS.get('SA1'),
    'SA2_2021_AUST_GDA2020': ASGS2021_URLS.get('SA2'),
    'SA3_2021_AUST_GDA2020': ASGS2021_URLS.get('SA3'),
    'SA4_2021_AUST_GDA2020': ASGS2021_URLS.get('SA4'),
    'STE_2021_AUST_GDA2020': ASGS2021_URLS.get('STE'),
    'POA_2021_AUST_GDA2020': ASGS2021_URLS.get('POA'),
    # Census Data
    'CENSUS_GCP_AUS_2021': CENSUS_URLS.get('GCP_ALL')
}
# Filter out None values if URLs were not found in direct_urls
DATA_URLS = {k: v for k, v in DATA_URLS.items() if v is not None}


# --- Processing Parameters ---

# Geographic levels to process (Shapefiles)
GEO_LEVELS_SHP_PROCESS = {
    'SA1': 'SA1_2021_AUST_GDA2020',
    'SA2': 'SA2_2021_AUST_GDA2020',
    'SA3': 'SA3_2021_AUST_GDA2020',
    'SA4': 'SA4_2021_AUST_GDA2020',
    'STATE': 'STE_2021_AUST_GDA2020',
    'POA': 'POA_2021_AUST_GDA2020'  # Added POA level
}

# Geographic levels for Census data processing
GEO_LEVELS_CENSUS_PROCESS = ['SA1', 'SA2'] # Example: Limit Census processing

# Census table patterns (using regex)
# Example: Find G01 CSVs for SA1 or SA2 levels
_census_geo_pattern = "|".join(GEO_LEVELS_CENSUS_PROCESS)
CENSUS_TABLE_PATTERNS = {
    "G01": rf"2021\s*Census_G01[_\s].*?({_census_geo_pattern})\.csv$",
    "G17": rf"2021\s*Census_G17[A-C]_.*?({_census_geo_pattern})\.csv$",
    "G18": rf"2021\s*Census_G18[_\s].*?({_census_geo_pattern})\.csv$",
    "G19": rf"2021\s*Census_G19[A-C]_.*?({_census_geo_pattern})\.csv$",
    "G20": rf"2021\s*Census_G20[A-B]_.*?({_census_geo_pattern})\.csv$",
    "G21": rf"2021\s*Census_G21[A-C]_.*?({_census_geo_pattern})\.csv$",
    "G25": rf"2021\s*Census_G25[_\s].*?({_census_geo_pattern})\.csv$"
}

# --- Data Schemas ---
# Define schemas for dimension and fact tables using Polars data types

# Geographic dimension schema
GEO_DIMENSION_SCHEMA = {
    'geo_sk': pl.UInt64,     # Surrogate key
    'geo_code': pl.Utf8,     # Natural key (ABS geographic code)
    'geo_level': pl.Utf8,    # Geographic level (SA1, SA2, etc.)
    'geometry': pl.Utf8,     # WKT geometry
    'centroid_lon': pl.Float64,  # Geometric centroid longitude
    'centroid_lat': pl.Float64,  # Geometric centroid latitude
    'pop_weighted_lon': pl.Float64,  # Population-weighted centroid longitude
    'pop_weighted_lat': pl.Float64,  # Population-weighted centroid latitude
    'postcode': pl.Utf8,     # Postal code (from POA correspondence)
    'state_code': pl.Utf8,   # State/territory code
    'state_name': pl.Utf8,   # State/territory name
    'area_sqkm': pl.Float64, # Area in square kilometers
    'etl_processed_at': pl.Datetime  # Processing timestamp
}

# Time dimension schema
TIME_DIMENSION_SCHEMA = {
    'time_sk': pl.Int64,        # Surrogate key (YYYYMMDD format)
    'full_date': pl.Date,       # The actual date (natural key)
    'year': pl.Int32,           # Calendar year
    'quarter': pl.Int32,        # Calendar quarter (1-4)
    'month': pl.Int32,          # Calendar month (1-12)
    'month_name': pl.Utf8,      # Month name (January, February, etc.)
    'day_of_month': pl.Int32,   # Day of month (1-31)
    'day_of_week': pl.Int32,    # Day of week (1=Monday, 7=Sunday)
    'day_name': pl.Utf8,        # Day name (Monday, Tuesday, etc.)
    'financial_year': pl.Utf8,  # Australian financial year (e.g., '2021/22')
    'is_weekday': pl.Boolean,   # True if Monday-Friday
    'is_census_year': pl.Boolean, # True if Census year (2011, 2016, 2021, etc.)
    'etl_processed_at': pl.Datetime  # Timestamp when the record was processed
}

# Health condition dimension schema
HEALTH_CONDITION_SCHEMA = {
    'condition_sk': pl.UInt64,      # Surrogate key
    'condition': pl.Utf8,           # Natural key (condition name)
    'condition_name': pl.Utf8,      # Full condition name
    'condition_category': pl.Utf8,  # Category (e.g., Physical, Mental)
    'etl_processed_at': pl.Datetime # Timestamp when the record was processed
}

# Demographic dimension schema
DEMOGRAPHIC_SCHEMA = {
    'demographic_sk': pl.UInt64,    # Surrogate key
    'age_group': pl.Utf8,           # Age group (e.g., '0-14', '15-24')
    'sex': pl.Utf8,                 # Sex code (M, F, P)
    'sex_name': pl.Utf8,            # Full sex name (Male, Female, Persons)
    'age_min': pl.Int32,            # Minimum age in group
    'age_max': pl.Int32,            # Maximum age in group
    'etl_processed_at': pl.Datetime # Timestamp when the record was processed
}

# Person characteristic dimension schema
PERSON_CHARACTERISTIC_SCHEMA = {
    'characteristic_sk': pl.UInt64,       # Surrogate key
    'characteristic_type': pl.Utf8,       # Type (CountryOfBirth, LabourForceStatus, Income)
    'characteristic_code': pl.Utf8,       # Natural key (abbreviated code)
    'characteristic_name': pl.Utf8,       # Full descriptive name
    'characteristic_category': pl.Utf8,   # Category (geographic, employment, economic)
    'etl_processed_at': pl.Datetime       # Timestamp when the record was processed
}

# --- Helper Functions ---

def get_required_geo_zips() -> Dict[str, str]:
    """Generate required geographic ZIP URLs based on GEO_LEVELS_SHP_PROCESS.

    Returns:
        Dict[str, str]: Dictionary mapping zip filenames to their download URLs.
    """
    required_zips = {}
    for level, prefix in GEO_LEVELS_SHP_PROCESS.items():
        if prefix in DATA_URLS:
            # Standard ABS naming convention seems to be PREFIX_SHP.zip
            # Adjust if POA or others have different naming (like POA_..._GDA2020_SHP.zip)
            if level == 'POA':
                 zip_filename = f"{prefix}_GDA2020_SHP.zip" # Handle specific POA naming
            else:
                 zip_filename = f"{prefix}_SHP.zip"
            required_zips[zip_filename] = DATA_URLS[prefix]
        else:
            print(f"Warning: URL for geographic level prefix '{prefix}' not found in DATA_URLS.")
    return required_zips

def get_required_census_zips() -> Dict[str, str]:
    """Get required Census data pack ZIP URLs based on CENSUS_TABLE_PATTERNS.
       Currently hardcoded for GCP_ALL pack which contains G01 for all levels.
       Could be extended if specific packs per level are needed.

    Returns:
        Dict[str, str]: Dictionary mapping zip filenames to their download URLs.
    """
    required_zips = {}
    # Assuming G01 (and potentially others needed later) are in the main GCP pack
    census_key = 'CENSUS_GCP_AUS_2021'
    if census_key in DATA_URLS:
        url = DATA_URLS[census_key]
        # Extract filename from URL
        filename = Path(url).name
        required_zips[filename] = url
    else:
         print(f"Warning: URL for Census key '{census_key}' not found in DATA_URLS.")

    # Add logic here if other specific packs are needed based on CENSUS_TABLE_PATTERNS
    # e.g., if G17 required a different zip file.

    return required_zips

# Deprecated: get_paths is replaced by the global PATHS dictionary
# def get_paths(base_dir: Path) -> Dict[str, Path]:
#     """Generate absolute paths based on BASE_DIR."""
#     return {
#         name: base_dir / path
#         for name, path in RELATIVE_PATHS.items()
#     }

# --- Initialization ---
def initialise_directories():
    """Initialise all required directory paths for the ETL process.
    This function creates all the directories specified in PATHS if they don't exist.
    """
    for path_name, path in PATHS.items():
        if isinstance(path, Path) and not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")

    logger.info("All required directories have been initialised")

# Create alias for backward compatibility
initialize_directories = initialise_directories

# Example of how to use the paths:
if __name__ == "__main__":
    print(f"Base Directory: {BASE_DIR}")
    print("\nCalculated Paths:")
    for name, path in PATHS.items():
        print(f"- {name}: {path}")

    print("\nRequired Geo Zips:")
    print(get_required_geo_zips())

    print("\nRequired Census Zips:")
    print(get_required_census_zips())

    # Example: Create directories if script is run directly
    # initialize_directories()

# --- Census Column Mappings ---
# These mappings help standardize column names across different census files
# and handle variations in column naming between releases

CENSUS_COLUMN_MAPPINGS = {
    "G17": {
        # Need for Assistance column mappings
        "assistance_needed_columns": [
            "Assistance_Needed_P", "Has_Need_Assistance_P", 
            "P_Tot_Has_need_for_assistance", "Tot_Need_for_assistance",
            "Has_Need_for_Assistance_P", "Assisted_Need_P"  # Added test column names
        ],
        "no_assistance_needed_columns": [
            "No_Assistance_Needed_P", "Does_Not_Need_Assistance_P", 
            "P_Tot_Does_not_have_need_for_assistance", "Tot_No_need_for_assistance",
            "Does_Not_Have_Need_for_Assistance_P", "No_Assistance_Need_P"  # Added test column names
        ],
        "assistance_not_stated_columns": [
            "Assistance_Not_Stated_P", "Need_For_Assistance_Not_Stated_P", 
            "P_Tot_Need_for_assistance_not_stated", "Tot_Not_stated",
            "Need_for_Assistance_Not_Stated_P", "Assistance_Need_NS_P"  # Added test column names
        ]
    },
    
    "G18": {
        # Unpaid Work column mappings
        "provided_care_columns": [
            "Provided_Unpaid_Care_P", "Cared_for_Child_P", 
            "P_Tot_Provided_unpaid_child_care", "Tot_Provided_unpaid_childcare",
            "Provided_Care_P"  # Added test column name
        ],
        "no_care_provided_columns": [
            "No_Unpaid_Care_Provided_P", "Did_Not_Care_For_Child_P", 
            "P_Tot_Did_not_provide_unpaid_child_care", "Tot_Did_not_provide_unpaid_childcare",
            "No_Care_Provided_P"  # Added test column name
        ],
        "care_not_stated_columns": [
            "Care_Not_Stated_P", "Care_For_Child_Not_Stated_P", 
            "P_Tot_Unpaid_child_care_not_stated", "Tot_Not_stated"
        ]
    },
    
    "G19": {
        # Health Conditions column mappings (summary level)
        "has_condition_columns": [
            "Has_Condition_P", "Has_Health_Condition_P", 
            "P_Tot_Has_long_term_health_condition", "Tot_Has_LTHC",
            "Has_Long_Term_Health_Condition_P"  # Added test column name
        ],
        "no_condition_columns": [
            "No_Condition_P", "No_Health_Condition_P", 
            "P_Tot_No_long_term_health_condition", "Tot_No_LTHC",
            "No_Long_Term_Health_Condition_P"  # Added test column name
        ],
        "condition_not_stated_columns": [
            "Condition_Not_Stated_P", "Health_Condition_Not_Stated_P", 
            "P_Tot_Long_term_health_condition_not_stated", "Tot_Not_stated",
            "Health_Condition_Not_Stated_P"  # Added test column name
        ]
    },
    
    "G20": {
        # Specific Health Conditions column patterns
        "condition_patterns": {
            "arthritis": ["Arth", "Arthritis"],
            "asthma": ["Asth", "Asthma"],
            "cancer": ["Can", "Canc", "Cancer"],
            "dementia": ["Dem", "Dementia"],
            "diabetes": ["Dia", "Diabetes"],
            "heart_disease": ["HD", "Heart"],
            "kidney_disease": ["Kid", "Kidney"],
            "lung_condition": ["LC", "Lung"],
            "mental_health": ["MH", "Mental"],
            "stroke": ["Stroke"],
            "other_condition": ["Oth", "Other"],
            "no_condition": ["No", "None"],
            "not_stated": ["NS", "Not_stated"]
        },
        # Age group patterns
        "age_group_patterns": {
            "0-14": ["0_14", "0to14"],
            "15-24": ["15_24", "15to24"],
            "25-34": ["25_34", "25to34"],
            "35-44": ["35_44", "35to44"],
            "45-54": ["45_54", "45to54"],
            "55-64": ["55_64", "55to64"],
            "65-74": ["65_74", "65to74"],
            "75-84": ["75_84", "75to84"],
            "85+": ["85_over", "85plus", "85p"]
        }
    },
    
    "G21": {
        # Characteristic Types
        "characteristic_types": {
            "COB": "CountryOfBirth",
            "LFS": "LabourForceStatus",
            "TPI": "Income",
            "CANA": "AssistanceNeeded",
            "ADFS": "DisabilityServices"
        },
        # Condition mappings (same as G20 but may have different codes)
        "condition_mappings": {
            "Arth": "arthritis",
            "Asth": "asthma",
            "Can_rem": "cancer",
            "Canc_rem": "cancer",
            "Dem_Alzh": "dementia",
            "Dia_ges_dia": "diabetes",
            "Dis_ges_dia": "diabetes",
            "HD_HA_ang": "heart_disease",
            "Kid_dis": "kidney_disease",
            "LC_COPD_emph": "lung_condition",
            "MHC_Dep_anx": "mental_health",
            "MHC_dep_anx": "mental_health",
            "Stroke": "stroke",
            "oth_LTHC": "other_condition",
            "no_LTHC": "no_condition",
            "LTHC_NS": "not_stated"
        }
    },
    
    "G25": {
        # Unpaid Assistance column mappings
        "provided_assistance_columns": [
            "Provided_Unpaid_Assistance_P",
            "Provided_Assistance_P",
            "Unpaid_Assistance_Provided_P",
            "P_Tot_Provided_unpaid_assistance",
            "P_Tot_Has_provided_unpaid_assistance",
            "Provided_unpaid_assistance",
            "P_Tot_prvided_unpaid_assist",
            "P_Tot_Prvided_unpaid_assist",
            "prvided_unpaid_assist"
        ],
        "no_assistance_provided_columns": [
            "No_Unpaid_Assistance_Provided_P",
            "No_Assistance_Provided_P",
            "Did_Not_Provide_Unpaid_Assistance_P",
            "P_Tot_No_unpaid_assistance",
            "P_Tot_Did_not_provide_unpaid_assistance",
            "Did_not_provide_unpaid_assistance",
            "P_Tot_No_prvided_unpaid_assist",
            "no_prvided_unpaid_assist"
        ],
        "assistance_not_stated_columns": [
            "Unpaid_Assistance_Not_Stated_P",
            "Assistance_Not_Stated_P",
            "Unpaid_Assistance_NS_P",
            "P_Tot_Unpaid_assistance_not_stated",
            "P_Tot_Unpaid_assistance_NS",
            "Unpaid_assistance_not_stated",
            "P_Tot_Unpaid_assist_NS",
            "unpaid_assist_NS"
        ]
    }
}