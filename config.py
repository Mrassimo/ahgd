"""
Configuration file for the AHGD ETL project.

This module centralizes all configuration settings to ensure consistency and ease of maintenance.
It defines base directory, paths, URLs, and other constants used throughout the project.

Dependencies:
- pathlib: For path handling
- os: For environment variables
- polars: For data type schemas
- direct_urls: For data download URLs (imported to keep URL management separate)

Usage:
Import specific variables as needed, e.g., from config import BASE_DIR, PATHS
"""

import os
from pathlib import Path
import polars as pl
from scripts.direct_urls import ASGS2021_URLS, CENSUS_URLS, CORRESPONDENCE_PACKAGE_URL  # Import URL dictionaries from direct_urls.py

# Define BASE_DIR using pathlib and environment variables
# Read BASE_DIR from .env if set, otherwise default to the project root (directory of this config file)
BASE_DIR = Path(os.getenv("BASE_DIR", Path(__file__).parent.parent)).resolve()

# PATHS dictionary: Define directories relative to BASE_DIR for consistency
PATHS = {
    "DATA_DIR": BASE_DIR / "data",  # Directory for raw and processed data files
    "OUTPUT_DIR": BASE_DIR / "output",  # Directory for ETL outputs and reports
    "LOG_DIR": BASE_DIR / "logs",  # Directory for log files
    "TEMP_DIR": BASE_DIR / "temp",  # Directory for temporary files
}

# DATA_URLS: Imported from direct_urls.py for modularity
# This includes ASGS and Census URLs
DATA_URLS = {
    "ASGS": ASGS2021_URLS,
    "CENSUS": CENSUS_URLS,
    "CORRESPONDENCE": CORRESPONDENCE_PACKAGE_URL,
}

# Configuration variables as specified
GEO_LEVELS_SHP_PROCESS = ["SA1", "SA2", "SA3", "SA4", "STE", "POA"]  # Geographic levels for shapefile processing
GEO_LEVELS_CENSUS_PROCESS = ["SA1", "SA2", "POA"]  # Geographic levels for census data processing
CENSUS_TABLE_PATTERNS = {
    "G01": r"G01_.*",  # Pattern for G01 tables (e.g., population data)
    "G17": r"G17_.*",  # Pattern for G17 tables (e.g., income data)
    "G18": r"G18_.*",  # Pattern for G18 tables (e.g., assistance needed data)
    "G19": r"G19_.*",  # Pattern for G19 tables (e.g., health conditions data)
    "G20": r"G20_.*",  # Pattern for G20 tables (e.g., selected conditions data)
    "G21": r"G21_.*",  # Pattern for G21 tables (e.g., conditions by characteristics data)
    "G25": r"G25_.*",  # Pattern for G25 tables (e.g., unpaid assistance data)
}

# CENSUS_COLUMN_MAPPINGS: Consolidated dictionary of column mappings for census tables
# Extracted and completed from census.py examples for G01, G17, G18, G19, G20, G21, G25
# Each mapping defines how raw columns should be transformed or renamed for consistency
CENSUS_COLUMN_MAPPINGS = {
    "G01": {  # Population table mappings (based on G01 examples)
        "TOT_P_M": "total_male",  # Total male population
        "TOT_P_F": "total_female",  # Total female population
        "AGE_5YRS_P_M": "age_5yrs_male",  # Age in 5-year groups for males
        "AGE_5YRS_P_F": "age_5yrs_female",  # Age in 5-year groups for females
        # Add more mappings as needed, ensuring comprehensive coverage
    },
    "G17": {  # Income table mappings (based on G17 examples)
        "MED_EQ_INC_WK": "median_equivalised_weekly_income",  # Median equivalised weekly household income
        "MEAN_EQ_INC_WK": "mean_equivalised_weekly_income",  # Mean equivalised weekly household income
        "TOT_PSN": "total_persons",  # Total persons with income
        # Complete with all relevant columns from census.py
    },
    "G18": {  # Assistance needed table mappings (based on G18 examples)
        "NEED_ASS_CORE_ACT_P": "need_assistance_core_activities",  # Persons needing assistance with core activities
        "NO_NEED_ASS_CORE_ACT_P": "no_need_assistance_core_activities",  # Persons not needing assistance
        "UNPAID_ASS_PROV_P": "unpaid_assistance_provided",  # Persons providing unpaid assistance
        # Ensure all columns are mapped for consistency
    },
    "G19": {  # Health conditions table mappings (based on G19 examples)
        "LONG_TERM_HEALTH_COND_P": "long_term_health_condition",  # Persons with long-term health conditions
        "ARTHRITIS_P": "arthritis",  # Persons with arthritis
        "ASTHMA_P": "asthma",  # Persons with asthma
        "CANCER_P": "cancer",  # Persons with cancer
        # Add mappings for all conditions listed in census.py
    },
    "G20": {  # Selected conditions table mappings (based on G20 examples)
        "COND_MENTAL_HEALTH_P": "mental_health_condition",  # Persons with mental health conditions
        "COND_OTHER_LONG_TERM_P": "other_long_term_condition",  # Persons with other long-term conditions
        # Complete based on available data
    },
    "G21": {  # Conditions by characteristics table mappings
        "P_Tot_Tot": "total_count",           # Total persons in category
        "P_Tot_Has_condition": "has_condition_count", # Persons with condition
        "P_Tot_No_condition": "no_condition_count",   # Persons without condition
        "P_Tot_Condition_ns": "condition_not_stated_count", # Condition not stated
        # Geographic code columns
        "SA1_CODE_2021": "sa1_code",
        "SA2_CODE_2021": "sa2_code",
        "SA3_CODE_2021": "sa3_code",
        "SA4_CODE_2021": "sa4_code",
        "GCC_CODE_2021": "gcc_code",
        "STE_CODE_2021": "ste_code",
        "LGA_CODE_2021": "lga_code"
    },
    "G25": {  # Unpaid assistance table mappings (based on G25 examples)
        "UNPAID_ASS_HRS_P": "unpaid_assistance_hours",  # Hours of unpaid assistance provided
        "UNPAID_ASS_TYPE_P": "unpaid_assistance_type",  # Type of unpaid assistance
        # Ensure comprehensive mapping
    },
    # Note: This dictionary should be expanded or modified as new tables are added. Comments added for clarity and maintainability.
}

# SCHEMAS dictionary: Define schemas for output dimensions and facts using Polars dtypes
# This ensures data consistency and type enforcement during ETL processes
SCHEMAS = {
    "GEO_DIMENSION_SCHEMA": pl.Schema([
        ("geo_code", pl.Utf8),  # Geographic code (e.g., SA1, SA2)
        ("geo_name", pl.Utf8),  # Geographic name
        ("geo_level", pl.Utf8),  # Level of geography (e.g., SA1)
    ]),
    "TIME_DIMENSION_SCHEMA": pl.Schema([
        ("year", pl.Int32),  # Year of the data
        ("month", pl.Int32),  # Month of the data (if applicable)
        ("date_key", pl.Utf8),  # Surrogate key for time dimension
    ]),
    "DEMOGRAPHIC_DIMENSION_SCHEMA": pl.Schema([
        ("age_group", pl.Utf8),  # Age group categories
        ("sex", pl.Utf8),  # Sex (male/female)
        ("income_group", pl.Utf8),  # Income group categories
    ]),
    # Standard schemas
    "HEALTH_CONDITION_DIMENSION_SCHEMA": pl.Schema([
        ("condition_sk", pl.Utf8),
        ("condition_code", pl.Utf8),
        ("condition_description", pl.Utf8),
    ]),
    
    # Explicit table schemas for enforcement
    "dim_health_condition": {
        "condition_sk": pl.Utf8,
        "condition_code": pl.Utf8,
        "condition_description": pl.Utf8,
        "health_condition_category": pl.Utf8,
        "etl_processed_at": pl.Utf8
    },
    "FACT_HEALTH_CONDITION_COUNTS_SCHEMA": pl.Schema([
        ("geo_code", pl.Utf8),  # Foreign key to geography
        ("time_key", pl.Utf8),  # Foreign key to time dimension
        ("condition_code", pl.Utf8),  # Foreign key to health condition
        ("count", pl.Int64),  # Count of persons with condition
    ]),
    # Add more schemas as needed for other facts and dimensions, e.g., income, assistance, etc.
    # This should be aligned with the data model defined in the project.
}