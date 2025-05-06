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
BASE_DIR = Path(os.getenv('BASE_DIR', '.')).resolve()

try:
    import sys
    # Ensure the project root is discoverable if BASE_DIR is not the parent
    project_root = BASE_DIR.parent if 'scripts' in str(BASE_DIR) else BASE_DIR
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Attempt to import URLs - handle case where script might not exist yet
    try:
        from scripts.direct_urls import ASGS2021_URLS, CENSUS_URLS
    except ImportError:
        # Fallback if direct_urls.py isn't available or populated
        ASGS2021_URLS = {}
        CENSUS_URLS = {}
        # Log a warning or info message if logger is available
        # logger.warning("Could not import URLs from scripts.direct_urls. Using empty fallbacks.")

except ImportError as e:
    # Broader import error handling
    ASGS2021_URLS = {}
    CENSUS_URLS = {}
    # logger.error(f"Error setting up path or importing URLs: {e}")

# Relative paths from BASE_DIR
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

# Absolute Paths
PATHS = {name: BASE_DIR/path_str for name, path_str in RELATIVE_PATHS.items()}

# Data Source URLs
DATA_URLS = {
    'SA1_2021_AUST_GDA2020': ASGS2021_URLS.get('SA1'),
    'SA2_2021_AUST_GDA2020': ASGS2021_URLS.get('SA2'),
    'SA3_2021_AUST_GDA2020': ASGS2021_URLS.get('SA3'),
    'SA4_2021_AUST_GDA2020': ASGS2021_URLS.get('SA4'),
    'STE_2021_AUST_GDA2020': ASGS2021_URLS.get('STE'),
    'POA_2021_AUST_GDA2020': ASGS2021_URLS.get('POA'),
    'CENSUS_GCP_AUS_2021': CENSUS_URLS.get('GCP_ALL')
}
DATA_URLS = {k: v for k, v in DATA_URLS.items() if v is not None}

# Processing Parameters
GEO_LEVELS_SHP_PROCESS = {
    'SA1': 'SA1_2021_AUST_GDA2020',
    'SA2': 'SA2_2021_AUST_GDA2020',
    'SA3': 'SA3_2021_AUST_GDA2020',
    'SA4': 'SA4_2021_AUST_GDA2020',
    'STATE': 'STE_2021_AUST_GDA2020',
    'POA': 'POA_2021_AUST_GDA2020'
}

GEO_LEVELS_CENSUS_PROCESS = ['SA1', 'SA2']

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

# Census Column Mappings
# Note: The 'measure_column_map' focuses on identifying the *key* columns needed to
# start processing or simple transformations. More complex transformations like
# unpivoting (G19, G20, G21, G25) rely on patterns within the specific processing functions.
# Updated GEO_COLUMN_OPTIONS with a more comprehensive list from G19 detailed processing
GEO_COLUMN_OPTIONS = [
    "SA1_CODE_2021", "SA2_CODE_2021", "SA3_CODE_2021", "SA4_CODE_2021",
    "SUA_CODE_2021", "LGA_CODE_2021", "STE_CODE_2021", "GCCSA_CODE_2021",
    "POA_CODE_2021", "CED_CODE_2021", "SED_CODE_2021", "SOS_CODE_2021",
    "SOSR_CODE_2021", "UCL_CODE_2021", "RA_CODE_2021", "region_id",
    'SA1_CODE21', 'SA1_Code_2021', 'SA2_CODE21', 'SA2_Code_2021', 'SA2_NAME_2021',
    'SA3_CODE21', 'SA3_Code_2021', 'SA4_CODE21', 'SA4_Code_2021',
    'GCC_CODE_2021', 'GCC_CODE21', 'STATE_CODE_2021', 'STATE_CODE21',
    'POA_CODE21', 'POSTCODE_2021', 'LGA_CODE21',
    'AUS_CODE_2021', 'AUS_CODE21', 'Australia_CODE_2021' # For national level if needed
]

CENSUS_COLUMN_MAPPINGS = {
    # G01: Basic Population Counts
    "G01": {
        "geo_column_options": GEO_COLUMN_OPTIONS,
        "measure_column_map": {
            'total_persons': ['Tot_P_P', 'Total_Persons_P', 'Persons_Total_P', 'P_Tot'],
            'total_male': ['Tot_P_M', 'Total_Persons_M', 'Males_Total_M', 'M_Tot'],
            'total_female': ['Tot_P_F', 'Total_Persons_F', 'Females_Total_F', 'F_Tot'],
            'total_indigenous': ['Indigenous_P_Tot_P', 'Tot_Indigenous_P', 'Indigenous_Tot_P', 'Indigenous_P']
        },
        "required_target_columns": ['total_persons', 'total_male', 'total_female']
    },

    # G17: Income (Weekly)
    "G17": {
        "geo_column_options": GEO_COLUMN_OPTIONS,
        "measure_column_map": {
            # Target names match fact_income schema
            'median_personal_income_weekly': ['Median_Tot_weekly_personal_income_A', 'Median_positive_zero_tot_inc_weekly_A', 'Median_Tot_Wkly_Personal_Incme'],
            'median_family_income_weekly': ['Median_Tot_weekly_family_income_A', 'Median_Tot_Wkly_Fam_Income'],
            'median_household_income_weekly': ['Median_Tot_weekly_household_income_A', 'Median_Tot_Wkly_Hhold_Income']
        },
        "required_target_columns": ['median_personal_income_weekly', 'median_family_income_weekly', 'median_household_income_weekly']
    },

    # G18: Need for Assistance
    "G18": {
        "geo_column_options": GEO_COLUMN_OPTIONS,
        "measure_column_map": {
            # These are used for unpivoting in process_g18_file
            'P_Needs_assistance': ['P_Need_for_assistance'],
            'M_Needs_assistance': ['M_Need_for_assistance'],
            'F_Needs_assistance': ['F_Need_for_assistance'],
            'P_No_need_assistance': ['P_No_need_for_assistance'],
            'M_No_need_assistance': ['M_No_need_for_assistance'],
            'F_No_need_assistance': ['F_No_need_for_assistance'],
            'P_Assistance_ns': ['P_Need_for_assistance_ns', 'P_Need_for_assistance_not_stated'],
            'M_Assistance_ns': ['M_Need_for_assistance_ns', 'M_Need_for_assistance_not_stated'],
            'F_Assistance_ns': ['F_Need_for_assistance_ns', 'F_Need_for_assistance_not_stated']
        },
        "required_target_columns": [
            'P_Needs_assistance', 'M_Needs_assistance', 'F_Needs_assistance',
            'P_No_need_assistance', 'M_No_need_assistance', 'F_No_need_assistance',
            'P_Assistance_ns', 'M_Assistance_ns', 'F_Assistance_ns'
        ] # Requires all source columns for unpivot
    },

    # G19: Long-Term Health Conditions (Summary Level)
    "G19": {
        "geo_column_options": GEO_COLUMN_OPTIONS,
        "measure_column_map": {
            # Source columns used by process_g19_detailed_file for unpivoting
            # Example: Target column 'arthritis_persons' maps to combinations
            'M_Arthritis': ['M_Arthritis'], 'F_Arthritis': ['F_Arthritis'], 'P_Arthritis': ['P_Arthritis'],
            'M_Asthma': ['M_Asthma'], 'F_Asthma': ['F_Asthma'], 'P_Asthma': ['P_Asthma'],
            'M_Cancer': ['M_Cancer'], 'F_Cancer': ['F_Cancer'], 'P_Cancer': ['P_Cancer'],
            'M_Dementia': ['M_Dementia_incl_Alzheimers'], 'F_Dementia': ['F_Dementia_incl_Alzheimers'], 'P_Dementia': ['P_Dementia_incl_Alzheimers'],
            'M_Diabetes': ['M_Diabetes_excl_gestational_diabetes'], 'F_Diabetes': ['F_Diabetes_excl_gestational_diabetes'], 'P_Diabetes': ['P_Diabetes_excl_gestational_diabetes'],
            'M_Heart_disease': ['M_Heart_disease_incl_heart_attack_or_angina'], 'F_Heart_disease': ['F_Heart_disease_incl_heart_attack_or_angina'], 'P_Heart_disease': ['P_Heart_disease_incl_heart_attack_or_angina'],
            'M_Kidney_disease': ['M_Kidney_disease'], 'F_Kidney_disease': ['F_Kidney_disease'], 'P_Kidney_disease': ['P_Kidney_disease'],
            'M_Lung_condition': ['M_Lung_condition_incl_COPD_emphysema'], 'F_Lung_condition': ['F_Lung_condition_incl_COPD_emphysema'], 'P_Lung_condition': ['P_Lung_condition_incl_COPD_emphysema'],
            'M_Mental_health': ['M_Mental_health_condition_incl_depression_or_anxiety'], 'F_Mental_health': ['F_Mental_health_condition_incl_depression_or_anxiety'], 'P_Mental_health': ['P_Mental_health_condition_incl_depression_or_anxiety'],
            'M_Stroke': ['M_Stroke'], 'F_Stroke': ['F_Stroke'], 'P_Stroke': ['P_Stroke'],
            'M_Other': ['M_Other_long_term_health_condition'], 'F_Other': ['F_Other_long_term_health_condition'], 'P_Other': ['P_Other_long_term_health_condition'],
            'M_No_LTC': ['M_No_long_term_health_condition'], 'F_No_LTC': ['F_No_long_term_health_condition'], 'P_No_LTC': ['P_No_long_term_health_condition'],
            'M_LTC_NS': ['M_Long_term_health_condition_ns'], 'F_LTC_NS': ['F_Long_term_health_condition_ns'], 'P_LTC_NS': ['P_Long_term_health_condition_ns']
        },
        # Requires all source columns for unpivoting in the specific processing function
        "required_target_columns": [
             'M_Arthritis', 'F_Arthritis', 'P_Arthritis', 'M_Asthma', 'F_Asthma', 'P_Asthma',
             'M_Cancer', 'F_Cancer', 'P_Cancer', 'M_Dementia', 'F_Dementia', 'P_Dementia',
             'M_Diabetes', 'F_Diabetes', 'P_Diabetes', 'M_Heart_disease', 'F_Heart_disease', 'P_Heart_disease',
             'M_Kidney_disease', 'F_Kidney_disease', 'P_Kidney_disease', 'M_Lung_condition', 'F_Lung_condition', 'P_Lung_condition',
             'M_Mental_health', 'F_Mental_health', 'P_Mental_health', 'M_Stroke', 'F_Stroke', 'P_Stroke',
             'M_Other', 'F_Other', 'P_Other', 'M_No_LTC', 'F_No_LTC', 'P_No_LTC',
             'M_LTC_NS', 'F_LTC_NS', 'P_LTC_NS'
        ]
    },

    # G20: Number of Long-Term Health Conditions
    "G20": {
        "geo_column_options": GEO_COLUMN_OPTIONS,
        "measure_column_map": {
             # Columns used by process_g20_unpivot_file
            'P_Tot_No_LTC': ['P_Tot_No_long_term_health_condition'],
            'P_Tot_1_LTC': ['P_Tot_1_long_term_health_condition'],
            'P_Tot_2_LTC': ['P_Tot_2_long_term_health_conditions'],
            'P_Tot_3_LTC': ['P_Tot_3_or_more_long_term_health_conditions'],
            'P_Tot_LTC_NS': ['P_Tot_Long_term_health_condition_ns'],
            'M_Tot_No_LTC': ['M_Tot_No_long_term_health_condition'],
            'M_Tot_1_LTC': ['M_Tot_1_long_term_health_condition'],
            'M_Tot_2_LTC': ['M_Tot_2_long_term_health_conditions'],
            'M_Tot_3_LTC': ['M_Tot_3_or_more_long_term_health_conditions'],
            'M_Tot_LTC_NS': ['M_Tot_Long_term_health_condition_ns'],
            'F_Tot_No_LTC': ['F_Tot_No_long_term_health_condition'],
            'F_Tot_1_LTC': ['F_Tot_1_long_term_health_condition'],
            'F_Tot_2_LTC': ['F_Tot_2_long_term_health_conditions'],
            'F_Tot_3_LTC': ['F_Tot_3_or_more_long_term_health_conditions'],
            'F_Tot_LTC_NS': ['F_Tot_Long_term_health_condition_ns']
        },
        "required_target_columns": [ # All source columns needed for unpivot
            'P_Tot_No_LTC', 'P_Tot_1_LTC', 'P_Tot_2_LTC', 'P_Tot_3_LTC', 'P_Tot_LTC_NS',
            'M_Tot_No_LTC', 'M_Tot_1_LTC', 'M_Tot_2_LTC', 'M_Tot_3_LTC', 'M_Tot_LTC_NS',
            'F_Tot_No_LTC', 'F_Tot_1_LTC', 'F_Tot_2_LTC', 'F_Tot_3_LTC', 'F_Tot_LTC_NS'
        ]
    },
# G20 Unpivot Specific Configuration
"G20_UNPIVOT": {
    "sex_patterns": {"M": "M", "F": "F", "P": "P"},
    "age_patterns": {
        "0_4": "0-4", "0_4_yrs": "0-4",
        "5_14": "5-14",
        "15_24": "15-24",
        "25_34": "25-34",
        "35_44": "35-44",
        "45_54": "45-54",
        "55_64": "55-64",
        "65_74": "65-74",
        "75_84": "75-84",
        "85ov": "85+", "85_over": "85+",
        "Tot": "total"
    },
    "condition_count_patterns": {
        "No_cond": "no_condition",
        "One_cond": "one_condition",
        "Two_cond": "two_conditions",
        "Three_more": "three_or_more",
        "Not_stated": "not_stated",
        "Tot": "total_persons"
    }
},
# G25: Unpaid Assistance to a Person with a Disability
"G25": {
    "geo_column_options": GEO_COLUMN_OPTIONS,
    "measure_column_map": {
        # Columns used by process_g25_unpivot_file
        'P_Unpaid_assistance_provided': ['P_Unpaid_assistance_to_a_person_with_a_disability_Yes'],
        'P_Unpaid_assistance_not_provided': ['P_Unpaid_assistance_to_a_person_with_a_disability_No'],
        'P_Unpaid_assistance_not_stated': ['P_Unpaid_assistance_to_a_person_with_a_disability_ns'],
        'M_Unpaid_assistance_provided': ['M_Unpaid_assistance_to_a_person_with_a_disability_Yes'],
        'M_Unpaid_assistance_not_provided': ['M_Unpaid_assistance_to_a_person_with_a_disability_No'],
        'M_Unpaid_assistance_not_stated': ['M_Unpaid_assistance_to_a_person_with_a_disability_ns'],
        'F_Unpaid_assistance_provided': ['F_Unpaid_assistance_to_a_person_with_a_disability_Yes'],
        'F_Unpaid_assistance_not_provided': ['F_Unpaid_assistance_to_a_person_with_a_disability_No'],
        'F_Unpaid_assistance_not_stated': ['F_Unpaid_assistance_to_a_person_with_a_disability_ns'],
        'P_Tot_Unpaid_assist_pop': ['P_Tot'], # Total population for this table context
        'M_Tot_Unpaid_assist_pop': ['M_Tot'],
        'F_Tot_Unpaid_assist_pop': ['F_Tot']
    },
    "required_target_columns": [ # All source columns needed for unpivot
        'P_Unpaid_assistance_provided', 'P_Unpaid_assistance_not_provided', 'P_Unpaid_assistance_not_stated',
        'M_Unpaid_assistance_provided', 'M_Unpaid_assistance_not_provided', 'M_Unpaid_assistance_not_stated',
        'F_Unpaid_assistance_provided', 'F_Unpaid_assistance_not_provided', 'F_Unpaid_assistance_not_stated',
        'P_Tot_Unpaid_assist_pop', 'M_Tot_Unpaid_assist_pop', 'F_Tot_Unpaid_assist_pop'
    ]
},

    # G21: Health Conditions by Characteristics (for _process_single_census_csv)
    # Note: This is separate from the G21_Unpivot config used by process_g21_unpivot_csv
    "G21": {
        "geo_column_options": [
            'SA1_CODE_2021',
            'SA2_CODE_2021',
            'SA3_CODE_2021',
            'SA4_CODE_2021',
            'GCC_CODE_2021',
            'STE_CODE_2021',
            'LGA_CODE_2021'
        ],
        "measure_column_map": {
            'total_count': ['P_Tot_Tot'],
            'has_condition_count': ['P_Tot_Has_condition'],
            'no_condition_count': ['P_Tot_No_condition'],
            'condition_not_stated_count': ['P_Tot_Condition_ns']
        },
        "required_target_columns": [
            'total_count',
            'has_condition_count',
            'no_condition_count',
            'condition_not_stated_count'
        ],
        # Mappings for the unpivot function (process_g21_unpivot_csv)
        "characteristic_types": {"COB": "CountryOfBirth", "LFS": "LabourForceStatus", "AGE": "Age", "SEX": "Sex", "Tot": "Total"},
        "condition_mappings": {
            "Arth": "arthritis", "Asth": "asthma", "Canc": "cancer", "Dem_Alzh": "dementia",
            "Dia_ges_dia": "diabetes", "HD_HA_ang": "heart_disease", "Kid_dis": "kidney_disease",
            "LC_COPD_emph": "lung_condition", "MHC_Dep_anx": "mental_health", "Stroke": "stroke",
            "Oth": "other_condition", "No_cond": "no_condition", "Cond_ns": "condition_not_stated",
            "Tot": "total" # For total counts within a characteristic
        }
    },

    # G21: Health Conditions by Characteristics (for _process_single_census_csv)
    # Note: This is separate from the G21_Unpivot config used by process_g21_unpivot_csv
    "G21": {
        "geo_column_options": [
            'SA1_CODE_2021',
            'SA2_CODE_2021',
            'SA3_CODE_2021',
            'SA4_CODE_2021',
            'GCC_CODE_2021',
            'STE_CODE_2021',
            'LGA_CODE_2021'
        ],
        "measure_column_map": {
            'total_count': ['P_Tot_Tot'],
            'has_condition_count': ['P_Tot_Has_condition'],
            'no_condition_count': ['P_Tot_No_condition'],
            'condition_not_stated_count': ['P_Tot_Condition_ns']
        },
        "required_target_columns": [
            'total_count',
            'has_condition_count',
            'no_condition_count',
            'condition_not_stated_count'
        ],
        # Mappings for the unpivot function (process_g21_unpivot_csv)
        "characteristic_types": {"COB": "CountryOfBirth", "LFS": "LabourForceStatus", "AGE": "Age", "SEX": "Sex", "Tot": "Total"},
        "condition_mappings": {
            "Arth": "arthritis", "Asth": "asthma", "Canc": "cancer", "Dem_Alzh": "dementia",
            "Dia_ges_dia": "diabetes", "HD_HA_ang": "heart_disease", "Kid_dis": "kidney_disease",
            "LC_COPD_emph": "lung_condition", "MHC_Dep_anx": "mental_health", "Stroke": "stroke",
            "Oth": "other_condition", "No_cond": "no_condition", "Cond_ns": "condition_not_stated",
            "Tot": "total" # For total counts within a characteristic
        }
    }
}

# G19 Unpivot Specific Configuration
G19_UNPIVOT = {
    "sex_prefixes": ["M", "F", "P"],  # Male, Female, Person
    "health_conditions_map": {
        "Arthritis": "arthritis", "Arth": "arthritis",
        "Asthma": "asthma", "Asth": "asthma",
        "Cancer": "cancer", "Can_rem": "cancer", "Canc": "cancer",
        "Dementia": "dementia", "Dem_Alzh": "dementia", "Dem": "dementia",
        "Diabetes": "diabetes", "Dia_ges_dia": "diabetes", "Dia": "diabetes",
        "Heart_disease": "heart_disease", "HD_HA_ang": "heart_disease", "HD": "heart_disease",
        "Kidney_disease": "kidney_disease", "Kid_dis": "kidney_disease", "Kid": "kidney_disease",
        "Lung_condition": "lung_condition", "LC_COPD_emph": "lung_condition", "LC": "lung_condition",
        "Mental_health": "mental_health", "MHC_Dep_anx": "mental_health", "MH": "mental_health",
        "Stroke": "stroke",
        "Other": "other_condition", "Oth": "other_condition",
        "No_condition": "no_condition", "No_LTHC": "no_condition", "None": "no_condition",
        "NS": "not_stated", "LTHC_NS": "not_stated"
    },
    "age_range_patterns": {
        "0_14": "0-14",
        "15_24": "15-24",
        "25_34": "25-34",
        "35_44": "35-44",
        "45_54": "45-54",
        "55_64": "55-64",
        "65_74": "65-74",
        "75_84": "75-84",
        "85_over": "85+",
        "Tot": "total"
    }
}

# G18 Unpivot Specific Configuration
G18_UNPIVOT = {
    "sex_prefixes": ["M", "F", "P"],  # Male, Female, Person
    "assistance_categories": {
        "Need_for_assistance": "needs_assistance",
        "No_need_for_assistance": "no_need_for_assistance",
        "Need_for_assistance_ns": "assistance_not_stated",
        "Not_stated": "assistance_not_stated" # Handle variation
    },
    "age_range_patterns": {
        "0_4_yrs": "0-4", "0_4": "0-4",
        "5_14": "5-14",
        "15_19": "15-19",
        "20_24": "20-24",
        "25_34": "25-34",
        "35_44": "35-44",
        "45_54": "45-54",
        "55_64": "55-64",
        "65_74": "65-74",
        "75_84": "75-84",
        "85_over": "85+", "85ov": "85+",
        "Tot": "total"
    }
}

# G17 Unpivot Specific Configuration
G17_UNPIVOT = {
    "sex_prefixes": ["M", "F", "P"],  # Male, Female, Person
    "income_categories": {
        "Neg_Nil_income": "negative_nil_income",
        "1_149": "income_1_149",
        "150_299": "income_150_299",
        "300_399": "income_300_399",
        "400_499": "income_400_499",
        "500_649": "income_500_649",
        "650_799": "income_650_799",
        "800_999": "income_800_999",
        "1000_1249": "income_1000_1249",
        "1250_1499": "income_1250_1499",
        "1500_1749": "income_1500_1749",
        "1750_1999": "income_1750_1999",
        "2000_2999": "income_2000_2999",
        "3000_3499": "income_3000_3499",
        "3500_more": "income_3500_plus",
        "PI_NS": "income_not_stated",
        "Tot": "total"
    },
    "age_range_patterns": {
        "15_19_yrs": "15-19",
        "20_24_yrs": "20-24",
        "25_34_yrs": "25-34",
        "35_44_yrs": "35-44",
        "45_54_yrs": "45-54",
        "55_64_yrs": "55-64",
        "65_74_yrs": "65-74",
        "75_84_yrs": "75-84",
        "85_yrs_ovr": "85+",
        "85ov": "85+",
        "Tot": "total"
    }
}

# Schemas for Output Parquet Files (using Polars types)
SCHEMAS = {
    # Dimension Tables
    "geo_dimension": {
        'geo_sk': pl.UInt64,
        'geo_code': pl.Utf8,
        'geo_level': pl.Categorical, # Or Utf8, Categorical can be more memory efficient
        'geometry': pl.Utf8, # WKT representation
        'longitude': pl.Float64, # Centroid longitude
        'latitude': pl.Float64, # Centroid latitude
        'etl_processed_at': pl.Datetime(time_unit="us") # Explicit time unit
    },
    "dim_time": {
        'time_sk': pl.Int64, # YYYYMMDD format
        'full_date': pl.Date,
        'year': pl.Int16, # Smaller int types where possible
        'quarter': pl.Int8,
        'month': pl.Int8,
        'day_of_month': pl.Int8,
        'day_of_week': pl.Int8, # 1-7 or 0-6 depending on function
        'day_name': pl.Utf8, # Or Categorical
        'month_name': pl.Utf8, # Or Categorical
        'is_weekday': pl.Boolean,
        'financial_year': pl.Utf8, # e.g., "2021/22"
        'is_census_year': pl.Boolean,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
     "dim_health_condition": {
        'condition_sk': pl.Utf8, # MD5 hash
        'condition_code': pl.Utf8, # Natural key from data
        'condition_description': pl.Utf8,
        'health_condition_category': pl.Categorical, # Or Utf8
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    "dim_demographic": {
        'demographic_sk': pl.Utf8, # MD5 hash
        'age_group_code': pl.Utf8, # e.g., '0_4', '5_14', 'Tot'
        'sex_code': pl.Categorical, # 'M', 'F', 'P'
        'age_group_description': pl.Utf8,
        'sex_description': pl.Categorical, # 'Male', 'Female', 'Person'
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    "dim_person_characteristic": {
        'characteristic_sk': pl.Utf8, # MD5 hash
        'characteristic_type': pl.Categorical, # e.g., 'Indigenous Status', 'Language'
        'characteristic_code': pl.Utf8, # e.g., 'Aboriginal', 'ENG'
        'characteristic_description': pl.Utf8,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },

    # Fact Tables (Raw/Intermediate - before joining all dims)
     "fact_population_raw": { # From G01
        'geo_code': pl.Utf8,
        'total_persons': pl.Int64,
        'total_males': pl.Int64,
        'total_females': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    "fact_income_raw": { # From G17
        'geo_code': pl.Utf8,
        'median_personal_income_weekly': pl.Int64, # Store as integer cents? Or Float64? Check source. Assume Int64 for now.
        'median_family_income_weekly': pl.Int64,
        'median_household_income_weekly': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
     "fact_assistance_needs_raw": { # From G18 (unpivoted)
        'geo_code': pl.Utf8,
        'sex_code': pl.Categorical, # 'M', 'F', 'P'
        'assistance_status': pl.Categorical, # 'Needs Assistance', 'No Need', 'Not Stated'
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    "fact_health_conditions_summary_raw": { # From G19 (unpivoted)
        'geo_code': pl.Utf8,
        'sex_code': pl.Categorical, # 'M', 'F', 'P'
        'condition_code': pl.Utf8, # From dim_health_condition
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    "fact_health_conditions_detailed_raw": { # From G21 (unpivoted)
        'geo_code': pl.Utf8,
        'sex_code': pl.Categorical, # 'M', 'F', 'P'
        'age_group_code': pl.Utf8, # e.g., '0_4', '5_14'
        'condition_code': pl.Utf8, # From dim_health_condition
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
     "fact_multiple_conditions_raw": { # From G20 (unpivoted)
        'geo_code': pl.Utf8,
        'sex_code': pl.Categorical, # 'M', 'F', 'P'
        'condition_count_category': pl.Categorical, # '0', '1', '2', '3+', 'NS'
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    "fact_unpaid_assistance_raw": { # From G25 (unpivoted)
        'geo_code': pl.Utf8,
        'sex_code': pl.Categorical, # 'M', 'F', 'P'
        'provides_assistance': pl.Categorical, # 'Yes', 'No', 'Not Stated'
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
     # Final Star Schema Fact Tables (with surrogate keys)
    "fact_population": {
        'geo_sk': pl.UInt64,
        'time_sk': pl.Int64,
        'demographic_sk': pl.Utf8, # Linked to 'P' in dim_demographic initially? Or separate SK for M/F? Check logic. Assume 'P' for now.
        'total_persons': pl.Int64,
        'total_males': pl.Int64,
        'total_females': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
     "fact_income": {
        'geo_sk': pl.UInt64,
        'time_sk': pl.Int64,
        # Income facts usually don't break down by demo/condition/characteristic
        'median_personal_income_weekly': pl.Int64,
        'median_family_income_weekly': pl.Int64,
        'median_household_income_weekly': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    "fact_assistance_needs": {
        'geo_sk': pl.UInt64,
        'time_sk': pl.Int64,
        'demographic_sk': pl.Utf8, # Links Age/Sex
        'characteristic_sk': pl.Utf8, # Links Needs Assistance status ('Needs', 'No Need', 'NS')
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    "fact_health_conditions_summary": { # Corresponds to G19 output
        'geo_sk': pl.UInt64,
        'time_sk': pl.Int64,
        'demographic_sk': pl.Utf8, # Links Sex ('M', 'F', 'P') - Age is not in G19
        'condition_sk': pl.Utf8, # Links specific condition
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    "fact_health_conditions_detailed": { # Corresponds to G21 output
        'geo_sk': pl.UInt64,
        'time_sk': pl.Int64,
        'demographic_sk': pl.Utf8, # Links Age/Sex
        'condition_sk': pl.Utf8, # Links specific condition
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
     "fact_multiple_conditions": { # Corresponds to G20 output
        'geo_sk': pl.UInt64,
        'time_sk': pl.Int64,
        'demographic_sk': pl.Utf8, # Links Sex ('M', 'F', 'P') - Age is not in G20
        'characteristic_sk': pl.Utf8, # Links condition count category ('0', '1', '2', '3+', 'NS')
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    "fact_unpaid_assistance": { # Corresponds to G25 output
        'geo_sk': pl.UInt64,
        'time_sk': pl.Int64,
        'demographic_sk': pl.Utf8, # Links Sex ('M', 'F', 'P') - Age is not in G25
        'characteristic_sk': pl.Utf8, # Links assistance provision ('Yes', 'No', 'NS')
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    },
    # Example of a potentially refined/combined fact table
    "fact_health_conditions_refined": {
        'geo_sk': pl.UInt64,
        'time_sk': pl.Int64,
        'condition_sk': pl.Utf8,
        'demographic_sk': pl.Utf8,
        'characteristic_sk': pl.Utf8, # Could link other characteristics if joined
        'person_count': pl.Int64,
        'etl_processed_at': pl.Datetime(time_unit="us")
    }
    # Add other refined facts as needed, e.g., fact_no_assistance derived from G25
    # "fact_no_assistance": { ... }
}

# Old schema variables - REMOVED
# GEO_DIMENSION_SCHEMA = { ... }
TIME_DIMENSION_SCHEMA = {
    "date": pl.Date,
    "year": pl.Int64,
    "month": pl.Int64,
    "day": pl.Int64,
    "day_of_week": pl.Int64,
    "day_of_year": pl.Int64,
    "week_of_year": pl.Int64,
    "quarter": pl.Int64,
    "financial_year": pl.Utf8,
    "is_census_year": pl.Boolean
}
def initialize_directories(config: 'CensusConfig'):
    """Create all required output directories based on the configuration.
    
    Creates all directories defined in the config.PATHS dictionary if they don't already exist.
    Uses Path.mkdir() with exist_ok=True to handle existing directories gracefully.
    
    Args:
        config: CensusConfig instance containing the paths to initialize
    """
    for path in config.PATHS.values():
        path.mkdir(parents=True, exist_ok=True)


def get_required_census_zips() -> Dict[str, str]:
    """Return a dictionary mapping census table IDs to their required zip file names.
    
    The function extracts zip file names from the `CENSUS_TABLE_PATTERNS` configuration.
    Returns:
        Dict[str, str]: Dictionary with table IDs as keys and corresponding zip file names as values.
    """
    return {
        table_id: pattern.split('\\')[-1].split('.')[0]
        for table_id, pattern in CENSUS_TABLE_PATTERNS.items()
    } if CENSUS_TABLE_PATTERNS else {}


def get_required_geo_zips() -> Dict[str, str]:
    """Return a dictionary mapping geography levels to their required SHP zip file names.
    
    The function extracts shapefile zip names from the `DATA_URLS` configuration.
    Returns:
        Dict[str, str]: Dictionary with geo levels as keys and corresponding SHP zip file names as values.
    """
    return {
        geo_level: url.split('/')[-1]
        for geo_level, url in DATA_URLS.items()
        if url and '_SHP' in url and url.endswith('.zip')
    }


if __name__ == "__main__":
    print(f"Base Directory: {BASE_DIR}")
    print("\nCalculated Paths:")
    for name, path in PATHS.items():
        print(f"- {name}: {path}")


class CensusConfig:
    """Configuration class for accessing Census ETL settings and paths.
    
    This class provides centralized access to all Census ETL configuration settings
    defined in this module, including paths, column mappings, and schema definitions.
    Path values can be configured via environment variables with sensible defaults.
    """

    def __init__(self):
        # Load paths from environment variables or use module defaults
        self.census_data_path = os.getenv('CENSUS_DATA_PATH', str(PATHS['CENSUS_DIR']))
        self.output_path = os.getenv('OUTPUT_PATH', str(PATHS['OUTPUT_DIR']))
        self.temp_path = os.getenv('TEMP_PATH', str(PATHS['TEMP_DIR']))

        # Make module-level configuration variables available as class attributes
        self.column_mappings = CENSUS_COLUMN_MAPPINGS
        self.g19_unpivot = G19_UNPIVOT
        self.g18_unpivot = G18_UNPIVOT
        self.g17_unpivot = G17_UNPIVOT
        self.schemas = SCHEMAS
        self.time_dimension_schema = TIME_DIMENSION_SCHEMA
        self.census_table_patterns = CENSUS_TABLE_PATTERNS
        self.geo_levels_census_process = GEO_LEVELS_CENSUS_PROCESS