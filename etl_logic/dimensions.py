"""Dimension table creation module for the AHGD ETL pipeline.

This module handles the generation and management of dimension tables
such as dim_health_condition and dim_demographic to support the fact tables
in the star schema design.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Set, Tuple
from datetime import datetime
import hashlib

import polars as pl

from . import config
from . import utils

def generate_surrogate_key(*args) -> str:
    """Generate a surrogate key from the given business keys using MD5 hash.
    
    Args:
        *args: Business key components to hash
        
    Returns:
        MD5 hash of the concatenated string representation of the inputs
    """
    if not args:
        raise ValueError("At least one argument required for surrogate key generation")
        
    # Convert all arguments to strings and handle NULL/None values
    str_args = []
    for arg in args:
        if arg is None:
            raise ValueError("NULL values not allowed in surrogate key components")
        str_args.append(str(arg))
    
    # Create combined key string and generate hash
    key_str = '_'.join(str_args)
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

logger = logging.getLogger('ahgd_etl')

def create_health_condition_dimension(source_data_path: Optional[Path] = None,
                                    output_dir: Path = Path("output")) -> Path:
    """
    Create a health condition dimension table.
    
    This function creates a dimension table for health conditions by extracting distinct
    condition values from fact_health_conditions_detailed.parquet if available, or using
    a predefined list of conditions if not.
    
    Args:
        source_data_path: Path to the fact table with health condition data.
                         If None, predefined values will be used.
        output_dir: Directory to save the dimension table.
        
    Returns:
        Path to the created health condition dimension file.
    """
    logger.info("Creating health condition dimension...")
    
    # Define output path
    output_path = output_dir / "dim_health_condition.parquet"
    
    try:
        # Initialize empty list of conditions
        conditions = []
        
        # If fact table is provided, extract unique conditions from it
        if source_data_path is not None and source_data_path.exists():
            logger.info(f"Extracting unique conditions from {source_data_path}")
            try:
                fact_df = pl.read_parquet(source_data_path)
                if 'health_condition' in fact_df.columns:
                    # Extract unique values
                    conditions = fact_df.select(pl.col('health_condition')).unique().to_series().to_list()
                    logger.info(f"Extracted {len(conditions)} unique conditions from fact table")
                else:
                    logger.warning("Required health_condition column not found in fact table")
            except Exception as e:
                logger.error(f"Error reading fact table: {e}")
        
        # If no conditions were found or no fact table provided, use predefined list
        if not conditions:
            logger.info("Using predefined list of health conditions")
            conditions = [
                "arthritis", "asthma", "cancer", "dementia", "diabetes", 
                "heart_disease", "kidney_disease", "lung_condition", 
                "mental_health", "stroke", "other_condition", "no_condition", "not_stated"
            ]
        
        # Create DataFrame
        dim_df = pl.DataFrame({"condition_code": conditions})
        
        # Add surrogate key using MD5 hash of condition code
        dim_df = dim_df.with_columns(
            pl.col('condition_code').map_elements(lambda x: generate_surrogate_key(x)).alias('condition_sk')
        )
        
        # Add additional attributes
        dim_df = dim_df.with_columns([
            # Add condition name (proper case)
            pl.col('condition_code').str.replace('_', ' ').str.to_titlecase().alias('condition_description'),
            
            # Add condition category (now in Title Case)
            pl.when(pl.col('condition_code').is_in(["arthritis", "heart_disease", "kidney_disease", "stroke"]))
              .then(pl.lit("Physical"))
              .when(pl.col('condition_code').is_in(["asthma", "lung_condition"]))
              .then(pl.lit("Respiratory"))
              .when(pl.col('condition_code').is_in(["diabetes"]))
              .then(pl.lit("Endocrine"))
              .when(pl.col('condition_code').is_in(["cancer", "dementia"]))
              .then(pl.lit("Cancer/Dementia"))
              .when(pl.col('condition_code').is_in(["mental_health"]))
              .then(pl.lit("Mental"))
              .when(pl.col('condition_code').is_in(["no_condition", "not_stated"]))
              .then(pl.lit("None"))
              .otherwise(pl.lit("Other"))
              .alias('health_condition_category'),
            
            # Add timestamp
            pl.lit(datetime.now()).alias('etl_processed_at')
        ])
        
        # Save dimension table
        dim_df.write_parquet(output_path)
        logger.info(f"Health condition dimension saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating health condition dimension: {e}")
        raise

def generate_health_condition_dimension(output_dir: Path, g21_path: Optional[Path] = None) -> bool:
    """Generate the health condition dimension table for the data warehouse.
    
    This function creates a standardized dimension table for health conditions with properly 
    categorized condition types (Physical, Mental, etc.). It can either use a predefined 
    list of standard health conditions or enhance that list by extracting additional 
    conditions from existing G21 data.
    
    The function performs the following:
    1. Defines standard health conditions based on ABS classifications
    2. If G21 data is provided, extracts unique conditions and adds them to the standard list
    3. Creates a DataFrame with surrogate keys and standardized attributes
    4. Saves the dimension table to the specified output directory
    
    Args:
        output_dir (Path): Directory where the dimension table will be saved. 
                          This directory must exist and be writable.
        g21_path (Optional[Path]): Path to the G21 data file containing health condition data.
                                  If provided and exists, the function will extract unique
                                  condition values from this file to enhance the standard list.
                                  If None or file doesn't exist, only predefined values are used.
        
    Returns:
        bool: True if the dimension table was successfully created and saved,
             False if any critical error occurred during processing.
              
    Note:
        The output file will be named 'dim_health_condition.parquet' and will include
        the following columns:
        - condition_sk: Surrogate key (integer)
        - health_condition_code: Natural key/condition code (string)
        - health_condition_description: Human-readable condition name (string)
        - health_condition_category: Category grouping (Physical, Mental, etc.) (string)
        - etl_processed_at: Timestamp when the record was created
    """
    logger.info("Creating health condition dimension table")
    
    # Standard health conditions based on ABS classifications
    conditions = [
        {"condition_code": "arthritis", "condition_description": "Arthritis", "health_condition_category": "Physical"},
        {"condition_code": "asthma", "condition_description": "Asthma", "health_condition_category": "Physical"},
        {"condition_code": "cancer", "condition_description": "Cancer", "health_condition_category": "Physical"},
        {"condition_code": "dementia", "condition_description": "Dementia and Alzheimer's", "health_condition_category": "Physical"},
        {"condition_code": "diabetes", "condition_description": "Diabetes", "health_condition_category": "Physical"},
        {"condition_code": "heart_disease", "condition_description": "Heart Disease", "health_condition_category": "Physical"},
        {"condition_code": "kidney_disease", "condition_description": "Kidney Disease", "health_condition_category": "Physical"},
        {"condition_code": "lung_condition", "condition_description": "Lung Condition", "health_condition_category": "Physical"},
        {"condition_code": "stroke", "condition_description": "Stroke", "health_condition_category": "Physical"},
        
        # Mental health conditions
        {"condition_code": "mental_health", "condition_description": "Mental Health Condition", "health_condition_category": "Mental"},
        
        # Other categories
        {"condition_code": "other_condition", "condition_description": "Other Long Term Health Condition", "health_condition_category": "Other"},
        {"condition_code": "no_condition", "condition_description": "No Long Term Health Condition", "health_condition_category": "None"},
        {"condition_code": "not_stated", "condition_description": "Long Term Health Condition Not Stated", "health_condition_category": "Not Stated"},
        
        # Default demographic groups (for compatibility with G20 demographic dimension)
        {"condition_code": "P", "condition_description": "Person", "health_condition_category": "Demographic"},
        {"condition_code": "M", "condition_description": "Male", "health_condition_category": "Demographic"},
        {"condition_code": "F", "condition_description": "Female", "health_condition_category": "Demographic"}
    ]
    
    # Check if G21 data file exists to extract additional conditions
    if g21_path and g21_path.exists():
        try:
            logger.info(f"Extracting condition values from {g21_path}")
            g21_data = pl.read_parquet(g21_path)
            
            # Get unique condition values
            condition_values = g21_data['health_condition'].unique().to_list()
            logger.info(f"Found {len(condition_values)} unique condition values in G21 data")
            
            # Create a set of existing conditions for easy lookup
            existing_conditions = {item["condition_code"] for item in conditions}
            
            # Add any new conditions not already in our list
            for condition in condition_values:
                if condition not in existing_conditions:
                    # Determine category based on simple heuristics
                    category = "Physical"  # Default
                    if "mental" in condition.lower() or "depression" in condition.lower() or "anxiety" in condition.lower():
                        category = "Mental"
                    elif "no_" in condition.lower() or "none" in condition.lower():
                        category = "None"
                    elif "not_stated" in condition.lower() or "ns" in condition.lower():
                        category = "Not Stated"
                    
                    # Create proper title-cased name
                    name = " ".join(w.capitalize() for w in condition.replace("_", " ").split())
                    
                    # Add to our list
                    conditions.append({
                        "condition_code": condition,
                        "health_condition_description": name,
                        "health_condition_category": category
                    })
                    logger.info(f"Added new condition: {condition} ({category})")
        except Exception as e:
            logger.warning(f"Could not extract conditions from G21 data: {e}")
    else:
        logger.info("No G21 data file provided, using predefined values only")
    
    try:
        # Create DataFrame from our list
        df = pl.DataFrame(conditions)
        
        # Add surrogate key using MD5 hash of condition code
        df = df.with_columns([
            pl.col("condition_code").map_elements(lambda x: generate_surrogate_key(x)).alias("condition_sk"),
            pl.lit(datetime.now()).alias("etl_processed_at")   # Add timestamp
        ])
        
        # Save to parquet
        output_path = output_dir / "dim_health_condition.parquet"
        df.write_parquet(output_path)
        logger.info(f"Health condition dimension created at: {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error creating health condition dimension: {e}")
        return False

def create_demographic_dimension(source_data_path: Optional[Path] = None,
                               output_dir: Path = Path("output")) -> Path:
    """
    Create a demographic dimension table for age groups and sex combinations.
    
    This dimension supports G20 and G21 health condition data by providing a reference
    table for all possible age/sex combinations.
    
    Args:
        source_data_path: Optional path to G20 fact data to extract values from.
                         If not provided, uses predefined values.
        output_dir: Output directory for the dimension table.
        
    Returns:
        Path to the created dimension file.
    """
    logger.info("Creating demographic dimension table")
    
    # Define output path
    output_path = output_dir / "dim_demographic.parquet"
    
    try:
        # If source data is available, extract unique combinations from it
        if source_data_path and source_data_path.exists():
            logger.info(f"Extracting demographic values from {source_data_path}")
            fact_df = pl.read_parquet(source_data_path)
            
            # Extract unique combinations
            if "age_group" in fact_df.columns and "sex" in fact_df.columns:
                # Group by age_group and sex to get unique combinations
                demo_df = fact_df.select(["age_group", "sex"]).unique()
                demo_df = _standardize_demographic_values(demo_df)
            else:
                logger.warning("Source data doesn't contain required columns, using predefined values")
                demo_df = _create_predefined_demographic_dimension()
        else:
            logger.info("No source data provided, using predefined values")
            demo_df = _create_predefined_demographic_dimension()
        
        # Add surrogate key using MD5 hash of age group + sex
        demo_df = demo_df.with_columns(
            pl.struct(['age_group', 'sex'])
            .map_elements(lambda x: generate_surrogate_key(x['age_group'], x['sex']))
            .alias('demographic_sk')
        )
        
        # Add derived attributes
        demo_df = _add_demographic_attributes(demo_df)
        
        # Save to parquet
        demo_df.write_parquet(output_path)
        logger.info(f"Demographic dimension saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating demographic dimension: {e}")
        raise

def _create_predefined_person_characteristic_dimension() -> pl.DataFrame:
    """
    Create a predefined person characteristic dimension with common values.
    
    Returns:
        DataFrame with characteristic_type, characteristic_code, and characteristic_name
    """
    # Country of Birth categories
    cob_entries = [
        # Main categories
        ("CountryOfBirth", "Aus", "Australia"),
        ("CountryOfBirth", "Bo_Ocea_Ant", "Other Oceania and Antarctica"),
        ("CountryOfBirth", "Bo_UK_CI_IM", "United Kingdom, Channel Islands and Isle of Man"),
        ("CountryOfBirth", "Bo_NW_Eu", "Other North-West Europe"),
        ("CountryOfBirth", "Bo_SE_Eu", "Southern and Eastern Europe"),
        ("CountryOfBirth", "Bo_NA_ME", "North Africa and the Middle East"),
        ("CountryOfBirth", "Bo_SE_Asia", "South-East Asia"),
        ("CountryOfBirth", "Bo_NE_Asia", "North-East Asia"),
        ("CountryOfBirth", "Bo_SC_Asia", "Southern and Central Asia"),
        ("CountryOfBirth", "Bo_Amer", "Americas"),
        ("CountryOfBirth", "Bo_SS_Afr", "Sub-Saharan Africa"),
        ("CountryOfBirth", "Bo_Tot_ob", "Total overseas born"),
        ("CountryOfBirth", "COB_NS", "Country of birth not stated"),
        ("CountryOfBirth", "Tot", "Total")
    ]
    
    # Labour Force Status categories
    lfs_entries = [
        ("LabourForceStatus", "Emp", "Employed"),
        ("LabourForceStatus", "Unemp", "Unemployed"),
        ("LabourForceStatus", "Not_LF", "Not in the labour force"),
        ("LabourForceStatus", "LFS_NS", "Labour force status not stated"),
        ("LabourForceStatus", "Tot", "Total")
    ]
    
    # Income categories
    income_entries = [
        ("Income", "Neg_Nil", "Negative/Nil income"),
        ("Income", "1_299", "$1-$299"),
        ("Income", "300_649", "$300-$649"),
        ("Income", "650_999", "$650-$999"),
        ("Income", "1000_1749", "$1,000-$1,749"),
        ("Income", "1750_2999", "$1,750-$2,999"),
        ("Income", "3000_more", "$3,000 or more"),
        ("Income", "Inc_NS", "Income not stated"),
        ("Income", "Tot", "Total")
    ]
    
    # Combine all entries
    all_entries = cob_entries + lfs_entries + income_entries
    
    # Create DataFrame
    df = pl.DataFrame({
        "characteristic_type": [entry[0] for entry in all_entries],
        "characteristic_code": [entry[1] for entry in all_entries],
        "characteristic_name": [entry[2] for entry in all_entries]
    })
    
    # Add category field based on characteristic_type
    df = df.with_columns([
        pl.when(pl.col("characteristic_type") == "CountryOfBirth").then(pl.lit("geographic"))
         .when(pl.col("characteristic_type") == "LabourForceStatus").then(pl.lit("employment"))
         .when(pl.col("characteristic_type") == "Income").then(pl.lit("economic"))
         .otherwise(pl.lit("other"))
         .alias("characteristic_category")
    ])
    
    return df

def create_person_characteristic_dimension(source_data_path: Optional[Path] = None,
                                         output_dir: Path = Path("output")) -> Path:
    """
    Create a person characteristic dimension table for G21 data.
    
    This dimension supports G21 health condition by characteristic data by providing
    a reference table for all possible characteristic types and values.
    
    Args:
        source_data_path: Optional path to G21 fact data to extract values from.
                         If not provided, uses predefined values.
        output_dir: Output directory for the dimension table.
        
    Returns:
        Path to the created dimension file.
    """
    logger.info("Creating person characteristic dimension table")
    
    # Define output path
    output_path = output_dir / "dim_person_characteristic.parquet"
    
    try:
        # If source data is available, extract unique combinations from it
        if source_data_path and source_data_path.exists():
            logger.info(f"Extracting characteristic values from {source_data_path}")
            fact_df = pl.read_parquet(source_data_path)
            
            # Extract unique combinations
            if "characteristic_type" in fact_df.columns and "characteristic_value" in fact_df.columns:
                # Group by characteristic_type and characteristic_value to get unique combinations
                char_df = fact_df.select(["characteristic_type", "characteristic_value"]).unique()
                logger.info(f"Extracted {len(char_df)} unique characteristic combinations")
                
                # Rename columns for consistency
                char_df = char_df.rename({"characteristic_value": "characteristic_code"})
                
                # Standardize characteristic names based on type and code
                char_df = char_df.with_columns([
                    # For age groups, convert 25_34 to 25-34
                    pl.when(pl.col("characteristic_type") == "age_group")
                      .then(pl.col("characteristic_code").str.replace("_", "-"))
                      .otherwise(pl.col("characteristic_code"))
                      .alias("characteristic_code"),
                    
                    # Create descriptive names
                    pl.when(pl.col("characteristic_type") == "age_group")
                      .then(pl.col("characteristic_code") + " years")
                      .when(pl.col("characteristic_type") == "sex")
                      .then(pl.col("characteristic_code").str.replace("M", "Male").str.replace("F", "Female"))
                      .when(pl.col("characteristic_type") == "country_of_birth")
                      .then(pl.col("characteristic_code").str.replace("AUS", "Australia").str.replace("_", " "))
                      .when(pl.col("characteristic_type") == "labour_force_status")
                      .then(pl.col("characteristic_code").str.replace("_", " ").str.title())
                      .otherwise(pl.col("characteristic_code").str.replace("_", " ").str.title())
                      .alias("characteristic_name"),
                    
                    # Add category based on characteristic type
                    pl.when(pl.col("characteristic_type") == "age_group").then(pl.lit("demographic"))
                      .when(pl.col("characteristic_type") == "sex").then(pl.lit("demographic"))
                      .when(pl.col("characteristic_type") == "country_of_birth").then(pl.lit("geographic"))
                      .when(pl.col("characteristic_type") == "labour_force_status").then(pl.lit("employment"))
                      .otherwise(pl.lit("other"))
                      .alias("characteristic_category")
                ])
            else:
                logger.warning("Source data doesn't contain required columns, using predefined values")
                char_df = _create_predefined_person_characteristic_dimension()
        else:
            logger.info("No source data provided, using predefined values")
            char_df = _create_predefined_person_characteristic_dimension()
        
        # Add surrogate key using MD5 hash of characteristic type + code
        char_df = char_df.with_columns(
            pl.struct(['characteristic_type', 'characteristic_code'])
            .map_elements(lambda x: generate_surrogate_key(x['characteristic_type'], x['characteristic_code']))
            .alias('characteristic_sk')
        )
        
        # Save to parquet
        char_df.write_parquet(output_path)
        logger.info(f"Person characteristic dimension saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating person characteristic dimension: {e}")
        raise

def _create_predefined_demographic_dimension() -> pl.DataFrame:
    """
    Create a predefined demographic dimension with standard age groups and sexes.
    
    Returns:
        DataFrame with age_group and sex columns
    """
    # Standard age groups
    age_groups = ["0-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+", "Tot"]
    
    # Standard sex values (M=Male, F=Female, P=Persons/Total)
    sexes = ["M", "F", "P"]
    
    # Create all combinations
    combinations = []
    for age in age_groups:
        for sex in sexes:
            combinations.append({"age_group": age, "sex": sex})
    
    # Create DataFrame
    return pl.DataFrame(combinations)

def _add_demographic_attributes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add derived attributes to the demographic dimension.
    
    Args:
        df: DataFrame with age_group and sex columns
         
    Returns:
        DataFrame with additional attributes
    """
    return df.with_columns([
        # Add full sex name
        pl.when(pl.col('sex') == 'M').then(pl.lit('Male'))
          .when(pl.col('sex') == 'F').then(pl.lit('Female'))
          .when(pl.col('sex') == 'P').then(pl.lit('Persons'))
          .otherwise(pl.lit('Unknown'))
          .alias('sex_name'),
        
        # Add age group min value
        pl.when(pl.col('age_group') == '0-14').then(pl.lit(0))
          .when(pl.col('age_group') == '15-24').then(pl.lit(15))
          .when(pl.col('age_group') == '25-34').then(pl.lit(25))
          .when(pl.col('age_group') == '35-44').then(pl.lit(35))
          .when(pl.col('age_group') == '45-54').then(pl.lit(45))
          .when(pl.col('age_group') == '55-64').then(pl.lit(55))
          .when(pl.col('age_group') == '65-74').then(pl.lit(65))
          .when(pl.col('age_group') == '75-84').then(pl.lit(75))
          .when(pl.col('age_group') == '85+').then(pl.lit(85))
          .otherwise(pl.lit(None))
          .alias('age_min'),
        
        # Add age group max value
        pl.when(pl.col('age_group') == '0-14').then(pl.lit(14))
          .when(pl.col('age_group') == '15-24').then(pl.lit(24))
          .when(pl.col('age_group') == '25-34').then(pl.lit(34))
          .when(pl.col('age_group') == '35-44').then(pl.lit(44))
          .when(pl.col('age_group') == '45-54').then(pl.lit(54))
          .when(pl.col('age_group') == '55-64').then(pl.lit(64))
          .when(pl.col('age_group') == '65-74').then(pl.lit(74))
          .when(pl.col('age_group') == '75-84').then(pl.lit(84))
          .when(pl.col('age_group') == '85+').then(pl.lit(120))
          .otherwise(pl.lit(None))
          .alias('age_max'),
        
        # Add timestamp
        pl.lit(datetime.now()).alias('etl_processed_at')
    ])

def _standardize_demographic_values(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize demographic values in a DataFrame.
    
    Args:
        df: DataFrame with age_group and sex columns
        
    Returns:
        DataFrame with standardized values
    """
    return df.with_columns([
        # Standardize age groups (convert 25_34 to 25-34)
        pl.col('age_group').str.replace('_', '-').alias('age_group'),
        
        # Standardize sex codes (uppercase)
        pl.col('sex').str.to_uppercase().alias('sex')
    ])

def get_health_condition_key(condition: Union[str, pl.Expr]) -> pl.Expr:
    """Look up health condition surrogate key for a given condition code.
    
    Args:
        condition: Condition code (string) or Polars expression containing condition codes
        
    Returns:
        Polars expression that can be used to look up health_condition_key values
    """
    # Load health condition dimension if not already loaded
    if not hasattr(get_health_condition_key, '_dim_cache'):
        dim_path = Path("output") / "dim_health_condition.parquet"
        if dim_path.exists():
            get_health_condition_key._dim_cache = pl.read_parquet(dim_path)
        else:
            # Create default dimension if it doesn't exist
            create_health_condition_dimension()
            get_health_condition_key._dim_cache = pl.read_parquet(dim_path)
    
    dim_df = get_health_condition_key._dim_cache
    
    # Create lookup expression
    return (
        pl.lit(dim_df.select(["condition", "condition_sk"]))
        .struct.field("lookup")
        .get(condition)
    )

def get_health_condition_sk(condition: Union[str, pl.Expr]) -> pl.Expr:
    """Alias for get_health_condition_key to maintain compatibility.
    
    Args:
        condition (Union[str, pl.Expr]): The health condition code or expression.
    
    Returns:
        pl.Expr: Expression to lookup the health condition key.
    """
    return get_health_condition_key(condition)

def get_demographic_sk(age_group: Union[str, pl.Expr], sex: Union[str, pl.Expr]) -> pl.Expr:
    """Get the surrogate key for a demographic combination (age group and sex).
    
    Args:
        age_group: Age group string or Polars expression (e.g. "25-34")
        sex: Sex code string or Polars expression (e.g. "M", "F", "P")
        
    Returns:
        Polars expression that will evaluate to the demographic surrogate key
        
    Note:
        This function assumes the demographic dimension table has been loaded
        and is available in the config module. If no match is found, returns -1.
    """
    # Get the demographic dimension table
    demo_df = config.get_dimension_table('demographic')
    
    if demo_df is None:
        return pl.lit(-1)
    
    # Create expression to find matching row and return its surrogate key
    return pl.when(pl.col('age_group').eq(age_group) & pl.col('sex').eq(sex)) \
             .then(pl.col('demographic_sk')) \
             .otherwise(pl.lit(-1))

def get_person_characteristic_sk(
    characteristic_type: Union[str, pl.Expr], 
    characteristic_code: Union[str, pl.Expr]
) -> pl.Expr:
    """Look up person characteristic surrogate key for given type and code.
    
    Args:
        characteristic_type: Characteristic type (string) or Polars expression
        characteristic_code: Characteristic code (string) or Polars expression
        
    Returns:
        Polars expression that can be used to look up characteristic_sk values
    """
    # Load person characteristic dimension if not already loaded
    if not hasattr(get_person_characteristic_sk, '_dim_cache'):
        dim_path = Path("output") / "dim_person_characteristic.parquet"
        if dim_path.exists():
            get_person_characteristic_sk._dim_cache = pl.read_parquet(dim_path)
        else:
            # Create default dimension if it doesn't exist
            create_person_characteristic_dimension()
            get_person_characteristic_sk._dim_cache = pl.read_parquet(dim_path)
    
    dim_df = get_person_characteristic_sk._dim_cache
    
    # Create lookup expression
    return dim_df.filter(
        (pl.col("characteristic_type") == characteristic_type) &
        (pl.col("characteristic_code") == characteristic_code)
    ).select("characteristic_sk").first()