"""Census G19 (Health Conditions) data processing module.

This module handles the processing of ABS Census G19 table data, which contains
information about long-term health conditions by age and sex.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List

import polars as pl

from .. import config
from .. import utils

logger = logging.getLogger('ahgd_etl')

def process_g19_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G19 Census CSV file.
    
    Args:
        csv_file (Path): Path to the CSV file to process
        
    Returns:
        Optional[pl.DataFrame]: Processed DataFrame or None if processing failed
    """
    # Define table code
    table_code = "G19"
    
    # Define geographic column options
    geo_column_options = [
        'SA1_CODE_2021',
        'SA2_CODE_2021',
        'SA3_CODE_2021',
        'SA4_CODE_2021',
        'GCC_CODE_2021',
        'STE_CODE_2021',
        'LGA_CODE_2021'
    ]
    
    try:
        # Read the CSV file
        df = pl.read_csv(csv_file, truncate_ragged_lines=True)
        logger.info(f"[{table_code}] Read {len(df)} rows from {csv_file.name}")
        
        # Identify geo_code column
        geo_code_column = utils.find_geo_column(df, geo_column_options)
        if not geo_code_column:
            logger.error(f"[{table_code}] Could not find geographic code column in {csv_file.name}")
            return None
        logger.info(f"[{table_code}] Found geographic code column: {geo_code_column}")
        
        # Define patterns for column interpretation
        sex_prefixes = ["M", "F", "P"]  # Male, Female, Person
        
        # Define health condition patterns and standardized names
        condition_patterns = {
            "Arthritis": "arthritis",
            "Arth": "arthritis",
            "Asthma": "asthma",
            "Asth": "asthma",
            "Cancer": "cancer",
            "Can_rem": "cancer",
            "Canc": "cancer",
            "Dementia": "dementia",
            "Dem_Alzh": "dementia",
            "Dem": "dementia",
            "Diabetes": "diabetes",
            "Dia_ges_dia": "diabetes",
            "Dia": "diabetes",
            "Heart_disease": "heart_disease",
            "HD_HA_ang": "heart_disease",
            "HD": "heart_disease",
            "Kidney_disease": "kidney_disease",
            "Kid_dis": "kidney_disease",
            "Kid": "kidney_disease",
            "Lung_condition": "lung_condition",
            "LC_COPD_emph": "lung_condition",
            "LC": "lung_condition",
            "Mental_health": "mental_health",
            "MHC_Dep_anx": "mental_health",
            "MH": "mental_health",
            "Stroke": "stroke",
            "Other": "other_condition",
            "Oth": "other_condition",
            "No_condition": "no_condition",
            "No_LTHC": "no_condition",
            "Not_stated": "not_stated",
            "LTHC_NS": "not_stated"
        }
        
        # Define age range patterns and standardized format
        age_range_patterns = {
            "0_4_yrs": "0-4",
            "0_4": "0-4",
            "5_14": "5-14",
            "15_19": "15-19",
            "20_24": "20-24",
            "25_34": "25-34",
            "35_44": "35-44",
            "45_54": "45-54", 
            "55_64": "55-64",
            "65_74": "65-74",
            "75_84": "75-84",
            "85_over": "85+",
            "85ov": "85+",
            "Tot": "total"
        }
        
        value_vars = [] # List of columns to unpivot
        parsed_cols = {} # Store parsing results
        
        # Process each column
        for col_name in df.columns:
            if col_name == geo_code_column:
                continue
                
            # Try to parse column in format: Sex_Condition_AgeGroup
            parts = col_name.split('_')
            if len(parts) < 2:
                continue
                
            # Extract sex (most consistent part)
            sex = parts[0]
            if sex not in sex_prefixes:
                continue
                
            # Join remaining parts to find condition
            remaining = '_'.join(parts[1:])
            
            parsed_condition = None
            age_part = None
            
            # Try to find condition
            for cond_pattern, cond_name in condition_patterns.items():
                if cond_pattern in remaining:
                    parsed_condition = cond_name
                    # Remove condition pattern to isolate age range
                    age_part = remaining.replace(cond_pattern, "").strip("_")
                    break
                    
            if not parsed_condition:
                continue
                
            # Try to find age range in remaining part
            parsed_age = None
            if age_part:
                for age_pattern, age_name in age_range_patterns.items():
                    if age_pattern in age_part:
                        parsed_age = age_name
                        break
                        
            if not parsed_age:
                parsed_age = "total"  # Default if no age range found
                
            # Create standardized column name
            std_col = f"{sex}_{parsed_age}_{parsed_condition}"
            parsed_cols[col_name] = std_col
            value_vars.append(col_name)
            
        if not value_vars:
            logger.error(f"[{table_code}] No valid columns found to process in {csv_file.name}")
            return None
            
        # Unpivot the data
        id_vars = [geo_code_column]
        df_long = df.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            variable_name="characteristic",
            value_name="count"
        )
        
        # Add parsed characteristics
        df_long = df_long.with_columns([
            pl.col("characteristic").map_dict(parsed_cols).alias("parsed_characteristic")
        ])
        
        # Split parsed characteristic into components
        df_long = df_long.with_columns([
            pl.col("parsed_characteristic").str.split("_").list.get(0).alias("sex"),
            pl.col("parsed_characteristic").str.split("_").list.get(1).alias("age_group"),
            pl.col("parsed_characteristic").str.split("_").list.get(2).alias("condition")
        ])
        
        # Group by geographic code and condition
        df_agg = df_long.group_by([
            geo_code_column, "sex", "age_group", "condition"
        ]).agg([
            pl.col("count").sum().alias("count")
        ])
        
        # Create the final DataFrame with condition counts
        df_final = df_agg.pivot(
            values="count",
            index=[geo_code_column, "sex", "age_group"],
            columns="condition",
            aggregate_function="sum"
        )
        
        # Rename columns to match target schema
        df_final = df_final.rename({
            "arthritis": "arthritis_count",
            "asthma": "asthma_count",
            "cancer": "cancer_count",
            "dementia": "dementia_count",
            "diabetes": "diabetes_count",
            "heart_disease": "heart_disease_count",
            "kidney_disease": "kidney_disease_count",
            "lung_condition": "lung_condition_count",
            "mental_health": "mental_health_count",
            "stroke": "stroke_count",
            "other_condition": "other_condition_count",
            "no_condition": "no_condition_count",
            "not_stated": "not_stated_count"
        })
        
        # Fill any missing columns with 0
        required_columns = [
            "arthritis_count", "asthma_count", "cancer_count",
            "dementia_count", "diabetes_count", "heart_disease_count",
            "kidney_disease_count", "lung_condition_count", "mental_health_count",
            "stroke_count", "other_condition_count", "no_condition_count",
            "not_stated_count"
        ]
        for col in required_columns:
            if col not in df_final.columns:
                df_final = df_final.with_columns(pl.lit(0).alias(col))
        
        logger.info(f"[{table_code}] Successfully processed {csv_file.name}")
        return df_final
        
    except Exception as e:
        logger.error(f"[{table_code}] Error processing {csv_file.name}: {str(e)}")
        return None

def process_census_g19_data(zip_dir: Path, temp_extract_base: Path, output_dir: Path,
                           geo_output_path: Path, time_sk: Optional[int] = None) -> bool:
    """Process G19 Census data files and create health conditions fact table.
    
    Args:
        zip_dir (Path): Directory containing Census zip files
        temp_extract_base (Path): Base directory for temporary file extraction
        output_dir (Path): Directory to write output files
        geo_output_path (Path): Path to geographic dimension file
        time_sk (Optional[int]): Time dimension surrogate key
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    return utils.process_census_table(
        table_id="G19",
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk,
        process_file_func=process_g19_file
    )
