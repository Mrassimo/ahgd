"""Census G18 (Need for Assistance) data processing module.

This module handles the processing of ABS Census G18 table data, which contains
information about core activity need for assistance by age and sex.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List

import polars as pl

from .. import config
from .. import utils

logger = logging.getLogger('ahgd_etl')

def process_g18_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G18 Census CSV file.
    
    Args:
        csv_file (Path): Path to the CSV file to process
        
    Returns:
        Optional[pl.DataFrame]: Processed DataFrame or None if processing failed
    """
    # Define table code
    table_code = "G18"
    
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
        
        # Define assistance categories and standardized names
        assistance_categories = {
            "Need_for_assistance": "needs_assistance",
            "Has_need_for_assistance": "needs_assistance",
            "No_need_for_assistance": "no_need_for_assistance",
            "Does_not_have_need_for_assistance": "no_need_for_assistance",
            "Need_for_assistance_not_stated": "assistance_not_stated",
            "Need_for_assistance_ns": "assistance_not_stated",
            "Not_stated": "assistance_not_stated"
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
                
            # Try to parse column in format: Sex_AgeGroup_AssistanceStatus
            parts = col_name.split('_')
            if len(parts) < 2:
                continue
                
            # Extract sex (most consistent part)
            sex = parts[0]
            if sex not in sex_prefixes:
                continue
                
            # Join remaining parts to find assistance status
            remaining = '_'.join(parts[1:])
            
            parsed_assistance = None
            age_part = None
            
            # Try to find assistance status
            for assist_pattern, assist_name in assistance_categories.items():
                if assist_pattern in remaining:
                    parsed_assistance = assist_name
                    # Remove assistance pattern to isolate age range
                    age_part = remaining.replace(assist_pattern, "").strip("_")
                    break
                    
            if not parsed_assistance:
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
            std_col = f"{sex}_{parsed_age}_{parsed_assistance}"
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
            pl.col("parsed_characteristic").str.split("_").list.get(2).alias("assistance_status")
        ])
        
        # Group by geographic code and assistance status
        df_agg = df_long.group_by([
            geo_code_column, "sex", "age_group", "assistance_status"
        ]).agg([
            pl.col("count").sum().alias("count")
        ])
        
        # Pivot the assistance status back to columns
        df_final = df_agg.pivot(
            values="count",
            index=[geo_code_column, "sex", "age_group"],
            columns="assistance_status",
            aggregate_function="sum"
        )
        
        # Rename columns to match target schema
        df_final = df_final.rename({
            "needs_assistance": "need_assistance_count",
            "no_need_for_assistance": "no_need_assistance_count",
            "assistance_not_stated": "need_assistance_ns_count"
        })
        
        # Fill any missing columns with 0
        required_columns = ["need_assistance_count", "no_need_assistance_count", "need_assistance_ns_count"]
        for col in required_columns:
            if col not in df_final.columns:
                df_final = df_final.with_columns(pl.lit(0).alias(col))
        
        logger.info(f"[{table_code}] Successfully processed {csv_file.name}")
        return df_final
        
    except Exception as e:
        logger.error(f"[{table_code}] Error processing {csv_file.name}: {str(e)}")
        return None

def process_census_g18_data(zip_dir: Path, temp_extract_base: Path, output_dir: Path,
                           geo_output_path: Path, time_sk: Optional[int] = None) -> bool:
    """Process G18 Census data files and create assistance need fact table.
    
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
        table_id="G18",
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk,
        process_file_func=process_g18_file
    )
