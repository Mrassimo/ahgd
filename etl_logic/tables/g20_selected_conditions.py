"""Census G20 (Selected Health Conditions) data processing module.

This module handles the processing of ABS Census G20 table data, which contains
information about selected long-term health conditions and their counts.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict

import polars as pl

from .. import config
from .. import utils
from ..census import process_census_table

logger = logging.getLogger('ahgd_etl')

def process_g20_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G20 Census CSV file for selected health conditions.
    
    Args:
        csv_file (Path): Path to the Census CSV file to process
        
    Returns:
        Optional[pl.DataFrame]: Processed DataFrame if successful, None if processing failed
    """
    # G20 column mapping
    geo_column_options = [
        'SA1_CODE_2021',
        'SA2_CODE_2021',
        'SA3_CODE_2021',
        'SA4_CODE_2021',
        'GCC_CODE_2021',
        'STE_CODE_2021',
        'LGA_CODE_2021'
    ]
    
    measure_column_map = {
        'arthritis_count': ['P_Tot_Arthritis'],
        'asthma_count': ['P_Tot_Asthma'],
        'cancer_count': ['P_Tot_Cancer'],
        'dementia_count': ['P_Tot_Dementia'],
        'diabetes_count': ['P_Tot_Diabetes'],
        'heart_disease_count': ['P_Tot_Heart_disease'],
        'kidney_disease_count': ['P_Tot_Kidney_disease'],
        'lung_condition_count': ['P_Tot_Lung_condition'],
        'mental_health_count': ['P_Tot_Mental_health'],
        'stroke_count': ['P_Tot_Stroke'],
        'other_condition_count': ['P_Tot_Other_condition']
    }
    
    required_target_columns = list(measure_column_map.keys())
    
    return utils._process_single_census_csv(
        csv_file=csv_file,
        geo_column_options=geo_column_options,
        measure_column_map=measure_column_map,
        required_target_columns=required_target_columns,
        table_code='G20'
    )

def process_g20_unpivot_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G20 Census CSV file into long format.
    
    This function transforms the wide-format health condition data into a long format
    where each row represents a specific condition count for a geographic area.
    
    Args:
        csv_file (Path): Path to the Census CSV file to process
        
    Returns:
        Optional[pl.DataFrame]: Processed DataFrame if successful, None if processing failed
    """
    try:
        # Read CSV with Polars
        df = pl.read_csv(csv_file, truncate_ragged_lines=True)
        
        # Find geographic code column
        geo_col = utils.find_geo_column(df, [
            'SA1_CODE_2021',
            'SA2_CODE_2021',
            'SA3_CODE_2021',
            'SA4_CODE_2021',
            'GCC_CODE_2021',
            'STE_CODE_2021',
            'LGA_CODE_2021'
        ])
        
        if not geo_col:
            logger.error(f"[G20] No geographic code column found in {csv_file}")
            return None
            
        # Define condition columns to unpivot
        condition_cols = [
            'P_Tot_Arthritis',
            'P_Tot_Asthma',
            'P_Tot_Cancer',
            'P_Tot_Dementia',
            'P_Tot_Diabetes',
            'P_Tot_Heart_disease',
            'P_Tot_Kidney_disease',
            'P_Tot_Lung_condition',
            'P_Tot_Mental_health',
            'P_Tot_Stroke',
            'P_Tot_Other_condition'
        ]
        
        # Unpivot the data
        melted_df = df.melt(
            id_vars=[geo_col],
            value_vars=condition_cols,
            variable_name='condition_code',
            value_name='count'
        )
        
        # Clean geographic codes
        melted_df = melted_df.with_columns([
            utils.clean_polars_geo_code(pl.col(geo_col)).alias('geo_code')
        ])
        
        # Extract condition name from column name
        parsed_df = melted_df.with_columns([
            pl.col('condition_code').str.extract(r'P_Tot_(.+)$').alias('condition'),
            utils.safe_polars_int(pl.col('count')).alias('count')
        ])
        
        # Select and rename final columns
        result_df = parsed_df.select([
            'geo_code',
            'condition',
            'count'
        ])
        
        # Validate the processed dataframe
        if len(result_df) == 0:
            logger.warning(f"[G20] No valid data rows in {csv_file} after processing")
            return None
            
        logger.info(f"[G20] Successfully processed unpivoted file {csv_file.name}: {len(result_df)} rows")
        return result_df
        
    except Exception as e:
        logger.error(f"[G20] Error processing unpivoted file {csv_file}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_census_g20_data(
    zip_dir: Path,
    temp_extract_base: Path,
    output_dir: Path,
    geo_output_path: Path,
    time_sk: Optional[int] = None
) -> bool:
    """Process G20 Census data files and create selected health conditions fact tables.
    
    Args:
        zip_dir (Path): Directory containing Census ZIP files
        temp_extract_base (Path): Base directory for temporary file extraction
        output_dir (Path): Directory for output files
        geo_output_path (Path): Path to geographic dimension Parquet file
        time_sk (Optional[int]): Time dimension surrogate key
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    # Process wide format
    success_wide = process_census_table(
        table_code='G20',
        process_file_function=process_g20_file,
        output_filename='fact_selected_conditions.parquet',
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )
    
    # Process long format
    success_long = process_census_table(
        table_code='G20',
        process_file_function=process_g20_unpivot_file,
        output_filename='fact_selected_conditions_long.parquet',
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )
    
    return success_wide and success_long
