"""Census G21 (Health Conditions by Characteristics) data processing module.

This module handles the processing of ABS Census G21 table data, which contains
information about health conditions broken down by various person characteristics
such as age, sex, country of birth, and labour force status.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict

import polars as pl

from .. import config
from .. import utils
from ..census import process_census_table

logger = logging.getLogger('ahgd_etl')

def process_g21_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G21 Census CSV file for health conditions by characteristics.
    
    Args:
        csv_file (Path): Path to the Census CSV file to process
        
    Returns:
        Optional[pl.DataFrame]: Processed DataFrame if successful, None if processing failed
    """
    # G21 column mapping
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
        'total_count': ['P_Tot_Tot'],
        'has_condition_count': ['P_Tot_Has_condition'],
        'no_condition_count': ['P_Tot_No_condition'],
        'condition_not_stated_count': ['P_Tot_Condition_ns']
    }
    
    required_target_columns = [
        'total_count',
        'has_condition_count',
        'no_condition_count',
        'condition_not_stated_count'
    ]
    
    return utils._process_single_census_csv(
        csv_file=csv_file,
        geo_column_options=geo_column_options,
        measure_column_map=measure_column_map,
        required_target_columns=required_target_columns,
        table_code='G21'
    )

def process_g21_unpivot_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G21 Census CSV file into long format by characteristics.
    
    This function transforms the wide-format health condition data into a long format
    where each row represents condition counts for a specific characteristic combination
    (e.g., age group, sex, country of birth) in a geographic area.
    
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
            logger.error(f"[G21] No geographic code column found in {csv_file}")
            return None
            
        # Define characteristic patterns to identify columns
        characteristic_patterns = {
            'age_group': r'([0-9]+_[0-9]+|Tot)',
            'sex': r'[MFP]',
            'country_of_birth': r'(Australia|Overseas|Tot)',
            'labour_force': r'(Employed|Unemployed|Not_in_labour_force|Tot)'
        }
        
        # Get all columns except geo code that contain count data
        count_cols = [col for col in df.columns if col != geo_col and col.startswith('P_')]
        
        # Unpivot the data
        melted_df = df.melt(
            id_vars=[geo_col],
            value_vars=count_cols,
            variable_name='characteristic_code',
            value_name='count'
        )
        
        # Clean geographic codes
        melted_df = melted_df.with_columns([
            utils.clean_polars_geo_code(pl.col(geo_col)).alias('geo_code')
        ])

        # Parse characteristic codes into components
        melted_df = melted_df.with_columns([
            pl.col('characteristic_code')
            .str.extract(r'P_(\w+)_(\w+)_(\w+)')
            .list.get(0)
            .alias('characteristic_type'),
            pl.col('characteristic_code')
            .str.extract(r'P_(\w+)_(\w+)_(\w+)')
            .list.get(1)
            .alias('characteristic_code'),
            pl.col('characteristic_code')
            .str.extract(r'P_(\w+)_(\w+)_(\w+)')
            .list.get(2)
            .alias('condition')
        ])

        # Standardize characteristic types
        type_mapping = {
            'Age': 'age_group',
            'Sex': 'sex',
            'COB': 'country_of_birth',
            'LFS': 'labour_force_status'
        }
        
        melted_df = melted_df.with_columns([
            pl.col('characteristic_type')
            .map_dict(type_mapping)
            .alias('characteristic_type')
        ])

        # Filter out any rows where parsing failed
        melted_df = melted_df.filter(
            pl.col('characteristic_type').is_not_null() &
            pl.col('characteristic_code').is_not_null() &
            pl.col('condition').is_not_null()
        )

        # Group by all dimensions and sum counts
        df_final = melted_df.group_by([
            'geo_code',
            'characteristic_type',
            'characteristic_code',
            'condition'
        ]).agg([
            pl.col('count').sum().alias('count')
        ])

        logger.info(f"[G21] Successfully processed {csv_file.name} into long format")
        return df_final
        
    except Exception as e:
        logger.error(f"[G21] Error processing unpivoted file {csv_file}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def process_census_g21_data(
    zip_dir: Path,
    temp_extract_base: Path,
    output_dir: Path,
    geo_output_path: Path,
    time_sk: Optional[int] = None
) -> bool:
    """Process G21 Census data files and create health conditions by characteristics fact tables.
    
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
        table_code='G21',
        process_file_function=process_g21_file,
        output_filename='fact_conditions_by_characteristics.parquet',
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )
    
    # Process long format
    success_long = process_census_table(
        table_code='G21',
        process_file_function=process_g21_unpivot_file,
        output_filename='fact_conditions_by_characteristics_long.parquet',
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )
    
    return success_wide and success_long
