"""Census G01 (Selected Person Characteristics) data processing module.

This module handles the processing of ABS Census G01 table data, which contains
selected person characteristics and population counts.
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

from .. import config
from .. import utils
from ..census import process_census_table

logger = logging.getLogger('ahgd_etl')

def process_g01_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G01 Census CSV file.
    
    Args:
        csv_file (Path): Path to the Census CSV file to process
        
    Returns:
        Optional[pl.DataFrame]: Processed DataFrame if successful, None if processing failed
    """
    # G01 column mapping
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
        'total_population': ['P_Tot_Tot'],
        'male_population': ['M_Tot_Tot'],
        'female_population': ['F_Tot_Tot']
    }
    
    required_target_columns = [
        'total_population',
        'male_population',
        'female_population'
    ]
    
    return utils._process_single_census_csv(
        csv_file=csv_file,
        geo_column_options=geo_column_options,
        measure_column_map=measure_column_map,
        required_target_columns=required_target_columns,
        table_code='G01'
    )

def process_census_g01_data(
    zip_dir: Path,
    temp_extract_base: Path,
    output_dir: Path,
    geo_output_path: Path,
    time_sk: Optional[int] = None
) -> bool:
    """Process G01 Census data files and create population fact table.
    
    Args:
        zip_dir (Path): Directory containing Census ZIP files
        temp_extract_base (Path): Base directory for temporary file extraction
        output_dir (Path): Directory for output files
        geo_output_path (Path): Path to geographic dimension Parquet file
        time_sk (Optional[int]): Time dimension surrogate key
        
    Returns:
        bool: True if processing successful, False otherwise
    """
    return process_census_table(
        table_code='G01',
        process_file_function=process_g01_file,
        output_filename='fact_population.parquet',
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )
