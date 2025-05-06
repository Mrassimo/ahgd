"""Census G01 (Selected Person Characteristics) data processing module.

This module handles the processing of ABS Census G01 table data, which contains
selected person characteristics and population counts.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List

import polars as pl

from .. import config
from .. import census

logger = logging.getLogger('ahgd_etl')

def process_g01_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a G01 Census CSV file.

    Args:
        csv_file (Path): Path to CSV file.

    Returns:
        Optional[pl.DataFrame]: Processed data or None if error.
    """
    # Define table code
    table_code = "G01"
    logger.debug(f"[{table_code}] Processing file: {csv_file.name} using specific G01 processor.")

    # Load configuration from config.py
    if table_code not in config.CENSUS_COLUMN_MAPPINGS:
        logger.error(f"[{table_code}] Configuration not found in config.CENSUS_COLUMN_MAPPINGS. Cannot process.")
        return None

    g01_config = config.CENSUS_COLUMN_MAPPINGS[table_code]

    # Ensure keys exist before accessing and assign
    geo_column_options = g01_config.get("geo_column_options", [])
    measure_column_map = g01_config.get("measure_column_map", {})
    required_target_columns = g01_config.get("required_target_columns", [])

    if not geo_column_options or not measure_column_map:
         logger.error(f"[{table_code}] Missing required configuration (geo options or measure map). Cannot process.")
         return None

    logger.debug(f"[{table_code}] Loaded mappings from config.")

    # Call the generic CSV processing function from census module
    try:
         result_df = census._process_single_census_csv(
             csv_file=csv_file,
             geo_column_options=geo_column_options,
             measure_column_map=measure_column_map,
             required_target_columns=required_target_columns,
             table_code=table_code
         )
         if result_df is None:
              logger.error(f"[{table_code}] _process_single_census_csv returned None for {csv_file.name}")
              return None
         logger.info(f"[{table_code}] Successfully processed {csv_file.name} using generic helper.")
         return result_df
    except AttributeError:
         logger.error(f"[{table_code}] Failed to find '_process_single_census_csv' in census module. Check import.", exc_info=True)
         return None
    except Exception as e:
         logger.error(f"[{table_code}] Unexpected error during generic processing of {csv_file.name}: {e}", exc_info=True)
         return None
