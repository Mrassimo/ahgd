"""Census G21 (Health Conditions by Characteristics) data processing module.

This module handles the processing of ABS Census G21 table data, which contains
information about health conditions broken down by various person characteristics
such as age, sex, country of birth, and labour force status.

NOTE - Potential Duplicate Sources:
1. Input CSVs may contain duplicate rows that need explicit deduplication
2. Combining multiple CSVs may create duplicates without grouping/distinct
3. The grain (geo_sk, time_sk, condition_sk, demo_sk, characteristic_sk) should
   be enforced before writing output
4. Surrogate key lookups assume uniqueness - need verification step
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict
import traceback

import polars as pl

from .. import config
from .. import census
from etl_logic.dimensions import (
    get_health_condition_sk,
    get_person_characteristic_sk, 
    get_demographic_sk
)
from ..census import process_census_table

logger = logging.getLogger('ahgd_etl')

def process_g21_csv_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G21 Census CSV file for health conditions by characteristics.
    
    Args:
        csv_file (Path): Path to the Census CSV file to process
        
    Returns:
        Optional[pl.DataFrame]: Processed DataFrame if successful, None if processing failed
    """
    # G21 column mapping
    # Get G21 column mapping from config
    g21_config = config.CENSUS_COLUMN_MAPPINGS.get("G21", {})
    geo_column_options = g21_config.get("geo_column_options", config.GEO_COLUMN_OPTIONS)
    measure_column_map = g21_config.get("measure_column_map", {})
    required_target_columns = g21_config.get("required_target_columns", [])

    if not measure_column_map or not required_target_columns:
        logger.error(f"[G21] Missing required mappings in config for process_g21_csv_file.")
        return None

    # Call the generic census processing function from the census module
    return census._process_single_census_csv(
        csv_file=csv_file,
        geo_column_options=geo_column_options,
        measure_column_map=measure_column_map,
        required_target_columns=required_target_columns,
        table_code="G21"
    )
def process_g21_file(zip_dir: Path, temp_extract_base: Path, output_dir: Path,
                      geo_output_path: Optional[Path] = None, time_sk: Optional[int] = None) -> bool:
    """Process Census G21 data (Health Conditions by Characteristics) and create fact table.
    
    Args:
        zip_dir (Path): Directory containing Census zip files
        temp_extract_base (Path): Base directory for temporary file extraction
        output_dir (Path): Directory to write output files
        geo_output_path (Optional[Path]): Path to geographic dimension file
        time_sk (Optional[int]): Time dimension surrogate key
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    table_code = "G21"
    output_filename = f"fact_health_conditions_by_characteristic.parquet"
    output_path = output_dir / output_filename
    
    # First process the census data normally
    success = process_census_table(
        table_code=table_code,
        process_file_function=process_g21_csv_file,
        output_filename=output_filename,
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk
    )
    
    if not success:
        return False
        
    # Now read back the output and ensure grain uniqueness
    try:
        df = pl.read_parquet(output_path)
        
        # Ensure no duplicates on grain (geo_sk, time_sk, condition_sk, demo_sk, characteristic_sk)
        unique_rows = df.group_by([
            'geo_sk', 'time_sk', 'condition_sk', 'demo_sk', 'characteristic_sk'
        ]).agg([
            pl.col('has_condition_count').sum(),
            pl.col('no_condition_count').sum(),
            pl.col('condition_not_stated_count').sum(),
            pl.col('total_count').sum()
        ])
        
        # Verify uniqueness
        unique_check = unique_rows.select([
            'geo_sk', 'time_sk', 'condition_sk', 'demo_sk', 'characteristic_sk'
        ]).unique().shape[0] == unique_rows.shape[0]
        
        if not unique_check:
            logger.error("Unable to resolve duplicate keys in G21 output")
            return False
            
        # Write back the deduplicated data
        unique_rows.write_parquet(output_path)
        return True
        
    except Exception as e:
        logger.error(f"Failed post-processing G21 output: {str(e)}")
        return False

def process_g21_file_generic_fallback(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a G21 Census CSV file (Type of Long-Term Health Condition by Selected Person Characteristics).
       Uses the generic _process_single_census_csv helper.
    """
    table_code = "G21"
    logger.debug(f"[{table_code}] Processing file: {csv_file.name} using specific G21 processor (generic fallback).")
    
    # Get G21 column mapping from config
    g21_config = config.CENSUS_COLUMN_MAPPINGS.get("G21", {})
    geo_column_options = g21_config.get("geo_column_options", config.GEO_COLUMN_OPTIONS)
    measure_column_map = g21_config.get("measure_column_map", {})
    required_target_columns = g21_config.get("required_target_columns", [])

    if not measure_column_map or not required_target_columns:
        logger.error(f"[{table_code}] Missing required mappings in config for process_g21_file_generic_fallback.")
        return None

    logger.warning(f"[{table_code}] Using generic _process_single_census_csv for G21. "
                   "This may not capture all detail. Consider using process_g21_unpivot_csv.")
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
         logger.error(f"[{table_code}] Failed to find '_process_single_census_csv' in census module.", exc_info=True)
         return None
    except Exception as e:
         logger.error(f"[{table_code}] Unexpected error during generic processing of {csv_file.name}: {e}", exc_info=True)
         return None

def process_g21_unpivot_csv(csv_file: Path) -> Optional[pl.DataFrame]:
    """
    Process a G21 file (Condition by Characteristic) to extract unpivoted health condition data.

    Args:
        csv_file: Path to the G21 CSV file.

    Returns:
        Optional[pl.DataFrame]: Processed DataFrame with unpivoted health condition data,
                                or None if processing failed.
    """
    table_code = "G21_Unpivot"
    logger.info(f"[{table_code}] Processing G21 file for unpivot: {csv_file.name}")

    try:
        df = pl.read_csv(csv_file, truncate_ragged_lines=True)
        logger.info(f"[{table_code}] Read {len(df)} rows from {csv_file.name}")

        # Identify geo_code column
        geo_column_options = ['region_id', 'SA1_CODE21', 'SA2_CODE21', 'SA1_CODE_2021', 'SA2_CODE_2021']
        geo_code_column = utils.find_geo_column(df, geo_column_options)
        if not geo_code_column:
            logger.error(f"[{table_code}] Could not find geographic code column in {csv_file.name}")
            return None
        logger.info(f"[{table_code}] Found geographic code column: {geo_code_column}")

        # Get G21 mappings from config
        # Assumes structure: config.CENSUS_COLUMN_MAPPINGS['G21'] = {
        #     "characteristic_types": {"COB": "CountryOfBirth", ...},
        #     "condition_mappings": {"Arth": "arthritis", ...}
        # }
        g21_mappings = config.CENSUS_COLUMN_MAPPINGS.get("G21", {})
        characteristic_types = g21_mappings.get("characteristic_types", {})
        condition_mappings = g21_mappings.get("condition_mappings", {})

        if not characteristic_types or not condition_mappings:
             logger.error(f"[{table_code}] Missing G21 mappings in config.py (characteristic_types or condition_mappings). Cannot parse columns.")
             # Attempt fallback to simplified internal mappings if desired, or return None
             return None

        value_vars = [] # List of columns to unpivot
        parsed_cols = {} # Store parsing results

        # Dynamically parse columns based on config mappings
        for col_name in df.columns:
            if col_name == geo_code_column: continue

            characteristic_type = None
            characteristic_value = None
            condition = None
            remaining_part = col_name

            # 1. Identify Characteristic Type and Value
            matched_char_key = None
            for char_key in sorted(characteristic_types.keys(), key=len, reverse=True):
                if remaining_part.startswith(f"{char_key}_"):
                    characteristic_type = characteristic_types[char_key]
                    remaining_part = remaining_part[len(char_key)+1:] # Remove prefix and underscore
                    matched_char_key = char_key
                    break
            
            if not characteristic_type:
                 # Handle cases like Tot_Tot_Arth where char type might be 'Tot'
                 if remaining_part.startswith("Tot_"):
                     characteristic_type = characteristic_types.get("Tot", "Total") # Use config or default
                     remaining_part = remaining_part[len("Tot_"):]
                     matched_char_key = "Tot"
                 else:
                     logger.debug(f"[{table_code}] Could not determine characteristic type for column: {col_name}")
                     continue

            # 2. Identify Condition from the rest
            matched_cond_key = None
            for cond_key in sorted(condition_mappings.keys(), key=len, reverse=True):
                 # Need to handle cases where condition is at end or start
                 if remaining_part.endswith(f"_{cond_key}"):
                     condition = condition_mappings[cond_key]
                     characteristic_value = remaining_part[:-len(cond_key)-1]
                     matched_cond_key = cond_key
                     break
                 elif remaining_part == cond_key: # Case like P_Tot_Tot (already handled char type 'Tot')
                      condition = condition_mappings[cond_key]
                      characteristic_value = matched_char_key # Use the char key ('Tot') as value
                      matched_cond_key = cond_key
                      break

            if not condition:
                 # Handle special Tot cases (e.g., COB_Aus_Tot)
                 if remaining_part.endswith("_Tot"):
                      condition = condition_mappings.get("Tot", "total")
                      characteristic_value = remaining_part[:-len("_Tot")]
                      matched_cond_key = "Tot"
                 else:
                    logger.debug(f"[{table_code}] Could not determine condition for remaining part '{remaining_part}' in column: {col_name}")
                    continue
            
            # Store parsed info
            value_vars.append(col_name)
            parsed_cols[col_name] = {
                "characteristic_type": characteristic_type,
                "characteristic_value": characteristic_value,
                "condition": condition
            }

        if not value_vars:
            logger.error(f"[{table_code}] No valid columns found to unpivot in {csv_file.name}")
            return None

        logger.info(f"[{table_code}] Unpivoting {len(value_vars)} columns.")

        # Unpivot using melt
        long_df = df.melt(
            id_vars=[geo_code_column],
            value_vars=value_vars,
            variable_name="column_info",
            value_name="count"
        )

        # Map parsed info back
        long_df = long_df.with_columns([
             utils.clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"), # Clean geo_code early
             pl.col("column_info").map_dict({k: v["characteristic_type"] for k, v in parsed_cols.items()}).alias("characteristic_type"),
             pl.col("column_info").map_dict({k: v["characteristic_value"] for k, v in parsed_cols.items()}).alias("characteristic_value"),
             pl.col("column_info").map_dict({k: v["condition"] for k, v in parsed_cols.items()}).alias("condition"),
             utils.safe_polars_int(pl.col("count")).alias("count")
        ])

        # Select final columns and filter
        result_df = long_df.select([
            "geo_code",
            "characteristic_type",
            "characteristic_value",
            pl.col("condition").alias("health_condition"),
            "count"
        ]).filter(pl.col("count").is_not_null() & (pl.col("count") > 0))

        if len(result_df) == 0:
            logger.warning(f"[{table_code}] No non-zero data after unpivoting {csv_file.name}")
            return None

        logger.info(f"[{table_code}] Created unpivoted G21 DataFrame with {len(result_df)} rows")
        return result_df

    except Exception as e:
        logger.error(f"[{table_code}] Error processing {table_code} file {csv_file.name}: {str(e)}", exc_info=True)
        return None
def process_g21_census_data(config: config.CensusConfig, census_data_path: Path, output_path: Path) -> bool:
    """Process all G21 Census data files and create health conditions by characteristics fact table.
    
    This function orchestrates the processing of G21 Census files by:
    - Finding and extracting relevant CSV files
    - Processing each file using process_g21_file
    - Combining results into standardized fact table format
    - Writing final output as Parquet
    
    Args:
        config: Census configuration object
        census_data_path: Directory containing Census data files
        output_path: Directory where output files should be written
        
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    return process_census_table(
        table_code="G21",
        process_file_function=process_g21_file,
        output_filename="fact_health_conditions_by_characteristic.parquet",
        zip_dir=census_data_path,
        temp_extract_base=config.extract_path,
        output_dir=output_path,
        geo_output_path=config.geography_output_path,
        time_sk=config.time_sk
    )
