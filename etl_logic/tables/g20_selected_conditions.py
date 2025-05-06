"""Census G20 (Selected Health Conditions) data processing module.

This module handles the processing of ABS Census G20 table data, which contains
information about selected long-term health conditions and their counts.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict
import traceback

import polars as pl

from .. import config
from ..config import CENSUS_COLUMN_MAPPINGS
from .. import census
from ..utils import process_census_table
from ..config import CensusConfig

logger = logging.getLogger('ahgd_etl')

def process_g20_file(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a G20 Census CSV file (Count of Selected Long-Term Health Conditions by Age by Sex).

    Note: This function uses the generic _process_single_census_csv helper and assumes
    a flat structure in the config mapping. For G20's complex column structure,
    using process_g20_unpivot_csv is generally recommended.

    Args:
        csv_file (Path): Path to CSV file.

    Returns:
        Optional[pl.DataFrame]: Processed data or None if error.
    """
    # Define table code
    table_code = "G20"
    logger.debug(f"[{table_code}] Processing file: {csv_file.name} using specific G20 processor (generic fallback).")

    # Define geographic column options (Consider moving to config)
    geo_column_options = ['region_id', 'SA1_CODE21', 'SA2_CODE21', 'SA1_CODE_2021', 'SA2_CODE_2021']

    # Get measure column mappings from config
    measure_column_map: Dict[str, List[str]] = {}
    required_target_columns: List[str] = []

    if table_code in config.CENSUS_COLUMN_MAPPINGS:
         g20_config = config.CENSUS_COLUMN_MAPPINGS[table_code]
         # Check if a simple, flat measure_column_map exists in config
         measure_column_map = g20_config.get("measure_column_map", {})
         required_target_columns = g20_config.get("required_target_columns", [])
         if not measure_column_map:
              logger.warning(f"[{table_code}] No flat 'measure_column_map' found in config. Using fallback.")
         else:
              logger.debug(f"[{table_code}] Loaded flat mappings from config.")
    else:
         logger.warning(f"[{table_code}] Mappings not found in config, using limited fallback.")

    # Fallback if config or flat map is missing
    if not measure_column_map:
        measure_column_map = {
            'no_condition_total': ['P_Tot_No_cond'],
            'one_condition_total': ['P_Tot_One_cond'],
            'two_conditions_total': ['P_Tot_Two_cond'],
            'three_or_more_total': ['P_Tot_Three_more'],
            'not_stated_total': ['P_Tot_Not_stated'],
        }
        required_target_columns = list(measure_column_map.keys())

    # Call the generic processing function
    logger.warning(f"[{table_code}] Using generic _process_single_census_csv for G20. "
                   "This may not capture all detail. Consider using process_g20_unpivot_csv.")
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

def process_g20_unpivot_csv(csv_file: Path) -> Optional[pl.DataFrame]:
    """Process a single G20 Census file with proper unpivoting for detailed breakdowns
       (count of conditions per person by age and sex).

    Args:
        csv_file: Path to the CSV file to process

    Returns:
        Optional[pl.DataFrame]: Processed DataFrame or None if processing failed
    """
    table_code = "G20_unpivot"
    logger.debug(f"[{table_code}] Processing file: {csv_file.name} using G20 unpivot processor.")
    geo_column_options = ["region_id", "SA1_CODE_2021", "SA2_CODE_2021", "SA3_CODE_2021", "SA4_CODE_2021"]

    try:
        # Read the CSV file
        df = pl.read_csv(csv_file, truncate_ragged_lines=True)
        logger.info(f"[{table_code}] Read {len(df)} rows from {csv_file.name}")

        # Find geo_code column
        geo_code_column = utils.find_geo_column(df, geo_column_options)
        if not geo_code_column:
            logger.error(f"[{table_code}] Could not find geographic code column in {csv_file.name}")
            return None
    except Exception as e:
        logger.error(f"[{table_code}] Error processing file {csv_file.name}: {e}")
        logger.error(traceback.format_exc())
        return None

def process_g20_census_data(config: CensusConfig, census_data_path: Path, output_path: Path) -> bool:
        """Process all G20 Census data files into a single output file.
    
        Wrapper function that orchestrates the processing of all G20 CSV files by:
        1. Finding all CSV files in the input directory
        2. Processing each file using process_g20_unpivot_csv
        3. Combining results into a single output file
    
        Args:
            config: Census configuration object
            census_data_path: Path to directory containing G20 CSV files
            output_path: Path to output directory
    
        Returns:
            bool: True if processing succeeded, False otherwise
        """
        return process_census_table(
            table_name="G20",
            processing_function=process_g20_unpivot_csv,
            config=config,
            census_data_path=census_data_path,
            output_path=output_path,
            output_file_name="fact_selected_conditions.parquet"
        )
        logger.info(f"[{table_code}] Found geographic code column: {geo_code_column}")
        # Get patterns for parsing column names from config
        g20_unpivot_config = CENSUS_COLUMN_MAPPINGS.get("G20_UNPIVOT", {})
        sex_patterns = g20_unpivot_config.get("sex_patterns", {})
        age_patterns = g20_unpivot_config.get("age_patterns", {})
        condition_count_patterns = g20_unpivot_config.get("condition_count_patterns", {})

        if not (sex_patterns and age_patterns and condition_count_patterns):
            logger.error(f"[{table_code}] Missing G20_UNPIVOT configuration in config.py")
            return None

        value_vars = []
        parsed_cols = {}

        # Identify and parse value columns
        for col_name in df.columns:
            if col_name == geo_code_column: continue

            # Basic check for structure (Sex_...)
            if not any(col_name.startswith(f"{sex}_") for sex in sex_patterns.keys()):
                 continue

            parts = col_name.split('_', 1)
            sex = sex_patterns.get(parts[0])
            if not sex: continue

            remaining = parts[1]
            age_range = None
            condition_count_category = None
            age_part = None
            matched_condition_key = None

            # Try to extract condition count category first
            for cond_key in sorted(condition_count_patterns.keys(), key=len, reverse=True):
                 if cond_key in remaining:
                      matched_condition_key = cond_key
                      break

            if matched_condition_key:
                condition_count_category = condition_count_patterns[matched_condition_key]
                try: # Determine age part
                     idx = remaining.index(matched_condition_key)
                     if idx == 0: age_part = remaining[len(matched_condition_key):].strip("_")
                     else: age_part = remaining[:idx].strip("_")
                except ValueError:
                     age_part = remaining.replace(matched_condition_key, "").strip("_")
            else:
                 continue # Failed to find condition count category

            # Determine age range from age_part
            matched_age_key = None
            for age_pat in sorted(age_patterns.keys(), key=len, reverse=True):
                 if age_pat == age_part or age_pat.replace("_","") == age_part:
                      matched_age_key = age_pat
                      break

            if matched_age_key:
                 age_range = age_patterns[matched_age_key]
            elif age_part == 'Tot':
                 age_range = 'total'
            else:
                 logger.debug(f"[{table_code}] Failed to parse age part '{age_part}' in column {col_name}")
                 continue # Failed to determine age range


            if sex and age_range and condition_count_category:
                parsed_cols[col_name] = {
                    "sex": sex,
                    "age_range": age_range,
                    "condition_count_category": condition_count_category
                }
                value_vars.append(col_name)
            else:
                 logger.debug(f"[{table_code}] Failed to fully parse column: {col_name}")


        if not value_vars:
            logger.error(f"[{table_code}] No valid columns found to process in {csv_file.name}")
            return None
        logger.info(f"[{table_code}] Unpivoting {len(value_vars)} columns.")

        # Unpivot the data
        df_long = df.melt(
            id_vars=[geo_code_column],
            value_vars=value_vars,
            variable_name="characteristic",
            value_name="count"
        )

        # Add parsed characteristics
        df_long = df_long.with_columns([
            pl.col("characteristic").map_dict({k: v["sex"] for k, v in parsed_cols.items()}).alias("sex"),
            pl.col("characteristic").map_dict({k: v["age_range"] for k, v in parsed_cols.items()}).alias("age_range"),
            pl.col("characteristic").map_dict({k: v["condition_count_category"] for k, v in parsed_cols.items()}).alias("condition_count_category")
        ])

        # Select final columns
        df_final = df_long.select([
            utils.clean_polars_geo_code(pl.col(geo_code_column)).alias("geo_code"),
            pl.col("sex"),
            pl.col("age_range"),
            pl.col("condition_count_category"),
            utils.safe_polars_int(pl.col("count")).alias("count")
        ]).filter(pl.col("count").is_not_null() & (pl.col("count") > 0))


        if len(df_final) == 0:
            logger.warning(f"[{table_code}] No non-zero data after processing {csv_file.name}")
            return None

        logger.info(f"[{table_code}] Successfully processed file {csv_file.name} with {len(df_final)} records")
        return df_final

