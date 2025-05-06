"""Census G25 (Unpaid Assistance) data processing module.

This module handles the processing of ABS Census G25 table data, which contains
information about unpaid assistance to persons with a disability by age and sex.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List
import traceback

import polars as pl

from .. import config
from .. import utils

from ..census import _process_single_census_csv
from ..dimensions import create_demographic_dimension
from .. import validation # Import the validation module
logger = logging.getLogger('ahgd_etl')

def process_census_g25_data(zip_dir: Path, temp_extract_base: Path, output_dir: Path,
                          geo_output_path: Path, time_sk: Optional[int] = None) -> bool:
    """Process G25 Census data files and create unpaid assistance fact table.
    
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
        table_id="G25",
        zip_dir=zip_dir,
        temp_extract_base=temp_extract_base,
        output_dir=output_dir,
        geo_output_path=geo_output_path,
        time_sk=time_sk,
        process_file_func=process_g25_file
    )
def process_g25_file(csv_file: Path, time_sk: Optional[int] = None) -> Optional[pl.DataFrame]:
    """Process a G25 Census CSV file for unpaid assistance.

    Args:
        csv_file (Path): Path to CSV file.
        time_sk (Optional[int]): Time dimension surrogate key.

    Returns:
        Optional[pl.DataFrame]: Processed G25 data or None if an error occurred.
    """
    table_code = 'G25'
    logger.info(f"[{table_code}] Starting G25 data processing from file: {csv_file.name}")

    try:
        # Define geographic column options
        geo_column_options = ['region_id', 'SA1_CODE21', 'SA2_CODE21', 'SA1_CODE_2021', 'SA2_CODE_2021',
                            'SA3_CODE_2021', 'SA4_CODE_2021', 'LGA_CODE_2021', 'STE_CODE_2021']

        # Get measure column mappings from config
        if table_code not in config.CENSUS_COLUMN_MAPPINGS:
            logger.error(f"[{table_code}] Configuration not found in config.CENSUS_COLUMN_MAPPINGS. Cannot process.")
            return None

        g25_config = config.CENSUS_COLUMN_MAPPINGS[table_code]
        measure_column_map = g25_config.get("measure_column_map", {})
        required_target_columns = g25_config.get("required_target_columns", [])

        if not measure_column_map:
            logger.error(f"[{table_code}] Missing required configuration (measure map). Cannot process.")
            return None

        logger.debug(f"[{table_code}] Loaded mappings from config.")

        # Call the generic CSV processing function
        g25_df = _process_single_census_csv(
            csv_file=csv_file,
            geo_column_options=geo_column_options,
            measure_column_map=measure_column_map,
            required_target_columns=required_target_columns,
            table_code=table_code
        )

        if g25_df is None:
            logger.error(f"[{table_code}] _process_single_census_csv returned None for {csv_file.name}")
            return None

        logger.info(f"[{table_code}] CSV processed successfully. Shape: {g25_df.shape}")

        # --- Data Cleaning ---
        logger.info(f"[{table_code}] Starting data cleaning...")
        cleaned_df = g25_df

        # 1. Column Renaming (Inferring common Census names, confirm with actual data)
        # TODO: Verify these are the correct source column names from the raw G25 file.
        rename_mapping = {
            "Age": "age_group", # Assuming 'Age' is the source column for age groups
            "Sex": "sex"        # Assuming 'Sex' is the source column for sex
            # region_id and measure columns (e.g., provided_assistance_count)
            # are assumed to be handled by the loading process (process_g25_file)
        }
        # Rename only columns that exist in the DataFrame
        actual_rename_mapping = {k: v for k, v in rename_mapping.items() if k in cleaned_df.columns}
        if actual_rename_mapping:
             cleaned_df = cleaned_df.rename(actual_rename_mapping)
             logger.debug(f"[{table_code}] Renamed columns: {actual_rename_mapping}")
        else:
             logger.warning(f"[{table_code}] No columns found to rename based on mapping: {rename_mapping}. Check source data headers.")


        # Define expected columns after potential renaming
        age_col = "age_group"
        sex_col = "sex"
        measure_cols = [col for col in cleaned_df.columns if col.endswith('_count')] # Assuming measure cols end with _count

        # 2. Handle Totals (Filter out summary rows)
        # TODO: Confirm 'Total' is the correct indicator in age_group and sex columns.
        initial_rows = cleaned_df.height
        if age_col in cleaned_df.columns:
            cleaned_df = cleaned_df.filter(~pl.col(age_col).str.contains("(?i)total"))
        if sex_col in cleaned_df.columns:
            cleaned_df = cleaned_df.filter(~pl.col(sex_col).str.contains("(?i)total"))
        rows_after_total_filter = cleaned_df.height
        logger.debug(f"[{table_code}] Removed {initial_rows - rows_after_total_filter} 'Total' rows.")

        # 3. Handle Missing/Not Applicable Data
        # Replace common ABS missing value indicators with None across all columns
        missing_indicators = ['..', 'n.a.', 'n.f.d.', '-', 'np']
        for col_name in cleaned_df.columns:
             if cleaned_df[col_name].dtype == pl.Utf8: # Only apply string replace logic to string columns
                  for indicator in missing_indicators:
                       cleaned_df = cleaned_df.with_columns(
                            pl.when(pl.col(col_name) == indicator)
                            .then(None)
                            .otherwise(pl.col(col_name))
                            .alias(col_name)
                       )
        logger.debug(f"[{table_code}] Replaced missing value indicators {missing_indicators} with None.")

        # 4. Basic Type Conversion
        # Convert measure columns to numeric (integer), coercing errors
        type_conversions = []
        for col_name in measure_cols:
            if col_name in cleaned_df.columns:
                type_conversions.append(pl.col(col_name).cast(pl.Int64, strict=False))
            else:
                 logger.warning(f"[{table_code}] Expected measure column '{col_name}' not found for type conversion.")

        # Convert categorical columns to String (Utf8)
        if age_col in cleaned_df.columns and cleaned_df[age_col].dtype != pl.Utf8:
             type_conversions.append(pl.col(age_col).cast(pl.Utf8))
        if sex_col in cleaned_df.columns and cleaned_df[sex_col].dtype != pl.Utf8:
             type_conversions.append(pl.col(sex_col).cast(pl.Utf8))
        # Add other categorical columns if needed

        if type_conversions:
             cleaned_df = cleaned_df.with_columns(type_conversions)
             logger.debug(f"[{table_code}] Performed type conversions for measure and categorical columns.")

        logger.info(f"[{table_code}] Data cleaning finished. Shape after cleaning: {cleaned_df.shape}")

        # --- Data Transformation ---
        logger.info(f"[{table_code}] Starting data transformation...")
        transformed_df = cleaned_df

        # Identify potential ID columns and measure columns
        id_vars = [col for col in transformed_df.columns if col in [age_col, sex_col, 'region_id']] # Add other potential IDs like region_id
        measure_vars = [col for col in transformed_df.columns if col.endswith('_count')] # Assumes measures end with _count

        if not id_vars:
             logger.error(f"[{table_code}] No ID variables (age, sex, region_id) found for melting. Skipping transformation.")
             # Decide how to handle this - return None, return cleaned_df, or raise error?
             # For now, pass cleaned_df to avoid breaking the flow, but log error.
             transformed_df = cleaned_df
        elif not measure_vars:
             logger.error(f"[{table_code}] No measure variables ending in '_count' found for melting. Skipping transformation.")
             transformed_df = cleaned_df
        else:
            logger.debug(f"[{table_code}] Melting data with id_vars: {id_vars}, measure_vars: {measure_vars}")
            # 1. Reshape data from wide to long
            transformed_df = transformed_df.melt(
                id_vars=id_vars,
                value_vars=measure_vars,
                variable_name="assistance_provided_status",
                value_name="person_count"
            )
            logger.debug(f"[{table_code}] Data melted. Shape: {transformed_df.shape}")

            # 2. Standardize 'assistance_provided_status' values
            status_mapping = {
                "provided_assistance_count": "Provided Assistance",
                "no_assistance_provided_count": "No Assistance Provided",
                "assistance_not_stated_count": "Assistance Not Stated"
            }
            transformed_df = transformed_df.with_columns(
                pl.col("assistance_provided_status").replace(status_mapping).alias("assistance_provided_status")
            )
            logger.debug(f"[{table_code}] Standardized 'assistance_provided_status' values.")

            # 3. Standardize 'sex' column (Example: Title Case)
            if sex_col in transformed_df.columns:
                 transformed_df = transformed_df.with_columns(
                      pl.col(sex_col).str.to_titlecase().alias(sex_col)
                 )
                 logger.debug(f"[{table_code}] Standardized 'sex' column to title case.")

            # 4. Standardize 'age_group' column (Add specific logic if needed)
            # TODO: Implement specific age group standardization if required (e.g., '5-14 years', '65+')
            # Example: transformed_df = transformed_df.with_columns(standardize_age_group(pl.col(age_col)).alias(age_col))
            logger.debug(f"[{table_code}] TODO: Implement specific age group standardization if needed.")


            # 5. Column Selection (Select final columns for the fact table)
            # Ensure 'region_id' exists or use the correct geo identifier column name
            geo_id_col = 'region_id' # Assume this is the standard geo identifier after loading/cleaning
            if geo_id_col not in transformed_df.columns:
                 # Attempt to find another common geo code if region_id is missing
                 potential_geo_cols = ['SA1_CODE21', 'SA2_CODE21', 'SA3_CODE21', 'SA4_CODE21', 'LGA_CODE21', 'STE_CODE21']
                 found_geo_col = next((col for col in potential_geo_cols if col in transformed_df.columns), None)
                 if found_geo_col:
                      logger.warning(f"[{table_code}] '{geo_id_col}' not found, using '{found_geo_col}' as geographic identifier.")
                      geo_id_col = found_geo_col
                 else:
                      logger.error(f"[{table_code}] Cannot find a suitable geographic identifier column ({geo_id_col} or alternatives). Transformation might be incomplete.")
                      # Handle error - maybe skip selection or raise? For now, proceed without geo_id if not found.


            final_columns = [col for col in [geo_id_col, age_col, sex_col, "assistance_provided_status", "person_count"] if col in transformed_df.columns]

            if len(final_columns) < 5:
                 missing_final_cols = set([geo_id_col, age_col, sex_col, "assistance_provided_status", "person_count"]) - set(final_columns)
                 logger.warning(f"[{table_code}] Missing expected final columns after transformation: {missing_final_cols}. Selecting available columns: {final_columns}")


            transformed_df = transformed_df.select(final_columns)
            logger.info(f"[{table_code}] Data transformation finished. Selected columns: {final_columns}. Shape: {transformed_df.shape}")


        # --- Data Validation ---
        logger.info(f"[{table_code}] Starting data validation...")

        # Define expected schema and values based on transformation steps
        # Use the determined geo_id_col, age_col, sex_col from earlier in the function
        expected_columns = [col for col in [geo_id_col, age_col, sex_col, "assistance_provided_status", "person_count"] if col in transformed_df.columns]
        # If transformation failed to produce all columns, validation will likely fail below, but we proceed with available columns.
        if len(expected_columns) != 5:
             logger.warning(f"[{table_code}] Validation: Expected 5 columns, but only found {len(expected_columns)} present in DataFrame after transformation: {expected_columns}. Proceeding with validation on available columns.")

        expected_dtypes = {
            # Allow flexibility for geo_id type (String or Int)
            geo_id_col: (pl.Utf8, pl.Int64) if geo_id_col in transformed_df.columns else None,
            age_col: pl.Utf8 if age_col in transformed_df.columns else None,
            sex_col: pl.Utf8 if sex_col in transformed_df.columns else None,
            "assistance_provided_status": pl.Utf8 if "assistance_provided_status" in transformed_df.columns else None,
            "person_count": pl.Int64 if "person_count" in transformed_df.columns else None
        }
        # Filter out None values for columns not present
        expected_dtypes = {k: v for k, v in expected_dtypes.items() if v is not None}

        expected_sex_values = ["Male", "Female"]
        expected_assistance_values = ["Provided Assistance", "No Assistance Provided", "Assistance Not Stated"]
        # TODO: Define expected_age_group_values if standardization is implemented and add check

        validation_passed = True
        error_messages = []

        # 1. Check columns existence (already implicitly handled by checking expected_columns length and subsequent checks)
        logger.info(f"[{table_code}] Validation: Checking DataFrame with columns: {transformed_df.columns}")

        # 2. Check data types
        for col, expected_type_or_tuple in expected_dtypes.items():
            if col not in transformed_df.columns:
                message = f"[{table_code}] Validation FAIL: Expected column '{col}' not found for type check."
                logger.error(message)
                error_messages.append(message)
                validation_passed = False
                continue # Skip type check if column missing

            actual_type = transformed_df[col].dtype
            is_type_ok = False
            if isinstance(expected_type_or_tuple, tuple):
                is_type_ok = actual_type in expected_type_or_tuple
            else:
                is_type_ok = actual_type == expected_type_or_tuple

            if not is_type_ok:
                message = f"[{table_code}] Validation FAIL: Column '{col}' has wrong dtype. Expected: {expected_type_or_tuple}, Actual: {actual_type}"
                logger.error(message)
                error_messages.append(message)
                validation_passed = False

        if validation_passed: # Log success only if all types passed so far
             logger.info(f"[{table_code}] Validation PASS: All checked column data types are valid.")

        # 3. Null Value Checks (using validation utility)
        # Check all expected columns that are actually present for nulls
        columns_to_check_nulls = [col for col in expected_columns if col in transformed_df.columns]
        if columns_to_check_nulls:
            if not validation.validate_null_values(transformed_df, f"{table_code}_transformed", columns_to_check_nulls):
                # Error logged within validate_null_values
                error_messages.append(f"[{table_code}] Validation FAIL: Null values found in required columns (see logs above).")
                validation_passed = False
        else:
             logger.warning(f"[{table_code}] Validation: Skipping null checks as no expected columns were found to check.")

        # 4. Categorical Value Checks
        # Check Sex (if column exists)
        if sex_col in transformed_df.columns:
            invalid_sex = transformed_df.filter(
                ~pl.col(sex_col).is_in(expected_sex_values) & pl.col(sex_col).is_not_null() # Exclude nulls from check
            )[sex_col].unique().to_list()
            if invalid_sex:
                message = f"[{table_code}] Validation FAIL: Invalid values found in '{sex_col}': {invalid_sex}"
                logger.error(message)
                error_messages.append(message)
                validation_passed = False
            else:
                logger.info(f"[{table_code}] Validation PASS: Values in '{sex_col}' are within expected set: {expected_sex_values}")
        else:
             logger.warning(f"[{table_code}] Validation: Skipping Sex categorical check as column '{sex_col}' is missing.")

        # Check Assistance Status (if column exists)
        if "assistance_provided_status" in transformed_df.columns:
            invalid_assistance = transformed_df.filter(
                ~pl.col("assistance_provided_status").is_in(expected_assistance_values) & pl.col("assistance_provided_status").is_not_null() # Exclude nulls
            )["assistance_provided_status"].unique().to_list()
            if invalid_assistance:
                message = f"[{table_code}] Validation FAIL: Invalid values found in 'assistance_provided_status': {invalid_assistance}"
                logger.error(message)
                error_messages.append(message)
                validation_passed = False
            else:
                logger.info(f"[{table_code}] Validation PASS: Values in 'assistance_provided_status' are within expected set: {expected_assistance_values}")
        else:
             logger.warning(f"[{table_code}] Validation: Skipping Assistance Status categorical check as column 'assistance_provided_status' is missing.")

        # TODO: Add check for age_group if expected values are defined

        # 5. Numeric Range Checks (using validation utility)
        if "person_count" in transformed_df.columns:
            if not validation.validate_range_values(transformed_df, f"{table_code}_transformed", ["person_count"]):
                # Error logged within validate_range_values
                error_messages.append(f"[{table_code}] Validation FAIL: Negative values found in 'person_count' (see logs above).")
                validation_passed = False
        else:
             logger.warning(f"[{table_code}] Validation: Skipping Range check as column 'person_count' is missing.")

        # Final check and error handling
        if not validation_passed:
            # Combine error messages for a comprehensive exception
            full_error_message = f"[{table_code}] Data validation failed after transformation. Errors:\n" + "\n".join(error_messages)
            logger.error(full_error_message)
            raise ValueError(full_error_message)
        else:
            logger.info(f"[{table_code}] Data validation finished successfully.")
            validated_df = transformed_df # Assign if validation passed


        # --- Dimension Integration (using cleaned and potentially transformed/validated data) ---
        # Create demographic dimension based on G25 data.
        # This function will likely extract demographic columns and create/lookup keys.
        logger.info(f"[{table_code}] Creating demographic dimension from G25 data.")
        # Ensure the necessary columns (age_group, sex) exist before calling create_demographic_dimension
        required_dim_cols = [age_col, sex_col] # Add others if needed by the dimension function
        missing_dim_cols = [col for col in required_dim_cols if col not in validated_df.columns]

        if missing_dim_cols:
             logger.error(f"[{table_code}] Missing required columns for dimension integration: {missing_dim_cols}. Skipping dimension integration.")
             g25_final_df = validated_df # Return the processed data without dimension keys
        else:
             logger.info(f"[{table_code}] Creating demographic dimension from cleaned G25 data.")
             # Pass the cleaned/transformed/validated DataFrame
             g25_final_df = create_demographic_dimension(validated_df, age_col=age_col, sex_col=sex_col)
             logger.info(f"[{table_code}] Demographic dimension integration complete.")


        logger.info(f"[{table_code}] G25 data processing finished. Final shape: {g25_final_df.shape}")
        return g25_final_df

    except Exception as e:
        logger.error(f"[{table_code}] An unexpected error occurred during G25 data processing: {e}", exc_info=True)
        return None
