"""
ABS Census demographic data transformer for AHGD ETL pipeline.

This module transforms raw Australian Bureau of Statistics census demographic data
(Table G01 - Basic Demographics) into standardised CensusDemographics schema format.
Handles age group aggregation, geographic integration, and demographic calculations.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from ...utils.logging import get_logger
from ...utils.config import get_config, get_config_manager
from ...utils.interfaces import (
    ProcessingMetadata,
    ProcessingStatus,
    TransformationError,
)
from schemas.census_schema import CensusDemographics


class DemographicTransformer:
    """
    Transforms raw ABS Census demographic data to CensusDemographics schema.
    
    Handles G01 (Basic Demographics) table processing with:
    - Age group standardisation & aggregation
    - Geographic hierarchy integration  
    - Demographic ratio calculations
    - Missing value imputation
    - Schema enforcement & validation
    """
    
    def __init__(self):
        """
        Initialise the demographic transformer.
        
        Follows existing transformer pattern - no base class inheritance.
        Uses settings and configuration for flexible, maintainable operation.
        """
        # Core configuration - follow existing patterns
        self.config_manager = get_config_manager()
        self._logger_name = __name__
        
        # Column mappings for robust field mapping
        self.column_mappings = self._load_column_mappings()
        
        # Target schema configuration
        self.target_schema = self._load_target_schema()
        
        # Geographic integration settings
        self.geographic_hierarchy = get_config("transformers.census.geographic_hierarchy", True)
        
        # Processing configuration
        self.operations_config = self._load_operations_config()
        self.imputation_strategy = get_config("transformers.census.impute_missing", "geographic_median")
        
        # State management for processing
        self.demographic_sk_counter = 10000  # Start demographic surrogates at 10K
        self.processing_metadata: Optional[ProcessingMetadata] = None
        
        # Error handling configuration
        self.stop_on_error = get_config("system.stop_on_error", False)
        
    @property  
    def logger(self):
        """
        Get logger instance (creates new instance to avoid serialization issues).
        
        Returns:
            Logger: Thread-safe logger instance
        """
        return get_logger(self._logger_name)
        
    def _load_column_mappings(self) -> Dict[str, List[str]]:
        """
        Load column mappings from configuration.
        
        Maps raw census column names to standardised field names with
        priority-based fallback for ABS naming variations.
        
        Returns:
            Dict[str, List[str]]: Mapping of target fields to source column candidates
        """
        # Default column mappings for ABS Census G01 table
        # Priority order: most recent ABS format first
        default_mappings = {
            "geographic_id": ["SA2_CODE_2021", "SA2_MAIN21", "SA2_CODE", "sa2_code"],
            "geographic_name": ["SA2_NAME_2021", "SA2_NAME21", "SA2_NAME", "sa2_name"],
            "state_territory": ["STATE_CODE_2021", "STE_CODE21", "STATE_CODE", "state_code"],
            "total_population": ["Tot_P_P", "Total_Persons", "total_population"],
            "males": ["Male_P", "Males", "males"],
            "females": ["Female_P", "Females", "females"],
            
            # Age group mappings - male + female = total
            "age_0_4_male": ["Age_0_4_yr_M", "Age_0_4_M", "age_0_4_male"],
            "age_0_4_female": ["Age_0_4_yr_F", "Age_0_4_F", "age_0_4_female"],
            "age_5_9_male": ["Age_5_9_yr_M", "Age_5_9_M", "age_5_9_male"],
            "age_5_9_female": ["Age_5_9_yr_F", "Age_5_9_F", "age_5_9_female"],
            "age_10_14_male": ["Age_10_14_yr_M", "Age_10_14_M", "age_10_14_male"],
            "age_10_14_female": ["Age_10_14_yr_F", "Age_10_14_F", "age_10_14_female"],
            "age_15_19_male": ["Age_15_19_yr_M", "Age_15_19_M", "age_15_19_male"],
            "age_15_19_female": ["Age_15_19_yr_F", "Age_15_19_F", "age_15_19_female"],
            
            # Indigenous status
            "indigenous": ["Indigenous_P", "Aboriginal_Torres_Strait_Islander", "indigenous"],
            "non_indigenous": ["Non_Indigenous_P", "Non_Indigenous", "non_indigenous"],
            "indigenous_not_stated": ["Indigenous_NS_P", "Indigenous_Not_Stated", "indigenous_not_stated"],
            
            # Dwelling data
            "total_private_dwellings": ["Total_dwell_P", "Total_Private_Dwellings", "total_private_dwellings"],
            "occupied_private_dwellings": ["OPD_P", "Occupied_Private_Dwellings", "occupied_private_dwellings"],
            "unoccupied_private_dwellings": ["UPD_P", "Unoccupied_Private_Dwellings", "unoccupied_private_dwellings"],
            "total_families": ["Total_families", "Total_Families", "total_families"],
        }
        
        # Try to load from configuration, fall back to defaults
        config_mappings = get_config("transformers.census.column_mappings", {})
        return {**default_mappings, **config_mappings}
        
    def _load_target_schema(self) -> Dict[str, Any]:
        """
        Load target schema configuration.
        
        Returns:
            Dict[str, Any]: Target schema specification
        """
        return get_config("schemas.census_demographics", {})
        
    def _load_operations_config(self) -> Dict[str, Any]:
        """
        Load operations configuration from pipeline settings.
        
        Returns:
            Dict[str, Any]: Operations configuration including ratios and indicators
        """
        default_operations = {
            "age_group_system": "5_year_groups",
            "include_broad_groups": True,
            "ratios": ["dependency_ratio", "sex_ratio", "child_ratio", "elderly_ratio"],
            "indicators": ["population_density", "median_age_deviation", "diversity_index"]
        }
        
        return get_config("transformers.census.operations", default_operations)
        
    def _standardise_input_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw census columns to standardised field names with validation.
        
        Handles multiple column name variations and validates required fields
        are present. Applies priority-based column matching for robustness.
        
        Args:
            raw_df: Raw census data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with standardised column names
            
        Raises:
            TransformationError: If required columns are missing
        """
        self.logger.info(f"Standardising input data with {len(raw_df)} records")
        
        # Create mapping dictionary from available columns
        standardised_columns = {}
        
        for target_field, source_candidates in self.column_mappings.items():
            matched_column = self._find_matching_column(raw_df, source_candidates)
            if matched_column:
                standardised_columns[matched_column] = target_field
                self.logger.debug(f"Mapped {matched_column} -> {target_field}")
            else:
                self.logger.warning(f"No source column found for {target_field}")
        
        # Apply column mappings
        try:
            mapped_df = raw_df.rename(columns=standardised_columns)
        except Exception as e:
            raise TransformationError(f"Column mapping failed: {e}")
        
        # Validate required columns are present
        required_columns = ["geographic_id", "total_population", "males", "females"]
        missing_required = [col for col in required_columns if col not in mapped_df.columns]
        
        if missing_required:
            raise TransformationError(f"Missing required columns: {missing_required}")
        
        # Log successful mapping
        self.logger.info(f"Successfully mapped {len(standardised_columns)} columns")
        
        return mapped_df
        
    def _find_matching_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """
        Find the first matching column from a list of candidates.
        
        Args:
            df: DataFrame to search in
            candidates: List of potential column names in priority order
            
        Returns:
            Optional[str]: First matching column name, or None if no match
        """
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None
        
    def _integrate_geographic_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate with geographic dimension for hierarchy resolution.
        
        Links to existing geo_dimension table to resolve SA2 → SA3 → SA4 → STE hierarchy
        and handles geographic boundary validation.
        
        Args:
            df: DataFrame with standardised geographic_id column
            
        Returns:
            pd.DataFrame: DataFrame enriched with geographic hierarchy data
            
        Raises:
            TransformationError: If geographic integration fails
        """
        self.logger.info("Integrating geographic hierarchy data")
        
        try:
            # Load geographic dimension for lookups (mock implementation for now)
            geo_dim = self._load_geographic_dimension()
            
            # Join with geographic hierarchy on geographic_id
            enriched_df = df.merge(
                geo_dim,
                left_on="geographic_id", 
                right_on="geo_id",
                how="left",
                suffixes=('', '_geo')
            )
            
            # Handle unmatched geographic codes
            unmatched_mask = enriched_df["geo_sk"].isna()
            unmatched_count = unmatched_mask.sum()
            
            if unmatched_count > 0:
                self.logger.warning(f"Found {unmatched_count} unmatched geographic codes")
                enriched_df = self._handle_unmatched_geography(enriched_df, unmatched_mask)
            
            self.logger.info(f"Successfully integrated geographic hierarchy for {len(enriched_df)} records")
            return enriched_df
            
        except Exception as e:
            self.logger.error(f"Geographic integration failed: {e}")
            if self.stop_on_error:
                raise TransformationError(f"Geographic integration failed: {e}")
            else:
                self.logger.warning("Continuing without geographic enrichment")
                return df
                
    def _load_geographic_dimension(self) -> pd.DataFrame:
        """
        Load geographic dimension data for hierarchy resolution.
        
        Returns:
            pd.DataFrame: Geographic dimension with hierarchy information
        """
        # Mock implementation - in production this would load from actual geo dimension
        mock_geo_data = {
            'geo_id': ['101021001', '101021002', '201011001', '301011001'],
            'geo_sk': [1001, 1002, 2001, 3001],
            'sa3_code': ['10102', '10102', '20101', '30101'],
            'sa3_name': ['Sydney Inner City', 'Sydney Inner City', 'Melbourne Inner', 'Brisbane Inner'],
            'sa4_code': ['101', '101', '201', '301'],
            'sa4_name': ['Sydney - City and Inner South', 'Sydney - City and Inner South', 'Melbourne - Inner', 'Brisbane - Inner'],
            'state_code': ['1', '1', '2', '3'],
            'state_name': ['New South Wales', 'New South Wales', 'Victoria', 'Queensland'],
            'area_sq_km': [2.53, 1.82, 2.07, 3.18]
        }
        return pd.DataFrame(mock_geo_data)
        
    def _handle_unmatched_geography(self, df: pd.DataFrame, unmatched_mask: pd.Series) -> pd.DataFrame:
        """
        Handle records with unmatched geographic codes.
        
        Args:
            df: DataFrame with geographic integration results
            unmatched_mask: Boolean mask indicating unmatched records
            
        Returns:
            pd.DataFrame: DataFrame with unmatched records handled
        """
        # Set unknown geo_sk for unmatched records
        unknown_geo_sk = -99  # Standard unknown member key
        df.loc[unmatched_mask, 'geo_sk'] = unknown_geo_sk
        
        # Log unmatched geographic codes for investigation
        unmatched_codes = df.loc[unmatched_mask, 'geographic_id'].unique()
        self.logger.warning(f"Unmatched geographic codes: {list(unmatched_codes)}")
        
        return df
        
    def _standardise_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate age/sex data into standardised 5-year age groups.
        
        Converts raw male/female age columns into 18 standardised age groups
        (age_0_4 through age_85_plus) with validation and consistency checking.
        
        Args:
            df: DataFrame with raw age group columns
            
        Returns:
            pd.DataFrame: DataFrame with standardised age group columns
        """
        self.logger.info("Standardising age groups into 5-year cohorts")
        
        # Define age group aggregations (male + female = total)
        age_aggregations = {
            "age_0_4": ["age_0_4_male", "age_0_4_female"],
            "age_5_9": ["age_5_9_male", "age_5_9_female"],
            "age_10_14": ["age_10_14_male", "age_10_14_female"],
            "age_15_19": ["age_15_19_male", "age_15_19_female"],
            "age_20_24": ["age_20_24_male", "age_20_24_female"],
            "age_25_29": ["age_25_29_male", "age_25_29_female"],
            "age_30_34": ["age_30_34_male", "age_30_34_female"],
            "age_35_39": ["age_35_39_male", "age_35_39_female"],
            "age_40_44": ["age_40_44_male", "age_40_44_female"],
            "age_45_49": ["age_45_49_male", "age_45_49_female"],
            "age_50_54": ["age_50_54_male", "age_50_54_female"],
            "age_55_59": ["age_55_59_male", "age_55_59_female"],
            "age_60_64": ["age_60_64_male", "age_60_64_female"],
            "age_65_69": ["age_65_69_male", "age_65_69_female"],
            "age_70_74": ["age_70_74_male", "age_70_74_female"],
            "age_75_79": ["age_75_79_male", "age_75_79_female"],
            "age_80_84": ["age_80_84_male", "age_80_84_female"],
            "age_85_plus": ["age_85_plus_male", "age_85_plus_female"]
        }
        
        # Vectorized aggregation with null handling
        for target_age_group, source_columns in age_aggregations.items():
            # Find available columns (some may be missing in different data sources)
            available_cols = [col for col in source_columns if col in df.columns]
            
            if available_cols:
                # Sum available columns, filling nulls with 0
                df[target_age_group] = df[available_cols].fillna(0).sum(axis=1).astype(int)
            else:
                self.logger.warning(f"No source data for {target_age_group}")
                df[target_age_group] = 0
        
        # Validate age group consistency
        df = self._validate_age_group_totals(df)
        
        self.logger.info("Successfully standardised 18 age groups")
        return df
        
    def _validate_age_group_totals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate that age group totals are consistent with total population.
        
        Args:
            df: DataFrame with age group columns
            
        Returns:
            pd.DataFrame: Validated DataFrame
        """
        # Calculate sum of all age groups
        age_columns = [f"age_{i}_{i+4}" for i in range(0, 85, 5)] + ["age_85_plus"]
        age_columns = [col for col in age_columns if col in df.columns]
        
        if age_columns:
            df['age_total_calculated'] = df[age_columns].sum(axis=1)
            
            # Check for significant discrepancies (allowing for small rounding differences)
            discrepancy = abs(df['age_total_calculated'] - df['total_population'])
            significant_discrepancies = discrepancy > 5
            
            if significant_discrepancies.any():
                discrepancy_count = significant_discrepancies.sum()
                self.logger.warning(f"Found {discrepancy_count} records with age group discrepancies > 5")
                
            # Drop the temporary validation column
            df = df.drop('age_total_calculated', axis=1)
            
        return df
        
    def _process_indigenous_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process Indigenous status data with validation and consistency checking.
        
        Args:
            df: DataFrame with Indigenous status columns
            
        Returns:
            pd.DataFrame: DataFrame with processed Indigenous status data
        """
        self.logger.info("Processing Indigenous status data")
        
        # Ensure Indigenous status columns are integers
        indigenous_columns = ['indigenous', 'non_indigenous', 'indigenous_not_stated']
        
        for col in indigenous_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
            else:
                self.logger.warning(f"Missing Indigenous status column: {col}")
                df[col] = 0
                
        # Validate Indigenous status totals
        df['indigenous_total'] = df[indigenous_columns].sum(axis=1)
        
        # Check for consistency with total population
        indigenous_discrepancy = abs(df['indigenous_total'] - df['total_population'])
        significant_discrepancies = indigenous_discrepancy > 5
        
        if significant_discrepancies.any():
            discrepancy_count = significant_discrepancies.sum()
            self.logger.warning(f"Found {discrepancy_count} records with Indigenous status discrepancies > 5")
            
        # Drop temporary validation column
        df = df.drop('indigenous_total', axis=1)
        
        return df
        
    def _process_dwelling_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process dwelling count data with validation.
        
        Args:
            df: DataFrame with dwelling count columns
            
        Returns:
            pd.DataFrame: DataFrame with processed dwelling data
        """
        self.logger.info("Processing dwelling count data")
        
        # Ensure dwelling columns are integers
        dwelling_columns = [
            'total_private_dwellings', 
            'occupied_private_dwellings', 
            'unoccupied_private_dwellings',
            'total_families'
        ]
        
        for col in dwelling_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)
            else:
                self.logger.warning(f"Missing dwelling column: {col}")
                df[col] = 0
                
        # Validate dwelling occupancy consistency
        if 'total_private_dwellings' in df.columns:
            occupancy_total = df['occupied_private_dwellings'] + df['unoccupied_private_dwellings']
            occupancy_discrepancy = abs(occupancy_total - df['total_private_dwellings'])
            significant_discrepancies = occupancy_discrepancy > 2
            
            if significant_discrepancies.any():
                discrepancy_count = significant_discrepancies.sum()
                self.logger.warning(f"Found {discrepancy_count} records with dwelling occupancy discrepancies > 2")
                
        return df
        
    def transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main transformation method - orchestrates the complete transformation pipeline.
        
        Executes the full demographic transformation pipeline following the data flow design:
        1. Column mapping and standardisation
        2. Geographic hierarchy integration
        3. Age group standardisation and aggregation
        4. Indigenous status processing
        5. Dwelling data processing
        
        Args:
            raw_df: Raw census data DataFrame
            
        Returns:
            pd.DataFrame: Transformed data conforming to CensusDemographics schema
            
        Raises:
            TransformationError: If transformation fails
        """
        try:
            # Initialize processing metadata
            self.processing_metadata = ProcessingMetadata(
                operation_id=f"demographic_transform_{int(time.time())}",
                operation_type="demographic_transformation",
                status=ProcessingStatus.RUNNING,
                start_time=datetime.now()
            )
            
            self.logger.info(f"Starting demographic transformation pipeline with {len(raw_df)} records")
            
            # Stage 1: Standardise input data and column mappings
            self.logger.info("Stage 1: Column standardisation and mapping")
            transformed_df = self._standardise_input_data(raw_df)
            
            # Stage 2: Geographic hierarchy integration (if enabled)
            if self.geographic_hierarchy:
                self.logger.info("Stage 2: Geographic hierarchy integration")
                transformed_df = self._integrate_geographic_hierarchy(transformed_df)
            else:
                self.logger.info("Stage 2: Skipping geographic integration (disabled)")
            
            # Stage 3: Age group standardisation and aggregation
            self.logger.info("Stage 3: Age group standardisation")
            transformed_df = self._standardise_age_groups(transformed_df)
            
            # Stage 4: Indigenous status processing
            self.logger.info("Stage 4: Indigenous status processing")
            transformed_df = self._process_indigenous_status(transformed_df)
            
            # Stage 5: Dwelling data processing
            self.logger.info("Stage 5: Dwelling data processing")
            transformed_df = self._process_dwelling_data(transformed_df)
            
            # Stage 6: Calculate demographic ratios (Advanced Features)
            self.logger.info("Stage 6: Calculating demographic ratios")
            transformed_df = self._calculate_demographic_ratios(transformed_df)
            
            # Stage 7: Derive demographic indicators (Advanced Features)
            self.logger.info("Stage 7: Deriving demographic indicators")
            transformed_df = self._derive_demographic_indicators(transformed_df)
            
            # Stage 8: Impute missing values (Data Hardening)
            if self.imputation_strategy == "geographic_median":
                self.logger.info("Stage 8: Imputing missing values using geographic median strategy")
                transformed_df = self._impute_missing_values(transformed_df)
            else:
                self.logger.info("Stage 8: Skipping missing value imputation (disabled)")
            
            # Stage 9: Enforce final schema compliance
            self.logger.info("Stage 9: Enforcing final schema compliance")
            transformed_df = self._enforce_schema(transformed_df)
            
            # Note: ETL metadata is now handled within _enforce_schema to ensure proper ordering
            
            # Update processing metadata
            self.processing_metadata.records_processed = len(transformed_df)
            self.processing_metadata.mark_completed()
            
            self.logger.info(f"Demographic transformation completed successfully: {len(transformed_df)} records processed in {self.processing_metadata.duration_seconds:.2f} seconds")
            
            return transformed_df
            
        except Exception as e:
            if self.processing_metadata:
                self.processing_metadata.mark_failed(str(e))
                
            self.logger.error(f"Demographic transformation failed: {e}")
            
            if self.stop_on_error:
                raise TransformationError(f"Demographic transformation failed: {e}")
            else:
                self.logger.warning("Continuing with partial transformation due to stop_on_error=False")
                return pd.DataFrame()  # Return empty DataFrame as fallback
    
    def _calculate_demographic_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate demographic ratios using vectorized operations.
        
        Implements Stage 4 of the transformation pipeline:
        - Dependency ratio: (young + elderly) / working age * 100
        - Sex ratio: males per 100 females
        - Child ratio: children (0-14) per 100 working age
        - Elderly ratio: elderly (65+) per 100 working age
        
        Args:
            df: DataFrame with age group columns
            
        Returns:
            DataFrame with added ratio columns
        """
        self.logger.debug("Calculating demographic ratios")
        
        # Define age group categories for efficient calculation
        working_age_groups = [
            'age_15_19', 'age_20_24', 'age_25_29', 'age_30_34', 
            'age_35_39', 'age_40_44', 'age_45_49', 'age_50_54', 
            'age_55_59', 'age_60_64'
        ]
        
        young_dependent_groups = ['age_0_4', 'age_5_9', 'age_10_14']
        
        elderly_dependent_groups = [
            'age_65_69', 'age_70_74', 'age_75_79', 
            'age_80_84', 'age_85_plus'
        ]
        
        # Calculate population segments using vectorized operations
        # Use .get() to handle potentially missing columns gracefully
        working_age_pop = df[[col for col in working_age_groups if col in df.columns]].sum(axis=1)
        young_dependents = df[[col for col in young_dependent_groups if col in df.columns]].sum(axis=1)
        elderly_dependents = df[[col for col in elderly_dependent_groups if col in df.columns]].sum(axis=1)
        total_dependents = young_dependents + elderly_dependents
        
        # Calculate dependency ratio (per 100 working age population)
        # Use np.where to handle division by zero
        df['dependency_ratio'] = np.where(
            working_age_pop > 0,
            (total_dependents / working_age_pop) * 100,
            0.0
        )
        
        # Calculate sex ratio (males per 100 females)
        df['sex_ratio'] = np.where(
            df['females'] > 0,
            (df['males'] / df['females']) * 100,
            0.0
        )
        
        # Calculate child ratio (children 0-14 per 100 working age)
        df['child_ratio'] = np.where(
            working_age_pop > 0,
            (young_dependents / working_age_pop) * 100,
            0.0
        )
        
        # Calculate elderly ratio (65+ per 100 working age)
        df['elderly_ratio'] = np.where(
            working_age_pop > 0,
            (elderly_dependents / working_age_pop) * 100,
            0.0
        )
        
        # Round ratios to 2 decimal places for readability
        ratio_columns = ['dependency_ratio', 'sex_ratio', 'child_ratio', 'elderly_ratio']
        for col in ratio_columns:
            df[col] = df[col].round(2)
        
        self.logger.debug(f"Calculated {len(ratio_columns)} demographic ratios")
        
        return df
    
    def _derive_demographic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive advanced demographic indicators.
        
        Implements Stage 5 of the transformation pipeline:
        - Population density: population per square kilometer
        - Median age deviation: deviation from national median age
        - Diversity index: Simpson's diversity index for Indigenous status
        
        Args:
            df: DataFrame with demographic and geographic data
            
        Returns:
            DataFrame with added indicator columns
        """
        self.logger.debug("Deriving demographic indicators")
        
        # Calculate population density (people per sq km)
        # Handle division by zero and missing area data
        if 'area_sq_km' in df.columns:
            df['population_density'] = np.where(
                df['area_sq_km'] > 0,
                df['total_population'] / df['area_sq_km'],
                0.0
            )
        else:
            self.logger.warning("area_sq_km column not found, setting population_density to 0")
            df['population_density'] = 0.0
        
        # Calculate median age deviation
        # This requires estimating median age from age group distributions
        df['median_age_deviation'] = self._calculate_median_age_deviation(df)
        
        # Calculate diversity index (Simpson's diversity index)
        # D = 1 - Σ(n_i/N)^2 where n_i is count in each group and N is total
        df['diversity_index'] = self._calculate_diversity_index(df)
        
        # Round indicators to appropriate precision
        df['population_density'] = df['population_density'].round(2)
        df['median_age_deviation'] = df['median_age_deviation'].round(2)
        df['diversity_index'] = df['diversity_index'].round(4)
        
        self.logger.debug("Calculated demographic indicators")
        
        return df
    
    def _calculate_median_age_deviation(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate deviation from median age using age group midpoints.
        
        Uses weighted average of age group midpoints to estimate median age,
        then calculates deviation from national median (assumed 38.4 years for Australia).
        """
        # Define age group midpoints for estimation
        age_midpoints = {
            'age_0_4': 2.5, 'age_5_9': 7.5, 'age_10_14': 12.5,
            'age_15_19': 17.5, 'age_20_24': 22.5, 'age_25_29': 27.5,
            'age_30_34': 32.5, 'age_35_39': 37.5, 'age_40_44': 42.5,
            'age_45_49': 47.5, 'age_50_54': 52.5, 'age_55_59': 57.5,
            'age_60_64': 62.5, 'age_65_69': 67.5, 'age_70_74': 72.5,
            'age_75_79': 77.5, 'age_80_84': 82.5, 'age_85_plus': 90.0
        }
        
        # Calculate weighted average age for each area
        total_weighted_age = pd.Series(0.0, index=df.index)
        total_population = pd.Series(0.0, index=df.index)
        
        for age_group, midpoint in age_midpoints.items():
            if age_group in df.columns:
                total_weighted_age += df[age_group] * midpoint
                total_population += df[age_group]
        
        # Calculate estimated median age
        estimated_median_age = np.where(
            total_population > 0,
            total_weighted_age / total_population,
            0.0
        )
        
        # Calculate deviation from national median (38.4 years for Australia)
        national_median_age = 38.4
        median_age_deviation = estimated_median_age - national_median_age
        
        return pd.Series(median_age_deviation, index=df.index)
    
    def _calculate_diversity_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Simpson's diversity index for Indigenous status categories.
        
        D = 1 - Σ(p_i)^2 where p_i is the proportion of each group.
        Higher values indicate more diversity (max = 0.667 for 3 equal groups).
        """
        # Get Indigenous status categories
        categories = ['indigenous', 'non_indigenous', 'indigenous_not_stated']
        
        # Calculate total for each row (handling missing columns)
        existing_categories = [cat for cat in categories if cat in df.columns]
        
        if not existing_categories:
            self.logger.warning("No Indigenous status columns found for diversity calculation")
            return pd.Series(0.0, index=df.index)
        
        # Calculate total population for Indigenous status
        total = df[existing_categories].sum(axis=1)
        
        # Calculate Simpson's diversity index
        diversity_index = pd.Series(1.0, index=df.index)
        
        for category in existing_categories:
            proportion = np.where(total > 0, df[category] / total, 0.0)
            diversity_index -= proportion ** 2
        
        # Handle edge cases
        diversity_index = np.where(total > 0, diversity_index, 0.0)
        
        return pd.Series(diversity_index, index=df.index)
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using hierarchical geographic median strategy.
        
        Implements geographic median imputation with fallback hierarchy:
        1. SA3-level median (Statistical Area Level 3)
        2. SA4-level median (Statistical Area Level 4)
        3. State-level median
        4. Global median (entire dataset)
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with imputed values
        """
        self.logger.debug("Starting hierarchical geographic median imputation")
        
        # Identify numeric columns for imputation (exclude identifiers)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove identifier columns from imputation
        exclude_columns = ['demographic_sk', 'geo_sk', 'census_year']
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Geographic hierarchy columns
        geo_hierarchy = ['sa3_code', 'sa4_code', 'state_territory']
        
        # Track imputation statistics
        imputation_stats = {}
        
        for col in numeric_columns:
            if df[col].isna().any():
                nulls_before = df[col].isna().sum()
                self.logger.debug(f"Imputing {nulls_before} missing values in column '{col}'")
                
                # Create a copy of the column for imputation
                imputed_col = df[col].copy()
                
                # Level 1: SA3-level median imputation
                if 'sa3_code' in df.columns:
                    sa3_medians = df.groupby('sa3_code')[col].median()
                    for sa3 in df['sa3_code'].unique():
                        mask = (df['sa3_code'] == sa3) & df[col].isna()
                        if mask.any() and not pd.isna(sa3_medians.get(sa3)):
                            imputed_col.loc[mask] = sa3_medians[sa3]
                
                # Level 2: SA4-level median imputation for remaining nulls
                if 'sa4_code' in df.columns and imputed_col.isna().any():
                    sa4_medians = df.groupby('sa4_code')[col].median()
                    for sa4 in df['sa4_code'].unique():
                        mask = (df['sa4_code'] == sa4) & imputed_col.isna()
                        if mask.any() and not pd.isna(sa4_medians.get(sa4)):
                            imputed_col.loc[mask] = sa4_medians[sa4]
                
                # Level 3: State-level median imputation for remaining nulls
                if 'state_territory' in df.columns and imputed_col.isna().any():
                    state_medians = df.groupby('state_territory')[col].median()
                    for state in df['state_territory'].unique():
                        mask = (df['state_territory'] == state) & imputed_col.isna()
                        if mask.any() and not pd.isna(state_medians.get(state)):
                            imputed_col.loc[mask] = state_medians[state]
                
                # Level 4: Global median imputation for any remaining nulls
                if imputed_col.isna().any():
                    global_median = df[col].median()
                    if not pd.isna(global_median):
                        imputed_col.fillna(global_median, inplace=True)
                    else:
                        # If all values are null, use 0 as fallback
                        imputed_col.fillna(0, inplace=True)
                
                # Update the column with imputed values
                df[col] = imputed_col
                
                # Track imputation statistics
                nulls_after = df[col].isna().sum()
                imputation_stats[col] = {
                    'imputed': nulls_before - nulls_after,
                    'remaining': nulls_after
                }
        
        # Log imputation summary
        total_imputed = sum(stats['imputed'] for stats in imputation_stats.values())
        if total_imputed > 0:
            self.logger.info(f"Imputed {total_imputed} missing values across {len(imputation_stats)} columns")
            for col, stats in imputation_stats.items():
                if stats['imputed'] > 0:
                    self.logger.debug(f"  {col}: imputed {stats['imputed']} values")
        
        return df
    
    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce final schema compliance for data types and column ordering.
        
        This is the final transformation step that ensures:
        1. All columns have correct data types
        2. Missing columns are added with appropriate defaults
        3. Columns are ordered according to CensusDemographics schema
        4. Extra columns not in schema are removed
        
        Args:
            df: DataFrame to enforce schema on
            
        Returns:
            Schema-compliant DataFrame
        """
        self.logger.debug("Enforcing final schema compliance")
        
        # Define the complete schema with data types
        schema_definition = {
            # Key identifiers (integers)
            'demographic_sk': 'int64',
            'geo_sk': 'int64',
            
            # Geographic identifiers (strings)
            'geographic_id': 'object',
            'geographic_name': 'object',
            'state_territory': 'object',
            'sa3_code': 'object',
            'sa3_name': 'object',
            'sa4_code': 'object', 
            'sa4_name': 'object',
            'gcc_code': 'object',
            'gcc_name': 'object',
            
            # Core demographics (integers)
            'total_population': 'int64',
            'males': 'int64',
            'females': 'int64',
            
            # Age groups (all integers)
            'age_0_4': 'int64',
            'age_5_9': 'int64',
            'age_10_14': 'int64',
            'age_15_19': 'int64',
            'age_20_24': 'int64',
            'age_25_29': 'int64',
            'age_30_34': 'int64',
            'age_35_39': 'int64',
            'age_40_44': 'int64',
            'age_45_49': 'int64',
            'age_50_54': 'int64',
            'age_55_59': 'int64',
            'age_60_64': 'int64',
            'age_65_69': 'int64',
            'age_70_74': 'int64',
            'age_75_79': 'int64',
            'age_80_84': 'int64',
            'age_85_plus': 'int64',
            
            # Indigenous status (integers)
            'indigenous': 'int64',
            'non_indigenous': 'int64',
            'indigenous_not_stated': 'int64',
            
            # Dwelling data (integers)
            'total_private_dwellings': 'int64',
            'occupied_private_dwellings': 'int64',
            'unoccupied_private_dwellings': 'int64',
            'total_families': 'int64',
            
            # Ratios (floats)
            'dependency_ratio': 'float64',
            'sex_ratio': 'float64',
            'child_ratio': 'float64',
            'elderly_ratio': 'float64',
            
            # Indicators (floats)
            'population_density': 'float64',
            'median_age_deviation': 'float64',
            'diversity_index': 'float64',
            
            # Metadata
            'census_year': 'int64',
            'table_code': 'object',
            'table_name': 'object',
            'etl_processed_at': 'datetime64[ns]'
        }
        
        # Create a new DataFrame with proper schema
        result_df = pd.DataFrame(index=df.index)
        
        # Process each column according to schema
        for col_name, dtype in schema_definition.items():
            if col_name in df.columns:
                # Column exists - enforce data type
                try:
                    if dtype == 'int64':
                        # Convert to numeric first, then to int, handling errors
                        result_df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0).astype('int64')
                    elif dtype == 'float64':
                        result_df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype('float64')
                    elif dtype == 'object':
                        result_df[col_name] = df[col_name].astype('object')
                    elif dtype == 'datetime64[ns]':
                        result_df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                except Exception as e:
                    self.logger.warning(f"Error converting column {col_name} to {dtype}: {e}")
                    # Use appropriate default based on type
                    if dtype == 'int64':
                        result_df[col_name] = 0
                    elif dtype == 'float64':
                        result_df[col_name] = 0.0
                    elif dtype == 'object':
                        result_df[col_name] = ''
                    else:
                        result_df[col_name] = pd.NaT
            else:
                # Column missing - add with appropriate default
                self.logger.debug(f"Adding missing column '{col_name}' with default value")
                
                if col_name == 'demographic_sk':
                    # Generate surrogate keys starting from counter
                    result_df[col_name] = range(self.demographic_sk_counter, 
                                               self.demographic_sk_counter + len(df))
                    self.demographic_sk_counter += len(df)
                elif col_name == 'geo_sk':
                    result_df[col_name] = -99  # Unknown geographic location
                elif col_name == 'census_year':
                    result_df[col_name] = 2021
                elif col_name == 'table_code':
                    result_df[col_name] = 'G01'
                elif col_name == 'table_name':
                    result_df[col_name] = 'Basic Demographic Profile'
                elif col_name == 'etl_processed_at':
                    result_df[col_name] = datetime.now()
                elif dtype == 'int64':
                    result_df[col_name] = 0
                elif dtype == 'float64':
                    result_df[col_name] = 0.0
                elif dtype == 'object':
                    result_df[col_name] = ''
        
        # Log schema enforcement summary
        added_columns = set(result_df.columns) - set(df.columns)
        removed_columns = set(df.columns) - set(result_df.columns)
        
        if added_columns:
            self.logger.info(f"Added {len(added_columns)} missing columns: {sorted(added_columns)}")
        if removed_columns:
            self.logger.info(f"Removed {len(removed_columns)} extra columns: {sorted(removed_columns)}")
        
        self.logger.debug(f"Schema enforcement complete: {len(result_df.columns)} columns")
        
        return result_df