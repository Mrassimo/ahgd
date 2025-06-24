"""
ABS Census housing data transformer for AHGD ETL pipeline.

This module transforms raw Australian Bureau of Statistics census housing data
(Tables G31-G42 - Housing characteristics) into standardised CensusHousing schema format.
Handles dwelling structure, tenure type, housing costs, and housing characteristics.
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
from schemas.census_schema import CensusHousing


class HousingTransformer:
    """
    Transforms raw ABS Census housing data to CensusHousing schema.
    
    Handles G31-G42 (Housing characteristics) table processing with:
    - Dwelling structure standardisation
    - Tenure type classification  
    - Housing cost processing
    - Internet and vehicle data processing
    - Schema enforcement & validation
    """
    
    def __init__(self):
        """
        Initialise the housing transformer.
        
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
        self.imputation_strategy = get_config("transformers.census.impute_missing", "category_mode")
        
        # State management for processing
        self.housing_sk_counter = 20000  # Start housing surrogates at 20K
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
        # Default column mappings for ABS Census G31-G42 tables
        # Priority order: most recent ABS format first
        default_mappings = {
            # Geographic identification
            "geographic_id": ["SA2_CODE_2021", "SA2_MAIN21", "SA2_CODE", "sa2_code"],
            "geographic_name": ["SA2_NAME_2021", "SA2_NAME21", "SA2_NAME", "sa2_name"],
            "state_territory": ["STATE_CODE_2021", "STE_CODE21", "STATE_CODE", "state_code"],
            
            # Dwelling structure (Table G28)
            "separate_house": ["Separate_house", "O_SeparateHouse_Dwgs", "separate_house_dwellings"],
            "semi_detached": ["Semi_detached_row_terrace", "O_SemiDetached_Dwgs", "semi_detached_dwellings"],
            "flat_apartment": ["Flat_unit_apartment", "O_FlatUnit_Dwgs", "flat_apartment_dwellings"],
            "other_dwelling": ["Other_dwelling", "O_Other_Dwgs", "other_dwelling_type"],
            "dwelling_structure_not_stated": ["Dwelling_structure_not_stated", "O_DwellingStructure_NS", "structure_not_stated"],
            
            # Tenure type (Table G25)
            "owned_outright": ["Owned_outright", "O_OR_Owned_Outright_H", "owned_outright_dwellings"],
            "owned_with_mortgage": ["Owned_with_mortgage", "O_OR_Owned_Mortgage_H", "owned_with_mortgage_dwellings"],
            "rented": ["Rented", "O_OR_Rented_H", "rented_dwellings"],
            "other_tenure": ["Other_tenure_type", "O_OR_Other_H", "other_tenure_dwellings"],
            "tenure_not_stated": ["Tenure_not_stated", "O_OR_TenureType_NS", "tenure_type_not_stated"],
            
            # Landlord type for rented dwellings
            "state_territory_housing": ["State_territory_housing_authority", "O_Rented_ST_HA", "state_housing_authority"],
            "private_landlord": ["Private_landlord", "O_Rented_Private", "private_landlord_rented"],
            "real_estate_agent": ["Real_estate_agent", "O_Rented_RE_Agent", "real_estate_agent_rented"],
            "other_landlord": ["Other_landlord_type", "O_Rented_Other", "other_landlord_type"],
            "landlord_not_stated": ["Landlord_type_not_stated", "O_Rented_Landlord_NS", "landlord_type_not_stated"],
            
            # Number of bedrooms (Table G26)
            "no_bedrooms": ["No_bedrooms", "O_Bedrooms_0_H", "zero_bedrooms"],
            "one_bedroom": ["One_bedroom", "O_Bedrooms_1_H", "one_bedroom_dwellings"],
            "two_bedrooms": ["Two_bedrooms", "O_Bedrooms_2_H", "two_bedroom_dwellings"],
            "three_bedrooms": ["Three_bedrooms", "O_Bedrooms_3_H", "three_bedroom_dwellings"],
            "four_bedrooms": ["Four_bedrooms", "O_Bedrooms_4_H", "four_bedroom_dwellings"],
            "five_plus_bedrooms": ["Five_or_more_bedrooms", "O_Bedrooms_5plus_H", "five_plus_bedroom_dwellings"],
            "bedrooms_not_stated": ["Bedrooms_not_stated", "O_Bedrooms_NS_H", "bedrooms_number_not_stated"],
            
            # Internet connection (Table G34)
            "internet_connection": ["Internet_connected", "O_Internet_Y_H", "internet_connection_dwellings"],
            "no_internet": ["No_internet_connection", "O_Internet_N_H", "no_internet_connection"],
            "internet_not_stated": ["Internet_not_stated", "O_Internet_NS_H", "internet_connection_not_stated"],
            
            # Motor vehicles (Table G37)
            "no_motor_vehicles": ["No_vehicles", "O_Veh_0_H", "no_motor_vehicles_dwellings"],
            "one_motor_vehicle": ["One_vehicle", "O_Veh_1_H", "one_motor_vehicle"],
            "two_motor_vehicles": ["Two_vehicles", "O_Veh_2_H", "two_motor_vehicles"],
            "three_plus_vehicles": ["Three_or_more_vehicles", "O_Veh_3plus_H", "three_plus_motor_vehicles"],
            "vehicles_not_stated": ["Vehicles_not_stated", "O_Veh_NS_H", "motor_vehicles_not_stated"],
            
            # Housing costs (Tables G31, G33)
            "median_mortgage_monthly": ["Median_mortgage_monthly", "Median_mort_repay_M", "median_monthly_mortgage"],
            "median_rent_weekly": ["Median_rent_weekly", "Median_rent_W", "median_weekly_rent"],
        }
        
        # Load custom mappings from configuration if available
        config_mappings = get_config("transformers.census.column_mappings", {})
        if config_mappings:
            # Merge with defaults, config takes precedence
            for field, mapping_list in config_mappings.items():
                if field in default_mappings:
                    # Prepend config mappings to defaults for higher priority
                    default_mappings[field] = mapping_list + default_mappings[field]
                else:
                    default_mappings[field] = mapping_list
        
        return default_mappings
    
    def _load_target_schema(self) -> Dict[str, Any]:
        """
        Load target schema configuration for CensusHousing.
        
        Returns:
            Dict[str, Any]: Schema configuration for validation and enforcement
        """
        schema_config = get_config("schemas.census_housing", {})
        
        # Default schema definition if not in config
        if not schema_config:
            schema_config = {
                'housing_sk': 'int64',
                'geo_sk': 'int64',
                'geographic_id': 'object',
                'geographic_level': 'object',
                'geographic_name': 'object',
                'state_territory': 'object',
                'census_year': 'int64',
                
                # Dwelling structure
                'separate_house': 'int64',
                'semi_detached': 'int64', 
                'flat_apartment': 'int64',
                'other_dwelling': 'int64',
                'dwelling_structure_not_stated': 'int64',
                
                # Tenure type
                'owned_outright': 'int64',
                'owned_with_mortgage': 'int64',
                'rented': 'int64',
                'other_tenure': 'int64',
                'tenure_not_stated': 'int64',
                
                # Landlord type
                'state_territory_housing': 'int64',
                'private_landlord': 'int64',
                'real_estate_agent': 'int64',
                'other_landlord': 'int64',
                'landlord_not_stated': 'int64',
                
                # Bedrooms
                'no_bedrooms': 'int64',
                'one_bedroom': 'int64',
                'two_bedrooms': 'int64',
                'three_bedrooms': 'int64',
                'four_bedrooms': 'int64',
                'five_plus_bedrooms': 'int64',
                'bedrooms_not_stated': 'int64',
                
                # Connectivity and transport
                'internet_connection': 'int64',
                'no_internet': 'int64',
                'internet_not_stated': 'int64',
                'no_motor_vehicles': 'int64',
                'one_motor_vehicle': 'int64',
                'two_motor_vehicles': 'int64',
                'three_plus_vehicles': 'int64',
                'vehicles_not_stated': 'int64',
                
                # Housing costs
                'median_mortgage_monthly': 'Int64',  # Nullable integer
                'median_rent_weekly': 'Int64',       # Nullable integer
                
                # ETL metadata
                'processed_timestamp': 'datetime64[ns]',
                'table_code': 'object',
                'table_name': 'object'
            }
        
        return schema_config
    
    def _load_operations_config(self) -> Dict[str, Any]:
        """
        Load operations configuration for housing processing.
        
        Returns:
            Dict[str, Any]: Configuration for housing-specific operations
        """
        operations_config = get_config("transformers.census.operations", {})
        
        # Default operations configuration
        if not operations_config:
            operations_config = {
                "dwelling_structure_system": "abs_standard",
                "tenure_classification": "abs_standard", 
                "include_landlord_types": True,
                "bedroom_categories": "standard_groups",
                "housing_ratios": ["home_ownership_rate", "rental_rate"],
                "housing_indicators": ["dwelling_diversity_index", "internet_penetration_rate"]
            }
        
        return operations_config
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main transformation method for housing data.
        
        Orchestrates the complete housing data transformation pipeline
        following the 9-stage process established in DemographicTransformer.
        
        Args:
            df (pd.DataFrame): Raw housing census data
            
        Returns:
            pd.DataFrame: Transformed data conforming to CensusHousing schema
            
        Raises:
            TransformationError: If transformation fails and stop_on_error is True
        """
        try:
            start_time = time.time()
            
            # Initialize processing metadata
            self.processing_metadata = ProcessingMetadata(
                operation_id=f"housing_transform_{int(time.time())}",
                operation_type="housing_transformation",
                status=ProcessingStatus.RUNNING,
                start_time=datetime.now()
            )
            
            self.logger.info("Starting housing data transformation", 
                           extra={"component": "HousingTransformer", "input_rows": len(df)})
            
            # Stage 1: Column standardisation and mapping
            self.logger.debug("Stage 1: Column standardisation and mapping")
            df = self._standardise_input_data(df)
            
            # Stage 2: Geographic hierarchy integration (if enabled)
            if self.geographic_hierarchy:
                self.logger.debug("Stage 2: Geographic hierarchy integration")
                df = self._integrate_geographic_hierarchy(df)
            
            # Stage 3: Dwelling structure processing
            self.logger.debug("Stage 3: Dwelling structure processing")
            df = self._process_dwelling_structure(df)
            
            # Stage 4: Tenure type processing
            self.logger.debug("Stage 4: Tenure type processing")
            df = self._process_tenure_type(df)
            
            # Stage 5: Internet connection processing
            self.logger.debug("Stage 5: Internet connection processing")
            df = self._process_internet_connection(df)
            
            # Stage 6: Vehicle data processing
            self.logger.debug("Stage 6: Vehicle data processing")
            df = self._process_vehicle_data(df)
            
            # Stage 7: Mortgage and rent payment processing
            self.logger.debug("Stage 7: Mortgage and rent payment processing")
            df = self._process_mortgage_and_rent_payments(df)
            
            # Stage 8: Missing value imputation
            self.logger.debug("Stage 8: Missing value imputation")
            df = self._impute_missing_values(df)
            
            # Stage 9: Schema enforcement and validation
            self.logger.debug("Stage 9: Schema enforcement and validation")
            df = self._enforce_schema(df)
            
            # Finalise processing metadata
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update records processed and mark completed
            self.processing_metadata.records_processed = len(df)
            self.processing_metadata.mark_completed()
            
            self.logger.info("Housing data transformation completed successfully",
                           extra={
                               "component": "HousingTransformer",
                               "input_rows": len(df),
                               "output_rows": len(df),
                               "processing_time_seconds": processing_time
                           })
            
            return df
            
        except Exception as e:
            # Update processing metadata
            if self.processing_metadata:
                self.processing_metadata.mark_failed(str(e))
            
            error_msg = str(e)
            self.logger.error(f"Housing transformation failed: {error_msg}", 
                            extra={"component": "HousingTransformer", "error": error_msg})
            
            if self.stop_on_error:
                raise TransformationError(f"Housing transformation failed: {error_msg}")
            else:
                # Log error and return original data
                self.logger.warning("Continuing with original data due to stop_on_error=False")
                return df
    
    def _standardise_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise input column names using configured mappings.
        
        Args:
            df (pd.DataFrame): Raw housing data with ABS column names
            
        Returns:
            pd.DataFrame: Data with standardised column names
        """
        standardised_df = df.copy()
        
        # Track mapping success for logging
        successful_mappings = 0
        failed_mappings = []
        
        # Apply column mappings
        for target_field, source_candidates in self.column_mappings.items():
            mapped_column = self._find_matching_column(standardised_df.columns, source_candidates)
            
            if mapped_column:
                if mapped_column != target_field:
                    # Rename the column to standardised name
                    standardised_df = standardised_df.rename(columns={mapped_column: target_field})
                    self.logger.debug(f"Mapped column: {mapped_column} -> {target_field}")
                successful_mappings += 1
            else:
                failed_mappings.append(target_field)
                # Add column with default value if it's a required field
                if target_field in self._get_required_fields():
                    standardised_df[target_field] = 0
                    self.logger.warning(f"Required field {target_field} not found, added with default value 0")
        
        # Log mapping summary
        self.logger.info(f"Column mapping completed: {successful_mappings} successful, {len(failed_mappings)} failed")
        
        if failed_mappings:
            self.logger.warning(f"Failed to map columns: {failed_mappings}")
        
        # Validate that we have essential columns for housing transformation
        essential_fields = ['geographic_id']
        missing_essential = [field for field in essential_fields if field not in standardised_df.columns]
        
        if missing_essential:
            raise TransformationError(f"Essential fields missing after mapping: {missing_essential}")
        
        return standardised_df
    
    def _find_matching_column(self, available_columns: List[str], candidates: List[str]) -> Optional[str]:
        """
        Find the first matching column from candidates in available columns.
        
        Args:
            available_columns: Columns available in the DataFrame
            candidates: List of candidate column names in priority order
            
        Returns:
            Optional[str]: First matching column name, or None if no match
        """
        for candidate in candidates:
            if candidate in available_columns:
                return candidate
        return None
    
    def _get_required_fields(self) -> List[str]:
        """
        Get list of required fields that must be present after mapping.
        
        Returns:
            List[str]: List of required field names
        """
        return [
            'geographic_id', 'geographic_name', 'state_territory',
            'separate_house', 'semi_detached', 'flat_apartment', 'other_dwelling',
            'owned_outright', 'owned_with_mortgage', 'rented', 'other_tenure'
        ]
    
    def _integrate_geographic_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate geographic hierarchy information (SA2->SA3->SA4->STE).
        
        Args:
            df (pd.DataFrame): Housing data with geographic identifiers
            
        Returns:
            pd.DataFrame: Data with integrated geographic hierarchy
        """
        enriched_df = df.copy()
        
        # Add geographic hierarchy columns for imputation fallback
        # These would typically be joined from a geographic lookup table
        # For now, we'll derive them from SA2 codes using ABS patterns
        
        if 'geographic_id' in enriched_df.columns:
            # Extract SA3 code (first 5 digits of SA2 code)
            enriched_df['sa3_code'] = enriched_df['geographic_id'].astype(str).str[:5]
            
            # Extract SA4 code (first 3 digits of SA2 code) 
            enriched_df['sa4_code'] = enriched_df['geographic_id'].astype(str).str[:3]
            
            # Extract state code (first digit of SA2 code)
            enriched_df['state_code'] = enriched_df['geographic_id'].astype(str).str[:1]
            
            # Set geographic level
            enriched_df['geographic_level'] = 'SA2'
            
            self.logger.debug("Geographic hierarchy integration completed")
        else:
            self.logger.warning("Geographic ID not available for hierarchy integration")
            # Add placeholder columns to prevent downstream errors
            enriched_df['sa3_code'] = 'UNKNOWN'
            enriched_df['sa4_code'] = 'UNKNOWN'
            enriched_df['state_code'] = 'UNKNOWN'
            enriched_df['geographic_level'] = 'UNKNOWN'
        
        return enriched_df
    
    def _process_dwelling_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process dwelling structure data (separate house, semi-detached, apartment, etc.).
        
        Args:
            df (pd.DataFrame): Housing data with dwelling structure columns
            
        Returns:
            pd.DataFrame: Data with processed dwelling structure information
        """
        processed_df = df.copy()
        
        # Define dwelling structure columns
        structure_columns = [
            'separate_house', 'semi_detached', 'flat_apartment', 'other_dwelling',
            'dwelling_structure_not_stated'
        ]
        
        # Ensure all structure columns exist with default values
        for column in structure_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0
                self.logger.debug(f"Added missing dwelling structure column: {column}")
        
        # Convert to numeric and handle any non-numeric values
        for column in structure_columns:
            processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce').fillna(0).astype(int)
        
        # Calculate total dwellings for validation
        processed_df['total_dwellings'] = processed_df[structure_columns].sum(axis=1)
        
        # Calculate dwelling structure percentages for analysis
        total_with_known = processed_df['total_dwellings'] - processed_df.get('dwelling_structure_not_stated', 0)
        
        for column in ['separate_house', 'semi_detached', 'flat_apartment', 'other_dwelling']:
            pct_column = f'{column}_pct'
            processed_df[pct_column] = np.where(
                total_with_known > 0,
                (processed_df[column] / total_with_known) * 100,
                0.0
            )
        
        # Validate dwelling structure data
        self._validate_dwelling_structure(processed_df)
        
        self.logger.debug("Dwelling structure processing completed")
        return processed_df
    
    def _validate_dwelling_structure(self, df: pd.DataFrame) -> None:
        """
        Validate dwelling structure data for consistency.
        
        Args:
            df (pd.DataFrame): DataFrame with dwelling structure data
        """
        # Check for negative values
        structure_columns = ['separate_house', 'semi_detached', 'flat_apartment', 'other_dwelling']
        for column in structure_columns:
            if column in df.columns:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    self.logger.warning(f"Found {negative_count} negative values in {column}")
        
        # Check for unreasonably high separate house percentages (>98%)
        if 'separate_house_pct' in df.columns:
            high_separate = (df['separate_house_pct'] > 98).sum()
            if high_separate > 0:
                self.logger.info(f"Found {high_separate} areas with very high separate house percentage (>98%)")
        
        # Check for areas with no dwellings
        if 'total_dwellings' in df.columns:
            no_dwellings = (df['total_dwellings'] == 0).sum()
            if no_dwellings > 0:
                self.logger.warning(f"Found {no_dwellings} areas with no dwelling data")
    
    def _process_tenure_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process tenure type data (owned outright, mortgage, rented, other).
        
        Args:
            df (pd.DataFrame): Housing data with tenure type columns
            
        Returns:
            pd.DataFrame: Data with processed tenure type information
        """
        processed_df = df.copy()
        
        # Define tenure type columns
        tenure_columns = [
            'owned_outright', 'owned_with_mortgage', 'rented', 'other_tenure',
            'tenure_not_stated'
        ]
        
        # Define landlord type columns (for rented dwellings)
        landlord_columns = [
            'state_territory_housing', 'private_landlord', 'real_estate_agent',
            'other_landlord', 'landlord_not_stated'
        ]
        
        # Ensure all tenure columns exist with default values
        for column in tenure_columns + landlord_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0
                self.logger.debug(f"Added missing tenure column: {column}")
        
        # Convert to numeric and handle any non-numeric values
        for column in tenure_columns + landlord_columns:
            processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce').fillna(0).astype(int)
        
        # Calculate total occupied dwellings for validation
        processed_df['total_occupied_dwellings'] = processed_df[tenure_columns].sum(axis=1)
        
        # Calculate ownership statistics
        processed_df['total_owned'] = processed_df['owned_outright'] + processed_df['owned_with_mortgage']
        
        # Calculate tenure percentages
        total_with_known_tenure = processed_df['total_occupied_dwellings'] - processed_df.get('tenure_not_stated', 0)
        
        for column in ['owned_outright', 'owned_with_mortgage', 'rented', 'other_tenure']:
            pct_column = f'{column}_pct'
            processed_df[pct_column] = np.where(
                total_with_known_tenure > 0,
                (processed_df[column] / total_with_known_tenure) * 100,
                0.0
            )
        
        # Calculate home ownership rate
        processed_df['home_ownership_rate'] = np.where(
            total_with_known_tenure > 0,
            (processed_df['total_owned'] / total_with_known_tenure) * 100,
            0.0
        )
        
        # Calculate rental market share
        processed_df['rental_rate'] = np.where(
            total_with_known_tenure > 0,
            (processed_df['rented'] / total_with_known_tenure) * 100,
            0.0
        )
        
        # Process landlord type data if rental data exists
        if processed_df['rented'].sum() > 0:
            processed_df = self._process_landlord_types(processed_df)
        
        # Validate tenure type data
        self._validate_tenure_type(processed_df)
        
        self.logger.debug("Tenure type processing completed")
        return processed_df
    
    def _process_landlord_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process landlord type data for rented dwellings.
        
        Args:
            df (pd.DataFrame): Housing data with landlord type columns
            
        Returns:
            pd.DataFrame: Data with processed landlord type information
        """
        processed_df = df.copy()
        
        landlord_columns = ['state_territory_housing', 'private_landlord', 'real_estate_agent', 'other_landlord']
        
        # Calculate total landlord classifications
        processed_df['total_landlord_classified'] = processed_df[landlord_columns].sum(axis=1)
        
        # Calculate landlord type percentages (as percentage of rented dwellings)
        for column in landlord_columns:
            pct_column = f'{column}_pct'
            processed_df[pct_column] = np.where(
                processed_df['rented'] > 0,
                (processed_df[column] / processed_df['rented']) * 100,
                0.0
            )
        
        return processed_df
    
    def _validate_tenure_type(self, df: pd.DataFrame) -> None:
        """
        Validate tenure type data for consistency.
        
        Args:
            df (pd.DataFrame): DataFrame with tenure type data
        """
        # Check for negative values
        tenure_columns = ['owned_outright', 'owned_with_mortgage', 'rented', 'other_tenure']
        for column in tenure_columns:
            if column in df.columns:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    self.logger.warning(f"Found {negative_count} negative values in {column}")
        
        # Check for unusual home ownership rates
        if 'home_ownership_rate' in df.columns:
            very_low_ownership = (df['home_ownership_rate'] < 20).sum()
            very_high_ownership = (df['home_ownership_rate'] > 95).sum()
            
            if very_low_ownership > 0:
                self.logger.info(f"Found {very_low_ownership} areas with very low home ownership (<20%)")
            if very_high_ownership > 0:
                self.logger.info(f"Found {very_high_ownership} areas with very high home ownership (>95%)")
        
        # Check rental vs landlord consistency
        if all(col in df.columns for col in ['rented', 'total_landlord_classified']):
            inconsistent = abs(df['rented'] - df['total_landlord_classified']) > 5
            inconsistent_count = inconsistent.sum()
            if inconsistent_count > 0:
                self.logger.warning(f"Found {inconsistent_count} areas with rental/landlord data inconsistency")
    
    def _process_internet_connection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process internet connection data.
        
        Args:
            df (pd.DataFrame): Housing data with internet connection columns
            
        Returns:
            pd.DataFrame: Data with processed internet connection information
        """
        processed_df = df.copy()
        
        # Define internet connection columns
        internet_columns = ['internet_connection', 'no_internet', 'internet_not_stated']
        
        # Ensure all internet columns exist with default values
        for column in internet_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0
                self.logger.debug(f"Added missing internet column: {column}")
        
        # Convert to numeric and handle any non-numeric values
        for column in internet_columns:
            processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce').fillna(0).astype(int)
        
        # Calculate total dwellings with internet data
        processed_df['total_internet_data'] = processed_df[internet_columns].sum(axis=1)
        
        # Calculate internet penetration rate
        total_with_known_internet = processed_df['total_internet_data'] - processed_df.get('internet_not_stated', 0)
        
        processed_df['internet_penetration_rate'] = np.where(
            total_with_known_internet > 0,
            (processed_df['internet_connection'] / total_with_known_internet) * 100,
            0.0
        )
        
        # Calculate percentages for analysis
        for column in ['internet_connection', 'no_internet']:
            pct_column = f'{column}_pct'
            processed_df[pct_column] = np.where(
                total_with_known_internet > 0,
                (processed_df[column] / total_with_known_internet) * 100,
                0.0
            )
        
        # Validate internet connection data
        self._validate_internet_connection(processed_df)
        
        self.logger.debug("Internet connection processing completed")
        return processed_df
    
    def _validate_internet_connection(self, df: pd.DataFrame) -> None:
        """
        Validate internet connection data for consistency.
        
        Args:
            df (pd.DataFrame): DataFrame with internet connection data
        """
        # Check for negative values
        internet_columns = ['internet_connection', 'no_internet']
        for column in internet_columns:
            if column in df.columns:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    self.logger.warning(f"Found {negative_count} negative values in {column}")
        
        # Check for unusual internet penetration rates
        if 'internet_penetration_rate' in df.columns:
            very_low_internet = (df['internet_penetration_rate'] < 50).sum()
            very_high_internet = (df['internet_penetration_rate'] > 99).sum()
            
            if very_low_internet > 0:
                self.logger.info(f"Found {very_low_internet} areas with low internet penetration (<50%)")
            if very_high_internet > 0:
                self.logger.info(f"Found {very_high_internet} areas with very high internet penetration (>99%)")
    
    def _process_vehicle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process motor vehicle data.
        
        Args:
            df (pd.DataFrame): Housing data with vehicle columns
            
        Returns:
            pd.DataFrame: Data with processed vehicle information
        """
        processed_df = df.copy()
        
        # Define vehicle columns
        vehicle_columns = [
            'no_motor_vehicles', 'one_motor_vehicle', 'two_motor_vehicles',
            'three_plus_vehicles', 'vehicles_not_stated'
        ]
        
        # Ensure all vehicle columns exist with default values
        for column in vehicle_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0
                self.logger.debug(f"Added missing vehicle column: {column}")
        
        # Convert to numeric and handle any non-numeric values
        for column in vehicle_columns:
            processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce').fillna(0).astype(int)
        
        # Calculate total dwellings with vehicle data
        processed_df['total_vehicle_data'] = processed_df[vehicle_columns].sum(axis=1)
        
        # Calculate vehicle statistics
        total_with_known_vehicles = processed_df['total_vehicle_data'] - processed_df.get('vehicles_not_stated', 0)
        
        # Calculate average vehicles per dwelling (estimated)
        processed_df['estimated_avg_vehicles'] = np.where(
            total_with_known_vehicles > 0,
            (processed_df['one_motor_vehicle'] * 1 + 
             processed_df['two_motor_vehicles'] * 2 + 
             processed_df['three_plus_vehicles'] * 3.5) / total_with_known_vehicles,  # Assume 3.5 for 3+
            0.0
        )
        
        # Calculate no vehicle percentage
        processed_df['no_vehicle_rate'] = np.where(
            total_with_known_vehicles > 0,
            (processed_df['no_motor_vehicles'] / total_with_known_vehicles) * 100,
            0.0
        )
        
        # Calculate multi-vehicle percentage (2+ vehicles)
        multi_vehicle = processed_df['two_motor_vehicles'] + processed_df['three_plus_vehicles']
        processed_df['multi_vehicle_rate'] = np.where(
            total_with_known_vehicles > 0,
            (multi_vehicle / total_with_known_vehicles) * 100,
            0.0
        )
        
        # Calculate percentages for each vehicle category
        for column in ['no_motor_vehicles', 'one_motor_vehicle', 'two_motor_vehicles', 'three_plus_vehicles']:
            pct_column = f'{column}_pct'
            processed_df[pct_column] = np.where(
                total_with_known_vehicles > 0,
                (processed_df[column] / total_with_known_vehicles) * 100,
                0.0
            )
        
        # Validate vehicle data
        self._validate_vehicle_data(processed_df)
        
        self.logger.debug("Vehicle data processing completed")
        return processed_df
    
    def _validate_vehicle_data(self, df: pd.DataFrame) -> None:
        """
        Validate vehicle data for consistency.
        
        Args:
            df (pd.DataFrame): DataFrame with vehicle data
        """
        # Check for negative values
        vehicle_columns = ['no_motor_vehicles', 'one_motor_vehicle', 'two_motor_vehicles', 'three_plus_vehicles']
        for column in vehicle_columns:
            if column in df.columns:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    self.logger.warning(f"Found {negative_count} negative values in {column}")
        
        # Check for unusual vehicle ownership patterns
        if 'no_vehicle_rate' in df.columns:
            very_high_no_vehicle = (df['no_vehicle_rate'] > 70).sum()
            very_low_no_vehicle = (df['no_vehicle_rate'] < 5).sum()
            
            if very_high_no_vehicle > 0:
                self.logger.info(f"Found {very_high_no_vehicle} areas with very high no-vehicle rate (>70%)")
            if very_low_no_vehicle > 0:
                self.logger.info(f"Found {very_low_no_vehicle} areas with very low no-vehicle rate (<5%)")
        
        # Check for unusual average vehicles per dwelling
        if 'estimated_avg_vehicles' in df.columns:
            very_high_avg = (df['estimated_avg_vehicles'] > 3.0).sum()
            if very_high_avg > 0:
                self.logger.info(f"Found {very_high_avg} areas with high average vehicles per dwelling (>3.0)")
    
    def _process_mortgage_and_rent_payments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process mortgage and rent payment data.
        
        Args:
            df (pd.DataFrame): Housing data with payment columns
            
        Returns:
            pd.DataFrame: Data with processed payment information
        """
        processed_df = df.copy()
        
        # Define payment columns
        payment_columns = ['median_mortgage_monthly', 'median_rent_weekly']
        
        # Process each payment column
        for column in payment_columns:
            if column in processed_df.columns:
                # Convert to numeric, handling any non-numeric values
                processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce')
                
                # Handle special cases (e.g., ABS uses specific codes for no data)
                # Convert any negative values or obvious data errors to NaN
                processed_df[column] = processed_df[column].where(processed_df[column] > 0)
            else:
                # Add column with NaN if not present
                processed_df[column] = np.nan
                self.logger.debug(f"Added missing payment column: {column}")
        
        # Calculate housing affordability metrics
        if all(col in processed_df.columns for col in ['median_rent_weekly', 'median_mortgage_monthly']):
            # Convert rent to monthly for comparison (weekly * 4.33)
            processed_df['median_rent_monthly'] = processed_df['median_rent_weekly'] * 4.33
            
            # Calculate rent to mortgage ratio
            processed_df['rent_to_mortgage_ratio'] = np.where(
                (processed_df['median_mortgage_monthly'].notna()) & 
                (processed_df['median_mortgage_monthly'] > 0),
                processed_df['median_rent_monthly'] / processed_df['median_mortgage_monthly'],
                np.nan
            )
            
            # Housing affordability indicator (higher rent suggests lower affordability for purchase)
            processed_df['housing_affordability_indicator'] = np.where(
                processed_df['rent_to_mortgage_ratio'].notna(),
                np.where(processed_df['rent_to_mortgage_ratio'] > 0.7, 'Low Purchase Affordability',
                        np.where(processed_df['rent_to_mortgage_ratio'] > 0.5, 'Moderate Purchase Affordability',
                                'High Purchase Affordability')),
                'Unknown'
            )
        
        # Calculate payment ranges for analysis
        self._calculate_payment_ranges(processed_df)
        
        # Validate payment data
        self._validate_payment_data(processed_df)
        
        self.logger.debug("Mortgage and rent payment processing completed")
        return processed_df
    
    def _calculate_payment_ranges(self, df: pd.DataFrame) -> None:
        """
        Calculate payment range categories for analysis.
        
        Args:
            df (pd.DataFrame): DataFrame with payment data
        """
        # Mortgage payment ranges (monthly)
        if 'median_mortgage_monthly' in df.columns:
            df['mortgage_range'] = pd.cut(
                df['median_mortgage_monthly'],
                bins=[0, 1500, 2500, 3500, 5000, float('inf')],
                labels=['Under $1,500', '$1,500-$2,500', '$2,500-$3,500', '$3,500-$5,000', 'Over $5,000'],
                include_lowest=True
            )
        
        # Rent payment ranges (weekly)
        if 'median_rent_weekly' in df.columns:
            df['rent_range'] = pd.cut(
                df['median_rent_weekly'],
                bins=[0, 300, 500, 700, 1000, float('inf')],
                labels=['Under $300', '$300-$500', '$500-$700', '$700-$1,000', 'Over $1,000'],
                include_lowest=True
            )
    
    def _validate_payment_data(self, df: pd.DataFrame) -> None:
        """
        Validate payment data for consistency and reasonableness.
        
        Args:
            df (pd.DataFrame): DataFrame with payment data
        """
        # Check for unreasonably high mortgage payments
        if 'median_mortgage_monthly' in df.columns:
            very_high_mortgage = (df['median_mortgage_monthly'] > 8000).sum()
            very_low_mortgage = ((df['median_mortgage_monthly'] > 0) & (df['median_mortgage_monthly'] < 500)).sum()
            
            if very_high_mortgage > 0:
                self.logger.info(f"Found {very_high_mortgage} areas with very high mortgage payments (>$8,000)")
            if very_low_mortgage > 0:
                self.logger.warning(f"Found {very_low_mortgage} areas with unusually low mortgage payments (<$500)")
        
        # Check for unreasonably high rent
        if 'median_rent_weekly' in df.columns:
            very_high_rent = (df['median_rent_weekly'] > 1500).sum()
            very_low_rent = ((df['median_rent_weekly'] > 0) & (df['median_rent_weekly'] < 100)).sum()
            
            if very_high_rent > 0:
                self.logger.info(f"Found {very_high_rent} areas with very high weekly rent (>$1,500)")
            if very_low_rent > 0:
                self.logger.warning(f"Found {very_low_rent} areas with unusually low weekly rent (<$100)")
        
        # Check rent to mortgage ratio consistency
        if 'rent_to_mortgage_ratio' in df.columns:
            ratio_data = df['rent_to_mortgage_ratio'].dropna()
            if len(ratio_data) > 0:
                unusual_ratio = ((ratio_data > 1.5) | (ratio_data < 0.2)).sum()
                if unusual_ratio > 0:
                    self.logger.info(f"Found {unusual_ratio} areas with unusual rent-to-mortgage ratios")
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using category mode imputation strategy.
        
        Args:
            df (pd.DataFrame): Housing data with potential missing values
            
        Returns:
            pd.DataFrame: Data with imputed missing values
        """
        imputed_df = df.copy()
        
        # Define categorical housing columns for mode imputation
        categorical_columns = [
            'separate_house', 'semi_detached', 'flat_apartment', 'other_dwelling',
            'owned_outright', 'owned_with_mortgage', 'rented', 'other_tenure',
            'no_bedrooms', 'one_bedroom', 'two_bedrooms', 'three_bedrooms',
            'four_bedrooms', 'five_plus_bedrooms',
            'internet_connection', 'no_internet',
            'no_motor_vehicles', 'one_motor_vehicle', 'two_motor_vehicles', 'three_plus_vehicles'
        ]
        
        # Define continuous columns for median imputation
        continuous_columns = ['median_mortgage_monthly', 'median_rent_weekly']
        
        imputation_count = 0
        
        # Impute categorical columns using hierarchical geographic mode
        for column in categorical_columns:
            if column in imputed_df.columns:
                original_missing = imputed_df[column].isna().sum()
                if original_missing > 0:
                    # SA3-level mode imputation
                    if 'sa3_code' in imputed_df.columns:
                        imputed_df[column] = imputed_df.groupby('sa3_code')[column].transform(
                            lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else 0)
                        )
                    
                    # SA4-level mode fallback
                    if 'sa4_code' in imputed_df.columns:
                        imputed_df[column] = imputed_df.groupby('sa4_code')[column].transform(
                            lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else 0)
                        )
                    
                    # State-level mode fallback
                    if 'state_territory' in imputed_df.columns:
                        imputed_df[column] = imputed_df.groupby('state_territory')[column].transform(
                            lambda x: x.fillna(x.mode().iloc[0] if len(x.mode()) > 0 else 0)
                        )
                    
                    # Global mode fallback
                    global_mode = imputed_df[column].mode()
                    if len(global_mode) > 0:
                        imputed_df[column] = imputed_df[column].fillna(global_mode.iloc[0])
                    else:
                        imputed_df[column] = imputed_df[column].fillna(0)
                    
                    final_missing = imputed_df[column].isna().sum()
                    imputed_count = original_missing - final_missing
                    
                    if imputed_count > 0:
                        self.logger.debug(f"Imputed {imputed_count} values for {column} using mode strategy")
        
        # Impute continuous columns using hierarchical geographic median
        for column in continuous_columns:
            if column in imputed_df.columns:
                original_missing = imputed_df[column].isna().sum()
                if original_missing > 0:
                    # SA3-level median imputation
                    if 'sa3_code' in imputed_df.columns:
                        imputed_df[column] = imputed_df.groupby('sa3_code')[column].transform(
                            lambda x: x.fillna(x.median())
                        )
                    
                    # SA4-level median fallback
                    if 'sa4_code' in imputed_df.columns:
                        imputed_df[column] = imputed_df.groupby('sa4_code')[column].transform(
                            lambda x: x.fillna(x.median())
                        )
                    
                    # State-level median fallback
                    if 'state_territory' in imputed_df.columns:
                        imputed_df[column] = imputed_df.groupby('state_territory')[column].transform(
                            lambda x: x.fillna(x.median())
                        )
                    
                    # Global median fallback
                    global_median = imputed_df[column].median()
                    if pd.notna(global_median):
                        imputed_df[column] = imputed_df[column].fillna(global_median)
                    
                    final_missing = imputed_df[column].isna().sum()
                    imputed_count = original_missing - final_missing
                    
                    if imputed_count > 0:
                        self.logger.debug(f"Imputed {imputed_count} values for {column} using median strategy")
        
        self.logger.info("Missing value imputation completed using hierarchical geographic strategy")
        return imputed_df
    
    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce final schema compliance for CensusHousing.
        
        Args:
            df (pd.DataFrame): Housing data ready for schema enforcement
            
        Returns:
            pd.DataFrame: Data conforming to CensusHousing schema
        """
        schema_df = df.copy()
        
        # Generate surrogate keys
        schema_df['housing_sk'] = range(self.housing_sk_counter, 
                                      self.housing_sk_counter + len(schema_df))
        self.housing_sk_counter += len(schema_df)
        
        # Generate geo surrogate key (placeholder - would typically join to dim_geography)
        schema_df['geo_sk'] = schema_df['geographic_id'].astype(str).str.replace(r'\D', '', regex=True).astype('int64', errors='ignore')
        
        # Add required schema columns with defaults if missing
        required_columns = {
            'census_year': 2021,
            'table_code': 'G31-G42',
            'table_name': 'Housing Characteristics',
            'processed_timestamp': datetime.now()
        }
        
        for column, default_value in required_columns.items():
            if column not in schema_df.columns:
                schema_df[column] = default_value
        
        # Ensure all target schema columns exist
        for column, data_type in self.target_schema.items():
            if column not in schema_df.columns:
                if data_type in ['int64', 'Int64']:
                    schema_df[column] = 0 if data_type == 'int64' else pd.NA
                elif data_type == 'object':
                    schema_df[column] = ''
                elif data_type == 'float64':
                    schema_df[column] = 0.0
                elif data_type == 'datetime64[ns]':
                    schema_df[column] = datetime.now()
                else:
                    schema_df[column] = None
                
                self.logger.debug(f"Added missing schema column: {column}")
        
        # Convert data types according to schema
        for column, target_type in self.target_schema.items():
            if column in schema_df.columns:
                try:
                    if target_type == 'int64':
                        schema_df[column] = pd.to_numeric(schema_df[column], errors='coerce').fillna(0).astype('int64')
                    elif target_type == 'Int64':  # Nullable integer
                        schema_df[column] = pd.to_numeric(schema_df[column], errors='coerce').astype('Int64')
                    elif target_type == 'float64':
                        schema_df[column] = pd.to_numeric(schema_df[column], errors='coerce').astype('float64')
                    elif target_type == 'object':
                        schema_df[column] = schema_df[column].astype('object').fillna('')
                    elif target_type == 'datetime64[ns]':
                        schema_df[column] = pd.to_datetime(schema_df[column], errors='coerce')
                        if schema_df[column].isna().any():
                            schema_df[column] = schema_df[column].fillna(datetime.now())
                except Exception as e:
                    self.logger.warning(f"Failed to convert {column} to {target_type}: {e}")
        
        # Order columns according to schema (keep only schema columns)
        schema_columns = list(self.target_schema.keys())
        existing_schema_columns = [col for col in schema_columns if col in schema_df.columns]
        schema_df = schema_df[existing_schema_columns]
        
        # Final validation
        self._validate_final_schema(schema_df)
        
        self.logger.info(f"Schema enforcement completed: {len(schema_df)} records, {len(schema_df.columns)} columns")
        return schema_df
    
    def _validate_final_schema(self, df: pd.DataFrame) -> None:
        """
        Validate final schema compliance.
        
        Args:
            df (pd.DataFrame): DataFrame with enforced schema
        """
        validation_errors = []
        
        # Check for required columns
        required_columns = ['housing_sk', 'geo_sk', 'geographic_id']
        for column in required_columns:
            if column not in df.columns:
                validation_errors.append(f"Required column missing: {column}")
            elif df[column].isna().any():
                na_count = df[column].isna().sum()
                validation_errors.append(f"Required column {column} has {na_count} null values")
        
        # Check data types
        type_errors = 0
        for column, expected_type in self.target_schema.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if expected_type == 'int64' and actual_type != 'int64':
                    type_errors += 1
                elif expected_type == 'object' and actual_type != 'object':
                    type_errors += 1
        
        if type_errors > 0:
            validation_errors.append(f"{type_errors} columns have incorrect data types")
        
        # Check for negative values in count columns
        count_columns = [col for col in df.columns if any(keyword in col for keyword in 
                        ['house', 'owned', 'rented', 'bedroom', 'internet', 'vehicle'])]
        
        for column in count_columns:
            if column in df.columns and df[column].dtype in ['int64', 'Int64']:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    validation_errors.append(f"Column {column} has {negative_count} negative values")
        
        # Log validation results
        if validation_errors:
            for error in validation_errors:
                self.logger.warning(f"Schema validation warning: {error}")
        else:
            self.logger.info("Final schema validation passed")
        
        # Log summary statistics
        self.logger.info(f"Final dataset summary: {len(df)} rows, {len(df.columns)} columns")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            self.logger.debug(f"Numeric columns: {len(numeric_columns)}")
        object_columns = df.select_dtypes(include=['object']).columns
        if len(object_columns) > 0:
            self.logger.debug(f"Text columns: {len(object_columns)}")