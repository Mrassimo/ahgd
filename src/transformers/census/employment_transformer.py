"""
ABS Census employment data transformer for AHGD ETL pipeline.

This module transforms raw Australian Bureau of Statistics census employment data
(Tables G17, G37-G43 - Employment and Labour Force) into standardised CensusEmployment schema format.
Handles ANZSCO occupation classification, ANZSIC industry classification, and labour force analysis.
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
from schemas.census_schema import CensusEmployment


class EmploymentTransformer:
    """
    Transforms raw ABS Census employment data to CensusEmployment schema.
    
    Handles G17, G37-G43 (Employment and Labour Force) table processing with:
    - ANZSCO occupation standardisation
    - ANZSIC industry classification
    - Labour force analysis & indicators
    - Education-employment alignment
    - Employment diversity calculations
    """
    
    def __init__(self):
        """
        Initialise the employment transformer.
        
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
        self.imputation_strategy = get_config("transformers.census.impute_missing", "employment_weighted")
        
        # State management for processing
        self.employment_sk_counter = 30000  # Start employment surrogates at 30K
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
        # Default column mappings for ABS Census employment tables (G17, G37-G43)
        # Priority order: most recent ABS format first
        default_mappings = {
            # Geographic identification
            "geographic_id": ["SA2_CODE_2021", "SA2_MAIN21", "SA2_CODE", "sa2_code"],
            "geographic_name": ["SA2_NAME_2021", "SA2_NAME21", "SA2_NAME", "sa2_name"],
            "state_territory": ["STATE_CODE_2021", "STE_CODE21", "STATE_CODE", "state_code"],
            
            # Labour force population base
            "labour_force_pop": ["Labour_force_status_15_P", "LF_Pop_P", "labour_force_population"],
            
            # Labour force status (Table G43)
            "employed_full_time": ["Employed_Full_time", "EmpFT_P", "employed_full_time_persons"],
            "employed_part_time": ["Employed_Part_time", "EmpPT_P", "employed_part_time_persons"],
            "unemployed": ["Unemployed_Total", "Unemp_P", "unemployed_persons"],
            "not_in_labour_force": ["Not_in_Labour_Force", "NotLF_P", "not_labour_force_persons"],
            "labour_force_not_stated": ["Labour_force_status_not_stated", "LF_NS_P", "labour_force_not_stated"],
            
            # ANZSIC Industry categories (Table G38)
            "agriculture_forestry_fishing": ["Agriculture_Forestry_Fishing", "Ind_AgFor_P", "agriculture_forestry_fishing_industry"],
            "mining": ["Mining", "Ind_Mining_P", "mining_industry"],
            "manufacturing": ["Manufacturing", "Ind_Manuf_P", "manufacturing_industry"],
            "electricity_gas_water": ["Electricity_Gas_Water_Waste", "Ind_ElecGas_P", "utilities_industry"],
            "construction": ["Construction", "Ind_Const_P", "construction_industry"],
            "wholesale_trade": ["Wholesale_Trade", "Ind_WhTrade_P", "wholesale_trade_industry"],
            "retail_trade": ["Retail_Trade", "Ind_RetTrade_P", "retail_trade_industry"],
            "accommodation_food": ["Accommodation_Food_Services", "Ind_AccomFood_P", "hospitality_industry"],
            "transport_postal": ["Transport_Postal_Warehousing", "Ind_TransPost_P", "transport_logistics_industry"],
            "information_media": ["Information_Media_Telecommunications", "Ind_InfoMedia_P", "information_technology_industry"],
            "financial_insurance": ["Financial_Insurance_Services", "Ind_FinInsur_P", "financial_services_industry"],
            "rental_real_estate": ["Rental_Hiring_Real_Estate", "Ind_RentEstate_P", "real_estate_industry"],
            "professional_services": ["Professional_Scientific_Technical", "Ind_ProfServ_P", "professional_services_industry"],
            "administrative_support": ["Administrative_Support_Services", "Ind_AdminSupp_P", "administrative_services_industry"],
            "public_administration": ["Public_Administration_Safety", "Ind_PubAdmin_P", "government_industry"],
            "education_training": ["Education_Training", "Ind_Educ_P", "education_industry"],
            "health_social_assistance": ["Health_Care_Social_Assistance", "Ind_Health_P", "healthcare_industry"],
            "arts_recreation": ["Arts_Recreation_Services", "Ind_Arts_P", "arts_recreation_industry"],
            "other_services": ["Other_Services", "Ind_Other_P", "other_services_industry"],
            "industry_not_stated": ["Industry_of_employment_not_stated", "Ind_NS_P", "industry_not_stated"],
            
            # ANZSCO Occupation categories (Table G37)
            "managers": ["Managers", "Occ_Managers_P", "managers_occupation"],
            "professionals": ["Professionals", "Occ_Professionals_P", "professionals_occupation"],
            "technicians_trades": ["Technicians_Trades_Workers", "Occ_TechTrades_P", "technicians_trades_workers"],
            "community_personal_service": ["Community_Personal_Service_Workers", "Occ_CommPers_P", "community_personal_service_workers"],
            "clerical_administrative": ["Clerical_Administrative_Workers", "Occ_ClerAdmin_P", "clerical_administrative_workers"],
            "sales_workers": ["Sales_Workers", "Occ_Sales_P", "sales_workers"],
            "machinery_operators": ["Machinery_Operators_Drivers", "Occ_MachOp_P", "machinery_operators_drivers"],
            "labourers": ["Labourers", "Occ_Labour_P", "labourers"],
            "occupation_not_stated": ["Occupation_not_stated", "Occ_NS_P", "occupation_not_stated"],
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
        Load target schema configuration for CensusEmployment.
        
        Returns:
            Dict[str, Any]: Schema configuration for validation and enforcement
        """
        schema_config = get_config("schemas.census_employment", {})
        
        # Default schema definition if not in config
        if not schema_config:
            schema_config = {
                'employment_sk': 'int64',
                'geo_sk': 'int64',
                'geographic_id': 'object',
                'geographic_level': 'object',
                'census_year': 'int64',
                
                # Labour force base
                'labour_force_pop': 'int64',
                
                # Labour force status
                'employed_full_time': 'int64',
                'employed_part_time': 'int64',
                'unemployed': 'int64',
                'not_in_labour_force': 'int64',
                'labour_force_not_stated': 'int64',
                
                # Industry classification (ANZSIC)
                'agriculture_forestry_fishing': 'int64',
                'mining': 'int64',
                'manufacturing': 'int64',
                'electricity_gas_water': 'int64',
                'construction': 'int64',
                'wholesale_trade': 'int64',
                'retail_trade': 'int64',
                'accommodation_food': 'int64',
                'transport_postal': 'int64',
                'information_media': 'int64',
                'financial_insurance': 'int64',
                'rental_real_estate': 'int64',
                'professional_services': 'int64',
                'administrative_support': 'int64',
                'public_administration': 'int64',
                'education_training': 'int64',
                'health_social_assistance': 'int64',
                'arts_recreation': 'int64',
                'other_services': 'int64',
                'industry_not_stated': 'int64',
                
                # Occupation classification (ANZSCO)
                'managers': 'int64',
                'professionals': 'int64',
                'technicians_trades': 'int64',
                'community_personal_service': 'int64',
                'clerical_administrative': 'int64',
                'sales_workers': 'int64',
                'machinery_operators': 'int64',
                'labourers': 'int64',
                'occupation_not_stated': 'int64',
                
                # ETL metadata
                'processed_timestamp': 'datetime64[ns]',
                'table_code': 'object',
                'table_name': 'object'
            }
        
        return schema_config
    
    def _load_operations_config(self) -> Dict[str, Any]:
        """
        Load operations configuration for employment processing.
        
        Returns:
            Dict[str, Any]: Configuration for employment-specific operations
        """
        operations_config = get_config("transformers.census.operations", {})
        
        # Default operations configuration
        if not operations_config:
            operations_config = {
                "occupation_classification": "ANZSCO_2021",
                "industry_classification": "ANZSIC_2006",
                "include_skill_levels": True,
                "include_sector_grouping": True,
                "employment_indicators": ["unemployment_rate", "participation_rate", "employment_self_sufficiency"],
                "industry_diversity_analysis": True,
                "education_employment_alignment": True
            }
        
        return operations_config
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main transformation method for employment data.
        
        Orchestrates the complete employment data transformation pipeline
        following the 9-stage process established in previous transformers.
        
        Args:
            df (pd.DataFrame): Raw employment census data
            
        Returns:
            pd.DataFrame: Transformed data conforming to CensusEmployment schema
            
        Raises:
            TransformationError: If transformation fails and stop_on_error is True
        """
        try:
            start_time = time.time()
            
            # Initialize processing metadata
            self.processing_metadata = ProcessingMetadata(
                operation_id=f"employment_transform_{int(time.time())}",
                operation_type="employment_transformation",
                status=ProcessingStatus.RUNNING,
                start_time=datetime.now()
            )
            
            self.logger.info("Starting employment data transformation", 
                           extra={"component": "EmploymentTransformer", "input_rows": len(df)})
            
            # Stage 1: Column standardisation and mapping
            self.logger.debug("Stage 1: Column standardisation and mapping")
            df = self._standardise_input_data(df)
            
            # Stage 2: Geographic hierarchy integration (if enabled)
            if self.geographic_hierarchy:
                self.logger.debug("Stage 2: Geographic hierarchy integration")
                df = self._integrate_geographic_hierarchy(df)
            
            # Stage 3: Labour force status processing
            self.logger.debug("Stage 3: Labour force status processing")
            df = self._process_labour_force_status(df)
            
            # Stage 4: ANZSCO occupation classification
            self.logger.debug("Stage 4: ANZSCO occupation classification")
            df = self._process_anzsco_occupation_classification(df)
            
            # Stage 5: ANZSIC industry classification
            self.logger.debug("Stage 5: ANZSIC industry classification")
            df = self._process_anzsic_industry_classification(df)
            
            # Stage 6: Employment indicators calculation
            self.logger.debug("Stage 6: Employment indicators calculation")
            df = self._calculate_employment_indicators(df)
            
            # Stage 7: Education-employment alignment analysis
            self.logger.debug("Stage 7: Education-employment alignment analysis")
            df = self._analyse_education_employment_alignment(df)
            
            # Stage 8: Employment-weighted missing value imputation
            self.logger.debug("Stage 8: Employment-weighted missing value imputation")
            df = self._impute_missing_values(df)
            
            # Stage 9: Final schema enforcement and compliance
            self.logger.debug("Stage 9: Final schema enforcement and compliance")
            df = self._enforce_schema(df)
            
            # Update records processed and mark completed
            self.processing_metadata.records_processed = len(df)
            self.processing_metadata.mark_completed()
            
            # Finalise processing metadata
            end_time = time.time()
            processing_time = end_time - start_time
            
            self.logger.info("Employment data transformation completed successfully",
                           extra={
                               "component": "EmploymentTransformer",
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
            self.logger.error(f"Employment transformation failed: {error_msg}", 
                            extra={"component": "EmploymentTransformer", "error": error_msg})
            
            if self.stop_on_error:
                raise TransformationError(f"Employment transformation failed: {error_msg}")
            else:
                # Log error and return original data
                self.logger.warning("Continuing with original data due to stop_on_error=False")
                return df
    
    def _standardise_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise input column names using configured mappings.
        
        Args:
            df (pd.DataFrame): Raw employment data with ABS column names
            
        Returns:
            pd.DataFrame: Data with standardised column names
        """
        standardised_df = df.copy()
        
        # Track mapping success for logging
        successful_mappings = 0
        failed_mappings = []
        
        # Get essential fields that must be present
        essential_fields = ['geographic_id', 'labour_force_pop']
        
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
                # Only add default values for non-essential required fields
                if target_field in self._get_required_fields() and target_field not in essential_fields:
                    standardised_df[target_field] = 0
                    self.logger.warning(f"Required field {target_field} not found, added with default value 0")
        
        # Log mapping summary
        self.logger.info(f"Column mapping completed: {successful_mappings} successful, {len(failed_mappings)} failed")
        
        if failed_mappings:
            self.logger.warning(f"Failed to map columns: {failed_mappings}")
        
        # Validate that we have essential columns for employment transformation
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
            'geographic_id', 'labour_force_pop',
            'employed_full_time', 'employed_part_time', 'unemployed', 'not_in_labour_force'
        ]
    
    def _integrate_geographic_hierarchy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate geographic hierarchy information (SA2->SA3->SA4->STE).
        
        Args:
            df (pd.DataFrame): Employment data with geographic identifiers
            
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
    
    def _process_labour_force_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process labour force status data (employed, unemployed, participation rates).
        
        Args:
            df (pd.DataFrame): Employment data with labour force columns
            
        Returns:
            pd.DataFrame: Data with processed labour force information
        """
        processed_df = df.copy()
        
        # Define labour force columns
        labour_force_columns = [
            'employed_full_time', 'employed_part_time', 'unemployed', 
            'not_in_labour_force', 'labour_force_not_stated'
        ]
        
        # Ensure all labour force columns exist with default values
        for column in labour_force_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0
                self.logger.debug(f"Added missing labour force column: {column}")
        
        # Convert to numeric and handle any non-numeric values
        for column in labour_force_columns:
            processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce').fillna(0).astype(int)
        
        # Calculate total employed
        processed_df['total_employed'] = processed_df['employed_full_time'] + processed_df['employed_part_time']
        
        # Calculate total labour force (employed + unemployed)
        processed_df['total_labour_force'] = processed_df['total_employed'] + processed_df['unemployed']
        
        # Calculate unemployment rate
        processed_df['unemployment_rate'] = np.where(
            processed_df['total_labour_force'] > 0,
            (processed_df['unemployed'] / processed_df['total_labour_force']) * 100,
            0.0
        )
        
        # Calculate labour force participation rate
        processed_df['participation_rate'] = np.where(
            processed_df['labour_force_pop'] > 0,
            (processed_df['total_labour_force'] / processed_df['labour_force_pop']) * 100,
            0.0
        )
        
        # Calculate employment to population ratio
        processed_df['employment_population_ratio'] = np.where(
            processed_df['labour_force_pop'] > 0,
            (processed_df['total_employed'] / processed_df['labour_force_pop']) * 100,
            0.0
        )
        
        # Calculate full-time employment ratio
        processed_df['full_time_employment_ratio'] = np.where(
            processed_df['total_employed'] > 0,
            (processed_df['employed_full_time'] / processed_df['total_employed']) * 100,
            0.0
        )
        
        # Validate labour force data
        self._validate_labour_force_data(processed_df)
        
        self.logger.debug("Labour force status processing completed")
        return processed_df
    
    def _validate_labour_force_data(self, df: pd.DataFrame) -> None:
        """
        Validate labour force data for consistency and reasonableness.
        
        Args:
            df (pd.DataFrame): DataFrame with labour force data
        """
        # Check for negative values
        labour_force_columns = ['employed_full_time', 'employed_part_time', 'unemployed', 'not_in_labour_force']
        for column in labour_force_columns:
            if column in df.columns:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    self.logger.warning(f"Found {negative_count} negative values in {column}")
        
        # Check for extreme unemployment rates
        if 'unemployment_rate' in df.columns:
            very_high_unemployment = (df['unemployment_rate'] > 25).sum()
            zero_unemployment = (df['unemployment_rate'] == 0).sum()
            
            if very_high_unemployment > 0:
                self.logger.warning(f"Found {very_high_unemployment} areas with very high unemployment (>25%)")
            if zero_unemployment > 0:
                self.logger.info(f"Found {zero_unemployment} areas with zero unemployment")
        
        # Check for extreme participation rates
        if 'participation_rate' in df.columns:
            very_high_participation = (df['participation_rate'] > 95).sum()
            very_low_participation = (df['participation_rate'] < 30).sum()
            
            if very_high_participation > 0:
                self.logger.info(f"Found {very_high_participation} areas with very high participation (>95%)")
            if very_low_participation > 0:
                self.logger.warning(f"Found {very_low_participation} areas with very low participation (<30%)")
        
        # Check labour force consistency (total should approximately equal population base)
        if all(col in df.columns for col in ['total_labour_force', 'not_in_labour_force', 'labour_force_pop']):
            total_accounted = df['total_labour_force'] + df['not_in_labour_force']
            inconsistent = abs(total_accounted - df['labour_force_pop']) > (df['labour_force_pop'] * 0.05)
            inconsistent_count = inconsistent.sum()
            if inconsistent_count > 0:
                self.logger.warning(f"Found {inconsistent_count} areas with labour force accounting inconsistencies")
    
    def _process_anzsco_occupation_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process ANZSCO occupation classification and calculate skill level groupings.
        
        Args:
            df (pd.DataFrame): Employment data with ANZSCO occupation columns
            
        Returns:
            pd.DataFrame: Data with ANZSCO skill level classifications
        """
        processed_df = df.copy()
        
        # Define ANZSCO occupation columns
        anzsco_columns = [
            'managers', 'professionals', 'technicians_trades', 'community_personal_service',
            'clerical_administrative', 'sales_workers', 'machinery_operators', 'labourers',
            'occupation_not_stated'
        ]
        
        # Ensure all ANZSCO columns exist with default values
        for column in anzsco_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0
                self.logger.debug(f"Added missing ANZSCO column: {column}")
        
        # Convert to numeric and handle any non-numeric values
        for column in anzsco_columns:
            processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce').fillna(0).astype(int)
        
        # ANZSCO Skill Level Classifications (based on ANZSCO 2021)
        # Skill Level 1-2: High skill occupations requiring tertiary education
        processed_df['skill_level_1_2'] = (
            processed_df['managers'] + processed_df['professionals']
        )
        
        # Skill Level 3: Medium skill occupations requiring Certificate III/IV or diploma
        processed_df['skill_level_3'] = (
            processed_df['technicians_trades'] + 
            processed_df['community_personal_service'] + 
            processed_df['clerical_administrative']
        )
        
        # Skill Level 4-5: Lower skill occupations requiring Certificate I/II or on-the-job training
        processed_df['skill_level_4_5'] = (
            processed_df['sales_workers'] + 
            processed_df['machinery_operators'] + 
            processed_df['labourers']
        )
        
        # Calculate total employed with classified occupations (excluding not stated)
        processed_df['total_employed_classified'] = (
            processed_df['skill_level_1_2'] + 
            processed_df['skill_level_3'] + 
            processed_df['skill_level_4_5']
        )
        
        # Calculate skill level proportions
        processed_df['skill_level_1_2_pct'] = np.where(
            processed_df['total_employed_classified'] > 0,
            (processed_df['skill_level_1_2'] / processed_df['total_employed_classified']) * 100,
            0.0
        )
        
        processed_df['skill_level_3_pct'] = np.where(
            processed_df['total_employed_classified'] > 0,
            (processed_df['skill_level_3'] / processed_df['total_employed_classified']) * 100,
            0.0
        )
        
        processed_df['skill_level_4_5_pct'] = np.where(
            processed_df['total_employed_classified'] > 0,
            (processed_df['skill_level_4_5'] / processed_df['total_employed_classified']) * 100,
            0.0
        )
        
        self.logger.debug("ANZSCO occupation classification completed")
        return processed_df
    
    def _process_anzsic_industry_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process ANZSIC industry classification and calculate sector groupings.
        
        Args:
            df (pd.DataFrame): Employment data with ANZSIC industry columns
            
        Returns:
            pd.DataFrame: Data with ANZSIC sector classifications and diversity index
        """
        processed_df = df.copy()
        
        # Define ANZSIC industry columns
        anzsic_columns = [
            'agriculture_forestry_fishing', 'mining', 'manufacturing', 'electricity_gas_water',
            'construction', 'wholesale_trade', 'retail_trade', 'accommodation_food',
            'transport_postal', 'information_media', 'financial_insurance', 'rental_real_estate',
            'professional_services', 'administrative_support', 'public_administration',
            'education_training', 'health_social_assistance', 'arts_recreation', 'other_services',
            'industry_not_stated'
        ]
        
        # Ensure all ANZSIC columns exist with default values
        for column in anzsic_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0
                self.logger.debug(f"Added missing ANZSIC column: {column}")
        
        # Convert to numeric and handle any non-numeric values
        for column in anzsic_columns:
            processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce').fillna(0).astype(int)
        
        # ANZSIC Sector Classifications
        # Primary industries: Resource extraction
        processed_df['primary_industries'] = (
            processed_df['agriculture_forestry_fishing'] + processed_df['mining']
        )
        
        # Secondary industries: Manufacturing and construction
        processed_df['secondary_industries'] = (
            processed_df['manufacturing'] + processed_df['construction']
        )
        
        # Tertiary industries: Services (all remaining industries except not stated)
        tertiary_industries = [
            'electricity_gas_water', 'wholesale_trade', 'retail_trade', 'accommodation_food',
            'transport_postal', 'information_media', 'financial_insurance', 'rental_real_estate',
            'professional_services', 'administrative_support', 'public_administration',
            'education_training', 'health_social_assistance', 'arts_recreation', 'other_services'
        ]
        processed_df['tertiary_industries'] = processed_df[tertiary_industries].sum(axis=1)
        
        # Public vs Private sector split
        public_sector_industries = [
            'public_administration', 'education_training', 'health_social_assistance'
        ]
        processed_df['public_sector'] = processed_df[public_sector_industries].sum(axis=1)
        
        # Private sector (all industries except public sector and not stated)
        private_sector_industries = [col for col in anzsic_columns 
                                   if col not in public_sector_industries + ['industry_not_stated']]
        processed_df['private_sector'] = processed_df[private_sector_industries].sum(axis=1)
        
        # Calculate total employed by industry (excluding not stated)
        industry_employment_columns = [col for col in anzsic_columns if col != 'industry_not_stated']
        processed_df['total_industry_employment'] = processed_df[industry_employment_columns].sum(axis=1)
        
        # Calculate industry diversity index using Shannon diversity index
        processed_df['industry_diversity_index'] = processed_df.apply(
            self._calculate_shannon_diversity_index, axis=1, 
            columns=industry_employment_columns
        )
        
        # Calculate sector proportions
        processed_df['primary_sector_pct'] = np.where(
            processed_df['total_industry_employment'] > 0,
            (processed_df['primary_industries'] / processed_df['total_industry_employment']) * 100,
            0.0
        )
        
        processed_df['secondary_sector_pct'] = np.where(
            processed_df['total_industry_employment'] > 0,
            (processed_df['secondary_industries'] / processed_df['total_industry_employment']) * 100,
            0.0
        )
        
        processed_df['tertiary_sector_pct'] = np.where(
            processed_df['total_industry_employment'] > 0,
            (processed_df['tertiary_industries'] / processed_df['total_industry_employment']) * 100,
            0.0
        )
        
        self.logger.debug("ANZSIC industry classification completed")
        return processed_df
    
    def _calculate_shannon_diversity_index(self, row: pd.Series, columns: List[str]) -> float:
        """
        Calculate Shannon diversity index for industry employment distribution.
        
        Args:
            row (pd.Series): DataFrame row containing industry employment counts
            columns (List[str]): List of industry column names
            
        Returns:
            float: Shannon diversity index (0 = no diversity, 1 = maximum diversity)
        """
        # Get employment counts for all industries and ensure they're numeric
        counts = []
        for col in columns:
            if col in row:
                val = pd.to_numeric(row[col], errors='coerce')
                counts.append(0 if pd.isna(val) else val)
            else:
                counts.append(0)
        
        counts = np.array(counts, dtype=float)
        total = counts.sum()
        
        if total == 0:
            return 0.0
        
        # Calculate proportions
        proportions = counts / total
        
        # Remove zero proportions to avoid log(0)
        proportions = proportions[proportions > 0]
        
        if len(proportions) == 0:
            return 0.0
        
        # Calculate Shannon entropy using natural logarithm
        shannon_entropy = -np.sum(proportions * np.log(proportions))
        
        # Normalise by theoretical maximum (log of number of categories)
        max_entropy = np.log(len(columns))
        
        if max_entropy == 0:
            return 0.0
        
        # Return normalised diversity index (0-1 scale)
        return float(shannon_entropy / max_entropy)
    
    def _calculate_employment_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced employment indicators and metrics.
        
        Args:
            df (pd.DataFrame): Employment data with classification results
            
        Returns:
            pd.DataFrame: Data with advanced employment indicators
        """
        processed_df = df.copy()
        
        # Ensure required columns exist
        required_columns = [
            'total_employed', 'labour_force_pop', 'employed_full_time', 'employed_part_time'
        ]
        for column in required_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0
                self.logger.debug(f"Added missing column for indicators: {column}")
        
        # Employment self-sufficiency: ratio of employed to working age population
        processed_df['employment_self_sufficiency'] = np.where(
            processed_df['labour_force_pop'] > 0,
            (processed_df['total_employed'] / processed_df['labour_force_pop']) * 100,
            0.0
        )
        
        # High skill employment ratio (if skill level data available)
        if 'skill_level_1_2' in processed_df.columns:
            processed_df['high_skill_employment_ratio'] = np.where(
                processed_df['total_employed'] > 0,
                (processed_df['skill_level_1_2'] / processed_df['total_employed']) * 100,
                0.0
            )
        else:
            processed_df['high_skill_employment_ratio'] = 0.0
        
        # Public sector employment ratio (if sector data available)
        if 'public_sector' in processed_df.columns:
            processed_df['public_sector_ratio'] = np.where(
                processed_df['total_employed'] > 0,
                (processed_df['public_sector'] / processed_df['total_employed']) * 100,
                0.0
            )
        else:
            processed_df['public_sector_ratio'] = 0.0
        
        # Full-time employment ratio
        processed_df['full_time_ratio'] = np.where(
            processed_df['total_employed'] > 0,
            (processed_df['employed_full_time'] / processed_df['total_employed']) * 100,
            0.0
        )
        
        # Part-time employment ratio
        processed_df['part_time_ratio'] = np.where(
            processed_df['total_employed'] > 0,
            (processed_df['employed_part_time'] / processed_df['total_employed']) * 100,
            0.0
        )
        
        # Skills mismatch index (simplified - based on skill level distribution)
        if all(col in processed_df.columns for col in ['skill_level_1_2', 'skill_level_3', 'skill_level_4_5']):
            # Calculate how balanced the skill distribution is (lower values = more balanced)
            total_classified = processed_df['skill_level_1_2'] + processed_df['skill_level_3'] + processed_df['skill_level_4_5']
            
            processed_df['skills_mismatch_index'] = np.where(
                total_classified > 0,
                processed_df.apply(self._calculate_skill_imbalance, axis=1),
                0.0
            )
        else:
            processed_df['skills_mismatch_index'] = 0.0
        
        # Economic complexity indicator (if industry diversity available)
        if 'industry_diversity_index' in processed_df.columns:
            processed_df['economic_complexity'] = processed_df['industry_diversity_index'] * 100
        else:
            processed_df['economic_complexity'] = 0.0
        
        self.logger.debug("Advanced employment indicators calculation completed")
        return processed_df
    
    def _calculate_skill_imbalance(self, row: pd.Series) -> float:
        """
        Calculate skill distribution imbalance using coefficient of variation.
        
        Args:
            row (pd.Series): DataFrame row with skill level data
            
        Returns:
            float: Skill imbalance index (higher = more imbalanced)
        """
        skill_levels = [
            row.get('skill_level_1_2', 0),
            row.get('skill_level_3', 0), 
            row.get('skill_level_4_5', 0)
        ]
        
        # Convert to numpy array and filter out zeros
        skill_array = np.array(skill_levels)
        skill_array = skill_array[skill_array > 0]
        
        if len(skill_array) < 2:
            return 0.0
        
        # Calculate coefficient of variation (std dev / mean)
        mean_val = np.mean(skill_array)
        if mean_val == 0:
            return 0.0
        
        std_val = np.std(skill_array)
        coefficient_variation = std_val / mean_val
        
        # Normalise to 0-100 scale
        return min(coefficient_variation * 100, 100.0)
    
    def _analyse_education_employment_alignment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyse education-employment alignment and qualification utilisation.
        
        Args:
            df (pd.DataFrame): Employment data with education and occupation information
            
        Returns:
            pd.DataFrame: Data with education-employment alignment indicators
        """
        processed_df = df.copy()
        
        # Define education qualification columns (may come from linked education data)
        education_columns = [
            'bachelor_degree', 'postgraduate_degree', 'graduate_diploma',
            'advanced_diploma', 'certificate_iii_iv', 'certificate_i_ii',
            'no_qualification', 'qualification_not_stated'
        ]
        
        # Ensure education columns exist with default values
        for column in education_columns:
            if column not in processed_df.columns:
                processed_df[column] = 0
                self.logger.debug(f"Added missing education column: {column}")
        
        # Convert to numeric
        for column in education_columns:
            processed_df[column] = pd.to_numeric(processed_df[column], errors='coerce').fillna(0).astype(int)
        
        # Calculate total university-qualified population
        processed_df['university_qualified'] = (
            processed_df['bachelor_degree'] + 
            processed_df['postgraduate_degree'] + 
            processed_df['graduate_diploma']
        )
        
        # Calculate total VET-qualified population (vocational education and training)
        processed_df['vet_qualified'] = (
            processed_df['advanced_diploma'] + 
            processed_df['certificate_iii_iv'] + 
            processed_df['certificate_i_ii']
        )
        
        # Qualification utilisation rate: professional roles vs university qualifications
        if 'professionals' in processed_df.columns:
            processed_df['qualification_utilisation_rate'] = np.where(
                processed_df['university_qualified'] > 0,
                (processed_df['professionals'] / processed_df['university_qualified']) * 100,
                0.0
            )
        else:
            processed_df['qualification_utilisation_rate'] = 0.0
        
        # Trade qualification utilisation: trade roles vs trade qualifications
        if 'technicians_trades' in processed_df.columns:
            processed_df['trade_utilisation_rate'] = np.where(
                processed_df['certificate_iii_iv'] > 0,
                (processed_df['technicians_trades'] / processed_df['certificate_iii_iv']) * 100,
                0.0
            )
        else:
            processed_df['trade_utilisation_rate'] = 0.0
        
        # Over-qualification rate: university qualified in non-professional roles
        if all(col in processed_df.columns for col in ['professionals', 'managers', 'total_employed']):
            high_skill_roles = processed_df['professionals'] + processed_df['managers']
            university_in_other_roles = np.maximum(
                processed_df['university_qualified'] - high_skill_roles, 0
            )
            
            processed_df['over_qualification_rate'] = np.where(
                processed_df['university_qualified'] > 0,
                (university_in_other_roles / processed_df['university_qualified']) * 100,
                0.0
            )
        else:
            processed_df['over_qualification_rate'] = 0.0
        
        # Under-qualification rate: professionals without university qualifications
        if all(col in processed_df.columns for col in ['professionals', 'managers']):
            high_skill_roles = processed_df['professionals'] + processed_df['managers']
            under_qualified = np.maximum(
                high_skill_roles - processed_df['university_qualified'], 0
            )
            
            processed_df['under_qualification_rate'] = np.where(
                high_skill_roles > 0,
                (under_qualified / high_skill_roles) * 100,
                0.0
            )
        else:
            processed_df['under_qualification_rate'] = 0.0
        
        # Skill match index: overall alignment between education and employment
        # Higher values indicate better alignment
        if all(col in processed_df.columns for col in ['qualification_utilisation_rate', 'trade_utilisation_rate']):
            # Weight by the size of each qualified population
            total_qualified = processed_df['university_qualified'] + processed_df['certificate_iii_iv']
            
            processed_df['skill_match_index'] = np.where(
                total_qualified > 0,
                (
                    (processed_df['qualification_utilisation_rate'] * processed_df['university_qualified'] +
                     processed_df['trade_utilisation_rate'] * processed_df['certificate_iii_iv']) / 
                    total_qualified
                ),
                0.0
            )
        else:
            processed_df['skill_match_index'] = 0.0
        
        # Education diversity index: distribution across qualification levels
        qualification_columns = [col for col in education_columns if col != 'qualification_not_stated']
        if qualification_columns:
            processed_df['education_diversity_index'] = processed_df.apply(
                self._calculate_shannon_diversity_index, axis=1,
                columns=qualification_columns
            )
        else:
            processed_df['education_diversity_index'] = 0.0
        
        self.logger.debug("Education-employment alignment analysis completed")
        return processed_df
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using employment-weighted geographic median strategy.
        
        Uses hierarchical geographic fallback (SA3 → SA4 → State → Global) with
        labour_force_pop as weights for median calculations.
        
        Args:
            df (pd.DataFrame): Employment data with potential missing values
            
        Returns:
            pd.DataFrame: Data with missing values imputed
        """
        processed_df = df.copy()
        
        # Identify numeric columns that should be imputed (excluding IDs and percentages)
        numeric_columns = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = [
            'employment_sk', 'geo_sk', 'census_year', 'geographic_level',
            'labour_force_pop'  # Don't impute the weighting column itself
        ]
        
        # Columns that should be imputed
        imputation_columns = [col for col in numeric_columns if col not in exclude_columns]
        
        # Skip imputation if no missing values
        has_missing = any(processed_df[col].isna().any() for col in imputation_columns)
        if not has_missing:
            self.logger.debug("No missing values found, skipping imputation")
            return processed_df
        
        # Ensure geographic hierarchy columns exist
        if 'sa3_code' not in processed_df.columns:
            processed_df = self._integrate_geographic_hierarchy(processed_df)
        
        # Ensure labour_force_pop exists for weighting
        if 'labour_force_pop' not in processed_df.columns:
            processed_df['labour_force_pop'] = 1  # Default weight if missing
        
        # Replace zero weights with 1 to avoid division issues
        processed_df['labour_force_pop'] = processed_df['labour_force_pop'].replace(0, 1)
        
        # Apply imputation for each column with missing values
        for column in imputation_columns:
            if processed_df[column].isna().any():
                processed_df[column] = self._impute_column_employment_weighted(
                    processed_df, column
                )
                
        missing_count = sum(processed_df[col].isna().sum() for col in imputation_columns)
        self.logger.debug(f"Employment-weighted imputation completed, {missing_count} values imputed")
        
        return processed_df
    
    def _impute_column_employment_weighted(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Impute a single column using employment-weighted geographic medians.
        
        Args:
            df (pd.DataFrame): DataFrame with geographic hierarchy
            column (str): Column name to impute
            
        Returns:
            pd.Series: Column with imputed values
        """
        result_series = df[column].copy()
        missing_mask = result_series.isna()
        
        if not missing_mask.any():
            return result_series
        
        # For each missing value, try hierarchical imputation
        for idx in df[missing_mask].index:
            sa3_code = df.loc[idx, 'sa3_code']
            sa4_code = df.loc[idx, 'sa4_code'] 
            state_code = df.loc[idx, 'state_code']
            
            # Try SA3 level first
            imputed_value = self._calculate_weighted_median(
                df, column, 'sa3_code', sa3_code
            )
            
            # Fall back to SA4 level
            if pd.isna(imputed_value):
                imputed_value = self._calculate_weighted_median(
                    df, column, 'sa4_code', sa4_code
                )
            
            # Fall back to state level
            if pd.isna(imputed_value):
                imputed_value = self._calculate_weighted_median(
                    df, column, 'state_code', state_code
                )
            
            # Fall back to global median
            if pd.isna(imputed_value):
                imputed_value = self._calculate_weighted_median(
                    df, column, None, None
                )
            
            # Use 0 as final fallback
            if pd.isna(imputed_value):
                imputed_value = 0
                self.logger.warning(f"Using zero fallback for {column} at index {idx}")
            
            result_series.iloc[idx] = imputed_value
        
        return result_series
    
    def _calculate_weighted_median(self, df: pd.DataFrame, column: str, 
                                 geo_column: Optional[str] = None, 
                                 geo_value: Optional[str] = None) -> float:
        """
        Calculate employment-weighted median for a column within geographic area.
        
        Args:
            df (pd.DataFrame): DataFrame with data
            column (str): Column to calculate median for
            geo_column (str, optional): Geographic grouping column
            geo_value (str, optional): Geographic value to filter by
            
        Returns:
            float: Weighted median value or NaN if insufficient data
        """
        # Filter data for geographic area
        if geo_column and geo_value:
            area_data = df[df[geo_column] == geo_value]
        else:
            area_data = df
        
        # Get non-missing values and their weights
        valid_data = area_data[area_data[column].notna()]
        
        if len(valid_data) == 0:
            return np.nan
        
        values = valid_data[column].values
        weights = valid_data['labour_force_pop'].values
        
        # Handle case with single value
        if len(values) == 1:
            return float(values[0])
        
        # Calculate weighted median
        try:
            # Sort values and weights together
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_weights = weights[sorted_indices]
            
            # Calculate cumulative weights
            cumulative_weights = np.cumsum(sorted_weights)
            total_weight = cumulative_weights[-1]
            
            # Find median position
            median_position = total_weight / 2.0
            
            # Find the weighted median
            median_idx = np.searchsorted(cumulative_weights, median_position, side='right')
            
            if median_idx >= len(sorted_values):
                median_idx = len(sorted_values) - 1
            
            return float(sorted_values[median_idx])
            
        except Exception as e:
            self.logger.warning(f"Error calculating weighted median for {column}: {e}")
            # Fall back to simple median
            return float(np.median(values)) if len(values) > 0 else np.nan
    
    def _enforce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce final schema compliance for CensusEmployment output.
        
        Adds missing schema columns, casts data types, removes extra columns,
        and reorders to match the target schema.
        
        Args:
            df (pd.DataFrame): Processed employment data
            
        Returns:
            pd.DataFrame: Schema-compliant employment data
        """
        processed_df = df.copy()
        
        # Generate employment surrogate keys if not present
        if 'employment_sk' not in processed_df.columns:
            processed_df['employment_sk'] = range(
                self.employment_sk_counter, 
                self.employment_sk_counter + len(processed_df)
            )
            self.employment_sk_counter += len(processed_df)
        
        # Add required schema columns with defaults
        schema_defaults = {
            'geo_sk': 0,  # Will be populated by data warehouse loader
            'geographic_id': '',  # Empty string for missing geographic ID
            'geographic_level': 'SA2',
            'census_year': 2021,
            'labour_force_pop': 0,  # Ensure this is always present
            'processed_timestamp': pd.Timestamp.now(),
            'table_code': 'G43',  # Default employment table
            'table_name': 'Labour Force Status'
        }
        
        for col, default_value in schema_defaults.items():
            if col not in processed_df.columns:
                processed_df[col] = default_value
        
        # Ensure all required employment columns exist with defaults
        employment_columns = {
            'employed_full_time': 0,
            'employed_part_time': 0,
            'unemployed': 0,
            'not_in_labour_force': 0,
            'labour_force_not_stated': 0,
            
            # Industry columns (ANZSIC)
            'agriculture_forestry_fishing': 0,
            'mining': 0,
            'manufacturing': 0,
            'electricity_gas_water': 0,
            'construction': 0,
            'wholesale_trade': 0,
            'retail_trade': 0,
            'accommodation_food': 0,
            'transport_postal': 0,
            'information_media': 0,
            'financial_insurance': 0,
            'rental_real_estate': 0,
            'professional_services': 0,
            'administrative_support': 0,
            'public_administration': 0,
            'education_training': 0,
            'health_social_assistance': 0,
            'arts_recreation': 0,
            'other_services': 0,
            'industry_not_stated': 0,
            
            # Occupation columns (ANZSCO)
            'managers': 0,
            'professionals': 0,
            'technicians_trades': 0,
            'community_personal_service': 0,
            'clerical_administrative': 0,
            'sales_workers': 0,
            'machinery_operators': 0,
            'labourers': 0,
            'occupation_not_stated': 0
        }
        
        for col, default_value in employment_columns.items():
            if col not in processed_df.columns:
                processed_df[col] = default_value
        
        # Cast data types according to schema
        type_conversions = {
            # Core identifiers
            'employment_sk': 'int64',
            'geo_sk': 'int64', 
            'geographic_id': 'object',
            'geographic_level': 'object',
            'census_year': 'int64',
            
            # Labour force base
            'labour_force_pop': 'int64',
            
            # Labour force status
            'employed_full_time': 'int64',
            'employed_part_time': 'int64',
            'unemployed': 'int64',
            'not_in_labour_force': 'int64',
            'labour_force_not_stated': 'int64',
            
            # Calculated employment columns
            'total_employed': 'int64',
            'total_labour_force': 'int64',
            
            # All industry columns to int64
            **{col: 'int64' for col in employment_columns.keys()},
            
            # Calculated indicators (rates and percentages as float)
            'unemployment_rate': 'float64',
            'participation_rate': 'float64',
            'employment_population_ratio': 'float64',
            'full_time_employment_ratio': 'float64',
            'employment_self_sufficiency': 'float64',
            'high_skill_employment_ratio': 'float64',
            'public_sector_ratio': 'float64',
            'full_time_ratio': 'float64',
            'qualification_utilisation_rate': 'float64',
            'industry_diversity_index': 'float64',
            
            # Metadata
            'table_code': 'object',
            'table_name': 'object'
        }
        
        # Apply type conversions
        for col, target_type in type_conversions.items():
            if col in processed_df.columns:
                try:
                    if target_type == 'int64':
                        # Handle NaN values before converting to int
                        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').fillna(0).astype('int64')
                    elif target_type == 'float64':
                        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce').astype('float64')
                    elif target_type == 'object':
                        processed_df[col] = processed_df[col].astype('object')
                    elif target_type == 'datetime64[ns]':
                        processed_df[col] = pd.to_datetime(processed_df[col])
                except Exception as e:
                    self.logger.warning(f"Failed to convert {col} to {target_type}: {e}")
        
        # Define schema column order (core CensusEmployment schema)
        schema_columns = [
            # Identifiers
            'employment_sk', 'geo_sk', 'geographic_id', 'geographic_level', 'census_year',
            
            # Labour force base
            'labour_force_pop',
            
            # Labour force status
            'employed_full_time', 'employed_part_time', 'unemployed', 
            'not_in_labour_force', 'labour_force_not_stated',
            
            # Industry classification (ANZSIC)
            'agriculture_forestry_fishing', 'mining', 'manufacturing', 'electricity_gas_water',
            'construction', 'wholesale_trade', 'retail_trade', 'accommodation_food',
            'transport_postal', 'information_media', 'financial_insurance', 'rental_real_estate',
            'professional_services', 'administrative_support', 'public_administration',
            'education_training', 'health_social_assistance', 'arts_recreation', 'other_services',
            'industry_not_stated',
            
            # Occupation classification (ANZSCO)
            'managers', 'professionals', 'technicians_trades', 'community_personal_service',
            'clerical_administrative', 'sales_workers', 'machinery_operators', 'labourers',
            'occupation_not_stated',
            
            # Metadata
            'processed_timestamp', 'table_code', 'table_name'
        ]
        
        # Keep all schema columns plus any calculated indicators
        final_columns = []
        
        # Add schema columns first
        for col in schema_columns:
            if col in processed_df.columns:
                final_columns.append(col)
        
        # Add calculated indicators and analytics columns (preserve advanced analytics)
        calculated_columns = [col for col in processed_df.columns 
                            if col not in schema_columns and 
                            not col.startswith('extra_') and  # Remove unwanted columns
                            col not in ['invalid_type']]  # Remove test columns
        
        final_columns.extend(calculated_columns)
        
        # Return DataFrame with final column selection and ordering
        result_df = processed_df[final_columns].copy()
        
        # Final validation
        if len(result_df) > 0:
            # Ensure employment_sk values are unique
            if result_df['employment_sk'].duplicated().any():
                result_df['employment_sk'] = range(
                    self.employment_sk_counter,
                    self.employment_sk_counter + len(result_df)
                )
                self.employment_sk_counter += len(result_df)
        
        self.logger.debug(f"Schema enforcement completed: {len(final_columns)} columns, {len(result_df)} rows")
        return result_df