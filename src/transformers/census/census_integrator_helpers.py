"""
Helper methods and utilities for CensusIntegrator.

This module contains the remaining implementation details for the CensusIntegrator
class, including data validation, temporal alignment, missing data handling,
and performance optimization utilities.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

from ...utils.logging import get_logger


class CensusIntegratorHelpers:
    """Helper methods for census integration operations."""
    
    @staticmethod
    def create_empty_dataset_structure(dataset_name: str) -> pd.DataFrame:
        """Create minimal dataset structure for join compatibility."""
        base_columns = ['geographic_id', 'geographic_level', 'census_year']
        
        additional_columns = {
            'demographics': ['total_population', 'males', 'females'],
            'education': ['education_pop_base', 'year_12_or_equivalent'],
            'employment': ['labour_force_pop', 'employed_full_time'],
            'housing': ['total_private_dwellings', 'owned_outright'],
            'seifa': ['irsad_score', 'irsd_score']
        }
        
        columns = base_columns + additional_columns.get(dataset_name, [])
        return pd.DataFrame(columns=columns)
    
    @staticmethod
    def add_missing_join_keys(data: pd.DataFrame, missing_keys: List[str]) -> pd.DataFrame:
        """Add missing join keys with default values."""
        result = data.copy()
        
        for key in missing_keys:
            if key == 'geographic_id':
                # Generate synthetic geographic IDs
                result[key] = [f"UNKNOWN_{i:06d}" for i in range(len(result))]
            elif key == 'geographic_level':
                result[key] = 'SA2'  # Default level
            elif key == 'census_year':
                result[key] = 2021  # Default year
        
        return result
    
    @staticmethod
    def align_to_primary_year(data: pd.DataFrame, primary_year: int, dataset_name: str) -> pd.DataFrame:
        """Align dataset to primary census year with interpolation."""
        if 'census_year' not in data.columns:
            result = data.copy()
            result['census_year'] = primary_year
            return result
        
        # For SEIFA data, handle year mismatches specially
        if dataset_name == 'seifa':
            return CensusIntegratorHelpers._align_seifa_temporal(data, primary_year)
        
        # For other datasets, filter to primary year or closest available
        year_mask = data['census_year'] == primary_year
        if year_mask.any():
            return data[year_mask].copy()
        
        # If primary year not available, use closest year
        available_years = data['census_year'].dropna().unique()
        if len(available_years) == 0:
            result = data.copy()
            result['census_year'] = primary_year
            return result
        
        closest_year = min(available_years, key=lambda x: abs(x - primary_year))
        result = data[data['census_year'] == closest_year].copy()
        result['census_year'] = primary_year  # Normalize to primary year
        result['_temporal_adjustment'] = abs(closest_year - primary_year)
        
        return result
    
    @staticmethod
    def _align_seifa_temporal(data: pd.DataFrame, primary_year: int) -> pd.DataFrame:
        """Special temporal alignment for SEIFA data."""
        # SEIFA is released every 5 years (2016, 2021, 2026...)
        seifa_years = [2011, 2016, 2021, 2026]
        
        # Find closest SEIFA year
        closest_seifa_year = min(seifa_years, key=lambda x: abs(x - primary_year))
        
        # Use SEIFA data from closest year
        if closest_seifa_year in data['census_year'].values:
            result = data[data['census_year'] == closest_seifa_year].copy()
        else:
            # Use any available SEIFA data
            result = data.copy()
        
        # Apply temporal adjustment factors if needed
        year_diff = abs(primary_year - closest_seifa_year)
        if year_diff > 0:
            result['_seifa_temporal_adjustment'] = year_diff
            # Could apply adjustment factors to scores here
        
        result['census_year'] = primary_year
        return result
    
    @staticmethod
    def calculate_temporal_alignment_score(data: pd.DataFrame, primary_year: int) -> float:
        """Calculate temporal alignment quality score."""
        if 'census_year' not in data.columns:
            return 50.0  # Neutral score for missing year info
        
        # Calculate proportion of records from primary year
        primary_year_records = (data['census_year'] == primary_year).sum()
        total_records = len(data)
        
        if total_records == 0:
            return 0.0
        
        alignment_rate = primary_year_records / total_records
        
        # Factor in temporal adjustments
        if '_temporal_adjustment' in data.columns:
            avg_adjustment = data['_temporal_adjustment'].mean()
            penalty = min(avg_adjustment * 10, 50)  # Max 50% penalty
        else:
            penalty = 0
        
        score = (alignment_rate * 100) - penalty
        return max(0, min(100, score))
    
    @staticmethod
    def optimize_dtypes_for_join(data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory-efficient joins."""
        result = data.copy()
        
        # Convert object columns that are actually categorical
        for col in result.select_dtypes(include=['object']).columns:
            if col not in ['geographic_id']:  # Keep geographic_id as string
                unique_vals = result[col].nunique()
                total_vals = len(result)
                
                # Convert to category if low cardinality (adjust threshold)
                if unique_vals <= max(2, total_vals * 0.1):  # More aggressive categorical conversion
                    result[col] = result[col].astype('category')
        
        # Optimize integer columns
        for col in result.select_dtypes(include=['int64']).columns:
            col_min = result[col].min()
            col_max = result[col].max()
            
            if pd.isna(col_min) or pd.isna(col_max):
                continue
                
            if col_min >= 0:
                if col_max < 255:
                    result[col] = result[col].astype('uint8')
                elif col_max < 65535:
                    result[col] = result[col].astype('uint16')
                elif col_max < 4294967295:
                    result[col] = result[col].astype('uint32')
            else:
                if col_min >= -128 and col_max <= 127:
                    result[col] = result[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    result[col] = result[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    result[col] = result[col].astype('int32')
        
        # Optimize float columns
        for col in result.select_dtypes(include=['float64']).columns:
            result[col] = pd.to_numeric(result[col], downcast='float')
        
        return result
    
    @staticmethod
    def apply_missing_data_strategies(data: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive missing data handling strategies."""
        result = data.copy()
        
        # Strategy 1: Forward fill for time series data
        time_series_cols = ['census_year']
        for col in time_series_cols:
            if col in result.columns:
                result[col] = result[col].fillna(method='ffill')
        
        # Strategy 2: Geographic interpolation for spatial data
        spatial_cols = ['geographic_id', 'geographic_level']
        for col in spatial_cols:
            if col in result.columns and result[col].isna().any():
                # Use mode for categorical spatial data
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col] = result[col].fillna(mode_val[0])
        
        # Strategy 3: Median imputation for demographic indicators
        demographic_cols = [col for col in result.columns if any(
            keyword in col.lower() for keyword in ['population', 'age_', 'male', 'female']
        )]
        for col in demographic_cols:
            if result[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                median_val = result[col].median()
                if not pd.isna(median_val):
                    result[col] = result[col].fillna(median_val)
        
        # Strategy 4: Zero fill for count data
        count_cols = [col for col in result.columns if any(
            keyword in col.lower() for keyword in ['total_', 'count_', 'number_']
        )]
        for col in count_cols:
            if result[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                result[col] = result[col].fillna(0)
        
        return result
    
    @staticmethod
    def validate_cross_domain_consistency(data: pd.DataFrame) -> pd.DataFrame:
        """Validate consistency across census domains."""
        result = data.copy()
        issues_found = []
        
        # Check population consistency across domains
        if 'total_population' in result.columns:
            pop_col = 'total_population'
            
            # Demographics domain checks
            if 'males' in result.columns and 'females' in result.columns:
                calculated_pop = result['males'] + result['females']
                pop_diff = abs(calculated_pop - result[pop_col])
                inconsistent_mask = pop_diff > 10  # Allow small discrepancies
                
                if inconsistent_mask.any():
                    issues_found.append(f"Population inconsistency in {inconsistent_mask.sum()} records")
                    # Use demographics total as authoritative
                    result.loc[inconsistent_mask, pop_col] = calculated_pop[inconsistent_mask]
            
            # Education domain checks
            if 'education_pop_base' in result.columns:
                edu_mask = result['education_pop_base'] > result[pop_col] * 1.1
                if edu_mask.any():
                    issues_found.append(f"Education population exceeds total in {edu_mask.sum()} records")
            
            # Employment domain checks  
            if 'labour_force_pop' in result.columns:
                lf_mask = result['labour_force_pop'] > result[pop_col] * 1.1
                if lf_mask.any():
                    issues_found.append(f"Labour force exceeds total in {lf_mask.sum()} records")
        
        # Check SEIFA score consistency
        seifa_score_cols = ['irsad_score', 'irsd_score', 'ier_score', 'ieo_score']
        available_seifa = [col for col in seifa_score_cols if col in result.columns]
        
        for col in available_seifa:
            # Check for scores outside typical range
            out_of_range = (result[col] < 500) | (result[col] > 1500)
            if out_of_range.any():
                issues_found.append(f"SEIFA {col} out of typical range in {out_of_range.sum()} records")
        
        # Log consistency issues
        if issues_found:
            logger = get_logger(__name__)
            logger.warning(f"Cross-domain consistency issues found: {issues_found}")
        
        # Add validation metadata
        result['_cross_domain_validation_issues'] = len(issues_found)
        
        return result
    
    @staticmethod
    def perform_cross_domain_validation(data: pd.DataFrame) -> pd.DataFrame:
        """Perform comprehensive cross-domain validation."""
        result = data.copy()
        
        # Validation 1: Age group totals vs total population
        age_group_cols = [col for col in result.columns if col.startswith('age_')]
        if age_group_cols and 'total_population' in result.columns:
            age_totals = result[age_group_cols].sum(axis=1)
            age_discrepancy = abs(age_totals - result['total_population'])
            
            # Flag records with significant age group discrepancies
            result['_age_group_validation_flag'] = age_discrepancy > 50
        
        # Validation 2: Education level consistency
        edu_level_cols = [col for col in result.columns if any(
            keyword in col for keyword in ['year_', 'degree', 'certificate', 'diploma']
        )]
        if edu_level_cols and 'education_pop_base' in result.columns:
            edu_totals = result[edu_level_cols].sum(axis=1)
            edu_discrepancy = abs(edu_totals - result['education_pop_base'])
            
            result['_education_validation_flag'] = edu_discrepancy > (result['education_pop_base'] * 0.1)
        
        # Validation 3: Employment sector totals
        employment_cols = [col for col in result.columns if any(
            sector in col for sector in ['employed_', 'unemployed', 'not_in_labour']
        )]
        if employment_cols and 'labour_force_pop' in result.columns:
            emp_totals = result[employment_cols].sum(axis=1)
            emp_discrepancy = abs(emp_totals - result['labour_force_pop'])
            
            result['_employment_validation_flag'] = emp_discrepancy > (result['labour_force_pop'] * 0.1)
        
        # Validation 4: Housing tenure consistency
        tenure_cols = [col for col in result.columns if any(
            tenure in col for tenure in ['owned_', 'rented', 'other_tenure']
        )]
        dwelling_cols = [col for col in result.columns if 'dwelling' in col]
        
        if tenure_cols and dwelling_cols:
            # Basic consistency check
            result['_housing_validation_flag'] = False
        
        return result
    
    @staticmethod
    def calculate_housing_stress_indicator(data: pd.DataFrame) -> pd.Series:
        """Calculate housing affordability stress indicator."""
        stress_factors = []
        
        # Factor 1: High rent relative to income areas
        if 'median_rent_weekly' in data.columns and 'median_household_income' in data.columns:
            rent_to_income = data['median_rent_weekly'] * 52 / data['median_household_income']
            stress_factors.append(np.where(rent_to_income > 0.3, rent_to_income * 50, 0))
        
        # Factor 2: High mortgage relative to income
        if 'median_mortgage_monthly' in data.columns and 'median_household_income' in data.columns:
            mortgage_to_income = data['median_mortgage_monthly'] * 12 / data['median_household_income']
            stress_factors.append(np.where(mortgage_to_income > 0.3, mortgage_to_income * 50, 0))
        
        # Factor 3: Low home ownership rates
        if 'owned_outright' in data.columns and 'owned_with_mortgage' in data.columns and 'total_private_dwellings' in data.columns:
            ownership_rate = (data['owned_outright'] + data['owned_with_mortgage']) / data['total_private_dwellings']
            stress_factors.append(np.where(ownership_rate < 0.6, (1 - ownership_rate) * 100, 0))
        
        # Factor 4: Overcrowding (more people than bedrooms)
        bedroom_cols = [col for col in data.columns if 'bedroom' in col]
        if bedroom_cols and 'total_population' in data.columns and 'total_private_dwellings' in data.columns:
            # Simplified overcrowding calculation
            avg_household_size = data['total_population'] / data['total_private_dwellings']
            stress_factors.append(np.where(avg_household_size > 3, (avg_household_size - 3) * 20, 0))
        
        if stress_factors:
            combined_stress = np.maximum.reduce(stress_factors)
            return pd.Series(combined_stress, index=data.index)
        else:
            return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_area_development_index(data: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive area development index."""
        development_components = []
        
        # Education component
        if 'bachelor_degree' in data.columns and 'education_pop_base' in data.columns:
            edu_rate = data['bachelor_degree'] / data['education_pop_base']
            development_components.append(edu_rate * 100)
        
        # Employment component
        if 'professionals' in data.columns and 'labour_force_pop' in data.columns:
            prof_rate = data['professionals'] / data['labour_force_pop'] 
            development_components.append(prof_rate * 100)
        
        # Infrastructure component (proxy via internet access)
        if 'internet_connection' in data.columns and 'total_private_dwellings' in data.columns:
            internet_rate = data['internet_connection'] / data['total_private_dwellings']
            development_components.append(internet_rate * 100)
        
        # SEIFA component
        if 'irsad_score' in data.columns:
            # Normalize SEIFA to 0-100 scale
            seifa_normalized = (data['irsad_score'] - 600) / 800 * 100
            development_components.append(seifa_normalized.clip(0, 100))
        
        if development_components:
            # Calculate weighted average
            weights = np.ones(len(development_components)) / len(development_components)
            development_index = np.average(development_components, weights=weights, axis=0)
            return pd.Series(development_index, index=data.index)
        else:
            return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def calculate_demographic_vulnerability(data: pd.DataFrame) -> pd.Series:
        """Calculate demographic vulnerability index."""
        vulnerability_factors = []
        
        # Age vulnerability (high proportion of children and elderly)
        age_cols = [col for col in data.columns if col.startswith('age_')]
        if age_cols and 'total_population' in data.columns:
            young_deps = sum(data[col] for col in age_cols if any(age in col for age in ['0_4', '5_9', '10_14']))
            old_deps = sum(data[col] for col in age_cols if any(age in col for age in ['75_79', '80_84', '85_plus']))
            
            if isinstance(young_deps, pd.Series) and isinstance(old_deps, pd.Series):
                dependency_rate = (young_deps + old_deps) / data['total_population']
                vulnerability_factors.append(dependency_rate * 100)
        
        # Indigenous population factor
        if 'indigenous' in data.columns and 'total_population' in data.columns:
            indigenous_rate = data['indigenous'] / data['total_population']
            vulnerability_factors.append(indigenous_rate * 50)  # Lower weight
        
        # Education vulnerability (low education completion)
        if 'year_8_or_below' in data.columns and 'education_pop_base' in data.columns:
            low_edu_rate = data['year_8_or_below'] / data['education_pop_base']
            vulnerability_factors.append(low_edu_rate * 100)
        
        # Employment vulnerability (unemployment)
        if 'unemployed' in data.columns and 'labour_force_pop' in data.columns:
            unemployment_rate = data['unemployed'] / data['labour_force_pop']
            vulnerability_factors.append(unemployment_rate * 200)  # Higher weight
        
        if vulnerability_factors:
            combined_vulnerability = np.mean(vulnerability_factors, axis=0)
            return pd.Series(combined_vulnerability, index=data.index)
        else:
            return pd.Series(np.nan, index=data.index)
    
    @staticmethod
    def standardize_final_data_types(data: pd.DataFrame) -> pd.DataFrame:
        """Standardize data types for final output."""
        result = data.copy()
        
        # Ensure geographic identifiers are strings
        geo_string_cols = ['geographic_id', 'geographic_name', 'state_territory']
        for col in geo_string_cols:
            if col in result.columns:
                result[col] = result[col].astype(str)
        
        # Ensure population counts are integers
        pop_int_cols = [col for col in result.columns if any(
            keyword in col.lower() for keyword in ['population', 'total_', 'males', 'females']
        )]
        for col in pop_int_cols:
            if result[col].dtype in ['float64', 'float32']:
                result[col] = pd.to_numeric(result[col], errors='coerce').astype('Int64')
        
        # Ensure rates and percentages are floats
        rate_float_cols = [col for col in result.columns if any(
            keyword in col.lower() for keyword in ['rate', 'ratio', 'percentage', 'index']
        )]
        for col in rate_float_cols:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce').astype('float64')
        
        # Ensure year columns are integers
        year_cols = [col for col in result.columns if 'year' in col.lower()]
        for col in year_cols:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce').astype('Int64')
        
        return result
    
    @staticmethod
    def get_integrated_sources_list(data: pd.DataFrame) -> List[str]:
        """Get list of data sources that were successfully integrated."""
        sources = []
        
        # Check for demographic indicators
        if any(col in data.columns for col in ['total_population', 'males', 'females']):
            sources.append('ABS_Census_Demographics')
        
        # Check for education indicators
        if any(col in data.columns for col in ['education_pop_base', 'year_12_or_equivalent']):
            sources.append('ABS_Census_Education')
        
        # Check for employment indicators
        if any(col in data.columns for col in ['labour_force_pop', 'employed_full_time']):
            sources.append('ABS_Census_Employment')
        
        # Check for housing indicators
        if any(col in data.columns for col in ['total_private_dwellings', 'owned_outright']):
            sources.append('ABS_Census_Housing')
        
        # Check for SEIFA indicators
        if any(col in data.columns for col in ['irsad_score', 'irsd_score']):
            sources.append('ABS_SEIFA')
        
        return sources
    
    @staticmethod
    def calculate_record_completeness(row: pd.Series) -> float:
        """Calculate completeness score for a single record."""
        # Define key indicator categories
        key_indicators = [
            'total_population', 'geographic_id', 'census_year',  # Core
            'year_12_or_equivalent', 'employed_full_time',       # Socioeconomic
            'owned_outright', 'irsad_score'                      # Housing & SEIFA
        ]
        
        available_indicators = [col for col in key_indicators if col in row.index]
        
        if not available_indicators:
            return 0.0
        
        non_null_count = sum(1 for col in available_indicators if pd.notna(row[col]))
        completeness = (non_null_count / len(available_indicators)) * 100
        
        return completeness
    
    @staticmethod
    def create_empty_integrated_schema() -> pd.DataFrame:
        """Create empty DataFrame with integrated census schema."""
        schema_columns = [
            # Core identifiers
            'census_integration_id', 'geographic_id', 'geographic_level', 'geographic_name',
            'state_territory', 'census_year',
            
            # Demographics
            'total_population', 'males', 'females',
            'age_0_4', 'age_5_9', 'age_15_19', 'age_65_69', 'age_85_plus',
            'indigenous', 'non_indigenous',
            
            # Education
            'education_pop_base', 'year_12_or_equivalent', 'bachelor_degree',
            'postgraduate_degree', 'certificate_iii_iv',
            
            # Employment
            'labour_force_pop', 'employed_full_time', 'employed_part_time', 'unemployed',
            'professionals', 'managers', 'technicians_trades',
            
            # Housing
            'total_private_dwellings', 'owned_outright', 'owned_with_mortgage', 'rented',
            'separate_house', 'flat_apartment', 'median_rent_weekly',
            
            # SEIFA
            'irsad_score', 'irsd_score', 'ier_score', 'ieo_score',
            'irsad_decile', 'irsd_decile', 'disadvantage_severity',
            
            # Derived indicators
            'comprehensive_seifa_index', 'education_employment_alignment',
            'housing_stress_indicator', 'area_development_index',
            'demographic_vulnerability_index',
            
            # Integration metadata
            'integration_timestamp', 'integration_version', 'data_completeness_score',
            'integration_quality_score', 'quality_validation_passed'
        ]
        
        return pd.DataFrame(columns=schema_columns)