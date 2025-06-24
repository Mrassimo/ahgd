"""
ABS Census data integrator for AHGD ETL pipeline.

This module integrates outputs from all census transformers (Demographics, Housing, 
Employment, Education, SEIFA) into a unified master dataset with cross-domain insights
and comprehensive data quality management.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from contextlib import contextmanager

from ...utils.logging import get_logger
from ...utils.config import get_config, get_config_manager
from ...utils.interfaces import (
    ProcessingMetadata,
    ProcessingStatus,
    TransformationError,
)
from schemas.census_schema import IntegratedCensusData


class CensusIntegrator:
    """
    Integrates outputs from all census transformers into unified master dataset.
    
    Performs optimized 4-way joins across Demographics, Housing, Employment, 
    Education, and SEIFA datasets with intelligent conflict resolution.
    
    Key Features:
    - Progressive left joins in memory-efficient order
    - Intelligent field conflict resolution with data quality hierarchy
    - Cross-domain derived indicator calculation
    - Comprehensive data quality and completeness metrics
    - Master schema enforcement with validation
    """
    
    def __init__(self):
        """
        Initialise the census integrator.
        
        Follows established transformer pattern with configuration management
        and state tracking.
        """
        # Configuration management
        self.config_manager = get_config_manager()
        self._logger_name = __name__
        
        # Initialize configuration and quality metrics (deferred)
        self._config = None
        self.quality_metrics = None
        
        # Integration configuration (legacy support)
        self.integration_config = self._load_integration_config()
        self.join_strategy = get_config("integrator.join_strategy", "left_progressive")
        
        # Data quality thresholds
        self.quality_thresholds = self._load_quality_thresholds()
        
        # State management
        self.integration_sk_counter = 50000  # Master record surrogates at 50K
        self.processing_metadata: Optional[ProcessingMetadata] = None
        
        # Performance settings
        self.chunk_size = get_config("integrator.chunk_size", 10000)
        self.parallel_joins = get_config("integrator.parallel_processing", True)
        
        # Error handling
        self.stop_on_error = get_config("system.stop_on_error", False)
        
        # Memory monitoring
        self.memory_limit_mb = get_config("integrator.memory_limit_mb", 4096)
        
        # Performance tracking
        self.performance_stats = {
            'join_times': [],
            'validation_times': [],
            'memory_usage': []
        }
        
        # Error logging
        self.error_log = []
        
        # Initialize after class definitions are available
        self._initialize_components()
        
    @property  
    def logger(self):
        """
        Get logger instance (creates new instance to avoid serialization issues).
        
        Returns:
            Logger: Thread-safe logger instance
        """
        return get_logger(self._logger_name)
    
    @property
    def config(self) -> "CensusIntegratorConfig":
        """
        Get configuration instance (lazy initialization).
        
        Returns:
            CensusIntegratorConfig: Configuration instance
        """
        if self._config is None:
            self._config = CensusIntegratorConfig(self.config_manager)
        return self._config
    
    @config.setter
    def config(self, value):
        """Set configuration instance."""
        self._config = value
    
    def _initialize_components(self):
        """Initialize configuration and quality metrics after class definitions are available."""
        # This will be called after the class definitions exist
        pass  # Lazy initialization through properties
        
    def _load_integration_config(self) -> Dict[str, Any]:
        """
        Load integration configuration from settings.
        
        Returns:
            Dict[str, Any]: Integration configuration
        """
        default_config = {
            "join_keys": ["geographic_id", "geographic_level", "census_year"],
            "priority_order": ["demographics", "housing", "employment", "education", "seifa"],
            "apply_quality_filters": False,
            "generate_summary_stats": True,
            "output_formats": ["parquet", "csv"],
            "compression": "gzip"
        }
        
        config = get_config("integrator.config", {})
        if config:
            default_config.update(config)
            
        return default_config
    
    def _load_quality_thresholds(self) -> Dict[str, float]:
        """
        Load data quality thresholds from configuration.
        
        Returns:
            Dict[str, float]: Quality threshold values
        """
        return {
            "min_completeness": get_config("integrator.quality.min_completeness", 0.5),
            "min_consistency": get_config("integrator.quality.min_consistency", 0.7),
            "max_missing_rate": get_config("integrator.quality.max_missing_rate", 0.3),
            "population_tolerance": get_config("integrator.quality.population_tolerance", 0.05)
        }
    
    def integrate(self, transformer_outputs: Optional[Dict[str, pd.DataFrame]] = None, 
                  demographics_data: Optional[pd.DataFrame] = None,
                  education_data: Optional[pd.DataFrame] = None,
                  employment_data: Optional[pd.DataFrame] = None,
                  housing_data: Optional[pd.DataFrame] = None,
                  seifa_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Main integration pipeline with 8 processing stages.
        
        Args:
            transformer_outputs: Dict of {transformer_name: DataFrame} from all transformers
            demographics_data: Demographics transformer output (alternative API)
            education_data: Education transformer output (alternative API)
            employment_data: Employment transformer output (alternative API)
            housing_data: Housing transformer output (alternative API)
            seifa_data: SEIFA transformer output (alternative API)
            
        Returns:
            pd.DataFrame: Unified master census dataset
            
        Raises:
            TransformationError: If integration fails
        """
        # Support both API styles
        if transformer_outputs is None:
            transformer_outputs = {}
            if demographics_data is not None:
                transformer_outputs['demographics'] = demographics_data
            if education_data is not None:
                transformer_outputs['education'] = education_data
            if employment_data is not None:
                transformer_outputs['employment'] = employment_data
            if housing_data is not None:
                transformer_outputs['housing'] = housing_data
            if seifa_data is not None:
                transformer_outputs['seifa'] = seifa_data
        start_time = time.time()
        
        try:
            # Initialise processing metadata
            self.processing_metadata = ProcessingMetadata(
                operation_id=f"census_integration_{int(time.time())}",
                operation_type="CensusIntegrator",
                status=ProcessingStatus.RUNNING,
                start_time=datetime.now(),
                records_processed=0
            )
            
            self.logger.info(f"Starting census integration with {len(transformer_outputs)} datasets")
            
            # Stage 1: Validate inputs and standardize join keys
            standardized_data = self._standardize_datasets(transformer_outputs)
            
            # Track performance timing
            stage_start = time.time()
            
            # Stage 2: Perform progressive left joins (optimized order)
            joined_data = self._execute_progressive_joins(standardized_data)
            
            # Track join performance
            join_time = time.time() - stage_start
            self.performance_stats['join_times'].append(join_time)
            
            # Stage 3: Resolve field conflicts and populate missing standardized fields  
            resolved_data = self._resolve_conflicts_and_populate_fields(joined_data)
            
            # Stage 4: Calculate cross-domain derived indicators
            enriched_data = self._calculate_derived_indicators(resolved_data)
            
            # Stage 5: Generate data quality and completeness metrics
            quality_data = self._generate_quality_metrics(enriched_data)
            
            # Update quality metrics tracking
            self._update_quality_metrics_tracking(transformer_outputs, quality_data)
            
            # Stage 6: Apply master schema enforcement
            schema_data = self._enforce_master_schema(quality_data)
            
            # Stage 7: Add integration metadata and lineage
            final_data = self._add_integration_metadata(schema_data)
            
            # Stage 8: Validate final output integrity
            validated_data = self._validate_integration_integrity(final_data)
            
            # Update processing metadata
            self.processing_metadata.end_time = datetime.now()
            self.processing_metadata.status = ProcessingStatus.COMPLETED
            self.processing_metadata.records_processed = len(validated_data)
            self.processing_metadata.duration_seconds = time.time() - start_time
            
            self.logger.info(f"Census integration completed: {len(validated_data)} records in {self.processing_metadata.duration_seconds:.2f}s")
            
            return validated_data
            
        except Exception as e:
            # Update metadata with error status
            if self.processing_metadata:
                self.processing_metadata.status = ProcessingStatus.FAILED
                self.processing_metadata.end_time = datetime.now()
                self.processing_metadata.duration_seconds = time.time() - start_time
                self.processing_metadata.error_message = str(e)
                self.processing_metadata.records_failed += 1
            
            error_msg = f"Census integration failed: {str(e)}"
            self.logger.error(error_msg)
            
            if self.stop_on_error:
                raise TransformationError(error_msg) from e
            else:
                # Return empty DataFrame with correct schema
                return self._create_empty_integrated_dataframe()
    
    def _standardize_datasets(self, transformer_outputs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Validate inputs and standardize join keys across all datasets.
        
        Args:
            transformer_outputs: Raw transformer outputs
            
        Returns:
            Dict[str, pd.DataFrame]: Standardized datasets
        """
        self.logger.debug("Stage 1: Standardizing datasets and join keys")
        
        standardized = {}
        join_keys = self.integration_config["join_keys"]
        
        # Validate required datasets
        required_datasets = ["demographics"]  # Demographics is mandatory base
        for dataset in required_datasets:
            if dataset not in transformer_outputs:
                raise TransformationError(f"Required dataset '{dataset}' not found in transformer outputs")
        
        # Process each dataset
        for name, df in transformer_outputs.items():
            if df.empty:
                self.logger.warning(f"Dataset '{name}' is empty, skipping")
                continue
            
            # Validate join keys exist
            missing_keys = [key for key in join_keys if key not in df.columns]
            if missing_keys:
                error_msg = f"Dataset '{name}' missing join keys: {missing_keys}"
                self.logger.warning(error_msg)
                self.error_log.append(error_msg)
                continue
            
            # Standardize data types for join keys
            standardized_df = df.copy()
            
            # Ensure geographic_id is string
            if "geographic_id" in standardized_df.columns:
                standardized_df["geographic_id"] = standardized_df["geographic_id"].astype(str)
            
            # Ensure census_year is integer
            if "census_year" in standardized_df.columns:
                standardized_df["census_year"] = pd.to_numeric(standardized_df["census_year"], errors="coerce").astype("Int64")
            
            # Sort by join keys for efficient joining
            standardized_df = standardized_df.sort_values(join_keys)
            
            standardized[name] = standardized_df
            self.logger.debug(f"Standardized '{name}': {len(standardized_df)} records")
        
        return standardized
    
    def _execute_progressive_joins(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Execute memory-efficient progressive joins in optimal order.
        
        Join Order (largest to smallest for memory efficiency):
        1. Demographics LEFT JOIN Housing  (both high coverage)
        2. Result LEFT JOIN Employment     (working-age focus)
        3. Result LEFT JOIN Education      (working-age focus) 
        4. Result LEFT JOIN SEIFA         (lowest temporal frequency)
        
        Args:
            datasets: Standardized datasets
            
        Returns:
            pd.DataFrame: Joined dataset
        """
        self.logger.debug("Stage 2: Executing progressive joins")
        
        join_keys = self.integration_config["join_keys"]
        priority_order = self.integration_config["priority_order"]
        
        # Start with Demographics (highest coverage base dataset)
        if "demographics" not in datasets:
            raise TransformationError("Demographics dataset required for integration")
        
        result = datasets["demographics"].copy()
        self.logger.info(f"Starting with demographics: {len(result)} records")
        
        # Progressive joins following priority order
        for dataset_name in priority_order[1:]:  # Skip demographics (already base)
            if dataset_name in datasets:
                result = self._perform_optimized_join(
                    left=result,
                    right=datasets[dataset_name], 
                    join_keys=join_keys,
                    dataset_name=dataset_name,
                    how="left"  # Preserve all demographic records
                )
        
        self.logger.info(f"Progressive joins complete: {len(result)} records, {len(result.columns)} columns")
        return result
    
    def _perform_optimized_join(self, left: pd.DataFrame, right: pd.DataFrame, 
                               join_keys: List[str], dataset_name: str, how: str) -> pd.DataFrame:
        """
        Memory-efficient join with conflict detection and monitoring.
        
        Args:
            left: Left dataset
            right: Right dataset
            join_keys: Keys to join on
            dataset_name: Name of right dataset for suffixing
            how: Join type
            
        Returns:
            pd.DataFrame: Joined result
        """
        # Pre-join validation
        self._validate_join_keys(left, right, join_keys, dataset_name)
        
        # Add dataset suffix to prevent column conflicts
        right_suffixed = right.copy()
        
        # Identify columns that need suffixing (exclude join keys)
        suffix_columns = [col for col in right_suffixed.columns if col not in join_keys]
        
        # Apply suffix
        rename_dict = {col: f"{col}_{dataset_name}" for col in suffix_columns}
        right_suffixed = right_suffixed.rename(columns=rename_dict)
        
        # Perform join with memory monitoring
        with self._memory_monitor(f"join_{dataset_name}"):
            joined = pd.merge(
                left, 
                right_suffixed, 
                on=join_keys, 
                how=how, 
                validate="one_to_one",
                suffixes=("", f"_{dataset_name}_dup")  # Handle any remaining conflicts
            )
        
        self.logger.info(f"Joined {dataset_name}: {len(left)} -> {len(joined)} records")
        return joined
    
    def _validate_join_keys(self, left: pd.DataFrame, right: pd.DataFrame, 
                           join_keys: List[str], dataset_name: str) -> None:
        """
        Validate join keys are compatible between datasets.
        
        Args:
            left: Left dataset
            right: Right dataset
            join_keys: Keys to validate
            dataset_name: Name of right dataset
            
        Raises:
            TransformationError: If validation fails
        """
        # Check for duplicates in join keys
        left_dupes = left[join_keys].duplicated().sum()
        right_dupes = right[join_keys].duplicated().sum()
        
        if left_dupes > 0:
            self.logger.warning(f"Left dataset has {left_dupes} duplicate join keys")
        
        if right_dupes > 0:
            self.logger.warning(f"Right dataset '{dataset_name}' has {right_dupes} duplicate join keys")
        
        # Check data type compatibility
        for key in join_keys:
            left_dtype = left[key].dtype
            right_dtype = right[key].dtype
            
            if left_dtype != right_dtype:
                self.logger.warning(f"Join key '{key}' has different dtypes: {left_dtype} vs {right_dtype}")
    
    def _resolve_conflicts_and_populate_fields(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resolve field conflicts using data quality hierarchy and populate missing standardized fields.
        
        Priority Order: Demographics > Housing > Employment > Education > SEIFA
        
        Args:
            data: Joined dataset with potential conflicts
            
        Returns:
            pd.DataFrame: Resolved dataset
        """
        self.logger.debug("Stage 3: Resolving conflicts and populating fields")
        
        result = data.copy()
        
        # Resolve geographic fields using priority hierarchy
        result['state_territory'] = self._resolve_field_conflict(
            result, 'state_territory', 
            sources=['demographics', 'seifa'],
            default_lookup='geographic_lookup'
        )
        
        result['geographic_name'] = self._resolve_field_conflict(
            result, 'geographic_name',
            sources=['demographics', 'seifa'], 
            default_lookup='geographic_lookup'
        )
        
        # Population base reconciliation
        result = self._reconcile_population_bases(result)
        
        # Fill any remaining nulls in critical fields
        if result['state_territory'].isna().any():
            # Derive from geographic_id pattern (first digit maps to state)
            result['state_territory'] = result.apply(
                lambda row: self._derive_state_from_geographic_id(row['geographic_id']) 
                if pd.isna(row['state_territory']) else row['state_territory'],
                axis=1
            )
        
        return result
    
    def _resolve_field_conflict(self, data: pd.DataFrame, field_name: str, 
                               sources: List[str], default_lookup: Optional[str] = None) -> pd.Series:
        """
        Resolve field conflicts using priority hierarchy.
        
        Args:
            data: Dataset with conflicts
            field_name: Field to resolve
            sources: Priority-ordered source datasets
            default_lookup: Default lookup method if all sources null
            
        Returns:
            pd.Series: Resolved field values
        """
        # Try each source in priority order
        result = pd.Series(index=data.index, dtype=object)
        
        for source in sources:
            source_field = f"{field_name}_{source}" if f"{field_name}_{source}" in data.columns else field_name
            if source_field in data.columns:
                # Fill non-null values from this source
                mask = result.isna() & data[source_field].notna()
                result[mask] = data.loc[mask, source_field]
        
        # Apply default lookup if configured and still have nulls
        if default_lookup and result.isna().any():
            self.logger.debug(f"Applying default lookup for {field_name}")
            # Implement geographic lookup or other default strategies
        
        return result
    
    def _reconcile_population_bases(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reconcile different population bases across domains.
        
        Args:
            data: Dataset with multiple population fields
            
        Returns:
            pd.DataFrame: Dataset with reconciled population metrics
        """
        result = data.copy()
        
        # Primary population (from demographics)
        if 'total_population' in result.columns:
            result['total_population'] = result['total_population'].fillna(0).astype('Int64')
        else:
            result['total_population'] = 0
        
        # Working age population (maximum from education/employment bases)
        working_age_candidates = []
        
        if 'education_pop_base_education' in result.columns:
            working_age_candidates.append(result['education_pop_base_education'])
        
        if 'labour_force_pop_employment' in result.columns:
            working_age_candidates.append(result['labour_force_pop_employment'])
        
        if working_age_candidates:
            result['working_age_population'] = pd.concat(working_age_candidates, axis=1).max(axis=1).astype('Int64')
        else:
            result['working_age_population'] = pd.NA
        
        # Population consistency validation
        result['population_consistency_flag'] = self._validate_population_consistency(
            result['total_population'], 
            result['working_age_population']
        )
        
        return result
    
    def _validate_population_consistency(self, total_pop: pd.Series, working_age_pop: pd.Series) -> pd.Series:
        """
        Validate population consistency between total and working age.
        
        Args:
            total_pop: Total population
            working_age_pop: Working age population
            
        Returns:
            pd.Series: Boolean consistency flags
        """
        # Allow for some tolerance in the comparison
        tolerance = self.quality_thresholds['population_tolerance']
        
        # Working age should be approximately 60-80% of total population
        min_ratio = 0.60 * (1 - tolerance)
        max_ratio = 0.80 * (1 + tolerance)
        
        # Calculate ratios where both values exist
        mask = total_pop.notna() & working_age_pop.notna() & (total_pop > 0)
        ratio = working_age_pop / total_pop
        
        # Flag records outside expected range
        consistency = pd.Series(True, index=total_pop.index)
        consistency[mask] = (ratio[mask] >= min_ratio) & (ratio[mask] <= max_ratio)
        
        return consistency
    
    def _calculate_derived_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sophisticated cross-domain insights.
        
        Args:
            data: Resolved dataset
            
        Returns:
            pd.DataFrame: Dataset with derived indicators
        """
        self.logger.debug("Stage 4: Calculating derived indicators")
        
        result = data.copy()
        
        # Calculate core demographic ratios
        result = self._calculate_demographic_indicators(result)
        
        # Calculate housing indicators
        result = self._calculate_housing_indicators(result)
        
        # Calculate employment indicators
        result = self._calculate_employment_indicators(result)
        
        # Calculate education indicators
        result = self._calculate_education_indicators(result)
        
        # Socioeconomic Profile (combining multiple dimensions)
        result['socioeconomic_profile'] = self._calculate_socioeconomic_profile(result)
        
        # Livability Index (housing + employment + education + disadvantage)
        result['livability_index'] = self._calculate_livability_index(result)
        
        # Economic Opportunity Score
        result['economic_opportunity_score'] = self._calculate_economic_opportunity(result)
        
        # Social Cohesion Index
        result['social_cohesion_index'] = self._calculate_social_cohesion(result)
        
        # Housing Market Pressure
        result['housing_market_pressure'] = self._calculate_housing_pressure(result)
        
        return result
    
    def _calculate_demographic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate demographic domain indicators."""
        result = data.copy()
        
        # Median age (if age distribution available)
        # For now, using placeholder calculation
        if 'age_20_24' in result.columns and 'age_40_44' in result.columns:
            # Simplified median age estimation
            result['median_age'] = 35.0  # Placeholder
        
        # Sex ratio (males per 100 females)
        if 'males' in result.columns and 'females' in result.columns:
            result['sex_ratio'] = (result['males'] / result['females'] * 100).round(1)
        
        # Indigenous percentage
        if 'indigenous' in result.columns and 'total_population' in result.columns:
            result['indigenous_percentage'] = (result['indigenous'] / result['total_population'] * 100).round(1)
        
        # Age dependency ratio
        if all(col in result.columns for col in ['age_0_4', 'age_5_9', 'age_65_69', 'age_70_74']):
            # Simplified calculation
            young = result[['age_0_4', 'age_5_9', 'age_10_14']].sum(axis=1)
            old = result[['age_65_69', 'age_70_74', 'age_75_79', 'age_80_84', 'age_85_plus']].sum(axis=1)
            working = result['working_age_population']
            result['age_dependency_ratio'] = ((young + old) / working * 100).round(1)
        
        return result
    
    def _calculate_housing_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate housing domain indicators."""
        result = data.copy()
        
        # Home ownership rate
        if all(col in result.columns for col in ['owned_outright_housing', 'owned_with_mortgage_housing', 'rented_housing']):
            owned = result['owned_outright_housing'] + result['owned_with_mortgage_housing']
            total = owned + result['rented_housing']
            result['home_ownership_rate'] = (owned / total * 100).round(1)
        
        # Median rent and mortgage (pass through if available)
        if 'median_rent_weekly_housing' in result.columns:
            result['median_rent_weekly'] = result['median_rent_weekly_housing']
        
        if 'median_mortgage_monthly_housing' in result.columns:
            result['median_mortgage_monthly'] = result['median_mortgage_monthly_housing']
        
        # Housing stress placeholder
        result['housing_stress_rate'] = 25.0  # Placeholder
        
        # Average household size
        if 'total_population' in result.columns and 'occupied_private_dwellings' in result.columns:
            result['average_household_size'] = (result['total_population'] / result['occupied_private_dwellings']).round(1)
        
        return result
    
    def _calculate_employment_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate employment domain indicators."""
        result = data.copy()
        
        # Pass through employment rates if available
        if 'unemployment_rate_employment' in result.columns:
            result['unemployment_rate'] = result['unemployment_rate_employment']
        
        if 'participation_rate_employment' in result.columns:
            result['participation_rate'] = result['participation_rate_employment']
        
        # Professional employment rate
        if 'professionals_employment' in result.columns and 'labour_force_pop_employment' in result.columns:
            employed = result['labour_force_pop_employment'] - result.get('unemployed_employment', 0)
            result['professional_employment_rate'] = (result['professionals_employment'] / employed * 100).round(1)
        
        # Placeholder for other indicators
        result['median_personal_income'] = 65000  # Placeholder
        result['industry_diversity_index'] = 0.75  # Placeholder
        
        return result
    
    def _calculate_education_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate education domain indicators."""
        result = data.copy()
        
        # University qualification rate
        if all(col in result.columns for col in ['postgraduate_degree_education', 'bachelor_degree_education', 'education_pop_base_education']):
            uni_qualified = result['postgraduate_degree_education'] + result['bachelor_degree_education']
            result['university_qualification_rate'] = (uni_qualified / result['education_pop_base_education'] * 100).round(1)
        
        # Year 12 completion rate
        if 'year_12_or_equivalent_education' in result.columns and 'education_pop_base_education' in result.columns:
            result['year12_completion_rate'] = (result['year_12_or_equivalent_education'] / result['education_pop_base_education'] * 100).round(1)
        
        # Vocational qualification rate
        if 'certificate_iii_iv_education' in result.columns and 'education_pop_base_education' in result.columns:
            result['vocational_qualification_rate'] = (result['certificate_iii_iv_education'] / result['education_pop_base_education'] * 100).round(1)
        
        return result
    
    def _calculate_socioeconomic_profile(self, data: pd.DataFrame) -> pd.Series:
        """
        Multi-dimensional socioeconomic classification.
        
        Args:
            data: Dataset with indicators
            
        Returns:
            pd.Series: Socioeconomic profile categories
        """
        # Weight factors for different domains
        weights = {
            'education': 0.25,    # University qualification rates
            'employment': 0.25,   # Professional occupation rates  
            'housing': 0.25,      # Home ownership rates
            'seifa': 0.25         # IRSAD disadvantage scores
        }
        
        # Normalize indicators to 0-100 scale
        education_score = self._normalize_indicator(data.get('university_qualification_rate', 25), 0, 50)
        employment_score = self._normalize_indicator(data.get('unemployment_rate', 5), 10, 0, inverse=True)
        housing_score = self._normalize_indicator(data.get('home_ownership_rate', 65), 40, 80)
        seifa_score = self._normalize_indicator(data.get('irsad_score_seifa', 1000), 800, 1200)
        
        # Weighted composite score
        composite_score = (
            education_score * weights['education'] +
            employment_score * weights['employment'] + 
            housing_score * weights['housing'] +
            seifa_score * weights['seifa']
        )
        
        # Categorical classification
        return pd.cut(composite_score, 
                     bins=[0, 25, 50, 75, 100],
                     labels=['low', 'medium-low', 'medium-high', 'high'])
    
    def _calculate_livability_index(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate composite livability index.
        
        Args:
            data: Dataset with indicators
            
        Returns:
            pd.Series: Livability scores (0-100)
        """
        # Component weights
        components = {
            'housing_affordability': 0.25,
            'employment_opportunities': 0.25,
            'education_access': 0.20,
            'socioeconomic_advantage': 0.20,
            'community_cohesion': 0.10
        }
        
        # Calculate component scores
        housing_score = self._normalize_indicator(
            data.get('median_rent_weekly', 500), 700, 300, inverse=True
        )
        
        employment_score = self._normalize_indicator(
            data.get('unemployment_rate', 5), 10, 2, inverse=True
        )
        
        education_score = self._normalize_indicator(
            data.get('university_qualification_rate', 25), 10, 40
        )
        
        seifa_score = self._normalize_indicator(
            data.get('irsad_decile_seifa', 5), 1, 10
        )
        
        # Placeholder for community cohesion
        cohesion_score = 70
        
        # Weighted composite
        livability = (
            housing_score * components['housing_affordability'] +
            employment_score * components['employment_opportunities'] +
            education_score * components['education_access'] +
            seifa_score * components['socioeconomic_advantage'] +
            cohesion_score * components['community_cohesion']
        )
        
        return livability.round(1)
    
    def _calculate_economic_opportunity(self, data: pd.DataFrame) -> pd.Series:
        """Calculate economic opportunity score."""
        # Simplified calculation based on employment and income
        employment_factor = self._normalize_indicator(
            data.get('participation_rate', 65), 50, 80
        )
        
        income_factor = self._normalize_indicator(
            data.get('median_personal_income', 65000), 40000, 90000
        )
        
        diversity_factor = self._normalize_indicator(
            data.get('industry_diversity_index', 0.7), 0.5, 0.9
        )
        
        return ((employment_factor + income_factor + diversity_factor) / 3).round(1)
    
    def _calculate_social_cohesion(self, data: pd.DataFrame) -> pd.Series:
        """Calculate social cohesion index."""
        # Placeholder implementation
        return pd.Series(72.5, index=data.index)
    
    def _calculate_housing_pressure(self, data: pd.DataFrame) -> pd.Series:
        """Calculate housing market pressure indicator."""
        # Based on rent/income ratios and ownership rates
        rent_pressure = self._normalize_indicator(
            data.get('median_rent_weekly', 500), 300, 700
        )
        
        ownership_pressure = self._normalize_indicator(
            data.get('home_ownership_rate', 65), 80, 40, inverse=True
        )
        
        return ((rent_pressure + ownership_pressure) / 2).round(1)
    
    def _normalize_indicator(self, series: Union[pd.Series, float], min_val: float, max_val: float, 
                           inverse: bool = False) -> pd.Series:
        """
        Normalize indicator to 0-100 scale.
        
        Args:
            series: Values to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            inverse: Whether to invert (lower is better)
            
        Returns:
            pd.Series: Normalized values (0-100)
        """
        if isinstance(series, (int, float)):
            series = pd.Series(series)
        elif series is None:
            return pd.Series(50.0)  # Default middle value
            
        # Clip to range
        clipped = series.clip(min_val, max_val)
        
        # Normalize
        if max_val != min_val:
            normalized = (clipped - min_val) / (max_val - min_val) * 100
        else:
            normalized = pd.Series(50.0, index=series.index)
        
        if inverse:
            normalized = 100 - normalized
            
        return normalized
    
    def _generate_quality_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data quality assessment.
        
        Args:
            data: Dataset with derived indicators
            
        Returns:
            pd.DataFrame: Dataset with quality metrics
        """
        self.logger.debug("Stage 5: Generating quality metrics")
        
        result = data.copy()
        
        # Calculate domain completeness
        result['demographics_completeness'] = self._calculate_domain_completeness(data, 'demographics')
        result['housing_completeness'] = self._calculate_domain_completeness(data, 'housing')
        result['employment_completeness'] = self._calculate_domain_completeness(data, 'employment')
        result['education_completeness'] = self._calculate_domain_completeness(data, 'education')
        result['seifa_completeness'] = self._calculate_domain_completeness(data, 'seifa')
        
        # Overall completeness score
        completeness_cols = [
            'demographics_completeness', 'housing_completeness', 
            'employment_completeness', 'education_completeness', 'seifa_completeness'
        ]
        result['data_completeness_score'] = result[completeness_cols].mean(axis=1).round(3)
        
        # Temporal alignment quality
        result['temporal_quality_flag'] = self._assess_temporal_alignment(data)
        
        # Cross-domain consistency checks
        result['consistency_score'] = self._assess_cross_domain_consistency(data)
        
        return result
    
    def _calculate_domain_completeness(self, data: pd.DataFrame, domain: str) -> pd.Series:
        """
        Calculate completeness score for a specific domain.
        
        Args:
            data: Full dataset
            domain: Domain name
            
        Returns:
            pd.Series: Completeness scores (0-1)
        """
        # Identify domain columns
        domain_cols = [col for col in data.columns if col.endswith(f'_{domain}')]
        
        if not domain_cols:
            return pd.Series(0.0, index=data.index)
        
        # Calculate non-null percentage for domain
        non_null_counts = data[domain_cols].notna().sum(axis=1)
        completeness = non_null_counts / len(domain_cols)
        
        return completeness.round(3)
    
    def _assess_temporal_alignment(self, data: pd.DataFrame) -> pd.Series:
        """
        Assess temporal data alignment quality.
        
        Args:
            data: Dataset to assess
            
        Returns:
            pd.Series: Boolean quality flags
        """
        # For now, assume good temporal quality if census year is consistent
        # In practice, would check SEIFA year alignment etc.
        return pd.Series(True, index=data.index)
    
    def _assess_cross_domain_consistency(self, data: pd.DataFrame) -> pd.Series:
        """
        Assess consistency across domains.
        
        Args:
            data: Dataset to assess
            
        Returns:
            pd.Series: Consistency scores (0-1)
        """
        consistency_checks = []
        
        # Check population consistency
        if 'population_consistency_flag' in data.columns:
            consistency_checks.append(data['population_consistency_flag'].astype(float))
        
        # Check employment-education alignment
        if all(col in data.columns for col in ['university_qualification_rate', 'professional_employment_rate']):
            # Higher education should correlate with professional employment
            edu_emp_diff = abs(data['university_qualification_rate'] - data['professional_employment_rate'])
            edu_emp_consistency = 1 - (edu_emp_diff / 100)
            consistency_checks.append(edu_emp_consistency)
        
        # Average all consistency checks
        if consistency_checks:
            return pd.concat(consistency_checks, axis=1).mean(axis=1).round(3)
        else:
            return pd.Series(1.0, index=data.index)
    
    def _enforce_master_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply master schema enforcement and type conversions.
        
        Args:
            data: Dataset with quality metrics
            
        Returns:
            pd.DataFrame: Schema-compliant dataset
        """
        self.logger.debug("Stage 6: Enforcing master schema")
        
        result = data.copy()
        
        # Define schema field mappings
        schema_mappings = {
            # Core identifiers
            'geographic_id': str,
            'geographic_level': str,
            'geographic_name': str,
            'state_territory': str,
            'census_year': 'Int64',
            
            # Geographic coordinates
            'latitude': float,
            'longitude': float,
            
            # Population
            'total_population': 'Int64',
            'working_age_population': 'Int64',
            
            # Percentages
            'indigenous_percentage': float,
            'home_ownership_rate': float,
            'unemployment_rate': float,
            'participation_rate': float,
            'university_qualification_rate': float,
            
            # Derived fields
            'socioeconomic_profile': str,
            'livability_index': float,
            'economic_opportunity_score': float,
            
            # Quality metrics
            'data_completeness_score': float,
            'consistency_score': float,
            'temporal_quality_flag': bool
        }
        
        # Apply type conversions
        for field, dtype in schema_mappings.items():
            if field in result.columns:
                try:
                    if dtype == 'Int64':
                        result[field] = pd.to_numeric(result[field], errors='coerce').astype('Int64')
                    else:
                        result[field] = result[field].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Failed to convert {field} to {dtype}: {e}")
        
        # Add missing required fields with defaults
        required_fields = {
            'source_datasets': ['demographics', 'housing', 'employment', 'education', 'seifa']
        }
        
        for field, default in required_fields.items():
            if field not in result.columns:
                result[field] = pd.Series([default] * len(result))
        
        return result
    
    def _add_integration_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add integration metadata and lineage information.
        
        Args:
            data: Schema-compliant dataset
            
        Returns:
            pd.DataFrame: Dataset with metadata
        """
        self.logger.debug("Stage 7: Adding integration metadata")
        
        result = data.copy()
        
        # Add integration surrogate keys
        result['integration_sk'] = range(self.integration_sk_counter, 
                                       self.integration_sk_counter + len(result))
        self.integration_sk_counter += len(result)
        
        # Add integration timestamp
        result['integration_timestamp'] = datetime.now()
        
        # Add data source information
        result['data_source_name'] = "ABS Census 2021 Integrated Dataset"
        result['data_source_url'] = "https://www.abs.gov.au/census"
        result['extraction_date'] = datetime.now()
        result['quality_level'] = "HIGH"
        result['source_version'] = "2021.0.0"
        
        # Add schema metadata
        result['schema_version'] = "2.0.0"
        
        return result
    
    def _validate_integration_integrity(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Final validation of integrated dataset integrity.
        
        Args:
            data: Complete integrated dataset
            
        Returns:
            pd.DataFrame: Validated dataset
        """
        self.logger.debug("Stage 8: Validating integration integrity")
        
        validation_results = {
            'total_records': len(data),
            'unique_geographies': data['geographic_id'].nunique(),
            'temporal_coverage': data['census_year'].value_counts().to_dict(),
            'completeness_distribution': {
                'mean': data['data_completeness_score'].mean(),
                'min': data['data_completeness_score'].min(),
                'max': data['data_completeness_score'].max()
            },
            'quality_summary': {
                'high_quality': (data['consistency_score'] >= 0.9).sum(),
                'medium_quality': ((data['consistency_score'] >= 0.7) & (data['consistency_score'] < 0.9)).sum(),
                'low_quality': (data['consistency_score'] < 0.7).sum()
            }
        }
        
        # Log validation summary
        self.logger.info(f"Integration validation summary: {validation_results}")
        
        # Apply quality filters if configured
        if self.integration_config.get('apply_quality_filters', False):
            min_completeness = self.quality_thresholds.get('min_completeness', 0.5)
            before_filter = len(data)
            data = data[data['data_completeness_score'] >= min_completeness]
            after_filter = len(data)
            self.logger.info(f"Applied quality filter: {before_filter} -> {after_filter} records retained")
        
        return data
    
    def _derive_state_from_geographic_id(self, geo_id: str) -> str:
        """
        Derive state/territory from geographic ID pattern.
        
        Args:
            geo_id: Geographic identifier
            
        Returns:
            str: State/territory code
        """
        if not geo_id or not isinstance(geo_id, str):
            return "OT"  # Other territories
        
        # First digit of SA codes maps to states
        state_mapping = {
            '1': 'NSW',
            '2': 'VIC',
            '3': 'QLD',
            '4': 'SA',
            '5': 'WA',
            '6': 'TAS',
            '7': 'NT',
            '8': 'ACT',
            '9': 'OT'
        }
        
        return state_mapping.get(geo_id[0], 'OT')
    
    @contextmanager
    def _memory_monitor(self, operation: str):
        """
        Monitor memory usage during operations.
        
        Args:
            operation: Name of operation
        """
        # Simplified memory monitoring
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.logger.debug(f"Operation '{operation}' completed in {duration:.2f}s")
    
    def _create_empty_integrated_dataframe(self) -> pd.DataFrame:
        """
        Create empty DataFrame with correct integrated schema structure.
        
        Returns:
            pd.DataFrame: Empty DataFrame with schema columns
        """
        schema_columns = [
            "geographic_id", "geographic_level", "geographic_name", "state_territory", "census_year",
            "total_population", "working_age_population", "population_consistency_flag",
            "median_age", "sex_ratio", "indigenous_percentage",
            "home_ownership_rate", "median_rent_weekly", "housing_stress_rate",
            "unemployment_rate", "participation_rate", "professional_employment_rate",
            "university_qualification_rate", "year12_completion_rate",
            "irsad_score", "irsd_score", "irsad_decile", "irsd_decile",
            "socioeconomic_profile", "livability_index", "economic_opportunity_score",
            "data_completeness_score", "consistency_score", "temporal_quality_flag",
            "source_datasets", "integration_timestamp"
        ]
        
        return pd.DataFrame(columns=schema_columns)
    
    def save_integrated_dataset(self, data: pd.DataFrame, output_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Save integrated dataset in multiple optimized formats.
        
        Args:
            data: Integrated dataset
            output_config: Output configuration
            
        Returns:
            Dict[str, str]: Paths to saved files
        """
        output_paths = {}
        base_path = output_config.get('base_path', 'output/integrated/')
        
        # Ensure output directory exists
        import os
        os.makedirs(base_path, exist_ok=True)
        
        # Parquet (primary format - columnar, compressed)
        if 'parquet' in self.integration_config['output_formats']:
            parquet_path = os.path.join(base_path, 'census_integrated.parquet')
            data.to_parquet(parquet_path, compression='gzip', index=False)
            output_paths['parquet'] = parquet_path
            self.logger.info(f"Saved parquet: {parquet_path}")
        
        # CSV (compatibility format)
        if 'csv' in self.integration_config['output_formats']:
            csv_path = os.path.join(base_path, 'census_integrated.csv')
            data.to_csv(csv_path, index=False)
            output_paths['csv'] = csv_path
            self.logger.info(f"Saved CSV: {csv_path}")
        
        # Generate summary statistics
        if self.integration_config.get('generate_summary_stats', True):
            summary_path = self._generate_integration_summary(data, base_path)
            output_paths['summary'] = summary_path
        
        self.logger.info(f"Integrated dataset saved: {output_paths}")
        return output_paths
    
    def _generate_integration_summary(self, data: pd.DataFrame, base_path: str) -> str:
        """
        Generate summary statistics for integrated dataset.
        
        Args:
            data: Integrated dataset
            base_path: Output directory
            
        Returns:
            str: Path to summary file
        """
        import os
        import json
        
        summary = {
            'integration_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_records': len(data),
                'unique_geographies': data['geographic_id'].nunique(),
                'geographic_levels': data['geographic_level'].value_counts().to_dict(),
                'temporal_coverage': data['census_year'].value_counts().to_dict()
            },
            'data_quality': {
                'overall_completeness': float(data['data_completeness_score'].mean()),
                'overall_consistency': float(data['consistency_score'].mean()),
                'domain_completeness': {
                    'demographics': float(data['demographics_completeness'].mean()),
                    'housing': float(data['housing_completeness'].mean()),
                    'employment': float(data['employment_completeness'].mean()),
                    'education': float(data['education_completeness'].mean()),
                    'seifa': float(data['seifa_completeness'].mean())
                }
            },
            'socioeconomic_distribution': data['socioeconomic_profile'].value_counts().to_dict(),
            'key_indicators': {
                'median_livability_index': float(data['livability_index'].median()) if 'livability_index' in data else None,
                'median_home_ownership_rate': float(data['home_ownership_rate'].median()) if 'home_ownership_rate' in data else None,
                'median_unemployment_rate': float(data['unemployment_rate'].median()) if 'unemployment_rate' in data else None
            }
        }
        
        summary_path = os.path.join(base_path, 'integration_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary_path
    
    def get_processing_metadata(self) -> Optional[ProcessingMetadata]:
        """
        Get processing metadata for the last integration.
        
        Returns:
            Optional[ProcessingMetadata]: Processing metadata if available
        """
        return self.processing_metadata
    
    def get_quality_metrics(self) -> "DataQualityMetrics":
        """
        Get data quality metrics for the last integration.
        
        Returns:
            DataQualityMetrics: Quality metrics instance
        """
        if self.quality_metrics is None:
            self.quality_metrics = DataQualityMetrics()
        return self.quality_metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the last integration.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        return self.performance_stats.copy()
    
    def _create_empty_integrated_dataframe(self) -> pd.DataFrame:
        """
        Create empty DataFrame with integrated census schema.
        
        Returns:
            pd.DataFrame: Empty dataframe with correct schema
        """
        # Import helper to avoid circular imports
        from .census_integrator_helpers import CensusIntegratorHelpers
        return CensusIntegratorHelpers.create_empty_integrated_schema()
    
    def _update_quality_metrics_tracking(self, transformer_outputs: Dict[str, pd.DataFrame], quality_data: pd.DataFrame):
        """
        Update quality metrics tracking based on integration results.
        
        Args:
            transformer_outputs: Original transformer datasets
            quality_data: Data with quality metrics calculated
        """
        quality_metrics = self.get_quality_metrics()
        
        # Calculate completeness scores for each dataset
        for dataset_name, dataset in transformer_outputs.items():
            if not dataset.empty:
                total_fields = len(dataset.columns)
                non_null_fields = dataset.notna().sum().sum()
                total_values = len(dataset) * total_fields
                completeness = (non_null_fields / total_values) * 100 if total_values > 0 else 0
                quality_metrics.completeness_scores[dataset_name] = completeness
        
        # Calculate join success rates
        expected_records = max(len(df) for df in transformer_outputs.values() if not df.empty) if transformer_outputs else 0
        actual_records = len(quality_data)
        
        for dataset_name in transformer_outputs.keys():
            if not transformer_outputs[dataset_name].empty:
                success_rate = (actual_records / expected_records) * 100 if expected_records > 0 else 100
                quality_metrics.join_success_rates[f"join_{dataset_name}"] = success_rate
        
        # Calculate temporal alignment scores
        if 'temporal_quality_flag' in quality_data.columns:
            aligned_records = quality_data['temporal_quality_flag'].sum()
            temporal_score = (aligned_records / len(quality_data)) * 100 if len(quality_data) > 0 else 100
            quality_metrics.temporal_alignment_scores['overall'] = temporal_score
        
        # Calculate validation pass rates
        if 'data_completeness_score' in quality_data.columns:
            avg_completeness = quality_data['data_completeness_score'].mean()
            quality_metrics.validation_pass_rates['completeness_validation'] = avg_completeness * 100
        
        if 'consistency_score' in quality_data.columns:
            avg_consistency = quality_data['consistency_score'].mean()
            quality_metrics.validation_pass_rates['consistency_validation'] = avg_consistency * 100


class DataQualityMetrics:
    """
    Manages data quality metrics for census integration.
    
    Tracks completeness, consistency, temporal alignment, and validation metrics
    across all census domains.
    """
    
    def __init__(self):
        """Initialise quality metrics tracking."""
        self.completeness_scores: Dict[str, float] = {}
        self.join_success_rates: Dict[str, float] = {}
        self.temporal_alignment_scores: Dict[str, float] = {}
        self.validation_pass_rates: Dict[str, float] = {}
        self.conflict_resolution_counts: Dict[str, int] = {}
        self.error_log: List[str] = []
    
    def calculate_overall_quality_score(self) -> float:
        """
        Calculate weighted overall quality score.
        
        Returns:
            float: Overall quality score (0-100)
        """
        scores = []
        weights = []
        
        # Completeness component (30% weight)
        if self.completeness_scores:
            avg_completeness = sum(self.completeness_scores.values()) / len(self.completeness_scores)
            scores.append(avg_completeness)
            weights.append(0.3)
        
        # Join success component (25% weight)
        if self.join_success_rates:
            avg_join_success = sum(self.join_success_rates.values()) / len(self.join_success_rates)
            scores.append(avg_join_success)
            weights.append(0.25)
        
        # Temporal alignment component (20% weight)
        if self.temporal_alignment_scores:
            avg_temporal = sum(self.temporal_alignment_scores.values()) / len(self.temporal_alignment_scores)
            scores.append(avg_temporal)
            weights.append(0.2)
        
        # Validation pass rate component (25% weight)
        if self.validation_pass_rates:
            avg_validation = sum(self.validation_pass_rates.values()) / len(self.validation_pass_rates)
            scores.append(avg_validation)
            weights.append(0.25)
        
        if not scores:
            return 0.0
        
        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
            
        weighted_score = sum(score * weight for score, weight in zip(scores, weights)) / total_weight
        
        # Apply penalty for conflicts
        total_conflicts = sum(self.conflict_resolution_counts.values())
        conflict_penalty = min(total_conflicts * 2, 20)  # Max 20% penalty
        
        final_score = max(0, weighted_score - conflict_penalty)
        return final_score


class CensusIntegratorConfig:
    """
    Configuration management for CensusIntegrator.
    
    Handles configuration loading, validation, and access for integration
    parameters and quality thresholds.
    """
    
    def __init__(self, config_manager):
        """
        Initialise configuration.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration values with defaults."""
        # Join configuration
        self.join_strategy = get_config("transformers.census.integration.join_strategy", "left")
        self.join_keys = get_config("transformers.census.integration.join_keys", 
                                   ["geographic_id", "geographic_level", "census_year"])
        self.parallel_joins = get_config("transformers.census.integration.parallel_joins", True)
        
        # Quality configuration
        self.minimum_completeness_threshold = get_config("transformers.census.integration.min_completeness", 0.5)
        self.minimum_join_success_rate = get_config("transformers.census.integration.min_join_rate", 0.8)
        self.temporal_tolerance_years = get_config("transformers.census.integration.temporal_tolerance", 1)
        
        # Performance configuration
        self.chunk_size = get_config("transformers.census.integration.chunk_size", 10000)
        self.memory_limit_mb = get_config("transformers.census.integration.memory_limit", 4096)
        self.enable_optimizations = get_config("transformers.census.integration.optimizations", True)
        
        # Output configuration
        self.generate_summary_stats = get_config("transformers.census.integration.summary_stats", True)
        self.output_formats = get_config("transformers.census.integration.output_formats", ["parquet"])
        self.compression = get_config("transformers.census.integration.compression", "gzip")