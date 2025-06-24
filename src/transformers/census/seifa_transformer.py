"""
ABS SEIFA socio-economic index data transformer for AHGD ETL pipeline.

This module transforms raw Australian Bureau of Statistics SEIFA (Socio-Economic Indexes for Areas)
data into standardised CensusSEIFA schema format. Handles IRSAD, IRSD, IER, and IEO indices
with score standardisation, ranking generation, and composite index creation.
"""

import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats

from ...utils.logging import get_logger
from ...utils.config import get_config, get_config_manager
from ...utils.interfaces import (
    ProcessingMetadata,
    ProcessingStatus,
    TransformationError,
)
from schemas.census_schema import CensusSEIFA


class SEIFATransformer:
    """
    Transforms raw ABS SEIFA data to CensusSEIFA schema.
    
    Handles SEIFA 2021 index processing with:
    - Score standardisation and normalisation
    - Ranking generation (national, state, regional)
    - Composite index creation with weighted components
    - Decile and percentile grouping
    - Geographic concordance and hierarchy integration
    """
    
    def __init__(self):
        """
        Initialise the SEIFA transformer.
        
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
        self.imputation_strategy = get_config("transformers.census.impute_missing", "score_median")
        
        # State management for processing
        self.seifa_sk_counter = 40000  # Start SEIFA surrogates at 40K
        self.processing_metadata: Optional[ProcessingMetadata] = None
        
        # Error handling configuration
        self.stop_on_error = get_config("system.stop_on_error", False)
        
        # SEIFA-specific configuration
        self.reference_population = self.operations_config.get("reference_population", "australia")
        self.normalisation_method = self.operations_config.get("normalisation", "z_score")
        self.composite_weights = self.operations_config.get("weights", {
            "economic": 0.4,
            "education": 0.3,
            "housing": 0.2,
            "accessibility": 0.1
        })
        
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
        
        Maps raw SEIFA column names to standardised field names with
        priority-based fallback for ABS naming variations across years.
        
        Returns:
            Dict[str, List[str]]: Mapping of target fields to source column candidates
        """
        # Default column mappings for ABS SEIFA data
        # Priority order: most recent ABS format first
        default_mappings = {
            # Geographic identification
            "geographic_id": ["SA2_CODE_2021", "SA2_MAIN21", "SA2_CODE", "sa2_code"],
            "geographic_name": ["SA2_NAME_2021", "SA2_NAME21", "SA2_NAME", "sa2_name"],
            "state_territory": ["STATE_CODE_2021", "STE_CODE21", "STATE_CODE", "state_code"],
            
            # SEIFA index scores
            "irsad_score": ["IRSAD_Score", "IRSAD_2021", "irsad_score"],
            "irsd_score": ["IRSD_Score", "IRSD_2021", "irsd_score"],
            "ier_score": ["IER_Score", "IER_2021", "ier_score"],
            "ieo_score": ["IEO_Score", "IEO_2021", "ieo_score"],
            
            # National rankings
            "irsad_rank": ["IRSAD_Rank", "IRSAD_Rank_Aust", "irsad_rank"],
            "irsd_rank": ["IRSD_Rank", "IRSD_Rank_Aust", "irsd_rank"],
            "ier_rank": ["IER_Rank", "IER_Rank_Aust", "ier_rank"],
            "ieo_rank": ["IEO_Rank", "IEO_Rank_Aust", "ieo_rank"],
            
            # Deciles
            "irsad_decile": ["IRSAD_Decile", "IRSAD_Decile_Aust", "irsad_decile"],
            "irsd_decile": ["IRSD_Decile", "IRSD_Decile_Aust", "irsd_decile"],
            "ier_decile": ["IER_Decile", "IER_Decile_Aust", "ier_decile"],
            "ieo_decile": ["IEO_Decile", "IEO_Decile_Aust", "ieo_decile"],
            
            # State rankings and deciles
            "irsad_state_rank": ["IRSAD_Rank_State", "irsad_state_rank"],
            "irsd_state_rank": ["IRSD_Rank_State", "irsd_state_rank"],
            "irsad_state_decile": ["IRSAD_Decile_State", "irsad_state_decile"],
            "irsd_state_decile": ["IRSD_Decile_State", "irsd_state_decile"],
            
            # Population data
            "population_base": ["Population", "Tot_P_P", "population_base", "usual_resident_population"],
        }
        
        # Allow configuration override
        config_mappings = get_config("transformers.census.column_mappings", {})
        if config_mappings:
            default_mappings.update(config_mappings)
            
        return default_mappings
    
    def _load_target_schema(self) -> Dict[str, Any]:
        """
        Load target schema configuration.
        
        Returns:
            Dict[str, Any]: Target schema configuration
        """
        return get_config("schemas.census_seifa", {
            "required_indices": ["irsd_score"],  # IRSD is minimum requirement
            "optional_indices": ["irsad_score", "ier_score", "ieo_score"],
            "score_range": {"min": 1, "max": 2000},
            "typical_range": {"min": 600, "max": 1400},
            "decile_range": {"min": 1, "max": 10}
        })
    
    def _load_operations_config(self) -> Dict[str, Any]:
        """
        Load operations configuration for SEIFA processing.
        
        Returns:
            Dict[str, Any]: Operations configuration
        """
        default_config = {
            "normalisation": "z_score",
            "reference_population": "australia",
            "weights": {
                "economic": 0.4,
                "education": 0.3,
                "housing": 0.2,
                "accessibility": 0.1
            },
            "ranking_levels": ["national", "state", "regional"],
            "percentile_groups": [10, 25, 50, 75, 90],
            "composite_indices": True,
            "geographic_concordance": True,
            "include_state_rankings": True
        }
        
        config_ops = get_config("transformers.census.operations", {})
        if config_ops:
            default_config.update(config_ops)
            
        return default_config
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw SEIFA data to standardised CensusSEIFA schema format.
        
        Args:
            data: Raw SEIFA data from ABS
            
        Returns:
            pd.DataFrame: Transformed data conforming to CensusSEIFA schema
            
        Raises:
            TransformationError: If transformation fails
        """
        start_time = time.time()
        
        try:
            # Initialise processing metadata
            self.processing_metadata = ProcessingMetadata(
                operation_id=f"seifa_transform_{int(time.time())}",
                operation_type="SEIFATransformer",
                status=ProcessingStatus.RUNNING,
                start_time=datetime.now(),
                records_processed=len(data)
            )
            
            self.logger.info(f"Starting SEIFA transformation with {len(data)} records")
            
            # Stage 1: Validate input data
            validated_data = self._validate_input_data(data)
            
            # Stage 2: Map columns to standard names
            mapped_data = self._map_columns(validated_data)
            
            # Stage 3: Standardise geographic identifiers
            geo_standardised = self._standardise_geographic_data(mapped_data)
            
            # Stage 4: Process SEIFA scores and normalisation
            score_processed = self._process_seifa_scores(geo_standardised)
            
            # Stage 5: Generate rankings and deciles
            ranked_data = self._generate_rankings_and_deciles(score_processed)
            
            # Stage 6: Create composite indices
            composite_data = self._create_composite_indices(ranked_data)
            
            # Stage 7: Add geographic hierarchy integration
            hierarchy_data = self._integrate_geographic_hierarchy(composite_data)
            
            # Stage 8: Handle missing values
            imputed_data = self._impute_missing_values(hierarchy_data)
            
            # Stage 9: Enforce target schema
            final_data = self._enforce_schema(imputed_data)
            
            # Update processing metadata
            self.processing_metadata.end_time = datetime.now()
            self.processing_metadata.status = ProcessingStatus.COMPLETED
            self.processing_metadata.records_processed = len(final_data)
            self.processing_metadata.duration_seconds = time.time() - start_time
            
            self.logger.info(f"SEIFA transformation completed: {len(final_data)} records in {self.processing_metadata.duration_seconds:.2f}s")
            
            return final_data
            
        except Exception as e:
            # Update metadata with error status
            if self.processing_metadata:
                self.processing_metadata.status = ProcessingStatus.FAILED
                self.processing_metadata.end_time = datetime.now()
                self.processing_metadata.duration_seconds = time.time() - start_time
                self.processing_metadata.error_message = str(e)
                self.processing_metadata.records_failed += 1
            
            error_msg = f"SEIFA transformation failed: {str(e)}"
            self.logger.error(error_msg)
            
            if self.stop_on_error:
                raise TransformationError(error_msg) from e
            else:
                # Return empty DataFrame with correct schema
                return self._create_empty_schema_dataframe()
    
    def _validate_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate input SEIFA data quality and structure.
        
        Args:
            data: Raw input data
            
        Returns:
            pd.DataFrame: Validated data
        """
        self.logger.debug("Stage 1: Validating input SEIFA data")
        
        if data.empty:
            raise TransformationError("Input SEIFA data is empty")
        
        # Check for required geographic columns
        required_geo_cols = ["geographic_id", "SA2_CODE_2021", "SA2_MAIN21", "SA2_CODE"]
        geo_col_found = any(col in data.columns for col in required_geo_cols)
        
        if not geo_col_found:
            available_cols = list(data.columns)
            raise TransformationError(f"No geographic identifier found. Available columns: {available_cols}")
        
        # Check for at least one SEIFA index
        seifa_score_cols = [
            "IRSAD_Score", "IRSD_Score", "IER_Score", "IEO_Score",
            "irsad_score", "irsd_score", "ier_score", "ieo_score"
        ]
        seifa_col_found = any(col in data.columns for col in seifa_score_cols)
        
        if not seifa_col_found:
            raise TransformationError("No SEIFA index scores found in data")
        
        self.logger.debug(f"Input validation completed: {len(data)} records")
        return data.copy()
    
    def _map_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Map raw column names to standardised field names.
        
        Args:
            data: Input data with raw column names
            
        Returns:
            pd.DataFrame: Data with standardised column names
        """
        self.logger.debug("Stage 2: Mapping columns to standard names")
        
        mapped_data = data.copy()
        mapping_log = []
        
        for target_field, source_candidates in self.column_mappings.items():
            mapped_column = None
            
            # Find first matching source column
            for candidate in source_candidates:
                if candidate in mapped_data.columns:
                    mapped_column = candidate
                    break
            
            if mapped_column:
                if mapped_column != target_field:
                    mapped_data = mapped_data.rename(columns={mapped_column: target_field})
                    mapping_log.append(f"{mapped_column} -> {target_field}")
            else:
                # Field not found - this is okay for optional fields
                self.logger.debug(f"Optional field {target_field} not found in source data")
        
        if mapping_log:
            self.logger.debug(f"Column mappings applied: {mapping_log}")
        
        return mapped_data
    
    def _standardise_geographic_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardise geographic identifiers and names.
        
        Args:
            data: Data with mapped column names
            
        Returns:
            pd.DataFrame: Data with standardised geographic fields
        """
        self.logger.debug("Stage 3: Standardising geographic data")
        
        result = data.copy()
        
        # Ensure geographic_id is string
        if "geographic_id" in result.columns:
            result["geographic_id"] = result["geographic_id"].astype(str)
        
        # Standardise state territory codes
        if "state_territory" in result.columns:
            result["state_territory"] = result["state_territory"].str.upper()
        
        # Set geographic level based on the data type
        if "geographic_level" not in result.columns:
            # Infer from geographic_id length (SA2 codes are typically 9 digits)
            if "geographic_id" in result.columns:
                id_lengths = result["geographic_id"].str.len()
                if id_lengths.mode().iloc[0] == 9:
                    result["geographic_level"] = "SA2"
                elif id_lengths.mode().iloc[0] == 6:
                    result["geographic_level"] = "SA3"
                elif id_lengths.mode().iloc[0] == 3:
                    result["geographic_level"] = "SA4"
                else:
                    result["geographic_level"] = "SA2"  # Default assumption
        
        # Set census year if not present
        if "census_year" not in result.columns:
            result["census_year"] = 2021  # Default to 2021 SEIFA
        
        return result
    
    def _process_seifa_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and normalise SEIFA scores.
        
        Args:
            data: Data with standardised geographic fields
            
        Returns:
            pd.DataFrame: Data with processed SEIFA scores
        """
        self.logger.debug("Stage 4: Processing SEIFA scores")
        
        result = data.copy()
        
        # SEIFA score columns to process
        score_columns = ["irsad_score", "irsd_score", "ier_score", "ieo_score"]
        
        for col in score_columns:
            if col in result.columns:
                # Convert to numeric, handling any non-numeric values
                result[col] = pd.to_numeric(result[col], errors="coerce")
                
                # Validate score ranges
                if result[col].notna().any():
                    min_score = result[col].min()
                    max_score = result[col].max()
                    
                    if min_score < 1 or max_score > 2000:
                        self.logger.warning(f"{col} has values outside expected range (1-2000): {min_score}-{max_score}")
                    
                    # Apply normalisation if configured
                    if self.normalisation_method == "z_score":
                        # Calculate z-scores only for non-null values
                        valid_mask = result[col].notna()
                        result[f"{col}_normalised"] = pd.Series(index=result.index, dtype='float64')
                        if valid_mask.sum() > 1:  # Need at least 2 values for z-score
                            result.loc[valid_mask, f"{col}_normalised"] = stats.zscore(result.loc[valid_mask, col])
        
        return result
    
    def _generate_rankings_and_deciles(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate rankings and deciles for SEIFA indices.
        
        Args:
            data: Data with processed scores
            
        Returns:
            pd.DataFrame: Data with rankings and deciles
        """
        self.logger.debug("Stage 5: Generating rankings and deciles")
        
        result = data.copy()
        
        # Score to ranking/decile mappings
        score_rank_mappings = [
            ("irsad_score", "irsad_rank", "irsad_decile"),
            ("irsd_score", "irsd_rank", "irsd_decile"),
            ("ier_score", "ier_rank", "ier_decile"),
            ("ieo_score", "ieo_rank", "ieo_decile")
        ]
        
        for score_col, rank_col, decile_col in score_rank_mappings:
            if score_col in result.columns and result[score_col].notna().any():
                # Generate national rankings (1 = lowest score/most disadvantaged)
                if rank_col not in result.columns:
                    result[rank_col] = result[score_col].rank(method="min", ascending=True)
                    result[rank_col] = result[rank_col].astype("Int64")  # Nullable integer
                
                # Generate deciles (1 = most disadvantaged, 10 = most advantaged)
                if decile_col not in result.columns:
                    result[decile_col] = pd.qcut(
                        result[score_col].rank(method="first"),
                        q=10,
                        labels=range(1, 11)
                    ).astype("Int64")
        
        # Generate state-level rankings if requested
        if self.operations_config.get("include_state_rankings", True) and "state_territory" in result.columns:
            self._generate_state_rankings(result)
        
        return result
    
    def _generate_state_rankings(self, data: pd.DataFrame) -> None:
        """
        Generate state-level rankings and deciles.
        
        Args:
            data: DataFrame to modify in-place
        """
        # Generate state rankings for IRSAD and IRSD (most commonly used)
        for score_col, state_rank_col, state_decile_col in [
            ("irsad_score", "irsad_state_rank", "irsad_state_decile"),
            ("irsd_score", "irsd_state_rank", "irsd_state_decile")
        ]:
            if score_col in data.columns and data[score_col].notna().any():
                # State rankings
                state_ranks = data.groupby("state_territory")[score_col].rank(
                    method="min", ascending=True
                )
                data[state_rank_col] = state_ranks.astype("Int64")
                
                # State deciles - handle cases where there might not be enough data for deciles
                try:
                    state_deciles = data.groupby("state_territory")[score_col].apply(
                        lambda x: pd.qcut(x.rank(method="first"), q=min(10, len(x)), labels=False) + 1
                    ).reset_index(level=0, drop=True)
                    data[state_decile_col] = state_deciles.astype("Int64")
                except ValueError:
                    # If we can't create deciles (e.g., too few unique values), use simple ranking
                    normalized_ranks = data.groupby("state_territory")[score_col].rank(
                        method="min", ascending=True, pct=True
                    )
                    data[state_decile_col] = (normalized_ranks * 10).clip(1, 10).round().astype("Int64")
    
    def _create_composite_indices(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite indices from individual SEIFA scores.
        
        Args:
            data: Data with individual SEIFA scores
            
        Returns:
            pd.DataFrame: Data with composite indices
        """
        self.logger.debug("Stage 6: Creating composite indices")
        
        result = data.copy()
        
        if not self.operations_config.get("composite_indices", True):
            return result
        
        # Calculate overall advantage score as weighted average
        score_weights = [
            ("irsad_score", self.composite_weights.get("economic", 0.4)),
            ("ier_score", self.composite_weights.get("education", 0.3)),
            ("ieo_score", self.composite_weights.get("housing", 0.2))
        ]
        
        # Create overall advantage score
        overall_scores = []
        total_weights = []
        
        for score_col, weight in score_weights:
            if score_col in result.columns:
                score_values = result[score_col].fillna(0) * weight
                overall_scores.append(score_values)
                total_weights.append(weight)
        
        if overall_scores:
            result["overall_advantage_score"] = sum(overall_scores) / sum(total_weights)
        
        # Generate disadvantage severity categories
        if "irsd_decile" in result.columns:
            result["disadvantage_severity"] = result["irsd_decile"].map({
                1: "very_high", 2: "very_high",
                3: "high", 4: "high",
                5: "moderate", 6: "moderate",
                7: "low", 8: "low",
                9: "very_low", 10: "very_low"
            })
        
        return result
    
    def _integrate_geographic_hierarchy(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate geographic hierarchy information.
        
        Args:
            data: Data with composite indices
            
        Returns:
            pd.DataFrame: Data with geographic hierarchy integration
        """
        self.logger.debug("Stage 7: Integrating geographic hierarchy")
        
        if not self.geographic_hierarchy:
            return data
        
        result = data.copy()
        
        # Add surrogate keys for SEIFA records
        result["seifa_sk"] = range(self.seifa_sk_counter, self.seifa_sk_counter + len(result))
        self.seifa_sk_counter += len(result)
        
        # Add data source information
        result["data_source_name"] = "ABS SEIFA 2021"
        result["data_source_url"] = "https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia"
        result["extraction_date"] = datetime.now()
        result["quality_level"] = "HIGH"
        result["source_version"] = "2021.0.0"
        
        # Add schema metadata
        result["schema_version"] = "2.0.0"
        result["last_updated"] = datetime.now()
        
        return result
    
    def _impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values using configured strategy.
        
        Args:
            data: Data with potential missing values
            
        Returns:
            pd.DataFrame: Data with imputed missing values
        """
        self.logger.debug("Stage 8: Imputing missing values")
        
        result = data.copy()
        
        if self.imputation_strategy == "score_median":
            # Impute missing scores with median values
            score_columns = ["irsad_score", "irsd_score", "ier_score", "ieo_score"]
            
            for col in score_columns:
                if col in result.columns and result[col].isna().any():
                    median_value = result[col].median()
                    missing_count = result[col].isna().sum()
                    result[col] = result[col].fillna(median_value)
                    self.logger.debug(f"Imputed {missing_count} missing values in {col} with median {median_value}")
        
        elif self.imputation_strategy == "geographic_median":
            # Impute within geographic groups
            if "state_territory" in result.columns:
                score_columns = ["irsad_score", "irsd_score", "ier_score", "ieo_score"]
                
                for col in score_columns:
                    if col in result.columns:
                        result[col] = result.groupby("state_territory")[col].transform(
                            lambda x: x.fillna(x.median())
                        )
        
        return result
    
    def _enforce_schema(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce final schema compliance and data types.
        
        Args:
            data: Data ready for schema enforcement
            
        Returns:
            pd.DataFrame: Schema-compliant data
        """
        self.logger.debug("Stage 9: Enforcing schema compliance")
        
        result = data.copy()
        
        # Ensure required fields exist
        required_fields = ["geographic_id", "geographic_level", "geographic_name", "state_territory", "census_year"]
        
        for field in required_fields:
            if field not in result.columns:
                if field == "geographic_name":
                    result[field] = "Unknown"
                elif field == "census_year":
                    result[field] = 2021
                else:
                    result[field] = None
        
        # Ensure data types
        integer_fields = [
            "census_year", "irsad_score", "irsd_score", "ier_score", "ieo_score",
            "irsad_rank", "irsd_rank", "ier_rank", "ieo_rank",
            "irsad_decile", "irsd_decile", "ier_decile", "ieo_decile",
            "irsad_state_rank", "irsd_state_rank", "irsad_state_decile", "irsd_state_decile",
            "population_base"
        ]
        
        for field in integer_fields:
            if field in result.columns:
                result[field] = pd.to_numeric(result[field], errors="coerce").astype("Int64")
        
        # Ensure string fields
        string_fields = ["geographic_id", "geographic_level", "geographic_name", "state_territory", "disadvantage_severity"]
        
        for field in string_fields:
            if field in result.columns:
                result[field] = result[field].astype(str)
        
        # Remove any extra columns not in schema
        schema_columns = [
            "geographic_id", "geographic_level", "geographic_name", "state_territory", "census_year",
            "irsad_score", "irsd_score", "ier_score", "ieo_score",
            "irsad_rank", "irsd_rank", "ier_rank", "ieo_rank",
            "irsad_decile", "irsd_decile", "ier_decile", "ieo_decile",
            "irsad_state_rank", "irsd_state_rank", "irsad_state_decile", "irsd_state_decile",
            "overall_advantage_score", "disadvantage_severity", "population_base",
            "seifa_sk", "data_source_name", "data_source_url", "extraction_date",
            "quality_level", "source_version", "schema_version", "last_updated"
        ]
        
        # Keep only schema columns that exist
        available_schema_cols = [col for col in schema_columns if col in result.columns]
        result = result[available_schema_cols]
        
        return result
    
    def _create_empty_schema_dataframe(self) -> pd.DataFrame:
        """
        Create empty DataFrame with correct schema structure.
        
        Returns:
            pd.DataFrame: Empty DataFrame with schema columns
        """
        schema_columns = [
            "geographic_id", "geographic_level", "geographic_name", "state_territory", "census_year",
            "irsad_score", "irsd_score", "ier_score", "ieo_score",
            "irsad_rank", "irsd_rank", "ier_rank", "ieo_rank",
            "irsad_decile", "irsd_decile", "ier_decile", "ieo_decile",
            "overall_advantage_score", "disadvantage_severity", "population_base"
        ]
        
        return pd.DataFrame(columns=schema_columns)
    
    def get_processing_metadata(self) -> Optional[ProcessingMetadata]:
        """
        Get processing metadata for the last transformation.
        
        Returns:
            Optional[ProcessingMetadata]: Processing metadata if available
        """
        return self.processing_metadata