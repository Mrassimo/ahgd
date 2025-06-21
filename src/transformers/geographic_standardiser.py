"""
Geographic standardisation transformer for the AHGD ETL pipeline.

This module provides comprehensive geographic standardisation capabilities,
mapping all geographic data to the SA2 framework as mandated by the Australian
Bureau of Statistics.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path

from .base import BaseTransformer, MissingValueStrategy
from ..utils.interfaces import (
    AuditTrail,
    DataBatch,
    DataRecord,
    GeographicValidationError,
    ProcessingMetadata,
    ProcessingStatus,
    ProgressCallback,
    TransformationError,
    ValidationResult,
    ValidationSeverity,
    ValidationError,
)
from ..validators import GeographicValidator, ValidationOrchestrator
from ..pipelines.validation_pipeline import ValidationMode, StageValidationResult


@dataclass
class GeographicMapping:
    """Represents a mapping between geographic units."""
    source_code: str
    target_sa2_code: str
    allocation_factor: float = 1.0  # For population-weighted mappings
    mapping_method: str = "direct"  # direct, area_weighted, population_weighted
    confidence: float = 1.0
    source_type: str = "unknown"  # postcode, lga, phn, sa1, sa3, sa4
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GeographicValidationResult:
    """Result of geographic validation."""
    is_valid: bool
    code: str
    code_type: str
    sa2_code: Optional[str] = None
    error_message: Optional[str] = None
    confidence: float = 1.0
    validation_method: str = "lookup"


class SA2MappingEngine:
    """
    Core engine for mapping various geographic units to SA2 framework.
    
    Handles the complexities of Australian geographic hierarchies and
    provides population-weighted allocations where necessary.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the SA2 mapping engine.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Mapping lookup tables (would be loaded from reference data)
        self._postcode_mappings: Dict[str, List[GeographicMapping]] = {}
        self._lga_mappings: Dict[str, List[GeographicMapping]] = {}
        self._phn_mappings: Dict[str, List[GeographicMapping]] = {}
        self._sa1_mappings: Dict[str, str] = {}  # SA1 to SA2 is 1:1
        self._sa3_mappings: Dict[str, List[str]] = {}  # SA3 contains multiple SA2s
        self._sa4_mappings: Dict[str, List[str]] = {}  # SA4 contains multiple SA2s
        
        # SA2 validation lookup
        self._valid_sa2_codes: Set[str] = set()
        
        # Cache for performance
        self._mapping_cache: Dict[str, List[GeographicMapping]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._load_reference_data()
    
    def map_to_sa2(self, code: str, code_type: str) -> List[GeographicMapping]:
        """
        Map a geographic code to SA2 framework.
        
        Args:
            code: Geographic code to map
            code_type: Type of code (postcode, lga, phn, sa1, sa3, sa4)
            
        Returns:
            List[GeographicMapping]: Mappings to SA2 codes
            
        Raises:
            GeographicValidationError: If code is invalid or cannot be mapped
        """
        cache_key = f"{code_type}:{code}"
        
        # Check cache first
        if cache_key in self._mapping_cache:
            self._cache_hits += 1
            return self._mapping_cache[cache_key]
        
        self._cache_misses += 1
        
        # Validate code format
        if not self._validate_code_format(code, code_type):
            raise GeographicValidationError(
                f"Invalid {code_type} code format: {code}"
            )
        
        # Perform mapping based on code type
        mappings = []
        
        if code_type == "postcode":
            mappings = self._map_postcode_to_sa2(code)
        elif code_type == "lga":
            mappings = self._map_lga_to_sa2(code)
        elif code_type == "phn":
            mappings = self._map_phn_to_sa2(code)
        elif code_type == "sa1":
            mappings = self._map_sa1_to_sa2(code)
        elif code_type == "sa3":
            mappings = self._map_sa3_to_sa2(code)
        elif code_type == "sa4":
            mappings = self._map_sa4_to_sa2(code)
        elif code_type == "sa2":
            # Direct SA2 - just validate
            if code in self._valid_sa2_codes:
                mappings = [GeographicMapping(
                    source_code=code,
                    target_sa2_code=code,
                    allocation_factor=1.0,
                    mapping_method="direct",
                    confidence=1.0,
                    source_type="sa2"
                )]
            else:
                raise GeographicValidationError(f"Invalid SA2 code: {code}")
        else:
            raise GeographicValidationError(f"Unsupported code type: {code_type}")
        
        if not mappings:
            raise GeographicValidationError(
                f"No SA2 mapping found for {code_type} code: {code}"
            )
        
        # Cache the result
        self._mapping_cache[cache_key] = mappings
        
        return mappings
    
    def validate_geographic_hierarchy(self, sa2_code: str) -> GeographicValidationResult:
        """
        Validate SA2 code against geographic hierarchy.
        
        Args:
            sa2_code: SA2 code to validate
            
        Returns:
            GeographicValidationResult: Validation result
        """
        if not isinstance(sa2_code, str) or len(sa2_code) != 9:
            return GeographicValidationResult(
                is_valid=False,
                code=sa2_code,
                code_type="sa2",
                error_message="SA2 codes must be 9 digits"
            )
        
        if not sa2_code.isdigit():
            return GeographicValidationResult(
                is_valid=False,
                code=sa2_code,
                code_type="sa2",
                error_message="SA2 codes must be numeric"
            )
        
        # Check if SA2 code exists in reference data
        if sa2_code not in self._valid_sa2_codes:
            return GeographicValidationResult(
                is_valid=False,
                code=sa2_code,
                code_type="sa2",
                error_message=f"SA2 code {sa2_code} not found in reference data"
            )
        
        return GeographicValidationResult(
            is_valid=True,
            code=sa2_code,
            code_type="sa2",
            sa2_code=sa2_code,
            confidence=1.0
        )
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_lookups if total_lookups > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._mapping_cache)
        }
    
    def _load_reference_data(self):
        """Load reference data for geographic mappings."""
        # This would load from actual reference files
        # For now, we'll simulate loading some basic data
        
        self.logger.info("Loading geographic reference data")
        
        # Load valid SA2 codes (normally from ABS correspondence files)
        sample_sa2_codes = [
            f"{state}{str(i).zfill(8)}" 
            for state in range(1, 9) 
            for i in range(10001, 10101)
        ]
        self._valid_sa2_codes = set(sample_sa2_codes[:2473])  # 2,473 SA2s in Australia
        
        # Load sample postcode mappings
        self._postcode_mappings = {
            "2000": [GeographicMapping("2000", "101021007", 0.8, "population_weighted", 0.95, "postcode")],
            "3000": [GeographicMapping("3000", "201021001", 1.0, "direct", 1.0, "postcode")],
            "4000": [GeographicMapping("4000", "301031015", 1.0, "direct", 1.0, "postcode")],
        }
        
        self.logger.info(f"Loaded {len(self._valid_sa2_codes)} SA2 codes")
    
    def _validate_code_format(self, code: str, code_type: str) -> bool:
        """Validate the format of a geographic code."""
        if not isinstance(code, str):
            return False
        
        if code_type == "postcode":
            return len(code) == 4 and code.isdigit()
        elif code_type in ["sa1", "sa2", "sa3", "sa4"]:
            expected_lengths = {"sa1": 11, "sa2": 9, "sa3": 5, "sa4": 3}
            return len(code) == expected_lengths[code_type] and code.isdigit()
        elif code_type in ["lga", "phn"]:
            return len(code) > 0  # More flexible validation for these
        
        return False
    
    def _map_postcode_to_sa2(self, postcode: str) -> List[GeographicMapping]:
        """Map postcode to SA2 with population weighting."""
        return self._postcode_mappings.get(postcode, [])
    
    def _map_lga_to_sa2(self, lga_code: str) -> List[GeographicMapping]:
        """Map Local Government Area to SA2."""
        return self._lga_mappings.get(lga_code, [])
    
    def _map_phn_to_sa2(self, phn_code: str) -> List[GeographicMapping]:
        """Map Primary Health Network to SA2."""
        return self._phn_mappings.get(phn_code, [])
    
    def _map_sa1_to_sa2(self, sa1_code: str) -> List[GeographicMapping]:
        """Map SA1 to SA2 (1:1 relationship)."""
        sa2_code = self._sa1_mappings.get(sa1_code)
        if sa2_code:
            return [GeographicMapping(
                source_code=sa1_code,
                target_sa2_code=sa2_code,
                allocation_factor=1.0,
                mapping_method="direct",
                confidence=1.0,
                source_type="sa1"
            )]
        return []
    
    def _map_sa3_to_sa2(self, sa3_code: str) -> List[GeographicMapping]:
        """Map SA3 to constituent SA2s."""
        sa2_codes = self._sa3_mappings.get(sa3_code, [])
        return [
            GeographicMapping(
                source_code=sa3_code,
                target_sa2_code=sa2_code,
                allocation_factor=1.0 / len(sa2_codes),  # Equal allocation
                mapping_method="area_weighted",
                confidence=0.9,
                source_type="sa3"
            )
            for sa2_code in sa2_codes
        ]
    
    def _map_sa4_to_sa2(self, sa4_code: str) -> List[GeographicMapping]:
        """Map SA4 to constituent SA2s."""
        sa2_codes = self._sa4_mappings.get(sa4_code, [])
        return [
            GeographicMapping(
                source_code=sa4_code,
                target_sa2_code=sa2_code,
                allocation_factor=1.0 / len(sa2_codes),  # Equal allocation
                mapping_method="area_weighted",
                confidence=0.8,
                source_type="sa4"
            )
            for sa2_code in sa2_codes
        ]


class PostcodeToSA2Mapper:
    """
    Specialised mapper for postcode to SA2 with population weighting.
    
    Handles the complex many-to-many relationship between postcodes and SA2s,
    using population-weighted allocation factors from ABS correspondence files.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the postcode mapper.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Population-weighted allocation table
        self._allocation_table: Dict[str, List[Tuple[str, float]]] = {}
        self._load_postcode_allocations()
    
    def map_postcode(self, postcode: str, allocation_method: str = "population") -> List[GeographicMapping]:
        """
        Map postcode to SA2s with appropriate allocation factors.
        
        Args:
            postcode: Australian postcode (4 digits)
            allocation_method: Method for allocation (population, area, equal)
            
        Returns:
            List[GeographicMapping]: SA2 mappings with allocation factors
        """
        if postcode not in self._allocation_table:
            self.logger.warning(f"Postcode {postcode} not found in allocation table")
            return []
        
        allocations = self._allocation_table[postcode]
        mappings = []
        
        for sa2_code, factor in allocations:
            mapping = GeographicMapping(
                source_code=postcode,
                target_sa2_code=sa2_code,
                allocation_factor=factor,
                mapping_method=f"{allocation_method}_weighted",
                confidence=0.95 if allocation_method == "population" else 0.8,
                source_type="postcode"
            )
            mappings.append(mapping)
        
        return mappings
    
    def validate_postcode(self, postcode: str) -> bool:
        """
        Validate Australian postcode format and existence.
        
        Args:
            postcode: Postcode to validate
            
        Returns:
            bool: True if valid postcode
        """
        if not isinstance(postcode, str) or len(postcode) != 4:
            return False
        
        if not postcode.isdigit():
            return False
        
        # Check postcode ranges for Australian states
        postcode_int = int(postcode)
        
        # Australian postcode ranges
        valid_ranges = [
            (1000, 2599),  # NSW
            (2600, 2920),  # ACT
            (3000, 3999),  # VIC
            (4000, 4999),  # QLD
            (5000, 5999),  # SA
            (6000, 6999),  # WA
            (7000, 7999),  # TAS
            (800, 999),    # NT
        ]
        
        return any(start <= postcode_int <= end for start, end in valid_ranges)
    
    def _load_postcode_allocations(self):
        """Load postcode to SA2 allocation factors."""
        # This would load from ABS correspondence files
        # For now, simulate some data
        self._allocation_table = {
            "2000": [("101021007", 0.8), ("101021008", 0.2)],
            "3000": [("201021001", 1.0)],
            "4000": [("301031015", 1.0)],
        }


class LGAToSA2Mapper:
    """
    Mapper for Local Government Areas to SA2 with area-based allocation.
    
    Uses area-weighted allocation where LGAs cross SA2 boundaries.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the LGA mapper.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # LGA to SA2 mappings with area weights
        self._lga_mappings: Dict[str, List[Tuple[str, float]]] = {}
        self._load_lga_mappings()
    
    def map_lga(self, lga_code: str) -> List[GeographicMapping]:
        """
        Map LGA to SA2s with area-based allocation.
        
        Args:
            lga_code: LGA code
            
        Returns:
            List[GeographicMapping]: SA2 mappings with area factors
        """
        if lga_code not in self._lga_mappings:
            self.logger.warning(f"LGA {lga_code} not found in mapping table")
            return []
        
        allocations = self._lga_mappings[lga_code]
        mappings = []
        
        for sa2_code, area_factor in allocations:
            mapping = GeographicMapping(
                source_code=lga_code,
                target_sa2_code=sa2_code,
                allocation_factor=area_factor,
                mapping_method="area_weighted",
                confidence=0.85,
                source_type="lga"
            )
            mappings.append(mapping)
        
        return mappings
    
    def _load_lga_mappings(self):
        """Load LGA to SA2 mappings."""
        # Simulate LGA mappings
        self._lga_mappings = {
            "LGA10050": [("101021007", 0.6), ("101021008", 0.4)],
            "LGA20110": [("201021001", 1.0)],
        }


class PHNToSA2Mapper:
    """
    Mapper for Primary Health Networks to SA2 for health service data.
    
    PHNs are health service boundaries that may not align perfectly with SA2s.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the PHN mapper.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # PHN to SA2 mappings
        self._phn_mappings: Dict[str, List[Tuple[str, float]]] = {}
        self._load_phn_mappings()
    
    def map_phn(self, phn_code: str) -> List[GeographicMapping]:
        """
        Map PHN to SA2s.
        
        Args:
            phn_code: PHN identifier
            
        Returns:
            List[GeographicMapping]: SA2 mappings
        """
        if phn_code not in self._phn_mappings:
            self.logger.warning(f"PHN {phn_code} not found in mapping table")
            return []
        
        allocations = self._phn_mappings[phn_code]
        mappings = []
        
        for sa2_code, factor in allocations:
            mapping = GeographicMapping(
                source_code=phn_code,
                target_sa2_code=sa2_code,
                allocation_factor=factor,
                mapping_method="health_service_weighted",
                confidence=0.8,
                source_type="phn"
            )
            mappings.append(mapping)
        
        return mappings
    
    def _load_phn_mappings(self):
        """Load PHN to SA2 mappings."""
        # Simulate PHN mappings
        self._phn_mappings = {
            "PHN001": [("101021007", 0.3), ("101021008", 0.4), ("101021009", 0.3)],
            "PHN002": [("201021001", 0.7), ("201021002", 0.3)],
        }


class GeographicStandardiser(BaseTransformer):
    """
    Main geographic standardisation transformer.
    
    Standardises all geographic data to the SA2 framework, handling multiple
    input formats and providing comprehensive mapping with allocation factors.
    """
    
    def __init__(
        self,
        transformer_id: str = "geographic_standardiser",
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the geographic standardiser.
        
        Args:
            transformer_id: Unique identifier for this transformer
            config: Configuration dictionary
            logger: Optional logger instance
        """
        config = config or {}
        super().__init__(transformer_id, config, logger)
        
        # Geographic mapping components
        self.sa2_mapping_engine = SA2MappingEngine(config, logger)
        self.postcode_mapper = PostcodeToSA2Mapper(config, logger)
        self.lga_mapper = LGAToSA2Mapper(config, logger)
        self.phn_mapper = PHNToSA2Mapper(config, logger)
        
        # Configuration
        self.geographic_column = config.get('geographic_column', 'geographic_code')
        self.geographic_type_column = config.get('geographic_type_column', 'geographic_type')
        self.output_sa2_column = config.get('output_sa2_column', 'sa2_code')
        self.output_allocation_column = config.get('output_allocation_column', 'allocation_factor')
        self.include_mapping_metadata = config.get('include_mapping_metadata', True)
        
        # Validation settings
        self.strict_validation = config.get('strict_validation', True)
        self.handle_invalid_codes = config.get('handle_invalid_codes', 'error')  # error, warn, skip
        
        # Performance settings
        self.batch_size = config.get('batch_size', 1000)
        self.enable_caching = config.get('enable_caching', True)
        
        # Validation configuration
        self.validation_enabled = config.get('validation_enabled', True)
        self.validation_mode = ValidationMode(config.get('validation_mode', 'selective'))
        self.geographic_quality_threshold = config.get('geographic_quality_threshold', 95.0)
        self.halt_on_validation_failure = config.get('halt_on_validation_failure', False)
        
        # Initialise validation components
        if self.validation_enabled:
            self.geographic_validator = GeographicValidator()
            self.validation_orchestrator = ValidationOrchestrator()
        else:
            self.geographic_validator = None
            self.validation_orchestrator = None
        
        # Statistics
        self._transformation_stats = {
            'total_records': 0,
            'successful_mappings': 0,
            'failed_mappings': 0,
            'multi_sa2_mappings': 0,
            'direct_sa2_records': 0,
            'mapping_methods': {},
            'validation_results': {}
        }
    
    def validate_geographic_data(
        self,
        data: pd.DataFrame,
        validation_context: Optional[Dict[str, Any]] = None
    ) -> StageValidationResult:
        """
        Validate geographic data before and after transformation.
        
        Args:
            data: DataFrame containing geographic data
            validation_context: Optional validation context
            
        Returns:
            StageValidationResult: Comprehensive validation results
        """
        if not self.validation_enabled or not self.geographic_validator:
            # Create a dummy successful result
            from ..pipelines.validation_pipeline import ValidationMetrics, QualityGateStatus
            return StageValidationResult(
                stage_name="geographic_standardisation",
                gate_status=QualityGateStatus.PASSED,
                validation_metrics=ValidationMetrics(
                    total_records=len(data),
                    validated_records=len(data),
                    passed_records=len(data),
                    failed_records=0,
                    warning_records=0,
                    validation_time_seconds=0.0,
                    rules_executed=0,
                    rules_passed=0,
                    rules_failed=0,
                    quality_score=100.0,
                    completeness_score=100.0,
                    accuracy_score=100.0,
                    consistency_score=100.0
                ),
                rule_results={},
                errors=[],
                warnings=[],
                recommendations=[]
            )
        
        try:
            start_time = datetime.now()
            
            # Run geographic validation
            validation_result = self.geographic_validator.validate_data_frame(data)
            
            # Calculate quality metrics
            total_records = len(data)
            validation_errors = getattr(validation_result, 'errors', [])
            validation_warnings = getattr(validation_result, 'warnings', [])
            
            # Calculate quality scores
            error_rate = len(validation_errors) / total_records if total_records > 0 else 0
            quality_score = max(0, (1 - error_rate) * 100)
            
            # Determine validation status
            from ..pipelines.validation_pipeline import QualityGateStatus, ValidationMetrics
            
            if quality_score >= self.geographic_quality_threshold:
                gate_status = QualityGateStatus.PASSED
            elif validation_errors:
                gate_status = QualityGateStatus.FAILED
            else:
                gate_status = QualityGateStatus.WARNING
            
            # Create detailed validation result
            validation_metrics = ValidationMetrics(
                total_records=total_records,
                validated_records=total_records,
                passed_records=total_records - len(validation_errors),
                failed_records=len(validation_errors),
                warning_records=len(validation_warnings),
                validation_time_seconds=(datetime.now() - start_time).total_seconds(),
                rules_executed=1,  # Geographic validation rule
                rules_passed=1 if quality_score >= self.geographic_quality_threshold else 0,
                rules_failed=0 if quality_score >= self.geographic_quality_threshold else 1,
                quality_score=quality_score,
                completeness_score=self._calculate_geographic_completeness(data),
                accuracy_score=self._calculate_geographic_accuracy(data),
                consistency_score=quality_score
            )
            
            stage_result = StageValidationResult(
                stage_name="geographic_standardisation",
                gate_status=gate_status,
                validation_metrics=validation_metrics,
                rule_results={"geographic_validation": quality_score >= self.geographic_quality_threshold},
                errors=[str(e) for e in validation_errors],
                warnings=[str(w) for w in validation_warnings],
                recommendations=self._generate_geographic_recommendations(validation_result),
                execution_time=validation_metrics.validation_time_seconds
            )
            
            # Store validation results in stats
            self._transformation_stats['validation_results'] = {
                'quality_score': quality_score,
                'gate_status': gate_status.value,
                'error_count': len(validation_errors),
                'warning_count': len(validation_warnings)
            }
            
            self.logger.info(
                f"Geographic validation completed: quality score {quality_score:.2f}%, status: {gate_status.value}"
            )
            
            return stage_result
            
        except Exception as e:
            self.logger.error(f"Geographic validation failed: {str(e)}")
            
            # Return failed validation result
            from ..pipelines.validation_pipeline import QualityGateStatus, ValidationMetrics
            
            return StageValidationResult(
                stage_name="geographic_standardisation",
                gate_status=QualityGateStatus.FAILED,
                validation_metrics=ValidationMetrics(
                    total_records=len(data),
                    validated_records=0,
                    passed_records=0,
                    failed_records=len(data),
                    warning_records=0,
                    validation_time_seconds=0.0,
                    rules_executed=1,
                    rules_passed=0,
                    rules_failed=1,
                    quality_score=0.0,
                    completeness_score=0.0,
                    accuracy_score=0.0,
                    consistency_score=0.0
                ),
                rule_results={"geographic_validation": False},
                errors=[f"Validation execution failed: {str(e)}"],
                warnings=[],
                recommendations=["Fix validation configuration and retry"],
                execution_time=0.0
            )
    
    def _calculate_geographic_completeness(self, data: pd.DataFrame) -> float:
        """Calculate geographic data completeness score."""
        if data.empty:
            return 0.0
        
        # Check for required geographic columns
        geographic_columns = [self.geographic_column, self.geographic_type_column]
        total_cells = 0
        non_null_cells = 0
        
        for col in geographic_columns:
            if col in data.columns:
                total_cells += len(data)
                non_null_cells += data[col].notna().sum()
        
        return (non_null_cells / total_cells * 100) if total_cells > 0 else 0.0
    
    def _calculate_geographic_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate geographic data accuracy score."""
        if data.empty:
            return 0.0
        
        # For geographic accuracy, we check the validity of codes
        # This is a simplified implementation
        if self.geographic_column in data.columns:
            valid_codes = data[self.geographic_column].notna().sum()
            total_codes = len(data)
            return (valid_codes / total_codes * 100) if total_codes > 0 else 0.0
        
        return 100.0
    
    def _generate_geographic_recommendations(self, validation_result: Any) -> List[str]:
        """Generate recommendations based on geographic validation results."""
        recommendations = []
        
        if hasattr(validation_result, 'invalid_coordinates_count'):
            if validation_result.invalid_coordinates_count > 0:
                recommendations.append(f"Fix {validation_result.invalid_coordinates_count} invalid coordinates")
        
        if hasattr(validation_result, 'missing_sa2_codes'):
            if validation_result.missing_sa2_codes > 0:
                recommendations.append(f"Resolve {validation_result.missing_sa2_codes} missing SA2 codes")
        
        recommendations.append("Ensure all geographic codes follow ABS standards")
        recommendations.append("Validate coordinate reference systems are consistent")
        
        return recommendations
    
    def transform(self, data: DataBatch, **kwargs) -> DataBatch:
        """
        Transform geographic data to SA2 standardised format.
        
        Args:
            data: Batch of data records to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            DataBatch: Transformed data with SA2 standardisation
            
        Raises:
            TransformationError: If transformation fails
        """
        if not data:
            return data
        
        self.logger.info(f"Starting geographic standardisation for {len(data)} records")
        
        # Reset statistics
        self._transformation_stats['total_records'] = len(data)
        
        # Convert data to DataFrame for validation
        if hasattr(data, 'to_dataframe'):
            df = data.to_dataframe()
        elif isinstance(data, list):
            import pandas as pd
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Pre-transformation validation
        if self.validation_enabled:
            self.logger.debug("Running pre-transformation validation")
            pre_validation_result = self.validate_geographic_data(
                df,
                validation_context={'stage': 'pre_transformation', 'transformer_id': self.transformer_id}
            )
            
            if (pre_validation_result.gate_status.value == 'failed' and 
                self.halt_on_validation_failure):
                raise ValidationError(
                    f"Pre-transformation validation failed: "
                    f"quality score {pre_validation_result.validation_metrics.quality_score:.2f}% "
                    f"below threshold {self.geographic_quality_threshold}%"
                )
            
            self._transformation_stats['validation_results']['pre_transformation'] = {
                'quality_score': pre_validation_result.validation_metrics.quality_score,
                'gate_status': pre_validation_result.gate_status.value
            }
        
        # Process in batches for better performance
        standardised_data = []
        
        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            processed_batch = self._process_batch(batch)
            standardised_data.extend(processed_batch)
            
            # Report progress
            if self._progress_callback:
                progress = min(i + self.batch_size, len(data))
                self._progress_callback(
                    progress, 
                    len(data), 
                    f"Processed {progress}/{len(data)} records"
                )
        
        # Post-transformation validation
        if self.validation_enabled and standardised_data:
            self.logger.debug("Running post-transformation validation")
            
            # Convert standardised data back to DataFrame
            if isinstance(standardised_data, list):
                import pandas as pd
                standardised_df = pd.DataFrame(standardised_data)
            else:
                standardised_df = standardised_data
            
            post_validation_result = self.validate_geographic_data(
                standardised_df,
                validation_context={'stage': 'post_transformation', 'transformer_id': self.transformer_id}
            )
            
            if (post_validation_result.gate_status.value == 'failed' and 
                self.halt_on_validation_failure):
                raise ValidationError(
                    f"Post-transformation validation failed: "
                    f"quality score {post_validation_result.validation_metrics.quality_score:.2f}% "
                    f"below threshold {self.geographic_quality_threshold}%"
                )
            
            self._transformation_stats['validation_results']['post_transformation'] = {
                'quality_score': post_validation_result.validation_metrics.quality_score,
                'gate_status': post_validation_result.gate_status.value
            }
        
        # Log completion with validation statistics
        completion_message = (
            f"Geographic standardisation completed. "
            f"Success: {self._transformation_stats['successful_mappings']}, "
            f"Failed: {self._transformation_stats['failed_mappings']}"
        )
        
        if self.validation_enabled and 'validation_results' in self._transformation_stats:
            validation_results = self._transformation_stats['validation_results']
            if 'post_transformation' in validation_results:
                post_quality = validation_results['post_transformation']['quality_score']
                completion_message += f", Post-transformation quality: {post_quality:.1f}%"
        
        self.logger.info(completion_message)
        
        return standardised_data
    
    def get_schema(self) -> Dict[str, str]:
        """
        Get the expected output schema.
        
        Returns:
            Dict[str, str]: Schema definition
        """
        schema = {
            self.output_sa2_column: "string",
            self.output_allocation_column: "float",
        }
        
        if self.include_mapping_metadata:
            schema.update({
                "mapping_method": "string",
                "mapping_confidence": "float",
                "source_geographic_code": "string",
                "source_geographic_type": "string",
            })
        
        return schema
    
    def get_transformation_statistics(self) -> Dict[str, Any]:
        """Get detailed transformation statistics."""
        return self._transformation_stats.copy()
    
    def _process_batch(self, batch: DataBatch) -> DataBatch:
        """
        Process a batch of records for geographic standardisation.
        
        Args:
            batch: Batch of records to process
            
        Returns:
            DataBatch: Processed batch
        """
        processed_records = []
        
        for record in batch:
            try:
                processed_record_list = self._process_record(record)
                processed_records.extend(processed_record_list)
                self._transformation_stats['successful_mappings'] += 1
                
                # Track multi-SA2 mappings
                if len(processed_record_list) > 1:
                    self._transformation_stats['multi_sa2_mappings'] += 1
                
            except Exception as e:
                self._transformation_stats['failed_mappings'] += 1
                
                if self.handle_invalid_codes == 'error':
                    raise TransformationError(f"Failed to process record: {e}") from e
                elif self.handle_invalid_codes == 'warn':
                    self.logger.warning(f"Failed to process record: {e}")
                # For 'skip', we just continue without adding the record
        
        return processed_records
    
    def _process_record(self, record: DataRecord) -> List[DataRecord]:
        """
        Process a single record for geographic standardisation.
        
        Args:
            record: Single data record
            
        Returns:
            List[DataRecord]: List of processed records (may be multiple for 1:many mappings)
        """
        # Extract geographic information
        geographic_code = record.get(self.geographic_column)
        geographic_type = record.get(self.geographic_type_column, 'unknown')
        
        if not geographic_code:
            raise TransformationError(f"Missing geographic code in column '{self.geographic_column}'")
        
        # Handle different geographic types
        geographic_code = str(geographic_code).strip()
        geographic_type = str(geographic_type).lower().strip()
        
        # Map to SA2
        try:
            mappings = self._map_to_sa2(geographic_code, geographic_type)
        except GeographicValidationError as e:
            if self.strict_validation:
                raise TransformationError(f"Geographic validation failed: {e}") from e
            else:
                self.logger.warning(f"Geographic validation warning: {e}")
                # Return original record with warning flags
                warning_record = record.copy()
                warning_record.update({
                    self.output_sa2_column: None,
                    self.output_allocation_column: 0.0,
                    "mapping_error": str(e)
                })
                return [warning_record]
        
        # Create output records
        output_records = []
        
        for mapping in mappings:
            # Create base record
            output_record = record.copy()
            
            # Add SA2 standardisation
            output_record[self.output_sa2_column] = mapping.target_sa2_code
            output_record[self.output_allocation_column] = mapping.allocation_factor
            
            # Add metadata if requested
            if self.include_mapping_metadata:
                output_record.update({
                    "mapping_method": mapping.mapping_method,
                    "mapping_confidence": mapping.confidence,
                    "source_geographic_code": mapping.source_code,
                    "source_geographic_type": mapping.source_type,
                })
            
            # Track mapping method statistics
            method = mapping.mapping_method
            self._transformation_stats['mapping_methods'][method] = (
                self._transformation_stats['mapping_methods'].get(method, 0) + 1
            )
            
            output_records.append(output_record)
        
        return output_records
    
    def _map_to_sa2(self, code: str, code_type: str) -> List[GeographicMapping]:
        """
        Map a geographic code to SA2 using appropriate mapper.
        
        Args:
            code: Geographic code
            code_type: Type of geographic code
            
        Returns:
            List[GeographicMapping]: SA2 mappings
        """
        # Normalise code type
        code_type_normalised = self._normalise_code_type(code_type)
        
        # Use specialised mappers for better performance and accuracy
        if code_type_normalised == "postcode":
            mappings = self.postcode_mapper.map_postcode(code)
        elif code_type_normalised == "lga":
            mappings = self.lga_mapper.map_lga(code)
        elif code_type_normalised == "phn":
            mappings = self.phn_mapper.map_phn(code)
        else:
            # Use general SA2 mapping engine
            mappings = self.sa2_mapping_engine.map_to_sa2(code, code_type_normalised)
        
        if not mappings:
            raise GeographicValidationError(
                f"No SA2 mapping found for {code_type} code: {code}"
            )
        
        return mappings
    
    def _normalise_code_type(self, code_type: str) -> str:
        """
        Normalise geographic code type names.
        
        Args:
            code_type: Raw code type string
            
        Returns:
            str: Normalised code type
        """
        code_type = code_type.lower().strip()
        
        # Handle common variations
        type_mappings = {
            'post_code': 'postcode',
            'postal_code': 'postcode',
            'zip': 'postcode',
            'zipcode': 'postcode',
            'local_government_area': 'lga',
            'lga_code': 'lga',
            'primary_health_network': 'phn',
            'phn_code': 'phn',
            'statistical_area_1': 'sa1',
            'statistical_area_2': 'sa2',
            'statistical_area_3': 'sa3',
            'statistical_area_4': 'sa4',
            'sa1_code': 'sa1',
            'sa2_code': 'sa2',
            'sa3_code': 'sa3',
            'sa4_code': 'sa4',
        }
        
        return type_mappings.get(code_type, code_type)
    
    def validate_geographic_data(self, data: DataBatch) -> List[ValidationResult]:
        """
        Validate geographic data before transformation.
        
        Args:
            data: Data batch to validate
            
        Returns:
            List[ValidationResult]: Validation results
        """
        validation_results = []
        
        for i, record in enumerate(data):
            # Check for required columns
            if self.geographic_column not in record:
                validation_results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    rule_id="missing_geographic_column",
                    message=f"Missing required column: {self.geographic_column}",
                    affected_records=[i]
                ))
            
            # Validate geographic code format
            geographic_code = record.get(self.geographic_column)
            if geographic_code:
                geographic_type = record.get(self.geographic_type_column, 'unknown')
                
                try:
                    # Try to validate using the mapping engine
                    self.sa2_mapping_engine.map_to_sa2(
                        str(geographic_code), 
                        self._normalise_code_type(str(geographic_type))
                    )
                except GeographicValidationError as e:
                    validation_results.append(ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.WARNING,
                        rule_id="invalid_geographic_code",
                        message=str(e),
                        affected_records=[i]
                    ))
        
        return validation_results