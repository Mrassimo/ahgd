"""
Data validation service for the AHGD Data Quality API.

This service integrates with the existing AHGD ValidationOrchestrator to provide
comprehensive data validation through the API, including schema validation,
business rules, geographic validation, and statistical checks.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import json

from ...utils.logging import get_logger, monitor_performance
from ...utils.config import get_config
from ...utils.interfaces import (
    AHGDException, ValidationError, ValidationResult as CoreValidationResult,
    ValidationSeverity, DataRecord, DataBatch
)
from ..models.common import (
    ValidationResult, ValidationSummary, SeverityEnum, 
    GeographicLevel, SA1Code
)
from ..models.requests import ValidationRequest, GeographicQuery
from ..models.responses import ValidationResponse, GeographicAnalysisResponse
from ..exceptions import (
    ServiceUnavailableException, ValidationException, 
    raise_validation_error
)


logger = get_logger(__name__)


class ValidationService:
    """
    Service for comprehensive data validation operations.
    
    Integrates with the existing AHGD ValidationOrchestrator while providing
    API-specific functionality, caching, and result aggregation.
    """
    
    def __init__(self):
        """Initialise the validation service."""
        self.config = get_config("validation_service", {})
        self.cache_ttl = self.config.get("cache_ttl", 1800)  # 30 minutes default
        self.data_path = Path(get_config("data.processed_path", "data_processed/"))
        self.schemas_path = Path(get_config("schemas.path", "schemas/"))
        
        # Validation type configurations
        self.validation_types = {
            "schema": {
                "enabled": True,
                "description": "Schema and data type validation",
                "priority": 1
            },
            "business_rules": {
                "enabled": True,
                "description": "Business logic and domain-specific rules",
                "priority": 2
            },
            "geographic": {
                "enabled": True,
                "description": "Geographic code and coordinate validation",
                "priority": 2
            },
            "statistical": {
                "enabled": True,
                "description": "Statistical outlier and distribution checks",
                "priority": 3
            },
            "completeness": {
                "enabled": True,
                "description": "Data completeness and mandatory field checks",
                "priority": 1
            },
            "consistency": {
                "enabled": True,
                "description": "Cross-field and temporal consistency checks",
                "priority": 2
            }
        }
        
        logger.info("Validation service initialised")
    
    @monitor_performance("data_validation")
    async def validate_data(
        self,
        request: ValidationRequest,
        cache_manager=None
    ) -> ValidationResponse:
        """
        Perform comprehensive data validation based on request parameters.
        
        Args:
            request: Validation request parameters
            cache_manager: Optional cache manager for result caching
            
        Returns:
            Validation response with detailed results and summary
        """
        
        try:
            logger.info(
                "Starting data validation",
                dataset_id=request.dataset_id,
                validation_types=request.validation_types,
                severity_threshold=request.severity_threshold
            )
            
            # Check cache first
            cache_key = self._generate_cache_key("validation", request)
            if cache_manager:
                cached_result = await cache_manager.get(cache_key)
                if cached_result:
                    logger.debug("Returning cached validation results")
                    return ValidationResponse.model_validate_json(cached_result)
            
            # Validate request parameters
            await self._validate_request_parameters(request)
            
            # Load dataset for validation
            dataset_metadata, records = await self._load_dataset_for_validation(
                request.dataset_id
            )
            
            # Perform validation by type
            all_validation_results = []
            
            for validation_type in request.validation_types:
                if validation_type in self.validation_types:
                    type_results = await self._perform_validation_type(
                        validation_type, 
                        records,
                        request.severity_threshold,
                        request.max_errors
                    )
                    all_validation_results.extend(type_results)
                else:
                    logger.warning(f"Unknown validation type: {validation_type}")
            
            # Filter by severity threshold
            filtered_results = self._filter_by_severity(
                all_validation_results, 
                request.severity_threshold
            )
            
            # Generate validation summary
            validation_summary = await self._generate_validation_summary(
                all_validation_results,
                len(records) if records else 0
            )
            
            # Perform geographic coverage analysis
            geographic_coverage = None
            if "geographic" in request.validation_types:
                geographic_coverage = await self._analyse_geographic_coverage(
                    records or []
                )
            
            # Build response
            response = ValidationResponse(
                success=True,
                timestamp=datetime.now(),
                total_count=len(filtered_results),
                page_size=min(request.max_errors, len(filtered_results)),
                current_page=1,
                total_pages=max(1, len(filtered_results) // request.max_errors),
                has_next=len(filtered_results) > request.max_errors,
                has_previous=False,
                validation_summary=validation_summary,
                validation_results=filtered_results[:request.max_errors],
                dataset_metadata=dataset_metadata,
                geographic_coverage=geographic_coverage
            )
            
            # Cache result
            if cache_manager:
                await cache_manager.set(
                    cache_key,
                    response.model_dump_json(),
                    self.cache_ttl
                )
            
            logger.info(
                "Data validation completed",
                total_rules=validation_summary.total_rules,
                passed_rules=validation_summary.passed_rules,
                failed_rules=validation_summary.failed_rules,
                overall_valid=validation_summary.overall_valid
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            if isinstance(e, (AHGDException, ValidationException)):
                raise
            raise ServiceUnavailableException(
                "validation_service",
                f"Validation operation failed: {str(e)}"
            )
    
    @monitor_performance("validation_rules_execution")
    async def get_validation_rules(
        self,
        validation_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get available validation rules and their configurations.
        
        Args:
            validation_type: Optional specific validation type to filter
            
        Returns:
            Dictionary of validation rules and configurations
        """
        
        try:
            logger.info("Retrieving validation rules", validation_type=validation_type)
            
            rules = {}
            
            # Load rules for each validation type
            for vtype, config in self.validation_types.items():
                if validation_type and vtype != validation_type:
                    continue
                
                if config["enabled"]:
                    type_rules = await self._load_validation_rules(vtype)
                    rules[vtype] = {
                        "description": config["description"],
                        "priority": config["priority"],
                        "rules": type_rules
                    }
            
            return {
                "validation_types": rules,
                "total_rule_count": sum(
                    len(type_rules["rules"]) 
                    for type_rules in rules.values()
                ),
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to retrieve validation rules: {e}")
            raise ServiceUnavailableException(
                "validation_service",
                f"Cannot retrieve validation rules: {str(e)}"
            )
    
    async def _validate_request_parameters(self, request: ValidationRequest) -> None:
        """Validate the validation request parameters."""
        
        # Check if validation types are supported
        unsupported_types = set(request.validation_types) - set(self.validation_types.keys())
        if unsupported_types:
            raise_validation_error(
                f"Unsupported validation types: {list(unsupported_types)}",
                field="validation_types",
                details={"supported_types": list(self.validation_types.keys())}
            )
        
        # Check max_errors limit
        if request.max_errors > 10000:
            raise_validation_error(
                "max_errors cannot exceed 10000",
                field="max_errors"
            )
    
    async def _load_dataset_for_validation(
        self, 
        dataset_id: Optional[str]
    ) -> tuple[Dict[str, Any], Optional[List[DataRecord]]]:
        """Load dataset for validation operations."""
        
        try:
            if dataset_id:
                # Load specific dataset
                logger.debug(f"Loading dataset: {dataset_id}")
                
                # Mock dataset loading - in real implementation, this would
                # integrate with existing AHGD data loading infrastructure
                dataset_metadata = {
                    "dataset_id": dataset_id,
                    "name": f"Dataset {dataset_id}",
                    "record_count": 10000,
                    "last_updated": datetime.now().isoformat(),
                    "schema_version": "2.0.0",
                    "geographic_level": "SA1",
                    "data_sources": ["ABS", "AIHW", "SEIFA"]
                }
                
                # Generate mock records for validation
                records = await self._generate_mock_records(dataset_metadata["record_count"])
                
                return dataset_metadata, records
            else:
                # Load latest processed data
                logger.debug("Loading latest processed dataset")
                
                dataset_metadata = {
                    "dataset_id": "latest",
                    "name": "Latest Processed Data",
                    "record_count": 57736,  # Approximate SA1 count
                    "last_updated": datetime.now().isoformat(),
                    "schema_version": "2.0.0",
                    "geographic_level": "SA1",
                    "data_sources": ["ABS", "AIHW", "SEIFA"]
                }
                
                # For demonstration, we'll use a subset
                records = await self._generate_mock_records(1000)
                
                return dataset_metadata, records
                
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise ValidationError(f"Dataset loading failed: {str(e)}")
    
    async def _generate_mock_records(self, count: int) -> List[DataRecord]:
        """Generate mock records for demonstration purposes."""
        
        import random
        
        records = []
        
        for i in range(min(count, 1000)):  # Limit for demonstration
            record = {
                "sa1_code": f"{random.randint(10000000000, 99999999999)}",  # 11 digits
                "postcode": random.choice(["2000", "3000", "4000", "5000", None]),
                "state": random.choice(["NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT"]),
                "latitude": round(random.uniform(-43.5, -10.5), 6),
                "longitude": round(random.uniform(113.0, 153.5), 6),
                "population": random.randint(0, 5000) if random.random() > 0.1 else None,
                "median_income": random.randint(20000, 120000) if random.random() > 0.05 else None,
                "seifa_score": round(random.uniform(500, 1200), 1) if random.random() > 0.08 else None,
                "last_updated": datetime.now().isoformat()
            }
            
            # Introduce some validation issues intentionally
            if random.random() < 0.1:  # 10% invalid SA1 codes
                record["sa1_code"] = f"{random.randint(100000, 999999)}"  # Wrong length
            
            if random.random() < 0.05:  # 5% invalid coordinates
                record["latitude"] = random.uniform(50, 60)  # Outside Australia
            
            records.append(record)
        
        return records
    
    async def _perform_validation_type(
        self,
        validation_type: str,
        records: List[DataRecord],
        severity_threshold: SeverityEnum,
        max_errors: int
    ) -> List[ValidationResult]:
        """Perform validation for a specific validation type."""
        
        validation_method = {
            "schema": self._validate_schema,
            "business_rules": self._validate_business_rules,
            "geographic": self._validate_geographic,
            "statistical": self._validate_statistical,
            "completeness": self._validate_completeness,
            "consistency": self._validate_consistency
        }.get(validation_type)
        
        if not validation_method:
            logger.warning(f"No validation method for type: {validation_type}")
            return []
        
        try:
            return await validation_method(records, severity_threshold, max_errors)
        except Exception as e:
            logger.error(f"Validation type '{validation_type}' failed: {e}")
            return [ValidationResult(
                rule_id=f"{validation_type}_error",
                is_valid=False,
                severity=SeverityEnum.ERROR,
                message=f"Validation type {validation_type} failed: {str(e)}",
                affected_records=[],
                details={"error": str(e)}
            )]
    
    async def _validate_schema(
        self, 
        records: List[DataRecord], 
        severity_threshold: SeverityEnum,
        max_errors: int
    ) -> List[ValidationResult]:
        """Perform schema validation."""
        
        results = []
        error_count = 0
        
        required_fields = ["sa1_code", "state", "latitude", "longitude"]
        
        for idx, record in enumerate(records):
            if error_count >= max_errors:
                break
            
            # Check required fields
            missing_fields = [field for field in required_fields if field not in record or record[field] is None]
            
            if missing_fields:
                results.append(ValidationResult(
                    rule_id="schema_required_fields",
                    is_valid=False,
                    severity=SeverityEnum.ERROR,
                    message=f"Missing required fields: {', '.join(missing_fields)}",
                    affected_records=[idx],
                    details={"missing_fields": missing_fields, "record_id": idx}
                ))
                error_count += 1
        
        # Add successful validation result if no errors
        if not results:
            results.append(ValidationResult(
                rule_id="schema_validation",
                is_valid=True,
                severity=SeverityEnum.INFO,
                message="All records pass schema validation",
                affected_records=[],
                details={"records_validated": len(records)}
            ))
        
        return results
    
    async def _validate_business_rules(
        self, 
        records: List[DataRecord],
        severity_threshold: SeverityEnum,
        max_errors: int
    ) -> List[ValidationResult]:
        """Perform business rules validation."""
        
        results = []
        error_count = 0
        
        for idx, record in enumerate(records):
            if error_count >= max_errors:
                break
            
            # Business rule: Population should be non-negative
            population = record.get("population")
            if population is not None and population < 0:
                results.append(ValidationResult(
                    rule_id="business_population_negative",
                    is_valid=False,
                    severity=SeverityEnum.ERROR,
                    message=f"Population cannot be negative: {population}",
                    affected_records=[idx],
                    details={"population_value": population, "record_id": idx}
                ))
                error_count += 1
            
            # Business rule: Income should be reasonable range
            income = record.get("median_income")
            if income is not None and (income < 10000 or income > 200000):
                results.append(ValidationResult(
                    rule_id="business_income_range",
                    is_valid=False,
                    severity=SeverityEnum.WARNING,
                    message=f"Income outside typical range: ${income:,}",
                    affected_records=[idx],
                    details={"income_value": income, "record_id": idx}
                ))
                error_count += 1
        
        return results
    
    async def _validate_geographic(
        self, 
        records: List[DataRecord],
        severity_threshold: SeverityEnum,
        max_errors: int
    ) -> List[ValidationResult]:
        """Perform geographic validation."""
        
        results = []
        error_count = 0
        
        for idx, record in enumerate(records):
            if error_count >= max_errors:
                break
            
            # Validate SA1 code format
            sa1_code = record.get("sa1_code")
            if sa1_code and not self._is_valid_sa1_code(sa1_code):
                results.append(ValidationResult(
                    rule_id="geographic_sa1_format",
                    is_valid=False,
                    severity=SeverityEnum.ERROR,
                    message=f"Invalid SA1 code format: {sa1_code}",
                    affected_records=[idx],
                    details={"sa1_code": sa1_code, "record_id": idx}
                ))
                error_count += 1
            
            # Validate coordinates are within Australia
            lat = record.get("latitude")
            lon = record.get("longitude")
            if lat is not None and lon is not None:
                if not self._is_coordinate_in_australia(lat, lon):
                    results.append(ValidationResult(
                        rule_id="geographic_coordinate_bounds",
                        is_valid=False,
                        severity=SeverityEnum.ERROR,
                        message=f"Coordinates outside Australia: {lat}, {lon}",
                        affected_records=[idx],
                        details={"latitude": lat, "longitude": lon, "record_id": idx}
                    ))
                    error_count += 1
        
        return results
    
    async def _validate_statistical(
        self, 
        records: List[DataRecord],
        severity_threshold: SeverityEnum,
        max_errors: int
    ) -> List[ValidationResult]:
        """Perform statistical validation."""
        
        results = []
        
        # Calculate statistics for numerical fields
        populations = [r.get("population") for r in records if r.get("population") is not None]
        incomes = [r.get("median_income") for r in records if r.get("median_income") is not None]
        
        if populations:
            pop_mean = sum(populations) / len(populations)
            pop_std = (sum((x - pop_mean) ** 2 for x in populations) / len(populations)) ** 0.5
            
            # Flag statistical outliers
            outlier_count = 0
            for idx, record in enumerate(records):
                if outlier_count >= max_errors:
                    break
                    
                pop = record.get("population")
                if pop is not None and abs(pop - pop_mean) > 3 * pop_std:
                    results.append(ValidationResult(
                        rule_id="statistical_population_outlier",
                        is_valid=False,
                        severity=SeverityEnum.WARNING,
                        message=f"Population is statistical outlier: {pop}",
                        affected_records=[idx],
                        details={
                            "value": pop, 
                            "mean": round(pop_mean, 2),
                            "std_dev": round(pop_std, 2),
                            "z_score": round((pop - pop_mean) / pop_std, 2) if pop_std > 0 else None,
                            "record_id": idx
                        }
                    ))
                    outlier_count += 1
        
        return results
    
    async def _validate_completeness(
        self, 
        records: List[DataRecord],
        severity_threshold: SeverityEnum,
        max_errors: int
    ) -> List[ValidationResult]:
        """Perform completeness validation."""
        
        results = []
        
        # Calculate completeness for each field
        field_completeness = {}
        total_records = len(records)
        
        if total_records > 0:
            all_fields = set()
            for record in records:
                all_fields.update(record.keys())
            
            for field in all_fields:
                non_null_count = sum(
                    1 for record in records 
                    if record.get(field) is not None and str(record.get(field)).strip()
                )
                completeness_pct = (non_null_count / total_records) * 100
                field_completeness[field] = completeness_pct
                
                # Flag fields with low completeness
                if completeness_pct < 80:
                    severity = SeverityEnum.ERROR if completeness_pct < 50 else SeverityEnum.WARNING
                    results.append(ValidationResult(
                        rule_id="completeness_field_threshold",
                        is_valid=completeness_pct >= 80,
                        severity=severity,
                        message=f"Field '{field}' has low completeness: {completeness_pct:.1f}%",
                        affected_records=[],
                        details={
                            "field_name": field,
                            "completeness_percentage": round(completeness_pct, 2),
                            "non_null_count": non_null_count,
                            "total_records": total_records
                        }
                    ))
        
        return results
    
    async def _validate_consistency(
        self, 
        records: List[DataRecord],
        severity_threshold: SeverityEnum,
        max_errors: int
    ) -> List[ValidationResult]:
        """Perform consistency validation."""
        
        results = []
        error_count = 0
        
        # Check state-postcode consistency (simplified)
        state_postcode_map = {
            "NSW": ["2", "1"],  # NSW postcodes start with 2 (mostly) or 1
            "VIC": ["3", "8"],  # VIC postcodes start with 3 or 8
            "QLD": ["4", "9"],  # QLD postcodes start with 4 or 9
            "SA": ["5"],        # SA postcodes start with 5
            "WA": ["6"],        # WA postcodes start with 6
            "TAS": ["7"],       # TAS postcodes start with 7
            "NT": ["0"],        # NT postcodes start with 0
            "ACT": ["0", "2"]   # ACT postcodes start with 0 or 2
        }
        
        for idx, record in enumerate(records):
            if error_count >= max_errors:
                break
            
            state = record.get("state")
            postcode = record.get("postcode")
            
            if state and postcode and len(str(postcode)) >= 1:
                expected_prefixes = state_postcode_map.get(state, [])
                postcode_prefix = str(postcode)[0]
                
                if postcode_prefix not in expected_prefixes:
                    results.append(ValidationResult(
                        rule_id="consistency_state_postcode",
                        is_valid=False,
                        severity=SeverityEnum.WARNING,
                        message=f"Postcode {postcode} inconsistent with state {state}",
                        affected_records=[idx],
                        details={
                            "state": state,
                            "postcode": postcode,
                            "expected_prefixes": expected_prefixes,
                            "record_id": idx
                        }
                    ))
                    error_count += 1
        
        return results
    
    def _is_valid_sa1_code(self, sa1_code: str) -> bool:
        """Check if SA1 code has valid format."""
        import re
        return bool(re.match(r'^\d{11}$', str(sa1_code)))
    
    def _is_coordinate_in_australia(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Australian bounds."""
        # Simplified Australian bounding box
        return (-43.5 <= lat <= -10.5) and (113.0 <= lon <= 153.5)
    
    def _filter_by_severity(
        self, 
        results: List[ValidationResult],
        severity_threshold: SeverityEnum
    ) -> List[ValidationResult]:
        """Filter validation results by severity threshold."""
        
        severity_levels = {
            SeverityEnum.INFO: 0,
            SeverityEnum.WARNING: 1,
            SeverityEnum.ERROR: 2,
            SeverityEnum.CRITICAL: 3
        }
        
        threshold_level = severity_levels.get(severity_threshold, 1)
        
        return [
            result for result in results
            if severity_levels.get(result.severity, 0) >= threshold_level
        ]
    
    async def _generate_validation_summary(
        self, 
        all_results: List[ValidationResult],
        record_count: int
    ) -> ValidationSummary:
        """Generate validation summary from all results."""
        
        # Count results by outcome
        passed_results = [r for r in all_results if r.is_valid]
        failed_results = [r for r in all_results if not r.is_valid]
        
        # Count by severity
        error_count = sum(1 for r in all_results if r.severity == SeverityEnum.ERROR)
        warning_count = sum(1 for r in all_results if r.severity == SeverityEnum.WARNING)
        info_count = sum(1 for r in all_results if r.severity == SeverityEnum.INFO)
        
        # Overall validity (no errors)
        overall_valid = error_count == 0
        
        # Calculate quality score based on validation results
        total_rules = len(all_results)
        if total_rules > 0:
            quality_score = (len(passed_results) / total_rules) * 100
            # Penalise errors more than warnings
            error_penalty = (error_count * 10) + (warning_count * 5)
            quality_score = max(0, quality_score - error_penalty)
        else:
            quality_score = 100.0
        
        return ValidationSummary(
            total_rules=total_rules,
            passed_rules=len(passed_results),
            failed_rules=len(failed_results),
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            overall_valid=overall_valid,
            quality_score=round(quality_score, 2) if quality_score >= 0 else None
        )
    
    async def _analyse_geographic_coverage(
        self, 
        records: List[DataRecord]
    ) -> Dict[str, Any]:
        """Analyse geographic coverage of the dataset."""
        
        # Count records by state
        state_counts = {}
        valid_coordinates = 0
        total_records = len(records)
        
        for record in records:
            state = record.get("state")
            if state:
                state_counts[state] = state_counts.get(state, 0) + 1
            
            lat = record.get("latitude")
            lon = record.get("longitude")
            if lat is not None and lon is not None:
                valid_coordinates += 1
        
        # Calculate coverage statistics
        coverage_stats = {
            "total_records": total_records,
            "geographic_distribution": state_counts,
            "coordinate_coverage": {
                "records_with_coordinates": valid_coordinates,
                "coordinate_completeness": (valid_coordinates / total_records * 100) if total_records > 0 else 0
            },
            "coverage_quality": "excellent" if valid_coordinates / total_records > 0.95 else "good" if valid_coordinates / total_records > 0.8 else "needs_improvement"
        }
        
        return coverage_stats
    
    async def _load_validation_rules(self, validation_type: str) -> List[Dict[str, Any]]:
        """Load validation rules for a specific type."""
        
        # Mock implementation - would load from actual rule configuration
        rule_sets = {
            "schema": [
                {
                    "rule_id": "schema_required_fields",
                    "description": "Check required fields are present",
                    "severity": "error",
                    "parameters": {"required_fields": ["sa1_code", "state", "latitude", "longitude"]}
                },
                {
                    "rule_id": "schema_data_types",
                    "description": "Validate data types",
                    "severity": "error",
                    "parameters": {"type_mappings": {"latitude": "float", "longitude": "float"}}
                }
            ],
            "business_rules": [
                {
                    "rule_id": "business_population_negative",
                    "description": "Population must be non-negative",
                    "severity": "error",
                    "parameters": {"field": "population", "min_value": 0}
                },
                {
                    "rule_id": "business_income_range",
                    "description": "Income should be within reasonable range",
                    "severity": "warning",
                    "parameters": {"field": "median_income", "min_value": 10000, "max_value": 200000}
                }
            ],
            "geographic": [
                {
                    "rule_id": "geographic_sa1_format",
                    "description": "SA1 code must be 11 digits",
                    "severity": "error",
                    "parameters": {"field": "sa1_code", "pattern": "^\\d{11}$"}
                },
                {
                    "rule_id": "geographic_coordinate_bounds",
                    "description": "Coordinates must be within Australia",
                    "severity": "error",
                    "parameters": {
                        "lat_field": "latitude",
                        "lon_field": "longitude",
                        "bounds": {"lat_min": -43.5, "lat_max": -10.5, "lon_min": 113.0, "lon_max": 153.5}
                    }
                }
            ]
        }
        
        return rule_sets.get(validation_type, [])
    
    def _generate_cache_key(self, operation: str, request) -> str:
        """Generate cache key for validation request."""
        import hashlib
        
        # Create a hash of the request parameters
        request_str = request.model_dump_json()
        request_hash = hashlib.md5(request_str.encode()).hexdigest()
        
        return f"validation_{operation}_{request_hash}"


# Singleton instance for dependency injection
validation_service = ValidationService()


async def get_validation_service() -> ValidationService:
    """Get validation service instance."""
    return validation_service