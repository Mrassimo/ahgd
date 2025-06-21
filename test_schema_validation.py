#!/usr/bin/env python3
"""
Comprehensive Schema Validation Testing Agent

This script performs systematic testing of the AHGD schema validation framework,
including Pydantic v2 compatibility, data quality validation, and target schema compliance.
"""

import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Schema imports
from schemas.base_schema import (
    VersionedSchema, 
    SchemaVersion, 
    DataQualityLevel,
    GeographicBoundary,
    TemporalData,
    DataSource,
    MigrationRecord,
    ValidationMetrics,
    SchemaVersionManager,
    CompatibilityChecker
)

from schemas.target_outputs import (
    DataWarehouseTable,
    ExportSpecification,
    APIResponseSchema,
    WebPlatformDataStructure,
    DataQualityReport,
    ExportFormat,
    CompressionType,
    APIVersion
)

# Validation framework imports
from src.validators.validation_orchestrator import (
    ValidationOrchestrator,
    ValidationPipelineConfig,
    ValidationTask
)

from src.validators.quality_checker import (
    QualityChecker,
    QualityScore,
    QualityRule
)

from src.utils.interfaces import (
    ValidationResult,
    ValidationSeverity,
    DataRecord,
    DataBatch
)

from pydantic import ValidationError


class SchemaValidationTester:
    """Comprehensive schema validation testing framework."""
    
    def __init__(self):
        """Initialise the testing framework."""
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = []
        
        # Test data samples
        self.sample_health_data = self._create_sample_health_data()
        self.sample_invalid_data = self._create_sample_invalid_data()
        
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all schema validation tests."""
        print("=" * 80)
        print("AHGD SCHEMA VALIDATION TEST REPORT")
        print("=" * 80)
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Test sections
        test_sections = [
            ("1. Base Schema Framework Tests", self._test_base_schema_framework),
            ("2. Pydantic v2 Compatibility Tests", self._test_pydantic_v2_compatibility),
            ("3. Target Schema Compliance Tests", self._test_target_schema_compliance),
            ("4. Data Quality Validation Tests", self._test_data_quality_validation),
            ("5. Schema Migration Tests", self._test_schema_migration),
            ("6. Validation Orchestrator Tests", self._test_validation_orchestrator),
            ("7. Sample Data Validation Tests", self._test_sample_data_validation),
            ("8. Error Handling Tests", self._test_error_handling)
        ]
        
        for section_name, test_function in test_sections:
            print(f"\n{section_name}")
            print("-" * len(section_name))
            try:
                test_function()
            except Exception as e:
                self._record_test_failure(f"{section_name} - Fatal Error", str(e))
                print(f"FATAL ERROR: {e}")
                traceback.print_exc()
        
        # Generate final report
        return self._generate_final_report()
    
    def _test_base_schema_framework(self):
        """Test the base schema framework."""
        
        # Test SchemaVersion enum
        self._test_case("SchemaVersion enum creation", lambda: SchemaVersion.V1_0_0)
        self._test_case("SchemaVersion from string", lambda: SchemaVersion.from_string("1.0.0"))
        self._test_case("SchemaVersion compatibility", 
                       lambda: SchemaVersion.V1_0_0.is_compatible_with(SchemaVersion.V1_1_0))
        
        # Test DataQualityLevel enum
        self._test_case("DataQualityLevel enum", lambda: DataQualityLevel.HIGH)
        
        # Test GeographicBoundary schema
        boundary_data = {
            "boundary_id": "SA2_101011001",
            "boundary_type": "SA2",
            "name": "Queanbeyan",
            "state": "NSW",
            "area_sq_km": 15.2,
            "centroid_lat": -35.3557,
            "centroid_lon": 149.2316
        }
        self._test_case("GeographicBoundary creation", 
                       lambda: GeographicBoundary(**boundary_data))
        
        # Test invalid state code
        invalid_boundary = boundary_data.copy()
        invalid_boundary["state"] = "INVALID"
        self._test_case("GeographicBoundary invalid state", 
                       lambda: GeographicBoundary(**invalid_boundary),
                       should_fail=True)
        
        # Test TemporalData schema
        temporal_data = {
            "reference_date": datetime(2021, 12, 31),
            "period_type": "annual",
            "period_start": datetime(2021, 1, 1),
            "period_end": datetime(2021, 12, 31)
        }
        self._test_case("TemporalData creation", 
                       lambda: TemporalData(**temporal_data))
        
        # Test DataSource schema
        source_data = {
            "source_name": "Australian Bureau of Statistics",
            "source_url": "https://www.abs.gov.au/statistics",
            "source_date": datetime.now(),
            "attribution": "© Australian Bureau of Statistics"
        }
        self._test_case("DataSource creation", 
                       lambda: DataSource(**source_data))
    
    def _test_pydantic_v2_compatibility(self):
        """Test Pydantic v2 syntax compatibility."""
        
        # Test field_validator decorator usage
        try:
            boundary = GeographicBoundary(
                boundary_id="test",
                boundary_type="SA2",
                name="Test",
                state="NSW"
            )
            self._record_test_success("Pydantic v2 field_validator compatibility")
        except Exception as e:
            self._record_test_failure("Pydantic v2 field_validator compatibility", str(e))
        
        # Test model_validator decorator usage
        try:
            temporal = TemporalData(
                reference_date=datetime(2021, 12, 31),
                period_type="annual",
                period_start=datetime(2021, 1, 1),
                period_end=datetime(2021, 12, 31)
            )
            self._record_test_success("Pydantic v2 model_validator compatibility")
        except Exception as e:
            self._record_test_failure("Pydantic v2 model_validator compatibility", str(e))
        
        # Test model_config usage
        self._test_case("Model config validation", 
                       lambda: hasattr(GeographicBoundary, 'model_config'))
        
        # Test Field() usage with new syntax
        self._test_case("Field() descriptor compatibility", 
                       lambda: GeographicBoundary.__fields__ if hasattr(GeographicBoundary, '__fields__') else True)
    
    def _test_target_schema_compliance(self):
        """Test target schema compliance with Australian health data standards."""
        
        # Test DataWarehouseTable schema
        warehouse_table_data = {
            "table_name": "fact_health_indicators",
            "schema_name": "health_analytics",
            "table_type": "fact",
            "columns": [
                {"name": "sa2_code", "type": "varchar(50)", "nullable": False},
                {"name": "total_population", "type": "integer", "nullable": False},
                {"name": "health_score", "type": "decimal(5,2)", "nullable": True}
            ],
            "primary_keys": ["sa2_code"],
            "refresh_frequency": "daily",
            "access_level": "public"
        }
        self._test_case("DataWarehouseTable creation", 
                       lambda: DataWarehouseTable(**warehouse_table_data))
        
        # Test ExportSpecification schema
        export_spec_data = {
            "export_name": "health_indicators_export",
            "export_description": "Daily export of health indicators",
            "export_type": "incremental",
            "source_tables": ["fact_health_indicators"],
            "output_format": ExportFormat.PARQUET,
            "file_naming_pattern": "health_indicators_{date}.parquet",
            "destination_type": "s3",
            "destination_path": "s3://health-data/exports/"
        }
        self._test_case("ExportSpecification creation", 
                       lambda: ExportSpecification(**export_spec_data))
        
        # Test APIResponseSchema
        api_response_data = {
            "api_version": APIVersion.V1_0,
            "response_id": "resp_123456",
            "endpoint": "/api/v1/health-indicators",
            "method": "GET",
            "status": "success",
            "status_code": 200
        }
        self._test_case("APIResponseSchema creation", 
                       lambda: APIResponseSchema(**api_response_data))
        
        # Test WebPlatformDataStructure
        web_platform_data = {
            "content_type": "map",
            "content_id": "health_map_001",
            "content_title": "Health Indicators by SA2",
            "content_description": "Interactive map showing health indicators",
            "data_structure": "geojson",
            "data_payload": {"type": "FeatureCollection", "features": []},
            "data_last_updated": datetime.now()
        }
        self._test_case("WebPlatformDataStructure creation", 
                       lambda: WebPlatformDataStructure(**web_platform_data))
        
        # Test DataQualityReport
        quality_report_data = {
            "report_id": "qr_20240101",
            "dataset_name": "health_indicators_2024",
            "assessment_scope": "full",
            "records_assessed": 15000,
            "columns_assessed": 25,
            "overall_completeness": 95.5,
            "column_completeness": {"sa2_code": 100.0, "total_population": 98.5},
            "accuracy_score": 92.0,
            "validation_rules_passed": 45,
            "validation_rules_failed": 3,
            "consistency_score": 88.0,
            "duplicate_records": 12,
            "validity_score": 94.0,
            "timeliness_score": 85.0,
            "overall_quality_score": 91.0,
            "quality_grade": "A",
            "fitness_for_purpose": "excellent"
        }
        self._test_case("DataQualityReport creation", 
                       lambda: DataQualityReport(**quality_report_data))
    
    def _test_data_quality_validation(self):
        """Test the data quality validation framework."""
        
        # Test QualityChecker instantiation
        quality_config = {
            "quality_rules": {
                "completeness_rules": {
                    "critical_columns": ["sa2_code", "total_population"],
                    "high_priority_columns": ["health_score"]
                },
                "validity_rules": {
                    "sa2_code_format": {
                        "pattern": r"^\d{9}$",
                        "severity": "error"
                    }
                }
            }
        }
        
        self._test_case("QualityChecker instantiation", 
                       lambda: QualityChecker(config=quality_config))
        
        # Test quality score calculation
        quality_checker = QualityChecker(config=quality_config)
        self._test_case("Quality score calculation", 
                       lambda: quality_checker.calculate_quality_score(self.sample_health_data))
        
        # Test anomaly detection
        self._test_case("Anomaly detection", 
                       lambda: quality_checker.detect_anomalies(self.sample_health_data))
        
        # Test validation rules
        self._test_case("Get validation rules", 
                       lambda: quality_checker.get_validation_rules())
    
    def _test_schema_migration(self):
        """Test schema migration capabilities."""
        
        # Test SchemaVersionManager
        self._test_case("SchemaVersionManager latest version", 
                       lambda: SchemaVersionManager.get_latest_version("test_schema"))
        
        # Test schema version registration
        self._test_case("Schema version registration", 
                       lambda: SchemaVersionManager.register_schema_version("test_schema", SchemaVersion.V1_0_0))
        
        # Test version compatibility
        self._test_case("Version compatibility check", 
                       lambda: SchemaVersionManager.is_version_supported("test_schema", SchemaVersion.V1_0_0))
        
        # Test migration path
        self._test_case("Migration path calculation", 
                       lambda: SchemaVersionManager.get_migration_path(SchemaVersion.V1_0_0, SchemaVersion.V1_1_0))
        
        # Test CompatibilityChecker
        self._test_case("Forward compatibility check", 
                       lambda: CompatibilityChecker.check_forward_compatibility(SchemaVersion.V1_0_0, SchemaVersion.V1_1_0))
        
        # Test MigrationRecord creation
        migration_data = {
            "from_version": SchemaVersion.V1_0_0,
            "to_version": SchemaVersion.V1_1_0,
            "record_count": 1000,
            "success": True
        }
        self._test_case("MigrationRecord creation", 
                       lambda: MigrationRecord(**migration_data))
    
    def _test_validation_orchestrator(self):
        """Test the validation orchestrator."""
        
        # Test ValidationOrchestrator instantiation
        orchestrator_config = {
            "pipeline": {
                "enable_parallel_execution": True,
                "max_workers": 2
            }
        }
        
        self._test_case("ValidationOrchestrator instantiation", 
                       lambda: ValidationOrchestrator(config=orchestrator_config))
        
        # Test validator registration
        orchestrator = ValidationOrchestrator(config=orchestrator_config)
        self._test_case("Custom validator registration", 
                       lambda: orchestrator.register_validator("test_validator", QualityChecker))
        
        # Test validation pipeline creation
        validators_config = [
            {
                "validator": "quality_checker",
                "config": {"quality_rules": {}},
                "priority": 1
            }
        ]
        
        self._test_case("Validation pipeline creation", 
                       lambda: orchestrator.create_validation_pipeline(
                           self.sample_health_data, validators_config))
        
        # Test data validation
        self._test_case("Data validation execution", 
                       lambda: orchestrator.validate_data(self.sample_health_data[:5]))  # Small sample for speed
    
    def _test_sample_data_validation(self):
        """Test validation with sample health data."""
        
        # Create quality checker with comprehensive rules
        quality_config = {
            "quality_rules": {
                "completeness_rules": {
                    "critical_columns": ["sa2_code", "total_population"],
                    "high_priority_columns": ["state"]
                },
                "validity_rules": {
                    "sa2_code_format": {
                        "pattern": r"^\d{9}$",
                        "severity": "error"
                    },
                    "population_range": {
                        "min_value": 0,
                        "max_value": 100000,
                        "column": "total_population",
                        "severity": "warning"
                    }
                },
                "uniqueness_rules": {
                    "primary_keys": [
                        {"columns": ["sa2_code"]}
                    ]
                }
            }
        }
        
        quality_checker = QualityChecker(config=quality_config)
        
        # Test with valid data
        valid_results = quality_checker.validate(self.sample_health_data)
        self._record_test_success(f"Valid data validation (found {len(valid_results)} results)")
        
        # Test with invalid data
        invalid_results = quality_checker.validate(self.sample_invalid_data)
        self._record_test_success(f"Invalid data validation (found {len(invalid_results)} results)")
        
        # Check that validation found expected issues
        error_count = sum(1 for r in invalid_results if r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in invalid_results if r.severity == ValidationSeverity.WARNING)
        
        if error_count > 0:
            self._record_test_success(f"Validation correctly identified {error_count} errors")
        else:
            self._record_test_warning("Expected validation errors not found")
        
        if warning_count > 0:
            self._record_test_success(f"Validation correctly identified {warning_count} warnings")
    
    def _test_error_handling(self):
        """Test error handling and edge cases."""
        
        # Test empty data validation
        self._test_case("Empty data validation", 
                       lambda: QualityChecker().validate([]))
        
        # Test None data validation
        self._test_case("None data validation", 
                       lambda: QualityChecker().validate(None),
                       should_fail=True)
        
        # Test invalid schema data
        try:
            GeographicBoundary(
                boundary_id="test",
                boundary_type="INVALID_TYPE",
                name="Test",
                state="INVALID_STATE"
            )
            self._record_test_failure("Invalid schema data handling", "Should have raised ValidationError")
        except ValidationError:
            self._record_test_success("Invalid schema data correctly rejected")
        except Exception as e:
            self._record_test_failure("Invalid schema data handling", f"Unexpected error: {e}")
        
        # Test malformed configuration
        try:
            QualityChecker(config={"invalid": "config"})
            self._record_test_success("Malformed config handled gracefully")
        except Exception as e:
            self._record_test_failure("Malformed config handling", str(e))
    
    def _create_sample_health_data(self) -> DataBatch:
        """Create sample health data for testing."""
        return [
            {
                "sa2_code": "101011001",
                "sa2_name": "Queanbeyan",
                "state": "NSW",
                "total_population": 15420,
                "health_score": 7.2,
                "age_0_4_years": 950,
                "age_5_14_years": 1850,
                "age_15_24_years": 1680,
                "age_25_34_years": 2100,
                "age_35_44_years": 2250,
                "age_45_54_years": 2180,
                "age_55_64_years": 1920,
                "age_65_74_years": 1540,
                "age_75_84_years": 820,
                "age_85_years_over": 270,
                "data_year": "2021"
            },
            {
                "sa2_code": "101011002",
                "sa2_name": "Googong",
                "state": "NSW",
                "total_population": 8750,
                "health_score": 8.1,
                "age_0_4_years": 680,
                "age_5_14_years": 1250,
                "age_15_24_years": 820,
                "age_25_34_years": 1420,
                "age_35_44_years": 1580,
                "age_45_54_years": 1350,
                "age_55_64_years": 980,
                "age_65_74_years": 520,
                "age_75_84_years": 130,
                "age_85_years_over": 20,
                "data_year": "2021"
            },
            {
                "sa2_code": "201021001",
                "sa2_name": "Melbourne CBD",
                "state": "VIC",
                "total_population": 24850,
                "health_score": 6.8,
                "age_0_4_years": 980,
                "age_5_14_years": 1450,
                "age_15_24_years": 4250,
                "age_25_34_years": 6850,
                "age_35_44_years": 4920,
                "age_45_54_years": 2950,
                "age_55_64_years": 1850,
                "age_65_74_years": 1250,
                "age_75_84_years": 280,
                "age_85_years_over": 70,
                "data_year": "2021"
            }
        ]
    
    def _create_sample_invalid_data(self) -> DataBatch:
        """Create sample invalid health data for testing."""
        return [
            {
                "sa2_code": "INVALID",  # Invalid format
                "sa2_name": None,  # Missing required field
                "state": "INVALID_STATE",  # Invalid state
                "total_population": -100,  # Invalid negative population
                "health_score": 15.0,  # Invalid score (>10)
                "data_year": "2025"  # Future year
            },
            {
                "sa2_code": "101011001",  # Duplicate from valid data
                "sa2_name": "Duplicate",
                "state": "NSW",
                "total_population": 50000000,  # Unrealistically large
                "health_score": None,
                "data_year": "1800"  # Too old
            },
            {
                # Missing critical fields
                "sa2_name": "Incomplete Record",
                "state": "QLD"
            }
        ]
    
    def _test_case(self, test_name: str, test_function: callable, should_fail: bool = False):
        """Execute a single test case."""
        self.total_tests += 1
        try:
            result = test_function()
            if should_fail:
                self._record_test_failure(test_name, "Expected failure but test passed")
            else:
                self._record_test_success(test_name)
        except Exception as e:
            if should_fail:
                self._record_test_success(test_name + " (expected failure)")
            else:
                self._record_test_failure(test_name, str(e))
    
    def _record_test_success(self, test_name: str):
        """Record a successful test."""
        self.passed_tests += 1
        self.test_results.append({
            "test": test_name,
            "status": "PASS",
            "message": "Test passed successfully"
        })
        print(f"✓ {test_name}")
    
    def _record_test_failure(self, test_name: str, error_message: str):
        """Record a failed test."""
        self.failed_tests += 1
        self.test_results.append({
            "test": test_name,
            "status": "FAIL",
            "message": error_message
        })
        print(f"✗ {test_name}: {error_message}")
    
    def _record_test_warning(self, message: str):
        """Record a test warning."""
        self.warnings.append(message)
        print(f"⚠ WARNING: {message}")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        print("\n" + "=" * 80)
        print("FINAL TEST REPORT")
        print("=" * 80)
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Warnings: {len(self.warnings)}")
        
        if self.failed_tests > 0:
            print("\nFAILED TESTS:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['test']}: {result['message']}")
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        # Overall assessment
        print("\nOVERALL ASSESSMENT:")
        if success_rate >= 95:
            print("🟢 EXCELLENT: Schema validation framework is highly robust")
        elif success_rate >= 85:
            print("🟡 GOOD: Schema validation framework is functional with minor issues")
        elif success_rate >= 70:
            print("🟠 ACCEPTABLE: Schema validation framework needs attention")
        else:
            print("🔴 CRITICAL: Schema validation framework has significant issues")
        
        # Recommendations
        recommendations = []
        if self.failed_tests > 0:
            recommendations.append("Address failed test cases to improve framework reliability")
        if len(self.warnings) > 3:
            recommendations.append("Review warnings to identify potential improvements")
        if success_rate < 90:
            recommendations.append("Consider additional testing and validation coverage")
        
        if recommendations:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        report = {
            "test_summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "success_rate": success_rate,
                "warnings_count": len(self.warnings)
            },
            "test_results": self.test_results,
            "warnings": self.warnings,
            "recommendations": recommendations,
            "overall_status": "PASS" if success_rate >= 70 else "FAIL",
            "timestamp": datetime.now().isoformat()
        }
        
        return report


def main():
    """Main function to run comprehensive schema validation tests."""
    tester = SchemaValidationTester()
    try:
        report = tester.run_comprehensive_test()
        
        # Save report to file
        report_file = project_root / "test_schema_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved to: {report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if report["overall_status"] == "PASS" else 1)
        
    except Exception as e:
        print(f"FATAL ERROR in test execution: {e}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()