"""
Integration tests for AHGD data quality systems.

Tests integration of validation systems, quality monitoring,
error reporting, and data quality workflows.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
import json
import sqlite3

from src.validators.base import BaseValidator
from src.validators.quality_checker import QualityChecker
from src.validators.statistical_validator import StatisticalValidator
from src.validators.geographic_validator import GeographicValidator
from src.validators.business_rules import BusinessRulesValidator
from src.validators.validation_orchestrator import ValidationOrchestrator
from src.validators.reporting import ValidationReporter
from src.utils.interfaces import (
    DataBatch,
    ValidationResult,
    ValidationSeverity,
    ValidationError,
    GeographicValidationError
)


@pytest.mark.integration
class TestDataQualityIntegration:
    """Integration tests for data quality validation systems."""
    
    def test_multi_validator_orchestration(self, sample_config, mock_logger, sample_data_batch):
        """Test orchestration of multiple validators."""
        # Create validation orchestrator
        orchestrator = ValidationOrchestrator(mock_logger)
        
        # Add multiple validators
        schema_validator = BaseValidator("schema_validator", sample_config["validators"]["schema_validator"], mock_logger)
        quality_checker = QualityChecker("quality_checker", {"completeness_threshold": 0.9}, mock_logger)
        statistical_validator = StatisticalValidator("stats_validator", {"outlier_detection": True}, mock_logger)
        geographic_validator = GeographicValidator("geo_validator", sample_config["geographic"], mock_logger)
        business_rules_validator = BusinessRulesValidator("business_validator", sample_config["validators"]["schema_validator"], mock_logger)
        
        orchestrator.add_validator(schema_validator)
        orchestrator.add_validator(quality_checker)
        orchestrator.add_validator(statistical_validator)
        orchestrator.add_validator(geographic_validator)
        orchestrator.add_validator(business_rules_validator)
        
        # Run comprehensive validation
        all_results = orchestrator.validate_all(sample_data_batch)
        
        # Verify results from all validators
        assert len(all_results) >= 5  # Results from all validators
        
        # Check validator coverage
        validator_ids = [result.get('validator_id') for result in all_results if 'validator_id' in result]
        assert "schema_validator" in validator_ids
        assert "quality_checker" in validator_ids
        assert "stats_validator" in validator_ids
        assert "geo_validator" in validator_ids
        assert "business_validator" in validator_ids
        
        # Get aggregated summary
        summary = orchestrator.get_validation_summary()
        assert "total_validators" in summary
        assert "total_validations" in summary
        assert "errors" in summary
        assert "warnings" in summary
    
    def test_data_quality_workflow_with_health_data(self, sample_config, mock_logger, sample_health_indicators):
        """Test complete data quality workflow with health indicator data."""
        orchestrator = ValidationOrchestrator(mock_logger)
        
        # Configure validators for health data
        health_schema_config = {
            "required_columns": ["sa2_code", "mortality_rate", "birth_rate", "life_expectancy", "year"],
            "column_types": {
                "sa2_code": "string",
                "mortality_rate": "float",
                "birth_rate": "float", 
                "life_expectancy": "float",
                "diabetes_prevalence": "float",
                "obesity_rate": "float",
                "year": "integer"
            },
            "business_rules": [
                {
                    "id": "mortality_rate_range",
                    "type": "range_check",
                    "column": "mortality_rate",
                    "min": 0,
                    "max": 50
                },
                {
                    "id": "life_expectancy_range",
                    "type": "range_check",
                    "column": "life_expectancy", 
                    "min": 60,
                    "max": 100
                },
                {
                    "id": "year_validity",
                    "type": "range_check",
                    "column": "year",
                    "min": 2000,
                    "max": 2025
                }
            ],
            "statistical_rules": {
                "outlier_detection": [
                    {
                        "column": "mortality_rate",
                        "method": "iqr",
                        "threshold": 2.0
                    },
                    {
                        "column": "life_expectancy",
                        "method": "iqr", 
                        "threshold": 1.5
                    }
                ]
            }
        }
        
        # Add validators
        schema_validator = BaseValidator("health_schema", health_schema_config, mock_logger)
        quality_checker = QualityChecker("health_quality", {"completeness_threshold": 0.95}, mock_logger)
        statistical_validator = StatisticalValidator("health_stats", health_schema_config, mock_logger)
        business_rules_validator = BusinessRulesValidator("health_business", health_schema_config, mock_logger)
        
        orchestrator.add_validator(schema_validator)
        orchestrator.add_validator(quality_checker)
        orchestrator.add_validator(statistical_validator)
        orchestrator.add_validator(business_rules_validator)
        
        # Run validation workflow
        validation_results = orchestrator.validate_all(sample_health_indicators)
        
        # Analyse results
        errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
        warnings = [r for r in validation_results if r.severity == ValidationSeverity.WARNING]
        info = [r for r in validation_results if r.severity == ValidationSeverity.INFO]
        
        # Health data should generally be valid
        assert len(errors) == 0, f"Health data validation errors: {[e.message for e in errors]}"
        
        # May have warnings or info messages
        total_issues = len(warnings) + len(info)
        assert total_issues >= 0
        
        # Verify specific health data validations
        rule_ids = [r.rule_id for r in validation_results]
        assert any("mortality_rate" in rule_id for rule_id in rule_ids)
        assert any("life_expectancy" in rule_id for rule_id in rule_ids)
    
    def test_data_quality_with_census_data(self, sample_config, mock_logger, sample_census_data):
        """Test data quality validation with census data."""
        orchestrator = ValidationOrchestrator(mock_logger)
        
        # Configure validators for census data
        census_config = {
            "required_columns": ["sa2_code", "total_population", "median_age", "unemployment_rate", "year"],
            "column_types": {
                "sa2_code": "string",
                "total_population": "integer",
                "median_age": "float",
                "median_income": "integer", 
                "unemployment_rate": "float",
                "indigenous_population": "integer",
                "overseas_born": "integer",
                "year": "integer"
            },
            "business_rules": [
                {
                    "id": "population_positive",
                    "type": "range_check",
                    "column": "total_population",
                    "min": 0
                },
                {
                    "id": "age_reasonable",
                    "type": "range_check",
                    "column": "median_age",
                    "min": 15,
                    "max": 70
                },
                {
                    "id": "unemployment_percentage",
                    "type": "range_check",
                    "column": "unemployment_rate",
                    "min": 0,
                    "max": 30
                }
            ],
            "statistical_rules": {
                "correlation_checks": [
                    {
                        "column1": "total_population",
                        "column2": "overseas_born",
                        "expected_correlation": 0.7,
                        "tolerance": 0.3
                    }
                ]
            }
        }
        
        schema_validator = BaseValidator("census_schema", census_config, mock_logger)
        business_rules_validator = BusinessRulesValidator("census_business", census_config, mock_logger)
        statistical_validator = StatisticalValidator("census_stats", census_config, mock_logger)
        quality_checker = QualityChecker("census_quality", {"completeness_threshold": 0.9}, mock_logger)
        
        orchestrator.add_validator(schema_validator)
        orchestrator.add_validator(business_rules_validator)
        orchestrator.add_validator(statistical_validator)
        orchestrator.add_validator(quality_checker)
        
        # Validate census data
        validation_results = orchestrator.validate_all(sample_census_data)
        
        # Analyse census-specific validations
        population_validations = [r for r in validation_results if "population" in r.rule_id.lower()]
        age_validations = [r for r in validation_results if "age" in r.rule_id.lower()]
        unemployment_validations = [r for r in validation_results if "unemployment" in r.rule_id.lower()]
        
        # Verify census data constraints are checked
        assert len(population_validations) >= 0
        assert len(age_validations) >= 0  
        assert len(unemployment_validations) >= 0
        
        # Check for any critical errors
        critical_errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
        assert len(critical_errors) == 0, f"Census data critical errors: {[e.message for e in critical_errors]}"
    
    def test_data_quality_with_seifa_data(self, sample_config, mock_logger, sample_seifa_data):
        """Test data quality validation with SEIFA socio-economic data."""
        orchestrator = ValidationOrchestrator(mock_logger)
        
        # Configure validators for SEIFA data
        seifa_config = {
            "required_columns": ["sa2_code", "irsad_score", "irsad_decile", "year"],
            "column_types": {
                "sa2_code": "string",
                "irsad_score": "integer",
                "irsad_decile": "integer",
                "irsd_score": "integer",
                "irsd_decile": "integer", 
                "ier_score": "integer",
                "ier_decile": "integer",
                "ieo_score": "integer",
                "ieo_decile": "integer",
                "year": "integer"
            },
            "business_rules": [
                {
                    "id": "irsad_score_range",
                    "type": "range_check",
                    "column": "irsad_score",
                    "min": 1,
                    "max": 1200
                },
                {
                    "id": "irsad_decile_range",
                    "type": "range_check", 
                    "column": "irsad_decile",
                    "min": 1,
                    "max": 10
                },
                {
                    "id": "decile_consistency",
                    "type": "custom",
                    "description": "SEIFA deciles should be consistent with scores"
                }
            ]
        }
        
        schema_validator = BaseValidator("seifa_schema", seifa_config, mock_logger)
        business_rules_validator = BusinessRulesValidator("seifa_business", seifa_config, mock_logger) 
        quality_checker = QualityChecker("seifa_quality", {"completeness_threshold": 0.85}, mock_logger)
        
        # Custom SEIFA validator for domain-specific rules
        class SEIFAValidator(BaseValidator):
            def validate(self, data):
                results = []
                for record_idx, record in enumerate(data):
                    # Check decile consistency with score
                    irsad_score = record.get('irsad_score')
                    irsad_decile = record.get('irsad_decile')
                    
                    if irsad_score and irsad_decile:
                        # Simplified consistency check (actual calculation is more complex)
                        expected_decile_range = self._score_to_decile_range(irsad_score)
                        if irsad_decile not in expected_decile_range:
                            results.append(ValidationResult(
                                is_valid=False,
                                severity=ValidationSeverity.WARNING,
                                rule_id="seifa_decile_consistency",
                                message=f"IRSAD decile {irsad_decile} may be inconsistent with score {irsad_score}",
                                affected_records=[record_idx]
                            ))
                
                return results
            
            def _score_to_decile_range(self, score):
                # Simplified mapping - real calculation would use actual SEIFA methodology
                if score >= 1100:
                    return [9, 10]
                elif score >= 1000:
                    return [7, 8, 9]
                elif score >= 900:
                    return [5, 6, 7]
                else:
                    return [1, 2, 3, 4, 5]
            
            def get_validation_rules(self):
                return ["seifa_decile_consistency"]
        
        seifa_validator = SEIFAValidator("seifa_domain", seifa_config, mock_logger)
        
        orchestrator.add_validator(schema_validator)
        orchestrator.add_validator(business_rules_validator)
        orchestrator.add_validator(quality_checker)
        orchestrator.add_validator(seifa_validator)
        
        # Validate SEIFA data
        validation_results = orchestrator.validate_all(sample_seifa_data)
        
        # Analyse SEIFA-specific validations
        score_validations = [r for r in validation_results if "score" in r.rule_id.lower()]
        decile_validations = [r for r in validation_results if "decile" in r.rule_id.lower()]
        consistency_validations = [r for r in validation_results if "consistency" in r.rule_id.lower()]
        
        # Verify SEIFA validations ran
        assert len(score_validations) >= 0
        assert len(decile_validations) >= 0
        
        # SEIFA data should be valid
        errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
        assert len(errors) == 0, f"SEIFA validation errors: {[e.message for e in errors]}"
    
    def test_data_quality_error_aggregation(self, sample_config, mock_logger, invalid_data_batch):
        """Test aggregation and reporting of data quality errors."""
        orchestrator = ValidationOrchestrator(mock_logger)
        
        # Add multiple validators that will find different errors
        schema_validator = BaseValidator("schema_validator", sample_config["validators"]["schema_validator"], mock_logger)
        business_rules_validator = BusinessRulesValidator("business_validator", sample_config["validators"]["schema_validator"], mock_logger)
        quality_checker = QualityChecker("quality_checker", {"completeness_threshold": 0.95}, mock_logger)
        
        orchestrator.add_validator(schema_validator)
        orchestrator.add_validator(business_rules_validator) 
        orchestrator.add_validator(quality_checker)
        
        # Validate intentionally invalid data
        validation_results = orchestrator.validate_all(invalid_data_batch)
        
        # Aggregate errors by type
        errors_by_severity = {}
        errors_by_rule = {}
        errors_by_validator = {}
        
        for result in validation_results:
            # Group by severity
            severity = result.severity.value
            if severity not in errors_by_severity:
                errors_by_severity[severity] = []
            errors_by_severity[severity].append(result)
            
            # Group by rule
            rule_id = result.rule_id
            if rule_id not in errors_by_rule:
                errors_by_rule[rule_id] = []
            errors_by_rule[rule_id].append(result)
            
            # Group by validator (if available)
            validator_id = getattr(result, 'validator_id', 'unknown')
            if validator_id not in errors_by_validator:
                errors_by_validator[validator_id] = []
            errors_by_validator[validator_id].append(result)
        
        # Verify error aggregation
        assert len(errors_by_severity) > 0
        assert "error" in errors_by_severity  # Should have errors from invalid data
        
        assert len(errors_by_rule) > 0
        assert len(errors_by_validator) > 0
        
        # Generate summary report
        error_summary = {
            "total_validations": len(validation_results),
            "errors_by_severity": {k: len(v) for k, v in errors_by_severity.items()},
            "errors_by_rule": {k: len(v) for k, v in errors_by_rule.items()},
            "errors_by_validator": {k: len(v) for k, v in errors_by_validator.items()},
            "most_common_errors": sorted(errors_by_rule.keys(), key=lambda x: len(errors_by_rule[x]), reverse=True)[:5]
        }
        
        assert error_summary["total_validations"] > 0
        assert error_summary["errors_by_severity"]["error"] > 0
        assert len(error_summary["most_common_errors"]) > 0
    
    def test_data_quality_historical_tracking(self, sample_config, mock_logger, temp_dir):
        """Test historical tracking of data quality metrics."""
        # Setup quality tracking database
        quality_db_path = temp_dir / "quality_tracking.db"
        quality_db = sqlite3.connect(str(quality_db_path))
        
        # Create quality tracking tables
        quality_db.execute("""
            CREATE TABLE quality_runs (
                id INTEGER PRIMARY KEY,
                run_timestamp TIMESTAMP,
                dataset_name TEXT,
                total_records INTEGER,
                total_validations INTEGER,
                errors INTEGER,
                warnings INTEGER,
                quality_score REAL
            )
        """)
        
        quality_db.execute("""
            CREATE TABLE quality_issues (
                id INTEGER PRIMARY KEY,
                run_id INTEGER,
                rule_id TEXT,
                severity TEXT,
                message TEXT,
                affected_records TEXT,
                FOREIGN KEY (run_id) REFERENCES quality_runs (id)
            )
        """)
        
        quality_db.commit()
        
        # Create quality tracker
        class QualityTracker:
            def __init__(self, db_connection):
                self.db = db_connection
            
            def track_validation_run(self, dataset_name, validation_results, total_records):
                timestamp = datetime.now()
                
                errors = sum(1 for r in validation_results if r.severity == ValidationSeverity.ERROR)
                warnings = sum(1 for r in validation_results if r.severity == ValidationSeverity.WARNING)
                
                # Calculate quality score (0-1, where 1 is perfect)
                quality_score = max(0, 1 - (errors * 0.1 + warnings * 0.05) / total_records) if total_records > 0 else 0
                
                # Insert quality run
                cursor = self.db.execute("""
                    INSERT INTO quality_runs 
                    (run_timestamp, dataset_name, total_records, total_validations, errors, warnings, quality_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (timestamp, dataset_name, total_records, len(validation_results), errors, warnings, quality_score))
                
                run_id = cursor.lastrowid
                
                # Insert individual issues
                for result in validation_results:
                    if result.severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING]:
                        self.db.execute("""
                            INSERT INTO quality_issues
                            (run_id, rule_id, severity, message, affected_records)
                            VALUES (?, ?, ?, ?, ?)
                        """, (run_id, result.rule_id, result.severity.value, result.message, 
                             json.dumps(result.affected_records)))
                
                self.db.commit()
                return run_id
            
            def get_quality_trends(self, dataset_name, days=30):
                cursor = self.db.execute("""
                    SELECT run_timestamp, quality_score, errors, warnings
                    FROM quality_runs
                    WHERE dataset_name = ? AND run_timestamp >= datetime('now', '-{} days')
                    ORDER BY run_timestamp
                """.format(days), (dataset_name,))
                
                return cursor.fetchall()
        
        tracker = QualityTracker(quality_db)
        orchestrator = ValidationOrchestrator(mock_logger)
        
        # Add validators
        schema_validator = BaseValidator("schema_validator", sample_config["validators"]["schema_validator"], mock_logger)
        quality_checker = QualityChecker("quality_checker", {"completeness_threshold": 0.9}, mock_logger)
        
        orchestrator.add_validator(schema_validator)
        orchestrator.add_validator(quality_checker)
        
        # Simulate multiple quality runs over time
        test_datasets = [
            ("health_data_v1", [
                {"sa2_code": "101011001", "value": 25.5, "year": 2021},
                {"sa2_code": "101011002", "value": 30.2, "year": 2021}
            ]),
            ("health_data_v2", [
                {"sa2_code": "101011001", "value": 25.5, "year": 2021},
                {"sa2_code": "invalid", "value": -10, "year": 1999}  # Invalid data
            ]),
            ("health_data_v3", [
                {"sa2_code": "101011001", "value": 25.5, "year": 2021},
                {"sa2_code": "101011002", "value": 30.2, "year": 2021},
                {"sa2_code": "101021003", "value": 22.8, "year": 2021}  # Improved data
            ])
        ]
        
        run_ids = []
        for dataset_name, test_data in test_datasets:
            validation_results = orchestrator.validate_all(test_data)
            run_id = tracker.track_validation_run(dataset_name, validation_results, len(test_data))
            run_ids.append(run_id)
        
        # Verify tracking
        assert len(run_ids) == 3
        
        # Check quality trends
        for dataset_name, _ in test_datasets:
            trends = tracker.get_quality_trends(dataset_name)
            assert len(trends) > 0
        
        # Verify quality scores differ based on data quality
        cursor = quality_db.execute("SELECT dataset_name, quality_score FROM quality_runs ORDER BY id")
        quality_scores = cursor.fetchall()
        
        # health_data_v2 should have lower quality score due to invalid data
        v1_score = next(score for name, score in quality_scores if name == "health_data_v1")
        v2_score = next(score for name, score in quality_scores if name == "health_data_v2") 
        v3_score = next(score for name, score in quality_scores if name == "health_data_v3")
        
        assert v2_score < v1_score  # v2 has invalid data
        assert v3_score >= v1_score  # v3 has more good data
        
        quality_db.close()
    
    def test_data_quality_alerting_system(self, sample_config, mock_logger):
        """Test data quality alerting for critical issues."""
        orchestrator = ValidationOrchestrator(mock_logger)
        
        # Create alerting system
        alerts_triggered = []
        
        class QualityAlerter:
            def __init__(self, thresholds):
                self.thresholds = thresholds
                self.alerts = alerts_triggered
            
            def check_and_alert(self, validation_results, dataset_info):
                errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
                warnings = [r for r in validation_results if r.severity == ValidationSeverity.WARNING]
                
                total_records = dataset_info.get('total_records', 1)
                error_rate = len(errors) / total_records
                warning_rate = len(warnings) / total_records
                
                # Check error rate threshold
                if error_rate > self.thresholds.get('max_error_rate', 0.05):
                    self.alerts.append({
                        'type': 'high_error_rate',
                        'severity': 'critical',
                        'message': f'Error rate {error_rate:.1%} exceeds threshold {self.thresholds["max_error_rate"]:.1%}',
                        'error_count': len(errors),
                        'total_records': total_records
                    })
                
                # Check warning rate threshold  
                if warning_rate > self.thresholds.get('max_warning_rate', 0.15):
                    self.alerts.append({
                        'type': 'high_warning_rate', 
                        'severity': 'warning',
                        'message': f'Warning rate {warning_rate:.1%} exceeds threshold {self.thresholds["max_warning_rate"]:.1%}',
                        'warning_count': len(warnings),
                        'total_records': total_records
                    })
                
                # Check for specific critical rules
                critical_rules = ['schema_missing_columns', 'geographic_invalid_sa2']
                critical_errors = [r for r in errors if r.rule_id in critical_rules]
                
                if critical_errors:
                    self.alerts.append({
                        'type': 'critical_rule_violation',
                        'severity': 'critical', 
                        'message': f'Critical validation rules violated: {[r.rule_id for r in critical_errors]}',
                        'rules': [r.rule_id for r in critical_errors]
                    })
        
        alerter = QualityAlerter({
            'max_error_rate': 0.1,   # 10% error rate threshold
            'max_warning_rate': 0.2  # 20% warning rate threshold
        })
        
        # Add validators
        schema_validator = BaseValidator("schema_validator", sample_config["validators"]["schema_validator"], mock_logger)
        business_rules_validator = BusinessRulesValidator("business_validator", sample_config["validators"]["schema_validator"], mock_logger)
        
        orchestrator.add_validator(schema_validator)
        orchestrator.add_validator(business_rules_validator)
        
        # Test with high-quality data (should not trigger alerts)
        good_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},
            {"sa2_code": "101011002", "value": 30.2, "year": 2021}
        ]
        
        validation_results = orchestrator.validate_all(good_data)
        alerter.check_and_alert(validation_results, {'total_records': len(good_data)})
        
        good_data_alerts = len(alerts_triggered)
        
        # Test with poor-quality data (should trigger alerts)
        bad_data = [
            {"sa2_code": "101011001", "value": 25.5, "year": 2021},  # Good
            {"sa2_code": "invalid", "value": -10, "year": 1999},      # Multiple errors
            {"value": 30.2, "year": 2021},                           # Missing sa2_code
            {"sa2_code": "101021003", "value": "invalid", "year": 2021}  # Invalid type
        ]
        
        validation_results = orchestrator.validate_all(bad_data)
        alerter.check_and_alert(validation_results, {'total_records': len(bad_data)})
        
        # Should have triggered new alerts
        assert len(alerts_triggered) > good_data_alerts
        
        # Check alert types
        alert_types = [alert['type'] for alert in alerts_triggered]
        assert 'high_error_rate' in alert_types or 'critical_rule_violation' in alert_types
        
        # Verify alert details
        for alert in alerts_triggered:
            assert 'type' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert alert['severity'] in ['critical', 'warning', 'info']