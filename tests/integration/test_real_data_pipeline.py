"""
Real data processing pipeline tests with mock Australian health and geographic data.

This module tests the complete ETL pipeline with realistic mock data that
represents Australian health indicators, geographic boundaries, and demographic data.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import pytest
import numpy as np

from src.pipelines.master_etl_pipeline import (
    MasterETLPipeline, PipelineStageDefinition, QualityAssuranceConfig, PipelineStageType, QualityLevel
)
from src.pipelines.validation_pipeline import ValidationMode
from src.extractors.aihw_extractor import AIHWMortalityExtractor
from src.transformers.geographic_standardiser import GeographicStandardiser
from src.transformers.data_integrator import MasterDataIntegrator
from src.validators import ValidationOrchestrator, QualityChecker
from src.utils.config import get_config
from src.utils.interfaces import ValidationError, TransformationError


class TestRealDataPipeline:
    """Test suite for real data processing pipeline with mock Australian data."""
    
    @pytest.fixture(scope="class")
    def mock_australian_health_data(self):
        """Generate mock Australian health data for testing."""
        # Generate realistic SA2 codes (Australian Bureau of Statistics format)
        sa2_codes = [
            f"1{state:01d}{region:03d}{area:03d}{suffix:04d}" 
            for state in range(1, 9)  # 8 states/territories
            for region in range(1, 6)  # 5 regions per state
            for area in range(1, 4)   # 3 areas per region  
            for suffix in range(1, 6) # 5 SA2s per area
        ][:500]  # Limit to 500 SA2 areas for testing
        
        # Generate health indicator data
        health_indicators = []
        for i, sa2_code in enumerate(sa2_codes):
            # Mortality data
            health_indicators.append({
                'SA2_CODE': sa2_code,
                'SA2_NAME': f'SA2 Area {i+1}',
                'STATE_CODE': sa2_code[1],
                'INDICATOR_TYPE': 'mortality_rate',
                'INDICATOR_VALUE': np.random.normal(8.5, 2.0),  # Deaths per 1000
                'CAUSE_OF_DEATH': np.random.choice(['cardiovascular', 'cancer', 'respiratory', 'accidents']),
                'AGE_GROUP': np.random.choice(['0-14', '15-44', '45-64', '65+']),
                'SEX': np.random.choice(['Male', 'Female']),
                'REFERENCE_YEAR': 2023,
                'DATA_SOURCE': 'AIHW_MORTALITY',
                'GEOGRAPHIC_CODE': sa2_code,
                'GEOGRAPHIC_TYPE': 'SA2'
            })
            
            # Health service accessibility
            health_indicators.append({
                'SA2_CODE': sa2_code,
                'SA2_NAME': f'SA2 Area {i+1}',
                'STATE_CODE': sa2_code[1],
                'INDICATOR_TYPE': 'health_service_accessibility',
                'INDICATOR_VALUE': np.random.normal(75.0, 15.0),  # Accessibility score 0-100
                'SERVICE_TYPE': np.random.choice(['hospital', 'gp', 'specialist', 'mental_health']),
                'DISTANCE_KM': np.random.exponential(10.0),
                'REFERENCE_YEAR': 2023,
                'DATA_SOURCE': 'AIHW_SERVICES',
                'GEOGRAPHIC_CODE': sa2_code,
                'GEOGRAPHIC_TYPE': 'SA2'
            })
        
        return pd.DataFrame(health_indicators)
    
    @pytest.fixture(scope="class")
    def mock_geographic_data(self):
        """Generate mock geographic boundary data."""
        # Generate coordinates within Australian bounds
        geographic_data = []
        
        for i in range(100):
            # Australian coordinate bounds
            latitude = np.random.uniform(-44.0, -10.0)
            longitude = np.random.uniform(112.0, 154.0)
            
            geographic_data.append({
                'SA2_CODE': f'1{i//20 + 1:01d}{(i//5) % 4 + 1:03d}{i%5 + 1:03d}{1000 + i:04d}',
                'SA2_NAME': f'Geographic Area {i+1}',
                'LATITUDE': latitude,
                'LONGITUDE': longitude,
                'AREA_SQKM': np.random.uniform(5.0, 500.0),
                'POPULATION': np.random.randint(1000, 50000),
                'COORDINATE_SYSTEM': 'GDA2020',
                'BOUNDARY_TYPE': 'SA2',
                'REFERENCE_DATE': '2021-07-01'
            })
        
        return pd.DataFrame(geographic_data)
    
    @pytest.fixture(scope="class") 
    def mock_census_data(self):
        """Generate mock Australian census data."""
        census_data = []
        
        for i in range(200):
            sa2_code = f'1{i//40 + 1:01d}{(i//10) % 4 + 1:03d}{i%10 + 1:03d}{2000 + i:04d}'
            
            census_data.append({
                'SA2_CODE': sa2_code,
                'TOTAL_POPULATION': np.random.randint(2000, 80000),
                'MEDIAN_AGE': np.random.normal(38.0, 8.0),
                'MEDIAN_HOUSEHOLD_INCOME': np.random.normal(65000, 20000),
                'INDIGENOUS_POPULATION_PERCENT': np.random.exponential(2.0),
                'UNIVERSITY_EDUCATION_PERCENT': np.random.normal(25.0, 10.0),
                'UNEMPLOYMENT_RATE': np.random.normal(6.5, 2.0),
                'SEIFA_ADVANTAGE_SCORE': np.random.normal(1000, 100),
                'REFERENCE_YEAR': 2021,
                'DATA_SOURCE': 'ABS_CENSUS'
            })
        
        return pd.DataFrame(census_data)
    
    @pytest.fixture
    def pipeline_config(self):
        """Create test pipeline configuration."""
        return {
            'name': 'test_real_data_pipeline',
            'quality_config': QualityAssuranceConfig(
                enabled=True,
                quality_level=QualityLevel.COMPREHENSIVE,
                validation_mode=ValidationMode.SELECTIVE,
                halt_on_critical_errors=False,  # Continue on errors for testing
                generate_quality_reports=True,
                monitor_performance=True,
                track_data_lineage=True,
                compliance_standards=['AIHW', 'ABS']
            ),
            'enable_checkpoints': True,
            'max_retries': 2
        }
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_complete_pipeline_with_mock_data(
        self, 
        mock_australian_health_data,
        mock_geographic_data,
        mock_census_data,
        pipeline_config,
        temp_output_dir
    ):
        """
        Test complete ETL pipeline with realistic mock Australian data.
        
        This test validates the entire pipeline from extraction through loading
        with data that represents real Australian health and geographic datasets.
        """
        # Create stage definitions for testing
        stage_definitions = [
            PipelineStageDefinition(
                stage_id="mock_data_extraction",
                stage_name="Mock Data Extraction", 
                stage_type=PipelineStageType.EXTRACTION,
                stage_class="tests.mocks.MockDataExtractor",
                dependencies=[],
                configuration={
                    'mock_data': mock_australian_health_data,
                    'validation_enabled': True,
                    'quality_threshold': 90.0
                },
                validation_required=True,
                quality_level=QualityLevel.STANDARD
            ),
            PipelineStageDefinition(
                stage_id="geographic_transformation",
                stage_name="Geographic Transformation",
                stage_type=PipelineStageType.TRANSFORMATION,
                stage_class="src.transformers.geographic_standardiser.GeographicStandardiser",
                dependencies=["mock_data_extraction"],
                configuration={
                    'validation_enabled': True,
                    'geographic_quality_threshold': 95.0,
                    'halt_on_validation_failure': False
                },
                validation_required=True,
                quality_level=QualityLevel.COMPREHENSIVE
            ),
            PipelineStageDefinition(
                stage_id="data_integration",
                stage_name="Data Integration",
                stage_type=PipelineStageType.INTEGRATION,
                stage_class="src.transformers.data_integrator.DataIntegrator",
                dependencies=["geographic_transformation"],
                configuration={
                    'integration_strategy': 'sa2_based',
                    'validation_enabled': True
                },
                validation_required=True,
                quality_level=QualityLevel.COMPREHENSIVE
            )
        ]
        
        # Create and run pipeline
        pipeline = MasterETLPipeline(
            name=pipeline_config['name'],
            stage_definitions=stage_definitions,
            quality_config=pipeline_config['quality_config'],
            enable_checkpoints=pipeline_config['enable_checkpoints'],
            max_retries=pipeline_config['max_retries']
        )
        
        # Inject mock data into pipeline context
        pipeline.current_data = mock_australian_health_data
        
        # Run complete ETL process
        results = pipeline.run_complete_etl(
            source_config={'mock_data': True},
            target_config={'output_dir': str(temp_output_dir)}
        )
        
        # Assert pipeline execution succeeded
        assert results['execution_status'] == 'completed'
        assert results['final_data_records'] > 0
        
        # Validate quality assessment
        quality_assessment = results['quality_assessment']
        assert quality_assessment['overall_quality_score'] >= 70.0  # Reasonable threshold for mock data
        assert quality_assessment['quality_grade'] in ['A', 'B', 'C']  # Acceptable grades
        
        # Validate stage results
        stage_results = results['stage_results']
        assert len(stage_results) >= 3  # At least 3 stages executed
        
        # Check that validation was performed
        validation_results = results['validation_results']
        assert len(validation_results) > 0
        
        # Verify data lineage tracking
        assert 'data_flow_status' in results
        assert results['execution_summary']['success_rate'] >= 80.0
    
    def test_pipeline_with_data_quality_issues(
        self,
        mock_australian_health_data,
        pipeline_config,
        temp_output_dir
    ):
        """
        Test pipeline behaviour with intentionally corrupted data.
        
        This test validates the pipeline's handling of data quality issues
        including missing values, invalid codes, and inconsistent data.
        """
        # Introduce data quality issues
        corrupted_data = mock_australian_health_data.copy()
        
        # Remove some SA2 codes (missing data)
        corrupted_data.loc[0:10, 'SA2_CODE'] = None
        
        # Invalid SA2 codes (wrong format)
        corrupted_data.loc[11:20, 'SA2_CODE'] = 'INVALID_CODE'
        
        # Extreme outliers in health indicators
        corrupted_data.loc[21:30, 'INDICATOR_VALUE'] = 999999.0
        
        # Inconsistent geographic types
        corrupted_data.loc[31:40, 'GEOGRAPHIC_TYPE'] = 'UNKNOWN'
        
        # Create pipeline with stricter validation
        strict_config = pipeline_config.copy()
        strict_config['quality_config'].validation_mode = ValidationMode.STRICT
        strict_config['quality_config'].halt_on_critical_errors = False  # Continue for testing
        
        stage_definitions = [
            PipelineStageDefinition(
                stage_id="corrupted_data_extraction",
                stage_name="Corrupted Data Extraction",
                stage_type=PipelineStageType.EXTRACTION,
                stage_class="tests.mocks.MockDataExtractor",
                dependencies=[],
                configuration={
                    'mock_data': corrupted_data,
                    'validation_enabled': True,
                    'quality_threshold': 90.0,
                    'halt_on_validation_failure': False
                },
                validation_required=True,
                quality_level=QualityLevel.COMPREHENSIVE
            )
        ]
        
        pipeline = MasterETLPipeline(
            name='test_corrupted_data_pipeline',
            stage_definitions=stage_definitions,
            quality_config=strict_config['quality_config']
        )
        
        # Inject corrupted data
        pipeline.current_data = corrupted_data
        
        # Run pipeline (should complete but with quality issues)
        results = pipeline.run_complete_etl()
        
        # Pipeline should complete but with reduced quality scores
        assert results['execution_status'] in ['completed', 'failed']
        
        # Quality score should be lower due to data issues
        quality_assessment = results['quality_assessment']
        
        # Validate that quality issues were detected
        if results['execution_status'] == 'completed':
            assert quality_assessment['overall_quality_score'] < 90.0
            assert len(quality_assessment['recommendations']) > 0
        
        # Check validation results captured the issues
        validation_results = results['validation_results']
        validation_failed = any(
            result['quality_score'] < 90.0 
            for result in validation_results.values()
        )
        assert validation_failed or results['execution_status'] == 'failed'
    
    def test_geographic_validation_integration(
        self,
        mock_geographic_data,
        pipeline_config,
        temp_output_dir
    ):
        """
        Test geographic validation integration with coordinate and boundary data.
        
        This test specifically validates the geographic transformation and
        validation components with Australian coordinate data.
        """
        # Create geographic standardiser
        geographic_standardiser = GeographicStandardiser(
            config={
                'validation_enabled': True,
                'geographic_quality_threshold': 95.0,
                'coordinate_system': 'GDA2020',
                'validate_australian_bounds': True
            }
        )
        
        # Run geographic validation
        validation_result = geographic_standardiser.validate_geographic_data(
            mock_geographic_data,
            validation_context={'stage': 'geographic_test'}
        )
        
        # Assert validation completed
        assert validation_result is not None
        assert validation_result.validation_metrics.total_records == len(mock_geographic_data)
        
        # Geographic data should pass validation (within Australian bounds)
        assert validation_result.validation_metrics.quality_score >= 85.0
        
        # Check for geographic-specific validation
        assert validation_result.stage_name == "geographic_standardisation"
        
        # Validate that Australian coordinate bounds were checked
        errors = validation_result.errors
        coordinate_errors = [e for e in errors if 'coordinate' in e.lower() or 'bound' in e.lower()]
        
        # Should have minimal coordinate errors with mock Australian data
        assert len(coordinate_errors) <= len(mock_geographic_data) * 0.1  # Max 10% errors
    
    def test_performance_with_large_dataset(
        self,
        pipeline_config,
        temp_output_dir
    ):
        """
        Test pipeline performance with larger mock dataset.
        
        This test validates performance characteristics and scalability
        of the validation-integrated pipeline.
        """
        # Generate larger mock dataset
        large_dataset_size = 10000
        large_mock_data = []
        
        for i in range(large_dataset_size):
            large_mock_data.append({
                'SA2_CODE': f'1{i//2000 + 1:01d}{(i//500) % 4 + 1:03d}{(i//100) % 5 + 1:03d}{i % 100 + 1000:04d}',
                'INDICATOR_TYPE': 'health_metric',
                'INDICATOR_VALUE': np.random.normal(50.0, 15.0),
                'REFERENCE_YEAR': 2023,
                'DATA_SOURCE': 'PERFORMANCE_TEST',
                'GEOGRAPHIC_CODE': f'CODE_{i}',
                'GEOGRAPHIC_TYPE': 'SA2'
            })
        
        large_df = pd.DataFrame(large_mock_data)
        
        # Create performance-optimised pipeline
        perf_config = pipeline_config.copy()
        perf_config['quality_config'].quality_level = QualityLevel.STANDARD  # Reduce validation overhead
        
        stage_definitions = [
            PipelineStageDefinition(
                stage_id="large_data_processing",
                stage_name="Large Data Processing",
                stage_type=PipelineStageType.EXTRACTION,
                stage_class="tests.mocks.MockDataExtractor",
                dependencies=[],
                configuration={
                    'mock_data': large_df,
                    'validation_enabled': True,
                    'batch_size': 1000,  # Process in batches
                    'quality_threshold': 85.0
                },
                validation_required=True,
                quality_level=QualityLevel.STANDARD
            )
        ]
        
        pipeline = MasterETLPipeline(
            name='test_performance_pipeline',
            stage_definitions=stage_definitions,
            quality_config=perf_config['quality_config']
        )
        
        # Measure execution time
        start_time = datetime.now()
        
        pipeline.current_data = large_df
        results = pipeline.run_complete_etl()
        
        end_time = datetime.now()
        execution_duration = (end_time - start_time).total_seconds()
        
        # Assert reasonable performance (should process 10K records in reasonable time)
        assert execution_duration < 300  # Less than 5 minutes
        assert results['final_data_records'] == large_dataset_size
        
        # Validate performance metrics were captured
        execution_summary = results['execution_summary']
        assert 'total_duration_seconds' in execution_summary
        assert execution_summary['total_duration_seconds'] == execution_duration
        
        # Quality should still be maintained despite large dataset
        quality_assessment = results['quality_assessment']
        assert quality_assessment['overall_quality_score'] >= 80.0
    
    def test_compliance_standards_validation(
        self,
        mock_australian_health_data,
        mock_census_data,
        pipeline_config
    ):
        """
        Test compliance with Australian health data standards.
        
        This test validates that the pipeline enforces compliance with
        AIHW, ABS, and other Australian health data standards.
        """
        # Merge health and census data for comprehensive testing
        combined_data = pd.merge(
            mock_australian_health_data,
            mock_census_data,
            on='SA2_CODE',
            how='inner'
        )
        
        # Create compliance-focused pipeline
        compliance_config = pipeline_config.copy()
        compliance_config['quality_config'].compliance_standards = ['AIHW', 'ABS', 'Medicare', 'PBS']
        compliance_config['quality_config'].validation_mode = ValidationMode.AUDIT
        
        stage_definitions = [
            PipelineStageDefinition(
                stage_id="compliance_validation",
                stage_name="Compliance Validation",
                stage_type=PipelineStageType.VALIDATION,
                stage_class="tests.mocks.MockComplianceValidator",
                dependencies=[],
                configuration={
                    'mock_data': combined_data,
                    'compliance_standards': compliance_config['quality_config'].compliance_standards,
                    'audit_mode': True
                },
                validation_required=True,
                quality_level=QualityLevel.AUDIT
            )
        ]
        
        pipeline = MasterETLPipeline(
            name='test_compliance_pipeline',
            stage_definitions=stage_definitions,
            quality_config=compliance_config['quality_config']
        )
        
        pipeline.current_data = combined_data
        results = pipeline.run_complete_etl()
        
        # Validate compliance assessment
        quality_assessment = results['quality_assessment']
        assert 'compliance_status' in quality_assessment
        
        compliance_status = quality_assessment['compliance_status']
        
        # Check each standard
        for standard in compliance_config['quality_config'].compliance_standards:
            assert standard in compliance_status
            # Should be compliant or have minor issues with mock data
            assert compliance_status[standard] in ['COMPLIANT', 'MINOR_ISSUES']
        
        # Overall compliance should be acceptable
        compliance_scores = [
            status for status in compliance_status.values() 
            if status == 'COMPLIANT'
        ]
        assert len(compliance_scores) >= len(compliance_status) * 0.75  # At least 75% compliant


@pytest.fixture(scope="session")
def mock_data_extractor():
    """Mock data extractor for testing."""
    class MockDataExtractor:
        def __init__(self, mock_data, **config):
            self.mock_data = mock_data
            self.config = config
        
        def extract(self):
            return self.mock_data
        
        def execute(self, input_data, context):
            return self.mock_data
    
    return MockDataExtractor


@pytest.fixture(scope="session") 
def mock_compliance_validator():
    """Mock compliance validator for testing."""
    class MockComplianceValidator:
        def __init__(self, mock_data, compliance_standards, **config):
            self.mock_data = mock_data
            self.compliance_standards = compliance_standards
            self.config = config
        
        def execute(self, input_data, context):
            # Simulate compliance validation
            compliance_results = {}
            for standard in self.compliance_standards:
                # Mock compliance check - assume mostly compliant
                compliance_results[standard] = 'COMPLIANT'
            
            return {
                'compliance_status': compliance_results,
                'validation_passed': True,
                'data': self.mock_data
            }
    
    return MockComplianceValidator