"""
Complete end-to-end ETL pipeline test.

This module provides comprehensive end-to-end testing of the complete
ETL pipeline with validation integration, covering the entire data
processing lifecycle from extraction to loading.
"""

import json
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import pytest
import numpy as np

from src.pipelines.master_etl_pipeline import (
    MasterETLPipeline, PipelineStageDefinition, QualityAssuranceConfig,
    PipelineStageType, DataFlowCheckpoint
)
from src.pipelines.validation_pipeline import ValidationMode, QualityLevel
from src.extractors.base import BaseExtractor
from src.transformers.base import BaseTransformer
from src.loaders.base import BaseLoader
from src.validators import ValidationOrchestrator
from src.utils.interfaces import ValidationError, TransformationError, ExtractionError
from src.utils.config import get_config
from src.utils.logging import get_logger


class MockAustralianDataExtractor(BaseExtractor):
    """Mock extractor that generates realistic Australian health data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("mock_australian_extractor", config)
        self.data_size = config.get('data_size', 5000)
        self.introduce_errors = config.get('introduce_errors', True)
        self.error_rate = config.get('error_rate', 0.05)  # 5% error rate
    
    def extract(self, source, **kwargs):
        """Generate mock Australian health and geographic data."""
        logger = get_logger(__name__)
        logger.info(f"Generating {self.data_size} mock Australian health records")
        
        # Australian states and territories
        states = ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT']
        state_codes = ['1', '2', '3', '4', '5', '6', '7', '8']
        
        data_records = []
        
        for i in range(self.data_size):
            state_idx = i % len(states)
            state_code = state_codes[state_idx]
            
            # Generate realistic SA2 code (11 digits)
            sa2_code = f"{state_code}{i//1000:02d}{(i//100)%10:01d}{(i//10)%10:01d}{i%10:01d}{1000 + (i%1000):04d}"
            
            # Introduce data quality issues based on configuration
            if self.introduce_errors and np.random.random() < self.error_rate:
                if np.random.random() < 0.3:
                    sa2_code = None  # Missing SA2 code
                elif np.random.random() < 0.3:
                    sa2_code = f"INVALID_{i}"  # Invalid format
                elif np.random.random() < 0.3:
                    sa2_code = "123"  # Too short
            
            # Generate health indicators with realistic Australian patterns
            mortality_rate = np.random.lognormal(2.0, 0.5)  # Realistic mortality distribution
            if self.introduce_errors and np.random.random() < self.error_rate:
                if np.random.random() < 0.5:
                    mortality_rate = np.random.uniform(100, 1000)  # Unrealistic outlier
                else:
                    mortality_rate = -1  # Invalid negative value
            
            # Health service accessibility (0-100 scale)
            accessibility = np.random.beta(2, 2) * 100
            if self.introduce_errors and np.random.random() < self.error_rate:
                accessibility = np.random.uniform(101, 200)  # Out of range
            
            # Geographic coordinates within Australian bounds
            if state_idx in [0, 1, 2]:  # Eastern states
                latitude = np.random.uniform(-37.0, -28.0)
                longitude = np.random.uniform(140.0, 154.0)
            elif state_idx == 4:  # Western Australia
                latitude = np.random.uniform(-35.0, -15.0)
                longitude = np.random.uniform(112.0, 130.0)
            else:  # Other states/territories
                latitude = np.random.uniform(-43.0, -12.0)
                longitude = np.random.uniform(115.0, 150.0)
            
            # Introduce coordinate errors
            if self.introduce_errors and np.random.random() < self.error_rate:
                if np.random.random() < 0.5:
                    latitude = np.random.uniform(-90, 90)  # Global coordinates
                    longitude = np.random.uniform(-180, 180)
                else:
                    latitude = None  # Missing coordinates
                    longitude = None
            
            record = {
                'SA2_CODE': sa2_code,
                'SA2_NAME': f'{states[state_idx]} Health Region {i+1}',
                'STATE_CODE': state_code,
                'STATE_NAME': states[state_idx],
                'MORTALITY_RATE_PER_1000': mortality_rate,
                'CARDIOVASCULAR_MORTALITY': np.random.poisson(3),
                'CANCER_MORTALITY': np.random.poisson(2),
                'RESPIRATORY_MORTALITY': np.random.poisson(1),
                'HEALTH_SERVICE_ACCESSIBILITY': accessibility,
                'HOSPITAL_BEDS_PER_1000': np.random.gamma(2, 2),
                'GP_ACCESSIBILITY_SCORE': np.random.uniform(0, 100),
                'MENTAL_HEALTH_SERVICES': np.random.poisson(5),
                'POPULATION_TOTAL': np.random.randint(1000, 50000),
                'POPULATION_INDIGENOUS_PCT': np.random.exponential(2),
                'MEDIAN_HOUSEHOLD_INCOME': np.random.normal(65000, 20000),
                'UNEMPLOYMENT_RATE_PCT': np.random.gamma(2, 3),
                'EDUCATION_UNIVERSITY_PCT': np.random.normal(25, 8),
                'SEIFA_ADVANTAGE_SCORE': np.random.normal(1000, 100),
                'LATITUDE': latitude,
                'LONGITUDE': longitude,
                'AREA_SQKM': np.random.exponential(50),
                'POPULATION_DENSITY': np.random.exponential(100),
                'REFERENCE_YEAR': 2023,
                'DATA_COLLECTION_DATE': (datetime.now() - timedelta(days=np.random.randint(0, 365))).isoformat(),
                'DATA_SOURCE': 'MOCK_AIHW',
                'QUALITY_INDICATOR': np.random.choice(['HIGH', 'MEDIUM', 'LOW'], p=[0.7, 0.25, 0.05]),
                'COMPLETENESS_SCORE': np.random.uniform(85, 100),
                'VALIDATION_FLAGS': [] if np.random.random() > 0.1 else ['REVIEW_REQUIRED'],
                'GEOGRAPHIC_TYPE': 'SA2',
                'COORDINATE_SYSTEM': 'GDA2020'
            }
            
            data_records.append(record)
        
        # Yield data in batches
        batch_size = 1000
        for i in range(0, len(data_records), batch_size):
            batch = data_records[i:i + batch_size]
            yield batch
    
    def get_source_metadata(self, source):
        """Return metadata about the mock data source."""
        from src.utils.interfaces import SourceMetadata
        return SourceMetadata(
            source_id="mock_australian_data",
            source_type="generated",
            row_count=self.data_size,
            column_count=25,
            file_size=self.data_size * 1024,  # Approximate size
            last_modified=datetime.now(),
            checksum="mock_checksum_001"
        )
    
    def validate_source(self, source):
        """Validate mock data source (always returns True)."""
        return True


class MockGeographicTransformer(BaseTransformer):
    """Mock transformer for geographic standardisation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("mock_geographic_transformer", config)
        self.validation_enabled = config.get('validation_enabled', True)
        self.introduce_transformation_errors = config.get('introduce_transformation_errors', False)
    
    def transform(self, data, **kwargs):
        """Apply mock geographic transformations."""
        logger = get_logger(__name__)
        logger.info(f"Transforming {len(data)} records for geographic standardisation")
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Standardise SA2 codes
        df['SA2_CODE_STANDARDISED'] = df['SA2_CODE'].apply(self._standardise_sa2_code)
        
        # Add allocation factors (for population weighting)
        df['ALLOCATION_FACTOR'] = 1.0
        
        # Add transformation metadata
        df['TRANSFORMATION_DATE'] = datetime.now().isoformat()
        df['TRANSFORMATION_METHOD'] = 'mock_standardisation'
        df['VALIDATION_STATUS'] = 'PROCESSED'
        
        # Validate coordinates and update if needed
        df['COORDINATE_VALIDATION'] = df.apply(self._validate_coordinates, axis=1)
        
        # Add derived geographic indicators
        df['REMOTENESS_CATEGORY'] = df.apply(self._calculate_remoteness, axis=1)
        df['URBAN_RURAL_INDICATOR'] = df.apply(self._classify_urban_rural, axis=1)
        
        # Introduce transformation errors if configured
        if self.introduce_transformation_errors:
            error_indices = np.random.choice(
                len(df), 
                size=int(len(df) * 0.02),  # 2% error rate
                replace=False
            )
            df.loc[error_indices, 'SA2_CODE_STANDARDISED'] = 'TRANSFORM_ERROR'
        
        # Run validation if enabled
        if self.validation_enabled:
            validation_results = self._validate_transformed_data(df)
            df['TRANSFORMATION_QUALITY_SCORE'] = validation_results['quality_score']
        
        logger.info(f"Geographic transformation completed for {len(df)} records")
        
        return df.to_dict('records') if isinstance(data, list) else df
    
    def _standardise_sa2_code(self, sa2_code):
        """Standardise SA2 code format."""
        if pd.isna(sa2_code) or sa2_code is None:
            return None
        
        sa2_str = str(sa2_code)
        
        # Remove any non-numeric characters
        numeric_only = ''.join(filter(str.isdigit, sa2_str))
        
        # Ensure 11-digit format
        if len(numeric_only) == 11:
            return numeric_only
        elif len(numeric_only) < 11:
            # Pad with zeros
            return numeric_only.zfill(11)
        else:
            # Truncate or return error
            return numeric_only[:11] if len(numeric_only) <= 15 else 'FORMAT_ERROR'
    
    def _validate_coordinates(self, row):
        """Validate geographic coordinates."""
        lat = row.get('LATITUDE')
        lon = row.get('LONGITUDE')
        
        if pd.isna(lat) or pd.isna(lon):
            return 'MISSING_COORDINATES'
        
        # Australian bounds check
        if -44.0 <= lat <= -10.0 and 112.0 <= lon <= 154.0:
            return 'VALID_AUSTRALIAN'
        else:
            return 'OUT_OF_BOUNDS'
    
    def _calculate_remoteness(self, row):
        """Calculate remoteness category based on population density."""
        density = row.get('POPULATION_DENSITY', 0)
        
        if density > 1000:
            return 'MAJOR_CITY'
        elif density > 100:
            return 'INNER_REGIONAL'
        elif density > 10:
            return 'OUTER_REGIONAL'
        elif density > 1:
            return 'REMOTE'
        else:
            return 'VERY_REMOTE'
    
    def _classify_urban_rural(self, row):
        """Classify area as urban or rural."""
        population = row.get('POPULATION_TOTAL', 0)
        density = row.get('POPULATION_DENSITY', 0)
        
        if population > 10000 and density > 100:
            return 'URBAN'
        elif population > 1000:
            return 'RURAL_TOWN'
        else:
            return 'RURAL_REMOTE'
    
    def _validate_transformed_data(self, df):
        """Validate transformed data quality."""
        total_records = len(df)
        
        # Check SA2 code standardisation success
        valid_sa2_codes = df['SA2_CODE_STANDARDISED'].notna().sum()
        standardisation_rate = (valid_sa2_codes / total_records) * 100
        
        # Check coordinate validation
        valid_coordinates = (df['COORDINATE_VALIDATION'] == 'VALID_AUSTRALIAN').sum()
        coordinate_validation_rate = (valid_coordinates / total_records) * 100
        
        # Overall quality score
        quality_score = (standardisation_rate + coordinate_validation_rate) / 2
        
        return {
            'quality_score': quality_score,
            'standardisation_rate': standardisation_rate,
            'coordinate_validation_rate': coordinate_validation_rate,
            'total_records': total_records
        }


class MockDataIntegrator(BaseTransformer):
    """Mock data integrator for combining health and geographic data."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("mock_data_integrator", config)
        self.integration_strategy = config.get('integration_strategy', 'sa2_based')
        self.calculate_derived_indicators = config.get('calculate_derived_indicators', True)
    
    def transform(self, data, **kwargs):
        """Integrate and derive health indicators."""
        logger = get_logger(__name__)
        logger.info(f"Integrating {len(data)} records with derived indicators")
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Calculate derived health indicators
        if self.calculate_derived_indicators:
            df = self._calculate_health_indicators(df)
        
        # Add integration metadata
        df['INTEGRATION_DATE'] = datetime.now().isoformat()
        df['INTEGRATION_METHOD'] = self.integration_strategy
        df['DERIVED_INDICATORS_CALCULATED'] = self.calculate_derived_indicators
        
        logger.info(f"Data integration completed for {len(df)} records")
        
        return df.to_dict('records') if isinstance(data, list) else df
    
    def _calculate_health_indicators(self, df):
        """Calculate derived health indicators."""
        # Health-to-population ratios
        df['HOSPITAL_BEDS_PER_CAPITA'] = (
            df['HOSPITAL_BEDS_PER_1000'] / 1000 * df['POPULATION_TOTAL']
        ).fillna(0)
        
        # Composite health access score
        df['COMPOSITE_HEALTH_ACCESS'] = (
            df['HEALTH_SERVICE_ACCESSIBILITY'] * 0.4 +
            df['GP_ACCESSIBILITY_SCORE'] * 0.3 +
            df['MENTAL_HEALTH_SERVICES'] * 0.3
        ).fillna(0)
        
        # Socioeconomic health correlation
        df['SOCIOECONOMIC_HEALTH_INDEX'] = (
            df['SEIFA_ADVANTAGE_SCORE'] / 1000 * 
            (100 - df['UNEMPLOYMENT_RATE_PCT']) / 100 *
            df['EDUCATION_UNIVERSITY_PCT'] / 100
        ).fillna(0)
        
        # Health equity indicator
        df['HEALTH_EQUITY_SCORE'] = (
            df['COMPOSITE_HEALTH_ACCESS'] / 
            np.maximum(df['MORTALITY_RATE_PER_1000'], 1)
        ).fillna(0)
        
        # Risk stratification
        df['HEALTH_RISK_CATEGORY'] = pd.cut(
            df['MORTALITY_RATE_PER_1000'],
            bins=[0, 5, 10, 20, float('inf')],
            labels=['LOW', 'MODERATE', 'HIGH', 'VERY_HIGH']
        ).astype(str)
        
        return df


class MockDataLoader(BaseLoader):
    """Mock data loader for testing output generation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("mock_data_loader", config)
        self.output_formats = config.get('output_formats', ['csv', 'json'])
        self.output_directory = Path(config.get('output_directory', '/tmp'))
        self.validate_on_load = config.get('validate_on_load', True)
    
    def load(self, data, **kwargs):
        """Load data to specified outputs."""
        logger = get_logger(__name__)
        logger.info(f"Loading {len(data)} records to {len(self.output_formats)} formats")
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        output_files = []
        
        # Ensure output directory exists
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        for output_format in self.output_formats:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"australian_health_data_{timestamp}.{output_format}"
            filepath = self.output_directory / filename
            
            try:
                if output_format == 'csv':
                    df.to_csv(filepath, index=False)
                elif output_format == 'json':
                    df.to_json(filepath, orient='records', indent=2)
                elif output_format == 'parquet':
                    df.to_parquet(filepath)
                
                output_files.append(str(filepath))
                logger.info(f"Data loaded to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to load data to {filepath}: {str(e)}")
                raise
        
        # Validate loaded files if enabled
        if self.validate_on_load:
            self._validate_loaded_files(output_files, len(df))
        
        return {
            'output_files': output_files,
            'records_loaded': len(df),
            'formats': self.output_formats,
            'load_timestamp': datetime.now().isoformat()
        }
    
    def _validate_loaded_files(self, output_files, expected_records):
        """Validate that files were loaded correctly."""
        for filepath in output_files:
            file_path = Path(filepath)
            
            if not file_path.exists():
                raise Exception(f"Output file not created: {filepath}")
            
            if file_path.stat().st_size == 0:
                raise Exception(f"Output file is empty: {filepath}")
            
            # Basic record count validation for CSV files
            if filepath.endswith('.csv'):
                try:
                    loaded_df = pd.read_csv(filepath)
                    if len(loaded_df) != expected_records:
                        raise Exception(
                            f"Record count mismatch in {filepath}: "
                            f"expected {expected_records}, got {len(loaded_df)}"
                        )
                except Exception as e:
                    raise Exception(f"Failed to validate {filepath}: {str(e)}")


class TestCompleteETLPipeline:
    """Comprehensive end-to-end ETL pipeline test suite."""
    
    @pytest.fixture(scope="class")
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create subdirectories
            (workspace / "data_raw").mkdir()
            (workspace / "data_processed").mkdir()
            (workspace / "logs").mkdir()
            (workspace / "checkpoints").mkdir()
            
            yield workspace
    
    @pytest.fixture
    def etl_config(self, temp_workspace):
        """Create comprehensive ETL configuration."""
        return {
            'name': 'complete_etl_test_pipeline',
            'workspace': temp_workspace,
            'data_size': 2000,  # Reasonable size for testing
            'quality_config': QualityAssuranceConfig(
                enabled=True,
                quality_level=QualityLevel.COMPREHENSIVE,
                validation_mode=ValidationMode.SELECTIVE,
                halt_on_critical_errors=False,  # Continue on errors for testing
                generate_quality_reports=True,
                monitor_performance=True,
                track_data_lineage=True,
                compliance_standards=['AIHW', 'ABS', 'Medicare']
            ),
            'enable_checkpoints': True,
            'checkpoint_interval': 500,
            'max_retries': 2,
            'parallel_stages': False,  # Sequential for deterministic testing
            'timeout_seconds': 300  # 5 minutes
        }
    
    def test_complete_etl_pipeline_success(self, etl_config, temp_workspace):
        """
        Test successful execution of complete ETL pipeline.
        
        This test validates the entire ETL process from extraction
        through loading with realistic Australian health data.
        """
        # Define complete pipeline stages
        stage_definitions = [
            PipelineStageDefinition(
                stage_id="australian_data_extraction",
                stage_name="Australian Health Data Extraction",
                stage_type=PipelineStageType.EXTRACTION,
                stage_class="tests.end_to_end.test_complete_etl.MockAustralianDataExtractor",
                dependencies=[],
                configuration={
                    'data_size': etl_config['data_size'],
                    'introduce_errors': True,
                    'error_rate': 0.03,  # 3% error rate
                    'validation_enabled': True,
                    'quality_threshold': 90.0
                },
                validation_required=True,
                quality_level=QualityLevel.STANDARD,
                timeout_seconds=120,
                retry_attempts=2
            ),
            PipelineStageDefinition(
                stage_id="geographic_standardisation",
                stage_name="Geographic Standardisation",
                stage_type=PipelineStageType.TRANSFORMATION,
                stage_class="tests.end_to_end.test_complete_etl.MockGeographicTransformer",
                dependencies=["australian_data_extraction"],
                configuration={
                    'validation_enabled': True,
                    'introduce_transformation_errors': False,
                    'standardisation_strategy': 'sa2_based',
                    'coordinate_validation': True
                },
                validation_required=True,
                quality_level=QualityLevel.COMPREHENSIVE,
                timeout_seconds=180,
                retry_attempts=2
            ),
            PipelineStageDefinition(
                stage_id="health_data_integration",
                stage_name="Health Data Integration",
                stage_type=PipelineStageType.INTEGRATION,
                stage_class="tests.end_to_end.test_complete_etl.MockDataIntegrator",
                dependencies=["geographic_standardisation"],
                configuration={
                    'integration_strategy': 'sa2_based',
                    'calculate_derived_indicators': True,
                    'validate_integration': True
                },
                validation_required=True,
                quality_level=QualityLevel.COMPREHENSIVE,
                timeout_seconds=240,
                retry_attempts=1
            ),
            PipelineStageDefinition(
                stage_id="data_export_loading",
                stage_name="Data Export and Loading",
                stage_type=PipelineStageType.LOADING,
                stage_class="tests.end_to_end.test_complete_etl.MockDataLoader",
                dependencies=["health_data_integration"],
                configuration={
                    'output_formats': ['csv', 'json'],
                    'output_directory': str(temp_workspace / "data_processed"),
                    'validate_on_load': True,
                    'compression': False  # Disable for testing
                },
                validation_required=True,
                quality_level=QualityLevel.STANDARD,
                timeout_seconds=120,
                retry_attempts=3
            )
        ]
        
        # Create data flow checkpoints
        checkpoints = [
            DataFlowCheckpoint(
                checkpoint_id="extraction_to_transformation",
                source_stage="australian_data_extraction",
                target_stage="geographic_standardisation",
                validation_rules=["non_empty", "min_records:100"],
                transformation_rules=["remove_duplicates"],
                quality_threshold=85.0
            ),
            DataFlowCheckpoint(
                checkpoint_id="transformation_to_integration",
                source_stage="geographic_standardisation",
                target_stage="health_data_integration",
                validation_rules=["non_empty", "valid_sa2_codes"],
                transformation_rules=["validate_coordinates"],
                quality_threshold=90.0
            ),
            DataFlowCheckpoint(
                checkpoint_id="integration_to_loading",
                source_stage="health_data_integration", 
                target_stage="data_export_loading",
                validation_rules=["non_empty", "completeness_check"],
                transformation_rules=[],
                quality_threshold=95.0
            )
        ]
        
        # Create master ETL pipeline
        pipeline = MasterETLPipeline(
            name=etl_config['name'],
            stage_definitions=stage_definitions,
            quality_config=etl_config['quality_config'],
            enable_checkpoints=etl_config['enable_checkpoints'],
            checkpoint_interval=etl_config['checkpoint_interval'],
            max_retries=etl_config['max_retries'],
            parallel_stages=etl_config['parallel_stages']
        )
        
        # Override data flow controller with test checkpoints
        from src.pipelines.master_etl_pipeline import DataFlowController
        pipeline.data_flow_controller = DataFlowController(checkpoints)
        
        # Record start time for performance measurement
        start_time = datetime.now()
        
        # Run complete ETL pipeline
        results = pipeline.run_complete_etl(
            source_config={
                'mock_mode': True,
                'workspace': str(temp_workspace)
            },
            target_config={
                'output_directory': str(temp_workspace / "data_processed"),
                'generate_metadata': True
            }
        )
        
        # Record completion time
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # ========================================
        # PIPELINE EXECUTION VALIDATION
        # ========================================
        
        # Pipeline should complete successfully
        assert results['execution_status'] == 'completed'
        assert results['pipeline_id'] == etl_config['name']
        
        # All stages should complete successfully
        stage_results = results['stage_results']
        assert len(stage_results) == len(stage_definitions)
        
        successful_stages = [
            stage for stage, result in stage_results.items()
            if result['state'] == 'completed'
        ]
        assert len(successful_stages) == len(stage_definitions)
        
        # ========================================
        # DATA VALIDATION
        # ========================================
        
        # Should have processed the expected number of records
        final_records = results['final_data_records']
        assert final_records > 0
        assert final_records <= etl_config['data_size']  # May be less due to filtering
        
        # Data flow should be tracked
        data_flow_status = results['data_flow_status']
        assert 'active_buffers' in data_flow_status
        assert 'flow_metrics' in data_flow_status
        
        # ========================================
        # QUALITY VALIDATION
        # ========================================
        
        # Quality assessment should be performed
        quality_assessment = results['quality_assessment']
        assert quality_assessment['overall_quality_score'] >= 0
        assert quality_assessment['overall_quality_score'] <= 100
        
        # Quality grade should be reasonable
        assert quality_assessment['quality_grade'] in ['A', 'B', 'C', 'D', 'F']
        
        # Should have quality recommendations
        assert 'recommendations' in quality_assessment
        
        # Stage-specific quality scores
        stage_quality_scores = quality_assessment['stage_quality_scores']
        assert len(stage_quality_scores) > 0
        
        for stage_name, quality_score in stage_quality_scores.items():
            assert 0 <= quality_score <= 100
        
        # ========================================
        # VALIDATION INTEGRATION
        # ========================================
        
        # Validation should have run at each stage
        validation_results = results['validation_results']
        assert len(validation_results) > 0
        
        # Each validation result should have quality score
        for stage_name, validation_result in validation_results.items():
            assert 'quality_score' in validation_result
            assert 0 <= validation_result['quality_score'] <= 100
        
        # ========================================
        # PERFORMANCE VALIDATION
        # ========================================
        
        # Execution should complete within reasonable time
        assert total_duration < etl_config['timeout_seconds']
        
        execution_summary = results['execution_summary']
        assert execution_summary['total_duration_seconds'] > 0
        assert execution_summary['success_rate'] >= 80.0  # At least 80% success
        
        # Performance should be reasonable for data size
        records_per_second = final_records / total_duration
        assert records_per_second >= 1  # At least 1 record per second
        
        # ========================================
        # OUTPUT VALIDATION
        # ========================================
        
        # Output files should be created
        output_dir = temp_workspace / "data_processed"
        output_files = list(output_dir.glob("australian_health_data_*.csv"))
        assert len(output_files) >= 1
        
        # Files should contain data
        for output_file in output_files:
            assert output_file.stat().st_size > 0
            
            # Validate CSV content
            try:
                output_df = pd.read_csv(output_file)
                assert len(output_df) > 0
                assert len(output_df.columns) > 10  # Should have many columns
                
                # Check for key columns
                expected_columns = ['SA2_CODE', 'MORTALITY_RATE_PER_1000', 'LATITUDE', 'LONGITUDE']
                for col in expected_columns:
                    if col not in output_df.columns:
                        # Check for standardised versions
                        standardised_col = f"{col}_STANDARDISED"
                        assert standardised_col in output_df.columns or col in output_df.columns
                
            except Exception as e:
                pytest.fail(f"Failed to validate output file {output_file}: {str(e)}")
        
        # ========================================
        # COMPLIANCE VALIDATION
        # ========================================
        
        # Compliance status should be assessed
        compliance_status = quality_assessment.get('compliance_status', {})
        
        # Should have status for configured standards
        for standard in etl_config['quality_config'].compliance_standards:
            if standard in compliance_status:
                assert compliance_status[standard] in ['COMPLIANT', 'MINOR_ISSUES', 'NON_COMPLIANT']
        
        print(f"✅ Complete ETL pipeline test passed!")
        print(f"   📊 Processed {final_records} records in {total_duration:.2f} seconds")
        print(f"   🎯 Overall quality score: {quality_assessment['overall_quality_score']:.1f}%")
        print(f"   📈 Success rate: {execution_summary['success_rate']:.1f}%")
        print(f"   📁 Generated {len(output_files)} output files")
    
    def test_pipeline_error_handling_and_recovery(self, etl_config, temp_workspace):
        """
        Test pipeline error handling and recovery mechanisms.
        
        This test validates that the pipeline properly handles various
        error conditions and recovers gracefully when possible.
        """
        # Create stage definitions with intentional error conditions
        error_stage_definitions = [
            PipelineStageDefinition(
                stage_id="error_prone_extraction",
                stage_name="Error Prone Data Extraction",
                stage_type=PipelineStageType.EXTRACTION,
                stage_class="tests.end_to_end.test_complete_etl.MockAustralianDataExtractor",
                dependencies=[],
                configuration={
                    'data_size': 500,  # Smaller for faster testing
                    'introduce_errors': True,
                    'error_rate': 0.15,  # High error rate (15%)
                    'validation_enabled': True,
                    'quality_threshold': 85.0,
                    'halt_on_validation_failure': False
                },
                validation_required=True,
                quality_level=QualityLevel.COMPREHENSIVE,
                timeout_seconds=60,
                retry_attempts=3
            ),
            PipelineStageDefinition(
                stage_id="error_prone_transformation",
                stage_name="Error Prone Transformation",
                stage_type=PipelineStageType.TRANSFORMATION,
                stage_class="tests.end_to_end.test_complete_etl.MockGeographicTransformer",
                dependencies=["error_prone_extraction"],
                configuration={
                    'validation_enabled': True,
                    'introduce_transformation_errors': True,  # Introduce errors
                    'halt_on_validation_failure': False
                },
                validation_required=True,
                quality_level=QualityLevel.COMPREHENSIVE,
                timeout_seconds=60,
                retry_attempts=2
            )
        ]
        
        # Create pipeline with error-tolerant configuration
        error_tolerant_config = etl_config.copy()
        error_tolerant_config['quality_config'].halt_on_critical_errors = False
        error_tolerant_config['quality_config'].validation_mode = ValidationMode.PERMISSIVE
        
        pipeline = MasterETLPipeline(
            name="error_handling_test_pipeline",
            stage_definitions=error_stage_definitions,
            quality_config=error_tolerant_config['quality_config'],
            enable_checkpoints=True,
            max_retries=3
        )
        
        # Run pipeline (should complete despite errors)
        results = pipeline.run_complete_etl()
        
        # Pipeline should complete or fail gracefully
        assert results['execution_status'] in ['completed', 'failed']
        
        # Should have attempted all stages
        stage_results = results['stage_results']
        assert len(stage_results) >= 1
        
        # Quality assessment should reflect issues
        quality_assessment = results['quality_assessment']
        
        if results['execution_status'] == 'completed':
            # Quality score should be lower due to errors
            assert quality_assessment['overall_quality_score'] < 90.0
            
            # Should have recommendations for improvement
            assert len(quality_assessment['recommendations']) > 0
        
        # Validation results should capture issues
        validation_results = results['validation_results']
        if validation_results:
            # At least some validation should show quality issues
            quality_scores = [
                result.get('quality_score', 100)
                for result in validation_results.values()
            ]
            assert any(score < 90 for score in quality_scores)
        
        print(f"✅ Error handling test completed: {results['execution_status']}")
        print(f"   🔍 Quality score with errors: {quality_assessment.get('overall_quality_score', 0):.1f}%")
    
    def test_pipeline_performance_at_scale(self, temp_workspace):
        """
        Test pipeline performance with larger dataset.
        
        This test validates that the pipeline performs adequately
        with larger volumes of data typical in production.
        """
        # Create performance test configuration
        perf_config = {
            'name': 'performance_test_pipeline',
            'data_size': 10000,  # Larger dataset
            'quality_config': QualityAssuranceConfig(
                enabled=True,
                quality_level=QualityLevel.STANDARD,  # Reduced validation overhead
                validation_mode=ValidationMode.SELECTIVE,
                halt_on_critical_errors=False,
                generate_quality_reports=False,  # Reduce overhead
                monitor_performance=True,
                track_data_lineage=False  # Reduce overhead
            ),
            'parallel_stages': True,  # Enable parallelism
            'max_workers': 2
        }
        
        # Create streamlined stage definitions for performance
        perf_stage_definitions = [
            PipelineStageDefinition(
                stage_id="high_volume_extraction",
                stage_name="High Volume Data Extraction",
                stage_type=PipelineStageType.EXTRACTION,
                stage_class="tests.end_to_end.test_complete_etl.MockAustralianDataExtractor",
                dependencies=[],
                configuration={
                    'data_size': perf_config['data_size'],
                    'introduce_errors': False,  # Clean data for performance test
                    'validation_enabled': True,
                    'quality_threshold': 85.0
                },
                validation_required=True,
                quality_level=QualityLevel.STANDARD,
                parallel_capable=False
            ),
            PipelineStageDefinition(
                stage_id="optimised_transformation",
                stage_name="Optimised Transformation",
                stage_type=PipelineStageType.TRANSFORMATION,
                stage_class="tests.end_to_end.test_complete_etl.MockGeographicTransformer",
                dependencies=["high_volume_extraction"],
                configuration={
                    'validation_enabled': True,
                    'introduce_transformation_errors': False,
                    'batch_processing': True
                },
                validation_required=True,
                quality_level=QualityLevel.STANDARD,
                parallel_capable=True
            )
        ]
        
        pipeline = MasterETLPipeline(
            name=perf_config['name'],
            stage_definitions=perf_stage_definitions,
            quality_config=perf_config['quality_config'],
            parallel_stages=perf_config['parallel_stages'],
            max_workers=perf_config['max_workers']
        )
        
        # Measure performance
        start_time = time.time()
        
        results = pipeline.run_complete_etl()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Performance assertions
        assert results['execution_status'] == 'completed'
        
        final_records = results['final_data_records']
        assert final_records == perf_config['data_size']
        
        # Performance metrics
        records_per_second = final_records / execution_time
        assert records_per_second >= 50  # At least 50 records per second
        
        # Execution should complete in reasonable time
        max_time = 300  # 5 minutes for 10K records
        assert execution_time < max_time
        
        execution_summary = results['execution_summary']
        efficiency = execution_summary.get('pipeline_efficiency', 'unknown')
        assert efficiency in ['high', 'medium', 'low']
        
        print(f"✅ Performance test completed!")
        print(f"   📊 Processed {final_records:,} records in {execution_time:.2f} seconds")
        print(f"   🚀 Throughput: {records_per_second:.1f} records/second")
        print(f"   ⚡ Efficiency: {efficiency}")
        
        # Performance should be logged
        stage_results = results['stage_results']
        for stage_name, stage_result in stage_results.items():
            stage_duration = stage_result.get('duration', 0)
            if stage_duration:
                output_records = stage_result.get('output_records', 0)
                stage_throughput = output_records / stage_duration if stage_duration > 0 else 0
                print(f"   📈 {stage_name}: {stage_throughput:.1f} records/second")


# Test fixtures and utilities
@pytest.fixture(scope="session", autouse=True)
def configure_test_logging():
    """Configure logging for test execution."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Reduce noise from non-essential loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)