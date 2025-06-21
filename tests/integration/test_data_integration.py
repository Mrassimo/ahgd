"""
Integration tests for data integration processes.

Tests the complete integration pipeline from raw data sources to 
MasterHealthRecord instances, ensuring target schema compliance.
"""

import pytest
import yaml
from datetime import datetime
from typing import Dict, List, Any

from src.pipelines.integration.master_integration_pipeline import MasterIntegrationPipeline
from src.pipelines.integration.health_integration_pipeline import HealthIntegrationPipeline
from src.pipelines.integration.demographic_integration_pipeline import DemographicIntegrationPipeline
from src.pipelines.integration.geographic_integration_pipeline import GeographicIntegrationPipeline
from src.transformers.data_integrator import MasterDataIntegrator
from src.utils.integration_rules import DataIntegrationRules, ConflictResolver
from schemas.integrated_schema import MasterHealthRecord, DataIntegrationLevel
from schemas.seifa_schema import SEIFAIndexType


class TestDataIntegration:
    """Test complete data integration process."""
    
    @pytest.fixture
    def integration_config(self):
        """Load integration configuration for testing."""
        return {
            'data_source_orchestration': {
                'parallel_execution': False,  # Sequential for testing
                'health_integration': {
                    'validation': {},
                    'standardisation': {},
                    'integration': {},
                    'quality_assessment': {}
                },
                'demographic_integration': {
                    'validation': {},
                    'standardisation': {},
                    'seifa_integration': {},
                    'quality_assessment': {}
                },
                'geographic_integration': {
                    'validation': {},
                    'standardisation': {},
                    'spatial_relationships': {},
                    'quality_assessment': {}
                }
            },
            'master_record_creation': {
                'master_integration': {},
                'integration_rules': {
                    'source_priorities': {
                        'census': 1,
                        'seifa': 2,
                        'health_indicators': 3
                    }
                },
                'conflict_resolution': {},
                'quality_selection': {},
                'missing_data_strategies': {
                    'geographic': 'flag',
                    'demographic': 'interpolate',
                    'health': 'interpolate',
                    'socioeconomic': 'default'
                }
            },
            'derived_indicator_calculation': {
                'derived_indicators': {
                    'confidence_level': 95.0,
                    'national_averages': {
                        'composite_health_index': 75.0
                    }
                }
            },
            'target_schema_validation': {
                'strict_validation': False
            }
        }
    
    @pytest.fixture
    def sample_multi_source_data(self):
        """Create sample multi-source data for testing."""
        return [
            {
                'sa2_code': '101011001',
                'sa2_name': 'Test SA2 Area',
                
                # Census data
                'total_population': 5432,
                'male_population': 2687,
                'female_population': 2745,
                'demographic_profile': {
                    'age_groups': {
                        'age_0_4': 324,
                        'age_5_14': 542,
                        'age_15_24': 634,
                        'age_25_44': 1876,
                        'age_45_64': 1543,
                        'age_65_plus': 513
                    },
                    'sex_distribution': {
                        'male_population': 2687,
                        'female_population': 2745
                    }
                },
                
                # Geographic data
                'geographic_hierarchy': {
                    'sa3_code': '10101',
                    'sa4_code': '101',
                    'state_code': 'NSW'
                },
                'boundary_data': {
                    'centroid_latitude': -33.8688,
                    'centroid_longitude': 151.2093,
                    'area_sq_km': 2.54,
                    'coordinate_system': 'GDA2020'
                },
                
                # SEIFA data
                'seifa_scores': {
                    SEIFAIndexType.IRSD: 1156,
                    SEIFAIndexType.IRSAD: 1098,
                    SEIFAIndexType.IER: 1023,
                    SEIFAIndexType.IEO: 1134
                },
                'seifa_deciles': {
                    SEIFAIndexType.IRSD: 8,
                    SEIFAIndexType.IRSAD: 7,
                    SEIFAIndexType.IER: 6,
                    SEIFAIndexType.IEO: 8
                },
                
                # Health data
                'health_indicators': {
                    'life_expectancy': 84.2,
                    'gp_services_per_1000': 2.1,
                    'specialist_services_per_1000': 0.8,
                    'bulk_billing_rate': 89.3,
                    'smoking_prevalence': 8.7,
                    'obesity_prevalence': 21.4
                },
                
                # Medicare/PBS data
                'medicare_pbs': {
                    'gp_services_per_1000': 2.0,
                    'bulk_billing_rate': 88.5,
                    'emergency_dept_presentations_per_1000': 245
                }
            }
        ]
    
    def test_master_integration_pipeline_execution(self, integration_config, sample_multi_source_data):
        """Test complete master integration pipeline execution."""
        # Create pipeline
        pipeline = MasterIntegrationPipeline(integration_config)
        
        # Execute pipeline
        result = pipeline.execute(sample_multi_source_data)
        
        # Verify results
        assert len(result) == 1
        
        record = result[0]
        
        # Test primary identification
        assert record['sa2_code'] == '101011001'
        assert record['sa2_name'] == 'Test SA2 Area'
        
        # Test geographic components
        assert 'geographic_hierarchy' in record
        assert record['geographic_hierarchy']['sa3_code'] == '10101'
        assert 'boundary_data' in record
        
        # Test demographic components
        assert record['total_population'] == 5432
        assert 'demographic_profile' in record
        
        # Test socioeconomic components
        assert 'seifa_scores' in record
        assert 'seifa_deciles' in record
        
        # Test integration metadata
        assert 'integration_level' in record
        assert 'data_completeness_score' in record
        assert 'integration_timestamp' in record
        
        # Test schema compliance
        assert record.get('schema_version') == "2.0.0"
    
    def test_health_integration_pipeline(self, sample_multi_source_data):
        """Test health-specific integration pipeline."""
        config = {
            'validation': {},
            'standardisation': {},
            'integration': {
                'health_field_priorities': {
                    'life_expectancy': ['health_indicators', 'aihw'],
                    'healthcare_utilisation': ['medicare_pbs', 'health_indicators']
                }
            },
            'quality_assessment': {}
        }
        
        pipeline = HealthIntegrationPipeline(config)
        result = pipeline.execute(sample_multi_source_data)
        
        assert len(result) == 1
        record = result[0]
        
        # Check health integration occurred
        assert 'health_integration_timestamp' in record
        assert 'health_sources_count' in record
        
        # Check specific health indicators were integrated
        assert record.get('life_expectancy') == 84.2
        assert record.get('gp_services_per_1000') is not None
        assert record.get('bulk_billing_rate') is not None
    
    def test_demographic_integration_pipeline(self, sample_multi_source_data):
        """Test demographic-specific integration pipeline."""
        config = {
            'validation': {},
            'standardisation': {},
            'seifa_integration': {},
            'quality_assessment': {}
        }
        
        pipeline = DemographicIntegrationPipeline(config)
        result = pipeline.execute(sample_multi_source_data)
        
        assert len(result) == 1
        record = result[0]
        
        # Check population consistency
        assert record['total_population'] == 5432
        
        # Check SEIFA integration
        assert 'seifa_scores' in record
        assert 'seifa_deciles' in record
        assert 'disadvantage_category' in record
        
        # Check demographic quality assessment
        assert 'demographic_completeness_scores' in record
    
    def test_geographic_integration_pipeline(self, sample_multi_source_data):
        """Test geographic-specific integration pipeline."""
        config = {
            'validation': {},
            'standardisation': {},
            'spatial_relationships': {},
            'quality_assessment': {}
        }
        
        pipeline = GeographicIntegrationPipeline(config)
        result = pipeline.execute(sample_multi_source_data)
        
        assert len(result) == 1
        record = result[0]
        
        # Check geographic hierarchy
        assert 'geographic_hierarchy' in record
        
        # Check boundary data
        assert 'boundary_data' in record
        
        # Check spatial classifications
        assert 'urbanisation' in record
        assert 'remoteness_category' in record
        
        # Check quality assessment
        assert 'geographic_completeness_scores' in record
    
    def test_conflict_resolution(self):
        """Test conflict resolution between data sources."""
        # Create test data with conflicting values
        conflicting_data = {
            'sa2_code': '101011001',
            'gp_services_per_1000_health_indicators': 2.1,
            'gp_services_per_1000_medicare_pbs': 2.0,
            'bulk_billing_rate_health_indicators': 89.3,
            'bulk_billing_rate_medicare_pbs': 88.5
        }
        
        conflict_resolver = ConflictResolver({
            'default_strategy': 'highest_quality'
        })
        
        # Test conflict resolution for GP services
        conflict = conflict_resolver.resolve_conflict(
            field_name='gp_services_per_1000',
            sa2_code='101011001',
            conflicting_data={
                'health_indicators': (2.1, 0.85, datetime.now()),
                'medicare_pbs': (2.0, 0.90, datetime.now())
            }
        )
        
        # Should select medicare_pbs due to higher quality
        assert conflict.resolved_value == 2.0
        assert 'highest quality' in conflict.resolution_reason.lower()
    
    def test_missing_data_handling(self):
        """Test handling of missing data in integration."""
        incomplete_data = [{
            'sa2_code': '101011001',
            'sa2_name': 'Test SA2 Area',
            'total_population': 5432,
            'geographic_hierarchy': {
                'sa3_code': '10101',
                'sa4_code': '101',
                'state_code': 'NSW'
            },
            # Missing: health indicators, SEIFA data, boundary data
        }]
        
        config = {
            'data_source_orchestration': {'parallel_execution': False},
            'master_record_creation': {
                'missing_data_strategies': {
                    'health': 'interpolate',
                    'socioeconomic': 'default'
                },
                'interpolation_enabled': True
            },
            'derived_indicator_calculation': {'derived_indicators': {}},
            'target_schema_validation': {'strict_validation': False}
        }
        
        pipeline = MasterIntegrationPipeline(config)
        result = pipeline.execute(incomplete_data)
        
        assert len(result) == 1
        record = result[0]
        
        # Should have handled missing data
        assert record['sa2_code'] == '101011001'
        assert 'integration_level' in record
        assert record['integration_level'] in [
            DataIntegrationLevel.MINIMAL.value,
            DataIntegrationLevel.MINIMAL
        ]
    
    def test_data_quality_scoring(self, sample_multi_source_data):
        """Test data quality scoring during integration."""
        config = {
            'data_source_orchestration': {'parallel_execution': False},
            'master_record_creation': {},
            'derived_indicator_calculation': {'derived_indicators': {}},
            'target_schema_validation': {}
        }
        
        pipeline = MasterIntegrationPipeline(config)
        result = pipeline.execute(sample_multi_source_data)
        
        record = result[0]
        
        # Should have quality scores
        assert 'data_completeness_score' in record
        assert isinstance(record['data_completeness_score'], (int, float))
        assert 0 <= record['data_completeness_score'] <= 100
        
        # Should have integration level
        assert 'integration_level' in record
    
    def test_derived_indicator_calculation(self, sample_multi_source_data):
        """Test calculation of derived health indicators."""
        config = {
            'data_source_orchestration': {'parallel_execution': False},
            'master_record_creation': {},
            'derived_indicator_calculation': {
                'derived_indicators': {
                    'confidence_level': 95.0,
                    'national_averages': {
                        'composite_health_index': 75.0
                    }
                }
            },
            'target_schema_validation': {}
        }
        
        pipeline = MasterIntegrationPipeline(config)
        result = pipeline.execute(sample_multi_source_data)
        
        record = result[0]
        
        # Should have calculated derived indicators
        assert 'composite_health_index' in record
        if record['composite_health_index'] is not None:
            assert isinstance(record['composite_health_index'], (int, float))
            assert 0 <= record['composite_health_index'] <= 100
    
    def test_target_schema_validation(self, sample_multi_source_data):
        """Test validation against target MasterHealthRecord schema."""
        config = {
            'data_source_orchestration': {'parallel_execution': False},
            'master_record_creation': {},
            'derived_indicator_calculation': {'derived_indicators': {}},
            'target_schema_validation': {
                'strict_validation': False,
                'validation_standards': []
            }
        }
        
        pipeline = MasterIntegrationPipeline(config)
        result = pipeline.execute(sample_multi_source_data)
        
        record = result[0]
        
        # Should have validation metadata
        assert 'schema_validation_passed' in record
        assert 'schema_validation_timestamp' in record
        
        # Check required fields are present
        required_fields = [
            'sa2_code', 'sa2_name', 'geographic_hierarchy', 
            'demographic_profile', 'total_population'
        ]
        
        for field in required_fields:
            assert field in record, f"Required field {field} missing"
    
    def test_integration_performance(self, sample_multi_source_data):
        """Test integration pipeline performance metrics."""
        # Create larger dataset for performance testing
        large_dataset = sample_multi_source_data * 100  # 100 records
        
        # Update SA2 codes to be unique
        for i, record in enumerate(large_dataset):
            record['sa2_code'] = f'10101{i:04d}'
        
        config = {
            'data_source_orchestration': {'parallel_execution': True},
            'master_record_creation': {},
            'derived_indicator_calculation': {'derived_indicators': {}},
            'target_schema_validation': {},
            'batch_size': 50,
            'enable_performance_monitoring': True
        }
        
        pipeline = MasterIntegrationPipeline(config)
        start_time = datetime.utcnow()
        
        result = pipeline.execute(large_dataset)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify performance
        assert len(result) == 100
        assert processing_time < 60  # Should complete within 60 seconds
        
        # Check metrics
        metrics = pipeline.get_integration_metrics()
        assert metrics['total_input_records'] == 100
        assert metrics['total_output_records'] == 100
        assert metrics['processing_time_seconds'] > 0
    
    def test_integration_error_handling(self):
        """Test error handling in integration pipeline."""
        # Create invalid data that should trigger errors
        invalid_data = [{
            'sa2_code': 'invalid',  # Invalid SA2 code format
            'total_population': -100,  # Negative population
            'demographic_profile': None  # Missing required field
        }]
        
        config = {
            'data_source_orchestration': {'parallel_execution': False},
            'master_record_creation': {},
            'derived_indicator_calculation': {'derived_indicators': {}},
            'target_schema_validation': {'strict_validation': False}
        }
        
        pipeline = MasterIntegrationPipeline(config)
        
        # Should handle errors gracefully
        result = pipeline.execute(invalid_data)
        
        # May return empty result or records with error flags
        if result:
            record = result[0]
            # Should have error information
            assert 'schema_validation_passed' in record
            assert record['schema_validation_passed'] is False
    
    def test_integration_metrics_calculation(self, sample_multi_source_data):
        """Test calculation of integration metrics."""
        config = {
            'data_source_orchestration': {'parallel_execution': False},
            'master_record_creation': {},
            'derived_indicator_calculation': {'derived_indicators': {}},
            'target_schema_validation': {}
        }
        
        pipeline = MasterIntegrationPipeline(config)
        result = pipeline.execute(sample_multi_source_data)
        
        metrics = pipeline.get_integration_metrics()
        
        # Verify all expected metrics are present
        expected_metrics = [
            'total_input_records',
            'total_output_records', 
            'validation_pass_rate',
            'average_completeness_score',
            'integration_levels',
            'processing_time_seconds'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
        
        # Verify metric values
        assert metrics['total_input_records'] == 1
        assert metrics['total_output_records'] == 1
        assert 0 <= metrics['validation_pass_rate'] <= 100
        assert 0 <= metrics['average_completeness_score'] <= 100
        assert metrics['processing_time_seconds'] > 0


class TestIntegrationRules:
    """Test integration rules and business logic."""
    
    def test_data_integration_rules(self):
        """Test data integration rules engine."""
        config = {
            'source_priorities': {
                'census': 1,
                'seifa': 2,
                'health_indicators': 3
            },
            'mandatory_fields': [
                'sa2_code', 'total_population'
            ],
            'quality_thresholds': {
                'total_population': 0.90
            }
        }
        
        rules_engine = DataIntegrationRules(config)
        
        # Test field priority
        priority = rules_engine.get_field_priority('total_population', 'census')
        assert priority == 1  # Census has highest priority
        
        # Test mandatory field check
        assert rules_engine.is_mandatory_field('sa2_code')
        assert rules_engine.is_mandatory_field('total_population')
        assert not rules_engine.is_mandatory_field('optional_field')
        
        # Test quality threshold
        threshold = rules_engine.get_quality_threshold('total_population')
        assert threshold == 0.90
    
    def test_integration_rule_evaluation(self):
        """Test evaluation of specific integration rules."""
        rules_config = {
            'rules': [{
                'name': 'population_validation',
                'type': 'validation',
                'source_fields': ['total_population', 'male_population', 'female_population'],
                'target_field': 'total_population',
                'condition': 'abs(total_population - (male_population + female_population)) < 10',
                'priority': 200,
                'enabled': True
            }]
        }
        
        rules_engine = DataIntegrationRules(rules_config)
        
        # Test valid data
        valid_data = {
            'sa2_code': '101011001',
            'total_population': 1000,
            'male_population': 485,
            'female_population': 515
        }
        
        applicable_rules = rules_engine.evaluate_integration_rules('101011001', {'test': valid_data})
        assert len(applicable_rules) == 1
        assert applicable_rules[0].rule_name == 'population_validation'
    
    def test_completeness_validation(self):
        """Test integration completeness validation."""
        config = {
            'mandatory_fields': [
                'sa2_code', 'sa2_name', 'total_population'
            ]
        }
        
        rules_engine = DataIntegrationRules(config)
        
        # Test complete record
        complete_record = {
            'sa2_code': '101011001',
            'sa2_name': 'Test Area',
            'total_population': 1000
        }
        
        is_valid, missing_fields = rules_engine.validate_integration_completeness(complete_record)
        assert is_valid
        assert len(missing_fields) == 0
        
        # Test incomplete record
        incomplete_record = {
            'sa2_code': '101011001'
            # Missing sa2_name and total_population
        }
        
        is_valid, missing_fields = rules_engine.validate_integration_completeness(incomplete_record)
        assert not is_valid
        assert 'sa2_name' in missing_fields
        assert 'total_population' in missing_fields


@pytest.mark.performance
class TestIntegrationPerformance:
    """Performance tests for integration processes."""
    
    def test_large_dataset_integration(self):
        """Test integration with large dataset (2,473 SA2 areas)."""
        # Create dataset representing all SA2 areas
        large_dataset = []
        
        for i in range(2473):  # All SA2 areas in Australia
            record = {
                'sa2_code': f'{i+101011001:09d}',
                'sa2_name': f'SA2 Area {i+1}',
                'total_population': 1000 + (i * 5),  # Varying population sizes
                'geographic_hierarchy': {
                    'sa3_code': f'{(i//10)+10101:05d}',
                    'sa4_code': f'{(i//100)+101:03d}',
                    'state_code': ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'][i % 8]
                },
                'boundary_data': {
                    'centroid_latitude': -35.0 + (i * 0.01),
                    'centroid_longitude': 149.0 + (i * 0.01),
                    'area_sq_km': 1.0 + (i * 0.1)
                }
            }
            large_dataset.append(record)
        
        config = {
            'data_source_orchestration': {'parallel_execution': True},
            'master_record_creation': {},
            'derived_indicator_calculation': {'derived_indicators': {}},
            'target_schema_validation': {'strict_validation': False},
            'batch_size': 500
        }
        
        pipeline = MasterIntegrationPipeline(config)
        start_time = datetime.utcnow()
        
        result = pipeline.execute(large_dataset)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Performance requirements
        assert len(result) == 2473
        assert processing_time < 300  # Should complete within 5 minutes
        
        # Calculate throughput
        throughput = len(result) / processing_time
        assert throughput > 8  # At least 8 records per second
        
        print(f"Processed {len(result)} records in {processing_time:.1f}s")
        print(f"Throughput: {throughput:.1f} records/second")
    
    def test_memory_usage(self):
        """Test memory usage during integration."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderately large dataset
        dataset = []
        for i in range(1000):
            record = {
                'sa2_code': f'{i+101011001:09d}',
                'total_population': 1000,
                'demographic_profile': {'age_groups': {f'age_{j}': 100 for j in range(10)}}
            }
            dataset.append(record)
        
        config = {
            'data_source_orchestration': {'parallel_execution': False},
            'master_record_creation': {},
            'derived_indicator_calculation': {'derived_indicators': {}},
            'target_schema_validation': {}
        }
        
        pipeline = MasterIntegrationPipeline(config)
        result = pipeline.execute(dataset)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 500  # Less than 500MB increase
        assert len(result) == 1000
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")


if __name__ == '__main__':
    # Run tests with performance markers
    pytest.main([__file__, '-v', '-m', 'not performance'])
    
    # Run performance tests separately
    print("\nRunning performance tests...")
    pytest.main([__file__, '-v', '-m', 'performance'])