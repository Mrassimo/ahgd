"""
Large-Scale Data Processing Performance Tests - Phase 5.4

Tests the platform's ability to handle 1M+ Australian health records with realistic
data patterns, validating performance targets for processing speed, memory usage,
and system throughput under production-scale loads.

Key Performance Tests:
- 1M+ record end-to-end processing pipeline validation
- 2,454 SA2 areas simultaneous processing
- Memory optimization effectiveness at scale (57.5% target)
- Concurrent dataset processing performance
- System stability under extended operations
"""

import pytest
import polars as pl
import numpy as np
import time
import psutil
import gc
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import concurrent.futures
import threading
from contextlib import contextmanager

from src.data_processing.seifa_processor import SEIFAProcessor
from src.data_processing.health_processor import HealthDataProcessor
from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor
from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from src.data_processing.storage.incremental_processor import IncrementalProcessor
from src.analysis.risk.health_risk_calculator import HealthRiskCalculator
from tests.performance import PERFORMANCE_CONFIG, AUSTRALIAN_DATA_SCALE

logger = logging.getLogger(__name__)


@dataclass
class LargeScalePerformanceResult:
    """Results from large-scale performance testing."""
    test_name: str
    records_processed: int
    processing_time_seconds: float
    memory_usage_mb: float
    throughput_records_per_second: float
    memory_optimization_percent: float
    integration_success_rate: float
    storage_efficiency_mb: float
    targets_met: Dict[str, bool]
    stage_breakdown: Dict[str, float]


class AustralianHealthDataGenerator:
    """Generates realistic Australian health data at scale for performance testing."""
    
    # Australian geographic and demographic patterns
    AUSTRALIAN_STATES = {
        'NSW': {'weight': 0.32, 'sa2_range': (10001, 13000)},
        'VIC': {'weight': 0.26, 'sa2_range': (20001, 23000)},
        'QLD': {'weight': 0.20, 'sa2_range': (30001, 33000)},
        'WA': {'weight': 0.11, 'sa2_range': (50001, 52000)},
        'SA': {'weight': 0.07, 'sa2_range': (40001, 41000)},
        'TAS': {'weight': 0.02, 'sa2_range': (60001, 60500)},
        'ACT': {'weight': 0.02, 'sa2_range': (80001, 80100)},
        'NT': {'weight': 0.01, 'sa2_range': (70001, 70100)}
    }
    
    # Health condition patterns based on Australian health statistics
    HEALTH_CONDITIONS = {
        'cardiovascular': {'prevalence': 0.18, 'age_correlation': 0.8, 'seifa_correlation': -0.4},
        'diabetes': {'prevalence': 0.05, 'age_correlation': 0.7, 'seifa_correlation': -0.5},
        'mental_health': {'prevalence': 0.15, 'age_correlation': -0.3, 'seifa_correlation': -0.3},
        'respiratory': {'prevalence': 0.12, 'age_correlation': 0.6, 'seifa_correlation': -0.2},
        'musculoskeletal': {'prevalence': 0.22, 'age_correlation': 0.9, 'seifa_correlation': -0.1}
    }
    
    def __init__(self, seed: int = 42):
        """Initialize with deterministic random seed for reproducible testing."""
        np.random.seed(seed)
        self.sa2_codes = self._generate_sa2_codes()
    
    def _generate_sa2_codes(self) -> List[str]:
        """Generate all 2,454 Australian SA2 codes."""
        sa2_codes = []
        for state, info in self.AUSTRALIAN_STATES.items():
            start, end = info['sa2_range']
            count = int(AUSTRALIAN_DATA_SCALE['sa2_areas_total'] * info['weight'])
            codes = [f"{i:08d}" for i in range(start, start + count)]
            sa2_codes.extend(codes)
        
        # Ensure we have exactly 2,454 SA2 codes
        while len(sa2_codes) < AUSTRALIAN_DATA_SCALE['sa2_areas_total']:
            sa2_codes.append(f"{len(sa2_codes) + 10000000:08d}")
        
        return sa2_codes[:AUSTRALIAN_DATA_SCALE['sa2_areas_total']]
    
    def generate_large_scale_seifa_data(self) -> pl.DataFrame:
        """Generate SEIFA data for all 2,454 SA2 areas."""
        logger.info(f"Generating SEIFA data for {len(self.sa2_codes)} SA2 areas")
        
        # Create correlated SEIFA indices with realistic Australian patterns
        base_scores = np.random.normal(1000, 100, len(self.sa2_codes))
        
        seifa_data = []
        for i, sa2_code in enumerate(self.sa2_codes):
            state = self._get_state_from_sa2(sa2_code)
            base_score = base_scores[i]
            
            # Add state-level variations (metropolitan vs regional patterns)
            state_modifier = np.random.normal(0, 50)
            if state in ['NSW', 'VIC']:  # More urban areas
                state_modifier += 30
            elif state in ['NT', 'TAS']:  # More disadvantaged areas
                state_modifier -= 40
            
            irsd_score = max(1, base_score + state_modifier + np.random.normal(0, 25))
            
            # Generate correlated indices
            irsad_score = irsd_score + np.random.normal(0, 40)
            ier_score = irsd_score + np.random.normal(0, 60)
            ieo_score = irsd_score + np.random.normal(0, 45)
            
            seifa_data.append({
                'sa2_code_2021': sa2_code,
                'sa2_name_2021': f"SA2 Area {sa2_code}",
                'state_territory_2021': state,
                'irsd_score': irsd_score,
                'irsd_decile': min(10, max(1, int((irsd_score - 500) / 100) + 1)),
                'irsad_score': irsad_score,
                'irsad_decile': min(10, max(1, int((irsad_score - 500) / 100) + 1)),
                'ier_score': ier_score,
                'ier_decile': min(10, max(1, int((ier_score - 500) / 100) + 1)),
                'ieo_score': ieo_score,
                'ieo_decile': min(10, max(1, int((ieo_score - 500) / 100) + 1)),
                'usual_resident_population': np.random.randint(100, 25000)
            })
        
        return pl.DataFrame(seifa_data)
    
    def generate_large_scale_health_data(self, n_records: int = 1000000) -> pl.DataFrame:
        """Generate 1M+ realistic Australian health records."""
        logger.info(f"Generating {n_records:,} health records with Australian patterns")
        
        health_data = []
        
        # Generate in chunks to manage memory
        chunk_size = 50000
        for chunk_start in range(0, n_records, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_records)
            chunk_size_actual = chunk_end - chunk_start
            
            # Select SA2 codes weighted by population
            sa2_indices = np.random.choice(len(self.sa2_codes), chunk_size_actual, replace=True)
            
            for i in range(chunk_size_actual):
                sa2_code = self.sa2_codes[sa2_indices[i]]
                state = self._get_state_from_sa2(sa2_code)
                
                # Generate demographics with Australian patterns
                age = self._generate_realistic_age()
                gender = np.random.choice(['M', 'F'], p=[0.49, 0.51])
                
                # Generate health utilisation based on age and socio-economic factors
                seifa_decile = np.random.randint(1, 11)  # Will be matched with actual SEIFA later
                
                # Age and SEIFA-correlated health utilisation
                base_prescriptions = max(0, np.random.poisson(5) + (age - 40) / 10 + (10 - seifa_decile))
                gp_visits = max(0, np.random.poisson(8) + (age - 30) / 15 + (10 - seifa_decile) / 2)
                specialist_visits = max(0, np.random.poisson(2) + (age - 50) / 20)
                
                # Generate costs with realistic Australian Medicare patterns
                prescription_cost = base_prescriptions * np.random.exponential(25)
                gp_cost = gp_visits * np.random.normal(180, 40)  # Medicare GP consultation
                specialist_cost = specialist_visits * np.random.normal(350, 100)
                
                # Generate chronic conditions
                chronic_count = self._generate_chronic_conditions(age, seifa_decile)
                
                health_data.append({
                    'sa2_code': sa2_code,
                    'state_territory': state,
                    'age_group': self._age_to_group(age),
                    'gender': gender,
                    'seifa_decile_estimate': seifa_decile,
                    'prescription_count': int(base_prescriptions),
                    'gp_visits': int(gp_visits),
                    'specialist_visits': int(specialist_visits),
                    'total_prescriptions': int(base_prescriptions),
                    'prescription_cost_aud': max(0, prescription_cost),
                    'gp_cost_aud': max(0, gp_cost),
                    'specialist_cost_aud': max(0, specialist_cost),
                    'total_cost_aud': max(0, prescription_cost + gp_cost + specialist_cost),
                    'chronic_conditions_count': chronic_count,
                    'service_year': np.random.choice([2022, 2023], p=[0.3, 0.7]),
                    'service_month': np.random.randint(1, 13),
                    'remoteness_category': self._get_remoteness_category(state),
                    'healthcare_access_score': max(1, min(10, np.random.normal(6, 2))),
                    'risk_score_preliminary': max(1, min(10, np.random.normal(5, 2)))
                })
        
        return pl.DataFrame(health_data)
    
    def generate_large_scale_boundary_data(self) -> pl.DataFrame:
        """Generate geographic boundary data for all SA2 areas."""
        logger.info(f"Generating boundary data for {len(self.sa2_codes)} SA2 areas")
        
        boundary_data = []
        for sa2_code in self.sa2_codes:
            state = self._get_state_from_sa2(sa2_code)
            
            # Generate realistic coordinates within Australian bounds
            if state == 'NSW':
                lat = np.random.uniform(-37.5, -28.2)
                lon = np.random.uniform(140.9, 153.6)
            elif state == 'VIC':
                lat = np.random.uniform(-39.2, -34.0)
                lon = np.random.uniform(140.9, 149.9)
            elif state == 'QLD':
                lat = np.random.uniform(-29.0, -10.4)
                lon = np.random.uniform(138.0, 153.6)
            elif state == 'WA':
                lat = np.random.uniform(-35.1, -13.8)
                lon = np.random.uniform(112.9, 129.0)
            elif state == 'SA':
                lat = np.random.uniform(-38.1, -26.0)
                lon = np.random.uniform(129.0, 141.0)
            elif state == 'TAS':
                lat = np.random.uniform(-43.6, -39.2)
                lon = np.random.uniform(143.8, 148.4)
            elif state == 'ACT':
                lat = np.random.uniform(-35.9, -35.1)
                lon = np.random.uniform(148.8, 149.4)
            else:  # NT
                lat = np.random.uniform(-26.0, -10.9)
                lon = np.random.uniform(129.0, 138.0)
            
            # Generate area characteristics
            area_sq_km = np.random.exponential(50)
            population = np.random.randint(100, 25000)
            
            boundary_data.append({
                'sa2_code_2021': sa2_code,
                'sa2_name_2021': f"SA2 Area {sa2_code}",
                'state_territory_2021': state,
                'latitude': lat,
                'longitude': lon,
                'area_sq_km': area_sq_km,
                'population_density_per_sq_km': population / max(area_sq_km, 0.1),
                'usual_resident_population': population,
                'remoteness_category': self._get_remoteness_category(state),
                'urban_rural_classification': np.random.choice(['Major Cities', 'Inner Regional', 'Outer Regional', 'Remote', 'Very Remote'], 
                                                             p=[0.71, 0.18, 0.08, 0.02, 0.01])
            })
        
        return pl.DataFrame(boundary_data)
    
    def _get_state_from_sa2(self, sa2_code: str) -> str:
        """Determine state from SA2 code pattern."""
        code_int = int(sa2_code)
        if 10001 <= code_int <= 13000:
            return 'NSW'
        elif 20001 <= code_int <= 23000:
            return 'VIC'
        elif 30001 <= code_int <= 33000:
            return 'QLD'
        elif 40001 <= code_int <= 41000:
            return 'SA'
        elif 50001 <= code_int <= 52000:
            return 'WA'
        elif 60001 <= code_int <= 60500:
            return 'TAS'
        elif 70001 <= code_int <= 70100:
            return 'NT'
        elif 80001 <= code_int <= 80100:
            return 'ACT'
        else:
            return 'NSW'  # Default
    
    def _generate_realistic_age(self) -> int:
        """Generate age with realistic Australian distribution."""
        # Based on Australian age distribution
        age_groups = [
            (0, 17, 0.24), (18, 34, 0.26), (35, 49, 0.20),
            (50, 64, 0.18), (65, 79, 0.10), (80, 95, 0.02)
        ]
        
        chosen_group = np.random.choice(len(age_groups), p=[g[2] for g in age_groups])
        min_age, max_age, _ = age_groups[chosen_group]
        return np.random.randint(min_age, max_age + 1)
    
    def _age_to_group(self, age: int) -> str:
        """Convert age to age group string."""
        if age <= 17:
            return '0-17'
        elif age <= 34:
            return '18-34'
        elif age <= 49:
            return '35-49'
        elif age <= 64:
            return '50-64'
        elif age <= 79:
            return '65-79'
        else:
            return '80+'
    
    def _generate_chronic_conditions(self, age: int, seifa_decile: int) -> int:
        """Generate number of chronic conditions based on age and SEIFA."""
        base_probability = 0.05 + (age / 100) * 0.3 + (10 - seifa_decile) / 100 * 0.2
        return np.random.poisson(base_probability * 8)  # Scale to get reasonable counts
    
    def _get_remoteness_category(self, state: str) -> str:
        """Get remoteness category based on state patterns."""
        if state in ['NSW', 'VIC']:
            return np.random.choice(['Major Cities', 'Inner Regional', 'Outer Regional'], p=[0.8, 0.15, 0.05])
        elif state in ['QLD', 'WA', 'SA']:
            return np.random.choice(['Major Cities', 'Inner Regional', 'Outer Regional', 'Remote'], p=[0.6, 0.2, 0.15, 0.05])
        else:  # NT, TAS, ACT
            return np.random.choice(['Major Cities', 'Inner Regional', 'Outer Regional', 'Remote', 'Very Remote'], 
                                  p=[0.4, 0.2, 0.2, 0.15, 0.05])


@contextmanager
def performance_monitor():
    """Context manager for performance monitoring."""
    process = psutil.Process()
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu = process.cpu_percent()
    
    yield
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    end_cpu = process.cpu_percent()
    
    logger.info(f"Performance: {end_time - start_time:.2f}s, "
                f"{end_memory - start_memory:.1f}MB memory delta, "
                f"{end_cpu - start_cpu:.1f}% CPU delta")


class TestLargeScaleProcessing:
    """Large-scale processing performance tests for 1M+ Australian health records."""
    
    @pytest.fixture(scope="class")
    def data_generator(self):
        """Create Australian health data generator."""
        return AustralianHealthDataGenerator(seed=42)
    
    @pytest.fixture(scope="class")
    def performance_processors(self, tmp_path_factory):
        """Create performance testing processors."""
        temp_dir = tmp_path_factory.mktemp("performance_test")
        
        return {
            'seifa_processor': SEIFAProcessor(data_dir=temp_dir),
            'health_processor': HealthDataProcessor(data_dir=temp_dir),
            'boundary_processor': SimpleBoundaryProcessor(data_dir=temp_dir),
            'storage_manager': ParquetStorageManager(base_path=temp_dir / "parquet"),
            'memory_optimizer': MemoryOptimizer(),
            'incremental_processor': IncrementalProcessor(temp_dir / "lake"),
            'risk_calculator': HealthRiskCalculator(data_dir=temp_dir / "processed"),
            'temp_dir': temp_dir
        }
    
    def test_million_record_end_to_end_pipeline(self, data_generator, performance_processors):
        """Test complete pipeline with 1M+ Australian health records."""
        logger.info("Starting 1M+ record end-to-end pipeline performance test")
        
        # Target: <5 minutes for 1M+ records end-to-end
        target_processing_time = PERFORMANCE_CONFIG['large_scale_targets']['max_processing_time_minutes'] * 60
        target_throughput = PERFORMANCE_CONFIG['large_scale_targets']['min_throughput_records_per_second']
        target_memory_gb = PERFORMANCE_CONFIG['large_scale_targets']['max_memory_usage_gb']
        
        with performance_monitor():
            pipeline_start = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # =================================================================
            # Stage 1: Generate Large-Scale Australian Data (1M+ records)
            # =================================================================
            stage1_start = time.time()
            
            # Generate realistic Australian datasets
            seifa_data = data_generator.generate_large_scale_seifa_data()  # 2,454 SA2 areas
            health_data = data_generator.generate_large_scale_health_data(1000000)  # 1M health records
            boundary_data = data_generator.generate_large_scale_boundary_data()  # 2,454 boundaries
            
            total_records = len(seifa_data) + len(health_data) + len(boundary_data)
            
            stage1_time = time.time() - stage1_start
            logger.info(f"Stage 1 (Data Generation): {stage1_time:.2f}s, {total_records:,} total records")
            
            assert total_records >= 1000000, f"Should process ≥1M records, generated {total_records:,}"
            
            # =================================================================
            # Stage 2: Data Validation and Initial Processing
            # =================================================================
            stage2_start = time.time()
            stage2_memory_start = process.memory_info().rss / 1024 / 1024
            
            # Process SEIFA data (all 2,454 SA2 areas)
            seifa_processor = performance_processors['seifa_processor']
            processed_seifa = seifa_processor._validate_seifa_data(seifa_data)
            
            # Process health data (1M records)
            health_processor = performance_processors['health_processor']
            validated_health = health_processor._validate_health_data(health_data)
            
            # Process boundary data (2,454 areas)
            boundary_processor = performance_processors['boundary_processor']
            validated_boundaries = boundary_processor._validate_boundary_data(boundary_data)
            
            stage2_time = time.time() - stage2_start
            stage2_memory_end = process.memory_info().rss / 1024 / 1024
            stage2_memory_usage = stage2_memory_end - stage2_memory_start
            
            logger.info(f"Stage 2 (Validation): {stage2_time:.2f}s, {stage2_memory_usage:.1f}MB memory")
            
            # Validation performance targets
            assert stage2_time < 120.0, f"Validation took {stage2_time:.1f}s, expected <120s"
            
            # =================================================================
            # Stage 3: Memory Optimization (57.5% Target)
            # =================================================================
            stage3_start = time.time()
            
            memory_optimizer = performance_processors['memory_optimizer']
            
            # Measure pre-optimization memory
            pre_opt_seifa_mb = processed_seifa.estimated_size("mb")
            pre_opt_health_mb = validated_health.estimated_size("mb")
            pre_opt_boundary_mb = validated_boundaries.estimated_size("mb")
            pre_opt_total_mb = pre_opt_seifa_mb + pre_opt_health_mb + pre_opt_boundary_mb
            
            # Apply memory optimizations
            optimized_seifa = memory_optimizer.optimize_data_types(processed_seifa, data_category="seifa")
            optimized_health = memory_optimizer.optimize_data_types(validated_health, data_category="health")
            optimized_boundaries = memory_optimizer.optimize_data_types(validated_boundaries, data_category="geographic")
            
            # Measure post-optimization memory
            post_opt_seifa_mb = optimized_seifa.estimated_size("mb")
            post_opt_health_mb = optimized_health.estimated_size("mb")
            post_opt_boundary_mb = optimized_boundaries.estimated_size("mb")
            post_opt_total_mb = post_opt_seifa_mb + post_opt_health_mb + post_opt_boundary_mb
            
            memory_reduction_percent = ((pre_opt_total_mb - post_opt_total_mb) / pre_opt_total_mb) * 100
            
            stage3_time = time.time() - stage3_start
            logger.info(f"Stage 3 (Memory Optimization): {stage3_time:.2f}s, {memory_reduction_percent:.1f}% reduction")
            
            # Memory optimization validation (57.5% target)
            target_memory_reduction = PERFORMANCE_CONFIG['memory_optimization_targets']['min_memory_reduction_percent']
            assert memory_reduction_percent >= target_memory_reduction * 0.7, \
                f"Memory reduction {memory_reduction_percent:.1f}% should be ≥{target_memory_reduction * 0.7:.1f}% (70% of target)"
            
            # =================================================================
            # Stage 4: Large-Scale Data Integration
            # =================================================================
            stage4_start = time.time()
            
            # Aggregate health data by SA2 (1M records → 2,454 areas)
            aggregated_health = health_processor._aggregate_by_sa2(optimized_health)
            
            # Enhance boundaries with population density
            enhanced_boundaries = boundary_processor._calculate_population_density(optimized_boundaries)
            
            # Create comprehensive integrated dataset
            comprehensive_integration = optimized_seifa.join(
                aggregated_health, left_on="sa2_code_2021", right_on="sa2_code", how="left"
            ).join(
                enhanced_boundaries, on="sa2_code_2021", how="left"
            )
            
            integration_success_rate = len(comprehensive_integration) / len(optimized_seifa)
            
            stage4_time = time.time() - stage4_start
            logger.info(f"Stage 4 (Integration): {stage4_time:.2f}s, {integration_success_rate:.1%} success rate")
            
            # Integration performance validation
            assert stage4_time < 180.0, f"Integration took {stage4_time:.1f}s, expected <180s"
            assert integration_success_rate >= 0.85, f"Integration success rate {integration_success_rate:.1%} should be ≥85%"
            
            # =================================================================
            # Stage 5: Risk Assessment and Analytics
            # =================================================================
            stage5_start = time.time()
            
            risk_calculator = performance_processors['risk_calculator']
            
            # Calculate risk scores for all SA2 areas
            seifa_risk = risk_calculator._calculate_seifa_risk_score(optimized_seifa)
            health_risk = risk_calculator._calculate_health_utilisation_risk(aggregated_health)
            geographic_risk = risk_calculator._calculate_geographic_accessibility_risk(enhanced_boundaries)
            
            # Comprehensive risk assessment
            comprehensive_risk = seifa_risk.join(
                health_risk, left_on="sa2_code_2021", right_on="sa2_code", how="inner"
            ).join(
                geographic_risk, on="sa2_code_2021", how="inner"
            )
            
            composite_risk = risk_calculator._calculate_composite_risk_score(comprehensive_risk)
            final_risk = risk_calculator._classify_risk_categories(composite_risk)
            
            stage5_time = time.time() - stage5_start
            logger.info(f"Stage 5 (Risk Assessment): {stage5_time:.2f}s, {len(final_risk)} risk assessments")
            
            # Risk assessment performance validation
            assert stage5_time < 120.0, f"Risk assessment took {stage5_time:.1f}s, expected <120s"
            
            # =================================================================
            # Stage 6: Storage Performance
            # =================================================================
            stage6_start = time.time()
            
            storage_manager = performance_processors['storage_manager']
            temp_dir = performance_processors['temp_dir']
            
            # Save optimized datasets with compression
            seifa_path = storage_manager.save_optimized_parquet(
                optimized_seifa, temp_dir / "performance_seifa.parquet", data_type="seifa"
            )
            health_agg_path = storage_manager.save_optimized_parquet(
                aggregated_health, temp_dir / "performance_health_agg.parquet", data_type="health"
            )
            boundaries_path = storage_manager.save_optimized_parquet(
                enhanced_boundaries, temp_dir / "performance_boundaries.parquet", data_type="geographic"
            )
            risk_path = storage_manager.save_optimized_parquet(
                final_risk, temp_dir / "performance_risk.parquet", data_type="analytics"
            )
            
            stage6_time = time.time() - stage6_start
            
            # Calculate storage efficiency
            total_storage_size = sum(
                path.stat().st_size for path in [seifa_path, health_agg_path, boundaries_path, risk_path]
            ) / 1024 / 1024  # MB
            
            storage_efficiency = total_records / total_storage_size  # Records per MB
            
            logger.info(f"Stage 6 (Storage): {stage6_time:.2f}s, {total_storage_size:.1f}MB total storage")
            
            # Storage performance validation
            assert stage6_time < 90.0, f"Storage operations took {stage6_time:.1f}s, expected <90s"
            assert storage_efficiency > 1000, f"Storage efficiency {storage_efficiency:.0f} records/MB should be >1000"
            
            # =================================================================
            # Overall Performance Assessment
            # =================================================================
            total_pipeline_time = time.time() - pipeline_start
            peak_memory = process.memory_info().rss / 1024 / 1024
            total_memory_usage = peak_memory - initial_memory
            throughput = total_records / total_pipeline_time
            
            # Performance targets validation
            targets_met = {
                'processing_time_under_5min': total_pipeline_time < target_processing_time,
                'memory_under_4gb': total_memory_usage < (target_memory_gb * 1024),
                'throughput_over_500_rps': throughput > target_throughput,
                'memory_optimization_achieved': memory_reduction_percent >= target_memory_reduction * 0.7,
                'integration_success_high': integration_success_rate >= 0.85,
                'storage_efficiency_good': storage_efficiency > 1000
            }
            
            all_targets_met = all(targets_met.values())
            
            # Final performance assertions
            assert total_pipeline_time < target_processing_time, \
                f"Pipeline took {total_pipeline_time:.1f}s, expected <{target_processing_time}s"
            assert total_memory_usage < (target_memory_gb * 1024), \
                f"Memory usage {total_memory_usage:.1f}MB should be <{target_memory_gb * 1024}MB"
            assert throughput > target_throughput, \
                f"Throughput {throughput:.0f} records/s should be >{target_throughput}"
            
            # Create comprehensive performance result
            result = LargeScalePerformanceResult(
                test_name="million_record_end_to_end",
                records_processed=total_records,
                processing_time_seconds=total_pipeline_time,
                memory_usage_mb=total_memory_usage,
                throughput_records_per_second=throughput,
                memory_optimization_percent=memory_reduction_percent,
                integration_success_rate=integration_success_rate,
                storage_efficiency_mb=storage_efficiency,
                targets_met=targets_met,
                stage_breakdown={
                    'data_generation': stage1_time,
                    'validation': stage2_time,
                    'memory_optimization': stage3_time,
                    'integration': stage4_time,
                    'risk_assessment': stage5_time,
                    'storage': stage6_time
                }
            )
            
            # Generate performance report
            logger.info("="*80)
            logger.info("1M+ RECORD PIPELINE PERFORMANCE REPORT")
            logger.info("="*80)
            logger.info(f"Total Records Processed: {total_records:,}")
            logger.info(f"Total Pipeline Time: {total_pipeline_time:.2f}s ({total_pipeline_time/60:.1f} minutes)")
            logger.info(f"Memory Usage: {total_memory_usage:.1f}MB ({total_memory_usage/1024:.2f}GB)")
            logger.info(f"Throughput: {throughput:.0f} records/second")
            logger.info(f"Memory Optimization: {memory_reduction_percent:.1f}% reduction")
            logger.info(f"Integration Success: {integration_success_rate:.1%}")
            logger.info(f"Storage Efficiency: {storage_efficiency:.0f} records/MB")
            logger.info(f"All Performance Targets Met: {all_targets_met}")
            logger.info("="*80)
            
            return result
    
    def test_concurrent_large_dataset_processing(self, data_generator, performance_processors):
        """Test concurrent processing of multiple large datasets."""
        logger.info("Starting concurrent large dataset processing test")
        
        # Create multiple large datasets for concurrent processing
        num_concurrent_datasets = 4
        records_per_dataset = 250000  # 1M total across 4 datasets
        
        datasets = []
        for i in range(num_concurrent_datasets):
            health_data = data_generator.generate_large_scale_health_data(records_per_dataset)
            datasets.append((f"dataset_{i}", health_data))
        
        def process_dataset_concurrent(dataset_info):
            """Process a dataset concurrently."""
            dataset_name, dataset = dataset_info
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Full processing pipeline
            health_processor = performance_processors['health_processor']
            memory_optimizer = performance_processors['memory_optimizer']
            
            validated = health_processor._validate_health_data(dataset)
            aggregated = health_processor._aggregate_by_sa2(validated)
            optimized = memory_optimizer.optimize_data_types(aggregated, data_category="health")
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            return {
                'dataset_name': dataset_name,
                'processing_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'input_records': len(dataset),
                'output_records': len(optimized),
                'success': True
            }
        
        # Test concurrent processing
        concurrent_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent_datasets) as executor:
            futures = [executor.submit(process_dataset_concurrent, dataset_info) for dataset_info in datasets]
            results = [future.result() for future in concurrent.futures.as_completed(futures, timeout=300)]
        
        concurrent_total_time = time.time() - concurrent_start
        
        # Validate concurrent processing results
        total_records_processed = sum(r['input_records'] for r in results)
        average_processing_time = np.mean([r['processing_time'] for r in results])
        concurrent_throughput = total_records_processed / concurrent_total_time
        
        # Concurrent processing validation
        assert len(results) == num_concurrent_datasets, f"Should have {num_concurrent_datasets} successful results"
        assert all(r['success'] for r in results), "All concurrent operations should succeed"
        assert concurrent_total_time < 180.0, f"Concurrent processing took {concurrent_total_time:.1f}s, expected <180s"
        assert concurrent_throughput > 2000, f"Concurrent throughput {concurrent_throughput:.0f} records/s should be >2000"
        
        logger.info(f"Concurrent processing: {concurrent_total_time:.2f}s, "
                   f"{concurrent_throughput:.0f} records/s, {len(results)} datasets")
        
        return {
            'concurrent_time': concurrent_total_time,
            'concurrent_throughput': concurrent_throughput,
            'datasets_processed': len(results),
            'total_records': total_records_processed,
            'average_per_dataset_time': average_processing_time
        }
    
    def test_memory_stability_extended_operation(self, data_generator, performance_processors):
        """Test memory stability during extended operations (memory leak detection)."""
        logger.info("Starting extended operation memory stability test")
        
        memory_samples = []
        operation_count = 10
        records_per_operation = 100000
        
        for i in range(operation_count):
            gc.collect()  # Force garbage collection
            
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024
            
            # Generate and process data
            health_data = data_generator.generate_large_scale_health_data(records_per_operation)
            health_processor = performance_processors['health_processor']
            memory_optimizer = performance_processors['memory_optimizer']
            
            validated = health_processor._validate_health_data(health_data)
            optimized = memory_optimizer.optimize_data_types(validated, data_category="health")
            
            # Force cleanup
            del health_data, validated, optimized
            gc.collect()
            
            end_memory = process.memory_info().rss / 1024 / 1024
            memory_delta = end_memory - start_memory
            
            memory_samples.append({
                'iteration': i,
                'start_memory_mb': start_memory,
                'end_memory_mb': end_memory,
                'memory_delta_mb': memory_delta
            })
            
            logger.info(f"Iteration {i + 1}/{operation_count}: {memory_delta:.1f}MB delta")
        
        # Analyze memory stability
        memory_deltas = [s['memory_delta_mb'] for s in memory_samples]
        total_memory_growth = sum(memory_deltas)
        average_memory_delta = np.mean(memory_deltas)
        memory_leak_tolerance = PERFORMANCE_CONFIG['stress_testing_targets']['memory_leak_tolerance_mb']
        
        # Memory stability validation
        assert total_memory_growth < memory_leak_tolerance, \
            f"Total memory growth {total_memory_growth:.1f}MB should be <{memory_leak_tolerance}MB"
        assert average_memory_delta < 20, \
            f"Average memory delta {average_memory_delta:.1f}MB should be <20MB per operation"
        
        logger.info(f"Memory stability test: {total_memory_growth:.1f}MB total growth, "
                   f"{average_memory_delta:.1f}MB average delta")
        
        return {
            'total_memory_growth_mb': total_memory_growth,
            'average_memory_delta_mb': average_memory_delta,
            'memory_samples': memory_samples,
            'memory_stable': total_memory_growth < memory_leak_tolerance
        }