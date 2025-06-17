"""
Data Generators - Phase 5.4

Large-scale data generation utilities for performance testing and load simulation.
Generates realistic Australian health data at scale with proper statistical
distributions, geographic patterns, and temporal correlations for comprehensive
performance testing scenarios.

Key Features:
- Scalable data generation (1M+ records)
- Realistic Australian health patterns
- Geographic distribution accuracy
- Temporal correlation modeling
- Memory-efficient batch generation
"""

import polars as pl
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Iterator
from dataclasses import dataclass
import concurrent.futures
import threading
import time
import gc

from tests.performance import AUSTRALIAN_DATA_SCALE

logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """Configuration for data generation."""
    total_records: int
    batch_size: int
    sa2_areas: int
    geographic_distribution: Dict[str, float]
    temporal_range_days: int
    correlation_strength: float
    memory_limit_mb: int
    enable_caching: bool


@dataclass
class GenerationStatistics:
    """Statistics from data generation process."""
    total_records_generated: int
    generation_time_seconds: float
    memory_usage_mb: float
    batches_processed: int
    throughput_records_per_second: float
    data_quality_score: float


class AustralianHealthDataGenerator:
    """
    High-performance generator for realistic Australian health data at scale.
    Optimized for memory efficiency and realistic statistical distributions.
    """
    
    # Australian demographic and health patterns
    AUSTRALIAN_PATTERNS = {
        'states': {
            'NSW': {'weight': 0.32, 'sa2_range': (10001, 13000), 'health_index': 1.05},
            'VIC': {'weight': 0.26, 'sa2_range': (20001, 23000), 'health_index': 1.08},
            'QLD': {'weight': 0.20, 'sa2_range': (30001, 33000), 'health_index': 0.95},
            'WA': {'weight': 0.11, 'sa2_range': (50001, 52000), 'health_index': 1.02},
            'SA': {'weight': 0.07, 'sa2_range': (40001, 41000), 'health_index': 0.98},
            'TAS': {'weight': 0.02, 'sa2_range': (60001, 60500), 'health_index': 0.92},
            'ACT': {'weight': 0.02, 'sa2_range': (80001, 80100), 'health_index': 1.15},
            'NT': {'weight': 0.01, 'sa2_range': (70001, 70100), 'health_index': 0.88}
        },
        'age_distributions': {
            '0-17': {'weight': 0.24, 'health_multiplier': 0.3},
            '18-34': {'weight': 0.26, 'health_multiplier': 0.5},
            '35-49': {'weight': 0.20, 'health_multiplier': 0.8},
            '50-64': {'weight': 0.18, 'health_multiplier': 1.2},
            '65-79': {'weight': 0.10, 'health_multiplier': 2.0},
            '80+': {'weight': 0.02, 'health_multiplier': 3.5}
        },
        'health_conditions': {
            'cardiovascular': {'prevalence': 0.18, 'age_correlation': 0.8, 'seifa_correlation': -0.4},
            'diabetes': {'prevalence': 0.05, 'age_correlation': 0.7, 'seifa_correlation': -0.5},
            'mental_health': {'prevalence': 0.15, 'age_correlation': -0.3, 'seifa_correlation': -0.3},
            'respiratory': {'prevalence': 0.12, 'age_correlation': 0.6, 'seifa_correlation': -0.2},
            'musculoskeletal': {'prevalence': 0.22, 'age_correlation': 0.9, 'seifa_correlation': -0.1}
        }
    }
    
    def __init__(self, 
                 config: Optional[DataGenerationConfig] = None,
                 seed: int = 42,
                 enable_parallel: bool = True):
        """
        Initialize the data generator.
        
        Args:
            config: Generation configuration
            seed: Random seed for reproducibility
            enable_parallel: Enable parallel generation
        """
        self.config = config or self._default_config()
        self.seed = seed
        self.enable_parallel = enable_parallel
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Pre-generate lookup tables for efficiency
        self.sa2_codes = self._generate_sa2_lookup()
        self.geographic_weights = self._calculate_geographic_weights()
        
        # Generation statistics
        self.stats = GenerationStatistics(
            total_records_generated=0,
            generation_time_seconds=0.0,
            memory_usage_mb=0.0,
            batches_processed=0,
            throughput_records_per_second=0.0,
            data_quality_score=0.0
        )
        
        logger.info(f"Data generator initialized for {self.config.total_records:,} records")
    
    def _default_config(self) -> DataGenerationConfig:
        """Create default generation configuration."""
        return DataGenerationConfig(
            total_records=1000000,
            batch_size=50000,
            sa2_areas=AUSTRALIAN_DATA_SCALE['sa2_areas_total'],
            geographic_distribution=self.AUSTRALIAN_PATTERNS['states'],
            temporal_range_days=365,
            correlation_strength=0.7,
            memory_limit_mb=2048,
            enable_caching=True
        )
    
    def _generate_sa2_lookup(self) -> List[str]:
        """Generate lookup table of SA2 codes."""
        sa2_codes = []
        
        for state, info in self.AUSTRALIAN_PATTERNS['states'].items():
            start, end = info['sa2_range']
            count = int(self.config.sa2_areas * info['weight'])
            codes = [f"{i:08d}" for i in range(start, start + count)]
            sa2_codes.extend(codes)
        
        # Ensure we have exactly the target number of SA2 codes
        while len(sa2_codes) < self.config.sa2_areas:
            sa2_codes.append(f"{len(sa2_codes) + 10000000:08d}")
        
        return sa2_codes[:self.config.sa2_areas]
    
    def _calculate_geographic_weights(self) -> np.ndarray:
        """Calculate geographic distribution weights."""
        weights = []
        for state_info in self.AUSTRALIAN_PATTERNS['states'].values():
            weights.extend([state_info['weight']] * int(self.config.sa2_areas * state_info['weight']))
        
        # Normalize weights
        weights = np.array(weights[:self.config.sa2_areas])
        return weights / weights.sum()
    
    def generate_large_scale_health_data(self, 
                                       batch_callback: Optional[callable] = None) -> Iterator[pl.DataFrame]:
        """
        Generate large-scale health data in batches for memory efficiency.
        
        Args:
            batch_callback: Optional callback function for each batch
            
        Yields:
            pl.DataFrame: Batches of health data
        """
        logger.info(f"Starting generation of {self.config.total_records:,} health records")
        
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        total_batches = (self.config.total_records + self.config.batch_size - 1) // self.config.batch_size
        
        for batch_num in range(total_batches):
            batch_start = batch_num * self.config.batch_size
            batch_end = min(batch_start + self.config.batch_size, self.config.total_records)
            batch_size = batch_end - batch_start
            
            logger.debug(f"Generating batch {batch_num + 1}/{total_batches} ({batch_size:,} records)")
            
            # Generate batch
            batch_data = self._generate_health_batch(batch_size, batch_start)
            
            # Update statistics
            self.stats.batches_processed += 1
            self.stats.total_records_generated += len(batch_data)
            
            # Execute callback if provided
            if batch_callback:
                batch_callback(batch_data, batch_num, total_batches)
            
            yield batch_data
            
            # Force garbage collection to manage memory
            if batch_num % 10 == 0:
                gc.collect()
        
        # Final statistics
        self.stats.generation_time_seconds = time.time() - start_time
        self.stats.memory_usage_mb = self._get_memory_usage() - start_memory
        self.stats.throughput_records_per_second = (
            self.stats.total_records_generated / self.stats.generation_time_seconds
        )
        self.stats.data_quality_score = self._assess_data_quality()
        
        logger.info(f"Generation complete: {self.stats.total_records_generated:,} records in "
                   f"{self.stats.generation_time_seconds:.2f}s "
                   f"({self.stats.throughput_records_per_second:.0f} records/s)")
    
    def _generate_health_batch(self, batch_size: int, batch_offset: int) -> pl.DataFrame:
        """Generate a single batch of health data."""
        
        # Geographic distribution
        sa2_indices = np.random.choice(len(self.sa2_codes), batch_size, replace=True, p=self.geographic_weights)
        sa2_codes = [self.sa2_codes[idx] for idx in sa2_indices]
        states = [self._get_state_from_sa2(code) for code in sa2_codes]
        
        # Demographics with realistic distributions
        age_groups = self._generate_age_distribution(batch_size)
        ages = [self._age_group_to_numeric(age_group) for age_group in age_groups]
        genders = np.random.choice(['M', 'F'], batch_size, p=[0.49, 0.51])
        
        # SEIFA correlation with geographic patterns
        seifa_deciles = self._generate_correlated_seifa(states, batch_size)
        
        # Health utilisation with realistic correlations
        health_metrics = self._generate_health_utilisation(ages, seifa_deciles, states, batch_size)
        
        # Temporal patterns
        temporal_data = self._generate_temporal_patterns(batch_size)
        
        # Chronic conditions with correlations
        chronic_conditions = self._generate_chronic_conditions(ages, seifa_deciles, batch_size)
        
        # Healthcare access and risk scoring
        access_scores = self._generate_access_scores(states, seifa_deciles, batch_size)
        risk_scores = self._generate_risk_scores(ages, seifa_deciles, chronic_conditions, batch_size)
        
        # Create DataFrame
        health_data = pl.DataFrame({
            # Geographic identifiers
            'sa2_code': sa2_codes,
            'sa2_name': [f"Statistical Area {code}" for code in sa2_codes],
            'state_territory': states,
            'postcode': self._generate_postcodes(states, batch_size),
            'remoteness_category': self._generate_remoteness(states, batch_size),
            
            # Demographics
            'age_group': age_groups,
            'age_numeric': ages,
            'gender': genders,
            'usual_resident_population': np.random.randint(100, 25000, batch_size),
            
            # SEIFA indices
            'seifa_irsd_decile': seifa_deciles,
            'seifa_irsad_decile': self._correlate_seifa_indices(seifa_deciles, 0.8),
            'seifa_ier_decile': self._correlate_seifa_indices(seifa_deciles, 0.7),
            'seifa_ieo_decile': self._correlate_seifa_indices(seifa_deciles, 0.6),
            
            # Health utilisation
            'prescription_count': health_metrics['prescriptions'],
            'gp_visits': health_metrics['gp_visits'],
            'specialist_visits': health_metrics['specialist_visits'],
            'total_prescriptions': health_metrics['prescriptions'],
            
            # Costs (Australian Medicare patterns)
            'prescription_cost_aud': health_metrics['prescription_costs'],
            'gp_cost_aud': health_metrics['gp_costs'],
            'specialist_cost_aud': health_metrics['specialist_costs'],
            'total_cost_aud': (health_metrics['prescription_costs'] + 
                              health_metrics['gp_costs'] + 
                              health_metrics['specialist_costs']),
            
            # Health conditions
            'chronic_conditions_count': chronic_conditions,
            'cardiovascular_risk': self._condition_risk('cardiovascular', ages, seifa_deciles),
            'diabetes_risk': self._condition_risk('diabetes', ages, seifa_deciles),
            'mental_health_risk': self._condition_risk('mental_health', ages, seifa_deciles),
            
            # Temporal data
            'service_year': temporal_data['years'],
            'service_month': temporal_data['months'],
            'service_quarter': temporal_data['quarters'],
            'service_date': temporal_data['dates'],
            'data_extraction_date': [datetime.now().strftime('%Y-%m-%d')] * batch_size,
            
            # Calculated metrics
            'healthcare_access_score': access_scores,
            'health_risk_score': risk_scores,
            'socioeconomic_disadvantage_score': 11 - seifa_deciles,  # Inverse of SEIFA
            'geographic_isolation_score': self._geographic_isolation_scores(states, batch_size),
            
            # Quality indicators
            'data_completeness_score': np.random.uniform(0.85, 1.0, batch_size),
            'record_quality_flag': np.random.choice(['high', 'medium', 'low'], batch_size, p=[0.8, 0.15, 0.05]),
            'validation_status': ['validated'] * batch_size
        })
        
        return health_data
    
    def _generate_age_distribution(self, batch_size: int) -> List[str]:
        """Generate realistic age distribution."""
        age_groups = list(self.AUSTRALIAN_PATTERNS['age_distributions'].keys())
        weights = [info['weight'] for info in self.AUSTRALIAN_PATTERNS['age_distributions'].values()]
        
        return np.random.choice(age_groups, batch_size, p=weights).tolist()
    
    def _age_group_to_numeric(self, age_group: str) -> int:
        """Convert age group to numeric age."""
        ranges = {
            '0-17': (0, 17), '18-34': (18, 34), '35-49': (35, 49),
            '50-64': (50, 64), '65-79': (65, 79), '80+': (80, 95)
        }
        min_age, max_age = ranges.get(age_group, (30, 40))
        return np.random.randint(min_age, max_age + 1)
    
    def _generate_correlated_seifa(self, states: List[str], batch_size: int) -> np.ndarray:
        """Generate SEIFA deciles with geographic correlation."""
        seifa_deciles = np.zeros(batch_size, dtype=int)
        
        for i, state in enumerate(states):
            # State-based SEIFA patterns
            state_base = {
                'NSW': 6, 'VIC': 6, 'QLD': 5, 'WA': 6,
                'SA': 5, 'TAS': 4, 'ACT': 8, 'NT': 4
            }.get(state, 5)
            
            # Add random variation around state base
            seifa_deciles[i] = np.clip(
                np.random.normal(state_base, 2.5), 1, 10
            ).astype(int)
        
        return seifa_deciles
    
    def _correlate_seifa_indices(self, base_deciles: np.ndarray, correlation: float) -> np.ndarray:
        """Generate correlated SEIFA indices."""
        noise_strength = 1 - correlation
        noise = np.random.normal(0, 2 * noise_strength, len(base_deciles))
        correlated = base_deciles + noise
        return np.clip(correlated, 1, 10).astype(int)
    
    def _generate_health_utilisation(self, ages: List[int], seifa_deciles: np.ndarray, 
                                   states: List[str], batch_size: int) -> Dict[str, np.ndarray]:
        """Generate realistic health utilisation patterns."""
        
        # Base utilisation with age and SEIFA correlation
        age_factors = np.array([self._age_health_factor(age) for age in ages])
        seifa_factors = (11 - seifa_deciles) / 10  # Higher disadvantage = higher utilisation
        
        # State-based health index
        state_factors = np.array([
            self.AUSTRALIAN_PATTERNS['states'][state]['health_index'] 
            for state in states
        ])
        
        # Generate utilisation with correlations
        base_prescriptions = np.random.poisson(3, batch_size)
        prescriptions = np.maximum(0, 
            (base_prescriptions * age_factors * seifa_factors * state_factors).astype(int)
        )
        
        base_gp_visits = np.random.poisson(6, batch_size)
        gp_visits = np.maximum(0,
            (base_gp_visits * age_factors * seifa_factors * 0.8).astype(int)
        )
        
        base_specialist_visits = np.random.poisson(1.5, batch_size)
        specialist_visits = np.maximum(0,
            (base_specialist_visits * age_factors * (seifa_deciles / 10) * 1.2).astype(int)
        )
        
        # Generate costs with realistic Australian Medicare patterns
        prescription_costs = prescriptions * np.random.exponential(28, batch_size)  # ~$28 avg
        gp_costs = gp_visits * np.random.normal(85, 25, batch_size)  # Medicare rebate ~$37, gap ~$48
        specialist_costs = specialist_visits * np.random.normal(280, 80, batch_size)  # Higher specialist costs
        
        return {
            'prescriptions': prescriptions,
            'gp_visits': gp_visits,
            'specialist_visits': specialist_visits,
            'prescription_costs': np.maximum(0, prescription_costs),
            'gp_costs': np.maximum(0, gp_costs),
            'specialist_costs': np.maximum(0, specialist_costs)
        }
    
    def _age_health_factor(self, age: int) -> float:
        """Calculate health utilisation factor based on age."""
        if age < 18:
            return 0.4
        elif age < 35:
            return 0.6
        elif age < 50:
            return 0.8
        elif age < 65:
            return 1.2
        elif age < 80:
            return 2.0
        else:
            return 3.5
    
    def _generate_temporal_patterns(self, batch_size: int) -> Dict[str, List]:
        """Generate realistic temporal patterns."""
        # Weighted towards recent years
        years = np.random.choice([2021, 2022, 2023], batch_size, p=[0.2, 0.3, 0.5])
        
        # Seasonal patterns for health services
        months = np.random.choice(range(1, 13), batch_size, p=[
            0.09, 0.08, 0.09, 0.08, 0.08, 0.07,  # Jan-Jun (flu season effect)
            0.07, 0.08, 0.08, 0.09, 0.10, 0.09   # Jul-Dec (higher utilisation)
        ])
        
        quarters = [(month - 1) // 3 + 1 for month in months]
        
        # Generate service dates
        dates = []
        for year, month in zip(years, months):
            day = np.random.randint(1, 29)  # Avoid month-end issues
            dates.append(f"{year}-{month:02d}-{day:02d}")
        
        return {
            'years': years.tolist(),
            'months': months.tolist(),
            'quarters': quarters,
            'dates': dates
        }
    
    def _generate_chronic_conditions(self, ages: List[int], seifa_deciles: np.ndarray, 
                                   batch_size: int) -> np.ndarray:
        """Generate chronic condition counts with realistic correlations."""
        chronic_counts = np.zeros(batch_size, dtype=int)
        
        for i, (age, seifa) in enumerate(zip(ages, seifa_deciles)):
            # Age-based base probability
            if age < 18:
                base_prob = 0.05
            elif age < 35:
                base_prob = 0.10
            elif age < 50:
                base_prob = 0.25
            elif age < 65:
                base_prob = 0.45
            elif age < 80:
                base_prob = 0.70
            else:
                base_prob = 0.85
            
            # SEIFA correlation (disadvantage increases conditions)
            seifa_multiplier = (12 - seifa) / 10
            adjusted_prob = base_prob * seifa_multiplier
            
            # Generate number of conditions
            chronic_counts[i] = np.random.poisson(adjusted_prob * 4)  # Scale to reasonable counts
        
        return chronic_counts
    
    def _condition_risk(self, condition: str, ages: List[int], seifa_deciles: np.ndarray) -> np.ndarray:
        """Calculate risk for specific health conditions."""
        condition_info = self.AUSTRALIAN_PATTERNS['health_conditions'][condition]
        
        base_prevalence = condition_info['prevalence']
        age_correlation = condition_info['age_correlation']
        seifa_correlation = condition_info['seifa_correlation']
        
        # Age factors (normalized 0-1)
        age_factors = np.array([(age / 100) ** age_correlation for age in ages])
        
        # SEIFA factors (disadvantage correlation)
        seifa_factors = ((11 - seifa_deciles) / 10) ** abs(seifa_correlation)
        if seifa_correlation > 0:
            seifa_factors = 1 - seifa_factors  # Inverse for positive correlation
        
        # Calculate risk scores
        risk_scores = base_prevalence * age_factors * seifa_factors * 10  # Scale to 0-10
        
        return np.clip(risk_scores, 0, 10)
    
    def _generate_access_scores(self, states: List[str], seifa_deciles: np.ndarray, 
                              batch_size: int) -> np.ndarray:
        """Generate healthcare access scores."""
        access_scores = np.zeros(batch_size)
        
        for i, (state, seifa) in enumerate(zip(states, seifa_deciles)):
            # State-based access (metropolitan vs rural)
            state_access = {
                'NSW': 7.5, 'VIC': 7.8, 'QLD': 6.8, 'WA': 6.5,
                'SA': 6.2, 'TAS': 5.8, 'ACT': 8.5, 'NT': 5.0
            }.get(state, 6.0)
            
            # SEIFA correlation (higher decile = better access)
            seifa_access = seifa / 10 * 3  # 0-3 point contribution
            
            # Random variation
            noise = np.random.normal(0, 0.8)
            
            access_scores[i] = np.clip(state_access + seifa_access + noise, 1, 10)
        
        return access_scores
    
    def _generate_risk_scores(self, ages: List[int], seifa_deciles: np.ndarray,
                            chronic_conditions: np.ndarray, batch_size: int) -> np.ndarray:
        """Generate composite health risk scores."""
        
        # Age component (0-3)
        age_risk = np.array([min(3, age / 30) for age in ages])
        
        # SEIFA component (0-3) - higher disadvantage = higher risk
        seifa_risk = (11 - seifa_deciles) / 10 * 3
        
        # Chronic conditions component (0-4)
        condition_risk = np.minimum(4, chronic_conditions)
        
        # Combine components
        total_risk = age_risk + seifa_risk + condition_risk
        
        # Add random variation and normalize to 1-10 scale
        noise = np.random.normal(0, 0.5, batch_size)
        final_risk = np.clip(total_risk + noise, 1, 10)
        
        return final_risk
    
    def _generate_postcodes(self, states: List[str], batch_size: int) -> List[str]:
        """Generate realistic postcodes by state."""
        postcodes = []
        
        postcode_ranges = {
            'NSW': (2000, 2999), 'VIC': (3000, 3999), 'QLD': (4000, 4999),
            'WA': (6000, 6999), 'SA': (5000, 5999), 'TAS': (7000, 7999),
            'ACT': (2600, 2699), 'NT': (800, 999)
        }
        
        for state in states:
            min_pc, max_pc = postcode_ranges.get(state, (2000, 2999))
            postcode = np.random.randint(min_pc, max_pc + 1)
            postcodes.append(f"{postcode:04d}")
        
        return postcodes
    
    def _generate_remoteness(self, states: List[str], batch_size: int) -> List[str]:
        """Generate remoteness categories by state."""
        remoteness_categories = ['Major Cities', 'Inner Regional', 'Outer Regional', 'Remote', 'Very Remote']
        
        # State-based remoteness distributions
        distributions = {
            'NSW': [0.75, 0.15, 0.08, 0.015, 0.005],
            'VIC': [0.73, 0.18, 0.07, 0.015, 0.005],
            'QLD': [0.68, 0.18, 0.10, 0.03, 0.01],
            'WA': [0.77, 0.10, 0.08, 0.03, 0.02],
            'SA': [0.73, 0.15, 0.08, 0.03, 0.01],
            'TAS': [0.40, 0.35, 0.20, 0.04, 0.01],
            'ACT': [1.0, 0.0, 0.0, 0.0, 0.0],
            'NT': [0.58, 0.18, 0.12, 0.08, 0.04]
        }
        
        remoteness = []
        for state in states:
            probs = distributions.get(state, [0.7, 0.2, 0.08, 0.015, 0.005])
            category = np.random.choice(remoteness_categories, p=probs)
            remoteness.append(category)
        
        return remoteness
    
    def _geographic_isolation_scores(self, states: List[str], batch_size: int) -> np.ndarray:
        """Generate geographic isolation scores."""
        isolation_base = {
            'NSW': 2.5, 'VIC': 2.3, 'QLD': 3.8, 'WA': 4.2,
            'SA': 3.5, 'TAS': 4.8, 'ACT': 1.0, 'NT': 5.5
        }
        
        scores = np.array([isolation_base.get(state, 3.0) for state in states])
        noise = np.random.normal(0, 1.0, batch_size)
        
        return np.clip(scores + noise, 1, 10)
    
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
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def _assess_data_quality(self) -> float:
        """Assess the quality of generated data."""
        # Simple quality assessment - could be expanded
        quality_factors = [
            1.0 if self.stats.total_records_generated > 0 else 0.0,  # Records generated
            1.0 if self.stats.throughput_records_per_second > 1000 else 0.5,  # Performance
            1.0 if self.stats.batches_processed > 0 else 0.0,  # Successful processing
            0.9  # Assume good data distribution (could validate this)
        ]
        
        return np.mean(quality_factors)
    
    def generate_consolidated_dataset(self, output_path: Optional[Path] = None) -> pl.DataFrame:
        """Generate a consolidated dataset from all batches."""
        logger.info("Generating consolidated dataset")
        
        all_data = []
        
        for batch_data in self.generate_large_scale_health_data():
            all_data.append(batch_data)
        
        # Concatenate all batches
        consolidated = pl.concat(all_data)
        
        if output_path:
            consolidated.write_parquet(output_path)
            logger.info(f"Consolidated dataset saved to {output_path}")
        
        return consolidated
    
    def generate_streaming_data(self, 
                              records_per_second: int = 1000,
                              duration_seconds: int = 60) -> Iterator[pl.DataFrame]:
        """
        Generate streaming data for real-time testing.
        
        Args:
            records_per_second: Rate of data generation
            duration_seconds: Duration of streaming
            
        Yields:
            pl.DataFrame: Small batches of streaming data
        """
        logger.info(f"Starting streaming data generation: {records_per_second} records/s for {duration_seconds}s")
        
        start_time = time.time()
        total_records = 0
        
        while (time.time() - start_time) < duration_seconds:
            batch_start_time = time.time()
            
            # Generate small batch
            batch_size = min(records_per_second, 100)  # Cap batch size
            batch_data = self._generate_health_batch(batch_size, total_records)
            
            total_records += batch_size
            yield batch_data
            
            # Control timing to maintain rate
            batch_time = time.time() - batch_start_time
            target_batch_time = batch_size / records_per_second
            
            if batch_time < target_batch_time:
                time.sleep(target_batch_time - batch_time)
        
        logger.info(f"Streaming complete: {total_records:,} records generated")


class ParallelDataGenerator:
    """Parallel data generator for maximum performance."""
    
    def __init__(self, num_workers: int = 4, **generator_kwargs):
        """
        Initialize parallel generator.
        
        Args:
            num_workers: Number of parallel workers
            **generator_kwargs: Arguments for individual generators
        """
        self.num_workers = num_workers
        self.generator_kwargs = generator_kwargs
        
    def generate_parallel_batches(self, 
                                total_records: int,
                                batch_size: int = 50000) -> Iterator[pl.DataFrame]:
        """Generate data using parallel workers."""
        
        total_batches = (total_records + batch_size - 1) // batch_size
        batches_per_worker = total_batches // self.num_workers
        
        def worker_generate(worker_id: int, num_batches: int):
            """Worker function for parallel generation."""
            worker_generator = AustralianHealthDataGenerator(
                seed=42 + worker_id,  # Unique seed per worker
                **self.generator_kwargs
            )
            
            worker_data = []
            for batch_num in range(num_batches):
                batch_data = worker_generator._generate_health_batch(
                    batch_size, 
                    worker_id * batches_per_worker * batch_size + batch_num * batch_size
                )
                worker_data.append(batch_data)
            
            return worker_data
        
        # Execute parallel generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(worker_generate, worker_id, batches_per_worker)
                for worker_id in range(self.num_workers)
            ]
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                worker_batches = future.result()
                for batch in worker_batches:
                    yield batch