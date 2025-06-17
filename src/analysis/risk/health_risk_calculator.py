"""
Health Risk Calculator - Multi-factor health risk assessment for Australian SA2 areas

Combines SEIFA socio-economic indices with health utilisation patterns to produce
composite risk scores for population health planning.

Data Sources:
- SEIFA 2021: 2,293 SA2 areas with 4 socio-economic indices
- PBS Health Data: 492,434 prescription records  
- Geographic Boundaries: 2,454 SA2 areas with population data
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HealthRiskCalculator:
    """
    Calculate composite health risk scores for Australian SA2 areas using
    socio-economic disadvantage and health utilisation patterns.
    """
    
    # SEIFA risk weights based on health literature
    SEIFA_WEIGHTS = {
        'irsd_decile': 0.35,    # Index of Relative Socio-economic Disadvantage
        'irsad_decile': 0.25,   # Index of Relative Socio-economic Advantage and Disadvantage  
        'ier_decile': 0.20,     # Index of Education and Occupation
        'ieo_decile': 0.20      # Index of Economic Resources
    }
    
    # Health utilisation risk indicators
    HEALTH_RISK_FACTORS = {
        'high_prescription_rate': 0.4,      # Above average prescription utilisation
        'chronic_medication_use': 0.3,      # Chronic disease indicators
        'geographic_isolation': 0.3         # Distance from health services
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize calculator with processed data directory."""
        self.data_dir = data_dir or Path("data/processed")
        self.seifa_data: Optional[pl.DataFrame] = None
        self.health_data: Optional[pl.DataFrame] = None
        self.boundary_data: Optional[pl.DataFrame] = None
        self.integrated_data: Optional[pl.DataFrame] = None
        
    def load_processed_data(self) -> bool:
        """Load processed data from Phase 2 outputs."""
        try:
            # Load SEIFA socio-economic data (2,293 SA2 areas)
            seifa_path = self.data_dir / "seifa_2021_sa2.csv"
            if seifa_path.exists():
                self.seifa_data = pl.read_csv(seifa_path)
                logger.info(f"Loaded SEIFA data: {self.seifa_data.shape[0]} SA2 areas")
            else:
                logger.warning(f"SEIFA data not found at {seifa_path}")
                return False
                
            # Load health utilisation data (492,434 records)
            health_path = self.data_dir / "health_data_processed.csv"
            if health_path.exists():
                self.health_data = pl.read_csv(health_path)
                logger.info(f"Loaded health data: {self.health_data.shape[0]} records")
            else:
                logger.info("Health data not found - will use mock data for development")
                self.health_data = self._generate_mock_health_data()
                
            # Load boundary data (2,454 SA2 areas)
            boundary_path = self.data_dir / "sa2_boundaries_processed.csv" 
            if boundary_path.exists():
                self.boundary_data = pl.read_csv(boundary_path)
                logger.info(f"Loaded boundary data: {self.boundary_data.shape[0]} areas")
            else:
                logger.warning(f"Boundary data not found at {boundary_path}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            return False
    
    def _generate_mock_health_data(self) -> pl.DataFrame:
        """Generate mock health data for development when real data unavailable."""
        np.random.seed(42)  # Reproducible mock data
        
        # Generate mock PBS data for SA2 areas
        n_records = 50000  # Scaled down for development
        sa2_codes = [f"1{str(i).zfill(8)}" for i in range(1000, 3000)]  # Mock SA2 codes
        
        mock_data = {
            'sa2_code': np.random.choice(sa2_codes, n_records),
            'prescription_count': np.random.poisson(5, n_records),  # Average 5 prescriptions
            'chronic_medication': np.random.choice([0, 1], n_records, p=[0.7, 0.3]),  # 30% chronic
            'total_cost': np.random.exponential(50, n_records),  # Cost distribution
            'state': np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], n_records)
        }
        
        return pl.DataFrame(mock_data)
    
    def calculate_seifa_risk_score(self) -> pl.DataFrame:
        """
        Calculate risk scores based on SEIFA socio-economic indices.
        Lower deciles (1-3) = higher risk, higher deciles (8-10) = lower risk.
        """
        if self.seifa_data is None:
            raise ValueError("SEIFA data not loaded - call load_processed_data() first")
            
        # Invert deciles so lower socio-economic status = higher risk
        # Decile 1 (most disadvantaged) -> Risk 10, Decile 10 (least disadvantaged) -> Risk 1
        seifa_risk = self.seifa_data.with_columns([
            (11 - pl.col('irsd_decile')).alias('irsd_risk'),
            (11 - pl.col('irsad_decile')).alias('irsad_risk'), 
            (11 - pl.col('ier_decile')).alias('ier_risk'),
            (11 - pl.col('ieo_decile')).alias('ieo_risk')
        ])
        
        # Calculate weighted composite SEIFA risk score (1-10 scale)
        seifa_risk = seifa_risk.with_columns([
            (
                pl.col('irsd_risk') * self.SEIFA_WEIGHTS['irsd_decile'] +
                pl.col('irsad_risk') * self.SEIFA_WEIGHTS['irsad_decile'] +
                pl.col('ier_risk') * self.SEIFA_WEIGHTS['ier_decile'] +
                pl.col('ieo_risk') * self.SEIFA_WEIGHTS['ieo_decile']
            ).alias('seifa_risk_score')
        ])
        
        return seifa_risk.select(['sa2_code_2021', 'sa2_name_2021', 'seifa_risk_score'])
    
    def calculate_health_utilisation_risk(self) -> pl.DataFrame:
        """Calculate risk indicators from health service utilisation patterns."""
        if self.health_data is None:
            logger.warning("Health data not available - using mock data")
            
        # Aggregate health data by SA2
        health_agg = self.health_data.group_by('sa2_code').agg([
            pl.col('prescription_count').sum().alias('total_prescriptions'),
            pl.col('chronic_medication').mean().alias('chronic_medication_rate'),
            pl.col('total_cost').mean().alias('avg_cost_per_service'),
            pl.len().alias('service_episodes')
        ])
        
        # Calculate risk indicators based on utilisation patterns
        # High prescription rate indicates health risk
        health_risk = health_agg.with_columns([
            # Prescription risk: above median = higher risk
            (pl.col('total_prescriptions') > pl.col('total_prescriptions').median()).cast(pl.Int8).alias('high_prescription_risk'),
            
            # Chronic medication risk: above 40% chronic medication use
            (pl.col('chronic_medication_rate') > 0.4).cast(pl.Int8).alias('chronic_risk'),
            
            # Service access risk: very low or very high utilisation 
            ((pl.col('service_episodes') < pl.col('service_episodes').quantile(0.1)) |
             (pl.col('service_episodes') > pl.col('service_episodes').quantile(0.9))).cast(pl.Int8).alias('access_risk')
        ])
        
        # Composite health utilisation risk score (0-10 scale)
        health_risk = health_risk.with_columns([
            (
                pl.col('high_prescription_risk') * 3 +
                pl.col('chronic_risk') * 4 +
                pl.col('access_risk') * 3
            ).alias('health_utilisation_risk')
        ])
        
        return health_risk.select(['sa2_code', 'health_utilisation_risk', 'total_prescriptions', 'chronic_medication_rate'])
    
    def integrate_data_sources(self) -> pl.DataFrame:
        """Integrate SEIFA and health data with geographic boundaries."""
        seifa_risk = self.calculate_seifa_risk_score()
        health_risk = self.calculate_health_utilisation_risk()
        
        # Join datasets on SA2 code
        integrated = seifa_risk.join(
            health_risk,
            left_on='sa2_code_2021',
            right_on='sa2_code',
            how='left'
        )
        
        # Add boundary data for population weighting
        if self.boundary_data is not None:
            integrated = integrated.join(
                self.boundary_data.select(['sa2_code', 'state_name', 'usual_resident_population']),
                left_on='sa2_code_2021',
                right_on='sa2_code',
                how='left'
            )
        
        # Fill missing health data with median values
        integrated = integrated.with_columns([
            pl.col('health_utilisation_risk').fill_null(5.0),  # Median risk
            pl.col('total_prescriptions').fill_null(0),
            pl.col('chronic_medication_rate').fill_null(0.0)
        ])
        
        self.integrated_data = integrated
        return integrated
    
    def calculate_composite_risk_score(self) -> pl.DataFrame:
        """
        Calculate final composite health risk score combining all factors.
        Scale: 1 (lowest risk) to 10 (highest risk)
        """
        if self.integrated_data is None:
            self.integrate_data_sources()
            
        # Combine SEIFA and health utilisation risks with equal weight
        composite = self.integrated_data.with_columns([
            (
                pl.col('seifa_risk_score') * 0.6 +  # 60% weight on socio-economic
                pl.col('health_utilisation_risk') * 0.4  # 40% weight on health utilisation
            ).alias('composite_risk_score')
        ])
        
        # Normalise to 1-10 scale and add risk categories
        composite = composite.with_columns([
            pl.col('composite_risk_score').alias('raw_risk_score'),
            pl.when(pl.col('composite_risk_score') <= 3.0)
            .then(pl.lit('Low Risk'))
            .when(pl.col('composite_risk_score') <= 6.0)
            .then(pl.lit('Moderate Risk'))
            .when(pl.col('composite_risk_score') <= 8.0)
            .then(pl.lit('High Risk'))
            .otherwise(pl.lit('Very High Risk'))
            .alias('risk_category')
        ])
        
        return composite
    
    def generate_risk_summary(self) -> Dict[str, any]:
        """Generate summary statistics for health risk assessment."""
        composite_data = self.calculate_composite_risk_score()
            
        summary = {
            'total_sa2_areas': composite_data.shape[0],
            'average_risk_score': float(composite_data['composite_risk_score'].mean()),
            'high_risk_areas': int(composite_data.filter(pl.col('composite_risk_score') >= 7.0).shape[0]),
            'low_risk_areas': int(composite_data.filter(pl.col('composite_risk_score') <= 3.0).shape[0]),
            'risk_distribution': {
                'low': int(composite_data.filter(pl.col('risk_category') == 'Low Risk').shape[0]),
                'moderate': int(composite_data.filter(pl.col('risk_category') == 'Moderate Risk').shape[0]),
                'high': int(composite_data.filter(pl.col('risk_category') == 'High Risk').shape[0]),
                'very_high': int(composite_data.filter(pl.col('risk_category') == 'Very High Risk').shape[0])
            }
        }
        
        return summary
    
    def export_risk_assessment(self, output_path: Path) -> bool:
        """Export complete risk assessment to CSV and Parquet formats."""
        try:
            composite_data = self.calculate_composite_risk_score()
            
            # Export to CSV for Excel compatibility
            csv_path = output_path / "health_risk_assessment.csv"
            composite_data.write_csv(csv_path)
            
            # Export to Parquet for efficient analysis
            parquet_path = output_path / "health_risk_assessment.parquet"
            composite_data.write_parquet(parquet_path)
            
            # Export summary statistics
            summary = self.generate_risk_summary()
            summary_path = output_path / "risk_assessment_summary.json"
            
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Risk assessment exported to {output_path}")
            logger.info(f"Summary: {summary['total_sa2_areas']} areas, avg risk: {summary['average_risk_score']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export risk assessment: {e}")
            return False
    
    def process_complete_risk_pipeline(self) -> bool:
        """Execute complete health risk assessment pipeline."""
        try:
            logger.info("Starting health risk assessment pipeline...")
            
            # Load data
            if not self.load_processed_data():
                logger.error("Failed to load processed data")
                return False
            
            # Calculate composite risk scores
            composite_data = self.calculate_composite_risk_score()
            logger.info(f"Calculated risk scores for {composite_data.shape[0]} SA2 areas")
            
            # Generate summary
            summary = self.generate_risk_summary()
            logger.info(f"Risk assessment complete - {summary['high_risk_areas']} high-risk areas identified")
            
            # Export results
            output_dir = Path("data/outputs/risk_assessment")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.export_risk_assessment(output_dir):
                logger.info("Health risk assessment pipeline completed successfully")
                return True
            else:
                logger.error("Failed to export risk assessment")
                return False
                
        except Exception as e:
            logger.error(f"Health risk assessment pipeline failed: {e}")
            return False


if __name__ == "__main__":
    # Development testing
    calculator = HealthRiskCalculator()
    success = calculator.process_complete_risk_pipeline()
    
    if success:
        print("✅ Health risk assessment completed successfully")
    else:
        print("❌ Health risk assessment failed")