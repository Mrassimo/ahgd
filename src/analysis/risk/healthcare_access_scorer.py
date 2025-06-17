"""
Healthcare Access Scorer - Geographic accessibility analysis for Australian health services

Calculates access scores based on distance to healthcare facilities, population density,
and service availability across SA2 areas.

Data Sources:
- SA2 Boundaries: Geographic centroids and areas
- Population Data: Usual resident population by SA2
- Mock Health Service Locations: GP clinics, hospitals, specialists
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HealthcareAccessScorer:
    """
    Calculate healthcare accessibility scores for Australian SA2 areas.
    Considers distance, population density, and service availability.
    """
    
    # Access scoring weights
    ACCESS_WEIGHTS = {
        'distance_to_gp': 0.4,        # Distance to nearest GP
        'population_density': 0.3,     # Population per square km
        'service_availability': 0.3    # Services per capita
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize access scorer with data directory."""
        self.data_dir = data_dir or Path("data/processed")
        self.boundary_data: Optional[pl.DataFrame] = None
        
    def load_boundary_data(self) -> bool:
        """Load SA2 boundary and population data."""
        try:
            boundary_path = self.data_dir / "sa2_boundaries_processed.csv"
            if boundary_path.exists():
                self.boundary_data = pl.read_csv(boundary_path)
                logger.info(f"Loaded boundary data: {self.boundary_data.shape[0]} SA2 areas")
                return True
            else:
                logger.warning(f"Boundary data not found at {boundary_path}")
                self.boundary_data = self._generate_mock_boundary_data()
                return True
                
        except Exception as e:
            logger.error(f"Failed to load boundary data: {e}")
            return False
    
    def _generate_mock_boundary_data(self) -> pl.DataFrame:
        """Generate mock boundary data for development."""
        np.random.seed(42)
        n_areas = 2000
        
        return pl.DataFrame({
            'sa2_code': [f"1{str(i).zfill(8)}" for i in range(1000, 1000 + n_areas)],
            'sa2_name': [f"Mock Area {i}" for i in range(n_areas)],
            'state_name': np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], n_areas),
            'usual_resident_population': np.random.randint(100, 8000, n_areas),
            'area_sqkm': np.random.exponential(50, n_areas) + 1,  # Avoid zero area
            'centroid_lat': np.random.uniform(-43, -10, n_areas),  # Australian latitude range
            'centroid_lon': np.random.uniform(113, 154, n_areas)   # Australian longitude range
        })
    
    def calculate_population_density_score(self) -> pl.DataFrame:
        """Calculate population density-based access scores."""
        if self.boundary_data is None:
            self.load_boundary_data()
            
        # Calculate population density
        density_data = self.boundary_data.with_columns([
            (pl.col('usual_resident_population') / pl.col('area_sqkm')).alias('population_density_per_sqkm')
        ])
        
        # Score based on density - moderate density = better access
        # Very low density = poor access (rural), very high density = poor access (overcrowding)
        density_data = density_data.with_columns([
            pl.when(pl.col('population_density_per_sqkm') < 1.0)
            .then(pl.lit(2))  # Rural - poor access
            .when(pl.col('population_density_per_sqkm') < 10.0)
            .then(pl.lit(6))  # Low density - moderate access
            .when(pl.col('population_density_per_sqkm') < 100.0)
            .then(pl.lit(8))  # Medium density - good access
            .when(pl.col('population_density_per_sqkm') < 1000.0)
            .then(pl.lit(7))  # High density - good access but some pressure
            .otherwise(pl.lit(4))  # Very high density - poor access due to overcrowding
            .alias('density_access_score')
        ])
        
        return density_data.select(['sa2_code', 'sa2_name', 'state_name', 'population_density_per_sqkm', 'density_access_score'])
    
    def calculate_geographic_access_score(self) -> pl.DataFrame:
        """Calculate geographic accessibility based on location characteristics."""
        if self.boundary_data is None:
            self.load_boundary_data()
            
        # Mock distance calculations - in real implementation would use actual service locations
        np.random.seed(42)  # Reproducible distances
        
        geo_access = self.boundary_data.with_columns([
            # Mock distance to nearest GP (km) - influenced by state and population
            pl.when(pl.col('state_name').is_in(['NSW', 'VIC']))
            .then(np.random.exponential(5, self.boundary_data.shape[0]))  # Better access in populous states
            .when(pl.col('state_name').is_in(['NT', 'WA']))
            .then(np.random.exponential(25, self.boundary_data.shape[0]))  # Poorer access in remote states
            .otherwise(np.random.exponential(12, self.boundary_data.shape[0]))
            .alias('distance_to_nearest_gp_km'),
            
            # Mock distance to nearest hospital
            pl.when(pl.col('usual_resident_population') > 5000)
            .then(np.random.exponential(8, self.boundary_data.shape[0]))   # Urban areas
            .otherwise(np.random.exponential(35, self.boundary_data.shape[0]))  # Rural areas
            .alias('distance_to_nearest_hospital_km')
        ])
        
        # Convert distances to access scores (1-10 scale, lower distance = higher score)
        geo_access = geo_access.with_columns([
            # GP access score
            pl.when(pl.col('distance_to_nearest_gp_km') <= 2)
            .then(pl.lit(10))  # Excellent access
            .when(pl.col('distance_to_nearest_gp_km') <= 5)
            .then(pl.lit(8))   # Good access
            .when(pl.col('distance_to_nearest_gp_km') <= 15)
            .then(pl.lit(6))   # Moderate access
            .when(pl.col('distance_to_nearest_gp_km') <= 30)
            .then(pl.lit(4))   # Poor access
            .otherwise(pl.lit(2))  # Very poor access
            .alias('gp_access_score'),
            
            # Hospital access score
            pl.when(pl.col('distance_to_nearest_hospital_km') <= 5)
            .then(pl.lit(10))
            .when(pl.col('distance_to_nearest_hospital_km') <= 15)
            .then(pl.lit(8))
            .when(pl.col('distance_to_nearest_hospital_km') <= 40)
            .then(pl.lit(6))
            .when(pl.col('distance_to_nearest_hospital_km') <= 80)
            .then(pl.lit(4))
            .otherwise(pl.lit(2))
            .alias('hospital_access_score')
        ])
        
        return geo_access.select(['sa2_code', 'distance_to_nearest_gp_km', 'distance_to_nearest_hospital_km', 
                                 'gp_access_score', 'hospital_access_score'])
    
    def calculate_service_availability_score(self) -> pl.DataFrame:
        """Calculate service availability based on population and state characteristics."""
        if self.boundary_data is None:
            self.load_boundary_data()
            
        # Mock service ratios based on population and state
        service_data = self.boundary_data.with_columns([
            # GPs per 1000 population (Australian average ~1.0-1.5)
            pl.when(pl.col('state_name') == 'ACT')
            .then(pl.lit(1.8))  # Best GP ratio
            .when(pl.col('state_name').is_in(['NSW', 'VIC']))
            .then(pl.lit(1.3))  # Good GP ratio
            .when(pl.col('state_name').is_in(['NT', 'TAS']))
            .then(pl.lit(0.8))  # Poor GP ratio
            .otherwise(pl.lit(1.1))  # Average GP ratio
            .alias('gps_per_1000_population'),
            
            # Specialists per 10000 population
            pl.when(pl.col('usual_resident_population') > 3000)
            .then(pl.lit(15.0))  # Urban areas
            .when(pl.col('usual_resident_population') > 1000)
            .then(pl.lit(8.0))   # Regional areas
            .otherwise(pl.lit(3.0))  # Rural areas
            .alias('specialists_per_10000_population')
        ])
        
        # Convert ratios to access scores
        service_data = service_data.with_columns([
            # GP availability score
            pl.when(pl.col('gps_per_1000_population') >= 1.5)
            .then(pl.lit(10))
            .when(pl.col('gps_per_1000_population') >= 1.2)
            .then(pl.lit(8))
            .when(pl.col('gps_per_1000_population') >= 1.0)
            .then(pl.lit(6))
            .when(pl.col('gps_per_1000_population') >= 0.8)
            .then(pl.lit(4))
            .otherwise(pl.lit(2))
            .alias('gp_availability_score'),
            
            # Specialist availability score
            pl.when(pl.col('specialists_per_10000_population') >= 12)
            .then(pl.lit(10))
            .when(pl.col('specialists_per_10000_population') >= 8)
            .then(pl.lit(8))
            .when(pl.col('specialists_per_10000_population') >= 5)
            .then(pl.lit(6))
            .when(pl.col('specialists_per_10000_population') >= 3)
            .then(pl.lit(4))
            .otherwise(pl.lit(2))
            .alias('specialist_availability_score')
        ])
        
        return service_data.select(['sa2_code', 'gps_per_1000_population', 'specialists_per_10000_population',
                                   'gp_availability_score', 'specialist_availability_score'])
    
    def calculate_composite_access_score(self) -> pl.DataFrame:
        """Calculate composite healthcare access score combining all factors."""
        density_scores = self.calculate_population_density_score()
        geographic_scores = self.calculate_geographic_access_score()
        service_scores = self.calculate_service_availability_score()
        
        # Join all scoring components
        composite = density_scores.join(geographic_scores, on='sa2_code').join(service_scores, on='sa2_code')
        
        # Calculate weighted composite score
        composite = composite.with_columns([
            (
                pl.col('gp_access_score') * 0.3 +
                pl.col('hospital_access_score') * 0.2 +
                pl.col('density_access_score') * 0.2 +
                pl.col('gp_availability_score') * 0.2 +
                pl.col('specialist_availability_score') * 0.1
            ).alias('composite_access_score')
        ])
        
        # Add access categories
        composite = composite.with_columns([
            pl.when(pl.col('composite_access_score') >= 8.0)
            .then(pl.lit('Excellent Access'))
            .when(pl.col('composite_access_score') >= 6.5)
            .then(pl.lit('Good Access'))
            .when(pl.col('composite_access_score') >= 5.0)
            .then(pl.lit('Moderate Access'))
            .when(pl.col('composite_access_score') >= 3.0)
            .then(pl.lit('Poor Access'))
            .otherwise(pl.lit('Very Poor Access'))
            .alias('access_category')
        ])
        
        return composite
    
    def generate_access_summary(self, access_data: pl.DataFrame) -> Dict[str, any]:
        """Generate summary statistics for healthcare access assessment."""
        summary = {
            'total_sa2_areas': access_data.shape[0],
            'average_access_score': float(access_data['composite_access_score'].mean()),
            'excellent_access_areas': int(access_data.filter(pl.col('access_category') == 'Excellent Access').shape[0]),
            'poor_access_areas': int(access_data.filter(pl.col('composite_access_score') <= 3.0).shape[0]),
            'access_distribution': {
                'excellent': int(access_data.filter(pl.col('access_category') == 'Excellent Access').shape[0]),
                'good': int(access_data.filter(pl.col('access_category') == 'Good Access').shape[0]),
                'moderate': int(access_data.filter(pl.col('access_category') == 'Moderate Access').shape[0]),
                'poor': int(access_data.filter(pl.col('access_category') == 'Poor Access').shape[0]),
                'very_poor': int(access_data.filter(pl.col('access_category') == 'Very Poor Access').shape[0])
            },
            'state_access_averages': {
                state: float(avg) for state, avg in 
                access_data.group_by('state_name').agg([
                    pl.col('composite_access_score').mean().alias('avg_access')
                ]).select(['state_name', 'avg_access']).iter_rows()
            }
        }
        
        return summary
    
    def export_access_assessment(self, output_path: Path) -> bool:
        """Export healthcare access assessment results."""
        try:
            access_data = self.calculate_composite_access_score()
            
            # Export to CSV
            csv_path = output_path / "healthcare_access_assessment.csv"
            access_data.write_csv(csv_path)
            
            # Export to Parquet
            parquet_path = output_path / "healthcare_access_assessment.parquet"
            access_data.write_parquet(parquet_path)
            
            # Export summary
            summary = self.generate_access_summary(access_data)
            summary_path = output_path / "access_assessment_summary.json"
            
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Access assessment exported to {output_path}")
            logger.info(f"Summary: {summary['total_sa2_areas']} areas, avg access: {summary['average_access_score']:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export access assessment: {e}")
            return False
    
    def process_complete_access_pipeline(self) -> bool:
        """Execute complete healthcare access assessment pipeline."""
        try:
            logger.info("Starting healthcare access assessment pipeline...")
            
            # Load data
            if not self.load_boundary_data():
                logger.error("Failed to load boundary data")
                return False
            
            # Calculate access scores
            access_data = self.calculate_composite_access_score()
            logger.info(f"Calculated access scores for {access_data.shape[0]} SA2 areas")
            
            # Generate summary
            summary = self.generate_access_summary(access_data)
            logger.info(f"Access assessment complete - {summary['poor_access_areas']} poor access areas identified")
            
            # Export results
            output_dir = Path("data/outputs/access_assessment")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.export_access_assessment(output_dir):
                logger.info("Healthcare access assessment pipeline completed successfully")
                return True
            else:
                logger.error("Failed to export access assessment")
                return False
                
        except Exception as e:
            logger.error(f"Healthcare access assessment pipeline failed: {e}")
            return False


if __name__ == "__main__":
    # Development testing
    scorer = HealthcareAccessScorer()
    success = scorer.process_complete_access_pipeline()
    
    if success:
        print("✅ Healthcare access assessment completed successfully")
    else:
        print("❌ Healthcare access assessment failed")