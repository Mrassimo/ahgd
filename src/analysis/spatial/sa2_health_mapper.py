"""
SA2 Health Mapper - Geographic health outcome mapping for Australian Statistical Areas

Maps health outcomes, risk scores, and service accessibility across SA2 geographic boundaries.
Integrates with SEIFA socio-economic data and health utilisation patterns.

Data Sources:
- SA2 Geographic Boundaries: Spatial geometry and centroids
- SEIFA Socio-Economic Data: Risk indicators by area
- Health Service Data: Utilisation and access patterns
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SA2HealthMapper:
    """
    Map health outcomes and risk indicators across Australian SA2 geographic areas.
    Provides spatial analysis and geographic health pattern identification.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize mapper with processed data directory."""
        self.data_dir = data_dir or Path("data/processed")
        self.boundary_data: Optional[pl.DataFrame] = None
        self.health_risk_data: Optional[pl.DataFrame] = None
        self.access_data: Optional[pl.DataFrame] = None
        
    def load_spatial_data(self) -> bool:
        """Load spatial and health data for mapping."""
        try:
            # Load boundary data
            boundary_path = self.data_dir / "sa2_boundaries_processed.csv"
            if boundary_path.exists():
                self.boundary_data = pl.read_csv(boundary_path)
                logger.info(f"Loaded boundary data: {self.boundary_data.shape[0]} SA2 areas")
            else:
                logger.warning("Boundary data not found - creating mock data")
                self.boundary_data = self._generate_mock_boundary_data()
                
            # Try to load health risk data if available
            risk_path = Path("data/outputs/risk_assessment/health_risk_assessment.csv")
            if risk_path.exists():
                self.health_risk_data = pl.read_csv(risk_path)
                logger.info(f"Loaded health risk data: {self.health_risk_data.shape[0]} areas")
            
            # Try to load access data if available
            access_path = Path("data/outputs/access_assessment/healthcare_access_assessment.csv")
            if access_path.exists():
                self.access_data = pl.read_csv(access_path)
                logger.info(f"Loaded access data: {self.access_data.shape[0]} areas")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load spatial data: {e}")
            return False
    
    def _generate_mock_boundary_data(self) -> pl.DataFrame:
        """Generate mock boundary data for development."""
        np.random.seed(42)
        n_areas = 1500
        
        return pl.DataFrame({
            'sa2_code': [f"1{str(i).zfill(8)}" for i in range(1000, 1000 + n_areas)],
            'sa2_name': [f"Mock Area {i}" for i in range(n_areas)],
            'state_name': np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], n_areas),
            'usual_resident_population': np.random.randint(100, 8000, n_areas),
            'area_sqkm': np.random.exponential(50, n_areas) + 1,
            'centroid_lat': np.random.uniform(-43, -10, n_areas),
            'centroid_lon': np.random.uniform(113, 154, n_areas)
        })
    
    def create_population_density_map(self) -> pl.DataFrame:
        """Create population density mapping for SA2 areas."""
        if self.boundary_data is None:
            self.load_spatial_data()
            
        density_map = self.boundary_data.with_columns([
            (pl.col('usual_resident_population') / pl.col('area_sqkm')).alias('population_density_per_sqkm')
        ])
        
        # Add density categories
        density_map = density_map.with_columns([
            pl.when(pl.col('population_density_per_sqkm') < 1.0)
            .then(pl.lit('Very Low Density'))
            .when(pl.col('population_density_per_sqkm') < 10.0)
            .then(pl.lit('Low Density'))
            .when(pl.col('population_density_per_sqkm') < 100.0)
            .then(pl.lit('Medium Density'))
            .when(pl.col('population_density_per_sqkm') < 1000.0)
            .then(pl.lit('High Density'))
            .otherwise(pl.lit('Very High Density'))
            .alias('density_category')
        ])
        
        return density_map
    
    def create_health_risk_map(self) -> Optional[pl.DataFrame]:
        """Create health risk mapping across SA2 areas."""
        if self.health_risk_data is None:
            logger.warning("Health risk data not available for mapping")
            return None
            
        if self.boundary_data is None:
            self.load_spatial_data()
            
        # Join health risk data with spatial boundaries
        risk_map = self.boundary_data.join(
            self.health_risk_data.select(['sa2_code_2021', 'composite_risk_score', 'risk_category']),
            left_on='sa2_code',
            right_on='sa2_code_2021',
            how='left'
        )
        
        return risk_map
    
    def create_access_map(self) -> Optional[pl.DataFrame]:
        """Create healthcare access mapping across SA2 areas."""
        if self.access_data is None:
            logger.warning("Healthcare access data not available for mapping")
            return None
            
        if self.boundary_data is None:
            self.load_spatial_data()
            
        # Join access data with spatial boundaries
        access_map = self.boundary_data.join(
            self.access_data.select(['sa2_code', 'composite_access_score', 'access_category']),
            on='sa2_code',
            how='left'
        )
        
        return access_map
    
    def identify_health_hotspots(self, risk_threshold: float = 7.0) -> pl.DataFrame:
        """Identify geographic health hotspots (high risk areas)."""
        risk_map = self.create_health_risk_map()
        
        if risk_map is None:
            logger.warning("Cannot identify hotspots - risk data not available")
            return pl.DataFrame()
            
        # Filter high-risk areas
        hotspots = risk_map.filter(
            pl.col('composite_risk_score') >= risk_threshold
        ).sort('composite_risk_score', descending=True)
        
        # Add hotspot ranking
        hotspots = hotspots.with_columns([
            pl.int_range(pl.len()).alias('hotspot_rank') + 1
        ])
        
        return hotspots
    
    def identify_access_deserts(self, access_threshold: float = 4.0) -> pl.DataFrame:
        """Identify healthcare access deserts (poor access areas)."""
        access_map = self.create_access_map()
        
        if access_map is None:
            logger.warning("Cannot identify access deserts - access data not available")
            return pl.DataFrame()
            
        # Filter poor access areas
        deserts = access_map.filter(
            pl.col('composite_access_score') <= access_threshold
        ).sort('composite_access_score')
        
        # Add desert ranking (worst access first)
        deserts = deserts.with_columns([
            pl.int_range(pl.len()).alias('desert_rank') + 1
        ])
        
        return deserts
    
    def create_state_level_summary(self) -> pl.DataFrame:
        """Create state-level health summary statistics."""
        if self.boundary_data is None:
            self.load_spatial_data()
            
        # Base state summary from population data
        state_summary = self.boundary_data.group_by('state_name').agg([
            pl.len().alias('total_sa2_areas'),
            pl.col('usual_resident_population').sum().alias('total_population'),
            pl.col('area_sqkm').sum().alias('total_area_sqkm'),
            (pl.col('usual_resident_population').sum() / pl.col('area_sqkm').sum()).alias('state_population_density')
        ])
        
        # Add health risk summary if available
        if self.health_risk_data is not None:
            # Create a mapping from sa2_code_2021 to state_name
            sa2_to_state = self.boundary_data.select(['sa2_code', 'state_name'])
            
            risk_by_state = self.health_risk_data.join(
                sa2_to_state,
                left_on='sa2_code_2021',
                right_on='sa2_code',
                how='left'
            ).group_by('state_name').agg([
                pl.col('composite_risk_score').mean().alias('avg_health_risk'),
                (pl.col('composite_risk_score') >= 7.0).sum().alias('high_risk_areas'),
                (pl.col('risk_category') == 'Very High Risk').sum().alias('very_high_risk_areas')
            ])
            
            state_summary = state_summary.join(risk_by_state, on='state_name', how='left')
        
        # Add access summary if available
        if self.access_data is not None:
            access_by_state = self.access_data.join(
                self.boundary_data.select(['sa2_code', 'state_name']),
                on='sa2_code',
                how='left'
            ).group_by('state_name').agg([
                pl.col('composite_access_score').mean().alias('avg_access_score'),
                (pl.col('composite_access_score') <= 4.0).sum().alias('poor_access_areas'),
                (pl.col('access_category') == 'Very Poor Access').sum().alias('very_poor_access_areas')
            ])
            
            state_summary = state_summary.join(access_by_state, on='state_name', how='left')
        
        return state_summary.sort('total_population', descending=True)
    
    def export_mapping_data(self, output_path: Path) -> bool:
        """Export mapping data in multiple formats for visualization."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export population density map
            density_map = self.create_population_density_map()
            density_map.write_csv(output_path / "population_density_map.csv")
            
            # Export health risk map if available
            risk_map = self.create_health_risk_map()
            if risk_map is not None:
                risk_map.write_csv(output_path / "health_risk_map.csv")
                
                # Export hotspots
                hotspots = self.identify_health_hotspots()
                if hotspots.shape[0] > 0:
                    hotspots.write_csv(output_path / "health_hotspots.csv")
            
            # Export access map if available
            access_map = self.create_access_map()
            if access_map is not None:
                access_map.write_csv(output_path / "healthcare_access_map.csv")
                
                # Export access deserts
                deserts = self.identify_access_deserts()
                if deserts.shape[0] > 0:
                    deserts.write_csv(output_path / "access_deserts.csv")
            
            # Export state summary
            state_summary = self.create_state_level_summary()
            state_summary.write_csv(output_path / "state_health_summary.csv")
            
            # Export GeoJSON format for web mapping (basic structure)
            self._export_geojson_structure(output_path)
            
            logger.info(f"Mapping data exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export mapping data: {e}")
            return False
    
    def _export_geojson_structure(self, output_path: Path) -> None:
        """Export basic GeoJSON structure for web mapping."""
        # Create simplified GeoJSON structure
        # In real implementation would use actual geometries
        
        density_map = self.create_population_density_map()
        
        # Create basic feature collection structure
        features = []
        for row in density_map.iter_rows(named=True):
            feature = {
                "type": "Feature",
                "properties": {
                    "sa2_code": row["sa2_code"],
                    "sa2_name": row["sa2_name"],
                    "state_name": row["state_name"],
                    "population": row["usual_resident_population"],
                    "density": row["population_density_per_sqkm"],
                    "density_category": row["density_category"]
                },
                "geometry": {
                    "type": "Point",  # Simplified - would use actual polygon geometries
                    "coordinates": [row["centroid_lon"], row["centroid_lat"]]
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features[:100]  # Limit for demo purposes
        }
        
        import json
        with open(output_path / "sa2_health_mapping.geojson", 'w') as f:
            json.dump(geojson, f, indent=2)
    
    def generate_mapping_summary(self) -> Dict[str, any]:
        """Generate summary statistics for mapping outputs."""
        summary = {
            'total_areas_mapped': 0,
            'states_covered': [],
            'population_coverage': 0,
            'density_distribution': {},
            'risk_mapping_available': False,
            'access_mapping_available': False
        }
        
        if self.boundary_data is not None:
            summary['total_areas_mapped'] = self.boundary_data.shape[0]
            summary['states_covered'] = self.boundary_data['state_name'].unique().to_list()
            summary['population_coverage'] = int(self.boundary_data['usual_resident_population'].sum())
            
            # Density distribution
            density_map = self.create_population_density_map()
            density_dist = density_map.group_by('density_category').agg([pl.count().alias('count')])
            summary['density_distribution'] = {
                row['density_category']: row['count'] 
                for row in density_dist.iter_rows(named=True)
            }
        
        summary['risk_mapping_available'] = self.health_risk_data is not None
        summary['access_mapping_available'] = self.access_data is not None
        
        return summary
    
    def process_complete_mapping_pipeline(self) -> bool:
        """Execute complete SA2 health mapping pipeline."""
        try:
            logger.info("Starting SA2 health mapping pipeline...")
            
            # Load spatial data
            if not self.load_spatial_data():
                logger.error("Failed to load spatial data")
                return False
            
            # Create mapping outputs
            output_dir = Path("data/outputs/health_mapping")
            if self.export_mapping_data(output_dir):
                logger.info("SA2 health mapping completed successfully")
                
                # Generate summary
                summary = self.generate_mapping_summary()
                logger.info(f"Mapped {summary['total_areas_mapped']} SA2 areas across {len(summary['states_covered'])} states")
                
                return True
            else:
                logger.error("Failed to export mapping data")
                return False
                
        except Exception as e:
            logger.error(f"SA2 health mapping pipeline failed: {e}")
            return False


if __name__ == "__main__":
    # Development testing
    mapper = SA2HealthMapper()
    success = mapper.process_complete_mapping_pipeline()
    
    if success:
        print("✅ SA2 health mapping completed successfully")
    else:
        print("❌ SA2 health mapping failed")