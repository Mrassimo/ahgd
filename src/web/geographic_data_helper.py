"""
Geographic Data Helper for Australian Health Analytics Dashboard

Provides real SA2 boundary data, centroids, and geographic processing
functions for the Streamlit dashboard. Replaces mock coordinate generation
with actual Australian Bureau of Statistics geographic data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import polars as pl
from loguru import logger
# import geopandas as gpd
# from shapely.geometry import Point, Polygon
# import numpy as np  # Temporarily disabled due to compatibility issues

class GeographicDataHelper:
    """
    Helper class for processing real SA2 geographic data for dashboard visualization.
    
    Features:
    - Loads real SA2 boundaries from processed datasets
    - Calculates centroids for map markers
    - Integrates health risk data with geographic locations
    - Optimized for dashboard performance with caching
    """
    
    def __init__(self, data_dir: Union[str, Path] = "data"):
        self.data_dir = Path(data_dir)
        self.web_exports_dir = self.data_dir / "web_exports"
        self.processed_dir = self.data_dir / "processed"
        self.outputs_dir = self.data_dir / "outputs"
        
        # Cache for loaded data
        self._sa2_boundaries_cache = None
        self._sa2_centroids_cache = None
        self._health_risk_cache = None
        self._seifa_data_cache = None
        self._platform_performance_cache = None
        
        logger.info("Geographic Data Helper initialized")
    
    def load_sa2_boundaries_geojson(self) -> Optional[Dict]:
        """
        Load SA2 boundaries from processed geographic data as raw GeoJSON.
        
        Returns GeoJSON dict with SA2 polygons, or None if data not available.
        """
        if self._sa2_boundaries_cache is not None:
            return self._sa2_boundaries_cache
        
        try:
            # Try to load from web exports first
            geojson_path = self.web_exports_dir / "geojson" / "sa2_boundaries" / "sa2_overview.geojson"
            
            if geojson_path.exists():
                logger.info(f"Loading SA2 boundaries from {geojson_path}")
                with open(geojson_path, 'r') as f:
                    self._sa2_boundaries_cache = json.load(f)
                return self._sa2_boundaries_cache
            
            # Try alternative location
            alt_path = self.processed_dir / "geographic" / "sa2_boundaries.geojson"
            if alt_path.exists():
                logger.info(f"Loading SA2 boundaries from {alt_path}")
                with open(alt_path, 'r') as f:
                    self._sa2_boundaries_cache = json.load(f)
                return self._sa2_boundaries_cache
            
            logger.warning("SA2 boundaries file not found - using mock data")
            return None
            
        except Exception as e:
            logger.error(f"Error loading SA2 boundaries: {e}")
            return None
    
    def calculate_sa2_centroids_from_geojson(self) -> Optional[pd.DataFrame]:
        """
        Calculate centroids for SA2 areas from GeoJSON data.
        
        Returns DataFrame with SA2_CODE, SA2_NAME, latitude, longitude.
        """
        if self._sa2_centroids_cache is not None:
            return self._sa2_centroids_cache
        
        boundaries_geojson = self.load_sa2_boundaries_geojson()
        if boundaries_geojson is None:
            logger.warning("Cannot calculate centroids - no boundary data available")
            return None
        
        try:
            centroids_data = []
            
            for feature in boundaries_geojson.get('features', []):
                properties = feature.get('properties', {})
                geometry = feature.get('geometry', {})
                
                sa2_code = properties.get('SA2_CODE21')
                sa2_name = properties.get('SA2_NAME21')
                
                if geometry.get('type') == 'Polygon' and geometry.get('coordinates'):
                    # Calculate centroid of polygon (simple average of vertices)
                    coords = geometry['coordinates'][0]  # Outer ring
                    if len(coords) > 0:
                        lons = [coord[0] for coord in coords]
                        lats = [coord[1] for coord in coords]
                        
                        centroid_lon = sum(lons) / len(lons)
                        centroid_lat = sum(lats) / len(lats)
                        
                        centroids_data.append({
                            'sa2_code': sa2_code,
                            'sa2_name': sa2_name,
                            'longitude': centroid_lon,
                            'latitude': centroid_lat
                        })
            
            if centroids_data:
                result = pd.DataFrame(centroids_data)
                self._sa2_centroids_cache = result
                logger.info(f"Calculated centroids for {len(result)} SA2 areas")
                return result
            else:
                logger.warning("No valid polygon features found in GeoJSON")
                return None
            
        except Exception as e:
            logger.error(f"Error calculating centroids: {e}")
            return None
    
    def load_health_risk_data(self) -> Optional[pl.DataFrame]:
        """
        Load real health risk assessment data.
        
        Returns Polars DataFrame with health risk scores and categories.
        """
        if self._health_risk_cache is not None:
            return self._health_risk_cache
        
        try:
            # Try CSV first (more reliable)
            csv_path = self.outputs_dir / "risk_assessment" / "health_risk_assessment.csv"
            
            if csv_path.exists():
                logger.info(f"Loading health risk data from {csv_path}")
                self._health_risk_cache = pl.read_csv(csv_path)
                return self._health_risk_cache
            
            # Try Parquet
            parquet_path = self.outputs_dir / "risk_assessment" / "health_risk_assessment.parquet"
            if parquet_path.exists():
                logger.info(f"Loading health risk data from {parquet_path}")
                self._health_risk_cache = pl.read_parquet(parquet_path)
                return self._health_risk_cache
            
            logger.warning("Health risk data not found")
            return None
            
        except Exception as e:
            logger.error(f"Error loading health risk data: {e}")
            return None
    
    def load_seifa_data(self) -> Optional[pl.DataFrame]:
        """
        Load SEIFA socio-economic data.
        
        Returns Polars DataFrame with SEIFA indices and demographics.
        """
        if self._seifa_data_cache is not None:
            return self._seifa_data_cache
        
        try:
            # Try CSV first
            csv_path = self.processed_dir / "seifa_2021_sa2.csv"
            
            if csv_path.exists():
                logger.info(f"Loading SEIFA data from {csv_path}")
                self._seifa_data_cache = pl.read_csv(csv_path)
                return self._seifa_data_cache
            
            # Try Parquet
            parquet_path = self.processed_dir / "seifa_2021_sa2.parquet"
            if parquet_path.exists():
                logger.info(f"Loading SEIFA data from {parquet_path}")
                self._seifa_data_cache = pl.read_parquet(parquet_path)
                return self._seifa_data_cache
            
            logger.warning("SEIFA data not found")
            return None
            
        except Exception as e:
            logger.error(f"Error loading SEIFA data: {e}")
            return None
    
    def load_platform_performance_data(self) -> Optional[Dict]:
        """
        Load platform performance metrics and statistics.
        
        Returns dictionary with processing statistics, record counts, etc.
        """
        if self._platform_performance_cache is not None:
            return self._platform_performance_cache
        
        try:
            perf_path = self.web_exports_dir / "json" / "performance" / "platform_performance.json"
            
            if perf_path.exists():
                logger.info(f"Loading platform performance data from {perf_path}")
                with open(perf_path, 'r') as f:
                    self._platform_performance_cache = json.load(f)
                return self._platform_performance_cache
            
            logger.warning("Platform performance data not found")
            return None
            
        except Exception as e:
            logger.error(f"Error loading platform performance data: {e}")
            return None
    
    def create_integrated_dataset(self, limit: Optional[int] = None) -> Optional[pl.DataFrame]:
        """
        Create integrated dataset combining health risk, SEIFA, and geographic data.
        
        Args:
            limit: Maximum number of records to return (for performance)
        
        Returns:
            Polars DataFrame with integrated health, socio-economic, and geographic data
        """
        try:
            # Load all datasets
            health_risk = self.load_health_risk_data()
            seifa_data = self.load_seifa_data()
            centroids = self.calculate_sa2_centroids()
            
            if health_risk is None:
                logger.warning("Cannot create integrated dataset - no health risk data")
                return None
            
            # Start with health risk data
            integrated = health_risk
            
            # Join with SEIFA data if available
            if seifa_data is not None:
                # Ensure consistent column names
                seifa_renamed = seifa_data.select([
                    pl.col("sa2_code_2021").alias("sa2_code_2021"),
                    pl.col("sa2_name_2021").alias("sa2_name_2021"),
                    pl.col("irsd_score"),
                    pl.col("irsd_decile"),
                    pl.col("irsad_score"),
                    pl.col("irsad_decile"),
                    pl.col("ier_score"),
                    pl.col("ier_decile"),
                    pl.col("ieo_score"),
                    pl.col("ieo_decile"),
                    pl.col("usual_resident_population").alias("population")
                ])
                
                integrated = integrated.join(
                    seifa_renamed,
                    on="sa2_code_2021",
                    how="left"
                )
                logger.info("Joined SEIFA data with health risk data")
            
            # Join with centroids if available
            centroids = self.calculate_sa2_centroids_from_geojson()
            if centroids is not None:
                centroids_pl = pl.from_pandas(centroids)
                centroids_renamed = centroids_pl.select([
                    pl.col("sa2_code").alias("sa2_code_2021"),
                    pl.col("longitude"),
                    pl.col("latitude")
                ])
                
                integrated = integrated.join(
                    centroids_renamed,
                    on="sa2_code_2021",
                    how="left"
                )
                logger.info("Joined centroid data with integrated dataset")
            
            # Apply limit if specified
            if limit:
                integrated = integrated.head(limit)
            
            logger.info(f"Created integrated dataset with {len(integrated)} records")
            return integrated
            
        except Exception as e:
            logger.error(f"Error creating integrated dataset: {e}")
            return None
    
    def get_mock_coordinates_fallback(self, count: int = 100) -> List[Tuple[float, float]]:
        """
        Fallback method to generate mock coordinates if real data unavailable.
        
        Args:
            count: Number of coordinate pairs to generate
        
        Returns:
            List of (latitude, longitude) tuples for Australia
        """
        logger.warning("Using mock coordinates fallback - real geographic data not available")
        
        # Australia bounding box
        lat_min, lat_max = -44.0, -10.0
        lon_min, lon_max = 113.0, 154.0
        
        coordinates = []
        for _ in range(count):
            # Weight towards populated areas (east coast)
            if np.random.random() < 0.6:  # 60% chance for populated areas
                lat = np.random.uniform(-37.0, -16.0)  # East coast latitude range
                lon = np.random.uniform(138.0, 154.0)  # East coast longitude range
            else:  # 40% chance for rest of Australia
                lat = np.random.uniform(lat_min, lat_max)
                lon = np.random.uniform(lon_min, lon_max)
            
            coordinates.append((lat, lon))
        
        return coordinates
    
    def clear_cache(self):
        """Clear all cached data to free memory."""
        self._sa2_boundaries_cache = None
        self._sa2_centroids_cache = None
        self._health_risk_cache = None
        self._seifa_data_cache = None
        self._platform_performance_cache = None
        logger.info("Geographic data cache cleared")


def create_sample_real_data_for_testing(data_dir: Union[str, Path] = "data") -> bool:
    """
    Create sample real data for testing if actual processed data is not available.
    
    This function generates realistic test data based on actual Australian SA2 areas
    to demonstrate the dashboard capabilities.
    
    Returns:
        True if sample data was created successfully, False otherwise
    """
    try:
        data_path = Path(data_dir)
        
        # Create sample health risk data with realistic SA2 codes and names
        sample_areas = [
            ("101021007", "Braidwood", "NSW"),
            ("101021008", "Karabar", "NSW"),
            ("101021009", "Queanbeyan", "NSW"),
            ("101021010", "Queanbeyan - East", "NSW"),
            ("101021012", "Queanbeyan West - Jerrabomberra", "NSW"),
            ("201011001", "Melbourne - Inner", "VIC"),
            ("201011002", "Melbourne - Inner East", "VIC"),
            ("201011003", "Melbourne - Inner South", "VIC"),
            ("301011001", "Brisbane - Inner", "QLD"),
            ("301011002", "Brisbane - Inner East", "QLD"),
            ("401011001", "Perth - Inner", "WA"),
            ("401011002", "Perth - North East", "WA"),
            ("501011001", "Adelaide - Central and Hills", "SA"),
            ("501011002", "Adelaide - North", "SA"),
            ("601011001", "Hobart - Inner", "TAS"),
        ]
        
        # Generate sample integrated dataset
        import random
        random.seed(42)  # For reproducible results
        sample_data = []
        
        for i, (sa2_code, sa2_name, state) in enumerate(sample_areas * 10):  # Create 150 records
            # Generate realistic coordinates based on state
            state_coords = {
                "NSW": (-33.8688, 151.2093),  # Sydney
                "VIC": (-37.8136, 144.9631),  # Melbourne
                "QLD": (-27.4698, 153.0251),  # Brisbane
                "WA": (-31.9505, 115.8605),   # Perth
                "SA": (-34.9285, 138.6007),   # Adelaide
                "TAS": (-42.8821, 147.3272),  # Hobart
            }
            
            base_lat, base_lon = state_coords.get(state, (-25.2744, 133.7751))
            
            # Add some variation around the city center
            lat = base_lat + random.uniform(-0.5, 0.5)
            lon = base_lon + random.uniform(-0.5, 0.5)
            
            sample_data.append({
                "sa2_code_2021": f"{sa2_code}_{i:03d}",
                "sa2_name_2021": f"{sa2_name} {i}",
                "state_name": state,
                "latitude": lat,
                "longitude": lon,
                "population": random.randint(500, 15000),
                "composite_risk_score": random.uniform(0.1, 0.9),
                "risk_category": random.choice(["Low Risk", "Moderate Risk", "High Risk"]),
                "irsd_score": random.uniform(800, 1200),
                "irsd_decile": random.randint(1, 11),
                "seifa_risk_score": random.uniform(3, 8),
                "health_utilisation_risk": random.uniform(0, 7),
                "total_prescriptions": random.randint(20, 100),
                "chronic_medication_rate": random.uniform(0.1, 0.6),
            })
        
        # Create DataFrame and save
        df = pl.DataFrame(sample_data)
        
        # Ensure output directory exists
        output_dir = data_path / "outputs" / "risk_assessment"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_path = output_dir / "health_risk_assessment.csv"
        df.write_csv(csv_path)
        
        logger.info(f"Created sample data with {len(df)} records at {csv_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return False