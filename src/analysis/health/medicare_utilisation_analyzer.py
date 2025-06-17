"""
Medicare Utilisation Analyzer - Analysis of Medicare service utilisation patterns

Analyzes Medicare Benefits Schedule (MBS) data to identify service utilisation patterns,
per capita usage, and healthcare access indicators across Australian populations.

Data Sources:
- MBS Service Data: Medicare service claims and utilisation
- Population Data: Demographics and SA2 resident populations
- Geographic Data: Service distribution and accessibility
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MedicareUtilisationAnalyzer:
    """
    Analyze Medicare service utilisation patterns and healthcare access indicators.
    """
    
    # Medicare service categories (MBS item groups)
    SERVICE_CATEGORIES = {
        'gp_consultations': 'General Practice consultations and procedures',
        'specialist_consultations': 'Specialist medical consultations',
        'diagnostic_imaging': 'Radiology and diagnostic imaging',
        'pathology': 'Pathology and laboratory services',
        'allied_health': 'Allied health professional services',
        'procedures': 'Surgical and medical procedures'
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize analyzer with processed data directory."""
        self.data_dir = data_dir or Path("data/processed")
        self.medicare_data: Optional[pl.DataFrame] = None
        self.population_data: Optional[pl.DataFrame] = None
        
    def load_medicare_data(self) -> bool:
        """Load Medicare utilisation data."""
        try:
            # Try to load real Medicare data
            medicare_path = self.data_dir / "medicare_utilisation.csv"
            if medicare_path.exists():
                self.medicare_data = pl.read_csv(medicare_path)
                logger.info(f"Loaded Medicare data: {self.medicare_data.shape[0]} records")
            else:
                logger.info("Real Medicare data not found - generating mock data")
                self.medicare_data = self._generate_mock_medicare_data()
                
            # Load population data for per capita calculations
            pop_path = self.data_dir / "sa2_boundaries_processed.csv"
            if pop_path.exists():
                self.population_data = pl.read_csv(pop_path).select([
                    'sa2_code', 'state_name', 'usual_resident_population'
                ])
                logger.info(f"Loaded population data: {self.population_data.shape[0]} SA2 areas")
            else:
                self.population_data = self._generate_mock_population_data()
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Medicare data: {e}")
            return False
    
    def _generate_mock_medicare_data(self) -> pl.DataFrame:
        """Generate mock Medicare utilisation data for development."""
        np.random.seed(42)
        n_records = 100000
        
        # Generate realistic SA2 codes
        sa2_codes = [f"1{str(i).zfill(8)}" for i in range(1000, 2500)]
        
        # Service types based on MBS item numbers
        service_types = [
            'GP Consultation', 'Specialist Consultation', 'Diagnostic Imaging',
            'Pathology', 'Allied Health', 'Minor Procedure', 'Major Procedure'
        ]
        
        # Generate mock MBS item numbers
        item_numbers = np.random.choice(range(1, 99999), n_records)
        
        mock_data = {
            'sa2_code': np.random.choice(sa2_codes, n_records),
            'mbs_item_number': item_numbers,
            'service_type': np.random.choice(service_types, n_records),
            'service_count': np.random.poisson(3, n_records) + 1,  # At least 1 service
            'total_benefits_paid': np.random.exponential(80, n_records) + 10,  # Realistic benefit amounts
            'patient_age_group': np.random.choice(['0-17', '18-34', '35-54', '55-74', '75+'], n_records),
            'service_quarter': np.random.choice(['2023-Q1', '2023-Q2', '2023-Q3', '2023-Q4'], n_records),
            'state': np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], n_records)
        }
        
        return pl.DataFrame(mock_data)
    
    def _generate_mock_population_data(self) -> pl.DataFrame:
        """Generate mock population data."""
        np.random.seed(42)
        n_areas = 1500
        
        return pl.DataFrame({
            'sa2_code': [f"1{str(i).zfill(8)}" for i in range(1000, 1000 + n_areas)],
            'state_name': np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], n_areas),
            'usual_resident_population': np.random.randint(100, 8000, n_areas)
        })
    
    def calculate_per_capita_utilisation(self) -> pl.DataFrame:
        """Calculate Medicare service utilisation per capita by SA2."""
        if self.medicare_data is None or self.population_data is None:
            self.load_medicare_data()
            
        # Aggregate Medicare data by SA2
        sa2_utilisation = self.medicare_data.group_by('sa2_code').agg([
            pl.col('service_count').sum().alias('total_services'),
            pl.col('total_benefits_paid').sum().alias('total_benefits'),
            pl.col('mbs_item_number').n_unique().alias('unique_services'),
            pl.count().alias('service_episodes')
        ])
        
        # Join with population data
        per_capita = sa2_utilisation.join(
            self.population_data,
            on='sa2_code',
            how='left'
        )
        
        # Calculate per capita metrics
        per_capita = per_capita.with_columns([
            (pl.col('total_services') / pl.col('usual_resident_population')).alias('services_per_capita'),
            (pl.col('total_benefits') / pl.col('usual_resident_population')).alias('benefits_per_capita'),
            (pl.col('service_episodes') / pl.col('usual_resident_population')).alias('episodes_per_capita')
        ])
        
        # Add utilisation categories
        per_capita = per_capita.with_columns([
            pl.when(pl.col('services_per_capita') >= pl.col('services_per_capita').quantile(0.8))
            .then(pl.lit('Very High Utilisation'))
            .when(pl.col('services_per_capita') >= pl.col('services_per_capita').quantile(0.6))
            .then(pl.lit('High Utilisation'))
            .when(pl.col('services_per_capita') >= pl.col('services_per_capita').quantile(0.4))
            .then(pl.lit('Moderate Utilisation'))
            .when(pl.col('services_per_capita') >= pl.col('services_per_capita').quantile(0.2))
            .then(pl.lit('Low Utilisation'))
            .otherwise(pl.lit('Very Low Utilisation'))
            .alias('utilisation_category')
        ])
        
        return per_capita
    
    def analyze_service_type_utilisation(self) -> pl.DataFrame:
        """Analyze utilisation patterns by service type."""
        if self.medicare_data is None:
            self.load_medicare_data()
            
        # Aggregate by service type and state
        service_analysis = self.medicare_data.group_by(['service_type', 'state']).agg([
            pl.col('service_count').sum().alias('total_services'),
            pl.col('total_benefits_paid').sum().alias('total_benefits'),
            pl.col('sa2_code').n_unique().alias('areas_served'),
            pl.count().alias('service_episodes')
        ])
        
        # Calculate service type percentages
        total_services_by_state = service_analysis.group_by('state').agg([
            pl.col('total_services').sum().alias('state_total_services')
        ])
        
        service_analysis = service_analysis.join(
            total_services_by_state,
            on='state'
        ).with_columns([
            (pl.col('total_services') / pl.col('state_total_services') * 100).alias('percentage_of_state_services')
        ])
        
        return service_analysis
    
    def identify_high_utilisation_areas(self, percentile_threshold: float = 0.9) -> pl.DataFrame:
        """Identify SA2 areas with high Medicare utilisation."""
        per_capita = self.calculate_per_capita_utilisation()
        
        threshold = per_capita['services_per_capita'].quantile(percentile_threshold)
        
        high_utilisation = per_capita.filter(
            pl.col('services_per_capita') >= threshold
        ).sort('services_per_capita', descending=True)
        
        return high_utilisation
    
    def identify_underserviced_areas(self, percentile_threshold: float = 0.1) -> pl.DataFrame:
        """Identify SA2 areas with low Medicare utilisation (potentially underserviced)."""
        per_capita = self.calculate_per_capita_utilisation()
        
        threshold = per_capita['services_per_capita'].quantile(percentile_threshold)
        
        underserviced = per_capita.filter(
            pl.col('services_per_capita') <= threshold
        ).sort('services_per_capita')
        
        return underserviced
    
    def analyze_age_group_utilisation(self) -> pl.DataFrame:
        """Analyze Medicare utilisation patterns by age group."""
        if self.medicare_data is None:
            self.load_medicare_data()
            
        age_analysis = self.medicare_data.group_by(['patient_age_group', 'state']).agg([
            pl.col('service_count').sum().alias('total_services'),
            pl.col('total_benefits_paid').sum().alias('total_benefits'),
            pl.col('total_benefits_paid').mean().alias('avg_benefit_per_service'),
            pl.count().alias('service_episodes')
        ])
        
        # Calculate age group service intensity
        age_analysis = age_analysis.with_columns([
            (pl.col('total_services') / pl.col('service_episodes')).alias('services_per_episode')
        ])
        
        return age_analysis
    
    def generate_utilisation_summary(self) -> Dict[str, any]:
        """Generate summary statistics for Medicare utilisation analysis."""
        if self.medicare_data is None:
            self.load_medicare_data()
            
        per_capita = self.calculate_per_capita_utilisation()
        
        summary = {
            'total_sa2_areas': per_capita.shape[0],
            'total_service_episodes': int(self.medicare_data.shape[0]),
            'total_services_provided': int(self.medicare_data['service_count'].sum()),
            'total_benefits_paid': float(self.medicare_data['total_benefits_paid'].sum()),
            'average_services_per_capita': float(per_capita['services_per_capita'].mean()),
            'average_benefits_per_capita': float(per_capita['benefits_per_capita'].mean()),
            'high_utilisation_areas': int(per_capita.filter(pl.col('utilisation_category') == 'Very High Utilisation').shape[0]),
            'low_utilisation_areas': int(per_capita.filter(pl.col('utilisation_category') == 'Very Low Utilisation').shape[0]),
            'utilisation_distribution': {
                cat: int(count) for cat, count in 
                per_capita.group_by('utilisation_category').agg([pl.count().alias('count')])
                .select(['utilisation_category', 'count']).iter_rows()
            },
            'service_type_distribution': {
                service: int(count) for service, count in
                self.medicare_data.group_by('service_type').agg([pl.col('service_count').sum().alias('count')])
                .select(['service_type', 'count']).iter_rows()
            }
        }
        
        return summary
    
    def export_utilisation_analysis(self, output_path: Path) -> bool:
        """Export Medicare utilisation analysis results."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export per capita utilisation
            per_capita = self.calculate_per_capita_utilisation()
            per_capita.write_csv(output_path / "medicare_per_capita_utilisation.csv")
            
            # Export service type analysis
            service_analysis = self.analyze_service_type_utilisation()
            service_analysis.write_csv(output_path / "medicare_service_type_analysis.csv")
            
            # Export high utilisation areas
            high_util = self.identify_high_utilisation_areas()
            high_util.write_csv(output_path / "high_utilisation_areas.csv")
            
            # Export underserviced areas
            underserviced = self.identify_underserviced_areas()
            underserviced.write_csv(output_path / "underserviced_areas.csv")
            
            # Export age group analysis
            age_analysis = self.analyze_age_group_utilisation()
            age_analysis.write_csv(output_path / "age_group_utilisation.csv")
            
            # Export summary
            summary = self.generate_utilisation_summary()
            summary_path = output_path / "medicare_utilisation_summary.json"
            
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Medicare utilisation analysis exported to {output_path}")
            logger.info(f"Summary: {summary['total_service_episodes']} episodes across {summary['total_sa2_areas']} areas")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export utilisation analysis: {e}")
            return False
    
    def process_complete_utilisation_pipeline(self) -> bool:
        """Execute complete Medicare utilisation analysis pipeline."""
        try:
            logger.info("Starting Medicare utilisation analysis pipeline...")
            
            # Load data
            if not self.load_medicare_data():
                logger.error("Failed to load Medicare data")
                return False
            
            # Generate analysis results
            output_dir = Path("data/outputs/medicare_analysis")
            if self.export_utilisation_analysis(output_dir):
                logger.info("Medicare utilisation analysis completed successfully")
                
                # Generate summary
                summary = self.generate_utilisation_summary()
                logger.info(f"Analyzed {summary['total_service_episodes']} service episodes")
                logger.info(f"Identified {summary['high_utilisation_areas']} high utilisation areas")
                
                return True
            else:
                logger.error("Failed to export utilisation analysis")
                return False
                
        except Exception as e:
            logger.error(f"Medicare utilisation analysis pipeline failed: {e}")
            return False


if __name__ == "__main__":
    # Development testing
    analyzer = MedicareUtilisationAnalyzer()
    success = analyzer.process_complete_utilisation_pipeline()
    
    if success:
        print("✅ Medicare utilisation analysis completed successfully")
    else:
        print("❌ Medicare utilisation analysis failed")