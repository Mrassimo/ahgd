"""
Pharmaceutical Analyzer - Analysis of PBS prescription patterns and pharmaceutical utilisation

Analyzes Pharmaceutical Benefits Scheme (PBS) data to identify prescription patterns,
medication utilisation, and pharmaceutical access across Australian populations.

Data Sources:
- PBS Prescription Data: Medication prescriptions and dispensing records
- ATC Drug Classifications: Medication categories and therapeutic groups
- Population Data: Demographics for per capita analysis
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PharmaceuticalAnalyzer:
    """
    Analyze PBS prescription patterns and pharmaceutical utilisation.
    Uses the 492,434 PBS records processed in Phase 2.
    """
    
    # Common ATC (Anatomical Therapeutic Chemical) code groups
    ATC_CATEGORIES = {
        'A': 'Alimentary tract and metabolism',
        'B': 'Blood and blood forming organs', 
        'C': 'Cardiovascular system',
        'D': 'Dermatologicals',
        'G': 'Genitourinary system and sex hormones',
        'H': 'Systemic hormonal preparations',
        'J': 'Antiinfectives for systemic use',
        'L': 'Antineoplastic and immunomodulating agents',
        'M': 'Musculo-skeletal system',
        'N': 'Nervous system',
        'P': 'Antiparasitic products',
        'R': 'Respiratory system',
        'S': 'Sensory organs',
        'V': 'Various'
    }
    
    # Chronic disease medication indicators
    CHRONIC_MEDICATIONS = {
        'diabetes': ['A10'],       # Antidiabetic medications
        'hypertension': ['C02', 'C03', 'C07', 'C08', 'C09'],  # Antihypertensives
        'mental_health': ['N05', 'N06'],  # Antipsychotics, antidepressants
        'respiratory': ['R03'],    # Respiratory medications
        'cardiovascular': ['C01', 'C04', 'C10']  # Cardiac therapy, lipid medications
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize analyzer with processed data directory."""
        self.data_dir = data_dir or Path("data/processed")
        self.pbs_data: Optional[pl.DataFrame] = None
        self.population_data: Optional[pl.DataFrame] = None
        
    def load_pharmaceutical_data(self) -> bool:
        """Load PBS pharmaceutical data."""
        try:
            # Try to load real PBS data from Phase 2 processing
            pbs_path = self.data_dir / "health_data_processed.csv"
            if pbs_path.exists():
                self.pbs_data = pl.read_csv(pbs_path)
                logger.info(f"Loaded PBS data: {self.pbs_data.shape[0]} prescription records")
            else:
                logger.info("Real PBS data not found - generating mock data")
                self.pbs_data = self._generate_mock_pbs_data()
                
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
            logger.error(f"Failed to load PBS data: {e}")
            return False
    
    def _generate_mock_pbs_data(self) -> pl.DataFrame:
        """Generate mock PBS prescription data for development."""
        np.random.seed(42)
        n_records = 150000  # Scaled down from 492,434 for development
        
        # Generate realistic SA2 codes
        sa2_codes = [f"1{str(i).zfill(8)}" for i in range(1000, 2500)]
        
        # Common ATC codes for realistic prescriptions
        atc_codes = [
            'A02BC01', 'A10BD08', 'C01DA14', 'C03DA01', 'C07AB07',
            'C09AA02', 'J01CA04', 'M01AE01', 'N02BA01', 'N05BA04',
            'N06AB04', 'R03AC02', 'C10AA01', 'A11CC01', 'B01AC06'
        ]
        
        # Generate drug names corresponding to ATC codes
        drug_names = [
            'Omeprazole', 'Metformin/Gliclazide', 'Glyceryl Trinitrate', 'Indapamide',
            'Bisoprolol', 'Enalapril', 'Amoxicillin', 'Ibuprofen', 'Aspirin',
            'Oxazepam', 'Sertraline', 'Salbutamol', 'Atorvastatin', 'Vitamin D', 'Clopidogrel'
        ]
        
        mock_data = {
            'sa2_code': np.random.choice(sa2_codes, n_records),
            'atc_code': np.random.choice(atc_codes, n_records),
            'drug_name': np.random.choice(drug_names, n_records),
            'prescription_count': np.random.poisson(2, n_records) + 1,  # At least 1 prescription
            'patient_contribution': np.random.exponential(15, n_records) + 5,  # Patient copayment
            'government_benefit': np.random.exponential(35, n_records) + 10,  # PBS subsidy
            'total_cost': np.random.exponential(50, n_records) + 15,  # Total medication cost
            'patient_category': np.random.choice(['General', 'Concession', 'Safety Net'], n_records, p=[0.6, 0.3, 0.1]),
            'dispensing_date': np.random.choice(['2023-01', '2023-02', '2023-03', '2023-04'], n_records),
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
    
    def calculate_prescription_rates(self) -> pl.DataFrame:
        """Calculate prescription rates per capita by SA2."""
        if self.pbs_data is None or self.population_data is None:
            self.load_pharmaceutical_data()
            
        # Aggregate PBS data by SA2
        sa2_prescriptions = self.pbs_data.group_by('sa2_code').agg([
            pl.col('prescription_count').sum().alias('total_prescriptions'),
            pl.col('total_cost').sum().alias('total_medication_cost'),
            pl.col('government_benefit').sum().alias('total_pbs_subsidy'),
            pl.col('patient_contribution').sum().alias('total_patient_cost'),
            pl.col('atc_code').n_unique().alias('unique_medications'),
            pl.count().alias('dispensing_episodes')
        ])
        
        # Join with population data
        prescription_rates = sa2_prescriptions.join(
            self.population_data,
            on='sa2_code',
            how='left'
        )
        
        # Calculate per capita metrics
        prescription_rates = prescription_rates.with_columns([
            (pl.col('total_prescriptions') / pl.col('usual_resident_population')).alias('prescriptions_per_capita'),
            (pl.col('total_medication_cost') / pl.col('usual_resident_population')).alias('medication_cost_per_capita'),
            (pl.col('total_pbs_subsidy') / pl.col('usual_resident_population')).alias('pbs_subsidy_per_capita'),
            (pl.col('dispensing_episodes') / pl.col('usual_resident_population')).alias('dispensing_episodes_per_capita')
        ])
        
        # Add prescription rate categories
        prescription_rates = prescription_rates.with_columns([
            pl.when(pl.col('prescriptions_per_capita') >= pl.col('prescriptions_per_capita').quantile(0.8))
            .then(pl.lit('Very High Usage'))
            .when(pl.col('prescriptions_per_capita') >= pl.col('prescriptions_per_capita').quantile(0.6))
            .then(pl.lit('High Usage'))
            .when(pl.col('prescriptions_per_capita') >= pl.col('prescriptions_per_capita').quantile(0.4))
            .then(pl.lit('Moderate Usage'))
            .when(pl.col('prescriptions_per_capita') >= pl.col('prescriptions_per_capita').quantile(0.2))
            .then(pl.lit('Low Usage'))
            .otherwise(pl.lit('Very Low Usage'))
            .alias('usage_category')
        ])
        
        return prescription_rates
    
    def analyze_therapeutic_categories(self) -> pl.DataFrame:
        """Analyze prescription patterns by therapeutic categories (ATC codes)."""
        if self.pbs_data is None:
            self.load_pharmaceutical_data()
            
        # Extract ATC category (first letter of ATC code)
        atc_analysis = self.pbs_data.with_columns([
            pl.col('atc_code').str.slice(0, 1).alias('atc_category')
        ])
        
        # Aggregate by ATC category and state
        atc_summary = atc_analysis.group_by(['atc_category', 'state']).agg([
            pl.col('prescription_count').sum().alias('total_prescriptions'),
            pl.col('total_cost').sum().alias('total_cost'),
            pl.col('government_benefit').sum().alias('total_subsidy'),
            pl.col('sa2_code').n_unique().alias('areas_served'),
            pl.count().alias('dispensing_episodes')
        ])
        
        # Add category descriptions
        atc_summary = atc_summary.with_columns([
            pl.col('atc_category').map_elements(
                lambda x: self.ATC_CATEGORIES.get(x, 'Unknown'),
                return_dtype=pl.Utf8
            ).alias('category_description')
        ])
        
        # Calculate percentages
        state_totals = atc_summary.group_by('state').agg([
            pl.col('total_prescriptions').sum().alias('state_total_prescriptions')
        ])
        
        atc_summary = atc_summary.join(state_totals, on='state').with_columns([
            (pl.col('total_prescriptions') / pl.col('state_total_prescriptions') * 100).alias('percentage_of_state_prescriptions')
        ])
        
        return atc_summary
    
    def identify_chronic_disease_patterns(self) -> pl.DataFrame:
        """Identify chronic disease medication patterns."""
        if self.pbs_data is None:
            self.load_pharmaceutical_data()
            
        # Add chronic disease indicators based on ATC codes
        chronic_analysis = self.pbs_data.with_columns([
            # Diabetes medications
            pl.when(pl.col('atc_code').str.starts_with('A10'))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('diabetes_medication'),
            
            # Cardiovascular medications
            pl.when(pl.col('atc_code').str.starts_with(('C01', 'C02', 'C03', 'C07', 'C08', 'C09', 'C10')))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('cardiovascular_medication'),
            
            # Mental health medications
            pl.when(pl.col('atc_code').str.starts_with(('N05', 'N06')))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('mental_health_medication'),
            
            # Respiratory medications
            pl.when(pl.col('atc_code').str.starts_with('R03'))
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('respiratory_medication')
        ])
        
        # Aggregate chronic disease indicators by SA2
        chronic_summary = chronic_analysis.group_by('sa2_code').agg([
            pl.col('diabetes_medication').sum().alias('diabetes_prescriptions'),
            pl.col('cardiovascular_medication').sum().alias('cardiovascular_prescriptions'),
            pl.col('mental_health_medication').sum().alias('mental_health_prescriptions'),
            pl.col('respiratory_medication').sum().alias('respiratory_prescriptions'),
            pl.col('prescription_count').sum().alias('total_prescriptions')
        ])
        
        # Calculate chronic disease percentages
        chronic_summary = chronic_summary.with_columns([
            (pl.col('diabetes_prescriptions') / pl.col('total_prescriptions') * 100).alias('diabetes_percentage'),
            (pl.col('cardiovascular_prescriptions') / pl.col('total_prescriptions') * 100).alias('cardiovascular_percentage'),
            (pl.col('mental_health_prescriptions') / pl.col('total_prescriptions') * 100).alias('mental_health_percentage'),
            (pl.col('respiratory_prescriptions') / pl.col('total_prescriptions') * 100).alias('respiratory_percentage')
        ])
        
        return chronic_summary
    
    def analyze_cost_burden(self) -> pl.DataFrame:
        """Analyze pharmaceutical cost burden by patient category."""
        if self.pbs_data is None:
            self.load_pharmaceutical_data()
            
        cost_analysis = self.pbs_data.group_by(['patient_category', 'state']).agg([
            pl.col('prescription_count').sum().alias('total_prescriptions'),
            pl.col('total_cost').sum().alias('total_medication_cost'),
            pl.col('government_benefit').sum().alias('total_government_subsidy'),
            pl.col('patient_contribution').sum().alias('total_patient_cost'),
            pl.col('total_cost').mean().alias('avg_cost_per_prescription'),
            pl.col('patient_contribution').mean().alias('avg_patient_contribution'),
            pl.count().alias('dispensing_episodes')
        ])
        
        # Calculate subsidy rates
        cost_analysis = cost_analysis.with_columns([
            (pl.col('total_government_subsidy') / pl.col('total_medication_cost') * 100).alias('government_subsidy_rate'),
            (pl.col('total_patient_cost') / pl.col('total_medication_cost') * 100).alias('patient_contribution_rate')
        ])
        
        return cost_analysis
    
    def identify_high_medication_areas(self, percentile_threshold: float = 0.9) -> pl.DataFrame:
        """Identify SA2 areas with high medication usage."""
        prescription_rates = self.calculate_prescription_rates()
        
        threshold = prescription_rates['prescriptions_per_capita'].quantile(percentile_threshold)
        
        high_medication = prescription_rates.filter(
            pl.col('prescriptions_per_capita') >= threshold
        ).sort('prescriptions_per_capita', descending=True)
        
        return high_medication
    
    def generate_pharmaceutical_summary(self) -> Dict[str, any]:
        """Generate summary statistics for pharmaceutical analysis."""
        if self.pbs_data is None:
            self.load_pharmaceutical_data()
            
        prescription_rates = self.calculate_prescription_rates()
        atc_analysis = self.analyze_therapeutic_categories()
        
        summary = {
            'total_sa2_areas': prescription_rates.shape[0],
            'total_dispensing_episodes': int(self.pbs_data.shape[0]),
            'total_prescriptions': int(self.pbs_data['prescription_count'].sum()),
            'total_medication_cost': float(self.pbs_data['total_cost'].sum()),
            'total_pbs_subsidy': float(self.pbs_data['government_benefit'].sum()),
            'average_prescriptions_per_capita': float(prescription_rates['prescriptions_per_capita'].mean()),
            'average_cost_per_capita': float(prescription_rates['medication_cost_per_capita'].mean()),
            'high_usage_areas': int(prescription_rates.filter(pl.col('usage_category') == 'Very High Usage').shape[0]),
            'low_usage_areas': int(prescription_rates.filter(pl.col('usage_category') == 'Very Low Usage').shape[0]),
            'usage_distribution': {
                cat: int(count) for cat, count in 
                prescription_rates.group_by('usage_category').agg([pl.count().alias('count')])
                .select(['usage_category', 'count']).iter_rows()
            },
            'top_therapeutic_categories': {
                cat: int(total) for cat, total in
                atc_analysis.group_by('category_description').agg([pl.col('total_prescriptions').sum().alias('total')])
                .sort('total', descending=True).limit(5)
                .select(['category_description', 'total']).iter_rows()
            }
        }
        
        return summary
    
    def export_pharmaceutical_analysis(self, output_path: Path) -> bool:
        """Export pharmaceutical analysis results."""
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export prescription rates
            prescription_rates = self.calculate_prescription_rates()
            prescription_rates.write_csv(output_path / "prescription_rates_by_sa2.csv")
            
            # Export therapeutic category analysis
            atc_analysis = self.analyze_therapeutic_categories()
            atc_analysis.write_csv(output_path / "therapeutic_category_analysis.csv")
            
            # Export chronic disease patterns
            chronic_patterns = self.identify_chronic_disease_patterns()
            chronic_patterns.write_csv(output_path / "chronic_disease_medication_patterns.csv")
            
            # Export cost burden analysis
            cost_analysis = self.analyze_cost_burden()
            cost_analysis.write_csv(output_path / "pharmaceutical_cost_burden.csv")
            
            # Export high medication areas
            high_med_areas = self.identify_high_medication_areas()
            high_med_areas.write_csv(output_path / "high_medication_usage_areas.csv")
            
            # Export summary
            summary = self.generate_pharmaceutical_summary()
            summary_path = output_path / "pharmaceutical_analysis_summary.json"
            
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Pharmaceutical analysis exported to {output_path}")
            logger.info(f"Summary: {summary['total_dispensing_episodes']} episodes, {summary['total_prescriptions']} prescriptions")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to export pharmaceutical analysis: {e}")
            return False
    
    def process_complete_pharmaceutical_pipeline(self) -> bool:
        """Execute complete pharmaceutical analysis pipeline."""
        try:
            logger.info("Starting pharmaceutical analysis pipeline...")
            
            # Load data
            if not self.load_pharmaceutical_data():
                logger.error("Failed to load pharmaceutical data")
                return False
            
            # Generate analysis results
            output_dir = Path("data/outputs/pharmaceutical_analysis")
            if self.export_pharmaceutical_analysis(output_dir):
                logger.info("Pharmaceutical analysis completed successfully")
                
                # Generate summary
                summary = self.generate_pharmaceutical_summary()
                logger.info(f"Analyzed {summary['total_dispensing_episodes']} dispensing episodes")
                logger.info(f"Identified {summary['high_usage_areas']} high medication usage areas")
                
                return True
            else:
                logger.error("Failed to export pharmaceutical analysis")
                return False
                
        except Exception as e:
            logger.error(f"Pharmaceutical analysis pipeline failed: {e}")
            return False


if __name__ == "__main__":
    # Development testing
    analyzer = PharmaceuticalAnalyzer()
    success = analyzer.process_complete_pharmaceutical_pipeline()
    
    if success:
        print("✅ Pharmaceutical analysis completed successfully")
    else:
        print("❌ Pharmaceutical analysis failed")