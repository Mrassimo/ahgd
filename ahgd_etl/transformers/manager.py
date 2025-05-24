"""
Transformer manager for orchestrating ETL processing.

This module provides a centralized manager for all transformer classes,
enabling coordinated execution of the ETL pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set, Tuple

import polars as pl

from .census import (
    G01PopulationTransformer,
    G17IncomeTransformer,
    G18AssistanceNeededTransformer, 
    G19HealthConditionsTransformer,
    G20SelectedConditionsTransformer,
    G21ConditionsByCharacteristicsTransformer,
    G25UnpaidAssistanceTransformer
)
from ..config import settings

class TransformerManager:
    """
    Manager for ETL transformers.
    
    This class provides centralized management of all ETL transformer classes,
    coordinating their execution and ensuring proper dependencies and order.
    """
    
    def __init__(self):
        """Initialize the transformer manager."""
        self.logger = logging.getLogger('ahgd_etl.transformers.manager')
        
        # Initialize transformers
        self.transformers = {
            'g01': G01PopulationTransformer(),
            'g17': G17IncomeTransformer(),
            'g18': G18AssistanceNeededTransformer(),
            'g19': G19HealthConditionsTransformer(),
            'g20': G20SelectedConditionsTransformer(),
            'g21': G21ConditionsByCharacteristicsTransformer(),
            'g25': G25UnpaidAssistanceTransformer()
        }
        
        # Define step dependencies
        self.dependencies = {
            'g01': ['download', 'geo', 'time'],
            'g17': ['download', 'geo', 'time'],
            'g18': ['download', 'geo', 'time'],
            'g19': ['download', 'geo', 'time'],
            'g20': ['download', 'geo', 'time', 'g19'],
            'g21': ['download', 'geo', 'time', 'g19'],
            'g25': ['download', 'geo', 'time'],
            'dimensions': ['g19', 'g20', 'g21'],
            'validate': ['g01', 'g17', 'g18', 'g19', 'g20', 'g21', 'g25', 'dimensions']
        }
        
        # Define output files for each step
        self.output_files = {
            'g01': 'fact_population.parquet',
            'g17': 'fact_income.parquet',
            'g18': 'fact_assistance_needed.parquet',
            'g19': 'fact_health_conditions.parquet',
            'g20': 'fact_health_conditions_refined.parquet',
            'g21': 'fact_health_conditions_by_characteristic_refined.parquet',
            'g25': 'fact_unpaid_care.parquet'
        }
    
    def get_transformer(self, table_code: str):
        """
        Get a transformer instance by table code.
        
        Args:
            table_code: Census table code (e.g., 'g01', 'g19')
            
        Returns:
            Transformer instance
        """
        return self.transformers.get(table_code.lower())
    
    def process_table(self, table_code: str, csv_file: Path, geo_output_path: Optional[Path] = None,
                      time_sk: Optional[str] = None) -> Optional[pl.DataFrame]:
        """
        Process a Census table file using the appropriate transformer.
        
        Args:
            table_code: Census table code (e.g., 'g01', 'g19')
            csv_file: Path to the CSV file to process
            geo_output_path: Path to the geographic dimension file (for lookups)
            time_sk: Time dimension surrogate key to use
            
        Returns:
            Processed DataFrame or None if processing failed
        """
        transformer = self.get_transformer(table_code)
        if not transformer:
            self.logger.error(f"No transformer found for table code '{table_code}'")
            return None
        
        start_time = time.time()
        result_df = transformer.process_file(csv_file, geo_output_path, time_sk)
        elapsed_time = time.time() - start_time
        
        if result_df is not None:
            self.logger.info(f"Processed {table_code} file {csv_file.name} in {elapsed_time:.2f} seconds")
        else:
            self.logger.error(f"Failed to process {table_code} file {csv_file.name}")
        
        return result_df
    
    def process_step(self, step: str, zip_dir: Path, temp_extract_base: Path, 
                    output_dir: Path, geo_output_path: Path, time_sk: Optional[str] = None) -> bool:
        """
        Process a complete ETL step (e.g., process all G19 files).
        
        Args:
            step: ETL step to process (e.g., 'g01', 'g19')
            zip_dir: Directory containing Census zip files
            temp_extract_base: Base directory for temporary file extraction
            output_dir: Directory to write output files
            geo_output_path: Path to the geographic dimension file
            time_sk: Time dimension surrogate key to use
            
        Returns:
            True if processing succeeded, False otherwise
        """
        from ..utils import process_census_table
        
        if step not in self.transformers:
            self.logger.error(f"No transformer found for step '{step}'")
            return False
        
        # Get the output filename
        output_filename = self.output_files.get(step)
        if not output_filename:
            self.logger.error(f"No output filename defined for step '{step}'")
            return False
        
        # Get the transformer instance
        transformer = self.get_transformer(step)
        
        # Process the table
        return process_census_table(
            table_code=step.upper(),
            process_file_function=transformer.process_file,
            output_filename=output_filename,
            zip_dir=zip_dir,
            temp_extract_base=temp_extract_base,
            output_dir=output_dir,
            geo_output_path=geo_output_path,
            time_sk=time_sk
        )
    
    def check_dependencies(self, step: str, completed_steps: Set[str]) -> List[str]:
        """
        Check if all dependencies for a step are satisfied.
        
        Args:
            step: Step to check dependencies for
            completed_steps: Set of steps that have been completed
            
        Returns:
            List of missing dependencies
        """
        dependencies = self.dependencies.get(step, [])
        return [dep for dep in dependencies if dep not in completed_steps]
    
    def get_next_step(self, steps: List[str], completed_steps: Set[str]) -> Optional[str]:
        """
        Get the next step to execute based on dependencies.
        
        Args:
            steps: List of steps to execute
            completed_steps: Set of steps that have been completed
            
        Returns:
            Next step to execute or None if no steps are ready
        """
        for step in steps:
            if step in completed_steps:
                continue
                
            missing_deps = self.check_dependencies(step, completed_steps)
            if not missing_deps:
                return step
        
        return None