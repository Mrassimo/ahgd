"""
ETL Pipeline Orchestrator

Manages the execution of ETL pipeline steps with proper dependency management,
error handling, and inline fixes.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from ahgd_etl.config import settings
from ahgd_etl.transformers.geo.geography import GeographyTransformer
from ahgd_etl.models.time_dimension import TimeDimensionBuilder
from ahgd_etl.models.dimension_builder import DimensionBuilder
from ahgd_etl.transformers.enhanced_transformers import (
    G01PopulationTransformer,
    G17IncomeTransformer,
    G18AssistanceNeededTransformer,
    G19HealthConditionsTransformer,
    G20SelectedConditionsTransformer,
    G21ConditionsByCharacteristicsTransformer,
    G25UnpaidAssistanceTransformer
)
from ahgd_etl.validators.data_quality import DataQualityValidator
from ahgd_etl.loaders.snowflake import SnowflakeLoader
from ahgd_etl.core.data_downloader import DataDownloader
from ahgd_etl.core.fix_manager import FixManager


class Orchestrator:
    """Orchestrates ETL pipeline execution with dependency management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.fix_manager = FixManager() if config.get("fix_inline", True) else None
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize pipeline components."""
        output_dir = self.config["output_dir"]
        
        # Data components
        self.downloader = DataDownloader(
            force_download=self.config.get("force_download", False)
        )
        
        # Transformers
        self.geo_transformer = GeographyTransformer(output_dir)
        self.time_builder = TimeDimensionBuilder(output_dir)
        self.dimension_builder = DimensionBuilder(output_dir, self.fix_manager)
        
        # Census transformers with inline fix support
        self.census_transformers = {
            "g01": G01PopulationTransformer(output_dir, self.fix_manager),
            "g17": G17IncomeTransformer(output_dir, self.fix_manager),
            "g18": G18AssistanceNeededTransformer(output_dir, self.fix_manager),
            "g19": G19HealthConditionsTransformer(output_dir, self.fix_manager),
            "g20": G20SelectedConditionsTransformer(output_dir, self.fix_manager),
            "g21": G21ConditionsByCharacteristicsTransformer(output_dir, self.fix_manager),
            "g25": G25UnpaidAssistanceTransformer(output_dir, self.fix_manager)
        }
        
        # Validator
        self.validator = DataQualityValidator(output_dir)
        
        # Step mapping
        self.step_handlers = {
            "download": self._run_download,
            "geo": self._run_geography,
            "time": self._run_time_dimension,
            "dimensions": self._run_dimensions,
            "g01": lambda: self._run_census("g01"),
            "g17": lambda: self._run_census("g17"),
            "g18": lambda: self._run_census("g18"),
            "g19": lambda: self._run_census("g19"),
            "g20": lambda: self._run_census("g20"),
            "g21": lambda: self._run_census("g21"),
            "g25": lambda: self._run_census("g25"),
            "validate": self._run_validation,
            "fix": self._run_fixes,
            "export": self._run_export
        }
        
    def run(self, steps: List[str]) -> bool:
        """
        Run the specified pipeline steps.
        
        Args:
            steps: List of step names to execute
            
        Returns:
            bool: True if all steps succeeded, False otherwise
        """
        overall_success = True
        
        for step in steps:
            if step not in self.step_handlers:
                self.logger.error(f"Unknown step: {step}")
                continue
                
            self.logger.info(f"{'=' * 60}")
            self.logger.info(f"Running step: {step.upper()}")
            self.logger.info(f"{'=' * 60}")
            
            start_time = time.time()
            
            try:
                handler = self.step_handlers[step]
                success = handler()
                
                duration = time.time() - start_time
                
                self.results[step] = {
                    "success": success,
                    "duration": duration,
                    "timestamp": datetime.now()
                }
                
                if success:
                    self.logger.info(f"Step {step} completed successfully in {duration:.2f}s")
                else:
                    self.logger.error(f"Step {step} failed after {duration:.2f}s")
                    overall_success = False
                    
                    if self.config.get("stop_on_error", False):
                        self.logger.error("Stopping pipeline due to error")
                        break
                        
            except Exception as e:
                duration = time.time() - start_time
                self.logger.error(f"Step {step} failed with exception: {str(e)}", exc_info=True)
                
                self.results[step] = {
                    "success": False,
                    "duration": duration,
                    "timestamp": datetime.now(),
                    "error": str(e)
                }
                
                overall_success = False
                
                if self.config.get("stop_on_error", False):
                    break
                    
        return overall_success
    
    def _run_download(self) -> bool:
        """Run data download step."""
        return self.downloader.download_all()
    
    def _run_geography(self) -> bool:
        """Run geography processing step."""
        return self.geo_transformer.process()
    
    def _run_time_dimension(self) -> bool:
        """Run time dimension generation step."""
        return self.time_builder.build()
    
    def _run_dimensions(self) -> bool:
        """Run dimension generation step."""
        success = True
        
        # Generate each dimension
        dimensions = [
            ("health_condition", self.dimension_builder.build_health_condition_dimension),
            ("demographic", self.dimension_builder.build_demographic_dimension),
            ("person_characteristic", self.dimension_builder.build_person_characteristic_dimension)
        ]
        
        for dim_name, builder_func in dimensions:
            self.logger.info(f"Building {dim_name} dimension...")
            if not builder_func():
                self.logger.error(f"Failed to build {dim_name} dimension")
                success = False
                
        return success
    
    def _run_census(self, table_code: str) -> bool:
        """Run census table processing."""
        transformer = self.census_transformers.get(table_code)
        if not transformer:
            self.logger.error(f"No transformer found for {table_code}")
            return False
            
        return transformer.process()
    
    def _run_validation(self) -> bool:
        """Run data validation step."""
        report = self.validator.validate_all()
        
        # Determine overall success
        critical_failures = [
            check for check, result in report.items()
            if not result["passed"] and "ref_integrity" in check
        ]
        
        success = len(critical_failures) == 0
        
        # Log summary
        self.logger.info("Validation Summary:")
        passed = sum(1 for r in report.values() if r["passed"])
        total = len(report)
        self.logger.info(f"Passed: {passed}/{total}")
        
        if critical_failures:
            self.logger.error(f"Critical failures: {critical_failures}")
            
        return success
    
    def _run_fixes(self) -> bool:
        """Run standalone fixes (when not inline)."""
        if not self.fix_manager:
            self.fix_manager = FixManager()
            
        return self.fix_manager.run_all_fixes(self.config["output_dir"])
    
    def _run_export(self) -> bool:
        """Run export step."""
        target = self.config.get("target_format", "parquet")
        
        if target == "snowflake":
            return self._export_to_snowflake()
        elif target == "csv":
            return self._export_to_csv()
        else:
            # Already in parquet format
            return True
    
    def _export_to_snowflake(self) -> bool:
        """Export data to Snowflake."""
        config_file = self.config.get("snowflake_config")
        if not config_file:
            self.logger.error("Snowflake config file required for export")
            return False
            
        loader = SnowflakeLoader(config_file)
        return loader.load_all(self.config["output_dir"])
    
    def _export_to_csv(self) -> bool:
        """Export data to CSV format."""
        # Implementation for CSV export
        self.logger.warning("CSV export not yet implemented")
        return False
    
    def export(self, target: str, config_file: Optional[str] = None) -> bool:
        """Export data to specified target."""
        self.config["target_format"] = target
        self.config["snowflake_config"] = config_file
        return self._run_export()
    
    def get_results(self) -> Dict[str, Any]:
        """Get execution results."""
        return self.results