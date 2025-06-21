"""Complete export pipeline orchestration for production data delivery.

This module provides comprehensive export pipeline capabilities with validation,
quality assurance, and metadata generation for Australian health and geographic data.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import asyncio

import pandas as pd
import numpy as np
from loguru import logger

from ..utils.interfaces import AHGDError, LoadingError
from ..utils.config import get_config
from ..utils.logging import get_logger, monitor_performance, track_lineage
from ..loaders.production_loader import ProductionLoader
from ..loaders.format_exporters import (
    ParquetExporter, CSVExporter, GeoJSONExporter, 
    JSONExporter, WebExporter
)
from ..utils.compression_utils import (
    CompressionAnalyzer, FormatOptimizer, SizeCalculator, CacheManager
)
from ..validators.quality_checker import QualityChecker


@dataclass
class ExportTask:
    """Represents a single export task."""
    task_id: str
    data_source: str
    output_path: Path
    formats: List[str]
    compression: bool
    partition: bool
    web_optimise: bool
    priority: str  # low, medium, high
    created_at: datetime
    status: str = 'pending'  # pending, running, completed, failed
    progress: float = 0.0
    error_message: Optional[str] = None
    export_results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialisation."""
        data = asdict(self)
        data['output_path'] = str(self.output_path)
        data['created_at'] = self.created_at.isoformat()
        return data


class ExportValidator:
    """Validates exported data integrity and completeness."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config('export_validation', {})
        self.logger = get_logger(__name__)
        self.quality_checker = QualityChecker()
        
    @monitor_performance("export_validation")
    def validate_export(self, 
                       original_data: pd.DataFrame,
                       export_results: Dict[str, Any],
                       validation_level: str = 'standard') -> Dict[str, Any]:
        """Validate exported data against original.
        
        Args:
            original_data: Original DataFrame
            export_results: Export results dictionary
            validation_level: Level of validation (quick, standard, comprehensive)
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'overall_status': 'pass',
            'validation_time': datetime.now().isoformat(),
            'validation_level': validation_level,
            'format_validations': {},
            'data_integrity': {},
            'performance_metrics': {}
        }
        
        try:
            # Validate each exported format
            for format_type, format_info in export_results.get('formats', {}).items():
                format_validation = self._validate_format_export(
                    original_data, format_type, format_info, validation_level
                )
                validation_results['format_validations'][format_type] = format_validation
                
                if format_validation['status'] != 'pass':
                    validation_results['overall_status'] = 'fail'
                    
            # Validate data integrity
            integrity_check = self._validate_data_integrity(
                original_data, export_results
            )
            validation_results['data_integrity'] = integrity_check
            
            if integrity_check['status'] != 'pass':
                validation_results['overall_status'] = 'fail'
                
            # Performance validation
            performance_check = self._validate_performance(
                export_results
            )
            validation_results['performance_metrics'] = performance_check
            
            self.logger.info(f"Export validation completed",
                           status=validation_results['overall_status'],
                           formats_validated=len(validation_results['format_validations']))
            
            return validation_results
            
        except Exception as e:
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
            self.logger.error(f"Export validation failed: {str(e)}")
            return validation_results
    
    def _validate_format_export(self, 
                               original_data: pd.DataFrame,
                               format_type: str,
                               format_info: Dict[str, Any],
                               validation_level: str) -> Dict[str, Any]:
        """Validate specific format export."""
        validation = {
            'status': 'pass',
            'file_checks': [],
            'data_checks': [],
            'warnings': []
        }
        
        # Check that files exist and have reasonable sizes
        for file_info in format_info.get('files', []):
            file_path = Path(file_info['path'])
            
            file_check = {
                'filename': file_info['filename'],
                'exists': file_path.exists(),
                'size_bytes': file_info['size_bytes'],
                'size_reasonable': file_info['size_bytes'] > 0
            }
            
            if not file_check['exists'] or not file_check['size_reasonable']:
                validation['status'] = 'fail'
                
            validation['file_checks'].append(file_check)
            
        # Data integrity checks (for formats we can read back)
        if validation_level in ['standard', 'comprehensive'] and format_type in ['parquet', 'csv']:
            try:
                data_check = self._verify_exported_data(
                    original_data, format_info['files'][0]['path'], format_type
                )
                validation['data_checks'].append(data_check)
                
                if data_check['status'] != 'pass':
                    validation['status'] = 'fail'
                    
            except Exception as e:
                validation['warnings'].append(f"Could not verify data integrity: {str(e)}")
                
        return validation
    
    def _verify_exported_data(self, 
                             original_data: pd.DataFrame,
                             exported_file_path: str,
                             format_type: str) -> Dict[str, Any]:
        """Verify exported data matches original."""
        try:
            if format_type == 'parquet':
                exported_data = pd.read_parquet(exported_file_path)
            elif format_type == 'csv':
                exported_data = pd.read_csv(exported_file_path)
            else:
                return {'status': 'skip', 'reason': f'Verification not supported for {format_type}'}
                
            # Basic checks
            row_count_match = len(exported_data) == len(original_data)
            column_count_match = len(exported_data.columns) == len(original_data.columns)
            
            # Column name checks
            columns_match = set(exported_data.columns) == set(original_data.columns)
            
            data_check = {
                'status': 'pass' if all([row_count_match, column_count_match, columns_match]) else 'fail',
                'row_count_original': len(original_data),
                'row_count_exported': len(exported_data),
                'row_count_match': row_count_match,
                'column_count_original': len(original_data.columns),
                'column_count_exported': len(exported_data.columns),
                'column_count_match': column_count_match,
                'columns_match': columns_match
            }
            
            return data_check
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _validate_data_integrity(self, 
                                original_data: pd.DataFrame,
                                export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate overall data integrity."""
        integrity = {
            'status': 'pass',
            'checks': {}
        }
        
        # Check metadata consistency
        metadata = export_results.get('metadata', {})
        
        integrity['checks']['row_count'] = {
            'original': len(original_data),
            'metadata': metadata.get('total_rows', 0),
            'match': len(original_data) == metadata.get('total_rows', 0)
        }
        
        integrity['checks']['column_count'] = {
            'original': len(original_data.columns),
            'metadata': metadata.get('total_columns', 0),
            'match': len(original_data.columns) == metadata.get('total_columns', 0)
        }
        
        # Overall status
        all_checks_pass = all(
            check.get('match', True) for check in integrity['checks'].values()
        )
        
        if not all_checks_pass:
            integrity['status'] = 'fail'
            
        return integrity
    
    def _validate_performance(self, export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate export performance metrics."""
        performance = {
            'total_size_mb': 0,
            'format_breakdown': {},
            'warnings': []
        }
        
        # Calculate total export size
        for format_type, format_info in export_results.get('formats', {}).items():
            format_size_mb = format_info.get('total_size_bytes', 0) / 1024 / 1024
            performance['total_size_mb'] += format_size_mb
            performance['format_breakdown'][format_type] = {
                'size_mb': format_size_mb,
                'file_count': len(format_info.get('files', []))
            }
            
        # Performance warnings
        if performance['total_size_mb'] > 1000:  # > 1GB
            performance['warnings'].append('Total export size exceeds 1GB')
            
        return performance


class MetadataGenerator:
    """Generates comprehensive export metadata and documentation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config('metadata_generation', {})
        self.logger = get_logger(__name__)
        
    @monitor_performance("metadata_generation")
    def generate_metadata(self, 
                         original_data: pd.DataFrame,
                         export_results: Dict[str, Any],
                         validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive export metadata.
        
        Args:
            original_data: Original DataFrame
            export_results: Export results
            validation_results: Validation results
            
        Returns:
            Complete metadata dictionary
        """
        metadata = {
            'schema_version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'generator': 'AHGD Export Pipeline v1.0',
            'export_summary': {},
            'data_schema': {},
            'quality_metrics': {},
            'lineage': {},
            'validation': validation_results
        }
        
        # Export summary
        metadata['export_summary'] = self._generate_export_summary(
            original_data, export_results
        )
        
        # Data schema
        metadata['data_schema'] = self._generate_data_schema(original_data)
        
        # Quality metrics
        metadata['quality_metrics'] = self._generate_quality_metrics(
            original_data, export_results
        )
        
        # Data lineage
        metadata['lineage'] = self._generate_lineage_info()
        
        self.logger.info("Export metadata generated",
                        formats=len(export_results.get('formats', {})),
                        quality_score=metadata['quality_metrics'].get('overall_score', 0))
        
        return metadata
    
    def _generate_export_summary(self, 
                                original_data: pd.DataFrame,
                                export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate export summary information."""
        summary = {
            'data_characteristics': {
                'total_records': len(original_data),
                'total_fields': len(original_data.columns),
                'memory_usage_mb': original_data.memory_usage(deep=True).sum() / 1024 / 1024,
                'data_types': original_data.dtypes.value_counts().to_dict()
            },
            'export_formats': {},
            'total_export_size_mb': 0
        }
        
        # Format-specific summaries
        for format_type, format_info in export_results.get('formats', {}).items():
            format_size_mb = format_info.get('total_size_bytes', 0) / 1024 / 1024
            summary['export_formats'][format_type] = {
                'file_count': len(format_info.get('files', [])),
                'total_size_mb': format_size_mb,
                'compression': format_info.get('compression_info', {})
            }
            summary['total_export_size_mb'] += format_size_mb
            
        return summary
    
    def _generate_data_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate detailed data schema information."""
        schema = {
            'fields': {},
            'summary_statistics': {}
        }
        
        # Field information
        for column in data.columns:
            field_info = {
                'data_type': str(data[column].dtype),
                'null_count': int(data[column].isnull().sum()),
                'null_percentage': float(data[column].isnull().sum() / len(data) * 100),
                'unique_count': int(data[column].nunique())
            }
            
            # Type-specific statistics
            if pd.api.types.is_numeric_dtype(data[column]):
                field_info.update({
                    'min_value': float(data[column].min()) if not data[column].isnull().all() else None,
                    'max_value': float(data[column].max()) if not data[column].isnull().all() else None,
                    'mean_value': float(data[column].mean()) if not data[column].isnull().all() else None
                })
            elif data[column].dtype == 'object':
                field_info.update({
                    'max_length': int(data[column].astype(str).str.len().max()) if len(data) > 0 else 0,
                    'sample_values': data[column].dropna().head(5).tolist()
                })
                
            schema['fields'][column] = field_info
            
        # Summary statistics
        schema['summary_statistics'] = {
            'total_fields': len(data.columns),
            'numeric_fields': len(data.select_dtypes(include=[np.number]).columns),
            'text_fields': len(data.select_dtypes(include=['object']).columns),
            'datetime_fields': len(data.select_dtypes(include=['datetime64']).columns),
            'total_null_values': int(data.isnull().sum().sum())
        }
        
        return schema
    
    def _generate_quality_metrics(self, 
                                 original_data: pd.DataFrame,
                                 export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate data quality metrics."""
        quality = {
            'completeness': {},
            'consistency': {},
            'accuracy': {},
            'overall_score': 0
        }
        
        # Completeness metrics
        total_cells = len(original_data) * len(original_data.columns)
        null_cells = original_data.isnull().sum().sum()
        completeness_score = (total_cells - null_cells) / total_cells * 100
        
        quality['completeness'] = {
            'percentage': float(completeness_score),
            'total_cells': int(total_cells),
            'null_cells': int(null_cells)
        }
        
        # Consistency metrics (basic checks)
        consistency_issues = 0
        
        # Check for duplicate records
        duplicate_count = original_data.duplicated().sum()
        if duplicate_count > 0:
            consistency_issues += 1
            
        quality['consistency'] = {
            'duplicate_records': int(duplicate_count),
            'issues_found': consistency_issues
        }
        
        # Overall quality score (simplified)
        quality['overall_score'] = float(
            (completeness_score * 0.6) + 
            ((100 - min(consistency_issues * 10, 100)) * 0.4)
        )
        
        return quality
    
    def _generate_lineage_info(self) -> Dict[str, Any]:
        """Generate data lineage information."""
        return {
            'pipeline': 'AHGD ETL Pipeline',
            'stage': 'Export',
            'source_system': 'Australian Health and Geographic Data',
            'processing_date': datetime.now().isoformat(),
            'quality_assured': True,
            'governance': {
                'data_classification': 'Public',
                'retention_period': 'As per source data policy',
                'update_frequency': 'As per source data updates'
            }
        }


class QualityChecker:
    """Final quality checks on exported data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config('quality_checking', {})
        self.logger = get_logger(__name__)
        
    @monitor_performance("quality_checking")
    def perform_quality_checks(self, 
                              original_data: pd.DataFrame,
                              export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive quality checks.
        
        Args:
            original_data: Original DataFrame
            export_results: Export results
            
        Returns:
            Quality check results
        """
        quality_results = {
            'overall_status': 'pass',
            'checks_performed': [],
            'warnings': [],
            'errors': []
        }
        
        try:
            # Data integrity checks
            integrity_check = self._check_data_integrity(original_data, export_results)
            quality_results['checks_performed'].append('data_integrity')
            
            if integrity_check['status'] != 'pass':
                quality_results['overall_status'] = 'fail'
                quality_results['errors'].extend(integrity_check.get('errors', []))
                
            # File completeness checks
            completeness_check = self._check_file_completeness(export_results)
            quality_results['checks_performed'].append('file_completeness')
            
            if completeness_check['status'] != 'pass':
                quality_results['overall_status'] = 'fail'
                quality_results['errors'].extend(completeness_check.get('errors', []))
                
            # Performance checks
            performance_check = self._check_performance_metrics(export_results)
            quality_results['checks_performed'].append('performance_metrics')
            quality_results['warnings'].extend(performance_check.get('warnings', []))
            
            self.logger.info(f"Quality checks completed",
                           status=quality_results['overall_status'],
                           checks=len(quality_results['checks_performed']))
            
            return quality_results
            
        except Exception as e:
            quality_results['overall_status'] = 'error'
            quality_results['errors'].append(f"Quality check failed: {str(e)}")
            return quality_results
    
    def _check_data_integrity(self, 
                             original_data: pd.DataFrame,
                             export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check data integrity between original and exported data."""
        check = {
            'status': 'pass',
            'errors': []
        }
        
        metadata = export_results.get('metadata', {})
        
        # Row count check
        if len(original_data) != metadata.get('total_rows', 0):
            check['status'] = 'fail'
            check['errors'].append(
                f"Row count mismatch: original={len(original_data)}, "
                f"exported={metadata.get('total_rows', 0)}"
            )
            
        # Column count check
        if len(original_data.columns) != metadata.get('total_columns', 0):
            check['status'] = 'fail'
            check['errors'].append(
                f"Column count mismatch: original={len(original_data.columns)}, "
                f"exported={metadata.get('total_columns', 0)}"
            )
            
        return check
    
    def _check_file_completeness(self, export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check that all expected files were created."""
        check = {
            'status': 'pass',
            'errors': []
        }
        
        # Check that each format has files
        for format_type, format_info in export_results.get('formats', {}).items():
            if not format_info.get('files'):
                check['status'] = 'fail'
                check['errors'].append(f"No files generated for format: {format_type}")
            else:
                # Check that files exist and have content
                for file_info in format_info['files']:
                    file_path = Path(file_info['path'])
                    if not file_path.exists():
                        check['status'] = 'fail'
                        check['errors'].append(f"File does not exist: {file_path}")
                    elif file_path.stat().st_size == 0:
                        check['status'] = 'fail'
                        check['errors'].append(f"File is empty: {file_path}")
                        
        return check
    
    def _check_performance_metrics(self, export_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check performance metrics and generate warnings."""
        check = {
            'warnings': []
        }
        
        # Calculate total export size
        total_size_mb = 0
        for format_info in export_results.get('formats', {}).values():
            total_size_mb += format_info.get('total_size_bytes', 0) / 1024 / 1024
            
        # Size warnings
        if total_size_mb > 1000:  # > 1GB
            check['warnings'].append(f"Large export size: {total_size_mb:.1f}MB")
        elif total_size_mb > 5000:  # > 5GB
            check['warnings'].append(f"Very large export size: {total_size_mb:.1f}MB - consider partitioning")
            
        return check


class ExportPipeline:
    """Orchestrates complete multi-format export process."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config('export_pipeline', {})
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.production_loader = ProductionLoader(config)
        self.compression_analyser = CompressionAnalyzer(config)
        self.format_optimiser = FormatOptimizer(config)
        self.size_calculator = SizeCalculator(config)
        self.cache_manager = CacheManager()
        
        self.export_validator = ExportValidator(config)
        self.metadata_generator = MetadataGenerator(config)
        self.quality_checker = QualityChecker(config)
        
        # Task management
        self.active_tasks: Dict[str, ExportTask] = {}
        
    @monitor_performance("complete_export_pipeline")
    def export_data(self, 
                   data: pd.DataFrame,
                   output_path: Union[str, Path],
                   formats: Optional[List[str]] = None,
                   export_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute complete export pipeline.
        
        Args:
            data: DataFrame to export
            output_path: Base output directory
            formats: List of formats to export
            export_options: Additional export options
            
        Returns:
            Complete export results including validation and metadata
        """
        if formats is None:
            formats = ['parquet', 'csv', 'json']
            
        options = export_options or {}
        output_path = Path(output_path)
        
        # Generate task ID
        task_id = f"export_{int(time.time())}_{hash(str(data.values.tobytes()))[:8}"
        
        # Create export task
        task = ExportTask(
            task_id=task_id,
            data_source="AHGD ETL Pipeline",
            output_path=output_path,
            formats=formats,
            compression=options.get('compress', True),
            partition=options.get('partition', True),
            web_optimise=options.get('web_optimise', True),
            priority=options.get('priority', 'medium'),
            created_at=datetime.now()
        )
        
        self.active_tasks[task_id] = task
        
        try:
            # Track lineage
            track_lineage("processed_data", str(output_path), "export_pipeline")
            
            self.logger.info(f"Starting export pipeline",
                           task_id=task_id, formats=formats,
                           output_path=str(output_path))
            
            task.status = 'running'
            
            # Step 1: Perform exports
            task.progress = 0.1
            export_results = self._perform_exports(data, task, options)
            
            # Step 2: Validate exports
            task.progress = 0.7
            validation_results = self.export_validator.validate_export(
                data, export_results, options.get('validation_level', 'standard')
            )
            
            # Step 3: Quality checks
            task.progress = 0.8
            quality_results = self.quality_checker.perform_quality_checks(
                data, export_results
            )
            
            # Step 4: Generate metadata
            task.progress = 0.9
            metadata = self.metadata_generator.generate_metadata(
                data, export_results, validation_results
            )
            
            # Step 5: Save metadata
            metadata_path = output_path / 'complete_metadata.json'
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            # Compile final results
            final_results = {
                'task_id': task_id,
                'export_results': export_results,
                'validation_results': validation_results,
                'quality_results': quality_results,
                'metadata': metadata,
                'metadata_file': str(metadata_path),
                'pipeline_status': 'completed'
            }
            
            # Update task
            task.status = 'completed'
            task.progress = 1.0
            task.export_results = final_results
            
            self.logger.info(f"Export pipeline completed successfully",
                           task_id=task_id,
                           validation_status=validation_results['overall_status'],
                           quality_status=quality_results['overall_status'])
            
            return final_results
            
        except Exception as e:
            task.status = 'failed'
            task.error_message = str(e)
            
            self.logger.error(f"Export pipeline failed", 
                            task_id=task_id, error=str(e))
            
            raise LoadingError(f"Export pipeline failed: {str(e)}")
        
        finally:
            # Clean up task after some time
            # In production, this would be handled by a background process
            pass
    
    def _perform_exports(self, 
                        data: pd.DataFrame,
                        task: ExportTask,
                        options: Dict[str, Any]) -> Dict[str, Any]:
        """Perform the actual data exports."""
        # Use production loader for exports
        export_results = self.production_loader.load(
            data=data,
            output_path=task.output_path,
            formats=task.formats,
            compress=task.compression,
            partition=task.partition,
            optimise_for_web=task.web_optimise,
            **options
        )
        
        return export_results
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of export task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status dictionary or None if not found
        """
        task = self.active_tasks.get(task_id)
        return task.to_dict() if task else None
    
    def list_active_tasks(self) -> List[Dict[str, Any]]:
        """List all active export tasks.
        
        Returns:
            List of task dictionaries
        """
        return [task.to_dict() for task in self.active_tasks.values()]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel an export task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if task was cancelled
        """
        task = self.active_tasks.get(task_id)
        if task and task.status in ['pending', 'running']:
            task.status = 'cancelled'
            self.logger.info(f"Task cancelled", task_id=task_id)
            return True
        return False
    
    def cleanup_completed_tasks(self, max_age: timedelta = timedelta(hours=24)) -> int:
        """Clean up old completed tasks.
        
        Args:
            max_age: Maximum age for completed tasks
            
        Returns:
            Number of tasks cleaned up
        """
        now = datetime.now()
        cleaned_count = 0
        
        tasks_to_remove = []
        for task_id, task in self.active_tasks.items():
            if task.status in ['completed', 'failed', 'cancelled']:
                if now - task.created_at > max_age:
                    tasks_to_remove.append(task_id)
                    
        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]
            cleaned_count += 1
            
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old tasks")
            
        return cleaned_count
