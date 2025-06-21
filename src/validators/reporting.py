"""
Validation Reporting Framework

This module provides comprehensive validation reporting capabilities including
HTML/PDF quality reports, data profiling reports, validation dashboard data,
and export capabilities for the AHGD validation framework.
"""

import logging
import json
import csv
import io
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, IO
from dataclasses import dataclass, field, asdict
import base64
import statistics

from ..utils.interfaces import (
    DataBatch, 
    DataRecord, 
    ValidationResult, 
    ValidationSeverity,
    ProcessingStatus
)


@dataclass
class ValidationSummary:
    """Summary of validation results."""
    total_records: int
    total_validations: int
    error_count: int
    warning_count: int
    info_count: int
    success_rate: float
    validation_coverage: float
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None


@dataclass
class DataProfileSummary:
    """Summary of data profiling results."""
    column_name: str
    data_type: str
    total_values: int
    non_null_values: int
    null_values: int
    completeness_rate: float
    unique_values: int
    uniqueness_rate: float
    min_value: Optional[Union[str, float]] = None
    max_value: Optional[Union[str, float]] = None
    mean_value: Optional[float] = None
    median_value: Optional[float] = None
    std_deviation: Optional[float] = None
    most_common_values: List[Tuple[Any, int]] = field(default_factory=list)


@dataclass
class QualityDimension:
    """Quality dimension assessment."""
    dimension_name: str
    score: float
    description: str
    issues_found: int
    recommendations: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    report_id: str
    report_type: str
    generated_at: datetime
    data_summary: Dict[str, Any]
    validation_summary: ValidationSummary
    quality_dimensions: List[QualityDimension]
    data_profiles: List[DataProfileSummary]
    validation_results: List[ValidationResult]
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationReporter:
    """
    Comprehensive validation reporting framework.
    
    This class provides various reporting formats and capabilities for
    validation results including HTML reports, data profiling, dashboard
    data preparation, and export functionality.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialise the validation reporter.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Reporting configuration
        self.reporting_config = self.config.get('reporting', {})
        self.include_sample_records = self.reporting_config.get('include_sample_records', True)
        self.max_sample_size = self.reporting_config.get('max_sample_size', 10)
        self.generate_charts = self.reporting_config.get('generate_charts', True)
        self.export_formats = self.reporting_config.get('export_formats', ['html', 'json'])
        
        # Quality scoring weights
        self.quality_weights = self.config.get('quality_weights', {
            'completeness': 0.25,
            'validity': 0.25,
            'consistency': 0.20,
            'accuracy': 0.15,
            'timeliness': 0.10,
            'uniqueness': 0.05
        })
        
    def generate_comprehensive_report(
        self,
        data: DataBatch,
        validation_results: List[ValidationResult],
        report_type: str = "comprehensive",
        include_profiling: bool = True,
        include_recommendations: bool = True
    ) -> ValidationReport:
        """
        Generate a comprehensive validation report.
        
        Args:
            data: Original data batch
            validation_results: Validation results
            report_type: Type of report to generate
            include_profiling: Whether to include data profiling
            include_recommendations: Whether to include recommendations
            
        Returns:
            ValidationReport: Comprehensive validation report
        """
        report_id = self._generate_report_id()
        generated_at = datetime.now()
        
        # Generate data summary
        data_summary = self._generate_data_summary(data)
        
        # Generate validation summary
        validation_summary = self._generate_validation_summary(validation_results, generated_at)
        
        # Generate quality dimensions assessment
        quality_dimensions = self._assess_quality_dimensions(validation_results, data)
        
        # Generate data profiles if requested
        data_profiles = []
        if include_profiling:
            data_profiles = self._generate_data_profiles(data)
        
        # Create comprehensive report
        report = ValidationReport(
            report_id=report_id,
            report_type=report_type,
            generated_at=generated_at,
            data_summary=data_summary,
            validation_summary=validation_summary,
            quality_dimensions=quality_dimensions,
            data_profiles=data_profiles,
            validation_results=validation_results,
            metadata={
                'include_profiling': include_profiling,
                'include_recommendations': include_recommendations,
                'config_used': self.config
            }
        )
        
        self.logger.info(f"Generated comprehensive report {report_id} with {len(validation_results)} validation results")
        
        return report
    
    def export_report_html(self, report: ValidationReport) -> str:
        """
        Export validation report as HTML.
        
        Args:
            report: Validation report to export
            
        Returns:
            str: HTML content
        """
        html_content = self._generate_html_report(report)
        self.logger.info(f"Generated HTML report for {report.report_id}")
        return html_content
    
    def export_report_json(self, report: ValidationReport) -> str:
        """
        Export validation report as JSON.
        
        Args:
            report: Validation report to export
            
        Returns:
            str: JSON content
        """
        # Convert report to dictionary with proper serialisation
        report_dict = self._serialize_report_to_dict(report)
        json_content = json.dumps(report_dict, indent=2, default=str)
        
        self.logger.info(f"Generated JSON report for {report.report_id}")
        return json_content
    
    def export_report_csv(self, report: ValidationReport) -> str:
        """
        Export validation results as CSV.
        
        Args:
            report: Validation report to export
            
        Returns:
            str: CSV content
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'Rule ID', 'Severity', 'Is Valid', 'Message', 
            'Affected Records Count', 'Timestamp'
        ])
        
        # Write validation results
        for result in report.validation_results:
            writer.writerow([
                result.rule_id,
                result.severity.value,
                result.is_valid,
                result.message,
                len(result.affected_records),
                result.timestamp.isoformat()
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        self.logger.info(f"Generated CSV report for {report.report_id}")
        return csv_content
    
    def generate_dashboard_data(
        self,
        validation_results: List[ValidationResult],
        data: Optional[DataBatch] = None
    ) -> Dict[str, Any]:
        """
        Generate data for validation dashboard.
        
        Args:
            validation_results: Validation results
            data: Optional data batch for additional metrics
            
        Returns:
            Dict[str, Any]: Dashboard data
        """
        dashboard_data = {
            'summary': self._generate_dashboard_summary(validation_results),
            'severity_breakdown': self._generate_severity_breakdown(validation_results),
            'rule_breakdown': self._generate_rule_breakdown(validation_results),
            'temporal_trends': self._generate_temporal_trends(validation_results),
            'quality_scores': self._generate_quality_scores(validation_results, data),
            'top_issues': self._generate_top_issues(validation_results),
            'recommendations': self._generate_recommendations(validation_results)
        }
        
        if data:
            dashboard_data['data_metrics'] = self._generate_data_metrics(data)
        
        self.logger.info(f"Generated dashboard data with {len(validation_results)} validation results")
        
        return dashboard_data
    
    def generate_executive_summary(
        self,
        report: ValidationReport
    ) -> Dict[str, Any]:
        """
        Generate executive summary of validation results.
        
        Args:
            report: Validation report
            
        Returns:
            Dict[str, Any]: Executive summary
        """
        summary = {
            'report_id': report.report_id,
            'generated_at': report.generated_at.isoformat(),
            'data_overview': {
                'total_records': report.data_summary.get('total_records', 0),
                'columns_analyzed': report.data_summary.get('total_columns', 0),
                'data_size_mb': report.data_summary.get('estimated_size_mb', 0)
            },
            'validation_overview': {
                'total_validations': report.validation_summary.total_validations,
                'success_rate': f"{report.validation_summary.success_rate:.1%}",
                'issues_found': report.validation_summary.error_count + report.validation_summary.warning_count,
                'critical_issues': report.validation_summary.error_count,
                'warnings': report.validation_summary.warning_count
            },
            'quality_assessment': {
                'overall_quality': self._calculate_overall_quality_score(report.quality_dimensions),
                'quality_grade': self._determine_quality_grade(report.quality_dimensions),
                'top_quality_issues': self._get_top_quality_issues(report.validation_results),
                'recommendations_count': sum(len(qd.recommendations) for qd in report.quality_dimensions)
            },
            'risk_assessment': {
                'data_quality_risk': self._assess_data_quality_risk(report),
                'compliance_risk': self._assess_compliance_risk(report),
                'operational_risk': self._assess_operational_risk(report)
            }
        }
        
        return summary
    
    # Private methods for report generation
    
    def _generate_report_id(self) -> str:
        """Generate unique report ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"validation_report_{timestamp}"
    
    def _generate_data_summary(self, data: DataBatch) -> Dict[str, Any]:
        """Generate summary of the data."""
        if not data:
            return {'total_records': 0, 'total_columns': 0}
        
        sample_record = data[0] if data else {}
        columns = list(sample_record.keys())
        
        # Estimate data size
        estimated_size_bytes = len(str(data[:100])) * (len(data) / 100) if len(data) > 100 else len(str(data))
        estimated_size_mb = estimated_size_bytes / (1024 * 1024)
        
        return {
            'total_records': len(data),
            'total_columns': len(columns),
            'columns': columns,
            'estimated_size_mb': round(estimated_size_mb, 2),
            'sample_record': sample_record if self.include_sample_records else None
        }
    
    def _generate_validation_summary(
        self,
        validation_results: List[ValidationResult],
        start_time: datetime
    ) -> ValidationSummary:
        """Generate validation summary."""
        error_count = sum(1 for r in validation_results if r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in validation_results if r.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for r in validation_results if r.severity == ValidationSeverity.INFO)
        
        total_validations = len(validation_results)
        success_rate = 1.0 - (error_count / total_validations) if total_validations > 0 else 1.0
        
        return ValidationSummary(
            total_records=len(set(
                record_id 
                for result in validation_results 
                for record_id in result.affected_records
            )),
            total_validations=total_validations,
            error_count=error_count,
            warning_count=warning_count,
            info_count=info_count,
            success_rate=success_rate,
            validation_coverage=1.0,  # Assume full coverage
            start_time=start_time,
            end_time=datetime.now()
        )
    
    def _assess_quality_dimensions(
        self,
        validation_results: List[ValidationResult],
        data: DataBatch
    ) -> List[QualityDimension]:
        """Assess quality across different dimensions."""
        dimensions = []
        
        # Group results by dimension
        dimension_results = defaultdict(list)
        
        for result in validation_results:
            # Categorise results by quality dimension
            dimension = self._categorise_by_quality_dimension(result)
            dimension_results[dimension].append(result)
        
        # Assess each dimension
        for dimension_name, results in dimension_results.items():
            score = self._calculate_dimension_score(results, data)
            issues_found = len([r for r in results if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.WARNING]])
            
            dimensions.append(QualityDimension(
                dimension_name=dimension_name,
                score=score,
                description=self._get_dimension_description(dimension_name),
                issues_found=issues_found,
                recommendations=self._generate_dimension_recommendations(dimension_name, results)
            ))
        
        return dimensions
    
    def _generate_data_profiles(self, data: DataBatch) -> List[DataProfileSummary]:
        """Generate data profiling summaries."""
        if not data:
            return []
        
        profiles = []
        sample_record = data[0]
        
        for column in sample_record.keys():
            profile = self._profile_column(data, column)
            profiles.append(profile)
        
        return profiles
    
    def _profile_column(self, data: DataBatch, column: str) -> DataProfileSummary:
        """Profile a single column."""
        values = [record.get(column) for record in data]
        non_null_values = [v for v in values if v is not None and v != ""]
        
        # Basic statistics
        total_values = len(values)
        non_null_count = len(non_null_values)
        null_count = total_values - non_null_count
        completeness_rate = non_null_count / total_values if total_values > 0 else 0
        
        unique_values = len(set(str(v) for v in non_null_values))
        uniqueness_rate = unique_values / non_null_count if non_null_count > 0 else 0
        
        # Data type inference
        data_type = self._infer_data_type(non_null_values)
        
        # Statistical measures for numeric data
        min_value = max_value = mean_value = median_value = std_deviation = None
        
        if data_type in ['integer', 'float'] and non_null_values:
            try:
                numeric_values = [float(v) for v in non_null_values if isinstance(v, (int, float))]
                if numeric_values:
                    min_value = min(numeric_values)
                    max_value = max(numeric_values)
                    mean_value = statistics.mean(numeric_values)
                    median_value = statistics.median(numeric_values)
                    if len(numeric_values) > 1:
                        std_deviation = statistics.stdev(numeric_values)
            except (ValueError, TypeError):
                pass
        
        # Most common values
        value_counts = Counter(str(v) for v in non_null_values)
        most_common_values = value_counts.most_common(5)
        
        return DataProfileSummary(
            column_name=column,
            data_type=data_type,
            total_values=total_values,
            non_null_values=non_null_count,
            null_values=null_count,
            completeness_rate=completeness_rate,
            unique_values=unique_values,
            uniqueness_rate=uniqueness_rate,
            min_value=min_value,
            max_value=max_value,
            mean_value=mean_value,
            median_value=median_value,
            std_deviation=std_deviation,
            most_common_values=most_common_values
        )
    
    def _generate_html_report(self, report: ValidationReport) -> str:
        """Generate HTML report content."""
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Quality Validation Report - {report.report_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .card {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; }}
        .metric {{ font-size: 2em; font-weight: bold; color: #495057; }}
        .label {{ color: #6c757d; font-size: 0.9em; }}
        .quality-score {{ text-align: center; }}
        .score-circle {{ width: 100px; height: 100px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; margin: 10px; }}
        .excellent {{ background: #28a745; color: white; }}
        .good {{ background: #17a2b8; color: white; }}
        .acceptable {{ background: #ffc107; color: black; }}
        .poor {{ background: #fd7e14; color: white; }}
        .critical {{ background: #dc3545; color: white; }}
        .validation-results {{ margin: 20px 0; }}
        .result-item {{ margin: 10px 0; padding: 10px; border-left: 4px solid; }}
        .error {{ border-left-color: #dc3545; background: #f8d7da; }}
        .warning {{ border-left-color: #ffc107; background: #fff3cd; }}
        .info {{ border-left-color: #17a2b8; background: #d1ecf1; }}
        .data-profiles {{ margin: 20px 0; }}
        .profile-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #dee2e6; padding: 8px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
        .recommendations {{ background: #e8f5e8; border: 1px solid #c3e6c3; border-radius: 8px; padding: 15px; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Data Quality Validation Report</h1>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Report Type:</strong> {report.report_type.title()}</p>
    </div>
    
    <div class="summary">
        <div class="card">
            <div class="metric">{report.data_summary.get('total_records', 0):,}</div>
            <div class="label">Records Analyzed</div>
        </div>
        <div class="card">
            <div class="metric">{report.validation_summary.total_validations:,}</div>
            <div class="label">Validations Performed</div>
        </div>
        <div class="card">
            <div class="metric">{report.validation_summary.error_count:,}</div>
            <div class="label">Critical Issues</div>
        </div>
        <div class="card">
            <div class="metric">{report.validation_summary.warning_count:,}</div>
            <div class="label">Warnings</div>
        </div>
        <div class="card quality-score">
            <div class="score-circle {self._get_quality_grade_class(report.quality_dimensions)}">
                <div>
                    <div class="metric">{self._calculate_overall_quality_score(report.quality_dimensions):.0%}</div>
                    <div class="label">Quality Score</div>
                </div>
            </div>
        </div>
    </div>
    
    {self._generate_quality_dimensions_html(report.quality_dimensions)}
    
    {self._generate_data_profiles_html(report.data_profiles)}
    
    {self._generate_validation_results_html(report.validation_results)}
    
    {self._generate_recommendations_html(report.quality_dimensions)}
    
    <div class="card">
        <h3>Report Metadata</h3>
        <table>
            <tr><th>Data Columns</th><td>{report.data_summary.get('total_columns', 0)}</td></tr>
            <tr><th>Data Size (MB)</th><td>{report.data_summary.get('estimated_size_mb', 0):.2f}</td></tr>
            <tr><th>Processing Duration</th><td>{getattr(report.validation_summary, 'duration_seconds', 0):.2f} seconds</td></tr>
            <tr><th>Success Rate</th><td>{report.validation_summary.success_rate:.1%}</td></tr>
        </table>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _generate_quality_dimensions_html(self, quality_dimensions: List[QualityDimension]) -> str:
        """Generate HTML for quality dimensions."""
        if not quality_dimensions:
            return ""
        
        html = '<div class="card"><h3>Quality Dimensions Assessment</h3><div class="profile-grid">'
        
        for dimension in quality_dimensions:
            score_class = self._get_score_class(dimension.score)
            html += f'''
            <div class="card">
                <h4>{dimension.dimension_name.title()}</h4>
                <div class="score-circle {score_class}">
                    <div class="metric">{dimension.score:.0%}</div>
                </div>
                <p>{dimension.description}</p>
                <p><strong>Issues Found:</strong> {dimension.issues_found}</p>
            </div>
            '''
        
        html += '</div></div>'
        return html
    
    def _generate_data_profiles_html(self, data_profiles: List[DataProfileSummary]) -> str:
        """Generate HTML for data profiles."""
        if not data_profiles:
            return ""
        
        html = '<div class="data-profiles"><h3>Data Profiling Results</h3><div class="profile-grid">'
        
        for profile in data_profiles[:10]:  # Limit to first 10 columns
            html += f'''
            <div class="card">
                <h4>{profile.column_name}</h4>
                <table>
                    <tr><th>Data Type</th><td>{profile.data_type}</td></tr>
                    <tr><th>Completeness</th><td>{profile.completeness_rate:.1%}</td></tr>
                    <tr><th>Uniqueness</th><td>{profile.uniqueness_rate:.1%}</td></tr>
                    <tr><th>Non-null Values</th><td>{profile.non_null_values:,}</td></tr>
                    <tr><th>Unique Values</th><td>{profile.unique_values:,}</td></tr>
            '''
            
            if profile.min_value is not None:
                html += f'<tr><th>Min Value</th><td>{profile.min_value}</td></tr>'
            if profile.max_value is not None:
                html += f'<tr><th>Max Value</th><td>{profile.max_value}</td></tr>'
            if profile.mean_value is not None:
                html += f'<tr><th>Mean</th><td>{profile.mean_value:.2f}</td></tr>'
            
            html += '</table></div>'
        
        html += '</div></div>'
        return html
    
    def _generate_validation_results_html(self, validation_results: List[ValidationResult]) -> str:
        """Generate HTML for validation results."""
        if not validation_results:
            return ""
        
        html = '<div class="validation-results"><h3>Validation Results</h3>'
        
        # Group by severity
        errors = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
        warnings = [r for r in validation_results if r.severity == ValidationSeverity.WARNING]
        
        # Show top errors
        if errors:
            html += '<h4>Critical Issues</h4>'
            for result in errors[:5]:  # Top 5 errors
                html += f'''
                <div class="result-item error">
                    <strong>{result.rule_id}</strong>: {result.message}
                    <br><small>Affected records: {len(result.affected_records)}</small>
                </div>
                '''
        
        # Show top warnings
        if warnings:
            html += '<h4>Warnings</h4>'
            for result in warnings[:5]:  # Top 5 warnings
                html += f'''
                <div class="result-item warning">
                    <strong>{result.rule_id}</strong>: {result.message}
                    <br><small>Affected records: {len(result.affected_records)}</small>
                </div>
                '''
        
        html += '</div>'
        return html
    
    def _generate_recommendations_html(self, quality_dimensions: List[QualityDimension]) -> str:
        """Generate HTML for recommendations."""
        all_recommendations = []
        for dimension in quality_dimensions:
            all_recommendations.extend(dimension.recommendations)
        
        if not all_recommendations:
            return ""
        
        html = '<div class="recommendations"><h3>Recommendations</h3><ul>'
        
        for recommendation in all_recommendations[:10]:  # Top 10 recommendations
            html += f'<li>{recommendation}</li>'
        
        html += '</ul></div>'
        return html
    
    def _serialize_report_to_dict(self, report: ValidationReport) -> Dict[str, Any]:
        """Serialize report to dictionary for JSON export."""
        return {
            'report_id': report.report_id,
            'report_type': report.report_type,
            'generated_at': report.generated_at.isoformat(),
            'data_summary': report.data_summary,
            'validation_summary': asdict(report.validation_summary),
            'quality_dimensions': [asdict(qd) for qd in report.quality_dimensions],
            'data_profiles': [asdict(dp) for dp in report.data_profiles],
            'validation_results': [
                {
                    'rule_id': r.rule_id,
                    'severity': r.severity.value,
                    'is_valid': r.is_valid,
                    'message': r.message,
                    'details': r.details,
                    'affected_records': r.affected_records,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in report.validation_results
            ],
            'performance_metrics': report.performance_metrics,
            'metadata': report.metadata
        }
    
    def _categorise_by_quality_dimension(self, result: ValidationResult) -> str:
        """Categorise validation result by quality dimension."""
        rule_id = result.rule_id.lower()
        
        if 'completeness' in rule_id or 'missing' in rule_id or 'null' in rule_id:
            return 'completeness'
        elif 'validity' in rule_id or 'format' in rule_id or 'pattern' in rule_id:
            return 'validity'
        elif 'consistency' in rule_id or 'relationship' in rule_id:
            return 'consistency'
        elif 'accuracy' in rule_id or 'reference' in rule_id:
            return 'accuracy'
        elif 'timeliness' in rule_id or 'freshness' in rule_id or 'currency' in rule_id:
            return 'timeliness'
        elif 'uniqueness' in rule_id or 'duplicate' in rule_id:
            return 'uniqueness'
        else:
            return 'other'
    
    def _calculate_dimension_score(self, results: List[ValidationResult], data: DataBatch) -> float:
        """Calculate quality score for a dimension."""
        if not results:
            return 1.0
        
        # Weight errors more heavily than warnings
        error_count = sum(1 for r in results if r.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for r in results if r.severity == ValidationSeverity.WARNING)
        
        total_issues = error_count * 2 + warning_count  # Errors count double
        max_possible_issues = len(data) if data else len(results)
        
        if max_possible_issues == 0:
            return 1.0
        
        score = max(0.0, 1.0 - (total_issues / max_possible_issues))
        return score
    
    def _get_dimension_description(self, dimension_name: str) -> str:
        """Get description for quality dimension."""
        descriptions = {
            'completeness': 'Measures the extent to which data is present and not missing',
            'validity': 'Measures whether data values conform to defined formats and constraints',
            'consistency': 'Measures whether data is uniform and follows business rules',
            'accuracy': 'Measures how well data represents real-world values',
            'timeliness': 'Measures whether data is up-to-date and available when needed',
            'uniqueness': 'Measures the absence of duplicate records and values',
            'other': 'Other data quality aspects not covered by standard dimensions'
        }
        return descriptions.get(dimension_name, 'Quality dimension assessment')
    
    def _generate_dimension_recommendations(
        self, 
        dimension_name: str, 
        results: List[ValidationResult]
    ) -> List[str]:
        """Generate recommendations for a quality dimension."""
        recommendations = []
        
        if dimension_name == 'completeness':
            null_issues = [r for r in results if 'null' in r.rule_id.lower() or 'missing' in r.rule_id.lower()]
            if null_issues:
                recommendations.append("Implement data collection improvements to reduce missing values")
                recommendations.append("Consider default value strategies for optional fields")
        
        elif dimension_name == 'validity':
            format_issues = [r for r in results if 'format' in r.rule_id.lower() or 'pattern' in r.rule_id.lower()]
            if format_issues:
                recommendations.append("Implement input validation at data entry points")
                recommendations.append("Review and update data format specifications")
        
        elif dimension_name == 'consistency':
            consistency_issues = [r for r in results if 'consistency' in r.rule_id.lower()]
            if consistency_issues:
                recommendations.append("Establish clear business rules and validation constraints")
                recommendations.append("Implement cross-field validation checks")
        
        elif dimension_name == 'uniqueness':
            duplicate_issues = [r for r in results if 'duplicate' in r.rule_id.lower()]
            if duplicate_issues:
                recommendations.append("Implement duplicate detection and resolution processes")
                recommendations.append("Review data integration procedures to prevent duplicates")
        
        return recommendations
    
    def _infer_data_type(self, values: List[Any]) -> str:
        """Infer data type from values."""
        if not values:
            return 'unknown'
        
        # Check if all values are integers
        try:
            all(isinstance(v, int) or (isinstance(v, str) and v.isdigit()) for v in values[:100])
            return 'integer'
        except:
            pass
        
        # Check if all values are floats
        try:
            all(isinstance(v, (int, float)) or (isinstance(v, str) and float(v)) for v in values[:100])
            return 'float'
        except:
            pass
        
        # Check if values look like dates
        date_indicators = ['date', 'time', 'year', '20', '19']
        if any(indicator in str(values[0]).lower() for indicator in date_indicators):
            return 'date'
        
        # Default to string
        return 'string'
    
    def _calculate_overall_quality_score(self, quality_dimensions: List[QualityDimension]) -> float:
        """Calculate overall quality score from dimensions."""
        if not quality_dimensions:
            return 0.0
        
        # Weight by configured weights
        total_score = 0.0
        total_weight = 0.0
        
        for dimension in quality_dimensions:
            weight = self.quality_weights.get(dimension.dimension_name, 0.1)
            total_score += dimension.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_quality_grade(self, quality_dimensions: List[QualityDimension]) -> str:
        """Determine quality grade based on dimensions."""
        overall_score = self._calculate_overall_quality_score(quality_dimensions)
        
        if overall_score >= 0.95:
            return "EXCELLENT"
        elif overall_score >= 0.85:
            return "GOOD"
        elif overall_score >= 0.70:
            return "ACCEPTABLE"
        elif overall_score >= 0.50:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _get_quality_grade_class(self, quality_dimensions: List[QualityDimension]) -> str:
        """Get CSS class for quality grade."""
        grade = self._determine_quality_grade(quality_dimensions)
        return grade.lower()
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score."""
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.70:
            return "acceptable"
        elif score >= 0.50:
            return "poor"
        else:
            return "critical"
    
    def _get_top_quality_issues(self, validation_results: List[ValidationResult]) -> List[str]:
        """Get top quality issues."""
        error_results = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
        return [r.rule_id for r in error_results[:5]]
    
    def _assess_data_quality_risk(self, report: ValidationReport) -> str:
        """Assess data quality risk level."""
        error_rate = report.validation_summary.error_count / max(report.validation_summary.total_validations, 1)
        
        if error_rate > 0.1:
            return "HIGH"
        elif error_rate > 0.05:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_compliance_risk(self, report: ValidationReport) -> str:
        """Assess compliance risk level."""
        # Look for business rule violations
        business_rule_errors = [
            r for r in report.validation_results 
            if r.severity == ValidationSeverity.ERROR and 'business' in r.rule_id.lower()
        ]
        
        if len(business_rule_errors) > 0:
            return "HIGH"
        else:
            return "LOW"
    
    def _assess_operational_risk(self, report: ValidationReport) -> str:
        """Assess operational risk level."""
        # Base on completeness and validity issues
        completeness_issues = sum(
            1 for r in report.validation_results 
            if 'completeness' in r.rule_id.lower() and r.severity == ValidationSeverity.ERROR
        )
        
        if completeness_issues > 10:
            return "HIGH"
        elif completeness_issues > 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    # Dashboard data generation methods
    
    def _generate_dashboard_summary(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate dashboard summary."""
        return {
            'total_validations': len(validation_results),
            'errors': sum(1 for r in validation_results if r.severity == ValidationSeverity.ERROR),
            'warnings': sum(1 for r in validation_results if r.severity == ValidationSeverity.WARNING),
            'info': sum(1 for r in validation_results if r.severity == ValidationSeverity.INFO),
            'affected_records': len(set(
                record_id 
                for result in validation_results 
                for record_id in result.affected_records
            ))
        }
    
    def _generate_severity_breakdown(self, validation_results: List[ValidationResult]) -> Dict[str, int]:
        """Generate severity breakdown for dashboard."""
        return {
            'ERROR': sum(1 for r in validation_results if r.severity == ValidationSeverity.ERROR),
            'WARNING': sum(1 for r in validation_results if r.severity == ValidationSeverity.WARNING),
            'INFO': sum(1 for r in validation_results if r.severity == ValidationSeverity.INFO)
        }
    
    def _generate_rule_breakdown(self, validation_results: List[ValidationResult]) -> Dict[str, int]:
        """Generate rule breakdown for dashboard."""
        rule_counts = Counter(r.rule_id for r in validation_results)
        return dict(rule_counts.most_common(10))
    
    def _generate_temporal_trends(self, validation_results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """Generate temporal trends for dashboard."""
        # Group by hour for recent results
        hourly_counts = defaultdict(int)
        
        for result in validation_results:
            hour_key = result.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_counts[hour_key] += 1
        
        return [
            {'timestamp': hour, 'count': count}
            for hour, count in sorted(hourly_counts.items())
        ]
    
    def _generate_quality_scores(
        self, 
        validation_results: List[ValidationResult], 
        data: Optional[DataBatch]
    ) -> Dict[str, float]:
        """Generate quality scores for dashboard."""
        dimension_results = defaultdict(list)
        
        for result in validation_results:
            dimension = self._categorise_by_quality_dimension(result)
            dimension_results[dimension].append(result)
        
        scores = {}
        for dimension, results in dimension_results.items():
            scores[dimension] = self._calculate_dimension_score(results, data)
        
        return scores
    
    def _generate_top_issues(self, validation_results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """Generate top issues for dashboard."""
        error_results = [r for r in validation_results if r.severity == ValidationSeverity.ERROR]
        
        return [
            {
                'rule_id': result.rule_id,
                'message': result.message,
                'affected_records': len(result.affected_records),
                'severity': result.severity.value
            }
            for result in error_results[:10]
        ]
    
    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations for dashboard."""
        recommendations = set()
        
        # Add generic recommendations based on common issues
        error_count = sum(1 for r in validation_results if r.severity == ValidationSeverity.ERROR)
        if error_count > 0:
            recommendations.add("Review and fix critical data quality issues")
        
        completeness_issues = [r for r in validation_results if 'completeness' in r.rule_id.lower()]
        if completeness_issues:
            recommendations.add("Improve data collection processes to reduce missing values")
        
        format_issues = [r for r in validation_results if 'format' in r.rule_id.lower()]
        if format_issues:
            recommendations.add("Implement stricter input validation controls")
        
        return list(recommendations)[:5]
    
    def _generate_data_metrics(self, data: DataBatch) -> Dict[str, Any]:
        """Generate data metrics for dashboard."""
        if not data:
            return {}
        
        sample_record = data[0]
        
        return {
            'total_records': len(data),
            'total_columns': len(sample_record),
            'estimated_size_mb': len(str(data[:100])) * (len(data) / 100) / (1024 * 1024) if len(data) > 100 else len(str(data)) / (1024 * 1024),
            'columns': list(sample_record.keys())
        }