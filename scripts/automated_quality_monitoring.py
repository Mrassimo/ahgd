#!/usr/bin/env python3
"""
📊 Automated Data Quality Monitoring System
Real-time monitoring and alerting for Australian Health Database
"""

import polars as pl
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class QualityMonitor:
    """Automated data quality monitoring and alerting system"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.monitoring_dir = self.data_dir.parent / 'monitoring'
        self.monitoring_dir.mkdir(exist_ok=True)
        
        # Quality thresholds
        self.thresholds = {
            'critical_completeness': 95.0,    # Block if below this
            'warning_completeness': 90.0,     # Alert if below this
            'max_null_percentage': 10.0,      # Per column
            'min_record_count': 100,          # Minimum records
            'uniqueness_sa2_codes': 95.0,     # SA2 code integrity
            'temporal_consistency': True       # Date validation
        }
        
        # Expected schemas
        self.expected_schemas = {
            'seifa_2021': ['sa2_code_2021', 'irsd_score', 'irsd_decile'],
            'sa2_boundaries': ['SA2_CODE21', 'SA2_NAME21', 'geometry'],
            'pbs_health': ['year', 'month', 'state'],
            'aihw_mortality': ['YEAR', 'deaths', 'crude_rate_per_100000'],
            'aihw_grim': ['year', 'deaths', 'crude_rate_per_100000'],
            'phidu_pha': ['pha_code', 'pha_name', 'state_territory']
        }
    
    def run_complete_monitoring(self) -> Dict:
        """Run complete data quality monitoring suite"""
        print("📊 Starting Automated Data Quality Monitoring...")
        print("=" * 60)
        
        monitoring_results = {
            'timestamp': datetime.now().isoformat(),
            'datasets': {},
            'alerts': [],
            'summary': {},
            'recommendations': []
        }
        
        # Monitor each dataset
        for dataset_name, expected_cols in self.expected_schemas.items():
            print(f"\n🔍 Monitoring {dataset_name.upper()}...")
            result = self.monitor_dataset(dataset_name, expected_cols)
            monitoring_results['datasets'][dataset_name] = result
            
            # Generate alerts based on results
            alerts = self.generate_alerts(dataset_name, result)
            monitoring_results['alerts'].extend(alerts)
        
        # Generate summary and recommendations
        monitoring_results['summary'] = self.generate_summary(monitoring_results['datasets'])
        monitoring_results['recommendations'] = self.generate_recommendations(monitoring_results)
        
        # Save monitoring report
        self.save_monitoring_report(monitoring_results)
        
        # Print results
        self.print_monitoring_summary(monitoring_results)
        
        return monitoring_results
    
    def monitor_dataset(self, dataset_name: str, expected_cols: List[str]) -> Dict:
        """Monitor individual dataset quality"""
        dataset_files = {
            'seifa_2021': 'seifa_2021_sa2.parquet',
            'sa2_boundaries': 'sa2_boundaries_2021.parquet',
            'pbs_health': 'pbs_current_processed.csv',
            'aihw_mortality': 'aihw_mort_table1.parquet',
            'aihw_grim': 'aihw_grim_data.parquet',
            'phidu_pha': 'phidu_pha_data.parquet'
        }
        
        file_name = dataset_files.get(dataset_name)
        if not file_name:
            return {'status': 'unknown_dataset'}
        
        file_path = self.data_dir / file_name
        if not file_path.exists():
            return {'status': 'file_not_found', 'file_path': str(file_path)}
        
        try:
            # Load dataset
            if file_path.suffix == '.csv':
                df = pl.read_csv(file_path)
            else:
                # Handle geospatial data
                if 'boundaries' in str(file_path).lower():
                    import pandas as pd
                    pandas_df = pd.read_parquet(file_path)
                    non_geo_cols = [col for col in pandas_df.columns if pandas_df[col].dtype.name != 'geometry']
                    df = pl.from_pandas(pandas_df[non_geo_cols])
                else:
                    df = pl.read_parquet(file_path)
            
            # Perform quality checks
            quality_checks = {
                'record_count': len(df),
                'column_count': len(df.columns),
                'completeness': self.calculate_completeness(df),
                'schema_compliance': self.check_schema_compliance(df, expected_cols),
                'null_analysis': self.analyze_null_patterns(df),
                'duplicate_analysis': self.check_duplicates(df),
                'data_type_validation': self.validate_data_types(df, dataset_name),
                'temporal_validation': self.validate_temporal_data(df, dataset_name),
                'geographic_validation': self.validate_geographic_data(df, dataset_name)
            }
            
            # Determine overall quality grade
            quality_checks['quality_grade'] = self.calculate_quality_grade(quality_checks)
            quality_checks['status'] = 'monitored'
            
            return quality_checks
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def calculate_completeness(self, df: pl.DataFrame) -> float:
        """Calculate overall data completeness"""
        total_cells = len(df) * len(df.columns)
        null_cells = sum(df[col].null_count() for col in df.columns)
        return ((total_cells - null_cells) / total_cells) * 100
    
    def check_schema_compliance(self, df: pl.DataFrame, expected_cols: List[str]) -> Dict:
        """Check if dataset complies with expected schema"""
        actual_cols = set(df.columns)
        expected_cols_set = set(expected_cols)
        
        return {
            'missing_columns': list(expected_cols_set - actual_cols),
            'extra_columns': list(actual_cols - expected_cols_set),
            'compliance_score': len(expected_cols_set & actual_cols) / len(expected_cols_set) * 100
        }
    
    def analyze_null_patterns(self, df: pl.DataFrame) -> Dict:
        """Analyze null value patterns"""
        null_analysis = {}
        
        for col in df.columns:
            null_count = df[col].null_count()
            null_pct = (null_count / len(df)) * 100
            
            null_analysis[col] = {
                'null_count': null_count,
                'null_percentage': round(null_pct, 2),
                'status': 'critical' if null_pct > self.thresholds['max_null_percentage'] else 'ok'
            }
        
        return null_analysis
    
    def check_duplicates(self, df: pl.DataFrame) -> Dict:
        """Check for duplicate records"""
        total_records = len(df)
        unique_records = len(df.unique())
        duplicate_count = total_records - unique_records
        duplicate_pct = (duplicate_count / total_records) * 100
        
        return {
            'total_records': total_records,
            'unique_records': unique_records,
            'duplicate_count': duplicate_count,
            'duplicate_percentage': round(duplicate_pct, 2),
            'status': 'warning' if duplicate_pct > 5 else 'ok'
        }
    
    def validate_data_types(self, df: pl.DataFrame, dataset_name: str) -> Dict:
        """Validate data types are appropriate"""
        type_validation = {'issues': []}
        
        # Check for common data type issues
        for col in df.columns:
            col_lower = col.lower()
            dtype = str(df[col].dtype)
            
            # SA2 codes should be strings
            if 'sa2' in col_lower and 'code' in col_lower:
                if dtype not in ['Utf8', 'String']:
                    type_validation['issues'].append(f"{col}: SA2 codes should be strings, found {dtype}")
            
            # Year columns should be integers
            if 'year' in col_lower:
                if dtype not in ['Int64', 'Int32']:
                    type_validation['issues'].append(f"{col}: Year should be integer, found {dtype}")
            
            # Rate columns should be numeric
            if 'rate' in col_lower or 'score' in col_lower:
                if dtype not in ['Float64', 'Float32', 'Int64', 'Int32']:
                    type_validation['issues'].append(f"{col}: Rates/scores should be numeric, found {dtype}")
        
        type_validation['status'] = 'issues' if type_validation['issues'] else 'ok'
        return type_validation
    
    def validate_temporal_data(self, df: pl.DataFrame, dataset_name: str) -> Dict:
        """Validate temporal data consistency"""
        temporal_validation = {'issues': []}
        
        # Check year columns
        year_cols = [col for col in df.columns if 'year' in col.lower()]
        
        for year_col in year_cols:
            try:
                min_year = df[year_col].min()
                max_year = df[year_col].max()
                
                # Check reasonable year ranges
                if min_year and min_year < 1900:
                    temporal_validation['issues'].append(f"{year_col}: Minimum year {min_year} seems too early")
                
                if max_year and max_year > datetime.now().year + 1:
                    temporal_validation['issues'].append(f"{year_col}: Maximum year {max_year} is in the future")
                    
            except Exception as e:
                temporal_validation['issues'].append(f"{year_col}: Could not validate - {str(e)}")
        
        temporal_validation['status'] = 'issues' if temporal_validation['issues'] else 'ok'
        return temporal_validation
    
    def validate_geographic_data(self, df: pl.DataFrame, dataset_name: str) -> Dict:
        """Validate geographic data integrity"""
        geo_validation = {'issues': []}
        
        # Check SA2 codes
        sa2_cols = [col for col in df.columns if 'sa2' in col.lower() and 'code' in col.lower()]
        
        for sa2_col in sa2_cols:
            try:
                # SA2 codes should be 9 digits
                if df[sa2_col].dtype == pl.Utf8:
                    lengths = df[sa2_col].str.len_chars()
                    invalid_lengths = lengths.filter(lengths != 9).len()
                    
                    if invalid_lengths > 0:
                        geo_validation['issues'].append(f"{sa2_col}: {invalid_lengths} codes are not 9 digits")
                
                # Check uniqueness for primary SA2 datasets
                if dataset_name in ['seifa_2021', 'sa2_boundaries']:
                    unique_pct = (df[sa2_col].n_unique() / len(df)) * 100
                    if unique_pct < self.thresholds['uniqueness_sa2_codes']:
                        geo_validation['issues'].append(f"{sa2_col}: Low uniqueness {unique_pct:.1f}%")
                        
            except Exception as e:
                geo_validation['issues'].append(f"{sa2_col}: Could not validate - {str(e)}")
        
        geo_validation['status'] = 'issues' if geo_validation['issues'] else 'ok'
        return geo_validation
    
    def calculate_quality_grade(self, quality_checks: Dict) -> str:
        """Calculate overall quality grade for dataset"""
        score = 100
        
        # Deduct for low completeness
        completeness = quality_checks.get('completeness', 0)
        if completeness < 95:
            score -= (95 - completeness) * 2
        
        # Deduct for schema issues
        schema_compliance = quality_checks.get('schema_compliance', {}).get('compliance_score', 100)
        if schema_compliance < 100:
            score -= (100 - schema_compliance) * 0.5
        
        # Deduct for high null percentages
        null_analysis = quality_checks.get('null_analysis', {})
        critical_nulls = sum(1 for col_data in null_analysis.values() 
                           if isinstance(col_data, dict) and col_data.get('status') == 'critical')
        score -= critical_nulls * 5
        
        # Deduct for duplicates
        duplicate_pct = quality_checks.get('duplicate_analysis', {}).get('duplicate_percentage', 0)
        if duplicate_pct > 5:
            score -= duplicate_pct
        
        # Deduct for data type issues
        type_issues = len(quality_checks.get('data_type_validation', {}).get('issues', []))
        score -= type_issues * 3
        
        # Deduct for temporal issues
        temporal_issues = len(quality_checks.get('temporal_validation', {}).get('issues', []))
        score -= temporal_issues * 2
        
        # Deduct for geographic issues
        geo_issues = len(quality_checks.get('geographic_validation', {}).get('issues', []))
        score -= geo_issues * 3
        
        # Assign grade
        if score >= 95:
            return 'A+ (Excellent)'
        elif score >= 85:
            return 'A (Very Good)'
        elif score >= 75:
            return 'B (Good)'
        elif score >= 65:
            return 'C (Fair)'
        else:
            return 'D (Poor)'
    
    def generate_alerts(self, dataset_name: str, monitoring_result: Dict) -> List[Dict]:
        """Generate alerts based on monitoring results"""
        alerts = []
        
        if monitoring_result.get('status') != 'monitored':
            alerts.append({
                'level': 'CRITICAL',
                'dataset': dataset_name,
                'message': f"Dataset could not be monitored: {monitoring_result.get('status', 'unknown error')}",
                'timestamp': datetime.now().isoformat()
            })
            return alerts
        
        # Check completeness
        completeness = monitoring_result.get('completeness', 0)
        if completeness < self.thresholds['critical_completeness']:
            alerts.append({
                'level': 'CRITICAL',
                'dataset': dataset_name,
                'message': f"Data completeness {completeness:.1f}% below critical threshold {self.thresholds['critical_completeness']}%",
                'timestamp': datetime.now().isoformat()
            })
        elif completeness < self.thresholds['warning_completeness']:
            alerts.append({
                'level': 'WARNING',
                'dataset': dataset_name,
                'message': f"Data completeness {completeness:.1f}% below warning threshold {self.thresholds['warning_completeness']}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check record count
        record_count = monitoring_result.get('record_count', 0)
        if record_count < self.thresholds['min_record_count']:
            alerts.append({
                'level': 'WARNING',
                'dataset': dataset_name,
                'message': f"Low record count: {record_count} records (minimum: {self.thresholds['min_record_count']})",
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for data type issues
        type_issues = monitoring_result.get('data_type_validation', {}).get('issues', [])
        if type_issues:
            alerts.append({
                'level': 'WARNING',
                'dataset': dataset_name,
                'message': f"Data type issues detected: {len(type_issues)} issues",
                'timestamp': datetime.now().isoformat(),
                'details': type_issues
            })
        
        return alerts
    
    def generate_summary(self, datasets: Dict) -> Dict:
        """Generate monitoring summary"""
        total_datasets = len(datasets)
        monitored_datasets = sum(1 for result in datasets.values() 
                               if result.get('status') == 'monitored')
        
        avg_completeness = 0
        quality_grades = []
        
        for result in datasets.values():
            if result.get('status') == 'monitored':
                avg_completeness += result.get('completeness', 0)
                quality_grades.append(result.get('quality_grade', ''))
        
        if monitored_datasets > 0:
            avg_completeness /= monitored_datasets
        
        return {
            'total_datasets': total_datasets,
            'monitored_datasets': monitored_datasets,
            'failed_datasets': total_datasets - monitored_datasets,
            'average_completeness': round(avg_completeness, 2),
            'quality_grades': quality_grades,
            'monitoring_success_rate': round((monitored_datasets / total_datasets) * 100, 1)
        }
    
    def generate_recommendations(self, monitoring_results: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Check overall quality
        summary = monitoring_results.get('summary', {})
        avg_completeness = summary.get('average_completeness', 0)
        
        if avg_completeness < 90:
            recommendations.append("🔧 Implement data imputation strategies for missing values")
        
        # Check alerts
        alerts = monitoring_results.get('alerts', [])
        critical_alerts = [a for a in alerts if a.get('level') == 'CRITICAL']
        warning_alerts = [a for a in alerts if a.get('level') == 'WARNING']
        
        if critical_alerts:
            recommendations.append(f"🚨 Address {len(critical_alerts)} critical data quality issues immediately")
        
        if warning_alerts:
            recommendations.append(f"⚠️ Review and fix {len(warning_alerts)} warning-level issues")
        
        # Check for failed monitoring
        failed_datasets = summary.get('failed_datasets', 0)
        if failed_datasets > 0:
            recommendations.append(f"🔍 Investigate {failed_datasets} datasets that failed monitoring")
        
        # General recommendations
        recommendations.extend([
            "📊 Schedule daily automated quality monitoring",
            "🔄 Implement real-time alerting for critical issues",
            "📈 Create quality trend analysis dashboard",
            "🛡️ Add data validation gates in processing pipeline"
        ])
        
        return recommendations
    
    def save_monitoring_report(self, monitoring_results: Dict) -> None:
        """Save monitoring report to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.monitoring_dir / f'quality_monitoring_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(monitoring_results, f, indent=2, default=str)
        
        # Also save latest report
        latest_file = self.monitoring_dir / 'latest_quality_report.json'
        with open(latest_file, 'w') as f:
            json.dump(monitoring_results, f, indent=2, default=str)
        
        print(f"\n📄 Monitoring report saved: {report_file}")
    
    def print_monitoring_summary(self, monitoring_results: Dict) -> None:
        """Print comprehensive monitoring summary"""
        print("\n📊 Data Quality Monitoring Summary")
        print("=" * 50)
        
        summary = monitoring_results.get('summary', {})
        print(f"📋 Datasets Monitored: {summary.get('monitored_datasets', 0)}/{summary.get('total_datasets', 0)}")
        print(f"📈 Average Completeness: {summary.get('average_completeness', 0):.1f}%")
        print(f"✅ Monitoring Success Rate: {summary.get('monitoring_success_rate', 0):.1f}%")
        
        # Print alerts
        alerts = monitoring_results.get('alerts', [])
        critical_alerts = [a for a in alerts if a.get('level') == 'CRITICAL']
        warning_alerts = [a for a in alerts if a.get('level') == 'WARNING']
        
        if critical_alerts:
            print(f"\n🚨 Critical Alerts: {len(critical_alerts)}")
            for alert in critical_alerts:
                print(f"    {alert['dataset']}: {alert['message']}")
        
        if warning_alerts:
            print(f"\n⚠️ Warning Alerts: {len(warning_alerts)}")
            for alert in warning_alerts:
                print(f"    {alert['dataset']}: {alert['message']}")
        
        if not critical_alerts and not warning_alerts:
            print("\n✅ No critical issues detected!")
        
        # Print quality grades
        print(f"\n🏆 Dataset Quality Grades:")
        for dataset_name, result in monitoring_results.get('datasets', {}).items():
            if result.get('status') == 'monitored':
                grade = result.get('quality_grade', 'Unknown')
                completeness = result.get('completeness', 0)
                print(f"    {dataset_name}: {grade} ({completeness:.1f}% complete)")


def main():
    """Run automated data quality monitoring"""
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    monitor = QualityMonitor(data_dir)
    results = monitor.run_complete_monitoring()
    
    print(f"\n🎉 Data Quality Monitoring Complete!")


if __name__ == "__main__":
    main()