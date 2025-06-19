#!/usr/bin/env python3
"""
🔍 Comprehensive Data Analysis & Schema Documentation
Ultra-deep analysis of processed Australian Health Data databases
"""

import polars as pl
import pandas as pd
from pathlib import Path
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveDataAnalyzer:
    """Ultra-comprehensive database analyzer for Australian Health Data"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.analysis_results = {}
        self.schema_documentation = {}
        
    def analyze_all_databases(self) -> Dict:
        """Perform comprehensive analysis of all processed databases"""
        print("🚀 Starting Ultra-Comprehensive Database Analysis...")
        print("=" * 80)
        
        # Core datasets to analyze
        datasets = {
            'seifa_2021': {
                'parquet': 'seifa_2021_sa2.parquet',
                'csv': 'seifa_2021_sa2.csv',
                'description': 'SEIFA Socio-Economic Disadvantage Indices'
            },
            'sa2_boundaries': {
                'parquet': 'sa2_boundaries_2021.parquet',
                'description': 'Statistical Area Level 2 Geographic Boundaries'
            },
            'pbs_health': {
                'csv': 'pbs_current_processed.csv',
                'description': 'Pharmaceutical Benefits Scheme Health Data'
            },
            'aihw_mortality': {
                'parquet': 'aihw_mort_table1.parquet',
                'description': 'AIHW Mortality Statistics Table 1'
            },
            'aihw_grim': {
                'parquet': 'aihw_grim_data.parquet',
                'description': 'AIHW General Record of Incidence of Mortality'
            },
            'phidu_pha': {
                'parquet': 'phidu_pha_data.parquet',
                'description': 'Public Health Information Development Unit Primary Health Area Data'
            }
        }
        
        for dataset_name, dataset_info in datasets.items():
            print(f"\n📊 Analyzing {dataset_name.upper()}...")
            self.analyze_dataset(dataset_name, dataset_info)
            
        # Generate comprehensive schema documentation
        self.generate_schema_documentation()
        
        # Export results
        self.export_analysis_results()
        
        return self.analysis_results
    
    def analyze_dataset(self, name: str, info: Dict) -> None:
        """Analyze individual dataset with ultra-comprehensive metrics"""
        dataset_analysis = {
            'metadata': info.copy(),
            'file_analysis': {},
            'schema_analysis': {},
            'data_quality': {},
            'statistical_summary': {},
            'relationships': {},
            'anomalies': [],
            'recommendations': []
        }
        
        # Try to load the dataset
        df = None
        file_path = None
        
        # Check parquet first, then CSV
        for format_type in ['parquet', 'csv']:
            if format_type in info:
                file_path = self.data_dir / info[format_type]
                if file_path.exists():
                    try:
                        if format_type == 'parquet':
                            # Check if this is a geospatial file (boundaries)
                            if 'boundaries' in str(file_path).lower():
                                print(f"  🗺️  Geospatial data detected, using pandas fallback")
                                import pandas as pd
                                pandas_df = pd.read_parquet(file_path)
                                # Remove geometry columns and convert to polars
                                non_geo_cols = [col for col in pandas_df.columns if pandas_df[col].dtype.name != 'geometry']
                                non_geo_df = pandas_df[non_geo_cols]
                                df = pl.from_pandas(non_geo_df)
                            else:
                                df = pl.read_parquet(file_path)
                        else:
                            df = pl.read_csv(file_path)
                        break
                    except Exception as e:
                        print(f"  ❌ Failed to load {format_type}: {e}")
                        # Try pandas fallback for any parquet file
                        if format_type == 'parquet':
                            try:
                                print(f"  🔄 Trying pandas fallback...")
                                import pandas as pd
                                pandas_df = pd.read_parquet(file_path)
                                # Convert to polars, excluding problematic columns
                                safe_df = pandas_df.select_dtypes(include=['number', 'object', 'bool'])
                                df = pl.from_pandas(safe_df)
                                break
                            except Exception as fallback_error:
                                print(f"  ❌ Pandas fallback also failed: {fallback_error}")
                        continue
        
        if df is None:
            print(f"  ⚠️  No valid files found for {name}")
            dataset_analysis['status'] = 'FILE_NOT_FOUND'
            self.analysis_results[name] = dataset_analysis
            return
        
        print(f"  ✅ Loaded {len(df)} records with {len(df.columns)} columns")
        
        # File Analysis
        dataset_analysis['file_analysis'] = {
            'file_path': str(file_path),
            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'record_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.estimated_size('mb')
        }
        
        # Schema Analysis
        schema_info = {}
        for col in df.columns:
            col_data = df[col]
            schema_info[col] = {
                'data_type': str(col_data.dtype),
                'null_count': col_data.null_count(),
                'null_percentage': round((col_data.null_count() / len(df)) * 100, 2),
                'unique_count': col_data.n_unique(),
                'unique_percentage': round((col_data.n_unique() / len(df)) * 100, 2)
            }
            
            # Add type-specific analysis
            if col_data.dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
                try:
                    stats = df[col].describe()
                    schema_info[col]['statistics'] = {
                        'min': float(col_data.min()) if col_data.min() is not None else None,
                        'max': float(col_data.max()) if col_data.max() is not None else None,
                        'mean': float(col_data.mean()) if col_data.mean() is not None else None,
                        'median': float(col_data.median()) if col_data.median() is not None else None,
                        'std': float(col_data.std()) if col_data.std() is not None else None
                    }
                except:
                    pass
            elif col_data.dtype == pl.Utf8:
                try:
                    avg_length = col_data.str.len_chars().mean()
                    schema_info[col]['string_analysis'] = {
                        'avg_length': float(avg_length) if avg_length is not None else None,
                        'max_length': col_data.str.len_chars().max(),
                        'min_length': col_data.str.len_chars().min(),
                        'sample_values': col_data.drop_nulls().unique().head(5).to_list()
                    }
                except:
                    pass
        
        dataset_analysis['schema_analysis'] = schema_info
        
        # Data Quality Assessment
        total_cells = len(df) * len(df.columns)
        null_cells = sum(col_data['null_count'] for col_data in schema_info.values())
        
        dataset_analysis['data_quality'] = {
            'completeness_score': round(((total_cells - null_cells) / total_cells) * 100, 2),
            'columns_with_nulls': len([col for col, data in schema_info.items() if data['null_count'] > 0]),
            'duplicate_rows': len(df) - len(df.unique()),
            'data_quality_grade': self.calculate_quality_grade(schema_info, len(df))
        }
        
        # Statistical Summary
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]]
        string_cols = [col for col in df.columns if df[col].dtype == pl.Utf8]
        
        dataset_analysis['statistical_summary'] = {
            'numeric_columns': len(numeric_cols),
            'string_columns': len(string_cols),
            'boolean_columns': len([col for col in df.columns if df[col].dtype == pl.Boolean]),
            'datetime_columns': len([col for col in df.columns if df[col].dtype in [pl.Date, pl.Datetime]]),
            'high_cardinality_columns': [col for col, data in schema_info.items() if data['unique_percentage'] > 95],
            'low_cardinality_columns': [col for col, data in schema_info.items() if data['unique_percentage'] < 5]
        }
        
        # Identify potential relationships (matching column names)
        dataset_analysis['relationships'] = self.identify_relationships(df, name)
        
        # Detect anomalies
        dataset_analysis['anomalies'] = self.detect_anomalies(df, schema_info)
        
        # Generate recommendations
        dataset_analysis['recommendations'] = self.generate_recommendations(df, schema_info, dataset_analysis)
        
        dataset_analysis['status'] = 'ANALYSIS_COMPLETE'
        self.analysis_results[name] = dataset_analysis
        
        print(f"  📈 Data Quality Grade: {dataset_analysis['data_quality']['data_quality_grade']}")
        print(f"  📊 Completeness: {dataset_analysis['data_quality']['completeness_score']}%")
        
    def calculate_quality_grade(self, schema_info: Dict, record_count: int) -> str:
        """Calculate data quality grade based on comprehensive metrics"""
        score = 100
        
        # Deduct for missing data
        avg_null_pct = sum(col['null_percentage'] for col in schema_info.values()) / len(schema_info)
        score -= avg_null_pct * 2
        
        # Deduct for low record count
        if record_count < 1000:
            score -= 20
        elif record_count < 10000:
            score -= 10
        
        # Bonus for high cardinality key columns
        key_cols = [col for col in schema_info.keys() if any(keyword in col.lower() for keyword in ['id', 'code', 'key'])]
        if key_cols and any(schema_info[col]['unique_percentage'] > 95 for col in key_cols):
            score += 10
        
        if score >= 90:
            return 'A+ (Excellent)'
        elif score >= 80:
            return 'A (Very Good)'
        elif score >= 70:
            return 'B (Good)'
        elif score >= 60:
            return 'C (Fair)'
        else:
            return 'D (Poor)'
    
    def identify_relationships(self, df: pl.DataFrame, dataset_name: str) -> Dict:
        """Identify potential relationships with other datasets"""
        relationships = {
            'potential_foreign_keys': [],
            'spatial_columns': [],
            'temporal_columns': [],
            'categorical_columns': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for common key patterns
            if any(pattern in col_lower for pattern in ['sa2', 'postcode', 'suburb', 'state', 'lga']):
                relationships['spatial_columns'].append(col)
            
            if any(pattern in col_lower for pattern in ['date', 'year', 'month', 'time']):
                relationships['temporal_columns'].append(col)
            
            if col.endswith('_code') or col.endswith('_id') or 'code' in col_lower:
                relationships['potential_foreign_keys'].append(col)
            
            # Check cardinality for categorical
            unique_ratio = df[col].n_unique() / len(df)
            if unique_ratio < 0.1 and df[col].n_unique() > 1:
                relationships['categorical_columns'].append(col)
        
        return relationships
    
    def detect_anomalies(self, df: pl.DataFrame, schema_info: Dict) -> List[str]:
        """Detect data anomalies and quality issues"""
        anomalies = []
        
        for col, info in schema_info.items():
            # Check for high null percentage
            if info['null_percentage'] > 50:
                anomalies.append(f"High missing data in {col}: {info['null_percentage']}%")
            
            # Check for single-value columns
            if info['unique_count'] == 1:
                anomalies.append(f"Column {col} has only one unique value")
            
            # Check for potential ID columns with low uniqueness
            if 'id' in col.lower() and info['unique_percentage'] < 95:
                anomalies.append(f"ID column {col} has low uniqueness: {info['unique_percentage']}%")
            
            # Check for numeric outliers
            if 'statistics' in info:
                stats = info['statistics']
                if stats['std'] and stats['mean']:
                    cv = stats['std'] / abs(stats['mean'])
                    if cv > 3:
                        anomalies.append(f"High coefficient of variation in {col}: {cv:.2f}")
        
        return anomalies
    
    def generate_recommendations(self, df: pl.DataFrame, schema_info: Dict, analysis: Dict) -> List[str]:
        """Generate actionable recommendations for data improvement"""
        recommendations = []
        
        # Data quality recommendations
        if analysis['data_quality']['completeness_score'] < 90:
            recommendations.append("Consider data imputation or source improvement for missing values")
        
        if analysis['data_quality']['duplicate_rows'] > 0:
            recommendations.append("Remove duplicate rows to ensure data integrity")
        
        # Schema optimization recommendations
        for col, info in schema_info.items():
            if info['data_type'] == 'Utf8' and 'string_analysis' in info:
                if info['string_analysis']['avg_length'] and info['string_analysis']['avg_length'] < 10:
                    recommendations.append(f"Consider categorical encoding for {col}")
        
        # Performance recommendations
        if analysis['file_analysis']['memory_usage_mb'] > 100:
            recommendations.append("Consider data partitioning or compression for large dataset")
        
        # Relationship recommendations
        if analysis['relationships']['spatial_columns']:
            recommendations.append("Enable spatial indexing for geographic columns")
        
        return recommendations
    
    def generate_schema_documentation(self) -> None:
        """Generate comprehensive schema documentation"""
        print("\n📚 Generating Ultra-Comprehensive Schema Documentation...")
        
        schema_doc = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'analyst': 'Australian Health Data Analytics Platform',
                'version': '2.0.0',
                'total_datasets': len(self.analysis_results),
                'total_records': sum(result.get('file_analysis', {}).get('record_count', 0) 
                                   for result in self.analysis_results.values()),
                'total_size_mb': sum(result.get('file_analysis', {}).get('file_size_mb', 0) 
                                   for result in self.analysis_results.values())
            },
            'datasets': {},
            'cross_dataset_analysis': self.perform_cross_dataset_analysis(),
            'architecture_recommendations': self.generate_architecture_recommendations()
        }
        
        # Add detailed dataset documentation
        for dataset_name, analysis in self.analysis_results.items():
            if analysis.get('status') == 'ANALYSIS_COMPLETE':
                schema_doc['datasets'][dataset_name] = {
                    'description': analysis['metadata']['description'],
                    'schema': self.format_schema_for_documentation(analysis['schema_analysis']),
                    'quality_assessment': analysis['data_quality'],
                    'relationships': analysis['relationships'],
                    'recommendations': analysis['recommendations']
                }
        
        self.schema_documentation = schema_doc
    
    def perform_cross_dataset_analysis(self) -> Dict:
        """Analyze relationships across datasets"""
        cross_analysis = {
            'common_columns': {},
            'potential_joins': [],
            'data_lineage': {},
            'integration_opportunities': []
        }
        
        # Find common columns across datasets
        all_columns = {}
        for dataset_name, analysis in self.analysis_results.items():
            if analysis.get('status') == 'ANALYSIS_COMPLETE':
                for col in analysis['schema_analysis'].keys():
                    if col not in all_columns:
                        all_columns[col] = []
                    all_columns[col].append(dataset_name)
        
        # Identify columns present in multiple datasets
        cross_analysis['common_columns'] = {
            col: datasets for col, datasets in all_columns.items() 
            if len(datasets) > 1
        }
        
        # Suggest potential joins
        for col, datasets in cross_analysis['common_columns'].items():
            if len(datasets) >= 2 and any(keyword in col.lower() for keyword in ['sa2', 'code', 'id']):
                cross_analysis['potential_joins'].append({
                    'join_column': col,
                    'datasets': datasets,
                    'join_type': 'inner' if 'sa2' in col.lower() else 'left'
                })
        
        return cross_analysis
    
    def generate_architecture_recommendations(self) -> List[str]:
        """Generate ultra-comprehensive architecture recommendations"""
        recommendations = [
            "🏗️ **Data Architecture Recommendations**",
            "",
            "**Storage Layer:**",
            "- Implement Bronze-Silver-Gold data lake architecture",
            "- Use Parquet format with ZSTD compression for optimal performance",
            "- Partition large datasets by geographic regions (state/territory)",
            "",
            "**Processing Layer:**",
            "- Continue using Polars for high-performance data processing",
            "- Implement incremental loading for large datasets",
            "- Add data quality monitoring with automated alerts",
            "",
            "**Integration Layer:**",
            "- Create standardized SA2 code mapping across all datasets",
            "- Implement CDC (Change Data Capture) for real-time updates",
            "- Add data lineage tracking for regulatory compliance",
            "",
            "**API Layer:**",
            "- Design RESTful APIs with GraphQL for flexible queries",
            "- Implement caching strategy with Redis for frequently accessed data",
            "- Add rate limiting and authentication for production use",
            "",
            "**Analytics Layer:**",
            "- Create materialized views for common analytical queries",
            "- Implement real-time streaming for health alerts",
            "- Add machine learning pipeline for predictive analytics"
        ]
        
        return recommendations
    
    def format_schema_for_documentation(self, schema_analysis: Dict) -> Dict:
        """Format schema for beautiful documentation"""
        formatted_schema = {}
        
        for col, info in schema_analysis.items():
            formatted_schema[col] = {
                'type': info['data_type'],
                'nullable': info['null_count'] > 0,
                'cardinality': info['unique_count'],
                'completeness': f"{100 - info['null_percentage']:.1f}%",
                'description': self.generate_column_description(col, info)
            }
            
            if 'statistics' in info:
                formatted_schema[col]['statistics'] = info['statistics']
            
            if 'string_analysis' in info:
                formatted_schema[col]['string_info'] = info['string_analysis']
        
        return formatted_schema
    
    def generate_column_description(self, col_name: str, info: Dict) -> str:
        """Generate intelligent column descriptions"""
        col_lower = col_name.lower()
        
        # Geographic columns
        if 'sa2' in col_lower:
            return "Statistical Area Level 2 identifier (ABS Geographic Standard)"
        elif 'postcode' in col_lower:
            return "Australian postal code"
        elif 'state' in col_lower:
            return "Australian state or territory code/name"
        
        # SEIFA columns
        elif 'irsd' in col_lower:
            return "Index of Relative Socio-economic Disadvantage (SEIFA 2021)"
        elif 'irsad' in col_lower:
            return "Index of Relative Socio-economic Advantage and Disadvantage (SEIFA 2021)"
        elif 'ier' in col_lower:
            return "Index of Economic Resources (SEIFA 2021)"
        elif 'ieo' in col_lower:
            return "Index of Education and Occupation (SEIFA 2021)"
        
        # Health columns
        elif 'mortality' in col_lower:
            return "Mortality rate or death statistics"
        elif 'pbs' in col_lower:
            return "Pharmaceutical Benefits Scheme related data"
        elif 'disease' in col_lower or 'condition' in col_lower:
            return "Health condition or disease indicator"
        
        # Default description based on data type
        elif info['data_type'] in ['Int64', 'Int32']:
            return f"Numeric integer field (range: {info.get('statistics', {}).get('min', 'unknown')} - {info.get('statistics', {}).get('max', 'unknown')})"
        elif info['data_type'] in ['Float64', 'Float32']:
            mean_val = info.get('statistics', {}).get('mean', 'unknown')
            if mean_val != 'unknown' and mean_val is not None:
                return f"Numeric decimal field (avg: {mean_val:.2f})"
            else:
                return "Numeric decimal field (avg: unknown)"
        else:
            return f"Text field ({info['unique_count']} unique values)"
    
    def export_analysis_results(self) -> None:
        """Export comprehensive analysis results"""
        output_dir = self.data_dir.parent / 'docs' / 'analysis'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export JSON report
        json_file = output_dir / 'comprehensive_database_analysis.json'
        with open(json_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Export schema documentation
        schema_file = output_dir / 'ultra_database_schema_documentation.json'
        with open(schema_file, 'w') as f:
            json.dump(self.schema_documentation, f, indent=2, default=str)
        
        # Export markdown report
        self.export_markdown_report(output_dir)
        
        print(f"\n📄 Analysis results exported to: {output_dir}")
        print(f"  - JSON Report: {json_file.name}")
        print(f"  - Schema Documentation: {schema_file.name}")
        print(f"  - Markdown Report: ultra_database_analysis_report.md")
    
    def export_markdown_report(self, output_dir: Path) -> None:
        """Export beautiful markdown report"""
        md_file = output_dir / 'ultra_database_analysis_report.md'
        
        with open(md_file, 'w') as f:
            f.write("# 🏥 Ultra-Comprehensive Australian Health Database Analysis\n\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Platform:** Australian Health Data Analytics v2.0.0\n\n")
            
            # Executive Summary
            f.write("## 📊 Executive Summary\n\n")
            total_records = self.schema_documentation['metadata']['total_records']
            total_size = self.schema_documentation['metadata']['total_size_mb']
            total_datasets = self.schema_documentation['metadata']['total_datasets']
            
            f.write(f"- **Total Datasets Analyzed:** {total_datasets}\n")
            f.write(f"- **Total Records:** {total_records:,}\n")
            f.write(f"- **Total Data Size:** {total_size:.1f} MB\n")
            f.write(f"- **Analysis Completeness:** {len([r for r in self.analysis_results.values() if r.get('status') == 'ANALYSIS_COMPLETE'])}/{total_datasets} datasets\n\n")
            
            # Dataset Details
            f.write("## 🗃️ Dataset Analysis Details\n\n")
            
            for dataset_name, analysis in self.analysis_results.items():
                if analysis.get('status') == 'ANALYSIS_COMPLETE':
                    f.write(f"### {dataset_name.upper()}\n\n")
                    f.write(f"**Description:** {analysis['metadata']['description']}\n\n")
                    
                    # File info
                    file_info = analysis['file_analysis']
                    f.write(f"- **Records:** {file_info['record_count']:,}\n")
                    f.write(f"- **Columns:** {file_info['column_count']}\n")
                    f.write(f"- **File Size:** {file_info['file_size_mb']:.1f} MB\n")
                    f.write(f"- **Memory Usage:** {file_info['memory_usage_mb']:.1f} MB\n")
                    
                    # Data quality
                    quality = analysis['data_quality']
                    f.write(f"- **Data Quality Grade:** {quality['data_quality_grade']}\n")
                    f.write(f"- **Completeness:** {quality['completeness_score']}%\n\n")
                    
                    # Schema table
                    f.write("#### Schema Details\n\n")
                    f.write("| Column | Type | Completeness | Cardinality | Description |\n")
                    f.write("|--------|------|--------------|-------------|-------------|\n")
                    
                    for col, info in analysis['schema_analysis'].items():
                        completeness = f"{100 - info['null_percentage']:.1f}%"
                        f.write(f"| {col} | {info['data_type']} | {completeness} | {info['unique_count']} | {self.generate_column_description(col, info)} |\n")
                    
                    f.write("\n")
                    
                    # Recommendations
                    if analysis['recommendations']:
                        f.write("#### Recommendations\n\n")
                        for rec in analysis['recommendations']:
                            f.write(f"- {rec}\n")
                        f.write("\n")
            
            # Cross-dataset analysis
            cross_analysis = self.schema_documentation['cross_dataset_analysis']
            f.write("## 🔗 Cross-Dataset Analysis\n\n")
            
            if cross_analysis['common_columns']:
                f.write("### Common Columns Across Datasets\n\n")
                for col, datasets in cross_analysis['common_columns'].items():
                    f.write(f"- **{col}:** Present in {', '.join(datasets)}\n")
                f.write("\n")
            
            if cross_analysis['potential_joins']:
                f.write("### Potential Join Opportunities\n\n")
                for join in cross_analysis['potential_joins']:
                    f.write(f"- **{join['join_column']}:** {join['join_type']} join between {', '.join(join['datasets'])}\n")
                f.write("\n")
            
            # Architecture recommendations
            f.write("## 🏗️ Architecture Recommendations\n\n")
            for rec in self.schema_documentation['architecture_recommendations']:
                f.write(f"{rec}\n")
        
        print(f"  ✅ Markdown report created: {md_file}")


def main():
    """Run comprehensive database analysis"""
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    analyzer = ComprehensiveDataAnalyzer(data_dir)
    results = analyzer.analyze_all_databases()
    
    print("\n🎉 Ultra-Comprehensive Database Analysis Complete!")
    print("=" * 80)
    print("\n📋 Analysis Summary:")
    
    successful_analyses = [name for name, result in results.items() 
                          if result.get('status') == 'ANALYSIS_COMPLETE']
    
    print(f"  ✅ Successfully analyzed: {len(successful_analyses)} datasets")
    print(f"  📊 Total records analyzed: {sum(result.get('file_analysis', {}).get('record_count', 0) for result in results.values()):,}")
    print(f"  💾 Total data size: {sum(result.get('file_analysis', {}).get('file_size_mb', 0) for result in results.values()):.1f} MB")
    
    if successful_analyses:
        print(f"\n📈 Dataset Quality Grades:")
        for name in successful_analyses:
            grade = results[name]['data_quality']['data_quality_grade']
            completeness = results[name]['data_quality']['completeness_score']
            print(f"    {name}: {grade} ({completeness}% complete)")


if __name__ == "__main__":
    main()