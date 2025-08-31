"""
AHGD V3: Data Export Manager
High-performance export functionality for health analytics data.

Features:
- Multiple export formats (CSV, Excel, Parquet, JSON, GeoJSON)
- Optimized Polars-based data processing
- Metadata preservation
- Compression and performance optimization
"""

import io
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import polars as pl
import pandas as pd
import streamlit as st


class ExportManager:
    """Manages data export operations with high performance."""
    
    def __init__(self):
        """Initialize export manager."""
        self.supported_formats = ['CSV', 'Excel', 'Parquet', 'JSON', 'GeoJSON']
        self.mime_types = {
            'CSV': 'text/csv',
            'Excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'Parquet': 'application/octet-stream',
            'JSON': 'application/json',
            'GeoJSON': 'application/geo+json'
        }

    def prepare_download(
        self, 
        data: pl.DataFrame, 
        export_format: str,
        include_metadata: bool = True,
        compress: bool = True
    ) -> bytes:
        """
        Prepare data for download in specified format.
        
        Args:
            data: Polars DataFrame to export
            export_format: Target export format
            include_metadata: Whether to include metadata
            compress: Whether to compress the output
            
        Returns:
            Bytes data ready for download
        """
        
        if export_format not in self.supported_formats:
            raise ValueError(f"Unsupported format: {export_format}")
        
        # Add metadata if requested
        if include_metadata:
            data = self._add_export_metadata(data)
        
        # Generate export data based on format
        if export_format == 'CSV':
            return self._export_csv(data, compress)
        elif export_format == 'Excel':
            return self._export_excel(data)
        elif export_format == 'Parquet':
            return self._export_parquet(data)
        elif export_format == 'JSON':
            return self._export_json(data, compress)
        elif export_format == 'GeoJSON':
            return self._export_geojson(data, compress)
        
        raise ValueError(f"Export format {export_format} not implemented")

    def _add_export_metadata(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add export metadata columns to the DataFrame."""
        
        return data.with_columns([
            pl.lit(datetime.now().isoformat()).alias('_export_timestamp'),
            pl.lit('AHGD_V3_Modern_Analytics_Platform').alias('_data_source'),
            pl.lit('Australian_Health_Geographic_Data').alias('_dataset_name'),
            pl.lit('1.0.0').alias('_schema_version')
        ])

    def _export_csv(self, data: pl.DataFrame, compress: bool = True) -> bytes:
        """Export data as CSV with optional compression."""
        
        # Convert to CSV using Polars (fast)
        csv_buffer = io.StringIO()
        data.write_csv(csv_buffer)
        csv_content = csv_buffer.getvalue().encode('utf-8')
        
        if compress:
            # Compress with ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(
                    f"ahgd_health_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    csv_content
                )
            return zip_buffer.getvalue()
        
        return csv_content

    def _export_excel(self, data: pl.DataFrame) -> bytes:
        """Export data as Excel workbook with multiple sheets."""
        
        excel_buffer = io.BytesIO()
        
        # Convert to pandas for Excel export (xlsxwriter integration)
        pandas_df = data.to_pandas()
        
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            # Main data sheet
            pandas_df.to_excel(writer, sheet_name='Health_Data', index=False)
            
            # Create summary sheet
            summary_data = self._generate_summary_stats(data)
            if summary_data:
                summary_data.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Add metadata sheet
            metadata = self._generate_export_metadata()
            pd.DataFrame([metadata]).to_excel(writer, sheet_name='Metadata', index=False)
            
            # Format worksheets
            workbook = writer.book
            
            # Add formatting
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#1f77b4',
                'font_color': 'white',
                'border': 1
            })
            
            # Apply header formatting
            for sheet_name in ['Health_Data', 'Summary_Statistics', 'Metadata']:
                worksheet = writer.sheets[sheet_name]
                for col_num, value in enumerate(pandas_df.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                worksheet.autofit()
        
        return excel_buffer.getvalue()

    def _export_parquet(self, data: pl.DataFrame) -> bytes:
        """Export data as Parquet (high-performance columnar format)."""
        
        parquet_buffer = io.BytesIO()
        
        # Use Polars native Parquet export (very fast)
        data.write_parquet(parquet_buffer, compression='snappy')
        
        return parquet_buffer.getvalue()

    def _export_json(self, data: pl.DataFrame, compress: bool = True) -> bytes:
        """Export data as JSON with optional compression."""
        
        # Convert to JSON using Polars
        json_data = data.write_json()
        json_bytes = json_data.encode('utf-8')
        
        if compress:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(
                    f"ahgd_health_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    json_bytes
                )
            return zip_buffer.getvalue()
        
        return json_bytes

    def _export_geojson(self, data: pl.DataFrame, compress: bool = True) -> bytes:
        """Export geographic data as GeoJSON."""
        
        # Check if geographic data is available
        required_geo_columns = ['centroid_longitude', 'centroid_latitude']
        has_geo_data = all(col in data.columns for col in required_geo_columns)
        
        if not has_geo_data:
            raise ValueError("Geographic data not available for GeoJSON export")
        
        # Create GeoJSON structure
        features = []
        
        for row in data.iter_rows(named=True):
            if row.get('centroid_longitude') and row.get('centroid_latitude'):
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            float(row['centroid_longitude']), 
                            float(row['centroid_latitude'])
                        ]
                    },
                    "properties": {
                        k: v for k, v in row.items() 
                        if k not in required_geo_columns and v is not None
                    }
                }
                features.append(feature)
        
        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
            "metadata": self._generate_export_metadata()
        }
        
        geojson_bytes = json.dumps(geojson_data, indent=2).encode('utf-8')
        
        if compress:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                zip_file.writestr(
                    f"ahgd_health_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                    geojson_bytes
                )
            return zip_buffer.getvalue()
        
        return geojson_bytes

    def _generate_summary_stats(self, data: pl.DataFrame) -> Optional[pl.DataFrame]:
        """Generate summary statistics for the dataset."""
        
        try:
            # Identify numeric columns
            numeric_columns = [
                col for col in data.columns 
                if data[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            
            if not numeric_columns:
                return None
            
            # Generate statistics for numeric columns
            stats_data = []
            
            for col in numeric_columns:
                col_stats = data.select([
                    pl.lit(col).alias('Column'),
                    pl.col(col).count().alias('Count'),
                    pl.col(col).mean().alias('Mean'),
                    pl.col(col).median().alias('Median'),
                    pl.col(col).std().alias('Std_Dev'),
                    pl.col(col).min().alias('Min'),
                    pl.col(col).max().alias('Max'),
                    pl.col(col).is_null().sum().alias('Missing_Count'),
                    (pl.col(col).is_null().sum() / pl.col(col).len() * 100).alias('Missing_Percent')
                ])
                
                stats_data.append(col_stats)
            
            # Combine all statistics
            return pl.concat(stats_data) if stats_data else None
            
        except Exception as e:
            st.error(f"Error generating summary statistics: {str(e)}")
            return None

    def _generate_export_metadata(self) -> Dict[str, Any]:
        """Generate comprehensive metadata for exports."""
        
        return {
            'export_timestamp': datetime.now().isoformat(),
            'platform': 'AHGD V3 - Modern Analytics Engineering Platform',
            'description': 'Australian Health Geography Data - Comprehensive health analytics',
            'data_sources': [
                'Australian Bureau of Statistics (ABS)',
                'Australian Institute of Health and Welfare (AIHW)', 
                'Bureau of Meteorology (BOM)',
                'Department of Health (Medicare/PBS)'
            ],
            'geographic_standard': 'Australian Statistical Geography Standard (ASGS) 2021',
            'processing_engine': 'Polars + DuckDB',
            'schema_version': '1.0.0',
            'contact_info': 'https://github.com/Mrassimo/ahgd',
            'license': 'Data subject to original source licensing terms',
            'citation': 'AHGD V3 Modern Analytics Platform. Australian health and geographic data integration.',
            'quality_notes': [
                'Age-standardised rates where applicable',
                'Small area data may be suppressed for privacy protection', 
                'Data quality scores included for each record',
                'Missing values preserved as null/None'
            ],
            'performance_notes': [
                '10x faster processing with Polars engine',
                'Columnar storage optimization with DuckDB',
                'Memory-efficient lazy evaluation'
            ]
        }

    def get_mime_type(self, export_format: str) -> str:
        """Get MIME type for export format."""
        return self.mime_types.get(export_format, 'application/octet-stream')

    def get_file_extension(self, export_format: str) -> str:
        """Get file extension for export format."""
        extensions = {
            'CSV': 'csv',
            'Excel': 'xlsx', 
            'Parquet': 'parquet',
            'JSON': 'json',
            'GeoJSON': 'geojson'
        }
        return extensions.get(export_format, 'data')

    def validate_export_data(self, data: pl.DataFrame, export_format: str) -> Dict[str, Any]:
        """Validate data before export and return validation results."""
        
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'record_count': data.height,
            'column_count': len(data.columns)
        }
        
        # Check for empty data
        if data.height == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Dataset is empty")
            return validation_results
        
        # Check for geographic requirements (GeoJSON)
        if export_format == 'GeoJSON':
            required_geo_cols = ['centroid_longitude', 'centroid_latitude']
            missing_geo_cols = [col for col in required_geo_cols if col not in data.columns]
            
            if missing_geo_cols:
                validation_results['is_valid'] = False
                validation_results['errors'].append(
                    f"GeoJSON export requires geographic columns: {missing_geo_cols}"
                )
        
        # Check for large datasets
        if data.height > 1000000:  # 1M records
            validation_results['warnings'].append(
                f"Large dataset ({data.height:,} records) may take time to export"
            )
        
        # Check column types
        problematic_columns = []
        for col in data.columns:
            if data[col].dtype == pl.Object:
                problematic_columns.append(col)
        
        if problematic_columns:
            validation_results['warnings'].append(
                f"Columns with complex data types may not export properly: {problematic_columns}"
            )
        
        # Memory usage estimation
        estimated_memory_mb = (data.height * len(data.columns) * 8) / (1024 * 1024)  # Rough estimate
        if estimated_memory_mb > 500:  # 500MB
            validation_results['warnings'].append(
                f"Export may require significant memory (~{estimated_memory_mb:.0f}MB)"
            )
        
        return validation_results