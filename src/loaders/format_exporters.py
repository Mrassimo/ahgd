"""Format-specific exporters for optimised data export.

This module provides specialised exporters for each supported data format,
optimised for Australian health and geographic data characteristics.

British English spelling is used throughout (optimise, standardise, etc.).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import geopandas as gpd
from shapely.geometry import Point, Polygon
import geojson
from loguru import logger

from ..utils.interfaces import LoadingError
from ..utils.config import get_config
from ..utils.logging import get_logger, monitor_performance


class BaseExporter(ABC):
    """Base class for format-specific exporters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
    @abstractmethod
    def export(self, 
              data: pd.DataFrame, 
              output_path: Path, 
              **kwargs) -> Dict[str, Any]:
        """Export data to specific format.
        
        Args:
            data: DataFrame to export
            output_path: Path for output file
            **kwargs: Format-specific options
            
        Returns:
            Dictionary with export metadata
        """
        pass
    
    @abstractmethod
    def get_optimal_settings(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get optimal export settings for given data.
        
        Args:
            data: DataFrame to analyse
            
        Returns:
            Dictionary of optimal settings
        """
        pass
    
    def validate_export(self, output_path: Path, original_data: pd.DataFrame) -> bool:
        """Validate exported file.
        
        Args:
            output_path: Path to exported file
            original_data: Original DataFrame for comparison
            
        Returns:
            True if export is valid
        """
        try:
            return output_path.exists() and output_path.stat().st_size > 0
        except Exception:
            return False


class ParquetExporter(BaseExporter):
    """Optimised Parquet export with compression and metadata."""
    
    def get_optimal_settings(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Determine optimal Parquet settings based on data characteristics."""
        data_size = data.memory_usage(deep=True).sum()
        row_count = len(data)
        
        # Compression selection
        if data_size > 100_000_000:  # > 100MB
            compression = 'snappy'  # Faster for large files
        elif data_size > 10_000_000:  # > 10MB
            compression = 'gzip'  # Good balance
        else:
            compression = 'brotli'  # Best compression for smaller files
            
        # Row group size optimisation
        if row_count > 1_000_000:
            row_group_size = 100_000
        elif row_count > 100_000:
            row_group_size = 50_000
        else:
            row_group_size = row_count
            
        return {
            'compression': compression,
            'row_group_size': row_group_size,
            'use_dictionary': True,
            'write_statistics': True,
            'coerce_timestamps': 'ms'
        }
    
    @monitor_performance("parquet_export")
    def export(self, 
              data: pd.DataFrame, 
              output_path: Path, 
              **kwargs) -> Dict[str, Any]:
        """Export DataFrame to optimised Parquet format."""
        try:
            # Get optimal settings
            settings = self.get_optimal_settings(data)
            settings.update(kwargs)  # Allow override
            
            # Prepare data for Parquet
            parquet_data = self._prepare_for_parquet(data)
            
            # Create metadata
            metadata = self._create_parquet_metadata(parquet_data)
            
            # Write Parquet file
            table = pa.Table.from_pandas(parquet_data, preserve_index=False)
            
            # Add custom metadata
            table = table.replace_schema_metadata(metadata)
            
            pq.write_table(
                table,
                output_path,
                compression=settings['compression'],
                row_group_size=settings['row_group_size'],
                use_dictionary=settings['use_dictionary'],
                write_statistics=settings['write_statistics'],
                coerce_timestamps=settings['coerce_timestamps']
            )
            
            # Return export info
            file_size = output_path.stat().st_size
            
            export_info = {
                'format': 'parquet',
                'file_size_bytes': file_size,
                'file_size_mb': file_size / 1024 / 1024,
                'rows': len(parquet_data),
                'columns': len(parquet_data.columns),
                'compression': settings['compression'],
                'settings': settings
            }
            
            self.logger.info(f"Parquet export completed", 
                           file_size_mb=export_info['file_size_mb'],
                           compression=settings['compression'])
            
            return export_info
            
        except Exception as e:
            raise LoadingError(f"Parquet export failed: {str(e)}")
    
    def _prepare_for_parquet(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for optimal Parquet storage."""
        parquet_data = data.copy()
        
        # Convert object columns to appropriate types
        for column in parquet_data.select_dtypes(include=['object']).columns:
            # Skip geometry columns
            if hasattr(parquet_data[column].iloc[0], '__geo_interface__'):
                continue
                
            # Try to convert to categorical for efficiency
            unique_ratio = parquet_data[column].nunique() / len(parquet_data)
            if unique_ratio < 0.1:  # Less than 10% unique values
                parquet_data[column] = parquet_data[column].astype('category')
                
        # Optimise numeric types
        for column in parquet_data.select_dtypes(include=['int64']).columns:
            col_min = parquet_data[column].min()
            col_max = parquet_data[column].max()
            
            # Downcast integer types where possible
            if col_min >= 0:
                if col_max < 256:
                    parquet_data[column] = parquet_data[column].astype('uint8')
                elif col_max < 65536:
                    parquet_data[column] = parquet_data[column].astype('uint16')
                elif col_max < 4294967296:
                    parquet_data[column] = parquet_data[column].astype('uint32')
            else:
                if col_min >= -128 and col_max < 128:
                    parquet_data[column] = parquet_data[column].astype('int8')
                elif col_min >= -32768 and col_max < 32768:
                    parquet_data[column] = parquet_data[column].astype('int16')
                elif col_min >= -2147483648 and col_max < 2147483648:
                    parquet_data[column] = parquet_data[column].astype('int32')
                    
        return parquet_data
    
    def _create_parquet_metadata(self, data: pd.DataFrame) -> Dict[str, str]:
        """Create metadata for Parquet file."""
        metadata = {
            'created_by': 'AHGD ETL Pipeline',
            'created_at': datetime.now().isoformat(),
            'schema_version': '1.0',
            'data_source': 'Australian Health and Geographic Data',
            'total_rows': str(len(data)),
            'total_columns': str(len(data.columns))
        }
        
        # Add column information
        for i, column in enumerate(data.columns):
            metadata[f'column_{i}_name'] = column
            metadata[f'column_{i}_type'] = str(data[column].dtype)
            
        return metadata


class CSVExporter(BaseExporter):
    """CSV export with proper encoding and chunk handling."""
    
    def get_optimal_settings(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Determine optimal CSV settings."""
        row_count = len(data)
        
        # Determine if chunking is needed
        chunk_size = None
        if row_count > 1_000_000:
            chunk_size = 100_000
        elif row_count > 500_000:
            chunk_size = 50_000
            
        return {
            'encoding': 'utf-8',
            'chunk_size': chunk_size,
            'compression': 'gzip' if row_count > 100_000 else None,
            'float_format': '%.6f',
            'date_format': '%Y-%m-%d %H:%M:%S'
        }
    
    @monitor_performance("csv_export")
    def export(self, 
              data: pd.DataFrame, 
              output_path: Path, 
              **kwargs) -> Dict[str, Any]:
        """Export DataFrame to optimised CSV format."""
        try:
            settings = self.get_optimal_settings(data)
            settings.update(kwargs)
            
            # Prepare data for CSV
            csv_data = self._prepare_for_csv(data)
            
            # Export with or without chunking
            if settings['chunk_size']:
                self._export_chunked_csv(csv_data, output_path, settings)
            else:
                csv_data.to_csv(
                    output_path,
                    index=False,
                    encoding=settings['encoding'],
                    float_format=settings['float_format'],
                    date_format=settings['date_format']
                )
                
            # Compress if requested
            if settings['compression']:
                import gzip
                with open(output_path, 'rb') as f_in:
                    with gzip.open(f"{output_path}.gz", 'wb') as f_out:
                        f_out.writelines(f_in)
                output_path.unlink()  # Remove uncompressed file
                output_path = Path(f"{output_path}.gz")
                
            file_size = output_path.stat().st_size
            
            export_info = {
                'format': 'csv',
                'file_size_bytes': file_size,
                'file_size_mb': file_size / 1024 / 1024,
                'rows': len(csv_data),
                'columns': len(csv_data.columns),
                'encoding': settings['encoding'],
                'compression': settings['compression'],
                'settings': settings
            }
            
            self.logger.info(f"CSV export completed", 
                           file_size_mb=export_info['file_size_mb'],
                           compression=settings['compression'])
            
            return export_info
            
        except Exception as e:
            raise LoadingError(f"CSV export failed: {str(e)}")
    
    def _prepare_for_csv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for CSV export."""
        csv_data = data.copy()
        
        # Handle special characters in string columns
        for column in csv_data.select_dtypes(include=['object']).columns:
            if csv_data[column].dtype == 'object':
                # Clean string data
                csv_data[column] = csv_data[column].astype(str).str.replace('\n', ' ').str.replace('\r', ' ')
                
        # Format datetime columns
        for column in csv_data.select_dtypes(include=['datetime64']).columns:
            csv_data[column] = csv_data[column].dt.strftime('%Y-%m-%d %H:%M:%S')
            
        return csv_data
    
    def _export_chunked_csv(self, 
                           data: pd.DataFrame, 
                           output_path: Path, 
                           settings: Dict[str, Any]) -> None:
        """Export large CSV in chunks."""
        chunk_size = settings['chunk_size']
        
        # Write header
        header_written = False
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size]
            
            chunk.to_csv(
                output_path,
                mode='a' if header_written else 'w',
                index=False,
                header=not header_written,
                encoding=settings['encoding'],
                float_format=settings['float_format'],
                date_format=settings['date_format']
            )
            
            header_written = True
            
            self.logger.debug(f"Exported chunk {i // chunk_size + 1}", 
                            rows=len(chunk))


class GeoJSONExporter(BaseExporter):
    """Spatial data export with geometry optimisation."""
    
    def get_optimal_settings(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Determine optimal GeoJSON settings."""
        # Check if data has geometry columns
        has_geometry = any(
            hasattr(data[col].iloc[0] if len(data) > 0 else None, '__geo_interface__')
            for col in data.columns
            if data[col].dtype == 'object'
        )
        
        return {
            'coordinate_precision': 6,  # Suitable for Australian coordinates
            'simplify_geometry': True,
            'tolerance': 0.0001,  # ~10m at equator
            'include_crs': True,
            'ensure_ascii': False
        }
    
    @monitor_performance("geojson_export")
    def export(self, 
              data: pd.DataFrame, 
              output_path: Path, 
              **kwargs) -> Dict[str, Any]:
        """Export DataFrame to GeoJSON format."""
        try:
            settings = self.get_optimal_settings(data)
            settings.update(kwargs)
            
            # Convert to GeoDataFrame if needed
            if not isinstance(data, gpd.GeoDataFrame):
                gdf = self._create_geodataframe(data)
            else:
                gdf = data.copy()
                
            # Optimise geometry if requested
            if settings['simplify_geometry'] and 'geometry' in gdf.columns:
                gdf['geometry'] = gdf['geometry'].simplify(settings['tolerance'])
                
            # Export to GeoJSON
            geojson_dict = self._to_geojson_dict(gdf, settings)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson_dict, f, 
                         ensure_ascii=settings['ensure_ascii'],
                         separators=(',', ':'))  # Compact format
                         
            file_size = output_path.stat().st_size
            
            export_info = {
                'format': 'geojson',
                'file_size_bytes': file_size,
                'file_size_mb': file_size / 1024 / 1024,
                'features': len(gdf),
                'crs': str(gdf.crs) if hasattr(gdf, 'crs') else 'EPSG:4326',
                'settings': settings
            }
            
            self.logger.info(f"GeoJSON export completed", 
                           file_size_mb=export_info['file_size_mb'],
                           features=export_info['features'])
            
            return export_info
            
        except Exception as e:
            raise LoadingError(f"GeoJSON export failed: {str(e)}")
    
    def _create_geodataframe(self, data: pd.DataFrame) -> gpd.GeoDataFrame:
        """Create GeoDataFrame from regular DataFrame."""
        # Look for coordinate columns
        lat_cols = [col for col in data.columns if any(term in col.lower() for term in ['lat', 'latitude'])]
        lon_cols = [col for col in data.columns if any(term in col.lower() for term in ['lon', 'lng', 'longitude'])]
        
        if lat_cols and lon_cols:
            lat_col = lat_cols[0]
            lon_col = lon_cols[0]
            
            # Create geometry from coordinates
            geometry = [Point(xy) for xy in zip(data[lon_col], data[lat_col])]
            gdf = gpd.GeoDataFrame(data.drop(columns=[lat_col, lon_col]), 
                                 geometry=geometry, crs='EPSG:4326')
        else:
            # No coordinates found, create empty geometry
            geometry = [Point(0, 0) for _ in range(len(data))]
            gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')
            
        return gdf
    
    def _to_geojson_dict(self, gdf: gpd.GeoDataFrame, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Convert GeoDataFrame to GeoJSON dictionary with optimisation."""
        features = []
        
        for idx, row in gdf.iterrows():
            # Create feature
            feature = {
                'type': 'Feature',
                'properties': {},
                'geometry': None
            }
            
            # Add properties (non-geometry columns)
            for col in gdf.columns:
                if col != 'geometry':
                    value = row[col]
                    # Handle special types
                    if pd.isna(value):
                        feature['properties'][col] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        feature['properties'][col] = float(value)
                    else:
                        feature['properties'][col] = str(value)
                        
            # Add geometry with coordinate precision
            if hasattr(row, 'geometry') and row.geometry is not None:
                geom_dict = row.geometry.__geo_interface__
                if settings['coordinate_precision'] is not None:
                    geom_dict = self._round_coordinates(
                        geom_dict, settings['coordinate_precision']
                    )
                feature['geometry'] = geom_dict
                
            features.append(feature)
            
        geojson_dict = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        # Add CRS if requested
        if settings['include_crs'] and hasattr(gdf, 'crs') and gdf.crs:
            geojson_dict['crs'] = {
                'type': 'name',
                'properties': {'name': str(gdf.crs)}
            }
            
        return geojson_dict
    
    def _round_coordinates(self, geom_dict: Dict[str, Any], precision: int) -> Dict[str, Any]:
        """Round coordinates to specified precision."""
        if geom_dict['type'] == 'Point':
            coords = geom_dict['coordinates']
            geom_dict['coordinates'] = [round(coord, precision) for coord in coords]
        elif geom_dict['type'] in ['LineString', 'MultiPoint']:
            coords = geom_dict['coordinates']
            geom_dict['coordinates'] = [[round(coord, precision) for coord in point] for point in coords]
        elif geom_dict['type'] in ['Polygon', 'MultiLineString']:
            coords = geom_dict['coordinates']
            geom_dict['coordinates'] = [[[round(coord, precision) for coord in point] for point in ring] for ring in coords]
        elif geom_dict['type'] == 'MultiPolygon':
            coords = geom_dict['coordinates']
            geom_dict['coordinates'] = [[[[round(coord, precision) for coord in point] for point in ring] for ring in polygon] for polygon in coords]
            
        return geom_dict


class JSONExporter(BaseExporter):
    """Structured JSON export for APIs and web platforms."""
    
    def get_optimal_settings(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Determine optimal JSON settings."""
        return {
            'orient': 'records',  # Most common for APIs
            'date_format': 'iso',
            'ensure_ascii': False,
            'indent': None,  # Compact format
            'include_metadata': True
        }
    
    @monitor_performance("json_export")
    def export(self, 
              data: pd.DataFrame, 
              output_path: Path, 
              **kwargs) -> Dict[str, Any]:
        """Export DataFrame to optimised JSON format."""
        try:
            settings = self.get_optimal_settings(data)
            settings.update(kwargs)
            
            # Prepare data for JSON
            json_data = self._prepare_for_json(data)
            
            # Create JSON structure
            if settings['include_metadata']:
                json_output = {
                    'metadata': {
                        'total_records': len(json_data),
                        'export_time': datetime.now().isoformat(),
                        'schema_version': '1.0',
                        'source': 'AHGD ETL Pipeline'
                    },
                    'data': json_data.to_dict(orient=settings['orient'])
                }
            else:
                json_output = json_data.to_dict(orient=settings['orient'])
                
            # Write JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_output, f,
                         ensure_ascii=settings['ensure_ascii'],
                         indent=settings['indent'],
                         default=str,  # Handle special types
                         separators=(',', ':') if settings['indent'] is None else (',', ': '))
                         
            file_size = output_path.stat().st_size
            
            export_info = {
                'format': 'json',
                'file_size_bytes': file_size,
                'file_size_mb': file_size / 1024 / 1024,
                'records': len(json_data),
                'orient': settings['orient'],
                'settings': settings
            }
            
            self.logger.info(f"JSON export completed", 
                           file_size_mb=export_info['file_size_mb'],
                           records=export_info['records'])
            
            return export_info
            
        except Exception as e:
            raise LoadingError(f"JSON export failed: {str(e)}")
    
    def _prepare_for_json(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for JSON export."""
        json_data = data.copy()
        
        # Handle datetime columns
        for column in json_data.select_dtypes(include=['datetime64']).columns:
            json_data[column] = json_data[column].dt.strftime('%Y-%m-%dT%H:%M:%S')
            
        # Handle NaN values
        json_data = json_data.fillna('')
        
        # Convert numpy types to native Python types
        for column in json_data.columns:
            if json_data[column].dtype == 'object':
                continue
            elif pd.api.types.is_integer_dtype(json_data[column]):
                json_data[column] = json_data[column].astype(int)
            elif pd.api.types.is_float_dtype(json_data[column]):
                json_data[column] = json_data[column].astype(float)
                
        return json_data


class WebExporter(BaseExporter):
    """Web-optimised formats with caching headers."""
    
    def get_optimal_settings(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Determine optimal web export settings."""
        return {
            'format': 'json',  # Most web-friendly
            'compression': 'gzip',
            'cache_duration': 3600,  # 1 hour
            'include_etag': True,
            'minify': True
        }
    
    @monitor_performance("web_export")
    def export(self, 
              data: pd.DataFrame, 
              output_path: Path, 
              **kwargs) -> Dict[str, Any]:
        """Export DataFrame in web-optimised format."""
        try:
            settings = self.get_optimal_settings(data)
            settings.update(kwargs)
            
            # Use appropriate exporter based on format
            if settings['format'] == 'json':
                exporter = JSONExporter()
                export_settings = {'indent': None if settings['minify'] else 2}
            elif settings['format'] == 'geojson':
                exporter = GeoJSONExporter()
                export_settings = {}
            else:
                raise LoadingError(f"Unsupported web format: {settings['format']}")
                
            # Export data
            export_info = exporter.export(data, output_path, **export_settings)
            
            # Generate cache headers
            cache_headers = self._generate_cache_headers(data, settings)
            
            # Write headers file
            headers_path = output_path.with_suffix('.headers.json')
            with open(headers_path, 'w', encoding='utf-8') as f:
                json.dump(cache_headers, f, indent=2)
                
            # Compress if requested
            if settings['compression'] == 'gzip':
                import gzip
                compressed_path = Path(f"{output_path}.gz")
                with open(output_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        f_out.writelines(f_in)
                        
                # Update export info
                export_info['compressed_file'] = str(compressed_path)
                export_info['compressed_size_bytes'] = compressed_path.stat().st_size
                
            export_info['cache_headers'] = cache_headers
            export_info['headers_file'] = str(headers_path)
            
            self.logger.info(f"Web export completed", 
                           format=settings['format'],
                           compression=settings['compression'])
            
            return export_info
            
        except Exception as e:
            raise LoadingError(f"Web export failed: {str(e)}")
    
    def _generate_cache_headers(self, 
                               data: pd.DataFrame, 
                               settings: Dict[str, Any]) -> Dict[str, str]:
        """Generate appropriate cache headers."""
        now = datetime.now()
        
        # Generate ETag based on data content
        data_hash = hash(str(data.values.tobytes()))
        etag = f'"{abs(data_hash)}"'
        
        headers = {
            'Content-Type': f'application/{settings["format"]}; charset=utf-8',
            'Cache-Control': f'public, max-age={settings["cache_duration"]}',
            'Last-Modified': now.strftime('%a, %d %b %Y %H:%M:%S GMT'),
            'Expires': (now + pd.Timedelta(seconds=settings['cache_duration'])).strftime('%a, %d %b %Y %H:%M:%S GMT')
        }
        
        if settings['include_etag']:
            headers['ETag'] = etag
            
        if settings['compression'] == 'gzip':
            headers['Content-Encoding'] = 'gzip'
            
        return headers
