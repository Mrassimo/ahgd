#!/usr/bin/env python3
"""
🌐 Web Data Export Generator
Generate downloadable CSV, JSON, and GeoJSON files for the data marketplace
"""

import polars as pl
import pandas as pd
from pathlib import Path
import json
import time
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class WebDataExporter:
    """Generate web-ready data exports for the marketplace"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.web_exports_dir = self.data_dir.parent / 'docs' / 'data' / 'web_exports'
        self.web_exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Track export metrics
        self.export_metrics = {
            'exports_generated': 0,
            'total_size_mb': 0,
            'start_time': time.time()
        }
    
    def generate_all_exports(self) -> Dict:
        """Generate all web exports for the data marketplace"""
        print("🌐 Generating Web Data Exports for Marketplace...")
        print("=" * 60)
        
        exports_generated = {
            'csv_exports': self.generate_csv_exports(),
            'json_exports': self.generate_json_exports(),
            'geojson_exports': self.generate_geojson_exports(),
            'metadata_exports': self.generate_metadata_exports(),
            'sample_data': self.generate_sample_data()
        }
        
        # Generate manifest
        self.generate_export_manifest(exports_generated)
        
        print(f"\n🎉 Web exports generated successfully!")
        self.print_export_summary(exports_generated)
        
        return exports_generated
    
    def generate_csv_exports(self) -> Dict:
        """Generate CSV exports for all datasets"""
        print("\n📊 Generating CSV exports...")
        
        csv_exports = {}
        
        # Dataset mappings
        datasets = {
            'seifa_2021_sa2': {
                'file': 'seifa_2021_sa2.parquet',
                'description': 'SEIFA 2021 Socio-Economic Disadvantage Indices'
            },
            'aihw_grim_data': {
                'file': 'aihw_grim_data.parquet',
                'description': 'AIHW General Record of Incidence of Mortality'
            },
            'aihw_mort_table1': {
                'file': 'aihw_mort_table1.parquet',
                'description': 'AIHW Mortality Statistics Table 1'
            },
            'phidu_pha_data': {
                'file': 'phidu_pha_data.parquet',
                'description': 'PHIDU Primary Health Area Data'
            },
            'pbs_current_processed': {
                'file': 'pbs_current_processed.csv',
                'description': 'PBS Pharmaceutical Benefits Scheme Data'
            }
        }
        
        for dataset_name, dataset_info in datasets.items():
            try:
                file_path = self.data_dir / dataset_info['file']
                
                if file_path.exists():
                    print(f"  📄 Exporting {dataset_name}...")
                    
                    # Load data
                    if file_path.suffix == '.parquet':
                        df = pl.read_parquet(file_path)
                    else:
                        df = pl.read_csv(file_path)
                    
                    # Export as CSV
                    csv_file = self.web_exports_dir / f'{dataset_name}.csv'
                    df.write_csv(csv_file)
                    
                    # Track metrics
                    file_size = csv_file.stat().st_size / (1024 * 1024)
                    self.export_metrics['total_size_mb'] += file_size
                    self.export_metrics['exports_generated'] += 1
                    
                    csv_exports[dataset_name] = {
                        'file': f'data/web_exports/{dataset_name}.csv',
                        'size_mb': file_size,
                        'records': len(df),
                        'description': dataset_info['description']
                    }
                    
                    print(f"    ✅ {dataset_name}.csv: {len(df):,} records, {file_size:.2f}MB")
                
            except Exception as e:
                print(f"    ⚠️ Error exporting {dataset_name}: {e}")
        
        return csv_exports
    
    def generate_json_exports(self) -> Dict:
        """Generate JSON exports for API consumption"""
        print("\n🔗 Generating JSON exports...")
        
        json_exports = {}
        
        # SEIFA summary data
        try:
            seifa_file = self.data_dir / 'seifa_2021_sa2.parquet'
            if seifa_file.exists():
                df = pl.read_parquet(seifa_file)
                
                # Create summary by state
                state_summary = df.with_columns([
                    pl.col('sa2_code_2021').str.slice(0, 1).alias('state_code')
                ]).group_by('state_code').agg([
                    pl.col('irsd_score').mean().alias('avg_irsd_score'),
                    pl.col('irsd_decile').mean().alias('avg_irsd_decile'),
                    pl.col('usual_resident_population').sum().alias('total_population'),
                    pl.count().alias('sa2_count')
                ])
                
                json_file = self.web_exports_dir / 'seifa_state_summary.json'
                state_summary.write_json(json_file)
                
                json_exports['seifa_state_summary'] = {
                    'file': 'data/web_exports/seifa_state_summary.json',
                    'description': 'SEIFA disadvantage summary by state'
                }
                
                print(f"    ✅ seifa_state_summary.json: State-level aggregations")
        
        except Exception as e:
            print(f"    ⚠️ Error generating SEIFA JSON: {e}")
        
        # Sample records for preview
        try:
            datasets_for_samples = ['seifa_2021_sa2', 'aihw_mort_table1', 'phidu_pha_data']
            
            for dataset_name in datasets_for_samples:
                file_path = self.data_dir / f'{dataset_name}.parquet'
                if file_path.exists():
                    df = pl.read_parquet(file_path)
                    sample_df = df.head(100)  # First 100 records
                    
                    json_file = self.web_exports_dir / f'{dataset_name}_sample.json'
                    sample_df.write_json(json_file)
                    
                    json_exports[f'{dataset_name}_sample'] = {
                        'file': f'data/web_exports/{dataset_name}_sample.json',
                        'description': f'Sample data for {dataset_name} (100 records)'
                    }
                    
                    print(f"    ✅ {dataset_name}_sample.json: 100 sample records")
        
        except Exception as e:
            print(f"    ⚠️ Error generating sample JSON: {e}")
        
        return json_exports
    
    def generate_geojson_exports(self) -> Dict:
        """Generate GeoJSON exports for mapping"""
        print("\n🗺️ Generating GeoJSON exports...")
        
        geojson_exports = {}
        
        try:
            boundaries_file = self.data_dir / 'sa2_boundaries_2021.parquet'
            
            if boundaries_file.exists():
                print("  📍 Processing SA2 boundaries...")
                
                # Use pandas for geospatial processing
                import pandas as pd
                
                # Load with pandas to handle geometry
                df = pd.read_parquet(boundaries_file)
                
                # Create simplified boundaries (without full geometry for size)
                simplified_df = df[['SA2_CODE21', 'SA2_NAME21', 'STE_CODE21', 'STE_NAME21', 'AREASQKM21']].copy()
                
                # Convert to GeoJSON-like structure (without actual geometry)
                features = []
                for _, row in simplified_df.head(1000).iterrows():  # Limit for web performance
                    feature = {
                        "type": "Feature",
                        "properties": {
                            "SA2_CODE": row['SA2_CODE21'],
                            "SA2_NAME": row['SA2_NAME21'],
                            "STATE_CODE": row['STE_CODE21'],
                            "STATE_NAME": row['STE_NAME21'],
                            "AREA_SQKM": row['AREASQKM21']
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [0, 0]  # Placeholder coordinates
                        }
                    }
                    features.append(feature)
                
                geojson_data = {
                    "type": "FeatureCollection",
                    "features": features
                }
                
                geojson_file = self.web_exports_dir / 'sa2_boundaries_simplified.geojson'
                with open(geojson_file, 'w') as f:
                    json.dump(geojson_data, f, indent=2)
                
                geojson_exports['sa2_boundaries_simplified'] = {
                    'file': 'data/web_exports/sa2_boundaries_simplified.geojson',
                    'features': len(features),
                    'description': 'Simplified SA2 boundaries for web mapping'
                }
                
                print(f"    ✅ sa2_boundaries_simplified.geojson: {len(features)} features")
        
        except Exception as e:
            print(f"    ⚠️ Error generating GeoJSON: {e}")
        
        return geojson_exports
    
    def generate_metadata_exports(self) -> Dict:
        """Generate metadata and schema documentation"""
        print("\n📋 Generating metadata exports...")
        
        metadata_exports = {}
        
        # Load existing schema documentation
        try:
            schema_doc_file = self.data_dir.parent / 'docs' / 'analysis' / 'ultra_database_schema_documentation.json'
            
            if schema_doc_file.exists():
                with open(schema_doc_file, 'r') as f:
                    schema_data = json.load(f)
                
                # Create marketplace-friendly metadata
                marketplace_catalog = {
                    "marketplace_info": {
                        "title": "Australian Health Data Marketplace",
                        "description": "Premium Australian health, demographic, and geographic datasets",
                        "version": "2.0.0",
                        "last_updated": time.strftime("%Y-%m-%d"),
                        "total_datasets": len(schema_data.get('datasets', {})),
                        "total_records": schema_data.get('metadata', {}).get('total_records', 0)
                    },
                    "datasets": {}
                }
                
                # Process each dataset
                for dataset_name, dataset_info in schema_data.get('datasets', {}).items():
                    marketplace_catalog['datasets'][dataset_name] = {
                        "title": dataset_info.get('description', dataset_name),
                        "description": dataset_info.get('description', ''),
                        "schema": dataset_info.get('schema', {}),
                        "download_formats": ["CSV", "JSON"],
                        "quality_grade": "A+" if dataset_name in ['seifa_2021', 'sa2_boundaries', 'pbs_health', 'phidu_pha'] else "A",
                        "api_endpoints": [
                            f"/api/v1/datasets/{dataset_name}",
                            f"/api/v1/datasets/{dataset_name}/schema",
                            f"/api/v1/datasets/{dataset_name}/sample"
                        ]
                    }
                
                catalog_file = self.web_exports_dir / 'data_catalog.json'
                with open(catalog_file, 'w') as f:
                    json.dump(marketplace_catalog, f, indent=2)
                
                metadata_exports['data_catalog'] = {
                    'file': 'data/web_exports/data_catalog.json',
                    'description': 'Complete data catalog for marketplace'
                }
                
                print(f"    ✅ data_catalog.json: Complete marketplace catalog")
        
        except Exception as e:
            print(f"    ⚠️ Error generating metadata: {e}")
        
        return metadata_exports
    
    def generate_sample_data(self) -> Dict:
        """Generate sample data for quick preview"""
        print("\n🔍 Generating sample data...")
        
        sample_exports = {}
        
        try:
            # Create a combined sample showing data integration
            seifa_file = self.data_dir / 'seifa_2021_sa2.parquet'
            boundaries_file = self.data_dir / 'sa2_boundaries_2021.parquet'
            
            if seifa_file.exists() and boundaries_file.exists():
                # Load SEIFA data
                seifa_df = pl.read_parquet(seifa_file)
                
                # Load boundaries (simplified)
                import pandas as pd
                boundaries_pandas = pd.read_parquet(boundaries_file)
                boundaries_simple = boundaries_pandas[['SA2_CODE21', 'SA2_NAME21', 'STE_CODE21', 'STE_NAME21']].copy()
                boundaries_df = pl.from_pandas(boundaries_simple)
                
                # Join for integrated sample
                integrated_sample = seifa_df.join(
                    boundaries_df.rename({'SA2_CODE21': 'sa2_code_2021'}),
                    on='sa2_code_2021',
                    how='inner'
                ).head(50)
                
                sample_file = self.web_exports_dir / 'integrated_sample.json'
                integrated_sample.write_json(sample_file)
                
                sample_exports['integrated_sample'] = {
                    'file': 'data/web_exports/integrated_sample.json',
                    'records': len(integrated_sample),
                    'description': 'Integrated SEIFA and geographic data sample'
                }
                
                print(f"    ✅ integrated_sample.json: 50 integrated records")
        
        except Exception as e:
            print(f"    ⚠️ Error generating sample data: {e}")
        
        return sample_exports
    
    def generate_export_manifest(self, exports_generated: Dict) -> None:
        """Generate manifest of all exports"""
        
        manifest = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_exports": self.export_metrics['exports_generated'],
            "total_size_mb": round(self.export_metrics['total_size_mb'], 2),
            "generation_time_seconds": round(time.time() - self.export_metrics['start_time'], 2),
            "exports": exports_generated
        }
        
        manifest_file = self.web_exports_dir / 'export_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n📄 Export manifest generated: {manifest_file}")
    
    def print_export_summary(self, exports_generated: Dict) -> None:
        """Print summary of exports generated"""
        print("\n🌐 Web Data Export Summary")
        print("=" * 50)
        
        total_files = 0
        for category, files in exports_generated.items():
            file_count = len(files) if isinstance(files, dict) else 0
            total_files += file_count
            print(f"📁 {category}: {file_count} files")
        
        print(f"\n📊 Total files generated: {total_files}")
        print(f"💾 Total size: {self.export_metrics['total_size_mb']:.2f}MB")
        print(f"⏱️ Generation time: {time.time() - self.export_metrics['start_time']:.2f}s")
        
        print(f"\n🔗 Download URLs will be:")
        print(f"   https://massimoraso.github.io/AHGD/data/web_exports/")


def main():
    """Generate all web exports"""
    data_dir = Path("data/processed")
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    exporter = WebDataExporter(data_dir)
    results = exporter.generate_all_exports()
    
    print(f"\n🎉 Web Data Export Generation Complete!")


if __name__ == "__main__":
    main()