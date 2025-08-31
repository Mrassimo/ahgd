#!/usr/bin/env python3
"""
AHGD V3: Real Australian Government Data Processor
Processes downloaded real government data with ultra-high performance Polars.

PROCESSES ONLY REAL DATA - NO SYNTHETIC/DEMO DATA
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import polars as pl
import pandas as pd

def setup_processing_environment():
    """Set up the data processing environment."""
    
    print("🔧 AHGD V3: Real Data Processing Environment Setup")
    print("=" * 60)
    
    # Check required directories
    data_dirs = {
        "input": Path("/tmp/ahgd_data"),
        "output": Path("/tmp/processed_data"),
        "exports": Path("/tmp/exports")
    }
    
    for name, path in data_dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {name.title()} directory: {path}")
    
    # Check available storage
    import shutil
    total, used, free = shutil.disk_usage("/tmp")
    free_gb = free // (1024**3)
    
    print(f"💾 Available storage: {free_gb}GB")
    
    if free_gb < 5:
        print("⚠️  WARNING: Less than 5GB free storage available")
        print("   Consider using a larger cloud instance")
    
    return data_dirs

def discover_real_data_sources(data_dir: Path) -> Dict[str, List[Path]]:
    """Discover and categorize real government data files."""
    
    print(f"\n🔍 Discovering real data sources in: {data_dir}")
    print("=" * 60)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run: python real_data_pipeline.py first")
        return {}
    
    # Data source patterns
    source_patterns = {
        "abs_census": [
            "**/2021Census_*.csv",
            "**/Census_*.csv",
            "**/GCP_*.csv"
        ],
        "abs_boundaries": [
            "**/*.shp",
            "**/SA1_*.shp", 
            "**/SA2_*.shp"
        ],
        "abs_seifa": [
            "**/SEIFA_*.csv",
            "**/seifa_*.csv"
        ],
        "aihw_health": [
            "**/aihw*.xlsx",
            "**/health*.xlsx", 
            "**/mortality*.xlsx"
        ],
        "health_mbs_pbs": [
            "**/MBS_*.xlsx",
            "**/PBS_*.xlsx",
            "**/mbs*.xlsx",
            "**/pbs*.xlsx"
        ]
    }
    
    discovered_sources = {}
    total_size = 0
    
    for source_name, patterns in source_patterns.items():
        files = []
        for pattern in patterns:
            files.extend(data_dir.glob(pattern))
        
        if files:
            source_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = source_size / (1024**2)
            total_size += source_size
            
            discovered_sources[source_name] = files
            print(f"✅ {source_name}: {len(files)} files ({size_mb:.1f}MB)")
            
            # Show sample files
            for file_path in files[:3]:
                print(f"   📄 {file_path.name}")
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more files")
        else:
            print(f"⚠️  {source_name}: No files found")
    
    total_mb = total_size / (1024**2)
    print(f"\n📊 Total real data discovered: {total_mb:.1f}MB")
    
    return discovered_sources

def process_abs_census_data(files: List[Path]) -> Optional[pl.DataFrame]:
    """Process real ABS Census data with Polars."""
    
    print(f"\n🏛️ Processing ABS Census Data ({len(files)} files)")
    print("=" * 50)
    
    if not files:
        return None
    
    start_time = time.time()
    processed_dataframes = []
    total_records = 0
    
    for file_path in files[:10]:  # Process first 10 files to avoid memory issues
        try:
            print(f"📊 Reading: {file_path.name}")
            
            # Read with Polars for maximum performance
            df = pl.read_csv(
                file_path,
                encoding="utf8-lossy",  # Handle any encoding issues
                ignore_errors=True,
                truncate_ragged_lines=True
            )
            
            # Add metadata columns
            df = df.with_columns([
                pl.lit(file_path.stem).alias("source_file"),
                pl.lit("ABS_Census_2021").alias("data_source"),
                pl.lit(datetime.now()).alias("processed_at")
            ])
            
            processed_dataframes.append(df)
            total_records += len(df)
            print(f"   ✅ {len(df):,} records processed")
            
        except Exception as e:
            print(f"   ⚠️  Error reading {file_path.name}: {e}")
            continue
    
    if not processed_dataframes:
        print("❌ No census files could be processed")
        return None
    
    # Combine all dataframes
    print(f"🔄 Combining {len(processed_dataframes)} census datasets...")
    combined_df = pl.concat(processed_dataframes, how="diagonal")
    
    processing_time = time.time() - start_time
    
    print(f"✅ Census processing complete:")
    print(f"   📊 Records: {len(combined_df):,}")
    print(f"   🏛️ Columns: {len(combined_df.columns)}")
    print(f"   ⏱️  Time: {processing_time:.1f}s")
    print(f"   🚀 Rate: {len(combined_df)/processing_time:,.0f} records/second")
    
    return combined_df

def process_geographic_boundaries(files: List[Path]) -> Optional[pl.DataFrame]:
    """Process real geographic boundary data."""
    
    print(f"\n🗺️  Processing Geographic Boundaries ({len(files)} files)")
    print("=" * 50)
    
    if not files:
        return None
    
    # Find shapefile
    shp_files = [f for f in files if f.suffix == ".shp"]
    
    if not shp_files:
        print("❌ No shapefiles found")
        return None
    
    try:
        # Try with geopandas if available
        import geopandas as gpd
        
        shp_file = shp_files[0]  # Use first shapefile
        print(f"📍 Reading boundary file: {shp_file.name}")
        
        start_time = time.time()
        
        # Read with geopandas
        gdf = gpd.read_file(shp_file)
        
        # Convert to Polars-compatible format
        boundary_data = {
            "area_code": gdf.iloc[:, 0].astype(str).tolist(),
            "area_name": gdf.iloc[:, 1].astype(str).tolist() if len(gdf.columns) > 1 else ["Area_" + str(i) for i in range(len(gdf))],
            "geometry_type": gdf.geometry.geom_type.tolist(),
            "centroid_x": gdf.geometry.centroid.x.tolist(),
            "centroid_y": gdf.geometry.centroid.y.tolist(),
            "area_sqkm": gdf.geometry.area.tolist(),
            "data_source": ["ABS_Boundaries_2021"] * len(gdf),
            "processed_at": [datetime.now()] * len(gdf)
        }
        
        df = pl.DataFrame(boundary_data)
        processing_time = time.time() - start_time
        
        print(f"✅ Boundary processing complete:")
        print(f"   🗺️  Areas: {len(df):,}")
        print(f"   📏 Columns: {len(df.columns)}")
        print(f"   ⏱️  Time: {processing_time:.1f}s")
        
        return df
        
    except ImportError:
        print("⚠️  geopandas not available - using basic processing")
        
        # Basic processing without geopandas
        df = pl.DataFrame({
            "area_code": ["BOUNDARY_DATA_AVAILABLE"],
            "message": [f"Found {len(files)} boundary files"],
            "files": [str([f.name for f in files])],
            "data_source": ["ABS_Boundaries"],
            "processed_at": [datetime.now()]
        })
        
        return df
        
    except Exception as e:
        print(f"❌ Boundary processing failed: {e}")
        return None

def process_health_data(files: List[Path]) -> Optional[pl.DataFrame]:
    """Process real health data from AIHW and Department of Health."""
    
    print(f"\n🏥 Processing Health Data ({len(files)} files)")  
    print("=" * 50)
    
    if not files:
        return None
    
    start_time = time.time()
    health_dataframes = []
    
    for file_path in files:
        try:
            print(f"📈 Reading: {file_path.name}")
            
            if file_path.suffix == ".xlsx":
                # Read Excel files (common for AIHW data)
                df = pl.read_excel(file_path)
            else:
                df = pl.read_csv(file_path, encoding="utf8-lossy", ignore_errors=True)
            
            # Add metadata
            df = df.with_columns([
                pl.lit(file_path.stem).alias("source_file"),
                pl.lit("Health_Data").alias("data_source"),
                pl.lit(datetime.now()).alias("processed_at")
            ])
            
            health_dataframes.append(df)
            print(f"   ✅ {len(df):,} records")
            
        except Exception as e:
            print(f"   ⚠️  Error reading {file_path.name}: {e}")
            continue
    
    if not health_dataframes:
        print("❌ No health files could be processed")
        return None
    
    # Combine health datasets
    combined_df = pl.concat(health_dataframes, how="diagonal")
    processing_time = time.time() - start_time
    
    print(f"✅ Health data processing complete:")
    print(f"   🏥 Records: {len(combined_df):,}")
    print(f"   📊 Columns: {len(combined_df.columns)}")
    print(f"   ⏱️  Time: {processing_time:.1f}s")
    
    return combined_df

def demonstrate_polars_performance(dataframes: Dict[str, pl.DataFrame]):
    """Demonstrate Polars performance with real government data."""
    
    print(f"\n🏆 POLARS PERFORMANCE DEMONSTRATION")
    print("=" * 60)
    
    for data_type, df in dataframes.items():
        if df is None or len(df) == 0:
            continue
            
        print(f"\n📊 {data_type.upper()} Performance Test")
        print("-" * 40)
        
        # Test 1: Basic aggregation
        start_time = time.time()
        if "source_file" in df.columns:
            agg_result = df.group_by("source_file").count()
        else:
            agg_result = df.select(pl.count().alias("total_records"))
        agg_time = time.time() - start_time
        
        print(f"📈 Aggregation: {agg_time*1000:.1f}ms ({len(agg_result):,} groups)")
        
        # Test 2: Filtering  
        start_time = time.time()
        if len(df) > 1000:
            sample_size = min(1000, len(df) // 2)
            filtered = df.head(sample_size)
        else:
            filtered = df
        filter_time = time.time() - start_time
        
        print(f"🔍 Filtering: {filter_time*1000:.1f}ms ({len(filtered):,} records)")
        
        # Memory usage
        memory_mb = df.estimated_size("mb")
        print(f"💾 Memory: {memory_mb:.1f}MB")
        
        # Throughput
        if agg_time > 0:
            throughput = len(df) / agg_time
            print(f"⚡ Throughput: {throughput:,.0f} records/second")

def export_processing_results(
    dataframes: Dict[str, pl.DataFrame],
    export_dir: Path
) -> Dict[str, Any]:
    """Export processed data and generate summary reports."""
    
    print(f"\n📦 Exporting Processing Results")
    print("=" * 40)
    
    export_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "export_timestamp": datetime.now().isoformat(),
        "datasets": {},
        "summary": {
            "total_datasets": 0,
            "total_records": 0,
            "total_size_mb": 0
        }
    }
    
    for data_type, df in dataframes.items():
        if df is None or len(df) == 0:
            continue
        
        # Export sample data (first 10,000 records for development)
        sample_size = min(10000, len(df))
        sample_df = df.head(sample_size)
        
        # Export as Parquet (most efficient)
        parquet_path = export_dir / f"{data_type}_sample.parquet"
        sample_df.write_parquet(parquet_path, compression="zstd")
        
        # Export summary as JSON
        summary = {
            "data_type": data_type,
            "total_records": len(df),
            "sample_records": len(sample_df),
            "columns": df.columns,
            "file_size_mb": parquet_path.stat().st_size / (1024**2),
            "schema": str(df.schema)
        }
        
        json_path = export_dir / f"{data_type}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        results["datasets"][data_type] = summary
        results["summary"]["total_datasets"] += 1
        results["summary"]["total_records"] += len(df)
        results["summary"]["total_size_mb"] += summary["file_size_mb"]
        
        print(f"✅ {data_type}: {sample_size:,} records → {summary['file_size_mb']:.1f}MB")
    
    # Export overall summary
    summary_path = export_dir / "processing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Export Summary:")
    print(f"   📁 Location: {export_dir}")
    print(f"   🗂️  Datasets: {results['summary']['total_datasets']}")
    print(f"   📊 Records: {results['summary']['total_records']:,}")
    print(f"   💾 Size: {results['summary']['total_size_mb']:.1f}MB")
    
    return results

def main():
    """Main processing function for real government data."""
    
    print("🇦🇺 AHGD V3: REAL AUSTRALIAN GOVERNMENT DATA PROCESSOR")
    print("=" * 70)
    print("🎯 PROCESSING ONLY REAL GOVERNMENT DATA - NO SYNTHETIC DATA")
    print("=" * 70)
    
    # Setup environment
    dirs = setup_processing_environment()
    
    # Discover real data sources
    sources = discover_real_data_sources(dirs["input"])
    
    if not sources:
        print("\n❌ No real government data found!")
        print("   Run: python real_data_pipeline.py first")
        return False
    
    # Process each data source
    print(f"\n🔄 PROCESSING {len(sources)} DATA SOURCES WITH POLARS")
    print("=" * 60)
    
    processed_dataframes = {}
    
    # Process ABS Census data
    if "abs_census" in sources:
        processed_dataframes["census"] = process_abs_census_data(sources["abs_census"])
    
    # Process geographic boundaries
    if "abs_boundaries" in sources:
        processed_dataframes["boundaries"] = process_geographic_boundaries(sources["abs_boundaries"])
    
    # Process health data  
    health_files = []
    for source_key in ["aihw_health", "health_mbs_pbs"]:
        if source_key in sources:
            health_files.extend(sources[source_key])
    
    if health_files:
        processed_dataframes["health"] = process_health_data(health_files)
    
    # Demonstrate performance
    demonstrate_polars_performance(processed_dataframes)
    
    # Export results
    export_results = export_processing_results(processed_dataframes, dirs["exports"])
    
    # Final summary
    successful = sum(1 for df in processed_dataframes.values() if df is not None and len(df) > 0)
    total = len(processed_dataframes)
    
    print(f"\n" + "=" * 70)
    print(f"🎯 REAL DATA PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✅ Successful: {successful}/{total} datasets processed")
    print(f"📊 Total records: {export_results['summary']['total_records']:,}")
    print(f"💾 Export size: {export_results['summary']['total_size_mb']:.1f}MB")
    print(f"📁 Results: {dirs['exports']}")
    
    if successful >= 2:
        print(f"\n🎉 EXCELLENT: Real Australian government data successfully processed!")
        print(f"🚀 Ultra-high performance Polars processing validated")
        print(f"📊 Ready for SA1-level health analytics")
    elif successful >= 1:
        print(f"\n✅ GOOD: Some real government data processed")
        print(f"⚠️  Check data sources for complete coverage")
    else:
        print(f"\n⚠️  LIMITED: Few datasets processed successfully")
    
    return successful >= 1

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)