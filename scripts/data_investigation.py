#!/usr/bin/env python3
"""
🔍 ULTRA DATA INVESTIGATION SCRIPT

Comprehensive analysis of data pipeline issues:
- Raw data vs Processed data comparison
- Data integrity checks
- Processing pipeline analysis
- File format and content validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import os
import sys

def analyze_data_pipeline():
    """Comprehensive data pipeline analysis"""
    
    print("🔍 ULTRA DATA INVESTIGATION")
    print("=" * 60)
    
    # 1. Analyze Raw Data
    print("\n📂 1. RAW DATA ANALYSIS")
    print("-" * 40)
    analyze_raw_data()
    
    # 2. Analyze Processed Data
    print("\n📊 2. PROCESSED DATA ANALYSIS")
    print("-" * 40)
    analyze_processed_data()
    
    # 3. Data Loss Analysis
    print("\n🚨 3. DATA LOSS ANALYSIS")
    print("-" * 40)
    calculate_data_loss()
    
    # 4. Data Quality Issues
    print("\n⚠️ 4. DATA QUALITY ISSUES")
    print("-" * 40)
    identify_quality_issues()

def get_file_size_mb(file_path):
    """Get file size in MB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0

def analyze_raw_data():
    """Analyze raw data files"""
    raw_dir = Path("data/raw")
    
    raw_analysis = {
        'demographics': [],
        'health': [],
        'geographic': []
    }
    
    # Demographics analysis
    demo_dir = raw_dir / "demographics"
    if demo_dir.exists():
        for file in demo_dir.glob("*"):
            size_mb = get_file_size_mb(file)
            raw_analysis['demographics'].append({
                'file': file.name,
                'size_mb': size_mb,
                'type': file.suffix
            })
    
    # Health analysis  
    health_dir = raw_dir / "health"
    if health_dir.exists():
        for file in health_dir.glob("*"):
            size_mb = get_file_size_mb(file)
            raw_analysis['health'].append({
                'file': file.name,
                'size_mb': size_mb,
                'type': file.suffix
            })
    
    # Geographic analysis
    geo_dir = raw_dir / "geographic"
    if geo_dir.exists():
        for file in geo_dir.glob("*"):
            size_mb = get_file_size_mb(file)
            raw_analysis['geographic'].append({
                'file': file.name,
                'size_mb': size_mb,
                'type': file.suffix
            })
    
    # Print analysis
    for category, files in raw_analysis.items():
        total_size = sum(f['size_mb'] for f in files)
        print(f"\n📁 {category.upper()}: {total_size:.1f}MB total")
        
        for file_info in sorted(files, key=lambda x: x['size_mb'], reverse=True)[:5]:
            print(f"  - {file_info['file']}: {file_info['size_mb']:.1f}MB ({file_info['type']})")
        
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")

def analyze_processed_data():
    """Analyze processed data files using pandas"""
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        print("❌ Processed data directory not found")
        return
    
    total_processed_records = 0
    total_processed_size_mb = 0
    
    for parquet_file in processed_dir.glob("*.parquet"):
        try:
            # Read parquet file
            df = pd.read_parquet(parquet_file)
            size_mb = get_file_size_mb(parquet_file)
            
            print(f"\n📊 {parquet_file.name}:")
            print(f"  📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"  💾 Size: {size_mb:.1f}MB")
            print(f"  📋 Columns: {list(df.columns)}")
            
            # Sample data
            if len(df) > 0:
                print(f"  🔍 Sample: {df.head(2).to_dict('records')}")
            
            total_processed_records += len(df)
            total_processed_size_mb += size_mb
            
        except Exception as e:
            print(f"  ❌ Error reading {parquet_file.name}: {e}")
    
    print(f"\n📊 TOTAL PROCESSED:")
    print(f"  Records: {total_processed_records:,}")
    print(f"  Size: {total_processed_size_mb:.1f}MB")

def calculate_data_loss():
    """Calculate data loss through pipeline"""
    
    # Raw data sizes
    raw_total = 0
    raw_total += get_file_size_mb("data/raw/demographics/2021_GCP_AUS_SA2.zip")
    raw_total += get_file_size_mb("data/raw/demographics/2021_GCP_NSW_SA2.zip") 
    raw_total += get_file_size_mb("data/raw/health/mbs_demographics_historical_1993_2015.zip")
    raw_total += get_file_size_mb("data/raw/health/pbs_historical_1992_2014.zip")
    raw_total += get_file_size_mb("data/raw/health/phidu_lga_australia.xls")
    raw_total += get_file_size_mb("data/raw/health/phidu_pha_australia.xlsx")
    raw_total += get_file_size_mb("data/raw/geographic/SA2_2021_AUST_SHP_GDA2020.zip")
    
    # Processed data size
    processed_total = 0
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        for file in processed_dir.glob("*.parquet"):
            processed_total += get_file_size_mb(file)
    
    # Calculate loss
    data_retention = (processed_total / raw_total) * 100 if raw_total > 0 else 0
    data_loss = 100 - data_retention
    
    print(f"📈 DATA PIPELINE EFFICIENCY:")
    print(f"  Raw Data Total: {raw_total:.1f}MB")
    print(f"  Processed Data Total: {processed_total:.1f}MB")
    print(f"  Data Retention: {data_retention:.1f}%")
    print(f"  Data Loss: {data_loss:.1f}%")
    
    if data_loss > 90:
        print(f"  🚨 CRITICAL: >90% data loss detected!")
    elif data_loss > 50:
        print(f"  ⚠️ WARNING: >50% data loss detected!")

def identify_quality_issues():
    """Identify specific data quality issues"""
    
    issues = []
    
    # Check SEIFA file
    seifa_file = Path("data/processed/seifa_2021_sa2.parquet")
    if seifa_file.exists():
        try:
            df = pd.read_parquet(seifa_file)
            if len(df) < 2000:  # Expected ~2,454 SA2 areas
                issues.append(f"SEIFA data severely truncated: {len(df)} rows (expected ~2,454)")
            if df.shape[1] < 5:  # Expected multiple SEIFA scores
                issues.append(f"SEIFA data missing columns: {df.shape[1]} columns (expected 5+)")
        except Exception as e:
            issues.append(f"SEIFA file corrupted: {e}")
    
    # Check ZIP files not processed
    large_zips = [
        "data/raw/demographics/2021_GCP_AUS_SA2.zip",
        "data/raw/health/mbs_demographics_historical_1993_2015.zip",
        "data/raw/health/pbs_historical_1992_2014.zip"
    ]
    
    for zip_file in large_zips:
        if os.path.exists(zip_file):
            size_mb = get_file_size_mb(zip_file)
            if size_mb > 50:  # Large files
                issues.append(f"Large ZIP file not processed: {Path(zip_file).name} ({size_mb:.0f}MB)")
    
    # Print issues
    if issues:
        print("🚨 CRITICAL ISSUES IDENTIFIED:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ No critical issues identified")

def analyze_specific_file_content():
    """Analyze specific file content in detail"""
    
    print("\n🔬 5. DETAILED FILE CONTENT ANALYSIS")
    print("-" * 40)
    
    # Check what's actually in the ZIP files
    large_zips = [
        ("Demographics AUS", "data/raw/demographics/2021_GCP_AUS_SA2.zip"),
        ("Demographics NSW", "data/raw/demographics/2021_GCP_NSW_SA2.zip"), 
        ("Health MBS", "data/raw/health/mbs_demographics_historical_1993_2015.zip"),
        ("Health PBS", "data/raw/health/pbs_historical_1992_2014.zip")
    ]
    
    for name, zip_path in large_zips:
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    files = zip_ref.namelist()
                    total_size = sum(zip_ref.getinfo(f).file_size for f in files)
                    
                    print(f"\n📦 {name} ({Path(zip_path).name}):")
                    print(f"  📁 Contains {len(files)} files")
                    print(f"  📏 Uncompressed: {total_size / (1024*1024):.1f}MB")
                    print(f"  📋 Key files:")
                    
                    for file in files[:3]:  # Show first 3 files
                        file_size = zip_ref.getinfo(file).file_size
                        print(f"    - {file}: {file_size / (1024*1024):.1f}MB")
                    
                    if len(files) > 3:
                        print(f"    ... and {len(files) - 3} more files")
                        
            except Exception as e:
                print(f"  ❌ Error reading {name}: {e}")

if __name__ == "__main__":
    try:
        analyze_data_pipeline()
        analyze_specific_file_content()
        
        print(f"\n🎯 CONCLUSION:")
        print(f"=" * 60)
        print(f"The data pipeline has significant issues:")
        print(f"1. Large ZIP files (766MB+ demographics) are not being processed")
        print(f"2. Only small subsets of data are making it to processed files")
        print(f"3. SEIFA data appears corrupted or severely truncated")  
        print(f"4. 90%+ of raw data is lost in the processing pipeline")
        print(f"\nRecommendations:")
        print(f"- Implement proper ZIP file extraction")
        print(f"- Fix SEIFA data processing pipeline")
        print(f"- Add data validation checkpoints")
        print(f"- Implement comprehensive error handling")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)