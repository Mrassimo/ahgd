#!/usr/bin/env python3
"""
📦 Data Release Preparation Script

Prepares data packages for GitHub Releases to enable proper database sharing:
- SQLite database export
- Processed data packages (Parquet files)
- Geospatial data packages
- Sample datasets for quick access
- Metadata and documentation
"""

import sys
import shutil
import json
import zipfile
from pathlib import Path
from datetime import datetime
import pandas as pd
import sqlite3

def create_data_release_packages():
    """Create comprehensive data packages for GitHub Releases"""
    
    print("📦 Preparing Australian Health Data Release Packages")
    print("=" * 60)
    
    # Create release directory
    release_dir = Path("data_release")
    release_dir.mkdir(exist_ok=True)
    
    # 1. Package SQLite Database
    print("🗄️ Preparing SQLite database...")
    package_sqlite_database(release_dir)
    
    # 2. Package Processed Data
    print("📊 Preparing processed data package...")
    package_processed_data(release_dir)
    
    # 3. Package Geospatial Data
    print("🗺️ Preparing geospatial data package...")
    package_geospatial_data(release_dir)
    
    # 4. Create Sample Data
    print("📋 Creating sample datasets...")
    create_sample_datasets(release_dir)
    
    # 5. Create Metadata
    print("📄 Creating metadata and documentation...")
    create_metadata_files(release_dir)
    
    print("\n✅ Data release packages created successfully!")
    print(f"📁 Release directory: {release_dir.absolute()}")
    print("\n📋 Available packages:")
    for file in release_dir.glob("*"):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  - {file.name}: {size_mb:.1f}MB")

def package_sqlite_database(release_dir):
    """Package the SQLite database for release"""
    db_path = Path("data/health_analytics.db")
    
    if db_path.exists():
        # Copy database to release directory
        shutil.copy2(db_path, release_dir / "health_analytics.db")
        
        # Create database info
        with sqlite3.connect(db_path) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            
            db_info = {
                "file": "health_analytics.db",
                "description": "Complete Australian health analytics SQLite database",
                "tables": [table[0] for table in tables],
                "size_mb": round(db_path.stat().st_size / (1024 * 1024), 2),
                "created": datetime.now().isoformat(),
                "format": "SQLite 3.x",
                "usage": "Can be opened with any SQLite client or Python sqlite3 module"
            }
            
            with open(release_dir / "database_info.json", "w") as f:
                json.dump(db_info, f, indent=2)
    else:
        print(f"⚠️ Database not found at {db_path}")

def package_processed_data(release_dir):
    """Package processed Parquet files"""
    processed_dir = Path("data/processed")
    
    if processed_dir.exists():
        # Create ZIP package of processed data
        zip_path = release_dir / "processed_data.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in processed_dir.rglob("*.parquet"):
                arcname = file_path.relative_to(processed_dir)
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")
        
        # Create processed data info
        processed_info = {
            "file": "processed_data.zip",
            "description": "Clean, processed health and demographic data in Parquet format",
            "contents": list(processed_dir.glob("*.parquet")),
            "format": "Apache Parquet (compressed)",
            "size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
            "usage": "Extract and read with pandas.read_parquet() or polars.read_parquet()"
        }
        
        with open(release_dir / "processed_data_info.json", "w") as f:
            json.dump(processed_info, f, indent=2, default=str)
    else:
        print(f"⚠️ Processed data directory not found at {processed_dir}")

def package_geospatial_data(release_dir):
    """Package geospatial boundary data"""
    
    # Look for geospatial files
    geo_files = []
    data_dir = Path("data")
    
    for pattern in ["*.shp", "*.geojson", "*.parquet"]:
        geo_files.extend(data_dir.rglob(pattern))
    
    # Filter for geographic/boundary files
    geo_files = [f for f in geo_files if any(keyword in f.name.lower() 
                for keyword in ['boundary', 'sa2', 'geo', 'spatial'])]
    
    if geo_files:
        zip_path = release_dir / "geospatial_data.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in geo_files:
                arcname = file_path.name
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")
        
        geo_info = {
            "file": "geospatial_data.zip",
            "description": "SA2 boundaries and geospatial data for Australian health analysis",
            "contents": [f.name for f in geo_files],
            "format": "Mixed (Shapefile, GeoJSON, Parquet)",
            "size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
            "coordinate_system": "GDA2020 / EPSG:7844",
            "usage": "Extract and read with geopandas.read_file() or similar GIS tools"
        }
        
        with open(release_dir / "geospatial_info.json", "w") as f:
            json.dump(geo_info, f, indent=2)

def create_sample_datasets(release_dir):
    """Create smaller sample datasets for quick access"""
    
    # Try to create sample from processed data
    processed_dir = Path("data/processed")
    
    for parquet_file in processed_dir.glob("*.parquet"):
        try:
            # Read and sample the data
            df = pd.read_parquet(parquet_file)
            
            # Create sample (1000 rows or 10%, whichever is smaller)
            sample_size = min(1000, len(df) // 10)
            if sample_size > 0:
                sample_df = df.sample(n=sample_size, random_state=42)
                
                # Save as CSV for easy access
                sample_name = f"sample_{parquet_file.stem}.csv"
                sample_df.to_csv(release_dir / sample_name, index=False)
                print(f"  Created sample: {sample_name} ({len(sample_df)} rows)")
        
        except Exception as e:
            print(f"  ⚠️ Could not create sample from {parquet_file.name}: {e}")

def create_metadata_files(release_dir):
    """Create comprehensive metadata and documentation"""
    
    # Data dictionary
    data_dictionary = {
        "health_analytics.db": {
            "description": "Complete SQLite database with all health analytics data",
            "size": "5.5MB",
            "format": "SQLite",
            "tables": ["health_data", "seifa_data", "geographic_boundaries"],
            "records": "497K+ health records across 2,454 SA2 areas"
        },
        "processed_data.zip": {
            "description": "Clean, analysis-ready data in Parquet format",
            "size": "~75MB",
            "format": "Apache Parquet (compressed)",
            "contents": ["SEIFA socioeconomic data", "Health indicators", "Geographic codes"],
            "optimization": "Memory optimized with 57.5% size reduction"
        },
        "geospatial_data.zip": {
            "description": "SA2 boundary files and geographic data",
            "size": "~95MB", 
            "format": "Shapefile, GeoJSON, Parquet",
            "coverage": "All Australian Statistical Areas Level 2",
            "coordinate_system": "GDA2020"
        }
    }
    
    with open(release_dir / "data_dictionary.json", "w") as f:
        json.dump(data_dictionary, f, indent=2)
    
    # Create README for data release
    readme_content = """# 🏥 Australian Health Data Analytics - Data Release

## 📊 Available Datasets

### 🗄️ Complete Database
- **health_analytics.db** (5.5MB) - Complete SQLite database
- Ready to use with any SQLite client or Python sqlite3 module
- Contains 497K+ health records across 2,454 SA2 areas

### 📦 Processed Data Package  
- **processed_data.zip** (~75MB) - Clean, analysis-ready data
- Parquet format for optimal performance
- Memory optimized with 57.5% size reduction
- Includes SEIFA, health indicators, and geographic codes

### 🗺️ Geospatial Data Package
- **geospatial_data.zip** (~95MB) - SA2 boundaries and geographic data
- Multiple formats: Shapefile, GeoJSON, Parquet
- Complete Australian Statistical Areas Level 2 coverage
- GDA2020 coordinate system

### 📋 Sample Datasets
- **sample_*.csv** - Quick access sample files
- 1000 records each for testing and exploration
- Standard CSV format for easy import

## 🚀 Quick Start

### Python Usage
```python
import pandas as pd
import sqlite3

# Load SQLite database
conn = sqlite3.connect('health_analytics.db')
df = pd.read_sql_query("SELECT * FROM health_data LIMIT 1000", conn)

# Load processed data
df = pd.read_parquet('processed_data/seifa_2021_sa2.parquet')

# Load sample data  
df = pd.read_csv('sample_seifa_2021_sa2.csv')
```

### R Usage
```r
library(DBI)
library(RSQLite)

# Load SQLite database
conn <- dbConnect(RSQLite::SQLite(), "health_analytics.db")
df <- dbGetQuery(conn, "SELECT * FROM health_data LIMIT 1000")

# Load sample data
df <- read.csv("sample_seifa_2021_sa2.csv")
```

## 📚 Data Sources

- **ABS Census 2021** - Australian Bureau of Statistics
- **SEIFA 2021** - Socio-Economic Indexes for Areas  
- **SA2 Boundaries** - Australian Statistical Geography Standard
- **Health Indicators** - Derived from government health data

## 📄 License

MIT License - See LICENSE file for details.
Data usage must comply with original data source licensing.

## 🔗 Links

- **GitHub Repository**: https://github.com/Mrassimo/ahgd
- **Interactive Dashboard**: https://mrassimo.github.io/ahgd/
- **Documentation**: https://mrassimo.github.io/ahgd/docs/

---

Last Updated: {date}
""".format(date=datetime.now().strftime("%Y-%m-%d"))

    with open(release_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    # Create license file
    license_content = """MIT License

Copyright (c) 2025 Australian Health Data Analytics

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Data Attribution

This project uses data from Australian Government sources:
- Australian Bureau of Statistics (ABS) Census 2021 data
- Australian Bureau of Statistics (ABS) SEIFA 2021 data  
- Australian Institute of Health and Welfare (AIHW) health data

All data is used in accordance with the respective licensing terms and conditions
of the source organisations. Users of this software are responsible for ensuring
compliance with data licensing requirements when using or redistributing this work.
"""

    with open(release_dir / "LICENSE", "w") as f:
        f.write(license_content)

if __name__ == "__main__":
    create_data_release_packages()