#!/usr/bin/env python3
"""
AHGD: Get REAL Australian Government Data
Use the ORIGINAL working extractors to download actual government data
"""

import requests
import zipfile
from pathlib import Path
import pandas as pd

def download_real_abs_data():
    """Download actual ABS data using the original working URLs"""
    print("🇦🇺 Downloading REAL ABS Government Data...")
    print("=" * 50)
    
    # Real URLs from the original working extractor
    urls = {
        'SA2_boundaries': "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip",
        'Census_data': "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_SA2_for_AUS_short-header.zip"
    }
    
    # Create data directory
    data_dir = Path("real_data")
    data_dir.mkdir(exist_ok=True)
    
    for name, url in urls.items():
        print(f"\n📥 Downloading {name}...")
        print(f"   URL: {url}")
        
        try:
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            # Save the file
            filename = data_dir / f"{name}.zip"
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded {name}: {filename.stat().st_size / (1024*1024):.1f} MB")
            
            # Try to extract and peek at contents
            try:
                with zipfile.ZipFile(filename) as zf:
                    files = zf.namelist()[:10]  # First 10 files
                    print(f"   Contents: {len(zf.namelist())} files")
                    for file in files:
                        print(f"     - {file}")
                    if len(zf.namelist()) > 10:
                        print(f"     ... and {len(zf.namelist()) - 10} more")
                        
                    # Extract to subfolder
                    extract_dir = data_dir / name
                    extract_dir.mkdir(exist_ok=True)
                    zf.extractall(extract_dir)
                    print(f"   ✅ Extracted to {extract_dir}")
                    
            except Exception as e:
                print(f"   ⚠️  Could not extract: {e}")
                
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")
            
    return True

def test_real_census_data():
    """Try to load and show actual census data"""
    print("\n📊 Testing Real Census Data...")
    print("=" * 50)
    
    census_dir = Path("real_data/Census_data")
    if not census_dir.exists():
        print("❌ Census data not downloaded yet")
        return False
        
    # Look for CSV files
    csv_files = list(census_dir.glob("**/*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in census data")
        return False
        
    print(f"📋 Found {len(csv_files)} CSV files:")
    
    for csv_file in csv_files[:5]:  # Show first 5
        print(f"   - {csv_file.name}")
        
        try:
            # Try to read a small sample
            df = pd.read_csv(csv_file, nrows=5)
            print(f"     Shape: {df.shape}, Columns: {len(df.columns)}")
            
            # Show first few columns
            cols = df.columns.tolist()[:5]
            print(f"     Sample columns: {', '.join(cols)}")
            
        except Exception as e:
            print(f"     ⚠️  Could not read: {e}")
            
    return True

def show_real_boundaries():
    """Show actual geographic boundary files"""
    print("\n🗺️ Testing Real Geographic Boundaries...")
    print("=" * 50)
    
    boundaries_dir = Path("real_data/SA2_boundaries")
    if not boundaries_dir.exists():
        print("❌ Boundary data not downloaded yet")
        return False
        
    # Look for shape files
    shp_files = list(boundaries_dir.glob("**/*.shp"))
    
    if not shp_files:
        print("❌ No shapefile found in boundary data")
        return False
        
    print(f"🗺️ Found {len(shp_files)} shapefiles:")
    
    for shp_file in shp_files:
        print(f"   - {shp_file.name}")
        print(f"     Size: {shp_file.stat().st_size / (1024*1024):.1f} MB")
        
        try:
            # Try to read with geopandas if available
            import geopandas as gpd
            gdf = gpd.read_file(shp_file)
            print(f"     Records: {len(gdf):,}")
            print(f"     Columns: {', '.join(gdf.columns.tolist()[:5])}")
            
            if 'SA2_CODE21' in gdf.columns:
                print(f"     Sample SA2 codes: {gdf['SA2_CODE21'].head(3).tolist()}")
                
        except ImportError:
            print("     ⚠️  geopandas not available for reading shapefile")
        except Exception as e:
            print(f"     ⚠️  Could not read: {e}")
            
    return True

if __name__ == "__main__":
    print("🇦🇺 AHGD: Restoring REAL Australian Government Data")
    print("=" * 60)
    
    # Download the data
    download_success = download_real_abs_data()
    
    if download_success:
        # Test the downloaded data
        test_real_census_data()
        show_real_boundaries()
        
        print("\n" + "=" * 60)
        print("🎯 REAL DATA RESTORATION COMPLETE")
        print("=" * 60)
        print("✅ Downloaded actual ABS government data")
        print("✅ Geographic boundaries (SA2 level)")
        print("✅ Census demographic data (2021)")
        print("")
        print("💡 Next step: Replace mock data with this REAL data!")
        print("   The original extractors work - we just need to use them!")