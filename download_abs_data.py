#!/usr/bin/env python3
"""Helper script to guide ABS data download process."""

import os
from pathlib import Path
import webbrowser


def main():
    """Guide user through ABS data download."""
    print("=" * 70)
    print("ABS Data Download Helper")
    print("=" * 70)
    print("\nThis script will help you download the required ABS data files.")
    print("Note: ABS requires registration/authentication for downloads.\n")
    
    # Create directories
    Path("data/raw/geographic").mkdir(parents=True, exist_ok=True)
    Path("data/raw/census").mkdir(parents=True, exist_ok=True)
    
    print("📁 Created data directories:")
    print("   - data/raw/geographic/")
    print("   - data/raw/census/\n")
    
    # Geographic data
    print("STEP 1: Download Geographic Boundary Files")
    print("-" * 50)
    print("Opening ABS ASGS download page...")
    
    geo_url = "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files"
    
    if input("\nOpen geographic data page in browser? (y/n): ").lower() == 'y':
        webbrowser.open(geo_url)
    else:
        print(f"Visit: {geo_url}")
    
    print("\nDownload these files:")
    print("  1. Statistical Area Level 1 (SA1) ASGS Ed 3 2021 - ~200MB")
    print("  2. Statistical Area Level 2 (SA2) ASGS Ed 3 2021 - ~100MB")
    print("  3. Statistical Area Level 3 (SA3) ASGS Ed 3 2021 - ~50MB")
    print("  4. Statistical Area Level 4 (SA4) ASGS Ed 3 2021 - ~25MB")
    print("  5. State and Territory (STE) ASGS Ed 3 2021 - ~10MB")
    print("  6. Postal Areas (POA) ASGS Ed 3 2021 - ~50MB (optional)")
    
    print(f"\n💾 Save all ZIP files to: {Path('data/raw/geographic').absolute()}")
    
    input("\nPress Enter when geographic files are downloaded...")
    
    # Census data
    print("\nSTEP 2: Download Census 2021 Data")
    print("-" * 50)
    print("Opening ABS Census DataPacks page...")
    
    census_url = "https://www.abs.gov.au/census/find-census-data/datapacks"
    
    if input("\nOpen census data page in browser? (y/n): ").lower() == 'y':
        webbrowser.open(census_url)
    else:
        print(f"Visit: {census_url}")
    
    print("\nDownload this file:")
    print("  • 2021 Census GCP All Geographies for AUS")
    print("    (General Community Profile - All Geographies)")
    print("    File: 2021_GCP_all_for_AUS_short-header.zip - ~1GB")
    
    print(f"\n💾 Save to: {Path('data/raw/census').absolute()}")
    print("   Then extract the ZIP file in place")
    
    input("\nPress Enter when census data is downloaded and extracted...")
    
    # Verify files
    print("\nSTEP 3: Verifying Downloads")
    print("-" * 50)
    
    geo_files = list(Path("data/raw/geographic").glob("*.zip"))
    census_files = list(Path("data/raw/census/extracted").glob("*/*.csv"))
    
    print(f"\n✅ Found {len(geo_files)} geographic ZIP files")
    for f in geo_files[:5]:  # Show first 5
        print(f"   - {f.name}")
    
    print(f"\n✅ Found {len(census_files)} census CSV files")
    if census_files:
        print("   Census data appears to be extracted correctly")
    else:
        print("   ⚠️  No CSV files found - make sure to extract the census ZIP")
    
    print("\n" + "=" * 70)
    print("Ready to run ETL pipeline!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run: python run_unified_etl.py --steps all")
    print("2. Monitor progress in the logs")
    print("3. Check output/ directory for results")
    
    if len(geo_files) == 0 and len(census_files) == 0:
        print("\n💡 TIP: No data files found. You can also run with mock data:")
        print("   python create_mock_data.py")
        print("   python run_unified_etl.py --steps all")


if __name__ == "__main__":
    main()