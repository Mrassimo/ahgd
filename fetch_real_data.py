#!/usr/bin/env python3
"""
AHGD V3: Real Data Fetcher
Test script to fetch actual Australian health data from government sources
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from extractors.polars_abs_extractor import PolarsABSExtractor
    from extractors.polars_aihw_extractor import PolarsAIHWExtractor
    from utils.config import get_config
    from utils.logging import get_logger
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Available modules:")
    import os

    for root, dirs, files in os.walk("src"):
        for file in files:
            if file.endswith(".py"):
                print(f"  {os.path.join(root, file)}")
    sys.exit(1)

logger = get_logger(__name__)


async def test_abs_data_extraction():
    """Test ABS (Australian Bureau of Statistics) data extraction"""
    print("🏛️ Testing ABS Data Extraction...")
    print("=" * 50)

    try:
        extractor = PolarsABSExtractor()

        # Test basic connection
        print("📡 Testing ABS API connection...")
        test_data = await extractor.test_api_connection()

        if test_data:
            print("✅ Connected to ABS API successfully")
            print(f"   Available datasets: {len(test_data.get('datasets', []))}")

            # Try to extract a small sample of census data
            print("\n📊 Extracting sample census data...")
            census_sample = await extractor.extract_census_sample(limit=100)

            if census_sample is not None and census_sample.height > 0:
                print(f"✅ Extracted {census_sample.height} census records")
                print(f"   Columns: {census_sample.columns}")
                print("\n📋 Sample data:")
                print(census_sample.head().to_pandas().to_string())

                return True
            else:
                print("❌ No census data retrieved")
                return False
        else:
            print("❌ Failed to connect to ABS API")
            return False

    except Exception as e:
        print(f"❌ ABS extraction failed: {e}")
        return False


async def test_aihw_data_extraction():
    """Test AIHW (Australian Institute of Health and Welfare) data extraction"""
    print("\n🏥 Testing AIHW Data Extraction...")
    print("=" * 50)

    try:
        extractor = PolarsAIHWExtractor()

        # Test health indicators extraction
        print("📡 Testing AIHW API connection...")
        test_data = await extractor.test_api_connection()

        if test_data:
            print("✅ Connected to AIHW API successfully")

            # Try to extract health indicators sample
            print("\n🏥 Extracting sample health indicators...")
            health_sample = await extractor.extract_health_indicators_sample(limit=50)

            if health_sample is not None and health_sample.height > 0:
                print(f"✅ Extracted {health_sample.height} health indicator records")
                print(f"   Columns: {health_sample.columns}")
                print("\n📋 Sample data:")
                print(health_sample.head().to_pandas().to_string())

                return True
            else:
                print("❌ No health data retrieved")
                return False
        else:
            print("❌ Failed to connect to AIHW API")
            return False

    except Exception as e:
        print(f"❌ AIHW extraction failed: {e}")
        return False


async def check_available_apis():
    """Check what government APIs are actually accessible"""
    print("\n🔍 Checking Available Government APIs...")
    print("=" * 50)

    import httpx

    apis_to_check = [
        {
            "name": "ABS Statistics API",
            "url": "https://api.data.abs.gov.au",
            "test_endpoint": "/datastructure",
        },
        {
            "name": "ABS Census API",
            "url": "https://api.census.abs.gov.au",
            "test_endpoint": "/health",
        },
        {
            "name": "AIHW Data API",
            "url": "https://www.aihw.gov.au/reports-data",
            "test_endpoint": "",
        },
    ]

    async with httpx.AsyncClient(timeout=10.0) as client:
        for api in apis_to_check:
            try:
                print(f"📡 Testing {api['name']}...")
                response = await client.get(api["url"] + api["test_endpoint"])

                if response.status_code == 200:
                    print(f"✅ {api['name']}: Available (Status: {response.status_code})")
                elif response.status_code == 404:
                    print(f"⚠️  {api['name']}: Endpoint not found but server responding")
                else:
                    print(f"⚠️  {api['name']}: Responding with status {response.status_code}")

            except Exception as e:
                print(f"❌ {api['name']}: Not accessible ({str(e)[:50]}...)")


async def main():
    print("🇦🇺 AHGD V3: Real Australian Health Data Extraction Test")
    print("=" * 60)

    # Check API availability first
    await check_available_apis()

    # Test extractors
    abs_success = await test_abs_data_extraction()
    aihw_success = await test_aihw_data_extraction()

    print("\n" + "=" * 60)
    print("🎯 EXTRACTION TEST SUMMARY")
    print("=" * 60)
    print(f"ABS Data Extraction:   {'✅ SUCCESS' if abs_success else '❌ FAILED'}")
    print(f"AIHW Data Extraction:  {'✅ SUCCESS' if aihw_success else '❌ FAILED'}")

    if abs_success or aihw_success:
        print(
            "\n🎉 Real data extraction is working! Run full pipeline to download complete datasets."
        )
    else:
        print("\n⚠️  No real data extracted. This may be due to:")
        print("   - API endpoints changed or require authentication")
        print("   - Network connectivity issues")
        print("   - Rate limiting from government APIs")
        print("   - Mock data sources need to be created for development")


if __name__ == "__main__":
    asyncio.run(main())
