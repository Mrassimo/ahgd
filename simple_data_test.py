#!/usr/bin/env python3
"""
AHGD V3: Simple Real Data Test
Direct test to fetch actual Australian government health data
"""

import polars as pl
import httpx
import asyncio
import json
from datetime import datetime

async def test_abs_api():
    """Test Australian Bureau of Statistics API"""
    print("🏛️ Testing ABS (Australian Bureau of Statistics) API...")
    print("=" * 60)
    
    # ABS has multiple APIs, let's test the main ones
    test_urls = [
        "https://api.data.abs.gov.au/datastructure",
        "https://www.abs.gov.au/api/v1/statistics",
        "https://explore.data.abs.gov.au/api/",
    ]
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        for url in test_urls:
            try:
                print(f"📡 Testing: {url}")
                response = await client.get(url)
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    print("   ✅ API responding successfully")
                    content = response.text[:200] + "..." if len(response.text) > 200 else response.text
                    print(f"   Content preview: {content}")
                    return True
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}...")
        
    return False

async def test_aihw_data():
    """Test Australian Institute of Health and Welfare data"""
    print("\n🏥 Testing AIHW (Australian Institute of Health and Welfare)...")
    print("=" * 60)
    
    # AIHW doesn't have a public API, but they provide downloadable datasets
    # Let's check their main data repositories
    test_urls = [
        "https://www.aihw.gov.au/reports-data/health-conditions-disability-deaths",
        "https://www.aihw.gov.au/reports-data/population-groups/indigenous-australians",
        "https://www.aihw.gov.au/getmedia/",
    ]
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        for url in test_urls:
            try:
                print(f"📡 Testing: {url}")
                response = await client.head(url)  # Use HEAD to avoid downloading large files
                print(f"   Status: {response.status_code}")
                
                if response.status_code == 200:
                    print("   ✅ AIHW data portal accessible")
                    return True
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:100]}...")
        
    return False

async def fetch_sample_abs_data():
    """Try to fetch actual sample data from ABS"""
    print("\n📊 Attempting to fetch real ABS data...")
    print("=" * 60)
    
    # ABS provides some open datasets - let's try to get population data
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Try the ABS.Stat API
            url = "https://stat.data.abs.gov.au/rest/v1/dataflow"
            print(f"📡 Fetching ABS dataflows: {url}")
            
            response = await client.get(url)
            if response.status_code == 200:
                print("✅ Successfully connected to ABS.Stat API")
                
                # The response should be XML with available dataflows
                content = response.text
                if "dataflow" in content.lower():
                    print("✅ Found dataflow information")
                    
                    # Extract some basic info
                    lines = content.split('\n')[:20]  # First 20 lines
                    for line in lines:
                        if 'id=' in line.lower() and ('population' in line.lower() or 'health' in line.lower() or 'demographic' in line.lower()):
                            print(f"   📋 Found relevant dataset: {line.strip()[:100]}...")
                    
                    return True
                else:
                    print("⚠️  Unexpected response format")
                    print(f"   Content preview: {content[:300]}...")
            else:
                print(f"❌ Failed to connect: Status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error fetching ABS data: {str(e)[:200]}...")
            
    return False

def create_mock_australian_health_data():
    """Create realistic mock Australian health data based on actual statistics"""
    print("\n🧪 Creating Mock Australian Health Data...")
    print("=" * 60)
    
    # Create realistic Australian health data based on published statistics
    import numpy as np
    
    # Australian states and territories
    states = ['NSW', 'VIC', 'QLD', 'WA', 'SA', 'TAS', 'ACT', 'NT']
    state_populations = [8166000, 6681000, 5200000, 2667000, 1771000, 542000, 432000, 249000]
    
    # Generate SA1 codes (Statistical Area 1) - realistic format
    sa1_codes = []
    health_data = []
    
    for i, (state, pop) in enumerate(zip(states, state_populations)):
        # Each state has multiple SA1s
        num_sa1s = max(50, int(pop / 50000))  # Roughly 1 SA1 per 50k people
        
        for j in range(num_sa1s):
            sa1_code = f"{i+1:01d}{j+1000:04d}{np.random.randint(1,99):02d}"  # Realistic SA1 format
            sa1_codes.append(sa1_code)
            
            # Generate realistic health indicators based on Australian health statistics
            health_data.append({
                'sa1_code': sa1_code,
                'state': state,
                'population': np.random.randint(200, 3000),  # SA1s typically 200-3000 people
                
                # Health indicators (based on Australian health statistics)
                'diabetes_prevalence': max(0, np.random.normal(5.1, 1.5)),  # Australia ~5.1%
                'obesity_rate': max(0, np.random.normal(31.3, 5.2)),  # Australia ~31.3%
                'hypertension_rate': max(0, np.random.normal(23.8, 4.1)),  # Australia ~23.8%
                'mental_health_score': max(1, min(10, np.random.normal(6.8, 1.8))),  # 1-10 scale
                
                # Access indicators
                'gp_per_1000': max(0, np.random.normal(1.2, 0.3)),  # GPs per 1000 people
                'hospital_distance_km': max(0.5, np.random.exponential(12.5)),  # Distance to hospital
                
                # Socioeconomic (SEIFA-like)
                'seifa_score': max(1, min(10, np.random.normal(5.5, 2.1))),  # 1-10 deciles
                'median_income': max(20000, np.random.normal(52000, 18000)),  # Australian median
                'education_score': max(1, min(10, np.random.normal(6.2, 1.9))),
                
                # Demographics  
                'median_age': max(18, np.random.normal(38.2, 8.4)),  # Australian median age
                'indigenous_percent': max(0, np.random.exponential(2.8)),  # Australia ~2.8%
                'overseas_born_percent': max(0, np.random.normal(29.8, 12.3)),  # Australia ~29.8%
                
                # Environmental
                'air_quality_index': max(0, min(500, np.random.normal(45, 15))),  # Good air quality
                'green_space_percent': max(0, min(100, np.random.normal(15.2, 8.7))),
                
                # Data quality metadata
                'data_collection_date': '2024-01-01',
                'data_source': 'ABS_Census_2021',
                'confidence_score': np.random.uniform(0.7, 1.0)
            })
    
    # Convert to Polars DataFrame
    df = pl.DataFrame(health_data)
    
    print(f"✅ Created mock dataset with {df.height:,} SA1 regions")
    print(f"   States covered: {', '.join(states)}")
    print(f"   Health indicators: {len([col for col in df.columns if 'rate' in col or 'score' in col or 'prevalence' in col])}")
    
    # Show sample
    print("\n📋 Sample data:")
    print(df.head().to_pandas().round(2).to_string())
    
    # Save sample data
    output_path = "sample_australian_health_data.parquet"
    df.write_parquet(output_path)
    print(f"\n💾 Sample data saved to: {output_path}")
    
    # Calculate some interesting statistics
    print("\n📊 Quick Statistics:")
    print(f"   Average diabetes prevalence: {df['diabetes_prevalence'].mean():.2f}%")
    print(f"   Average obesity rate: {df['obesity_rate'].mean():.2f}%")
    print(f"   Median SEIFA score: {df['seifa_score'].median():.1f}")
    print(f"   Total population covered: {df['population'].sum():,}")
    
    return df

async def main():
    print("🇦🇺 AHGD V3: REAL Australian Health Data Investigation")
    print("=" * 70)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test government APIs
    abs_accessible = await test_abs_api()
    aihw_accessible = await test_aihw_data()
    
    # Try to fetch real data
    real_data_success = False
    if abs_accessible:
        real_data_success = await fetch_sample_abs_data()
    
    print("\n" + "=" * 70)
    print("🎯 REAL DATA INVESTIGATION SUMMARY")
    print("=" * 70)
    print(f"ABS API Accessible:     {'✅ YES' if abs_accessible else '❌ NO'}")
    print(f"AIHW Data Accessible:   {'✅ YES' if aihw_accessible else '❌ NO'}")
    print(f"Real Data Fetched:      {'✅ YES' if real_data_success else '❌ NO'}")
    
    if not real_data_success:
        print("\n⚠️  Unable to fetch real government data.")
        print("   This is common due to:")
        print("   • Government APIs require specific authentication")
        print("   • Rate limiting and access restrictions")
        print("   • Data is available as downloads, not APIs")
        print("   • APIs have changed since implementation")
        
        print("\n🔄 Creating realistic mock data instead...")
        mock_data = create_mock_australian_health_data()
        
        print("\n✅ SOLUTION: Use the mock data as starting point.")
        print("   • Based on real Australian health statistics")
        print("   • Includes realistic SA1 codes and indicators")
        print("   • Can be replaced with real data later")
        print("   • Perfect for development and testing")
        
        return mock_data
    else:
        print("\n🎉 SUCCESS: Real government data is accessible!")
        return None

if __name__ == "__main__":
    result = asyncio.run(main())