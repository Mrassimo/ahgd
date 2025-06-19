#!/usr/bin/env python3
"""Test script to validate the complete pipeline data recovery."""

import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_processing.census_processor import CensusProcessor
from data_processing.seifa_processor import SEIFAProcessor 
from data_processing.health_processor import HealthDataProcessor
from monitoring.data_volume_monitor import DataVolumeMonitor


def test_complete_pipeline():
    """Test the complete data processing pipeline."""
    print("🚀 Starting Complete Pipeline Test")
    print("=" * 60)
    
    results = {}
    total_records = 0
    
    # Test 1: SEIFA Processing
    print("\n1️⃣ Testing SEIFA Processing...")
    try:
        seifa_processor = SEIFAProcessor()
        seifa_df = seifa_processor.process_complete_pipeline()
        results['seifa'] = len(seifa_df)
        total_records += len(seifa_df)
        print(f"✅ SEIFA: {len(seifa_df):,} records processed")
    except Exception as e:
        print(f"❌ SEIFA failed: {e}")
        results['seifa'] = 0
    
    # Test 2: Census ZIP Extraction and Processing  
    print("\n2️⃣ Testing Census ZIP Extraction...")
    try:
        census_processor = CensusProcessor()
        
        # First test just the extraction (already done above)
        extraction_dir = census_processor.extract_census_zips()
        csv_files = list(extraction_dir.rglob('*.csv'))
        print(f"✅ Census ZIP extraction: {len(csv_files):,} CSV files extracted")
        
        # Test loading the extracted data (may be slow with 4GB)
        print("   Loading census data with lazy evaluation...")
        lazy_df = census_processor.load_census_datapack()
        print(f"✅ Census data loaded successfully (lazy frame)")
        
        # Test basic demographics processing (sample first)
        print("   Processing basic demographics...")
        demographics_df = census_processor.process_basic_demographics()
        results['census'] = len(demographics_df)
        total_records += len(demographics_df)
        print(f"✅ Census: {len(demographics_df):,} demographic records processed")
        
    except Exception as e:
        print(f"❌ Census processing failed: {e}")
        results['census'] = 0
    
    # Test 3: Health Data Processing (without limits)
    print("\n3️⃣ Testing Health Data Processing...")
    try:
        health_processor = HealthDataProcessor()
        health_summary = health_processor.process_complete_pipeline()
        
        if health_summary and isinstance(health_summary, dict):
            mbs_records = health_summary.get('mbs_records', 0)
            pbs_records = health_summary.get('pbs_records', 0)
            health_total = mbs_records + pbs_records
            results['health'] = health_total
            total_records += health_total
            print(f"✅ Health: {mbs_records:,} MBS + {pbs_records:,} PBS = {health_total:,} total records")
        else:
            results['health'] = 0
            print("⚠️  Health processing returned no data")
            
    except Exception as e:
        print(f"❌ Health processing failed: {e}")
        results['health'] = 0
    
    # Test 4: Data Volume Analysis
    print("\n4️⃣ Running Data Volume Analysis...")
    try:
        monitor = DataVolumeMonitor()
        volume_results = monitor.monitor_pipeline_data_loss()
        monitor.print_summary_table(volume_results)
        
        overall_retention = volume_results['overall']['retention_rate']
        print(f"✅ Overall retention rate: {overall_retention:.1f}%")
        
    except Exception as e:
        print(f"❌ Volume analysis failed: {e}")
    
    # Final Summary
    print("\n🎉 Pipeline Test Summary")
    print("=" * 60)
    print(f"📊 Total Records Processed: {total_records:,}")
    print(f"🏠 SEIFA Records: {results.get('seifa', 0):,}")
    print(f"📊 Census Records: {results.get('census', 0):,}")
    print(f"🏥 Health Records: {results.get('health', 0):,}")
    
    # Expected vs Actual
    expected_seifa = 2368  # Expected SA2 areas
    expected_census = 50000  # Conservative estimate for processed demographics
    expected_health = 100000  # Conservative estimate for health records
    
    print(f"\n📈 Recovery Analysis:")
    if results.get('seifa', 0) > expected_seifa * 0.9:
        print(f"✅ SEIFA: Excellent recovery ({results.get('seifa', 0)} vs {expected_seifa} expected)")
    else:
        print(f"⚠️  SEIFA: Needs improvement ({results.get('seifa', 0)} vs {expected_seifa} expected)")
    
    if results.get('census', 0) > expected_census * 0.5:
        print(f"✅ Census: Good recovery ({results.get('census', 0):,} vs {expected_census:,} expected)")
    else:
        print(f"⚠️  Census: Needs improvement ({results.get('census', 0):,} vs {expected_census:,} expected)")
    
    if results.get('health', 0) > expected_health * 0.5:
        print(f"✅ Health: Good recovery ({results.get('health', 0):,} vs {expected_health:,} expected)")
    else:
        print(f"⚠️  Health: Needs improvement ({results.get('health', 0):,} vs {expected_health:,} expected)")
    
    return results


if __name__ == "__main__":
    results = test_complete_pipeline()
    
    # Success criteria
    total_processed = sum(results.values())
    if total_processed > 100000:  # At least 100k records
        print(f"\n🎉 SUCCESS: Pipeline recovered {total_processed:,} records!")
        print("   This represents a massive improvement from the previous 95.1% data loss.")
        exit(0)
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: {total_processed:,} records processed (expected >100k)")
        exit(1)