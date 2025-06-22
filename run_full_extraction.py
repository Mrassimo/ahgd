#!/usr/bin/env python3
"""
Run FULL data extraction from all Australian government sources.
This will extract REAL data for all 2,473 SA2 areas across Australia.
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.extractors.extractor_registry import ExtractorRegistry
from src.utils.logging import get_logger

logger = get_logger(__name__)

def main():
    print("🇦🇺 AHGD FULL DATA EXTRACTION")
    print("=" * 50)
    print("This will extract REAL data from Australian government sources")
    print("Expected: 2,473 SA2 areas across all of Australia")
    print("Estimated time: 30-60 minutes")
    print("Estimated size: 500MB-1GB")
    print()
    
    # Check for --no-confirm flag
    if '--no-confirm' not in sys.argv:
        try:
            response = input("Ready to extract full Australian health data? (yes/no): ")
            if response.lower() != 'yes':
                print("Extraction cancelled.")
                return 0
        except EOFError:
            print("\n⚠️  Running in non-interactive mode. Use --no-confirm to skip confirmation.")
            print("Proceeding with extraction...")
    else:
        print("✅ Running with --no-confirm flag, proceeding with extraction...")
    
    print("\n🚀 Starting full extraction...")
    start_time = time.time()
    
    # Initialize registry
    registry = ExtractorRegistry()
    extractors = registry.list_extractors()
    print(f"\n📊 Found {len(extractors)} data sources to extract from")
    
    # Create output directory
    output_dir = Path("data_raw_full")
    output_dir.mkdir(exist_ok=True)
    
    # Get extraction order based on dependencies
    extraction_order = registry.get_extraction_order()
    print(f"\n📋 Extraction order determined by dependencies:")
    for i, ext_type in enumerate(extraction_order, 1):
        print(f"  {i}. {ext_type}")
    
    # Track results
    results = {
        'start_time': datetime.now().isoformat(),
        'extractors': {},
        'total_records': 0,
        'total_files': 0,
        'errors': []
    }
    
    # Extract from each source
    for ext_type in extraction_order:
        ext_id = str(ext_type).split('.')[-1].lower()
        print(f"\n{'='*50}")
        print(f"📥 Extracting: {ext_type}")
        print(f"{'='*50}")
        
        try:
            # Get extractor instance
            extractor = registry.get_extractor(ext_id)
            if not extractor:
                print(f"❌ Failed to create extractor for {ext_id}")
                results['errors'].append(f"Failed to create {ext_id}")
                continue
            
            # Run extraction
            print(f"🔄 Running {ext_id} extraction...")
            source_output = output_dir / ext_id
            
            # Call the actual extract method
            extraction_result = extractor.extract(
                output_dir=str(source_output),
                format='parquet',  # Use parquet for efficiency
                compress=True
            )
            
            # Check results
            if source_output.exists():
                files = list(source_output.rglob("*.parquet"))
                record_count = 0
                
                # Count records in parquet files
                try:
                    import pandas as pd
                    for file in files:
                        df = pd.read_parquet(file)
                        record_count += len(df)
                except:
                    record_count = "unknown"
                
                print(f"✅ Success: {len(files)} files, {record_count} records")
                results['extractors'][ext_id] = {
                    'status': 'success',
                    'files': len(files),
                    'records': record_count,
                    'path': str(source_output)
                }
                results['total_files'] += len(files)
                if isinstance(record_count, int):
                    results['total_records'] += record_count
            else:
                print(f"⚠️  No data extracted for {ext_id}")
                results['extractors'][ext_id] = {
                    'status': 'no_data'
                }
                
        except Exception as e:
            print(f"❌ Error extracting {ext_id}: {str(e)}")
            results['errors'].append(f"{ext_id}: {str(e)}")
            results['extractors'][ext_id] = {
                'status': 'error',
                'error': str(e)
            }
    
    # Summary
    elapsed = time.time() - start_time
    results['end_time'] = datetime.now().isoformat()
    results['elapsed_minutes'] = round(elapsed / 60, 2)
    
    print(f"\n{'='*50}")
    print("📊 EXTRACTION SUMMARY")
    print(f"{'='*50}")
    print(f"⏱️  Time: {results['elapsed_minutes']} minutes")
    print(f"📁 Files: {results['total_files']}")
    print(f"📈 Records: {results['total_records']:,}")
    print(f"✅ Success: {sum(1 for e in results['extractors'].values() if e.get('status') == 'success')}/{len(extraction_order)}")
    
    if results['errors']:
        print(f"\n❌ Errors ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")
    
    # Save results
    with open(output_dir / "extraction_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Full report saved to: {output_dir}/extraction_report.json")
    
    # Check total size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    print(f"💾 Total data size: {total_size / 1024 / 1024:.1f} MB")
    
    if results['total_records'] > 100000:  # If we got substantial data
        print("\n✅ Full extraction complete! Ready for processing pipeline.")
    else:
        print("\n⚠️  Extraction complete but data seems limited. Check report for details.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())