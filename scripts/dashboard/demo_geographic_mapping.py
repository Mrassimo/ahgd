#!/usr/bin/env python3
"""
Demonstration of Postcode to SA2 Mapping Functionality

This script demonstrates the key features of the postcode to SA2 mapping
system for health data integration.
"""

import pandas as pd
from geographic_mapping import PostcodeToSA2Mapper, postcode_to_sa2, aggregate_postcode_data_to_sa2

def demo_basic_mapping():
    """Demonstrate basic postcode to SA2 mapping."""
    print("=== Basic Postcode to SA2 Mapping ===")
    
    # Test with major Australian cities
    test_postcodes = ["2000", "3000", "4000", "5000", "6000", "7000"]
    city_names = ["Sydney", "Melbourne", "Brisbane", "Adelaide", "Perth", "Hobart"]
    
    for postcode, city in zip(test_postcodes, city_names):
        print(f"\n{city} CBD (Postcode {postcode}):")
        mappings = postcode_to_sa2(postcode)
        
        if mappings:
            for i, mapping in enumerate(mappings):
                print(f"  {i+1}. {mapping['sa2_name']} (SA2: {mapping['sa2_code']})")
                print(f"     Weight: {mapping['weight']:.3f}, Quality: {mapping['quality']}")
        else:
            print("  No mapping found")


def demo_health_data_aggregation():
    """Demonstrate aggregation of health data from postcode to SA2 level."""
    print("\n" + "="*60)
    print("=== Health Data Aggregation Example ===")
    
    # Create sample health service data at postcode level
    health_data = pd.DataFrame({
        'postcode': ['2000', '2001', '2002', '3000', '3001', '3002', '4000', '4001'],
        'gp_clinics': [15, 8, 12, 25, 18, 14, 20, 10],
        'hospital_beds': [200, 150, 100, 400, 250, 180, 300, 120],
        'pharmacies': [8, 5, 7, 12, 9, 8, 10, 6],
        'population': [15000, 12000, 14000, 25000, 18000, 16000, 20000, 13000]
    })
    
    print("\nOriginal postcode-level health data:")
    print(health_data.to_string(index=False))
    
    # Aggregate to SA2 level using weighted sum
    print("\nAggregating to SA2 level using population-weighted allocation...")
    
    sa2_data = aggregate_postcode_data_to_sa2(
        health_data,
        postcode_col='postcode',
        value_cols=['gp_clinics', 'hospital_beds', 'pharmacies', 'population'],
        method='weighted_sum'
    )
    
    print(f"\nAggregated data (showing first 10 SA2 areas):")
    display_cols = ['sa2_code', 'sa2_name', 'gp_clinics', 'hospital_beds', 'pharmacies', 'population']
    print(sa2_data[display_cols].head(10).to_string(index=False))
    
    print(f"\nSummary:")
    print(f"- Original postcodes: {len(health_data)}")
    print(f"- Resulting SA2 areas: {len(sa2_data)}")
    print(f"- Total GP clinics: {health_data['gp_clinics'].sum():.0f} → {sa2_data['gp_clinics'].sum():.0f}")
    print(f"- Total hospital beds: {health_data['hospital_beds'].sum():.0f} → {sa2_data['hospital_beds'].sum():.0f}")


def demo_coverage_validation():
    """Demonstrate coverage validation for a dataset."""
    print("\n" + "="*60)
    print("=== Coverage Validation Example ===")
    
    mapper = PostcodeToSA2Mapper()
    
    # Simulate a real health dataset with mix of urban and rural postcodes
    health_dataset_postcodes = [
        # Major cities (likely to be mapped)
        "2000", "2001", "2010", "2015", "2020",
        "3000", "3001", "3141", "3182", "3199",
        "4000", "4001", "4101", "4151", "4169",
        # Regional areas (may or may not be mapped)
        "2480", "2850", "3450", "4350", "5280",
        # Potentially problematic postcodes
        "1234", "9999", "0001"
    ]
    
    print(f"Validating coverage for {len(health_dataset_postcodes)} postcodes from health dataset...")
    
    coverage = mapper.validate_mapping_coverage(health_dataset_postcodes)
    
    print(f"\nCoverage Results:")
    print(f"- Total postcodes: {coverage['total_postcodes']}")
    print(f"- Successfully mapped: {coverage['mapped_postcodes']}")
    print(f"- Unmapped postcodes: {coverage['unmapped_postcodes']}")
    print(f"- Coverage percentage: {coverage['coverage_percentage']:.1f}%")
    
    if coverage['unmapped_postcode_list']:
        print(f"\nUnmapped postcodes:")
        for pc in coverage['unmapped_postcode_list']:
            print(f"  - {pc}")


def demo_mapping_quality():
    """Demonstrate mapping quality assessment."""
    print("\n" + "="*60)
    print("=== Mapping Quality Assessment ===")
    
    mapper = PostcodeToSA2Mapper()
    
    # Get quality summary
    quality_summary = mapper.get_mapping_quality_summary()
    print("Quality distribution across all mappings:")
    print(quality_summary.to_string(index=False))
    
    # Show examples of different mapping complexities
    print("\nMapping complexity examples:")
    
    complex_postcodes = ["2000", "3000", "6000"]  # Multi-SA2 mappings
    simple_postcodes = ["5000", "0800"]           # Single SA2 mappings
    
    print("\nComplex mappings (multiple SA2s per postcode):")
    for pc in complex_postcodes:
        mappings = postcode_to_sa2(pc)
        print(f"  {pc}: {len(mappings)} SA2 areas")
        
    print("\nSimple mappings (single SA2 per postcode):")
    for pc in simple_postcodes:
        mappings = postcode_to_sa2(pc)
        print(f"  {pc}: {len(mappings)} SA2 area(s)")


def demo_real_world_scenario():
    """Demonstrate a real-world health data integration scenario."""
    print("\n" + "="*60)
    print("=== Real-World Scenario: Hospital Discharge Data Integration ===")
    
    # Simulate hospital discharge data with postcodes
    discharge_data = pd.DataFrame({
        'postcode': ['2000', '2000', '2001', '2001', '2010', '3000', '3000', '3141', '3182', '4000'],
        'diagnosis_category': ['Respiratory', 'Cardiac', 'Respiratory', 'Injury', 'Cardiac', 
                              'Respiratory', 'Mental Health', 'Cardiac', 'Injury', 'Respiratory'],
        'length_of_stay': [3, 5, 2, 7, 4, 3, 8, 5, 1, 4],
        'cost': [2500, 4500, 1800, 6200, 3200, 2800, 7500, 4100, 1200, 3500]
    })
    
    print("Hospital discharge data (patient postcodes):")
    print(discharge_data.to_string(index=False))
    
    # Aggregate by diagnosis category for each postcode first
    postcode_summary = discharge_data.groupby(['postcode', 'diagnosis_category']).agg({
        'length_of_stay': 'mean',
        'cost': 'mean'
    }).reset_index()
    
    print(f"\nPostcode-level summary by diagnosis:")
    print(postcode_summary.to_string(index=False))
    
    # Now aggregate to SA2 level
    print(f"\nAggregating to SA2 level for geographic analysis...")
    
    # For this example, aggregate totals by postcode first
    postcode_totals = discharge_data.groupby('postcode').agg({
        'length_of_stay': 'sum',
        'cost': 'sum'
    }).reset_index()
    postcode_totals['discharge_count'] = discharge_data.groupby('postcode').size().values
    
    sa2_totals = aggregate_postcode_data_to_sa2(
        postcode_totals,
        postcode_col='postcode',
        value_cols=['length_of_stay', 'cost', 'discharge_count'],
        method='weighted_sum'
    )
    
    print(f"\nSA2-level hospital discharge summary:")
    display_cols = ['sa2_code', 'sa2_name', 'discharge_count', 'length_of_stay', 'cost']
    print(sa2_totals[display_cols].head(8).to_string(index=False))
    
    print(f"\nNow this data can be linked with SA2-level population and socioeconomic data!")


def main():
    """Run all demonstrations."""
    print("Postcode to SA2 Mapping - Demonstration Script")
    print("=" * 60)
    print("This script demonstrates the key functionality of the postcode to SA2")
    print("mapping system for Australian health data integration.")
    
    try:
        demo_basic_mapping()
        demo_health_data_aggregation()
        demo_coverage_validation()
        demo_mapping_quality()
        demo_real_world_scenario()
        
        print("\n" + "="*60)
        print("✓ All demonstrations completed successfully!")
        print("\nThe postcode to SA2 mapping system is ready for health data integration.")
        print("See docs/postcode_sa2_mapping.md for detailed documentation.")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()