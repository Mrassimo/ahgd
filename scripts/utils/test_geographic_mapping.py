#!/usr/bin/env python3
"""
Test and validation functions for geographic mapping.

This module contains comprehensive tests for the postcode to SA2 mapping
functionality to ensure accuracy and robustness.
"""

import pandas as pd
import numpy as np
from geographic_mapping import PostcodeToSA2Mapper
import sys
from pathlib import Path

def test_known_mappings():
    """Test with known postcode-SA2 relationships."""
    print("=== Testing Known Postcode-SA2 Mappings ===")
    
    mapper = PostcodeToSA2Mapper()
    
    # Test cases with known postcodes from major cities
    test_cases = [
        {"postcode": "2000", "expected_city": "Sydney", "description": "Sydney CBD"},
        {"postcode": "3000", "expected_city": "Melbourne", "description": "Melbourne CBD"},
        {"postcode": "4000", "expected_city": "Brisbane", "description": "Brisbane CBD"},
        {"postcode": "5000", "expected_city": "Adelaide", "description": "Adelaide CBD"},
        {"postcode": "6000", "expected_city": "Perth", "description": "Perth CBD"},
        {"postcode": "7000", "expected_city": "Hobart", "description": "Hobart CBD"},
        {"postcode": "0800", "expected_city": "Darwin", "description": "Darwin CBD"},
        {"postcode": "2600", "expected_city": "Deakin", "description": "Canberra CBD"},
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        postcode = test_case["postcode"]
        expected_city = test_case["expected_city"]
        description = test_case["description"]
        
        result = mapper.postcode_to_sa2(postcode)
        
        if result:
            primary_mapping = result[0]  # Highest weighted mapping
            sa2_name = primary_mapping['sa2_name']
            weight = primary_mapping['weight']
            
            # Check if expected city name appears in SA2 name
            city_found = expected_city.lower() in sa2_name.lower()
            
            status = "✓ PASS" if city_found else "✗ FAIL"
            print(f"{status} | {postcode} ({description})")
            print(f"      Primary SA2: {sa2_name} (weight: {weight:.3f})")
            print(f"      Total mappings: {len(result)}")
            
            if not city_found:
                all_passed = False
                print(f"      Expected city '{expected_city}' not found in SA2 name")
        else:
            print(f"✗ FAIL | {postcode} ({description}) - No mapping found")
            all_passed = False
        
        print()
    
    print(f"Overall result: {'All tests passed!' if all_passed else 'Some tests failed'}")
    return all_passed


def test_mapping_completeness():
    """Test the completeness of the mapping coverage."""
    print("=== Testing Mapping Completeness ===")
    
    mapper = PostcodeToSA2Mapper()
    
    # Generate a range of valid Australian postcodes
    test_postcodes = []
    
    # NSW: 1000-2999
    test_postcodes.extend([f"{i:04d}" for i in range(1000, 3000, 100)])
    
    # VIC: 3000-3999, 8000-8999
    test_postcodes.extend([f"{i:04d}" for i in range(3000, 4000, 100)])
    test_postcodes.extend([f"{i:04d}" for i in range(8000, 9000, 100)])
    
    # QLD: 4000-4999, 9000-9999
    test_postcodes.extend([f"{i:04d}" for i in range(4000, 5000, 100)])
    test_postcodes.extend([f"{i:04d}" for i in range(9000, 10000, 100)])
    
    # SA: 5000-5999
    test_postcodes.extend([f"{i:04d}" for i in range(5000, 6000, 100)])
    
    # WA: 6000-6999
    test_postcodes.extend([f"{i:04d}" for i in range(6000, 7000, 100)])
    
    # TAS: 7000-7999
    test_postcodes.extend([f"{i:04d}" for i in range(7000, 8000, 100)])
    
    # NT: 0800-0999
    test_postcodes.extend([f"{i:04d}" for i in range(800, 1000, 100)])
    
    # ACT: 0200-0299, 2600-2699
    test_postcodes.extend([f"{i:04d}" for i in range(200, 300, 50)])
    test_postcodes.extend([f"{i:04d}" for i in range(2600, 2700, 50)])
    
    coverage = mapper.validate_mapping_coverage(test_postcodes)
    
    print(f"Total postcodes tested: {coverage['total_postcodes']}")
    print(f"Mapped postcodes: {coverage['mapped_postcodes']}")
    print(f"Unmapped postcodes: {coverage['unmapped_postcodes']}")
    print(f"Coverage percentage: {coverage['coverage_percentage']:.1f}%")
    
    if coverage['unmapped_postcode_list']:
        print("\nUnmapped postcodes:")
        for pc in coverage['unmapped_postcode_list'][:10]:  # Show first 10
            print(f"  - {pc}")
        if len(coverage['unmapped_postcode_list']) > 10:
            print(f"  ... and {len(coverage['unmapped_postcode_list']) - 10} more")
    
    return coverage['coverage_percentage'] >= 30  # Expect at least 30% coverage (many test postcodes may not exist)


def test_weight_consistency():
    """Test that weights are properly normalised and sum appropriately."""
    print("\n=== Testing Weight Consistency ===")
    
    mapper = PostcodeToSA2Mapper()
    
    # Test postcodes that span multiple SA2s
    test_postcodes = ["2000", "3000", "4000", "6000"]  # Major cities likely to span multiple SA2s
    
    all_passed = True
    
    for postcode in test_postcodes:
        result = mapper.postcode_to_sa2(postcode)
        
        if len(result) > 1:  # Only test multi-mapping postcodes
            total_weight = sum(mapping['weight'] for mapping in result)
            
            # Weights should sum to approximately 1.0 (allowing for small floating point errors)
            weight_ok = abs(total_weight - 1.0) < 0.001
            
            status = "✓ PASS" if weight_ok else "✗ FAIL"
            print(f"{status} | {postcode}: {len(result)} mappings, total weight: {total_weight:.6f}")
            
            if not weight_ok:
                all_passed = False
        else:
            print(f"ℹ INFO | {postcode}: Single mapping (weight: {result[0]['weight']:.3f})")
    
    print(f"\nWeight consistency: {'All tests passed!' if all_passed else 'Some weights inconsistent'}")
    return all_passed


def test_aggregation_functionality():
    """Test the data aggregation functionality."""
    print("\n=== Testing Data Aggregation Functionality ===")
    
    mapper = PostcodeToSA2Mapper()
    
    # Create sample postcode-level data
    sample_data = pd.DataFrame({
        'postcode': ['2000', '2001', '3000', '3001', '4000', '5000'],
        'population': [15000, 12000, 25000, 18000, 20000, 16000],
        'median_income': [65000, 58000, 72000, 55000, 62000, 59000],
        'hospitals': [2, 1, 3, 2, 2, 1]
    })
    
    print("Sample data:")
    print(sample_data)
    print()
    
    # Test aggregation
    try:
        aggregated = mapper.aggregate_postcode_data_to_sa2(
            sample_data,
            postcode_col='postcode',
            value_cols=['population', 'median_income', 'hospitals'],
            method='weighted_sum'
        )
        
        print("Aggregated to SA2 level:")
        print(aggregated.head())
        print(f"\nRows after aggregation: {len(aggregated)}")
        
        # Check that we have SA2 codes and names
        has_sa2_code = 'sa2_code' in aggregated.columns
        has_sa2_name = 'sa2_name' in aggregated.columns
        has_data = len(aggregated) > 0
        
        status = "✓ PASS" if (has_sa2_code and has_sa2_name and has_data) else "✗ FAIL"
        print(f"\n{status} | Aggregation functionality")
        
        return has_sa2_code and has_sa2_name and has_data
        
    except Exception as e:
        print(f"✗ FAIL | Aggregation failed with error: {e}")
        return False


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Testing Edge Cases ===")
    
    mapper = PostcodeToSA2Mapper()
    
    # Test invalid postcodes
    invalid_postcodes = ["0000", "9999", "1234567", "ABCD"]
    
    for postcode in invalid_postcodes:
        result = mapper.postcode_to_sa2(postcode)
        print(f"Invalid postcode {postcode}: {len(result)} mappings")
    
    # Test with empty dataframe
    try:
        empty_df = pd.DataFrame(columns=['postcode', 'value'])
        result = mapper.aggregate_postcode_data_to_sa2(empty_df)
        print(f"Empty DataFrame: {len(result)} rows after aggregation")
    except Exception as e:
        print(f"Empty DataFrame error: {e}")
    
    return True


def run_comprehensive_validation():
    """Run all validation tests."""
    print("Running Comprehensive Validation of Postcode-SA2 Mapping")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Known Mappings", test_known_mappings()))
    test_results.append(("Mapping Completeness", test_mapping_completeness()))
    test_results.append(("Weight Consistency", test_weight_consistency()))
    test_results.append(("Aggregation Functionality", test_aggregation_functionality()))
    test_results.append(("Edge Cases", test_edge_cases()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} | {test_name}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{len(test_results)} tests passed")
    
    if passed_tests == len(test_results):
        print("🎉 All validation tests passed! The mapping is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
    
    return passed_tests == len(test_results)


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)