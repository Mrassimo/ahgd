#!/usr/bin/env python3
"""
Example: Using the Advanced Statistical Validator

This example demonstrates how to use the AdvancedStatisticalValidator
for comprehensive statistical validation of Australian health data.

Note: This example shows the intended usage. Due to scipy dependency
compatibility issues in the current environment, you may need to
resolve numpy/scipy version conflicts before running.
"""

import logging
from typing import Dict, Any, List

# Example configuration for the Advanced Statistical Validator
ADVANCED_STATISTICAL_CONFIG = {
    'outlier_detection': {
        'iqr_multiplier': 1.5,
        'z_score_threshold': 3.0,
        'modified_z_threshold': 3.5,
        'isolation_contamination': 0.05
    },
    'distribution_validation': {
        'sex_ratio_tolerance': 0.05,  # 5% tolerance for sex distribution
        'age_distribution_bins': [0, 15, 25, 45, 65, 85, 120],
        'normality_alpha': 0.05
    },
    'reporting': {
        'include_visualisations': True,
        'include_recommendations': True,
        'summary_statistics': True,
        'include_report': True
    },
    'custom_ranges': {
        # You can add custom range validations here
        'custom_health_indicator': {
            'min_value': 0.0,
            'max_value': 100.0,
            'typical_min': 10.0,
            'typical_max': 80.0,
            'description': "Custom health indicator percentage"
        }
    }
}

def example_usage():
    """Example of how to use the AdvancedStatisticalValidator."""
    
    # Sample Australian health data
    sample_health_data = [
        {
            'sa2_code': '101011001',
            'life_expectancy_at_birth': 82.5,
            'life_expectancy_male': 80.1,
            'life_expectancy_female': 84.8,
            'mortality_rate_all_causes': 520.3,
            'infant_mortality_rate': 3.2,
            'smoking_rate': 15.4,
            'obesity_rate': 28.7,
            'diabetes_prevalence': 6.8,
            'seifa_index_disadvantage': 985.2,
            'seifa_index_advantage': 1024.7,
            'population_total': 12547,
            'median_age': 38.5,
            'population_density': 245.6,
            'gp_per_1000_population': 1.8,
            'hospital_beds_per_1000': 3.4,
            'population_male': 6200,
            'population_female': 6347
        },
        {
            'sa2_code': '101011002',
            'life_expectancy_at_birth': 81.2,
            'life_expectancy_male': 78.9,
            'life_expectancy_female': 83.6,
            'mortality_rate_all_causes': 545.1,
            'infant_mortality_rate': 4.1,
            'smoking_rate': 18.2,
            'obesity_rate': 31.4,
            'diabetes_prevalence': 8.2,
            'seifa_index_disadvantage': 892.7,
            'seifa_index_advantage': 967.3,
            'population_total': 8932,
            'median_age': 42.1,
            'population_density': 156.3,
            'gp_per_1000_population': 1.5,
            'hospital_beds_per_1000': 4.2,
            'population_male': 4401,
            'population_female': 4531
        },
        # Add problematic data for demonstration
        {
            'sa2_code': '101011003',
            'life_expectancy_at_birth': 95.0,  # Outlier - too high
            'life_expectancy_male': 92.0,
            'life_expectancy_female': 98.0,
            'mortality_rate_all_causes': 200.0,  # Inconsistent with high life expectancy
            'infant_mortality_rate': 1.0,
            'smoking_rate': 45.0,  # Very high outlier
            'obesity_rate': 60.0,  # Very high outlier
            'diabetes_prevalence': 25.0,  # Very high outlier
            'seifa_index_disadvantage': 1500.0,  # Outside valid range
            'seifa_index_advantage': 400.0,  # Outside valid range
            'population_total': 200000,  # Very large outlier
            'median_age': 25.0,
            'population_density': 10000.0,  # Very high density
            'gp_per_1000_population': 0.1,  # Very low
            'hospital_beds_per_1000': 15.0,  # Very high
            'population_male': 8000,  # Imbalanced sex ratio
            'population_female': 2000
        }
    ]
    
    try:
        # Import the validator (may fail due to scipy compatibility issues)
        from src.validators.advanced_statistical import AdvancedStatisticalValidator
        
        # Create logger
        logger = logging.getLogger('advanced_statistical_example')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Create validator instance
        validator = AdvancedStatisticalValidator(
            validator_id="example_advanced_statistical",
            config=ADVANCED_STATISTICAL_CONFIG,
            logger=logger
        )
        
        print("🔍 Running Advanced Statistical Validation...")
        print(f"📊 Validating {len(sample_health_data)} health records...")
        
        # Perform comprehensive validation
        validation_results = validator.validate(sample_health_data)
        
        # Analyse results
        error_count = sum(1 for r in validation_results if r.severity.value == 'error')
        warning_count = sum(1 for r in validation_results if r.severity.value == 'warning')
        info_count = sum(1 for r in validation_results if r.severity.value == 'info')
        
        print(f"\n📋 Validation Results Summary:")
        print(f"   Total issues found: {len(validation_results)}")
        print(f"   🔴 Errors: {error_count}")
        print(f"   🟡 Warnings: {warning_count}")
        print(f"   🔵 Information: {info_count}")
        
        # Show examples of each type of validation
        print(f"\n🔍 Detailed Findings:")
        
        # Range violations
        range_violations = [r for r in validation_results if 'range_validation' in r.rule_id]
        if range_violations:
            print(f"\n📏 Range Violations ({len(range_violations)}):")
            for result in range_violations[:3]:  # Show first 3
                print(f"   • {result.message}")
                
        # Outliers
        outlier_detections = [r for r in validation_results if 'outlier' in r.rule_id]
        if outlier_detections:
            print(f"\n📈 Outlier Detections ({len(outlier_detections)}):")
            for result in outlier_detections[:3]:  # Show first 3
                print(f"   • {result.message}")
        
        # Correlation issues
        correlation_issues = [r for r in validation_results if 'correlation' in r.rule_id]
        if correlation_issues:
            print(f"\n🔗 Correlation Issues ({len(correlation_issues)}):")
            for result in correlation_issues[:3]:  # Show first 3
                print(f"   • {result.message}")
        
        # Distribution issues
        distribution_issues = [r for r in validation_results if 'distribution' in r.rule_id]
        if distribution_issues:
            print(f"\n📊 Distribution Issues ({len(distribution_issues)}):")
            for result in distribution_issues[:3]:  # Show first 3
                print(f"   • {result.message}")
        
        # Show validation rules available
        available_rules = validator.get_validation_rules()
        print(f"\n🛠️ Available Validation Rules ({len(available_rules)}):")
        rule_types = set()
        for rule in available_rules:
            if 'range_validation' in rule:
                rule_types.add('Range Validation')
            elif 'outlier_detection' in rule:
                rule_types.add('Outlier Detection')
            elif 'correlation' in rule:
                rule_types.add('Correlation Analysis')
            elif 'distribution' in rule:
                rule_types.add('Distribution Validation')
        
        for rule_type in sorted(rule_types):
            print(f"   • {rule_type}")
        
        print(f"\n✅ Advanced statistical validation completed successfully!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Note: There may be numpy/scipy compatibility issues in the current environment.")
        print("The validator implementation is complete but requires compatible scipy installation.")
        return False
    
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_configuration_options():
    """Show available configuration options for the validator."""
    
    print("🔧 Advanced Statistical Validator Configuration Options:")
    print()
    
    print("📊 Outlier Detection:")
    print("   • iqr_multiplier: IQR method multiplier (default: 1.5)")
    print("   • z_score_threshold: Z-score threshold (default: 3.0)")
    print("   • modified_z_threshold: Modified Z-score threshold (default: 3.5)")
    print("   • isolation_contamination: Isolation Forest contamination rate (default: 0.05)")
    print()
    
    print("📈 Distribution Validation:")
    print("   • sex_ratio_tolerance: Tolerance for sex distribution imbalance (default: 0.05)")
    print("   • age_distribution_bins: Age group boundaries for distribution checks")
    print("   • normality_alpha: Significance level for normality tests (default: 0.05)")
    print()
    
    print("📋 Reporting:")
    print("   • include_visualisations: Generate visualisation suggestions")
    print("   • include_recommendations: Generate actionable recommendations")
    print("   • summary_statistics: Calculate summary statistics")
    print("   • include_report: Generate comprehensive statistical report")
    print()
    
    print("🎯 Built-in Health Data Ranges:")
    ranges = [
        "life_expectancy_at_birth (60-95 years, typical: 70-90)",
        "mortality_rate_all_causes (100-2000 per 100k, typical: 300-1200)",
        "infant_mortality_rate (0-50 per 1k births, typical: 2-10)",
        "smoking_rate (0-100%, typical: 5-30%)",
        "obesity_rate (0-100%, typical: 15-40%)",
        "diabetes_prevalence (0-100%, typical: 2-15%)",
        "seifa_index_disadvantage (400-1400, typical: 600-1200)",
        "population_total (0-100k, typical: 100-50k)",
        "median_age (15-80 years, typical: 25-55)",
        "population_density (0-50k per km², typical: 0.1-5k)"
    ]
    
    for range_desc in ranges:
        print(f"   • {range_desc}")


if __name__ == "__main__":
    print("🏥 Advanced Statistical Validator for Australian Health Data")
    print("=" * 60)
    print()
    
    show_configuration_options()
    print()
    print("=" * 60)
    print()
    
    success = example_usage()
    
    if not success:
        print()
        print("💡 To resolve scipy compatibility issues:")
        print("   1. Consider downgrading numpy: pip install 'numpy<2.0'")
        print("   2. Or upgrade scipy: pip install --upgrade scipy")
        print("   3. Or use conda for better dependency management")
    
    print()
    print("📝 For more information, see:")
    print("   • src/validators/advanced_statistical.py")
    print("   • tests/unit/test_validators.py (TestAdvancedStatisticalValidator)")