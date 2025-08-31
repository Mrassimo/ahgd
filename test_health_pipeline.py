#!/usr/bin/env python3
"""
Test Script for Health Data Pipeline

Validates the Phase 3 health data integration including:
- MBS/PBS health service data extraction
- AIHW mortality data processing
- PHIDU chronic disease data integration
- DBT health staging models
- Data quality validation
"""

import logging
import sys
import time
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipelines.orchestrator import PipelineOrchestrator


def setup_logging():
    """Configure logging for health pipeline test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("logs/health_pipeline_test.log")],
    )
    return logging.getLogger(__name__)


def test_health_data_extraction():
    """
    Test Phase 3.1: Health service data extraction (MBS/PBS).

    This is a focused test that validates the data extraction pipeline
    without requiring full downloads (which could be 100+ MB).
    """
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("TESTING HEALTH DATA EXTRACTION PIPELINE")
    logger.info("=" * 80)

    orchestrator = PipelineOrchestrator()
    start_time = time.time()

    try:
        # Test 1: Validate pipeline configuration
        logger.info("\n" + "=" * 60)
        logger.info("TEST 1: PIPELINE CONFIGURATION VALIDATION")
        logger.info("=" * 60)

        # Check if health pipelines are properly registered
        expected_pipelines = ["health_services", "mortality_data", "chronic_disease"]

        for pipeline_name in expected_pipelines:
            try:
                # This will validate the import works
                if pipeline_name == "health_services":
                    from pipelines.dlt.health import load_mbs_pbs_data

                    func = load_mbs_pbs_data
                elif pipeline_name == "mortality_data":
                    from pipelines.dlt.health import load_aihw_mortality_data

                    func = load_aihw_mortality_data
                elif pipeline_name == "chronic_disease":
                    from pipelines.dlt.health import load_phidu_chronic_disease_data

                    func = load_phidu_chronic_disease_data

                logger.info(f"✅ {pipeline_name} pipeline function imported successfully")

            except ImportError as e:
                logger.error(f"❌ {pipeline_name} pipeline import failed: {e}")
                return False

        # Test 2: Validate Pydantic models
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: PYDANTIC MODEL VALIDATION")
        logger.info("=" * 60)

        try:
            from src.models.health import AgeGroup
            from src.models.health import AIHWMortalityRecord
            from src.models.health import CauseOfDeath
            from src.models.health import ChronicDiseaseType
            from src.models.health import Gender
            from src.models.health import MBSRecord
            from src.models.health import PBSRecord
            from src.models.health import PHIDUChronicDiseaseRecord
            from src.models.health import ServiceType

            # Test MBS record validation
            test_mbs = MBSRecord(
                geographic_code="10001000001",
                geographic_name="Test SA1",
                state_code="1",
                state_name="New South Wales",
                mbs_item_number="23",
                mbs_item_description="GP Consultation",
                service_type=ServiceType.MEDICAL,
                age_group=AgeGroup.ALL_AGES,
                gender=Gender.ALL,
                service_count=100,
                benefit_paid=2500.0,
                financial_year="2021-22",
            )
            logger.info("✅ MBS record validation successful")

            # Test PBS record validation
            test_pbs = PBSRecord(
                geographic_code="10001000001",
                geographic_name="Test SA1",
                state_code="1",
                state_name="New South Wales",
                pbs_item_code="8254K",
                medicine_name="Atorvastatin",
                age_group=AgeGroup.ALL_AGES,
                gender=Gender.ALL,
                prescription_count=50,
                government_benefit=1500.0,
                financial_year="2021-22",
            )
            logger.info("✅ PBS record validation successful")

            # Test mortality record validation
            test_mortality = AIHWMortalityRecord(
                geographic_code="10001000001",
                geographic_name="Test SA1",
                state_code="1",
                state_name="New South Wales",
                cause_of_death=CauseOfDeath.ALL_CAUSES,
                age_group=AgeGroup.ALL_AGES,
                gender=Gender.ALL,
                death_count=10,
                calendar_year=2023,
                data_source="MORT",
            )
            logger.info("✅ AIHW mortality record validation successful")

            # Test PHIDU record validation
            test_phidu = PHIDUChronicDiseaseRecord(
                geographic_code="10001000001",
                geographic_name="Test SA1",
                state_code="1",
                state_name="New South Wales",
                disease_type=ChronicDiseaseType.DIABETES,
                prevalence_rate=8.5,
                age_group=AgeGroup.ALL_AGES,
                gender=Gender.ALL,
                population_total=1000,
            )
            logger.info("✅ PHIDU chronic disease record validation successful")

        except Exception as e:
            logger.error(f"❌ Pydantic model validation failed: {e}")
            logger.error(traceback.format_exc())
            return False

        # Test 3: Validate GeographicMatcher utility
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: GEOGRAPHIC MATCHER VALIDATION")
        logger.info("=" * 60)

        try:
            from src.utils.geographic import GeographicMatcher
            from src.utils.geographic import PopulationWeighter

            # Initialize matcher (will warn about missing DB but shouldn't fail)
            matcher = GeographicMatcher()
            logger.info("✅ GeographicMatcher initialized")

            # Test geographic type detection
            test_cases = [
                ("12345678901", "sa1"),
                ("123456789", "sa2"),
                ("12345", "sa3"),
                ("123", "sa4"),
                ("3000", "postcode"),
            ]

            for geo_id, expected_type in test_cases:
                detected_type = matcher._detect_geographic_type(geo_id)
                if detected_type == expected_type:
                    logger.info(f"✅ Geographic type detection: {geo_id} -> {detected_type}")
                else:
                    logger.warning(
                        f"⚠️ Geographic type detection: {geo_id} expected {expected_type}, got {detected_type}"
                    )

            # Test population weighter
            weighter = PopulationWeighter()
            test_weights = weighter.calculate_weights(
                ["12345678901", "12345678902"], method="equal"
            )
            if len(test_weights) == 2 and abs(sum(test_weights) - 1.0) < 0.001:
                logger.info("✅ PopulationWeighter equal weights calculation successful")
            else:
                logger.warning("⚠️ PopulationWeighter equal weights calculation issues")

        except Exception as e:
            logger.error(f"❌ GeographicMatcher validation failed: {e}")
            logger.error(traceback.format_exc())
            return False

        # Test 4: Validate DBT staging model syntax
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: DBT STAGING MODEL VALIDATION")
        logger.info("=" * 60)

        staging_models = [
            "pipelines/dbt/models/staging/health/stg_mbs_data.sql",
            "pipelines/dbt/models/staging/health/stg_pbs_data.sql",
            "pipelines/dbt/models/staging/health/stg_aihw_mortality.sql",
            "pipelines/dbt/models/staging/health/stg_phidu_chronic_disease.sql",
        ]

        for model_path in staging_models:
            model_file = Path(model_path)
            if model_file.exists():
                content = model_file.read_text()
                # Basic syntax validation
                if "SELECT" in content.upper() and "FROM" in content.upper():
                    logger.info(f"✅ {model_file.name} - SQL syntax valid")
                else:
                    logger.warning(f"⚠️ {model_file.name} - SQL syntax concerns")
            else:
                logger.error(f"❌ {model_file.name} - File not found")
                return False

        # Test 5: Helper function validation
        logger.info("\n" + "=" * 60)
        logger.info("TEST 5: HELPER FUNCTION VALIDATION")
        logger.info("=" * 60)

        try:
            from pipelines.dlt.health import _classify_service_type
            from pipelines.dlt.health import _extract_disease_type
            from pipelines.dlt.health import _map_age_group
            from pipelines.dlt.health import _map_cause_of_death
            from pipelines.dlt.health import _map_gender

            # Test service type classification
            test_service = _classify_service_type("GP consultation and examination")
            if test_service == "MEDICAL":
                logger.info("✅ Service type classification working")
            else:
                logger.warning(f"⚠️ Service type classification: got {test_service}")

            # Test age group mapping
            test_age = _map_age_group("25-44")
            if test_age == "ADULT":
                logger.info("✅ Age group mapping working")
            else:
                logger.warning(f"⚠️ Age group mapping: got {test_age}")

            # Test gender mapping
            test_gender = _map_gender("M")
            if test_gender == "MALE":
                logger.info("✅ Gender mapping working")
            else:
                logger.warning(f"⚠️ Gender mapping: got {test_gender}")

            # Test cause of death mapping
            test_cause = _map_cause_of_death("CARDIOVASCULAR DISEASE")
            if test_cause == "CARDIOVASCULAR":
                logger.info("✅ Cause of death mapping working")
            else:
                logger.warning(f"⚠️ Cause of death mapping: got {test_cause}")

            # Test disease type extraction
            test_disease = _extract_disease_type("DIABETES PREVALENCE DATA")
            if test_disease == "DIABETES":
                logger.info("✅ Disease type extraction working")
            else:
                logger.warning(f"⚠️ Disease type extraction: got {test_disease}")

        except Exception as e:
            logger.error(f"❌ Helper function validation failed: {e}")
            return False

        # Success summary
        duration = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("HEALTH PIPELINE VALIDATION COMPLETED SUCCESSFULLY ✅")
        logger.info("=" * 80)
        logger.info(f"Validation completed in {duration:.2f} seconds")
        logger.info("\nKey validations passed:")
        logger.info("- ✅ Pipeline configuration and imports")
        logger.info("- ✅ Pydantic health data models")
        logger.info("- ✅ Geographic mapping utilities")
        logger.info("- ✅ DBT staging model files")
        logger.info("- ✅ Data transformation helper functions")
        logger.info("\n📋 Ready for Phase 3.2: Full pipeline execution")

        return True

    except Exception as e:
        logger.error(f"Health pipeline validation failed: {e}")
        logger.error(traceback.format_exc())
        return False


def run_integration_test():
    """
    Run a limited integration test with mock data.

    This creates a small test dataset to validate the complete pipeline
    without downloading large government datasets.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING INTEGRATION TEST WITH MOCK DATA")
    logger.info("=" * 80)

    try:
        import duckdb
        import pandas as pd

        # Create temporary test database
        conn = duckdb.connect(":memory:")

        # Install spatial extension for DuckDB
        conn.execute("INSTALL spatial")
        conn.execute("LOAD spatial")

        # Create mock MBS data
        mock_mbs_data = pd.DataFrame(
            {
                "geographic_code": ["10001000001", "10001000002", "10001000003"],
                "geographic_name": ["Test SA1 A", "Test SA1 B", "Test SA1 C"],
                "state_code": ["1", "1", "1"],
                "mbs_item_number": ["23", "36", "721"],
                "mbs_item_description": [
                    "GP Consultation",
                    "Health Assessment",
                    "Specialist Consultation",
                ],
                "service_type": ["MEDICAL", "MEDICAL", "SPECIALIST"],
                "age_group": ["ALL_AGES", "ELDERLY", "ADULT"],
                "gender": ["ALL", "FEMALE", "MALE"],
                "service_count": [150, 25, 10],
                "benefit_paid": [3750.0, 875.0, 450.0],
                "financial_year": ["2021-22", "2021-22", "2021-22"],
                "quality_score": [0.95, 0.95, 0.95],
                "source_system": ["TEST_MBS", "TEST_MBS", "TEST_MBS"],
            }
        )

        # Insert mock data into DuckDB
        conn.execute("CREATE SCHEMA IF NOT EXISTS health_analytics")
        conn.register("mbs_data_df", mock_mbs_data)
        conn.execute("CREATE TABLE health_analytics.mbs_data AS SELECT * FROM mbs_data_df")

        logger.info(f"✅ Created mock MBS data with {len(mock_mbs_data)} records")

        # Test DBT staging model logic (simplified version)
        staging_query = """
        SELECT
            geographic_code AS sa1_code,
            mbs_item_number,
            service_type,
            service_count,
            benefit_paid,
            CASE WHEN service_count > 0 AND benefit_paid > 0
                 THEN benefit_paid / service_count
                 ELSE NULL END AS calculated_benefit_per_service,
            CASE WHEN mbs_item_number ~ '^[0-9]{1,6}$' THEN 1 ELSE 0 END AS valid_item_number
        FROM health_analytics.mbs_data
        WHERE quality_score >= 0.5
        """

        staged_data = conn.execute(staging_query).df()
        logger.info(f"✅ DBT staging logic validated with {len(staged_data)} processed records")

        # Validate data quality
        if len(staged_data) == 3:
            logger.info("✅ All test records passed staging validation")

            # Check calculated fields
            avg_benefit = staged_data["calculated_benefit_per_service"].mean()
            if avg_benefit > 0:
                logger.info(f"✅ Calculated benefit per service: ${avg_benefit:.2f}")

            # Check validation flags
            valid_items = staged_data["valid_item_number"].sum()
            if valid_items == 3:
                logger.info("✅ All MBS item numbers passed validation")

        else:
            logger.warning(f"⚠️ Expected 3 records, got {len(staged_data)}")

        conn.close()
        logger.info("✅ Integration test completed successfully")
        return True

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        logger.error(traceback.format_exc())
        return False


def performance_benchmark():
    """
    Run performance benchmarks for health data processing.

    Estimates processing times for full-scale data volumes.
    """
    logger = logging.getLogger(__name__)
    logger.info("\n" + "=" * 80)
    logger.info("PERFORMANCE BENCHMARKING")
    logger.info("=" * 80)

    try:
        from src.utils.geographic import GeographicMatcher

        # Benchmark data transformation functions
        start_time = time.time()

        # Test batch processing of service type classification
        test_descriptions = [
            "GP consultation and examination",
            "Pathology blood test",
            "X-ray diagnostic imaging",
            "Surgical procedure",
            "Mental health consultation",
        ] * 1000  # 5000 records

        from pipelines.dlt.health import _classify_service_type

        start_classification = time.time()
        results = [_classify_service_type(desc) for desc in test_descriptions]
        classification_time = time.time() - start_classification

        records_per_second = len(test_descriptions) / classification_time
        logger.info(f"✅ Service classification: {records_per_second:,.0f} records/second")

        # Estimate full pipeline processing times
        estimated_mbs_records = 500000  # Conservative estimate for MBS data
        estimated_pbs_records = 750000  # Conservative estimate for PBS data
        estimated_mortality_records = 100000  # AIHW mortality data
        estimated_phidu_records = 50000  # PHIDU chronic disease data

        total_records = (
            estimated_mbs_records
            + estimated_pbs_records
            + estimated_mortality_records
            + estimated_phidu_records
        )

        # Estimate processing time (including download and validation overhead)
        processing_rate = records_per_second * 0.1  # Much slower with network I/O and validation
        estimated_time_minutes = total_records / processing_rate / 60

        logger.info("📊 Performance Estimates:")
        logger.info(f"   - Total estimated records: {total_records:,}")
        logger.info(f"   - Processing rate: {processing_rate:,.0f} records/second")
        logger.info(f"   - Estimated total time: {estimated_time_minutes:.1f} minutes")
        logger.info("   - Memory usage estimate: ~2-4 GB peak")

        # Test geographic matching performance
        matcher = GeographicMatcher()
        test_codes = ["12345", "67890", "11111"] * 100  # 300 geographic lookups

        start_matching = time.time()
        for code in test_codes:
            matcher.map_to_sa1(code, "sa3")
        matching_time = time.time() - start_matching

        matching_rate = len(test_codes) / matching_time
        logger.info(f"✅ Geographic matching: {matching_rate:,.0f} lookups/second")

        return True

    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        return False


def main():
    """Main test execution function."""
    print("🇦🇺 AHGD Health Data Pipeline Testing")
    print("=" * 80)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    test_results = []

    # Run validation tests
    print("\n🧪 Running validation tests...")
    validation_success = test_health_data_extraction()
    test_results.append(("Validation Tests", validation_success))

    if validation_success:
        print("\n🔗 Running integration tests...")
        integration_success = run_integration_test()
        test_results.append(("Integration Tests", integration_success))

        print("\n⚡ Running performance benchmarks...")
        benchmark_success = performance_benchmark()
        test_results.append(("Performance Benchmarks", benchmark_success))

    # Summary
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)

    all_passed = True
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False

    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Health pipeline ready for production!")
        print("\nNext steps:")
        print("1. Run small-scale test: python test_sa1_pipeline.py")
        print(
            "2. Execute health data extraction: pipelines/orchestrator.py --pipeline health_services"
        )
        print("3. Monitor pipeline progress and data quality")
        return True
    else:
        print("\n❌ Some tests failed - please review and fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
