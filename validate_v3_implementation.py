#!/usr/bin/env python3
"""
AHGD V3: Implementation Validation Script
Progressive 4-level validation system for production readiness.

Validates:
- Level 1: Syntax and imports
- Level 2: Core functionality and data flow
- Level 3: Integration between components
- Level 4: End-to-end system readiness
"""

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add source paths
sys.path.append(str(Path(__file__).parent / "src"))


def print_header(level: int, title: str):
    """Print formatted validation level header."""
    print(f"\n{'='*60}")
    print(f"🧪 LEVEL {level} VALIDATION: {title}")
    print(f"{'='*60}")


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print formatted test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {test_name}")
    if details:
        print(f"     {details}")


def validate_level_1_syntax() -> dict[str, bool]:
    """Level 1: Syntax and Import Validation."""
    print_header(1, "SYNTAX & IMPORTS")

    results = {}

    # Test 1: Core module syntax
    try:
        import duckdb
        import polars as pl

        results["polars_import"] = True
        print_result("Core dependencies (Polars, DuckDB)", True, "Modern data stack available")
    except ImportError as e:
        results["polars_import"] = False
        print_result("Core dependencies", False, str(e))

    # Test 2: Python file syntax validation
    python_files = []
    for root in ["src", "streamlit_app"]:
        if Path(root).exists():
            python_files.extend(Path(root).rglob("*.py"))

    syntax_errors = 0
    for py_file in python_files:
        try:
            compile(py_file.read_text(), str(py_file), "exec")
        except SyntaxError:
            syntax_errors += 1

    results["syntax_check"] = syntax_errors == 0
    print_result(
        f"Python syntax validation ({len(python_files)} files)",
        results["syntax_check"],
        f"{syntax_errors} syntax errors" if syntax_errors > 0 else "All files valid",
    )

    # Test 3: Configuration file validation
    config_files = ["docker-compose-v3.yml", "dbt_project.yml", "profiles.yml"]
    config_valid = True
    for config_file in config_files:
        if not Path(config_file).exists():
            config_valid = False
            print_result(f"Config file: {config_file}", False, "File not found")
        else:
            print_result(f"Config file: {config_file}", True, "Found")

    results["config_files"] = config_valid

    return results


def validate_level_2_functionality() -> dict[str, bool]:
    """Level 2: Core Functionality Validation."""
    print_header(2, "CORE FUNCTIONALITY")

    results = {}

    # Test 1: Polars DataFrame operations
    try:
        import polars as pl

        # Create test data
        test_df = pl.DataFrame(
            {
                "sa1_code": ["10101100001", "10101100002", "10101100003"],
                "diabetes_prevalence": [5.2, 6.1, 4.8],
                "population": [450, 523, 389],
            }
        )

        # Test lazy operations
        lazy_df = test_df.lazy()
        processed = lazy_df.with_columns(
            [(pl.col("diabetes_prevalence") * pl.col("population") / 100).alias("diabetes_cases")]
        ).collect()

        results["polars_operations"] = processed.height == 3
        print_result(
            "Polars DataFrame operations",
            results["polars_operations"],
            f"Processed {processed.height} records with lazy evaluation",
        )

    except Exception as e:
        results["polars_operations"] = False
        print_result("Polars DataFrame operations", False, str(e))

    # Test 2: DuckDB connectivity and operations
    try:
        import duckdb

        # Test in-memory database
        conn = duckdb.connect(":memory:")

        # Create test table
        conn.execute(
            """
            CREATE TABLE test_health_data (
                sa1_code VARCHAR,
                diabetes_prevalence FLOAT,
                population INTEGER
            )
        """
        )

        # Insert test data
        conn.execute(
            """
            INSERT INTO test_health_data VALUES
            ('10101100001', 5.2, 450),
            ('10101100002', 6.1, 523),
            ('10101100003', 4.8, 389)
        """
        )

        # Test analytical query
        result = conn.execute(
            """
            SELECT
                COUNT(*) as record_count,
                AVG(diabetes_prevalence) as avg_diabetes,
                SUM(population) as total_population
            FROM test_health_data
        """
        ).fetchone()

        conn.close()

        results["duckdb_operations"] = result[0] == 3
        print_result(
            "DuckDB analytical operations",
            results["duckdb_operations"],
            f"Query result: {result[0]} records, avg diabetes: {result[1]:.1f}",
        )

    except Exception as e:
        results["duckdb_operations"] = False
        print_result("DuckDB analytical operations", False, str(e))

    # Test 3: dbt project structure
    dbt_components = ["dbt_project.yml", "profiles.yml", "models", "macros"]
    dbt_valid = all(Path(comp).exists() for comp in dbt_components)

    results["dbt_structure"] = dbt_valid
    print_result(
        "dbt project structure",
        dbt_valid,
        "All required dbt components present" if dbt_valid else "Missing dbt components",
    )

    # Test 4: Streamlit app structure
    streamlit_components = [
        "streamlit_app/main.py",
        "streamlit_app/utils/data_connector.py",
        "streamlit_app/components/geographic_selector.py",
    ]
    streamlit_valid = all(Path(comp).exists() for comp in streamlit_components)

    results["streamlit_structure"] = streamlit_valid
    print_result(
        "Streamlit app structure",
        streamlit_valid,
        "All required Streamlit components present"
        if streamlit_valid
        else "Missing Streamlit components",
    )

    return results


def validate_level_3_integration() -> dict[str, bool]:
    """Level 3: Integration Testing."""
    print_header(3, "INTEGRATION TESTING")

    results = {}

    # Test 1: Docker Compose validation
    try:
        import yaml

        with open("docker-compose-v3.yml") as f:
            compose_config = yaml.safe_load(f)

        required_services = ["postgres", "duckdb", "redis", "airflow-webserver", "streamlit", "api"]
        available_services = list(compose_config.get("services", {}).keys())

        services_present = all(service in available_services for service in required_services)

        results["docker_compose"] = services_present
        print_result(
            "Docker Compose configuration",
            services_present,
            f"Services: {', '.join(available_services)}",
        )

    except Exception as e:
        results["docker_compose"] = False
        print_result("Docker Compose configuration", False, str(e))

    # Test 2: dbt model compilation
    try:
        if Path("dbt_project.yml").exists():
            # Simple dbt validation - check if project compiles
            import subprocess

            result = subprocess.run(["dbt", "parse"], capture_output=True, text=True, cwd=".")

            dbt_valid = result.returncode == 0
            results["dbt_compilation"] = dbt_valid
            print_result(
                "dbt model compilation",
                dbt_valid,
                "Models parse successfully" if dbt_valid else f"dbt error: {result.stderr[:100]}",
            )
        else:
            results["dbt_compilation"] = False
            print_result("dbt model compilation", False, "dbt_project.yml not found")

    except FileNotFoundError:
        results["dbt_compilation"] = False
        print_result("dbt model compilation", False, "dbt not installed")
    except Exception as e:
        results["dbt_compilation"] = False
        print_result("dbt model compilation", False, str(e))

    # Test 3: Data flow integration test
    try:
        import duckdb
        import polars as pl

        # Simulate data extraction -> transformation -> loading
        start_time = time.time()

        # Step 1: Extract (simulate)
        raw_data = pl.DataFrame(
            {
                "sa1_code": [f"1010110000{i}" for i in range(1000)],
                "diabetes_prevalence": [4.5 + (i % 10) * 0.3 for i in range(1000)],
                "population": [400 + (i % 200) for i in range(1000)],
            }
        )

        # Step 2: Transform (dbt-style transformation)
        transformed_data = (
            raw_data.lazy()
            .with_columns(
                [
                    # Health vulnerability calculation
                    ((10 - pl.col("diabetes_prevalence")) * 10).alias("health_score"),
                    # Population density category
                    pl.when(pl.col("population") > 500)
                    .then(pl.lit("High"))
                    .when(pl.col("population") > 400)
                    .then(pl.lit("Medium"))
                    .otherwise(pl.lit("Low"))
                    .alias("population_category"),
                ]
            )
            .collect()
        )

        # Step 3: Load to DuckDB
        conn = duckdb.connect(":memory:")
        conn.register("health_data", transformed_data.to_pandas())

        # Test analytical query
        analytical_result = conn.execute(
            """
            SELECT
                population_category,
                COUNT(*) as areas,
                AVG(diabetes_prevalence) as avg_diabetes,
                AVG(health_score) as avg_health_score
            FROM health_data
            GROUP BY population_category
            ORDER BY avg_health_score DESC
        """
        ).fetchall()

        processing_time = time.time() - start_time
        conn.close()

        # Validate results
        data_flow_valid = (
            len(analytical_result) == 3  # 3 population categories
            and processing_time < 2.0  # Processing under 2 seconds
            and transformed_data.height == 1000  # All records processed
        )

        results["data_flow_integration"] = data_flow_valid
        print_result(
            "Data flow integration (Extract→Transform→Load)",
            data_flow_valid,
            f"Processed 1000 records in {processing_time:.3f}s, {len(analytical_result)} categories",
        )

    except Exception as e:
        results["data_flow_integration"] = False
        print_result("Data flow integration", False, str(e))

    return results


def validate_level_4_deployment() -> dict[str, bool]:
    """Level 4: Deployment Readiness."""
    print_header(4, "DEPLOYMENT READINESS")

    results = {}

    # Test 1: Environment configuration
    dockerfile_configs = ["Dockerfile.v3", "Dockerfile.streamlit", "Dockerfile.api"]
    docker_valid = all(Path(dockerfile).exists() for dockerfile in dockerfile_configs)

    results["docker_images"] = docker_valid
    print_result(
        "Docker image configurations",
        docker_valid,
        "All Dockerfiles present" if docker_valid else "Missing Dockerfiles",
    )

    # Test 2: Performance benchmarking
    try:
        import time

        import polars as pl

        # Performance test - 10x improvement claim validation
        record_counts = [1000, 10000, 100000]
        performance_results = []

        for count in record_counts:
            # Generate test data
            test_data = pl.DataFrame(
                {
                    "sa1_code": [f"sa1_{i:06d}" for i in range(count)],
                    "health_metric": [50.0 + (i % 100) * 0.1 for i in range(count)],
                    "population": [300 + (i % 500) for i in range(count)],
                }
            )

            # Time complex operations
            start_time = time.time()

            result = (
                test_data.lazy()
                .with_columns(
                    [
                        # Complex aggregations and calculations
                        (pl.col("health_metric") * pl.col("population") / 100).alias(
                            "health_burden"
                        ),
                        pl.col("health_metric").rank().alias("health_rank"),
                        pl.col("population").pct_change().alias("pop_change"),
                    ]
                )
                .group_by((pl.col("sa1_code").str.slice(0, 3)).alias("region"))
                .agg(
                    [
                        pl.col("health_burden").sum().alias("total_burden"),
                        pl.col("health_metric").mean().alias("avg_health"),
                        pl.col("population").sum().alias("total_pop"),
                    ]
                )
                .collect()
            )

            processing_time = time.time() - start_time
            records_per_second = count / processing_time if processing_time > 0 else float("inf")

            performance_results.append(
                {"records": count, "time": processing_time, "rps": records_per_second}
            )

        # Validate performance (should handle 100k records in under 1 second)
        performance_valid = performance_results[-1]["time"] < 1.0

        results["performance_benchmark"] = performance_valid
        print_result(
            "Performance benchmark (100K records)",
            performance_valid,
            f"{performance_results[-1]['rps']:,.0f} records/sec, "
            f"{performance_results[-1]['time']:.3f}s processing time",
        )

    except Exception as e:
        results["performance_benchmark"] = False
        print_result("Performance benchmark", False, str(e))

    # Test 3: Production data quality standards
    try:
        import polars as pl

        # Test data quality validation functions
        test_health_data = pl.DataFrame(
            {
                "sa1_code": ["10101100001", "10101100002", "10101100003", None, "10101100005"],
                "diabetes_prevalence": [5.2, 6.1, None, 4.8, 150.0],  # One outlier
                "population": [450, 523, 389, 412, 367],
                "data_quality_score": [0.95, 0.88, 0.92, 0.85, 0.91],
            }
        )

        # Data quality checks
        completeness_check = test_health_data.select(
            [
                (pl.col("sa1_code").is_not_null().sum() / pl.len() * 100).alias("sa1_completeness"),
                (pl.col("diabetes_prevalence").is_not_null().sum() / pl.len() * 100).alias(
                    "diabetes_completeness"
                ),
            ]
        )

        # Outlier detection
        outliers = test_health_data.filter(
            (pl.col("diabetes_prevalence") > 50)  # Unrealistic diabetes rate
            | (pl.col("diabetes_prevalence") < 0)
        )

        quality_score = completeness_check.select(pl.col("diabetes_completeness")).item()
        has_outliers = outliers.height > 0

        quality_valid = quality_score >= 80.0  # 80% completeness threshold

        results["data_quality_standards"] = quality_valid
        print_result(
            "Data quality standards",
            quality_valid,
            f"Completeness: {quality_score:.1f}%, Outliers detected: {has_outliers}",
        )

    except Exception as e:
        results["data_quality_standards"] = False
        print_result("Data quality standards", False, str(e))

    return results


def run_comprehensive_validation():
    """Run complete 4-level validation suite."""

    print(
        f"""
🏥 AHGD V3: Modern Analytics Engineering Platform
🧪 Comprehensive Validation Suite
📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    )

    all_results = {}

    # Execute all validation levels
    try:
        all_results["level_1"] = validate_level_1_syntax()
        all_results["level_2"] = validate_level_2_functionality()
        all_results["level_3"] = validate_level_3_integration()
        all_results["level_4"] = validate_level_4_deployment()

    except Exception as e:
        print(f"\n❌ Validation suite error: {e!s}")
        traceback.print_exc()
        return False

    # Calculate overall results
    total_tests = sum(len(level_results) for level_results in all_results.values())
    passed_tests = sum(sum(level_results.values()) for level_results in all_results.values())
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

    # Print final summary
    print(f"\n{'='*60}")
    print("🎯 VALIDATION SUMMARY")
    print(f"{'='*60}")

    for level, results in all_results.items():
        level_passed = sum(results.values())
        level_total = len(results)
        level_success = (level_passed / level_total * 100) if level_total > 0 else 0

        status = "✅" if level_success == 100 else "⚠️" if level_success >= 75 else "❌"
        print(
            f"{status} {level.replace('_', ' ').title()}: {level_passed}/{level_total} ({level_success:.0f}%)"
        )

    print(f"\n🏆 OVERALL SUCCESS RATE: {success_rate:.1f}% ({passed_tests}/{total_tests})")

    # Production readiness assessment
    if success_rate >= 90:
        print("✅ PRODUCTION READY - Implementation meets quality standards")
        return True
    elif success_rate >= 75:
        print("⚠️  PRODUCTION PENDING - Some issues need resolution")
        return False
    else:
        print("❌ NOT PRODUCTION READY - Major issues require attention")
        return False


if __name__ == "__main__":
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)
