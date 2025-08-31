#!/usr/bin/env python3
"""
AHGD V3: Comprehensive Performance Benchmarking Suite
Benchmarks modern Polars stack vs legacy pandas implementation.

Provides detailed performance metrics for:
- Data processing throughput
- Memory efficiency
- Query response times
- Concurrent user capacity
- Storage optimization
"""

import gc

# Add project root to path
import sys
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Optional

import pandas as pd
import polars as pl
import psutil

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.storage.parquet_manager import ParquetStorageManager
from src.utils.logging import get_logger

logger = get_logger("performance_benchmark")


@dataclass
class BenchmarkResult:
    """Individual benchmark result with comprehensive metrics."""

    name: str
    operation: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    records_processed: int = 0
    memory_peak_mb: float = 0.0
    cpu_percent: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def records_per_second(self) -> float:
        """Calculate processing throughput."""
        if self.duration_seconds > 0:
            return self.records_processed / self.duration_seconds
        return 0.0

    @property
    def memory_per_record_kb(self) -> float:
        """Calculate memory efficiency."""
        if self.records_processed > 0:
            return (self.memory_peak_mb * 1024) / self.records_processed
        return 0.0


@dataclass
class ComparisonResult:
    """Comparison between Polars and pandas performance."""

    operation_name: str
    polars_result: BenchmarkResult
    pandas_result: BenchmarkResult

    @property
    def speed_improvement(self) -> float:
        """Calculate speed improvement factor (Polars vs pandas)."""
        if self.pandas_result.duration_seconds > 0:
            return self.pandas_result.duration_seconds / self.polars_result.duration_seconds
        return 0.0

    @property
    def memory_improvement(self) -> float:
        """Calculate memory improvement factor."""
        if self.polars_result.memory_peak_mb > 0:
            return self.pandas_result.memory_peak_mb / self.polars_result.memory_peak_mb
        return 0.0

    @property
    def throughput_improvement(self) -> float:
        """Calculate throughput improvement factor."""
        if self.pandas_result.records_per_second > 0:
            return self.polars_result.records_per_second / self.pandas_result.records_per_second
        return 0.0


class PerformanceBenchmarkSuite:
    """
    Comprehensive performance benchmarking suite for AHGD V3.

    Benchmarks:
    - Data loading and parsing
    - Filtering and aggregation operations
    - Memory usage and efficiency
    - Concurrent processing capacity
    - Storage format optimization
    """

    def __init__(self, data_size: str = "medium"):
        """
        Initialize benchmark suite.

        Args:
            data_size: Benchmark data size ("small", "medium", "large", "xl")
        """
        self.data_size = data_size
        self.results: list[BenchmarkResult] = []
        self.comparisons: list[ComparisonResult] = []

        # Configure data sizes
        self.size_configs = {
            "small": {"rows": 10000, "concurrent_users": 5},
            "medium": {"rows": 100000, "concurrent_users": 10},
            "large": {"rows": 1000000, "concurrent_users": 25},
            "xl": {"rows": 5000000, "concurrent_users": 50},
        }

        self.config = self.size_configs[data_size]

        # Initialize monitoring
        self.process = psutil.Process()
        self.parquet_manager = ParquetStorageManager("./data/benchmark_cache")

        logger.info(f"Initialized benchmark suite with {data_size} dataset")

    def run_comprehensive_benchmark(self) -> dict[str, Any]:
        """
        Run complete performance benchmark suite.

        Returns:
            Comprehensive benchmark results and analysis
        """
        logger.info("🚀 Starting comprehensive AHGD V3 performance benchmark")

        benchmark_start = time.time()

        # 1. Data Processing Benchmarks
        logger.info("📊 Running data processing benchmarks...")
        self._benchmark_data_processing()

        # 2. Query Performance Benchmarks
        logger.info("🔍 Running query performance benchmarks...")
        self._benchmark_query_performance()

        # 3. Memory Efficiency Benchmarks
        logger.info("💾 Running memory efficiency benchmarks...")
        self._benchmark_memory_efficiency()

        # 4. Concurrent Processing Benchmarks
        logger.info("⚡ Running concurrent processing benchmarks...")
        self._benchmark_concurrent_processing()

        # 5. Storage Format Benchmarks
        logger.info("📦 Running storage format benchmarks...")
        self._benchmark_storage_formats()

        total_time = time.time() - benchmark_start

        # Generate comprehensive report
        report = self._generate_benchmark_report(total_time)

        logger.info(f"✅ Benchmark suite completed in {total_time:.1f}s")
        return report

    def _benchmark_data_processing(self):
        """Benchmark core data processing operations."""

        # Generate test data
        test_data = self._generate_test_health_data(self.config["rows"])

        # Benchmark 1: Data Loading (Polars vs Pandas)
        polars_loading = self._benchmark_operation(
            "polars_data_loading",
            lambda: self._polars_load_data(test_data),
            "Data loading with Polars",
        )

        pandas_loading = self._benchmark_operation(
            "pandas_data_loading",
            lambda: self._pandas_load_data(test_data),
            "Data loading with Pandas",
        )

        self.comparisons.append(ComparisonResult("data_loading", polars_loading, pandas_loading))

        # Benchmark 2: Filtering Operations
        df_polars = pl.DataFrame(test_data)
        df_pandas = pd.DataFrame(test_data)

        polars_filtering = self._benchmark_operation(
            "polars_filtering",
            lambda: self._polars_filter_operations(df_polars),
            "Complex filtering with Polars",
        )

        pandas_filtering = self._benchmark_operation(
            "pandas_filtering",
            lambda: self._pandas_filter_operations(df_pandas),
            "Complex filtering with Pandas",
        )

        self.comparisons.append(
            ComparisonResult("filtering_operations", polars_filtering, pandas_filtering)
        )

        # Benchmark 3: Aggregation Operations
        polars_aggregation = self._benchmark_operation(
            "polars_aggregation",
            lambda: self._polars_aggregation_operations(df_polars),
            "Complex aggregations with Polars",
        )

        pandas_aggregation = self._benchmark_operation(
            "pandas_aggregation",
            lambda: self._pandas_aggregation_operations(df_pandas),
            "Complex aggregations with Pandas",
        )

        self.comparisons.append(
            ComparisonResult("aggregation_operations", polars_aggregation, pandas_aggregation)
        )

    def _benchmark_query_performance(self):
        """Benchmark query response performance."""

        # Create test dataset in multiple formats
        test_data = self._generate_test_health_data(self.config["rows"])
        df_polars = pl.DataFrame(test_data)

        # Store in Parquet for realistic testing
        parquet_path = self.parquet_manager.store_processed_data(
            df_polars, "benchmark_health_data", geographic_level="sa1"
        )

        # Benchmark typical API queries
        query_benchmarks = [
            ("sa1_lookup", lambda: self._query_sa1_profile(df_polars)),
            ("health_search", lambda: self._query_health_search(df_polars)),
            ("geographic_filter", lambda: self._query_geographic_filter(df_polars)),
            ("aggregation_query", lambda: self._query_health_aggregation(df_polars)),
        ]

        for query_name, query_func in query_benchmarks:
            result = self._benchmark_operation(
                f"query_{query_name}", query_func, f"Query performance: {query_name}"
            )

            # Add query-specific metadata
            result.metadata.update(
                {
                    "query_type": query_name,
                    "data_size": self.config["rows"],
                    "response_time_target_ms": 500,  # Target <500ms
                }
            )

    def _benchmark_memory_efficiency(self):
        """Benchmark memory usage and efficiency."""

        # Test memory usage scaling
        memory_test_sizes = [1000, 10000, 100000, 500000]

        for size in memory_test_sizes:
            if size > self.config["rows"]:
                continue

            test_data = self._generate_test_health_data(size)

            # Polars memory benchmark
            polars_memory = self._benchmark_operation(
                f"polars_memory_{size}",
                lambda data=test_data: self._polars_memory_test(data),
                f"Memory efficiency test: {size:,} records",
            )
            polars_memory.metadata["test_size"] = size

            # Pandas memory benchmark
            pandas_memory = self._benchmark_operation(
                f"pandas_memory_{size}",
                lambda data=test_data: self._pandas_memory_test(data),
                f"Pandas memory test: {size:,} records",
            )
            pandas_memory.metadata["test_size"] = size

            self.comparisons.append(
                ComparisonResult(f"memory_efficiency_{size}", polars_memory, pandas_memory)
            )

    def _benchmark_concurrent_processing(self):
        """Benchmark concurrent processing capacity."""

        test_data = self._generate_test_health_data(self.config["rows"])
        concurrent_users = self.config["concurrent_users"]

        # Simulate concurrent API requests
        concurrent_polars = self._benchmark_operation(
            "concurrent_polars",
            lambda: self._simulate_concurrent_requests_polars(test_data, concurrent_users),
            f"Concurrent processing: {concurrent_users} users",
        )
        concurrent_polars.metadata["concurrent_users"] = concurrent_users

        concurrent_pandas = self._benchmark_operation(
            "concurrent_pandas",
            lambda: self._simulate_concurrent_requests_pandas(test_data, concurrent_users),
            f"Concurrent pandas processing: {concurrent_users} users",
        )
        concurrent_pandas.metadata["concurrent_users"] = concurrent_users

        self.comparisons.append(
            ComparisonResult("concurrent_processing", concurrent_polars, concurrent_pandas)
        )

    def _benchmark_storage_formats(self):
        """Benchmark storage format performance."""

        test_data = self._generate_test_health_data(self.config["rows"])
        df = pl.DataFrame(test_data)

        storage_formats = [
            ("parquet", lambda: self._test_parquet_storage(df)),
            ("csv", lambda: self._test_csv_storage(df)),
            ("json", lambda: self._test_json_storage(df)),
        ]

        for format_name, storage_func in storage_formats:
            result = self._benchmark_operation(
                f"storage_{format_name}", storage_func, f"Storage benchmark: {format_name.upper()}"
            )
            result.metadata["storage_format"] = format_name

    def _benchmark_operation(self, name: str, operation_func, description: str) -> BenchmarkResult:
        """
        Benchmark a single operation with comprehensive metrics.

        Args:
            name: Operation identifier
            operation_func: Function to benchmark
            description: Human-readable description

        Returns:
            Detailed benchmark result
        """
        logger.debug(f"Benchmarking: {description}")

        # Reset memory tracking
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        result = BenchmarkResult(name=name, operation=description, start_time=datetime.now())

        try:
            # Start CPU monitoring
            cpu_percent_start = self.process.cpu_percent()

            # Execute operation
            start_time = time.time()
            operation_result = operation_func()
            end_time = time.time()

            # Calculate metrics
            result.end_time = datetime.now()
            result.duration_seconds = end_time - start_time
            result.success = True

            # Memory measurement
            peak_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            result.memory_peak_mb = peak_memory - initial_memory

            # CPU measurement
            result.cpu_percent = self.process.cpu_percent() - cpu_percent_start

            # Extract record count if available
            if hasattr(operation_result, "height"):  # Polars DataFrame
                result.records_processed = operation_result.height
            elif hasattr(operation_result, "__len__"):  # List or pandas
                result.records_processed = len(operation_result)
            elif isinstance(operation_result, tuple) and len(operation_result) > 1:
                result.records_processed = operation_result[1]  # (result, count)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            logger.error(f"Benchmark failed for {name}: {e!s}")

        self.results.append(result)
        return result

    # Data Generation and Test Operations
    def _generate_test_health_data(self, n_rows: int) -> dict[str, list]:
        """Generate realistic health data for benchmarking."""
        import random

        # Seed for reproducible benchmarks
        random.seed(42)

        states = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]

        data = {
            "sa1_code": [
                f"{random.randint(101, 801)}{random.randint(10000, 99999):05d}"
                for _ in range(n_rows)
            ],
            "area_name": [f"Test Area {i}" for i in range(n_rows)],
            "state": [random.choice(states) for _ in range(n_rows)],
            "population": [random.randint(200, 2000) for _ in range(n_rows)],
            "diabetes_prevalence": [round(random.uniform(2.0, 15.0), 1) for _ in range(n_rows)],
            "life_expectancy": [round(random.uniform(75.0, 90.0), 1) for _ in range(n_rows)],
            "seifa_irsad": [random.randint(500, 1200) for _ in range(n_rows)],
            "mental_health_services": [
                round(random.uniform(10.0, 100.0), 1) for _ in range(n_rows)
            ],
            "healthcare_access": [round(random.uniform(1.0, 10.0), 1) for _ in range(n_rows)],
        }

        return data

    # Polars Operations
    def _polars_load_data(self, data: dict) -> pl.DataFrame:
        """Load data using Polars."""
        return pl.DataFrame(data)

    def _polars_filter_operations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Complex filtering operations with Polars."""
        return df.filter(
            (pl.col("diabetes_prevalence") > 5.0)
            & (pl.col("life_expectancy") < 85.0)
            & (pl.col("state").is_in(["NSW", "VIC"]))
        ).with_columns(
            [
                (pl.col("diabetes_prevalence") * 2).alias("risk_factor"),
                pl.col("population").rank().alias("population_rank"),
            ]
        )

    def _polars_aggregation_operations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Complex aggregation operations with Polars."""
        return (
            df.group_by(["state"])
            .agg(
                [
                    pl.col("diabetes_prevalence").mean().alias("avg_diabetes"),
                    pl.col("life_expectancy").max().alias("max_life_expectancy"),
                    pl.col("population").sum().alias("total_population"),
                    pl.col("seifa_irsad").std().alias("seifa_std"),
                ]
            )
            .sort("avg_diabetes", descending=True)
        )

    # Pandas Operations (for comparison)
    def _pandas_load_data(self, data: dict) -> pd.DataFrame:
        """Load data using Pandas."""
        return pd.DataFrame(data)

    def _pandas_filter_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complex filtering operations with Pandas."""
        filtered = df[
            (df["diabetes_prevalence"] > 5.0)
            & (df["life_expectancy"] < 85.0)
            & (df["state"].isin(["NSW", "VIC"]))
        ].copy()

        filtered["risk_factor"] = filtered["diabetes_prevalence"] * 2
        filtered["population_rank"] = filtered["population"].rank()

        return filtered

    def _pandas_aggregation_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complex aggregation operations with Pandas."""
        return (
            df.groupby("state")
            .agg(
                {
                    "diabetes_prevalence": "mean",
                    "life_expectancy": "max",
                    "population": "sum",
                    "seifa_irsad": "std",
                }
            )
            .rename(
                columns={
                    "diabetes_prevalence": "avg_diabetes",
                    "life_expectancy": "max_life_expectancy",
                    "population": "total_population",
                    "seifa_irsad": "seifa_std",
                }
            )
            .sort_values("avg_diabetes", ascending=False)
            .reset_index()
        )

    def _generate_benchmark_report(self, total_time: float) -> dict[str, Any]:
        """Generate comprehensive benchmark report."""

        # Calculate summary statistics
        successful_results = [r for r in self.results if r.success]

        report = {
            "benchmark_summary": {
                "total_time_seconds": total_time,
                "data_size": self.data_size,
                "test_records": self.config["rows"],
                "concurrent_users_tested": self.config["concurrent_users"],
                "total_operations": len(self.results),
                "successful_operations": len(successful_results),
                "failed_operations": len(self.results) - len(successful_results),
            },
            "performance_improvements": {},
            "detailed_results": {},
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": sys.version,
                "polars_version": pl.__version__,
                "pandas_version": pd.__version__,
            },
            "recommendations": [],
        }

        # Analyze comparisons
        for comparison in self.comparisons:
            improvement_data = {
                "speed_improvement": f"{comparison.speed_improvement:.1f}x faster",
                "memory_improvement": f"{comparison.memory_improvement:.1f}x more efficient",
                "throughput_improvement": f"{comparison.throughput_improvement:.1f}x higher throughput",
            }

            report["performance_improvements"][comparison.operation_name] = improvement_data

            # Add recommendations based on results
            if comparison.speed_improvement > 10:
                report["recommendations"].append(
                    f"🚀 {comparison.operation_name}: Polars provides {comparison.speed_improvement:.1f}x speed improvement - highly recommended for production"
                )
            elif comparison.memory_improvement > 2:
                report["recommendations"].append(
                    f"💾 {comparison.operation_name}: Polars uses {comparison.memory_improvement:.1f}x less memory - beneficial for large datasets"
                )

        # Add detailed results
        for result in successful_results:
            report["detailed_results"][result.name] = {
                "duration_seconds": result.duration_seconds,
                "records_processed": result.records_processed,
                "records_per_second": result.records_per_second,
                "memory_peak_mb": result.memory_peak_mb,
                "memory_per_record_kb": result.memory_per_record_kb,
                "cpu_percent": result.cpu_percent,
            }

        return report


def main():
    """Run the comprehensive benchmark suite."""
    import argparse

    parser = argparse.ArgumentParser(description="AHGD V3 Performance Benchmark Suite")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large", "xl"],
        default="medium",
        help="Benchmark data size",
    )
    parser.add_argument("--output", type=str, help="Output file for results")

    args = parser.parse_args()

    # Run benchmark
    benchmark = PerformanceBenchmarkSuite(data_size=args.size)
    results = benchmark.run_comprehensive_benchmark()

    # Print summary
    print("\n" + "=" * 80)
    print("🚀 AHGD V3 Performance Benchmark Results")
    print("=" * 80)

    print("\n📊 Test Configuration:")
    print(f"   Data size: {results['benchmark_summary']['data_size']}")
    print(f"   Records tested: {results['benchmark_summary']['test_records']:,}")
    print(f"   Concurrent users: {results['benchmark_summary']['concurrent_users_tested']}")
    print(f"   Total time: {results['benchmark_summary']['total_time_seconds']:.1f}s")

    print("\n🔥 Performance Improvements (Polars vs Pandas):")
    for operation, improvements in results["performance_improvements"].items():
        print(f"   {operation}:")
        print(f"     • Speed: {improvements['speed_improvement']}")
        print(f"     • Memory: {improvements['memory_improvement']}")
        print(f"     • Throughput: {improvements['throughput_improvement']}")

    print("\n💡 Recommendations:")
    for rec in results["recommendations"]:
        print(f"   {rec}")

    print("\n" + "=" * 80)

    # Save results if output specified
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📁 Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
