"""
Performance Benchmarking Suite - Comprehensive benchmarking and monitoring for Phase 4.4

Provides complete performance benchmarking for the Australian health data storage optimization
pipeline, integrating all Phase 4 components for comprehensive performance analysis.

Key Features:
- Full storage pipeline benchmarking (Parquet, Memory, Incremental, Lazy loading)
- Regression testing for performance optimization
- Comparative analysis with baseline performance
- Automated performance reporting and recommendations
- Integration testing across all storage components
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import time
import json
import tempfile
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
# Skip visualization imports for now due to numpy compatibility issues
plt = None
sns = None
VISUALIZATION_AVAILABLE = False
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import all our storage optimization components
from .parquet_storage_manager import ParquetStorageManager
from .lazy_data_loader import LazyDataLoader
from .memory_optimizer import MemoryOptimizer
from .incremental_processor import IncrementalProcessor
from .storage_performance_monitor import StoragePerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark test result."""
    test_name: str
    component: str
    data_size_mb: float
    rows_processed: int
    execution_time_seconds: float
    memory_usage_mb: float
    throughput_mb_per_second: float
    optimization_applied: List[str]
    performance_score: float
    baseline_comparison: Optional[float] = None


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    suite_name: str
    timestamp: str
    total_tests: int
    total_execution_time: float
    average_performance_score: float
    benchmark_results: List[BenchmarkResult]
    performance_summary: Dict[str, Any]
    recommendations: List[str]


class PerformanceBenchmarkingSuite:
    """
    Comprehensive performance benchmarking suite for Australian health data storage optimization.
    Tests all Phase 4 components and provides detailed performance analysis.
    """
    
    # Benchmark configuration
    BENCHMARK_CONFIG = {
        "data_sizes": [1000, 10000, 50000, 100000],  # Row counts for testing
        "test_iterations": 3,                         # Number of iterations per test
        "memory_limit_gb": 4.0,                      # Memory limit for testing
        "performance_threshold": 0.8,                # Minimum acceptable performance score
        "regression_tolerance": 0.1,                 # 10% performance degradation tolerance
    }
    
    # Australian health data patterns for realistic testing
    HEALTH_DATA_PATTERNS = {
        "sa2_codes": [f"1{str(i).zfill(8)}" for i in range(10010, 15000)],
        "states": ['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'],
        "age_groups": ['0-17', '18-34', '35-49', '50-64', '65-79', '80+'],
        "postcodes": [str(i) for i in range(2000, 8999)],
    }
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize benchmarking suite with output directory."""
        self.output_dir = output_dir or Path("data/performance_benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all storage components
        self.parquet_manager = ParquetStorageManager()
        self.lazy_loader = LazyDataLoader()
        self.memory_optimizer = MemoryOptimizer(enable_profiling=True)
        self.incremental_processor = IncrementalProcessor()
        self.performance_monitor = StoragePerformanceMonitor()
        
        # Benchmark results storage
        self.benchmark_results: List[BenchmarkResult] = []
        self.baseline_results: Dict[str, float] = {}
        
        # Load baseline results if available
        self._load_baseline_results()
        
        logger.info(f"Initialized performance benchmarking suite at {self.output_dir}")
    
    def _load_baseline_results(self) -> None:
        """Load baseline performance results for comparison."""
        try:
            baseline_file = self.output_dir / "baseline_results.json"
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    self.baseline_results = json.load(f)
                logger.info(f"Loaded {len(self.baseline_results)} baseline results")
        except Exception as e:
            logger.warning(f"Failed to load baseline results: {e}")
    
    def generate_test_data(self, n_rows: int) -> pl.DataFrame:
        """Generate realistic Australian health test data."""
        np.random.seed(42)  # Reproducible benchmarks
        
        return pl.DataFrame({
            # Geographic identifiers
            "sa2_code": np.random.choice(self.HEALTH_DATA_PATTERNS["sa2_codes"], n_rows),
            "sa2_name": [f"Statistical Area {i}" for i in range(n_rows)],
            "state_territory": np.random.choice(self.HEALTH_DATA_PATTERNS["states"], n_rows),
            "postcode": np.random.choice(self.HEALTH_DATA_PATTERNS["postcodes"], n_rows),
            "lga_name": np.random.choice([f"LGA {i}" for i in range(1, 100)], n_rows),
            
            # SEIFA indices (1-10 deciles)
            "seifa_irsd_decile": np.random.randint(1, 11, n_rows),
            "seifa_irsad_decile": np.random.randint(1, 11, n_rows),
            "seifa_ier_decile": np.random.randint(1, 11, n_rows),
            "seifa_ieo_decile": np.random.randint(1, 11, n_rows),
            
            # Health metrics
            "prescription_count": np.random.poisson(5, n_rows),
            "gp_visits": np.random.poisson(8, n_rows),
            "specialist_visits": np.random.poisson(2, n_rows),
            "total_cost_aud": np.random.exponential(250, n_rows),
            "chronic_conditions": np.random.randint(0, 8, n_rows),
            
            # Demographics
            "age_group": np.random.choice(self.HEALTH_DATA_PATTERNS["age_groups"], n_rows),
            "gender": np.random.choice(['M', 'F', 'O'], n_rows),
            "usual_resident_population": np.random.randint(50, 15000, n_rows),
            
            # Risk scores
            "health_risk_score": np.random.uniform(1.0, 10.0, n_rows),
            "access_score": np.random.uniform(1.0, 10.0, n_rows),
            
            # Temporal data
            "service_date": ["2023-01-01"] * n_rows,
            "data_extraction_date": ["2023-12-01"] * n_rows,
        })
    
    def benchmark_parquet_storage(self, data_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark Parquet storage manager performance."""
        console = Console()
        console.print("[blue]🗄️  Benchmarking Parquet Storage Manager...")
        
        results = []
        
        for size in data_sizes:
            test_data = self.generate_test_data(size)
            data_size_mb = test_data.estimated_size("mb")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Test multiple iterations for accuracy
                execution_times = []
                memory_usages = []
                optimizations = []
                
                for iteration in range(self.BENCHMARK_CONFIG["test_iterations"]):
                    file_path = temp_path / f"test_parquet_{size}_{iteration}.parquet"
                    
                    # Start monitoring
                    start_memory = self.memory_optimizer.system_monitor.get_memory_usage_gb() * 1024
                    start_time = time.time()
                    
                    # Execute Parquet optimization
                    metrics = self.parquet_manager.write_parquet_optimized(test_data, file_path)
                    
                    # End monitoring
                    end_time = time.time()
                    end_memory = self.memory_optimizer.system_monitor.get_memory_usage_gb() * 1024
                    
                    execution_times.append(end_time - start_time)
                    memory_usages.append(end_memory - start_memory)
                    if iteration == 0:  # Collect optimizations from first iteration
                        optimizations = metrics.get("optimizations_applied", [])
                
                # Calculate averages
                avg_execution_time = np.mean(execution_times)
                avg_memory_usage = np.mean(memory_usages)
                throughput = data_size_mb / avg_execution_time if avg_execution_time > 0 else 0
                
                # Calculate performance score (higher is better)
                performance_score = min(1.0, (100.0 / avg_execution_time) * (data_size_mb / max(1, avg_memory_usage)))
                
                # Create benchmark result
                result = BenchmarkResult(
                    test_name=f"parquet_storage_{size}_rows",
                    component="ParquetStorageManager",
                    data_size_mb=data_size_mb,
                    rows_processed=size,
                    execution_time_seconds=avg_execution_time,
                    memory_usage_mb=avg_memory_usage,
                    throughput_mb_per_second=throughput,
                    optimization_applied=optimizations,
                    performance_score=performance_score
                )
                
                # Compare with baseline if available
                baseline_key = f"parquet_{size}"
                if baseline_key in self.baseline_results:
                    result.baseline_comparison = performance_score / self.baseline_results[baseline_key]
                
                results.append(result)
                
                console.print(f"  📊 {size:,} rows: {avg_execution_time:.3f}s, {throughput:.1f}MB/s")
        
        return results
    
    def benchmark_memory_optimization(self, data_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark memory optimizer performance."""
        console = Console()
        console.print("[blue]🧠 Benchmarking Memory Optimizer...")
        
        results = []
        
        for size in data_sizes:
            test_data = self.generate_test_data(size)
            original_size_mb = test_data.estimated_size("mb")
            
            # Test multiple iterations
            execution_times = []
            memory_savings = []
            optimizations_counts = []
            
            for iteration in range(self.BENCHMARK_CONFIG["test_iterations"]):
                start_time = time.time()
                
                # Execute memory optimization
                optimized_df, stats = self.memory_optimizer.optimize_dataframe_memory(test_data, "health")
                
                end_time = time.time()
                
                execution_times.append(end_time - start_time)
                memory_savings.append(stats.get("memory_savings_mb", 0))
                optimizations_counts.append(len(stats.get("optimizations_applied", [])))
            
            # Calculate averages
            avg_execution_time = np.mean(execution_times)
            avg_memory_savings = np.mean(memory_savings)
            avg_optimizations = np.mean(optimizations_counts)
            
            # Performance score based on memory savings and speed
            performance_score = (avg_memory_savings / original_size_mb) * (10.0 / max(0.1, avg_execution_time))
            
            result = BenchmarkResult(
                test_name=f"memory_optimization_{size}_rows",
                component="MemoryOptimizer", 
                data_size_mb=original_size_mb,
                rows_processed=size,
                execution_time_seconds=avg_execution_time,
                memory_usage_mb=avg_memory_savings,  # Memory saved
                throughput_mb_per_second=original_size_mb / avg_execution_time,
                optimization_applied=[f"{avg_optimizations:.1f} optimizations applied"],
                performance_score=performance_score
            )
            
            # Compare with baseline
            baseline_key = f"memory_{size}"
            if baseline_key in self.baseline_results:
                result.baseline_comparison = performance_score / self.baseline_results[baseline_key]
            
            results.append(result)
            
            savings_percent = (avg_memory_savings / original_size_mb) * 100
            console.print(f"  💾 {size:,} rows: {avg_execution_time:.3f}s, {savings_percent:.1f}% memory saved")
        
        return results
    
    def benchmark_lazy_loading(self, data_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark lazy data loader performance."""
        console = Console()
        console.print("[blue]⚡ Benchmarking Lazy Data Loader...")
        
        results = []
        
        for size in data_sizes:
            test_data = self.generate_test_data(size)
            data_size_mb = test_data.estimated_size("mb")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                file_path = temp_path / f"test_lazy_{size}.parquet"
                test_data.write_parquet(file_path)
                
                # Test multiple iterations
                execution_times = []
                memory_usages = []
                
                for iteration in range(self.BENCHMARK_CONFIG["test_iterations"]):
                    start_memory = self.memory_optimizer.system_monitor.get_memory_usage_gb() * 1024
                    start_time = time.time()
                    
                    # Execute lazy loading with query
                    lazy_df = self.lazy_loader.load_lazy_dataset(file_path, "parquet")
                    query_df = lazy_df.filter(pl.col("prescription_count") > 3).group_by("state_territory").agg([
                        pl.col("prescription_count").sum().alias("total_prescriptions")
                    ])
                    result_df = self.lazy_loader.execute_lazy_query(query_df, cache_key=f"test_{iteration}")
                    
                    end_time = time.time()
                    end_memory = self.memory_optimizer.system_monitor.get_memory_usage_gb() * 1024
                    
                    execution_times.append(end_time - start_time)
                    memory_usages.append(end_memory - start_memory)
                
                # Calculate averages
                avg_execution_time = np.mean(execution_times)
                avg_memory_usage = np.mean(memory_usages)
                throughput = data_size_mb / avg_execution_time if avg_execution_time > 0 else 0
                
                # Performance score
                performance_score = (data_size_mb / max(1, avg_memory_usage)) * (10.0 / max(0.1, avg_execution_time))
                
                result = BenchmarkResult(
                    test_name=f"lazy_loading_{size}_rows",
                    component="LazyDataLoader",
                    data_size_mb=data_size_mb,
                    rows_processed=size,
                    execution_time_seconds=avg_execution_time,
                    memory_usage_mb=avg_memory_usage,
                    throughput_mb_per_second=throughput,
                    optimization_applied=["lazy_evaluation", "query_caching"],
                    performance_score=performance_score
                )
                
                # Compare with baseline
                baseline_key = f"lazy_{size}"
                if baseline_key in self.baseline_results:
                    result.baseline_comparison = performance_score / self.baseline_results[baseline_key]
                
                results.append(result)
                
                console.print(f"  ⚡ {size:,} rows: {avg_execution_time:.3f}s, {throughput:.1f}MB/s")
        
        return results
    
    def benchmark_incremental_processing(self, data_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark incremental processor performance."""
        console = Console()
        console.print("[blue]🔄 Benchmarking Incremental Processor...")
        
        results = []
        
        for size in data_sizes:
            test_data = self.generate_test_data(size)
            data_size_mb = test_data.estimated_size("mb")
            
            with tempfile.TemporaryDirectory() as temp_dir:
                processor = IncrementalProcessor(Path(temp_dir))
                
                # Test multiple iterations
                execution_times = []
                
                for iteration in range(self.BENCHMARK_CONFIG["test_iterations"]):
                    start_time = time.time()
                    
                    # Execute Bronze-Silver-Gold pipeline
                    bronze_version = processor.ingest_to_bronze(
                        test_data, "health", {"source": f"benchmark_{iteration}"}
                    )
                    silver_version = processor.process_to_silver("health", bronze_version)
                    gold_version = processor.aggregate_to_gold(
                        "health", silver_version, {"group_by_sa2": True}
                    )
                    
                    end_time = time.time()
                    execution_times.append(end_time - start_time)
                
                # Calculate averages
                avg_execution_time = np.mean(execution_times)
                throughput = data_size_mb / avg_execution_time if avg_execution_time > 0 else 0
                
                # Performance score based on data processing throughput
                performance_score = throughput / max(1, avg_execution_time)
                
                result = BenchmarkResult(
                    test_name=f"incremental_processing_{size}_rows",
                    component="IncrementalProcessor",
                    data_size_mb=data_size_mb,
                    rows_processed=size,
                    execution_time_seconds=avg_execution_time,
                    memory_usage_mb=0,  # Not directly measured
                    throughput_mb_per_second=throughput,
                    optimization_applied=["bronze_silver_gold", "data_versioning"],
                    performance_score=performance_score
                )
                
                # Compare with baseline
                baseline_key = f"incremental_{size}"
                if baseline_key in self.baseline_results:
                    result.baseline_comparison = performance_score / self.baseline_results[baseline_key]
                
                results.append(result)
                
                console.print(f"  🔄 {size:,} rows: {avg_execution_time:.3f}s, {throughput:.1f}MB/s")
        
        return results
    
    def run_comprehensive_benchmark(self) -> BenchmarkSuite:
        """Run complete benchmark suite across all storage components."""
        console = Console()
        
        console.print(Panel.fit(
            "[bold blue]🚀 Storage Optimization Benchmark Suite[/bold blue]\n"
            "Testing all Phase 4 components with Australian health data patterns",
            title="Phase 4.4: Performance Benchmarking"
        ))
        
        start_time = time.time()
        data_sizes = self.BENCHMARK_CONFIG["data_sizes"]
        
        # Run all component benchmarks
        all_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Parquet Storage Benchmarking
            task1 = progress.add_task("Benchmarking Parquet storage...", total=None)
            parquet_results = self.benchmark_parquet_storage(data_sizes)
            all_results.extend(parquet_results)
            progress.update(task1, completed=True, description="Parquet storage benchmarking complete")
            
            # Memory Optimization Benchmarking  
            task2 = progress.add_task("Benchmarking memory optimization...", total=None)
            memory_results = self.benchmark_memory_optimization(data_sizes)
            all_results.extend(memory_results)
            progress.update(task2, completed=True, description="Memory optimization benchmarking complete")
            
            # Lazy Loading Benchmarking
            task3 = progress.add_task("Benchmarking lazy loading...", total=None)
            lazy_results = self.benchmark_lazy_loading(data_sizes)
            all_results.extend(lazy_results)
            progress.update(task3, completed=True, description="Lazy loading benchmarking complete")
            
            # Incremental Processing Benchmarking
            task4 = progress.add_task("Benchmarking incremental processing...", total=None)
            incremental_results = self.benchmark_incremental_processing(data_sizes)
            all_results.extend(incremental_results)
            progress.update(task4, completed=True, description="Incremental processing benchmarking complete")
        
        total_time = time.time() - start_time
        
        # Calculate performance summary
        performance_summary = self._calculate_performance_summary(all_results)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(all_results)
        
        # Create benchmark suite
        suite = BenchmarkSuite(
            suite_name="Phase_4_Storage_Optimization_Benchmark",
            timestamp=datetime.now().isoformat(),
            total_tests=len(all_results),
            total_execution_time=total_time,
            average_performance_score=np.mean([r.performance_score for r in all_results]),
            benchmark_results=all_results,
            performance_summary=performance_summary,
            recommendations=recommendations
        )
        
        # Save results
        self._save_benchmark_results(suite)
        
        # Display results
        self._display_benchmark_results(suite)
        
        return suite
    
    def _calculate_performance_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate comprehensive performance summary."""
        summary = {
            "by_component": {},
            "by_data_size": {},
            "overall_metrics": {},
            "optimization_impact": {}
        }
        
        # Group by component
        for component in ["ParquetStorageManager", "MemoryOptimizer", "LazyDataLoader", "IncrementalProcessor"]:
            component_results = [r for r in results if r.component == component]
            if component_results:
                summary["by_component"][component] = {
                    "avg_execution_time": np.mean([r.execution_time_seconds for r in component_results]),
                    "avg_throughput": np.mean([r.throughput_mb_per_second for r in component_results]),
                    "avg_performance_score": np.mean([r.performance_score for r in component_results]),
                    "test_count": len(component_results)
                }
        
        # Group by data size
        for size in self.BENCHMARK_CONFIG["data_sizes"]:
            size_results = [r for r in results if r.rows_processed == size]
            if size_results:
                summary["by_data_size"][str(size)] = {
                    "avg_execution_time": np.mean([r.execution_time_seconds for r in size_results]),
                    "avg_throughput": np.mean([r.throughput_mb_per_second for r in size_results]),
                    "avg_performance_score": np.mean([r.performance_score for r in size_results])
                }
        
        # Overall metrics
        summary["overall_metrics"] = {
            "total_data_processed_mb": sum([r.data_size_mb for r in results]),
            "total_rows_processed": sum([r.rows_processed for r in results]),
            "average_execution_time": np.mean([r.execution_time_seconds for r in results]),
            "average_throughput": np.mean([r.throughput_mb_per_second for r in results]),
            "average_performance_score": np.mean([r.performance_score for r in results])
        }
        
        return summary
    
    def _generate_performance_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze performance across components
        component_scores = {}
        for result in results:
            if result.component not in component_scores:
                component_scores[result.component] = []
            component_scores[result.component].append(result.performance_score)
        
        # Find lowest performing components
        avg_scores = {comp: np.mean(scores) for comp, scores in component_scores.items()}
        min_score_component = min(avg_scores.keys(), key=lambda k: avg_scores[k])
        
        if avg_scores[min_score_component] < self.BENCHMARK_CONFIG["performance_threshold"]:
            recommendations.append(
                f"Consider optimizing {min_score_component} - performance score {avg_scores[min_score_component]:.2f} "
                f"below threshold {self.BENCHMARK_CONFIG['performance_threshold']}"
            )
        
        # Check for regression against baseline
        regression_count = 0
        for result in results:
            if result.baseline_comparison and result.baseline_comparison < (1 - self.BENCHMARK_CONFIG["regression_tolerance"]):
                regression_count += 1
        
        if regression_count > 0:
            recommendations.append(f"Performance regression detected in {regression_count} tests - investigate recent changes")
        
        # Data size scaling recommendations
        large_data_results = [r for r in results if r.rows_processed >= 50000]
        if large_data_results:
            avg_large_throughput = np.mean([r.throughput_mb_per_second for r in large_data_results])
            if avg_large_throughput < 20:  # Threshold for acceptable throughput
                recommendations.append("Consider implementing streaming processing for large datasets (>50K rows)")
        
        # Memory usage recommendations
        high_memory_results = [r for r in results if r.memory_usage_mb > 100]
        if len(high_memory_results) > len(results) * 0.3:  # More than 30% of tests use high memory
            recommendations.append("High memory usage detected - consider batch processing or lazy evaluation optimizations")
        
        if not recommendations:
            recommendations.append("Performance is optimal - all components meeting performance targets")
        
        return recommendations
    
    def _save_benchmark_results(self, suite: BenchmarkSuite) -> None:
        """Save benchmark results to files."""
        try:
            # Save complete results as JSON
            results_file = self.output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(asdict(suite), f, indent=2, default=str)
            
            # Update baseline results
            baseline_updates = {}
            for result in suite.benchmark_results:
                if result.component == "ParquetStorageManager":
                    baseline_updates[f"parquet_{result.rows_processed}"] = result.performance_score
                elif result.component == "MemoryOptimizer":
                    baseline_updates[f"memory_{result.rows_processed}"] = result.performance_score
                elif result.component == "LazyDataLoader":
                    baseline_updates[f"lazy_{result.rows_processed}"] = result.performance_score
                elif result.component == "IncrementalProcessor":
                    baseline_updates[f"incremental_{result.rows_processed}"] = result.performance_score
            
            # Save updated baselines
            self.baseline_results.update(baseline_updates)
            baseline_file = self.output_dir / "baseline_results.json"
            with open(baseline_file, 'w') as f:
                json.dump(self.baseline_results, f, indent=2)
            
            logger.info(f"Benchmark results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def _display_benchmark_results(self, suite: BenchmarkSuite) -> None:
        """Display benchmark results in formatted tables."""
        console = Console()
        
        # Overall summary table
        summary_table = Table(show_header=True, header_style="bold green")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")
        
        summary_table.add_row("Total Tests", str(suite.total_tests))
        summary_table.add_row("Total Execution Time", f"{suite.total_execution_time:.2f}s")
        summary_table.add_row("Average Performance Score", f"{suite.average_performance_score:.3f}")
        summary_table.add_row("Data Processed", f"{suite.performance_summary['overall_metrics']['total_data_processed_mb']:.1f}MB")
        summary_table.add_row("Rows Processed", f"{suite.performance_summary['overall_metrics']['total_rows_processed']:,}")
        
        console.print("\n📊 Benchmark Summary")
        console.print(summary_table)
        
        # Component performance table
        component_table = Table(show_header=True, header_style="bold blue")
        component_table.add_column("Component", style="cyan")
        component_table.add_column("Avg Time (s)", style="yellow")
        component_table.add_column("Avg Throughput (MB/s)", style="green") 
        component_table.add_column("Performance Score", style="magenta")
        component_table.add_column("Tests", style="white")
        
        for component, stats in suite.performance_summary["by_component"].items():
            component_table.add_row(
                component.replace("Manager", "").replace("Processor", ""),
                f"{stats['avg_execution_time']:.3f}",
                f"{stats['avg_throughput']:.1f}",
                f"{stats['avg_performance_score']:.3f}",
                str(stats['test_count'])
            )
        
        console.print("\n🔧 Component Performance")
        console.print(component_table)
        
        # Recommendations
        console.print("\n💡 Performance Recommendations")
        for i, rec in enumerate(suite.recommendations, 1):
            console.print(f"  {i}. {rec}")
        
        # Final status
        overall_score = suite.average_performance_score
        if overall_score >= 0.8:
            status = "[bold green]EXCELLENT[/bold green]"
        elif overall_score >= 0.6:
            status = "[bold yellow]GOOD[/bold yellow]"
        else:
            status = "[bold red]NEEDS IMPROVEMENT[/bold red]"
        
        console.print(Panel.fit(
            f"[bold]Phase 4.4 Benchmark Complete![/bold]\n\n"
            f"Overall Performance: {status}\n"
            f"Average Score: {overall_score:.3f}\n"
            f"Total Tests: {suite.total_tests}\n\n"
            f"All storage optimization components benchmarked successfully.",
            title="Benchmark Complete"
        ))


if __name__ == "__main__":
    # Development testing
    suite = PerformanceBenchmarkingSuite()
    results = suite.run_comprehensive_benchmark()
    
    print(f"✅ Benchmark suite completed: {results.total_tests} tests, "
          f"average score {results.average_performance_score:.3f}")