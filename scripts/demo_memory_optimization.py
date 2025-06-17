#!/usr/bin/env python3
"""
Memory Optimization Demo - Phase 4.3 Demonstration

Demonstrates advanced memory optimization strategies for Australian health data
processing, building on the 497,181+ records from Phase 2.

This script showcases:
- DataFrame memory optimization with Australian health data patterns
- Streaming processing for large datasets
- Memory pressure detection and response
- Performance recommendations
"""

import polars as pl
import numpy as np
from pathlib import Path
import time
import logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.storage.memory_optimizer import MemoryOptimizer


def create_realistic_australian_health_dataset(n_records: int = 100000) -> pl.DataFrame:
    """Create realistic Australian health dataset for testing."""
    console = Console()
    console.print("[blue]🏥 Creating realistic Australian health dataset...")
    
    np.random.seed(42)  # Reproducible results
    
    # Australian-specific data patterns
    sa2_codes = [f"1{str(i).zfill(8)}" for i in range(10010, 13000)]  # NSW SA2 codes
    vic_sa2_codes = [f"2{str(i).zfill(8)}" for i in range(10010, 12000)]  # VIC SA2 codes  
    qld_sa2_codes = [f"3{str(i).zfill(8)}" for i in range(10010, 12000)]  # QLD SA2 codes
    all_sa2_codes = sa2_codes + vic_sa2_codes + qld_sa2_codes
    
    australian_postcodes = list(range(2000, 2999)) + list(range(3000, 3999)) + list(range(4000, 4999))
    
    dataset = pl.DataFrame({
        # Geographic identifiers (high cardinality, good for categorical)
        "sa2_code": np.random.choice(all_sa2_codes, n_records),
        "sa2_name": [f"Statistical Area {i}" for i in range(n_records)],
        "state_territory": np.random.choice(['NSW', 'VIC', 'QLD', 'SA', 'WA', 'TAS', 'NT', 'ACT'], n_records),
        "postcode": np.random.choice([str(p) for p in australian_postcodes], n_records),
        "lga_name": np.random.choice([f"Local Government Area {i}" for i in range(1, 100)], n_records),
        
        # SEIFA indices (1-10 deciles - perfect for int8)
        "seifa_irsd_decile": np.random.randint(1, 11, n_records),
        "seifa_irsad_decile": np.random.randint(1, 11, n_records),
        "seifa_ier_decile": np.random.randint(1, 11, n_records),
        "seifa_ieo_decile": np.random.randint(1, 11, n_records),
        
        # Health utilization metrics (typical int64 by default)
        "prescription_count": np.random.poisson(5, n_records),
        "gp_visits": np.random.poisson(8, n_records),
        "specialist_visits": np.random.poisson(2, n_records),
        "emergency_visits": np.random.poisson(1, n_records),
        "chronic_conditions": np.random.randint(0, 8, n_records),
        
        # Financial data (typical float64 by default)
        "total_cost_aud": np.random.exponential(250, n_records),
        "pbs_benefit_aud": np.random.exponential(45, n_records),
        "patient_contribution_aud": np.random.exponential(15, n_records),
        "bulk_billing_rate": np.random.uniform(0.0, 1.0, n_records),
        
        # Demographics (good for categorical)
        "age_group": np.random.choice(['0-17', '18-34', '35-49', '50-64', '65-79', '80+'], n_records),
        "gender": np.random.choice(['M', 'F', 'O'], n_records),
        "indigenous_status": np.random.choice(['Aboriginal', 'Torres Strait Islander', 'Both', 'Neither'], n_records),
        "country_of_birth": np.random.choice(['Australia', 'UK', 'China', 'India', 'Other'], n_records),
        
        # Population data (int64 by default, but could be int32)
        "usual_resident_population": np.random.randint(50, 15000, n_records),
        "dwellings": np.random.randint(20, 8000, n_records),
        
        # Risk and access scores (float64 by default, could be float32)
        "health_risk_score": np.random.uniform(1.0, 10.0, n_records),
        "access_score": np.random.uniform(1.0, 10.0, n_records),
        "remoteness_score": np.random.uniform(0.0, 5.0, n_records),
        
        # Temporal data
        "service_date": ["2023-01-01"] * n_records,
        "data_extraction_date": ["2023-12-01"] * n_records,
        "financial_year": ["2022-23"] * n_records,
    })
    
    console.print(f"[green]✅ Created dataset with {n_records:,} records and {len(dataset.columns)} columns")
    return dataset


def demonstrate_memory_optimization():
    """Demonstrate memory optimization capabilities."""
    console = Console()
    
    # Header
    console.print(Panel.fit(
        "[bold blue]🧠 Memory Optimization Demo - Phase 4.3[/bold blue]\n"
        "Advanced memory optimization for Australian health data analytics",
        title="Phase 4.3: Memory Optimization"
    ))
    
    # Step 1: Create test dataset
    console.print("\n[bold]Step 1: Create Large Australian Health Dataset[/bold]")
    dataset = create_realistic_australian_health_dataset(75000)  # 75K records for demo
    
    original_size_mb = dataset.estimated_size("mb")
    console.print(f"📊 Original dataset: {original_size_mb:.2f}MB, {dataset.shape[0]:,} rows, {dataset.shape[1]} columns")
    
    # Show data types before optimization
    console.print("\n[dim]Data types before optimization:[/dim]")
    dtype_table = Table(show_header=True, header_style="bold magenta")
    dtype_table.add_column("Column", style="cyan")
    dtype_table.add_column("Data Type", style="yellow")
    dtype_table.add_column("Memory Impact", style="green")
    
    for col in dataset.columns[:10]:  # Show first 10 columns
        dtype = str(dataset[col].dtype)
        memory_impact = "High" if dtype in ["Int64", "Float64", "Utf8"] else "Medium"
        dtype_table.add_row(col, dtype, memory_impact)
    
    console.print(dtype_table)
    
    # Step 2: Initialize Memory Optimizer
    console.print("\n[bold]Step 2: Initialize Memory Optimizer[/bold]")
    optimizer = MemoryOptimizer(memory_limit_gb=4.0, enable_profiling=True)
    
    system_info = optimizer.system_monitor.get_memory_info()
    console.print(f"💾 System Memory: {system_info.get('total_gb', 0):.1f}GB total, "
                 f"{system_info.get('available_gb', 0):.1f}GB available")
    console.print(f"⚙️  Memory Limit: {optimizer.memory_limit_gb:.1f}GB")
    
    # Step 3: Optimize DataFrame Memory
    console.print("\n[bold]Step 3: Optimize DataFrame Memory[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Optimizing memory usage...", total=None)
        
        start_time = time.time()
        optimized_df, optimization_stats = optimizer.optimize_dataframe_memory(dataset, "health")
        optimization_time = time.time() - start_time
        
        progress.update(task, completed=True, description="Memory optimization completed!")
    
    # Display optimization results
    optimized_size_mb = optimization_stats["optimized_size_mb"]
    memory_savings_mb = optimization_stats["memory_savings_mb"]
    memory_savings_percent = optimization_stats["memory_savings_percent"]
    
    results_table = Table(show_header=True, header_style="bold green")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Before", style="yellow")
    results_table.add_column("After", style="green")
    results_table.add_column("Improvement", style="bold green")
    
    results_table.add_row(
        "Memory Usage",
        f"{original_size_mb:.2f} MB",
        f"{optimized_size_mb:.2f} MB",
        f"-{memory_savings_mb:.2f} MB ({memory_savings_percent:.1f}%)"
    )
    results_table.add_row(
        "Optimization Time",
        "N/A",
        f"{optimization_time:.2f} seconds",
        f"{optimization_time:.2f}s"
    )
    results_table.add_row(
        "Optimizations Applied",
        "0",
        str(len(optimization_stats["optimizations_applied"])),
        f"+{len(optimization_stats['optimizations_applied'])} optimizations"
    )
    
    console.print(results_table)
    
    # Show specific optimizations applied
    console.print("\n[dim]Optimizations applied:[/dim]")
    for optimization in optimization_stats["optimizations_applied"][:8]:  # Show first 8
        console.print(f"  • {optimization}")
    
    if len(optimization_stats["optimizations_applied"]) > 8:
        console.print(f"  • ... and {len(optimization_stats['optimizations_applied']) - 8} more")
    
    # Step 4: Demonstrate Streaming Processing
    console.print("\n[bold]Step 4: Demonstrate Streaming Processing[/bold]")
    
    # Save dataset to temporary file for streaming demo
    temp_file = Path("temp_health_data.parquet")
    optimized_df.write_parquet(temp_file)
    
    def health_analytics_processing(df: pl.DataFrame) -> pl.DataFrame:
        """Sample health analytics processing function."""
        return df.filter(pl.col("prescription_count") > 3).group_by(["state_territory", "age_group"]).agg([
            pl.col("prescription_count").sum().alias("total_prescriptions"),
            pl.col("total_cost_aud").mean().alias("avg_cost_per_service"),
            pl.col("health_risk_score").mean().alias("avg_risk_score"),
            pl.len().alias("area_count")
        ])
    
    console.print("🔄 Processing large dataset with adaptive streaming...")
    
    streaming_result = optimizer.process_large_dataset_streaming(
        temp_file,
        health_analytics_processing,
        batch_size=10000
    )
    
    if "stats" in streaming_result:
        stats = streaming_result["stats"]
        console.print(f"✅ Streaming completed: {stats['total_rows_processed']:,} rows in "
                     f"{stats['batches_processed']} batches")
        
        if "result" in streaming_result and streaming_result["result"] is not None:
            result_df = streaming_result["result"]
            console.print(f"📈 Analysis result: {result_df.shape[0]} state/age groups analyzed")
    
    # Clean up temp file
    if temp_file.exists():
        temp_file.unlink()
    
    # Step 5: Memory Optimization Summary
    console.print("\n[bold]Step 5: Memory Optimization Summary[/bold]")
    
    summary = optimizer.get_memory_optimization_summary()
    
    summary_table = Table(show_header=True, header_style="bold blue")
    summary_table.add_column("Summary Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")
    
    summary_table.add_row("Operations Tracked", str(summary.get("total_operations_tracked", 0)))
    summary_table.add_row("Average Memory Usage", f"{summary.get('average_peak_memory_mb', 0):.1f} MB")
    summary_table.add_row("Memory Efficiency", f"{summary.get('average_efficiency_ratio', 0):.3f}")
    summary_table.add_row("Current Memory Usage", f"{summary.get('current_memory_usage_gb', 0):.2f} GB")
    summary_table.add_row("Memory Pressure Level", f"{summary.get('memory_pressure_level', 0):.1%}")
    
    console.print(summary_table)
    
    # Step 6: Optimization Recommendations
    console.print("\n[bold]Step 6: Optimization Recommendations[/bold]")
    
    recommendations = optimizer.get_memory_optimization_recommendations()
    
    if recommendations:
        rec_table = Table(show_header=True, header_style="bold magenta")
        rec_table.add_column("Priority", style="red")
        rec_table.add_column("Category", style="cyan")
        rec_table.add_column("Recommendation", style="white")
        rec_table.add_column("Est. Savings", style="green")
        
        for rec in recommendations[:5]:  # Show top 5 recommendations
            rec_table.add_row(
                rec.priority.upper(),
                rec.category.title(),
                rec.title,
                f"{rec.estimated_memory_savings_mb:.0f} MB"
            )
        
        console.print(rec_table)
    else:
        console.print("[green]✅ No optimization recommendations - memory usage is optimal!")
    
    # Final summary
    console.print(Panel.fit(
        f"[bold green]🎉 Memory Optimization Demo Completed![/bold green]\n\n"
        f"• Memory savings: {memory_savings_mb:.2f}MB ({memory_savings_percent:.1f}% reduction)\n"
        f"• Optimizations applied: {len(optimization_stats['optimizations_applied'])}\n"
        f"• Streaming processing: {stats['total_rows_processed']:,} rows processed\n"
        f"• Recommendations: {len(recommendations)} optimization suggestions\n\n"
        f"[dim]Phase 4.3 memory optimization provides production-ready memory management\n"
        f"for Australian health data analytics with 497,181+ records.[/dim]",
        title="Phase 4.3 Complete"
    ))


if __name__ == "__main__":
    demonstrate_memory_optimization()