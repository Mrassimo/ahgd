"""
Performance Dashboard - Visual reporting and monitoring for Phase 4.4

Provides comprehensive performance dashboards and reports for the Australian health data
storage optimization pipeline, building on the benchmarking suite results.

Key Features:
- Interactive performance dashboards with visualizations
- Trend analysis and performance regression detection
- Component comparison and optimization tracking
- Automated performance reports
- Real-time monitoring integration
"""

import polars as pl
import numpy as np
# Skip problematic visualization imports for now
plt = None
sns = None
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
from datetime import datetime, timedelta
from dataclasses import asdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# Skip pandas for now due to compatibility issues
# Will use Polars for data processing instead

from .performance_benchmarking_suite import BenchmarkSuite, BenchmarkResult, PerformanceBenchmarkingSuite

logger = logging.getLogger(__name__)


class PerformanceDashboard:
    """
    Performance dashboard for visualizing and monitoring storage optimization results.
    Provides comprehensive reporting and trend analysis for Phase 4 components.
    """
    
    def __init__(self, benchmarks_dir: Optional[Path] = None):
        """Initialize performance dashboard with benchmark results directory."""
        self.benchmarks_dir = benchmarks_dir or Path("data/performance_benchmarks")
        self.benchmarks_dir.mkdir(parents=True, exist_ok=True)
        
        # Dashboard configuration
        self.dashboard_config = {
            "figure_width": 1200,
            "figure_height": 600,
            "color_palette": "viridis",
            "performance_threshold": 0.8,
            "regression_threshold": 0.9
        }
        
        # Load historical benchmark results
        self.historical_results = self._load_historical_results()
        
        logger.info(f"Initialized performance dashboard with {len(self.historical_results)} historical benchmarks")
    
    def _load_historical_results(self) -> List[BenchmarkSuite]:
        """Load all historical benchmark results."""
        historical_results = []
        
        try:
            for results_file in self.benchmarks_dir.glob("benchmark_results_*.json"):
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                    
                    # Convert back to BenchmarkSuite object
                    suite = BenchmarkSuite(**data)
                    historical_results.append(suite)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {results_file}: {e}")
            
            # Sort by timestamp
            historical_results.sort(key=lambda x: x.timestamp)
            
        except Exception as e:
            logger.error(f"Failed to load historical results: {e}")
        
        return historical_results
    
    def create_performance_overview_dashboard(self, suite: BenchmarkSuite) -> str:
        """Create comprehensive performance overview dashboard."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Component Performance Comparison", 
                    "Data Size Scaling Analysis",
                    "Execution Time vs Data Size",
                    "Performance Score Distribution"
                ],
                specs=[[{"secondary_y": True}, {"secondary_y": True}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Prepare data
            df = self._convert_benchmark_to_dataframe(suite.benchmark_results)
            
            # 1. Component Performance Comparison (bar chart)
            component_perf = df.groupby('component').agg({
                'performance_score': 'mean',
                'execution_time_seconds': 'mean',
                'throughput_mb_per_second': 'mean'
            }).reset_index()
            
            fig.add_trace(
                go.Bar(
                    x=component_perf['component'],
                    y=component_perf['performance_score'],
                    name="Performance Score",
                    marker_color="lightblue"
                ),
                row=1, col=1
            )
            
            # 2. Data Size Scaling Analysis
            for component in df['component'].unique():
                component_data = df[df['component'] == component]
                fig.add_trace(
                    go.Scatter(
                        x=component_data['rows_processed'],
                        y=component_data['throughput_mb_per_second'],
                        mode='lines+markers',
                        name=f"{component} Throughput",
                        line=dict(width=2)
                    ),
                    row=1, col=2
                )
            
            # 3. Execution Time vs Data Size
            fig.add_trace(
                go.Scatter(
                    x=df['rows_processed'],
                    y=df['execution_time_seconds'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=df['performance_score'],
                        colorscale='viridis',
                        showscale=True,
                        colorbar=dict(title="Performance Score")
                    ),
                    text=df['component'],
                    name="Execution Time",
                    hovertemplate="<b>%{text}</b><br>Rows: %{x:,}<br>Time: %{y:.3f}s<br>Score: %{marker.color:.3f}<extra></extra>"
                ),
                row=2, col=1
            )
            
            # 4. Performance Score Distribution
            fig.add_trace(
                go.Histogram(
                    x=df['performance_score'],
                    nbinsx=20,
                    name="Score Distribution",
                    marker_color="lightgreen"
                ),
                row=2, col=2
            )
            
            # Add performance threshold line
            fig.add_hline(
                y=self.dashboard_config["performance_threshold"], 
                line_dash="dash", 
                line_color="red",
                annotation_text="Performance Threshold",
                row=1, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"Phase 4.4 Storage Optimization Performance Dashboard<br><sub>Benchmark Date: {suite.timestamp[:10]}</sub>",
                height=800,
                width=1400,
                showlegend=True,
                template="plotly_white"
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Component", row=1, col=1)
            fig.update_yaxes(title_text="Performance Score", row=1, col=1)
            fig.update_xaxes(title_text="Rows Processed", row=1, col=2)
            fig.update_yaxes(title_text="Throughput (MB/s)", row=1, col=2)
            fig.update_xaxes(title_text="Rows Processed", row=2, col=1)
            fig.update_yaxes(title_text="Execution Time (s)", row=2, col=1)
            fig.update_xaxes(title_text="Performance Score", row=2, col=2)
            fig.update_yaxes(title_text="Frequency", row=2, col=2)
            
            # Save dashboard
            dashboard_file = self.benchmarks_dir / f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(str(dashboard_file))
            
            logger.info(f"Performance dashboard created: {dashboard_file}")
            return str(dashboard_file)
            
        except Exception as e:
            logger.error(f"Failed to create performance dashboard: {e}")
            return ""
    
    def create_trend_analysis_dashboard(self) -> str:
        """Create trend analysis dashboard from historical results."""
        if len(self.historical_results) < 2:
            logger.warning("Need at least 2 historical benchmarks for trend analysis")
            return ""
        
        try:
            # Create subplots for trend analysis
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "Performance Score Trends Over Time",
                    "Component Performance Evolution", 
                    "Execution Time Trends",
                    "Regression Detection"
                ]
            )
            
            # Prepare historical data
            trend_data = self._prepare_trend_data()
            
            # 1. Overall performance score trends
            for component in trend_data['component'].unique():
                component_data = trend_data[trend_data['component'] == component]
                fig.add_trace(
                    go.Scatter(
                        x=component_data['timestamp'],
                        y=component_data['avg_performance_score'],
                        mode='lines+markers',
                        name=f"{component} Score",
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
            
            # 2. Component comparison over time
            latest_data = trend_data[trend_data['timestamp'] == trend_data['timestamp'].max()]
            fig.add_trace(
                go.Bar(
                    x=latest_data['component'],
                    y=latest_data['avg_performance_score'],
                    name="Latest Performance",
                    marker_color="lightblue"
                ),
                row=1, col=2
            )
            
            # 3. Execution time trends
            for component in trend_data['component'].unique():
                component_data = trend_data[trend_data['component'] == component]
                fig.add_trace(
                    go.Scatter(
                        x=component_data['timestamp'],
                        y=component_data['avg_execution_time'],
                        mode='lines+markers',
                        name=f"{component} Time",
                        line=dict(width=2)
                    ),
                    row=2, col=1
                )
            
            # 4. Regression detection (performance degradation)
            regression_data = self._detect_performance_regression()
            if not regression_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=regression_data['timestamp'],
                        y=regression_data['regression_factor'],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color='red',
                            symbol='x'
                        ),
                        name="Performance Regression",
                        hovertemplate="<b>Regression Detected</b><br>Date: %{x}<br>Factor: %{y:.2f}<extra></extra>"
                    ),
                    row=2, col=2
                )
                
                # Add regression threshold line
                fig.add_hline(
                    y=self.dashboard_config["regression_threshold"],
                    line_dash="dash",
                    line_color="red", 
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Storage Optimization Performance Trends Analysis",
                height=800,
                width=1400,
                showlegend=True,
                template="plotly_white"
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=1, col=1)
            fig.update_yaxes(title_text="Performance Score", row=1, col=1)
            fig.update_xaxes(title_text="Component", row=1, col=2)
            fig.update_yaxes(title_text="Performance Score", row=1, col=2)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Execution Time (s)", row=2, col=1)
            fig.update_xaxes(title_text="Date", row=2, col=2)
            fig.update_yaxes(title_text="Regression Factor", row=2, col=2)
            
            # Save trend dashboard
            trend_file = self.benchmarks_dir / f"trend_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(str(trend_file))
            
            logger.info(f"Trend analysis dashboard created: {trend_file}")
            return str(trend_file)
            
        except Exception as e:
            logger.error(f"Failed to create trend analysis dashboard: {e}")
            return ""
    
    def _convert_benchmark_to_dataframe(self, results: List[BenchmarkResult]) -> pl.DataFrame:
        """Convert benchmark results to Polars DataFrame for visualization."""
        data = []
        for result in results:
            data.append({
                'test_name': result.test_name,
                'component': result.component,
                'data_size_mb': result.data_size_mb,
                'rows_processed': result.rows_processed,
                'execution_time_seconds': result.execution_time_seconds,
                'memory_usage_mb': result.memory_usage_mb,
                'throughput_mb_per_second': result.throughput_mb_per_second,
                'performance_score': result.performance_score,
                'optimization_count': len(result.optimization_applied),
                'baseline_comparison': result.baseline_comparison or 1.0
            })
        return pl.DataFrame(data)
    
    def _prepare_trend_data(self) -> pl.DataFrame:
        """Prepare historical data for trend analysis."""
        trend_data = []
        
        for suite in self.historical_results:
            # Group by component for each benchmark suite
            df = self._convert_benchmark_to_dataframe(suite.benchmark_results)
            component_stats = df.groupby('component').agg({
                'performance_score': 'mean',
                'execution_time_seconds': 'mean',
                'throughput_mb_per_second': 'mean'
            }).reset_index()
            
            for _, row in component_stats.iterrows():
                trend_data.append({
                    'timestamp': suite.timestamp[:10],  # Date only
                    'component': row['component'],
                    'avg_performance_score': row['performance_score'],
                    'avg_execution_time': row['execution_time_seconds'],
                    'avg_throughput': row['throughput_mb_per_second']
                })
        
        return pl.DataFrame(trend_data)
    
    def _detect_performance_regression(self) -> pl.DataFrame:
        """Detect performance regressions in historical data."""
        if len(self.historical_results) < 2:
            return pl.DataFrame()
        
        regression_data = []
        
        for i in range(1, len(self.historical_results)):
            current_suite = self.historical_results[i]
            previous_suite = self.historical_results[i-1]
            
            # Compare average performance scores
            current_avg = current_suite.average_performance_score
            previous_avg = previous_suite.average_performance_score
            
            regression_factor = current_avg / previous_avg if previous_avg > 0 else 1.0
            
            if regression_factor < self.dashboard_config["regression_threshold"]:
                regression_data.append({
                    'timestamp': current_suite.timestamp[:10],
                    'regression_factor': regression_factor,
                    'current_score': current_avg,
                    'previous_score': previous_avg
                })
        
        return pl.DataFrame(regression_data)
    
    def generate_performance_report(self, suite: BenchmarkSuite) -> str:
        """Generate comprehensive performance report."""
        try:
            report_lines = []
            
            # Header
            report_lines.append("# Phase 4.4 Storage Optimization Performance Report")
            report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"**Benchmark Date:** {suite.timestamp[:10]}")
            report_lines.append("")
            
            # Executive Summary
            report_lines.append("## Executive Summary")
            report_lines.append(f"- **Total Tests:** {suite.total_tests}")
            report_lines.append(f"- **Total Execution Time:** {suite.total_execution_time:.2f} seconds")
            report_lines.append(f"- **Average Performance Score:** {suite.average_performance_score:.3f}")
            
            # Performance status
            if suite.average_performance_score >= 0.8:
                status = "🟢 EXCELLENT"
            elif suite.average_performance_score >= 0.6:
                status = "🟡 GOOD" 
            else:
                status = "🔴 NEEDS IMPROVEMENT"
            report_lines.append(f"- **Overall Status:** {status}")
            report_lines.append("")
            
            # Component Performance
            report_lines.append("## Component Performance Analysis")
            report_lines.append("")
            
            for component, stats in suite.performance_summary["by_component"].items():
                report_lines.append(f"### {component}")
                report_lines.append(f"- Average Execution Time: {stats['avg_execution_time']:.3f}s")
                report_lines.append(f"- Average Throughput: {stats['avg_throughput']:.1f} MB/s")
                report_lines.append(f"- Performance Score: {stats['avg_performance_score']:.3f}")
                report_lines.append(f"- Tests Completed: {stats['test_count']}")
                
                # Performance rating
                score = stats['avg_performance_score']
                if score >= 0.8:
                    rating = "🟢 Excellent"
                elif score >= 0.6:
                    rating = "🟡 Good"
                else:
                    rating = "🔴 Needs Improvement"
                report_lines.append(f"- Rating: {rating}")
                report_lines.append("")
            
            # Data Size Analysis
            report_lines.append("## Data Size Scaling Analysis")
            report_lines.append("")
            
            for size, stats in suite.performance_summary["by_data_size"].items():
                report_lines.append(f"### {size} Rows")
                report_lines.append(f"- Average Execution Time: {stats['avg_execution_time']:.3f}s")
                report_lines.append(f"- Average Throughput: {stats['avg_throughput']:.1f} MB/s")
                report_lines.append(f"- Performance Score: {stats['avg_performance_score']:.3f}")
                report_lines.append("")
            
            # Optimization Impact
            report_lines.append("## Optimization Impact")
            total_optimizations = sum(len(r.optimization_applied) for r in suite.benchmark_results)
            memory_savings = [r for r in suite.benchmark_results if r.component == "MemoryOptimizer"]
            
            if memory_savings:
                avg_memory_saved = np.mean([r.memory_usage_mb for r in memory_savings])
                report_lines.append(f"- **Total Optimizations Applied:** {total_optimizations}")
                report_lines.append(f"- **Average Memory Savings:** {avg_memory_saved:.2f} MB")
            
            # Key optimizations by component
            optimization_summary = {}
            for result in suite.benchmark_results:
                for opt in result.optimization_applied:
                    if opt not in optimization_summary:
                        optimization_summary[opt] = 0
                    optimization_summary[opt] += 1
            
            if optimization_summary:
                report_lines.append("- **Most Common Optimizations:**")
                for opt, count in sorted(optimization_summary.items(), key=lambda x: x[1], reverse=True)[:5]:
                    report_lines.append(f"  - {opt}: {count} times")
            report_lines.append("")
            
            # Recommendations
            report_lines.append("## Performance Recommendations")
            for i, rec in enumerate(suite.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
            
            # Historical Comparison
            if len(self.historical_results) > 1:
                report_lines.append("## Historical Comparison")
                previous_suite = self.historical_results[-2] if len(self.historical_results) >= 2 else None
                
                if previous_suite:
                    score_change = suite.average_performance_score - previous_suite.average_performance_score
                    time_change = suite.total_execution_time - previous_suite.total_execution_time
                    
                    if score_change > 0:
                        trend = f"🟢 +{score_change:.3f}"
                    elif score_change < -0.05:
                        trend = f"🔴 {score_change:.3f}"
                    else:
                        trend = f"🟡 {score_change:.3f}"
                    
                    report_lines.append(f"- Performance Score Change: {trend}")
                    report_lines.append(f"- Execution Time Change: {time_change:.2f}s")
                report_lines.append("")
            
            # Technical Details
            report_lines.append("## Technical Details")
            report_lines.append(f"- **Total Data Processed:** {suite.performance_summary['overall_metrics']['total_data_processed_mb']:.1f} MB")
            report_lines.append(f"- **Total Rows Processed:** {suite.performance_summary['overall_metrics']['total_rows_processed']:,}")
            report_lines.append(f"- **Average Throughput:** {suite.performance_summary['overall_metrics']['average_throughput']:.1f} MB/s")
            report_lines.append("")
            
            # Footer
            report_lines.append("---")
            report_lines.append("*This report was automatically generated by the Phase 4.4 Performance Benchmarking Suite*")
            
            # Save report
            report_content = "\n".join(report_lines)
            report_file = self.benchmarks_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            with open(report_file, 'w') as f:
                f.write(report_content)
            
            logger.info(f"Performance report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return ""
    
    def create_optimization_impact_visualization(self, suite: BenchmarkSuite) -> str:
        """Create visualization showing optimization impact."""
        try:
            # Create comparison of optimized vs baseline performance
            fig = go.Figure()
            
            # Get results with baseline comparisons
            results_with_baseline = [r for r in suite.benchmark_results if r.baseline_comparison is not None]
            
            if not results_with_baseline:
                logger.warning("No baseline comparisons available for optimization impact visualization")
                return ""
            
            components = list(set(r.component for r in results_with_baseline))
            
            for component in components:
                component_results = [r for r in results_with_baseline if r.component == component]
                
                baseline_scores = [r.performance_score / r.baseline_comparison for r in component_results]
                current_scores = [r.performance_score for r in component_results]
                data_sizes = [r.rows_processed for r in component_results]
                
                # Add baseline performance
                fig.add_trace(go.Scatter(
                    x=data_sizes,
                    y=baseline_scores,
                    mode='lines+markers',
                    name=f"{component} (Baseline)",
                    line=dict(dash='dash', width=2),
                    opacity=0.7
                ))
                
                # Add current optimized performance
                fig.add_trace(go.Scatter(
                    x=data_sizes,
                    y=current_scores,
                    mode='lines+markers',
                    name=f"{component} (Optimized)",
                    line=dict(width=3)
                ))
            
            fig.update_layout(
                title="Storage Optimization Impact Analysis<br><sub>Baseline vs Optimized Performance</sub>",
                xaxis_title="Data Size (Rows)",
                yaxis_title="Performance Score",
                height=600,
                width=1000,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Save impact visualization
            impact_file = self.benchmarks_dir / f"optimization_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            fig.write_html(str(impact_file))
            
            logger.info(f"Optimization impact visualization created: {impact_file}")
            return str(impact_file)
            
        except Exception as e:
            logger.error(f"Failed to create optimization impact visualization: {e}")
            return ""


def create_comprehensive_performance_dashboard(benchmarks_dir: Optional[Path] = None) -> Dict[str, str]:
    """Create comprehensive performance dashboard suite."""
    dashboard = PerformanceDashboard(benchmarks_dir)
    
    # Run fresh benchmark if needed
    benchmarking_suite = PerformanceBenchmarkingSuite()
    latest_results = benchmarking_suite.run_comprehensive_benchmark()
    
    # Create all dashboard components
    outputs = {}
    
    # Main performance dashboard
    outputs["main_dashboard"] = dashboard.create_performance_overview_dashboard(latest_results)
    
    # Trend analysis dashboard
    outputs["trend_dashboard"] = dashboard.create_trend_analysis_dashboard()
    
    # Performance report
    outputs["performance_report"] = dashboard.generate_performance_report(latest_results)
    
    # Optimization impact visualization
    outputs["impact_visualization"] = dashboard.create_optimization_impact_visualization(latest_results)
    
    return outputs


if __name__ == "__main__":
    # Development testing
    outputs = create_comprehensive_performance_dashboard()
    
    print("📊 Performance Dashboard Suite Created:")
    for component, file_path in outputs.items():
        if file_path:
            print(f"  • {component}: {file_path}")
        else:
            print(f"  • {component}: ❌ Failed to create")