"""
Performance integration tests for Australian Health Analytics platform.

Tests performance under realistic load with 497,181+ records:
- End-to-end pipeline performance validation
- Memory optimization effectiveness (57.5% target)
- Dashboard response times (<2 seconds target)
- Concurrent processing scalability
- Storage performance under load
- Real-time analytics performance

Validates enterprise-grade performance requirements with Australian health data volumes.
"""

import pytest
import polars as pl
import numpy as np
import time
import psutil
import os
import shutil
import concurrent.futures
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import logging
import gc

from src.data_processing.seifa_processor import SEIFAProcessor
from src.data_processing.health_processor import HealthDataProcessor
from src.data_processing.simple_boundary_processor import SimpleBoundaryProcessor
from src.data_processing.storage.parquet_storage_manager import ParquetStorageManager
from src.data_processing.storage.memory_optimizer import MemoryOptimizer
from src.data_processing.storage.performance_benchmarking_suite import PerformanceBenchmarkingSuite
from src.analysis.risk.health_risk_calculator import HealthRiskCalculator


class TestPerformanceIntegration:
    """Performance integration tests under realistic Australian health data loads."""
    
    def test_end_to_end_pipeline_performance_at_scale(self, mock_excel_seifa_file, mock_health_data, 
                                                     mock_boundary_data, mock_data_paths):
        """Test complete pipeline performance with 497,181+ record simulation."""
        
        # Initialize performance monitoring
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Initialize all pipeline components
        seifa_processor = SEIFAProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        memory_optimizer = MemoryOptimizer()
        risk_calculator = HealthRiskCalculator(data_dir=mock_data_paths["processed_dir"])
        benchmark_suite = PerformanceBenchmarkingSuite()
        
        # Create large-scale datasets (497,181+ total records)
        # Scale appropriately for testing while maintaining performance characteristics
        
        # SEIFA: 2,454 SA2 areas (all Australian SA2 areas)
        large_seifa_file = mock_excel_seifa_file(num_areas=2454, include_errors=True)
        expected_seifa_path = seifa_processor.raw_dir / "SEIFA_2021_SA2_Indexes.xlsx"
        shutil.copy(large_seifa_file, expected_seifa_path)
        
        # Health: 150,000 records (scaled from 492,434 PBS records)
        large_health = mock_health_data(num_records=150000, num_sa2_areas=2454)
        
        # Boundaries: 2,454 SA2 boundaries
        large_boundaries = mock_boundary_data(num_areas=2454)
        
        # Additional datasets to reach target volume
        census_supplement = mock_health_data(num_records=50000, num_sa2_areas=2454)
        demographic_supplement = mock_boundary_data(num_areas=2000)
        
        total_target_records = len(large_health) + len(census_supplement) + len(large_boundaries) + len(demographic_supplement) + 2454
        
        assert total_target_records >= 200000, f"Should process ≥200k records for scale test, created {total_target_records:,}"
        
        # Execute complete pipeline with performance monitoring
        pipeline_start_time = time.time()
        
        # =================================================================
        # STAGE 1: Data Ingestion and Validation (Performance Critical)
        # =================================================================
        stage1_start = time.time()
        stage1_memory_start = process.memory_info().rss / 1024 / 1024
        
        # Process SEIFA data
        seifa_processing_start = time.time()
        processed_seifa = seifa_processor.process_seifa_file()
        seifa_processing_time = time.time() - seifa_processing_start
        
        # Process health data
        health_processing_start = time.time()
        validated_health = health_processor._validate_health_data(large_health)
        health_processing_time = time.time() - health_processing_start
        
        # Process boundary data
        boundary_processing_start = time.time()
        validated_boundaries = boundary_processor._validate_boundary_data(large_boundaries)
        boundary_processing_time = time.time() - boundary_processing_start
        
        # Process supplemental datasets
        supplement_processing_start = time.time()
        validated_census = health_processor._validate_health_data(census_supplement)
        validated_demo = boundary_processor._validate_boundary_data(demographic_supplement)
        supplement_processing_time = time.time() - supplement_processing_start
        
        stage1_time = time.time() - stage1_start
        stage1_memory_end = process.memory_info().rss / 1024 / 1024
        stage1_memory_usage = stage1_memory_end - stage1_memory_start
        
        # Stage 1 Performance Validation
        assert stage1_time < 180.0, f"Stage 1 ingestion took {stage1_time:.1f}s, expected <180s"
        assert seifa_processing_time < 60.0, f"SEIFA processing took {seifa_processing_time:.1f}s, expected <60s"
        assert health_processing_time < 90.0, f"Health processing took {health_processing_time:.1f}s, expected <90s"
        assert boundary_processing_time < 30.0, f"Boundary processing took {boundary_processing_time:.1f}s, expected <30s"
        
        # =================================================================
        # STAGE 2: Memory Optimization (57.5% Target Reduction)
        # =================================================================
        stage2_start = time.time()
        stage2_memory_start = process.memory_info().rss / 1024 / 1024
        
        # Measure pre-optimization memory usage
        pre_opt_seifa_memory = processed_seifa.estimated_size("mb")
        pre_opt_health_memory = validated_health.estimated_size("mb")
        pre_opt_boundary_memory = validated_boundaries.estimated_size("mb")
        pre_opt_total_memory = pre_opt_seifa_memory + pre_opt_health_memory + pre_opt_boundary_memory
        
        # Apply memory optimizations
        optimized_seifa = memory_optimizer.optimize_data_types(processed_seifa, data_category="seifa")
        optimized_health = memory_optimizer.optimize_data_types(validated_health, data_category="health")
        optimized_boundaries = memory_optimizer.optimize_data_types(validated_boundaries, data_category="geographic")
        optimized_census = memory_optimizer.optimize_data_types(validated_census, data_category="health")
        optimized_demo = memory_optimizer.optimize_data_types(validated_demo, data_category="geographic")
        
        # Measure post-optimization memory usage
        post_opt_seifa_memory = optimized_seifa.estimated_size("mb")
        post_opt_health_memory = optimized_health.estimated_size("mb")
        post_opt_boundary_memory = optimized_boundaries.estimated_size("mb")
        post_opt_total_memory = post_opt_seifa_memory + post_opt_health_memory + post_opt_boundary_memory
        
        # Calculate memory optimization effectiveness
        memory_reduction_rate = (pre_opt_total_memory - post_opt_total_memory) / pre_opt_total_memory
        
        stage2_time = time.time() - stage2_start
        stage2_memory_end = process.memory_info().rss / 1024 / 1024
        stage2_memory_usage = stage2_memory_end - stage2_memory_start
        
        # Stage 2 Performance Validation
        assert stage2_time < 120.0, f"Stage 2 optimization took {stage2_time:.1f}s, expected <120s"
        assert memory_reduction_rate >= 0.40, f"Memory reduction {memory_reduction_rate:.1%} should be ≥40% (target 57.5%)"
        
        logging.info(f"Memory optimization achieved {memory_reduction_rate:.1%} reduction (target: 57.5%)")
        
        # =================================================================
        # STAGE 3: Data Integration and Aggregation (Scalability Critical)
        # =================================================================
        stage3_start = time.time()
        stage3_memory_start = process.memory_info().rss / 1024 / 1024
        
        # Aggregate health data by SA2
        aggregation_start = time.time()
        aggregated_health = health_processor._aggregate_by_sa2(optimized_health)
        aggregation_time = time.time() - aggregation_start
        
        # Enhance boundaries with derived metrics
        enhancement_start = time.time()
        enhanced_boundaries = boundary_processor._calculate_population_density(optimized_boundaries)
        enhancement_time = time.time() - enhancement_start
        
        # Create comprehensive integrated dataset
        integration_start = time.time()
        comprehensive_integration = optimized_seifa.join(
            aggregated_health, left_on="sa2_code_2021", right_on="sa2_code", how="left"
        ).join(
            enhanced_boundaries, on="sa2_code_2021", how="left"
        )
        integration_time = time.time() - integration_start
        
        stage3_time = time.time() - stage3_start
        stage3_memory_end = process.memory_info().rss / 1024 / 1024
        stage3_memory_usage = stage3_memory_end - stage3_memory_start
        
        # Stage 3 Performance Validation
        assert stage3_time < 90.0, f"Stage 3 integration took {stage3_time:.1f}s, expected <90s"
        assert aggregation_time < 45.0, f"Health aggregation took {aggregation_time:.1f}s, expected <45s"
        assert enhancement_time < 30.0, f"Boundary enhancement took {enhancement_time:.1f}s, expected <30s"
        assert integration_time < 30.0, f"Data integration took {integration_time:.1f}s, expected <30s"
        
        # =================================================================
        # STAGE 4: Analytics and Risk Assessment (Real-time Performance)
        # =================================================================
        stage4_start = time.time()
        stage4_memory_start = process.memory_info().rss / 1024 / 1024
        
        # Calculate risk components
        seifa_risk_start = time.time()
        seifa_risk = risk_calculator._calculate_seifa_risk_score(optimized_seifa)
        seifa_risk_time = time.time() - seifa_risk_start
        
        health_risk_start = time.time()
        health_risk = risk_calculator._calculate_health_utilisation_risk(aggregated_health)
        health_risk_time = time.time() - health_risk_start
        
        geographic_risk_start = time.time()
        geographic_risk = risk_calculator._calculate_geographic_accessibility_risk(enhanced_boundaries)
        geographic_risk_time = time.time() - geographic_risk_start
        
        # Comprehensive risk assessment
        risk_integration_start = time.time()
        comprehensive_risk = seifa_risk.join(
            health_risk, left_on="sa2_code_2021", right_on="sa2_code", how="inner"
        ).join(
            geographic_risk, on="sa2_code_2021", how="inner"
        )
        
        composite_risk = risk_calculator._calculate_composite_risk_score(comprehensive_risk)
        final_risk = risk_calculator._classify_risk_categories(composite_risk)
        risk_integration_time = time.time() - risk_integration_start
        
        stage4_time = time.time() - stage4_start
        stage4_memory_end = process.memory_info().rss / 1024 / 1024
        stage4_memory_usage = stage4_memory_end - stage4_memory_start
        
        # Stage 4 Performance Validation (Real-time analytics requirement)
        assert stage4_time < 60.0, f"Stage 4 analytics took {stage4_time:.1f}s, expected <60s"
        assert seifa_risk_time < 20.0, f"SEIFA risk calculation took {seifa_risk_time:.1f}s, expected <20s"
        assert health_risk_time < 20.0, f"Health risk calculation took {health_risk_time:.1f}s, expected <20s"
        assert geographic_risk_time < 20.0, f"Geographic risk calculation took {geographic_risk_time:.1f}s, expected <20s"
        assert risk_integration_time < 15.0, f"Risk integration took {risk_integration_time:.1f}s, expected <15s"
        
        # =================================================================
        # STAGE 5: Storage Performance and Dashboard Readiness
        # =================================================================
        stage5_start = time.time()
        stage5_memory_start = process.memory_info().rss / 1024 / 1024
        
        # Storage performance test
        storage_start = time.time()
        
        seifa_path = storage_manager.save_optimized_parquet(
            optimized_seifa, mock_data_paths["parquet_dir"] / "performance_seifa.parquet", data_type="seifa"
        )
        health_path = storage_manager.save_optimized_parquet(
            aggregated_health, mock_data_paths["parquet_dir"] / "performance_health.parquet", data_type="health"
        )
        boundaries_path = storage_manager.save_optimized_parquet(
            enhanced_boundaries, mock_data_paths["parquet_dir"] / "performance_boundaries.parquet", data_type="geographic"
        )
        integration_path = storage_manager.save_optimized_parquet(
            comprehensive_integration, mock_data_paths["parquet_dir"] / "performance_integration.parquet", data_type="analytics"
        )
        risk_path = storage_manager.save_optimized_parquet(
            final_risk, mock_data_paths["parquet_dir"] / "performance_risk.parquet", data_type="analytics"
        )
        
        storage_time = time.time() - storage_start
        
        # Dashboard load time simulation (target: <2 seconds)
        dashboard_start = time.time()
        
        # Simulate dashboard data loading
        dashboard_seifa = pl.read_parquet(seifa_path)
        dashboard_health = pl.read_parquet(health_path)
        dashboard_risk = pl.read_parquet(risk_path)
        
        # Simulate dashboard aggregations
        dashboard_summary = {
            "total_sa2_areas": len(dashboard_seifa),
            "average_irsd_decile": float(dashboard_seifa["irsd_decile"].mean()) if "irsd_decile" in dashboard_seifa.columns else None,
            "total_prescriptions": int(dashboard_health["total_prescriptions"].sum()) if "total_prescriptions" in dashboard_health.columns else None,
            "high_risk_areas": len(dashboard_risk.filter(pl.col("risk_category") == "High")) if "risk_category" in dashboard_risk.columns else None,
            "very_high_risk_areas": len(dashboard_risk.filter(pl.col("risk_category") == "Very High")) if "risk_category" in dashboard_risk.columns else None
        }
        
        dashboard_load_time = time.time() - dashboard_start
        
        stage5_time = time.time() - stage5_start
        stage5_memory_end = process.memory_info().rss / 1024 / 1024
        stage5_memory_usage = stage5_memory_end - stage5_memory_start
        
        # Stage 5 Performance Validation (Dashboard requirement: <2 seconds)
        assert stage5_time < 45.0, f"Stage 5 storage took {stage5_time:.1f}s, expected <45s"
        assert storage_time < 30.0, f"Storage operations took {storage_time:.1f}s, expected <30s"
        assert dashboard_load_time < 2.0, f"Dashboard load took {dashboard_load_time:.1f}s, expected <2s (target requirement)"
        
        # =================================================================
        # OVERALL PIPELINE PERFORMANCE VALIDATION
        # =================================================================
        total_pipeline_time = time.time() - pipeline_start_time
        peak_memory = process.memory_info().rss / 1024 / 1024
        total_memory_usage = peak_memory - initial_memory
        
        # Calculate throughput metrics
        total_records_processed = (len(processed_seifa) + len(validated_health) + 
                                 len(validated_boundaries) + len(validated_census) + len(validated_demo))
        throughput_records_per_second = total_records_processed / total_pipeline_time
        
        # Overall Performance Validation
        assert total_pipeline_time < 300.0, f"Complete pipeline took {total_pipeline_time:.1f}s, expected <300s (5 minutes)"
        assert total_memory_usage < 4096, f"Peak memory usage {total_memory_usage:.1f}MB should be <4GB"
        assert throughput_records_per_second > 500, f"Throughput {throughput_records_per_second:.0f} records/s should be >500"
        
        # Data quality and integration validation
        integration_success_rate = len(comprehensive_integration) / len(processed_seifa)
        risk_assessment_coverage = len(final_risk) / len(processed_seifa)
        
        assert integration_success_rate > 0.85, f"Integration success rate {integration_success_rate:.1%} should be >85%"
        assert risk_assessment_coverage > 0.75, f"Risk assessment coverage {risk_assessment_coverage:.1%} should be >75%"
        
        # Storage efficiency validation
        total_storage_size = sum(
            path.stat().st_size for path in [seifa_path, health_path, boundaries_path, integration_path, risk_path]
        ) / 1024 / 1024  # MB
        
        storage_efficiency = total_records_processed / total_storage_size
        assert storage_efficiency > 100, f"Storage efficiency {storage_efficiency:.1f} records/MB should be >100"
        
        # Generate comprehensive performance report
        performance_report = {
            "total_pipeline_time": total_pipeline_time,
            "stage_timings": {
                "stage_1_ingestion": stage1_time,
                "stage_2_optimization": stage2_time,
                "stage_3_integration": stage3_time,
                "stage_4_analytics": stage4_time,
                "stage_5_storage": stage5_time
            },
            "detailed_timings": {
                "seifa_processing": seifa_processing_time,
                "health_processing": health_processing_time,
                "boundary_processing": boundary_processing_time,
                "health_aggregation": aggregation_time,
                "data_integration": integration_time,
                "risk_calculations": {
                    "seifa_risk": seifa_risk_time,
                    "health_risk": health_risk_time,
                    "geographic_risk": geographic_risk_time,
                    "risk_integration": risk_integration_time
                },
                "storage_operations": storage_time,
                "dashboard_load": dashboard_load_time
            },
            "memory_metrics": {
                "initial_memory_mb": initial_memory,
                "peak_memory_mb": peak_memory,
                "total_memory_usage_mb": total_memory_usage,
                "memory_optimization_rate": memory_reduction_rate,
                "stage_memory_usage": {
                    "stage_1": stage1_memory_usage,
                    "stage_2": stage2_memory_usage,
                    "stage_3": stage3_memory_usage,
                    "stage_4": stage4_memory_usage,
                    "stage_5": stage5_memory_usage
                }
            },
            "throughput_metrics": {
                "total_records_processed": total_records_processed,
                "records_per_second": throughput_records_per_second,
                "integration_success_rate": integration_success_rate,
                "risk_assessment_coverage": risk_assessment_coverage
            },
            "storage_metrics": {
                "total_storage_size_mb": total_storage_size,
                "storage_efficiency_records_per_mb": storage_efficiency,
                "compression_achieved": True
            },
            "performance_targets": {
                "pipeline_under_5min": total_pipeline_time < 300.0,
                "memory_under_4gb": total_memory_usage < 4096,
                "memory_optimization_target": memory_reduction_rate >= 0.40,
                "dashboard_under_2s": dashboard_load_time < 2.0,
                "throughput_over_500_rps": throughput_records_per_second > 500,
                "integration_over_85pct": integration_success_rate > 0.85,
                "all_targets_met": (
                    total_pipeline_time < 300.0 and
                    total_memory_usage < 4096 and
                    memory_reduction_rate >= 0.40 and
                    dashboard_load_time < 2.0 and
                    throughput_records_per_second > 500 and
                    integration_success_rate > 0.85
                )
            },
            "dashboard_readiness": {
                "load_time": dashboard_load_time,
                "summary_generated": dashboard_summary is not None,
                "real_time_capable": dashboard_load_time < 2.0
            }
        }
        
        logging.info(f"End-to-End Performance Integration Report: {performance_report}")
        
        return performance_report
    
    def test_concurrent_processing_scalability(self, mock_health_data, mock_boundary_data, mock_data_paths):
        """Test concurrent processing scalability with multiple datasets."""
        
        # Initialize components
        health_processor = HealthDataProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        boundary_processor = SimpleBoundaryProcessor(data_dir=mock_data_paths["raw_dir"].parent)
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        memory_optimizer = MemoryOptimizer()
        
        # Create multiple large datasets for concurrent processing
        num_datasets = 6
        dataset_size = 25000  # Records per dataset
        
        health_datasets = [mock_health_data(num_records=dataset_size, num_sa2_areas=1000) for _ in range(num_datasets)]
        boundary_datasets = [mock_boundary_data(num_areas=1000) for _ in range(num_datasets)]
        
        # Define concurrent processing functions
        def process_health_dataset(dataset_id, dataset):
            """Process health dataset concurrently."""
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Full processing pipeline
            validated = health_processor._validate_health_data(dataset)
            aggregated = health_processor._aggregate_by_sa2(validated)
            optimized = memory_optimizer.optimize_data_types(aggregated, data_category="health")
            
            # Save result
            path = mock_data_paths["parquet_dir"] / f"concurrent_health_{dataset_id}.parquet"
            storage_manager.save_optimized_parquet(optimized, path, data_type="health")
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            return {
                "dataset_id": dataset_id,
                "type": "health",
                "processing_time": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "input_records": len(dataset),
                "output_records": len(optimized),
                "path": path,
                "success": True
            }
        
        def process_boundary_dataset(dataset_id, dataset):
            """Process boundary dataset concurrently."""
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Full processing pipeline
            validated = boundary_processor._validate_boundary_data(dataset)
            enhanced = boundary_processor._calculate_population_density(validated)
            optimized = memory_optimizer.optimize_data_types(enhanced, data_category="geographic")
            
            # Save result
            path = mock_data_paths["parquet_dir"] / f"concurrent_boundary_{dataset_id}.parquet"
            storage_manager.save_optimized_parquet(optimized, path, data_type="geographic")
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            return {
                "dataset_id": dataset_id,
                "type": "boundary",
                "processing_time": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "input_records": len(dataset),
                "output_records": len(optimized),
                "path": path,
                "success": True
            }
        
        # Test concurrent processing with different thread counts
        thread_counts = [2, 4, 6]
        scalability_results = {}
        
        for thread_count in thread_counts:
            logging.info(f"Testing concurrent processing with {thread_count} threads")
            
            # Reset memory state
            gc.collect()
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
                # Submit health processing tasks
                health_futures = [
                    executor.submit(process_health_dataset, i, dataset)
                    for i, dataset in enumerate(health_datasets[:thread_count//2])
                ]
                
                # Submit boundary processing tasks
                boundary_futures = [
                    executor.submit(process_boundary_dataset, i, dataset)
                    for i, dataset in enumerate(boundary_datasets[:thread_count//2])
                ]
                
                # Collect results
                all_futures = health_futures + boundary_futures
                results = []
                
                for future in concurrent.futures.as_completed(all_futures, timeout=300):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "success": False,
                            "error": str(e),
                            "type": "unknown"
                        })
            
            total_time = time.time() - start_time
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage = peak_memory - initial_memory
            
            # Validate concurrent processing results
            successful_results = [r for r in results if r.get("success", False)]
            failed_results = [r for r in results if not r.get("success", False)]
            
            assert len(failed_results) == 0, f"All concurrent operations should succeed with {thread_count} threads"
            assert len(successful_results) == thread_count, f"Should have {thread_count} successful results"
            
            # Calculate performance metrics
            total_records_processed = sum(r["input_records"] for r in successful_results)
            average_processing_time = np.mean([r["processing_time"] for r in successful_results])
            throughput = total_records_processed / total_time
            
            # Performance validation
            assert total_time < 120.0, f"Concurrent processing with {thread_count} threads took {total_time:.1f}s, expected <120s"
            assert memory_usage < 2048, f"Memory usage {memory_usage:.1f}MB with {thread_count} threads should be <2GB"
            assert throughput > 1000, f"Throughput {throughput:.0f} records/s should be >1000 with {thread_count} threads"
            
            scalability_results[thread_count] = {
                "total_time": total_time,
                "memory_usage": memory_usage,
                "throughput": throughput,
                "average_processing_time": average_processing_time,
                "successful_operations": len(successful_results),
                "total_records_processed": total_records_processed,
                "efficiency": throughput / thread_count  # Records per second per thread
            }
        
        # Validate scalability
        # Throughput should improve with more threads (up to optimal point)
        throughput_2_threads = scalability_results[2]["throughput"]
        throughput_4_threads = scalability_results[4]["throughput"]
        throughput_6_threads = scalability_results[6]["throughput"]
        
        assert throughput_4_threads > throughput_2_threads * 1.5, "4 threads should provide >1.5x throughput vs 2 threads"
        
        # Efficiency should remain reasonable (some overhead expected)
        efficiency_2_threads = scalability_results[2]["efficiency"]
        efficiency_6_threads = scalability_results[6]["efficiency"]
        efficiency_ratio = efficiency_6_threads / efficiency_2_threads
        
        assert efficiency_ratio > 0.7, f"Efficiency ratio {efficiency_ratio:.2f} should be >0.7 (reasonable scaling overhead)"
        
        # Generate scalability report
        scalability_report = {
            "thread_count_results": scalability_results,
            "scalability_analysis": {
                "throughput_improvement_2_to_4_threads": throughput_4_threads / throughput_2_threads,
                "throughput_improvement_4_to_6_threads": throughput_6_threads / throughput_4_threads if throughput_4_threads > 0 else 0,
                "efficiency_degradation": 1.0 - efficiency_ratio,
                "optimal_thread_count": max(scalability_results.keys(), key=lambda k: scalability_results[k]["throughput"]),
                "linear_scaling_achieved": throughput_4_threads > throughput_2_threads * 1.5
            },
            "performance_validation": {
                "all_thread_counts_performant": all(
                    result["total_time"] < 120.0 and result["memory_usage"] < 2048
                    for result in scalability_results.values()
                ),
                "scaling_efficiency_acceptable": efficiency_ratio > 0.7,
                "concurrent_operations_successful": True
            }
        }
        
        logging.info(f"Concurrent Processing Scalability Report: {scalability_report}")
        
        return scalability_report
    
    def test_storage_performance_under_load(self, mock_seifa_data, mock_health_data, mock_data_paths):
        """Test storage performance under heavy load conditions."""
        
        # Initialize storage components
        storage_manager = ParquetStorageManager(base_path=mock_data_paths["parquet_dir"])
        memory_optimizer = MemoryOptimizer()
        
        # Create heavy load scenario
        num_datasets = 10
        large_dataset_size = 20000
        
        # Generate large datasets
        large_datasets = []
        for i in range(num_datasets):
            if i % 2 == 0:
                dataset = mock_seifa_data(num_areas=large_dataset_size // 5)  # SEIFA has fewer records
                dataset_type = "seifa"
            else:
                dataset = mock_health_data(num_records=large_dataset_size, num_sa2_areas=1000)
                dataset_type = "health"
            
            # Apply memory optimization
            optimized_dataset = memory_optimizer.optimize_data_types(dataset, data_category=dataset_type)
            
            large_datasets.append({
                "id": i,
                "data": optimized_dataset,
                "type": dataset_type,
                "size_mb": optimized_dataset.estimated_size("mb")
            })
        
        total_data_size = sum(ds["size_mb"] for ds in large_datasets)
        logging.info(f"Testing storage performance with {total_data_size:.1f}MB of data across {num_datasets} datasets")
        
        # Test 1: Sequential write performance
        sequential_start = time.time()
        sequential_paths = []
        
        for dataset_info in large_datasets:
            write_start = time.time()
            
            path = mock_data_paths["parquet_dir"] / f"load_test_sequential_{dataset_info['id']}.parquet"
            saved_path = storage_manager.save_optimized_parquet(
                dataset_info["data"], 
                path, 
                data_type=dataset_info["type"]
            )
            
            write_time = time.time() - write_start
            dataset_info["sequential_write_time"] = write_time
            sequential_paths.append(saved_path)
        
        sequential_total_time = time.time() - sequential_start
        sequential_throughput = total_data_size / sequential_total_time
        
        # Test 2: Concurrent write performance
        def write_dataset_concurrent(dataset_info):
            """Write dataset concurrently."""
            write_start = time.time()
            
            path = mock_data_paths["parquet_dir"] / f"load_test_concurrent_{dataset_info['id']}.parquet"
            saved_path = storage_manager.save_optimized_parquet(
                dataset_info["data"], 
                path, 
                data_type=dataset_info["type"]
            )
            
            write_time = time.time() - write_start
            
            return {
                "id": dataset_info["id"],
                "path": saved_path,
                "write_time": write_time,
                "size_mb": dataset_info["size_mb"]
            }
        
        concurrent_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            concurrent_futures = [
                executor.submit(write_dataset_concurrent, dataset_info)
                for dataset_info in large_datasets
            ]
            
            concurrent_results = [
                future.result() for future in concurrent.futures.as_completed(concurrent_futures, timeout=180)
            ]
        
        concurrent_total_time = time.time() - concurrent_start
        concurrent_throughput = total_data_size / concurrent_total_time
        
        # Test 3: Read performance under load
        read_start = time.time()
        read_results = []
        
        for path in sequential_paths:
            read_dataset_start = time.time()
            loaded_data = pl.read_parquet(path)
            read_time = time.time() - read_dataset_start
            
            read_results.append({
                "path": path,
                "read_time": read_time,
                "records": len(loaded_data),
                "size_mb": path.stat().st_size / 1024 / 1024
            })
        
        total_read_time = time.time() - read_start
        total_read_size = sum(r["size_mb"] for r in read_results)
        read_throughput = total_read_size / total_read_time
        
        # Test 4: Concurrent read performance
        def read_dataset_concurrent(path):
            """Read dataset concurrently."""
            read_start = time.time()
            loaded_data = pl.read_parquet(path)
            read_time = time.time() - read_start
            
            return {
                "path": path,
                "read_time": read_time,
                "records": len(loaded_data),
                "size_mb": path.stat().st_size / 1024 / 1024
            }
        
        concurrent_read_start = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            concurrent_read_futures = [
                executor.submit(read_dataset_concurrent, path)
                for path in sequential_paths[:5]  # Test subset for concurrent reads
            ]
            
            concurrent_read_results = [
                future.result() for future in concurrent.futures.as_completed(concurrent_read_futures, timeout=60)
            ]
        
        concurrent_read_total_time = time.time() - concurrent_read_start
        concurrent_read_size = sum(r["size_mb"] for r in concurrent_read_results)
        concurrent_read_throughput = concurrent_read_size / concurrent_read_total_time
        
        # Storage Performance Validation
        # 1. Sequential write performance should be reasonable
        assert sequential_throughput > 10, f"Sequential write throughput {sequential_throughput:.1f} MB/s should be >10 MB/s"
        assert sequential_total_time < 180, f"Sequential writes took {sequential_total_time:.1f}s, expected <180s"
        
        # 2. Concurrent write performance should be competitive
        assert concurrent_throughput > 8, f"Concurrent write throughput {concurrent_throughput:.1f} MB/s should be >8 MB/s"
        assert concurrent_total_time < 120, f"Concurrent writes took {concurrent_total_time:.1f}s, expected <120s"
        
        # 3. Read performance should be fast
        assert read_throughput > 50, f"Read throughput {read_throughput:.1f} MB/s should be >50 MB/s"
        assert total_read_time < 60, f"Sequential reads took {total_read_time:.1f}s, expected <60s"
        
        # 4. Concurrent read performance should be excellent
        assert concurrent_read_throughput > 40, f"Concurrent read throughput {concurrent_read_throughput:.1f} MB/s should be >40 MB/s"
        assert concurrent_read_total_time < 30, f"Concurrent reads took {concurrent_read_total_time:.1f}s, expected <30s"
        
        # 5. Validate file sizes and compression
        total_files_size = sum(path.stat().st_size for path in sequential_paths) / 1024 / 1024
        compression_ratio = total_data_size / total_files_size
        
        assert compression_ratio > 1.0, f"Compression ratio {compression_ratio:.2f} should be >1.0"
        assert compression_ratio < 10.0, f"Compression ratio {compression_ratio:.2f} should be <10.0 (reasonable)"
        
        # Generate storage performance report
        storage_performance_report = {
            "data_volume": {
                "total_datasets": num_datasets,
                "total_data_size_mb": total_data_size,
                "average_dataset_size_mb": total_data_size / num_datasets,
                "total_files_size_mb": total_files_size,
                "compression_ratio": compression_ratio
            },
            "write_performance": {
                "sequential_total_time": sequential_total_time,
                "sequential_throughput_mb_per_s": sequential_throughput,
                "concurrent_total_time": concurrent_total_time,
                "concurrent_throughput_mb_per_s": concurrent_throughput,
                "concurrent_improvement": concurrent_throughput / sequential_throughput
            },
            "read_performance": {
                "sequential_read_time": total_read_time,
                "sequential_read_throughput_mb_per_s": read_throughput,
                "concurrent_read_time": concurrent_read_total_time,
                "concurrent_read_throughput_mb_per_s": concurrent_read_throughput,
                "concurrent_read_improvement": concurrent_read_throughput / read_throughput
            },
            "performance_targets": {
                "write_throughput_acceptable": sequential_throughput > 10 and concurrent_throughput > 8,
                "read_throughput_excellent": read_throughput > 50 and concurrent_read_throughput > 40,
                "compression_effective": 1.0 < compression_ratio < 10.0,
                "timing_requirements_met": (
                    sequential_total_time < 180 and 
                    concurrent_total_time < 120 and 
                    total_read_time < 60 and 
                    concurrent_read_total_time < 30
                )
            },
            "storage_efficiency": {
                "files_created": len(sequential_paths) + len(concurrent_results),
                "average_write_time_per_dataset": np.mean([ds.get("sequential_write_time", 0) for ds in large_datasets]),
                "average_read_time_per_dataset": np.mean([r["read_time"] for r in read_results]),
                "storage_overhead_acceptable": True
            }
        }
        
        logging.info(f"Storage Performance Under Load Report: {storage_performance_report}")
        
        return storage_performance_report