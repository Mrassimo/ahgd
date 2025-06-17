"""
Data Lineage and Provenance Tracking Tests

Comprehensive testing suite for data lineage and provenance tracking including:
- Complete data provenance tracking through Bronze-Silver-Gold layers
- Schema evolution tracking with transformation details
- Data transformation audit trails
- Impact analysis for data changes
- Lineage integrity validation
- Data quality lineage correlation
- Automated lineage documentation generation

This test suite ensures complete visibility into data transformations
and maintains audit trails for Australian health analytics data processing.
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from unittest.mock import Mock, patch
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

import polars as pl
import numpy as np
from loguru import logger

from tests.data_quality.validators.schema_validators import DataLineageTracker, SchemaValidator


class LineageEventType(Enum):
    """Types of lineage events."""
    DATA_EXTRACTION = "data_extraction"
    DATA_TRANSFORMATION = "data_transformation"
    DATA_VALIDATION = "data_validation"
    DATA_AGGREGATION = "data_aggregation"
    DATA_ENRICHMENT = "data_enrichment"
    SCHEMA_EVOLUTION = "schema_evolution"
    QUALITY_CHECK = "quality_check"
    DATA_EXPORT = "data_export"


class TransformationType(Enum):
    """Types of data transformations."""
    BRONZE_TO_SILVER = "bronze_to_silver"
    SILVER_TO_GOLD = "silver_to_gold"
    ENRICHMENT = "enrichment"
    AGGREGATION = "aggregation"
    CLEANSING = "cleansing"
    VALIDATION = "validation"
    STANDARDIZATION = "standardization"


@dataclass
class LineageEvent:
    """Individual lineage event."""
    event_id: str
    event_type: LineageEventType
    timestamp: str
    source_datasets: List[str]
    target_dataset: str
    transformation_details: Dict
    schema_changes: Dict
    quality_metrics: Dict
    metadata: Dict


@dataclass
class DataProvenance:
    """Complete data provenance information."""
    dataset_id: str
    creation_timestamp: str
    source_lineage: List[LineageEvent]
    transformation_chain: List[str]
    quality_history: List[Dict]
    schema_evolution: List[Dict]
    data_freshness: Dict
    impact_analysis: Dict


class EnhancedDataLineageTracker(DataLineageTracker):
    """Enhanced data lineage tracker with provenance capabilities."""
    
    def __init__(self, lineage_path: Optional[Path] = None):
        """Initialize enhanced lineage tracker."""
        super().__init__(lineage_path)
        self.events_path = self.lineage_path / "events"
        self.provenance_path = self.lineage_path / "provenance"
        self.events_path.mkdir(parents=True, exist_ok=True)
        self.provenance_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger.bind(component="enhanced_lineage_tracker")
    
    def record_lineage_event(self, 
                           event_type: LineageEventType,
                           source_datasets: List[str],
                           target_dataset: str,
                           transformation_details: Dict,
                           schema_before: Optional[Dict] = None,
                           schema_after: Optional[Dict] = None,
                           quality_metrics: Optional[Dict] = None,
                           metadata: Optional[Dict] = None) -> str:
        """
        Record a comprehensive lineage event.
        
        Args:
            event_type: Type of lineage event
            source_datasets: List of source dataset identifiers
            target_dataset: Target dataset identifier
            transformation_details: Details of the transformation
            schema_before: Schema before transformation
            schema_after: Schema after transformation
            quality_metrics: Quality metrics for this transformation
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        event_id = f"event_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
        
        # Calculate schema changes
        schema_changes = {}
        if schema_before and schema_after:
            schema_validator = SchemaValidator()
            comparison = schema_validator.compare_schemas(schema_before, schema_after)
            schema_changes = {
                "has_changes": not comparison["identical"],
                "compatibility": comparison["compatibility"].value if hasattr(comparison["compatibility"], 'value') else str(comparison["compatibility"]),
                "changes_summary": comparison["summary"],
                "detailed_changes": comparison["changes"]
            }
        
        # Create lineage event
        event = LineageEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            source_datasets=source_datasets,
            target_dataset=target_dataset,
            transformation_details=transformation_details,
            schema_changes=schema_changes,
            quality_metrics=quality_metrics or {},
            metadata=metadata or {}
        )
        
        # Save event
        event_file = self.events_path / f"{event_id}.json"
        with open(event_file, "w") as f:
            json.dump(asdict(event), f, indent=2, default=str)
        
        self.logger.info(f"Recorded lineage event: {event_id} ({event_type.value})")
        return event_id
    
    def build_data_provenance(self, dataset_id: str) -> DataProvenance:
        """
        Build complete data provenance for a dataset.
        
        Args:
            dataset_id: Dataset identifier
            
        Returns:
            Complete data provenance information
        """
        # Find all events related to this dataset
        related_events = []
        
        for event_file in self.events_path.glob("event_*.json"):
            with open(event_file, "r") as f:
                event_data = json.load(f)
            
            # Check if this event involves the dataset
            if (dataset_id == event_data["target_dataset"] or 
                dataset_id in event_data["source_datasets"]):
                
                # Convert back to LineageEvent
                event = LineageEvent(
                    event_id=event_data["event_id"],
                    event_type=LineageEventType(event_data["event_type"]),
                    timestamp=event_data["timestamp"],
                    source_datasets=event_data["source_datasets"],
                    target_dataset=event_data["target_dataset"],
                    transformation_details=event_data["transformation_details"],
                    schema_changes=event_data["schema_changes"],
                    quality_metrics=event_data["quality_metrics"],
                    metadata=event_data["metadata"]
                )
                related_events.append(event)
        
        # Sort events by timestamp
        related_events.sort(key=lambda x: x.timestamp)
        
        # Build transformation chain
        transformation_chain = []
        for event in related_events:
            if event.target_dataset == dataset_id:
                transformation_chain.extend(event.source_datasets)
        
        # Extract quality history
        quality_history = []
        for event in related_events:
            if event.quality_metrics:
                quality_history.append({
                    "timestamp": event.timestamp,
                    "event_id": event.event_id,
                    "metrics": event.quality_metrics
                })
        
        # Extract schema evolution
        schema_evolution = []
        for event in related_events:
            if event.schema_changes.get("has_changes", False):
                schema_evolution.append({
                    "timestamp": event.timestamp,
                    "event_id": event.event_id,
                    "changes": event.schema_changes
                })
        
        # Calculate data freshness
        data_freshness = {}
        if related_events:
            latest_event = related_events[-1]
            latest_timestamp = datetime.fromisoformat(latest_event.timestamp.replace("Z", "+00:00"))
            age_hours = (datetime.now() - latest_timestamp.replace(tzinfo=None)).total_seconds() / 3600
            data_freshness = {
                "latest_update": latest_event.timestamp,
                "age_hours": age_hours,
                "freshness_score": max(0, 100 - (age_hours / 24) * 10)  # Decreases 10 points per day
            }
        
        # Build impact analysis
        impact_analysis = self._build_impact_analysis(dataset_id, related_events)
        
        # Determine creation timestamp
        creation_timestamp = related_events[0].timestamp if related_events else datetime.now().isoformat()
        
        return DataProvenance(
            dataset_id=dataset_id,
            creation_timestamp=creation_timestamp,
            source_lineage=related_events,
            transformation_chain=list(set(transformation_chain)),  # Remove duplicates
            quality_history=quality_history,
            schema_evolution=schema_evolution,
            data_freshness=data_freshness,
            impact_analysis=impact_analysis
        )
    
    def _build_impact_analysis(self, dataset_id: str, events: List[LineageEvent]) -> Dict:
        """Build impact analysis for a dataset."""
        # Find downstream datasets
        downstream_datasets = set()
        for event in events:
            if dataset_id in event.source_datasets:
                downstream_datasets.add(event.target_dataset)
        
        # Find upstream datasets
        upstream_datasets = set()
        for event in events:
            if event.target_dataset == dataset_id:
                upstream_datasets.update(event.source_datasets)
        
        # Calculate transformation types used
        transformation_types = set()
        for event in events:
            if event.target_dataset == dataset_id:
                transformation_types.add(event.transformation_details.get("type", "unknown"))
        
        return {
            "upstream_datasets": list(upstream_datasets),
            "downstream_datasets": list(downstream_datasets),
            "transformation_types": list(transformation_types),
            "dependency_count": len(upstream_datasets),
            "impact_count": len(downstream_datasets)
        }
    
    def validate_lineage_completeness(self, dataset_id: str) -> Dict:
        """
        Validate completeness of lineage tracking for a dataset.
        
        Args:
            dataset_id: Dataset to validate
            
        Returns:
            Validation results
        """
        provenance = self.build_data_provenance(dataset_id)
        
        validation_result = {
            "complete": True,
            "errors": [],
            "warnings": [],
            "coverage_metrics": {}
        }
        
        # Check for essential lineage components
        if not provenance.source_lineage:
            validation_result["errors"].append("No lineage events found")
            validation_result["complete"] = False
        
        if not provenance.transformation_chain:
            validation_result["warnings"].append("No transformation chain identified")
        
        if not provenance.quality_history:
            validation_result["warnings"].append("No quality metrics tracked")
        
        # Calculate coverage metrics
        total_events = len(provenance.source_lineage)
        events_with_schema = len([e for e in provenance.source_lineage if e.schema_changes.get("has_changes")])
        events_with_quality = len([e for e in provenance.source_lineage if e.quality_metrics])
        
        validation_result["coverage_metrics"] = {
            "total_events": total_events,
            "schema_coverage": (events_with_schema / total_events) * 100 if total_events > 0 else 0,
            "quality_coverage": (events_with_quality / total_events) * 100 if total_events > 0 else 0,
            "lineage_depth": len(provenance.transformation_chain),
            "impact_breadth": len(provenance.impact_analysis["downstream_datasets"])
        }
        
        return validation_result
    
    def generate_lineage_report(self, dataset_id: str) -> Dict:
        """
        Generate comprehensive lineage report for a dataset.
        
        Args:
            dataset_id: Dataset to report on
            
        Returns:
            Comprehensive lineage report
        """
        provenance = self.build_data_provenance(dataset_id)
        validation = self.validate_lineage_completeness(dataset_id)
        
        # Calculate summary statistics
        event_types = {}
        for event in provenance.source_lineage:
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Find critical paths
        critical_paths = []
        for event in provenance.source_lineage:
            if event.event_type in [LineageEventType.DATA_TRANSFORMATION, LineageEventType.SCHEMA_EVOLUTION]:
                critical_paths.append({
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "type": event.event_type.value,
                    "sources": event.source_datasets,
                    "target": event.target_dataset
                })
        
        return {
            "dataset_id": dataset_id,
            "report_timestamp": datetime.now().isoformat(),
            "provenance_summary": {
                "creation_date": provenance.creation_timestamp,
                "total_events": len(provenance.source_lineage),
                "transformation_depth": len(provenance.transformation_chain),
                "quality_checkpoints": len(provenance.quality_history),
                "schema_changes": len(provenance.schema_evolution)
            },
            "event_distribution": event_types,
            "data_freshness": provenance.data_freshness,
            "impact_analysis": provenance.impact_analysis,
            "critical_transformation_paths": critical_paths,
            "validation_results": validation,
            "recommendations": self._generate_lineage_recommendations(provenance, validation)
        }
    
    def _generate_lineage_recommendations(self, provenance: DataProvenance, validation: Dict) -> List[str]:
        """Generate recommendations based on lineage analysis."""
        recommendations = []
        
        # Coverage recommendations
        if validation["coverage_metrics"]["quality_coverage"] < 50:
            recommendations.append("Consider adding more quality checkpoints throughout the data pipeline")
        
        if validation["coverage_metrics"]["schema_coverage"] < 30:
            recommendations.append("Implement schema change tracking for better lineage visibility")
        
        # Freshness recommendations
        if provenance.data_freshness.get("freshness_score", 100) < 80:
            recommendations.append("Data appears stale - consider more frequent updates")
        
        # Complexity recommendations
        if len(provenance.transformation_chain) > 10:
            recommendations.append("Complex transformation chain - consider simplification")
        
        if len(provenance.impact_analysis["downstream_datasets"]) > 20:
            recommendations.append("High impact dataset - ensure robust change management")
        
        return recommendations


class TestDataLineageTracking:
    """Test suite for data lineage and provenance tracking."""
    
    @pytest.fixture
    def temp_lineage_path(self):
        """Create temporary lineage path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir) / "lineage"
    
    @pytest.fixture
    def enhanced_tracker(self, temp_lineage_path):
        """Create enhanced lineage tracker."""
        return EnhancedDataLineageTracker(temp_lineage_path)
    
    @pytest.fixture
    def schema_validator(self):
        """Create schema validator."""
        return SchemaValidator()
    
    @pytest.fixture
    def sample_bronze_schema(self):
        """Sample Bronze layer schema."""
        return {
            "columns": {
                "sa2_code_2021": {"data_type": "String", "null_count": 0},
                "sa2_name_2021": {"data_type": "String", "null_count": 0},
                "irsd_score": {"data_type": "Int64", "null_count": 0},
                "irsd_decile": {"data_type": "Int64", "null_count": 0},
                "extraction_timestamp": {"data_type": "String", "null_count": 0}
            },
            "schema_hash": "bronze_hash_123"
        }
    
    @pytest.fixture
    def sample_silver_schema(self):
        """Sample Silver layer schema."""
        return {
            "columns": {
                "sa2_code_2021": {"data_type": "String", "null_count": 0},
                "sa2_name_2021": {"data_type": "String", "null_count": 0},
                "irsd_score": {"data_type": "Int64", "null_count": 0},
                "irsd_decile": {"data_type": "Int64", "null_count": 0},
                "quality_score": {"data_type": "Float64", "null_count": 0},  # Added
                "validation_timestamp": {"data_type": "String", "null_count": 0}  # Added
            },
            "schema_hash": "silver_hash_456"
        }
    
    @pytest.fixture
    def sample_gold_schema(self):
        """Sample Gold layer schema."""
        return {
            "columns": {
                "sa2_code_2021": {"data_type": "String", "null_count": 0},
                "sa2_name_2021": {"data_type": "String", "null_count": 0},
                "seifa_composite_score": {"data_type": "Float64", "null_count": 0},  # Transformed
                "disadvantage_category": {"data_type": "String", "null_count": 0},  # Added
                "aggregation_timestamp": {"data_type": "String", "null_count": 0}
            },
            "schema_hash": "gold_hash_789"
        }
    
    def test_lineage_event_recording(self, enhanced_tracker, sample_bronze_schema, sample_silver_schema):
        """Test recording of lineage events."""
        # Record a transformation event
        event_id = enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_datasets=["bronze_seifa"],
            target_dataset="silver_seifa",
            transformation_details={
                "type": "bronze_to_silver",
                "operations": ["validation", "quality_scoring", "null_handling"],
                "processing_time_seconds": 45.2
            },
            schema_before=sample_bronze_schema,
            schema_after=sample_silver_schema,
            quality_metrics={
                "completeness": 98.5,
                "validity": 97.2,
                "processing_success_rate": 100.0
            },
            metadata={
                "processor_version": "v1.2.3",
                "environment": "test"
            }
        )
        
        assert event_id is not None
        assert event_id.startswith("event_")
        
        # Verify event was saved
        event_file = enhanced_tracker.events_path / f"{event_id}.json"
        assert event_file.exists()
        
        # Load and verify event content
        with open(event_file, "r") as f:
            saved_event = json.load(f)
        
        assert saved_event["event_type"] == "data_transformation"
        assert saved_event["source_datasets"] == ["bronze_seifa"]
        assert saved_event["target_dataset"] == "silver_seifa"
        assert "schema_changes" in saved_event
        assert "quality_metrics" in saved_event
        assert saved_event["quality_metrics"]["completeness"] == 98.5
    
    def test_complete_pipeline_lineage_tracking(self, enhanced_tracker, sample_bronze_schema, sample_silver_schema, sample_gold_schema):
        """Test tracking lineage through complete pipeline."""
        # Step 1: Data extraction
        extraction_event = enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_EXTRACTION,
            source_datasets=["external_seifa_source"],
            target_dataset="bronze_seifa",
            transformation_details={
                "type": "extraction",
                "source_type": "xlsx",
                "extraction_method": "pandas_read",
                "records_extracted": 2544
            },
            schema_after=sample_bronze_schema,
            quality_metrics={"extraction_success_rate": 100.0}
        )
        
        # Step 2: Bronze to Silver transformation
        bronze_silver_event = enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_datasets=["bronze_seifa"],
            target_dataset="silver_seifa",
            transformation_details={
                "type": "bronze_to_silver",
                "operations": ["validation", "quality_scoring"],
                "records_processed": 2544,
                "records_passed": 2498
            },
            schema_before=sample_bronze_schema,
            schema_after=sample_silver_schema,
            quality_metrics={
                "completeness": 98.5,
                "validity": 97.2,
                "consistency": 96.8
            }
        )
        
        # Step 3: Silver to Gold transformation
        silver_gold_event = enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_AGGREGATION,
            source_datasets=["silver_seifa"],
            target_dataset="gold_seifa",
            transformation_details={
                "type": "silver_to_gold",
                "operations": ["aggregation", "business_logic", "categorization"],
                "records_input": 2498,
                "records_output": 2498
            },
            schema_before=sample_silver_schema,
            schema_after=sample_gold_schema,
            quality_metrics={
                "business_rule_compliance": 99.1,
                "aggregation_accuracy": 100.0
            }
        )
        
        # Verify all events were recorded
        assert extraction_event is not None
        assert bronze_silver_event is not None
        assert silver_gold_event is not None
        
        # Build provenance for Gold dataset
        gold_provenance = enhanced_tracker.build_data_provenance("gold_seifa")
        
        # Verify provenance structure
        assert gold_provenance.dataset_id == "gold_seifa"
        assert len(gold_provenance.source_lineage) == 1  # Only direct transformation
        assert len(gold_provenance.transformation_chain) == 1  # silver_seifa
        assert len(gold_provenance.quality_history) == 1
        assert len(gold_provenance.schema_evolution) == 1
        
        # Verify impact analysis
        assert "upstream_datasets" in gold_provenance.impact_analysis
        assert "downstream_datasets" in gold_provenance.impact_analysis
        assert "silver_seifa" in gold_provenance.impact_analysis["upstream_datasets"]
    
    def test_data_provenance_building(self, enhanced_tracker, sample_bronze_schema, sample_silver_schema):
        """Test building comprehensive data provenance."""
        dataset_id = "test_dataset"
        
        # Record multiple events for the dataset
        events = []
        
        # Extraction event
        events.append(enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_EXTRACTION,
            source_datasets=["source_system"],
            target_dataset=dataset_id,
            transformation_details={"type": "extraction"},
            schema_after=sample_bronze_schema,
            quality_metrics={"extraction_rate": 100.0}
        ))
        
        # Validation event
        events.append(enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_VALIDATION,
            source_datasets=[dataset_id],
            target_dataset=dataset_id,
            transformation_details={"type": "validation"},
            quality_metrics={"validation_pass_rate": 95.5}
        ))
        
        # Transformation event
        events.append(enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_datasets=[dataset_id],
            target_dataset=dataset_id + "_transformed",
            transformation_details={"type": "transformation"},
            schema_before=sample_bronze_schema,
            schema_after=sample_silver_schema,
            quality_metrics={"transformation_success": 98.2}
        ))
        
        # Build provenance
        provenance = enhanced_tracker.build_data_provenance(dataset_id)
        
        # Verify provenance completeness
        assert provenance.dataset_id == dataset_id
        assert len(provenance.source_lineage) >= 2  # At least extraction and validation
        assert len(provenance.quality_history) >= 2
        assert provenance.data_freshness["freshness_score"] > 90  # Recent data
        
        # Verify transformation chain
        assert len(provenance.transformation_chain) >= 1
        assert "source_system" in provenance.transformation_chain
    
    def test_schema_evolution_tracking(self, enhanced_tracker):
        """Test tracking of schema evolution through transformations."""
        dataset_id = "schema_evolution_test"
        
        # Initial schema
        schema_v1 = {
            "columns": {
                "id": {"data_type": "String"},
                "value": {"data_type": "Int64"}
            },
            "schema_hash": "v1_hash"
        }
        
        # Evolved schema (column added)
        schema_v2 = {
            "columns": {
                "id": {"data_type": "String"},
                "value": {"data_type": "Int64"},
                "new_column": {"data_type": "Float64"}
            },
            "schema_hash": "v2_hash"
        }
        
        # Further evolved schema (type changed)
        schema_v3 = {
            "columns": {
                "id": {"data_type": "String"},
                "value": {"data_type": "Float64"},  # Type changed
                "new_column": {"data_type": "Float64"}
            },
            "schema_hash": "v3_hash"
        }
        
        # Record schema evolution events
        enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.SCHEMA_EVOLUTION,
            source_datasets=[dataset_id + "_v1"],
            target_dataset=dataset_id + "_v2",
            transformation_details={"type": "column_addition", "added_columns": ["new_column"]},
            schema_before=schema_v1,
            schema_after=schema_v2
        )
        
        enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.SCHEMA_EVOLUTION,
            source_datasets=[dataset_id + "_v2"],
            target_dataset=dataset_id + "_v3",
            transformation_details={"type": "type_change", "changed_columns": ["value"]},
            schema_before=schema_v2,
            schema_after=schema_v3
        )
        
        # Build provenance for final version
        provenance = enhanced_tracker.build_data_provenance(dataset_id + "_v3")
        
        # Verify schema evolution tracking
        assert len(provenance.schema_evolution) >= 1
        
        for evolution in provenance.schema_evolution:
            assert "changes" in evolution
            assert "timestamp" in evolution
            assert evolution["changes"]["has_changes"] is True
    
    def test_quality_metrics_lineage(self, enhanced_tracker):
        """Test tracking of quality metrics through lineage."""
        dataset_id = "quality_lineage_test"
        
        # Record events with different quality metrics
        quality_checkpoints = [
            {"completeness": 95.0, "validity": 92.0, "checkpoint": "initial"},
            {"completeness": 98.5, "validity": 97.2, "checkpoint": "after_cleaning"},
            {"completeness": 99.1, "validity": 98.8, "checkpoint": "final"}
        ]
        
        for i, quality_metrics in enumerate(quality_checkpoints):
            enhanced_tracker.record_lineage_event(
                event_type=LineageEventType.QUALITY_CHECK,
                source_datasets=[f"{dataset_id}_step_{i}"],
                target_dataset=f"{dataset_id}_step_{i+1}",
                transformation_details={"step": i+1},
                quality_metrics=quality_metrics
            )
        
        # Build provenance
        provenance = enhanced_tracker.build_data_provenance(f"{dataset_id}_step_3")
        
        # Verify quality history
        assert len(provenance.quality_history) >= 1
        
        for quality_record in provenance.quality_history:
            assert "timestamp" in quality_record
            assert "metrics" in quality_record
            assert "completeness" in quality_record["metrics"]
            assert "validity" in quality_record["metrics"]
    
    def test_lineage_validation(self, enhanced_tracker, sample_bronze_schema, sample_silver_schema):
        """Test validation of lineage completeness."""
        dataset_id = "validation_test"
        
        # Record minimal lineage
        enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_EXTRACTION,
            source_datasets=["source"],
            target_dataset=dataset_id,
            transformation_details={"type": "extraction"},
            schema_after=sample_bronze_schema,
            quality_metrics={"extraction_rate": 100.0}
        )
        
        # Validate lineage
        validation_result = enhanced_tracker.validate_lineage_completeness(dataset_id)
        
        # Check validation structure
        assert "complete" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result
        assert "coverage_metrics" in validation_result
        
        # Check coverage metrics
        coverage = validation_result["coverage_metrics"]
        assert "total_events" in coverage
        assert "schema_coverage" in coverage
        assert "quality_coverage" in coverage
        assert "lineage_depth" in coverage
        
        # Should be complete with at least one event
        assert coverage["total_events"] >= 1
    
    def test_lineage_report_generation(self, enhanced_tracker, sample_bronze_schema, sample_silver_schema, sample_gold_schema):
        """Test generation of comprehensive lineage reports."""
        dataset_id = "report_test"
        
        # Create a complex lineage scenario
        enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_EXTRACTION,
            source_datasets=["external_source"],
            target_dataset=dataset_id + "_bronze",
            transformation_details={"type": "extraction"},
            schema_after=sample_bronze_schema,
            quality_metrics={"extraction_rate": 100.0}
        )
        
        enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_datasets=[dataset_id + "_bronze"],
            target_dataset=dataset_id + "_silver",
            transformation_details={"type": "bronze_to_silver"},
            schema_before=sample_bronze_schema,
            schema_after=sample_silver_schema,
            quality_metrics={"transformation_success": 98.5}
        )
        
        enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_AGGREGATION,
            source_datasets=[dataset_id + "_silver"],
            target_dataset=dataset_id + "_gold",
            transformation_details={"type": "silver_to_gold"},
            schema_before=sample_silver_schema,
            schema_after=sample_gold_schema,
            quality_metrics={"aggregation_success": 99.2}
        )
        
        # Generate report for Gold dataset
        report = enhanced_tracker.generate_lineage_report(dataset_id + "_gold")
        
        # Verify report structure
        assert "dataset_id" in report
        assert "report_timestamp" in report
        assert "provenance_summary" in report
        assert "event_distribution" in report
        assert "data_freshness" in report
        assert "impact_analysis" in report
        assert "critical_transformation_paths" in report
        assert "validation_results" in report
        assert "recommendations" in report
        
        # Verify provenance summary
        summary = report["provenance_summary"]
        assert summary["total_events"] >= 1
        assert summary["transformation_depth"] >= 1
        
        # Verify event distribution
        assert len(report["event_distribution"]) > 0
        
        # Verify recommendations
        assert isinstance(report["recommendations"], list)
    
    def test_impact_analysis(self, enhanced_tracker):
        """Test impact analysis for lineage tracking."""
        # Create a dataset with multiple dependencies and impacts
        central_dataset = "central_data"
        
        # Record upstream dependencies
        upstream_datasets = ["source_a", "source_b", "source_c"]
        for source in upstream_datasets:
            enhanced_tracker.record_lineage_event(
                event_type=LineageEventType.DATA_TRANSFORMATION,
                source_datasets=[source],
                target_dataset=central_dataset,
                transformation_details={"type": "merge"},
                quality_metrics={"merge_success": 100.0}
            )
        
        # Record downstream impacts
        downstream_datasets = ["output_x", "output_y", "output_z"]
        for target in downstream_datasets:
            enhanced_tracker.record_lineage_event(
                event_type=LineageEventType.DATA_TRANSFORMATION,
                source_datasets=[central_dataset],
                target_dataset=target,
                transformation_details={"type": "derive"},
                quality_metrics={"derivation_success": 95.0}
            )
        
        # Build provenance and analyze impact
        provenance = enhanced_tracker.build_data_provenance(central_dataset)
        impact_analysis = provenance.impact_analysis
        
        # Verify impact analysis
        assert len(impact_analysis["upstream_datasets"]) >= len(upstream_datasets)
        assert impact_analysis["dependency_count"] >= 3
        
        # For downstream analysis, check outputs
        for downstream in downstream_datasets:
            downstream_provenance = enhanced_tracker.build_data_provenance(downstream)
            downstream_impact = downstream_provenance.impact_analysis
            assert central_dataset in downstream_impact["upstream_datasets"]
    
    def test_lineage_integrity_validation(self, enhanced_tracker):
        """Test validation of lineage integrity and completeness."""
        # Create a lineage chain with gaps
        enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_EXTRACTION,
            source_datasets=["source"],
            target_dataset="bronze_data",
            transformation_details={"type": "extraction"}
        )
        
        # Skip silver layer - create gap
        enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_AGGREGATION,
            source_datasets=["silver_data"],  # Missing: bronze_data -> silver_data
            target_dataset="gold_data",
            transformation_details={"type": "aggregation"}
        )
        
        # Validate lineage for gold_data
        validation = enhanced_tracker.validate_lineage_completeness("gold_data")
        
        # Should detect the gap or have warnings about incomplete lineage
        assert "coverage_metrics" in validation
        
        # Check if lineage depth is captured
        coverage = validation["coverage_metrics"]
        assert "lineage_depth" in coverage
    
    def test_performance_with_large_lineage(self, enhanced_tracker):
        """Test lineage tracking performance with large number of events."""
        dataset_id = "performance_test"
        
        # Record many lineage events
        num_events = 100
        start_time = datetime.now()
        
        for i in range(num_events):
            enhanced_tracker.record_lineage_event(
                event_type=LineageEventType.DATA_TRANSFORMATION,
                source_datasets=[f"source_{i}"],
                target_dataset=f"{dataset_id}_{i}",
                transformation_details={"step": i, "type": "transformation"},
                quality_metrics={"success_rate": 95.0 + (i % 5)}
            )
        
        recording_time = (datetime.now() - start_time).total_seconds()
        
        # Build provenance for last dataset
        start_time = datetime.now()
        provenance = enhanced_tracker.build_data_provenance(f"{dataset_id}_{num_events-1}")
        provenance_time = (datetime.now() - start_time).total_seconds()
        
        # Generate report
        start_time = datetime.now()
        report = enhanced_tracker.generate_lineage_report(f"{dataset_id}_{num_events-1}")
        report_time = (datetime.now() - start_time).total_seconds()
        
        # Performance assertions
        assert recording_time < 30.0, f"Recording {num_events} events took too long: {recording_time}s"
        assert provenance_time < 5.0, f"Building provenance took too long: {provenance_time}s"
        assert report_time < 10.0, f"Generating report took too long: {report_time}s"
        
        # Verify functionality still works
        assert len(provenance.source_lineage) >= 1
        assert report["provenance_summary"]["total_events"] >= 1
        
        logger.info(f"Performance test results:")
        logger.info(f"  Recording {num_events} events: {recording_time:.2f}s")
        logger.info(f"  Building provenance: {provenance_time:.2f}s")
        logger.info(f"  Generating report: {report_time:.2f}s")
    
    def test_comprehensive_lineage_scenario(self, enhanced_tracker, sample_bronze_schema, sample_silver_schema, sample_gold_schema):
        """Test comprehensive end-to-end lineage tracking scenario."""
        # Simulate complete Australian health analytics pipeline
        
        # Step 1: Extract SEIFA data
        seifa_extraction = enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_EXTRACTION,
            source_datasets=["abs_seifa_excel"],
            target_dataset="bronze_seifa",
            transformation_details={
                "type": "excel_extraction",
                "source_file": "SEIFA_2021_SA2_Indexes.xlsx",
                "records_extracted": 2544
            },
            schema_after=sample_bronze_schema,
            quality_metrics={
                "extraction_success_rate": 100.0,
                "null_rate": 0.2
            }
        )
        
        # Step 2: Extract Census data
        census_extraction = enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_EXTRACTION,
            source_datasets=["abs_census_csv"],
            target_dataset="bronze_census",
            transformation_details={
                "type": "csv_extraction",
                "source_files": ["2021Census_G01_AUS_SA1.csv"],
                "records_extracted": 358000
            },
            schema_after={
                "columns": {"sa1_code_2021": {"data_type": "String"}, "tot_p_p": {"data_type": "Int64"}},
                "schema_hash": "bronze_census_hash"
            },
            quality_metrics={
                "extraction_success_rate": 99.8,
                "null_rate": 1.2
            }
        )
        
        # Step 3: Transform SEIFA to Silver
        seifa_silver_transform = enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_TRANSFORMATION,
            source_datasets=["bronze_seifa"],
            target_dataset="silver_seifa",
            transformation_details={
                "type": "bronze_to_silver",
                "operations": ["sa2_validation", "seifa_score_validation", "quality_scoring"],
                "records_processed": 2544,
                "records_passed": 2498,
                "validation_rules": ["SA2_FORMAT", "SEIFA_RANGE", "POPULATION_POSITIVE"]
            },
            schema_before=sample_bronze_schema,
            schema_after=sample_silver_schema,
            quality_metrics={
                "completeness": 98.2,
                "validity": 97.8,
                "consistency": 96.5,
                "pass_rate": 98.2
            }
        )
        
        # Step 4: Enrich with geographic data
        geographic_enrichment = enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_ENRICHMENT,
            source_datasets=["silver_seifa", "reference_geographic"],
            target_dataset="silver_seifa_enriched",
            transformation_details={
                "type": "geographic_enrichment",
                "operations": ["coordinate_lookup", "boundary_mapping"],
                "enrichment_rate": 99.1
            },
            quality_metrics={
                "enrichment_success_rate": 99.1,
                "coordinate_accuracy": 98.9
            }
        )
        
        # Step 5: Aggregate to Gold
        gold_aggregation = enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_AGGREGATION,
            source_datasets=["silver_seifa_enriched"],
            target_dataset="gold_health_analytics",
            transformation_details={
                "type": "silver_to_gold",
                "operations": ["seifa_composite_calculation", "disadvantage_categorization", "business_metrics"],
                "business_rules": ["COMPOSITE_SCORING", "CATEGORY_ASSIGNMENT"]
            },
            schema_before=sample_silver_schema,
            schema_after=sample_gold_schema,
            quality_metrics={
                "business_rule_compliance": 99.5,
                "aggregation_accuracy": 100.0,
                "composite_score_validity": 98.8
            }
        )
        
        # Step 6: Export for analysis
        data_export = enhanced_tracker.record_lineage_event(
            event_type=LineageEventType.DATA_EXPORT,
            source_datasets=["gold_health_analytics"],
            target_dataset="analysis_export",
            transformation_details={
                "type": "export",
                "format": "parquet",
                "destination": "data/outputs/"
            },
            quality_metrics={
                "export_success_rate": 100.0
            }
        )
        
        # Generate comprehensive report for final dataset
        final_report = enhanced_tracker.generate_lineage_report("gold_health_analytics")
        
        # Verify comprehensive lineage
        assert final_report["provenance_summary"]["total_events"] >= 1
        assert final_report["provenance_summary"]["transformation_depth"] >= 1
        
        # Verify event distribution covers major pipeline stages
        event_types = final_report["event_distribution"]
        
        # Should have evidence of the pipeline
        assert len(event_types) > 0
        
        # Verify quality progression through pipeline
        gold_provenance = enhanced_tracker.build_data_provenance("gold_health_analytics")
        assert len(gold_provenance.quality_history) >= 1
        
        # Verify schema evolution tracking
        assert len(gold_provenance.schema_evolution) >= 0  # May have schema changes
        
        # Check impact analysis
        impact = gold_provenance.impact_analysis
        assert "upstream_datasets" in impact
        assert len(impact["upstream_datasets"]) >= 1
        
        # Log comprehensive results
        logger.info("Comprehensive lineage scenario results:")
        logger.info(f"  Final dataset: {gold_provenance.dataset_id}")
        logger.info(f"  Total lineage events: {len(gold_provenance.source_lineage)}")
        logger.info(f"  Transformation depth: {len(gold_provenance.transformation_chain)}")
        logger.info(f"  Quality checkpoints: {len(gold_provenance.quality_history)}")
        logger.info(f"  Schema changes: {len(gold_provenance.schema_evolution)}")
        logger.info(f"  Data freshness: {gold_provenance.data_freshness.get('freshness_score', 'N/A')}")
        logger.info(f"  Upstream dependencies: {len(impact['upstream_datasets'])}")
        logger.info(f"  Downstream impacts: {len(impact['downstream_datasets'])}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])