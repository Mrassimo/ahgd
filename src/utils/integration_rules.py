"""
Data integration rules and business logic for AHGD project.

This module defines business rules for data integration decisions,
conflict resolution strategies, quality-based selection, and temporal alignment.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field

from ..utils.logging import get_logger
from schemas.base_schema import DataQualityLevel


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicting data."""
    HIGHEST_QUALITY = "highest_quality"
    MOST_RECENT = "most_recent"
    HIGHEST_PRIORITY = "highest_priority"
    WEIGHTED_AVERAGE = "weighted_average"
    CONSENSUS = "consensus"
    MANUAL = "manual"


class TemporalAlignmentStrategy(str, Enum):
    """Strategies for aligning temporal data."""
    EXACT_MATCH = "exact_match"
    NEAREST_AVAILABLE = "nearest_available"
    INTERPOLATION = "interpolation"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    SEASONAL_ADJUSTMENT = "seasonal_adjustment"


@dataclass
class IntegrationRule:
    """Definition of a data integration rule."""
    
    rule_name: str
    rule_type: str  # validation, transformation, aggregation, filtering
    source_fields: List[str]
    target_field: str
    condition: Optional[str] = None
    transformation: Optional[str] = None
    priority: int = 100
    enabled: bool = True
    description: str = ""
    
    def applies_to(self, data: Dict[str, Any]) -> bool:
        """Check if this rule applies to given data."""
        # Check if all required source fields are present
        for field in self.source_fields:
            if field not in data or data[field] is None:
                return False
        
        # Evaluate condition if present
        if self.condition:
            try:
                # Safe evaluation with limited scope
                return eval(self.condition, {"__builtins__": {}}, data)
            except Exception:
                return False
        
        return True


@dataclass
class DataConflict:
    """Record of a data conflict between sources."""
    
    field_name: str
    sa2_code: str
    conflicting_values: Dict[str, Any]
    source_qualities: Dict[str, float]
    source_timestamps: Dict[str, datetime]
    resolution_strategy: ConflictResolutionStrategy
    resolved_value: Any = None
    resolution_reason: str = ""
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DataIntegrationRules:
    """
    Business rules engine for data integration decisions.
    
    Manages rules for field selection, priority ordering, and validation
    during the data integration process.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the data integration rules engine.
        
        Args:
            config: Configuration including rule definitions
            logger: Optional logger instance
        """
        self.logger = get_logger(__name__) if logger is None else logger
        
        # Load integration rules
        self.rules = self._load_integration_rules(config.get('rules', []))
        
        # Field-specific rules
        self.field_priorities = config.get('field_priorities', {})
        self.mandatory_fields = config.get('mandatory_fields', [])
        self.quality_thresholds = config.get('quality_thresholds', {})
        
        # Source priorities (default ordering)
        self.source_priorities = config.get('source_priorities', {
            'census': 1,
            'seifa': 2,
            'health_indicators': 3,
            'geographic_boundaries': 4,
            'medicare_pbs': 5,
            'environmental': 6
        })
        
        # Temporal alignment configuration
        self.temporal_tolerance_days = config.get('temporal_tolerance_days', 365)
        self.reference_date = config.get('reference_date', datetime.now())
        
        # Conflict resolution configuration
        self.default_resolution_strategy = ConflictResolutionStrategy(
            config.get('default_resolution_strategy', 'highest_quality')
        )
        self.field_resolution_strategies = config.get('field_resolution_strategies', {})
        
    def evaluate_integration_rules(
        self, 
        sa2_code: str, 
        source_data: Dict[str, Dict[str, Any]]
    ) -> List[IntegrationRule]:
        """
        Evaluate which integration rules apply to given data.
        
        Args:
            sa2_code: SA2 identifier
            source_data: Dictionary of source data by source name
            
        Returns:
            List of applicable integration rules
        """
        applicable_rules = []
        
        # Flatten source data for rule evaluation
        flattened_data = {'sa2_code': sa2_code}
        for source, data in source_data.items():
            for field, value in data.items():
                flattened_data[f"{source}_{field}"] = value
        
        # Check each rule
        for rule in self.rules:
            if rule.enabled and rule.applies_to(flattened_data):
                applicable_rules.append(rule)
        
        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        return applicable_rules
    
    def get_field_priority(self, field_name: str, source_name: str) -> int:
        """
        Get the priority for a specific field from a specific source.
        
        Args:
            field_name: Name of the field
            source_name: Name of the data source
            
        Returns:
            Priority value (lower number = higher priority)
        """
        # Check field-specific priorities
        if field_name in self.field_priorities:
            field_config = self.field_priorities[field_name]
            if isinstance(field_config, dict) and 'sources' in field_config:
                source_order = field_config['sources']
                if source_name in source_order:
                    return source_order.index(source_name)
        
        # Fall back to general source priorities
        return self.source_priorities.get(source_name, 999)
    
    def is_mandatory_field(self, field_name: str) -> bool:
        """Check if a field is mandatory for integration."""
        return field_name in self.mandatory_fields
    
    def get_quality_threshold(self, field_name: str) -> float:
        """Get the minimum quality threshold for a field."""
        if field_name in self.quality_thresholds:
            return self.quality_thresholds[field_name]
        
        # Default thresholds by field category
        if 'mortality' in field_name:
            return 0.95  # High quality required for mortality data
        elif 'population' in field_name:
            return 0.90  # High quality for population data
        elif 'prevalence' in field_name:
            return 0.85  # Moderate quality for prevalence
        else:
            return 0.80  # Default threshold
    
    def validate_integration_completeness(
        self, 
        integrated_record: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that an integrated record meets completeness requirements.
        
        Args:
            integrated_record: The integrated data record
            
        Returns:
            Tuple of (is_valid, list_of_missing_fields)
        """
        missing_fields = []
        
        for field in self.mandatory_fields:
            if field not in integrated_record or integrated_record[field] is None:
                missing_fields.append(field)
        
        is_valid = len(missing_fields) == 0
        
        return is_valid, missing_fields
    
    def _load_integration_rules(self, rule_configs: List[Dict[str, Any]]) -> List[IntegrationRule]:
        """Load integration rules from configuration."""
        rules = []
        
        for config in rule_configs:
            rule = IntegrationRule(
                rule_name=config['name'],
                rule_type=config['type'],
                source_fields=config['source_fields'],
                target_field=config['target_field'],
                condition=config.get('condition'),
                transformation=config.get('transformation'),
                priority=config.get('priority', 100),
                enabled=config.get('enabled', True),
                description=config.get('description', '')
            )
            rules.append(rule)
        
        return rules


class ConflictResolver:
    """
    Handles conflicting data from multiple sources.
    
    Implements various strategies for resolving conflicts when the same
    field has different values from different sources.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the conflict resolver.
        
        Args:
            config: Configuration for conflict resolution
            logger: Optional logger instance
        """
        self.logger = get_logger(__name__) if logger is None else logger
        
        # Resolution strategies
        self.default_strategy = ConflictResolutionStrategy(
            config.get('default_strategy', 'highest_quality')
        )
        self.field_strategies = config.get('field_strategies', {})
        
        # Consensus thresholds
        self.consensus_threshold = config.get('consensus_threshold', 0.7)
        self.minimum_sources_for_consensus = config.get('minimum_sources_for_consensus', 3)
        
        # Quality weights for weighted averaging
        self.quality_weight_power = config.get('quality_weight_power', 2.0)
        
    def resolve_conflict(
        self,
        field_name: str,
        sa2_code: str,
        conflicting_data: Dict[str, Tuple[Any, float, datetime]]
    ) -> DataConflict:
        """
        Resolve a conflict for a specific field.
        
        Args:
            field_name: Name of the conflicting field
            sa2_code: SA2 identifier
            conflicting_data: Dict of source_name -> (value, quality, timestamp)
            
        Returns:
            DataConflict record with resolution
        """
        # Extract components
        conflicting_values = {source: data[0] for source, data in conflicting_data.items()}
        source_qualities = {source: data[1] for source, data in conflicting_data.items()}
        source_timestamps = {source: data[2] for source, data in conflicting_data.items()}
        
        # Determine resolution strategy
        strategy = self._get_resolution_strategy(field_name)
        
        # Create conflict record
        conflict = DataConflict(
            field_name=field_name,
            sa2_code=sa2_code,
            conflicting_values=conflicting_values,
            source_qualities=source_qualities,
            source_timestamps=source_timestamps,
            resolution_strategy=strategy
        )
        
        # Resolve based on strategy
        if strategy == ConflictResolutionStrategy.HIGHEST_QUALITY:
            self._resolve_by_quality(conflict)
        elif strategy == ConflictResolutionStrategy.MOST_RECENT:
            self._resolve_by_recency(conflict)
        elif strategy == ConflictResolutionStrategy.HIGHEST_PRIORITY:
            self._resolve_by_priority(conflict)
        elif strategy == ConflictResolutionStrategy.WEIGHTED_AVERAGE:
            self._resolve_by_weighted_average(conflict)
        elif strategy == ConflictResolutionStrategy.CONSENSUS:
            self._resolve_by_consensus(conflict)
        else:
            self._mark_for_manual_resolution(conflict)
        
        return conflict
    
    def _get_resolution_strategy(self, field_name: str) -> ConflictResolutionStrategy:
        """Get the resolution strategy for a specific field."""
        if field_name in self.field_strategies:
            return ConflictResolutionStrategy(self.field_strategies[field_name])
        
        # Default strategies by field type
        if any(term in field_name for term in ['rate', 'percentage', 'prevalence']):
            return ConflictResolutionStrategy.WEIGHTED_AVERAGE
        elif any(term in field_name for term in ['code', 'name', 'category']):
            return ConflictResolutionStrategy.CONSENSUS
        else:
            return self.default_strategy
    
    def _resolve_by_quality(self, conflict: DataConflict) -> None:
        """Resolve conflict by selecting highest quality source."""
        best_source = max(conflict.source_qualities.items(), key=lambda x: x[1])[0]
        conflict.resolved_value = conflict.conflicting_values[best_source]
        conflict.resolution_reason = f"Selected {best_source} with highest quality score {conflict.source_qualities[best_source]:.3f}"
        conflict.confidence_score = conflict.source_qualities[best_source]
    
    def _resolve_by_recency(self, conflict: DataConflict) -> None:
        """Resolve conflict by selecting most recent data."""
        most_recent_source = max(conflict.source_timestamps.items(), key=lambda x: x[1])[0]
        conflict.resolved_value = conflict.conflicting_values[most_recent_source]
        conflict.resolution_reason = f"Selected {most_recent_source} with most recent timestamp {conflict.source_timestamps[most_recent_source]}"
        
        # Confidence based on data age
        age_days = (datetime.utcnow() - conflict.source_timestamps[most_recent_source]).days
        conflict.confidence_score = max(0.5, 1.0 - (age_days / 365.0))
    
    def _resolve_by_priority(self, conflict: DataConflict) -> None:
        """Resolve conflict by source priority ordering."""
        # This would use external priority configuration
        # For now, use quality as proxy for priority
        self._resolve_by_quality(conflict)
        conflict.resolution_reason = conflict.resolution_reason.replace("quality", "priority")
    
    def _resolve_by_weighted_average(self, conflict: DataConflict) -> None:
        """Resolve numeric conflicts by quality-weighted averaging."""
        # Check if all values are numeric
        numeric_values = {}
        for source, value in conflict.conflicting_values.items():
            try:
                numeric_values[source] = float(value)
            except (TypeError, ValueError):
                # Fall back to quality-based resolution
                self._resolve_by_quality(conflict)
                return
        
        # Calculate weighted average
        weighted_sum = 0
        weight_sum = 0
        
        for source, value in numeric_values.items():
            quality = conflict.source_qualities[source]
            weight = quality ** self.quality_weight_power
            weighted_sum += value * weight
            weight_sum += weight
        
        conflict.resolved_value = weighted_sum / weight_sum if weight_sum > 0 else 0
        conflict.resolution_reason = f"Weighted average of {len(numeric_values)} sources"
        conflict.confidence_score = sum(conflict.source_qualities.values()) / len(conflict.source_qualities)
    
    def _resolve_by_consensus(self, conflict: DataConflict) -> None:
        """Resolve by finding consensus among sources."""
        # Count occurrences of each value
        value_counts = {}
        for value in conflict.conflicting_values.values():
            value_str = str(value)
            value_counts[value_str] = value_counts.get(value_str, 0) + 1
        
        # Check if there's a clear consensus
        total_sources = len(conflict.conflicting_values)
        if total_sources >= self.minimum_sources_for_consensus:
            for value_str, count in value_counts.items():
                if count / total_sources >= self.consensus_threshold:
                    # Find original value (not string representation)
                    for source, orig_value in conflict.conflicting_values.items():
                        if str(orig_value) == value_str:
                            conflict.resolved_value = orig_value
                            break
                    
                    conflict.resolution_reason = f"Consensus among {count}/{total_sources} sources"
                    conflict.confidence_score = count / total_sources
                    return
        
        # No consensus found, fall back to quality
        self._resolve_by_quality(conflict)
        conflict.resolution_reason += " (no consensus found)"
    
    def _mark_for_manual_resolution(self, conflict: DataConflict) -> None:
        """Mark conflict for manual resolution."""
        conflict.resolved_value = None
        conflict.resolution_reason = "Requires manual resolution"
        conflict.confidence_score = 0.0


class QualityBasedSelector:
    """
    Selects best data based on quality scores.
    
    Implements sophisticated quality assessment and selection logic
    for choosing between multiple data sources.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the quality-based selector.
        
        Args:
            config: Configuration for quality assessment
            logger: Optional logger instance
        """
        self.logger = get_logger(__name__) if logger is None else logger
        
        # Quality dimensions and weights
        self.quality_dimensions = config.get('quality_dimensions', {
            'completeness': 0.3,
            'accuracy': 0.3,
            'timeliness': 0.2,
            'consistency': 0.1,
            'reliability': 0.1
        })
        
        # Source reliability scores
        self.source_reliability = config.get('source_reliability', {
            'census': 0.95,
            'seifa': 0.98,
            'health_indicators': 0.90,
            'geographic_boundaries': 0.99,
            'medicare_pbs': 0.85,
            'environmental': 0.80
        })
        
        # Timeliness decay parameters
        self.timeliness_half_life_days = config.get('timeliness_half_life_days', 365)
        
    def select_best_source(
        self,
        field_name: str,
        candidate_sources: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, float]:
        """
        Select the best source for a specific field.
        
        Args:
            field_name: Name of the field
            candidate_sources: Dict of source_name -> source_data
            
        Returns:
            Tuple of (selected_source_name, quality_score)
        """
        source_scores = {}
        
        for source_name, source_data in candidate_sources.items():
            if field_name in source_data and source_data[field_name] is not None:
                quality_score = self._calculate_quality_score(
                    source_name, 
                    source_data, 
                    field_name
                )
                source_scores[source_name] = quality_score
        
        if not source_scores:
            return None, 0.0
        
        # Select highest scoring source
        best_source = max(source_scores.items(), key=lambda x: x[1])
        return best_source
    
    def _calculate_quality_score(
        self,
        source_name: str,
        source_data: Dict[str, Any],
        field_name: str
    ) -> float:
        """Calculate comprehensive quality score for a data source."""
        scores = {}
        
        # Completeness score
        scores['completeness'] = self._calculate_completeness(source_data, field_name)
        
        # Accuracy score (based on validation rules)
        scores['accuracy'] = self._calculate_accuracy(source_data, field_name)
        
        # Timeliness score
        scores['timeliness'] = self._calculate_timeliness(source_data)
        
        # Consistency score
        scores['consistency'] = self._calculate_consistency(source_data, field_name)
        
        # Reliability score (source-based)
        scores['reliability'] = self.source_reliability.get(source_name, 0.5)
        
        # Calculate weighted overall score
        overall_score = 0.0
        for dimension, weight in self.quality_dimensions.items():
            if dimension in scores:
                overall_score += scores[dimension] * weight
        
        return overall_score
    
    def _calculate_completeness(self, source_data: Dict[str, Any], field_name: str) -> float:
        """Calculate completeness score for a field."""
        # Check if field exists and is not null
        if field_name not in source_data or source_data[field_name] is None:
            return 0.0
        
        # Check related fields for context
        related_fields_count = 0
        total_related_fields = 0
        
        # Define related field groups
        if 'mortality' in field_name:
            related_fields = ['all_cause_mortality', 'cardiovascular_mortality', 'cancer_mortality']
        elif 'seifa' in field_name:
            related_fields = ['seifa_irsd', 'seifa_irsad', 'seifa_ier', 'seifa_ieo']
        else:
            related_fields = []
        
        for related_field in related_fields:
            total_related_fields += 1
            if related_field in source_data and source_data[related_field] is not None:
                related_fields_count += 1
        
        if total_related_fields > 0:
            return related_fields_count / total_related_fields
        else:
            return 1.0  # Field exists, no related fields to check
    
    def _calculate_accuracy(self, source_data: Dict[str, Any], field_name: str) -> float:
        """Calculate accuracy score based on validation."""
        # This would implement validation logic
        # For now, return high score if value is within expected ranges
        value = source_data.get(field_name)
        
        if value is None:
            return 0.0
        
        # Range checks for common field types
        try:
            if 'percentage' in field_name or 'rate' in field_name:
                numeric_value = float(value)
                if 0 <= numeric_value <= 100:
                    return 1.0
                else:
                    return 0.5  # Out of expected range
            elif 'population' in field_name:
                numeric_value = int(value)
                if numeric_value >= 0:
                    return 1.0
                else:
                    return 0.0
        except (TypeError, ValueError):
            return 0.5  # Can't validate, assume moderate accuracy
        
        return 0.9  # Default high accuracy
    
    def _calculate_timeliness(self, source_data: Dict[str, Any]) -> float:
        """Calculate timeliness score based on data age."""
        # Look for timestamp fields
        timestamp_fields = ['last_updated', 'collection_date', 'reference_date']
        
        latest_timestamp = None
        for field in timestamp_fields:
            if field in source_data and source_data[field] is not None:
                try:
                    timestamp = datetime.fromisoformat(str(source_data[field]).replace('Z', '+00:00'))
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                except:
                    continue
        
        if latest_timestamp is None:
            return 0.5  # Unknown timeliness
        
        # Calculate age in days
        age_days = (datetime.utcnow() - latest_timestamp).days
        
        # Apply exponential decay
        half_life = self.timeliness_half_life_days
        timeliness_score = 0.5 ** (age_days / half_life)
        
        return timeliness_score
    
    def _calculate_consistency(self, source_data: Dict[str, Any], field_name: str) -> float:
        """Calculate internal consistency score."""
        # Check for logical consistency within the data
        consistency_checks = []
        
        # Population consistency checks
        if 'total_population' in source_data and 'male_population' in source_data and 'female_population' in source_data:
            total = source_data.get('total_population', 0)
            male = source_data.get('male_population', 0)
            female = source_data.get('female_population', 0)
            
            if total > 0:
                difference = abs(total - (male + female))
                consistency_score = 1.0 - (difference / total)
                consistency_checks.append(max(0, consistency_score))
        
        # SEIFA consistency (all indices should be present together)
        seifa_indices = ['seifa_irsd', 'seifa_irsad', 'seifa_ier', 'seifa_ieo']
        seifa_present = sum(1 for idx in seifa_indices if idx in source_data and source_data[idx] is not None)
        if seifa_present > 0:
            consistency_checks.append(seifa_present / len(seifa_indices))
        
        if consistency_checks:
            return sum(consistency_checks) / len(consistency_checks)
        else:
            return 0.9  # Default high consistency


class TemporalAligner:
    """
    Aligns data from different time periods.
    
    Handles temporal misalignment in data sources by implementing various
    alignment strategies including interpolation and seasonal adjustment.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the temporal aligner.
        
        Args:
            config: Configuration for temporal alignment
            logger: Optional logger instance
        """
        self.logger = get_logger(__name__) if logger is None else logger
        
        # Reference period configuration
        self.reference_date = config.get('reference_date', datetime.now())
        self.alignment_tolerance_days = config.get('alignment_tolerance_days', 365)
        
        # Alignment strategies by field type
        self.field_strategies = config.get('field_strategies', {})
        self.default_strategy = TemporalAlignmentStrategy(
            config.get('default_strategy', 'nearest_available')
        )
        
        # Seasonal patterns (for seasonal adjustment)
        self.seasonal_patterns = config.get('seasonal_patterns', {})
        
    def align_temporal_data(
        self,
        field_name: str,
        temporal_data: List[Tuple[datetime, Any, float]]
    ) -> Tuple[Any, float, str]:
        """
        Align temporal data to reference period.
        
        Args:
            field_name: Name of the field
            temporal_data: List of (timestamp, value, quality) tuples
            
        Returns:
            Tuple of (aligned_value, confidence_score, alignment_method)
        """
        if not temporal_data:
            return None, 0.0, "no_data"
        
        # Sort by timestamp
        temporal_data.sort(key=lambda x: x[0])
        
        # Get alignment strategy
        strategy = self._get_alignment_strategy(field_name)
        
        if strategy == TemporalAlignmentStrategy.EXACT_MATCH:
            return self._align_exact_match(temporal_data)
        elif strategy == TemporalAlignmentStrategy.NEAREST_AVAILABLE:
            return self._align_nearest_available(temporal_data)
        elif strategy == TemporalAlignmentStrategy.INTERPOLATION:
            return self._align_interpolation(temporal_data)
        elif strategy == TemporalAlignmentStrategy.FORWARD_FILL:
            return self._align_forward_fill(temporal_data)
        elif strategy == TemporalAlignmentStrategy.BACKWARD_FILL:
            return self._align_backward_fill(temporal_data)
        elif strategy == TemporalAlignmentStrategy.SEASONAL_ADJUSTMENT:
            return self._align_seasonal_adjustment(field_name, temporal_data)
        else:
            return self._align_nearest_available(temporal_data)
    
    def _get_alignment_strategy(self, field_name: str) -> TemporalAlignmentStrategy:
        """Get temporal alignment strategy for a field."""
        if field_name in self.field_strategies:
            return TemporalAlignmentStrategy(self.field_strategies[field_name])
        
        # Default strategies by field type
        if any(term in field_name for term in ['population', 'count']):
            return TemporalAlignmentStrategy.INTERPOLATION
        elif any(term in field_name for term in ['rate', 'percentage']):
            return TemporalAlignmentStrategy.SEASONAL_ADJUSTMENT
        else:
            return self.default_strategy
    
    def _align_exact_match(self, temporal_data: List[Tuple[datetime, Any, float]]) -> Tuple[Any, float, str]:
        """Align by finding exact temporal match."""
        for timestamp, value, quality in temporal_data:
            if abs((timestamp - self.reference_date).days) < 1:
                return value, quality, "exact_match"
        
        # No exact match found
        return None, 0.0, "no_exact_match"
    
    def _align_nearest_available(self, temporal_data: List[Tuple[datetime, Any, float]]) -> Tuple[Any, float, str]:
        """Align by selecting nearest available data point."""
        min_distance = float('inf')
        best_match = None
        
        for timestamp, value, quality in temporal_data:
            distance = abs((timestamp - self.reference_date).days)
            if distance < min_distance and distance <= self.alignment_tolerance_days:
                min_distance = distance
                best_match = (value, quality, distance)
        
        if best_match:
            value, quality, distance = best_match
            # Adjust confidence based on temporal distance
            confidence = quality * (1.0 - distance / (self.alignment_tolerance_days * 2))
            return value, confidence, f"nearest_available_{distance}d"
        
        return None, 0.0, "no_data_within_tolerance"
    
    def _align_interpolation(self, temporal_data: List[Tuple[datetime, Any, float]]) -> Tuple[Any, float, str]:
        """Align using linear interpolation between data points."""
        # Find bracketing points
        before = None
        after = None
        
        for timestamp, value, quality in temporal_data:
            if timestamp <= self.reference_date:
                before = (timestamp, value, quality)
            elif timestamp > self.reference_date and after is None:
                after = (timestamp, value, quality)
        
        if before and after:
            # Check if values are numeric
            try:
                before_value = float(before[1])
                after_value = float(after[1])
                
                # Linear interpolation
                time_span = (after[0] - before[0]).days
                time_offset = (self.reference_date - before[0]).days
                
                if time_span > 0:
                    interpolation_weight = time_offset / time_span
                    interpolated_value = (
                        before_value * (1 - interpolation_weight) +
                        after_value * interpolation_weight
                    )
                    
                    # Average quality of bracketing points
                    avg_quality = (before[2] + after[2]) / 2
                    
                    # Adjust confidence based on time span
                    confidence = avg_quality * (1.0 - min(time_span / 730, 0.5))  # Max 50% reduction for 2+ years
                    
                    return interpolated_value, confidence, f"interpolated_{time_span}d"
            except (TypeError, ValueError):
                pass  # Non-numeric values, can't interpolate
        
        # Fall back to nearest available
        return self._align_nearest_available(temporal_data)
    
    def _align_forward_fill(self, temporal_data: List[Tuple[datetime, Any, float]]) -> Tuple[Any, float, str]:
        """Align by forward filling from most recent past data."""
        best_match = None
        
        for timestamp, value, quality in reversed(temporal_data):
            if timestamp <= self.reference_date:
                distance = (self.reference_date - timestamp).days
                if distance <= self.alignment_tolerance_days:
                    confidence = quality * (1.0 - distance / (self.alignment_tolerance_days * 2))
                    return value, confidence, f"forward_fill_{distance}d"
        
        return None, 0.0, "no_past_data"
    
    def _align_backward_fill(self, temporal_data: List[Tuple[datetime, Any, float]]) -> Tuple[Any, float, str]:
        """Align by backward filling from nearest future data."""
        for timestamp, value, quality in temporal_data:
            if timestamp > self.reference_date:
                distance = (timestamp - self.reference_date).days
                if distance <= self.alignment_tolerance_days:
                    confidence = quality * (1.0 - distance / (self.alignment_tolerance_days * 2))
                    return value, confidence, f"backward_fill_{distance}d"
        
        return None, 0.0, "no_future_data"
    
    def _align_seasonal_adjustment(
        self, 
        field_name: str, 
        temporal_data: List[Tuple[datetime, Any, float]]
    ) -> Tuple[Any, float, str]:
        """Align with seasonal adjustment."""
        # Get seasonal pattern if available
        seasonal_pattern = self.seasonal_patterns.get(field_name)
        
        if not seasonal_pattern:
            # No seasonal pattern, use interpolation
            return self._align_interpolation(temporal_data)
        
        # Find nearest data point
        value, confidence, method = self._align_nearest_available(temporal_data)
        
        if value is not None and method.startswith("nearest_available"):
            try:
                # Apply seasonal adjustment
                # This is a simplified implementation
                numeric_value = float(value)
                
                # Get seasonal factor (would be more sophisticated in practice)
                source_month = None
                for timestamp, val, _ in temporal_data:
                    if val == value:
                        source_month = timestamp.month
                        break
                
                if source_month:
                    target_month = self.reference_date.month
                    
                    # Simple seasonal adjustment
                    source_factor = seasonal_pattern.get(source_month, 1.0)
                    target_factor = seasonal_pattern.get(target_month, 1.0)
                    
                    adjusted_value = numeric_value * (target_factor / source_factor)
                    
                    return adjusted_value, confidence * 0.9, f"seasonal_adjusted_{method}"
            except (TypeError, ValueError):
                pass  # Non-numeric value, can't adjust
        
        return value, confidence, method


# Utility functions

def create_default_integration_rules() -> List[Dict[str, Any]]:
    """Create default set of integration rules."""
    return [
        {
            "name": "population_validation",
            "type": "validation",
            "source_fields": ["total_population", "male_population", "female_population"],
            "target_field": "total_population",
            "condition": "abs(total_population - (male_population + female_population)) < 10",
            "priority": 200,
            "enabled": True,
            "description": "Validate population totals consistency"
        },
        {
            "name": "seifa_completeness",
            "type": "validation",
            "source_fields": ["seifa_irsd", "seifa_irsad", "seifa_ier", "seifa_ieo"],
            "target_field": "seifa_complete",
            "condition": "all(x is not None for x in [seifa_irsd, seifa_irsad, seifa_ier, seifa_ieo])",
            "priority": 150,
            "enabled": True,
            "description": "Check SEIFA indices completeness"
        },
        {
            "name": "mortality_rate_standardisation",
            "type": "transformation",
            "source_fields": ["crude_mortality_rate", "age_distribution"],
            "target_field": "age_standardised_mortality_rate",
            "transformation": "age_standardise_rate",
            "priority": 180,
            "enabled": True,
            "description": "Calculate age-standardised mortality rate"
        }
    ]


def create_default_field_priorities() -> Dict[str, Any]:
    """Create default field priority configuration."""
    return {
        "total_population": {
            "sources": ["census", "health_indicators", "seifa"],
            "minimum_quality": 0.90
        },
        "life_expectancy": {
            "sources": ["health_indicators", "aihw", "abs"],
            "minimum_quality": 0.85
        },
        "seifa_irsd_score": {
            "sources": ["seifa", "abs"],
            "minimum_quality": 0.95
        },
        "gp_services_per_1000": {
            "sources": ["medicare_pbs", "health_indicators"],
            "minimum_quality": 0.80
        }
    }


def create_mandatory_fields_list() -> List[str]:
    """Create list of mandatory fields for integration."""
    return [
        # Primary identification
        "sa2_code",
        "sa2_name",
        
        # Geographic hierarchy
        "geographic_hierarchy",
        "boundary_data",
        
        # Core demographics
        "total_population",
        "demographic_profile",
        
        # Key indicators
        "seifa_scores",
        "seifa_deciles",
        
        # Data quality
        "data_completeness_score",
        "integration_level"
    ]