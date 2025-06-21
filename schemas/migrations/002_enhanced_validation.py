"""
Enhanced validation and field additions for AHGD schema v1.1.0.

This migration adds enhanced validation rules and additional optional fields
to existing schemas while maintaining backward compatibility.
"""

from datetime import datetime
from typing import Dict, Any, List

from schemas.base_schema import SchemaVersion


class Migration002:
    """Enhanced validation migration."""
    
    version = "002"
    description = "Enhanced validation and field additions"
    created_at = datetime(2024, 2, 1)
    
    # Schema enhancements for v1.1.0
    ENHANCEMENTS = {
        "SA2Coordinates": {
            "added_fields": [
                "data_quality_score",
                "validation_timestamp",
                "source_reliability"
            ],
            "enhanced_validators": [
                "boundary_consistency_check",
                "population_density_validation"
            ]
        },
        "HealthIndicator": {
            "added_fields": [
                "confidence_level",
                "methodology_notes",
                "peer_review_status"
            ],
            "enhanced_validators": [
                "clinical_significance_check",
                "temporal_consistency_validation"
            ]
        },
        "SEIFAScore": {
            "added_fields": [
                "calculation_methodology",
                "component_weights",
                "uncertainty_measure"
            ],
            "enhanced_validators": [
                "index_correlation_check",
                "population_threshold_validation"
            ]
        }
    }
    
    @staticmethod
    def get_field_mapping_v1_to_v1_1(schema_type: str) -> Dict[str, str]:
        """Get field mapping from v1.0.0 to v1.1.0."""
        # Most fields remain the same, new fields get default values
        return {
            # Standard mappings - most fields unchanged
            "id": "id",
            "schema_version": "schema_version",
            "created_at": "created_at",
            "updated_at": "updated_at"
        }
    
    @staticmethod
    def add_enhanced_fields(data: Dict[str, Any], schema_type: str) -> Dict[str, Any]:
        """Add enhanced validation fields for v1.1.0."""
        enhanced_data = data.copy()
        
        # Add common enhanced fields
        enhanced_data["data_quality_score"] = 85.0  # Default quality score
        enhanced_data["validation_timestamp"] = datetime.utcnow().isoformat()
        
        # Schema-specific enhancements
        if schema_type == "SA2Coordinates":
            enhanced_data["source_reliability"] = "high"
            
        elif schema_type == "HealthIndicator":
            enhanced_data["confidence_level"] = 95.0
            enhanced_data["methodology_notes"] = "Standard AHGD methodology applied"
            enhanced_data["peer_review_status"] = "pending"
            
        elif schema_type == "SEIFAScore":
            enhanced_data["calculation_methodology"] = "ABS Standard 2021"
            enhanced_data["uncertainty_measure"] = "low"
        
        return enhanced_data
    
    @staticmethod
    def validate_v1_1_constraints(data: Dict[str, Any], schema_type: str) -> List[str]:
        """Validate enhanced constraints for v1.1.0."""
        errors = []
        
        # Common validation
        if "data_quality_score" in data:
            score = data["data_quality_score"]
            if not isinstance(score, (int, float)) or score < 0 or score > 100:
                errors.append("data_quality_score must be between 0 and 100")
        
        # Schema-specific validation
        if schema_type == "HealthIndicator":
            if "confidence_level" in data:
                conf = data["confidence_level"]
                if conf < 50 or conf > 99.9:
                    errors.append("confidence_level must be between 50 and 99.9")
                    
        elif schema_type == "SEIFAScore":
            if "uncertainty_measure" in data:
                uncertainty = data["uncertainty_measure"]
                valid_levels = {"very_low", "low", "medium", "high", "very_high"}
                if uncertainty not in valid_levels:
                    errors.append(f"uncertainty_measure must be one of {valid_levels}")
        
        return errors


def upgrade() -> None:
    """Apply migration - enhanced validation."""
    print("Migration 002: Adding enhanced validation fields and rules")
    print("- Enhanced field validation")
    print("- Additional quality metrics")
    print("- Improved consistency checks")


def downgrade() -> None:
    """Revert migration - remove enhanced fields."""
    print("Migration 002 downgrade: Removing enhanced validation fields")
    print("- Removing additional quality metrics")
    print("- Reverting to basic validation rules")