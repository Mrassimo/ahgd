"""
Australian Health Data Validators

Comprehensive validation utilities for Australian health data standards including:
- SA2 geographic codes (9-digit format with state validation)
- SEIFA 2021 socio-economic indices (deciles, scores, rankings)
- PBS prescription data (ATC codes, dispensing patterns)
- ABS Census 2021 data structures and formats
- Geographic boundary coordinate validation
- Population density and age distribution patterns

This module provides the core validation logic for ensuring Australian health data
compliance with government standards and realistic health patterns.
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
import json

import polars as pl
import numpy as np
from loguru import logger


class AustralianHealthDataValidator:
    """Core validator for Australian health data standards."""
    
    # Australian state/territory codes for SA2 validation
    VALID_STATE_CODES = {
        1: "NSW",  # New South Wales
        2: "VIC",  # Victoria  
        3: "QLD",  # Queensland
        4: "SA",   # South Australia
        5: "WA",   # Western Australia
        6: "TAS",  # Tasmania
        7: "NT",   # Northern Territory
        8: "ACT",  # Australian Capital Territory
    }
    
    # SEIFA 2021 valid ranges
    SEIFA_SCORE_RANGE = (800, 1200)
    SEIFA_DECILE_RANGE = (1, 10)
    SEIFA_RANKING_RANGE = (1, 2544)  # Total SA2 areas in Australia
    
    # Australian geographic bounds
    AUSTRALIA_BOUNDS = {
        "latitude": (-44.0, -10.0),
        "longitude": (113.0, 154.0)
    }
    
    # Australian postcode ranges by state
    POSTCODE_RANGES = {
        "NSW": [(1000, 1999), (2000, 2599), (2619, 2899), (2921, 2999)],
        "ACT": [(200, 299), (2600, 2618), (2900, 2920)],
        "VIC": [(3000, 3999), (8000, 8999)],
        "QLD": [(4000, 4999), (9000, 9999)],
        "SA": [(5000, 5999)],
        "WA": [(6000, 6799), (6800, 6999)],
        "TAS": [(7000, 7799), (7800, 7999)],
        "NT": [(800, 899), (900, 999)]
    }
    
    def __init__(self):
        """Initialize the validator with Australian health data standards."""
        self.validation_rules = self._load_validation_rules()
        self.logger = logger.bind(component="australian_health_validator")
    
    def validate_sa2_code(self, sa2_code: Union[str, int]) -> Dict[str, Union[bool, str]]:
        """
        Validate SA2 code format and state prefix.
        
        Args:
            sa2_code: SA2 code to validate (9-digit string or integer)
            
        Returns:
            Dict with validation result and details
        """
        # Convert to string if integer
        sa2_str = str(sa2_code) if isinstance(sa2_code, int) else sa2_code
        
        result = {
            "valid": False,
            "code": sa2_str,
            "errors": []
        }
        
        # Check if string and length
        if not isinstance(sa2_str, str):
            result["errors"].append("SA2 code must be string or integer")
            return result
            
        if len(sa2_str) != 9:
            result["errors"].append(f"SA2 code must be 9 digits, got {len(sa2_str)}")
            return result
        
        # Check if all digits
        if not sa2_str.isdigit():
            result["errors"].append("SA2 code must contain only digits")
            return result
        
        # Check valid state prefix
        state_code = int(sa2_str[0])
        if state_code not in self.VALID_STATE_CODES:
            result["errors"].append(f"Invalid state code {state_code}, must be 1-8")
            return result
        
        # All validations passed
        result["valid"] = True
        result["state"] = self.VALID_STATE_CODES[state_code]
        result["state_code"] = state_code
        
        return result
    
    def validate_seifa_2021_data(self, seifa_record: Dict) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate SEIFA 2021 data record for compliance with ABS methodology.
        
        Args:
            seifa_record: Dictionary containing SEIFA data fields
            
        Returns:
            Dict with validation results and error details
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Required fields for SEIFA 2021
        required_fields = ["sa2_code_2021", "irsd_score", "irsd_decile", 
                          "irsad_score", "irsad_decile", "ier_score", "ier_decile",
                          "ieo_score", "ieo_decile", "usual_resident_population"]
        
        # Check required fields
        for field in required_fields:
            if field not in seifa_record or seifa_record[field] is None:
                result["errors"].append(f"Missing required field: {field}")
                result["valid"] = False
        
        if not result["valid"]:
            return result
        
        # Validate SA2 code
        sa2_validation = self.validate_sa2_code(seifa_record["sa2_code_2021"])
        if not sa2_validation["valid"]:
            result["errors"].extend([f"SA2 code: {error}" for error in sa2_validation["errors"]])
            result["valid"] = False
        
        # Validate SEIFA scores and deciles
        seifa_indices = ["irsd", "irsad", "ier", "ieo"]
        
        for index in seifa_indices:
            score_field = f"{index}_score"
            decile_field = f"{index}_decile"
            
            # Validate score range
            score = seifa_record.get(score_field)
            if score is not None:
                if not (self.SEIFA_SCORE_RANGE[0] <= score <= self.SEIFA_SCORE_RANGE[1]):
                    result["errors"].append(
                        f"{score_field} {score} outside valid range {self.SEIFA_SCORE_RANGE}"
                    )
                    result["valid"] = False
            
            # Validate decile range
            decile = seifa_record.get(decile_field)
            if decile is not None:
                if not (self.SEIFA_DECILE_RANGE[0] <= decile <= self.SEIFA_DECILE_RANGE[1]):
                    result["errors"].append(
                        f"{decile_field} {decile} outside valid range {self.SEIFA_DECILE_RANGE}"
                    )
                    result["valid"] = False
        
        # Validate population
        population = seifa_record.get("usual_resident_population")
        if population is not None:
            if population <= 0:
                result["errors"].append("Population must be positive")
                result["valid"] = False
            elif population > 50000:
                result["warnings"].append(f"Unusually high population for SA2: {population}")
            elif population < 100:
                result["warnings"].append(f"Unusually low population for SA2: {population}")
        
        return result
    
    def validate_australian_coordinates(self, latitude: float, longitude: float) -> Dict[str, Union[bool, str]]:
        """
        Validate geographic coordinates are within Australian bounds.
        
        Args:
            latitude: Latitude value
            longitude: Longitude value
            
        Returns:
            Dict with validation result and details
        """
        result = {
            "valid": True,
            "errors": []
        }
        
        lat_min, lat_max = self.AUSTRALIA_BOUNDS["latitude"]
        lon_min, lon_max = self.AUSTRALIA_BOUNDS["longitude"]
        
        if not (lat_min <= latitude <= lat_max):
            result["errors"].append(
                f"Latitude {latitude} outside Australian bounds ({lat_min}, {lat_max})"
            )
            result["valid"] = False
        
        if not (lon_min <= longitude <= lon_max):
            result["errors"].append(
                f"Longitude {longitude} outside Australian bounds ({lon_min}, {lon_max})"
            )
            result["valid"] = False
        
        return result
    
    def validate_atc_code(self, atc_code: str) -> Dict[str, Union[bool, str]]:
        """
        Validate ATC (Anatomical Therapeutic Chemical) code format.
        
        Args:
            atc_code: ATC code to validate (e.g., "A02BC01")
            
        Returns:
            Dict with validation result and details
        """
        result = {
            "valid": False,
            "code": atc_code,
            "errors": []
        }
        
        if not isinstance(atc_code, str):
            result["errors"].append("ATC code must be string")
            return result
        
        if len(atc_code) != 7:
            result["errors"].append(f"ATC code must be 7 characters, got {len(atc_code)}")
            return result
        
        # ATC code pattern: A02BC01 (Letter-Digit-Digit-Letter-Letter-Digit-Digit)
        pattern = r'^[A-N][0-9][0-9][A-Z][A-Z][0-9][0-9]$'
        
        if not re.match(pattern, atc_code):
            result["errors"].append("ATC code format invalid, expected: A02BC01")
            return result
        
        # Validate first letter (anatomical group)
        anatomical_groups = set("ABCDEFGHJLMNPQRSUV")
        if atc_code[0] not in anatomical_groups:
            result["errors"].append(f"Invalid anatomical group: {atc_code[0]}")
            return result
        
        result["valid"] = True
        result["anatomical_group"] = atc_code[0]
        result["therapeutic_group"] = atc_code[1:3]
        result["chemical_group"] = atc_code[3:5]
        result["substance"] = atc_code[5:7]
        
        return result
    
    def validate_australian_postcode(self, postcode: Union[str, int], state: Optional[str] = None) -> Dict[str, Union[bool, str]]:
        """
        Validate Australian postcode format and state consistency.
        
        Args:
            postcode: Postcode to validate
            state: Optional state/territory code for validation
            
        Returns:
            Dict with validation result and details
        """
        # Convert to string
        postcode_str = str(postcode).zfill(4) if isinstance(postcode, int) else postcode
        
        result = {
            "valid": False,
            "postcode": postcode_str,
            "errors": []
        }
        
        if len(postcode_str) != 4:
            result["errors"].append(f"Postcode must be 4 digits, got {len(postcode_str)}")
            return result
        
        if not postcode_str.isdigit():
            result["errors"].append("Postcode must contain only digits")
            return result
        
        postcode_int = int(postcode_str)
        
        # Check overall Australian postcode range
        if postcode_int < 200 or postcode_int > 9999:
            result["errors"].append(f"Postcode {postcode_int} outside Australian range (200-9999)")
            return result
        
        # Validate against state if provided
        if state:
            state_upper = state.upper()
            if state_upper in self.POSTCODE_RANGES:
                valid_for_state = False
                for min_code, max_code in self.POSTCODE_RANGES[state_upper]:
                    if min_code <= postcode_int <= max_code:
                        valid_for_state = True
                        break
                
                if not valid_for_state:
                    result["errors"].append(f"Postcode {postcode_int} not valid for {state_upper}")
                    return result
        
        result["valid"] = True
        result["inferred_state"] = self._infer_state_from_postcode(postcode_int)
        
        return result
    
    def validate_census_2021_data(self, census_record: Dict) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate ABS Census 2021 data structure and field values.
        
        Args:
            census_record: Dictionary containing Census data fields
            
        Returns:
            Dict with validation results and error details
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check SA1/SA2 code if present
        for sa_field in ["sa1_code_2021", "sa2_code_2021"]:
            if sa_field in census_record:
                sa_code = census_record[sa_field]
                if sa_field == "sa2_code_2021":
                    sa_validation = self.validate_sa2_code(sa_code)
                    if not sa_validation["valid"]:
                        result["errors"].extend([f"{sa_field}: {error}" for error in sa_validation["errors"]])
                        result["valid"] = False
                elif sa_field == "sa1_code_2021":
                    if not (isinstance(sa_code, str) and len(sa_code) == 11 and sa_code.isdigit()):
                        result["errors"].append(f"SA1 code must be 11-digit string, got {sa_code}")
                        result["valid"] = False
        
        # Validate population counts
        population_fields = ["tot_p_m", "tot_p_f", "tot_p_p"]  # Male, Female, Persons
        if all(field in census_record for field in population_fields):
            male_pop = census_record["tot_p_m"]
            female_pop = census_record["tot_p_f"]
            total_pop = census_record["tot_p_p"]
            
            if male_pop + female_pop != total_pop:
                result["errors"].append(
                    f"Population sum mismatch: Male({male_pop}) + Female({female_pop}) ≠ Total({total_pop})"
                )
                result["valid"] = False
        
        # Validate age group consistency
        age_fields = [field for field in census_record.keys() if field.startswith("age_")]
        if age_fields:
            age_totals = sum(census_record[field] for field in age_fields if isinstance(census_record[field], (int, float)))
            if "tot_p_p" in census_record and abs(age_totals - census_record["tot_p_p"]) > 0.01 * census_record["tot_p_p"]:
                result["warnings"].append("Age group totals don't match total population")
        
        return result
    
    def validate_health_service_data(self, health_record: Dict) -> Dict[str, Union[bool, List[str]]]:
        """
        Validate health service utilisation data patterns.
        
        Args:
            health_record: Dictionary containing health service data
            
        Returns:
            Dict with validation results and error details
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate service date
        if "service_date" in health_record:
            service_date = health_record["service_date"]
            if isinstance(service_date, str):
                try:
                    parsed_date = datetime.strptime(service_date, "%Y-%m-%d")
                    # Check date is not in future
                    if parsed_date > datetime.now():
                        result["errors"].append("Service date cannot be in future")
                        result["valid"] = False
                    # Check date is not too old (pre-Medicare)
                    elif parsed_date < datetime(1975, 1, 1):
                        result["errors"].append("Service date cannot be before Medicare (1975)")
                        result["valid"] = False
                except ValueError:
                    result["errors"].append(f"Invalid service date format: {service_date}")
                    result["valid"] = False
        
        # Validate provider numbers
        if "provider_number" in health_record:
            provider_num = str(health_record["provider_number"])
            if not (len(provider_num) == 6 and provider_num.isdigit()):
                result["errors"].append("Provider number must be 6-digit string")
                result["valid"] = False
        
        # Validate benefit amounts
        benefit_fields = ["schedule_fee", "benefit_paid", "patient_contribution"]
        for field in benefit_fields:
            if field in health_record:
                amount = health_record[field]
                if isinstance(amount, (int, float)):
                    if amount < 0:
                        result["errors"].append(f"{field} cannot be negative: {amount}")
                        result["valid"] = False
                    elif amount > 10000:  # Unusually high
                        result["warnings"].append(f"Unusually high {field}: ${amount}")
        
        return result
    
    def _load_validation_rules(self) -> Dict:
        """Load validation rules from configuration."""
        # Default validation rules
        return {
            "sa2_code_length": 9,
            "seifa_score_range": self.SEIFA_SCORE_RANGE,
            "seifa_decile_range": self.SEIFA_DECILE_RANGE,
            "australia_bounds": self.AUSTRALIA_BOUNDS,
            "max_population_per_sa2": 50000,
            "min_population_per_sa2": 100
        }
    
    def _infer_state_from_postcode(self, postcode: int) -> Optional[str]:
        """Infer state/territory from postcode."""
        for state, ranges in self.POSTCODE_RANGES.items():
            for min_code, max_code in ranges:
                if min_code <= postcode <= max_code:
                    return state
        return None


class DataQualityMetricsCalculator:
    """Calculate comprehensive data quality metrics for Australian health data."""
    
    def __init__(self, validator: AustralianHealthDataValidator):
        """Initialize with validator instance."""
        self.validator = validator
        self.logger = logger.bind(component="data_quality_metrics")
    
    def calculate_completeness_metrics(self, df: pl.DataFrame) -> Dict[str, float]:
        """
        Calculate data completeness metrics.
        
        Args:
            df: Polars DataFrame to analyse
            
        Returns:
            Dict with completeness metrics
        """
        total_cells = df.width * df.height
        null_cells = sum(df[col].null_count() for col in df.columns)
        
        completeness_overall = ((total_cells - null_cells) / total_cells) * 100
        
        # Per-column completeness
        column_completeness = {}
        for col in df.columns:
            null_count = df[col].null_count()
            completeness = ((df.height - null_count) / df.height) * 100
            column_completeness[col] = completeness
        
        return {
            "overall_completeness": completeness_overall,
            "column_completeness": column_completeness,
            "total_cells": total_cells,
            "null_cells": null_cells
        }
    
    def calculate_validity_metrics(self, df: pl.DataFrame, data_type: str = "seifa") -> Dict[str, float]:
        """
        Calculate data validity metrics based on Australian standards.
        
        Args:
            df: Polars DataFrame to analyse
            data_type: Type of data (seifa, health, census, geographic)
            
        Returns:
            Dict with validity metrics
        """
        validity_results = {}
        
        if data_type == "seifa" and "sa2_code_2021" in df.columns:
            # Validate SA2 codes
            sa2_codes = df["sa2_code_2021"].to_list()
            valid_sa2_count = 0
            
            for code in sa2_codes:
                if code is not None:
                    validation = self.validator.validate_sa2_code(code)
                    if validation["valid"]:
                        valid_sa2_count += 1
            
            validity_results["sa2_code_validity"] = (valid_sa2_count / len(sa2_codes)) * 100
            
            # Validate SEIFA scores
            seifa_indices = ["irsd", "irsad", "ier", "ieo"]
            for index in seifa_indices:
                score_col = f"{index}_score"
                decile_col = f"{index}_decile"
                
                if score_col in df.columns:
                    scores = df[score_col].drop_nulls().to_list()
                    valid_scores = sum(1 for score in scores 
                                     if self.validator.SEIFA_SCORE_RANGE[0] <= score <= self.validator.SEIFA_SCORE_RANGE[1])
                    validity_results[f"{score_col}_validity"] = (valid_scores / len(scores)) * 100 if scores else 0
                
                if decile_col in df.columns:
                    deciles = df[decile_col].drop_nulls().to_list()
                    valid_deciles = sum(1 for decile in deciles 
                                      if self.validator.SEIFA_DECILE_RANGE[0] <= decile <= self.validator.SEIFA_DECILE_RANGE[1])
                    validity_results[f"{decile_col}_validity"] = (valid_deciles / len(deciles)) * 100 if deciles else 0
        
        return validity_results
    
    def calculate_consistency_metrics(self, df: pl.DataFrame) -> Dict[str, float]:
        """
        Calculate data consistency metrics.
        
        Args:
            df: Polars DataFrame to analyse
            
        Returns:
            Dict with consistency metrics
        """
        consistency_results = {}
        
        # SA2 code uniqueness
        if "sa2_code_2021" in df.columns:
            sa2_codes = df["sa2_code_2021"].drop_nulls()
            unique_count = sa2_codes.n_unique()
            total_count = len(sa2_codes)
            consistency_results["sa2_uniqueness"] = (unique_count / total_count) * 100
        
        # SEIFA score-decile consistency
        seifa_indices = ["irsd", "irsad", "ier", "ieo"]
        for index in seifa_indices:
            score_col = f"{index}_score"
            decile_col = f"{index}_decile"
            
            if score_col in df.columns and decile_col in df.columns:
                # Check correlation between scores and deciles
                scores_deciles = df.select([score_col, decile_col]).drop_nulls()
                if len(scores_deciles) > 1:
                    correlation = scores_deciles.select(
                        pl.corr(score_col, decile_col)
                    ).item()
                    consistency_results[f"{index}_score_decile_correlation"] = correlation * 100 if correlation else 0
        
        return consistency_results
    
    def calculate_timeliness_metrics(self, df: pl.DataFrame, date_column: str) -> Dict[str, Union[float, str]]:
        """
        Calculate data timeliness metrics.
        
        Args:
            df: Polars DataFrame to analyse
            date_column: Name of the date column
            
        Returns:
            Dict with timeliness metrics
        """
        if date_column not in df.columns:
            return {"error": f"Date column {date_column} not found"}
        
        dates = df[date_column].drop_nulls()
        if len(dates) == 0:
            return {"error": "No valid dates found"}
        
        # Calculate data freshness
        latest_date = dates.max()
        today = datetime.now().date()
        
        if isinstance(latest_date, str):
            try:
                latest_date = datetime.strptime(latest_date, "%Y-%m-%d").date()
            except ValueError:
                return {"error": "Invalid date format"}
        
        days_since_latest = (today - latest_date).days
        
        # Calculate data coverage period
        earliest_date = dates.min()
        if isinstance(earliest_date, str):
            earliest_date = datetime.strptime(earliest_date, "%Y-%m-%d").date()
        
        coverage_days = (latest_date - earliest_date).days
        
        return {
            "days_since_latest_data": days_since_latest,
            "data_coverage_days": coverage_days,
            "latest_date": str(latest_date),
            "earliest_date": str(earliest_date),
            "freshness_score": max(0, 100 - (days_since_latest / 30) * 10)  # Decreases by 10 points per month
        }
    
    def generate_comprehensive_quality_report(self, df: pl.DataFrame, data_type: str = "seifa") -> Dict:
        """
        Generate comprehensive data quality report.
        
        Args:
            df: Polars DataFrame to analyse
            data_type: Type of data being analysed
            
        Returns:
            Comprehensive quality report dictionary
        """
        report = {
            "dataset_info": {
                "rows": df.height,
                "columns": df.width,
                "data_type": data_type,
                "analysis_timestamp": datetime.now().isoformat()
            },
            "completeness": self.calculate_completeness_metrics(df),
            "validity": self.calculate_validity_metrics(df, data_type),
            "consistency": self.calculate_consistency_metrics(df)
        }
        
        # Add timeliness if date columns exist
        date_columns = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
        if date_columns:
            report["timeliness"] = self.calculate_timeliness_metrics(df, date_columns[0])
        
        # Calculate overall quality score
        completeness_score = report["completeness"]["overall_completeness"]
        validity_scores = list(report["validity"].values())
        avg_validity = sum(validity_scores) / len(validity_scores) if validity_scores else 100
        consistency_scores = list(report["consistency"].values())
        avg_consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 100
        
        overall_score = (completeness_score + avg_validity + avg_consistency) / 3
        report["overall_quality_score"] = overall_score
        
        # Add quality classification
        if overall_score >= 95:
            report["quality_classification"] = "Excellent"
        elif overall_score >= 90:
            report["quality_classification"] = "Good"
        elif overall_score >= 80:
            report["quality_classification"] = "Acceptable"
        elif overall_score >= 70:
            report["quality_classification"] = "Poor"
        else:
            report["quality_classification"] = "Unacceptable"
        
        return report


if __name__ == "__main__":
    # Example usage
    validator = AustralianHealthDataValidator()
    
    # Test SA2 code validation
    sa2_result = validator.validate_sa2_code("101021007")
    print(f"SA2 validation: {sa2_result}")
    
    # Test SEIFA data validation
    seifa_record = {
        "sa2_code_2021": "101021007",
        "irsd_score": 1050,
        "irsd_decile": 8,
        "irsad_score": 1080,
        "irsad_decile": 7,
        "ier_score": 1000,
        "ier_decile": 6,
        "ieo_score": 1150,
        "ieo_decile": 9,
        "usual_resident_population": 15000
    }
    
    seifa_result = validator.validate_seifa_2021_data(seifa_record)
    print(f"SEIFA validation: {seifa_result}")