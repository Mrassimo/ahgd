"""
SA1-focused geographic processor for the AHGD ETL pipeline.

This module provides comprehensive SA1-based geographic processing capabilities,
treating SA1s as the core geographic building blocks as per ABS 2021 standards.
SA1s can be aggregated up to SA2, SA3, SA4 levels as needed.
"""

import logging
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from typing import Any
from typing import Optional

import polars as pl

from ..utils.interfaces import TransformationError
from .base import BaseTransformer


@dataclass
class SA1Mapping:
    """Represents a mapping involving SA1 geographic units."""

    source_code: str
    target_sa1_code: str
    allocation_factor: float = 1.0  # For population-weighted mappings
    mapping_method: str = "direct"  # direct, area_weighted, population_weighted
    confidence: float = 1.0
    source_type: str = "unknown"  # postcode, lga, mesh_block, address
    created_at: datetime = field(default_factory=datetime.now)

    # SA1 hierarchy information
    sa2_code: Optional[str] = None
    sa3_code: Optional[str] = None
    sa4_code: Optional[str] = None
    state_code: Optional[str] = None


@dataclass
class SA1ValidationResult:
    """Result of SA1 validation."""

    is_valid: bool
    sa1_code: str
    hierarchy_codes: dict[str, str] = field(default_factory=dict)
    error_message: Optional[str] = None
    confidence: float = 1.0
    validation_method: str = "lookup"


class SA1ProcessingEngine:
    """
    Core engine for SA1-based geographic processing.

    Treats SA1s as the primary geographic unit and provides utilities
    for mapping from various sources to SA1s and aggregating to higher levels.
    """

    def __init__(self, config: dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialise the SA1 processing engine.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # SA1 lookup and hierarchy tables
        self._sa1_hierarchy: dict[str, dict[str, str]] = {}  # SA1 -> {SA2, SA3, SA4, STATE}
        self._valid_sa1_codes: set[str] = set()

        # Mapping lookup tables for various sources to SA1
        self._postcode_mappings: dict[str, list[SA1Mapping]] = {}
        self._mesh_block_mappings: dict[str, str] = {}  # Mesh Block to SA1 is 1:1
        self._address_mappings: dict[str, str] = {}  # Address to SA1 lookup

        # Reverse mappings (SA2->SA1, SA3->SA1, SA4->SA1)
        self._sa2_to_sa1s: dict[str, list[str]] = {}
        self._sa3_to_sa1s: dict[str, list[str]] = {}
        self._sa4_to_sa1s: dict[str, list[str]] = {}

        # Cache for performance
        self._mapping_cache: dict[str, list[SA1Mapping]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        self._load_reference_data()

    def process_sa1_data(self, input_data: pl.DataFrame) -> pl.DataFrame:
        """
        Process geographic data with SA1 as the primary unit.

        Args:
            input_data: Input DataFrame with geographic codes

        Returns:
            pl.DataFrame: Processed data with SA1 codes and hierarchy
        """
        self.logger.info(f"Processing {len(input_data)} records for SA1 standardisation")

        # Detect geographic code columns
        geographic_columns = self._detect_geographic_columns(input_data)
        self.logger.info(f"Detected geographic columns: {geographic_columns}")

        # Process each record
        processed_records = []
        for row in input_data.iter_rows(named=True):
            try:
                processed_row = self._process_record_to_sa1(row, geographic_columns)
                processed_records.append(processed_row)
            except Exception as e:
                self.logger.error(f"Failed to process record: {row}, error: {e!s}")
                # Add error record with original data
                error_row = dict(row)
                error_row.update(
                    {
                        "sa1_code": None,
                        "processing_error": str(e),
                        "processing_status": "error",
                    }
                )
                processed_records.append(error_row)

        result_df = pl.DataFrame(processed_records)
        self.logger.info(f"Successfully processed {len(result_df)} records")
        return result_df

    def validate_sa1_hierarchy(self, sa1_code: str) -> SA1ValidationResult:
        """
        Validate SA1 code and return hierarchy information.

        Args:
            sa1_code: 11-digit SA1 code to validate

        Returns:
            SA1ValidationResult: Validation result with hierarchy
        """
        if not sa1_code or not isinstance(sa1_code, str):
            return SA1ValidationResult(
                is_valid=False,
                sa1_code=sa1_code or "",
                error_message="SA1 code is empty or invalid type",
            )

        # Validate format (11 digits)
        if not (sa1_code.isdigit() and len(sa1_code) == 11):
            return SA1ValidationResult(
                is_valid=False,
                sa1_code=sa1_code,
                error_message="SA1 code must be exactly 11 digits",
            )

        # Check if SA1 exists in our reference data
        if sa1_code not in self._valid_sa1_codes:
            return SA1ValidationResult(
                is_valid=False,
                sa1_code=sa1_code,
                error_message="SA1 code not found in reference data",
            )

        # Extract hierarchy codes from SA1 structure
        hierarchy = self._extract_hierarchy_from_sa1(sa1_code)

        return SA1ValidationResult(
            is_valid=True, sa1_code=sa1_code, hierarchy_codes=hierarchy, confidence=1.0
        )

    def aggregate_sa1_to_sa2(
        self, sa1_data: pl.DataFrame, value_columns: list[str]
    ) -> pl.DataFrame:
        """
        Aggregate SA1 data to SA2 level.

        Args:
            sa1_data: DataFrame with SA1-level data
            value_columns: Columns to aggregate (sum)

        Returns:
            pl.DataFrame: SA2-level aggregated data
        """
        if "sa1_code" not in sa1_data.columns:
            raise ValueError("Input data must contain 'sa1_code' column")

        # Add SA2 codes
        sa1_with_hierarchy = sa1_data.with_columns(
            [pl.col("sa1_code").str.slice(0, 9).alias("sa2_code")]
        )

        # Aggregate by SA2
        aggregated = sa1_with_hierarchy.group_by("sa2_code").agg(
            [
                *[pl.col(col).sum().alias(col) for col in value_columns],
                pl.col("sa1_code").count().alias("sa1_count"),
            ]
        )

        self.logger.info(f"Aggregated {len(sa1_data)} SA1 records to {len(aggregated)} SA2 records")
        return aggregated

    def get_sa1_neighbours(self, sa1_code: str, distance_km: float = 5.0) -> list[str]:
        """
        Find neighbouring SA1s within specified distance.

        Args:
            sa1_code: Target SA1 code
            distance_km: Maximum distance in kilometres

        Returns:
            List[str]: List of neighbouring SA1 codes
        """
        # Simplified implementation - would use spatial index in production
        neighbours = []

        # Get SA1 hierarchy to find same SA2 SA1s first
        hierarchy = self._extract_hierarchy_from_sa1(sa1_code)
        sa2_code = hierarchy.get("sa2_code", "")

        if sa2_code and sa2_code in self._sa2_to_sa1s:
            same_sa2_sa1s = [code for code in self._sa2_to_sa1s[sa2_code] if code != sa1_code]
            neighbours.extend(same_sa2_sa1s[:10])  # Limit for performance

        return neighbours

    def standardise_geographic_data(self, input_data: pl.DataFrame) -> pl.DataFrame:
        """
        Main method to standardise geographic data to SA1 framework.

        Args:
            input_data: Input DataFrame with various geographic codes

        Returns:
            pl.DataFrame: Standardised data with SA1 codes and hierarchy
        """
        self.logger.info("Starting geographic standardisation to SA1 framework")
        start_time = time.time()

        try:
            # Process data to SA1
            standardised_data = self.process_sa1_data(input_data)

            # Add full hierarchy information
            standardised_data = self._add_geographic_hierarchy(standardised_data)

            # Validate results
            validation_summary = self._validate_standardisation_results(standardised_data)

            processing_time = time.time() - start_time
            self.logger.info(
                f"Geographic standardisation completed in {processing_time:.2f}s. "
                f"Validation: {validation_summary}"
            )

            return standardised_data

        except Exception as e:
            self.logger.error(f"Geographic standardisation failed: {e!s}")
            raise TransformationError(f"SA1 standardisation failed: {e!s}") from e

    def validate_sa1_codes(self, sa1_codes: list[str]) -> dict[str, bool]:
        """
        Validate multiple SA1 codes efficiently.

        Args:
            sa1_codes: List of SA1 codes to validate

        Returns:
            Dict[str, bool]: Validation results for each code
        """
        results = {}
        for code in sa1_codes:
            validation = self.validate_sa1_hierarchy(code)
            results[code] = validation.is_valid

        return results

    def get_cache_statistics(self) -> dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._mapping_cache),
        }

    def _detect_geographic_columns(self, data: pl.DataFrame) -> dict[str, str]:
        """Detect which columns contain geographic codes."""
        geographic_columns = {}

        for column in data.columns:
            column_lower = column.lower()
            if "postcode" in column_lower or "pcode" in column_lower:
                geographic_columns[column] = "postcode"
            elif "sa1" in column_lower:
                geographic_columns[column] = "sa1"
            elif "sa2" in column_lower:
                geographic_columns[column] = "sa2"
            elif "lga" in column_lower:
                geographic_columns[column] = "lga"
            elif "mesh" in column_lower and "block" in column_lower:
                geographic_columns[column] = "mesh_block"

        return geographic_columns

    def _process_record_to_sa1(
        self, record: dict[str, Any], geographic_columns: dict[str, str]
    ) -> dict[str, Any]:
        """Process a single record to extract SA1 information."""
        processed_record = dict(record)
        sa1_code = None
        processing_method = None

        # Priority order: SA1 direct, mesh_block, postcode, SA2->SA1
        for column, code_type in geographic_columns.items():
            value = record.get(column)
            if not value:
                continue

            try:
                if code_type == "sa1":
                    # Direct SA1 - validate
                    validation = self.validate_sa1_hierarchy(str(value))
                    if validation.is_valid:
                        sa1_code = validation.sa1_code
                        processing_method = "direct_sa1"
                        break

                elif code_type == "mesh_block":
                    # Mesh block to SA1 mapping
                    mapped_sa1 = self._mesh_block_mappings.get(str(value))
                    if mapped_sa1:
                        sa1_code = mapped_sa1
                        processing_method = "mesh_block_mapping"
                        break

                elif code_type == "postcode":
                    # Postcode to SA1 mapping (may return multiple)
                    mappings = self._map_postcode_to_sa1(str(value))
                    if mappings:
                        # Take first mapping (could implement better selection logic)
                        sa1_code = mappings[0].target_sa1_code
                        processing_method = "postcode_mapping"
                        break

                elif code_type == "sa2":
                    # SA2 contains multiple SA1s - would need additional info to select
                    # For now, take first SA1 in SA2
                    sa1_codes = self._sa2_to_sa1s.get(str(value), [])
                    if sa1_codes:
                        sa1_code = sa1_codes[0]
                        processing_method = "sa2_fallback"
                        break

            except Exception as e:
                self.logger.warning(f"Failed to process {code_type} {value}: {e!s}")
                continue

        # Add SA1 and processing information
        processed_record["sa1_code"] = sa1_code
        processed_record["processing_method"] = processing_method
        processed_record["processing_status"] = "success" if sa1_code else "no_mapping"

        return processed_record

    def _extract_hierarchy_from_sa1(self, sa1_code: str) -> dict[str, str]:
        """Extract geographic hierarchy codes from SA1 code structure."""
        if not sa1_code or len(sa1_code) != 11:
            return {}

        # Use cached hierarchy if available
        if sa1_code in self._sa1_hierarchy:
            return self._sa1_hierarchy[sa1_code]

        # Extract from SA1 code structure
        state_code = self._get_state_code_from_digit(sa1_code[0])
        sa4_code = sa1_code[:3]
        sa3_code = sa1_code[:5]
        sa2_code = sa1_code[:9]

        hierarchy = {
            "sa1_code": sa1_code,
            "sa2_code": sa2_code,
            "sa3_code": sa3_code,
            "sa4_code": sa4_code,
            "state_code": state_code,
        }

        return hierarchy

    def _get_state_code_from_digit(self, digit: str) -> str:
        """Convert numeric state digit to state code."""
        state_mapping = {
            "1": "NSW",
            "2": "VIC",
            "3": "QLD",
            "4": "SA",
            "5": "WA",
            "6": "TAS",
            "7": "NT",
            "8": "ACT",
        }
        return state_mapping.get(digit, "UNKNOWN")

    def _add_geographic_hierarchy(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add complete geographic hierarchy to SA1 data."""
        if "sa1_code" not in data.columns:
            return data

        # Add hierarchy columns
        hierarchy_data = []
        for row in data.iter_rows(named=True):
            sa1_code = row.get("sa1_code")
            if sa1_code:
                hierarchy = self._extract_hierarchy_from_sa1(sa1_code)
                row.update(hierarchy)
            hierarchy_data.append(row)

        return pl.DataFrame(hierarchy_data)

    def _validate_standardisation_results(self, data: pl.DataFrame) -> dict[str, Any]:
        """Validate standardisation results."""
        total_records = len(data)

        if "processing_status" in data.columns:
            status_counts = data.get_column("processing_status").value_counts()
            success_count = (
                status_counts.filter(pl.col("processing_status") == "success")
                .select("count")
                .to_series()
                .sum()
            )
        else:
            success_count = data.filter(pl.col("sa1_code").is_not_null()).height

        success_rate = success_count / total_records if total_records > 0 else 0

        return {
            "total_records": total_records,
            "successful_mappings": success_count,
            "success_rate": success_rate,
            "failed_mappings": total_records - success_count,
        }

    def _map_postcode_to_sa1(self, postcode: str) -> list[SA1Mapping]:
        """Map postcode to SA1(s) - placeholder implementation."""
        # In production, this would use ABS correspondence files
        mappings = self._postcode_mappings.get(postcode, [])
        return mappings

    def _load_reference_data(self):
        """Load SA1 reference data and mappings."""
        self.logger.info("Loading SA1 reference data...")

        # In production, this would load from ABS data files
        # For now, populate with some test data
        self._populate_test_reference_data()

        self.logger.info(
            f"Loaded reference data: {len(self._valid_sa1_codes)} SA1 codes, "
            f"{len(self._sa2_to_sa1s)} SA2 mappings"
        )

    def _populate_test_reference_data(self):
        """Populate test reference data for development."""
        # Test SA1 codes from our fixtures
        test_sa1_codes = [
            "10102100701",
            "10102100702",
            "10102100703",
            "20203200801",
            "20203200802",
            "20203200803",
            "30504500901",
            "30504500902",
            "40102800501",
            "40102800502",
        ]

        for sa1_code in test_sa1_codes:
            self._valid_sa1_codes.add(sa1_code)

            # Build hierarchy
            hierarchy = self._extract_hierarchy_from_sa1(sa1_code)
            self._sa1_hierarchy[sa1_code] = hierarchy

            # Build reverse mappings
            sa2_code = hierarchy.get("sa2_code")
            if sa2_code:
                if sa2_code not in self._sa2_to_sa1s:
                    self._sa2_to_sa1s[sa2_code] = []
                self._sa2_to_sa1s[sa2_code].append(sa1_code)


class SA1GeographicTransformer(BaseTransformer):
    """
    SA1-focused geographic transformation component.

    This transformer processes input data and standardises it to use SA1
    as the primary geographic unit, with supporting hierarchy information.
    """

    def __init__(self, config: dict[str, Any] = None):
        """Initialise SA1 geographic transformer."""
        super().__init__(transformer_id="sa1_geographic_transformer", config=config or {})
        self.sa1_engine = SA1ProcessingEngine(self.config, self.logger)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        """
        Transform data using SA1 geographic standardisation.

        Args:
            data: Input DataFrame with geographic information

        Returns:
            pl.DataFrame: Transformed data with SA1 standardisation
        """
        return self.sa1_engine.standardise_geographic_data(data)

    def get_transformation_metadata(self) -> dict[str, Any]:
        """Get metadata about the transformation process."""
        cache_stats = self.sa1_engine.get_cache_statistics()
        return {
            "transformer_type": "SA1GeographicTransformer",
            "primary_geographic_unit": "SA1",
            "cache_statistics": cache_stats,
            "supported_input_types": ["postcode", "sa1", "sa2", "mesh_block"],
            "british_english_spelling": True,
        }

    def get_schema(self):
        """Return the SA1 schema for this transformer."""
        from schemas.sa1_schema import SA1Coordinates

        return SA1Coordinates
