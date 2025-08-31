"""
Geographic Utility Classes for SA1 Mapping

Provides geographic matching and population weighting for mapping various
geographic levels (postcode, LGA, SA3, SA4, PHA) down to SA1 level.
"""

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class GeographicMatcher:
    """
    Handles geographic mapping and population weighting for SA1-level analysis.

    Maps various geographic identifiers (postcodes, LGA codes, SA3/SA4 codes,
    Population Health Areas) to SA1 codes using population-based weighting.
    """

    def __init__(self, db_path: str = "health_analytics.db"):
        self.db_path = db_path
        self._sa1_lookup = None
        self._postcode_mapping = None
        self._lga_mapping = None
        self._sa3_mapping = None
        self._pha_mapping = None
        self._load_mappings()

    def _load_mappings(self):
        """Load geographic mapping tables from database."""
        try:
            conn = duckdb.connect(self.db_path)

            # Load SA1 lookup table
            try:
                self._sa1_lookup = conn.execute(
                    """
                    SELECT sa1_code, sa1_name, sa2_code, sa3_code, sa4_code,
                           state_code, population_total
                    FROM stg_sa1_boundaries
                    WHERE data_quality_score > 0.8
                """
                ).df()
                logger.info(f"Loaded {len(self._sa1_lookup)} SA1 boundaries")
            except Exception as e:
                logger.warning(f"Could not load SA1 boundaries: {e}")
                self._sa1_lookup = pd.DataFrame()

            # Load postcode to SA1 mappings (if available)
            try:
                self._postcode_mapping = conn.execute(
                    """
                    SELECT postcode, sa1_code, population_weight
                    FROM postcode_sa1_mapping
                """
                ).df()
            except Exception as e:
                logger.warning(f"Postcode mapping not available: {e}")
                self._postcode_mapping = pd.DataFrame()

            # Load LGA to SA1 mappings
            try:
                self._lga_mapping = conn.execute(
                    """
                    SELECT lga_code, sa1_code, population_weight
                    FROM lga_sa1_mapping
                """
                ).df()
            except Exception as e:
                logger.warning(f"LGA mapping not available: {e}")
                self._lga_mapping = pd.DataFrame()

            # Load SA3 to SA1 mappings (hierarchical relationship)
            if not self._sa1_lookup.empty:
                self._sa3_mapping = (
                    self._sa1_lookup.groupby("sa3_code")
                    .agg({"sa1_code": list, "population_total": "sum"})
                    .reset_index()
                )

            conn.close()

        except Exception as e:
            logger.error(f"Failed to load geographic mappings: {e}")
            self._initialize_empty_mappings()

    def _initialize_empty_mappings(self):
        """Initialize empty mapping tables as fallback."""
        self._sa1_lookup = pd.DataFrame()
        self._postcode_mapping = pd.DataFrame()
        self._lga_mapping = pd.DataFrame()
        self._sa3_mapping = pd.DataFrame()

    def map_to_sa1(self, geographic_id: str, source_type: str = "auto") -> list[tuple[str, float]]:
        """
        Map any geographic identifier to SA1 codes with population weights.

        Args:
            geographic_id: The geographic identifier (postcode, LGA, SA3, etc.)
            source_type: Type of source geography ('postcode', 'lga', 'sa3', 'sa4', 'auto')

        Returns:
            List of tuples (sa1_code, population_weight)
        """
        if not geographic_id:
            return []

        geographic_id = str(geographic_id).strip()

        # Auto-detect source type if not specified
        if source_type == "auto":
            source_type = self._detect_geographic_type(geographic_id)

        # Route to appropriate mapping method
        if source_type == "postcode":
            return self._map_postcode_to_sa1(geographic_id)
        elif source_type == "lga":
            return self._map_lga_to_sa1(geographic_id)
        elif source_type == "sa3":
            return self._map_sa3_to_sa1(geographic_id)
        elif source_type == "sa4":
            return self._map_sa4_to_sa1(geographic_id)
        elif source_type == "sa1":
            # Already SA1 level
            return [(geographic_id, 1.0)]
        else:
            logger.warning(f"Unknown source type '{source_type}' for {geographic_id}")
            return []

    def _detect_geographic_type(self, geographic_id: str) -> str:
        """Auto-detect the type of geographic identifier."""
        # Remove any non-alphanumeric characters
        clean_id = "".join(c for c in geographic_id if c.isalnum())

        # SA1 codes are 11 digits
        if len(clean_id) == 11 and clean_id.isdigit():
            return "sa1"

        # SA2 codes are 9 digits
        elif len(clean_id) == 9 and clean_id.isdigit():
            return "sa2"

        # SA3 codes are 5 digits
        elif len(clean_id) == 5 and clean_id.isdigit():
            return "sa3"

        # SA4 codes are 3 digits
        elif len(clean_id) == 3 and clean_id.isdigit():
            return "sa4"

        # Postcodes are usually 4 digits
        elif len(clean_id) == 4 and clean_id.isdigit():
            return "postcode"

        # LGA codes can vary
        elif len(clean_id) >= 3:
            return "lga"

        else:
            logger.warning(f"Could not detect type for geographic ID: {geographic_id}")
            return "unknown"

    def _map_postcode_to_sa1(self, postcode: str) -> list[tuple[str, float]]:
        """Map postcode to SA1 codes with population weights."""
        if self._postcode_mapping.empty:
            logger.warning(f"No postcode mapping available for {postcode}")
            return []

        matches = self._postcode_mapping[self._postcode_mapping["postcode"] == postcode]

        if matches.empty:
            logger.warning(f"No SA1 mappings found for postcode {postcode}")
            return []

        # Normalize weights to sum to 1.0
        total_weight = matches["population_weight"].sum()
        if total_weight > 0:
            matches = matches.copy()
            matches["normalized_weight"] = matches["population_weight"] / total_weight
            return list(zip(matches["sa1_code"], matches["normalized_weight"]))
        else:
            return []

    def _map_lga_to_sa1(self, lga_code: str) -> list[tuple[str, float]]:
        """Map LGA code to SA1 codes with population weights."""
        if self._lga_mapping.empty:
            logger.warning(f"No LGA mapping available for {lga_code}")
            return []

        matches = self._lga_mapping[self._lga_mapping["lga_code"] == lga_code]

        if matches.empty:
            logger.warning(f"No SA1 mappings found for LGA {lga_code}")
            return []

        # Normalize weights
        total_weight = matches["population_weight"].sum()
        if total_weight > 0:
            matches = matches.copy()
            matches["normalized_weight"] = matches["population_weight"] / total_weight
            return list(zip(matches["sa1_code"], matches["normalized_weight"]))
        else:
            return []

    def _map_sa3_to_sa1(self, sa3_code: str) -> list[tuple[str, float]]:
        """Map SA3 code to SA1 codes using hierarchical relationship."""
        if self._sa1_lookup.empty:
            logger.warning(f"No SA1 lookup available for SA3 {sa3_code}")
            return []

        # Filter SA1s within this SA3
        sa1s_in_sa3 = self._sa1_lookup[self._sa1_lookup["sa3_code"] == sa3_code]

        if sa1s_in_sa3.empty:
            logger.warning(f"No SA1s found for SA3 {sa3_code}")
            return []

        # Use population as weights
        total_population = sa1s_in_sa3["population_total"].sum()

        if total_population > 0:
            weights = sa1s_in_sa3["population_total"] / total_population
            return list(zip(sa1s_in_sa3["sa1_code"], weights))
        else:
            # Equal weights if no population data
            equal_weight = 1.0 / len(sa1s_in_sa3)
            return [(code, equal_weight) for code in sa1s_in_sa3["sa1_code"]]

    def _map_sa4_to_sa1(self, sa4_code: str) -> list[tuple[str, float]]:
        """Map SA4 code to SA1 codes using hierarchical relationship."""
        if self._sa1_lookup.empty:
            logger.warning(f"No SA1 lookup available for SA4 {sa4_code}")
            return []

        # Filter SA1s within this SA4
        sa1s_in_sa4 = self._sa1_lookup[self._sa1_lookup["sa4_code"] == sa4_code]

        if sa1s_in_sa4.empty:
            logger.warning(f"No SA1s found for SA4 {sa4_code}")
            return []

        # Use population as weights
        total_population = sa1s_in_sa4["population_total"].sum()

        if total_population > 0:
            weights = sa1s_in_sa4["population_total"] / total_population
            return list(zip(sa1s_in_sa4["sa1_code"], weights))
        else:
            # Equal weights if no population data
            equal_weight = 1.0 / len(sa1s_in_sa4)
            return [(code, equal_weight) for code in sa1s_in_sa4["sa1_code"]]

    def map_pha_to_sa1(self, pha_code: str) -> list[tuple[str, float]]:
        """
        Map Population Health Area (PHA) to SA1 codes.

        PHAs are PHIDU-specific geographic areas that require special mapping
        to SA1 level using concordance tables.
        """
        if self._pha_mapping is None:
            self._load_pha_mapping()

        if self._pha_mapping.empty:
            logger.warning(f"No PHA mapping available for {pha_code}")
            return []

        matches = self._pha_mapping[self._pha_mapping["pha_code"] == pha_code]

        if matches.empty:
            logger.warning(f"No SA1 mappings found for PHA {pha_code}")
            return []

        # Use mapping percentages as weights
        total_weight = matches["mapping_percentage"].sum()
        if total_weight > 0:
            matches = matches.copy()
            matches["normalized_weight"] = matches["mapping_percentage"] / total_weight
            return list(zip(matches["sa1_code"], matches["normalized_weight"]))
        else:
            return []

    def _load_pha_mapping(self):
        """Load PHA to SA1 mapping table."""
        try:
            conn = duckdb.connect(self.db_path)
            self._pha_mapping = conn.execute(
                """
                SELECT pha_code, sa1_code, mapping_percentage
                FROM pha_sa1_mapping
                WHERE mapping_percentage > 0
            """
            ).df()
            conn.close()
            logger.info(f"Loaded {len(self._pha_mapping)} PHA-SA1 mappings")
        except Exception as e:
            logger.warning(f"Could not load PHA mapping: {e}")
            self._pha_mapping = pd.DataFrame()

    def get_sa1_name(self, sa1_code: str) -> str:
        """Get the name for an SA1 code."""
        if self._sa1_lookup.empty:
            return f"SA1 {sa1_code}"

        match = self._sa1_lookup[self._sa1_lookup["sa1_code"] == sa1_code]

        if not match.empty:
            return match.iloc[0]["sa1_name"]
        else:
            return f"SA1 {sa1_code}"

    def get_sa1_hierarchy(self, sa1_code: str) -> dict[str, str]:
        """Get the full geographic hierarchy for an SA1 code."""
        if self._sa1_lookup.empty:
            return {}

        match = self._sa1_lookup[self._sa1_lookup["sa1_code"] == sa1_code]

        if not match.empty:
            row = match.iloc[0]
            return {
                "sa1_code": row["sa1_code"],
                "sa1_name": row["sa1_name"],
                "sa2_code": row["sa2_code"],
                "sa3_code": row["sa3_code"],
                "sa4_code": row["sa4_code"],
                "state_code": row["state_code"],
            }
        else:
            return {}

    def validate_sa1_code(self, sa1_code: str) -> bool:
        """Validate that an SA1 code exists and is properly formatted."""
        # Format validation
        if not sa1_code or len(sa1_code) != 11 or not sa1_code.isdigit():
            return False

        # Check if exists in lookup
        if not self._sa1_lookup.empty:
            return sa1_code in self._sa1_lookup["sa1_code"].values

        return True  # Assume valid if no lookup available


class PopulationWeighter:
    """
    Handles population-based weighting for disaggregating health data
    from larger geographic areas to SA1 level.
    """

    def __init__(self, db_path: str = "health_analytics.db"):
        self.db_path = db_path
        self._population_data = None
        self._load_population_data()

    def _load_population_data(self):
        """Load population data for SA1 areas."""
        try:
            conn = duckdb.connect(self.db_path)
            self._population_data = conn.execute(
                """
                SELECT sa1_code, population_total, population_density,
                       sa2_code, sa3_code, sa4_code
                FROM stg_sa1_boundaries
                WHERE population_total > 0
            """
            ).df()
            conn.close()
            logger.info(f"Loaded population data for {len(self._population_data)} SA1s")
        except Exception as e:
            logger.warning(f"Could not load population data: {e}")
            self._population_data = pd.DataFrame()

    def calculate_weights(self, sa1_codes: list[str], method: str = "population") -> list[float]:
        """
        Calculate weights for a list of SA1 codes.

        Args:
            sa1_codes: List of SA1 codes
            method: Weighting method ('population', 'equal', 'density')

        Returns:
            List of weights (sum to 1.0)
        """
        if not sa1_codes:
            return []

        if method == "equal":
            weight = 1.0 / len(sa1_codes)
            return [weight] * len(sa1_codes)

        if self._population_data.empty:
            logger.warning("No population data available, using equal weights")
            weight = 1.0 / len(sa1_codes)
            return [weight] * len(sa1_codes)

        if method == "population":
            populations = []
            for sa1_code in sa1_codes:
                match = self._population_data[self._population_data["sa1_code"] == sa1_code]
                if not match.empty:
                    populations.append(match.iloc[0]["population_total"])
                else:
                    populations.append(0)

            total_pop = sum(populations)
            if total_pop > 0:
                return [pop / total_pop for pop in populations]
            else:
                weight = 1.0 / len(sa1_codes)
                return [weight] * len(sa1_codes)

        elif method == "density":
            densities = []
            for sa1_code in sa1_codes:
                match = self._population_data[self._population_data["sa1_code"] == sa1_code]
                if not match.empty:
                    densities.append(match.iloc[0]["population_density"] or 1.0)
                else:
                    densities.append(1.0)

            total_density = sum(densities)
            return [density / total_density for density in densities]

        else:
            logger.warning(f"Unknown weighting method '{method}', using equal weights")
            weight = 1.0 / len(sa1_codes)
            return [weight] * len(sa1_codes)
