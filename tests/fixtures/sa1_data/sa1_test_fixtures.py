"""
SA1 test data fixtures and generators for AHGD testing.

This module provides utilities to generate realistic SA1 test data
following ABS 2021 standards and the SA1 schema validation rules.
"""

import random
from datetime import datetime
from typing import Any
from typing import Optional

import polars as pl

from schemas.base_schema import DataQualityLevel
from schemas.base_schema import DataSource
from schemas.base_schema import GeographicBoundary
from schemas.base_schema import SchemaVersion
from schemas.sa1_schema import SA1Coordinates


class SA1TestDataGenerator:
    """Generate realistic SA1 test data for validation and testing."""

    # Australian state/territory mapping
    STATE_MAPPINGS = {
        "1": {"code": "NSW", "name": "New South Wales"},
        "2": {"code": "VIC", "name": "Victoria"},
        "3": {"code": "QLD", "name": "Queensland"},
        "4": {"code": "SA", "name": "South Australia"},
        "5": {"code": "WA", "name": "Western Australia"},
        "6": {"code": "TAS", "name": "Tasmania"},
        "7": {"code": "NT", "name": "Northern Territory"},
        "8": {"code": "ACT", "name": "Australian Capital Territory"},
    }

    # Typical SA1 characteristics by remoteness category
    REMOTENESS_PROFILES = {
        "Major Cities": {
            "population_range": (300, 700),
            "area_range": (0.1, 3.0),
            "dwelling_ratio": 0.45,  # dwellings per person
        },
        "Inner Regional": {
            "population_range": (250, 600),
            "area_range": (1.0, 20.0),
            "dwelling_ratio": 0.48,
        },
        "Outer Regional": {
            "population_range": (200, 500),
            "area_range": (5.0, 100.0),
            "dwelling_ratio": 0.52,
        },
        "Remote": {
            "population_range": (150, 400),
            "area_range": (20.0, 1000.0),
            "dwelling_ratio": 0.55,
        },
        "Very Remote": {
            "population_range": (100, 300),
            "area_range": (50.0, 10000.0),
            "dwelling_ratio": 0.60,
        },
    }

    def __init__(self, seed: int = 42):
        """Initialise generator with random seed for reproducible results."""
        self.random = random.Random(seed)

    def generate_sa1_code(
        self,
        state_digit: str = None,
        sa4_code: str = None,
        sa3_code: str = None,
        sa2_code: str = None,
    ) -> str:
        """Generate a valid 11-digit SA1 code following ABS structure."""
        if not state_digit:
            state_digit = self.random.choice(list(self.STATE_MAPPINGS.keys()))

        if not sa4_code:
            # Generate 3-digit SA4 code (state + 2 digits)
            sa4_code = f"{state_digit}{self.random.randint(1, 99):02d}"

        if not sa3_code:
            # Generate 5-digit SA3 code (SA4 + 2 digits)
            sa3_code = f"{sa4_code}{self.random.randint(1, 99):02d}"

        if not sa2_code:
            # Generate 9-digit SA2 code (SA3 + 4 digits)
            sa2_code = f"{sa3_code}{self.random.randint(1, 9999):04d}"

        # Generate 11-digit SA1 code (SA2 + 2 digits)
        sa1_suffix = self.random.randint(1, 99)
        sa1_code = f"{sa2_code}{sa1_suffix:02d}"

        return sa1_code

    def generate_sa1_name(self, state_code: str, remoteness: str = "Major Cities") -> str:
        """Generate realistic SA1 name based on state and remoteness."""

        # Major city examples by state
        city_patterns = {
            "NSW": ["Sydney", "Newcastle", "Wollongong", "Central Coast"],
            "VIC": ["Melbourne", "Geelong", "Ballarat", "Bendigo"],
            "QLD": ["Brisbane", "Gold Coast", "Cairns", "Townsville"],
            "SA": ["Adelaide", "Mount Gambier", "Whyalla", "Port Augusta"],
            "WA": ["Perth", "Bunbury", "Geraldton", "Kalgoorlie"],
            "TAS": ["Hobart", "Launceston", "Devonport", "Burnie"],
            "NT": ["Darwin", "Alice Springs", "Katherine", "Tennant Creek"],
            "ACT": ["Canberra", "Tuggeranong", "Belconnen", "Weston Creek"],
        }

        suburbs = {
            "Major Cities": ["CBD", "Central", "East", "West", "North", "South"],
            "Inner Regional": ["Central", "East", "West", "Industrial", "Residential"],
            "Outer Regional": ["Central", "Rural", "Township", "Outskirts"],
            "Remote": ["Central", "Station", "Community", "Settlement"],
            "Very Remote": ["Community", "Station", "Outpost", "Remote"],
        }

        city = self.random.choice(city_patterns.get(state_code, ["Unknown"]))
        suburb = self.random.choice(suburbs[remoteness])

        return f"{city} - {suburb}"

    def generate_coordinates(self, state_code: str, remoteness: str) -> tuple[float, float]:
        """Generate realistic coordinates based on state and remoteness."""

        # Approximate coordinate bounds by state (centroid regions)
        state_bounds = {
            "NSW": {"lat": (-37.5, -28.0), "lon": (141.0, 154.0)},
            "VIC": {"lat": (-39.2, -34.0), "lon": (141.0, 150.0)},
            "QLD": {"lat": (-29.0, -9.0), "lon": (138.0, 154.0)},
            "SA": {"lat": (-38.0, -26.0), "lon": (129.0, 141.0)},
            "WA": {"lat": (-35.0, -13.8), "lon": (113.0, 129.0)},
            "TAS": {"lat": (-43.6, -40.6), "lon": (144.0, 148.5)},
            "NT": {"lat": (-26.0, -11.0), "lon": (129.0, 138.0)},
            "ACT": {"lat": (-35.9, -35.1), "lon": (148.7, 149.4)},
        }

        bounds = state_bounds.get(state_code, state_bounds["NSW"])

        # Adjust coordinates based on remoteness (more remote = more dispersed)
        if remoteness in ["Remote", "Very Remote"]:
            lat = self.random.uniform(bounds["lat"][0], bounds["lat"][1])
            lon = self.random.uniform(bounds["lon"][0], bounds["lon"][1])
        else:
            # Urban areas - concentrate around major cities
            lat_mid = sum(bounds["lat"]) / 2
            lon_mid = sum(bounds["lon"]) / 2
            lat_range = (bounds["lat"][1] - bounds["lat"][0]) * 0.3
            lon_range = (bounds["lon"][1] - bounds["lon"][0]) * 0.3

            lat = self.random.uniform(lat_mid - lat_range / 2, lat_mid + lat_range / 2)
            lon = self.random.uniform(lon_mid - lon_range / 2, lon_mid + lon_range / 2)

        return round(lat, 6), round(lon, 6)

    def generate_sa1_coordinates(
        self,
        state_code: Optional[str] = None,
        remoteness: Optional[str] = None,
        population: Optional[int] = None,
    ) -> SA1Coordinates:
        """Generate a complete SA1Coordinates object with realistic data."""

        # Select random state if not provided
        if not state_code:
            state_digit = self.random.choice(list(self.STATE_MAPPINGS.keys()))
            state_code = self.STATE_MAPPINGS[state_digit]["code"]
        else:
            state_digit = next(k for k, v in self.STATE_MAPPINGS.items() if v["code"] == state_code)

        # Select random remoteness if not provided
        if not remoteness:
            remoteness = self.random.choice(list(self.REMOTENESS_PROFILES.keys()))

        profile = self.REMOTENESS_PROFILES[remoteness]

        # Generate population if not provided
        if not population:
            population = self.random.randint(*profile["population_range"])

        # Generate area and dwellings
        area_sq_km = round(self.random.uniform(*profile["area_range"]), 3)
        dwellings = int(population * profile["dwelling_ratio"])

        # Generate coordinates
        lat, lon = self.generate_coordinates(state_code, remoteness)

        # Generate hierarchical codes
        sa1_code = self.generate_sa1_code(state_digit)
        sa2_code = sa1_code[:9]
        sa3_code = sa1_code[:5]
        sa4_code = sa1_code[:3]

        # Generate name
        sa1_name = self.generate_sa1_name(state_code, remoteness)

        # Create boundary data
        boundary_data = GeographicBoundary(
            boundary_id=sa1_code,
            boundary_type="SA1",
            name=sa1_name,
            state=state_code,
            area_sq_km=area_sq_km,
            centroid_lat=lat,
            centroid_lon=lon,
            geometry={
                "type": "Polygon",
                "coordinates": [
                    [
                        [lon - 0.01, lat - 0.01],
                        [lon + 0.01, lat - 0.01],
                        [lon + 0.01, lat + 0.01],
                        [lon - 0.01, lat + 0.01],
                        [lon - 0.01, lat - 0.01],
                    ]
                ],
            },
        )

        # Create data source
        data_source = DataSource(
            source_name="Australian Bureau of Statistics",
            source_url="https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3",
            source_date=datetime(2021, 7, 1),
            source_version="ASGS Edition 3",
            attribution="© Australian Bureau of Statistics 2021",
            license="Creative Commons Attribution 2.5 Australia",
        )

        return SA1Coordinates(
            sa1_code=sa1_code,
            sa1_name=sa1_name,
            boundary_data=boundary_data,
            population=population,
            dwellings=dwellings,
            sa2_code=sa2_code,
            sa3_code=sa3_code,
            sa4_code=sa4_code,
            state_code=state_code,
            remoteness_category=remoteness,
            data_source=data_source,
            schema_version=SchemaVersion.V2_0_0,
            data_quality=DataQualityLevel.HIGH,
        )

    def generate_test_dataset(self, count: int = 20, **kwargs) -> list[SA1Coordinates]:
        """Generate a dataset of SA1 coordinates for testing."""
        return [self.generate_sa1_coordinates(**kwargs) for _ in range(count)]

    def generate_polars_dataframe(self, count: int = 20, **kwargs) -> pl.DataFrame:
        """Generate SA1 test data as Polars DataFrame."""
        sa1_records = self.generate_test_dataset(count, **kwargs)

        records = []
        for sa1 in sa1_records:
            record = {
                "sa1_code": sa1.sa1_code,
                "sa1_name": sa1.sa1_name,
                "population": sa1.population,
                "dwellings": sa1.dwellings,
                "area_sq_km": sa1.boundary_data.area_sq_km,
                "centroid_lat": sa1.boundary_data.centroid_lat,
                "centroid_lon": sa1.boundary_data.centroid_lon,
                "sa2_code": sa1.sa2_code,
                "sa3_code": sa1.sa3_code,
                "sa4_code": sa1.sa4_code,
                "state_code": sa1.state_code,
                "remoteness_category": sa1.remoteness_category,
            }
            records.append(record)

        return pl.DataFrame(records)


def get_sample_sa1_data() -> dict[str, Any]:
    """Get sample SA1 data for basic validation tests."""
    return {
        "sa1_code": "10102100701",
        "sa1_name": "Sydney - Haymarket - The Rocks (Test)",
        "boundary_data": {
            "boundary_id": "10102100701",
            "boundary_type": "SA1",
            "name": "Sydney - Haymarket - The Rocks (Test)",
            "state": "NSW",
            "area_sq_km": 0.85,
            "centroid_lat": -33.8688,
            "centroid_lon": 151.2093,
        },
        "population": 420,
        "dwellings": 185,
        "sa2_code": "101021007",
        "sa3_code": "10102",
        "sa4_code": "101",
        "state_code": "NSW",
        "remoteness_category": "Major Cities",
        "data_source": {
            "source_name": "Australian Bureau of Statistics",
            "source_url": "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3",
            "source_date": "2021-07-01T00:00:00",
            "source_version": "ASGS Edition 3",
            "attribution": "© Australian Bureau of Statistics 2021",
            "license": "Creative Commons Attribution 2.5 Australia",
        },
    }


def validate_test_data(sa1_data: dict[str, Any]) -> list[str]:
    """Validate SA1 test data and return any errors."""
    try:
        sa1 = SA1Coordinates(**sa1_data)
        return sa1.validate_data_integrity()
    except Exception as e:
        return [f"Validation error: {e!s}"]


# Pre-defined test cases for common scenarios
TEST_CASES = {
    "urban_major_city": {
        "state_code": "NSW",
        "remoteness": "Major Cities",
        "population": 450,
    },
    "regional_town": {
        "state_code": "VIC",
        "remoteness": "Inner Regional",
        "population": 350,
    },
    "remote_community": {"state_code": "WA", "remoteness": "Remote", "population": 250},
    "very_remote": {"state_code": "NT", "remoteness": "Very Remote", "population": 180},
    "small_population": {"population": 150},
    "large_population": {"population": 750},
}
