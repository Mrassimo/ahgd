"""
⚠️  DEPRECATED: Legacy DLT Pipeline for Geographic Boundary Data

⚠️  This pandas-based pipeline has been REPLACED by polars_abs_extractor.py
⚠️  New extractor provides 10-100x performance improvement with Polars
⚠️  This file will be removed in a future version

For new implementations, use:
    from src.extractors.polars_abs_extractor import PolarsABSExtractor

Legacy functionality (DEPRECATED):
- SA1 boundaries (61,845 areas)
- SA2 boundaries (2,454 areas)
- Geographic relationships and hierarchies
- Spatial data processing and validation
"""

import logging

# Import Pydantic models for validation
import sys
import tempfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import dlt
import geopandas as gpd
import httpx
from shapely.validation import make_valid

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.geographic import SA1Boundary
from src.models.geographic import SA2Boundary

logger = logging.getLogger(__name__)


# ABS Data URLs (need to be updated with actual direct URLs)
SA1_BOUNDARIES_URL = "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA1_2021_AUST_GDA2020.zip"
SA2_BOUNDARIES_URL = "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/SA2_2021_AUST_SHP_GDA2020.zip"

# Chunk size for processing large datasets
CHUNK_SIZE = 5000  # Process 5000 SA1s at a time


@dlt.source(name="abs_geographic")
def geographic_boundaries_source():
    """
    DLT source for Australian geographic boundary data.

    Yields resources for SA1 and SA2 boundaries with full validation.
    """

    return [
        sa1_boundaries_resource(),
        sa2_boundaries_resource(),
        geographic_relationships_resource(),
    ]


@dlt.resource(
    name="sa1_boundaries",
    write_disposition="merge",
    primary_key="sa1_code",
    columns={
        "sa1_code": {"data_type": "text", "nullable": False},
        "geometry_wkt": {"data_type": "text"},
        "population_total": {"data_type": "bigint"},
        "area_sqkm": {"data_type": "double"},
    },
)
def sa1_boundaries_resource() -> Iterator[dict[str, Any]]:
    """
    Extract and process SA1 boundary data.

    Downloads SA1 boundaries, validates geometry, and yields
    records in chunks for efficient processing of 61K+ areas.
    """

    logger.info("Starting SA1 boundaries extraction")

    # Download and extract shapefile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download SA1 boundaries
        logger.info(f"Downloading SA1 boundaries from {SA1_BOUNDARIES_URL}")
        response = httpx.get(
            SA1_BOUNDARIES_URL,
            timeout=600,  # 10 minute timeout for large file
            follow_redirects=True,
        )
        response.raise_for_status()

        # Extract ZIP file
        zip_path = temp_path / "sa1_boundaries.zip"
        zip_path.write_bytes(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_path)

        # Find shapefile
        shapefiles = list(temp_path.glob("**/*.shp"))
        if not shapefiles:
            raise ValueError("No shapefile found in SA1 boundaries archive")

        shapefile_path = shapefiles[0]
        logger.info(f"Processing shapefile: {shapefile_path}")

        # Read with GeoPandas
        gdf = gpd.read_file(shapefile_path)
        logger.info(f"Loaded {len(gdf)} SA1 boundaries")

        # Process in chunks for memory efficiency
        for chunk_start in range(0, len(gdf), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(gdf))
            chunk = gdf.iloc[chunk_start:chunk_end]

            logger.info(f"Processing SA1 chunk {chunk_start}-{chunk_end}")

            for idx, row in chunk.iterrows():
                try:
                    # Validate and repair geometry if needed
                    geom = row.geometry
                    if not geom.is_valid:
                        geom = make_valid(geom)

                    # Convert to WKT for storage
                    geometry_wkt_str = geom.wkt

                    # Extract SA2 code from SA1 code (first 9 digits)
                    sa1_code = str(row.get("SA1_CODE21", row.get("SA1_MAIN16", "")))
                    sa2_code = sa1_code[:9] if len(sa1_code) >= 9 else None

                    # Create validated SA1 boundary record
                    sa1_data = {
                        "sa1_code": sa1_code,
                        "sa1_name": str(row.get("SA1_NAME21", sa1_code)),
                        "sa2_code": sa2_code,
                        "sa3_code": str(row.get("SA3_CODE21", ""))[:5],
                        "sa3_name": str(row.get("SA3_NAME21", "")),
                        "sa4_code": str(row.get("SA4_CODE21", ""))[:3],
                        "sa4_name": str(row.get("SA4_NAME21", "")),
                        "state_code": sa1_code[0] if sa1_code else None,
                        "state_name": str(row.get("STE_NAME21", "")),
                        "geographic_code": sa1_code,  # For base model
                        "geographic_name": str(row.get("SA1_NAME21", sa1_code)),
                        "area_sqkm": float(row.get("AREASQKM21", 0)),
                        "geometry_wkt": geometry_wkt_str,
                        "centroid_longitude": float(geom.centroid.x),
                        "centroid_latitude": float(geom.centroid.y),
                        "change_flag": str(row.get("CHG_FLAG21", "0")),
                        "change_label": str(row.get("CHG_LBL21", "")),
                    }

                    # Validate with Pydantic model
                    try:
                        validated = SA1Boundary(**sa1_data)
                        yield validated.model_dump()
                    except Exception as e:
                        logger.warning(f"Validation failed for SA1 {sa1_code}: {e}")
                        # Yield with data quality flag
                        sa1_data["has_missing_data"] = True
                        sa1_data["validation_errors"] = [str(e)]
                        yield sa1_data

                except Exception as e:
                    logger.error(f"Error processing SA1 boundary at index {idx}: {e}")
                    continue

        logger.info("Completed SA1 boundaries extraction")


@dlt.resource(
    name="sa2_boundaries",
    write_disposition="merge",
    primary_key="sa2_code",
    columns={
        "sa2_code": {"data_type": "text", "nullable": False},
        "geometry_wkt": {"data_type": "text"},
        "population_total": {"data_type": "bigint"},
        "area_sqkm": {"data_type": "double"},
    },
)
def sa2_boundaries_resource() -> Iterator[dict[str, Any]]:
    """
    Extract and process SA2 boundary data.

    Downloads SA2 boundaries and validates geometry for
    2,454 statistical areas.
    """

    logger.info("Starting SA2 boundaries extraction")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download SA2 boundaries
        logger.info(f"Downloading SA2 boundaries from {SA2_BOUNDARIES_URL}")
        response = httpx.get(
            SA2_BOUNDARIES_URL,
            timeout=300,  # 5 minute timeout
            follow_redirects=True,
        )
        response.raise_for_status()

        # Extract and process
        zip_path = temp_path / "sa2_boundaries.zip"
        zip_path.write_bytes(response.content)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_path)

        # Find shapefile
        shapefiles = list(temp_path.glob("**/*.shp"))
        if not shapefiles:
            raise ValueError("No shapefile found in SA2 boundaries archive")

        shapefile_path = shapefiles[0]
        logger.info(f"Processing shapefile: {shapefile_path}")

        # Read with GeoPandas
        gdf = gpd.read_file(shapefile_path)
        logger.info(f"Loaded {len(gdf)} SA2 boundaries")

        for idx, row in gdf.iterrows():
            try:
                # Validate geometry
                geom = row.geometry
                if not geom.is_valid:
                    geom = make_valid(geom)

                sa2_code = str(row.get("SA2_CODE21", ""))

                # Create SA2 boundary record
                sa2_data = {
                    "sa2_code": sa2_code,
                    "sa2_name": str(row.get("SA2_NAME21", "")),
                    "sa3_code": str(row.get("SA3_CODE21", ""))[:5],
                    "sa3_name": str(row.get("SA3_NAME21", "")),
                    "sa4_code": str(row.get("SA4_CODE21", ""))[:3],
                    "sa4_name": str(row.get("SA4_NAME21", "")),
                    "gcc_code": str(row.get("GCC_CODE21", "")),
                    "gcc_name": str(row.get("GCC_NAME21", "")),
                    "state_code": sa2_code[0] if sa2_code else None,
                    "state_name": str(row.get("STE_NAME21", "")),
                    "geographic_code": sa2_code,  # For base model
                    "geographic_name": str(row.get("SA2_NAME21", "")),
                    "area_sqkm": float(row.get("AREASQKM21", 0)),
                    "geometry_wkt": geom.wkt,
                    "centroid_longitude": float(geom.centroid.x),
                    "centroid_latitude": float(geom.centroid.y),
                    "change_flag": str(row.get("CHG_FLAG21", "0")),
                    "change_label": str(row.get("CHG_LBL21", "")),
                }

                # Validate with Pydantic
                try:
                    validated = SA2Boundary(**sa2_data)
                    yield validated.model_dump()
                except Exception as e:
                    logger.warning(f"Validation failed for SA2 {sa2_code}: {e}")
                    sa2_data["has_missing_data"] = True
                    sa2_data["validation_errors"] = [str(e)]
                    yield sa2_data

            except Exception as e:
                logger.error(f"Error processing SA2 boundary at index {idx}: {e}")
                continue

        logger.info("Completed SA2 boundaries extraction")


@dlt.resource(
    name="geographic_relationships",
    write_disposition="merge",
    primary_key=["source_code", "target_code"],
    columns={
        "source_code": {"data_type": "text", "nullable": False},
        "target_code": {"data_type": "text", "nullable": False},
        "relationship_type": {"data_type": "text"},
    },
)
def geographic_relationships_resource() -> Iterator[dict[str, Any]]:
    """
    Build geographic relationships between SA1s and SA2s.

    Creates mapping table for hierarchical aggregation and analysis.
    """

    logger.info("Building geographic relationships")

    # This would typically come from a correspondence file or be derived
    # from the SA1 codes themselves (SA2 code is first 9 digits of SA1)

    # For now, we'll build it from the SA1 boundaries we just loaded
    # In production, this would query the loaded SA1 data

    # Placeholder - in real implementation, would query the database
    # or use the SA1 boundaries already processed

    yield {
        "source_type": "SA1",
        "source_code": "PLACEHOLDER",
        "target_type": "SA2",
        "target_code": "PLACEHOLDER",
        "relationship_type": "exact",
        "allocation_percentage": 100.0,
        "geographic_code": "PLACEHOLDER",  # For base model
        "geographic_name": "Relationship",
        "state_code": "1",
        "state_name": "NSW",
    }

    logger.info("Completed geographic relationships")


def load_sa1_boundaries():
    """
    Main function to load SA1 boundary data.

    Called by the orchestrator to execute the SA1 boundaries pipeline.
    """

    # Configure DLT pipeline
    pipeline = dlt.pipeline(
        pipeline_name="sa1_boundaries",
        destination="duckdb",
        dataset_name="geographic_data",
        credentials="health_analytics.db",
    )

    # Run the pipeline
    source = geographic_boundaries_source()

    # Select only SA1 boundaries for this run
    sa1_resource = source.resources["sa1_boundaries"]

    info = pipeline.run(sa1_resource, loader_file_format="parquet", write_disposition="merge")

    logger.info(f"SA1 boundaries pipeline completed: {info}")

    return info


def load_sa2_boundaries():
    """
    Main function to load SA2 boundary data.

    Called by the orchestrator to execute the SA2 boundaries pipeline.
    """

    pipeline = dlt.pipeline(
        pipeline_name="sa2_boundaries",
        destination="duckdb",
        dataset_name="geographic_data",
        credentials="health_analytics.db",
    )

    source = geographic_boundaries_source()
    sa2_resource = source.resources["sa2_boundaries"]

    info = pipeline.run(sa2_resource, loader_file_format="parquet", write_disposition="merge")

    logger.info(f"SA2 boundaries pipeline completed: {info}")

    return info


def load_geographic_relationships():
    """
    Main function to load geographic relationship mappings.
    """

    pipeline = dlt.pipeline(
        pipeline_name="geographic_relationships",
        destination="duckdb",
        dataset_name="geographic_data",
        credentials="health_analytics.db",
    )

    source = geographic_boundaries_source()
    relationships_resource = source.resources["geographic_relationships"]

    info = pipeline.run(
        relationships_resource, loader_file_format="parquet", write_disposition="merge"
    )

    logger.info(f"Geographic relationships pipeline completed: {info}")

    return info


if __name__ == "__main__":
    # For testing - run SA1 boundaries pipeline
    logging.basicConfig(level=logging.INFO)
    load_sa1_boundaries()
