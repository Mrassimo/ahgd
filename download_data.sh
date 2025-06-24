#!/bin/bash
# AHGD Production Data Acquisition Script V3
# This script uses curl with flags to handle redirects and output to named files.

echo "--- Creating download directory ---"
mkdir -p data_downloads
cd data_downloads

echo "--- Downloading ABS Census 2021 DataPack (~49MB) ---"
curl -L "https://www.abs.gov.au/census/find-census-data/datapacks/download/2021_GCP_ALL_for_AUS_short-header.zip" -o "census_datapack.zip"

echo "--- Downloading ABS 2021 SA2 Geographic Boundaries (~48MB) ---"
curl -L "https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files/ASGS_2021_SA2_SHP_GDA2020.zip" -o "sa2_boundaries.zip"

echo "--- Downloading ABS 2021 SEIFA Data ---"
curl -L "https://www.abs.gov.au/statistics/people/people-and-communities/socio-economic-indexes-areas-seifa-australia/2021/seifa-2021-sa2-indexes.xlsx" -o "seifa_indexes.xlsx"

echo "--- Download script finished. Verifying files... ---"
ls -lh