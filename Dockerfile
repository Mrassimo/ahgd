# Dockerfile for AHGD v2 Airflow Environment
FROM apache/airflow:2.8.1-python3.9

USER root
# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && rm -r /var/lib/apt/lists/*

USER airflow
# Install core data stack dependencies first
RUN pip install --no-cache-dir \
    polars \
    duckdb \
    dbt-duckdb \
    numpy

# Copy and install other requirements
COPY requirements.txt /requirements.txt
COPY requirements-dev.txt /requirements-dev.txt

RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install --no-cache-dir -r /requirements-dev.txt
