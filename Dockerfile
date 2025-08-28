# Dockerfile for AHGD v2 Airflow Environment
FROM apache/airflow:2.8.1-python3.9

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && rm -r /var/lib/apt/lists/*
USER airflow

COPY requirements.txt /requirements.txt
COPY requirements-dev.txt /requirements-dev.txt

RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install --no-cache-dir -r /requirements-dev.txt
