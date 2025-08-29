#!/bin/bash

# Initialize the Airflow database
airflow db init

# Create an admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start the Airflow webserver and scheduler
airflow webserver & airflow scheduler
