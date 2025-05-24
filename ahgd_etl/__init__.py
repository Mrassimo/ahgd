"""Australian Healthcare Geographic Database (AHGD) ETL Pipeline.

A robust, automated ETL pipeline for processing Australian Bureau of Statistics
(ABS) geographic and Census data into a dimensional data warehouse optimized
for healthcare analytics.
"""

__version__ = "1.0.0"
__author__ = "AHGD ETL Team"

from .config import get_settings

__all__ = ["get_settings"]