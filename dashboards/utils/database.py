"""
Database connection utilities for AHGD dashboard.

This module provides connection management and query execution
functions for DuckDB database operations.
"""

import duckdb
import polars as pl
from pathlib import Path
from typing import Optional, Any
import streamlit as st


class DuckDBConnection:
    """Manages DuckDB connection with caching for Streamlit."""

    def __init__(self, db_path: str = "ahgd.db"):
        """
        Initialise DuckDB connection.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB database not found at {self.db_path}")

    @st.cache_resource
    def get_connection(_self):
        """
        Get cached DuckDB connection.

        Returns:
            duckdb.DuckDBPyConnection: Database connection
        """
        return duckdb.connect(str(_self.db_path), read_only=True)

    def query(self, sql: str, params: Optional[dict] = None) -> pl.DataFrame:
        """
        Execute SQL query and return results as Polars DataFrame.

        Args:
            sql: SQL query string
            params: Optional query parameters

        Returns:
            Polars DataFrame with query results
        """
        conn = self.get_connection()
        try:
            if params:
                result = conn.execute(sql, params).pl()
            else:
                result = conn.execute(sql).pl()
            return result
        except Exception as e:
            st.error(f"Query failed: {str(e)}")
            raise

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_health_summary(_self) -> dict:
        """
        Get summary statistics for health indicators.

        Returns:
            Dictionary with summary metrics
        """
        conn = _self.get_connection()

        summary = {
            "total_sa2": conn.execute(
                "SELECT COUNT(DISTINCT sa2_code) FROM master_health_record"
            ).fetchone()[0],
            "avg_mortality_rate": conn.execute(
                "SELECT AVG(mortality_rate) FROM master_health_record"
            ).fetchone()[0],
            "avg_utilisation_rate": conn.execute(
                "SELECT AVG(utilisation_rate) FROM master_health_record"
            ).fetchone()[0],
            "total_population": conn.execute(
                "SELECT SUM(total_population) FROM master_health_record"
            ).fetchone()[0],
            "avg_composite_index": conn.execute(
                "SELECT AVG(composite_health_index) FROM derived_health_indicators"
            ).fetchone()[0],
        }

        return summary

    @st.cache_data(ttl=300)
    def get_master_health_record(_self, limit: Optional[int] = None) -> pl.DataFrame:
        """
        Get master health record data.

        Args:
            limit: Optional limit on number of records

        Returns:
            Polars DataFrame with health record data
        """
        sql = "SELECT * FROM master_health_record"
        if limit:
            sql += f" LIMIT {limit}"

        return _self.query(sql)

    @st.cache_data(ttl=300)
    def get_derived_indicators(_self, limit: Optional[int] = None) -> pl.DataFrame:
        """
        Get derived health indicators.

        Args:
            limit: Optional limit on number of records

        Returns:
            Polars DataFrame with derived indicators
        """
        sql = "SELECT * FROM derived_health_indicators"
        if limit:
            sql += f" LIMIT {limit}"

        return _self.query(sql)

    @st.cache_data(ttl=300)
    def get_by_state(_self, state_code: str) -> pl.DataFrame:
        """
        Get health records filtered by state.

        Args:
            state_code: State code to filter by

        Returns:
            Polars DataFrame filtered by state
        """
        sql = """
            SELECT * FROM master_health_record
            WHERE state_code = ?
        """
        return _self.query(sql, {"state_code": state_code})

    @st.cache_data(ttl=300)
    def get_by_remoteness(_self, remoteness: str) -> pl.DataFrame:
        """
        Get health records filtered by remoteness category.

        Args:
            remoteness: Remoteness category

        Returns:
            Polars DataFrame filtered by remoteness
        """
        sql = """
            SELECT * FROM master_health_record
            WHERE remoteness_category = ?
        """
        return _self.query(sql, {"remoteness": remoteness})

    @st.cache_data(ttl=300)
    def get_states(_self) -> list:
        """
        Get list of unique states.

        Returns:
            List of state codes
        """
        conn = _self.get_connection()
        result = conn.execute(
            "SELECT DISTINCT state_code FROM master_health_record ORDER BY state_code"
        ).fetchall()
        return [row[0] for row in result if row[0]]

    @st.cache_data(ttl=300)
    def get_remoteness_categories(_self) -> list:
        """
        Get list of remoteness categories.

        Returns:
            List of remoteness categories
        """
        conn = _self.get_connection()
        result = conn.execute(
            """
            SELECT DISTINCT remoteness_category
            FROM master_health_record
            WHERE remoteness_category IS NOT NULL
            ORDER BY remoteness_category
        """
        ).fetchall()
        return [row[0] for row in result]

    @st.cache_data(ttl=300)
    def get_correlation_matrix(_self) -> pl.DataFrame:
        """
        Calculate correlation matrix for health and socioeconomic indicators.

        Returns:
            Polars DataFrame with correlation matrix
        """
        sql = """
            SELECT
                CORR(mortality_rate, median_household_income) as mortality_income_corr,
                CORR(mortality_rate, unemployment_rate) as mortality_unemployment_corr,
                CORR(mortality_rate, seifa_irsad_score) as mortality_seifa_corr,
                CORR(utilisation_rate, median_household_income) as utilisation_income_corr,
                CORR(utilisation_rate, unemployment_rate) as utilisation_unemployment_corr,
                CORR(utilisation_rate, seifa_irsad_score) as utilisation_seifa_corr
            FROM master_health_record
        """
        return _self.query(sql)

    @st.cache_data(ttl=300)
    def get_top_bottom_regions(_self, metric: str = "composite_health_index", n: int = 10) -> dict:
        """
        Get top and bottom performing regions by metric.

        Args:
            metric: Metric to rank by
            n: Number of regions to return

        Returns:
            Dictionary with 'top' and 'bottom' DataFrames
        """
        top_sql = f"""
            SELECT sa2_code, {metric}
            FROM derived_health_indicators
            ORDER BY {metric} DESC
            LIMIT {n}
        """

        bottom_sql = f"""
            SELECT sa2_code, {metric}
            FROM derived_health_indicators
            ORDER BY {metric} ASC
            LIMIT {n}
        """

        return {
            "top": _self.query(top_sql),
            "bottom": _self.query(bottom_sql),
        }

    def close(self):
        """Close database connection."""
        conn = self.get_connection()
        if conn:
            conn.close()


# Convenience function for getting connection
@st.cache_resource
def get_db_connection(db_path: str = "ahgd.db") -> DuckDBConnection:
    """
    Get cached database connection instance.

    Args:
        db_path: Path to DuckDB database

    Returns:
        DuckDBConnection instance
    """
    return DuckDBConnection(db_path)
