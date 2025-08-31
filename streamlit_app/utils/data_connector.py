"""
AHGD V3: High-Performance Data Connector for Streamlit
DuckDB-based data access layer providing fast queries for dashboard components.

Features:
- Optimized DuckDB queries with Polars integration
- Caching for improved performance
- Geographic data aggregation
- Health metrics calculations
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import streamlit as st

import polars as pl
import duckdb
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from utils.logging import get_logger


@st.cache_resource
def get_duckdb_connection():
    """Create cached DuckDB connection for Streamlit app."""
    db_path = os.getenv("DUCKDB_PATH", "./duckdb_data/ahgd_v3.db")
    
    try:
        conn = duckdb.connect(db_path)
        
        # Optimize for dashboard queries
        conn.execute("SET memory_limit='2GB'")
        conn.execute("SET threads=2")  # Conservative for Streamlit
        conn.execute("SET enable_progress_bar=false")
        
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None


class DuckDBConnector:
    """High-performance data connector for AHGD dashboard."""
    
    def __init__(self):
        """Initialize connector with optimized DuckDB connection."""
        self.logger = get_logger("streamlit_data_connector")
        self.connection = get_duckdb_connection()
        
        if self.connection is None:
            st.error("❌ Database connection failed")
            st.stop()
        
        self.logger.info("DuckDB connector initialized for Streamlit")

    def check_connection(self) -> bool:
        """Check if database connection is healthy."""
        try:
            if self.connection:
                result = self.connection.execute("SELECT 1").fetchone()
                return result[0] == 1
        except:
            pass
        return False

    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_available_areas(_self, geographic_level: str) -> List[str]:
        """
        Get list of available geographic areas for selection.
        
        Args:
            geographic_level: Geographic level (state, sa4, sa3, sa2, sa1)
            
        Returns:
            List of available area codes/names
        """
        try:
            if geographic_level == 'state':
                query = """
                    SELECT DISTINCT state_name
                    FROM marts.mart_sa1_health_profile 
                    WHERE state_name IS NOT NULL
                    ORDER BY state_name
                """
            elif geographic_level == 'sa4':
                query = """
                    SELECT DISTINCT sa4_name
                    FROM marts.mart_sa1_health_profile 
                    WHERE sa4_name IS NOT NULL
                    ORDER BY sa4_name
                """
            elif geographic_level == 'sa3':
                query = """
                    SELECT DISTINCT sa3_name
                    FROM marts.mart_sa1_health_profile 
                    WHERE sa3_name IS NOT NULL
                    ORDER BY sa3_name
                """
            elif geographic_level == 'sa2':
                query = """
                    SELECT DISTINCT sa2_code, sa2_name
                    FROM marts.mart_sa1_health_profile 
                    WHERE sa2_code IS NOT NULL
                    ORDER BY sa2_name
                """
            else:  # sa1
                query = """
                    SELECT DISTINCT sa1_code, sa1_name
                    FROM marts.mart_sa1_health_profile 
                    WHERE sa1_code IS NOT NULL
                    ORDER BY sa1_name
                    LIMIT 1000  -- Limit SA1 for performance
                """
            
            result = _self.connection.execute(query).pl()
            
            if geographic_level in ['sa2', 'sa1']:
                # Return code-name pairs for lower levels
                return [f"{row[0]} - {row[1]}" for row in result.rows()]
            else:
                # Return names for higher levels
                return result.get_column(0).to_list()
                
        except Exception as e:
            _self.logger.error(f"Error fetching areas for {geographic_level}: {str(e)}")
            return []

    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def get_summary_metrics(
        _self,
        geographic_level: str,
        selected_areas: List[str],
        health_metric: str,
        date_range: Tuple[int, int]
    ) -> Optional[pl.DataFrame]:
        """
        Get summary health metrics for dashboard overview.
        
        Args:
            geographic_level: Geographic aggregation level
            selected_areas: List of selected area names/codes
            health_metric: Primary health metric to analyze
            date_range: Year range tuple (start, end)
            
        Returns:
            Polars DataFrame with summary statistics
        """
        try:
            # Build WHERE clause for area selection
            where_clause = "WHERE 1=1"
            
            if selected_areas:
                if geographic_level == 'state':
                    area_filter = "'" + "','".join(selected_areas) + "'"
                    where_clause += f" AND state_name IN ({area_filter})"
                elif geographic_level == 'sa4':
                    area_filter = "'" + "','".join(selected_areas) + "'"
                    where_clause += f" AND sa4_name IN ({area_filter})"
                # Add more geographic level filters as needed
            
            query = f"""
                SELECT 
                    sa1_code,
                    sa1_name,
                    state_name,
                    total_population,
                    {health_metric},
                    data_completeness_score,
                    health_vulnerability_index,
                    healthcare_access_category
                FROM marts.mart_sa1_health_profile
                {where_clause}
                AND {health_metric} IS NOT NULL
                ORDER BY {health_metric} DESC
            """
            
            result = _self.connection.execute(query).pl()
            
            _self.logger.info(f"Retrieved {result.height} records for summary metrics")
            return result
            
        except Exception as e:
            _self.logger.error(f"Error getting summary metrics: {str(e)}")
            return None

    @st.cache_data(ttl=300)
    def get_geographic_data(
        _self,
        geographic_level: str,
        health_metric: str,
        selected_areas: List[str] = None
    ) -> Optional[pl.DataFrame]:
        """
        Get geographic boundary data with health metrics for mapping.
        
        Args:
            geographic_level: Geographic level for aggregation
            health_metric: Health metric to include
            selected_areas: Optional area filter
            
        Returns:
            DataFrame with geographic and health data
        """
        try:
            # Aggregation logic based on geographic level
            if geographic_level == 'state':
                agg_query = f"""
                    SELECT 
                        state_name,
                        AVG(centroid_longitude) as centroid_longitude,
                        AVG(centroid_latitude) as centroid_latitude,
                        AVG({health_metric}) as {health_metric},
                        SUM(total_population) as total_population,
                        AVG(health_vulnerability_index) as health_vulnerability_index
                    FROM marts.mart_sa1_health_profile
                    WHERE {health_metric} IS NOT NULL
                    GROUP BY state_name
                """
            else:
                # SA1 level data
                agg_query = f"""
                    SELECT 
                        sa1_code,
                        sa1_name,
                        state_name,
                        centroid_longitude,
                        centroid_latitude,
                        {health_metric},
                        total_population,
                        health_vulnerability_index
                    FROM marts.mart_sa1_health_profile
                    WHERE {health_metric} IS NOT NULL
                    LIMIT 5000  -- Limit for map performance
                """
            
            result = _self.connection.execute(agg_query).pl()
            
            _self.logger.info(f"Retrieved geographic data: {result.height} areas")
            return result
            
        except Exception as e:
            _self.logger.error(f"Error getting geographic data: {str(e)}")
            return None

    @st.cache_data(ttl=600)
    def get_temporal_trends(
        _self,
        geographic_level: str,
        selected_areas: List[str],
        health_metric: str,
        date_range: Tuple[int, int]
    ) -> Optional[pl.DataFrame]:
        """
        Get temporal trends data for health metrics.
        
        Note: This is a placeholder as the current data model doesn't include
        temporal data. In a full implementation, this would query historical tables.
        """
        try:
            # Generate sample temporal data for demonstration
            # In production, this would query actual historical tables
            years = list(range(date_range[0], date_range[1] + 1))
            
            # Get current metrics and simulate temporal variation
            current_data = _self.get_summary_metrics(
                geographic_level, selected_areas, health_metric, date_range
            )
            
            if current_data is None or current_data.height == 0:
                return None
            
            # Create simulated temporal data
            temporal_data = []
            base_value = current_data.select(pl.col(health_metric).mean()).item()
            
            for year in years:
                # Simple simulation - in reality, query historical tables
                variation = 0.95 + (year - date_range[0]) * 0.02  # Small upward trend
                temporal_data.append({
                    'year': year,
                    health_metric: base_value * variation
                })
            
            return pl.DataFrame(temporal_data)
            
        except Exception as e:
            _self.logger.error(f"Error getting temporal trends: {str(e)}")
            return None

    @st.cache_data(ttl=600)
    def get_correlation_matrix(
        _self,
        selected_areas: List[str] = None
    ) -> Optional[pl.DataFrame]:
        """
        Calculate correlation matrix for health indicators.
        
        Args:
            selected_areas: Optional area filter
            
        Returns:
            Correlation matrix as DataFrame
        """
        try:
            # Health metrics for correlation analysis
            health_metrics = [
                'diabetes_prevalence_rate',
                'mental_health_service_rate', 
                'cardiovascular_disease_rate',
                'gp_visits_per_capita_annual',
                'irsd_score',
                'health_vulnerability_index'
            ]
            
            # Build correlation query
            select_columns = [f"COALESCE({metric}, 0) as {metric}" for metric in health_metrics]
            
            query = f"""
                SELECT {', '.join(select_columns)}
                FROM marts.mart_sa1_health_profile
                WHERE diabetes_prevalence_rate IS NOT NULL
                LIMIT 10000  -- Performance limit
            """
            
            data = _self.connection.execute(query).pl()
            
            if data.height == 0:
                return None
            
            # Calculate correlation matrix using Polars
            correlation_data = {}
            for metric1 in health_metrics:
                correlation_data[metric1] = []
                for metric2 in health_metrics:
                    if metric1 in data.columns and metric2 in data.columns:
                        corr = data.select([
                            pl.corr(metric1, metric2).alias('correlation')
                        ]).item()
                        correlation_data[metric1].append(corr if corr is not None else 0)
                    else:
                        correlation_data[metric1].append(0)
            
            return pl.DataFrame(correlation_data)
            
        except Exception as e:
            _self.logger.error(f"Error calculating correlation matrix: {str(e)}")
            return None

    @st.cache_data(ttl=60)  # Short cache for exports
    def get_export_data(
        _self,
        geographic_level: str,
        selected_areas: List[str] = None,
        health_metric: str = None,
        date_range: Tuple[int, int] = None
    ) -> Optional[pl.DataFrame]:
        """
        Get comprehensive data for export functionality.
        
        Args:
            geographic_level: Geographic aggregation level
            selected_areas: Optional area selection
            health_metric: Optional specific health metric
            date_range: Optional date range
            
        Returns:
            Complete dataset for export
        """
        try:
            # Build comprehensive export query
            where_clauses = ["1=1"]
            
            if selected_areas and geographic_level == 'state':
                area_filter = "'" + "','".join(selected_areas) + "'"
                where_clauses.append(f"state_name IN ({area_filter})")
            
            where_clause = " AND ".join(where_clauses)
            
            export_query = f"""
                SELECT 
                    sa1_code,
                    sa1_name,
                    sa2_code,
                    sa3_code,
                    sa4_code,
                    state_name,
                    total_population,
                    median_age,
                    median_income_weekly,
                    diabetes_prevalence_rate,
                    mental_health_service_rate,
                    cardiovascular_disease_rate,
                    gp_visits_per_capita_annual,
                    irsd_score,
                    irsd_decile,
                    health_vulnerability_index,
                    healthcare_access_category,
                    data_completeness_score,
                    last_updated
                FROM marts.mart_sa1_health_profile
                WHERE {where_clause}
                ORDER BY state_name, sa1_name
            """
            
            result = _self.connection.execute(export_query).pl()
            
            _self.logger.info(f"Prepared export data: {result.height} records")
            return result
            
        except Exception as e:
            _self.logger.error(f"Error preparing export data: {str(e)}")
            return None

    def get_data_freshness(self) -> Dict[str, Any]:
        """Get information about data freshness and update status."""
        try:
            freshness_query = """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN diabetes_prevalence_rate IS NOT NULL THEN 1 END) as diabetes_records,
                    COUNT(CASE WHEN mental_health_service_rate IS NOT NULL THEN 1 END) as mental_health_records,
                    MAX(last_updated) as last_update,
                    AVG(data_completeness_score) as avg_completeness
                FROM marts.mart_sa1_health_profile
            """
            
            result = self.connection.execute(freshness_query).fetchone()
            
            return {
                'total_records': result[0],
                'diabetes_coverage': result[1] / result[0] if result[0] > 0 else 0,
                'mental_health_coverage': result[2] / result[0] if result[0] > 0 else 0,
                'last_updated': result[3],
                'avg_completeness': result[4]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting data freshness: {str(e)}")
            return {}

    def __del__(self):
        """Clean up database connection."""
        if hasattr(self, 'connection') and self.connection:
            try:
                self.connection.close()
            except:
                pass