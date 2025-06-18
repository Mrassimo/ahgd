"""
Performance Monitoring Dashboard for Australian Health Analytics Dashboard

Comprehensive dashboard for visualizing:
- Performance metrics and trends
- System health status
- Alert history and status
- Resource utilization
- Database performance
- User engagement analytics
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .monitoring import PerformanceMonitor, get_performance_monitor
from .cache import CacheManager, get_cache_manager
from .health import HealthChecker, HealthStatus, get_health_checker
from .alerts import AlertManager, get_alert_manager, AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for performance dashboard"""
    refresh_interval_seconds: int = 30
    max_data_points: int = 100
    show_debug_info: bool = False
    auto_refresh: bool = True
    theme: str = "streamlit"  # streamlit, dark, light


class PerformanceDashboard:
    """Main performance monitoring dashboard"""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        
        # Initialize components
        self.monitor = get_performance_monitor()
        self.cache_manager = get_cache_manager()
        self.health_checker = None  # Will be initialized when needed
        self.alert_manager = get_alert_manager()
        
        # Dashboard state
        self.last_refresh = datetime.now()
        self.dashboard_start_time = datetime.now()
    
    def render_dashboard(self):
        """Render the complete performance dashboard"""
        if not STREAMLIT_AVAILABLE:
            logger.error("Streamlit not available for dashboard rendering")
            return
        
        try:
            # Dashboard header
            st.set_page_config(
                page_title="Performance Monitoring",
                page_icon="📊",
                layout="wide"
            )
            
            st.title("🚀 Performance Monitoring Dashboard")
            st.markdown("Real-time monitoring of Australian Health Analytics Dashboard performance")
            
            # Control panel
            self._render_control_panel()
            
            # Main dashboard sections
            self._render_overview_section()
            self._render_system_metrics_section()
            self._render_performance_metrics_section()
            self._render_health_status_section()
            self._render_alerts_section()
            self._render_cache_metrics_section()
            self._render_database_performance_section()
            
            # Debug section (if enabled)
            if self.config.show_debug_info:
                self._render_debug_section()
            
            # Auto-refresh
            if self.config.auto_refresh:
                time.sleep(self.config.refresh_interval_seconds)
                st.rerun()
                
        except Exception as e:
            st.error(f"Error rendering performance dashboard: {e}")
            logger.error(f"Dashboard rendering error: {e}")
    
    def _render_control_panel(self):
        """Render dashboard control panel"""
        with st.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Refresh Now"):
                    st.rerun()
            
            with col2:
                auto_refresh = st.checkbox("Auto Refresh", value=self.config.auto_refresh)
                self.config.auto_refresh = auto_refresh
            
            with col3:
                refresh_interval = st.selectbox(
                    "Refresh Interval",
                    [10, 30, 60, 120],
                    index=1,
                    format_func=lambda x: f"{x}s"
                )
                self.config.refresh_interval_seconds = refresh_interval
            
            with col4:
                show_debug = st.checkbox("Show Debug Info", value=self.config.show_debug_info)
                self.config.show_debug_info = show_debug
    
    def _render_overview_section(self):
        """Render system overview section"""
        st.header("📈 System Overview")
        
        # Get current system status
        try:
            summary = self.monitor.get_performance_summary(hours=1)
            cache_stats = self.cache_manager.get_stats()
            alert_stats = self.alert_manager.get_alert_statistics()
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                uptime_hours = (datetime.now() - self.dashboard_start_time).total_seconds() / 3600
                st.metric("Uptime", f"{uptime_hours:.1f}h")
            
            with col2:
                total_metrics = summary.get('total_metrics', 0)
                st.metric("Metrics Collected", f"{total_metrics:,}")
            
            with col3:
                cache_hit_rate = cache_stats.get('hit_rate', 0)
                st.metric("Cache Hit Rate", f"{cache_hit_rate:.1%}")
            
            with col4:
                active_alerts = alert_stats.get('active_alerts', 0)
                color = "normal" if active_alerts == 0 else "inverse"
                st.metric("Active Alerts", active_alerts, delta_color=color)
            
            # System status indicators
            st.subheader("System Status")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Performance status
                current_system = summary.get('current_system', {})
                cpu_percent = current_system.get('cpu_percent', 0)
                
                if cpu_percent < 70:
                    st.success(f"🟢 CPU: {cpu_percent:.1f}% (Good)")
                elif cpu_percent < 85:
                    st.warning(f"🟡 CPU: {cpu_percent:.1f}% (High)")
                else:
                    st.error(f"🔴 CPU: {cpu_percent:.1f}% (Critical)")
            
            with col2:
                # Memory status
                memory_percent = current_system.get('memory_percent', 0)
                
                if memory_percent < 80:
                    st.success(f"🟢 Memory: {memory_percent:.1f}% (Good)")
                elif memory_percent < 90:
                    st.warning(f"🟡 Memory: {memory_percent:.1f}% (High)")
                else:
                    st.error(f"🔴 Memory: {memory_percent:.1f}% (Critical)")
            
            with col3:
                # Cache status
                cache_backends = len(cache_stats.get('backends', []))
                
                if cache_backends > 0:
                    st.success(f"🟢 Cache: {cache_backends} backends active")
                else:
                    st.warning("🟡 Cache: No backends active")
                    
        except Exception as e:
            st.error(f"Error loading overview: {e}")
    
    def _render_system_metrics_section(self):
        """Render system metrics section"""
        st.header("💻 System Metrics")
        
        try:
            # Get recent metrics
            since = datetime.now() - timedelta(hours=1)
            metrics = self.monitor.metrics_collector.get_metrics(category='system', since=since)
            
            if not metrics:
                st.warning("No system metrics available")
                return
            
            # Convert to DataFrame
            if PANDAS_AVAILABLE and PLOTLY_AVAILABLE:
                metric_data = []
                for metric in metrics:
                    if isinstance(metric.value, (int, float)):
                        metric_data.append({
                            'timestamp': metric.timestamp,
                            'name': metric.name,
                            'value': metric.value
                        })
                
                if metric_data:
                    df = pd.DataFrame(metric_data)
                    
                    # CPU and Memory charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        cpu_data = df[df['name'] == 'cpu_usage_percent']
                        if not cpu_data.empty:
                            fig = px.line(
                                cpu_data, 
                                x='timestamp', 
                                y='value',
                                title='CPU Usage Over Time',
                                labels={'value': 'CPU %', 'timestamp': 'Time'}
                            )
                            fig.add_hline(y=80, line_dash="dash", line_color="orange", 
                                        annotation_text="High Usage (80%)")
                            fig.add_hline(y=95, line_dash="dash", line_color="red",
                                        annotation_text="Critical (95%)")
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        memory_data = df[df['name'] == 'memory_usage_percent']
                        if not memory_data.empty:
                            fig = px.line(
                                memory_data,
                                x='timestamp',
                                y='value',
                                title='Memory Usage Over Time',
                                labels={'value': 'Memory %', 'timestamp': 'Time'}
                            )
                            fig.add_hline(y=80, line_dash="dash", line_color="orange",
                                        annotation_text="High Usage (80%)")
                            fig.add_hline(y=90, line_dash="dash", line_color="red",
                                        annotation_text="Critical (90%)")
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Process metrics
                    st.subheader("Process Metrics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        process_memory_data = df[df['name'] == 'process_memory_rss_mb']
                        if not process_memory_data.empty:
                            fig = px.area(
                                process_memory_data,
                                x='timestamp',
                                y='value',
                                title='Process Memory Usage (RSS)',
                                labels={'value': 'Memory (MB)', 'timestamp': 'Time'}
                            )
                            fig.update_layout(height=250)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        thread_data = df[df['name'] == 'process_threads']
                        if not thread_data.empty:
                            fig = px.line(
                                thread_data,
                                x='timestamp',
                                y='value',
                                title='Process Thread Count',
                                labels={'value': 'Threads', 'timestamp': 'Time'}
                            )
                            fig.update_layout(height=250)
                            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading system metrics: {e}")
    
    def _render_performance_metrics_section(self):
        """Render application performance metrics"""
        st.header("⚡ Application Performance")
        
        try:
            # Get performance metrics
            since = datetime.now() - timedelta(hours=1)
            perf_metrics = self.monitor.metrics_collector.get_metrics(category='performance', since=since)
            streamlit_metrics = self.monitor.metrics_collector.get_metrics(category='streamlit', since=since)
            
            if not perf_metrics and not streamlit_metrics:
                st.info("No performance metrics available yet. Metrics will appear as you use the application.")
                return
            
            # Function performance metrics
            if perf_metrics and PANDAS_AVAILABLE and PLOTLY_AVAILABLE:
                perf_data = []
                for metric in perf_metrics:
                    if isinstance(metric.value, (int, float)) and 'function_duration_' in metric.name:
                        func_name = metric.name.replace('function_duration_', '')
                        perf_data.append({
                            'timestamp': metric.timestamp,
                            'function': func_name,
                            'duration': metric.value * 1000  # Convert to ms
                        })
                
                if perf_data:
                    df_perf = pd.DataFrame(perf_data)
                    
                    st.subheader("Function Performance")
                    
                    # Function duration scatter plot
                    fig = px.scatter(
                        df_perf,
                        x='timestamp',
                        y='duration',
                        color='function',
                        title='Function Execution Times',
                        labels={'duration': 'Duration (ms)', 'timestamp': 'Time'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Function performance summary
                    func_summary = df_perf.groupby('function')['duration'].agg(['count', 'mean', 'max']).round(2)
                    func_summary.columns = ['Calls', 'Avg Duration (ms)', 'Max Duration (ms)']
                    st.dataframe(func_summary, use_container_width=True)
            
            # Streamlit metrics
            if streamlit_metrics and PANDAS_AVAILABLE and PLOTLY_AVAILABLE:
                st.subheader("Page Performance")
                
                page_data = []
                for metric in streamlit_metrics:
                    if isinstance(metric.value, (int, float)) and 'page_load_time_' in metric.name:
                        page_name = metric.name.replace('page_load_time_', '')
                        page_data.append({
                            'timestamp': metric.timestamp,
                            'page': page_name,
                            'load_time': metric.value
                        })
                
                if page_data:
                    df_pages = pd.DataFrame(page_data)
                    
                    # Page load times
                    fig = px.box(
                        df_pages,
                        x='page',
                        y='load_time',
                        title='Page Load Time Distribution',
                        labels={'load_time': 'Load Time (seconds)', 'page': 'Page'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading performance metrics: {e}")
    
    def _render_health_status_section(self):
        """Render health status section"""
        st.header("🏥 Health Status")
        
        try:
            # Initialize health checker if needed
            if self.health_checker is None:
                try:
                    self.health_checker = get_health_checker()
                except Exception as e:
                    st.error(f"Could not initialize health checker: {e}")
                    return
            
            # Get current health status
            health_status = self.health_checker.run_all_checks(include_slow_checks=False)
            
            # Overall status
            status_colors = {
                HealthStatus.HEALTHY: "🟢",
                HealthStatus.WARNING: "🟡",
                HealthStatus.CRITICAL: "🔴",
                HealthStatus.UNKNOWN: "⚪"
            }
            
            status_color = status_colors.get(health_status.status, "⚪")
            st.subheader(f"{status_color} Overall Health: {health_status.status.value.title()}")
            
            # Health checks summary
            col1, col2, col3, col4 = st.columns(4)
            
            summary = health_status.get_summary()
            
            with col1:
                st.metric("Healthy", summary.get('healthy', 0))
            
            with col2:
                st.metric("Warnings", summary.get('warning', 0))
            
            with col3:
                st.metric("Critical", summary.get('critical', 0))
            
            with col4:
                st.metric("Unknown", summary.get('unknown', 0))
            
            # Individual health checks
            st.subheader("Health Check Details")
            
            for check in health_status.checks:
                status_icon = status_colors.get(check.status, "⚪")
                
                with st.expander(f"{status_icon} {check.name.replace('_', ' ').title()}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Status:** {check.status.value.title()}")
                        st.write(f"**Message:** {check.message}")
                        
                        if check.metadata:
                            st.write("**Details:**")
                            for key, value in check.metadata.items():
                                st.write(f"  • {key}: {value}")
                    
                    with col2:
                        st.metric("Duration", f"{check.duration_ms:.1f}ms")
                        st.write(f"**Time:** {check.timestamp.strftime('%H:%M:%S')}")
        
        except Exception as e:
            st.error(f"Error loading health status: {e}")
    
    def _render_alerts_section(self):
        """Render alerts section"""
        st.header("🚨 Alerts & Notifications")
        
        try:
            # Get alert statistics
            alert_stats = self.alert_manager.get_alert_statistics()
            
            # Alert summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rules", alert_stats.get('total_rules', 0))
            
            with col2:
                st.metric("Enabled Rules", alert_stats.get('enabled_rules', 0))
            
            with col3:
                st.metric("Active Alerts", alert_stats.get('active_alerts', 0))
            
            with col4:
                st.metric("Rules Triggered (24h)", alert_stats.get('rules_triggered_24h', 0))
            
            # Alerts by severity
            st.subheader("Alerts by Severity (24h)")
            
            severity_data = alert_stats.get('alerts_by_severity', {})
            if severity_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Severity chart
                    if PLOTLY_AVAILABLE:
                        fig = px.pie(
                            values=list(severity_data.values()),
                            names=list(severity_data.keys()),
                            title="Alert Severity Distribution"
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Severity table
                    severity_df = pd.DataFrame([
                        {'Severity': k.title(), 'Count': v}
                        for k, v in severity_data.items()
                    ]) if PANDAS_AVAILABLE else None
                    
                    if severity_df is not None:
                        st.dataframe(severity_df, use_container_width=True)
            
            # Recent alerts
            st.subheader("Recent Alert Rules")
            
            recent_rules = alert_stats.get('recent_rules', [])
            if recent_rules:
                for rule_name in recent_rules[-5:]:  # Last 5 rules
                    st.write(f"• {rule_name}")
            else:
                st.info("No recent alerts")
            
            # Alert configuration
            with st.expander("Alert Configuration"):
                available_channels = alert_stats.get('available_channels', [])
                st.write(f"**Available Channels:** {', '.join(available_channels)}")
                
                if st.button("Test Alert System"):
                    # Create a test alert
                    from .monitoring import PerformanceMetric
                    test_metric = PerformanceMetric(
                        name="test_metric",
                        value=100,
                        timestamp=datetime.now(),
                        category="test"
                    )
                    
                    self.alert_manager.evaluate_metric(test_metric)
                    st.success("Test alert sent!")
        
        except Exception as e:
            st.error(f"Error loading alerts section: {e}")
    
    def _render_cache_metrics_section(self):
        """Render cache performance metrics"""
        st.header("💾 Cache Performance")
        
        try:
            # Get cache statistics
            cache_stats = self.cache_manager.get_stats()
            
            # Cache overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                hit_rate = cache_stats.get('hit_rate', 0)
                st.metric("Hit Rate", f"{hit_rate:.1%}")
            
            with col2:
                total_hits = cache_stats.get('metrics', {}).get('hits', 0)
                st.metric("Total Hits", f"{total_hits:,}")
            
            with col3:
                total_misses = cache_stats.get('metrics', {}).get('misses', 0)
                st.metric("Total Misses", f"{total_misses:,}")
            
            with col4:
                backends = cache_stats.get('backends', [])
                st.metric("Active Backends", len(backends))
            
            # Backend details
            st.subheader("Cache Backend Details")
            
            backend_stats = cache_stats.get('backend_stats', {})
            
            for backend_name, stats in backend_stats.items():
                if isinstance(stats, dict) and 'error' not in stats:
                    with st.expander(f"{backend_name.title()} Cache"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'entries' in stats:
                                st.metric("Entries", stats['entries'])
                            
                            if 'total_size_bytes' in stats:
                                size_mb = stats['total_size_bytes'] / 1024 / 1024
                                st.metric("Size", f"{size_mb:.1f} MB")
                        
                        with col2:
                            if 'hit_rate' in stats:
                                st.metric("Hit Rate", f"{stats['hit_rate']:.1%}")
                            
                            if 'compression_ratio' in stats:
                                st.metric("Compression", f"{stats['compression_ratio']:.1%}")
                        
                        # Additional stats
                        st.json(stats)
        
        except Exception as e:
            st.error(f"Error loading cache metrics: {e}")
    
    def _render_database_performance_section(self):
        """Render database performance metrics"""
        st.header("🗄️ Database Performance")
        
        try:
            # Get database metrics
            since = datetime.now() - timedelta(hours=1)
            db_metrics = self.monitor.metrics_collector.get_metrics(category='database', since=since)
            
            if not db_metrics:
                st.info("No database metrics available yet.")
                return
            
            if PANDAS_AVAILABLE and PLOTLY_AVAILABLE:
                # Process database metrics
                query_data = []
                for metric in db_metrics:
                    if isinstance(metric.value, (int, float)) and 'db_query_duration_' in metric.name:
                        query_name = metric.name.replace('db_query_duration_', '')
                        query_data.append({
                            'timestamp': metric.timestamp,
                            'query': query_name,
                            'duration': metric.value * 1000  # Convert to ms
                        })
                
                if query_data:
                    df_queries = pd.DataFrame(query_data)
                    
                    # Query performance chart
                    fig = px.scatter(
                        df_queries,
                        x='timestamp',
                        y='duration',
                        color='query',
                        title='Database Query Performance',
                        labels={'duration': 'Duration (ms)', 'timestamp': 'Time'}
                    )
                    fig.add_hline(y=1000, line_dash="dash", line_color="orange",
                                annotation_text="Slow Query Threshold (1s)")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Query summary
                    st.subheader("Query Performance Summary")
                    query_summary = df_queries.groupby('query')['duration'].agg(['count', 'mean', 'max']).round(2)
                    query_summary.columns = ['Executions', 'Avg Duration (ms)', 'Max Duration (ms)']
                    st.dataframe(query_summary, use_container_width=True)
                    
                    # Slow queries
                    slow_queries = df_queries[df_queries['duration'] > 1000]
                    if not slow_queries.empty:
                        st.subheader("⚠️ Slow Queries (>1s)")
                        st.dataframe(slow_queries, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error loading database metrics: {e}")
    
    def _render_debug_section(self):
        """Render debug information section"""
        st.header("🔧 Debug Information")
        
        try:
            with st.expander("System Information"):
                import sys
                import platform
                
                debug_info = {
                    'Python Version': sys.version,
                    'Platform': platform.platform(),
                    'CPU Count': os.cpu_count() if hasattr(os, 'cpu_count') else 'Unknown',
                    'Current Working Directory': str(Path.cwd()),
                    'Dashboard Start Time': self.dashboard_start_time.isoformat(),
                    'Last Refresh': self.last_refresh.isoformat()
                }
                
                st.json(debug_info)
            
            with st.expander("Component Status"):
                component_status = {
                    'Performance Monitor': 'Active' if self.monitor else 'Inactive',
                    'Cache Manager': 'Active' if self.cache_manager else 'Inactive',
                    'Health Checker': 'Active' if self.health_checker else 'Inactive',
                    'Alert Manager': 'Active' if self.alert_manager else 'Inactive',
                    'Streamlit Available': STREAMLIT_AVAILABLE,
                    'Pandas Available': PANDAS_AVAILABLE,
                    'Plotly Available': PLOTLY_AVAILABLE
                }
                
                st.json(component_status)
            
            with st.expander("Session State"):
                if STREAMLIT_AVAILABLE:
                    session_info = {
                        'Session State Keys': len(st.session_state.keys()),
                        'Keys': list(st.session_state.keys())[:20]  # First 20 keys
                    }
                    st.json(session_info)
        
        except Exception as e:
            st.error(f"Error loading debug information: {e}")


def create_performance_dashboard_page():
    """Create standalone performance dashboard page"""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not available for dashboard")
        return
    
    dashboard = PerformanceDashboard()
    dashboard.render_dashboard()


def render_performance_sidebar():
    """Render performance monitoring sidebar widget"""
    if not STREAMLIT_AVAILABLE:
        return
    
    try:
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Performance")
        
        # Quick performance indicators
        monitor = get_performance_monitor()
        summary = monitor.get_performance_summary(hours=1)
        
        # CPU indicator
        current_system = summary.get('current_system', {})
        cpu_percent = current_system.get('cpu_percent', 0)
        
        if cpu_percent < 70:
            cpu_status = f"🟢 {cpu_percent:.0f}%"
        elif cpu_percent < 85:
            cpu_status = f"🟡 {cpu_percent:.0f}%"
        else:
            cpu_status = f"🔴 {cpu_percent:.0f}%"
        
        st.sidebar.metric("CPU", cpu_status)
        
        # Memory indicator
        memory_percent = current_system.get('memory_percent', 0)
        
        if memory_percent < 80:
            memory_status = f"🟢 {memory_percent:.0f}%"
        elif memory_percent < 90:
            memory_status = f"🟡 {memory_percent:.0f}%"
        else:
            memory_status = f"🔴 {memory_percent:.0f}%"
        
        st.sidebar.metric("Memory", memory_status)
        
        # Cache hit rate
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_stats()
        hit_rate = cache_stats.get('hit_rate', 0)
        st.sidebar.metric("Cache Hit Rate", f"{hit_rate:.0%}")
        
        # Performance dashboard link
        if st.sidebar.button("📊 Open Performance Dashboard"):
            # This would typically open the dashboard in a new tab/page
            # For now, we'll show a message
            st.sidebar.success("Performance dashboard would open here")
    
    except Exception as e:
        st.sidebar.error(f"Performance widget error: {e}")


if __name__ == "__main__":
    # Test the dashboard
    create_performance_dashboard_page()