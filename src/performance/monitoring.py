"""
Performance Monitoring System for Australian Health Analytics Dashboard

Comprehensive performance monitoring including:
- Application performance metrics
- Database query performance tracking
- Memory usage monitoring
- Dashboard loading time metrics
- User interaction analytics
- Real-time performance visualization
"""

import time
import psutil
import threading
import logging
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from collections import defaultdict, deque
from datetime import datetime, timedelta
import functools
import weakref

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
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: Union[float, int, str]
    timestamp: datetime
    category: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class PerformanceAlert:
    """Performance alert configuration"""
    metric_name: str
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes
    last_triggered: Optional[datetime] = None


class MetricsCollector:
    """Collects and stores performance metrics"""
    
    def __init__(self, max_metrics: int = 10000, storage_path: Optional[Path] = None):
        self.max_metrics = max_metrics
        self.storage_path = storage_path
        self.metrics: deque = deque(maxlen=max_metrics)
        self.metrics_by_category: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable[[PerformanceAlert, PerformanceMetric], None]] = []
        self._lock = threading.Lock()
        
        # Initialize persistent storage if path provided
        if storage_path:
            self._init_storage()
    
    def _init_storage(self):
        """Initialize SQLite database for persistent metrics storage"""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(str(self.storage_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        value REAL,
                        timestamp TEXT NOT NULL,
                        category TEXT NOT NULL,
                        tags TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                    ON metrics(timestamp)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_category 
                    ON metrics(category)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_name 
                    ON metrics(name)
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize metrics storage: {e}")
            self.storage_path = None
    
    def add_metric(self, name: str, value: Union[float, int, str], 
                   category: str = "general", tags: Optional[Dict[str, str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetric:
        """Add a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            category=category,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics.append(metric)
            self.metrics_by_category[category].append(metric)
            
            # Store in persistent storage
            if self.storage_path:
                self._store_metric(metric)
            
            # Check alerts
            self._check_alerts(metric)
        
        return metric
    
    def _store_metric(self, metric: PerformanceMetric):
        """Store metric in persistent storage"""
        try:
            with sqlite3.connect(str(self.storage_path)) as conn:
                conn.execute("""
                    INSERT INTO metrics (name, value, timestamp, category, tags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric.name,
                    float(metric.value) if isinstance(metric.value, (int, float)) else None,
                    metric.timestamp.isoformat(),
                    metric.category,
                    json.dumps(metric.tags),
                    json.dumps(metric.metadata)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
    
    def get_metrics(self, category: Optional[str] = None, 
                   since: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[PerformanceMetric]:
        """Get metrics with optional filtering"""
        with self._lock:
            if category:
                metrics = list(self.metrics_by_category[category])
            else:
                metrics = list(self.metrics)
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            if limit:
                metrics = metrics[-limit:]
            
            return metrics
    
    def get_metrics_from_storage(self, category: Optional[str] = None,
                               since: Optional[datetime] = None,
                               limit: Optional[int] = 1000) -> List[Dict[str, Any]]:
        """Get metrics from persistent storage"""
        if not self.storage_path:
            return []
        
        try:
            with sqlite3.connect(str(self.storage_path)) as conn:
                query = "SELECT * FROM metrics WHERE 1=1"
                params = []
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                if since:
                    query += " AND timestamp >= ?"
                    params.append(since.isoformat())
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to get metrics from storage: {e}")
            return []
    
    def add_alert(self, alert: PerformanceAlert):
        """Add performance alert"""
        self.alerts.append(alert)
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert, PerformanceMetric], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def _check_alerts(self, metric: PerformanceMetric):
        """Check if metric triggers any alerts"""
        for alert in self.alerts:
            if not alert.enabled or alert.metric_name != metric.name:
                continue
            
            # Check cooldown
            if (alert.last_triggered and 
                datetime.now() - alert.last_triggered < timedelta(seconds=alert.cooldown_seconds)):
                continue
            
            # Check threshold
            if not isinstance(metric.value, (int, float)):
                continue
            
            triggered = False
            if alert.operator == '>' and metric.value > alert.threshold:
                triggered = True
            elif alert.operator == '<' and metric.value < alert.threshold:
                triggered = True
            elif alert.operator == '>=' and metric.value >= alert.threshold:
                triggered = True
            elif alert.operator == '<=' and metric.value <= alert.threshold:
                triggered = True
            elif alert.operator == '==' and metric.value == alert.threshold:
                triggered = True
            elif alert.operator == '!=' and metric.value != alert.threshold:
                triggered = True
            
            if triggered:
                alert.last_triggered = datetime.now()
                logger.warning(f"Performance alert triggered: {alert.message}")
                
                # Notify callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert, metric)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
    
    def get_summary(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """Get metrics summary"""
        with self._lock:
            metrics = self.get_metrics(since=since)
            
            if not metrics:
                return {'total_metrics': 0}
            
            categories = defaultdict(int)
            values_by_name = defaultdict(list)
            
            for metric in metrics:
                categories[metric.category] += 1
                if isinstance(metric.value, (int, float)):
                    values_by_name[metric.name].append(metric.value)
            
            # Calculate statistics
            stats = {}
            for name, values in values_by_name.items():
                if values:
                    stats[name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'latest': values[-1]
                    }
            
            return {
                'total_metrics': len(metrics),
                'categories': dict(categories),
                'time_range': {
                    'start': metrics[0].timestamp.isoformat() if metrics else None,
                    'end': metrics[-1].timestamp.isoformat() if metrics else None
                },
                'statistics': stats
            }


class SystemMetricsCollector:
    """Collects system-level performance metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.process = psutil.Process()
        self.collection_active = False
        self.collection_thread = None
        self.collection_interval = 5  # seconds
    
    def start_collection(self, interval: int = 5):
        """Start automatic system metrics collection"""
        if self.collection_active:
            return
        
        self.collection_interval = interval
        self.collection_active = True
        self.collection_thread = threading.Thread(target=self._collection_worker, daemon=True)
        self.collection_thread.start()
        logger.info("Started system metrics collection")
    
    def stop_collection(self):
        """Stop automatic system metrics collection"""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2)
        logger.info("Stopped system metrics collection")
    
    def _collection_worker(self):
        """Worker thread for collecting system metrics"""
        while self.collection_active:
            try:
                self.collect_current_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(self.collection_interval)
    
    def collect_current_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.add_metric("cpu_usage_percent", cpu_percent, "system")
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.add_metric("memory_usage_percent", memory.percent, "system")
            self.metrics_collector.add_metric("memory_available_mb", memory.available / 1024 / 1024, "system")
            self.metrics_collector.add_metric("memory_used_mb", memory.used / 1024 / 1024, "system")
            
            # Process-specific metrics
            with self.process.oneshot():
                proc_memory = self.process.memory_info()
                self.metrics_collector.add_metric("process_memory_rss_mb", proc_memory.rss / 1024 / 1024, "process")
                self.metrics_collector.add_metric("process_memory_vms_mb", proc_memory.vms / 1024 / 1024, "process")
                
                try:
                    proc_cpu = self.process.cpu_percent()
                    self.metrics_collector.add_metric("process_cpu_percent", proc_cpu, "process")
                except:
                    pass  # CPU percent might not be available immediately
                
                self.metrics_collector.add_metric("process_threads", self.process.num_threads(), "process")
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            self.metrics_collector.add_metric("disk_usage_percent", 
                                             disk_usage.used / disk_usage.total * 100, "system")
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class DatabaseMetricsCollector:
    """Collects database performance metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.query_times: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    @contextmanager
    def track_query(self, query_name: str, query_sql: Optional[str] = None):
        """Context manager to track database query performance"""
        start_time = time.time()
        error_occurred = False
        
        try:
            yield
        except Exception as e:
            error_occurred = True
            self.metrics_collector.add_metric(
                f"db_query_error_{query_name}", 1, "database",
                metadata={"error": str(e), "query": query_sql}
            )
            raise
        finally:
            duration = time.time() - start_time
            
            with self._lock:
                self.query_times[query_name].append(duration)
                # Keep only last 100 query times per query
                if len(self.query_times[query_name]) > 100:
                    self.query_times[query_name] = self.query_times[query_name][-100:]
            
            # Record metrics
            self.metrics_collector.add_metric(
                f"db_query_duration_{query_name}", duration, "database",
                metadata={"query": query_sql, "error": error_occurred}
            )
            
            # Record aggregate metrics
            avg_duration = sum(self.query_times[query_name]) / len(self.query_times[query_name])
            self.metrics_collector.add_metric(
                f"db_query_avg_duration_{query_name}", avg_duration, "database"
            )
    
    def get_query_stats(self) -> Dict[str, Dict[str, float]]:
        """Get query performance statistics"""
        with self._lock:
            stats = {}
            for query_name, times in self.query_times.items():
                if times:
                    stats[query_name] = {
                        'count': len(times),
                        'avg': sum(times) / len(times),
                        'min': min(times),
                        'max': max(times),
                        'recent_avg': sum(times[-10:]) / min(len(times), 10)
                    }
            return stats


class StreamlitMetricsCollector:
    """Collects Streamlit-specific performance metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.page_load_times: Dict[str, List[float]] = defaultdict(list)
        self.user_interactions: Dict[str, int] = defaultdict(int)
    
    @contextmanager
    def track_page_load(self, page_name: str):
        """Track page loading performance"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.page_load_times[page_name].append(duration)
            
            # Keep only last 50 load times
            if len(self.page_load_times[page_name]) > 50:
                self.page_load_times[page_name] = self.page_load_times[page_name][-50:]
            
            self.metrics_collector.add_metric(
                f"page_load_time_{page_name}", duration, "streamlit"
            )
            
            # Average load time
            avg_time = sum(self.page_load_times[page_name]) / len(self.page_load_times[page_name])
            self.metrics_collector.add_metric(
                f"page_avg_load_time_{page_name}", avg_time, "streamlit"
            )
    
    def track_user_interaction(self, interaction_type: str, component: str = ""):
        """Track user interactions"""
        key = f"{interaction_type}_{component}" if component else interaction_type
        self.user_interactions[key] += 1
        
        self.metrics_collector.add_metric(
            f"user_interaction_{key}", self.user_interactions[key], "streamlit"
        )
    
    def track_session_info(self):
        """Track Streamlit session information"""
        if not STREAMLIT_AVAILABLE:
            return
        
        try:
            # Session state size
            session_size = len(st.session_state.keys())
            self.metrics_collector.add_metric("session_state_size", session_size, "streamlit")
            
            # Estimate session memory usage
            session_memory = 0
            for key, value in st.session_state.items():
                try:
                    session_memory += len(str(value))
                except:
                    pass
            
            self.metrics_collector.add_metric("session_memory_estimate", session_memory, "streamlit")
            
        except Exception as e:
            logger.error(f"Error tracking session info: {e}")


class PerformanceMonitor:
    """Main performance monitoring coordinator"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.metrics_collector = MetricsCollector(storage_path=storage_path)
        self.system_collector = SystemMetricsCollector(self.metrics_collector)
        self.db_collector = DatabaseMetricsCollector(self.metrics_collector)
        self.streamlit_collector = StreamlitMetricsCollector(self.metrics_collector)
        
        # Performance thresholds and alerts
        self._setup_default_alerts()
        
        # Start automatic system monitoring
        self.start_monitoring()
    
    def _setup_default_alerts(self):
        """Setup default performance alerts"""
        alerts = [
            PerformanceAlert(
                metric_name="cpu_usage_percent",
                threshold=80.0,
                operator=">",
                severity="high",
                message="High CPU usage detected"
            ),
            PerformanceAlert(
                metric_name="memory_usage_percent",
                threshold=85.0,
                operator=">",
                severity="high",
                message="High memory usage detected"
            ),
            PerformanceAlert(
                metric_name="process_memory_rss_mb",
                threshold=1024.0,  # 1GB
                operator=">",
                severity="medium",
                message="Application using more than 1GB memory"
            )
        ]
        
        for alert in alerts:
            self.metrics_collector.add_alert(alert)
        
        # Add default alert callback
        self.metrics_collector.add_alert_callback(self._default_alert_handler)
    
    def _default_alert_handler(self, alert: PerformanceAlert, metric: PerformanceMetric):
        """Default alert handler"""
        logger.warning(f"PERFORMANCE ALERT: {alert.message} - {metric.name}: {metric.value}")
        
        # If running in Streamlit, show warning
        if STREAMLIT_AVAILABLE:
            try:
                st.warning(f"Performance Alert: {alert.message}")
            except:
                pass  # Might not be in Streamlit context
    
    def start_monitoring(self, system_interval: int = 5):
        """Start automatic performance monitoring"""
        self.system_collector.start_collection(interval=system_interval)
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop automatic performance monitoring"""
        self.system_collector.stop_collection()
        logger.info("Performance monitoring stopped")
    
    def track_function_performance(self, name: Optional[str] = None):
        """Decorator to track function performance"""
        def decorator(func):
            func_name = name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    success = True
                    return result
                except Exception as e:
                    success = False
                    self.metrics_collector.add_metric(
                        f"function_error_{func_name}", 1, "performance",
                        metadata={"error": str(e)}
                    )
                    raise
                finally:
                    duration = time.time() - start_time
                    self.metrics_collector.add_metric(
                        f"function_duration_{func_name}", duration, "performance",
                        metadata={"success": success}
                    )
            
            return wrapper
        return decorator
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        since = datetime.now() - timedelta(hours=hours)
        summary = self.metrics_collector.get_summary(since=since)
        
        # Add query statistics
        summary['database_queries'] = self.db_collector.get_query_stats()
        
        # Add system status
        try:
            summary['current_system'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').used / psutil.disk_usage('/').total * 100
            }
        except:
            summary['current_system'] = {'error': 'Could not get current system stats'}
        
        return summary
    
    def create_performance_dashboard(self) -> Optional[Any]:
        """Create performance monitoring dashboard"""
        if not STREAMLIT_AVAILABLE or not PLOTLY_AVAILABLE:
            logger.warning("Streamlit or Plotly not available for performance dashboard")
            return None
        
        try:
            st.header("Performance Monitoring Dashboard")
            
            # Get recent metrics
            since = datetime.now() - timedelta(hours=1)
            metrics = self.metrics_collector.get_metrics(since=since)
            
            if not metrics:
                st.warning("No performance metrics available")
                return
            
            # Convert to DataFrame for easier plotting
            if PANDAS_AVAILABLE:
                df_data = []
                for metric in metrics:
                    if isinstance(metric.value, (int, float)):
                        df_data.append({
                            'timestamp': metric.timestamp,
                            'name': metric.name,
                            'value': metric.value,
                            'category': metric.category
                        })
                
                if df_data:
                    df = pd.DataFrame(df_data)
                    
                    # System metrics
                    st.subheader("System Metrics")
                    system_metrics = df[df['category'] == 'system']
                    
                    if not system_metrics.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            cpu_data = system_metrics[system_metrics['name'] == 'cpu_usage_percent']
                            if not cpu_data.empty:
                                fig = px.line(cpu_data, x='timestamp', y='value', title='CPU Usage %')
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            memory_data = system_metrics[system_metrics['name'] == 'memory_usage_percent']
                            if not memory_data.empty:
                                fig = px.line(memory_data, x='timestamp', y='value', title='Memory Usage %')
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Database metrics
                    st.subheader("Database Performance")
                    db_metrics = df[df['category'] == 'database']
                    
                    if not db_metrics.empty:
                        query_duration_metrics = db_metrics[db_metrics['name'].str.contains('duration')]
                        if not query_duration_metrics.empty:
                            fig = px.scatter(query_duration_metrics, x='timestamp', y='value', 
                                           color='name', title='Query Duration Times')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance summary
                    st.subheader("Performance Summary")
                    summary = self.get_performance_summary()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Metrics", summary.get('total_metrics', 0))
                    with col2:
                        if 'current_system' in summary:
                            st.metric("Current CPU %", 
                                    f"{summary['current_system'].get('cpu_percent', 0):.1f}")
                    with col3:
                        if 'current_system' in summary:
                            st.metric("Current Memory %", 
                                    f"{summary['current_system'].get('memory_percent', 0):.1f}")
                    
                    # Detailed statistics
                    with st.expander("Detailed Statistics"):
                        st.json(summary)
            
        except Exception as e:
            st.error(f"Error creating performance dashboard: {e}")
            logger.error(f"Error creating performance dashboard: {e}")
    
    # Context managers for tracking
    def track_database_query(self, query_name: str, query_sql: Optional[str] = None):
        """Track database query performance"""
        return self.db_collector.track_query(query_name, query_sql)
    
    def track_page_load(self, page_name: str):
        """Track page loading performance"""
        return self.streamlit_collector.track_page_load(page_name)
    
    def track_user_interaction(self, interaction_type: str, component: str = ""):
        """Track user interaction"""
        self.streamlit_collector.track_user_interaction(interaction_type, component)
    
    def add_custom_metric(self, name: str, value: Union[float, int, str], 
                         category: str = "custom", **kwargs):
        """Add custom performance metric"""
        return self.metrics_collector.add_metric(name, value, category, **kwargs)


def create_performance_monitor(storage_path: Optional[Path] = None) -> PerformanceMonitor:
    """Factory function to create performance monitor"""
    return PerformanceMonitor(storage_path=storage_path)


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor(storage_path: Optional[Path] = None) -> PerformanceMonitor:
    """Get or create global performance monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = create_performance_monitor(storage_path)
    return _global_monitor


# Decorators for easy performance tracking
def track_performance(name: Optional[str] = None):
    """Decorator to track function performance"""
    monitor = get_performance_monitor()
    return monitor.track_function_performance(name)


if __name__ == "__main__":
    # Test performance monitoring
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        storage_path = Path(f.name)
    
    print("Testing performance monitoring...")
    
    monitor = create_performance_monitor(storage_path)
    
    # Test function tracking
    @monitor.track_function_performance()
    def test_function(n: int) -> int:
        time.sleep(0.1)
        return sum(range(n))
    
    # Test database tracking
    with monitor.track_database_query("test_query", "SELECT * FROM test"):
        time.sleep(0.05)
    
    # Run test function
    result = test_function(100)
    
    # Get summary
    time.sleep(6)  # Wait for system metrics
    summary = monitor.get_performance_summary()
    
    print(f"Performance summary: {json.dumps(summary, indent=2, default=str)}")
    
    # Cleanup
    monitor.stop_monitoring()
    storage_path.unlink()
    
    print("Performance monitoring test completed!")