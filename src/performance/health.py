"""
Health Check and Production Deployment Features for Australian Health Analytics Dashboard

Features:
- Health check endpoints
- Graceful error handling and recovery
- Load balancer compatibility
- Session state management
- Resource usage optimization
- System status monitoring
"""

import os
import sys
import time
import json
import logging
import threading
import sqlite3
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
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
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'metadata': self.metadata
        }


@dataclass
class SystemStatus:
    """Overall system status"""
    status: HealthStatus
    checks: List[HealthCheck] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'status': self.status.value,
            'checks': [check.to_dict() for check in self.checks],
            'timestamp': self.timestamp.isoformat(),
            'uptime_seconds': self.uptime_seconds,
            'version': self.version,
            'summary': self.get_summary()
        }
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of check statuses"""
        summary = {status.value: 0 for status in HealthStatus}
        for check in self.checks:
            summary[check.status.value] += 1
        return summary


class DatabaseHealthChecker:
    """Database connectivity and performance health checker"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def check_connectivity(self) -> HealthCheck:
        """Check database connectivity"""
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path, timeout=5) as conn:
                cursor = conn.execute("SELECT 1")
                cursor.fetchone()
                
            duration = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="database_connectivity",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                duration_ms=duration
            )
            
        except sqlite3.OperationalError as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="database_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {e}",
                duration_ms=duration
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="database_connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Unexpected database error: {e}",
                duration_ms=duration
            )
    
    def check_performance(self) -> HealthCheck:
        """Check database query performance"""
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                # Run a sample query
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM sqlite_master WHERE type='table'
                """)
                table_count = cursor.fetchone()[0]
                
            duration = (time.time() - start_time) * 1000
            
            # Determine status based on query time
            if duration < 100:  # < 100ms
                status = HealthStatus.HEALTHY
                message = f"Database performance good ({duration:.1f}ms)"
            elif duration < 500:  # < 500ms
                status = HealthStatus.WARNING
                message = f"Database performance slow ({duration:.1f}ms)"
            else:  # >= 500ms
                status = HealthStatus.CRITICAL
                message = f"Database performance critical ({duration:.1f}ms)"
            
            return HealthCheck(
                name="database_performance",
                status=status,
                message=message,
                duration_ms=duration,
                metadata={'table_count': table_count}
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="database_performance",
                status=HealthStatus.CRITICAL,
                message=f"Database performance check failed: {e}",
                duration_ms=duration
            )
    
    def check_data_integrity(self) -> HealthCheck:
        """Check data integrity"""
        start_time = time.time()
        
        try:
            with sqlite3.connect(self.db_path, timeout=15) as conn:
                # Check for foreign key violations
                cursor = conn.execute("PRAGMA foreign_key_check")
                fk_violations = cursor.fetchall()
                
                # Check for corrupted data
                cursor = conn.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
                
            duration = (time.time() - start_time) * 1000
            
            if fk_violations or integrity_result != "ok":
                return HealthCheck(
                    name="database_integrity",
                    status=HealthStatus.CRITICAL,
                    message="Database integrity issues detected",
                    duration_ms=duration,
                    metadata={
                        'fk_violations': len(fk_violations),
                        'integrity_check': integrity_result
                    }
                )
            else:
                return HealthCheck(
                    name="database_integrity",
                    status=HealthStatus.HEALTHY,
                    message="Database integrity OK",
                    duration_ms=duration
                )
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="database_integrity",
                status=HealthStatus.CRITICAL,
                message=f"Database integrity check failed: {e}",
                duration_ms=duration
            )


class SystemResourceChecker:
    """System resource health checker"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def check_memory_usage(self) -> HealthCheck:
        """Check memory usage"""
        start_time = time.time()
        
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            
            # Process memory
            process_memory = self.process.memory_info()
            process_memory_mb = process_memory.rss / 1024 / 1024
            
            duration = (time.time() - start_time) * 1000
            
            # Determine status
            if system_memory.percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical system memory usage: {system_memory.percent:.1f}%"
            elif system_memory.percent > 80:
                status = HealthStatus.WARNING
                message = f"High system memory usage: {system_memory.percent:.1f}%"
            elif process_memory_mb > 2048:  # > 2GB
                status = HealthStatus.WARNING
                message = f"High process memory usage: {process_memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {system_memory.percent:.1f}% system, {process_memory_mb:.1f}MB process"
            
            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                duration_ms=duration,
                metadata={
                    'system_memory_percent': system_memory.percent,
                    'system_memory_available_mb': system_memory.available / 1024 / 1024,
                    'process_memory_mb': process_memory_mb,
                    'process_memory_percent': process_memory.rss / system_memory.total * 100
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {e}",
                duration_ms=duration
            )
    
    def check_cpu_usage(self) -> HealthCheck:
        """Check CPU usage"""
        start_time = time.time()
        
        try:
            # Get CPU usage (1 second interval)
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Process CPU usage
            try:
                process_cpu = self.process.cpu_percent()
            except:
                process_cpu = 0  # Might not be available immediately
            
            duration = (time.time() - start_time) * 1000
            
            # Determine status
            if cpu_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 80:
                status = HealthStatus.WARNING
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"CPU usage normal: {cpu_percent:.1f}%"
            
            return HealthCheck(
                name="cpu_usage",
                status=status,
                message=message,
                duration_ms=duration,
                metadata={
                    'system_cpu_percent': cpu_percent,
                    'process_cpu_percent': process_cpu,
                    'cpu_count': psutil.cpu_count()
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="cpu_usage",
                status=HealthStatus.UNKNOWN,
                message=f"CPU check failed: {e}",
                duration_ms=duration
            )
    
    def check_disk_space(self) -> HealthCheck:
        """Check disk space"""
        start_time = time.time()
        
        try:
            disk_usage = psutil.disk_usage('/')
            usage_percent = disk_usage.used / disk_usage.total * 100
            free_gb = disk_usage.free / 1024**3
            
            duration = (time.time() - start_time) * 1000
            
            # Determine status
            if usage_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Critical disk usage: {usage_percent:.1f}% ({free_gb:.1f}GB free)"
            elif usage_percent > 85:
                status = HealthStatus.WARNING
                message = f"High disk usage: {usage_percent:.1f}% ({free_gb:.1f}GB free)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk usage normal: {usage_percent:.1f}% ({free_gb:.1f}GB free)"
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                duration_ms=duration,
                metadata={
                    'usage_percent': usage_percent,
                    'free_gb': free_gb,
                    'total_gb': disk_usage.total / 1024**3,
                    'used_gb': disk_usage.used / 1024**3
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Disk check failed: {e}",
                duration_ms=duration
            )


class DataHealthChecker:
    """Data availability and quality health checker"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    def check_data_files(self) -> HealthCheck:
        """Check required data files existence"""
        start_time = time.time()
        
        try:
            required_files = [
                'sa2_boundaries_2021.parquet',
                'aihw_grim_data.parquet',
                'phidu_pha_data.parquet',
                'seifa_2021_sa2.parquet'
            ]
            
            processed_dir = self.data_dir / 'processed'
            missing_files = []
            file_sizes = {}
            
            for filename in required_files:
                file_path = processed_dir / filename
                if file_path.exists():
                    file_sizes[filename] = file_path.stat().st_size / 1024 / 1024  # MB
                else:
                    missing_files.append(filename)
            
            duration = (time.time() - start_time) * 1000
            
            if missing_files:
                return HealthCheck(
                    name="data_files",
                    status=HealthStatus.CRITICAL,
                    message=f"Missing data files: {', '.join(missing_files)}",
                    duration_ms=duration,
                    metadata={
                        'missing_files': missing_files,
                        'found_files': list(file_sizes.keys()),
                        'file_sizes_mb': file_sizes
                    }
                )
            else:
                total_size = sum(file_sizes.values())
                return HealthCheck(
                    name="data_files",
                    status=HealthStatus.HEALTHY,
                    message=f"All data files present ({total_size:.1f}MB total)",
                    duration_ms=duration,
                    metadata={
                        'file_count': len(required_files),
                        'total_size_mb': total_size,
                        'file_sizes_mb': file_sizes
                    }
                )
                
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="data_files",
                status=HealthStatus.UNKNOWN,
                message=f"Data file check failed: {e}",
                duration_ms=duration
            )
    
    def check_data_freshness(self) -> HealthCheck:
        """Check data freshness"""
        start_time = time.time()
        
        try:
            processed_dir = self.data_dir / 'processed'
            
            if not processed_dir.exists():
                duration = (time.time() - start_time) * 1000
                return HealthCheck(
                    name="data_freshness",
                    status=HealthStatus.CRITICAL,
                    message="Processed data directory not found",
                    duration_ms=duration
                )
            
            # Find newest file
            data_files = list(processed_dir.glob('*.parquet'))
            if not data_files:
                duration = (time.time() - start_time) * 1000
                return HealthCheck(
                    name="data_freshness",
                    status=HealthStatus.CRITICAL,
                    message="No data files found",
                    duration_ms=duration
                )
            
            newest_file = max(data_files, key=lambda p: p.stat().st_mtime)
            newest_time = datetime.fromtimestamp(newest_file.stat().st_mtime)
            age_hours = (datetime.now() - newest_time).total_seconds() / 3600
            
            duration = (time.time() - start_time) * 1000
            
            # Determine status based on age
            if age_hours < 24:  # < 1 day
                status = HealthStatus.HEALTHY
                message = f"Data is fresh (newest file: {age_hours:.1f} hours old)"
            elif age_hours < 168:  # < 1 week
                status = HealthStatus.WARNING
                message = f"Data is aging (newest file: {age_hours:.1f} hours old)"
            else:  # > 1 week
                status = HealthStatus.CRITICAL
                message = f"Data is stale (newest file: {age_hours:.1f} hours old)"
            
            return HealthCheck(
                name="data_freshness",
                status=status,
                message=message,
                duration_ms=duration,
                metadata={
                    'newest_file': newest_file.name,
                    'newest_file_age_hours': age_hours,
                    'newest_file_timestamp': newest_time.isoformat(),
                    'total_files': len(data_files)
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="data_freshness",
                status=HealthStatus.UNKNOWN,
                message=f"Data freshness check failed: {e}",
                duration_ms=duration
            )


class StreamlitHealthChecker:
    """Streamlit application health checker"""
    
    def check_session_state(self) -> HealthCheck:
        """Check session state health"""
        start_time = time.time()
        
        if not STREAMLIT_AVAILABLE:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="streamlit_session",
                status=HealthStatus.UNKNOWN,
                message="Streamlit not available",
                duration_ms=duration
            )
        
        try:
            session_size = len(st.session_state.keys())
            session_memory = 0
            
            # Estimate session memory usage
            for key, value in st.session_state.items():
                try:
                    session_memory += sys.getsizeof(value)
                except:
                    pass
            
            session_memory_mb = session_memory / 1024 / 1024
            
            duration = (time.time() - start_time) * 1000
            
            # Determine status
            if session_memory_mb > 100:  # > 100MB
                status = HealthStatus.WARNING
                message = f"Large session state: {session_size} keys, {session_memory_mb:.1f}MB"
            elif session_size > 100:  # > 100 keys
                status = HealthStatus.WARNING
                message = f"Many session keys: {session_size} keys, {session_memory_mb:.1f}MB"
            else:
                status = HealthStatus.HEALTHY
                message = f"Session state OK: {session_size} keys, {session_memory_mb:.1f}MB"
            
            return HealthCheck(
                name="streamlit_session",
                status=status,
                message=message,
                duration_ms=duration,
                metadata={
                    'session_keys': session_size,
                    'session_memory_mb': session_memory_mb,
                    'session_keys_list': list(st.session_state.keys())[:20]  # First 20 keys
                }
            )
            
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return HealthCheck(
                name="streamlit_session",
                status=HealthStatus.UNKNOWN,
                message=f"Session state check failed: {e}",
                duration_ms=duration
            )


class HealthChecker:
    """Main health checker coordinator"""
    
    def __init__(self, db_path: str, data_dir: Path):
        self.db_path = db_path
        self.data_dir = data_dir
        self.start_time = time.time()
        
        # Initialize individual checkers
        self.db_checker = DatabaseHealthChecker(db_path)
        self.system_checker = SystemResourceChecker()
        self.data_checker = DataHealthChecker(data_dir)
        self.streamlit_checker = StreamlitHealthChecker()
        
        # Health check history
        self.check_history: List[SystemStatus] = []
        self.max_history = 100
    
    def run_all_checks(self, include_slow_checks: bool = True) -> SystemStatus:
        """Run all health checks"""
        checks = []
        
        # Fast checks (always run)
        checks.extend([
            self.system_checker.check_memory_usage(),
            self.system_checker.check_disk_space(),
            self.db_checker.check_connectivity(),
            self.data_checker.check_data_files()
        ])
        
        # Slower checks (optional)
        if include_slow_checks:
            checks.extend([
                self.system_checker.check_cpu_usage(),  # Takes 1 second
                self.db_checker.check_performance(),
                self.db_checker.check_data_integrity(),
                self.data_checker.check_data_freshness(),
                self.streamlit_checker.check_session_state()
            ])
        
        # Determine overall status
        status_priority = {
            HealthStatus.CRITICAL: 3,
            HealthStatus.WARNING: 2,
            HealthStatus.HEALTHY: 1,
            HealthStatus.UNKNOWN: 0
        }
        
        overall_status = HealthStatus.HEALTHY
        for check in checks:
            if status_priority[check.status] > status_priority[overall_status]:
                overall_status = check.status
        
        system_status = SystemStatus(
            status=overall_status,
            checks=checks,
            uptime_seconds=time.time() - self.start_time
        )
        
        # Add to history
        self.check_history.append(system_status)
        if len(self.check_history) > self.max_history:
            self.check_history = self.check_history[-self.max_history:]
        
        return system_status
    
    def get_health_endpoint_response(self, include_slow_checks: bool = False) -> Dict[str, Any]:
        """Get health check response for HTTP endpoint"""
        try:
            status = self.run_all_checks(include_slow_checks)
            
            # Convert to response format
            response = status.to_dict()
            
            # Add HTTP status code
            if status.status == HealthStatus.HEALTHY:
                response['http_status'] = 200
            elif status.status == HealthStatus.WARNING:
                response['http_status'] = 200  # Still operational
            else:
                response['http_status'] = 503  # Service unavailable
            
            return response
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': HealthStatus.CRITICAL.value,
                'message': f"Health check system error: {e}",
                'timestamp': datetime.now().isoformat(),
                'http_status': 503
            }
    
    def get_readiness_check(self) -> Dict[str, Any]:
        """Get readiness check (for load balancers)"""
        try:
            # Only check critical components for readiness
            checks = [
                self.db_checker.check_connectivity(),
                self.data_checker.check_data_files()
            ]
            
            # Fail if any critical check fails
            ready = all(check.status in [HealthStatus.HEALTHY, HealthStatus.WARNING] 
                       for check in checks)
            
            return {
                'ready': ready,
                'status': 'ready' if ready else 'not_ready',
                'checks': [check.to_dict() for check in checks],
                'timestamp': datetime.now().isoformat(),
                'http_status': 200 if ready else 503
            }
            
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return {
                'ready': False,
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat(),
                'http_status': 503
            }
    
    def get_liveness_check(self) -> Dict[str, Any]:
        """Get liveness check (for container orchestration)"""
        try:
            # Simple check that the process is alive and responsive
            memory_check = self.system_checker.check_memory_usage()
            
            # Consider alive if we can check memory and it's not critical
            alive = memory_check.status != HealthStatus.CRITICAL
            
            return {
                'alive': alive,
                'status': 'alive' if alive else 'dead',
                'uptime_seconds': time.time() - self.start_time,
                'memory_status': memory_check.to_dict(),
                'timestamp': datetime.now().isoformat(),
                'http_status': 200 if alive else 503
            }
            
        except Exception as e:
            logger.error(f"Liveness check failed: {e}")
            return {
                'alive': False,
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat(),
                'http_status': 503
            }
    
    def create_health_dashboard(self):
        """Create health monitoring dashboard in Streamlit"""
        if not STREAMLIT_AVAILABLE:
            logger.warning("Streamlit not available for health dashboard")
            return
        
        try:
            st.header("System Health Dashboard")
            
            # Get current status
            status = self.run_all_checks(include_slow_checks=True)
            
            # Overall status
            status_colors = {
                HealthStatus.HEALTHY: "🟢",
                HealthStatus.WARNING: "🟡", 
                HealthStatus.CRITICAL: "🔴",
                HealthStatus.UNKNOWN: "⚪"
            }
            
            st.subheader(f"{status_colors[status.status]} Overall Status: {status.status.value.title()}")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Uptime", f"{status.uptime_seconds / 3600:.1f} hours")
            
            with col2:
                healthy_count = sum(1 for check in status.checks if check.status == HealthStatus.HEALTHY)
                st.metric("Healthy Checks", f"{healthy_count}/{len(status.checks)}")
            
            with col3:
                warning_count = sum(1 for check in status.checks if check.status == HealthStatus.WARNING)
                st.metric("Warnings", warning_count)
            
            with col4:
                critical_count = sum(1 for check in status.checks if check.status == HealthStatus.CRITICAL)
                st.metric("Critical Issues", critical_count)
            
            # Individual checks
            st.subheader("Health Checks")
            
            for check in status.checks:
                with st.expander(f"{status_colors[check.status]} {check.name.replace('_', ' ').title()}"):
                    st.write(f"**Status:** {check.status.value.title()}")
                    st.write(f"**Message:** {check.message}")
                    st.write(f"**Duration:** {check.duration_ms:.1f}ms")
                    
                    if check.metadata:
                        st.write("**Details:**")
                        st.json(check.metadata)
            
            # Health history
            if len(self.check_history) > 1:
                st.subheader("Health History")
                
                # Simple status over time
                history_data = []
                for historical_status in self.check_history[-24:]:  # Last 24 checks
                    history_data.append({
                        'timestamp': historical_status.timestamp,
                        'status': historical_status.status.value,
                        'healthy_checks': sum(1 for check in historical_status.checks 
                                            if check.status == HealthStatus.HEALTHY),
                        'total_checks': len(historical_status.checks)
                    })
                
                if PANDAS_AVAILABLE and history_data:
                    df = pd.DataFrame(history_data)
                    st.line_chart(df.set_index('timestamp')[['healthy_checks', 'total_checks']])
            
            # Quick actions
            st.subheader("Quick Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Refresh Health Checks"):
                    st.rerun()
            
            with col2:
                if st.button("Clear Session State"):
                    for key in list(st.session_state.keys()):
                        if not key.startswith('_'):  # Keep system keys
                            del st.session_state[key]
                    st.success("Session state cleared")
                    st.rerun()
            
        except Exception as e:
            st.error(f"Error creating health dashboard: {e}")
            logger.error(f"Error creating health dashboard: {e}")


def create_health_checker(db_path: str, data_dir: Path) -> HealthChecker:
    """Factory function to create health checker"""
    return HealthChecker(db_path, data_dir)


# Global health checker instance
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker(db_path: Optional[str] = None, 
                      data_dir: Optional[Path] = None) -> HealthChecker:
    """Get or create global health checker"""
    global _global_health_checker
    
    if _global_health_checker is None:
        if db_path is None or data_dir is None:
            # Try to get from config
            try:
                from ..config import get_global_config
                config = get_global_config()
                db_path = db_path or str(config.database.path)
                data_dir = data_dir or config.data_source.processed_data_dir.parent
            except:
                raise ValueError("Database path and data directory required")
        
        _global_health_checker = create_health_checker(db_path, data_dir)
    
    return _global_health_checker


if __name__ == "__main__":
    # Test health checking functionality
    import tempfile
    
    # Create temporary database and data directory
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        processed_dir = data_dir / 'processed'
        processed_dir.mkdir()
        
        # Create some test data files
        for filename in ['sa2_boundaries_2021.parquet', 'aihw_grim_data.parquet']:
            test_file = processed_dir / filename
            test_file.write_text("test data")
        
        # Create test database
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'test')")
        conn.commit()
        conn.close()
        
        print("Testing health checks...")
        
        health_checker = create_health_checker(db_path, data_dir)
        
        # Run health checks
        status = health_checker.run_all_checks()
        
        print(f"Overall status: {status.status.value}")
        print(f"Number of checks: {len(status.checks)}")
        
        for check in status.checks:
            print(f"  {check.name}: {check.status.value} - {check.message}")
        
        # Test endpoint responses
        health_response = health_checker.get_health_endpoint_response()
        readiness_response = health_checker.get_readiness_check()
        liveness_response = health_checker.get_liveness_check()
        
        print(f"\nHealth endpoint status: {health_response.get('http_status')}")
        print(f"Readiness: {readiness_response.get('ready')}")
        print(f"Liveness: {liveness_response.get('alive')}")
        
        # Cleanup
        Path(db_path).unlink()
        
        print("Health check test completed!")