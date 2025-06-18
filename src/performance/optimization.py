"""
Query Optimization and Data Loading System for Australian Health Analytics Dashboard

Features:
- Database query optimization and result caching
- Lazy loading for large datasets
- Pagination for large result sets
- Background data processing
- Compressed data storage formats
- Intelligent data prefetching
"""

import sqlite3
import threading
import asyncio
import time
import logging
import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Iterator, Generator
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, Future
from functools import wraps, lru_cache
import weakref
from abc import ABC, abstractmethod

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from .cache import CacheManager, cached
from .monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """Database query execution plan"""
    query: str
    parameters: List[Any] = field(default_factory=list)
    estimated_rows: Optional[int] = None
    estimated_cost: Optional[float] = None
    cache_key: Optional[str] = None
    ttl: Optional[int] = None
    indexes_used: List[str] = field(default_factory=list)
    optimization_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaginationConfig:
    """Configuration for paginated data loading"""
    page_size: int = 1000
    max_pages: Optional[int] = None
    prefetch_pages: int = 2
    cache_pages: bool = True
    lazy_load: bool = True


class QueryOptimizer:
    """Database query optimizer with caching and performance monitoring"""
    
    def __init__(self, db_path: str, cache_manager: Optional[CacheManager] = None,
                 monitor: Optional[PerformanceMonitor] = None):
        self.db_path = db_path
        self.cache_manager = cache_manager
        self.monitor = monitor
        self.connection_pool: List[sqlite3.Connection] = []
        self.pool_lock = threading.Lock()
        self.query_stats: Dict[str, Dict[str, float]] = {}
        self.prepared_statements: Dict[str, str] = {}
        
        # Initialize connection pool
        self._init_connection_pool()
        
        # Query optimization settings
        self.enable_query_cache = True
        self.default_cache_ttl = 3600  # 1 hour
        self.max_result_size_mb = 100
    
    def _init_connection_pool(self, pool_size: int = 5):
        """Initialize database connection pool"""
        try:
            for _ in range(pool_size):
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                
                # Enable query optimization
                conn.execute("PRAGMA optimize")
                conn.execute("PRAGMA cache_size = -64000")  # 64MB cache
                conn.execute("PRAGMA temp_store = MEMORY")
                conn.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                
                self.connection_pool.append(conn)
                
            logger.info(f"Initialized database connection pool with {pool_size} connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get database connection from pool"""
        conn = None
        try:
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                else:
                    # Create new connection if pool is empty
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
            
            yield conn
            
        finally:
            if conn:
                with self.pool_lock:
                    self.connection_pool.append(conn)
    
    def analyze_query(self, query: str, parameters: Optional[List[Any]] = None) -> QueryPlan:
        """Analyze query and create execution plan"""
        try:
            with self.get_connection() as conn:
                # Get query plan
                explain_query = f"EXPLAIN QUERY PLAN {query}"
                plan_rows = conn.execute(explain_query, parameters or []).fetchall()
                
                # Parse plan information
                indexes_used = []
                estimated_rows = None
                
                for row in plan_rows:
                    detail = row[3].lower() if len(row) > 3 else ""
                    if "using index" in detail:
                        # Extract index name
                        parts = detail.split("using index")
                        if len(parts) > 1:
                            index_name = parts[1].strip().split()[0]
                            indexes_used.append(index_name)
                    
                    if "scan" in detail and "rows" in detail:
                        # Try to extract row estimate
                        try:
                            parts = detail.split()
                            for i, part in enumerate(parts):
                                if part == "rows" and i > 0:
                                    estimated_rows = int(parts[i-1])
                                    break
                        except:
                            pass
                
                # Generate cache key
                cache_key = None
                if self.cache_manager:
                    import hashlib
                    key_data = query + str(parameters or [])
                    cache_key = hashlib.sha256(key_data.encode()).hexdigest()
                
                return QueryPlan(
                    query=query,
                    parameters=parameters or [],
                    estimated_rows=estimated_rows,
                    cache_key=cache_key,
                    ttl=self.default_cache_ttl,
                    indexes_used=indexes_used
                )
                
        except Exception as e:
            logger.warning(f"Could not analyze query: {e}")
            return QueryPlan(query=query, parameters=parameters or [])
    
    def execute_query(self, query: str, parameters: Optional[List[Any]] = None,
                     cache_ttl: Optional[int] = None,
                     return_dataframe: bool = True) -> Union[List[Dict], Any]:
        """Execute optimized query with caching"""
        plan = self.analyze_query(query, parameters)
        cache_ttl = cache_ttl or plan.ttl
        
        # Try to get from cache first
        if self.cache_manager and plan.cache_key:
            cached_result = self.cache_manager.get(plan.cache_key)
            if cached_result is not None:
                logger.debug(f"Query cache hit: {plan.cache_key[:16]}...")
                return cached_result
        
        # Execute query with monitoring
        start_time = time.time()
        monitor_context = None
        
        if self.monitor:
            monitor_context = self.monitor.track_database_query(
                f"query_{hash(query) % 10000}", query
            )
            monitor_context.__enter__()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, parameters or [])
                rows = cursor.fetchall()
                
                # Convert to desired format
                if return_dataframe and PANDAS_AVAILABLE:
                    if rows:
                        columns = [desc[0] for desc in cursor.description]
                        data = [dict(zip(columns, row)) for row in rows]
                        result = pd.DataFrame(data)
                    else:
                        result = pd.DataFrame()
                else:
                    result = [dict(row) for row in rows]
                
                # Cache result if caching is enabled
                if self.cache_manager and plan.cache_key and cache_ttl:
                    self.cache_manager.set(plan.cache_key, result, cache_ttl)
                
                # Update query statistics
                duration = time.time() - start_time
                self._update_query_stats(query, duration, len(rows))
                
                return result
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            if monitor_context:
                monitor_context.__exit__(None, None, None)
    
    def _update_query_stats(self, query: str, duration: float, row_count: int):
        """Update query performance statistics"""
        query_hash = str(hash(query) % 10000)
        
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = {
                'total_duration': 0,
                'total_executions': 0,
                'total_rows': 0,
                'avg_duration': 0,
                'avg_rows': 0
            }
        
        stats = self.query_stats[query_hash]
        stats['total_duration'] += duration
        stats['total_executions'] += 1
        stats['total_rows'] += row_count
        stats['avg_duration'] = stats['total_duration'] / stats['total_executions']
        stats['avg_rows'] = stats['total_rows'] / stats['total_executions']
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query performance statistics"""
        return {
            'queries': dict(self.query_stats),
            'connection_pool_size': len(self.connection_pool),
            'cache_enabled': self.enable_query_cache
        }
    
    def optimize_table(self, table_name: str):
        """Optimize table with ANALYZE and VACUUM"""
        try:
            with self.get_connection() as conn:
                conn.execute(f"ANALYZE {table_name}")
                conn.execute("VACUUM")
                conn.commit()
                logger.info(f"Optimized table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to optimize table {table_name}: {e}")
    
    def create_index(self, table_name: str, columns: List[str], 
                    index_name: Optional[str] = None, unique: bool = False):
        """Create database index for query optimization"""
        try:
            if not index_name:
                index_name = f"idx_{table_name}_{'_'.join(columns)}"
            
            unique_clause = "UNIQUE " if unique else ""
            columns_clause = ", ".join(columns)
            
            with self.get_connection() as conn:
                conn.execute(
                    f"CREATE {unique_clause}INDEX IF NOT EXISTS {index_name} "
                    f"ON {table_name} ({columns_clause})"
                )
                conn.commit()
                logger.info(f"Created index: {index_name}")
                
        except Exception as e:
            logger.error(f"Failed to create index: {e}")


class LazyDataFrame:
    """Lazy-loaded DataFrame with pagination and caching"""
    
    def __init__(self, query_func: Callable[[int, int], Any], 
                 total_rows: Optional[int] = None,
                 page_size: int = 1000,
                 cache_manager: Optional[CacheManager] = None):
        self.query_func = query_func
        self.total_rows = total_rows
        self.page_size = page_size
        self.cache_manager = cache_manager
        self.loaded_pages: Dict[int, Any] = {}
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prefetch")
        
        # Metadata
        self._columns = None
        self._dtypes = None
        self._shape = None
    
    def _get_page_cache_key(self, page_num: int) -> str:
        """Generate cache key for page"""
        return f"lazy_df_page_{id(self)}_{page_num}"
    
    def _load_page(self, page_num: int) -> Any:
        """Load a specific page of data"""
        # Check cache first
        if self.cache_manager:
            cache_key = self._get_page_cache_key(page_num)
            cached_page = self.cache_manager.get(cache_key)
            if cached_page is not None:
                return cached_page
        
        # Load from source
        offset = page_num * self.page_size
        page_data = self.query_func(offset, self.page_size)
        
        # Cache the page
        if self.cache_manager:
            cache_key = self._get_page_cache_key(page_num)
            self.cache_manager.set(cache_key, page_data, ttl=1800)  # 30 mins
        
        return page_data
    
    def get_page(self, page_num: int) -> Any:
        """Get a specific page of data"""
        if page_num in self.loaded_pages:
            return self.loaded_pages[page_num]
        
        page_data = self._load_page(page_num)
        self.loaded_pages[page_num] = page_data
        
        # Prefetch next pages in background
        self._prefetch_pages(page_num + 1, 2)
        
        return page_data
    
    def _prefetch_pages(self, start_page: int, num_pages: int):
        """Prefetch pages in background"""
        def prefetch_worker():
            for page_num in range(start_page, start_page + num_pages):
                if page_num not in self.loaded_pages:
                    try:
                        self.loaded_pages[page_num] = self._load_page(page_num)
                    except Exception as e:
                        logger.warning(f"Failed to prefetch page {page_num}: {e}")
        
        self.prefetch_executor.submit(prefetch_worker)
    
    def __len__(self) -> int:
        """Get total number of rows"""
        if self.total_rows is not None:
            return self.total_rows
        
        # Estimate by loading first page
        first_page = self.get_page(0)
        if hasattr(first_page, '__len__'):
            if len(first_page) < self.page_size:
                self.total_rows = len(first_page)
            else:
                # Estimate based on first page
                self.total_rows = len(first_page) * 10  # Rough estimate
        
        return self.total_rows or 0
    
    def __getitem__(self, key) -> Any:
        """Get rows by index or slice"""
        if isinstance(key, slice):
            return self._get_slice(key)
        elif isinstance(key, int):
            return self._get_row(key)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
    def _get_row(self, index: int) -> Any:
        """Get a single row by index"""
        page_num = index // self.page_size
        row_in_page = index % self.page_size
        
        page_data = self.get_page(page_num)
        
        if hasattr(page_data, 'iloc'):  # pandas DataFrame
            return page_data.iloc[row_in_page] if row_in_page < len(page_data) else None
        elif isinstance(page_data, list):
            return page_data[row_in_page] if row_in_page < len(page_data) else None
        else:
            return None
    
    def _get_slice(self, slice_obj: slice) -> Any:
        """Get a slice of rows"""
        start, stop, step = slice_obj.indices(len(self))
        
        if step != 1:
            raise NotImplementedError("Step slicing not supported")
        
        # Determine which pages we need
        start_page = start // self.page_size
        end_page = (stop - 1) // self.page_size if stop > 0 else 0
        
        # Load required pages
        pages_data = []
        for page_num in range(start_page, end_page + 1):
            page_data = self.get_page(page_num)
            pages_data.append(page_data)
        
        # Combine pages and slice
        if PANDAS_AVAILABLE and pages_data and hasattr(pages_data[0], 'concat'):
            combined = pd.concat(pages_data, ignore_index=True)
            return combined.iloc[start % self.page_size:(stop - start_page * self.page_size)]
        else:
            # Handle as list
            combined = []
            for page_data in pages_data:
                if isinstance(page_data, list):
                    combined.extend(page_data)
                elif hasattr(page_data, 'to_dict'):
                    combined.extend(page_data.to_dict('records'))
            
            return combined[start % self.page_size:(stop - start_page * self.page_size)]
    
    def head(self, n: int = 5) -> Any:
        """Get first n rows"""
        return self[:n]
    
    def tail(self, n: int = 5) -> Any:
        """Get last n rows"""
        total = len(self)
        return self[max(0, total - n):total]
    
    def columns(self) -> List[str]:
        """Get column names"""
        if self._columns is None:
            first_page = self.get_page(0)
            if hasattr(first_page, 'columns'):
                self._columns = list(first_page.columns)
            elif isinstance(first_page, list) and first_page:
                if isinstance(first_page[0], dict):
                    self._columns = list(first_page[0].keys())
        
        return self._columns or []
    
    def shape(self) -> Tuple[int, int]:
        """Get shape (rows, columns)"""
        if self._shape is None:
            self._shape = (len(self), len(self.columns()))
        return self._shape
    
    def info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'shape': self.shape(),
            'columns': self.columns(),
            'page_size': self.page_size,
            'loaded_pages': len(self.loaded_pages),
            'cache_enabled': self.cache_manager is not None
        }


class PaginatedQuery:
    """Paginated query interface for large datasets"""
    
    def __init__(self, optimizer: QueryOptimizer, base_query: str,
                 count_query: Optional[str] = None,
                 pagination_config: Optional[PaginationConfig] = None):
        self.optimizer = optimizer
        self.base_query = base_query
        self.count_query = count_query
        self.config = pagination_config or PaginationConfig()
        self._total_count = None
    
    def get_total_count(self) -> int:
        """Get total number of rows"""
        if self._total_count is not None:
            return self._total_count
        
        if self.count_query:
            count_query = self.count_query
        else:
            # Generate count query from base query
            count_query = f"SELECT COUNT(*) as count FROM ({self.base_query}) as subquery"
        
        try:
            result = self.optimizer.execute_query(count_query, return_dataframe=False)
            self._total_count = result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Failed to get total count: {e}")
            self._total_count = 0
        
        return self._total_count
    
    def get_page(self, page_num: int, page_size: Optional[int] = None) -> Any:
        """Get a specific page of results"""
        page_size = page_size or self.config.page_size
        offset = page_num * page_size
        
        paginated_query = f"{self.base_query} LIMIT {page_size} OFFSET {offset}"
        
        return self.optimizer.execute_query(paginated_query)
    
    def iter_pages(self, max_pages: Optional[int] = None) -> Iterator[Any]:
        """Iterate through all pages"""
        total_count = self.get_total_count()
        total_pages = (total_count + self.config.page_size - 1) // self.config.page_size
        
        max_pages = min(max_pages or total_pages, total_pages)
        
        for page_num in range(max_pages):
            yield self.get_page(page_num)
    
    def to_lazy_dataframe(self) -> LazyDataFrame:
        """Convert to lazy-loaded DataFrame"""
        def query_func(offset: int, limit: int) -> Any:
            paginated_query = f"{self.base_query} LIMIT {limit} OFFSET {offset}"
            return self.optimizer.execute_query(paginated_query)
        
        return LazyDataFrame(
            query_func=query_func,
            total_rows=self.get_total_count(),
            page_size=self.config.page_size,
            cache_manager=self.optimizer.cache_manager
        )


class DataLoader:
    """High-level data loading interface with optimization"""
    
    def __init__(self, optimizer: QueryOptimizer, 
                 cache_manager: Optional[CacheManager] = None,
                 monitor: Optional[PerformanceMonitor] = None):
        self.optimizer = optimizer
        self.cache_manager = cache_manager
        self.monitor = monitor
        self.background_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="data_loader")
    
    def load_sa2_data(self, states: Optional[List[str]] = None,
                     lazy: bool = False, page_size: int = 1000) -> Union[Any, LazyDataFrame]:
        """Load SA2 geographic data with optional state filtering"""
        base_query = """
        SELECT sa2_code, sa2_name, state_code, state_name, 
               geometry, area_sqkm, population
        FROM sa2_boundaries
        """
        
        parameters = []
        if states:
            placeholders = ",".join("?" * len(states))
            base_query += f" WHERE state_code IN ({placeholders})"
            parameters.extend(states)
        
        if lazy:
            paginated = PaginatedQuery(
                self.optimizer, base_query,
                pagination_config=PaginationConfig(page_size=page_size)
            )
            return paginated.to_lazy_dataframe()
        else:
            return self.optimizer.execute_query(base_query, parameters)
    
    def load_health_data(self, metric_types: Optional[List[str]] = None,
                        sa2_codes: Optional[List[str]] = None,
                        lazy: bool = False) -> Union[Any, LazyDataFrame]:
        """Load health analytics data with filtering"""
        base_query = """
        SELECT h.sa2_code, h.metric_name, h.metric_value, h.year,
               s.sa2_name, s.state_code, s.population
        FROM health_metrics h
        JOIN sa2_boundaries s ON h.sa2_code = s.sa2_code
        """
        
        conditions = []
        parameters = []
        
        if metric_types:
            placeholders = ",".join("?" * len(metric_types))
            conditions.append(f"h.metric_name IN ({placeholders})")
            parameters.extend(metric_types)
        
        if sa2_codes:
            placeholders = ",".join("?" * len(sa2_codes))
            conditions.append(f"h.sa2_code IN ({placeholders})")
            parameters.extend(sa2_codes)
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        if lazy:
            paginated = PaginatedQuery(self.optimizer, base_query)
            return paginated.to_lazy_dataframe()
        else:
            return self.optimizer.execute_query(base_query, parameters)
    
    def load_demographic_data(self, sa2_codes: Optional[List[str]] = None,
                            lazy: bool = False) -> Union[Any, LazyDataFrame]:
        """Load demographic data"""
        base_query = """
        SELECT d.sa2_code, d.total_population, d.median_age, d.median_income,
               d.unemployment_rate, d.education_level,
               s.sa2_name, s.state_code
        FROM demographics d
        JOIN sa2_boundaries s ON d.sa2_code = s.sa2_code
        """
        
        parameters = []
        if sa2_codes:
            placeholders = ",".join("?" * len(sa2_codes))
            base_query += f" WHERE d.sa2_code IN ({placeholders})"
            parameters.extend(sa2_codes)
        
        if lazy:
            paginated = PaginatedQuery(self.optimizer, base_query)
            return paginated.to_lazy_dataframe()
        else:
            return self.optimizer.execute_query(base_query, parameters)
    
    def preload_data(self, data_types: List[str], states: Optional[List[str]] = None):
        """Preload data in background for faster access"""
        def preload_worker():
            try:
                if 'sa2' in data_types:
                    self.load_sa2_data(states=states)
                if 'health' in data_types:
                    self.load_health_data()
                if 'demographics' in data_types:
                    self.load_demographic_data()
                logger.info("Data preloading completed")
            except Exception as e:
                logger.error(f"Data preloading failed: {e}")
        
        self.background_executor.submit(preload_worker)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data"""
        summaries = {}
        
        try:
            # SA2 data summary
            sa2_count = self.optimizer.execute_query(
                "SELECT COUNT(*) as count FROM sa2_boundaries",
                return_dataframe=False
            )[0]['count']
            summaries['sa2_areas'] = sa2_count
            
            # Health metrics summary
            health_count = self.optimizer.execute_query(
                "SELECT COUNT(DISTINCT metric_name) as count FROM health_metrics",
                return_dataframe=False
            )[0]['count']
            summaries['health_metrics'] = health_count
            
            # State summary
            states = self.optimizer.execute_query(
                "SELECT DISTINCT state_code FROM sa2_boundaries ORDER BY state_code",
                return_dataframe=False
            )
            summaries['states'] = [s['state_code'] for s in states]
            
        except Exception as e:
            logger.error(f"Failed to get data summary: {e}")
            summaries['error'] = str(e)
        
        return summaries


# Utility functions for optimization
def optimize_dataframe_memory(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Optimize pandas DataFrame memory usage"""
    if not PANDAS_AVAILABLE:
        return df
    
    original_memory = df.memory_usage(deep=True).sum()
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize string columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If many repeating values
            df[col] = df[col].astype('category')
    
    optimized_memory = df.memory_usage(deep=True).sum()
    memory_reduction = (original_memory - optimized_memory) / original_memory * 100
    
    logger.info(f"Memory optimization: {memory_reduction:.1f}% reduction "
               f"({original_memory / 1024**2:.1f}MB -> {optimized_memory / 1024**2:.1f}MB)")
    
    return df


def compress_data(data: Any, compression: str = 'gzip') -> bytes:
    """Compress data for storage"""
    serialized = pickle.dumps(data)
    
    if compression == 'gzip':
        return gzip.compress(serialized)
    else:
        return serialized


def decompress_data(compressed_data: bytes, compression: str = 'gzip') -> Any:
    """Decompress data from storage"""
    if compression == 'gzip':
        serialized = gzip.decompress(compressed_data)
    else:
        serialized = compressed_data
    
    return pickle.loads(serialized)


if __name__ == "__main__":
    # Test optimization functionality
    import tempfile
    from .cache import CacheManager, CacheConfig
    from .monitoring import PerformanceMonitor
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Create test data
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE test_data (
            id INTEGER PRIMARY KEY,
            name TEXT,
            value REAL,
            state_code TEXT
        )
    """)
    
    # Generate sample data
    test_records = [(i, f"Record_{i}", i * 1.5, f"S{i % 5}") for i in range(10000)]
    conn.executemany("INSERT INTO test_data VALUES (?, ?, ?, ?)", test_records)
    conn.commit()
    conn.close()
    
    print("Testing query optimization...")
    
    # Initialize components
    cache_config = CacheConfig(file_cache_enabled=True)
    cache_manager = CacheManager(cache_config)
    monitor = PerformanceMonitor()
    
    optimizer = QueryOptimizer(db_path, cache_manager, monitor)
    data_loader = DataLoader(optimizer, cache_manager, monitor)
    
    # Test query optimization
    query = "SELECT * FROM test_data WHERE state_code = ? ORDER BY value DESC"
    
    # First execution (no cache)
    start_time = time.time()
    result1 = optimizer.execute_query(query, ["S1"])
    first_duration = time.time() - start_time
    
    # Second execution (should use cache)
    start_time = time.time()
    result2 = optimizer.execute_query(query, ["S1"])
    second_duration = time.time() - start_time
    
    print(f"First query: {first_duration:.3f}s, Second query: {second_duration:.3f}s")
    print(f"Cache speedup: {first_duration / max(second_duration, 0.001):.1f}x")
    
    # Test pagination
    paginated = PaginatedQuery(optimizer, "SELECT * FROM test_data", page_size=100)
    total_count = paginated.get_total_count()
    first_page = paginated.get_page(0)
    
    print(f"Total records: {total_count}")
    print(f"First page shape: {first_page.shape if hasattr(first_page, 'shape') else len(first_page)}")
    
    # Test lazy loading
    lazy_df = paginated.to_lazy_dataframe()
    print(f"Lazy DataFrame info: {lazy_df.info()}")
    print(f"First 5 rows shape: {len(lazy_df.head(5))}")
    
    # Get statistics
    stats = optimizer.get_query_statistics()
    print(f"Query statistics: {stats}")
    
    monitor.stop_monitoring()
    
    # Cleanup
    Path(db_path).unlink()
    
    print("Optimization test completed!")