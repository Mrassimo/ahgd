"""
Advanced Caching System for Australian Health Analytics Dashboard

Provides multi-tier caching with Redis compatibility, persistent storage,
and intelligent cache invalidation strategies.

Features:
- Streamlit session-based caching
- File-based persistent caching
- Redis-compatible caching (production)
- Smart cache invalidation
- Compression for large objects
- TTL-based expiration
- Memory usage optimization
"""

import os
import pickle
import hashlib
import time
import logging
import gzip
import json
from pathlib import Path
from typing import Any, Optional, Dict, Union, Callable, TypeVar, Tuple, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from abc import ABC, abstractmethod
import threading
from functools import wraps
import weakref

import streamlit as st

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Type variables for generic caching
T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration settings"""
    # General settings
    enabled: bool = True
    default_ttl: int = 3600  # 1 hour
    max_memory_mb: int = 512
    compression_enabled: bool = True
    compression_threshold: int = 1024  # Compress objects larger than 1KB
    
    # File cache settings
    file_cache_enabled: bool = True
    file_cache_dir: Optional[Path] = None
    max_file_cache_size_mb: int = 1024
    file_cache_cleanup_interval: int = 3600
    
    # Redis settings
    redis_enabled: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    redis_ssl: bool = False
    redis_connection_timeout: int = 5
    
    # Streamlit cache settings
    streamlit_cache_enabled: bool = True
    streamlit_ttl: int = 1800  # 30 minutes
    streamlit_max_entries: int = 100
    
    # Performance settings
    async_invalidation: bool = True
    background_cleanup: bool = True
    metrics_enabled: bool = True
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.file_cache_dir is None:
            from ..config import get_project_root
            self.file_cache_dir = get_project_root() / "cache"


class CacheBackend(ABC):
    """Abstract base class for cache backends"""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class StreamlitCacheBackend(CacheBackend):
    """Streamlit session state cache backend"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_key_prefix = "_cache_"
        self.metadata_key = "_cache_metadata_"
        
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key for session state"""
        return f"{self.cache_key_prefix}{key}"
    
    def _get_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get cache metadata from session state"""
        return st.session_state.get(self.metadata_key, {})
    
    def _set_metadata(self, metadata: Dict[str, Dict[str, Any]]):
        """Set cache metadata in session state"""
        st.session_state[self.metadata_key] = metadata
    
    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        if 'expires_at' not in metadata:
            return False
        return time.time() > metadata['expires_at']
    
    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        metadata = self._get_metadata()
        expired_keys = []
        
        for key, meta in metadata.items():
            if self._is_expired(meta):
                expired_keys.append(key)
                cache_key = self._get_cache_key(key)
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
        
        # Update metadata
        for key in expired_keys:
            del metadata[key]
        
        if expired_keys:
            self._set_metadata(metadata)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Streamlit session cache"""
        try:
            self._cleanup_expired()
            
            cache_key = self._get_cache_key(key)
            if cache_key not in st.session_state:
                return None
            
            metadata = self._get_metadata()
            if key in metadata and self._is_expired(metadata[key]):
                del st.session_state[cache_key]
                del metadata[key]
                self._set_metadata(metadata)
                return None
            
            return st.session_state[cache_key]
            
        except Exception as e:
            logger.error(f"Error getting from Streamlit cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Streamlit session cache"""
        try:
            # Enforce max entries limit
            metadata = self._get_metadata()
            if len(metadata) >= self.config.streamlit_max_entries and key not in metadata:
                # Remove oldest entry
                oldest_key = min(metadata.keys(), key=lambda k: metadata[k].get('created_at', 0))
                self.delete(oldest_key)
            
            cache_key = self._get_cache_key(key)
            st.session_state[cache_key] = value
            
            # Update metadata
            ttl = ttl or self.config.streamlit_ttl
            metadata[key] = {
                'created_at': time.time(),
                'expires_at': time.time() + ttl if ttl > 0 else None,
                'ttl': ttl,
                'size': len(str(value))  # Rough size estimate
            }
            self._set_metadata(metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting Streamlit cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Streamlit session cache"""
        try:
            cache_key = self._get_cache_key(key)
            if cache_key in st.session_state:
                del st.session_state[cache_key]
            
            metadata = self._get_metadata()
            if key in metadata:
                del metadata[key]
                self._set_metadata(metadata)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting from Streamlit cache: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            metadata = self._get_metadata()
            for key in list(metadata.keys()):
                cache_key = self._get_cache_key(key)
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
            
            self._set_metadata({})
            return True
            
        except Exception as e:
            logger.error(f"Error clearing Streamlit cache: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            self._cleanup_expired()
            cache_key = self._get_cache_key(key)
            return cache_key in st.session_state
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            metadata = self._get_metadata()
            total_size = sum(meta.get('size', 0) for meta in metadata.values())
            
            return {
                'backend': 'streamlit',
                'entries': len(metadata),
                'total_size_bytes': total_size,
                'max_entries': self.config.streamlit_max_entries,
                'default_ttl': self.config.streamlit_ttl
            }
        except Exception:
            return {'backend': 'streamlit', 'error': 'Could not get stats'}


class FileCacheBackend(CacheBackend):
    """File-based persistent cache backend"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_dir = config.file_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self._lock = threading.Lock()
        
        # Load existing metadata
        self._metadata = self._load_metadata()
        
        # Start background cleanup if enabled
        if config.background_cleanup:
            self._start_cleanup_thread()
    
    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata from file"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save cache metadata: {e}")
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key"""
        # Use hash of key to avoid filesystem issues
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def _is_expired(self, metadata: Dict[str, Any]) -> bool:
        """Check if cache entry is expired"""
        if 'expires_at' not in metadata:
            return False
        return time.time() > metadata['expires_at']
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if enabled and above threshold"""
        if (self.config.compression_enabled and 
            len(data) > self.config.compression_threshold):
            return gzip.compress(data)
        return data
    
    def _decompress_data(self, data: bytes, compressed: bool) -> bytes:
        """Decompress data if needed"""
        if compressed:
            return gzip.decompress(data)
        return data
    
    def _cleanup_expired(self):
        """Remove expired cache files"""
        with self._lock:
            expired_keys = []
            for key, metadata in self._metadata.items():
                if self._is_expired(metadata):
                    expired_keys.append(key)
                    cache_file = self._get_cache_file(key)
                    try:
                        if cache_file.exists():
                            cache_file.unlink()
                    except Exception as e:
                        logger.error(f"Error removing expired cache file {cache_file}: {e}")
            
            # Update metadata
            for key in expired_keys:
                del self._metadata[key]
            
            if expired_keys:
                self._save_metadata()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(self.config.file_cache_cleanup_interval)
                    self._cleanup_expired()
                except Exception as e:
                    logger.error(f"Error in cache cleanup thread: {e}")
        
        thread = threading.Thread(target=cleanup_worker, daemon=True)
        thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from file cache"""
        try:
            with self._lock:
                if key not in self._metadata:
                    return None
                
                metadata = self._metadata[key]
                if self._is_expired(metadata):
                    self.delete(key)
                    return None
                
                cache_file = self._get_cache_file(key)
                if not cache_file.exists():
                    # Metadata exists but file doesn't, clean up
                    del self._metadata[key]
                    self._save_metadata()
                    return None
                
                with open(cache_file, 'rb') as f:
                    data = f.read()
                
                # Decompress if needed
                data = self._decompress_data(data, metadata.get('compressed', False))
                
                # Deserialize
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Error getting from file cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in file cache"""
        try:
            with self._lock:
                # Serialize data
                data = pickle.dumps(value)
                
                # Compress if enabled
                compressed = (self.config.compression_enabled and 
                            len(data) > self.config.compression_threshold)
                if compressed:
                    data = self._compress_data(data)
                
                # Write to file
                cache_file = self._get_cache_file(key)
                with open(cache_file, 'wb') as f:
                    f.write(data)
                
                # Update metadata
                ttl = ttl or self.config.default_ttl
                self._metadata[key] = {
                    'created_at': time.time(),
                    'expires_at': time.time() + ttl if ttl > 0 else None,
                    'ttl': ttl,
                    'size': len(data),
                    'compressed': compressed,
                    'original_size': len(pickle.dumps(value))
                }
                
                self._save_metadata()
                return True
                
        except Exception as e:
            logger.error(f"Error setting file cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from file cache"""
        try:
            with self._lock:
                cache_file = self._get_cache_file(key)
                if cache_file.exists():
                    cache_file.unlink()
                
                if key in self._metadata:
                    del self._metadata[key]
                    self._save_metadata()
                
                return True
                
        except Exception as e:
            logger.error(f"Error deleting from file cache: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        try:
            with self._lock:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.error(f"Error removing cache file {cache_file}: {e}")
                
                # Clear metadata
                self._metadata = {}
                self._save_metadata()
                
                return True
                
        except Exception as e:
            logger.error(f"Error clearing file cache: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            with self._lock:
                if key not in self._metadata:
                    return False
                
                metadata = self._metadata[key]
                if self._is_expired(metadata):
                    self.delete(key)
                    return False
                
                cache_file = self._get_cache_file(key)
                return cache_file.exists()
                
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            with self._lock:
                total_size = sum(meta.get('size', 0) for meta in self._metadata.values())
                total_original_size = sum(meta.get('original_size', 0) for meta in self._metadata.values())
                compressed_entries = sum(1 for meta in self._metadata.values() if meta.get('compressed', False))
                
                return {
                    'backend': 'file',
                    'entries': len(self._metadata),
                    'total_size_bytes': total_size,
                    'total_original_size_bytes': total_original_size,
                    'compressed_entries': compressed_entries,
                    'compression_ratio': (total_original_size - total_size) / total_original_size if total_original_size > 0 else 0,
                    'cache_dir': str(self.cache_dir),
                    'max_size_mb': self.config.max_file_cache_size_mb
                }
        except Exception:
            return {'backend': 'file', 'error': 'Could not get stats'}


class RedisCacheBackend(CacheBackend):
    """Redis cache backend for production deployment"""
    
    def __init__(self, config: CacheConfig):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis not available. Install with: pip install redis")
        
        self.config = config
        self.redis_client = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis server"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                ssl=self.config.redis_ssl,
                socket_timeout=self.config.redis_connection_timeout,
                decode_responses=False  # We handle our own encoding
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Connected to Redis cache backend")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            raise
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        data = pickle.dumps(value)
        if (self.config.compression_enabled and 
            len(data) > self.config.compression_threshold):
            return gzip.compress(data)
        return data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from Redis storage"""
        # Try to decompress first
        try:
            decompressed = gzip.decompress(data)
            return pickle.loads(decompressed)
        except (gzip.BadGzipFile, OSError):
            # Not compressed
            return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data is None:
                return None
            
            return self._deserialize(data)
            
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            data = self._serialize(value)
            ttl = ttl or self.config.default_ttl
            
            if ttl > 0:
                return self.redis_client.setex(key, ttl, data)
            else:
                return self.redis_client.set(key, data)
                
        except Exception as e:
            logger.error(f"Error setting Redis cache: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all cache entries"""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.flushdb())
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis_client:
            return False
        
        try:
            return bool(self.redis_client.exists(key))
        except Exception:
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.redis_client:
            return {'backend': 'redis', 'error': 'Not connected'}
        
        try:
            info = self.redis_client.info()
            return {
                'backend': 'redis',
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1)
            }
        except Exception:
            return {'backend': 'redis', 'error': 'Could not get stats'}


class CacheManager:
    """Multi-tier cache manager with intelligent fallback"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.backends: Dict[str, CacheBackend] = {}
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        # Initialize enabled backends
        self._initialize_backends()
        
        # Set up cache key namespacing
        self.namespace = "ahgd_cache"
    
    def _initialize_backends(self):
        """Initialize available cache backends"""
        # Always try Streamlit cache first (for session persistence)
        if self.config.streamlit_cache_enabled:
            try:
                self.backends['streamlit'] = StreamlitCacheBackend(self.config)
                logger.info("Initialized Streamlit cache backend")
            except Exception as e:
                logger.warning(f"Could not initialize Streamlit cache: {e}")
        
        # File cache for persistence across sessions
        if self.config.file_cache_enabled:
            try:
                self.backends['file'] = FileCacheBackend(self.config)
                logger.info("Initialized file cache backend")
            except Exception as e:
                logger.warning(f"Could not initialize file cache: {e}")
        
        # Redis for production deployment
        if self.config.redis_enabled and REDIS_AVAILABLE:
            try:
                self.backends['redis'] = RedisCacheBackend(self.config)
                logger.info("Initialized Redis cache backend")
            except Exception as e:
                logger.warning(f"Could not initialize Redis cache: {e}")
        
        if not self.backends:
            logger.warning("No cache backends available - caching disabled")
    
    def _make_key(self, key: str) -> str:
        """Create namespaced cache key"""
        return f"{self.namespace}:{key}"
    
    def _get_backends_by_priority(self) -> List[Tuple[str, CacheBackend]]:
        """Get backends in priority order (fastest to slowest)"""
        priority_order = ['streamlit', 'redis', 'file']
        ordered_backends = []
        
        for backend_name in priority_order:
            if backend_name in self.backends:
                ordered_backends.append((backend_name, self.backends[backend_name]))
        
        return ordered_backends
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with multi-tier fallback"""
        if not self.config.enabled or not self.backends:
            return None
        
        namespaced_key = self._make_key(key)
        
        for backend_name, backend in self._get_backends_by_priority():
            try:
                value = backend.get(namespaced_key)
                if value is not None:
                    self.metrics['hits'] += 1
                    
                    # Populate faster caches if value found in slower cache
                    if backend_name != 'streamlit' and 'streamlit' in self.backends:
                        self.backends['streamlit'].set(namespaced_key, value)
                    
                    return value
                    
            except Exception as e:
                logger.error(f"Error getting from {backend_name} cache: {e}")
                self.metrics['errors'] += 1
        
        self.metrics['misses'] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in all available caches"""
        if not self.config.enabled or not self.backends:
            return False
        
        namespaced_key = self._make_key(key)
        success = False
        
        for backend_name, backend in self.backends.items():
            try:
                if backend.set(namespaced_key, value, ttl):
                    success = True
            except Exception as e:
                logger.error(f"Error setting {backend_name} cache: {e}")
                self.metrics['errors'] += 1
        
        if success:
            self.metrics['sets'] += 1
        
        return success
    
    def delete(self, key: str) -> bool:
        """Delete value from all caches"""
        if not self.config.enabled or not self.backends:
            return False
        
        namespaced_key = self._make_key(key)
        success = False
        
        for backend_name, backend in self.backends.items():
            try:
                if backend.delete(namespaced_key):
                    success = True
            except Exception as e:
                logger.error(f"Error deleting from {backend_name} cache: {e}")
                self.metrics['errors'] += 1
        
        if success:
            self.metrics['deletes'] += 1
        
        return success
    
    def clear(self) -> bool:
        """Clear all caches"""
        if not self.config.enabled or not self.backends:
            return False
        
        success = False
        
        for backend_name, backend in self.backends.items():
            try:
                if backend.clear():
                    success = True
            except Exception as e:
                logger.error(f"Error clearing {backend_name} cache: {e}")
                self.metrics['errors'] += 1
        
        return success
    
    def exists(self, key: str) -> bool:
        """Check if key exists in any cache"""
        if not self.config.enabled or not self.backends:
            return False
        
        namespaced_key = self._make_key(key)
        
        for backend_name, backend in self._get_backends_by_priority():
            try:
                if backend.exists(namespaced_key):
                    return True
            except Exception as e:
                logger.error(f"Error checking existence in {backend_name} cache: {e}")
                self.metrics['errors'] += 1
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'enabled': self.config.enabled,
            'backends': list(self.backends.keys()),
            'metrics': self.metrics.copy(),
            'hit_rate': self.metrics['hits'] / max(self.metrics['hits'] + self.metrics['misses'], 1),
            'backend_stats': {}
        }
        
        for backend_name, backend in self.backends.items():
            try:
                stats['backend_stats'][backend_name] = backend.get_stats()
            except Exception as e:
                stats['backend_stats'][backend_name] = {'error': str(e)}
        
        return stats
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        # This is a simplified implementation
        # In production, you might want more sophisticated pattern matching
        invalidated = 0
        
        if 'redis' in self.backends:
            try:
                redis_backend = self.backends['redis']
                keys = redis_backend.redis_client.keys(f"{self.namespace}:*{pattern}*")
                for key in keys:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    original_key = key_str.replace(f"{self.namespace}:", "")
                    if self.delete(original_key):
                        invalidated += 1
            except Exception as e:
                logger.error(f"Error invalidating pattern {pattern}: {e}")
        
        return invalidated


# Decorator for caching function results
def cached(key_func: Optional[Callable[..., str]] = None, 
          ttl: Optional[int] = None,
          cache_manager: Optional[CacheManager] = None):
    """
    Decorator for caching function results
    
    Args:
        key_func: Function to generate cache key from arguments
        ttl: Time to live for cached result
        cache_manager: Cache manager instance to use
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get or create cache manager
            manager = cache_manager or _get_default_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = "|".join(key_parts)
            
            # Try to get from cache
            cached_result = manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result and cache it
            result = func(*args, **kwargs)
            manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def _get_default_cache_manager() -> CacheManager:
    """Get or create default cache manager"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def get_cache_manager(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get cache manager instance"""
    if config is None:
        return _get_default_cache_manager()
    return CacheManager(config)


@contextmanager
def cache_context(cache_manager: CacheManager):
    """Context manager for temporary cache manager"""
    global _global_cache_manager
    old_manager = _global_cache_manager
    _global_cache_manager = cache_manager
    try:
        yield cache_manager
    finally:
        _global_cache_manager = old_manager


# Streamlit-specific caching utilities
def st_cache_data_enhanced(func: Optional[Callable] = None, *, 
                          ttl: Optional[int] = None,
                          max_entries: Optional[int] = None,
                          show_spinner: Union[bool, str] = True,
                          persist: str = "disk"):
    """
    Enhanced Streamlit cache decorator with additional backends
    
    This combines Streamlit's native caching with our multi-tier cache system
    """
    def decorator(f):
        # Apply Streamlit caching first
        streamlit_cached = st.cache_data(
            ttl=ttl,
            max_entries=max_entries,
            show_spinner=show_spinner,
            persist=persist
        )(f)
        
        # Then apply our enhanced caching
        enhanced_cached = cached(ttl=ttl)(streamlit_cached)
        
        return enhanced_cached
    
    if func is None:
        return decorator
    else:
        return decorator(func)


if __name__ == "__main__":
    # Test cache functionality
    config = CacheConfig(
        file_cache_enabled=True,
        redis_enabled=False,  # Set to True if Redis is available
        compression_enabled=True
    )
    
    cache_manager = CacheManager(config)
    
    # Test basic operations
    print("Testing cache operations...")
    
    # Set some test data
    test_data = {"message": "Hello, World!", "numbers": list(range(100))}
    success = cache_manager.set("test_key", test_data, ttl=60)
    print(f"Set operation: {'Success' if success else 'Failed'}")
    
    # Get data
    retrieved_data = cache_manager.get("test_key")
    print(f"Get operation: {'Success' if retrieved_data == test_data else 'Failed'}")
    
    # Check existence
    exists = cache_manager.exists("test_key")
    print(f"Exists check: {'Success' if exists else 'Failed'}")
    
    # Get stats
    stats = cache_manager.get_stats()
    print(f"Cache stats: {stats}")
    
    # Test decorator
    @cached(ttl=30)
    def expensive_computation(n: int) -> int:
        """Simulate expensive computation"""
        time.sleep(0.1)  # Simulate work
        return sum(range(n))
    
    print("\nTesting cached decorator...")
    start_time = time.time()
    result1 = expensive_computation(1000)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    result2 = expensive_computation(1000)
    second_call_time = time.time() - start_time
    
    print(f"First call: {first_call_time:.3f}s, Second call: {second_call_time:.3f}s")
    print(f"Results match: {result1 == result2}")
    print(f"Speed improvement: {first_call_time / max(second_call_time, 0.001):.1f}x")
    
    print("\nCache system test completed!")