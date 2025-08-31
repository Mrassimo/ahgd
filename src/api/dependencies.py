"""
Dependency injection for the AHGD Data Quality API.

This module provides FastAPI dependency injection for authentication, database connections,
services, and other shared resources following the existing AHGD patterns.
"""

from functools import lru_cache
from typing import Annotated
from typing import Any
from typing import Optional

import httpx
from fastapi import Depends
from fastapi import Header
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer

from ..utils.config import get_config
from ..utils.logging import get_logger
from .exceptions import AuthenticationException
from .exceptions import AuthorisationException
from .exceptions import ServiceUnavailableException
from .models.common import SystemHealth

logger = get_logger(__name__)

# Security scheme for Bearer token authentication
security = HTTPBearer(auto_error=False)


# Configuration Dependencies
@lru_cache
def get_api_config() -> dict[str, Any]:
    """Get API configuration."""
    return {
        "rate_limiting": get_config("api.rate_limiting", True),
        "max_requests_per_minute": get_config("api.max_requests_per_minute", 100),
        "enable_auth": get_config("api.enable_authentication", False),
        "auth_service_url": get_config("api.auth_service_url"),
        "enable_metrics": get_config("api.enable_metrics", True),
        "database_url": get_config("database.url"),
        "redis_url": get_config("cache.redis_url"),
        "external_services": get_config("external_services", {}),
    }


def get_database_config() -> dict[str, Any]:
    """Get database configuration."""
    return {
        "url": get_config("database.url"),
        "pool_size": get_config("database.pool_size", 10),
        "max_overflow": get_config("database.max_overflow", 20),
        "echo": get_config("database.echo", False),
    }


# Database Dependencies
class DatabaseManager:
    """Database connection manager."""

    def __init__(self):
        self._pool = None
        self._config = get_database_config()

    async def initialize(self):
        """Initialize database pool."""
        if self._pool is None:
            logger.info("Initializing database connection pool")
            # Here we would initialize the actual database pool
            # For now, it's a placeholder
            self._pool = "initialized"

    async def close(self):
        """Close database connections."""
        if self._pool:
            logger.info("Closing database connection pool")
            self._pool = None

    async def get_connection(self):
        """Get database connection."""
        if not self._pool:
            await self.initialize()

        # Return connection - placeholder for now
        return self._pool

    async def health_check(self) -> bool:
        """Check database health."""
        try:
            # Placeholder health check
            return self._pool is not None
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# Global database manager instance
_db_manager = DatabaseManager()


async def get_database() -> Any:
    """Get database connection dependency."""
    try:
        return await _db_manager.get_connection()
    except Exception as e:
        logger.error(f"Failed to get database connection: {e}")
        raise ServiceUnavailableException("database", "Database connection unavailable")


# Cache Dependencies
class CacheManager:
    """Redis cache manager."""

    def __init__(self):
        self._client = None
        self._config = get_config("cache", {})

    async def initialize(self):
        """Initialize cache client."""
        if self._client is None and self._config.get("redis_url"):
            logger.info("Initializing Redis cache client")
            # Here we would initialize the actual Redis client
            # For now, it's a placeholder
            self._client = "initialized"

    async def close(self):
        """Close cache client."""
        if self._client:
            logger.info("Closing cache client")
            self._client = None

    async def get_client(self):
        """Get cache client."""
        if not self._client:
            await self.initialize()
        return self._client

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        try:
            client = await self.get_client()
            if client:
                # Placeholder - would use actual Redis client
                return None
            return None
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: str, expire_seconds: int = 3600) -> bool:
        """Set value in cache."""
        try:
            client = await self.get_client()
            if client:
                # Placeholder - would use actual Redis client
                return True
            return False
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False


# Global cache manager instance
_cache_manager = CacheManager()


async def get_cache() -> CacheManager:
    """Get cache manager dependency."""
    return _cache_manager


# Authentication Dependencies
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    config: dict[str, Any] = Depends(get_api_config),
) -> Optional[dict[str, Any]]:
    """
    Get current authenticated user.

    Returns None if authentication is disabled.
    Raises AuthenticationException if auth is enabled but token is invalid.
    """

    # If authentication is disabled, return anonymous user
    if not config.get("enable_auth", False):
        return {
            "user_id": "anonymous",
            "username": "anonymous",
            "roles": ["read"],
            "is_authenticated": False,
        }

    # If auth is enabled but no credentials provided
    if not credentials:
        raise AuthenticationException("Authentication token required")

    # Validate token
    try:
        user_data = await validate_auth_token(credentials.credentials, config)
        return user_data
    except Exception as e:
        logger.warning(f"Authentication failed: {e}")
        raise AuthenticationException("Invalid authentication token")


async def validate_auth_token(token: str, config: dict[str, Any]) -> dict[str, Any]:
    """Validate authentication token with auth service."""

    auth_service_url = config.get("auth_service_url")
    if not auth_service_url:
        # Fallback to simple token validation for development
        if token == "dev-token":
            return {
                "user_id": "dev-user",
                "username": "developer",
                "roles": ["admin"],
                "is_authenticated": True,
            }
        else:
            raise ValueError("Invalid token")

    # Call external auth service
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{auth_service_url}/validate",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5.0,
            )

            if response.status_code == 200:
                return response.json()
            else:
                raise ValueError(f"Auth service returned {response.status_code}")

        except httpx.TimeoutException:
            raise ServiceUnavailableException("auth_service", "Authentication service timeout")
        except Exception as e:
            raise ValueError(f"Auth service error: {e}")


def require_authenticated_user(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    """Require an authenticated user."""

    if not user or not user.get("is_authenticated", False):
        raise AuthenticationException("Authentication required")

    return user


def require_admin_user(
    user: dict[str, Any] = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Require an admin user."""

    user_roles = user.get("roles", [])
    if "admin" not in user_roles:
        raise AuthorisationException("Admin privileges required")

    return user


def require_write_permission(
    user: dict[str, Any] = Depends(require_authenticated_user),
) -> dict[str, Any]:
    """Require write permission."""

    user_roles = user.get("roles", [])
    if not any(role in ["admin", "write", "editor"] for role in user_roles):
        raise AuthorisationException("Write permission required")

    return user


# Request Context Dependencies
def get_request_id(request: Request) -> str:
    """Get or generate request ID."""

    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        import uuid

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

    return request_id


def get_client_ip(request: Request) -> str:
    """Get client IP address."""

    # Check for forwarded headers first
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    # Fallback to direct connection
    if hasattr(request.client, "host"):
        return request.client.host

    return "unknown"


def get_user_agent(user_agent: Annotated[Optional[str], Header()] = None) -> str:
    """Get user agent string."""
    return user_agent or "unknown"


# Service Health Dependencies
class HealthChecker:
    """System health checker."""

    def __init__(self):
        self._last_check = None
        self._cached_health = None
        self._check_interval = 60  # seconds

    async def get_system_health(self) -> SystemHealth:
        """Get current system health status."""

        import time

        current_time = time.time()

        # Use cached result if recent
        if (
            self._cached_health
            and self._last_check
            and current_time - self._last_check < self._check_interval
        ):
            return self._cached_health

        # Perform health checks
        try:
            # Check database
            db_healthy = await _db_manager.health_check()

            # Check cache
            cache_healthy = await self._check_cache_health()

            # Check external services
            services_healthy = await self._check_external_services()

            # Determine overall status
            if db_healthy and cache_healthy and services_healthy:
                status = "healthy"
            elif db_healthy:
                status = "degraded"
            else:
                status = "unhealthy"

            health = SystemHealth(
                status=status,
                active_pipelines=0,  # Placeholder
                pending_validations=0,  # Placeholder
            )

            # Cache result
            self._cached_health = health
            self._last_check = current_time

            return health

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return SystemHealth(
                status="unhealthy",
                active_pipelines=0,
                pending_validations=0,
            )

    async def _check_cache_health(self) -> bool:
        """Check cache health."""
        try:
            cache = await get_cache()
            # Simple ping test
            await cache.set("health_check", "ok", 10)
            result = await cache.get("health_check")
            return result is not None
        except Exception:
            return False

    async def _check_external_services(self) -> bool:
        """Check external services health."""
        try:
            config = get_api_config()
            external_services = config.get("external_services", {})

            if not external_services:
                return True

            # Check each service
            async with httpx.AsyncClient(timeout=5.0) as client:
                for service_name, service_url in external_services.items():
                    try:
                        response = await client.get(f"{service_url}/health")
                        if response.status_code != 200:
                            logger.warning(
                                f"Service {service_name} unhealthy: {response.status_code}"
                            )
                            return False
                    except Exception as e:
                        logger.warning(f"Service {service_name} unreachable: {e}")
                        return False

            return True

        except Exception:
            return True  # Don't fail if external service checks fail


# Global health checker
_health_checker = HealthChecker()


async def get_system_health() -> SystemHealth:
    """Get system health dependency."""
    return await _health_checker.get_system_health()


# Rate Limiting Dependencies
class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self):
        self._requests = {}
        self._config = get_api_config()

    async def check_rate_limit(self, client_ip: str, user_id: str) -> bool:
        """Check if request is within rate limits."""

        if not self._config.get("rate_limiting", True):
            return True

        import time

        current_time = time.time()
        window_start = current_time - 60  # 1 minute window

        # Clean old entries
        keys_to_remove = [
            key
            for key, requests in self._requests.items()
            if all(req_time < window_start for req_time in requests)
        ]
        for key in keys_to_remove:
            del self._requests[key]

        # Check current requests
        key = f"{client_ip}:{user_id}"
        requests = self._requests.get(key, [])

        # Remove old requests from current key
        requests = [req_time for req_time in requests if req_time >= window_start]

        # Check limit
        max_requests = self._config.get("max_requests_per_minute", 100)
        if len(requests) >= max_requests:
            return False

        # Add current request
        requests.append(current_time)
        self._requests[key] = requests

        return True


# Global rate limiter
_rate_limiter = RateLimiter()


async def check_rate_limit(
    client_ip: str = Depends(get_client_ip), user: dict[str, Any] = Depends(get_current_user)
) -> bool:
    """Rate limiting dependency."""

    user_id = user.get("user_id", "anonymous")
    allowed = await _rate_limiter.check_rate_limit(client_ip, user_id)

    if not allowed:
        from .exceptions import raise_rate_limit_error

        raise_rate_limit_error(60)  # Suggest retry after 1 minute

    return True


# Service Dependencies (placeholders for actual service implementations)
async def get_quality_service():
    """Get quality metrics service."""
    # This will be implemented when we create the actual service
    return None


async def get_validation_service():
    """Get validation service."""
    # This will be implemented when we create the actual service
    return None


async def get_pipeline_service():
    """Get pipeline management service."""
    # This will be implemented when we create the actual service
    return None


# Lifecycle management
async def initialize_dependencies():
    """Initialize all dependency managers."""
    logger.info("Initializing API dependencies")

    try:
        await _db_manager.initialize()
        await _cache_manager.initialize()
        logger.info("Dependencies initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {e}")
        raise


async def cleanup_dependencies():
    """Clean up all dependency managers."""
    logger.info("Cleaning up API dependencies")

    try:
        await _db_manager.close()
        await _cache_manager.close()
        logger.info("Dependencies cleaned up successfully")
    except Exception as e:
        logger.error(f"Failed to clean up dependencies: {e}")


# Export commonly used dependencies
__all__ = [
    "get_api_config",
    "get_database_config",
    "get_database",
    "get_cache",
    "get_current_user",
    "require_authenticated_user",
    "require_admin_user",
    "require_write_permission",
    "get_request_id",
    "get_client_ip",
    "get_user_agent",
    "get_system_health",
    "check_rate_limit",
    "get_quality_service",
    "get_validation_service",
    "get_pipeline_service",
    "initialize_dependencies",
    "cleanup_dependencies",
]
