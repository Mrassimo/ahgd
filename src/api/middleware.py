"""
Custom middleware for the AHGD Data Quality API.

This module provides middleware components for request processing,
following British English conventions and integrating with existing
AHGD logging and monitoring infrastructure.
"""

import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from typing import Optional

from fastapi import Request
from fastapi import Response
from fastapi import status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..utils.config import get_config
from ..utils.config import is_production
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request tracing and correlation IDs.

    Adds trace IDs to requests for monitoring and debugging purposes.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.trace_header = "X-Trace-ID"
        self.correlation_header = "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with tracing information.

        Args:
            request: HTTP request
            call_next: Next middleware/endpoint

        Returns:
            HTTP response with trace headers
        """

        # Generate or extract trace ID
        trace_id = request.headers.get(self.trace_header) or str(uuid.uuid4())
        correlation_id = request.headers.get(self.correlation_header) or str(uuid.uuid4())

        # Store trace information in request state
        request.state.trace_id = trace_id
        request.state.correlation_id = correlation_id
        request.state.request_start_time = time.time()

        # Set context for logging
        logger.set_context(
            trace_id=trace_id,
            correlation_id=correlation_id,
            method=request.method,
            path=str(request.url.path),
        )

        # Process request
        response = await call_next(request)

        # Add trace headers to response
        response.headers[self.trace_header] = trace_id
        response.headers[self.correlation_header] = correlation_id

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured request logging.

    Integrates with AHGD logging infrastructure for comprehensive
    request monitoring and analysis.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.exclude_paths = {"/health/ping", "/health/liveness", "/metrics"}
        self.slow_request_threshold = get_config("api.logging.slow_request_threshold", 2.0)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with comprehensive logging.

        Args:
            request: HTTP request
            call_next: Next middleware/endpoint

        Returns:
            HTTP response with logging
        """

        start_time = time.time()

        # Skip logging for health check endpoints
        if str(request.url.path) in self.exclude_paths:
            return await call_next(request)

        # Log request start
        logger.info(
            "API request started",
            method=request.method,
            path=str(request.url.path),
            query_params=str(request.query_params),
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent", "unknown"),
        )

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            error_message = None

        except Exception as e:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            error_message = str(e)
            # Re-raise the exception to be handled by exception handlers
            raise

        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Determine log level based on status and duration
            if status_code >= 500:
                log_level = "error"
            elif status_code >= 400:
                log_level = "warning"
            elif duration > self.slow_request_threshold:
                log_level = "warning"
            else:
                log_level = "info"

            # Log request completion
            getattr(logger, log_level)(
                "API request completed",
                method=request.method,
                path=str(request.url.path),
                status_code=status_code,
                duration_seconds=duration,
                slow_request=duration > self.slow_request_threshold,
                error_message=error_message,
                response_size=getattr(response, "body", b"").__len__()
                if "response" in locals()
                else 0,
            )

        return response


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for API rate limiting.

    Implements sliding window rate limiting with configurable
    limits per client IP address.
    """

    def __init__(
        self,
        app: ASGIApp,
        calls: int = 100,
        period: int = 60,
        exempt_paths: Optional[set[str]] = None,
    ):
        """
        Initialise rate limiting middleware.

        Args:
            app: ASGI application
            calls: Number of calls allowed per period
            period: Time period in seconds
            exempt_paths: Paths exempt from rate limiting
        """
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.exempt_paths = exempt_paths or {"/health/ping", "/health/liveness"}

        # Rate limiting storage (in production, use Redis)
        self.client_requests: dict[str, list] = defaultdict(list)
        self.cleanup_interval = 300  # Cleanup every 5 minutes
        self.last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting.

        Args:
            request: HTTP request
            call_next: Next middleware/endpoint

        Returns:
            HTTP response or rate limit error
        """

        # Skip rate limiting for exempt paths
        if str(request.url.path) in self.exempt_paths:
            return await call_next(request)

        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"

        # Check if we need to cleanup old entries
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_old_requests()
            self.last_cleanup = current_time

        # Check rate limit
        if not await self._is_rate_limit_ok(client_ip, current_time):
            # Rate limit exceeded
            retry_after = self._calculate_retry_after(client_ip, current_time)

            logger.warning(
                "Rate limit exceeded",
                client_ip=client_ip,
                path=str(request.url.path),
                calls_limit=self.calls,
                period_seconds=self.period,
                retry_after=retry_after,
            )

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Maximum {self.calls} requests per {self.period} seconds.",
                        "details": {
                            "limit": self.calls,
                            "period": self.period,
                            "retry_after": retry_after,
                        },
                    }
                },
                headers={"Retry-After": str(retry_after)},
            )

        # Record this request
        self.client_requests[client_ip].append(current_time)

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        remaining_requests = await self._get_remaining_requests(client_ip, current_time)
        reset_time = current_time + self.period

        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining_requests)
        response.headers["X-RateLimit-Reset"] = str(int(reset_time))

        return response

    async def _is_rate_limit_ok(self, client_ip: str, current_time: float) -> bool:
        """
        Check if client is within rate limits.

        Args:
            client_ip: Client IP address
            current_time: Current timestamp

        Returns:
            True if within limits, False otherwise
        """

        # Get recent requests for this client
        recent_requests = [
            req_time
            for req_time in self.client_requests[client_ip]
            if current_time - req_time < self.period
        ]

        # Update the client's request list
        self.client_requests[client_ip] = recent_requests

        # Check if within limits
        return len(recent_requests) < self.calls

    async def _get_remaining_requests(self, client_ip: str, current_time: float) -> int:
        """
        Get remaining requests for client.

        Args:
            client_ip: Client IP address
            current_time: Current timestamp

        Returns:
            Number of remaining requests
        """

        recent_requests = [
            req_time
            for req_time in self.client_requests[client_ip]
            if current_time - req_time < self.period
        ]

        return max(0, self.calls - len(recent_requests))

    def _calculate_retry_after(self, client_ip: str, current_time: float) -> int:
        """
        Calculate retry-after seconds.

        Args:
            client_ip: Client IP address
            current_time: Current timestamp

        Returns:
            Seconds to wait before retry
        """

        if not self.client_requests[client_ip]:
            return self.period

        # Find the oldest request within the period
        oldest_request = min(
            [
                req_time
                for req_time in self.client_requests[client_ip]
                if current_time - req_time < self.period
            ]
        )

        # Calculate when the oldest request will expire
        retry_after = int(oldest_request + self.period - current_time) + 1
        return max(1, retry_after)

    async def _cleanup_old_requests(self):
        """Clean up old request records to prevent memory leaks."""

        current_time = time.time()
        cutoff_time = current_time - self.period * 2  # Keep extra buffer

        # Clean up old entries
        for client_ip in list(self.client_requests.keys()):
            self.client_requests[client_ip] = [
                req_time for req_time in self.client_requests[client_ip] if req_time > cutoff_time
            ]

            # Remove clients with no recent requests
            if not self.client_requests[client_ip]:
                del self.client_requests[client_ip]

        logger.debug("Rate limit cleanup completed", active_clients=len(self.client_requests))


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers.

    Adds standard security headers to all responses for
    improved security posture.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

        # Security headers configuration
        self.security_headers = {
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            # XSS protection
            "X-XSS-Protection": "1; mode=block",
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            # Content Security Policy (basic)
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self'; "
                "connect-src 'self' ws: wss:; "
                "object-src 'none';"
            ),
            # Permissions policy
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "accelerometer=(), "
                "gyroscope=()"
            ),
        }

        # Add HSTS in production
        if is_production():
            self.security_headers[
                "Strict-Transport-Security"
            ] = "max-age=31536000; includeSubDomains; preload"

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and add security headers.

        Args:
            request: HTTP request
            call_next: Next middleware/endpoint

        Returns:
            HTTP response with security headers
        """

        response = await call_next(request)

        # Add security headers
        for header_name, header_value in self.security_headers.items():
            response.headers[header_name] = header_value

        # Add server identification (minimal)
        response.headers["Server"] = "AHGD-API"

        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for performance monitoring and metrics collection.

    Collects performance metrics and integrates with the AHGD
    monitoring infrastructure.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics_enabled = get_config("api.monitoring.metrics_enabled", True)
        self.detailed_metrics = get_config("api.monitoring.detailed_metrics", not is_production())

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with performance monitoring.

        Args:
            request: HTTP request
            call_next: Next middleware/endpoint

        Returns:
            HTTP response with performance metrics
        """

        if not self.metrics_enabled:
            return await call_next(request)

        start_time = time.time()

        # Get pipeline monitor from app state if available
        pipeline_monitor = getattr(request.app.state, "pipeline_monitor", None)

        try:
            response = await call_next(request)

            # Calculate metrics
            duration = time.time() - start_time

            # Record metrics if monitor is available
            if pipeline_monitor:
                # Record request metrics
                pipeline_monitor.metrics_collector.record_metric(
                    "api_request_duration",
                    duration,
                    labels={
                        "method": request.method,
                        "endpoint": str(request.url.path),
                        "status_code": str(response.status_code),
                    },
                )

                pipeline_monitor.metrics_collector.record_metric(
                    "api_request_count",
                    1,
                    labels={
                        "method": request.method,
                        "endpoint": str(request.url.path),
                        "status_code": str(response.status_code),
                    },
                )

            # Add performance headers for debugging
            if self.detailed_metrics:
                response.headers["X-Response-Time"] = f"{duration:.3f}s"
                response.headers["X-Process-Time"] = str(int(duration * 1000))

            return response

        except Exception as e:
            # Record error metrics
            duration = time.time() - start_time

            if pipeline_monitor:
                pipeline_monitor.metrics_collector.record_metric(
                    "api_request_errors",
                    1,
                    labels={
                        "method": request.method,
                        "endpoint": str(request.url.path),
                        "error_type": type(e).__name__,
                    },
                )

            raise


# Middleware factory functions
def create_rate_limiting_middleware(calls: int = 100, period: int = 60) -> type:
    """
    Factory function to create rate limiting middleware with custom limits.

    Args:
        calls: Number of calls allowed
        period: Time period in seconds

    Returns:
        Configured middleware class
    """

    class ConfiguredRateLimitingMiddleware(RateLimitingMiddleware):
        def __init__(self, app: ASGIApp):
            super().__init__(app, calls=calls, period=period)

    return ConfiguredRateLimitingMiddleware


def create_logging_middleware(exclude_paths: Optional[set[str]] = None) -> type:
    """
    Factory function to create logging middleware with custom configuration.

    Args:
        exclude_paths: Paths to exclude from logging

    Returns:
        Configured middleware class
    """

    class ConfiguredLoggingMiddleware(LoggingMiddleware):
        def __init__(self, app: ASGIApp):
            super().__init__(app)
            if exclude_paths:
                self.exclude_paths = exclude_paths

    return ConfiguredLoggingMiddleware
