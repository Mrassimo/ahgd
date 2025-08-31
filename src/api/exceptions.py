"""
API-specific exception handlers for the AHGD Data Quality API.

This module defines custom exceptions and their handlers, integrating with
the existing AHGD error handling patterns while providing REST-appropriate
error responses.
"""

import traceback
from typing import Dict, Any, Optional, Union

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import ValidationError

from ..utils.interfaces import AHGDException, ValidationError as AHGDValidationError
from ..utils.logging import get_logger
from .models.common import ErrorResponse, ErrorDetail

logger = get_logger(__name__)


class AHGDAPIException(Exception):
    """Base exception for AHGD API-specific errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        Initialise API exception.
        
        Args:
            message: Error message
            status_code: HTTP status code
            error_code: Internal error code
            details: Additional error details
            headers: Response headers
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        self.headers = headers or {}


class ValidationException(AHGDAPIException):
    """Exception for validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details
        )
        self.field = field


class AuthenticationException(AHGDAPIException):
    """Exception for authentication errors."""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_REQUIRED",
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorisationException(AHGDAPIException):
    """Exception for authorisation errors (British spelling)."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="INSUFFICIENT_PERMISSIONS"
        )


class RateLimitException(AHGDAPIException):
    """Exception for rate limiting errors."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None
    ):
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            headers=headers
        )


class PipelineException(AHGDAPIException):
    """Exception for pipeline-related errors."""
    
    def __init__(
        self,
        message: str,
        pipeline_name: Optional[str] = None,
        stage_name: Optional[str] = None
    ):
        details = {}
        if pipeline_name:
            details["pipeline_name"] = pipeline_name
        if stage_name:
            details["stage_name"] = stage_name
        
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="PIPELINE_ERROR",
            details=details
        )


class ResourceNotFoundException(AHGDAPIException):
    """Exception for resource not found errors."""
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str
    ):
        super().__init__(
            message=f"{resource_type} with ID '{resource_id}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="RESOURCE_NOT_FOUND",
            details={
                "resource_type": resource_type,
                "resource_id": resource_id
            }
        )


class ServiceUnavailableException(AHGDAPIException):
    """Exception for service unavailable errors."""
    
    def __init__(
        self,
        service_name: str,
        message: Optional[str] = None
    ):
        super().__init__(
            message=message or f"{service_name} service is currently unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="SERVICE_UNAVAILABLE",
            details={"service_name": service_name}
        )


async def ahgd_api_exception_handler(request: Request, exc: AHGDAPIException) -> JSONResponse:
    """
    Handle AHGD API-specific exceptions.
    
    Args:
        request: FastAPI request
        exc: Exception instance
        
    Returns:
        JSON error response
    """
    
    # Log the exception
    logger.error(
        "API exception occurred",
        error_code=exc.error_code,
        status_code=exc.status_code,
        message=exc.message,
        path=str(request.url),
        method=request.method,
        details=exc.details
    )
    
    # Create error response
    error_detail = ErrorDetail(
        code=exc.error_code,
        message=exc.message,
        field=getattr(exc, 'field', None),
        details=exc.details
    )
    
    response = ErrorResponse(
        error=error_detail,
        trace_id=getattr(request.state, 'trace_id', None)
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response.dict(),
        headers=exc.headers
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle Pydantic validation errors.
    
    Args:
        request: FastAPI request
        exc: Validation error
        
    Returns:
        JSON error response
    """
    
    # Extract validation details
    errors = []
    for error in exc.errors():
        field = ".".join(str(x) for x in error["loc"]) if error["loc"] else None
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    # Log validation error
    logger.warning(
        "Request validation failed",
        path=str(request.url),
        method=request.method,
        errors=errors
    )
    
    # Create error response
    error_detail = ErrorDetail(
        code="VALIDATION_ERROR",
        message="Request validation failed",
        details={"errors": errors}
    )
    
    response = ErrorResponse(
        error=error_detail,
        trace_id=getattr(request.state, 'trace_id', None)
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response.dict()
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions.
    
    Args:
        request: FastAPI request
        exc: HTTP exception
        
    Returns:
        JSON error response
    """
    
    # Map status codes to error codes
    error_code_map = {
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        406: "NOT_ACCEPTABLE",
        415: "UNSUPPORTED_MEDIA_TYPE",
        500: "INTERNAL_SERVER_ERROR"
    }
    
    error_code = error_code_map.get(exc.status_code, "HTTP_ERROR")
    
    # Log HTTP exception
    logger.error(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=str(request.url),
        method=request.method
    )
    
    # Create error response
    error_detail = ErrorDetail(
        code=error_code,
        message=str(exc.detail),
        details={"status_code": exc.status_code}
    )
    
    response = ErrorResponse(
        error=error_detail,
        trace_id=getattr(request.state, 'trace_id', None)
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response.dict()
    )


async def ahgd_core_exception_handler(request: Request, exc: AHGDException) -> JSONResponse:
    """
    Handle AHGD core infrastructure exceptions.
    
    Args:
        request: FastAPI request
        exc: AHGD core exception
        
    Returns:
        JSON error response
    """
    
    # Map AHGD core exceptions to HTTP status codes
    status_code_map = {
        "ValidationError": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "ExtractionError": status.HTTP_503_SERVICE_UNAVAILABLE,
        "TransformationError": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "LoadingError": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "ConfigurationError": status.HTTP_500_INTERNAL_SERVER_ERROR
    }
    
    error_type = type(exc).__name__
    http_status = status_code_map.get(error_type, status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    # Log core exception
    logger.error(
        "AHGD core exception occurred",
        error_type=error_type,
        message=str(exc),
        path=str(request.url),
        method=request.method
    )
    
    # Create error response
    error_detail = ErrorDetail(
        code=f"AHGD_{error_type.upper()}",
        message=str(exc),
        details={"error_type": error_type}
    )
    
    response = ErrorResponse(
        error=error_detail,
        trace_id=getattr(request.state, 'trace_id', None)
    )
    
    return JSONResponse(
        status_code=http_status,
        content=response.dict()
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions.
    
    Args:
        request: FastAPI request
        exc: Unhandled exception
        
    Returns:
        JSON error response
    """
    
    # Generate trace ID if not present
    trace_id = getattr(request.state, 'trace_id', None)
    if not trace_id:
        import uuid
        trace_id = str(uuid.uuid4())
    
    # Log the unexpected exception with full traceback
    logger.error(
        "Unexpected exception occurred",
        exception_type=type(exc).__name__,
        message=str(exc),
        path=str(request.url),
        method=request.method,
        trace_id=trace_id,
        traceback=traceback.format_exc()
    )
    
    # Create generic error response (don't expose internal details in production)
    from ..utils.config import is_production
    
    if is_production():
        message = "An internal error occurred"
        details = {"trace_id": trace_id}
    else:
        message = str(exc)
        details = {
            "trace_id": trace_id,
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc().split('\n')
        }
    
    error_detail = ErrorDetail(
        code="INTERNAL_SERVER_ERROR",
        message=message,
        details=details
    )
    
    response = ErrorResponse(
        error=error_detail,
        trace_id=trace_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.dict()
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Set up exception handlers for the FastAPI application.
    
    Args:
        app: FastAPI instance
    """
    
    # AHGD API-specific exceptions
    app.add_exception_handler(AHGDAPIException, ahgd_api_exception_handler)
    
    # Validation exceptions
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ValidationError, validation_exception_handler)
    
    # HTTP exceptions
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    
    # AHGD core exceptions
    app.add_exception_handler(AHGDException, ahgd_core_exception_handler)
    
    # Generic exception handler (catch-all)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("Exception handlers configured successfully")


# Exception utilities
def raise_not_found(resource_type: str, resource_id: str) -> None:
    """
    Convenience function to raise resource not found exception.
    
    Args:
        resource_type: Type of resource
        resource_id: Resource identifier
    """
    raise ResourceNotFoundException(resource_type, resource_id)


def raise_validation_error(message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
    """
    Convenience function to raise validation exception.
    
    Args:
        message: Error message
        field: Field that failed validation
        details: Additional details
    """
    raise ValidationException(message, field, details)


def raise_pipeline_error(message: str, pipeline_name: Optional[str] = None, stage_name: Optional[str] = None) -> None:
    """
    Convenience function to raise pipeline exception.
    
    Args:
        message: Error message
        pipeline_name: Pipeline name
        stage_name: Stage name
    """
    raise PipelineException(message, pipeline_name, stage_name)


def raise_rate_limit_error(retry_after: Optional[int] = None) -> None:
    """
    Convenience function to raise rate limit exception.
    
    Args:
        retry_after: Seconds to wait before retry
    """
    raise RateLimitException(retry_after=retry_after)


def raise_service_unavailable(service_name: str, message: Optional[str] = None) -> None:
    """
    Convenience function to raise service unavailable exception.
    
    Args:
        service_name: Name of unavailable service
        message: Custom error message
    """
    raise ServiceUnavailableException(service_name, message)