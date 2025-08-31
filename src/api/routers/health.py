"""
Health check endpoints for the AHGD Data Quality API.
"""

from fastapi import APIRouter, status
from datetime import datetime

from ..models.common import APIResponse, SystemHealth

router = APIRouter()

@router.get("/ping", status_code=status.HTTP_200_OK)
async def health_ping() -> APIResponse:
    """Simple health check for load balancers."""
    return APIResponse(
        message="Service is healthy",
        timestamp=datetime.now()
    )

@router.get("/liveness", status_code=status.HTTP_200_OK) 
async def health_liveness() -> APIResponse:
    """Kubernetes liveness probe."""
    return APIResponse(
        message="Service is live",
        timestamp=datetime.now()
    )

@router.get("/readiness", status_code=status.HTTP_200_OK)
async def health_readiness() -> APIResponse:
    """Kubernetes readiness probe."""
    return APIResponse(
        message="Service is ready",
        timestamp=datetime.now()
    )

@router.get("/status", response_model=SystemHealth)
async def health_status() -> SystemHealth:
    """Detailed system health status."""
    return SystemHealth(
        status="healthy",
        timestamp=datetime.now()
    )