"""
Pipeline management endpoints.
"""

from fastapi import APIRouter

from ..models.common import APIResponse

router = APIRouter()


@router.get("/status")
async def get_pipeline_status() -> APIResponse:
    """Get pipeline status - placeholder implementation."""
    return APIResponse(message="Pipeline endpoints - implementation pending")
