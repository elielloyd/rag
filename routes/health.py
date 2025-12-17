"""Health check routes."""

from fastapi import APIRouter

from models import HealthResponse
from config import settings
from services import QdrantService

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health status of the application.
    
    Returns the status of Gemini API configuration and Qdrant connection.
    """
    # Check Gemini configuration
    gemini_configured = bool(settings.gemini_api_key)
    
    # Check Qdrant connection
    qdrant_connected = False
    try:
        qdrant_service = QdrantService()
        qdrant_connected = qdrant_service.is_connected()
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if gemini_configured else "degraded",
        gemini_configured=gemini_configured,
        qdrant_connected=qdrant_connected,
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "TrueClaim Preprocessing API",
        "version": "1.0.0",
        "description": "Image preprocessing pipeline using Gemini 3 API",
        "docs": "/docs",
        "health": "/health",
    }
