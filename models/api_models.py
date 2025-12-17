from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(description="Health status")
    gemini_configured: bool = Field(description="Whether Gemini API is configured")
    qdrant_connected: bool = Field(description="Whether Qdrant is connected")
