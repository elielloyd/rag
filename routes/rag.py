"""Routes for RAG (Retrieval Augmented Generation) pipeline."""

from fastapi import APIRouter, HTTPException

from models.rag_models import (
    RAGEstimateRequest,
    RAGEstimateResponse,
)
from services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG Estimate Generation"])


def get_rag_service() -> RAGService:
    """Get or create the RAG service instance."""
    return RAGService()


@router.post("/estimate", response_model=RAGEstimateResponse)
async def generate_rag_estimate(request: RAGEstimateRequest):
    """
    Generate a repair estimate using RAG pipeline.
    
    This endpoint implements the complete RAG flow:
    1. Fetches images from the provided S3 bucket URL
    2. Analyzes each image for damage using Gemini 3
    3. Filters to only images with detected damage
    4. Retrieves similar historical estimates from Qdrant vector database
    5. Combines retrieved chunks with PSS data and damage descriptions
    6. Generates a final repair estimate
    
    Args:
        request: RAGEstimateRequest containing:
            - bucket_url: S3 bucket URL with vehicle images
            - damage_description: Optional human-provided damage description
            - pss_url: Optional S3 URL to PSS (Parts and Service Standards) JSON file
            - vehicle_info: Optional vehicle information
    
    Returns:
        RAGEstimateResponse with:
            - Damage detection results per image
            - Retrieved similar chunks
            - Generated estimate with line items
            - Processing metadata
    
    Example request:
        {
            "bucket_url": "s3://bucket/claims/claim-id/images/",
            "damage_description": "Rear-end collision damage to bumper",
            "vehicle_info": {
                "vin": "1234567890",
                "make": "Toyota",
                "model": "Camry",
                "year": 2020,
                "body_type": "Sedan"
            },
            "pss_url": "s3://ehsan-poc-estimate-true-claim/pss/subaru_outback_2020_2024.json"
        }
    """
    if not request.bucket_url:
        raise HTTPException(
            status_code=400,
            detail="bucket_url must be provided"
        )
    
    service = get_rag_service()
    response = service.run_rag_pipeline(request)
    
    if not response.success:
        raise HTTPException(
            status_code=500,
            detail=response.error or "Failed to generate estimate"
        )
    
    return response
