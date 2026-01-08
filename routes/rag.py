"""Routes for RAG (Retrieval Augmented Generation) pipeline."""

from fastapi import APIRouter, HTTPException, Depends

from models.rag_models import (
    RAGEstimateRequest,
    RAGEstimateResponse,
)
from services.rag_service import RAGService
from middleware.auth import verify_api_key

router = APIRouter(
    prefix="/rag",
    tags=["RAG Estimate Generation"],
    dependencies=[Depends(verify_api_key)]
)


def get_rag_service() -> RAGService:
    """Get or create the RAG service instance."""
    return RAGService()


@router.post("/estimate", response_model=RAGEstimateResponse)
async def generate_rag_estimate(request: RAGEstimateRequest):
    """
    Generate a repair estimate using RAG pipeline.
    
    This endpoint accepts pre-processed damage information and generates an estimate:
    1. Uses provided damage descriptions and merged description
    2. Retrieves similar historical estimates from Qdrant vector database
    3. Combines retrieved chunks with PSS data and damage descriptions
    4. Generates a final repair estimate
    
    Args:
        request: RAGEstimateRequest containing:
            - vehicle_info: Vehicle information (VIN, make, model, year, body_type)
            - side: Side of vehicle where damage is located
            - images: List of image URLs
            - damage_descriptions: List of damage descriptions
            - merged_damage_description: Merged narrative of all damages
            - pss_data: PSS (Parts and Service Standards) data as a dictionary/JSON object
            - custom_estimate_prompt: Optional custom prompt template (uses placeholders: {vehicle_info}, {damage_descriptions}, {human_description}, {retrieved_chunks}, {pss_data})
    
    Returns:
        RAGEstimateResponse with:
            - Generated estimate with line items
            - Processing metadata
    
    Example request:
        {
            "vehicle_info": {
                "vin": "1234567890",
                "make": "Subaru",
                "model": "Outback",
                "year": 2020,
                "body_type": "SUV"
            },
            "side": "rear",
            "images": ["s3://bucket/image1.jpg"],
            "damage_descriptions": [
                {
                    "location": "rear",
                    "part": "Rear Bumper",
                    "severity": "Medium",
                    "type": "Dent",
                    "start_position": "left",
                    "end_position": "center",
                    "description": "Dent on rear bumper"
                }
            ],
            "merged_damage_description": "Rear bumper damage with dent",
            "pss_data": {
                "parts": [...],
                "operations": [...]
            }
        }
    """
    if not request.damage_descriptions and not request.merged_damage_description:
        raise HTTPException(
            status_code=400,
            detail="Either damage_descriptions or merged_damage_description must be provided"
        )
    
    service = get_rag_service()
    response = service.run_rag_pipeline(request)
    
    if not response.success:
        raise HTTPException(
            status_code=500,
            detail=response.error or "Failed to generate estimate"
        )
    
    return response
