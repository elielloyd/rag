"""Routes for vehicle damage analysis."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from models import (
    ClassifyImagesRequest,
    ClassifyImagesResponse,
    AnalyzeSideImagesRequest,
    VehicleDamageAnalysisRequest,
    VehicleDamageAnalysisResponse,
    ChunkOutput,
)
from services import VehicleDamageService, QdrantService
from middleware.auth import verify_api_key

router = APIRouter(
    prefix="/vehicle-damage",
    tags=["Vehicle Damage Analysis"],
    dependencies=[Depends(verify_api_key)]
)


def get_vehicle_damage_service() -> VehicleDamageService:
    """Get or create the vehicle damage service instance."""
    return VehicleDamageService()


def get_qdrant_service() -> QdrantService:
    """Get or create the Qdrant service instance."""
    return QdrantService()


@router.post("/classify", response_model=ClassifyImagesResponse)
async def classify_images(request: ClassifyImagesRequest):
    """
    Step 1: Classify vehicle images by side.
    
    This endpoint reads images from S3 and classifies each image
    by vehicle side (front, rear, left, right, roof).
    
    Args:
        request: ClassifyImagesRequest containing:
            - bucket_url: S3 bucket URL containing vehicle images
            - image_urls: Optional list of specific S3 image URLs
    
    Returns:
        ClassifyImagesResponse with images grouped by side.
    """
    if not request.bucket_url and not request.image_urls:
        raise HTTPException(
            status_code=400,
            detail="Either bucket_url or image_urls must be provided"
        )
    
    service = get_vehicle_damage_service()
    
    response = service.classify_images_only(
        bucket_url=request.bucket_url,
        custom_classification_prompt=request.custom_classification_prompt,
    )
    
    if not response.success:
        raise HTTPException(
            status_code=500,
            detail=response.error or "Failed to classify images"
        )
    
    return response





@router.post("/analyze-side", response_model=ChunkOutput, response_model_exclude_none=True)
async def analyze_side_images(request: AnalyzeSideImagesRequest):
    """
    Step 2: Analyze images for a specific side and produce chunk output.
    
    Takes a side and list of images along with vehicle_info and approved_estimate,
    then produces damage descriptions using Gemini API.
    
    Args:
        request: AnalyzeSideImagesRequest containing:
            - side: Side of the vehicle (Front, Rear, Left, Right, Roof)
            - images: List of S3 image URLs for this side
            - vehicle_info: Vehicle information (returned as-is)
            - approved_estimate: Approved estimate operations (returned as-is)
    
    Returns:
        ChunkOutput matching the expected format with damage_descriptions from Gemini.
    
    Example request:
        {
            "side": "Front",
            "images": ["s3://bucket/image1.jpeg", "s3://bucket/image2.jpeg"],
            "vehicle_info": {...},
            "approved_estimate": {...}
        }
    """
    if not request.images:
        raise HTTPException(
            status_code=400,
            detail="images must be provided"
        )
    
    valid_sides = ["front", "rear", "left", "right", "roof"]
    if request.side.lower() not in valid_sides:
        raise HTTPException(
            status_code=400,
            detail=f"side must be one of: {valid_sides}"
        )
    
    service = get_vehicle_damage_service()
    
    chunk = service.analyze_side_images(
        side=request.side,
        images=request.images,
        vehicle_info=request.vehicle_info,
        approved_estimate=request.approved_estimate,
        custom_damage_analysis_prompt=request.custom_damage_analysis_prompt,
        custom_merge_damage_prompt=request.custom_merge_damage_prompt,
    )
    
    # Save chunk to Qdrant
    try:
        qdrant_service = get_qdrant_service()
        if qdrant_service.is_connected():
            qdrant_service.upload_damage_chunk(chunk)
    except Exception as e:
        print(f"Warning: Failed to save chunk to Qdrant: {e}")
    
    return chunk


@router.post("/analyze/chunks", response_model=list[ChunkOutput], response_model_exclude_none=True)
async def analyze_vehicle_damage_chunks(request: VehicleDamageAnalysisRequest):
    """
    Analyze vehicle damage and return chunks per side.
    
    Classifies images by side and analyzes each side separately,
    returning a list of ChunkOutput objects (one per side with images).
    
    Args:
        request: VehicleDamageAnalysisRequest containing:
            - bucket_url: S3 bucket URL containing vehicle images
            - vehicle_info: Vehicle information (VIN, make, model, year, body_type)
            - approved_estimate: Approved estimate operations by part category
    
    Returns:
        List of ChunkOutput, one per side that has images.
    """
    if not request.bucket_url:
        raise HTTPException(
            status_code=400,
            detail="bucket_url must be provided"
        )
    
    service = get_vehicle_damage_service()
    
    # First classify images by side
    classify_response = service.classify_images_only(
        bucket_url=request.bucket_url,
        custom_classification_prompt=request.custom_classification_prompt,
    )
    
    if not classify_response.success:
        raise HTTPException(
            status_code=500,
            detail=classify_response.error or "Failed to classify images"
        )
    
    chunks = []
    qdrant_service = get_qdrant_service()
    
    # Analyze each side that has images
    for side, images in classify_response.classified_images.items():
        if not images or side == "unknown":
            continue
        
        chunk = service.analyze_side_images(
            side=side,
            images=images,
            vehicle_info=request.vehicle_info,
            approved_estimate=request.approved_estimate,
            custom_damage_analysis_prompt=request.custom_damage_analysis_prompt,
            custom_merge_damage_prompt=request.custom_merge_damage_prompt,
        )
        
        # Save chunk to Qdrant
        try:
            if qdrant_service.is_connected():
                qdrant_service.upload_damage_chunk(chunk)
        except Exception as e:
            print(f"Warning: Failed to save chunk to Qdrant: {e}")
        
        chunks.append(chunk)
    
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No valid images found for any side"
        )
    
    return chunks


@router.post("/save-chunk")
async def save_chunk_to_qdrant(chunk: ChunkOutput):
    """
    Save a damage analysis chunk to Qdrant vector database.
    
    The merged_damage_description is used as the content for semantic search.
    All other vehicle and estimate info is stored as metadata.
    
    Args:
        chunk: ChunkOutput containing the damage analysis to save.
    
    Returns:
        Success status and the generated chunk ID.
    """
    qdrant_service = get_qdrant_service()
    
    if not qdrant_service.is_connected():
        raise HTTPException(status_code=503, detail="Qdrant is not connected")
    
    try:
        chunk_id = qdrant_service.upload_damage_chunk(chunk)
        return {
            "success": True,
            "chunk_id": chunk_id,
            "message": "Chunk saved to Qdrant successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save chunk: {str(e)}")
