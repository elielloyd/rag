"""Models for RAG (Retrieval Augmented Generation) pipeline."""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from .vehicle_damage import VehicleInfo, DamageDescription, EstimateOperation


class DamageDetectionResult(BaseModel):
    """Result from damage detection on a single image."""
    image_url: str = Field(description="S3 URL of the analyzed image")
    has_damage: bool = Field(description="Whether damage was detected in the image")
    side: str = Field(description="Detected side of vehicle (front, rear, left, right, roof, unknown)")
    damages: list[DamageDescription] = Field(default_factory=list, description="List of detected damages")
    confidence: float = Field(default=0.0, description="Confidence score for damage detection")


class RetrievedChunk(BaseModel):
    """A chunk retrieved from Qdrant vector database."""
    score: float = Field(description="Similarity score")
    content: str = Field(description="The damage description content")
    vehicle_info: dict = Field(description="Vehicle info from the retrieved chunk")
    side: str = Field(description="Side of vehicle from retrieved chunk")
    damage_descriptions: list[dict] = Field(default_factory=list, description="Damage descriptions from chunk")
    approved_estimate: dict = Field(default_factory=dict, description="Approved estimate from chunk")


class RAGEstimateRequest(BaseModel):
    """Request model for RAG-based estimate generation."""
    vehicle_info: Optional[VehicleInfo] = Field(
        default=None,
        description="Vehicle information (VIN, make, model, year, body_type)"
    )
    side: Optional[str] = Field(
        default=None,
        description="Side of the vehicle where damage is located (front, rear, left, right, roof)"
    )
    images: Optional[list[str]] = Field(
        default=None,
        description="List of image URLs or base64 encoded images"
    )
    damage_descriptions: Optional[list[DamageDescription]] = Field(
        default=None,
        description="List of damage descriptions with location, part, severity, type, positions, and description"
    )
    merged_damage_description: Optional[str] = Field(
        default=None,
        description="Merged narrative description of all damages"
    )
    pss_url: Optional[str] = Field(
        default=None,
        description="S3 URL to PSS (Parts and Service Standards) JSON file",
        json_schema_extra={"example": "s3://ehsan-poc-estimate-true-claim/pss/subaru_outback_2020_2024.json"}
    )
    custom_estimate_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for estimate generation. If provided, this will be used instead of the default prompt. Use placeholders: {vehicle_info}, {damage_descriptions}, {human_description}, {retrieved_chunks}, {pss_data}"
    )


class EstimateOperation(BaseModel):
    """A single operation in the generated estimate."""
    Description: str = Field(description="Part or operation description")
    Operation: str = Field(description="Type of operation (Remove / Install, Remove / Replace, Repair, Overhaul, etc.)")
    LaborHours: Optional[float] = Field(default=None, description="Labor hours - only present for Repair operations")
    
    model_config = {"extra": "allow", "exclude_none": True}


class GeneratedEstimate(BaseModel):
    """The final generated estimate in approved_estimate format."""
    estimate: dict[str, list[EstimateOperation]] = Field(
        default_factory=dict,
        description="Estimate operations grouped by part category (e.g., 'Front Bumper', 'Rear Bumper')"
    )


class RAGEstimateResponse(BaseModel):
    """Response model for RAG-based estimate generation."""
    success: bool = Field(description="Whether the estimate generation was successful")
    
    # Input analysis
    images_analyzed: int = Field(default=0, description="Number of images analyzed")
    images_with_damage: int = Field(default=0, description="Number of images with detected damage")
    damage_detections: list[DamageDetectionResult] = Field(
        default_factory=list, 
        description="Damage detection results per image"
    )
    
    # # Retrieval results
    # retrieved_chunks: list[RetrievedChunk] = Field(
    #     default_factory=list,
    #     description="Similar chunks retrieved from Qdrant"
    # )
    
    # Generated estimate
    generated_estimate: Optional[GeneratedEstimate] = Field(
        default=None,
        description="The final generated estimate"
    )
    
    # Metadata
    vehicle_info: Optional[VehicleInfo] = Field(default=None, description="Vehicle information used")
    human_damage_description: Optional[str] = Field(default=None, description="Human-provided damage description")
    pss_data_used: bool = Field(default=False, description="Whether PSS data was used")
    processing_time_seconds: float = Field(default=0.0, description="Total processing time")
    error: Optional[str] = Field(default=None, description="Error message if generation failed")


class DamageDetectionRequest(BaseModel):
    """Request model for damage detection only (without estimate generation)."""
    bucket_url: Optional[str] = Field(
        default=None,
        description="S3 bucket URL containing vehicle images"
    )
    image_urls: Optional[list[str]] = Field(
        default=None,
        description="List of specific S3 image URLs to analyze"
    )
    vehicle_info: Optional[VehicleInfo] = Field(
        default=None,
        description="Vehicle information for context"
    )


class DamageDetectionResponse(BaseModel):
    """Response model for damage detection only."""
    success: bool = Field(description="Whether detection was successful")
    total_images: int = Field(default=0, description="Total images processed")
    images_with_damage: int = Field(default=0, description="Images with detected damage")
    detections: list[DamageDetectionResult] = Field(
        default_factory=list,
        description="Detection results per image"
    )
    merged_damage_description: str = Field(
        default="",
        description="Merged narrative of all detected damages"
    )
    processing_time_seconds: float = Field(default=0.0, description="Processing time")
    error: Optional[str] = Field(default=None, description="Error message if failed")
