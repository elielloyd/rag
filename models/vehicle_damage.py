"""Models for vehicle damage analysis."""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class VehicleSide(str, Enum):
    """Vehicle image classification categories."""
    FRONT = "front"
    REAR = "rear"
    LEFT = "left"
    RIGHT = "right"
    ROOF = "roof"
    INTERIOR = "interior"
    ENGINE_ELECTRICAL = "engine / electrical"
    STEERING_SUSPENSION = "steering / suspension"
    AC = "a/c"
    FRAME_FLOOR = "frame / floor"

    
    UNKNOWN = "unknown"


class VehicleInfo(BaseModel):
    """Vehicle information model."""
    vin: str = Field(description="Vehicle Identification Number")
    make: str = Field(description="Vehicle manufacturer")
    model: str = Field(description="Vehicle model")
    year: int = Field(description="Vehicle year")
    body_type: str = Field(description="Vehicle body type (e.g., Sedan, SUV)")


class EstimateOperation(BaseModel):
    """Single operation in an approved estimate."""
    Description: str = Field(description="Part or operation description")
    Operation: str = Field(description="Type of operation (e.g., Remove / Install, Repair)")
    LabourHours: Optional[float] = Field(default=None, description="Labour hours - only present for Repair operations")
    
    model_config = {"extra": "allow", "exclude_none": True}


class DamageDescription(BaseModel):
    """Individual damage description from Gemini analysis."""
    location: str = Field(description="Location on the vehicle (e.g., Front Right Corner)")
    part: str = Field(description="Affected part name (e.g., Front Bumper Cover)")
    severity: str = Field(description="Damage severity (Minor, Medium, Major)")
    type: str = Field(description="Type of damage (e.g., Scuffing, Scratches, Dent)")
    start_position: str = Field(description="Starting position of damage")
    end_position: str = Field(description="Ending position of damage")
    description: str = Field(description="Detailed description of the damage")


class ClassifyImagesRequest(BaseModel):
    """Request model for classifying images by vehicle side."""
    bucket_url: Optional[str] = Field(
        default="s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/",
        description="S3 bucket URL containing vehicle images",
        json_schema_extra={"example": "s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/"}
    )
    custom_classification_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for image classification. If provided, this will be used instead of the default prompt."
    )

class ClassifyImagesResponse(BaseModel):
    """Response model for image classification."""
    success: bool = Field(description="Whether the classification was successful")
    classified_images: dict[str, list[str]] = Field(description="Images classified by side (front, rear, left, right, roof, unknown)")
    total_images: int = Field(description="Total number of images processed")
    processing_time_seconds: float = Field(description="Processing time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if classification failed")



class AnalyzeSideImagesRequest(BaseModel):
    """Request model for analyzing images of a specific side."""
    side: str = Field(default="rear", description="Side of the vehicle (front, rear, left, right, roof)")
    images: list[str] = Field(
        default=[
            "s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/11_af787c1a-c89a-4c0d-ade8-cff3731bfd65.jpeg",
            "s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/1_0be7ae45-f870-406b-b1f7-01aaf229ba1a.jpeg",
            "s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/2_11595b3d-1f86-4d6d-9d1e-547015464420.jpeg",
            "s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/3_150db621-efbc-4de8-90f4-646fd946d957.jpeg",
            "s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/4_282bd1ba-9253-4241-8abc-eeab2a3db1c4.jpeg",
            "s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/5_2b77d545-a1b9-413e-a7e1-3883fe38d6d2.jpeg",
            "s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/6_3379f29a-a516-4f14-aeee-97777e693c53.jpeg",
            "s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/8_91c0e141-d7dd-4e52-8dc3-28adfd6e12bc.jpeg"
        ],
        description="List of S3 image URLs for this side"
    )
    vehicle_info: VehicleInfo = Field(
        default=VehicleInfo(
            vin="4S4BTDNC3L3195200",
            make="SUBARU",
            model="OUTBACK 2.5i LIMITED FAMILIALE TI",
            year=2020,
            body_type="Sedan"
        ),
        description="Vehicle information"
    )
    approved_estimate: dict[str, list[EstimateOperation]] = Field(
        default={
            "Rear Bumper": [
                EstimateOperation(Description="Rear Bumper Cover", Operation="Remove / Replace")
            ]
        },
        description="Approved estimate operations by part category"
    )
    custom_damage_analysis_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for damage analysis. Use placeholders: {year}, {make}, {model}, {body_type}, {side}, {approved_estimate}"
    )
    custom_merge_damage_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for merging damage descriptions. Use placeholders: {year}, {make}, {model}, {body_type}, {damage_descriptions}"
    )

    n8n_uuid: Optional[str] = Field(
        default=None,
        description="N8N UUID of the claim"
    )

    mitchell_url_key: Optional[str] = Field(
        default=None,
        description="Mitchell URL key of the claim"
    )
    account_id: Optional[int] = Field(
        default=None,
        description="Account ID"
    )


class VehicleDamageAnalysisRequest(BaseModel):
    """Request model for vehicle damage analysis."""
    bucket_url: Optional[str] = Field(
        default="s3://ehsan-poc-estimate-true-claim/claims/test-claim/images/",
        description="S3 bucket URL containing vehicle images (e.g., s3://bucket/claims/id/images/)"
    )
    vehicle_info: VehicleInfo = Field(
        default=VehicleInfo(
            vin="4S4BTDNC3L3195200",
            make="SUBARU",
            model="OUTBACK 2.5i LIMITED FAMILIALE TI",
            year=2020,
            body_type="Sedan"
        ),
        description="Vehicle information"
    )
    approved_estimate: dict[str, list[EstimateOperation]] = Field(
        default={
            "Rear Bumper": [
                EstimateOperation(Description="Rear Bumper Cover", Operation="Remove / Replace")
            ]
        },
        description="Approved estimate operations by part category"
    )
    custom_classification_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for image classification. If provided, this will be used instead of the default prompt."
    )
    custom_damage_analysis_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for damage analysis. Use placeholders: {year}, {make}, {model}, {body_type}, {side}, {approved_estimate}"
    )
    custom_merge_damage_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for merging damage descriptions. Use placeholders: {year}, {make}, {model}, {body_type}, {damage_descriptions}"
    )
    n8n_uuid: Optional[str] = Field(
        default=None,
        description="N8N UUID of the claim"
    )
    mitchell_url_key: Optional[str] = Field(
        default=None,
        description="Mitchell URL key of the claim"
    )
    account_id: Optional[int] = Field(
        default=None,
        description="Account ID"
    )


class VehicleDamageAnalysisResponse(BaseModel):
    """Response model for vehicle damage analysis."""
    success: bool = Field(description="Whether the analysis was successful")
    vehicle_info: VehicleInfo = Field(description="Vehicle information")
    classified_images: dict[str, list[str]] = Field(description="Images classified by side (front, rear, left, right, roof)")
    damage_descriptions: list[DamageDescription] = Field(default_factory=list, description="All damage descriptions")
    merged_damage_description: str = Field(default="", description="Merged narrative of all damages")
    approved_estimate: dict[str, list[EstimateOperation]] = Field(description="Approved estimate operations")
    processing_time_seconds: float = Field(description="Total processing time")
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")


class ChunkOutput(BaseModel):
    """Complete chunk output matching the expected format."""
    vehicle_info: VehicleInfo = Field(description="Vehicle information")
    side: str = Field(description="Primary side analyzed")
    images: list[str] = Field(description="S3 URLs of analyzed images")
    damage_descriptions: list[DamageDescription] = Field(description="All damage descriptions")
    merged_damage_description: str = Field(description="Merged narrative of all damages")
    approved_estimate: dict[str, list[EstimateOperation]] = Field(description="Approved estimate operations")
    n8n_uuid: Optional[str] = Field(default=None, description="N8N UUID of the claim")
    mitchell_url_key: Optional[str] = Field(default=None, description="Mitchell URL key of the claim")
    account_id: Optional[int] = Field(default=None, description="Account ID")
    
    def model_dump(self, **kwargs):
        """Override to exclude None values from nested EstimateOperation."""
        kwargs.setdefault('exclude_none', True)
        return super().model_dump(**kwargs)
