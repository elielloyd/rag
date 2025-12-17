"""Service for vehicle damage analysis using Gemini API and S3."""

import base64
import time
from typing import Optional, Literal
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field
from PIL import Image

from config import settings
from models.vehicle_damage import (
    VehicleSide,
    VehicleInfo,
    DamageDescription,
    ClassifyImagesResponse,
    VehicleDamageAnalysisResponse,
    ChunkOutput,
    EstimateOperation,
)
from prompts.vehicle_damage import (
    get_classification_prompt,
    get_damage_analysis_prompt,
    get_merge_damage_prompt,
)
from services.s3_service import S3Service
import os 

os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY'] = settings.langsmith_api_key
os.environ['LANGSMITH_PROJECT'] = settings.langsmith_project or 'default'


class ClassificationResult(BaseModel):
    """Structured output for image classification."""
    side: Literal["front", "rear", "left", "right", "roof", "unknown"] = Field(
        description="Classified side of the vehicle"
    )
    confidence: float = Field(
        description="Classification confidence score (0-1)"
    )


class DamageItem(BaseModel):
    """Single damage item in the analysis."""
    location: str = Field(description="Location on the vehicle")
    part: str = Field(description="Affected part name")
    severity: Literal["Minor", "Medium", "Major"] = Field(description="Damage severity")
    type: str = Field(description="Type of damage")
    start_position: str = Field(description="Starting position of damage")
    end_position: str = Field(description="Ending position of damage")
    description: str = Field(description="Detailed description of the damage")


class DamageAnalysisResult(BaseModel):
    """Structured output for damage analysis."""
    damage_descriptions: list[DamageItem] = Field(
        description="List of damage descriptions"
    )

class VehicleDamageService:
    """Service for analyzing vehicle damage from S3 images using Gemini API."""
    
    def __init__(self):
        """Initialize the service with Gemini client and S3 service."""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        
        self.model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            temperature=1.0,
            media_resolution="MEDIA_RESOLUTION_HIGH",
        )
        self.classification_model = self.model.with_structured_output(
            schema=ClassificationResult,
            method="json_schema",
        )
        self.damage_analysis_model = self.model.with_structured_output(
            schema=DamageAnalysisResult,
            method="json_schema",
        )
        self.s3_service = S3Service()
    
    
    def classify_image(self, image_data: bytes, mime_type: str = "image/jpeg") -> tuple[VehicleSide, float]:
        """
        Classify a vehicle image to determine which side it shows.
        
        Args:
            image_data: Raw image bytes
            mime_type: MIME type of the image
        
        Returns:
            Tuple of (VehicleSide, confidence)
        """
        prompt = get_classification_prompt()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "base64": image_base64,
                    "mime_type": mime_type,
                },
            ]
        )
        
        result: ClassificationResult = self.classification_model.invoke([message])
        side = VehicleSide(result.side)
        confidence = result.confidence
        
        return side, confidence
    
    def analyze_damage(
        self,
        images_data: list[tuple[bytes, str]],
        vehicle_info: VehicleInfo,
        side: VehicleSide,
        approved_estimate: dict,
    ) -> list[DamageDescription]:
        """
        Analyze damage from multiple images of the same vehicle side.
        
        Args:
            images_data: List of (image_bytes, mime_type) tuples
            vehicle_info: Vehicle information
            side: Side of the vehicle being analyzed
            approved_estimate: Approved estimate operations
        
        Returns:
            List of DamageDescription objects
        """
        if not images_data:
            return []
        
        prompt = get_damage_analysis_prompt(
            year=vehicle_info.year,
            make=vehicle_info.make,
            model=vehicle_info.model,
            body_type=vehicle_info.body_type,
            side=side.value,
            approved_estimate=approved_estimate,
        )
        
        content_parts = [{"type": "text", "text": prompt}]
        for image_data, mime_type in images_data:
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            content_parts.append({
                "type": "image",
                "base64": image_base64,
                "mime_type": mime_type,
            })
        
        message = HumanMessage(content=content_parts)
        
        result: DamageAnalysisResult = self.damage_analysis_model.invoke([message])
        damage_descriptions = []
        for damage_item in result.damage_descriptions:
            damage_descriptions.append(DamageDescription(**damage_item.model_dump()))
        
        return damage_descriptions
    
    def merge_damage_descriptions(
        self,
        vehicle_info: VehicleInfo,
        damage_descriptions: list[DamageDescription],
    ) -> str:
        """
        Create a merged narrative from all damage descriptions.
        
        Args:
            vehicle_info: Vehicle information
            damage_descriptions: List of all damage descriptions
        
        Returns:
            Merged damage description narrative
        """
        if not damage_descriptions:
            return "No visible damage detected on the vehicle."
        
        prompt = get_merge_damage_prompt(
            year=vehicle_info.year,
            make=vehicle_info.make,
            model=vehicle_info.model,
            body_type=vehicle_info.body_type,
            damage_descriptions=[d.model_dump() for d in damage_descriptions],
        )
        
        message = HumanMessage(content=prompt)
        response = self.model.invoke([message])
        
        return response.text.strip()
    
    def analyze_vehicle_damage(
        self,
        bucket_url: str,
        vehicle_info: VehicleInfo = None,
        approved_estimate: dict = None,
    ) -> VehicleDamageAnalysisResponse:
        """
        Complete vehicle damage analysis pipeline.
        
        Args:
            bucket_url: S3 bucket URL containing vehicle images
            vehicle_info: Vehicle information
            approved_estimate: Approved estimate operations
        
        Returns:
            VehicleDamageAnalysisResponse with all analysis results
        """
        start_time = time.time()
        
        try:
            all_image_urls = self.s3_service.list_images_from_url(bucket_url)
            
            if not all_image_urls:
                return VehicleDamageAnalysisResponse(
                    success=False,
                    vehicle_info=vehicle_info,
                    classified_images={},
                    damage_descriptions=[],
                    merged_damage_description="",
                    approved_estimate=approved_estimate or {},
                    processing_time_seconds=time.time() - start_time,
                    error="No images found in the specified location",
                )
            
            classified_images: dict[str, list[str]] = {
                "front": [],
                "rear": [],
                "left": [],
                "right": [],
                "roof": [],
                "unknown": [],
            }
            image_data_by_side: dict[str, list[tuple[bytes, str]]] = {
                "front": [],
                "rear": [],
                "left": [],
                "right": [],
                "roof": [],
            }
            
            for s3_url in all_image_urls:
                try:
                    image_data, mime_type = self.s3_service.get_image(s3_url)
                    side, confidence = self.classify_image(image_data, mime_type)
                    classified_images[side.value].append(s3_url)
                    
                    if side != VehicleSide.UNKNOWN:
                        image_data_by_side[side.value].append((image_data, mime_type))
                except Exception as e:
                    print(f"Error processing image {s3_url}: {e}")
                    classified_images["unknown"].append(s3_url)
            
            all_damage_descriptions: list[DamageDescription] = []
            
            for side_str, images_data in image_data_by_side.items():
                if images_data:
                    side = VehicleSide(side_str)
                    damages = self.analyze_damage(
                        images_data=images_data,
                        vehicle_info=vehicle_info,
                        side=side,
                        approved_estimate=approved_estimate or {},
                    )
                    all_damage_descriptions.extend(damages)
            
            merged_description = self.merge_damage_descriptions(
                vehicle_info=vehicle_info,
                damage_descriptions=all_damage_descriptions,
            )
            
            processing_time = time.time() - start_time
            
            return VehicleDamageAnalysisResponse(
                success=True,
                vehicle_info=vehicle_info,
                classified_images=classified_images,
                damage_descriptions=all_damage_descriptions,
                merged_damage_description=merged_description,
                approved_estimate=approved_estimate or {},
                processing_time_seconds=processing_time,
            )
            
        except Exception as e:
            return VehicleDamageAnalysisResponse(
                success=False,
                vehicle_info=vehicle_info,
                classified_images={},
                damage_descriptions=[],
                merged_damage_description="",
                approved_estimate=approved_estimate or {},
                processing_time_seconds=time.time() - start_time,
                error=str(e),
            )
    
    def _classify_single_image(self, s3_url: str) -> tuple[str, VehicleSide]:
        """
        Classify a single image from S3. Helper for parallel processing.
        
        Args:
            s3_url: S3 URL of the image
        
        Returns:
            Tuple of (s3_url, VehicleSide)
        """
        try:
            image_data, mime_type = self.s3_service.get_image(s3_url)
            side, confidence = self.classify_image(image_data, mime_type)
            return s3_url, side
        except Exception as e:
            print(f"Error processing image {s3_url}: {e}")
            return s3_url, VehicleSide.UNKNOWN
    
    def classify_images_only(
        self,
        bucket_url: Optional[str] = None,
        image_urls: Optional[list[str]] = None,
        max_workers: int = 10,
    ) -> ClassifyImagesResponse:
        """
        Classify images by vehicle side without damage analysis.
        Uses parallel processing for faster classification.
        
        Args:
            bucket_url: S3 bucket URL containing vehicle images
            image_urls: Optional list of specific S3 image URLs
            max_workers: Maximum number of parallel workers (default: 10)
        
        Returns:
            ClassifyImagesResponse with classified images by side
        """
        start_time = time.time()
        
        try:
            if image_urls:
                all_image_urls = image_urls
            elif bucket_url:
                all_image_urls = self.s3_service.list_images_from_url(bucket_url)
            else:
                raise ValueError("Either bucket_url or image_urls must be provided")
            
            if not all_image_urls:
                return ClassifyImagesResponse(
                    success=False,
                    classified_images={},
                    total_images=0,
                    processing_time_seconds=time.time() - start_time,
                    error="No images found in the specified location",
                )
            
            classified_images: dict[str, list[str]] = {
                "front": [],
                "rear": [],
                "left": [],
                "right": [],
                "roof": [],
                "unknown": [],
            }
            
            # Process images in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(self._classify_single_image, s3_url): s3_url
                    for s3_url in all_image_urls
                }
                
                for future in as_completed(futures):
                    try:
                        s3_url, side = future.result()
                        classified_images[side.value].append(s3_url)
                    except Exception as e:
                        s3_url = futures[future]
                        print(f"Error processing image {s3_url}: {e}")
                        classified_images["unknown"].append(s3_url)
            
            return ClassifyImagesResponse(
                success=True,
                classified_images=classified_images,
                total_images=len(all_image_urls),
                processing_time_seconds=time.time() - start_time,
            )
            
        except Exception as e:
            return ClassifyImagesResponse(
                success=False,
                classified_images={},
                total_images=0,
                processing_time_seconds=time.time() - start_time,
                error=str(e),
            )
    
    def analyze_side_images(
        self,
        side: str,
        images: list[str],
        vehicle_info: VehicleInfo,
        approved_estimate: dict[str, list[EstimateOperation]],
    ) -> ChunkOutput:
        """
        Analyze images for a specific side and produce chunk output.
        
        Args:
            side: Side of the vehicle (front, rear, left, right, roof)
            images: List of S3 image URLs for this side
            vehicle_info: Vehicle information
            approved_estimate: Approved estimate operations
        
        Returns:
            ChunkOutput with damage descriptions from Gemini
        """
        # Download images
        images_data: list[tuple[bytes, str]] = []
        for s3_url in images:
            try:
                image_data, mime_type = self.s3_service.get_image(s3_url)
                images_data.append((image_data, mime_type))
            except Exception as e:
                print(f"Error downloading image {s3_url}: {e}")
        
        # Analyze damage
        damage_descriptions: list[DamageDescription] = []
        if images_data:
            vehicle_side = VehicleSide(side.lower())
            damage_descriptions = self.analyze_damage(
                images_data=images_data,
                vehicle_info=vehicle_info,
                side=vehicle_side,
                approved_estimate=approved_estimate,
            )
        
        # Merge damage descriptions
        merged_description = self.merge_damage_descriptions(
            vehicle_info=vehicle_info,
            damage_descriptions=damage_descriptions,
        )
        
        return ChunkOutput(
            vehicle_info=vehicle_info,
            side=side.capitalize(),
            images=images,
            damage_descriptions=damage_descriptions,
            merged_damage_description=merged_description,
            approved_estimate=approved_estimate,
        )
    
    def generate_chunk_output(
        self,
        response: VehicleDamageAnalysisResponse,
        primary_side: str = "Front",
    ) -> ChunkOutput:
        """
        Generate the final chunk output in the expected format.
        
        Args:
            response: VehicleDamageAnalysisResponse from analysis
            primary_side: Primary side to report (default: Front)
        
        Returns:
            ChunkOutput matching the expected format
        """
        all_images = []
        for side, urls in response.classified_images.items():
            all_images.extend(urls)
        
        return ChunkOutput(
            vehicle_info=response.vehicle_info,
            side=primary_side,
            images=all_images,
            damage_descriptions=response.damage_descriptions,
            merged_damage_description=response.merged_damage_description,
            approved_estimate=response.approved_estimate,
        )
