"""Service for RAG (Retrieval Augmented Generation) pipeline."""

import base64
import time
from typing import Optional, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import HumanMessage
from pydantic import BaseModel, Field

from config import settings
from models.vehicle_damage import VehicleInfo, DamageDescription
from models.rag_models import (
    DamageDetectionResult,
    RetrievedChunk,
    RAGEstimateRequest,
    RAGEstimateResponse,
    GeneratedEstimate,
    DamageDetectionResponse,
)
from prompts.rag_prompts import (
    get_damage_detection_prompt,
    get_damage_detection_with_context_prompt,
    get_estimate_generation_prompt,
)
from services.s3_service import S3Service
from services.qdrant_service import QdrantService

import os

os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGSMITH_API_KEY'] = settings.langsmith_api_key
os.environ['LANGSMITH_PROJECT'] = settings.langsmith_project or 'default'


class DamageDetectionOutput(BaseModel):
    """Structured output for damage detection."""
    side: Literal[
        "front", "rear", "left", "right", "roof", "unknown",
       "interior", "engine / electrical", "steering / suspension", "a/c", "frame / floor"
    ] = Field(
        description="Detected side of the vehicle"
    )
    has_damage: bool = Field(
        description="Whether damage was detected in the image"
    )
    confidence: float = Field(
        description="Confidence score for the detection (0-1)"
    )
    damages: list[dict] = Field(
        default_factory=list,
        description="List of detected damages with location, part, severity, type, start_position, end_position, description"
    )


class EstimateOperationOutput(BaseModel):
    """Single operation in the estimate output."""
    Description: str = Field(description="Part or operation description")
    Operation: str = Field(description="Type of operation")
    LaborHours: Optional[float] = Field(default=None, description="Labor hours - only for Repair operations")
    PartId: Optional[str] = Field(default=None, description="ID of the part from PSS data that needs to be replaced or repaired")


class EstimateOutput(BaseModel):
    """Structured output for estimate generation in approved_estimate format."""
    estimate: dict[str, list[dict]] = Field(
        default_factory=dict,
        description="Estimate operations grouped by part category. Each key is a part category (e.g., 'Rear Bumper'), and each value is a list of operations with Description, Operation, optionally LaborHours, and PartId (ID from PSS data)."
    )


class RAGService:
    """Service for RAG-based damage estimation pipeline."""
    
    def __init__(self):
        """Initialize the RAG service with required components."""
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        
        # Initialize Gemini model for damage detection
        self.model = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            temperature=1.0,
            media_resolution="MEDIA_RESOLUTION_HIGH",
        )
        
        # Structured output models
        self.damage_detection_model = self.model.with_structured_output(
            schema=DamageDetectionOutput,
            method="json_schema",
        )
        self.estimate_model = self.model.with_structured_output(
            schema=EstimateOutput,
            method="json_schema",
        )
        
        # Initialize services
        self.s3_service = S3Service()
        self.qdrant_service = QdrantService()
    
    def detect_damage_single_image(
        self,
        image_data: bytes,
        mime_type: str,
        s3_url: str,
        vehicle_info: Optional[VehicleInfo] = None,
        human_description: Optional[str] = None,
    ) -> DamageDetectionResult:
        """
        Detect damage in a single image.
        
        Args:
            image_data: Raw image bytes
            mime_type: MIME type of the image
            s3_url: S3 URL of the image
            vehicle_info: Optional vehicle information for context
            human_description: Optional human-provided damage description
        
        Returns:
            DamageDetectionResult with detected damages
        """
        # Get appropriate prompt
        if vehicle_info:
            prompt = get_damage_detection_with_context_prompt(
                year=vehicle_info.year,
                make=vehicle_info.make,
                model=vehicle_info.model,
                body_type=vehicle_info.body_type,
                human_description=human_description,
            )
        else:
            prompt = get_damage_detection_prompt()
        
        # Encode image
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Create message with image
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
        
        # Get detection result
        result: DamageDetectionOutput = self.damage_detection_model.invoke([message])
        
        # Convert damages to DamageDescription objects
        damages = []
        for damage_dict in result.damages:
            try:
                damages.append(DamageDescription(
                    location=damage_dict.get('location', 'Unknown'),
                    part=damage_dict.get('part', 'Unknown'),
                    severity=damage_dict.get('severity', 'Unknown'),
                    type=damage_dict.get('type', 'Unknown'),
                    start_position=damage_dict.get('start_position', 'N/A'),
                    end_position=damage_dict.get('end_position', 'N/A'),
                    description=damage_dict.get('description', ''),
                ))
            except Exception as e:
                print(f"Warning: Could not parse damage: {e}")
        
        return DamageDetectionResult(
            image_url=s3_url,
            has_damage=result.has_damage,
            side=result.side,
            damages=damages,
            confidence=result.confidence,
        )
    
    def _detect_damage_worker(
        self,
        s3_url: str,
        vehicle_info: Optional[VehicleInfo] = None,
        human_description: Optional[str] = None,
    ) -> DamageDetectionResult:
        """Worker function for parallel damage detection."""
        try:
            image_data, mime_type = self.s3_service.get_image(s3_url)
            return self.detect_damage_single_image(
                image_data=image_data,
                mime_type=mime_type,
                s3_url=s3_url,
                vehicle_info=vehicle_info,
                human_description=human_description,
            )
        except Exception as e:
            print(f"Error detecting damage in {s3_url}: {e}")
            return DamageDetectionResult(
                image_url=s3_url,
                has_damage=False,
                side="unknown",
                damages=[],
                confidence=0.0,
            )
    
    def detect_damage_batch(
        self,
        bucket_url: Optional[str] = None,
        image_urls: Optional[list[str]] = None,
        vehicle_info: Optional[VehicleInfo] = None,
        human_description: Optional[str] = None,
        max_workers: int = 10,
    ) -> DamageDetectionResponse:
        """
        Detect damage in multiple images.
        
        Args:
            bucket_url: S3 bucket URL containing images
            image_urls: Optional list of specific image URLs
            vehicle_info: Optional vehicle information
            human_description: Optional human-provided description
            max_workers: Maximum parallel workers
        
        Returns:
            DamageDetectionResponse with all detection results
        """
        start_time = time.time()
        
        try:
            # Get image URLs
            if image_urls:
                all_image_urls = image_urls
            elif bucket_url:
                all_image_urls = self.s3_service.list_images_from_url(bucket_url)
            else:
                return DamageDetectionResponse(
                    success=False,
                    error="Either bucket_url or image_urls must be provided",
                    processing_time_seconds=time.time() - start_time,
                )
            
            if not all_image_urls:
                return DamageDetectionResponse(
                    success=False,
                    error="No images found",
                    processing_time_seconds=time.time() - start_time,
                )
            
            # Detect damage in parallel
            detections: list[DamageDetectionResult] = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self._detect_damage_worker,
                        s3_url,
                        vehicle_info,
                        human_description,
                    ): s3_url
                    for s3_url in all_image_urls
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    detections.append(result)
            
            # Count images with damage
            images_with_damage = sum(1 for d in detections if d.has_damage)
            
            # Create merged description
            all_damages = []
            for detection in detections:
                all_damages.extend(detection.damages)
            
            merged_description = self._merge_damage_descriptions(all_damages, vehicle_info)
            
            return DamageDetectionResponse(
                success=True,
                total_images=len(all_image_urls),
                images_with_damage=images_with_damage,
                detections=detections,
                merged_damage_description=merged_description,
                processing_time_seconds=time.time() - start_time,
            )
            
        except Exception as e:
            return DamageDetectionResponse(
                success=False,
                error=str(e),
                processing_time_seconds=time.time() - start_time,
            )
    
    def _merge_damage_descriptions(
        self,
        damages: list[DamageDescription],
        vehicle_info: Optional[VehicleInfo] = None,
    ) -> str:
        """Create a merged narrative from all damage descriptions."""
        if not damages:
            return "No visible damage detected."
        
        # Group damages by part
        damages_by_part = {}
        for damage in damages:
            part = damage.part
            if part not in damages_by_part:
                damages_by_part[part] = []
            damages_by_part[part].append(damage)
        
        # Build narrative
        parts = []
        for part, part_damages in damages_by_part.items():
            severities = [d.severity for d in part_damages]
            types = list(set(d.type for d in part_damages))
            max_severity = "Major" if "Major" in severities else ("Medium" if "Medium" in severities else "Minor")
            parts.append(f"{part} ({max_severity} {', '.join(types)})")
        
        vehicle_str = ""
        if vehicle_info:
            vehicle_str = f"The {vehicle_info.year} {vehicle_info.make} {vehicle_info.model} shows "
        else:
            vehicle_str = "The vehicle shows "
        
        return vehicle_str + "damage to: " + "; ".join(parts) + "."
    
    def _extract_pss_parts(self, pss_data: Optional[dict]) -> dict[str, dict]:
        """
        Extract all parts from PSS data into a searchable dictionary.
        
        Returns a dict mapping part descriptions to their IDs and details.
        Format: {part_description: {"id": part_id, "full_description": full_desc, ...}}
        """
        if not pss_data:
            return {}
        
        parts_map = {}
        
        try:
            categories = pss_data.get("Categories", [])
            for category in categories:
                subcategories = category.get("SubCategories", [])
                for subcategory in subcategories:
                    parts = subcategory.get("Parts", [])
                    for part in parts:
                        part_details = part.get("PartDetails", [])
                        for detail in part_details:
                            # Use FullDescription as primary key
                            full_desc = detail.get("FullDescription", "")
                            part_id = detail.get("Id", "")
                            
                            if full_desc and part_id:
                                parts_map[full_desc.lower()] = {
                                    "id": str(part_id),
                                    "full_description": full_desc,
                                    "description": detail.get("Part", {}).get("Description", ""),
                                    "available_operations": detail.get("AvailableOperations", []),
                                }
                            
                            # Also map by Part Description for flexibility
                            part_desc = detail.get("Part", {}).get("Description", "")
                            if part_desc and part_id and part_desc.lower() not in parts_map:
                                parts_map[part_desc.lower()] = {
                                    "id": str(part_id),
                                    "full_description": full_desc,
                                    "description": part_desc,
                                    "available_operations": detail.get("AvailableOperations", []),
                                }
        except Exception as e:
            print(f"Warning: Error extracting PSS parts: {e}")
        
        return parts_map
    
    def _match_part_with_pss(
        self,
        part_description: str,
        pss_parts_map: dict[str, dict],
    ) -> Optional[str]:
        """
        Match a part description with PSS data and return the PartId.
        
        Uses fuzzy matching to find the best match.
        """
        if not part_description or not pss_parts_map:
            return None
        
        part_lower = part_description.lower().strip()
        
        # Exact match first
        if part_lower in pss_parts_map:
            return pss_parts_map[part_lower]["id"]
        
        # Partial match - check if description contains key terms
        for pss_desc, part_info in pss_parts_map.items():
            # Check if part description contains key words from PSS part
            pss_words = set(pss_desc.split())
            desc_words = set(part_lower.split())
            
            # If significant overlap, consider it a match
            if len(pss_words.intersection(desc_words)) >= 2:
                return part_info["id"]
            
            # Check if PSS description contains the part description or vice versa
            if part_lower in pss_desc or pss_desc in part_lower:
                return part_info["id"]
        
        return None
    
    def retrieve_similar_chunks(
        self,
        damage_description: str,
        top_k: int = 5,
        score_threshold: Optional[float] = 0.5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve similar damage chunks from Qdrant.
        
        Args:
            damage_description: The damage description to search for
            top_k: Number of results to retrieve
            score_threshold: Minimum similarity score
        
        Returns:
            List of RetrievedChunk objects
        """
        if not self.qdrant_service.is_connected():
            print("Warning: Qdrant is not connected")
            return []
        
        try:
            results = self.qdrant_service.search(
                query=damage_description,
                limit=top_k,
                score_threshold=score_threshold,
            )
            
            chunks = []
            for result in results:
                payload = result.get('payload', {})
                chunks.append(RetrievedChunk(
                    score=result.get('score', 0.0),
                    content=payload.get('content', ''),
                    vehicle_info=payload.get('vehicle_info', {}),
                    side=payload.get('side', 'unknown'),
                    damage_descriptions=payload.get('damage_descriptions', []),
                    approved_estimate=payload.get('approved_estimate', {}),
                ))
            
            return chunks
            
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []
    
    def generate_estimate(
        self,
        damage_descriptions: list[DamageDescription],
        retrieved_chunks: list[RetrievedChunk],
        vehicle_info: Optional[VehicleInfo] = None,
        human_description: Optional[str] = None,
        pss_data: Optional[dict] = None,
        custom_prompt: Optional[str] = None,
    ) -> GeneratedEstimate:
        """
        Generate an estimate based on damage and retrieved chunks.
        
        Args:
            damage_descriptions: Detected damages
            retrieved_chunks: Similar chunks from Qdrant
            vehicle_info: Vehicle information
            human_description: Human-provided description
            pss_data: Parts and Service Standards data
            custom_prompt: Optional custom prompt template with placeholders
        
        Returns:
            GeneratedEstimate object
        """
        # Convert damages to dict format
        damage_dicts = [d.model_dump() for d in damage_descriptions]
        
        # Convert chunks to dict format
        chunk_dicts = [
            {
                'score': c.score,
                'content': c.content,
                'vehicle_info': c.vehicle_info,
                'side': c.side,
                'approved_estimate': c.approved_estimate,
            }
            for c in retrieved_chunks
        ]
        
        # Use custom prompt if provided, otherwise use default
        if custom_prompt:
            from prompts.rag_prompts import (
                format_vehicle_info,
                format_damage_descriptions,
                format_retrieved_chunks,
                format_pss_data,
            )
            # Format the custom prompt with the same well-formatted context as default
            prompt = custom_prompt.format(
                vehicle_info=format_vehicle_info(vehicle_info.model_dump() if vehicle_info else None),
                damage_descriptions=format_damage_descriptions(damage_dicts),
                human_description=human_description or "Not provided",
                retrieved_chunks=format_retrieved_chunks(chunk_dicts),
                pss_data=format_pss_data(pss_data),
            )
        else:
            prompt = get_estimate_generation_prompt(
                vehicle_info=vehicle_info.model_dump() if vehicle_info else None,
                damage_descriptions=damage_dicts,
                human_description=human_description,
                retrieved_chunks=chunk_dicts,
                pss_data=pss_data,
            )
        
        # Generate estimate
        message = HumanMessage(content=prompt)
        print(f"DEBUG: Sending prompt to LLM (length: {len(prompt)} chars)")
        
        try:
            result: EstimateOutput = self.estimate_model.invoke([message])
            print(f"DEBUG: LLM response estimate keys: {list(result.estimate.keys()) if result.estimate else 'empty'}")
            print(f"DEBUG: Full LLM response: {result.estimate}")
        except Exception as e:
            print(f"DEBUG: LLM invocation error: {e}")
            # Return empty estimate on error
            return GeneratedEstimate(estimate={})
        
        # Extract PSS parts for matching
        pss_parts_map = self._extract_pss_parts(pss_data)
        
        # Convert to EstimateOperation objects and build the estimate dict
        from models.rag_models import EstimateOperation
        
        estimate_dict = {}
        for category, operations in result.estimate.items():
            estimate_dict[category] = []
            for op in operations:
                # Only include LaborHours if Operation is "Repair"
                labor_hours = op.get('LaborHours') if op.get('Operation') == 'Repair' else None
                
                # Get PartId - use from LLM response if provided, otherwise match with PSS data
                part_id = str(op.get('PartId','') or '')
                part_description = op.get('Description', '')
                
                # If LLM didn't provide PartId, try to match with PSS data
                if not part_id:
                    part_id = self._match_part_with_pss(part_description, pss_parts_map)
                    
                    # If no match found, try to match with the category name
                    if not part_id:
                        part_id = self._match_part_with_pss(category, pss_parts_map)
                
                estimate_dict[category].append(
                    EstimateOperation(
                        Description=op.get('Description', 'Unknown'),
                        Operation=op.get('Operation', 'Unknown'),
                        LaborHours=labor_hours,
                        PartId=str(part_id or ''),
                    )
                )
        
        return GeneratedEstimate(estimate=estimate_dict)
    
    def run_rag_pipeline(
        self,
        request: RAGEstimateRequest,
    ) -> RAGEstimateResponse:
        """
        Run the complete RAG pipeline for estimate generation.
        
        Flow:
        1. Use provided damage descriptions directly
        2. Retrieve similar chunks from Qdrant using merged_damage_description
        3. Generate estimate using retrieved chunks + PSS data
        
        Args:
            request: RAGEstimateRequest with all input parameters
        
        Returns:
            RAGEstimateResponse with complete results
        """
        start_time = time.time()
        
        try:
            # Use provided PSS data directly
            pss_data = request.pss_data
            
            # Use provided damage descriptions directly
            all_damages: list[DamageDescription] = request.damage_descriptions or []
            
            print(f"DEBUG: Received {len(all_damages)} damage descriptions")
            print(f"DEBUG: PSS data loaded: {pss_data is not None}")
            
            # Build search query from merged_damage_description or individual descriptions
            search_query = request.merged_damage_description
            if not search_query and all_damages:
                # Create merged description from individual damages if not provided
                search_query = self._merge_damage_descriptions(all_damages, request.vehicle_info)
            
            print(f"DEBUG: Search query: {search_query}")
            
            if not search_query:
                return RAGEstimateResponse(
                    success=False,
                    error="No damage description provided for retrieval",
                    processing_time_seconds=time.time() - start_time,
                )
            
            # Step 1: Retrieve similar chunks from Qdrant
            retrieved_chunks = self.retrieve_similar_chunks(
                damage_description=search_query,
            )
            
            print(f"DEBUG: Retrieved {len(retrieved_chunks)} chunks from Qdrant")
            
            # Step 2: Generate estimate
            generated_estimate = self.generate_estimate(
                damage_descriptions=all_damages,
                retrieved_chunks=retrieved_chunks,
                vehicle_info=request.vehicle_info,
                human_description=request.merged_damage_description,
                pss_data=pss_data,
                custom_prompt=request.custom_estimate_prompt,
            )
            
            print(f"DEBUG: Generated estimate with {len(generated_estimate.estimate)} categories")
            
            return RAGEstimateResponse(
                success=True,
                images_analyzed=len(request.images) if request.images else 0,
                images_with_damage=len(request.images) if request.images else 0,
                damage_detections=[],
                generated_estimate=generated_estimate,
                vehicle_info=request.vehicle_info,
                human_damage_description=request.merged_damage_description,
                pss_data_used=pss_data is not None,
                processing_time_seconds=time.time() - start_time,
            )
            
        except Exception as e:
            return RAGEstimateResponse(
                success=False,
                error=str(e),
                processing_time_seconds=time.time() - start_time,
            )
