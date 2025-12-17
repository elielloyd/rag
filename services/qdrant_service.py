"""Service for interacting with Qdrant vector database."""

import json
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime

import numpy as np
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from config import settings
from models.vehicle_damage import ChunkOutput


class QdrantService:
    """Service for storing and retrieving image descriptions from Qdrant."""
    
    # Gemini embedding model outputs 3072 dimensions by default
    # Using 768 for efficiency while maintaining quality
    VECTOR_SIZE = 768
    EMBEDDING_MODEL = "gemini-embedding-001"
    
    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the Qdrant client and Gemini embedding client.
        
        Args:
            collection_name: Name of the collection to use. Defaults to settings value.
        """
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        self.collection_name = collection_name or settings.qdrant_collection_name
        
        # Initialize Gemini client for embeddings
        self.genai_client = genai.Client(api_key=settings.gemini_api_key)
    
    def _ensure_collection_exists(self):
        """Ensure the collection exists, creating it if necessary."""
        try:
            self.client.get_collection(self.collection_name)
        except (UnexpectedResponse, Exception):
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
    
    def _generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding using Gemini embedding model.
        
        Args:
            text: Text to embed.
        
        Returns:
            List of floats representing the embedding.
        """
        result = self.genai_client.models.embed_content(
            model=self.EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=self.VECTOR_SIZE
            )
        )
        
        # Get the embedding values and normalize for better similarity
        embedding_values = result.embeddings[0].values
        
        # Normalize the embedding
        embedding_np = np.array(embedding_values)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
        
        return embedding_np.tolist()
    
    def _generate_query_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for search queries using Gemini embedding model.
        Uses RETRIEVAL_QUERY task type for better search results.
        
        Args:
            text: Query text to embed.
        
        Returns:
            List of floats representing the embedding.
        """
        result = self.genai_client.models.embed_content(
            model=self.EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.VECTOR_SIZE
            )
        )
        
        # Get the embedding values and normalize
        embedding_values = result.embeddings[0].values
        embedding_np = np.array(embedding_values)
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
        
        return embedding_np.tolist()
    
    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Search for similar images based on a text query.
        
        Args:
            query: Text query to search for.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score threshold.
        
        Returns:
            List of matching results with scores.
        """
        self._ensure_collection_exists()
        
        # Generate embedding for query using RETRIEVAL_QUERY task type
        query_embedding = self._generate_query_embedding(query)
        
        # Search using query_points (newer Qdrant API)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
        )
        
        return [
            {
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]
    
    def get_collection_info(self) -> dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary with collection information.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "status": info.status.value if info.status else "unknown",
                "vector_size": self.VECTOR_SIZE,
            }
        except Exception as e:
            return {
                "name": self.collection_name,
                "error": str(e),
            }
    
    def is_connected(self) -> bool:
        """Check if Qdrant is connected and accessible."""
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
    
    def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            self.client.delete_collection(self.collection_name)
            return True
        except Exception:
            return False

    def upload_damage_chunk(self, chunk: ChunkOutput) -> str:
        """
        Upload a vehicle damage chunk to Qdrant.
        
        The merged_damage_description is used as the content for embedding.
        All other vehicle and estimate info is stored as metadata.
        
        Args:
            chunk: The ChunkOutput to upload.
        
        Returns:
            The ID of the uploaded point.
        """
        self._ensure_collection_exists()
        
        # Generate embedding from merged_damage_description
        embedding = self._generate_embedding(chunk.merged_damage_description)
        
        # Create payload with vehicle info and estimate as metadata
        payload = {
            "content": chunk.merged_damage_description,
            "vehicle_info": {
                "vin": chunk.vehicle_info.vin,
                "make": chunk.vehicle_info.make,
                "model": chunk.vehicle_info.model,
                "year": chunk.vehicle_info.year,
                "body_type": chunk.vehicle_info.body_type,
            },
            "side": chunk.side,
            "images": chunk.images,
            "damage_descriptions": [
                {
                    "location": d.location,
                    "part": d.part,
                    "severity": d.severity,
                    "type": d.type,
                    "start_position": d.start_position,
                    "end_position": d.end_position,
                    "description": d.description,
                }
                for d in chunk.damage_descriptions
            ],
            "approved_estimate": {
                category: [
                    {
                        "Description": op.Description,
                        "Operation": op.Operation,
                        **({"LabourHours": op.LabourHours} if op.LabourHours is not None else {}),
                    }
                    for op in operations
                ]
                for category, operations in chunk.approved_estimate.items()
            },
            "uploaded_at": datetime.now().isoformat(),
        }
        
        # Generate a numeric ID from VIN + side + timestamp
        unique_id = f"{chunk.vehicle_info.vin}_{chunk.side}_{datetime.now().isoformat()}"
        point_id = int(hashlib.md5(unique_id.encode()).hexdigest()[:16], 16)
        
        # Upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
        
        return unique_id
