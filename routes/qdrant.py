"""Routes for Qdrant vector database operations."""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from services import QdrantService
from middleware.auth import verify_api_key

router = APIRouter(
    prefix="/qdrant",
    tags=["Qdrant"],
    dependencies=[Depends(verify_api_key)]
)


def get_qdrant_service(collection_name: Optional[str] = None) -> QdrantService:
    """Get or create the Qdrant service instance."""
    return QdrantService(collection_name=collection_name)


@router.get("/search")
async def search_qdrant(
    query: str,
    limit: int = 10,
    collection_name: Optional[str] = None,
):
    """
    Search for similar images based on a text query.
    
    Args:
        query: Text query to search for.
        limit: Maximum number of results to return.
        collection_name: Optional collection name to search in.
    """
    service = get_qdrant_service(collection_name)
    
    if not service.is_connected():
        raise HTTPException(status_code=503, detail="Qdrant is not connected")
    
    try:
        results = service.search(query=query, limit=limit)
        return {
            "query": query,
            "results": results,
            "count": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collection/info")
async def get_collection_info(collection_name: Optional[str] = None):
    """
    Get information about the Qdrant collection.
    
    Args:
        collection_name: Optional collection name. Uses default if not specified.
    """
    service = get_qdrant_service(collection_name)
    
    if not service.is_connected():
        raise HTTPException(status_code=503, detail="Qdrant is not connected")
    
    return service.get_collection_info()


@router.delete("/collection")
async def delete_collection(collection_name: Optional[str] = None):
    """
    Delete the Qdrant collection.
    
    Args:
        collection_name: Optional collection name. Uses default if not specified.
    """
    service = get_qdrant_service(collection_name)
    
    if not service.is_connected():
        raise HTTPException(status_code=503, detail="Qdrant is not connected")
    
    success = service.delete_collection()
    
    if success:
        return {"message": f"Collection '{service.collection_name}' deleted successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete collection")
