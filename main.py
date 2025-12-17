"""FastAPI application for image preprocessing using Gemini 3 API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes import qdrant_router, health_router, vehicle_damage_router

# Create FastAPI application
app = FastAPI(
    title="TrueClaim Preprocessing API",
    description="Image preprocessing pipeline using Gemini 3 API with high resolution settings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router)
app.include_router(qdrant_router)
app.include_router(vehicle_damage_router)


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Ensure data directories exist
    settings.ensure_directories()
    
    # Log configuration status
    if settings.gemini_api_key:
        print("✓ Gemini API key configured")
    else:
        print("✗ Warning: GEMINI_API_KEY not set")
    
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        print("✓ AWS S3 credentials configured")
    else:
        print("✗ Warning: AWS S3 credentials not set")
    
    print(f"✓ Data directory: {settings.data_dir}")
    print(f"✓ Images directory: {settings.images_dir}")
    print(f"✓ Outputs directory: {settings.outputs_dir}")
    print(f"✓ Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
