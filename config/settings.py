from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    # Gemini API Configuration
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-pro-preview"
    
    # AWS S3 Configuration
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    aws_s3_bucket: str = "ehsan-poc-estimate-true-claim"
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "image_descriptions"
    
    # Application Configuration
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = False
    
    # Data paths
    data_dir: Path = Path("./data")
    images_dir: Path = Path("./data/images")
    outputs_dir: Path = Path("./data/outputs")

    # Langsmith tracing
    langsmith_tracing: Optional[bool] = True
    langsmith_endpoint: Optional[str] = "https://api.smith.langchain.com"
    langsmith_api_key: Optional[str] = ""
    langsmith_project: Optional[str] = ""
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
