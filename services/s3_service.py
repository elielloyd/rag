"""Service for interacting with AWS S3 to read vehicle images."""

import boto3
from botocore.exceptions import ClientError
from typing import Optional
from urllib.parse import urlparse
import io

from config import settings


class S3Service:
    """Service for reading images from AWS S3 buckets."""
    
    def __init__(self):
        """Initialize the S3 client with AWS credentials."""
        self.client = boto3.client(
            's3',
            region_name=settings.aws_region,
        )
        self.default_bucket = settings.aws_s3_bucket
    
    def parse_s3_url(self, s3_url: str, use_default_bucket: bool = True) -> tuple[str, str]:
        """
        Parse an S3 URL into bucket name and key.
        
        Args:
            s3_url: S3 URL in format s3://bucket/key or https://bucket.s3.region.amazonaws.com/key
            use_default_bucket: If True, use the configured default bucket instead of the one in URL
        
        Returns:
            Tuple of (bucket_name, key)
        """
        if s3_url.startswith('s3://'):
            # Format: s3://bucket/key
            parsed = urlparse(s3_url)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
        elif 's3.amazonaws.com' in s3_url or 's3.' in s3_url:
            # Format: https://bucket.s3.region.amazonaws.com/key
            parsed = urlparse(s3_url)
            # Extract bucket from hostname
            hostname_parts = parsed.netloc.split('.')
            bucket = hostname_parts[0]
            key = parsed.path.lstrip('/')
        else:
            raise ValueError(f"Invalid S3 URL format: {s3_url}")
        
        # Use default bucket if configured and requested
        if use_default_bucket and self.default_bucket:
            bucket = self.default_bucket
        
        return bucket, key
    
    def get_image(self, s3_url: str) -> tuple[bytes, str]:
        """
        Download an image from S3.
        
        Args:
            s3_url: S3 URL of the image
        
        Returns:
            Tuple of (image_bytes, mime_type)
        """
        bucket, key = self.parse_s3_url(s3_url)
        
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            image_data = response['Body'].read()
            content_type = response.get('ContentType', 'image/jpeg')
            return image_data, content_type
        except ClientError as e:
            raise Exception(f"Failed to download image from S3: {e}")
    
    def list_images_in_folder(self, bucket: str, prefix: str) -> list[str]:
        """
        List all image files in an S3 folder.
        
        Args:
            bucket: S3 bucket name
            prefix: Folder prefix (path)
        
        Returns:
            List of S3 URLs for images
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
        images = []
        
        try:
            paginator = self.client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    ext = key.lower().split('.')[-1] if '.' in key else ''
                    if f'.{ext}' in image_extensions:
                        images.append(f"s3://{bucket}/{key}")
        except ClientError as e:
            raise Exception(f"Failed to list images from S3: {e}")
        
        return images
    
    def list_images_from_url(self, s3_folder_url: str) -> list[str]:
        """
        List all image files from an S3 folder URL.
        
        Args:
            s3_folder_url: S3 URL of the folder (e.g., s3://bucket/claims/id/images/)
        
        Returns:
            List of S3 URLs for images
        """
        bucket, prefix = self.parse_s3_url(s3_folder_url)
        return self.list_images_in_folder(bucket, prefix)
    
    def get_json(self, s3_url: str) -> dict:
        """
        Download and parse a JSON file from S3.
        
        Args:
            s3_url: S3 URL of the JSON file
        
        Returns:
            Parsed JSON as a dictionary
        """
        import json
        
        bucket, key = self.parse_s3_url(s3_url)
        
        try:
            response = self.client.get_object(Bucket=bucket, Key=key)
            json_data = response['Body'].read().decode('utf-8')
            return json.loads(json_data)
        except ClientError as e:
            raise Exception(f"Failed to download JSON from S3: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON from S3: {e}")
    
    def is_configured(self) -> bool:
        """Check if the S3 service is properly configured."""
        return bool(settings.aws_access_key_id and settings.aws_secret_access_key)
