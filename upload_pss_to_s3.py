"""Script to upload PSS (Parts and Service Standards) JSON file to S3."""

import boto3
from botocore.exceptions import ClientError
from config import settings


def upload_file_to_s3(local_file_path: str, s3_key: str, bucket_name: str = None):
    """
    Upload a file to S3.
    
    Args:
        local_file_path: Path to the local file to upload
        s3_key: The S3 key (path) where the file will be stored
        bucket_name: S3 bucket name (defaults to settings.aws_s3_bucket)
    """
    bucket_name = bucket_name or settings.aws_s3_bucket
    
    s3 = boto3.client(
        's3',
        region_name=settings.aws_region,
    )
    
    try:
        print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}...")
        
        s3.upload_file(
            local_file_path,
            bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'application/json'}
        )
        
        print(f"✓ Successfully uploaded to s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
        
    except ClientError as e:
        print(f"✗ Error uploading file: {e}")
        return None
    except FileNotFoundError:
        print(f"✗ Error: File not found: {local_file_path}")
        return None


if __name__ == "__main__":
    # Upload optimized_pss.json as PSS data
    local_file = "optimized_pss.json"
    s3_key = "pss/optimized_pss.json"
    
    result = upload_file_to_s3(local_file, s3_key)
    
    if result:
        print(f"\nFile available at: {result}")
        print(f"\nYou can use this URL in the RAG API as pss_url parameter.")
