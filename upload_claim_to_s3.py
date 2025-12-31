"""Script to upload claim images and PSS data to S3."""

import boto3
import os
import json
from botocore.exceptions import ClientError
from config import settings
from extractpss_new import extract_required_pss_data


def upload_file_to_s3(local_file_path: str, s3_key: str, bucket_name: str = None, content_type: str = None):
    """
    Upload a file to S3.
    
    Args:
        local_file_path: Path to the local file to upload
        s3_key: The S3 key (path) where the file will be stored
        bucket_name: S3 bucket name (defaults to settings.aws_s3_bucket)
        content_type: Content type for the file
    """
    bucket_name = bucket_name or settings.aws_s3_bucket
    
    s3 = boto3.client(
        's3',
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )
    
    try:
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        
        s3.upload_file(local_file_path, bucket_name, s3_key, ExtraArgs=extra_args if extra_args else None)
        print(f"✓ Uploaded: {local_file_path} -> s3://{bucket_name}/{s3_key}")
        return f"s3://{bucket_name}/{s3_key}"
        
    except ClientError as e:
        print(f"✗ Error uploading {local_file_path}: {e}")
        return None
    except FileNotFoundError:
        print(f"✗ File not found: {local_file_path}")
        return None


def upload_folder_images(local_folder: str, s3_prefix: str, bucket_name: str = None):
    """
    Upload all images from a local folder to S3.
    
    Args:
        local_folder: Path to the local folder containing images
        s3_prefix: S3 prefix (folder path) where images will be stored
        bucket_name: S3 bucket name
    
    Returns:
        List of uploaded S3 URLs
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}
    content_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    
    uploaded_urls = []
    
    for filename in os.listdir(local_folder):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            local_path = os.path.join(local_folder, filename)
            s3_key = f"{s3_prefix}/{filename}"
            content_type = content_types.get(ext, 'application/octet-stream')
            
            url = upload_file_to_s3(local_path, s3_key, bucket_name, content_type)
            if url:
                uploaded_urls.append(url)
    
    return uploaded_urls


def process_and_upload_pss(pss_file_path: str, s3_key: str, bucket_name: str = None):
    """
    Process PSS file through extractpss_new and upload to S3.
    
    Args:
        pss_file_path: Path to the raw PSS JSON file
        s3_key: S3 key for the optimized PSS file
        bucket_name: S3 bucket name
    
    Returns:
        S3 URL of the uploaded file
    """
    bucket_name = bucket_name or settings.aws_s3_bucket
    
    print(f"\nProcessing PSS file: {pss_file_path}")
    
    # Load and process PSS data
    with open(pss_file_path, 'r') as f:
        full_pss_data = json.load(f)
    
    optimized_pss = extract_required_pss_data(full_pss_data)
    
    # Save temporarily
    temp_file = "temp_optimized_pss.json"
    with open(temp_file, 'w') as f:
        json.dump(optimized_pss, f, indent=2)
    
    print(f"✓ PSS optimized: {len(json.dumps(optimized_pss))} bytes")
    
    # Upload to S3
    url = upload_file_to_s3(temp_file, s3_key, bucket_name, 'application/json')
    
    # Clean up temp file
    os.remove(temp_file)
    
    return url


if __name__ == "__main__":
    # Configuration
    claim_folder = "claim_images_6"
    claim_id = "claim-6"
    pss_file = "claim_images_6/4_pss.json"
    
    print("=" * 60)
    print("Uploading Claim Images to S3")
    print("=" * 60)
    
    # Upload images
    s3_images_prefix = f"claims/{claim_id}/images"
    uploaded_images = upload_folder_images(claim_folder, s3_images_prefix)
    
    print(f"\n✓ Uploaded {len(uploaded_images)} images")
    print(f"Images available at: s3://{settings.aws_s3_bucket}/{s3_images_prefix}/")
    
    print("\n" + "=" * 60)
    print("Processing and Uploading PSS Data")
    print("=" * 60)
    
    # Process and upload PSS
    pss_s3_key = f"pss/{claim_id}_optimized_pss.json"
    pss_url = process_and_upload_pss(pss_file, pss_s3_key)
    
    if pss_url:
        print(f"\n✓ PSS data available at: {pss_url}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Images URL: s3://{settings.aws_s3_bucket}/{s3_images_prefix}/")
    print(f"PSS URL: {pss_url}")
    print("\nUse these URLs in your RAG API request:")
    print(f'  "bucket_url": "s3://{settings.aws_s3_bucket}/{s3_images_prefix}/"')
    print(f'  "pss_url": "{pss_url}"')
