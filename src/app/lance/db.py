"""
LanceDB connection management for S3 Vectors.

This module handles LanceDB connections to MinIO/S3 buckets.
"""

import lancedb
from app.util import config


def connect_bucket(bucket_name: str) -> lancedb.DBConnection:
    """
    Connect to LanceDB using S3/MinIO storage.
    
    Args:
        bucket_name: The bucket name (without vb- prefix)
        
    Returns:
        LanceDB connection
    """
    # Build S3 URI for the bucket
    bucket_uri = f"s3://{config.S3_BUCKET_PREFIX}{bucket_name}/"
    
    # Configure S3 storage options
    storage_options = {
        "aws_access_key_id": config.LANCE_ACCESS_KEY,
        "aws_secret_access_key": config.LANCE_SECRET_KEY,
        "aws_endpoint_url": config.LANCE_S3_ENDPOINT,
        "aws_region": config.LANCE_S3_REGION,
        "aws_allow_http": str(config.LANCE_ALLOW_HTTP).lower()
    }
    
    return lancedb.connect(bucket_uri, storage_options=storage_options)


def table_path(index_name: str) -> str:
    """
    Get the table name for an index.
    
    Args:
        index_name: The index name
        
    Returns:
        Table name for Lance (alphanumeric, underscores, hyphens, periods only)
    """
    # Create a safe table name by replacing invalid characters
    safe_name = index_name.replace("/", "_").replace(":", "_").replace(" ", "_").replace("-", "_")
    # Ensure it's a valid Lance table name (alphanumeric, underscores, hyphens, periods only)
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', safe_name)
    # Remove consecutive underscores
    safe_name = re.sub(r'_+', '_', safe_name)
    # Remove leading/trailing underscores
    safe_name = safe_name.strip('_')
    # Just return the safe table name, not a full path
    return safe_name
