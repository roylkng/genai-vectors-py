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
    Get the table path for an index.
    
    Args:
        index_name: The index name
        
    Returns:
        Safe table name for Lance (alphanumeric, underscores, hyphens, periods only)
    """
    # Create a safe table name by replacing invalid characters
    safe_name = index_name.replace("/", "_").replace(":", "_").replace(" ", "_")
    return f"index_{safe_name}_table"
