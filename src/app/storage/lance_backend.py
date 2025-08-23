"""
Enhanced S3 storage backend with Lance integration and new object layout.

This module extends the existing S3Storage class to support the new Lance-based
object layout while maintaining backward compatibility during migration.
"""

from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from .s3_backend import S3Storage as BaseS3Storage
from app.util import config


class LanceS3Storage(BaseS3Storage):
    """
    Enhanced S3 storage backend with Lance integration.
    
    Supports the new object layout:
    vb-<bucketName>/
    ├─ _meta/
    │  ├─ bucket.json
    │  └─ policy.json  
    └─ indexes/
       └─ <indexName>/
          ├─ table/           # Lance dataset
          └─ _index_config.json
    """
    
    def __init__(self):
        super().__init__()
    
    # Bucket metadata operations
    def put_bucket_metadata(self, bucket_name: str, metadata: Dict[str, Any]) -> None:
        """Store bucket metadata in _meta/bucket.json"""
        key = f"{config.META_DIR}/{config.BUCKET_CONFIG_KEY}"
        self.put_json(bucket_name, key, metadata)
    
    def get_bucket_metadata(self, bucket_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve bucket metadata from _meta/bucket.json"""
        key = f"{config.META_DIR}/{config.BUCKET_CONFIG_KEY}"
        return self.get_json(bucket_name, key)
    
    def delete_bucket_metadata(self, bucket_name: str) -> None:
        """Delete bucket metadata"""
        key = f"{config.META_DIR}/{config.BUCKET_CONFIG_KEY}"
        self.delete_object(bucket_name, key)
    
    # Bucket policy operations
    def put_bucket_policy(self, bucket_name: str, policy: Dict[str, Any]) -> None:
        """Store bucket policy in _meta/policy.json"""
        key = f"{config.META_DIR}/{config.POLICY_CONFIG_KEY}"
        self.put_json(bucket_name, key, policy)
    
    def get_bucket_policy(self, bucket_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve bucket policy from _meta/policy.json"""
        key = f"{config.META_DIR}/{config.POLICY_CONFIG_KEY}"
        return self.get_json(bucket_name, key)
    
    def delete_bucket_policy(self, bucket_name: str) -> None:
        """Delete bucket policy"""
        key = f"{config.META_DIR}/{config.POLICY_CONFIG_KEY}"
        self.delete_object(bucket_name, key)
    
    # Index configuration operations
    def put_index_config(self, bucket_name: str, index_name: str, config_data: Dict[str, Any]) -> None:
        """Store index configuration"""
        key = f"{config.INDEX_DIR}/{index_name}/{config.INDEX_CONFIG_KEY}"
        self.put_json(bucket_name, key, config_data)
    
    def get_index_config(self, bucket_name: str, index_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve index configuration"""
        key = f"{config.INDEX_DIR}/{index_name}/{config.INDEX_CONFIG_KEY}"
        return self.get_json(bucket_name, key)
    
    def delete_index_config(self, bucket_name: str, index_name: str) -> None:
        """Delete index configuration"""
        key = f"{config.INDEX_DIR}/{index_name}/{config.INDEX_CONFIG_KEY}"
        self.delete_object(bucket_name, key)
    
    # Enhanced bucket operations
    def create_vector_bucket(self, bucket_name: str, encryption_config: Optional[Dict] = None) -> None:
        """
        Create a vector bucket with metadata.
        
        Args:
            bucket_name: Name of the bucket to create
            encryption_config: Optional encryption configuration
        """
        # Create the underlying bucket
        self.ensure_bucket(bucket_name)
        
        # Store bucket metadata
        metadata = {
            "name": bucket_name,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "encryptionConfiguration": encryption_config or {}
        }
        self.put_bucket_metadata(bucket_name, metadata)
    
    def delete_vector_bucket(self, bucket_name: str, force: bool = False) -> None:
        """
        Delete a vector bucket and optionally the underlying S3 bucket.
        
        Args:
            bucket_name: Name of the bucket to delete
            force: Whether to delete the underlying S3 bucket
        """
        # Delete all vector bucket content
        self.delete_prefix(bucket_name, f"{config.INDEX_DIR}/")
        self.delete_prefix(bucket_name, f"{config.META_DIR}/")
        
        # Legacy cleanup
        self.delete_prefix(bucket_name, f"{config.LEGACY_STAGED_DIR}/")
        
        # Optionally delete the S3 bucket itself
        if force or config.DELETE_BUCKET_ON_DELETE:
            try:
                bucket_with_prefix = self.bucket_name(bucket_name)
                self.client.delete_bucket(Bucket=bucket_with_prefix)
            except Exception:
                pass  # Ignore errors when deleting bucket
    
    def list_vector_buckets_with_metadata(self) -> List[Dict[str, Any]]:
        """
        List vector buckets with their metadata.
        
        Returns:
            List of bucket information dictionaries
        """
        bucket_names = self.list_vector_buckets()
        buckets = []
        
        for name in bucket_names:
            metadata = self.get_bucket_metadata(name)
            if metadata:
                bucket_info = {
                    "vectorBucketName": name,
                    "vectorBucketArn": self._generate_bucket_arn(name),
                    "creationTime": metadata.get("createdAt", self._get_iso_timestamp()),
                    "encryptionConfiguration": metadata.get("encryptionConfiguration", {})
                }
            else:
                # Fallback for buckets without metadata (legacy or external)
                bucket_info = {
                    "vectorBucketName": name,
                    "vectorBucketArn": self._generate_bucket_arn(name),
                    "creationTime": self._get_iso_timestamp(),
                    "encryptionConfiguration": {}
                }
            buckets.append(bucket_info)
        
        return sorted(buckets, key=lambda x: x["vectorBucketName"])
    
    # Index discovery operations
    def list_indexes_with_metadata(self, bucket_name: str) -> List[Dict[str, Any]]:
        """
        List indexes in a bucket with their metadata.
        
        Args:
            bucket_name: Name of the bucket
            
        Returns:
            List of index information dictionaries
        """
        index_names = set()
        prefix = f"{config.INDEX_DIR}/"
        
        # Discover indexes from both Lance tables and legacy configs
        for key in self.list_prefix(bucket_name, prefix):
            parts = key.split("/")
            if len(parts) >= 3:  # indexes/<name>/...
                index_names.add(parts[1])
        
        indexes = []
        for name in sorted(index_names):
            index_config = self.get_index_config(bucket_name, name)
            if index_config:
                index_info = {
                    "indexName": name,
                    "indexArn": self._generate_index_arn(bucket_name, name),
                    "vectorBucketName": bucket_name,
                    "creationTime": index_config.get("createdAt", self._get_iso_timestamp()),
                    "dimension": index_config.get("dimension"),
                    "dataType": index_config.get("dataType", "float32"),
                    "distanceMetric": index_config.get("distanceMetric", "cosine"),
                    "metadataConfiguration": index_config.get("metadataConfiguration", {})
                }
            else:
                # Fallback for legacy indexes
                legacy_config = self.get_json(bucket_name, f"{config.INDEX_DIR}/{name}/config.json")
                if legacy_config:
                    index_info = {
                        "indexName": name,
                        "indexArn": self._generate_index_arn(bucket_name, name),
                        "vectorBucketName": bucket_name,
                        "creationTime": self._get_iso_timestamp(),
                        "dimension": legacy_config.get("dimension"),
                        "dataType": legacy_config.get("dataType", "float32"),
                        "distanceMetric": legacy_config.get("distanceMetric", "cosine"),
                        "metadataConfiguration": legacy_config.get("metadataConfiguration", {})
                    }
                else:
                    continue  # Skip incomplete indexes
            
            indexes.append(index_info)
        
        return indexes
    
    # Utility methods
    def _generate_bucket_arn(self, bucket_name: str) -> str:
        """Generate ARN for a vector bucket"""
        return f"arn:aws:s3vectors:{config.S3_REGION}:123456789012:bucket/{bucket_name}"
    
    def _generate_index_arn(self, bucket_name: str, index_name: str) -> str:
        """Generate ARN for a vector index"""
        return f"arn:aws:s3vectors:{config.S3_REGION}:123456789012:index/{bucket_name}/{index_name}"
    
    def _get_iso_timestamp(self) -> str:
        """Get current timestamp in ISO 8601 format"""
        return datetime.now(timezone.utc).isoformat()
    
    # Migration helpers
    def has_legacy_data(self, bucket_name: str, index_name: str) -> bool:
        """
        Check if an index has legacy (pre-Lance) data.
        
        Args:
            bucket_name: Vector bucket name
            index_name: Vector index name
            
        Returns:
            True if legacy data exists
        """
        # Check for legacy manifest
        legacy_manifest = self.get_json(bucket_name, f"{config.INDEX_DIR}/{index_name}/{config.LEGACY_MANIFEST_KEY}")
        if legacy_manifest:
            return True
        
        # Check for legacy idmap
        try:
            self.download_bytes(bucket_name, f"{config.INDEX_DIR}/{index_name}/{config.LEGACY_IDMAP_KEY}")
            return True
        except Exception:
            pass
        
        # Check for staged data
        staged_keys = list(self.list_prefix(bucket_name, f"{config.LEGACY_STAGED_DIR}/{index_name}/"))
        return len(staged_keys) > 0
    
    def get_lance_table_path(self, bucket_name: str, index_name: str) -> str:
        """
        Get the Lance table path for an index.
        
        Args:
            bucket_name: Vector bucket name
            index_name: Vector index name
            
        Returns:
            Path to Lance table within the bucket
        """
        return f"{config.INDEX_DIR}/{index_name}/{config.TABLE_DIR}"
    
    def table_exists(self, bucket_name: str, index_name: str) -> bool:
        """
        Check if a Lance table exists for the given index.
        
        Args:
            bucket_name: Vector bucket name
            index_name: Vector index name
            
        Returns:
            True if Lance table exists
        """
        table_prefix = self.get_lance_table_path(bucket_name, index_name)
        # Check for any Lance table files
        for _ in self.list_prefix(bucket_name, table_prefix + "/"):
            return True
        return False
