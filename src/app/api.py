"""
S3 Vectors API implementation using Lance vector database.
Direct Lance integration without feature flags or legacy support.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
import json
from datetime import datetime

from .models import (
    CreateVectorBucketRequest, CreateVectorBucketResponse,
    ListVectorBucketsResponse, GetVectorBucketResponse,
    CreateIndexRequest, CreateIndexResponse, 
    ListIndexesResponse, GetIndexResponse,
    PutVectorsRequest, PutVectorsResponse,
    GetVectorsRequest, GetVectorsResponse,
    QueryVectorsRequest, QueryVectorsResponse,
    ListVectorsRequest, ListVectorsResponse,
    DeleteVectorsRequest, DeleteVectorsResponse
)

from .lance.db import connect_bucket, table_path
from .lance import index_ops
from .storage.s3_backend import S3Storage
from .util import config

router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "implementation": "lance"}

# ===============================
# Bucket Operations  
# ===============================

@router.put("/buckets/{bucket_name}")
async def create_vector_bucket(bucket_name: str, request: CreateVectorBucketRequest):
    """Create a new vector bucket"""
    try:
        s3 = S3Storage()
        
        # Ensure underlying S3 bucket exists with vb- prefix
        s3.ensure_bucket(bucket_name)
        
        # Store bucket metadata
        bucket_config = {
            "name": bucket_name,
            "created": datetime.utcnow().isoformat(),
            "engine": "lance",
            "version": "1.0"
        }
        
        s3.put_json(
            bucket_name,
            f"{config.META_DIR}/bucket.json",
            bucket_config
        )
        
        return CreateVectorBucketResponse(
            bucketName=bucket_name,
            bucketArn=f"arn:aws:s3-vectors:::bucket/{bucket_name}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create bucket: {str(e)}"
        )

@router.get("/buckets")
async def list_vector_buckets() -> ListVectorBucketsResponse:
    """List all vector buckets"""
    try:
        s3 = S3Storage()
        
        # List S3 buckets with vb- prefix
        all_buckets = s3.list_buckets()
        vector_buckets = []
        
        for bucket in all_buckets:
            if bucket.startswith(config.S3_BUCKET_PREFIX):
                bucket_name = bucket[len(config.S3_BUCKET_PREFIX):]
                
                # Try to get bucket metadata
                try:
                    bucket_data = s3.get_json(bucket_name, f"{config.META_DIR}/bucket.json")
                    bucket_info = bucket_data if bucket_data else {}
                    created = bucket_info.get("created", datetime.utcnow().isoformat())
                except:
                    created = datetime.utcnow().isoformat()
                
                vector_buckets.append({
                    "name": bucket_name,
                    "creationDate": created,
                    "arn": f"arn:aws:s3-vectors:::bucket/{bucket_name}"
                })
        
        return ListVectorBucketsResponse(buckets=vector_buckets)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list buckets: {str(e)}"
        )

@router.get("/buckets/{bucket_name}")
async def get_vector_bucket(bucket_name: str) -> GetVectorBucketResponse:
    """Get vector bucket information"""
    try:
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        # Check if bucket exists
        if not s3.bucket_exists(s3_bucket):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found"
            )
        
        # Get bucket metadata
        try:
            bucket_data = s3.get_json(bucket_name, f"{config.META_DIR}/bucket.json")
            bucket_info = bucket_data if bucket_data else {}
        except:
            # Fallback for buckets without metadata
            bucket_info = {
                "name": bucket_name,
                "created": datetime.utcnow().isoformat(),
                "engine": "lance"
            }
        
        return GetVectorBucketResponse(
            name=bucket_name,
            arn=f"arn:aws:s3-vectors:::bucket/{bucket_name}",
            creationDate=bucket_info.get("created", datetime.utcnow().isoformat())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get bucket: {str(e)}"
        )

@router.delete("/buckets/{bucket_name}")
async def delete_vector_bucket(bucket_name: str):
    """Delete a vector bucket"""
    try:
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        if not s3.bucket_exists(s3_bucket):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found"
            )
        
        # Delete all vector indexes and metadata
        s3.delete_prefix(s3_bucket, config.INDEX_DIR)
        s3.delete_prefix(s3_bucket, config.META_DIR)
        
        # Note: We don't delete the underlying S3 bucket
        # in case it has other non-vector data
        
        return {"message": f"Vector bucket {bucket_name} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete bucket: {str(e)}"
        )

# ===============================
# Index Operations
# ===============================

@router.post("/buckets/{bucket_name}/indexes/{index_name}")
async def create_index(
    bucket_name: str, 
    index_name: str, 
    request: CreateIndexRequest
) -> CreateIndexResponse:
    """Create a vector index"""
    try:
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        if not s3.bucket_exists(s3_bucket):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found"
            )
        
        # Create Lance table
        db = connect_bucket(bucket_name)
        # Initialize indexer
        table_uri = table_path(index_name)
        
        # Extract non-filterable keys from request
        nonfilterable_keys = []
        if request.metadataConfiguration and request.metadataConfiguration.nonFilterableMetadataKeys:
            nonfilterable_keys = request.metadataConfiguration.nonFilterableMetadataKeys
        await index_ops.create_table(db, table_uri, request.dimension, nonfilterable_keys=nonfilterable_keys)
        
        # Store index metadata
        index_config = {
            "name": index_name,
            "dimension": request.dimension,
            "created": datetime.utcnow().isoformat(),
            "engine": "lance",
            "indexType": config.LANCE_INDEX_TYPE,
            "metricType": "cosine",
            "nonFilterableMetadataKeys": nonfilterable_keys
        }
        
        s3.put_json(
            bucket_name,
            f"{config.INDEX_DIR}/{index_name}/_index_config.json",
            index_config
        )
        
        return CreateIndexResponse(
            name=index_name,
            dimension=request.dimension,
            arn=f"arn:aws:s3-vectors:::bucket/{bucket_name}/index/{index_name}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create index: {str(e)}"
        )

@router.get("/buckets/{bucket_name}/indexes")
async def list_indexes(bucket_name: str) -> ListIndexesResponse:
    """List all indexes in a bucket"""
    try:
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        if not s3.bucket_exists(s3_bucket):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found"
            )
        
        # List index directories
        index_objects = s3.list_objects_with_prefix(s3_bucket, f"{config.INDEX_DIR}/")
        index_names = set()
        
        for obj_key in index_objects:
            # Extract index name from path like "indexes/my-index/..."
            parts = obj_key.split('/')
            if len(parts) >= 2 and parts[0] == config.INDEX_DIR.rstrip('/'):
                index_names.add(parts[1])
        
        indexes = []
        for index_name in sorted(index_names):
            # Get index metadata
            try:
                config_data = s3.get_json(
                    bucket_name, 
                    f"{config.INDEX_DIR}/{index_name}/_index_config.json"
                )
                index_info = config_data if config_data else {}
            except:
                # Fallback for indexes without metadata
                index_info = {
                    "name": index_name,
                    "dimension": 128,  # Default
                    "created": datetime.utcnow().isoformat()
                }
            
            indexes.append({
                "name": index_name,
                "dimension": index_info.get("dimension", 128),
                "arn": f"arn:aws:s3-vectors:::bucket/{bucket_name}/index/{index_name}",
                "creationDate": index_info.get("created", datetime.utcnow().isoformat())
            })
        
        return ListIndexesResponse(indexes=indexes)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list indexes: {str(e)}"
        )

@router.get("/buckets/{bucket_name}/indexes/{index_name}")
async def get_index(bucket_name: str, index_name: str) -> GetIndexResponse:
    """Get index information"""
    try:
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        if not s3.bucket_exists(s3_bucket):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found"
            )
        
        # Check if index exists
        try:
            config_data = s3.get_json(
                bucket_name,
                f"{config.INDEX_DIR}/{index_name}/_index_config.json"
            )
            if not config_data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Index {index_name} not found"
                )
            index_info = config_data
        except:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index {index_name} not found"
            )
        
        return GetIndexResponse(
            name=index_name,
            dimension=index_info.get("dimension", 128),
            arn=f"arn:aws:s3-vectors:::bucket/{bucket_name}/index/{index_name}",
            creationDate=index_info.get("created", datetime.utcnow().isoformat())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get index: {str(e)}"
        )

@router.delete("/buckets/{bucket_name}/indexes/{index_name}")
async def delete_index(bucket_name: str, index_name: str):
    """Delete an index"""
    try:
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        if not s3.bucket_exists(s3_bucket):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found"
            )
        
        # Delete index data and metadata
        s3.delete_prefix(s3_bucket, f"{config.INDEX_DIR}/{index_name}/")
        
        return {"message": f"Index {index_name} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete index: {str(e)}"
        )

# ===============================
# Vector Operations
# ===============================

@router.post("/buckets/{bucket_name}/indexes/{index_name}/vectors")
async def put_vectors(
    bucket_name: str,
    index_name: str, 
    request: PutVectorsRequest
) -> PutVectorsResponse:
    """Add or update vectors in an index"""
    try:
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        if not s3.bucket_exists(s3_bucket):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found"
            )
        
        # Connect to Lance
        db = connect_bucket(bucket_name)
        table_uri = table_path(index_name)
        
        # Convert vectors to dictionary format for Lance
        vectors_data = []
        for v in request.vectors:
            vector_dict = {
                "key": v.key,
                "vector": v.data.float32,
                "metadata": v.metadata or {}
            }
            vectors_data.append(vector_dict)
        
        # Upsert vectors
        await index_ops.upsert_vectors(db, table_uri, vectors_data)
        
        # Rebuild index if configured (smart indexing)
        await index_ops.rebuild_index(db, table_uri, config.LANCE_INDEX_TYPE)
        
        return PutVectorsResponse(
            vectorCount=len(request.vectors),
            vectorIds=[v.key for v in request.vectors]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to put vectors: {str(e)}"
        )

@router.post("/buckets/{bucket_name}/indexes/{index_name}/query")
async def query_vectors(
    bucket_name: str,
    index_name: str,
    request: QueryVectorsRequest
) -> QueryVectorsResponse:
    """Query vectors using similarity search with enhanced filtering"""
    try:
        # Import AWS-compatible error handling
        from .errors import (
            ResourceNotFoundException, ValidationException,
            validate_bucket_name, validate_index_name, validate_top_k
        )
        
        # Validate inputs
        validate_bucket_name(bucket_name)
        validate_index_name(index_name)
        validate_top_k(request.topK)
        
        if not request.queryVector or not request.queryVector.float32:
            raise ValidationException("Query vector is required")
        
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        if not s3.bucket_exists(s3_bucket):
            raise ResourceNotFoundException("VectorBucket", bucket_name)
        
        # Check if index exists
        index_config = s3.get_json(
            bucket_name,
            f"{config.INDEX_DIR}/{index_name}/_index_config.json"
        )
        if not index_config:
            raise ResourceNotFoundException("Index", f"{bucket_name}/{index_name}")
        
        # Connect to Lance
        db = connect_bucket(bucket_name)
        table_uri = table_path(index_name)
        
        # Search vectors with enhanced filtering
        results = await index_ops.search_vectors(
            db, table_uri, 
            query_vector=request.queryVector.float32,
            top_k=request.topK,
            filter_condition=request.filter.root if request.filter else None,
            return_data=request.returnData or False,
            return_metadata=request.returnMetadata or False,
            return_distance=request.returnDistance or True
        )
        
        return QueryVectorsResponse(vectors=results)
        
    except (ResourceNotFoundException, ValidationException):
        raise
    except Exception as e:
        from .errors import InternalServiceException
        raise InternalServiceException(f"Failed to query vectors: {str(e)}")

@router.post("/buckets/{bucket_name}/indexes/{index_name}/vectors:get")
async def get_vectors(
    bucket_name: str,
    index_name: str,
    request: GetVectorsRequest
) -> GetVectorsResponse:
    """Get vectors by keys (batch lookup)"""
    try:
        # Import AWS-compatible error handling
        from .errors import (
            ResourceNotFoundException, ValidationException, 
            validate_vector_keys, validate_bucket_name, validate_index_name
        )
        
        # Validate inputs
        validate_bucket_name(bucket_name)
        validate_index_name(index_name)
        validate_vector_keys(request.keys)
        
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        if not s3.bucket_exists(s3_bucket):
            raise ResourceNotFoundException("VectorBucket", bucket_name)
        
        # Check if index exists
        index_config = s3.get_json(
            bucket_name,
            f"{config.INDEX_DIR}/{index_name}/_index_config.json"
        )
        if not index_config:
            raise ResourceNotFoundException("Index", f"{bucket_name}/{index_name}")
        
        # Connect to Lance
        db = connect_bucket(bucket_name)
        table_uri = table_path(index_name)
        
        # Get vectors
        vectors = await index_ops.get_vectors(
            db, table_uri, request.keys,
            return_data=request.returnData,
            return_metadata=request.returnMetadata
        )
        
        return GetVectorsResponse(vectors=vectors)
        
    except (ResourceNotFoundException, ValidationException):
        raise
    except Exception as e:
        from .errors import InternalServiceException
        raise InternalServiceException(f"Failed to get vectors: {str(e)}")


@router.post("/buckets/{bucket_name}/indexes/{index_name}/vectors:list")
async def list_vectors(
    bucket_name: str,
    index_name: str,
    request: ListVectorsRequest
) -> ListVectorsResponse:
    """List vectors with optional pagination"""
    try:
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        if not s3.bucket_exists(s3_bucket):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found"
            )
        
        # Connect to Lance
        db = connect_bucket(bucket_name)
        table_uri = table_path(index_name)
        
        # List vectors with segmentation
        vectors = await index_ops.list_vectors(
            db, table_uri, 
            segment_id=request.segmentId,
            segment_count=request.segmentCount,
            max_results=request.maxResults
        )
        
        return ListVectorsResponse(
            vectors=vectors,
            nextToken=None  # Simplified pagination
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list vectors: {str(e)}"
        )

@router.post("/buckets/{bucket_name}/indexes/{index_name}/vectors:delete")
async def delete_vectors(
    bucket_name: str,
    index_name: str,
    request: DeleteVectorsRequest
) -> DeleteVectorsResponse:
    """Delete vectors by keys"""
    try:
        s3 = S3Storage()
        s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
        
        if not s3.bucket_exists(s3_bucket):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Bucket {bucket_name} not found"
            )
        
        # Connect to Lance
        db = connect_bucket(bucket_name)
        table_uri = table_path(index_name)
        
        # Delete vectors
        deleted_count = await index_ops.delete_vectors(db, table_uri, request.vectorKeys)
        
        return DeleteVectorsResponse(
            deletedVectorCount=deleted_count,
            deletedVectorKeys=request.vectorKeys[:deleted_count]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete vectors: {str(e)}"
        )
