from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import json
from datetime import datetime, timezone

from .models import (
    CreateVectorBucketRequest, CreateVectorBucketResponse,
    ListVectorBucketsRequest, ListVectorBucketsResponse,
    GetVectorBucketRequest, GetVectorBucketResponse,
    DeleteVectorBucketRequest, DeleteVectorBucketResponse,
    CreateIndexRequest, CreateIndexResponse,
    ListIndexesRequest, ListIndexesResponse,
    GetIndexRequest, GetIndexResponse,
    DeleteIndexRequest, DeleteIndexResponse,
    PutVectorsRequest, PutVectorsResponse,
    GetVectorsRequest, GetVectorsResponse,
    DeleteVectorsRequest, DeleteVectorsResponse,
    ListVectorsRequest, ListVectorsResponse,
    QueryVectorsRequest, QueryVectorsResponse,
    VectorBucket, VectorBucketSummary, Index, IndexSummary,
    ListOutputVector, QueryOutputVector, VectorData
)
from .util import config
from .storage.s3_backend import S3Storage
from .metadata.filter_engine import matches
from .index.indexer import process_new_slices, search, get_vectors_by_ids, get_vectors_by_keys, delete_by_keys, list_vectors

router = APIRouter()

def _get_iso_timestamp() -> str:
    """Get current timestamp in ISO 8601 format for S3 Vectors API compatibility"""
    return datetime.now(timezone.utc).isoformat()

def _cfg_key(index: str) -> str:
    return f"{config.INDEX_DIR}/{index}/config.json"

def _get_bucket_name(req) -> str:
    """Extract bucket name from request, preferring vectorBucketName over vectorBucketArn"""
    if hasattr(req, 'vectorBucketName') and req.vectorBucketName:
        return req.vectorBucketName
    if hasattr(req, 'vectorBucketArn') and req.vectorBucketArn:
        # Extract name from ARN if needed - for now just use the ARN as name
        return req.vectorBucketArn.split(':')[-1] if ':' in req.vectorBucketArn else req.vectorBucketArn
    raise HTTPException(status_code=400, detail="vectorBucketName or vectorBucketArn required")

def _get_index_name(req) -> str:
    """Extract index name from request, preferring indexName over indexArn"""
    if hasattr(req, 'indexName') and req.indexName:
        return req.indexName
    if hasattr(req, 'indexArn') and req.indexArn:
        # Extract name from ARN if needed
        return req.indexArn.split(':')[-1] if ':' in req.indexArn else req.indexArn
    raise HTTPException(status_code=400, detail="indexName or indexArn required")

def _generate_arn(resource_type: str, bucket_name: str, resource_name: str = None) -> str:
    """Generate ARN for S3 Vectors resources"""
    base = f"arn:aws:s3vectors:us-east-1:123456789012:{resource_type}/{bucket_name}"
    return f"{base}/{resource_name}" if resource_name else base

# ---------- Buckets ----------
@router.post("/CreateVectorBucket", response_model=CreateVectorBucketResponse)
def create_vector_bucket(req: CreateVectorBucketRequest) -> CreateVectorBucketResponse:
    s3 = S3Storage()
    s3.ensure_bucket(req.vectorBucketName)
    return CreateVectorBucketResponse()

@router.post("/ListVectorBuckets", response_model=ListVectorBucketsResponse)
def list_vector_buckets(req: ListVectorBucketsRequest) -> ListVectorBucketsResponse:
    s3 = S3Storage()
    names = sorted(s3.list_vector_buckets())
    
    # Apply prefix filter if provided
    if req.prefix:
        names = [n for n in names if n.startswith(req.prefix)]
    
    start = int(req.nextToken or 0)
    max_results = req.maxResults or 100
    end = min(len(names), start + max_results)
    page = names[start:end]
    next_token = str(end) if end < len(names) else None
    
    # Convert to VectorBucketSummary objects
    buckets = []
    for name in page:
        buckets.append(VectorBucketSummary(
            creationTime=_get_iso_timestamp(),  # ISO 8601 formatted timestamp
            vectorBucketArn=_generate_arn("bucket", name),
            vectorBucketName=name
        ))
    
    return ListVectorBucketsResponse(vectorBuckets=buckets, nextToken=next_token)

@router.post("/GetVectorBucket", response_model=GetVectorBucketResponse)
def get_vector_bucket(req: GetVectorBucketRequest) -> GetVectorBucketResponse:
    bucket_name = _get_bucket_name(req)
    s3 = S3Storage()
    
    try:
        s3.ensure_bucket(bucket_name)  # NOP if exists
    except OSError:
        return GetVectorBucketResponse()  # Return empty response if bucket doesn't exist
    
    # Create VectorBucket object
    bucket = VectorBucket(
        creationTime=_get_iso_timestamp(),  # ISO 8601 formatted timestamp
        vectorBucketArn=_generate_arn("bucket", bucket_name),
        vectorBucketName=bucket_name
    )
    
    return GetVectorBucketResponse(vectorBucket=bucket)

@router.post("/DeleteVectorBucket", response_model=DeleteVectorBucketResponse)
def delete_vector_bucket(req: DeleteVectorBucketRequest) -> DeleteVectorBucketResponse:
    bucket_name = _get_bucket_name(req)
    s3 = S3Storage()
    # delete our prefix content only
    s3.delete_prefix(bucket_name, f"{config.INDEX_DIR}/")
    s3.delete_prefix(bucket_name, f"{config.STAGED_DIR}/")
    # NOTE: not deleting the bucket itself (leave to operator)
    return DeleteVectorBucketResponse()

# ---------- Indexes ----------
@router.post("/CreateIndex", response_model=CreateIndexResponse)
def create_index(req: CreateIndexRequest) -> CreateIndexResponse:
    if req.dimension < 1 or req.dimension > config.MAX_DIM:
        raise HTTPException(status_code=400, detail="Invalid dimension")
    
    bucket_name = _get_bucket_name(req)
    s3 = S3Storage()
    s3.ensure_bucket(bucket_name)
    
    cfg = {
        "dimension": req.dimension,
        "dataType": req.dataType,
        "distanceMetric": req.distanceMetric,
        "algorithm": "hybrid",  # default for our implementation
        "hnswThreshold": 100_000,  # default
        "nList": 1024,  # default
        "m": 16,  # default
        "nbits": 8,  # default
        "metadataConfiguration": req.metadataConfiguration or {}
    }
    s3.put_json(bucket_name, _cfg_key(req.indexName), cfg)
    
    # write empty manifest
    s3.put_json(bucket_name, f"{config.INDEX_DIR}/{req.indexName}/{config.MANIFEST_KEY}", {
        "index": req.indexName, "dataType": req.dataType, "dimension": req.dimension,
        "distanceMetric": req.distanceMetric, "vectors": 0
    })
    return CreateIndexResponse()

@router.post("/ListIndexes", response_model=ListIndexesResponse)
def list_indexes(req: ListIndexesRequest) -> ListIndexesResponse:
    bucket_name = _get_bucket_name(req)
    s3 = S3Storage()
    names = set()
    prefix = f"{config.INDEX_DIR}/"
    for k in s3.list_prefix(bucket_name, prefix):
        parts = k.split("/")
        if len(parts) >= 3:
            names.add(parts[1])
    
    names = sorted(names)
    
    # Apply prefix filter if provided
    if req.prefix:
        names = [n for n in names if n.startswith(req.prefix)]
    
    start = int(req.nextToken or 0)
    max_results = req.maxResults or 100
    end = min(len(names), start + max_results)
    page = names[start:end]
    next_token = str(end) if end < len(names) else None
    
    # Convert to IndexSummary objects
    indexes = []
    for name in page:
        indexes.append(IndexSummary(
            creationTime=_get_iso_timestamp(),  # ISO 8601 formatted timestamp
            indexArn=_generate_arn("index", bucket_name, name),
            indexName=name,
            vectorBucketName=bucket_name
        ))
    
    return ListIndexesResponse(indexes=indexes, nextToken=next_token)

@router.post("/GetIndex", response_model=GetIndexResponse)
def get_index(req: GetIndexRequest) -> GetIndexResponse:
    bucket_name = _get_bucket_name(req)
    index_name = _get_index_name(req)
    
    s3 = S3Storage()
    cfg = s3.get_json(bucket_name, _cfg_key(index_name))
    if not cfg:
        return GetIndexResponse()  # Return empty if index doesn't exist
    
    index = Index(
        creationTime=_get_iso_timestamp(),  # ISO 8601 formatted timestamp
        dataType=cfg["dataType"],
        dimension=cfg["dimension"],
        distanceMetric=cfg["distanceMetric"],
        indexArn=_generate_arn("index", bucket_name, index_name),
        indexName=index_name,
        vectorBucketName=bucket_name,
        metadataConfiguration=cfg.get("metadataConfiguration")
    )
    
    return GetIndexResponse(index=index)

@router.post("/DeleteIndex", response_model=DeleteIndexResponse)
def delete_index(req: DeleteIndexRequest) -> DeleteIndexResponse:
    bucket_name = _get_bucket_name(req)
    index_name = _get_index_name(req)
    
    s3 = S3Storage()
    s3.delete_prefix(bucket_name, f"{config.INDEX_DIR}/{index_name}/")
    s3.delete_prefix(bucket_name, f"{config.STAGED_DIR}/{index_name}/")
    return DeleteIndexResponse()

# ---------- Vectors ----------
@router.post("/PutVectors", response_model=PutVectorsResponse)
def put_vectors(req: PutVectorsRequest) -> PutVectorsResponse:
    if len(req.vectors) == 0:
        return PutVectorsResponse()
    if len(req.vectors) > config.MAX_BATCH:
        raise HTTPException(status_code=400, detail="Too many vectors in one request")

    bucket_name = _get_bucket_name(req)
    index_name = _get_index_name(req)
    
    s3 = S3Storage()
    cfg = s3.get_json(bucket_name, _cfg_key(index_name))
    if not cfg:
        raise HTTPException(status_code=404, detail="Index not found")
    dim = int(cfg["dimension"])

    # validate + write slice
    rows = []
    for v in req.vectors:
        vec = v.data.float32 or []
        if len(vec) != dim:
            raise HTTPException(status_code=400, detail="Vector dimension mismatch")
        md_bytes = len(json.dumps(v.metadata or {}).encode("utf-8"))
        if md_bytes > config.MAX_TOTAL_METADATA_BYTES:
            raise HTTPException(status_code=400, detail="Metadata too large")
        rows.append({"key": v.key, "vec": vec, "meta": v.metadata or {}})

    s3.write_slice(bucket_name, index_name, rows)

    # callback indexing
    process_new_slices(
        vector_bucket=bucket_name,
        index=index_name,
        dim=dim,
        metric=cfg.get("distanceMetric", "cosine"),
        algorithm=cfg.get("algorithm", "hybrid"),
        hnsw_threshold=int(cfg.get("hnswThreshold", 100_000)),
        nlist=int(cfg.get("nList", 1024)),
        m=int(cfg.get("m", 16)),
        nbits=int(cfg.get("nbits", 8)),
    )
    return PutVectorsResponse()

@router.post("/GetVectors", response_model=GetVectorsResponse)
def get_vectors(req: GetVectorsRequest) -> GetVectorsResponse:
    bucket_name = _get_bucket_name(req)
    index_name = _get_index_name(req)
    
    raw_vectors = get_vectors_by_keys(bucket_name, index_name, req.keys)
    
    # Convert to ListOutputVector format
    vectors = []
    for v in raw_vectors:
        vector_data = None
        if req.returnData:
            vector_data = VectorData(float32=v.get("Data", {}).get("float32"))  # Fixed field access
        
        metadata = v.get("Metadata") if req.returnMetadata else None  # Fixed field name from "meta" to "Metadata"
        
        vectors.append(ListOutputVector(
            key=v["Key"],  # Fixed field name from "key" to "Key"
            data=vector_data,
            metadata=metadata
        ))
    
    return GetVectorsResponse(vectors=vectors)

@router.post("/DeleteVectors", response_model=DeleteVectorsResponse)
def delete_vectors(req: DeleteVectorsRequest) -> DeleteVectorsResponse:
    bucket_name = _get_bucket_name(req)
    index_name = _get_index_name(req)
    
    delete_by_keys(bucket_name, index_name, req.keys)
    return DeleteVectorsResponse()

@router.post("/ListVectors", response_model=ListVectorsResponse)
def list_vectors_api(req: ListVectorsRequest) -> ListVectorsResponse:
    bucket_name = _get_bucket_name(req)
    index_name = _get_index_name(req)
    
    max_results = req.maxResults or 1000
    items, nxt = list_vectors(bucket_name, index_name, max_results, req.nextToken)
    
    # Convert to ListOutputVector format
    vectors = []
    for v in items:
        vector_data = None
        if req.returnData:
            vector_data = VectorData(float32=v.get("Data", {}).get("float32"))  # Fixed field access
        
        metadata = v.get("Metadata") if req.returnMetadata else None  # Fixed field name from "meta" to "Metadata"
        
        vectors.append(ListOutputVector(
            key=v["Key"],  # Fixed field name from "key" to "Key"
            data=vector_data,
            metadata=metadata
        ))
    
    return ListVectorsResponse(vectors=vectors, nextToken=nxt)

# ---------- Query ----------
@router.post("/QueryVectors", response_model=QueryVectorsResponse)
def query_vectors(req: QueryVectorsRequest) -> QueryVectorsResponse:
    bucket_name = _get_bucket_name(req)
    index_name = _get_index_name(req)
    
    topk = min(req.topK, config.MAX_TOPK)
    
    # Extract query vector
    if not req.queryVector or not req.queryVector.float32:
        raise HTTPException(status_code=400, detail="queryVector with float32 data required")
    
    # do ANN search
    ids_dists = search(bucket_name, index_name, req.queryVector.float32, topk, getattr(req, 'nprobe', None))
    if not ids_dists: 
        return QueryVectorsResponse(vectors=[])
    
    ids = [i for i, _ in ids_dists]
    rows = get_vectors_by_ids(bucket_name, index_name, ids)
    
    # Build output with distances
    out = []
    for (i, d), row in zip(ids_dists, rows):
        vector_data = None
        if req.returnData:
            vector_data = VectorData(float32=row.get("Data", {}).get("float32"))  # Fixed field access
        
        metadata = row.get("Metadata") if req.returnMetadata else None  # Fixed field name from "meta" to "Metadata"
        distance = d if req.returnDistance else None
        
        out.append(QueryOutputVector(
            key=row["Key"],  # Changed from "key" to "Key" to match indexer.py response format
            distance=distance,
            data=vector_data,
            metadata=metadata
        ))

    # Apply filtering (post-filter)
    if req.filter:
        flt = req.filter.root
        out = [r for r in out if matches(r.metadata or {}, flt)]

    return QueryVectorsResponse(vectors=out[:topk])
