from pydantic import BaseModel, Field, RootModel
from typing import List, Optional, Dict, Any, Literal

Metric = Literal["cosine", "euclidean"]
Algorithm = Literal["hnsw_flat", "ivfpq", "hybrid"]

class FilterableKey(BaseModel):
    name: str
    type: Literal["int64","float64","bool","string","string[]"]

# ===== Request Models =====
class CreateVectorBucketRequest(BaseModel):
    vectorBucketName: str
    encryptionConfiguration: Optional[Dict[str, Any]] = None

class ListVectorBucketsRequest(BaseModel):
    maxResults: Optional[int] = None
    nextToken: Optional[str] = None
    prefix: Optional[str] = None

class GetVectorBucketRequest(BaseModel):
    vectorBucketArn: Optional[str] = None
    vectorBucketName: Optional[str] = None

class DeleteVectorBucketRequest(BaseModel):
    vectorBucketArn: Optional[str] = None
    vectorBucketName: Optional[str] = None

class CreateIndexRequest(BaseModel):
    vectorBucketArn: Optional[str] = None
    vectorBucketName: Optional[str] = None
    indexName: str
    dataType: Literal["float32"]
    dimension: int = Field(ge=1, le=4096)
    distanceMetric: Literal["euclidean", "cosine"]
    metadataConfiguration: Optional[Dict[str, Any]] = None

class ListIndexesRequest(BaseModel):
    vectorBucketArn: Optional[str] = None
    vectorBucketName: Optional[str] = None
    prefix: Optional[str] = None
    maxResults: Optional[int] = None
    nextToken: Optional[str] = None

class GetIndexRequest(BaseModel):
    vectorBucketName: Optional[str] = None
    indexName: Optional[str] = None
    indexArn: Optional[str] = None

class DeleteIndexRequest(BaseModel):
    vectorBucketName: Optional[str] = None
    indexName: Optional[str] = None
    indexArn: Optional[str] = None

class VectorData(BaseModel):
    float32: Optional[List[float]] = None

class PutInputVector(BaseModel):
    key: str
    data: VectorData
    metadata: Optional[Dict[str, Any]] = None

class PutVectorsRequest(BaseModel):
    vectorBucketName: Optional[str] = None
    indexName: Optional[str] = None
    indexArn: Optional[str] = None
    vectors: List[PutInputVector]

class GetVectorsRequest(BaseModel):
    vectorBucketName: Optional[str] = None
    indexName: Optional[str] = None
    indexArn: Optional[str] = None
    keys: List[str]
    returnData: Optional[bool] = None
    returnMetadata: Optional[bool] = None

class DeleteVectorsRequest(BaseModel):
    vectorBucketName: Optional[str] = None
    indexName: Optional[str] = None
    indexArn: Optional[str] = None
    keys: List[str]

class ListVectorsRequest(BaseModel):
    vectorBucketName: Optional[str] = None
    indexName: Optional[str] = None
    indexArn: Optional[str] = None
    maxResults: Optional[int] = None
    nextToken: Optional[str] = None
    returnData: Optional[bool] = None
    returnMetadata: Optional[bool] = None
    segmentCount: Optional[int] = None
    segmentIndex: Optional[int] = None

class MetadataFilter(RootModel[Dict[str, Any]]): ...

class QueryVectorsRequest(BaseModel):
    vectorBucketName: Optional[str] = None
    indexName: Optional[str] = None
    indexArn: Optional[str] = None
    queryVector: Optional[VectorData] = None
    filter: Optional[MetadataFilter] = None
    topK: int
    returnDistance: Optional[bool] = None
    returnData: Optional[bool] = None
    returnMetadata: Optional[bool] = None

# ===== Response Models =====
class EncryptionConfiguration(BaseModel):
    sseType: Optional[str] = None
    kmsKeyArn: Optional[str] = None

class VectorBucket(BaseModel):
    creationTime: str  # ISO 8601 formatted date-time string
    vectorBucketArn: str
    vectorBucketName: str
    encryptionConfiguration: Optional[EncryptionConfiguration] = None

class VectorBucketSummary(BaseModel):
    creationTime: str  # ISO 8601 formatted date-time string
    vectorBucketArn: str
    vectorBucketName: str

class IndexSummary(BaseModel):
    creationTime: str  # ISO 8601 formatted date-time string
    indexArn: str
    indexName: str
    vectorBucketName: str

class MetadataConfiguration(BaseModel):
    nonFilterableMetadataKeys: Optional[List[str]] = None

class Index(BaseModel):
    creationTime: str  # ISO 8601 formatted date-time string
    dataType: str
    dimension: int
    distanceMetric: str
    indexArn: str
    indexName: str
    vectorBucketName: str
    metadataConfiguration: Optional[MetadataConfiguration] = None

class ListOutputVector(BaseModel):
    key: str
    data: Optional[VectorData] = None
    metadata: Optional[Dict[str, Any]] = None

class QueryOutputVector(BaseModel):
    key: str
    distance: Optional[float] = None
    data: Optional[VectorData] = None
    metadata: Optional[Dict[str, Any]] = None

# Response containers
class CreateVectorBucketResponse(BaseModel):
    pass

class GetVectorBucketResponse(BaseModel):
    vectorBucket: Optional[VectorBucket] = None

class ListVectorBucketsResponse(BaseModel):
    vectorBuckets: Optional[List[VectorBucketSummary]] = None
    nextToken: Optional[str] = None

class DeleteVectorBucketResponse(BaseModel):
    pass

class CreateIndexResponse(BaseModel):
    pass

class GetIndexResponse(BaseModel):
    index: Optional[Index] = None

class ListIndexesResponse(BaseModel):
    indexes: Optional[List[IndexSummary]] = None
    nextToken: Optional[str] = None

class DeleteIndexResponse(BaseModel):
    pass

class PutVectorsResponse(BaseModel):
    pass

class GetVectorsResponse(BaseModel):
    vectors: Optional[List[ListOutputVector]] = None

class ListVectorsResponse(BaseModel):
    vectors: Optional[List[ListOutputVector]] = None
    nextToken: Optional[str] = None

class DeleteVectorsResponse(BaseModel):
    pass

class QueryVectorsResponse(BaseModel):
    vectors: Optional[List[QueryOutputVector]] = None
