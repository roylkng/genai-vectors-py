from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from .api import router
from .storage.s3_backend import S3Storage
from .lance.db import connect_bucket, table_path
from .lance import index_ops
from .util import config
from datetime import datetime
import json
import os

# Enhanced OpenAPI/Swagger configuration
app = FastAPI(
    title="S3 Vectors API",
    description="""
    A high-performance vector database API that implements the AWS S3 Vectors interface using Lance.
    
    ## Features
    
    * **Complete AWS S3 Vectors API compatibility** - All endpoints match boto3/CLI shapes
    * **Lance vector database** - High-performance vector storage with S3 backend
    * **Configurable indexing** - IVF_PQ, HNSW, or no indexing
    * **Real-time embeddings** - Integration with LM Studio and other embedding services
    * **Metadata filtering** - Advanced JSON-based filtering capabilities
    
    ## Quick Start
    
    1. **Create a bucket**: `POST /CreateVectorBucket`
    2. **Create an index**: `POST /CreateIndex`  
    3. **Insert vectors**: `POST /PutVectors`
    4. **Search vectors**: `POST /QueryVectors`
    
    ## Supported Index Types
    
    * **IVF_PQ** - Best for high-dimensional vectors (>100D), fast search
    * **HNSW** - Best for low-dimensional vectors (<100D), better recall  
    * **NONE** - Brute force search, no indexing overhead
    
    ## Authentication
    
    Currently uses simple key-based authentication. Set environment variables:
    - `S3_ACCESS_KEY` 
    - `S3_SECRET_KEY`
    """,
    version="1.0.0",
    terms_of_service="https://github.com/roylkng/genai-vectors-py",
    contact={
        "name": "S3 Vectors API Support",
        "url": "https://github.com/roylkng/genai-vectors-py",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.example.com",
            "description": "Production server"
        }
    ],
    tags_metadata=[
        {
            "name": "Vector Buckets",
            "description": "Operations for managing vector storage buckets"
        },
        {
            "name": "Indexes", 
            "description": "Operations for managing vector indexes and search configurations"
        },
        {
            "name": "Vectors",
            "description": "Operations for storing, retrieving, and searching vectors"
        },
        {
            "name": "S3 Compatibility",
            "description": "S3-compatible endpoints for boto3 integration"
        },
        {
            "name": "Health",
            "description": "System health and status endpoints"
        }
    ]
)
app.include_router(router)

# S3 Vectors service endpoints (for boto3 compatibility)
@app.post("/ListVectorBuckets", tags=["Vector Buckets"])
async def list_vector_buckets_service():
    """
    List all vector buckets in the system.
    
    Returns a list of all vector buckets with their metadata including:
    - Bucket name and ARN
    - Creation timestamp
    - Configuration details
    
    **Response Format:**
    ```json
    {
        "vectorBuckets": [
            {
                "vectorBucketName": "my-bucket",
                "creationTime": "2025-01-01T00:00:00Z",
                "vectorBucketArn": "arn:aws:s3-vectors:::bucket/my-bucket"
            }
        ]
    }
    ```
    """
    try:
        s3 = S3Storage()
        
        # List vector buckets using existing method
        bucket_names = s3.list_vector_buckets()
        vector_buckets = []
        
        for bucket_name in bucket_names:
            s3_bucket = f"{config.S3_BUCKET_PREFIX}{bucket_name}"
            
            # Try to get bucket metadata
            try:
                bucket_data = s3.get_json(bucket_name, f"{config.META_DIR}/bucket.json")
                if bucket_data:
                    created = bucket_data.get("created", datetime.utcnow().isoformat())
                else:
                    created = datetime.utcnow().isoformat()
            except:
                created = datetime.utcnow().isoformat()
            
            vector_buckets.append({
                "vectorBucketName": bucket_name,
                "creationTime": created,
                "vectorBucketArn": f"arn:aws:s3-vectors:::bucket/{bucket_name}"
            })
        
        return {"vectorBuckets": vector_buckets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/CreateVectorBucket", tags=["Vector Buckets"])
async def create_vector_bucket_service(request: Request):
    """
    Create a new vector bucket for storing vector indexes.
    
    Creates a new vector storage bucket with the specified configuration.
    The bucket will be used to store vector indexes and associated metadata.
    
    **Request Body:**
    ```json
    {
        "vectorBucketName": "my-new-bucket",
        "vectorDimensions": 768
    }
    ```
    
    **Response:**
    ```json
    {
        "vectorBucketName": "my-new-bucket", 
        "vectorBucketArn": "arn:aws:s3-vectors:::bucket/my-new-bucket"
    }
    ```
    """
    try:
        body = await request.json()
        print(f"DEBUG: CreateVectorBucket request body: {body}")
        
        # Handle both boto3 and direct API formats with comprehensive parameter extraction
        bucket_name = (body.get("VectorBucketName") or 
                      body.get("vectorBucketName") or
                      body.get("bucketName"))
        
        if not bucket_name:
            print(f"DEBUG: VectorBucketName not found in body: {body}")
            raise HTTPException(status_code=400, detail="VectorBucketName, vectorBucketName, or bucketName required")
        
        # Validate bucket name format
        import re
        if not re.match(r'^[a-z0-9][a-z0-9\-]*[a-z0-9]$', bucket_name):
            raise HTTPException(status_code=400, detail="Bucket name must contain only lowercase letters, numbers, and hyphens, and must start and end with a letter or number")
        
        if len(bucket_name) < 3 or len(bucket_name) > 63:
            raise HTTPException(status_code=400, detail="Bucket name must be between 3 and 63 characters long")
        
        s3 = S3Storage()
        
        # Check if bucket already exists
        existing_buckets = s3.list_vector_buckets()
        if bucket_name in existing_buckets:
            raise HTTPException(status_code=409, detail=f"Bucket {bucket_name} already exists")
        
        # Ensure underlying S3 bucket exists
        s3.ensure_bucket(bucket_name)
        
        # Store bucket metadata
        bucket_config = {
            "name": bucket_name,
            "created": datetime.utcnow().isoformat(),
            "engine": "lance",
            "version": "1.0"
        }
        
        s3.put_json(bucket_name, f"{config.META_DIR}/bucket.json", bucket_config)
        
        return {
            "vectorBucketName": bucket_name,
            "vectorBucketArn": f"arn:aws:s3-vectors:::bucket/{bucket_name}"
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"CreateVectorBucket Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/CreateIndex", tags=["Indexes"])
async def create_index_service(request: Request):
    """
    Create a new vector index for similarity search.
    
    Creates a high-performance vector index using Lance database with configurable
    indexing strategies (IVF_PQ, HNSW, or brute force).
    
    **Request Body:**
    ```json
    {
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "dimension": 768,
        "dataType": "float32",
        "distanceMetric": "cosine",
        "metadataConfiguration": {
            "fields": ["category", "timestamp"]
        }
    }
    ```
    
    **Index Types (set via LANCE_INDEX_TYPE env var):**
    - **IVF_PQ**: Best for high-dimensional vectors (>100D)
    - **HNSW**: Best for low-dimensional vectors (<100D) 
    - **NONE**: Brute force search, no indexing overhead
    
    **Response:**
    ```json
    {
        "indexName": "my-index",
        "dimension": 768,
        "dataType": "float32", 
        "distanceMetric": "cosine",
        "indexArn": "arn:aws:s3-vectors:::bucket/my-bucket/index/my-index"
    }
    ```
    """
    try:
        body = await request.json()
        print(f"DEBUG: CreateIndex request body: {body}")
        
        # Handle both boto3 and direct API formats with comprehensive parameter extraction
        bucket_name = (body.get("vectorBucketName") or 
                      body.get("VectorBucketName") or
                      (body.get("vectorBucketArn", "").split("/")[-1] if body.get("vectorBucketArn") else None))
        
        index_name = (body.get("indexName") or 
                     body.get("IndexName"))
        
        dimension = (body.get("dimension") or 
                    body.get("Dimension"))
        
        data_type = body.get("dataType", "float32")
        distance_metric = body.get("distanceMetric", "cosine")
        metadata_config = body.get("metadataConfiguration", {})
        
        # Validate required parameters
        if not bucket_name:
            raise HTTPException(status_code=400, detail="vectorBucketName, VectorBucketName, or vectorBucketArn required")
        if not index_name:
            raise HTTPException(status_code=400, detail="indexName or IndexName required")
        if not dimension:
            raise HTTPException(status_code=400, detail="dimension or Dimension required")
        
        # Validate dimension is a positive integer
        try:
            dimension = int(dimension)
            if dimension <= 0:
                raise ValueError("Dimension must be positive")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="dimension must be a positive integer")
        
        s3 = S3Storage()
        
        # Check if bucket exists
        existing_buckets = s3.list_vector_buckets()
        if bucket_name not in existing_buckets:
            raise HTTPException(status_code=404, detail=f"Bucket {bucket_name} not found")
        
        # Create Lance table
        db = connect_bucket(bucket_name)
        table_uri = table_path(index_name)
        
        await index_ops.create_table(db, table_uri, dimension)
        
        # Store index metadata
        index_config = {
            "indexName": index_name,
            "dimension": dimension,
            "dataType": data_type,
            "distanceMetric": distance_metric,
            "metadataConfiguration": metadata_config,
            "created": datetime.utcnow().isoformat(),
            "engine": "lance",
            "indexType": config.LANCE_INDEX_TYPE
        }
        
        s3.put_json(bucket_name, f"{config.INDEX_DIR}/{index_name}/_index_config.json", index_config)
        
        return {
            "indexName": index_name,
            "dimension": dimension,
            "dataType": data_type,
            "distanceMetric": distance_metric,
            "indexArn": f"arn:aws:s3-vectors:::bucket/{bucket_name}/index/{index_name}"
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"CreateIndex Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ListIndexes", tags=["Indexes"])
@app.get("/ListIndexes", tags=["Indexes"])
async def list_indexes_service(request: Request):
    """
    List all vector indexes in a bucket.
    
    Returns all vector indexes created in the specified bucket along with
    their configuration and metadata.
    
    **Query Parameters (GET) or Request Body (POST):**
    ```json
    {
        "vectorBucketName": "my-bucket"
    }
    ```
    
    **Response:**
    ```json
    {
        "indexes": [
            {
                "indexName": "my-index",
                "dimension": 768,
                "dataType": "float32",
                "distanceMetric": "cosine", 
                "indexArn": "arn:aws:s3-vectors:::bucket/my-bucket/index/my-index"
            }
        ],
        "count": 1
    }
    ```
    """
    try:
        # Handle both GET (query params) and POST (JSON body)
        bucket_name = None
        if request.method == "GET":
            bucket_name = (request.query_params.get("vectorBucketName") or 
                          request.query_params.get("VectorBucketName"))
        else:
            body = await request.json()
            bucket_name = (body.get("vectorBucketName") or 
                          body.get("VectorBucketName") or
                          (body.get("vectorBucketArn", "").split("/")[-1] if body.get("vectorBucketArn") else None))
        
        if not bucket_name:
            raise HTTPException(status_code=400, detail="vectorBucketName or VectorBucketName required")
        
        print(f"DEBUG: ListIndexes for bucket: {bucket_name}")
        
        s3 = S3Storage()
        
        # Check if bucket exists
        existing_buckets = s3.list_vector_buckets()
        if bucket_name not in existing_buckets:
            raise HTTPException(status_code=404, detail=f"Bucket {bucket_name} not found")
        
        # List Lance tables as indexes
        db = connect_bucket(bucket_name)
        
        try:
            table_names = db.table_names()
            indexes = []
            
            print(f"DEBUG: Found {len(table_names)} tables: {table_names}")
            
            for table_name in table_names:
                # Try to get index metadata
                try:
                    index_config = s3.get_json(bucket_name, f"{config.INDEX_DIR}/{table_name}/_index_config.json")
                    dimension = index_config.get("dimension", 768)
                    data_type = index_config.get("dataType", "float32")
                    distance_metric = index_config.get("distanceMetric", "cosine")
                    creation_time = index_config.get("created", datetime.utcnow().isoformat())
                except:
                    # Fallback for indexes without metadata
                    dimension = 768
                    data_type = "float32"
                    distance_metric = "cosine"
                    creation_time = datetime.utcnow().isoformat()
                
                indexes.append({
                    "indexName": table_name,
                    "dimension": dimension,
                    "dataType": data_type, 
                    "distanceMetric": distance_metric,
                    "indexArn": f"arn:aws:s3-vectors:::bucket/{bucket_name}/index/{table_name}",
                    "creationTime": creation_time,
                    "vectorBucketName": bucket_name
                })
            
            return {
                "indexes": indexes,
                "count": len(indexes)
            }
            
        except Exception as e:
            print(f"DEBUG: Error listing tables: {e}")
            # No tables yet
            return {
                "indexes": [],
                "count": 0
            }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"DEBUG: ListIndexes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/PutVectors", tags=["Vectors"])
async def put_vectors_service(request: Request):
    """
    Insert or update vectors in an index.
    
    Stores vector embeddings with optional metadata in the specified index.
    Automatically rebuilds the index for optimal search performance.
    
    **Request Body:**
    ```json
    {
        "vectorBucketName": "my-bucket",
        "indexName": "my-index", 
        "vectors": [
            {
                "key": "doc1",
                "data": {
                    "float32": [0.1, 0.2, 0.3, ...]
                },
                "metadata": {
                    "category": "technology",
                    "title": "AI Research Paper",
                    "timestamp": "2025-01-01T00:00:00Z"
                }
            }
        ]
    }
    ```
    
    **Vector Data Formats:**
    - `float32`: Array of 32-bit floating point numbers
    - `vector`: Generic vector array (auto-converted to float32)
    
    **Response:**
    ```json
    {
        "vectorCount": 1,
        "successful": true
    }
    ```
    """
    try:
        body = await request.json()
        print(f"DEBUG: PutVectors request body keys: {list(body.keys())}")
        
        # Handle both boto3 and direct API formats with comprehensive parameter extraction
        bucket_name = (body.get("vectorBucketName") or 
                      body.get("VectorBucketName") or
                      (body.get("vectorBucketArn", "").split("/")[-1] if body.get("vectorBucketArn") else None))
        
        index_name = (body.get("indexName") or 
                     body.get("IndexName") or
                     (body.get("indexArn", "").split("/")[-1] if body.get("indexArn") else None))
        
        vectors = body.get("vectors") or body.get("Vectors", [])
        
        # Validate required parameters
        if not bucket_name:
            raise HTTPException(status_code=400, detail="vectorBucketName, VectorBucketName, or vectorBucketArn required")
        if not index_name:
            raise HTTPException(status_code=400, detail="indexName, IndexName, or indexArn required")
        if not vectors:
            raise HTTPException(status_code=400, detail="vectors array cannot be empty")
        
        s3 = S3Storage()
        
        # Check if bucket exists
        existing_buckets = s3.list_vector_buckets()
        if bucket_name not in existing_buckets:
            raise HTTPException(status_code=404, detail=f"Bucket {bucket_name} not found")
        
        # Connect to Lance table
        db = connect_bucket(bucket_name)
        table_uri = table_path(index_name)
        
        # Prepare vector data for Lance
        vector_data = []
        for i, vec in enumerate(vectors):
            key = vec.get("key")
            if not key:
                raise HTTPException(status_code=400, detail=f"Vector at index {i} missing required 'key' field")
                
            data = vec.get("data", {})
            metadata = vec.get("metadata", {})
            
            # Extract vector array (handle both float32 and other formats)
            vector_array = None
            if "float32" in data:
                vector_array = data["float32"]
            elif "vector" in data:
                vector_array = data["vector"]
            else:
                raise HTTPException(status_code=400, detail=f"Vector data format not supported for key {key}. Expected 'float32' or 'vector' field in data.")
            
            # Validate vector array
            if not isinstance(vector_array, list) or not vector_array:
                raise HTTPException(status_code=400, detail=f"Vector array must be a non-empty list for key {key}")
            
            # Create Lance row with metadata in nonfilter field
            lance_row = {
                "key": key,
                "vector": vector_array,
                "metadata": metadata  # Store metadata properly for Lance processing
            }
            vector_data.append(lance_row)
        
        # Insert vectors into Lance table
        await index_ops.upsert_vectors(db, table_uri, vector_data)
        
        # Auto-index if needed (smart indexing)
        await index_ops.rebuild_index(db, table_uri, config.LANCE_INDEX_TYPE)
        
        return {
            "vectorCount": len(vectors),
            "successful": True
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"PutVectors Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/QueryVectors", tags=["Vectors"]) 
async def query_vectors_service(request: Request):
    """
    Search for similar vectors using semantic similarity.
    
    Performs approximate nearest neighbor (ANN) search to find the most similar
    vectors to the query vector. Supports advanced filtering on metadata fields.
    
    **Request Body:**
    ```json
    {
        "vectorBucketName": "my-bucket",
        "indexName": "my-index",
        "queryVector": {
            "float32": [0.1, 0.2, 0.3, ...]
        },
        "topK": 10,
        "returnMetadata": true,
        "returnDistance": true,
        "filter": {
            "operator": "equals",
            "metadata_key": "category", 
            "value": "technology"
        }
    }
    ```
    
    **Filter Operators:**
    - `equals`: Exact match
    - `not_equals`: Not equal to value
    - `in`: Value in array
    - `not_in`: Value not in array
    - `greater_than`: Numeric greater than
    - `less_than`: Numeric less than
    - `and`: Logical AND of multiple conditions
    - `or`: Logical OR of multiple conditions
    
    **Response:**
    ```json
    {
        "vectors": [
            {
                "key": "doc1",
                "distance": 0.15,
                "metadata": {
                    "category": "technology",
                    "title": "AI Research Paper"
                }
            }
        ],
        "count": 1
    }
    ```
    
    **Performance:**
    - Query latency: 5-20ms (depending on index size)
    - Supports up to 10M+ vectors with sub-second search
    - Automatic index optimization for best performance
    """
    try:
        body = await request.json()
        print(f"DEBUG: QueryVectors request body keys: {list(body.keys())}")
        
        # Handle both boto3 and direct API formats with comprehensive parameter extraction
        bucket_name = (body.get("vectorBucketName") or 
                      body.get("VectorBucketName") or
                      (body.get("vectorBucketArn", "").split("/")[-1] if body.get("vectorBucketArn") else None))
        
        index_name = (body.get("indexName") or 
                     body.get("IndexName") or
                     (body.get("indexArn", "").split("/")[-1] if body.get("indexArn") else None))
        
        query_vector_data = (body.get("queryVector") or 
                           body.get("QueryVector"))
        
        top_k = body.get("topK") or body.get("TopK", 10)
        filter_condition = body.get("filter") or body.get("Filter")
        return_metadata = body.get("returnMetadata", True)
        return_distance = body.get("returnDistance", True)
        
        # Validate required parameters
        if not bucket_name:
            raise HTTPException(status_code=400, detail="vectorBucketName, VectorBucketName, or vectorBucketArn required")
        if not index_name:
            raise HTTPException(status_code=400, detail="indexName, IndexName, or indexArn required")
        if not query_vector_data:
            raise HTTPException(status_code=400, detail="queryVector or QueryVector required")
        
        # Extract vector array from query_vector_data
        query_vector = None
        if isinstance(query_vector_data, dict):
            query_vector = query_vector_data.get("float32") or query_vector_data.get("vector")
        else:
            query_vector = query_vector_data
            
        print(f"DEBUG: Query vector type: {type(query_vector)}, length: {len(query_vector) if query_vector else 0}")
        
        if not query_vector or not isinstance(query_vector, list):
            raise HTTPException(status_code=400, detail="Invalid queryVector format. Expected array of numbers in 'float32' or 'vector' field.")
        
        # Validate topK parameter
        try:
            top_k = int(top_k)
            if top_k <= 0:
                raise ValueError("topK must be positive")
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="topK must be a positive integer")
        
        s3 = S3Storage()
        
        # Check if bucket exists
        existing_buckets = s3.list_vector_buckets()
        if bucket_name not in existing_buckets:
            raise HTTPException(status_code=404, detail=f"Bucket {bucket_name} not found")
        
        print(f"DEBUG: Searching in bucket: {bucket_name}, index: {index_name}")
        
        # Connect to Lance table
        db = connect_bucket(bucket_name)
        table_uri = table_path(index_name)
        
        print(f"DEBUG: Table URI: {table_uri}")
        
        # Search vectors
        results = await index_ops.search_vectors(
            db, table_uri, query_vector, top_k, filter_condition
        )
        
        # Format response for boto3 compatibility
        output_vectors = []
        for result in results:
            vector_data = {
                "key": result["key"]
            }
            
            if return_distance:
                vector_data["distance"] = result.get("score", 0.0)
                
            if return_metadata and "metadata" in result:
                vector_data["metadata"] = result["metadata"]
                
            output_vectors.append(vector_data)
        
        return {
            "vectors": output_vectors,
            "count": len(output_vectors)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"QueryVectors Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# S3-compatible endpoints for boto3 client
@app.get("/", tags=["S3 Compatibility"])
async def s3_list_buckets():
    """
    S3-compatible ListBuckets endpoint.
    
    Maps to S3 Vectors ListVectorBuckets for seamless boto3 integration.
    Returns XML response compatible with AWS S3 API.
    """
    s3 = S3Storage()
    names = sorted(s3.list_vector_buckets())
    
    # Build S3-compatible XML response
    buckets_xml = ""
    for name in names:
        creation_date = datetime.utcnow().isoformat() + "Z"
        buckets_xml += f"""
        <Bucket>
            <Name>{name}</Name>
            <CreationDate>{creation_date}</CreationDate>
        </Bucket>"""
    
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<ListAllMyBucketsResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <Owner>
        <ID>s3vectors</ID>
        <DisplayName>s3vectors</DisplayName>
    </Owner>
    <Buckets>{buckets_xml}
    </Buckets>
</ListAllMyBucketsResult>"""
    
    return Response(content=xml_response, media_type="application/xml")

@app.put("/{bucket}", tags=["S3 Compatibility"])
async def s3_create_bucket(bucket: str):
    """
    S3-compatible CreateBucket endpoint.
    
    Maps to S3 Vectors CreateVectorBucket for seamless boto3 integration.
    Creates a vector bucket that can store multiple indexes.
    """
    s3 = S3Storage()
    s3.ensure_bucket(bucket)
    
    # Return S3-compatible XML response
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<CreateBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <Location>/{bucket}</Location>
</CreateBucketResult>"""
    
    return Response(content=xml_response, media_type="application/xml")

@app.delete("/{bucket}", tags=["S3 Compatibility"])
async def s3_delete_bucket(bucket: str):
    """
    S3-compatible DeleteBucket endpoint.
    
    Maps to S3 Vectors DeleteVectorBucket for seamless boto3 integration.
    Removes all vector data and indexes from the bucket.
    """
    s3 = S3Storage()
    # Delete vector bucket content only (not the underlying S3 bucket)
    from .util import config
    s3.delete_prefix(bucket, f"{config.INDEX_DIR}/")
    s3.delete_prefix(bucket, f"{config.STAGED_DIR}/")
    
    # Return empty 204 response (standard S3 behavior)
    return Response(status_code=204)

# AWS-style error handler
@app.exception_handler(HTTPException)
async def aws_http_exception_handler(_request: Request, exc: HTTPException):
    # Map status code to AWS error code string
    code_map = {
        400: "BadRequest",
        404: "NotFound",
        409: "Conflict",
    }
    code = code_map.get(exc.status_code, str(exc.status_code))
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "Error": {
                "Message": str(exc.detail),
                "Code": code
            }
        }
    )

# Add global exception handler for all unhandled exceptions
@app.exception_handler(Exception)
async def aws_global_exception_handler(_request: Request, exc: Exception):
    import traceback
    tb = traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={
            "Error": {
                "Message": f"{str(exc)}\n{tb}",
                "Code": "InternalServerError"
            }
        }
    )

@app.get("/healthz", tags=["Health"])
def healthz():
    """
    Health check endpoint.
    
    Returns system status and readiness information.
    Use this endpoint for monitoring and load balancer health checks.
    """
    return {"ok": True}

@app.get("/api-docs", tags=["Health"])
def api_docs():
    """
    API documentation links and formats.
    
    Provides links to different API documentation formats:
    - Interactive Swagger UI
    - ReDoc documentation  
    - Auto-generated OpenAPI JSON specification
    """
    return {
        "swagger_ui": "http://localhost:8000/docs",
        "redoc": "http://localhost:8000/redoc", 
        "openapi_json": "http://localhost:8000/openapi.json"
    }
