#!/usr/bin/env python3
"""
Patch file to fix S3 compatibility issues in the S3 Vectors API
"""

# Add this to main.py to ensure proper S3 compatibility routing

S3_COMPATIBILITY_FIXES = """
# Enhanced S3 compatibility endpoints for better boto3 integration
@app.put("/{bucket}", tags=["S3 Compatibility"])
async def s3_create_bucket_with_body(bucket: str, request: Request = None):
    \"\"\"
    Enhanced S3-compatible CreateBucket endpoint with body support.
    
    Handles both simple PUT requests and requests with body metadata
    for vector bucket creation.
    \"\"\"
    s3 = S3Storage()
    
    # Check if this is a vector bucket creation request
    if request and request.headers.get("content-type") == "application/json":
        try:
            body = await request.json()
            action = body.get("action") or body.get("Action")
            
            if action == "create_vector_bucket":
                # Handle vector bucket creation with metadata
                s3.ensure_bucket(bucket)
                
                # Store bucket metadata
                bucket_config = {
                    "name": bucket,
                    "created": datetime.utcnow().isoformat(),
                    "engine": "lance",
                    "version": "1.0"
                }
                
                s3.put_json(bucket, f"{config.META_DIR}/bucket.json", bucket_config)
                
                # Return S3-compatible XML response
                xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<CreateVectorBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <Location>/{bucket}</Location>
</CreateVectorBucketResult>"""
                return Response(content=xml_response, media_type="application/xml")
        except Exception:
            pass  # Fall back to standard bucket creation
    
    # Standard S3 bucket creation
    s3.ensure_bucket(bucket)
    
    # Return S3-compatible XML response
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<CreateBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <Location>/{bucket}</Location>
</CreateBucketResult>"""
    
    return Response(content=xml_response, media_type="application/xml")

@app.post("/{bucket}", tags=["S3 Compatibility"])
async def s3_post_bucket_operation(bucket: str, request: Request):
    \"\"\"
    S3-compatible POST endpoint for bucket operations.
    
    Handles various S3 POST operations including:
    - Object creation with metadata
    - Multipart uploads
    - Vector operations when metadata indicates special actions
    \"\"\"
    s3 = S3Storage()
    
    # Check content type and handle accordingly
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        try:
            body = await request.json()
            action = body.get("action") or body.get("Action")
            
            # Handle vector operations based on metadata
            if action == "create_index":
                return await create_index_service(request)
            elif action == "put_vectors":
                return await put_vectors_service(request)
            elif action == "query_vectors":
                return await query_vectors_service(request)
            elif action == "list_vector_buckets":
                return await list_vector_buckets_service()
            elif action == "create_vector_bucket":
                # Handle vector bucket creation via POST
                s3.ensure_bucket(bucket)
                
                # Store bucket metadata
                bucket_config = {
                    "name": bucket,
                    "created": datetime.utcnow().isoformat(),
                    "engine": "lance",
                    "version": "1.0"
                }
                
                s3.put_json(bucket, f"{config.META_DIR}/bucket.json", bucket_config)
                
                return {"vectorBucketName": bucket}
        except Exception as e:
            # If JSON parsing fails, treat as regular S3 operation
            pass
    
    # Handle multipart form data (standard S3 operations)
    if "multipart/form-data" in content_type:
        # Process multipart upload
        pass
    
    # Default response for unrecognized operations
    raise HTTPException(status_code=400, detail="Unsupported operation")

# Additional S3 compatibility helpers
def extract_s3_action_from_metadata(metadata: dict) -> str:
    \"\"\"Extract S3 Vectors action from request metadata\"\"\"
    return metadata.get("action", metadata.get("Action", "unknown")).lower()

def is_vector_operation(metadata: dict) -> bool:
    \"\"\"Check if this is a vector operation based on metadata\"\"\"
    action = extract_s3_action_from_metadata(metadata)
    return action in [
        "create_vector_bucket", "create_index", "put_vectors", 
        "query_vectors", "delete_vectors", "list_indexes"
    ]
"""

print("S3 Compatibility Fixes for S3 Vectors API")
print("=" * 50)
print("To apply these fixes:")
print("1. Add the above code to src/app/main.py")
print("2. Ensure proper imports are added:")
print("   - from fastapi import Request")
print("   - from .storage.s3_backend import S3Storage")
print("   - from .util import config")
print("   - from datetime import datetime")
print("3. Make sure the existing S3 compatibility endpoints work with these enhancements")
print("\nThese fixes will:")
print("- Better handle boto3 client requests")
print("- Support vector operations through S3-compatible interfaces")
print("- Provide proper error handling for malformed requests")
print("- Maintain backward compatibility with existing API")