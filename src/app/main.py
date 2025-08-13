from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from .api import router
from .storage.s3_backend import S3Storage
from datetime import datetime

app = FastAPI(title="s3vec-py")
app.include_router(router)

# S3-compatible endpoints for boto3 client
@app.get("/")
async def s3_list_buckets():
    """S3-compatible ListBuckets endpoint - maps to S3 Vectors ListVectorBuckets"""
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

@app.put("/{bucket}")
async def s3_create_bucket(bucket: str):
    """S3-compatible CreateBucket endpoint - maps to S3 Vectors CreateVectorBucket"""
    s3 = S3Storage()
    s3.ensure_bucket(bucket)
    
    # Return S3-compatible XML response
    xml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<CreateBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
    <Location>/{bucket}</Location>
</CreateBucketResult>"""
    
    return Response(content=xml_response, media_type="application/xml")

@app.delete("/{bucket}")
async def s3_delete_bucket(bucket: str):
    """S3-compatible DeleteBucket endpoint - maps to S3 Vectors DeleteVectorBucket"""
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

@app.get("/healthz")
def healthz():
    return {"ok": True}
