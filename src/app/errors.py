"""AWS-compatible error responses for S3 Vectors API."""

from fastapi import HTTPException
from typing import Optional, Dict, Any
from pydantic import BaseModel


class AWSError(BaseModel):
    """AWS error response format"""
    Code: str
    Message: str
    Resource: Optional[str] = None
    RequestId: Optional[str] = None


class AWSErrorResponse(BaseModel):
    """AWS error wrapper"""
    Error: AWSError


# AWS S3 Vectors specific exceptions
class S3VectorsException(HTTPException):
    """Base exception for S3 Vectors API"""
    
    def __init__(self, code: str, message: str, resource: Optional[str] = None, status_code: int = 400):
        self.aws_code = code
        self.aws_message = message
        self.aws_resource = resource
        
        super().__init__(
            status_code=status_code,
            detail={
                "Error": {
                    "Code": code,
                    "Message": message,
                    "Resource": resource
                }
            }
        )


class ValidationException(S3VectorsException):
    """Invalid request parameters"""
    
    def __init__(self, message: str, resource: Optional[str] = None):
        super().__init__("ValidationException", message, resource, 400)


class ResourceNotFoundException(S3VectorsException):
    """Resource not found"""
    
    def __init__(self, resource_type: str, resource_name: str):
        message = f"{resource_type} '{resource_name}' does not exist"
        super().__init__("ResourceNotFoundException", message, resource_name, 404)


class ConflictException(S3VectorsException):
    """Resource already exists"""
    
    def __init__(self, resource_type: str, resource_name: str):
        message = f"{resource_type} '{resource_name}' already exists"
        super().__init__("ConflictException", message, resource_name, 409)


class AccessDeniedException(S3VectorsException):
    """Access denied"""
    
    def __init__(self, action: str, resource: Optional[str] = None):
        message = f"Access denied for action '{action}'"
        if resource:
            message += f" on resource '{resource}'"
        super().__init__("AccessDeniedException", message, resource, 403)


class ThrottlingException(S3VectorsException):
    """Rate limiting"""
    
    def __init__(self, message: str = "Request rate exceeded"):
        super().__init__("ThrottlingException", message, None, 429)


class ServiceUnavailableException(S3VectorsException):
    """Service temporarily unavailable"""
    
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__("ServiceUnavailableException", message, None, 503)


class InternalServiceException(S3VectorsException):
    """Internal service error"""
    
    def __init__(self, message: str = "Internal service error"):
        super().__init__("InternalServiceException", message, None, 500)


# Validation helpers
def validate_dimension(dimension: int) -> None:
    """Validate vector dimension within AWS limits"""
    if dimension < 1 or dimension > 4096:
        raise ValidationException(
            f"Vector dimension must be between 1 and 4096, got {dimension}"
        )


def validate_metadata_size(metadata: Dict[str, Any]) -> None:
    """Validate metadata size within AWS limits"""
    if not metadata:
        return
        
    # Convert to JSON string to check byte size
    import json
    metadata_json = json.dumps(metadata)
    size_bytes = len(metadata_json.encode('utf-8'))
    
    if size_bytes > 8192:  # 8KB limit
        raise ValidationException(
            f"Metadata size exceeds 8KB limit, got {size_bytes} bytes"
        )
    
    if len(metadata) > 50:  # 50 key limit
        raise ValidationException(
            f"Metadata key count exceeds 50 limit, got {len(metadata)} keys"
        )


def validate_batch_size(vectors: list) -> None:
    """Validate batch size within AWS limits using configured maximum"""
    from .util import config
    if len(vectors) > config.MAX_BATCH:
        raise ValidationException(
            f"Batch size exceeds {config.MAX_BATCH} limit, got {len(vectors)} vectors"
        )


def validate_top_k(top_k: int) -> None:
    """Validate topK parameter using configured limit"""
    from .util import config
    if top_k < 1 or top_k > config.MAX_TOPK:
        raise ValidationException(
            f"topK must be between 1 and {config.MAX_TOPK}, got {top_k}"
        )


def validate_vector_keys(keys: list) -> None:
    """Validate vector keys"""
    if not keys:
        raise ValidationException("Vector keys cannot be empty")
    
    if len(keys) > 100:  # AWS limit for get_vectors
        raise ValidationException(
            f"Cannot request more than 100 keys at once, got {len(keys)}"
        )
    
    for key in keys:
        if not isinstance(key, str):
            raise ValidationException(f"Vector key must be string, got {type(key)}")
        
        if len(key) > 512:  # Reasonable key length limit
            raise ValidationException("Vector key exceeds 512 character limit")


def validate_index_name(name: str) -> None:
    """Validate index name format"""
    if not name:
        raise ValidationException("Index name cannot be empty")
    
    if len(name) > 255:
        raise ValidationException("Index name exceeds 255 character limit")
    
    # AWS naming pattern: alphanumeric, hyphens, underscores
    import re
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        raise ValidationException(
            "Index name must contain only alphanumeric characters, hyphens, and underscores"
        )


def validate_bucket_name(name: str) -> None:
    """Validate bucket name format"""
    if not name:
        raise ValidationException("Bucket name cannot be empty")
    
    if len(name) < 3 or len(name) > 63:
        raise ValidationException("Bucket name must be between 3 and 63 characters")
    
    # AWS S3 bucket naming rules (simplified)
    import re
    if not re.match(r'^[a-z0-9.-]+$', name):
        raise ValidationException(
            "Bucket name must contain only lowercase letters, numbers, dots, and hyphens"
        )
    
    if name.startswith('.') or name.endswith('.'):
        raise ValidationException("Bucket name cannot start or end with a dot")
    
    if name.startswith('-') or name.endswith('-'):
        raise ValidationException("Bucket name cannot start or end with a hyphen")
