"""
Custom S3 Vectors client that mimics boto3 interface but calls our HTTP API.
"""
import requests
import json
from typing import List, Dict, Any, Optional

class S3VectorsClient:
    """Custom client that implements the S3 Vectors API interface"""
    
    def __init__(self, endpoint_url: str, aws_access_key_id: str = None, 
                 aws_secret_access_key: str = None, region_name: str = None):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.access_key = aws_access_key_id
        self.secret_key = aws_secret_access_key
        self.region = region_name
        
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request to the API"""
        url = f"{self.endpoint_url}/{endpoint.lstrip('/')}"
        
        headers = {"Content-Type": "application/json"}
        
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, 
                                       json=data, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, 
                                      json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            
            # Handle both JSON and empty responses
            if response.content:
                return response.json()
            else:
                return {"success": True}
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
    
    def create_vector_bucket(self, vectorBucketName: str, **kwargs) -> Dict:
        """Create a new vector bucket"""
        data = {"vectorBucketName": vectorBucketName}
        return self._make_request("POST", "/CreateVectorBucket", data)
    
    def list_vector_buckets(self, **kwargs) -> Dict:
        """List all vector buckets"""
        return self._make_request("POST", "/ListVectorBuckets")
    
    def create_index(self, vectorBucketName: str, indexName: str, 
                    dimension: int, dataType: str = "float32", 
                    distanceMetric: str = "cosine", **kwargs) -> Dict:
        """Create a vector index"""
        data = {
            "vectorBucketName": vectorBucketName,
            "indexName": indexName,
            "dimension": dimension,
            "dataType": dataType,
            "distanceMetric": distanceMetric
        }
        return self._make_request("POST", "/CreateIndex", data)
    
    def list_indexes(self, vectorBucketName: str, **kwargs) -> Dict:
        """List indexes in a bucket"""
        data = {"vectorBucketName": vectorBucketName}
        return self._make_request("POST", "/ListIndexes", data)
    
    def put_vectors(self, vectorBucketName: str, indexName: str, 
                   vectors: List[Dict], **kwargs) -> Dict:
        """Insert or update vectors"""
        data = {
            "vectorBucketName": vectorBucketName,
            "indexName": indexName,
            "vectors": vectors
        }
        return self._make_request("POST", "/PutVectors", data)
    
    def query_vectors(self, vectorBucketName: str, indexName: str,
                     queryVector: Dict, topK: int = 10,
                     returnMetadata: bool = True, 
                     returnDistance: bool = True,
                     filter: Dict = None, **kwargs) -> Dict:
        """Query vectors for similarity search"""
        data = {
            "vectorBucketName": vectorBucketName,
            "indexName": indexName,
            "queryVector": queryVector,
            "topK": topK,
            "returnMetadata": returnMetadata,
            "returnDistance": returnDistance
        }
        if filter:
            data["filter"] = filter
            
        return self._make_request("POST", "/QueryVectors", data)

def create_s3vectors_client(endpoint_url: str, aws_access_key_id: str = None,
                           aws_secret_access_key: str = None, 
                           region_name: str = None) -> S3VectorsClient:
    """Factory function to create S3 Vectors client"""
    return S3VectorsClient(
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id, 
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name
    )
