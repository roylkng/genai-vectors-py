#!/usr/bin/env python3
"""
Debug script to test the S3 Vectors API step by step
"""

import sys
import os
import time
import requests
import numpy as np

# Add the src directory to the path
sys.path.append('/home/rajan/Desktop/work/genai-vectors-py/src')

from app.s3vectors_client import create_s3vectors_client

def get_text_embedding(text):
    """Generate text embedding using a simple random generator (fallback)."""
    # In a real implementation, this would call an embedding service
    # For testing purposes, we'll just return a random vector of size 768
    return np.random.rand(768).tolist()

def debug_request(client, method, endpoint, data=None):
    """Debug HTTP requests to see what's happening."""
    print(f"DEBUG: Making {method} request to {endpoint}")
    if data:
        print(f"DEBUG: Request data: {data}")
    
    try:
        response = client._make_request(method, endpoint, data)
        print(f"DEBUG: Response: {response}")
        return response
    except Exception as e:
        print(f"DEBUG: Request failed with error: {e}")
        return None

def main():
    """Debug the S3 Vectors API step by step."""
    print("🔧 Setting up S3 Vectors client...")
    
    # Connect to S3 Vectors using our custom client
    s3vectors_client = create_s3vectors_client(
        endpoint_url='http://localhost:8000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123',
        region_name='us-east-1'
    )
    
    print("✅ S3 Vectors client ready")
    
    # Create bucket and index
    bucket_name = f"debug-demo-{int(time.time())}"
    index_name = "debug-index"
    
    print(f"🏗️ Creating bucket: {bucket_name}")
    response = debug_request(s3vectors_client, "PUT", f"/buckets/{bucket_name}", 
                            {"vectorBucketName": bucket_name})
    
    print(f"🏗️ Creating index: {index_name}")
    response = debug_request(s3vectors_client, "POST", 
                            f"/buckets/{bucket_name}/indexes/{index_name}",
                            {
                                "vectorBucketName": bucket_name,
                                "indexName": index_name,
                                "dimension": 768,
                                "dataType": "float32",
                                "distanceMetric": "cosine"
                            })
    
    # Insert sample document
    print("📦 Inserting sample document...")
    embedding = get_text_embedding("This is a test document")
    response = debug_request(s3vectors_client, "POST",
                            f"/buckets/{bucket_name}/indexes/{index_name}/vectors",
                            {
                                "vectorBucketName": bucket_name,
                                "indexName": index_name,
                                "vectors": [
                                    {
                                        "key": "test-doc",
                                        "data": {"float32": embedding},
                                        "metadata": {"category": "test"}
                                    }
                                ]
                            })
    
    # Test query
    print("🔍 Testing query...")
    query_embedding = get_text_embedding("test query")
    response = debug_request(s3vectors_client, "POST",
                            f"/buckets/{bucket_name}/indexes/{index_name}/query",
                            {
                                "vectorBucketName": bucket_name,
                                "indexName": index_name,
                                "queryVector": {"float32": query_embedding},
                                "topK": 3,
                                "returnMetadata": True
                            })
    
    if response:
        print(f"✅ Query response: {response}")
        vectors = response.get('vectors', [])
        print(f"✅ Found {len(vectors)} vectors")
        for i, vector in enumerate(vectors):
            print(f"  Vector {i+1}: {vector}")

if __name__ == "__main__":
    main()