#!/usr/bin/env python3
"""
Debug script to check how metadata is being stored and retrieved
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
    # For testing purposes, we'll return a normalized random vector of size 768
    vector = np.random.rand(768) - 0.5  # Center around 0
    # Normalize to unit vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()

def main():
    """Debug how metadata is being stored and retrieved."""
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
    bucket_name = f"debug-metadata-{int(time.time())}"
    index_name = "debug-index"
    
    print(f"🏗️ Creating bucket: {bucket_name}")
    try:
        response = s3vectors_client.create_vector_bucket(
            vectorBucketName=bucket_name
        )
        print(f"✅ Created bucket: {bucket_name}")
    except Exception as e:
        print(f"❌ Error creating bucket: {e}")
        raise
    
    print(f"🏗️ Creating index: {index_name}")
    try:
        response = s3vectors_client.create_index(
            vectorBucketName=bucket_name,
            indexName=index_name,
            dimension=768,
            dataType="float32",
            distanceMetric="cosine"
        )
        print(f"✅ Created index: {index_name}")
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        raise
    
    print("✅ Setup complete")
    
    # Insert a single document with metadata
    print("📦 Inserting document with metadata...")
    embedding = get_text_embedding("This is a test document")
    vector_data = {
        "key": "test-doc",
        "data": {"float32": embedding},
        "metadata": {"category": "test", "topic": "debug"}
    }
    
    print(f"Inserting vector: {vector_data}")
    
    try:
        response = s3vectors_client.put_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            vectors=[vector_data]
        )
        print(f"✅ Inserted document")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error inserting document: {e}")
        raise
    
    # Test different query configurations
    print("\n🔍 Testing different query configurations...")
    
    query_embedding = get_text_embedding("test query")
    
    # Test 1: Query with returnMetadata=True
    print("\n--- Test 1: Query with returnMetadata=True ---")
    try:
        results = s3vectors_client.query_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            queryVector={"float32": query_embedding},
            topK=1,
            returnMetadata=True
        )
        print(f"Response: {results}")
    except Exception as e:
        print(f"❌ Error in search: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Query with returnMetadata=False
    print("\n--- Test 2: Query with returnMetadata=False ---")
    try:
        results = s3vectors_client.query_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            queryVector={"float32": query_embedding},
            topK=1,
            returnMetadata=False
        )
        print(f"Response: {results}")
    except Exception as e:
        print(f"❌ Error in search: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Query with returnMetadata not specified (should default to True)
    print("\n--- Test 3: Query with returnMetadata not specified ---")
    try:
        results = s3vectors_client.query_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            queryVector={"float32": query_embedding},
            topK=1
            # returnMetadata not specified
        )
        print(f"Response: {results}")
    except Exception as e:
        print(f"❌ Error in search: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Debug test completed!")

if __name__ == "__main__":
    main()