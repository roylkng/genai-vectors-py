#!/usr/bin/env python3
"""
Simple test to verify metadata retrieval works with typed columns
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
    # For testing purposes, we'll return a normalized random vector of size 10
    vector = np.random.rand(10) - 0.5  # Center around 0
    # Normalize to unit vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()

def main():
    """Test that metadata retrieval works with typed columns."""
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
    bucket_name = f"typed-metadata-test-{int(time.time())}"
    index_name = "test-index"
    
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
            dimension=10,  # Smaller dimension for testing
            dataType="float32",
            distanceMetric="cosine"
        )
        print(f"✅ Created index: {index_name}")
    except Exception as e:
        print(f"❌ Error creating index: {e}")
        raise
    
    print("✅ Setup complete")
    
    # Insert a document with metadata
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
    except Exception as e:
        print(f"❌ Error inserting document: {e}")
        raise
    
    # Test query with metadata retrieval
    print("\n🔍 Testing query with metadata retrieval...")
    
    query_embedding = get_text_embedding("test query")
    
    try:
        results = s3vectors_client.query_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            queryVector={"float32": query_embedding},
            topK=1,
            returnMetadata=True
        )
        
        print(f"Full response: {results}")
        
        if results and 'vectors' in results:
            vectors_result = results['vectors']
            print(f"Found {len(vectors_result)} results:")
            
            for j, result in enumerate(vectors_result, 1):
                key = result.get('key', 'Unknown')
                metadata = result.get('metadata', {})
                distance = result.get('distance', 0.0)
                similarity = 1 - distance  # Convert distance to similarity
                print(f"  {j}. {key}")
                print(f"     Similarity: {similarity:.3f}")
                print(f"     Metadata: {metadata}")
                if metadata:
                    print(f"     Category: {metadata.get('category', 'N/A')}")
                    print(f"     Topic: {metadata.get('topic', 'N/A')}")
        else:
            print("No vectors found in response")
            
    except Exception as e:
        print(f"❌ Error in search: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()