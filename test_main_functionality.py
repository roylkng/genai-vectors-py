#!/usr/bin/env python3
"""
Test to verify the main S3 Vectors functionality after cleanup and fixes.
"""

import sys
import os
sys.path.append('/home/rajan/Desktop/work/genai-vectors-py/src')

import json
import time
import numpy as np
from app.s3vectors_client import create_s3vectors_client
from app.index_builder import build_index_if_needed

def get_text_embedding(text):
    """Generate text embedding using a simple random generator (fallback)."""
    # In a real implementation, this would call an embedding service
    # For testing purposes, we'll just return a normalized random vector of size 768
    vector = np.random.rand(768) - 0.5  # Center around 0
    # Normalize to unit vector
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector.tolist()

def test_main_functionality():
    """Test the main S3 Vectors functionality."""
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
    bucket_name = f"test-main-func-{int(time.time())}"
    index_name = "test-index"
    
    print(f"🏗️ Creating bucket: {bucket_name}")
    try:
        response = s3vectors_client.create_vector_bucket(
            vectorBucketName=bucket_name
        )
        print(f"✅ Created bucket: {bucket_name}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"📦 Using existing bucket: {bucket_name}")
        else:
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
        if "already exists" in str(e).lower():
            print(f"📊 Using existing index: {index_name}")
        else:
            print(f"❌ Error creating index: {e}")
            raise
    
    print("✅ Setup complete")
    
    # Insert sample documents
    documents = [
        {
            "key": "doc1",
            "text": "Python is a high-level programming language with dynamic semantics.",
            "metadata": {"category": "programming", "topic": "python"}
        },
        {
            "key": "doc2", 
            "text": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"category": "AI", "topic": "machine_learning"}
        }
    ]
    
    print("📦 Inserting sample documents...")
    vectors = []
    for doc in documents:
        embedding = get_text_embedding(doc['text'])
        vectors.append({
            "key": doc['key'],
            "data": {"float32": embedding},
            "metadata": doc['metadata']
        })
    
    response = s3vectors_client.put_vectors(
        vectorBucketName=bucket_name,
        indexName=index_name,
        vectors=vectors
    )
    print(f"✅ Inserted {len(vectors)} documents")
    
    # Test semantic search
    test_query = "What is artificial intelligence?"
    print(f"\n🔍 Testing semantic search with query: {test_query}")
    
    query_embedding = get_text_embedding(test_query)
    
    try:
        results = s3vectors_client.query_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            queryVector={"float32": query_embedding},
            topK=3,
            returnMetadata=True
        )
        
        vectors_result = results.get('vectors', [])
        print(f"  Found {len(vectors_result)} results:")
        
        for j, result in enumerate(vectors_result, 1):
            key = result.get('key', 'Unknown')
            metadata = result.get('metadata', {})
            distance = result.get('distance', 0.0)
            similarity = 1 - distance  # Convert distance to similarity
            category = metadata.get('category', 'N/A')
            topic = metadata.get('topic', 'N/A')
            print(f"    {j}. {key} (similarity: {similarity:.3f}, category: {category}, topic: {topic})")
        
    except Exception as e:
        print(f"  ❌ Error in search: {e}")
        import traceback
        traceback.print_exc()
    
    # Test index building
    print("\n🔧 Testing index building...")
    try:
        index_result = build_index_if_needed(bucket_name, index_name)
        print(f"  Index build result: {index_result}")
    except Exception as e:
        print(f"  ❌ Error building index: {e}")
    
    print("\n✅ Main functionality test completed!")

if __name__ == "__main__":
    test_main_functionality()