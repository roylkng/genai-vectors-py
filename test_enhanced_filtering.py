#!/usr/bin/env python3
"""Test enhanced filtering functionality"""

import asyncio
import sys
import os
sys.path.append('src')

from app.api import create_vector_bucket, create_index, put_vectors, query_vectors
from app.models import (
    CreateVectorBucketRequest, CreateIndexRequest, PutVectorsRequest, QueryVectorsRequest,
    VectorData, MetadataFilter, FilterCondition, LogicalFilter
)

async def test_enhanced_filtering():
    """Test the new enhanced filtering capabilities"""
    
    bucket_name = 'test-filter-bucket'
    index_name = 'filter-test-index'
    
    try:
        print("=== Testing Enhanced Filtering ===")
        
        # Step 0: Create bucket
        print("0. Creating bucket...")
        bucket_request = CreateVectorBucketRequest(vectorBucketName=bucket_name)
        await create_vector_bucket(bucket_name, bucket_request)
        print("✅ Bucket created")
        
        # Step 1: Create index
        print("1. Creating index...")
        index_request = CreateIndexRequest(
            indexName=index_name,
            dataType='float32',
            dimension=3,
            distanceMetric='cosine'
        )
        await create_index(bucket_name, index_name, index_request)
        print("✅ Index created")
        
        # Step 2: Insert test vectors with various metadata
        print("2. Inserting test vectors...")
        vectors_data = {
            'vectors': [
                {
                    'key': 'doc-1',
                    'data': {'float32': [0.1, 0.2, 0.3]},
                    'metadata': {'category': 'tech', 'score': 85, 'published': True, 'tags': ['ai', 'ml']}
                },
                {
                    'key': 'doc-2', 
                    'data': {'float32': [0.4, 0.5, 0.6]},
                    'metadata': {'category': 'science', 'score': 92, 'published': True, 'tags': ['research']}
                },
                {
                    'key': 'doc-3',
                    'data': {'float32': [0.7, 0.8, 0.9]},
                    'metadata': {'category': 'tech', 'score': 78, 'published': False, 'tags': ['web', 'frontend']}
                },
                {
                    'key': 'doc-4',
                    'data': {'float32': [0.2, 0.4, 0.1]},
                    'metadata': {'category': 'business', 'score': 95, 'published': True, 'tags': ['strategy']}
                }
            ]
        }
        put_request = PutVectorsRequest(**vectors_data)
        await put_vectors(bucket_name, index_name, put_request)
        print("✅ Vectors inserted")
        
        # Step 3: Test various filter types
        
        # Test 1: Simple equality filter
        print("\n3. Testing simple equality filter (category = 'tech')...")
        filter_condition = FilterCondition(
            metadata_key='category',
            operator='equals',
            value='tech'
        )
        query_request = QueryVectorsRequest(
            queryVector=VectorData(float32=[0.1, 0.2, 0.3]),
            topK=10,
            filter=MetadataFilter(root=filter_condition)
        )
        result = await query_vectors(bucket_name, index_name, query_request)
        print(f"Found {len(result.vectors)} vectors with category='tech'")
        
        # Test 2: Numeric comparison filter
        print("\n4. Testing numeric filter (score > 80)...")
        filter_condition = FilterCondition(
            metadata_key='score',
            operator='greater_than',
            value=80
        )
        query_request = QueryVectorsRequest(
            queryVector=VectorData(float32=[0.1, 0.2, 0.3]),
            topK=10,
            filter=MetadataFilter(root=filter_condition)
        )
        result = await query_vectors(bucket_name, index_name, query_request)
        print(f"Found {len(result.vectors)} vectors with score > 80")
        
        # Test 3: IN operator
        print("\n5. Testing IN filter (category in ['tech', 'science'])...")
        filter_condition = FilterCondition(
            metadata_key='category',
            operator='in',
            value=['tech', 'science']
        )
        query_request = QueryVectorsRequest(
            queryVector=VectorData(float32=[0.1, 0.2, 0.3]),
            topK=10,
            filter=MetadataFilter(root=filter_condition)
        )
        result = await query_vectors(bucket_name, index_name, query_request)
        print(f"Found {len(result.vectors)} vectors with category in ['tech', 'science']")
        
        # Test 4: Boolean filter
        print("\n6. Testing boolean filter (published = true)...")
        filter_condition = FilterCondition(
            metadata_key='published',
            operator='equals',
            value=True
        )
        query_request = QueryVectorsRequest(
            queryVector=VectorData(float32=[0.1, 0.2, 0.3]),
            topK=10,
            filter=MetadataFilter(root=filter_condition)
        )
        result = await query_vectors(bucket_name, index_name, query_request)
        print(f"Found {len(result.vectors)} vectors with published=true")
        
        # Test 5: AND logical filter
        print("\n7. Testing AND filter (category='tech' AND score > 80)...")
        and_filter = LogicalFilter(
            operator='and',
            conditions=[
                FilterCondition(metadata_key='category', operator='equals', value='tech'),
                FilterCondition(metadata_key='score', operator='greater_than', value=80)
            ]
        )
        query_request = QueryVectorsRequest(
            queryVector=VectorData(float32=[0.1, 0.2, 0.3]),
            topK=10,
            filter=MetadataFilter(root=and_filter)
        )
        result = await query_vectors(bucket_name, index_name, query_request)
        print(f"Found {len(result.vectors)} vectors with category='tech' AND score > 80")
        
        # Test 6: OR logical filter
        print("\n8. Testing OR filter (score > 90 OR published = false)...")
        or_filter = LogicalFilter(
            operator='or',
            conditions=[
                FilterCondition(metadata_key='score', operator='greater_than', value=90),
                FilterCondition(metadata_key='published', operator='equals', value=False)
            ]
        )
        query_request = QueryVectorsRequest(
            queryVector=VectorData(float32=[0.1, 0.2, 0.3]),
            topK=10,
            filter=MetadataFilter(root=or_filter)
        )
        result = await query_vectors(bucket_name, index_name, query_request)
        print(f"Found {len(result.vectors)} vectors with score > 90 OR published = false")
        
        print("\n✅ All enhanced filtering tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_filtering())
