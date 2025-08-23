#!/usr/bin/env python3
"""
Example script showing how to properly use boto3 client with the S3 Vectors API
"""

import boto3
import json
import numpy as np

def create_boto3_client():
    """Create a boto3 client that works with our S3 Vectors API"""
    return boto3.client(
        's3',  # This is correct - we're implementing the S3 interface
        endpoint_url='http://localhost:8000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123',
        region_name='us-east-1'
    )

def test_boto3_integration():
    """Test the boto3 client integration"""
    # Create client
    client = create_boto3_client()
    
    # Create a bucket
    try:
        # Using the S3-compatible endpoint (this should work)
        response = client.create_bucket(Bucket='test-bucket')
        print("✅ Bucket created successfully")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error creating bucket: {e}")
        return
    
    # List buckets
    try:
        response = client.list_buckets()
        print("\n✅ Buckets listed successfully")
        print(f"Buckets: {response.get('Buckets', [])}")
    except Exception as e:
        print(f"❌ Error listing buckets: {e}")
    
    # Put an object (this is how we'll create indexes and put vectors)
    try:
        # Create a vector bucket using the S3 Vectors pattern
        response = client.put_object(
            Bucket='test-bucket',
            Key='',
            Body=b'',
            Metadata={'action': 'create_vector_bucket'}
        )
        print("\n✅ Vector bucket created via S3 interface")
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error creating vector bucket: {e}")

def test_direct_api():
    """Test using the direct API endpoints"""
    import requests
    
    base_url = "http://localhost:8000"
    
    # Create a vector bucket using direct API
    try:
        response = requests.put(
            f"{base_url}/buckets/test-bucket-direct",
            json={"vectorBucketName": "test-bucket-direct"}
        )
        response.raise_for_status()
        print("✅ Direct API bucket created successfully")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error creating bucket via direct API: {e}")

if __name__ == "__main__":
    print("Testing S3 Vectors API with boto3 client...")
    print("=" * 50)
    
    test_boto3_integration()
    
    print("\n" + "=" * 50)
    print("Testing direct API endpoints...")
    print("=" * 50)
    
    test_direct_api()