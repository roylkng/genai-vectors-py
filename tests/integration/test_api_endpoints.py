"""
Integration tests for the S3 Vectors API endpoints.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from app.main import app
from fastapi.testclient import TestClient
from app.util import config


class TestAPIEndpoints:
    """Integration tests for S3 Vectors API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "implementation" in data

    def test_healthz_endpoint(self, client):
        """Test the healthz endpoint."""
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True

    @patch('app.api.S3Storage')
    @patch('app.api.connect_bucket')
    def test_create_vector_bucket(self, mock_connect_bucket, mock_s3_storage, client):
        """Test creating a vector bucket."""
        # Mock S3 storage
        mock_s3 = Mock()
        mock_s3.bucket_exists.return_value = False
        mock_s3_storage.return_value = mock_s3
        
        # Mock LanceDB
        mock_db = Mock()
        mock_connect_bucket.return_value = mock_db
        
        # Test create bucket
        response = client.put("/buckets/test-bucket", json={
            "vectorBucketName": "test-bucket"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "vectorBucketName" in data

    @patch('app.api.S3Storage')
    def test_list_vector_buckets(self, mock_s3_storage, client):
        """Test listing vector buckets."""
        # Mock S3 storage
        mock_s3 = Mock()
        mock_s3.list_buckets.return_value = [
            {"Name": f"{config.S3_BUCKET_PREFIX}bucket1"},
            {"Name": f"{config.S3_BUCKET_PREFIX}bucket2"}
        ]
        mock_s3_storage.return_value = mock_s3
        
        # Test list buckets
        response = client.get("/buckets")
        
        assert response.status_code == 200
        data = response.json()
        assert "vectorBuckets" in data
        assert len(data["vectorBuckets"]) == 2

    @patch('app.api.S3Storage')
    @patch('app.api.connect_bucket')
    def test_create_index(self, mock_connect_bucket, mock_s3_storage, client):
        """Test creating an index."""
        # Mock S3 storage
        mock_s3 = Mock()
        mock_s3.bucket_exists.return_value = True
        mock_s3_storage.return_value = mock_s3
        
        # Mock LanceDB
        mock_db = Mock()
        mock_connect_bucket.return_value = mock_db
        
        # Test create index
        response = client.post("/buckets/test-bucket/indexes/test-index", json={
            "vectorBucketName": "test-bucket",
            "indexName": "test-index",
            "dimension": 128,
            "dataType": "float32",
            "distanceMetric": "cosine"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "test-index"

    @patch('app.api.S3Storage')
    @patch('app.api.connect_bucket')
    def test_put_vectors(self, mock_connect_bucket, mock_s3_storage, client):
        """Test putting vectors."""
        # Mock S3 storage
        mock_s3 = Mock()
        mock_s3.bucket_exists.return_value = True
        mock_s3_storage.return_value = mock_s3
        
        # Mock LanceDB
        mock_db = Mock()
        mock_connect_bucket.return_value = mock_db
        
        # Test put vectors
        vectors = [
            {
                "key": "doc1",
                "data": {"float32": [0.1] * 128},
                "metadata": {"category": "test"}
            }
        ]
        
        response = client.post("/buckets/test-bucket/indexes/test-index/vectors", json={
            "vectorBucketName": "test-bucket",
            "indexName": "test-index",
            "vectors": vectors
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data == {}

    @patch('app.api.S3Storage')
    @patch('app.api.connect_bucket')
    def test_query_vectors(self, mock_connect_bucket, mock_s3_storage, client):
        """Test querying vectors."""
        # Mock S3 storage
        mock_s3 = Mock()
        mock_s3.bucket_exists.return_value = True
        mock_s3_storage.return_value = mock_s3
        
        # Mock LanceDB with search results
        mock_db = Mock()
        mock_table = Mock()
        mock_db.open_table.return_value = mock_table
        mock_connect_bucket.return_value = mock_db
        
        # Mock search results
        mock_results = Mock()
        mock_results.to_pandas.return_value = Mock()
        mock_table.search.return_value.limit.return_value.nprobes.return_value.refine_factor.return_value.to_pandas.return_value = Mock()
        
        # Test query vectors
        response = client.post("/buckets/test-bucket/indexes/test-index/query", json={
            "vectorBucketName": "test-bucket",
            "indexName": "test-index",
            "queryVector": {"float32": [0.1] * 128},
            "topK": 10,
            "returnMetadata": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "vectors" in data

    @patch('app.api.S3Storage')
    @patch('app.api.connect_bucket')
    def test_list_vectors_pagination(self, mock_connect_bucket, mock_s3_storage, client):
        """Test listing vectors with pagination."""
        # Mock S3 storage
        mock_s3 = Mock()
        mock_s3.bucket_exists.return_value = True
        mock_s3_storage.return_value = mock_s3
        
        # Mock LanceDB
        mock_db = Mock()
        mock_table = Mock()
        mock_db.open_table.return_value = mock_table
        mock_connect_bucket.return_value = mock_db
        
        # Mock list results
        mock_table.search.return_value.where.return_value.limit.return_value.to_pandas.return_value = Mock()
        
        # Test list vectors
        response = client.post("/buckets/test-bucket/indexes/test-index/vectors:list", json={
            "vectorBucketName": "test-bucket",
            "indexName": "test-index",
            "maxResults": 50,
            "nextToken": None
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "vectors" in data
        assert "nextToken" in data

    def test_openapi_spec(self, client):
        """Test that OpenAPI spec is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data

    def test_docs_endpoints(self, client):
        """Test that docs endpoints are available."""
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])