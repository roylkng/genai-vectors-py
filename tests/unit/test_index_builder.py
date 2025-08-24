"""
Unit tests for the index builder module.
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app.index_builder import build_index_if_needed


class TestIndexBuilder:
    """Test cases for the index builder module."""

    def test_build_index_if_needed_success_ivf_pq(self, mock_s3_storage):
        """Test successful IVF_PQ index building."""
        # Mock S3 storage response
        mock_s3_storage.get_text.return_value = '{"dimension": 128, "indexType": "IVF_PQ", "distanceMetric": "cosine"}'
        
        # Mock LanceDB table
        mock_table = Mock()
        mock_table.list_indices.return_value = []  # No existing indices
        mock_table.count_rows.return_value = 50000  # Large enough for IVF_PQ
        
        with patch('app.lance.db.connect_bucket') as mock_connect:
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db
            
            result = build_index_if_needed("test-bucket", "test-index")
            
            assert result["status"] == "READY"
            assert result["indexType"] == "IVF_PQ"
            mock_table.create_index.assert_called_once()

    def test_build_index_if_needed_success_hnsw(self, mock_s3_storage):
        """Test successful HNSW index building."""
        # Mock S3 storage response
        mock_s3_storage.get_text.return_value = '{"dimension": 128, "indexType": "HNSW", "distanceMetric": "cosine"}'
        
        # Mock LanceDB table
        mock_table = Mock()
        mock_table.list_indices.return_value = []  # No existing indices
        mock_table.count_rows.return_value = 1000  # Small enough for HNSW
        
        with patch('app.lance.db.connect_bucket') as mock_connect:
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db
            
            result = build_index_if_needed("test-bucket", "test-index")
            
            assert result["status"] == "READY"
            assert result["indexType"] == "HNSW"
            mock_table.create_index.assert_called_once()

    def test_build_index_if_needed_auto_heuristic_ivf_pq(self, mock_s3_storage):
        """Test AUTO index type selects IVF_PQ for large datasets."""
        # Mock S3 storage response
        mock_s3_storage.get_text.return_value = '{"dimension": 256, "indexType": "AUTO", "distanceMetric": "cosine"}'
        
        # Mock LanceDB table
        mock_table = Mock()
        mock_table.list_indices.return_value = []  # No existing indices
        mock_table.count_rows.return_value = 300000  # Large enough for IVF_PQ
        
        with patch('app.lance.db.connect_bucket') as mock_connect:
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db
            
            result = build_index_if_needed("test-bucket", "test-index")
            
            assert result["status"] == "READY"
            assert result["indexType"] == "IVF_PQ"

    def test_build_index_if_needed_auto_heuristic_hnsw(self, mock_s3_storage):
        """Test AUTO index type selects HNSW for small datasets."""
        # Mock S3 storage response
        mock_s3_storage.get_text.return_value = '{"dimension": 128, "indexType": "AUTO", "distanceMetric": "cosine"}'
        
        # Mock LanceDB table
        mock_table = Mock()
        mock_table.list_indices.return_value = []  # No existing indices
        mock_table.count_rows.return_value = 1000  # Small enough for HNSW
        
        with patch('app.lance.db.connect_bucket') as mock_connect:
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db
            
            result = build_index_if_needed("test-bucket", "test-index")
            
            assert result["status"] == "READY"
            assert result["indexType"] == "HNSW"

    def test_build_index_if_needed_skip_existing(self, mock_s3_storage):
        """Test that existing indices are skipped."""
        # Mock S3 storage response
        mock_s3_storage.get_text.return_value = '{"dimension": 128, "indexType": "IVF_PQ", "distanceMetric": "cosine"}'
        
        # Mock LanceDB table with existing index
        mock_table = Mock()
        mock_table.list_indices.return_value = [Mock(name="vector_idx")]  # Existing index
        
        with patch('app.lance.db.connect_bucket') as mock_connect:
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db
            
            result = build_index_if_needed("test-bucket", "test-index")
            
            assert result["status"] == "READY"
            assert result["note"] == "index exists"
            mock_table.create_index.assert_not_called()

    def test_build_index_if_needed_none_disabled(self, mock_s3_storage):
        """Test that NONE index type skips indexing."""
        # Mock S3 storage response
        mock_s3_storage.get_text.return_value = '{"dimension": 128, "indexType": "NONE", "distanceMetric": "cosine"}'
        
        # Mock LanceDB table
        mock_table = Mock()
        mock_table.list_indices.return_value = []  # No existing indices
        
        with patch('app.lance.db.connect_bucket') as mock_connect:
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db
            
            result = build_index_if_needed("test-bucket", "test-index")
            
            assert result["status"] == "SKIPPED"
            assert result["note"] == "index disabled"
            mock_table.create_index.assert_not_called()

    def test_build_index_if_needed_s3_error(self, mock_s3_storage):
        """Test error handling when S3 storage fails."""
        # Mock S3 storage failure
        mock_s3_storage.get_text.side_effect = Exception("S3 connection failed")
        
        result = build_index_if_needed("test-bucket", "test-index")
        
        assert result["status"] == "ERROR"
        assert "S3 connection failed" in result["error"]

    def test_build_index_if_needed_invalid_index_type(self, mock_s3_storage):
        """Test error handling for unknown index types."""
        # Mock S3 storage response with invalid index type
        mock_s3_storage.get_text.return_value = '{"dimension": 128, "indexType": "INVALID_TYPE", "distanceMetric": "cosine"}'
        
        # Mock LanceDB table
        mock_table = Mock()
        mock_table.list_indices.return_value = []  # No existing indices
        
        with patch('app.lance.db.connect_bucket') as mock_connect:
            mock_db = Mock()
            mock_db.open_table.return_value = mock_table
            mock_connect.return_value = mock_db
            
            result = build_index_if_needed("test-bucket", "test-index")
            
            assert result["status"] == "ERROR"
            assert "Unknown indexType INVALID_TYPE" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__])