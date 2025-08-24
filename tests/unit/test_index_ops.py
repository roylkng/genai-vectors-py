"""
Unit tests for the index operations module.
"""

import pytest
import sys
import os
from unittest.mock import Mock, AsyncMock, patch
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from app.lance.index_ops import list_vectors


class TestIndexOps:
    """Test cases for the index operations module."""

    @pytest.mark.asyncio
    async def test_list_vectors_basic_pagination(self):
        """Test basic pagination with NextToken/MaxResults."""
        # Mock database and table
        mock_db = Mock()
        mock_table = Mock()
        mock_db.open_table.return_value = mock_table
        
        # Mock pandas DataFrame with sample data
        sample_data = pd.DataFrame({
            'key': ['doc1', 'doc2', 'doc3'],
            'vector': [[0.1]*128, [0.2]*128, [0.3]*128]
        })
        mock_table.search.return_value.where.return_value.limit.return_value.to_pandas.return_value = sample_data
        
        # Test basic pagination
        items, next_token = await list_vectors(mock_db, "test-table", max_results=10, next_token=None)
        
        assert len(items) == 3
        assert items[0]['key'] == 'doc1'
        assert items[1]['key'] == 'doc2'
        assert items[2]['key'] == 'doc3'
        # Should have next token since we didn't hit the limit
        assert next_token is None

    @pytest.mark.asyncio
    async def test_list_vectors_with_next_token(self):
        """Test pagination with continuation token."""
        # Mock database and table
        mock_db = Mock()
        mock_table = Mock()
        mock_db.open_table.return_value = mock_table
        
        # Mock pandas DataFrame with sample data that fills the page
        sample_keys = [f'doc{i}' for i in range(10)]
        sample_vectors = [[0.1*i]*128 for i in range(10)]
        sample_data = pd.DataFrame({
            'key': sample_keys,
            'vector': sample_vectors
        })
        mock_table.search.return_value.where.return_value.limit.return_value.to_pandas.return_value = sample_data
        
        # Test pagination with continuation
        items, next_token = await list_vectors(mock_db, "test-table", max_results=10, next_token="doc5")
        
        assert len(items) == 10
        assert next_token == 'doc9'  # Last key should be next token
        mock_table.search.return_value.where.assert_called_once()
        where_call = mock_table.search.return_value.where.call_args[0][0]
        assert "key > 'doc5'" in where_call

    @pytest.mark.asyncio
    async def test_list_vectors_sql_injection_protection(self):
        """Test SQL injection protection in continuation tokens."""
        # Mock database and table
        mock_db = Mock()
        mock_table = Mock()
        mock_db.open_table.return_value = mock_table
        
        # Test malicious continuation token
        malicious_token = "doc1'; DROP TABLE users;"
        sample_data = pd.DataFrame({
            'key': ['doc1', 'doc2'],
            'vector': [[0.1]*128, [0.2]*128]
        })
        mock_table.search.return_value.where.return_value.limit.return_value.to_pandas.return_value = sample_data
        
        # Should escape the malicious token
        items, next_token = await list_vectors(mock_db, "test-table", max_results=5, next_token=malicious_token)
        
        # Verify the escaped token was used in the query
        mock_table.search.return_value.where.assert_called_once()
        where_call = mock_table.search.return_value.where.call_args[0][0]
        assert "doc1''; DROP TABLE users;" in where_call  # Escaped single quotes

    @pytest.mark.asyncio
    async def test_list_vectors_empty_result(self):
        """Test pagination with empty result set."""
        # Mock database and table
        mock_db = Mock()
        mock_table = Mock()
        mock_db.open_table.return_value = mock_table
        
        # Mock empty DataFrame
        sample_data = pd.DataFrame(columns=['key', 'vector'])
        mock_table.search.return_value.where.return_value.limit.return_value.to_pandas.return_value = sample_data
        
        # Test empty result
        items, next_token = await list_vectors(mock_db, "test-table", max_results=10, next_token=None)
        
        assert len(items) == 0
        assert next_token is None

    @pytest.mark.asyncio
    async def test_list_vectors_exact_page_limit(self):
        """Test pagination with exactly full page."""
        # Mock database and table
        mock_db = Mock()
        mock_table = Mock()
        mock_db.open_table.return_value = mock_table
        
        # Create exactly 5 items (matching max_results)
        sample_keys = [f'doc{i}' for i in range(5)]
        sample_vectors = [[0.1*i]*128 for i in range(5)]
        sample_data = pd.DataFrame({
            'key': sample_keys,
            'vector': sample_vectors
        })
        mock_table.search.return_value.where.return_value.limit.return_value.to_pandas.return_value = sample_data
        
        # Test exact page limit
        items, next_token = await list_vectors(mock_db, "test-table", max_results=5, next_token=None)
        
        assert len(items) == 5
        # Should have next token since we filled the page exactly
        assert next_token == 'doc4'

    @pytest.mark.asyncio
    async def test_list_vectors_sorting(self):
        """Test that results are properly sorted by key."""
        # Mock database and table
        mock_db = Mock()
        mock_table = Mock()
        mock_db.open_table.return_value = mock_table
        
        # Mock unsorted DataFrame
        sample_data = pd.DataFrame({
            'key': ['zebra', 'alpha', 'beta', 'gamma'],
            'vector': [[0.1]*128, [0.2]*128, [0.3]*128, [0.4]*128]
        })
        mock_table.search.return_value.limit.return_value.to_pandas.return_value = sample_data
        
        # Test sorting (no next_token)
        items, next_token = await list_vectors(mock_db, "test-table", max_results=10, next_token=None)
        
        # Results should be sorted by key
        assert len(items) == 4
        assert items[0]['key'] == 'alpha'
        assert items[1]['key'] == 'beta'
        assert items[2]['key'] == 'gamma'
        assert items[3]['key'] == 'zebra'

    @pytest.mark.asyncio
    async def test_list_vectors_error_handling(self):
        """Test error handling in list_vectors."""
        # Mock database that raises exception
        mock_db = Mock()
        mock_db.open_table.side_effect = Exception("Database connection failed")
        
        # Test error handling
        with pytest.raises(Exception) as exc_info:
            await list_vectors(mock_db, "test-table", max_results=10, next_token=None)
        
        assert "Database connection failed" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__])