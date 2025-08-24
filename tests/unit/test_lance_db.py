"""
Unit tests for the Lance DB module.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from app.lance.db import table_path


class TestLanceDB:
    """Test cases for the Lance DB module."""

    def test_table_path_basic(self):
        """Test basic table path generation."""
        result = table_path("test-index")
        assert result == "indexes/test-index/table"

    def test_table_path_with_special_characters(self):
        """Test table path generation with special characters."""
        # Test slashes
        result = table_path("test/index/name")
        assert result == "indexes/test_index_name/table"
        
        # Test colons
        result = table_path("test:index")
        assert result == "indexes/test_index/table"
        
        # Test spaces
        result = table_path("test index")
        assert result == "indexes/test_index/table"
        
        # Test dots
        result = table_path("test.index.name")
        assert result == "indexes/test.index.name/table"

    def test_table_path_complex_names(self):
        """Test table path generation with complex names."""
        # Test mixed special characters
        result = table_path("complex/index:name with spaces")
        assert result == "indexes/complex_index_name_with_spaces/table"
        
        # Test multiple slashes
        result = table_path("very/deep/nested/index")
        assert result == "indexes/very_deep_nested_index/table"
        
        # Test underscore preservation
        result = table_path("already_has_underscores")
        assert result == "indexes/already_has_underscores/table"

    def test_table_path_unicode(self):
        """Test table path generation with unicode characters."""
        # Test unicode characters (should be preserved)
        result = table_path("test-индекс")
        assert result == "indexes/test-индекс/table"
        
        # Test accented characters
        result = table_path("test-índice")
        assert result == "indexes/test-índice/table"

    def test_table_path_empty_and_edge_cases(self):
        """Test table path generation with edge cases."""
        # Test empty string
        result = table_path("")
        assert result == "indexes//table"
        
        # Test single character
        result = table_path("a")
        assert result == "indexes/a/table"
        
        # Test only special characters
        result = table_path("///")
        assert result == "indexes/___/table"


if __name__ == "__main__":
    pytest.main([__file__])