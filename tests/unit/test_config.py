"""
Unit tests for configuration and limits.
"""

import pytest
import sys
import os
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from app.util import config
from app.errors import validate_top_k, validate_batch_size, validate_dimension
from app.errors import ValidationException


class TestConfiguration:
    """Test cases for configuration and limits."""

    def test_config_constants(self):
        """Test that configuration constants are set correctly."""
        assert config.MAX_BATCH == 500
        assert config.MAX_TOPK == 100
        assert config.MAX_DIM == 4096
        assert config.MAX_METADATA_BYTES == 8192
        assert config.MAX_METADATA_KEYS == 50
        assert config.INDEX_DIR == "indexes"
        assert config.TABLE_DIR == "table"

    @patch.dict(os.environ, {"MAX_BATCH": "1000", "MAX_TOPK": "50"})
    def test_config_env_override(self):
        """Test that environment variables override defaults."""
        # Reload config to pick up environment variables
        import importlib
        importlib.reload(config)
        
        assert config.MAX_BATCH == 1000
        assert config.MAX_TOPK == 50

    def test_validate_top_k_valid(self):
        """Test valid top_k values."""
        # Test boundary values
        validate_top_k(1)  # Minimum
        validate_top_k(config.MAX_TOPK)  # Maximum
        validate_top_k(50)  # Middle value

    def test_validate_top_k_invalid(self):
        """Test invalid top_k values."""
        # Test below minimum
        with pytest.raises(ValidationException):
            validate_top_k(0)
        
        # Test above maximum
        with pytest.raises(ValidationException):
            validate_top_k(config.MAX_TOPK + 1)
        
        # Test negative
        with pytest.raises(ValidationException):
            validate_top_k(-1)

    def test_validate_batch_size_valid(self):
        """Test valid batch sizes."""
        # Test empty batch
        validate_batch_size([])
        
        # Test at limit
        vectors = [{"key": f"doc{i}"} for i in range(config.MAX_BATCH)]
        validate_batch_size(vectors)
        
        # Test below limit
        vectors = [{"key": f"doc{i}"} for i in range(10)]
        validate_batch_size(vectors)

    def test_validate_batch_size_invalid(self):
        """Test invalid batch sizes."""
        # Test above limit
        vectors = [{"key": f"doc{i}"} for i in range(config.MAX_BATCH + 1)]
        with pytest.raises(ValidationException):
            validate_batch_size(vectors)

    def test_validate_dimension_valid(self):
        """Test valid dimension values."""
        # Test boundary values
        validate_dimension(1)  # Minimum
        validate_dimension(config.MAX_DIM)  # Maximum
        validate_dimension(128)  # Common dimension
        validate_dimension(768)  # Common dimension
        validate_dimension(1536)  # Common dimension

    def test_validate_dimension_invalid(self):
        """Test invalid dimension values."""
        # Test below minimum
        with pytest.raises(ValidationException):
            validate_dimension(0)
        
        # Test above maximum
        with pytest.raises(ValidationException):
            validate_dimension(config.MAX_DIM + 1)
        
        # Test negative
        with pytest.raises(ValidationException):
            validate_dimension(-1)


if __name__ == "__main__":
    pytest.main([__file__])