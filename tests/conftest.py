"""
Pytest configuration and fixtures for S3 Vectors tests.
"""

import sys
import os
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test configuration
TEST_BUCKET = "test-bucket"
TEST_INDEX = "test-index"
TEST_DIMENSION = 128


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_lancedb():
    """Mock LanceDB for unit tests."""
    with patch('lancedb.connect') as mock_connect:
        mock_db = Mock()
        mock_db.create_table.return_value = Mock()
        mock_db.open_table.return_value = Mock()
        mock_connect.return_value = mock_db
        yield mock_db


@pytest.fixture
def mock_s3_storage():
    """Mock S3 storage for unit tests."""
    with patch('app.storage.s3_backend.S3Storage') as mock_s3:
        mock_instance = Mock()
        mock_instance.bucket_exists.return_value = True
        mock_instance.get_json.return_value = {
            "dimension": TEST_DIMENSION,
            "distanceMetric": "cosine"
        }
        mock_s3.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def sample_vectors():
    """Sample vectors for testing."""
    return [
        {
            "key": "doc1",
            "vector": [0.1] * TEST_DIMENSION,
            "metadata": {"category": "test", "score": 0.95}
        },
        {
            "key": "doc2", 
            "vector": [0.2] * TEST_DIMENSION,
            "metadata": {"category": "example", "score": 0.87}
        }
    ]


@pytest.fixture
def sample_filter():
    """Sample filter for testing."""
    return {
        "operator": "and",
        "conditions": [
            {
                "operator": "equals",
                "metadata_key": "category",
                "value": "test"
            },
            {
                "operator": "greater_than",
                "metadata_key": "score",
                "value": 0.8
            }
        ]
    }