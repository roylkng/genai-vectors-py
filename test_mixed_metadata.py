#!/usr/bin/env python3
"""
Test with mixed filterable and non-filterable metadata
"""

import sys
import os
sys.path.append('/home/rajan/Desktop/work/genai-vectors-py/src')

import lancedb
import pyarrow as pa
import json
import numpy as np
from app.lance.db import connect_bucket, table_path
from app.lance.schema import create_vector_schema, create_filterable_types, prepare_batch_data

def test_mixed_metadata():
    """Test with mixed filterable and non-filterable metadata"""
    # Create test data
    vectors = [
        {
            "key": "test-doc",
            "vector": np.random.rand(10).tolist(),  # Small vector for testing
            "metadata": {
                "category": "test",      # This should be filterable
                "topic": "debug",        # This should be filterable
                "description": "A long description that won't be filterable",  # This should go to metadata_json
                "tags": ["tag1", "tag2"] # This should go to metadata_json
            }
        }
    ]
    
    # Test filterable types inference
    filterable_types = create_filterable_types(vectors)
    print(f"Filterable types: {filterable_types}")
    
    # Test batch preparation
    batch_data = prepare_batch_data(vectors, 10, filterable_types)
    print(f"Batch data schema: {batch_data.schema}")
    print("Batch data columns:")
    for i, column in enumerate(batch_data.column_names):
        print(f"  {column}: {batch_data.column(i)}")

if __name__ == "__main__":
    test_mixed_metadata()