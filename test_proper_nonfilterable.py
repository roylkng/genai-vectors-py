#!/usr/bin/env python3
"""
Test with proper non-filterable key handling
"""

import sys
import os
sys.path.append('/home/rajan/Desktop/work/genai-vectors-py/src')

import lancedb
import pyarrow as pa
import json
import numpy as np
from typing import List, Dict, Any
from app.lance.db import connect_bucket, table_path
from app.lance.schema import create_vector_schema, prepare_batch_data

def create_filterable_types_with_nonfilterable(vectors: List[Dict[str, Any]], nonfilterable_keys: List[str]) -> Dict[str, pa.DataType]:
    """Infer pyarrow types for filterable keys from the first batch, excluding non-filterable keys."""
    from app.lance.schema import infer_arrow_type
    
    types = {}
    for item in vectors:
        metadata = item.get("metadata", {})
        for k, v in metadata.items():
            # Only include keys that are not in the non-filterable list
            if k not in types and v is not None and k not in nonfilterable_keys:
                types[k] = infer_arrow_type(v)
    return types

def test_proper_nonfilterable():
    """Test with proper non-filterable key handling"""
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
    
    # Specify non-filterable keys
    nonfilterable_keys = ["description", "tags"]
    
    # Test filterable types inference
    filterable_types = create_filterable_types_with_nonfilterable(vectors, nonfilterable_keys)
    print(f"Filterable types: {filterable_types}")
    
    # Test batch preparation
    batch_data = prepare_batch_data(vectors, 10, filterable_types)
    print(f"Batch data schema: {batch_data.schema}")
    print("Batch data columns:")
    for i, column in enumerate(batch_data.column_names):
        print(f"  {column}: {batch_data.column(i)}")
    
    # Check the metadata_json column
    metadata_json_col = batch_data.column("metadata_json")
    if metadata_json_col:
        json_value = str(metadata_json_col[0])  # First row
        print(f"metadata_json value: {json_value}")
        if json_value and json_value != "None":
            try:
                # Remove quotes if present
                if json_value.startswith('"') and json_value.endswith('"'):
                    json_value = json_value[1:-1]
                parsed = json.loads(json_value)
                print(f"Parsed metadata_json: {parsed}")
            except Exception as e:
                print(f"Error parsing metadata_json: {e}")

if __name__ == "__main__":
    test_proper_nonfilterable()