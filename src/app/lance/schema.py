"""
Lance schema management for S3 Vectors.

Simple schema: key, vector, and nonfilter JSON column.
"""

import pyarrow as pa
import json
from typing import Dict, List, Any


def create_vector_schema(dimension: int) -> pa.Schema:
    """
    Create a simple Lance table schema for vector storage.
    
    Args:
        dimension: Vector dimension
        
    Returns:
        PyArrow schema with key, vector, and nonfilter columns
    """
    return pa.schema([
        pa.field("key", pa.string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32()), nullable=False),  # Remove dimension constraint
        pa.field("nonfilter", pa.string(), nullable=True)  # JSON metadata
    ])


def prepare_batch_data(vectors: List[Dict[str, Any]], dimension: int) -> pa.Table:
    """
    Prepare vector data for Lance insertion.
    
    Args:
        vectors: List of vector data with key, vector, and metadata
        dimension: Expected vector dimension
        
    Returns:
        PyArrow Table for Lance insertion
    """
    keys = []
    vector_arrays = []
    nonfilter_json = []
    
    for item in vectors:
        keys.append(item["key"])
        
        # Ensure vector is correct dimension
        vector = item["vector"]
        if len(vector) != dimension:
            raise ValueError(f"Vector dimension mismatch: expected {dimension}, got {len(vector)}")
        vector_arrays.append(vector)
        
        # Store all metadata as JSON in nonfilter column
        metadata = item.get("metadata", {})
        nonfilter_json.append(json.dumps(metadata) if metadata else None)
    
    # Create PyArrow table
    data = {
        "key": keys,
        "vector": vector_arrays,
        "nonfilter": nonfilter_json
    }
    
    schema = create_vector_schema(dimension)
    return pa.table(data, schema=schema)
