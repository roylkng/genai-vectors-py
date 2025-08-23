"""
Lance schema management for S3 Vectors.

Simple schema: key, vector, and nonfilter JSON column.
"""

import pyarrow as pa
import json
from typing import Dict, List, Any


def create_vector_schema(dimension: int, filterable_types: Dict[str, pa.DataType] = None) -> pa.Schema:
    """
    Create a Lance table schema for vector storage with filterable columns (typed) and metadata_json.
    Args:
        dimension: Vector dimension
        filterable_types: Dict of filterable metadata keys to pyarrow types
    Returns:
        PyArrow schema with key, vector, filterable columns, and metadata_json
    """
    fields = [
        pa.field("key", pa.string(), nullable=False),
        pa.field("vector", pa.list_(pa.float32()), nullable=False),
        pa.field("metadata_json", pa.string(), nullable=True),
    ]
    if filterable_types:
        for k, typ in filterable_types.items():
            fields.append(pa.field(k, typ, nullable=True))
    return pa.schema(fields)


def infer_arrow_type(val):
    if isinstance(val, bool):
        return pa.bool_()
    elif isinstance(val, int):
        return pa.int64()
    elif isinstance(val, float):
        return pa.float64()
    else:
        return pa.string()

def create_filterable_types(vectors: List[Dict[str, Any]]) -> Dict[str, pa.DataType]:
    """Infer pyarrow types for filterable keys from the first batch."""
    types = {}
    for item in vectors:
        metadata = item.get("metadata", {})
        for k, v in metadata.items():
            if k not in types and v is not None:
                types[k] = infer_arrow_type(v)
    return types

def prepare_batch_data(vectors: List[Dict[str, Any]], dimension: int, filterable_types: Dict[str, pa.DataType]) -> pa.Table:
    """
    Prepare vector data for Lance insertion with filterable columns (typed) and metadata_json.
    """
    import json
    keys = []
    vector_arrays = []
    meta_columns = {k: [] for k in filterable_types}
    metadata_json = []
    for item in vectors:
        keys.append(item["key"])
        vector = item["vector"]
        if len(vector) != dimension:
            raise ValueError(f"Vector dimension mismatch: expected {dimension}, got {len(vector)}")
        vector_arrays.append(vector)
        metadata = item.get("metadata", {})
        filterable = {k: metadata[k] for k in filterable_types if k in metadata}
        nonfilterable = {k: v for k, v in metadata.items() if k not in filterable_types}
        for k, typ in filterable_types.items():
            v = filterable.get(k)
            if v is None:
                meta_columns[k].append(None)
            elif typ == pa.bool_():
                meta_columns[k].append(bool(v))
            elif typ == pa.int64():
                meta_columns[k].append(int(v))
            elif typ == pa.float64():
                meta_columns[k].append(float(v))
            else:
                meta_columns[k].append(str(v))
        metadata_json.append(json.dumps(nonfilterable) if nonfilterable else None)
    data = {
        "key": keys,
        "vector": vector_arrays,
        "metadata_json": metadata_json,
    }
    data.update(meta_columns)
    schema = create_vector_schema(dimension, filterable_types)
    return pa.table(data, schema=schema)
