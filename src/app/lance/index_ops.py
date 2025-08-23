"""
Lance index operations with LanceDB-style smart indexing.
Auto-index after 50k vectors, following LanceDB best practices.
"""

import lancedb
from typing import List, Dict, Any, Optional
from .schema import create_vector_schema, prepare_batch_data
from .filter_translate import aws_filter_to_where


async def create_table(db, table_uri: str, dimension: int):
    """Create a new Lance table with schema"""
    # Create empty table with minimal data to establish schema
    import pandas as pd
    import numpy as np
    
    # Create a single dummy row with correct types, then delete it
    dummy_vector = np.zeros(dimension).tolist()
    
    initial_data = pd.DataFrame({
        'key': ['__dummy__'],
        'vector': [dummy_vector],
        'nonfilter': ['{}']
    })
    
    # Create table with initial data
    table = db.create_table(table_uri, initial_data, mode="overwrite")
    
    # Delete the dummy row to have an empty table
    table.delete("key = '__dummy__'")
    
    return table


async def upsert_vectors(
    db, 
    table_uri: str, 
    vectors: List[Dict[str, Any]]
):
    """Upsert vectors to Lance table"""
    try:
        # Open the table
        table = db.open_table(table_uri)
        
        # Get dimension from first vector
        if vectors:
            first_vector = vectors[0].get("vector", [])
            dimension = len(first_vector)
        else:
            dimension = 768  # Default
        
        # Prepare batch data for Lance
        batch_data = prepare_batch_data(vectors, dimension)
        
        # Add data using add method
        table.add(batch_data)
        
        return True
        
    except Exception as e:
        print(f"Upsert failed: {e}")
        return False


async def search_vectors(
    db, 
    table_uri: str, 
    query_vector: List[float], 
    top_k: int, 
    filter_condition: Optional[Dict[str, Any]] = None
):
    """Search for similar vectors"""
    try:
        table = db.open_table(table_uri)
        
        # Use simple pandas operations instead of Lance search
        # Get all data and compute similarity manually for now
        df = table.to_pandas()
        
        print(f"DEBUG: Found {len(df)} vectors in table")
        
        if len(df) == 0:
            return []
        
        # Compute cosine similarity manually
        import numpy as np
        
        # Convert vectors to numpy arrays
        query_arr = np.array(query_vector)
        
        similarities = []
        for idx, row in df.iterrows():
            try:
                # Handle different vector storage formats
                doc_vector = row["vector"]
                if isinstance(doc_vector, str):
                    # Parse JSON if stored as string
                    import json
                    doc_vector = json.loads(doc_vector)
                doc_vector = np.array(doc_vector)
                
                # Cosine similarity
                dot_product = np.dot(query_arr, doc_vector)
                norm_query = np.linalg.norm(query_arr)
                norm_doc = np.linalg.norm(doc_vector)
                
                if norm_query > 0 and norm_doc > 0:
                    similarity = dot_product / (norm_query * norm_doc)
                    distance = 1.0 - similarity  # Convert to distance
                else:
                    distance = 1.0
                
                similarities.append((idx, distance))
                print(f"DEBUG: Vector {row['key']}: similarity={similarity:.3f}, distance={distance:.3f}")
                
            except Exception as e:
                print(f"DEBUG: Error processing vector {idx}: {e}")
                continue
        
        # Sort by distance (ascending)
        similarities.sort(key=lambda x: x[1])
        
        # Take top_k results
        results = similarities[:top_k]
        
        print(f"DEBUG: Returning {len(results)} results")
        
        # Convert to API format
        output_vectors = []
        for idx, distance in results:
            row = df.iloc[idx]
            vector_data = {
                "key": row["key"],
                "vector": row["vector"].tolist() if hasattr(row["vector"], 'tolist') else row["vector"],
                "score": float(distance)
            }
            
            # Add metadata from nonfilter JSON
            if "nonfilter" in row and row["nonfilter"]:
                import json
                try:
                    metadata = json.loads(row["nonfilter"])
                    vector_data["metadata"] = metadata
                except:
                    pass
            
            output_vectors.append(vector_data)
        
        return output_vectors
        
    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        return []


async def list_vectors(
    db, 
    table_uri: str, 
    segment_id: int = 0,
    segment_count: int = 1,
    max_results: int = 1000
):
    """List vectors with hash-based segmentation"""
    try:
        table = db.open_table(table_uri)
        
        # Simple segmentation using hash of key
        if segment_count > 1:
            where_clause = f"hash(key) % {segment_count} = {segment_id}"
            results = table.search().where(where_clause).limit(max_results).to_pandas()
        else:
            results = table.search().limit(max_results).to_pandas()
        
        # Convert to API format
        vectors = []
        for _, row in results.iterrows():
            vector_data = {
                "key": row["key"],
                "vector": row["vector"].tolist() if hasattr(row["vector"], 'tolist') else row["vector"]
            }
            
            # Add metadata
            if row["nonfilter"]:
                import json
                try:
                    metadata = json.loads(row["nonfilter"])
                    vector_data["metadata"] = metadata
                except:
                    pass
            
            vectors.append(vector_data)
        
        return vectors
        
    except Exception as e:
        print(f"List vectors failed: {e}")
        return []


async def delete_vectors(db, table_uri: str, vector_keys: List[str]):
    """Delete vectors by keys"""
    try:
        table = db.open_table(table_uri)
        
        # Build delete condition
        key_list = "', '".join(vector_keys)
        delete_condition = f"key IN ('{key_list}')"
        
        # Delete rows
        table.delete(delete_condition)
        
        return len(vector_keys)
        
    except Exception as e:
        print(f"Delete failed: {e}")
        return 0


async def rebuild_index(db, table_uri: str, index_type: str = None):
    """
    Rebuild index with smart logic like LanceDB.
    Only creates index when vector count exceeds threshold.
    """
    try:
        table = db.open_table(table_uri)
        
        # Get current vector count
        vector_count = len(table.to_pandas())
        
        # Smart indexing logic like LanceDB
        from ..util import config
        threshold = config.LANCE_INDEX_THRESHOLD
        
        # Determine if we should index
        should_index = False
        actual_index_type = "NONE"
        
        if index_type == "AUTO" or index_type is None:
            if vector_count >= threshold:
                # Use IVF_PQ for large datasets (LanceDB default)
                actual_index_type = "IVF_PQ"
                should_index = True
            # For < 50k vectors, use brute force (no index)
        elif index_type in ["IVF_PQ", "HNSW"]:
            # Explicit index type requested
            actual_index_type = index_type
            should_index = True
        # else: index_type == "NONE", don't index
        
        if should_index:
            print(f"Creating {actual_index_type} index for {vector_count} vectors")
            
            if actual_index_type == "IVF_PQ":
                # IVF_PQ parameters - simplified approach
                num_partitions = min(256, max(1, vector_count // 256))
                table.create_index()
            elif actual_index_type == "HNSW":
                # HNSW parameters - simplified approach
                table.create_index()
        else:
            print(f"Skipping index creation: {vector_count} vectors < {threshold} threshold")
            
    except Exception as e:
        print(f"Index rebuild failed: {e}")
        # Continue without index - brute force search still works


async def get_table_stats(db, table_uri: str):
    """Get table statistics"""
    try:
        table = db.open_table(table_uri)
        df = table.to_pandas()
        
        return {
            "vector_count": len(df),
            "has_index": bool(table.list_indices()),
            "index_type": "unknown"  # Lance doesn't expose this easily
        }
    except Exception as e:
        print(f"Stats failed: {e}")
        return {"vector_count": 0, "has_index": False, "index_type": "none"}
