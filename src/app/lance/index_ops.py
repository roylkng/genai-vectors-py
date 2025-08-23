"""
Lance index operations with LanceDB-style smart indexing.
Auto-index after 50k vectors, following LanceDB best practices.
"""

import lancedb
from typing import List, Dict, Any, Optional
from .schema import create_vector_schema, prepare_batch_data
from .filter_translate import aws_filter_to_where


async def create_table(db, table_uri: str, dimension: int, nonfilterable_keys: Optional[List[str]] = None):
    """Create a new Lance table with schema, supporting non-filterable metadata keys."""
    import numpy as np
    from .schema import create_vector_schema
    # At creation, only non-filterable keys are known; filterable keys are added later
    filterable_keys = []
    schema = create_vector_schema(dimension, filterable_keys)
    import pyarrow as pa
    dummy_vector = np.zeros(dimension).tolist()
    data = {
        'key': ['__dummy__'],
        'vector': [dummy_vector],
        'metadata_json': [None],
    }
    table = db.create_table(table_uri, pa.table(data, schema=schema), mode="overwrite")
    table.delete("key = '__dummy__'")
    return table


async def upsert_vectors(
    db, 
    table_uri: str, 
    vectors: List[Dict[str, Any]]
):
    """Upsert vectors to Lance table"""
    try:
        # Try to open the table, create if it doesn't exist
        try:
            table = db.open_table(table_uri)
        except Exception:
            # Table doesn't exist, create it
            if vectors:
                first_vector = vectors[0].get("vector", [])
                dimension = len(first_vector)
            else:
                dimension = 768  # Default
            table = await create_table(db, table_uri, dimension)
        # Get dimension from first vector
        if vectors:
            first_vector = vectors[0].get("vector", [])
            dimension = len(first_vector)
        else:
            dimension = 768  # Default
        # Infer filterable types from this batch
        from .schema import create_filterable_types, prepare_batch_data
        filterable_types = create_filterable_types(vectors)
        # Add new columns for new filterable keys with correct types
        existing_fields = set(table.schema.names)
        new_fields = set(filterable_types.keys()) - existing_fields
        if new_fields:
            import pyarrow as pa
            table.add_columns([pa.field(k, filterable_types[k], nullable=True) for k in new_fields])
        # Prepare batch data for Lance
        batch_data = prepare_batch_data(vectors, dimension, filterable_types)
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
    filter_condition: Optional[Dict[str, Any]] = None,
    return_data: bool = True,
    return_metadata: bool = True,
    return_distance: bool = True
):
    """Search for similar vectors with enhanced filtering"""
    try:
        table = db.open_table(table_uri)
        
        # Start with basic search
        search_query = table.search(query_vector)
        
        # Apply filter if provided
        if filter_condition:
            # Convert Pydantic model to dict for filter translation
            if hasattr(filter_condition, 'model_dump'):
                filter_dict = filter_condition.model_dump()
            else:
                filter_dict = filter_condition
            
            where_clause = aws_filter_to_where(filter_dict)
            if where_clause and where_clause != "TRUE":
                print(f"DEBUG: Applying filter: {where_clause}")
                search_query = search_query.where(where_clause)
        
        # Execute search with limit
        search_query = search_query.limit(top_k)
        
        try:
            # Try Lance native search first
            results_df = search_query.to_pandas()
            print(f"DEBUG: Lance search returned {len(results_df)} results")
        except Exception as lance_error:
            print(f"DEBUG: Lance search failed, falling back to manual search: {lance_error}")
            # Fallback to manual similarity computation
            return await _manual_search_vectors(
                db, table_uri, query_vector, top_k, filter_condition,
                return_data, return_metadata, return_distance
            )
        
        # Convert to API format
        output_vectors = []
        for _, row in results_df.iterrows():
            vector_data = {"key": row["key"]}
            
            # Include distance if requested
            if return_distance and "_distance" in row:
                vector_data["distance"] = float(row["_distance"])
            elif return_distance:
                # Calculate distance manually if not provided
                import numpy as np
                query_arr = np.array(query_vector)
                doc_vector = np.array(row["vector"])
                dot_product = np.dot(query_arr, doc_vector)
                norm_query = np.linalg.norm(query_arr)
                norm_doc = np.linalg.norm(doc_vector)
                if norm_query > 0 and norm_doc > 0:
                    similarity = dot_product / (norm_query * norm_doc)
                    distance = 1.0 - similarity
                else:
                    distance = 1.0
                vector_data["distance"] = float(distance)
            
            # Include vector data if requested
            if return_data:
                vector_data["data"] = {
                    "float32": row["vector"].tolist() if hasattr(row["vector"], 'tolist') else row["vector"]
                }
            
            # Include metadata if requested
            if return_metadata and "nonfilter" in row and row["nonfilter"]:
                import json
                try:
                    metadata = json.loads(row["nonfilter"])
                    vector_data["metadata"] = metadata
                except json.JSONDecodeError:
                    pass
            
            output_vectors.append(vector_data)
        
        return output_vectors
        
    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to manual search
        return await _manual_search_vectors(
            db, table_uri, query_vector, top_k, filter_condition,
            return_data, return_metadata, return_distance
        )


async def _manual_search_vectors(
    db, table_uri: str, query_vector: List[float], top_k: int,
    filter_condition: Optional[Dict[str, Any]] = None,
    return_data: bool = True, return_metadata: bool = True, return_distance: bool = True
):
    """Manual vector search with filtering (fallback method)"""
    try:
        table = db.open_table(table_uri)
        df = table.to_pandas()
        
        print(f"DEBUG: Manual search on {len(df)} vectors")
        
        if len(df) == 0:
            return []
        
        # Apply filter first if provided
        if filter_condition:
            # Convert Pydantic model to dict for filter translation
            if hasattr(filter_condition, 'model_dump'):
                filter_dict = filter_condition.model_dump()
            else:
                filter_dict = filter_condition
            where_clause = aws_filter_to_where(filter_dict)
            if where_clause and where_clause != "TRUE":
                # Simple Python-based filtering for fallback
                filtered_df = _apply_python_filter(df, filter_dict)
                df = filtered_df.reset_index(drop=True)
                print(f"DEBUG: After filtering: {len(df)} vectors")
        
        # Compute cosine similarity manually
        import numpy as np
        query_arr = np.array(query_vector)
        
        similarities = []
        for idx, row in df.iterrows():
            try:
                doc_vector = row["vector"]
                if isinstance(doc_vector, str):
                    import json
                    doc_vector = json.loads(doc_vector)
                doc_vector = np.array(doc_vector)
                
                # Cosine similarity
                dot_product = np.dot(query_arr, doc_vector)
                norm_query = np.linalg.norm(query_arr)
                norm_doc = np.linalg.norm(doc_vector)
                
                if norm_query > 0 and norm_doc > 0:
                    similarity = dot_product / (norm_query * norm_doc)
                    distance = 1.0 - similarity
                else:
                    distance = 1.0
                
                similarities.append((idx, distance))
                
            except Exception as e:
                print(f"DEBUG: Error processing vector {idx}: {e}")
                continue
        
        # Sort by distance and take top_k
        similarities.sort(key=lambda x: x[1])
        results = similarities[:top_k]
        
        # Convert to API format
        output_vectors = []
        for idx, distance in results:
            row = df.iloc[idx]
            vector_data = {"key": row["key"]}
            
            if return_distance:
                vector_data["distance"] = float(distance)
            
            if return_data:
                vector_data["data"] = {
                    "float32": row["vector"].tolist() if hasattr(row["vector"], 'tolist') else row["vector"]
                }
            
            if return_metadata and "nonfilter" in row and row["nonfilter"]:
                import json
                try:
                    metadata = json.loads(row["nonfilter"])
                    vector_data["metadata"] = metadata
                except:
                    pass
            
            output_vectors.append(vector_data)
        
        return output_vectors
        
    except Exception as e:
        print(f"Manual search failed: {e}")
        return []


def _apply_python_filter(df, filter_condition: Dict[str, Any]):
    """Apply filter using Python (fallback when SQL fails)"""
    try:
        import json
        import pandas as pd
        
        def check_condition(row, condition):
            operator = condition.get("operator")
            
            if operator == "and":
                conditions = condition.get("conditions", condition.get("operands", []))
                return all(check_condition(row, cond) for cond in conditions)
            
            elif operator == "or":
                conditions = condition.get("conditions", condition.get("operands", []))
                return any(check_condition(row, cond) for cond in conditions)
            
            elif operator == "not":
                operand = condition.get("operand", condition.get("condition"))
                if operand:
                    return not check_condition(row, operand)
                return True
            
            else:
                # Leaf condition
                metadata_key = condition.get("metadata_key")
                filter_value = condition.get("value")
                
                if not metadata_key:
                    return True
                
                # Parse metadata from nonfilter column
                try:
                    metadata = json.loads(row["nonfilter"]) if row["nonfilter"] else {}
                except:
                    metadata = {}
                
                actual_value = metadata.get(metadata_key)
                
                if operator == "equals":
                    return actual_value == filter_value
                elif operator == "not_equals":
                    return actual_value != filter_value
                elif operator == "in":
                    return actual_value in filter_value if isinstance(filter_value, list) else False
                elif operator == "not_in":
                    return actual_value not in filter_value if isinstance(filter_value, list) else True
                elif operator in ["greater_than", "gt"]:
                    try:
                        return float(actual_value) > float(filter_value)
                    except:
                        return False
                elif operator in ["greater_equal", "gte"]:
                    try:
                        return float(actual_value) >= float(filter_value)
                    except:
                        return False
                elif operator in ["less_than", "lt"]:
                    try:
                        return float(actual_value) < float(filter_value)
                    except:
                        return False
                elif operator in ["less_equal", "lte"]:
                    try:
                        return float(actual_value) <= float(filter_value)
                    except:
                        return False
                elif operator == "exists":
                    return (actual_value is not None) == filter_value
                else:
                    return True
        
        # Apply filter to each row
        mask = df.apply(lambda row: check_condition(row, filter_condition), axis=1)
        return df[mask]
        
    except Exception as e:
        print(f"Python filter failed: {e}")
        return df  # Return unfiltered data on error


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


async def get_vectors(
    db, 
    table_uri: str, 
    keys: List[str],
    return_data: bool = True,
    return_metadata: bool = True
):
    """Get vectors by specific keys (batch lookup)"""
    try:
        table = db.open_table(table_uri)
        
        # Build where clause for multiple keys
        key_list = "', '".join(keys)
        where_clause = f"key IN ('{key_list}')"
        
        # Query for specific keys
        df = table.search().where(where_clause).to_pandas()
        
        print(f"DEBUG: Requested {len(keys)} keys, found {len(df)} vectors")
        
        # Convert to API format
        vectors = []
        found_keys = set()
        
        for _, row in df.iterrows():
            vector_data = {"key": row["key"]}
            found_keys.add(row["key"])
            
            # Include vector data if requested
            if return_data:
                vector_data["data"] = {
                    "float32": row["vector"].tolist() if hasattr(row["vector"], 'tolist') else row["vector"]
                }
            
            # Include metadata if requested
            if return_metadata and "nonfilter" in row and row["nonfilter"]:
                import json
                try:
                    metadata = json.loads(row["nonfilter"])
                    vector_data["metadata"] = metadata
                except json.JSONDecodeError:
                    pass
            
            vectors.append(vector_data)
        
        # Log missing keys for debugging
        missing_keys = set(keys) - found_keys
        if missing_keys:
            print(f"DEBUG: Missing keys: {missing_keys}")
        
        return vectors
        
    except Exception as e:
        print(f"Get vectors failed: {e}")
        import traceback
        traceback.print_exc()
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
