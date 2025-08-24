"""
Lance index operations for the S3 Vectors–style service.

Key changes (Phase-1):
- Keep request path pure: no implicit index creation here.
- Prefer Lance native search; Pandas path is a guarded fallback.
- Pagination uses key-ordered NextToken/MaxResults (no bogus .search()).
- Schema evolves by adding typed filterable columns on demand.
- Safer SQL literal handling for key-based WHERE/DELETE IN clauses.
- Cheap row counts (no full table materialization).
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

from app.errors import InternalServiceException
from .schema import create_vector_schema, create_filterable_types, prepare_batch_data
from .filter_translate import aws_filter_to_where

logger = logging.getLogger("lance.index_ops")


# ---------- small helpers ----------

def _sql_literal(s: str) -> str:
    """Escape a Python string for use in simple SQL string literals."""
    return (s or "").replace("'", "''")


def _count_rows(tbl) -> int:
    """Return row count without materializing full table to Pandas."""
    try:
        return tbl.count_rows()
    except Exception:
        # Fallback: count via Arrow with a narrow projection
        return tbl.to_arrow(columns=["key"]).num_rows


# ---------- table & write path ----------

async def create_table(db, table_uri: str, dimension: int, nonfilterable_keys: Optional[List[str]] = None):
    """
    Create a new Lance table with base schema.
    nonfilterable_keys are stored in metadata_json (typed filterables are added later).
    """
    try:
        import numpy as np
        import pyarrow as pa

        # Base schema: key, vector, metadata_json
        schema = create_vector_schema(dimension, filterable_types={})

        # Materialize schema with a dummy write (then delete the row)
        dummy = {
            "key": ["__dummy__"],
            "vector": [np.zeros(dimension, dtype="float32").tolist()],
            "metadata_json": [None],
        }
        tbl = db.create_table(table_uri, pa.table(dummy, schema=schema), mode="overwrite")
        tbl.delete("key = '__dummy__'")
        return tbl
    except Exception as e:
        raise InternalServiceException(f"Create table failed: {e}")


async def upsert_vectors(
    db,
    table_uri: str,
    vectors: List[Dict[str, Any]],
) -> bool:
    """
    Upsert vectors into a Lance table.
    - Auto-creates the table if missing (dim inferred from first vector, default 768).
    - Adds new typed filterable columns on demand (schema evolution).
    """
    try:
        # Open or create table
        try:
            tbl = db.open_table(table_uri)
        except Exception:
            dim = len(vectors[0].get("vector", [])) if vectors else 768
            tbl = await create_table(db, table_uri, dim)

        # Determine dimension from batch (fallback 768)
        dim = len(vectors[0].get("vector", [])) if vectors else 768

        # Infer filterable types from this batch
        ftypes = create_filterable_types(vectors)

        # Add any new typed filterable columns (nullable)
        try:
            import pyarrow as pa
            existing = set(tbl.schema.names)
            to_add = [pa.field(k, ftypes[k], nullable=True) for k in (set(ftypes.keys()) - existing)]
            if to_add:
                tbl.add_columns(to_add)
        except Exception as add_err:
            # Likely a concurrent writer added them first; re-open table and continue
            logger.debug(f"add_columns race (safe to ignore): {add_err}")
            tbl = db.open_table(table_uri)

        # Prepare and append batch
        batch = prepare_batch_data(vectors, dim, ftypes)
        tbl.add(batch)
        return True

    except Exception as e:
        raise InternalServiceException(f"Upsert failed: {e}")


# ---------- read path (search / list / get / delete) ----------

async def search_vectors(
    db,
    table_uri: str,
    query_vector: List[float],
    top_k: int,
    filter_condition: Optional[Dict[str, Any]] = None,
    return_data: bool = True,
    return_metadata: bool = True,
    return_distance: bool = True,
) -> List[Dict[str, Any]]:
    """
    Vector search with optional prefilter.
    - Primary path: Lance native ANN/flat search.
    - Fallback: Pandas-based cosine scan (guarded by config flag and errors).
    """
    try:
        from app.util import config

        tbl = db.open_table(table_uri)

        # Build Lance-native search
        q = tbl.search(query_vector)

        # Translate AWS-style filter to a WHERE clause (translator should prefer typed cols)
        if filter_condition:
            fdict = filter_condition.model_dump() if hasattr(filter_condition, "model_dump") else filter_condition
            where = aws_filter_to_where(fdict)
            if where and where.upper() != "TRUE":
                q = q.where(where)

        q = q.limit(top_k)

        try:
            df = q.to_pandas()
        except Exception as lance_err:
            logger.warning(f"Lance search failed, fallback may apply: {lance_err}")
            if getattr(config, "ENABLE_PANDAS_FALLBACK", False):
                return await _manual_search_vectors(
                    db, table_uri, query_vector, top_k, filter_condition,
                    return_data, return_metadata, return_distance
                )
            raise

        # Convert to API format
        out: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item: Dict[str, Any] = {"key": row["key"]}

            # distance
            if return_distance:
                if "_distance" in row:
                    item["distance"] = float(row["_distance"])
                else:
                    # compute cosine distance if missing
                    try:
                        import numpy as np
                        qv = np.asarray(query_vector, dtype="float32")
                        dv = row["vector"]
                        dv = np.asarray(dv.tolist() if hasattr(dv, "tolist") else dv, dtype="float32")
                        sim = float(np.dot(qv, dv) / (np.linalg.norm(qv) * np.linalg.norm(dv) + 1e-12))
                        item["distance"] = float(1.0 - sim)
                    except Exception:
                        item["distance"] = 1.0

            # vector data
            if return_data:
                vec = row["vector"]
                item["data"] = {"float32": vec.tolist() if hasattr(vec, "tolist") else vec}

            # metadata (typed columns + metadata_json)
            if return_metadata:
                md: Dict[str, Any] = {}

                # metadata_json blob first
                if "metadata_json" in row and row["metadata_json"]:
                    import json
                    try:
                        md.update(json.loads(row["metadata_json"]))
                    except Exception:
                        pass

                # then typed columns
                for col in row.index:
                    if col in {"key", "vector", "metadata_json", "_distance", "_rowid"}:
                        continue
                    val = row[col]
                    if val is not None:
                        if hasattr(val, "item"):
                            val = val.item()
                        md[col] = val

                if md:
                    item["metadata"] = md

            out.append(item)

        return out

    except Exception as e:
        raise InternalServiceException(f"Search failed: {e}")


async def _manual_search_vectors(
    db,
    table_uri: str,
    query_vector: List[float],
    top_k: int,
    filter_condition: Optional[Dict[str, Any]] = None,
    return_data: bool = True,
    return_metadata: bool = True,
    return_distance: bool = True,
) -> List[Dict[str, Any]]:
    """Fallback: full scan in Pandas with optional Python-side filter + cosine sorting."""
    try:
        tbl = db.open_table(table_uri)
        df = tbl.to_pandas()
        if df.empty:
            return []

        # Apply Python-side filter if provided
        if filter_condition:
            fdict = filter_condition.model_dump() if hasattr(filter_condition, "model_dump") else filter_condition
            df = _apply_python_filter(df, fdict).reset_index(drop=True)

        import numpy as np
        qv = np.asarray(query_vector, dtype="float32")

        # Compute cosine distance per row
        dists: List[Tuple[int, float]] = []
        for i, row in df.iterrows():
            dv = row["vector"]
            if isinstance(dv, str):
                import json
                try:
                    dv = json.loads(dv)
                except Exception:
                    continue
            dv = np.asarray(dv.tolist() if hasattr(dv, "tolist") else dv, dtype="float32")
            sim = float(np.dot(qv, dv) / (np.linalg.norm(qv) * np.linalg.norm(dv) + 1e-12))
            dists.append((i, 1.0 - sim))

        dists.sort(key=lambda x: x[1])
        top = dists[:top_k]

        # Build output
        out: List[Dict[str, Any]] = []
        for i, dist in top:
            row = df.iloc[i]
            item: Dict[str, Any] = {"key": row["key"]}
            if return_distance:
                item["distance"] = float(dist)
            if return_data:
                vec = row["vector"]
                item["data"] = {"float32": vec.tolist() if hasattr(vec, "tolist") else vec}
            if return_metadata:
                md: Dict[str, Any] = {}
                if "metadata_json" in row and row["metadata_json"]:
                    import json
                    try:
                        md.update(json.loads(row["metadata_json"]))
                    except Exception:
                        pass
                for col in row.index:
                    if col in {"key", "vector", "metadata_json"}:
                        continue
                    val = row[col]
                    if val is not None:
                        if hasattr(val, "item"):
                            val = val.item()
                        md[col] = val
                if md:
                    item["metadata"] = md
            out.append(item)

        return out

    except Exception as e:
        raise InternalServiceException(f"Manual search failed: {e}")


def _apply_python_filter(df, condition: Dict[str, Any]):
    """Very small Python-side filter engine; used only in fallback path."""
    try:
        import json

        def check(row, cond):
            op = cond.get("operator")
            if op == "and":
                return all(check(row, c) for c in (cond.get("conditions") or cond.get("operands") or []))
            if op == "or":
                return any(check(row, c) for c in (cond.get("conditions") or cond.get("operands") or []))
            if op == "not":
                inner = cond.get("operand") or cond.get("condition")
                return not check(row, inner) if inner else True

            # leaf
            key = cond.get("metadata_key")
            val = cond.get("value")
            if not key:
                return True

            try:
                blob = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
            except Exception:
                blob = {}

            actual = blob.get(key)
            if op == "equals":
                return actual == val
            if op == "not_equals":
                return actual != val
            if op == "in":
                return actual in val if isinstance(val, list) else False
            if op == "not_in":
                return actual not in val if isinstance(val, list) else True
            if op in ("gt", "greater_than"):
                try: return float(actual) > float(val)
                except Exception: return False
            if op in ("gte", "greater_equal"):
                try: return float(actual) >= float(val)
                except Exception: return False
            if op in ("lt", "less_than"):
                try: return float(actual) < float(val)
                except Exception: return False
            if op in ("lte", "less_equal"):
                try: return float(actual) <= float(val)
                except Exception: return False
            if op == "exists":
                return (actual is not None) == bool(val)
            return True

        mask = df.apply(lambda r: check(r, condition), axis=1)
        return df[mask]
    except Exception as e:
        raise InternalServiceException(f"Python filter failed: {e}")


async def list_vectors(
    db,
    table_uri: str,
    max_results: int = 1000,
    next_token: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], Optional[str]]:
    """
    List vector keys with key-ordered pagination.
    Returns (items, nextToken). No query vector involved.
    """
    try:
        tbl = db.open_table(table_uri)
        df = tbl.to_pandas()[["key"]].sort_values("key")

        if next_token is not None:
            df = df[df["key"] > next_token]

        page = df.head(max_results)
        items = [{"key": k} for k in page["key"].tolist()]
        next_tok = page["key"].iloc[-1] if len(page) == max_results else None
        return items, next_tok
    except Exception as e:
        raise InternalServiceException(f"List vectors failed: {e}")


async def get_vectors(
    db,
    table_uri: str,
    keys: List[str],
    return_data: bool = True,
    return_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    Batch fetch by keys. For phase-1, use Pandas filtering.
    (Phase-2: replace with a predicate-pushdown scanner when available.)
    """
    try:
        tbl = db.open_table(table_uri)
        df = tbl.to_pandas()

        if df.empty or not keys:
            return []

        keyset = set(keys)
        df = df[df["key"].isin(keyset)]

        out: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            item: Dict[str, Any] = {"key": row["key"]}

            if return_data:
                vec = row["vector"]
                item["data"] = {"float32": vec.tolist() if hasattr(vec, "tolist") else vec}

            if return_metadata:
                md: Dict[str, Any] = {}
                if row.get("metadata_json"):
                    import json
                    try:
                        md.update(json.loads(row["metadata_json"]))
                    except Exception:
                        pass
                for col in row.index:
                    if col in {"key", "vector", "metadata_json", "_distance", "_rowid"}:
                        continue
                    val = row[col]
                    if val is not None:
                        if hasattr(val, "item"):
                            val = val.item()
                        md[col] = val
                if md:
                    item["metadata"] = md

            out.append(item)

        return out

    except Exception as e:
        raise InternalServiceException(f"Get vectors failed: {e}")


async def delete_vectors(db, table_uri: str, vector_keys: List[str]) -> int:
    """Delete vectors by keys (IN (...) with safe literals)."""
    try:
        tbl = db.open_table(table_uri)
        if not vector_keys:
            return 0
        key_list = ", ".join(f"'{_sql_literal(k)}'" for k in vector_keys)
        tbl.delete(f"key IN ({key_list})")
        return len(vector_keys)
    except Exception as e:
        raise InternalServiceException(f"Delete failed: {e}")


# ---------- stats (no heavy scans) ----------

async def get_table_stats(db, table_uri: str) -> Dict[str, Any]:
    """Return lightweight table stats."""
    try:
        tbl = db.open_table(table_uri)
        return {
            "vector_count": _count_rows(tbl),
            "has_index": bool(tbl.list_indices()),
            "index_type": "unknown",  # Lance does not currently expose a printable type
        }
    except Exception as e:
        raise InternalServiceException(f"Stats failed: {e}")
