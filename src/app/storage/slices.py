import io, json
from typing import List, Dict, Any
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None
    pq = None

# slice schema: key (string), vec (list<float>), meta (json string)
def rows_to_parquet_bytes(rows: List[Dict[str, Any]]) -> io.BytesIO:
    if pa is None:
        return rows_to_jsonl_bytes(rows)
    keys = [r["key"] for r in rows]
    vecs = [r["vec"] for r in rows]
    metas = [json.dumps(r.get("meta", {})) for r in rows]
    arr_key = pa.array(keys, type=pa.string())
    arr_meta = pa.array(metas, type=pa.string())
    # list<float32>
    arr_vec = pa.array(vecs, type=pa.list_(pa.float32()))
    table = pa.table({"key": arr_key, "vec": arr_vec, "meta": arr_meta})
    bio = io.BytesIO()
    pq.write_table(table, bio, compression="zstd")
    bio.seek(0)
    return bio

def rows_to_jsonl_bytes(rows: List[Dict[str, Any]]) -> io.BytesIO:
    bio = io.BytesIO()
    for r in rows:
        bio.write(json.dumps(r).encode("utf-8") + b"\n")
    bio.seek(0)
    return bio
