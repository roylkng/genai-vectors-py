"""
Index builder for S3 Vectors API.

Separate callable for explicit index building to avoid brute-force scans at scale.
Can be called from thread/CLI/cron jobs.
"""

from app.lance.db import connect_bucket, table_path
from app.storage.s3_backend import S3Storage
from app.util import config
import lancedb
import json
from typing import Dict, Any


def build_index_if_needed(bucket: str, index: str) -> Dict[str, Any]:
    """
    Build vector index if needed for efficient similarity search.
    
    Args:
        bucket: Vector bucket name
        index: Index name
        
    Returns:
        Status dictionary with index building results
    """
    try:
        s3 = S3Storage()
        cfg_key = f"{config.INDEX_DIR}/{index}/_index_config.json"
        cfg = s3.get_json(bucket, cfg_key)
        dim = cfg["dimension"]
        metric = cfg.get("distanceMetric", "cosine").lower()  # "cosine" | "l2"
        itype = cfg.get("indexType", "AUTO")                 # "AUTO"|"IVF_PQ"|"HNSW"|"NONE"
        params = cfg.get("indexParams", {})                  # persist per-index

        db = connect_bucket(bucket)
        table_uri = table_path(index)                          # colocate data+config
        tbl = db.open_table(table_uri)

        # already has a vector index?
        if any(ix.name == "vector_idx" for ix in tbl.list_indices()):
            return {"status": "READY", "note": "index exists"}

        # choose config (AUTO heuristic)
        if itype == "AUTO":
            n = tbl.count_rows()
            itype = "IVF_PQ" if n >= 200_000 and dim >= 256 else "HNSW"

        if itype == "IVF_PQ":
            nlist = params.get("numPartitions") or max(256, int(2*(tbl.count_rows()**0.5)))
            m = params.get("numSubVectors") or max(8, dim//8)
            cfg_obj = lancedb.Index.ivfPq(num_partitions=nlist, num_sub_vectors=m, distance_type=metric)
        elif itype == "HNSW":
            M = params.get("M", 16)
            efc = params.get("efConstruction", 200)
            cfg_obj = lancedb.Index.hnsw(m=M, ef_construction=efc, distance_type=metric)
        elif itype == "NONE":
            return {"status": "SKIPPED", "note": "index disabled"}
        else:
            raise ValueError(f"Unknown indexType {itype}")

        tbl.create_index("vector", config=cfg_obj, name="vector_idx")
        return {"status": "READY", "indexType": itype}
        
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}