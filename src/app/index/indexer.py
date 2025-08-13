import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None; pq = None

from ..util import config
from ..storage.s3_backend import S3Storage
from .faiss_backends import HNSWBackend, IVFPQBackend

def _load_idmap(storage: S3Storage, bucket: str, index: str) -> Optional["pa.Table"]:
    if pq is None: return None
    try:
        data = storage.download_bytes(bucket, storage.idmap_key(index))
    except Exception:
        return None
    bio = pa.py_buffer(data)
    return pq.read_table(bio)

def _write_idmap(storage: S3Storage, bucket: str, index: str, table: "pa.Table") -> None:
    if pq is None:
        # fallback JSON (not recommended)
        rows = [dict(zip(table.column_names, r)) for r in zip(*[table[c].to_pylist() for c in table.column_names])]
        storage.put_json(bucket, storage.idmap_key(index), {"rows": rows})
        return
    import io
    out = io.BytesIO()
    pq.write_table(table, out, compression="zstd")
    storage.upload_bytes(bucket, storage.idmap_key(index), out.getvalue())

def _append_to_idmap(idmap: Optional["pa.Table"], new_keys: List[str], new_vecs: List[List[float]], new_meta: List[str]) -> "pa.Table":
    if pa is None: raise RuntimeError("pyarrow required")
    start_id = 0 if idmap is None else idmap.num_rows
    ids = list(range(start_id, start_id + len(new_keys)))
    arrays = {
        "id": pa.array(ids, type=pa.int64()),
        "key": pa.array(new_keys, type=pa.string()),
        "vec": pa.array(new_vecs, type=pa.list_(pa.float32())),
        "meta": pa.array(new_meta, type=pa.string()),
        "alive": pa.array([True]*len(ids), type=pa.bool_()),
    }
    new_tbl = pa.table(arrays)
    return new_tbl if idmap is None else pa.concat_tables([idmap, new_tbl], promote=True)

def _list_staged(storage: S3Storage, bucket: str, index: str) -> List[str]:
    return [k for k in storage.list_prefix(bucket, f"{config.STAGED_DIR}/{index}/")]

def _load_slice(storage: S3Storage, bucket: str, key: str) -> Tuple[List[str], List[List[float]], List[str]]:
    # returns lists: keys, vecs, metas(json string)
    data = storage.download_bytes(bucket, key)
    if key.endswith(".parquet") and pq is not None:
        bio = pa.py_buffer(data)
        tbl = pq.read_table(bio)
        keys = tbl["key"].to_pylist()
        vecs = tbl["vec"].to_pylist()
        metas = tbl["meta"].to_pylist()
        return keys, vecs, metas
    # jsonl fallback
    import json
    keys, vecs, metas = [], [], []
    for line in data.splitlines():
        r = json.loads(line)
        keys.append(r["key"]); vecs.append(r["vec"]); metas.append(json.dumps(r.get("meta", {})))
    return keys, vecs, metas

def _store_index(storage: S3Storage, bucket: str, index: str, algo: str, backend) -> None:
    if algo == "hnsw_flat":
        path = storage.index_file_key(index, "hnsw")
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=True) as tf:
            backend.save(tf.name)
            with open(tf.name, "rb") as f:
                storage.upload_bytes(bucket, path, f.read())
    else:  # ivfpq
        path = storage.index_file_key(index, "faiss")
        import tempfile, faiss
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=True) as tf:
            faiss.write_index(backend.index, tf.name)
            with open(tf.name, "rb") as f:
                storage.upload_bytes(bucket, path, f.read())

def _update_manifest(storage: S3Storage, bucket: str, index: str, algo: str, dim: int, metric: str, counts: int) -> None:
    mkey = storage.manifest_key(index)
    man = storage.get_json(bucket, mkey) or {}
    man.update({
        "index": index,
        "algo": algo,
        "dimension": dim,
        "metric": metric,
        "vectors": counts
    })
    storage.put_json(bucket, mkey, man)

def process_new_slices(vector_bucket: str, index: str, dim: int, metric: str,
                       algorithm: str, hnsw_threshold: int, nlist: int, m: int, nbits: int) -> int:
    """Pull staged slices, append to idmap, rebuild/add index, update manifest. Returns new vector count added."""
    s3 = S3Storage()
    s3.ensure_bucket(vector_bucket)

    # 1) Load staged
    staged = _list_staged(s3, vector_bucket, index)
    if not staged: return 0

    # 2) Load current idmap
    idmap = _load_idmap(s3, vector_bucket, index)  # Table[id:int64, key:str, vec:list<float>, meta:str, alive:bool]

    add_count = 0
    all_keys, all_vecs, all_meta = [], [], []
    for sk in staged:
        keys, vecs, metas = _load_slice(s3, vector_bucket, sk)
        add_count += len(keys)
        all_keys.extend(keys)
        all_vecs.extend(vecs)
        all_meta.extend(metas)

    # 3) Append to idmap and persist
    idmap = _append_to_idmap(idmap, all_keys, all_vecs, all_meta)
    _write_idmap(s3, vector_bucket, index, idmap)

    # 4) Build or extend index
    import numpy as np
    X = np.asarray(idmap["vec"].to_pylist(), dtype=np.float32)
    ids = np.asarray(idmap["id"].to_pylist(), dtype=np.int64)
    total = X.shape[0]
    use_hnsw = (algorithm == "hnsw_flat") or (algorithm == "hybrid" and total < hnsw_threshold)

    if use_hnsw:
        backend = HNSWBackend(dim=dim, metric=metric)
        backend.build(X, ids)
        _store_index(s3, vector_bucket, index, "hnsw_flat", backend)
        _update_manifest(s3, vector_bucket, index, "hnsw_flat", dim, metric, total)
    else:
        backend = IVFPQBackend(dim=dim, metric=metric, nlist=nlist, m=m, nbits=nbits)
        backend.build(X, ids)
        _store_index(s3, vector_bucket, index, "ivfpq", backend)
        _update_manifest(s3, vector_bucket, index, "ivfpq", dim, metric, total)

    # 5) Clear staged
    s3.delete_prefix(vector_bucket, f"{config.STAGED_DIR}/{index}/")
    return add_count

def search(vector_bucket: str, index: str, query: List[float], topk: int, nprobe: Optional[int]) -> List[Tuple[int, float]]:
    """Return [(id, distance)]"""
    s3 = S3Storage()
    man = s3.get_json(vector_bucket, s3.manifest_key(index)) or {}
    algo = man.get("algo")
    dim = int(man.get("dimension", len(query)))
    metric = man.get("metric", "cosine")

    # Load idmap
    if pq is None:
        raise RuntimeError("pyarrow required for idmap")
    idmap_tbl = _load_idmap(s3, vector_bucket, index)
    if idmap_tbl is None or idmap_tbl.num_rows == 0:
        return []
    alive = idmap_tbl["alive"].to_pylist()
    # Prepare backend and load index file
    import numpy as np, tempfile, os, faiss
    q = np.asarray([query], dtype=np.float32)
    if algo == "hnsw_flat":
        bk = HNSWBackend(dim=dim, metric=metric)
        with tempfile.NamedTemporaryFile(suffix=".hnsw", delete=False) as tf:
            tf.write(s3.download_bytes(vector_bucket, s3.index_file_key(index, "hnsw")))
            path = tf.name
        try:
            bk.load(path)
        finally:
            os.unlink(path)
        ids, dists = bk.search(q, topk=topk, nprobe=None)
    else:
        bk = IVFPQBackend(dim=dim, metric=metric,
                          nlist=man.get("nList", 1024), m=man.get("m", 16), nbits=man.get("nbits", 8))
        with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tf:
            tf.write(s3.download_bytes(vector_bucket, s3.index_file_key(index, "faiss")))
            path = tf.name
        try:
            bk.load(path)
        finally:
            os.unlink(path)
        ids, dists = bk.search(q, topk=topk, nprobe=nprobe)

    out = []
    for i, d in zip(ids, dists):
        if i < 0: continue
        if i >= idmap_tbl.num_rows: continue
        if not alive[i]: continue
        out.append((int(i), float(d)))
    return out

def get_vectors_by_ids(vector_bucket: str, index: str, ids: List[int]) -> List[Dict[str, Any]]:
    s3 = S3Storage()
    tbl = _load_idmap(s3, vector_bucket, index)
    if tbl is None: return []
    keys = tbl["key"].to_pylist()
    vecs = tbl["vec"].to_pylist()
    metas = tbl["meta"].to_pylist()
    alive = tbl["alive"].to_pylist()
    out = []
    for i in ids:
        if i < 0 or i >= len(keys): continue
        if not alive[i]: continue
        out.append({"Key": keys[i], "Data": {"float32": vecs[i]}, "Metadata": json.loads(metas[i])})
    return out

def get_vectors_by_keys(vector_bucket: str, index: str, keys: List[str]) -> List[Dict[str, Any]]:
    s3 = S3Storage()
    tbl = _load_idmap(s3, vector_bucket, index)
    if tbl is None: return []
    k2i = {k: i for i, k in enumerate(tbl["key"].to_pylist())}
    ids = [k2i.get(k, -1) for k in keys]
    return get_vectors_by_ids(vector_bucket, index, [i for i in ids if i >= 0])

def delete_by_keys(vector_bucket: str, index: str, keys: List[str]) -> int:
    s3 = S3Storage()
    tbl = _load_idmap(s3, vector_bucket, index)
    if tbl is None: return 0
    k2i = {k: i for i, k in enumerate(tbl["key"].to_pylist())}
    to_kill = [k2i.get(k) for k in keys if k in k2i]
    if not to_kill: return 0
    # flip alive to False and write back
    alive = tbl["alive"].to_pylist()
    for i in to_kill:
        if i is not None and 0 <= i < len(alive):
            alive[i] = False
    import pyarrow as pa, pyarrow.parquet as pq, io
    new_tbl = tbl.set_column(tbl.schema.get_field_index("alive"), "alive", pa.array(alive, type=pa.bool_()))
    _write_idmap(s3, vector_bucket, index, new_tbl)
    return len(to_kill)

def list_vectors(vector_bucket: str, index: str, max_results: int, next_token: Optional[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    s3 = S3Storage()
    tbl = _load_idmap(s3, vector_bucket, index)
    if tbl is None or tbl.num_rows == 0: return [], None
    start = int(next_token or 0)
    end = min(tbl.num_rows, start + max_results)
    keys = tbl["key"].to_pylist()[start:end]
    metas = [json.loads(m) for m in tbl["meta"].to_pylist()[start:end]]
    alive = tbl["alive"].to_pylist()[start:end]
    vecs = [{"Key": k, "Metadata": md} for k, md, a in zip(keys, metas, alive) if a]
    nxt = str(end) if end < tbl.num_rows else None
    return vecs, nxt
