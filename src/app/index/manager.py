import threading
import numpy as np
from typing import List, Tuple, Dict, Optional

from .faiss_backends import HNSWBackend, IVFPQBackend

class IndexManager:
    """
    In-memory manager:
      - maps external keys -> int IDs
      - stores vectors + metadata arrays
      - supports hybrid backend selection based on total count
      - logical deletes via alive mask
    """
    def __init__(self, dim: int, metric: str, algorithm: str,
                 hnsw_threshold: int|None = None, nlist: int|None = None, m: int|None = None, nbits: int|None = None):
        self.dim = dim
        self.metric = metric
        self.algorithm = algorithm
        # Set a default threshold if not provided (AWS s3vectors does not specify)
        if hnsw_threshold is None:
            self.hnsw_threshold = 10000
        else:
            self.hnsw_threshold = hnsw_threshold
        self.nlist = nlist or 1024
        self.m = m or 16
        self.nbits = nbits or 8

        self._lock = threading.RLock()

        self._key_to_id: Dict[str, int] = {}
        self._id_to_key: List[str] = []
        self._vecs: Optional[np.ndarray] = None
        self._meta: List[dict] = []
        self._alive: List[bool] = []
        self._next_id = 0

        self.backend = None

    def _choose_backend(self, total: int):
        if self.backend is None:
            if self.algorithm == "hnsw_flat" or (self.algorithm == "hybrid" and total < self.hnsw_threshold):
                self.backend = HNSWBackend(self.dim, metric=self.metric)
            else:
                self.backend = IVFPQBackend(self.dim, metric=self.metric, nlist=self.nlist, m=self.m, nbits=self.nbits)

    def add_batch(self, batch: List[Tuple[str, List[float], dict]]):
        with self._lock:
            new_ids = []
            new_vecs = []
            new_meta = []
            for k, vec, md in batch:
                if k in self._key_to_id:
                    # overwrite: mark old deleted and append as new version (simple approach)
                    old_id = self._key_to_id[k]
                    self._alive[old_id] = False
                idv = self._next_id
                self._next_id += 1
                self._key_to_id[k] = idv
                self._id_to_key.append(k)
                new_ids.append(idv)
                new_vecs.append(vec)
                new_meta.append(md)
                self._alive.append(True)
            new_vecs = np.asarray(new_vecs, dtype=np.float32).reshape(len(new_vecs), self.dim)
            self._meta.extend(new_meta)
            # append vectors
            if self._vecs is None:
                self._vecs = new_vecs
            else:
                self._vecs = np.vstack([self._vecs, new_vecs])
            # build or add to backend
            total = self._vecs.shape[0]
            self._choose_backend(total)
            if hasattr(self.backend, "build") and total == len(new_ids):
                self.backend.build(self._vecs, np.asarray(list(range(total)), dtype=np.int64))
            else:
                self.backend.add(new_vecs, np.asarray(new_ids, dtype=np.int64))

    def delete_keys(self, keys: List[str]) -> int:
        removed = 0
        with self._lock:
            for k in keys:
                idv = self._key_to_id.get(k)
                if idv is not None and self._alive[idv]:
                    self._alive[idv] = False
                    removed += 1
        return removed

    def get_vectors(self, keys: List[str]) -> List[dict]:
        out = []
        with self._lock:
            for k in keys:
                idv = self._key_to_id.get(k)
                if idv is None or not self._alive[idv]:
                    continue
                vec = self._vecs[idv].tolist()
                out.append({"Key": k, "Data": {"float32": vec}, "Metadata": self._meta[idv]})
        return out

    def list_vectors(self, max_results: int, start_token: Optional[int]) -> Tuple[List[dict], Optional[int]]:
        with self._lock:
            start = int(start_token or 0)
            res = []
            i = start
            while i < len(self._id_to_key) and len(res) < max_results:
                k = self._id_to_key[i]
                if self._alive[i]:
                    res.append({"Key": k, "Metadata": self._meta[i]})
                i += 1
            next_tok = i if i < len(self._id_to_key) else None
            return res, next_tok

    def search(self, q: List[float], topk: int, nprobe: Optional[int] = None) -> List[dict]:
        if self._vecs is None or self._vecs.shape[0] == 0:
            return []
        qv = np.asarray([q], dtype=np.float32)
        ids, dist = self.backend.search(qv, topk=topk, nprobe=nprobe)
        out = []
        for i, d in zip(ids, dist):
            if i < 0 or i >= len(self._alive) or not self._alive[i]:
                continue
            out.append({"key": self._id_to_key[i], "distance": float(d), "metadata": self._meta[i]})
        return out

    def stats(self) -> dict:
        total = 0 if self._vecs is None else self._vecs.shape[0]
        alive = sum(1 for a in self._alive if a)
        return {"total": total, "alive": alive}
