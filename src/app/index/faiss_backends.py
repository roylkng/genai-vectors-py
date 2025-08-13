import numpy as np
import faiss
import hnswlib
from typing import Optional, Tuple

def _normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / norms

class HNSWBackend:
    def __init__(self, dim: int, metric: str = "cosine", ef_construction: int = 200, M: int = 16):
        space = "cosine" if metric == "cosine" else "l2"
        self.index = hnswlib.Index(space=space, dim=dim)
        self.ef_construction = ef_construction
        self.M = M
        self.dim = dim
        self.metric = metric
        self._count = 0
        self.index.init_index(max_elements=1, ef_construction=ef_construction, M=M)  # will grow

    def build(self, X: np.ndarray, ids: np.ndarray):
        self.index.resize_index(X.shape[0])
        self.index.add_items(X, ids)
        self._count = X.shape[0]

    def add(self, X: np.ndarray, ids: np.ndarray):
        tgt = self._count + X.shape[0]
        self.index.resize_index(tgt)
        self.index.add_items(X, ids)
        self._count = tgt

    def save(self, path: str):
        self.index.save_index(path)

    def load(self, path: str):
        self.index.load_index(path, max_elements=self._count or 1)

    def search(self, q: np.ndarray, topk: int, nprobe: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        self.index.set_ef(max(topk * 2, 32))
        lbls, dists = self.index.knn_query(q, k=topk)
        return lbls[0].astype(np.int64), dists[0].astype(np.float32)

class IVFPQBackend:
    def __init__(self, dim: int, metric: str = "cosine", nlist: int = 1024, m: int = 16, nbits: int = 8):
        self.dim = dim
        self.metric = metric
        self.nlist = max(1, nlist)
        self.m = max(1, m)
        self.nbits = max(4, nbits)
        self.metric_type = faiss.METRIC_INNER_PRODUCT if metric == "cosine" else faiss.METRIC_L2
        self.quantizer = faiss.IndexFlatIP(dim) if metric == "cosine" else faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFPQ(self.quantizer, dim, self.nlist, self.m, self.nbits, self.metric_type)
        self.trained = False
        self.nprobe = 8

    def train(self, X: np.ndarray):
        if self.metric == "cosine":
            X = _normalize_rows(X)
        if not self.index.is_trained:
            self.index.train(X)
        self.trained = True

    def add(self, X: np.ndarray, ids: np.ndarray):
        if self.metric == "cosine":
            X = _normalize_rows(X)
        if not self.trained: self.train(X)
        self.index.add_with_ids(X, ids.astype(np.int64))

    def build(self, X: np.ndarray, ids: np.ndarray):
        if self.metric == "cosine":
            X = _normalize_rows(X)
        self.train(X)
        self.index.add_with_ids(X, ids.astype(np.int64))

    def set_nprobe(self, nprobe: int):
        self.index.nprobe = int(nprobe)

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)

    def search(self, q: np.ndarray, topk: int, nprobe: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        if self.metric == "cosine":
            q = _normalize_rows(q)
        if nprobe is not None: self.index.nprobe = int(nprobe)
        D, I = self.index.search(q, topk)
        return I[0].astype(np.int64), D[0].astype(np.float32)
