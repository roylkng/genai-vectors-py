import numpy as np
from typing import Optional, Tuple
from .backends import IndexBackend

class HNSWFlat(IndexBackend):
    def __init__(self, metric: str = "cosine") -> None:
        self.metric = metric
        self.X = None  # (N, d)
        self.ids = None  # (N,)

    def build(self, X: np.ndarray) -> None:
        self.X = X.astype(np.float32, copy=False)
        self.ids = np.arange(self.X.shape[0], dtype=np.int64)

    def add(self, X: np.ndarray, ids: np.ndarray) -> None:
        if self.X is None:
            self.build(X); self.ids = ids
            return
        self.X = np.vstack([self.X, X])
        self.ids = np.concatenate([self.ids, ids])

    def _dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
            return 1.0 - (A @ B.T)
        else:
            # Euclidean
            a2 = np.sum(A*A, axis=1, keepdims=True)
            b2 = np.sum(B*B, axis=1, keepdims=True).T
            return a2 + b2 - 2*A@B.T

    def search(self, q: np.ndarray, topk: int, nprobe: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        D = self._dist(q, self.X)  # (1, N)
        idx = np.argpartition(D[0], topk)[:topk]
        idx_sorted = idx[np.argsort(D[0, idx])]
        return self.ids[idx_sorted], D[0, idx_sorted]
