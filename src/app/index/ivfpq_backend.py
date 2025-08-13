import numpy as np
from typing import Optional, Tuple
from sklearn.cluster import KMeans

from .backends import IndexBackend

class IVFPQSim(IndexBackend):
    def __init__(self, metric: str = "cosine", nlist: int = 1024, m: int = 16, nbits: int = 8) -> None:
        self.metric = metric
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.coarse = None  # kmeans for IVF
        self.codebooks = None  # PQ codebooks per sub-vector
        self.lists = {}   # list_id -> (codes, ids, residual_means)
        self.d = None

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        return X.astype(np.float32, copy=False)

    def build(self, X: np.ndarray) -> None:
        X = self._normalize(X); self.d = X.shape[1]
        nl = min(self.nlist, max(1, X.shape[0]//39))  # training heuristic
        self.coarse = KMeans(n_clusters=nl, n_init=3, random_state=0).fit(X)
        # simple PQ: split dims into m parts and k=2^nbits centers per part
        subdim = self.d // self.m
        self.codebooks = []
        for i in range(self.m):
            part = X[:, i*subdim:(i+1)*subdim]
            k = min(2**self.nbits, max(2, part.shape[0]))
            self.codebooks.append(KMeans(n_clusters=k, n_init=2, random_state=0).fit(part))
        # assign all points
        self.add(X, np.arange(X.shape[0], dtype=np.int64))

    def _encode(self, X: np.ndarray):
        subdim = self.d // self.m
        codes = []
        for i, cb in enumerate(self.codebooks):
            part = X[:, i*subdim:(i+1)*subdim]
            codes.append(cb.predict(part).astype(np.int32))
        return np.stack(codes, axis=1)  # (N, m)

    def add(self, X: np.ndarray, ids: np.ndarray) -> None:
        X = self._normalize(X)
        coarse_ids = self.coarse.predict(X)
        codes = self._encode(X)
        for ci in np.unique(coarse_ids):
            mask = coarse_ids == ci
            sub = codes[mask]
            ids_sub = ids[mask]
            if ci not in self.lists:
                self.lists[ci] = (sub, ids_sub)
            else:
                oldc, oldi = self.lists[ci]
                self.lists[ci] = (np.vstack([oldc, sub]), np.concatenate([oldi, ids_sub]))

    def _dist_code(self, q: np.ndarray, code_row: np.ndarray) -> float:
        # simplified asymmetric distance: use PQ centers to approx dist
        subdim = self.d // self.m
        acc = 0.0
        for i, cb in enumerate(self.codebooks):
            center = cb.cluster_centers_[code_row[i]]
            part = q[i*subdim:(i+1)*subdim]
            if self.metric == "cosine":
                denom = (np.linalg.norm(center)*np.linalg.norm(part) + 1e-9)
                sim = (center @ part)/denom
                acc += (1.0 - sim)
            else:
                acc += np.sum((center - part)**2)
        return float(acc)

    def search(self, q: np.ndarray, topk: int, nprobe: Optional[int]=None):
        q = self._normalize(q)
        nprobe = nprobe or min(8, len(self.lists) or 1)
        # find closest coarse centroids
        cents = self.coarse.cluster_centers_
        if self.metric == "cosine":
            cq = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
            cc = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
            dcoarse = 1.0 - (cq @ cc.T)
        else:
            a2 = np.sum(q*q, axis=1, keepdims=True)
            b2 = np.sum(cents*cents, axis=1, keepdims=True).T
            dcoarse = a2 + b2 - 2*q@cents.T
        probe = np.argpartition(dcoarse[0], nprobe)[:nprobe]
        # scan probed lists
        cand = []
        for li in probe:
            if li not in self.lists: continue
            codes, ids = self.lists[li]
            for row, idv in zip(codes, ids):
                d = self._dist_code(q[0], row)
                cand.append((d, idv))
        cand.sort(key=lambda x: x[0])
        top = cand[:topk]
        if not top:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        ids = np.array([t[1] for t in top], dtype=np.int64)
        dist = np.array([t[0] for t in top], dtype=np.float32)
        return ids, dist
