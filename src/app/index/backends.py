from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

class IndexBackend(ABC):
    @abstractmethod
    def build(self, X: np.ndarray) -> None: ...
    @abstractmethod
    def add(self, X: np.ndarray, ids: np.ndarray) -> None: ...
    @abstractmethod
    def search(self, q: np.ndarray, topk: int, nprobe: Optional[int]=None) -> Tuple[np.ndarray, np.ndarray]: ...
