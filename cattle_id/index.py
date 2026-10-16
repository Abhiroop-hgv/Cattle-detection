from pathlib import Path
from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    def add(self, cattle_id: str, vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray) -> list[tuple[str, float]]:
        """Best cosine score per cattle_id, sorted descending."""
        ...
    def save(self) -> None: ...


class NumpyIndex:
    """Exact cosine search. Fine up to ~100k vectors; swap for Vertex AI Vector Search later
    by implementing the same three methods."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.ids: list[str] = []
        self.vecs = np.zeros((0, 0), dtype=np.float32)
        if path and Path(path).exists():
            d = np.load(path, allow_pickle=False)
            self.vecs, self.ids = d["vecs"], list(d["ids"])

    def add(self, cattle_id: str, vectors: np.ndarray) -> None:
        vectors = np.atleast_2d(vectors).astype(np.float32)
        self.vecs = vectors if self.vecs.size == 0 else np.vstack([self.vecs, vectors])
        self.ids += [cattle_id] * len(vectors)

    def search(self, vector: np.ndarray) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        sims = self.vecs @ (vector / np.linalg.norm(vector))
        best: dict[str, float] = {}
        for cid, s in zip(self.ids, sims):
            best[cid] = max(best.get(cid, -1.0), float(s))
        return sorted(best.items(), key=lambda kv: -kv[1])

    def save(self) -> None:
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.path, vecs=self.vecs, ids=np.array(self.ids))
