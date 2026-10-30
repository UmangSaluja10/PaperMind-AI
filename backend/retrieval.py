"""Per-paper FAISS index (inner product over normalised vectors == cosine)."""
import faiss
import numpy as np

import config


class PaperIndex:
    def __init__(self, paper_id, passages, index):
        self.paper_id = paper_id
        self.passages = passages
        self.index = index
        self._vectors = None

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = self.index.reconstruct_n(0, self.index.ntotal)
        return self._vectors

    @classmethod
    def build(cls, paper_id, passages, vectors: np.ndarray):
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        faiss.write_index(index, str(config.INDEX_DIR / f"{paper_id}.faiss"))
        obj = cls(paper_id, passages, index)
        obj._vectors = vectors
        return obj

    @classmethod
    def load(cls, paper_id, passages):
        index = faiss.read_index(str(config.INDEX_DIR / f"{paper_id}.faiss"))
        return cls(paper_id, passages, index)

    def search(self, query_vec: np.ndarray, k: int = 5):
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        scores, ids = self.index.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [
            {**self.passages[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i >= 0
        ]

    def delete_file(self):
        (config.INDEX_DIR / f"{self.paper_id}.faiss").unlink(missing_ok=True)
