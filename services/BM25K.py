import re
import numpy as np
from typing import List, Tuple, Dict, Any
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self.bm25k = None
        self.chunks: List[Dict[str, Any]] = []

    def _tok(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def build(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            return
        self.chunks = list(chunks)
        tokenized_corpus = [self._tok(c.get("code", "")) for c in self.chunks]
        self.bm25k = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, Dict[str, Any], float]]:
        if self.bm25k is None or not self.chunks:
            return []
        tokenized_query = self._tok(query)
        ranks = self.bm25k.get_scores(tokenized_query)
        top_rank = np.argsort(ranks)[::-1][:top_k]
        return [
            (self.chunks[i]["code"], self.chunks[i].get("metadata", {}), float(ranks[i]))
            for i in top_rank if ranks[i] > 0
        ]
