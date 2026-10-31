from typing import List, Tuple, Dict, Any


def reciprocal_rank_fusion(
    dense: List[Tuple[str, Dict[str, Any], float]],
    sparse: List[Tuple[str, Dict[str, Any], float]],
    k: int = 60,
    dw: float = 0.6,
    sw: float = 0.4
) -> List[Tuple[str, Dict[str, Any], float]]:
    scores, docs = {}, {}
    for rank, (text, meta, _) in enumerate(dense):
        cid = meta.get("chunk_id") or f"{meta.get('file', '')}:{meta.get('name', '')}:{rank}"
        scores[cid] = scores.get(cid, 0.0) + dw / (k + rank + 1)
        docs[cid] = (text, meta)
    for rank, (text, meta, _) in enumerate(sparse):
        cid = meta.get("chunk_id") or f"{meta.get('file', '')}:{meta.get('name', '')}:{rank}"
        scores[cid] = scores.get(cid, 0.0) + sw / (k + rank + 1)
        docs[cid] = (text, meta)
    return [
        (docs[c][0], docs[c][1], scores[c])
        for c in sorted(scores, key=lambda x: scores[x], reverse=True)
    ]


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self._model = None
        self.model_name = model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, Dict[str, Any], float]],
        top_n: int = 5
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        if not candidates:
            return []
        self._load()
        scores = self._model.predict([(query, t) for t, _, _ in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [(t, m, float(s)) for s, (t, m, _) in ranked[:top_n]]

