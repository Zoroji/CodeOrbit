from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx
import numpy as np

from services.pg_vector_store import PgVectorStore
from services.BM25K import BM25Index
from services.reranker import reciprocal_rank_fusion, Reranker

_ENTITY_LABELS = {
    "FILE", "CLASS", "FUNCTION", "METHOD", "INTERFACE",
    "GLOBAL", "ENTRYPOINT", "IMPORT", "SCHEMA", "TYPE_DEF",
    "ENDPOINT", "TEST", "UI_COMPONENT", "STYLESHEET",
    "EXTERNAL_PKG", "DOCSTRING",
}

_GRAPH_FILE = "graph.gml"


class GraphRAG:
    def __init__(
        self,
        persist_dir: str = "./graph_store",
        top_k_retrieve: int = 20,
        top_n_rerank: int = 5,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.top_k_retrieve = top_k_retrieve
        self.top_n_rerank   = top_n_rerank

        self.vector_store = PgVectorStore()
        self.bm25_index   = BM25Index()
        self.reranker     = Reranker()

        self.graph: nx.DiGraph                  = nx.DiGraph()
        self._entity_to_chunks: Dict[str, set]  = {}
        self._all_chunks: List[Dict[str, Any]]  = []
        self._ingested_sources: Dict[str, bool] = {}
        self._nlp                               = None

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError("Run: python -m spacy download en_core_web_sm")
        return self._nlp

    def extract_entities(self, text: str) -> List[str]:
        nlp = self._get_nlp()
        doc = nlp(text)
        seen, entities = set(), []
        for ent in doc.ents:
            if ent.label_ in _ENTITY_LABELS:
                name = ent.text.strip().lower()
                if len(name) > 2 and not re.match(r"^\d+[\d\s,\.]*$", name) and name not in seen:
                    seen.add(name)
                    entities.append(name)
        return entities

    def _chunk_key(self, chunk: Dict[str, Any]) -> str:
        meta = chunk.get("metadata", {})
        return meta.get("chunk_id") or f"{meta.get('file','')}:{meta.get('name','')}:{meta.get('start_line','')}"

    def _build_graph_from_chunks(self, chunks: List[Dict[str, Any]]):
        for chunk in chunks:
            meta     = chunk.get("metadata", {})
            ckey     = self._chunk_key(chunk)
            ctype    = meta.get("type", "GLOBAL").upper()
            name     = meta.get("name", ckey)
            file_    = meta.get("file", "")
            parent   = meta.get("parent")

            if not self.graph.has_node(ckey):
                self.graph.add_node(ckey, label=ctype, name=name, file=file_, chunk_id=meta.get("chunk_id", ""))

            if file_ and not self.graph.has_node(file_):
                self.graph.add_node(file_, label="FILE", name=file_)
            if file_:
                self.graph.add_edge(file_, ckey, relation="DEFINES")

            if parent:
                parent_key = f"{file_}:{parent}"
                if not self.graph.has_node(parent_key):
                    self.graph.add_node(parent_key, label="CLASS", name=parent, file=file_)
                self.graph.add_edge(parent_key, ckey, relation="CONTAINS")

            imports = meta.get("imports", [])
            for imp in imports:
                if not self.graph.has_node(imp):
                    self.graph.add_node(imp, label="IMPORT", name=imp)
                self.graph.add_edge(ckey, imp, relation="IMPORTS")

            for entity in self.extract_entities(chunk.get("code", "")):
                if entity not in self._entity_to_chunks:
                    self._entity_to_chunks[entity] = set()
                self._entity_to_chunks[entity].add(ckey)

    def ingest(self, chunks: List[Dict[str, Any]], source_id: str):
        if self._ingested_sources.get(source_id):
            print(f"Source '{source_id}' already ingested. Skipping.")
            return

        self.vector_store.index_chunks(chunks, source_id)
        self.bm25_index.build(chunks)
        self._all_chunks.extend(chunks)
        self._build_graph_from_chunks(chunks)
        self._ingested_sources[source_id] = True
        self._save_graph()

    def _save_graph(self):
        path = self.persist_dir / _GRAPH_FILE
        nx.write_gml(self.graph, str(path))

    def load_graph(self):
        path = self.persist_dir / _GRAPH_FILE
        if path.exists():
            self.graph = nx.read_gml(str(path))

    def _get_neighboring_chunks(self, center_keys: List[str], hops: int = 1) -> List[str]:
        neighbors = set(center_keys)
        for key in center_keys:
            if self.graph.has_node(key):
                for _ in range(hops):
                    for nbr in list(nx.neighbors(self.graph, key)):
                        neighbors.add(nbr)
        return list(neighbors)

    def query(self, question: str, source_id: Optional[str] = None, top_n: Optional[int] = None) -> List[Tuple[str, Dict[str, Any], float]]:
        top_n = top_n or self.top_n_rerank

        dense_hits  = self.vector_store.dense_search(question, source_id=source_id, top_k=self.top_k_retrieve)
        sparse_hits = self.bm25_index.search(question, top_k=self.top_k_retrieve)
        fused       = reciprocal_rank_fusion(dense_hits, sparse_hits)

        seed_keys = [
            self._chunk_key({"metadata": meta})
            for _, meta, _ in fused[:self.top_k_retrieve]
        ]
        expanded_keys = self._get_neighboring_chunks(seed_keys, hops=1)

        chunk_map = {self._chunk_key(c): c for c in self._all_chunks}
        expanded_candidates = [
            (chunk_map[k]["code"], chunk_map[k]["metadata"], 0.0)
            for k in expanded_keys if k in chunk_map
        ]

        candidates = (expanded_candidates or fused)[:self.top_k_retrieve]
        return self.reranker.rerank(question, candidates, top_n=top_n)

