from __future__ import annotations
import os
import json
from typing import List, Tuple, Optional
import numpy as np
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector # this tells how to serialize and deserialize the data between py and psql

#fist get the extension for psql 
INIT_SQL = """
CREATE EXTENSION IF NOT EXISTS vector; 
CREATE TABLE IF NOT EXISTS code_chunks (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    chunk_type  TEXT,
    name        TEXT,
    file        TEXT,
    start_line  INTEGER,
    end_line    INTEGER,
    token_count INTEGER,
    code        TEXT,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(768)
);
CREATE INDEX IF NOT EXISTS idx_chunks_hnsw
    ON code_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_source
    ON code_chunks (source);
"""

UPSERT_SQL = """
INSERT INTO code_chunks (id, source, chunk_type, name, file, start_line, end_line, token_count, code, metadata, embedding)
VALUES (%(id)s, %(source)s, %(chunk_type)s, %(name)s, %(file)s, %(start_line)s, %(end_line)s, %(token_count)s, %(code)s, %(metadata)s, %(embedding)s)
ON CONFLICT (id) DO UPDATE SET
    code      = EXCLUDED.code,
    metadata  = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding;
"""

SEARCH_SQL = """
SELECT id, code, metadata, 1 - (embedding <=> %(query)s) AS similarity
FROM code_chunks
WHERE (%(source)s::text IS NULL OR source = %(source)s)
ORDER BY embedding <=> %(query)s
LIMIT %(top_k)s;
"""

COUNT_SQL = "SELECT count(*) FROM code_chunks WHERE (%(source)s::text IS NULL OR source = %(source)s);"

SOURCE_EXISTS_SQL = "SELECT 1 FROM code_chunks WHERE source = %(source)s LIMIT 1;"


def _configure_connection(conn):
    register_vector(conn)


class PgVectorStore:
    def __init__(
        self,
        conninfo: Optional[str] = None,
        min_size: int = 1,
        max_size: int = 10
    ):
        self._conninfo = conninfo or os.getenv(
            "DATABASE_URL",
            "postgresql://codeorbit:codeorbit@localhost:5433/codeorbit"
        )
        self._pool = ConnectionPool(
            conninfo=self._conninfo,
            min_size=min_size,
            max_size=max_size,
            configure=_configure_connection,
            open=True
        )
        self._embedder = None
        self._init_schema()

    def _init_schema(self):
        with self._pool.connection() as conn:
            conn.execute(INIT_SQL)

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print("Loading CodeRankEmbed model...")
            self._embedder = SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True)
        return self._embedder

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        model = self._get_embedder()
        prefixed = [f"search_document: {t}" for t in texts]
        return model.encode(prefixed, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        model = self._get_embedder()
        return model.encode(f"search_query: {query}", normalize_embeddings=True)

    def index_chunks(self, chunks: List[dict], source_id: str):
        # ponytail: filter out micro/empty chunks before embedding. Upgrade: pre-validation schema.
        chunks = [
            c for c in (chunks or [])
            if len(c.get("code", "").strip()) >= 10 and any(ch.isalnum() for ch in c.get("code", ""))
        ]
        if not chunks:
            return

        with self._pool.connection() as conn:
            row = conn.execute(SOURCE_EXISTS_SQL, {"source": source_id}).fetchone()
            if row:
                print(f"Source '{source_id}' already indexed. Skipping.")
                return

        codes = [c["code"] for c in chunks]
        print(f"Embedding {len(codes)} chunks for '{source_id}'...")
        embeddings = self.embed_texts(codes)

        with self._pool.connection() as conn:
            with conn.transaction(): #ACID principle
                with conn.cursor() as cur:
                    for i, chunk in enumerate(chunks):
                        meta = chunk["metadata"]
                        cur.execute(UPSERT_SQL, {
                            "id":          meta["chunk_id"],
                            "source":      source_id,
                            "chunk_type":  meta.get("type"),
                            "name":        meta.get("name"),
                            "file":        meta.get("file"),
                            "start_line":  meta.get("start_line"),
                            "end_line":    meta.get("end_line"),
                            "token_count": meta.get("chunk_size"),
                            "code":        chunk["code"],
                            "metadata":    json.dumps(meta),
                            "embedding":   embeddings[i],
                        })
        print(f"Indexed {len(chunks)} chunks into pgvector for '{source_id}'.")

    def dense_search(
        self,
        query: str,
        source_id: Optional[str] = None,
        top_k: int = 20
    ) -> List[Tuple[str, dict, float]]:
        with self._pool.connection() as conn:
            count = conn.execute(COUNT_SQL, {"source": source_id}).fetchone()[0]
            if count == 0:
                return []

            query_emb = self.embed_query(query)
            rows = conn.execute(
                SEARCH_SQL,
                {"query": query_emb, "source": source_id, "top_k": min(top_k, count)}
            ).fetchall()
            return [(row[1], row[2], row[3]) for row in rows]

    def close(self):
        if self._pool:
            self._pool.close()
