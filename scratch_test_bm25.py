import sys
from pathlib import Path
from services.BM25K import BM25Index
from services.chunker import chunk_repository

def main():
    repo_path = Path("test_extracted_repo/Zoroji_Graph-RAG")
    if not repo_path.exists():
        print("Repo not found")
        return

    chunks = chunk_repository(repo_path, "Zoroji", "Graph-RAG")
    print(f"Total chunks: {len(chunks)}")

    idx = BM25Index()
    idx.build(chunks)

    query = "How is knowledge graph constructed or retrieved?"
    hits = idx.search(query, top_k=5)
    print(f"BM25 Search Hits for '{query}': {len(hits)}")
    for i, (code, meta, score) in enumerate(hits, 1):
        print(f"[{i}] Score: {score:.4f} | Name: {meta.get('name')} | File: {meta.get('file')} (lines {meta.get('start_line')}-{meta.get('end_line')})")

if __name__ == "__main__":
    main()
