import os
import sys
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import tempfile
import shutil

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from services.zipball import extract_github_zipball, parse_github_url
from services.chunker import chunk_repository
from services.pg_vector_store import PgVectorStore

load_dotenv()

app = FastAPI(
    title="CodeOrbit API",
    description="Developer intelligence, polyglot code chunking, and change-impact analysis engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_DIR = Path(tempfile.gettempdir()) / "codeorbit"
REPOS_DIR = STORAGE_DIR / "repos"
REPOS_DIR.mkdir(parents=True, exist_ok=True)

# Production connection pool vector store
vector_store = PgVectorStore()


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "CodeOrbit Backend"}


@app.get("/api/search")
async def search(
    query: str = Query(..., description="Natural language search query"),
    repo_id: str | None = Query(None, description="Optional repository id filter"),
    top_k: int = Query(5, description="Number of top matches to return")
):
    results = await asyncio.to_thread(vector_store.dense_search, query, repo_id, top_k)
    return [
        {"code": code, "metadata": meta, "similarity": round(float(score), 4)}
        for code, meta, score in results
    ]


@app.get("/api/ingest/stream")
async def stream_ingest(url: str = Query(..., description="Public GitHub repository URL")):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Downloads, extracts, semantically chunks, and indexes a repository into pgvector with real-time updates.
    """
    async def event_generator():
        try:
            # 1. Validation & Parsing
            try:
                owner, repo = parse_github_url(url)
                repo_id = f"{owner}_{repo}"
            except ValueError as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                return

            # 2. Extracting Zipball
            yield f"event: progress\ndata: {json.dumps({'status': 'extracting', 'message': f'Connecting to GitHub for {owner}/{repo}...', 'repo_id': repo_id})}\n\n"

            try:
                dest_dir, owner, repo, was_cached = await asyncio.to_thread(
                    extract_github_zipball, url, REPOS_DIR
                )
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': 'Failed to download repository', 'details': str(e)})}\n\n"
                return

            status_msg = "Loaded from local cache" if was_cached else "Extracted code files to disk"
            yield f"event: progress\ndata: {json.dumps({'status': 'extracted', 'message': status_msg, 'cached': was_cached, 'repo_id': repo_id})}\n\n"

            # 3. Tree-sitter Polyglot Semantic Chunking
            yield f"event: progress\ndata: {json.dumps({'status': 'chunking', 'message': 'Parsing syntax tree with Tree-sitter...', 'repo_id': repo_id})}\n\n"

            chunks = await asyncio.to_thread(
                chunk_repository, dest_dir, owner, repo, 500
            )

            # Delete temp folder
            await asyncio.to_thread(shutil.rmtree, dest_dir, ignore_errors=True)

            yield f"event: progress\ndata: {json.dumps({'status': 'chunked', 'message': f'Generated {len(chunks)} chunks', 'total_chunks': len(chunks), 'repo_id': repo_id})}\n\n"

            # 4. Vector Embedding & Indexing into pgvector
            yield f"event: progress\ndata: {json.dumps({'status': 'indexing', 'message': f'Embedding & storing {len(chunks)} chunks in pgvector...', 'repo_id': repo_id})}\n\n"

            await asyncio.to_thread(vector_store.index_chunks, chunks, repo_id)

            yield f"event: progress\ndata: {json.dumps({'status': 'indexed', 'message': f'Indexed into pgvector with CodeRankEmbed', 'repo_id': repo_id})}\n\n"

            # 5. Complete
            yield f"event: complete\ndata: {json.dumps({'status': 'completed', 'repo_id': repo_id, 'total_chunks': len(chunks)})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': 'Unexpected ingestion failure', 'details': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


def run_dry_run(target_url: str = "https://github.com/Zoroji/Graph-RAG.git"):
    """
    Direct dry-run pipeline:
    Downloads repo -> Polyglot chunks -> Embeds with CodeRankEmbed -> Stores in pgvector -> Executes a dense search.
    """
    print("=" * 70)
    print(f"🚀 CodeOrbit Ingestion Dry-Run Pipeline")
    print(f"📂 Target: {target_url}")
    print("=" * 70)

    owner, repo = parse_github_url(target_url)
    repo_id = f"{owner}_{repo}"

    print(f"\n[1/4] Extracting repository {owner}/{repo}...")
    dest_dir, _, _, was_cached = extract_github_zipball(target_url, REPOS_DIR)
    print(f"   ✓ Extracted to {dest_dir} (cached: {was_cached})")

    print(f"\n[2/4] Chunking repository with Tree-sitter...")
    chunks = chunk_repository(dest_dir, owner, repo, 500)
    print(f"   ✓ Created {len(chunks)} semantic chunks")

    shutil.rmtree(dest_dir, ignore_errors=True)

    print(f"\n[3/4] Indexing chunks into pgvector (ConnectionPool)...")
    vector_store.index_chunks(chunks, repo_id)
    print(f"   ✓ Stored in PostgreSQL with CodeRankEmbed embeddings")

    print(f"\n[4/4] Testing Dense Semantic Search...")
    sample_query = "How is knowledge graph constructed or retrieved?"
    print(f"   🔍 Query: '{sample_query}'")
    hits = vector_store.dense_search(sample_query, source_id=repo_id, top_k=3)

    print(f"\n✨ Top {len(hits)} Search Results from pgvector:")
    print("-" * 70)
    for i, (code, meta, score) in enumerate(hits, 1):
        print(f"[{i}] Similarity: {score:.4f} | Type: {meta.get('type')} | Name: {meta.get('name')}")
        print(f"    File: {meta.get('file')} (lines {meta.get('start_line')}-{meta.get('end_line')})")
        preview = code.strip().splitlines()
        first_few = "\n".join(f"      {line}" for line in preview[:4])
        print(first_few)
        if len(preview) > 4:
            print(f"      ... ({len(preview) - 4} more lines)")
        print("-" * 70)

    print("\n✅ Dry run completed successfully!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--dry-run", "dry-run"):
        url = sys.argv[2] if len(sys.argv) > 2 else "https://github.com/Zoroji/Graph-RAG.git"
        run_dry_run(url)
    else:
        import uvicorn
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_excludes=["storage/*", "storage/**", "*.json", "test_extracted_repo/*"]
        )
