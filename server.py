import os
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import tempfile
import shutil

from services.zipball import extract_github_zipball, parse_github_url
from services.chunker import chunk_repository

load_dotenv()

app = FastAPI(
    title="CodeOrbit API",
    description="Developer intelligence, polyglot code chunking, and change-impact analysis engine",
    version="1.0.0"
)

# Enable CORS for React frontend (Vite / Next.js / Create React App)
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


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "CodeOrbit Backend"}


@app.get("/api/ingest/stream")
async def stream_ingest(url: str = Query(..., description="Public GitHub repository URL")):
    """
    Server-Sent Events (SSE) streaming endpoint.
    Downloads, extracts, and semantically chunks a repository with real-time progress updates.
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

            # 4. Handoff to Indexing (Paused here for next milestone)
            yield f"event: progress\ndata: {json.dumps({'status': 'indexing', 'message': 'Ready for vector & graph indexing...', 'repo_id': repo_id})}\n\n"

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_excludes=["storage/*", "storage/**", "*.json", "test_extracted_repo/*"]
    )
