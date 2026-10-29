# CodeOrbit

---

########### 1. EXTRACTING CODE FROM GITHUB URL ###########

We extract source code from public GitHub repositories using GitHub's **Zipball API** instead of a slow `git clone`. This skips cloning heavy `.git` commit histories, large media assets, and package folders.

The compressed archive streams directly into memory (`io.BytesIO`), where only supported code files (`.py`, `.ts`, `.tsx`, `.js`, `.go`, `.rs`, `.java`, etc.) are selectively extracted to temporary SSD storage (`storage/repos/{owner}_{repo}`) for subsequent AST parsing and chunking. A local cache check ensures 0.00s instant access on repeat visits.

### PSEUDO CODE: GITHUB CODE EXTRACTION (ZIPBALL)

```python
START

# STEP 1: PARSE GITHUB URL
READ repo_url from request
VALIDATE format is "https://github.com/{owner}/{repo}"
EXTRACT owner, repo_name
SET repo_id = "{owner}_{repo_name}"
SET target_directory = "storage/repos/{repo_id}"

# STEP 2: CHECK LOCAL SSD CACHE
IF target_directory EXISTS on SSD and contains files:
    RETURN target_directory (Instant 0.00s Cache Hit)

# STEP 3: STREAM ZIPBALL ARCHIVE
SET api_url = "https://api.github.com/repos/{owner}/{repo_name}/zipball"
SET headers = {"Accept": "application/vnd.github+json"}

IF github_token is available:
    ADD "Authorization: Bearer {github_token}" to headers

SEND HTTP GET to api_url (stream=True)
IF response status is NOT 200:
    RETURN error (HTTP 403 Rate Limit or 404 Not Found)

LOAD response.bytes into in-memory buffer (RAM io.BytesIO)

# STEP 4: SELECTIVE EXTRACTION TO SSD
CREATE target_directory on SSD
OPEN zip_archive from in-memory buffer
IDENTIFY root_wrapper_prefix from zip archive

FOR EACH file_entry in zip_archive:
    IF file_entry is directory:
        CONTINUE

    STRIP root_wrapper_prefix from file_entry.path
    SET relative_path = file_entry.path

    # Filter out junk directories (.git, node_modules, venv, dist, build)
    IF relative_path starts with any EXCLUDED_DIRECTORY:
        CONTINUE

    # Filter: retain only valid programming languages and configs
    IF relative_path extension NOT in SUPPORTED_EXTENSIONS:
        CONTINUE

    # Guard: prevent extracting massive binaries or minified bundles (> 2MB)
    IF file_entry.size > 2MB:
        CONTINUE

    # Write clean code file to SSD
    SET destination_file = target_directory / relative_path
    CREATE destination_file parent directories on SSD
    WRITE file_entry bytes to destination_file

# STEP 5: RETURN PATH FOR CHUNKING
RETURN target_directory

END
```

---

########### 2. CHUNKING STRATEGY #######################

We use **Tree-sitter** for polyglot semantic code chunking (Python, JavaScript, TypeScript, TSX). 
Tree-sitter builds a concrete syntax tree (CST) that pinpoints exact byte ranges and line boundaries for every function, class, import, and statement across multiple languages.

### PSEUDO CODE FOR CHUNKING (TREE-SITTER VERSION)

```python
START

# STEP 1: INITIALIZE & PARSER SELECTION
READ source file as bytes
SELECT grammar dynamically from file extension (.py, .js, .jsx, .ts, .tsx)
INITIALIZE Tree-sitter Parser with selected Language
PARSE source_bytes into Tree-sitter Syntax Tree (root_node)

CREATE empty collections:
    imports_nodes
    globals_nodes
    output_chunks

SET TARGET_CHUNK_SIZE = 500 tokens

# STEP 2: PROCESS ROOT SYNTAX NODES
FOR EACH child node in root_node.children (where child is named):

    # A. IMPORTS (Python & JS/TS)
    IF child.type in ["import_statement", "import_from_statement"]:
        ADD child to imports_nodes

    # B. CLASSES (class_definition / class_declaration)
    ELSE IF child.type in ["class_definition", "class_declaration"]:

        EXTRACT class name from child.child_by_field_name("name")
        EXTRACT docstring / leading comments
        EXTRACT class-level data members (attributes / field_definitions)
        FIND method nodes inside class body (function_definition / method_definition)

        # Build interface summary without method implementations
        FOR EACH method node:
            EXTRACT method signature line only (e.g. "def foo(...): ...")
            DO NOT include method body

        CREATE CLASS OVERVIEW chunk containing:
            class name
            docstring / comments
            class-level attributes
            method signature stubs

        ADD CLASS OVERVIEW to output_chunks

        # Store complete method implementations separately
        FOR EACH method node inside class body:
            EXTRACT complete method source code
            ATTACH parent = class name

            IF method tokens <= TARGET_CHUNK_SIZE:
                CREATE METHOD chunk (index 1 of 1)
                ADD to output_chunks
            ELSE:
                SPLIT method statements recursively at block/statement boundaries
                FOR EACH split statement group:
                    CREATE METHOD chunk (index i of total)
                    ADD to output_chunks

    # C. TOP-LEVEL FUNCTIONS (function_definition / function_declaration / arrow_function)
    ELSE IF child.type in ["function_definition", "function_declaration", "arrow_function"]:

        EXTRACT complete function source code

        IF function tokens <= TARGET_CHUNK_SIZE:
            CREATE FUNCTION chunk (index 1 of 1)
            ADD to output_chunks
        ELSE:
            SPLIT function statements recursively at block/statement boundaries
            FOR EACH split statement group:
                CREATE FUNCTION chunk (index i of total)
                ADD to output_chunks

    # D. GLOBAL VARIABLES & CONSTANTS
    ELSE IF child.type in ["expression_statement", "lexical_declaration", "variable_declaration"]:
        IF child represents assignment / variable declaration:
            ADD child to globals_nodes

# STEP 3: CREATE AGGREGATED FILE-LEVEL CHUNKS
IF imports_nodes is not empty:
    COMBINE all import statements into one single IMPORTS chunk
    ADD to output_chunks

IF globals_nodes is not empty:
    COMBINE all global variable assignments into one single GLOBALS chunk
    ADD to output_chunks

# STEP 4: RETURN RESULTS
RETURN output_chunks

END
```

---

## 🛠️ Backend Module 1: Ingestion & Streaming Chunking Pipeline

### 1. Module Overview
This module exposes a real-time **Server-Sent Events (SSE)** streaming ingestion pipeline in FastAPI. It connects to public GitHub repositories, selectively extracts source code without downloading Git history, parses AST/CST with Tree-sitter, and generates semantic chunks ready for vector/graph indexing.

### 2. Files Worked On
* [server.py](file:///e:/2026%20mnc/codeorbit/server.py): FastAPI backend providing CORS and `GET /api/ingest/stream?url=...` SSE streaming endpoint.
* [services/zipball.py](file:///e:/2026%20mnc/codeorbit/services/zipball.py): In-memory ZIP archive streaming, selective filtering (ignoring `.git`, `node_modules`, binaries), and local directory caching.
* [services/chunker.py](file:///e:/2026%20mnc/codeorbit/services/chunker.py): Production Tree-sitter polyglot semantic chunking service with full metadata tracking.
* [requirements.txt](file:///e:/2026%20mnc/codeorbit/requirements.txt): Environment dependencies.

### 3. Metadata Schema Per Chunk
Each generated chunk is tagged with self-contained metadata:
```json
{
  "code": "export function calculateRiskLevel(...) { ... }",
  "metadata": {
    "chunk_id": "sha256_hash_id",
    "file": "src/utils/metrics.ts",
    "name": "calculateRiskLevel",
    "type": "function",
    "parent": null,
    "start_line": 45,
    "end_line": 53,
    "chunk_size": 89,
    "segment_idx": 1,
    "total_chunks": 1,
    "github_developer": "Zoroji",
    "repo_name": "Graph-RAG",
    "github_source_code_url": "https://github.com/Zoroji/Graph-RAG/blob/main/src/utils/metrics.ts#L45-L53"
  }
}
```

### 4. Short Pseudocode (Streaming Pipeline)
```python
START GET /api/ingest/stream?url=...

# 1. Validation & Cache Check
PARSE owner, repo from url
YIELD event: progress -> "extracting"

IF repo already cached in storage/repos/{owner}_{repo}:
    USE cached files
ELSE:
    STREAM zipball from api.github.com/repos/{owner}/{repo}/zipball
    EXTRACT code files only (filter .py, .ts, .go, etc.) to storage/repos/{owner}_{repo}

YIELD event: progress -> "extracted"

# 2. Polyglot Tree-Sitter Chunking
YIELD event: progress -> "chunking"
FOR EACH code file in repository:
    PARSE syntax tree with Tree-sitter
    EXTRACT imports, class overviews, method chunks, function chunks
    IF function tokens > 500:
        RECURSIVELY split by statement boundaries
    COLLECT into in-memory chunks list

SAVE chunks payload to storage/repos/{owner}_{repo}/chunks.json
YIELD event: progress -> "chunked" (total_chunks)

# 3. Handoff to Indexer
YIELD event: progress -> "indexing"
YIELD event: complete -> "completed" (ready for Qdrant & NetworkX)

END
```


---

########### 3. PGVECTOR & INDEXING ###########

We use **PostgreSQL with the `pgvector` extension** and `psycopg_pool.ConnectionPool` for production-grade vector storage and high-concurrency similarity search. Code embeddings are generated using **CodeRankEmbed** (768 dimensions, normalized).

### PSEUDO CODE: PGVECTOR STORE & DENSE RETRIEVAL

```python
START

# STEP 1: INITIALIZATION & CONNECTION POOL SETUP
READ conninfo from DATABASE_URL (or default "postgresql://codeorbit:codeorbit@localhost:5433/codeorbit")

DEFINE function configure_connection(conn):
    CALL register_vector(conn)  # Teaches psycopg how to serialize/deserialize numpy arrays <-> pgvector

INITIALIZE ConnectionPool:
    conninfo = conninfo
    min_size = 1
    max_size = 10
    configure = configure_connection
    open = True

# STEP 2: SCHEMA INITIALIZATION
ACQUIRE conn FROM pool:
    EXECUTE SQL:
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

        # HNSW multi-layer graph index for ultra-fast cosine similarity search
        CREATE INDEX IF NOT EXISTS idx_chunks_hnsw
            ON code_chunks USING hnsw (embedding vector_cosine_ops);

        # B-Tree index to filter by repo shelf in O(log N)
        CREATE INDEX IF NOT EXISTS idx_chunks_source
            ON code_chunks (source);

# STEP 3: INDEXING CHUNKS (index_chunks)
FUNCTION index_chunks(chunks, source_id):
    IF chunks is empty:
        RETURN

    ACQUIRE conn FROM pool:
        IF EXISTS row WHERE source = source_id:
            PRINT "Source already indexed. Skipping."
            RETURN

    # Generate 768-dim normalized embeddings with CodeRankEmbed
    EXTRACT code_texts FROM chunks
    ADD prefix "search_document: " TO EACH code_text
    embeddings = MODEL.encode(code_texts, normalize_embeddings=True)

    # Atomic batch insertion
    ACQUIRE conn FROM pool:
        START conn.transaction():
            FOR EACH (chunk, embedding) IN (chunks, embeddings):
                EXECUTE SQL:
                    INSERT INTO code_chunks (
                        id, source, chunk_type, name, file,
                        start_line, end_line, token_count,
                        code, metadata, embedding
                    ) VALUES (...)
                    ON CONFLICT (id) DO UPDATE SET
                        code = EXCLUDED.code,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding;

    PRINT "Indexed chunks into pgvector."

# STEP 4: DENSE SEARCHING (dense_search)
FUNCTION dense_search(query, source_id, top_k=20):
    # Embed search query with Nomic prefix
    query_emb = MODEL.encode("search_query: " + query, normalize_embeddings=True)

    ACQUIRE conn FROM pool:
        EXECUTE SQL:
            SELECT id, code, metadata, 1 - (embedding <=> query_emb) AS similarity
            FROM code_chunks
            WHERE (source_id IS NULL OR source = source_id)
            ORDER BY embedding <=> query_emb
            LIMIT top_k;

        FETCH matching rows
        RETURN list of (code, metadata, similarity)

END
```


#psql is run on docker get your docker up 
# initalial command
# docker run --name codeorbit-pgvector -e POSTGRES_USER=codeorbit -e POSTGRES_PASSWORD=codeorbit -e POSTGRES_DB=codeorbit -p 5433:5432 -v codeorbit_pgdata:/var/lib/postgresql/data -d pgvector/pgvector:pg16

# after that use for restarting
# docker start codeorbit-pgvector
# to enter terminal forpsql
# docker exec -it codeorbit-pgvector psql -U codeorbit -d codeorbit