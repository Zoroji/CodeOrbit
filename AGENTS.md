# AGENTS.md — CodeOrbit Project Operating Blueprint

> **Role & Partnership Charter**:
> - **The Developer (User)** is the Lead Architect and Primary Coder. They write and own the implementation.
> - **The Assistant (Antigravity)** is the **Plumber, Researcher, and Debugging Partner**.
> - **Intervention Protocol**: Antigravity assists with research, algorithmic strategies, AST edge cases, graph modeling tradeoffs, debugging hypotheses, and architectural plumbing. Do not unilaterally generate massive codebases unless explicitly instructed ("ship this" / "give me the code"). Favor: `Question → Direction → Hint → Strategy → Pseudocode → Targeted Snippet`.

---

## 1. Project Overview & Core Value Proposition

**CodeOrbit** (formerly RepoLens) is an AI-powered developer intelligence and change-impact analysis engine. Instead of dumping raw files into an LLM context window (which burns tokens and hallucinates call paths), CodeOrbit:
1. **Parses & Maps**: Builds an explicit directed Knowledge Graph of the codebase using AST (Abstract Syntax Tree) analysis (files, classes, functions, routes, imports, calls, tests).
2. **Hybrid GraphRAG**: Couples structural graph traversals with dense semantic search (vector store) and exact token matching (BM25) reranked through a Cross-Encoder.
3. **Change-Impact Analysis (Signature Feature)**: Given a Git diff or changed function, traces reverse dependency trees, pinpoints affected downstream routes/services, and recommends existing tests that cover the blast radius.
4. **Visual Exploration**: Delivers an interactive visual dependency graph and chat workbench built with React, Bootstrap, Framer Motion, GSAP, and Groq LLM.

---

## 2. Tech Stack & Architectural Decisions

### 2.1 Backend & AI
- **Language**: Python 3.10+
- **API Framework**: FastAPI (async endpoints for repository scanning, graph queries, impact analysis, streaming chat)
- **Code Parsing**: Python standard library `ast` + `symtable` (zero external parsing dependencies for Python code; easily extensible to Tree-sitter later for polyglot support)
- **LLM Provider**: **Groq API** (`groq_api_key`) exclusively for ultra-fast reasoning and summarization (e.g. `llama-3.3-70b-versatile` or `mixtral-8x7b-32768`)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (or `BAAI/bge-base-en-v1.5` / code-specific embeddings)
- **Vector DB**: `ChromaDB` (persistent local storage, zero cloud costs)
- **Keyword Search**: `rank-bm25` (with code-aware tokenization handling `snake_case`, `camelCase`, and dot-paths)
- **Reranker**: `sentence-transformers` CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

### 2.2 Graph Engine Evaluation (Free & Storage-Optimized)

| Graph Solution | Storage & Setup | Memory / Overhead | Query Interface | Verdict for CodeOrbit |
| :--- | :--- | :--- | :--- | :--- |
| **NetworkX** | File-based (`pickle`, `GraphML`, `JSON`) | Kept in memory; 10k nodes ~ 15MB RAM | Python API (`neighbors`, `dfs_tree`, `shortest_path`) | **Recommended for v1**: Zero setup, 100% free, instantaneous graph traversals, easy export to frontend JSON. |
| **Kùzu** | Embedded C++ engine (stored in a single directory like SQLite) | Very low; memory-mapped disk storage | OpenCypher (`MATCH (f:Function)-[:CALLS]->(g)`) | **Recommended for v1.5 / Scaling**: 100% free, embeddable in Python (`pip install kuzu`), no Docker needed, full Cypher query power without Neo4j overhead. |
| **Neo4j Community** | Standalone server (Docker / local service) | Heavy (~1–2 GB RAM for JVM) | Cypher & Neo4j Browser | Optional: Great visual dashboard, but requires Docker/service management and adds operational friction. |

**Decision**: 
- **Milestone 1 & 2**: Start with **NetworkX** serialized to disk (or cached per repo).
- **Milestone 3+**: If complex multi-hop Cypher queries are needed, plug in **Kùzu** as an embedded, zero-maintenance, disk-backed Cypher engine.

### 2.3 Frontend & Visualization
- **Core**: React.js
- **Styling**: Bootstrap 5 + custom Vanilla CSS for modern dark-mode aesthetic
- **Animations**: Framer Motion & GSAP
- **Graph Visualization**: `React Flow` (or `@visx/network` / `Cytoscape.js`) for interactive node dragging, edge highlighting, call-path tracing, and impact blast-radius halos.

---

## 3. Core Graph Schema

### 3.1 Node Types
- `File`: Relative file path, lines of code, docstring
- `Class`: Name, docstring, file path, line range
- `Function` / `Method`: Name, signature, is_async, file path, line range, docstring
- `Endpoint`: HTTP method (`GET`, `POST`, etc.), route path (`/api/v1/profile`), handler function ref
- `Test`: Test function/class name, target subject heuristic, file path

### 3.2 Edge Types (Directed)
- `(File) -[:DEFINES]-> (Class | Function)`
- `(File) -[:IMPORTS]-> (File | Module)`
- `(Function) -[:CALLS]-> (Function | Method)`
- `(Endpoint) -[:EXPOSES]-> (Function)`
- `(Class) -[:INHERITS_FROM]-> (Class)`
- `(Test) -[:TESTS]-> (Function | Class | Module)`
- `(Component) -[:DEPENDS_ON]-> (Component)`

---

## 4. Feature Flow & Retrieval Pipeline

### 4.1 Ingestion Pipeline
```
[Git URL / Local Path / ZIP]
         │
         ▼
[File Discovery & Filter] (.py files, exclude .venv, .git, node_modules)
         │
    ┌────┴─────────────────────────────┐
    ▼                                  ▼
[AST Visitor & Symbol Table]     [Code Chunking] (by class/function boundary)
    │                                  │
    ▼                                  ▼
[NetworkX Graph Assembly]         [Dual Indexing]
 - File -> Definitions             - ChromaDB (Dense BGE embeddings)
 - Imports & Resolution            - BM25 (Sparse code tokens)
 - Call graph resolution
 - Route decorator detection
```

### 4.2 Query & Q&A Flow (Graph-Augmented RAG)
1. **Candidate Retrieval**: Dense vector search + BM25 keyword search over function chunks and docstrings.
2. **Graph Seeding**: Map top hits to corresponding graph nodes (`Function`, `Class`, `Endpoint`).
3. **Graph Expansion**:
   - Forward 1–2 hops: What does this function call? What models does it touch?
   - Reverse 1–2 hops: Who calls this function? What routes expose it?
4. **Context Synthesis & Rerank**: Collate seed chunks + graph neighbor chunks → Cross-Encoder reranks top 5–7 most relevant snippets.
5. **LLM Generation**: Prompt Groq (`llama-3.3-70b-versatile`) with structural context + code excerpts with strict requirement to cite `path/to/file.py:lines`.

### 4.3 Change-Impact Analysis Flow
1. **Diff Parsing**: Parse `git diff` or compare two commit SHAs / branches.
2. **Changed Symbol Identification**: Map modified line ranges to AST functions/classes.
3. **Reverse Dependency Traversal (Blast Radius)**:
   - Perform reverse BFS/DFS in the call graph starting from modified symbols: `G.reverse()`.
   - Collect upstream callers, affected routes (`EXPOSES`), and services.
4. **Test Discovery**: Find test nodes connected via `TESTS` edges or whose test functions call any node in the blast radius.
5. **Confidence & Evidence Report**: Output structured impact tree (Directly Modified → Downstream Dependents → Endpoints at Risk → Recommended Test Suites).

---

## 5. Development Milestones

- [ ] **Milestone 1: Repository Parser & Code Graph Engine**
  - Implement recursive Python AST visitor extracting functions, classes, decorators (FastAPI routes), calls, and imports.
  - Implement import resolution (translating `from .services import user_service` to graph file nodes).
  - Assemble directed NetworkX graph and write unit tests on sample miniature repositories.
  
- [ ] **Milestone 2: Hybrid Indexing & Graph-Augmented Q&A**
  - Function-aware code chunking (chunk by semantic blocks rather than arbitrary character count).
  - Setup ChromaDB dense index + BM25 sparse index.
  - Graph-assisted retrieval (seeds → neighbors → CrossEncoder reranking).
  - Groq LLM integration with citation prompt engineering.
  
- [ ] **Milestone 3: Change-Impact Analysis Engine**
  - Git diff parser extracting modified line ranges.
  - Mapping changed lines to AST symbol entities.
  - Reverse traversal blast-radius calculator + test coverage mapper.
  - Impact report schema generation (JSON/Markdown).

- [ ] **Milestone 4: Full-Stack Interface & Visualizer**
  - FastAPI server exposing: `/api/ingest`, `/api/query`, `/api/impact`, `/api/graph`.
  - React frontend with Bootstrap & modern styling.
  - Interactive graph visualizer (e.g. React Flow) showing live nodes, highlighted paths, and impact halos.
  - GSAP / Framer Motion transitions and responsive chat/report drawer.

---

## 6. Known Challenges & Research Solutions

1. **Dynamic Python Calls**: AST cannot resolve dynamic imports (`importlib.import_module`), runtime reflection (`getattr`), or dynamic dependency injection.
   - *Mitigation*: Mark relationships with confidence attributes (`confidence: "explicit"` vs `"inferred"` vs `"dynamic_unresolved"`).
2. **Unqualified Method Calls**: In `self.repo.save()`, AST sees call to attribute `save`, but doesn't have runtime type of `self.repo`.
   - *Mitigation*: Local class scope resolution + heuristic lookup across candidate class definitions.
3. **Chunk Boundary Alignment**: Standard RAG chunks split functions halfway through.
   - *Mitigation*: Use AST node line ranges (`lineno` to `end_lineno`) to chunk cleanly on function/class boundaries.
