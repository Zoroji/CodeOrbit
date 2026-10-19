import ast
import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ─── Token Counter ────────────────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    # ponytail: fallback word tokenizer if tiktoken missing. Upgrade: BPE tokenizer.
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)

TARGET_SIZE = 500
TARGET_FILE = Path("test_extracted_repo/Zoroji_Graph-RAG/Simple_rag/rag_pipeline.py")
OUTPUT_JSON = Path(__file__).parent / "temp.json"


# ─── Statement-Level Recursive Splitter (for oversized functions) ─────────────
def split_large_statement(stmt: ast.stmt, source_code: str, target_size: int = TARGET_SIZE) -> list[str]:
    """If a single statement has a body (loops/ifs/try), split its children; otherwise line-slice."""
    if hasattr(stmt, "body") and isinstance(stmt.body, list):
        sub_chunks, curr = [], []
        for s in stmt.body:
            code = ast.get_source_segment(source_code, s) or ""
            if curr and count_tokens("\n".join(curr) + "\n" + code) > target_size:
                sub_chunks.append("\n".join(curr))
                curr = []
            if count_tokens(code) > target_size:
                sub_chunks.extend(split_large_statement(s, source_code, target_size))
            else:
                curr.append(code)
        if curr:
            sub_chunks.append("\n".join(curr))
        return sub_chunks

    # ponytail: indivisible statements (e.g. huge dict/literal) sliced by lines. Upgrade: AST sub-expression visitor.
    stmt_code = ast.get_source_segment(source_code, stmt) or ""
    lines = stmt_code.splitlines()
    sub_chunks, curr = [], []
    for line in lines:
        if curr and count_tokens("\n".join(curr) + "\n" + line) > target_size:
            sub_chunks.append("\n".join(curr))
            curr = []
        curr.append(line)
    if curr:
        sub_chunks.append("\n".join(curr))
    return sub_chunks


# ─── Definition Chunker (Function / Method) ───────────────────────────────────
def make_definition_chunks(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_code: str,
    file_path: Path,
    parent: str | None = None,
    target_size: int = TARGET_SIZE
) -> list[dict]:
    func_code = ast.get_source_segment(source_code, node) or ""
    tokens = count_tokens(func_code)
    qual_name = f"{parent}.{node.name}" if parent else node.name

    base_meta = {
        "type": "method" if parent else "function",
        "file_path": str(file_path),
        "name": qual_name,
        "parent": parent,
        "start_line": node.lineno,
        "end_line": getattr(node, "end_lineno", node.lineno),
    }

    # Fits in single chunk
    if tokens <= target_size:
        return [{
            "code": func_code,
            "metadata": {
                **base_meta,
                "token_count": tokens,
                "chunk_index": 1,
                "total_chunks": 1,
            }
        }]

    # Oversized -> split statements
    chunks, curr = [], []
    for stmt in node.body:
        stmt_code = ast.get_source_segment(source_code, stmt) or ""
        stmt_tokens = count_tokens(stmt_code)

        if curr and count_tokens("\n".join(curr) + "\n" + stmt_code) > target_size:
            chunks.append("\n".join(curr))
            curr = []

        if stmt_tokens > target_size:
            chunks.extend(split_large_statement(stmt, source_code, target_size))
        else:
            curr.append(stmt_code)

    if curr:
        chunks.append("\n".join(curr))

    total = len(chunks)
    return [{
        "code": chunk_text,
        "metadata": {
            **base_meta,
            "token_count": count_tokens(chunk_text),
            "chunk_index": idx,
            "total_chunks": total,
        }
    } for idx, chunk_text in enumerate(chunks, start=1)]


# ─── Class Overview Builder ───────────────────────────────────────────────────
def make_class_overview(node: ast.ClassDef, source_code: str, file_path: Path) -> dict:
    """Builds a lightweight structural overview: docstring, class attributes, and method signatures."""
    docstring = ast.get_docstring(node) or ""
    bases = [ast.get_source_segment(source_code, b) or "" for b in node.bases]
    bases_str = f"({', '.join(bases)})" if bases else ""

    lines = [f"class {node.name}{bases_str}:"]
    if docstring:
        lines.append(f'    """{docstring.strip()}"""')

    # Class-level attributes and method signatures
    method_names = []
    for item in node.body:
        if isinstance(item, (ast.Assign, ast.AnnAssign)):
            attr_code = ast.get_source_segment(source_code, item) or ""
            if attr_code:
                lines.append(f"    {attr_code.strip()}")
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_names.append(item.name)
            # Extract only the signature line (def foo(...):)
            first_line = (ast.get_source_segment(source_code, item) or "").splitlines()[0]
            lines.append(f"    {first_line.strip()} ...")

    overview_code = "\n".join(lines)
    return {
        "code": overview_code,
        "metadata": {
            "type": "class_overview",
            "file_path": str(file_path),
            "name": node.name,
            "parent": None,
            "methods": method_names,
            "start_line": node.lineno,
            "end_line": getattr(node, "end_lineno", node.lineno),
            "token_count": count_tokens(overview_code),
            "chunk_index": 1,
            "total_chunks": 1,
        }
    }


# ─── Core Pipeline Runner ─────────────────────────────────────────────────────
def chunk_file(file_path: Path, target_size: int = TARGET_SIZE) -> list[dict]:
    source = file_path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source)

    imports_nodes = []
    globals_nodes = []
    chunks = []

    for node in tree.body:
        # A. Imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports_nodes.append(node)

        # B. Global variables & constants
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            globals_nodes.append(node)

        # C. Classes
        elif isinstance(node, ast.ClassDef):
            # 1. Class overview chunk (API surface without method bodies)
            chunks.append(make_class_overview(node, source, file_path))

            # 2. Methods as individual definition chunks
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.extend(
                        make_definition_chunks(child, source, file_path, parent=node.name, target_size=target_size)
                    )

        # D. Top-level standalone functions
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.extend(
                make_definition_chunks(node, source, file_path, parent=None, target_size=target_size)
            )

    # E. Emit file-level chunks
    if imports_nodes:
        import_codes = [ast.get_source_segment(source, n) or "" for n in imports_nodes]
        imports_text = "\n".join(code.strip() for code in import_codes if code)
        chunks.append({
            "code": imports_text,
            "metadata": {
                "type": "imports",
                "file_path": str(file_path),
                "name": f"{file_path.name}::imports",
                "parent": None,
                "start_line": imports_nodes[0].lineno,
                "end_line": getattr(imports_nodes[-1], "end_lineno", imports_nodes[-1].lineno),
                "token_count": count_tokens(imports_text),
                "chunk_index": 1,
                "total_chunks": 1,
            }
        })

    if globals_nodes:
        global_codes = [ast.get_source_segment(source, n) or "" for n in globals_nodes]
        globals_text = "\n".join(code.strip() for code in global_codes if code)
        chunks.append({
            "code": globals_text,
            "metadata": {
                "type": "globals",
                "file_path": str(file_path),
                "name": f"{file_path.name}::globals",
                "parent": None,
                "start_line": globals_nodes[0].lineno,
                "end_line": getattr(globals_nodes[-1], "end_lineno", globals_nodes[-1].lineno),
                "token_count": count_tokens(globals_text),
                "chunk_index": 1,
                "total_chunks": 1,
            }
        })

    return chunks


def main():
    if not TARGET_FILE.exists():
        print(f"❌ File not found: {TARGET_FILE}")
        return

    print("=" * 68)
    print(f"✂️  AST Semantic Chunking Pipeline")
    print(f"📄 Target File: {TARGET_FILE}")
    print("=" * 68)

    chunks = chunk_file(TARGET_FILE, TARGET_SIZE)

    # Save to temp.json
    OUTPUT_JSON.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    # Terminal Dashboard
    type_counts = {}
    for c in chunks:
        t = c["metadata"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n📦 Generated {len(chunks)} Total Chunks:")
    for c_type, count in type_counts.items():
        print(f"   • {c_type:<16}: {count} chunk(s)")

    print(f"\n💾 Saved full chunk payload to: {OUTPUT_JSON.name}")
    print("\n🔍 Sample Overview of Extracted Chunks:")
    print("-" * 68)
    for c in chunks[:8]:
        m = c["metadata"]
        print(f"[{m['type'].upper():<14}] {m['name']:<35} ({m['token_count']} tok, lines {m['start_line']}–{m['end_line']})")

    # Highlight any multi-chunk splits
    splits = [c for c in chunks if c["metadata"]["total_chunks"] > 1]
    if splits:
        print("\n⚡ Statement-Split Functions:")
        for s in splits:
            m = s["metadata"]
            print(f"   • {m['name']} (part {m['chunk_index']}/{m['total_chunks']}) -> {m['token_count']} tokens")

    print("=" * 68)


if __name__ == "__main__":
    main()
