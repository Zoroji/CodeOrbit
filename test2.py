import sys
import json
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ─── Tree-sitter Language Bindings ────────────────────────────────────────────
try:
    from tree_sitter import Language, Parser
    import tree_sitter_python
    import tree_sitter_javascript
    import tree_sitter_typescript

    LANGUAGES = {
        ".py": Language(tree_sitter_python.language()),
        ".js": Language(tree_sitter_javascript.language()),
        ".jsx": Language(tree_sitter_javascript.language()),
        ".ts": Language(tree_sitter_typescript.language_typescript()),
        ".tsx": Language(tree_sitter_typescript.language_tsx()),
    }
except ImportError as e:
    print(f"❌ Missing required tree-sitter packages: {e}")
    print("👉 Run: pip install -r requirements.txt")
    sys.exit(1)

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


def get_parser(language) -> Parser:
    """Initialize Tree-sitter Parser supporting both 0.22+ and older APIs."""
    try:
        return Parser(language)
    except TypeError:
        p = Parser()
        p.set_language(language)
        return p


def get_node_text(node, source_bytes: bytes) -> str:
    """Extract raw text slice for a given Tree-sitter node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


# ─── Statement-Level Recursive Splitter (Oversized Functions) ─────────────────
def split_large_tree_node(node, source_bytes: bytes, target_size: int = TARGET_SIZE) -> list[str]:
    """If a Tree-sitter statement node has children (loops, conditionals), split children."""
    body_node = node.child_by_field_name("body") or node
    children = [c for c in body_node.children if c.is_named]

    if len(children) > 1:
        sub_chunks, curr = [], []
        for child in children:
            code = get_node_text(child, source_bytes)
            if curr and count_tokens("\n".join(curr) + "\n" + code) > target_size:
                sub_chunks.append("\n".join(curr))
                curr = []
            if count_tokens(code) > target_size:
                sub_chunks.extend(split_large_tree_node(child, source_bytes, target_size))
            else:
                curr.append(code)
        if curr:
            sub_chunks.append("\n".join(curr))
        return sub_chunks

    # ponytail: indivisible statements sliced by lines. Upgrade: sub-expression traversal.
    lines = get_node_text(node, source_bytes).splitlines()
    sub_chunks, curr = [], []
    for line in lines:
        if curr and count_tokens("\n".join(curr) + "\n" + line) > target_size:
            sub_chunks.append("\n".join(curr))
            curr = []
        curr.append(line)
    if curr:
        sub_chunks.append("\n".join(curr))
    return sub_chunks


# ─── Function & Method Chunker ────────────────────────────────────────────────
def make_definition_chunks(
    node,
    source_bytes: bytes,
    file_path: Path,
    parent: str | None = None,
    target_size: int = TARGET_SIZE
) -> list[dict]:
    func_code = get_node_text(node, source_bytes)
    name_node = node.child_by_field_name("name")
    func_name = get_node_text(name_node, source_bytes) if name_node else "anonymous"
    qual_name = f"{parent}.{func_name}" if parent else func_name

    start_line = node.start_point[0] + 1
    end_line = node.end_point[0] + 1
    tokens = count_tokens(func_code)

    base_meta = {
        "type": "method" if parent else "function",
        "file_path": str(file_path),
        "name": qual_name,
        "parent": parent,
        "start_line": start_line,
        "end_line": end_line,
    }

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

    # Oversized -> split statements in body
    body_node = node.child_by_field_name("body")
    statements = [c for c in body_node.children if c.is_named] if body_node else [node]

    chunks, curr = [], []
    for stmt in statements:
        stmt_code = get_node_text(stmt, source_bytes)
        stmt_tokens = count_tokens(stmt_code)

        if curr and count_tokens("\n".join(curr) + "\n" + stmt_code) > target_size:
            chunks.append("\n".join(curr))
            curr = []

        if stmt_tokens > target_size:
            chunks.extend(split_large_tree_node(stmt, source_bytes, target_size))
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
def make_class_overview(node, source_bytes: bytes, file_path: Path) -> tuple[dict, list]:
    """Extracts lightweight class overview and returns (overview_chunk, list_of_method_nodes)."""
    name_node = node.child_by_field_name("name")
    class_name = get_node_text(name_node, source_bytes) if name_node else "UnnamedClass"

    body_node = node.child_by_field_name("body")
    lines = [f"class {class_name}:"]
    method_nodes = []
    method_names = []

    if body_node:
        for child in body_node.children:
            if not child.is_named:
                continue

            # Python: function_definition | JS/TS: method_definition
            if child.type in ("function_definition", "method_definition"):
                method_nodes.append(child)
                m_name_node = child.child_by_field_name("name")
                m_name = get_node_text(m_name_node, source_bytes) if m_name_node else "method"
                method_names.append(m_name)
                # Signature: first line of method
                sig_line = get_node_text(child, source_bytes).splitlines()[0].strip()
                lines.append(f"    {sig_line} ...")

            # Class docstring or data member
            elif child.type in ("expression_statement", "lexical_declaration", "variable_declaration", "field_definition"):
                txt = get_node_text(child, source_bytes).strip()
                # If multi-line docstring, capture first line
                if "\n" in txt:
                    txt = txt.splitlines()[0] + ' """...'
                lines.append(f"    {txt}")

    overview_code = "\n".join(lines)
    overview_chunk = {
        "code": overview_code,
        "metadata": {
            "type": "class_overview",
            "file_path": str(file_path),
            "name": class_name,
            "parent": None,
            "methods": method_names,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "token_count": count_tokens(overview_code),
            "chunk_index": 1,
            "total_chunks": 1,
        }
    }
    return overview_chunk, method_nodes


# ─── Main Polyglot Tree-Sitter Chunker ─────────────────────────────────────────
def chunk_with_tree_sitter(file_path: Path, target_size: int = TARGET_SIZE) -> list[dict]:
    suffix = file_path.suffix.lower()
    language = LANGUAGES.get(suffix)

    if language is None:
        raise ValueError(f"Unsupported language extension: {suffix}")

    parser = get_parser(language)
    source_bytes = file_path.read_bytes()
    tree = parser.parse(source_bytes)
    root = tree.root_node

    imports_nodes = []
    globals_nodes = []
    chunks = []

    # Node types mapping across Python & JavaScript/TypeScript
    IMPORT_TYPES = {"import_statement", "import_from_statement"}
    GLOBAL_TYPES = {"expression_statement", "lexical_declaration", "variable_declaration"}
    CLASS_TYPES = {"class_definition", "class_declaration"}
    FUNCTION_TYPES = {"function_definition", "function_declaration", "arrow_function"}

    for child in root.children:
        if not child.is_named:
            continue

        # A. Imports
        if child.type in IMPORT_TYPES:
            imports_nodes.append(child)

        # B. Classes
        elif child.type in CLASS_TYPES:
            class_overview, methods = make_class_overview(child, source_bytes, file_path)
            chunks.append(class_overview)
            c_name = class_overview["metadata"]["name"]

            for m_node in methods:
                chunks.extend(
                    make_definition_chunks(m_node, source_bytes, file_path, parent=c_name, target_size=target_size)
                )

        # C. Top-level Functions
        elif child.type in FUNCTION_TYPES:
            chunks.extend(
                make_definition_chunks(child, source_bytes, file_path, parent=None, target_size=target_size)
            )

        # D. Global Variables / Assignments
        elif child.type in GLOBAL_TYPES:
            # Check if this expression is an assignment
            if child.type == "expression_statement":
                if any(c.type in ("assignment", "augmented_assignment") for c in child.children):
                    globals_nodes.append(child)
            else:
                globals_nodes.append(child)

    # E. Emit File-Level Aggregated Chunks
    if imports_nodes:
        imports_text = "\n".join(get_node_text(n, source_bytes).strip() for n in imports_nodes)
        chunks.append({
            "code": imports_text,
            "metadata": {
                "type": "imports",
                "file_path": str(file_path),
                "name": f"{file_path.name}::imports",
                "parent": None,
                "start_line": imports_nodes[0].start_point[0] + 1,
                "end_line": imports_nodes[-1].end_point[0] + 1,
                "token_count": count_tokens(imports_text),
                "chunk_index": 1,
                "total_chunks": 1,
            }
        })

    if globals_nodes:
        globals_text = "\n".join(get_node_text(n, source_bytes).strip() for n in globals_nodes)
        chunks.append({
            "code": globals_text,
            "metadata": {
                "type": "globals",
                "file_path": str(file_path),
                "name": f"{file_path.name}::globals",
                "parent": None,
                "start_line": globals_nodes[0].start_point[0] + 1,
                "end_line": globals_nodes[-1].end_point[0] + 1,
                "token_count": count_tokens(globals_text),
                "chunk_index": 1,
                "total_chunks": 1,
            }
        })

    return chunks


def main():
    target = TARGET_FILE
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])

    if not target.exists():
        print(f"❌ File not found: {target}")
        return

    print("=" * 68)
    print(f"🌳 Tree-Sitter Polyglot Semantic Chunking")
    print(f"📄 Parsing: {target} ({target.suffix})")
    print("=" * 68)

    chunks = chunk_with_tree_sitter(target, TARGET_SIZE)

    # Override temp.json
    OUTPUT_JSON.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    # Terminal Dashboard
    type_counts = {}
    for c in chunks:
        t = c["metadata"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n📦 Generated {len(chunks)} Total Chunks:")
    for c_type, count in type_counts.items():
        print(f"   • {c_type:<16}: {count} chunk(s)")

    print(f"\n💾 Overwrote full chunk payload to: {OUTPUT_JSON.resolve()}")
    print("\n🔍 Sample Overview of Extracted Chunks:")
    print("-" * 68)
    for c in chunks[:8]:
        m = c["metadata"]
        print(f"[{m['type'].upper():<14}] {m['name']:<35} ({m['token_count']} tok, lines {m['start_line']}–{m['end_line']})")

    splits = [c for c in chunks if c["metadata"]["total_chunks"] > 1]
    if splits:
        print("\n⚡ Statement-Split Functions (> 500 tokens):")
        for s in splits:
            m = s["metadata"]
            print(f"   • {m['name']} (part {m['chunk_index']}/{m['total_chunks']}) -> {m['token_count']} tokens")

    print("=" * 68)


if __name__ == "__main__":
    main()
