import sys
import json
import importlib
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ─── Comprehensive Extension to Language Registry ─────────────────────────────
# Maps file extensions to standard Tree-sitter language names
EXTENSION_TO_LANG = {
    # Python
    ".py": "python",
    ".pyi": "python",
    # JavaScript & TypeScript
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    # Systems & Compiled Languages
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "c_sharp",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    # Scripting & Web Languages
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    # Data & Markup Formats
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
}

# ─── Polyglot AST Syntax Node Maps ────────────────────────────────────────────
# Generalized categories across Python, JS/TS, Go, Rust, Java, C++, C#, etc.
IMPORT_NODE_TYPES = {
    "import_statement",
    "import_from_statement",
    "import_declaration",
    "use_declaration",
    "using_directive",
    "preproc_include",
    "namespace_use_declaration",
}

CLASS_NODE_TYPES = {
    "class_definition",
    "class_declaration",
    "struct_item",
    "enum_item",
    "trait_item",
    "impl_item",
    "type_declaration",
    "class_specifier",
    "struct_specifier",
    "interface_declaration",
    "protocol_declaration",
    "record_declaration",
    "module",
}

FUNCTION_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "function_item",
    "method_declaration",
    "method_definition",
    "constructor_declaration",
    "arrow_function",
    "singleton_method",
}

GLOBAL_VAR_TYPES = {
    "expression_statement",
    "lexical_declaration",
    "variable_declaration",
    "const_item",
    "static_item",
    "var_declaration",
}

TYPE_DEF_TYPES = {
    "type_alias_declaration",
    "type_spec",
}

# ─── Dynamic Tree-sitter Language Loader ──────────────────────────────────────
_LOADED_LANGUAGES = {}
_HAS_LANG_PACK = None

def get_language_and_parser(extension: str):
    """
    Dynamically loads grammar and parser for any supported file extension.
    1. Checks if tree-sitter-language-pack (370+ languages) is available.
    2. Falls back to modular tree-sitter-<language> packages.
    """
    global _HAS_LANG_PACK

    ext = extension.lower()
    lang_name = EXTENSION_TO_LANG.get(ext)
    if not lang_name:
        raise ValueError(f"Unsupported file extension: '{ext}'")

    if lang_name in _LOADED_LANGUAGES:
        return _LOADED_LANGUAGES[lang_name]

    from tree_sitter import Language, Parser

    # Strategy A: Universal Language Pack (370+ languages)
    if _HAS_LANG_PACK is not False:
        try:
            import tree_sitter_language_pack as tslp
            _HAS_LANG_PACK = True
            lang = tslp.get_language(lang_name)
            parser = tslp.get_parser(lang_name)
            _LOADED_LANGUAGES[lang_name] = (lang, parser)
            return lang, parser
        except ImportError:
            _HAS_LANG_PACK = False

    # Strategy B: Modular tree-sitter-<language> packages
    try:
        if lang_name == "typescript":
            import tree_sitter_typescript as ts
            lang = Language(ts.language_typescript())
        elif lang_name == "tsx":
            import tree_sitter_typescript as ts
            lang = Language(ts.language_tsx())
        elif lang_name == "c_sharp":
            import tree_sitter_c_sharp as tscs
            lang = Language(tscs.language())
        else:
            mod = importlib.import_module(f"tree_sitter_{lang_name}")
            lang = Language(mod.language())

        # Support both 0.22+ and older Parser API
        try:
            parser = Parser(lang)
        except TypeError:
            parser = Parser()
            parser.set_language(lang)

        _LOADED_LANGUAGES[lang_name] = (lang, parser)
        return lang, parser

    except ImportError:
        raise RuntimeError(
            f"Parser for language '{lang_name}' (extension '{ext}') is not installed.\n"
            f"👉 Install the 370+ language bundle:  pip install tree-sitter-language-pack\n"
            f"👉 Or install the individual grammar: pip install tree-sitter-{lang_name}"
        )


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
OUTPUT_JSON = Path(__file__).parent / "temp.json"


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
    explicit_name: str | None = None,
    parent: str | None = None,
    target_size: int = TARGET_SIZE
) -> list[dict]:
    func_code = get_node_text(node, source_bytes)
    if explicit_name:
        func_name = explicit_name
    else:
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


# ─── Class & Struct Overview Builder ──────────────────────────────────────────
def make_class_overview(node, source_bytes: bytes, file_path: Path) -> tuple[dict, list]:
    name_node = node.child_by_field_name("name")
    class_name = get_node_text(name_node, source_bytes) if name_node else "UnnamedType"

    body_node = node.child_by_field_name("body")
    lines = [f"class/type {class_name}:"]
    method_nodes = []
    method_names = []

    if body_node:
        for child in body_node.children:
            if not child.is_named:
                continue

            if child.type in FUNCTION_NODE_TYPES:
                method_nodes.append(child)
                m_name_node = child.child_by_field_name("name")
                m_name = get_node_text(m_name_node, source_bytes) if m_name_node else "method"
                method_names.append(m_name)
                sig_line = get_node_text(child, source_bytes).splitlines()[0].strip()
                lines.append(f"    {sig_line} ...")

            elif child.type in GLOBAL_VAR_TYPES or child.type.endswith(("_field", "_definition", "specifier")):
                txt = get_node_text(child, source_bytes).strip()
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


# ─── Universal Syntax Node Classifier ─────────────────────────────────────────
def process_syntax_node(node, source_bytes: bytes, file_path: Path, target_size: int = TARGET_SIZE):
    effective_node = node

    # Handle JS/TS export wrappers
    if node.type == "export_statement":
        named_children = [c for c in node.children if c.is_named]
        if named_children:
            effective_node = named_children[0]
        else:
            return "global", node

    node_type = effective_node.type

    # 1. Imports / Inclusions
    if node_type in IMPORT_NODE_TYPES:
        return "import", node

    # 2. Classes, Structs, Traits, Protocols, Modules
    if node_type in CLASS_NODE_TYPES:
        return "class", node

    # 3. Standard Functions & Methods
    if node_type in FUNCTION_NODE_TYPES:
        name_node = effective_node.child_by_field_name("name")
        fn_name = get_node_text(name_node, source_bytes) if name_node else None
        return "function", (node, fn_name)

    # 4. Type Aliases & Interfaces
    if node_type in TYPE_DEF_TYPES:
        return "type", node

    # 5. JS/TS Lexical & Variable Declarations (const / let / var)
    if node_type in ("lexical_declaration", "variable_declaration"):
        for declarator in effective_node.children:
            if declarator.type == "variable_declarator":
                val = declarator.child_by_field_name("value")
                if val and val.type in ("arrow_function", "function_expression"):
                    name_node = declarator.child_by_field_name("name")
                    fn_name = get_node_text(name_node, source_bytes) if name_node else "anonymous"
                    return "function", (node, fn_name)
        return "global", node

    # 6. Global Assignments / Constants
    if node_type in GLOBAL_VAR_TYPES:
        return "global", node

    return "other", node


# ─── Universal Polyglot Chunker ───────────────────────────────────────────────
def chunk_with_tree_sitter(file_path: Path, target_size: int = TARGET_SIZE) -> list[dict]:
    suffix = file_path.suffix.lower()
    lang, parser = get_language_and_parser(suffix)

    source_bytes = file_path.read_bytes()
    tree = parser.parse(source_bytes)
    root = tree.root_node

    imports_nodes = []
    globals_nodes = []
    types_nodes = []
    chunks = []

    for child in root.children:
        if not child.is_named:
            continue

        category, payload = process_syntax_node(child, source_bytes, file_path, target_size)

        if category == "import":
            imports_nodes.append(payload)

        elif category == "class":
            class_overview, methods = make_class_overview(payload, source_bytes, file_path)
            chunks.append(class_overview)
            c_name = class_overview["metadata"]["name"]
            for m_node in methods:
                chunks.extend(
                    make_definition_chunks(m_node, source_bytes, file_path, parent=c_name, target_size=target_size)
                )

        elif category == "function":
            func_node, explicit_name = payload
            chunks.extend(
                make_definition_chunks(func_node, source_bytes, file_path, explicit_name=explicit_name, parent=None, target_size=target_size)
            )

        elif category == "type":
            types_nodes.append(payload)

        elif category == "global":
            globals_nodes.append(payload)

    # File-Level Aggregated Chunks
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

    if types_nodes:
        types_text = "\n\n".join(get_node_text(n, source_bytes).strip() for n in types_nodes)
        chunks.append({
            "code": types_text,
            "metadata": {
                "type": "types",
                "file_path": str(file_path),
                "name": f"{file_path.name}::types",
                "parent": None,
                "start_line": types_nodes[0].start_point[0] + 1,
                "end_line": types_nodes[-1].end_point[0] + 1,
                "token_count": count_tokens(types_text),
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
    target = Path("test.tsx")
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])

    if not target.exists():
        print(f"❌ File not found: {target}")
        return

    ext = target.suffix.lower()
    lang_name = EXTENSION_TO_LANG.get(ext, "unknown")

    print("=" * 68)
    print(f"🌐 Universal Polyglot Tree-Sitter Chunker")
    print(f"📄 Parsing: {target} (Language: {lang_name}, Extension: {ext})")
    print("=" * 68)

    chunks = chunk_with_tree_sitter(target, TARGET_SIZE)

    # Save payload to temp.json
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
    for c in chunks:
        m = c["metadata"]
        status = f"chunk {m['chunk_index']}/{m['total_chunks']}" if m['total_chunks'] > 1 else "whole"
        print(f"[{m['type'].upper():<10}] {m['name']:<35} ({m['token_count']} tok, lines {m['start_line']}–{m['end_line']}) [{status}]")

    print("=" * 68)


if __name__ == "__main__":
    main()
