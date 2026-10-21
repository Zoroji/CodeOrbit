import hashlib
import importlib
from pathlib import Path

# ─── Extension to Language Mapping ────────────────────────────────────────────
EXTENSION_TO_LANG = {
    # Python
    ".py": "python", ".pyi": "python",
    # JavaScript & TypeScript
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx",
    # Systems & Compiled Languages
    ".go": "go", ".rs": "rust", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".cs": "c_sharp", ".kt": "kotlin", ".swift": "swift", ".scala": "scala",
    # Scripting & Web
    ".rb": "ruby", ".php": "php", ".sh": "bash", ".sql": "sql",
    ".html": "html", ".css": "css", ".scss": "scss"
}

IMPORT_NODE_TYPES = {
    "import_statement", "import_from_statement", "import_declaration",
    "use_declaration", "using_directive", "preproc_include", "namespace_use_declaration"
}

CLASS_NODE_TYPES = {
    "class_definition", "class_declaration", "struct_item", "enum_item",
    "trait_item", "impl_item", "type_declaration", "class_specifier",
    "struct_specifier", "interface_declaration", "protocol_declaration", "record_declaration"
}

FUNCTION_NODE_TYPES = {
    "function_definition", "function_declaration", "function_item",
    "method_declaration", "method_definition", "constructor_declaration",
    "arrow_function", "singleton_method"
}

GLOBAL_VAR_TYPES = {
    "expression_statement", "lexical_declaration", "variable_declaration",
    "const_item", "static_item", "var_declaration"
}

TYPE_DEF_TYPES = {
    "type_alias_declaration", "type_spec"
}

# ─── Dynamic Grammar Loader ───────────────────────────────────────────────────
_LOADED_LANGUAGES = {}
_HAS_LANG_PACK = None

def get_language_and_parser(extension: str):
    global _HAS_LANG_PACK
    ext = extension.lower()
    lang_name = EXTENSION_TO_LANG.get(ext)
    if not lang_name:
        return None, None

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

    # Strategy B: Modular packages
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

        try:
            parser = Parser(lang)
        except TypeError:
            parser = Parser()
            parser.set_language(lang)

        _LOADED_LANGUAGES[lang_name] = (lang, parser)
        return lang, parser
    except ImportError:
        return None, None


# ─── Token Counter ────────────────────────────────────────────────────────────
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return int(len(text.split()) * 1.3)


def get_node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def generate_chunk_id(repo_id: str, file_rel_path: str, chunk_name: str, chunk_index: int) -> str:
    raw = f"{repo_id}:{file_rel_path}:{chunk_name}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


# ─── Unified Chunk Factory ────────────────────────────────────────────────────
def _create_chunk(
    code: str,
    chunk_type: str,
    name: str,
    file_rel_path: str,
    start_line: int,
    end_line: int,
    repo_owner: str,
    repo_name: str,
    parent: str | None = None,
    methods: list[str] | None = None,
    segment_idx: int = 1,
    total_chunks: int = 1,
    branch: str = "main"
) -> dict:
    repo_id = f"{repo_owner}_{repo_name}"
    github_url = f"https://github.com/{repo_owner}/{repo_name}/blob/{branch}/{file_rel_path}#L{start_line}-L{end_line}"

    meta = {
        "chunk_id": generate_chunk_id(repo_id, file_rel_path, name, segment_idx),
        "type": chunk_type,
        "name": name,
        "file": file_rel_path,
        "parent": parent,
        "start_line": start_line,
        "end_line": end_line,
        "chunk_size": count_tokens(code),
        "segment_idx": segment_idx,
        "total_chunks": total_chunks,
        "github_developer": repo_owner,
        "repo_name": repo_name,
        "github_source_code_url": github_url,
    }
    if methods is not None:
        meta["methods"] = methods

    return {"code": code, "metadata": meta}


def split_large_tree_node(node, source_bytes: bytes, target_size: int = 500) -> list[str]:
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


def make_definition_chunks(
    node,
    source_bytes: bytes,
    file_rel_path: str,
    repo_owner: str,
    repo_name: str,
    explicit_name: str | None = None,
    parent: str | None = None,
    target_size: int = 500
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
    chunk_type = "method" if parent else "function"

    if tokens <= target_size:
        return [
            _create_chunk(
                code=func_code, chunk_type=chunk_type, name=qual_name,
                file_rel_path=file_rel_path, start_line=start_line, end_line=end_line,
                repo_owner=repo_owner, repo_name=repo_name, parent=parent,
                segment_idx=1, total_chunks=1
            )
        ]

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
    return [
        _create_chunk(
            code=chunk_text, chunk_type=chunk_type, name=qual_name,
            file_rel_path=file_rel_path, start_line=start_line, end_line=end_line,
            repo_owner=repo_owner, repo_name=repo_name, parent=parent,
            segment_idx=idx, total_chunks=total
        )
        for idx, chunk_text in enumerate(chunks, start=1)
    ]


def make_class_overview(
    node,
    source_bytes: bytes,
    file_rel_path: str,
    repo_owner: str,
    repo_name: str
) -> tuple[dict, list]:
    name_node = node.child_by_field_name("name")
    class_name = get_node_text(name_node, source_bytes) if name_node else "UnnamedClass"

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
    overview_chunk = _create_chunk(
        code=overview_code, chunk_type="class_overview", name=class_name,
        file_rel_path=file_rel_path, start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
        repo_owner=repo_owner, repo_name=repo_name, parent=None, methods=method_names,
        segment_idx=1, total_chunks=1
    )
    return overview_chunk, method_nodes


def process_syntax_node(node, source_bytes: bytes):
    effective_node = node
    if node.type == "export_statement":
        named_children = [c for c in node.children if c.is_named]
        if named_children:
            effective_node = named_children[0]
        else:
            return "global", node

    node_type = effective_node.type
    if node_type in IMPORT_NODE_TYPES:
        return "import", node
    if node_type in CLASS_NODE_TYPES:
        return "class", node
    if node_type in FUNCTION_NODE_TYPES:
        name_node = effective_node.child_by_field_name("name")
        fn_name = get_node_text(name_node, source_bytes) if name_node else None
        return "function", (node, fn_name)
    if node_type in TYPE_DEF_TYPES:
        return "type", node
    if node_type in ("lexical_declaration", "variable_declaration"):
        for declarator in effective_node.children:
            if declarator.type == "variable_declarator":
                val = declarator.child_by_field_name("value")
                if val and val.type in ("arrow_function", "function_expression"):
                    name_node = declarator.child_by_field_name("name")
                    fn_name = get_node_text(name_node, source_bytes) if name_node else "anonymous"
                    return "function", (node, fn_name)
        return "global", node
    if node_type in GLOBAL_VAR_TYPES:
        return "global", node

    return "other", node


def chunk_single_file(
    file_path: Path,
    repo_dir: Path,
    repo_owner: str,
    repo_name: str,
    target_size: int = 500
) -> list[dict]:
    suffix = file_path.suffix.lower()
    lang, parser = get_language_and_parser(suffix)
    if not parser:
        return []

    source_bytes = file_path.read_bytes()
    tree = parser.parse(source_bytes)
    root = tree.root_node

    rel_path = str(file_path.relative_to(repo_dir)).replace("\\", "/")
    imports_nodes, globals_nodes, types_nodes, chunks = [], [], [], []

    for child in root.children:
        if not child.is_named:
            continue

        category, payload = process_syntax_node(child, source_bytes)

        if category == "import":
            imports_nodes.append(payload)
        elif category == "class":
            class_overview, methods = make_class_overview(payload, source_bytes, rel_path, repo_owner, repo_name)
            chunks.append(class_overview)
            c_name = class_overview["metadata"]["name"]
            for m_node in methods:
                chunks.extend(
                    make_definition_chunks(m_node, source_bytes, rel_path, repo_owner, repo_name, parent=c_name, target_size=target_size)
                )
        elif category == "function":
            func_node, explicit_name = payload
            chunks.extend(
                make_definition_chunks(func_node, source_bytes, rel_path, repo_owner, repo_name, explicit_name=explicit_name, parent=None, target_size=target_size)
            )
        elif category == "type":
            types_nodes.append(payload)
        elif category == "global":
            globals_nodes.append(payload)

    if imports_nodes:
        imports_text = "\n".join(get_node_text(n, source_bytes).strip() for n in imports_nodes)
        chunks.append(
            _create_chunk(
                code=imports_text, chunk_type="imports", name=f"{file_path.name}::imports",
                file_rel_path=rel_path, start_line=imports_nodes[0].start_point[0] + 1,
                end_line=imports_nodes[-1].end_point[0] + 1, repo_owner=repo_owner, repo_name=repo_name
            )
        )

    if types_nodes:
        types_text = "\n\n".join(get_node_text(n, source_bytes).strip() for n in types_nodes)
        chunks.append(
            _create_chunk(
                code=types_text, chunk_type="types", name=f"{file_path.name}::types",
                file_rel_path=rel_path, start_line=types_nodes[0].start_point[0] + 1,
                end_line=types_nodes[-1].end_point[0] + 1, repo_owner=repo_owner, repo_name=repo_name
            )
        )

    if globals_nodes:
        globals_text = "\n".join(get_node_text(n, source_bytes).strip() for n in globals_nodes)
        chunks.append(
            _create_chunk(
                code=globals_text, chunk_type="globals", name=f"{file_path.name}::globals",
                file_rel_path=rel_path, start_line=globals_nodes[0].start_point[0] + 1,
                end_line=globals_nodes[-1].end_point[0] + 1, repo_owner=repo_owner, repo_name=repo_name
            )
        )

    return chunks


def chunk_repository(repo_dir: Path, repo_owner: str, repo_name: str, target_size: int = 500) -> list[dict]:
    """Recursively walks repository directory and chunks every supported code file."""
    all_chunks = []
    for file_path in repo_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in EXTENSION_TO_LANG:
            try:
                chunks = chunk_single_file(file_path, repo_dir, repo_owner, repo_name, target_size)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"⚠️ Error chunking {file_path}: {e}")
                continue
    return all_chunks
