"""
Podcast Vector RAG - Streamlit Interface
Dense Vector (ChromaDB) + Lexical BM25 + Cross-Encoder Reranking

Run: streamlit run app.py
"""

import os
import re
import math
import uuid
import tempfile
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)
ASSETS_DIR = BASE_DIR / "assets"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

try:
    st.set_page_config(
        page_title="Vector RAG - Podcast Intelligence",
        page_icon=":material/dataset:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

# ─── CSS Design System (Vector RAG Theme: Indigo & Slate) ──────────────────────
st.markdown("""
<style>
    /* Global Typography & Palette */
    :root {
        --rag-primary: #4f46e5;
        --rag-primary-dark: #3730a3;
        --rag-accent: #818cf8;
        --rag-bg-tint: #f5f3ff;
        --rag-border: #e0e7ff;
    }

    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #3730a3 0%, #4f46e5 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .system-badge-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.6rem;
    }

    .system-badge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 6px;
        background: #4f46e5;
        color: #ffffff;
    }

    .system-subbadge {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        background: #e0e7ff;
        color: #3730a3;
    }

    .subtitle {
        color: #4b5563;
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    .arch-card {
        background: #fafafa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1.2rem;
    }

    .chunk-box {
        background: #fbfbfd;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #4f46e5;
        padding: 0.85rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #1f2937;
        margin-bottom: 0.6rem;
        line-height: 1.5;
    }

    .timestamp-badge {
        background: #ede9fe;
        color: #5b21b6;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .source-badge {
        background: #e0e7ff;
        color: #3730a3;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .score-badge {
        background: #f3f4f6;
        color: #6b7280;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .stat-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.9rem;
        text-align: center;
    }

    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #4f46e5;
    }

    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .source-tag {
        display: inline-block;
        background: #4f46e5;
        color: #ffffff;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.74rem;
        font-weight: 500;
        margin: 2px;
    }

    .srt-cue-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }

    .srt-cue-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }

    .srt-cue-id {
        font-size: 0.75rem;
        font-weight: 700;
        color: #9ca3af;
    }

    .srt-cue-text {
        font-size: 0.92rem;
        color: #111827;
        line-height: 1.45;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────
def init_state():
    defaults = {
        "vrag_messages": [],
        "vrag": None,
        "vrag_ingested": False,
        "vrag_stats_list": [],
        "vrag_active_sources": [],
        "vrag_custom_api_key": "",
        "vrag_data_mode": "library",
        "persist_dir": str(BASE_DIR / "rag_store"),
        "srt_current_file": None,
        "srt_page_num": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─── SRT Helpers ──────────────────────────────────────────────────────────────
def get_available_srt_files():
    """Discover all .srt transcript files in standard directories."""
    candidates = [
        TRANSCRIPTS_DIR,
        BASE_DIR,
        BASE_DIR.parent / "graph_rag" / "transcripts",
    ]
    discovered = []
    seen = set()
    for directory in candidates:
        if directory.exists() and directory.is_dir():
            for srt_path in sorted(directory.glob("*.srt")):
                if srt_path.name not in seen:
                    seen.add(srt_path.name)
                    discovered.append(srt_path)
    return discovered


def parse_srt(file_path: Path):
    """Parse an .srt file into structured subtitle cues."""
    content = file_path.read_text(encoding="utf-8-sig", errors="replace")
    blocks = re.split(r"\n\s*\n", content.strip())
    cues = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        cue_id = lines[0] if time_index > 0 else str(len(cues) + 1)
        time_line = lines[time_index]
        parts = time_line.split("-->")
        start_ts = parts[0].strip() if len(parts) > 0 else ""
        end_ts = parts[1].strip() if len(parts) > 1 else ""
        text = " ".join(lines[time_index + 1:])
        text = re.sub(r"<[^>]+>", "", text).strip()
        if text:
            cues.append({
                "id": cue_id,
                "start": start_ts,
                "end": end_ts,
                "text": text,
            })
    return cues


# ─── Secure API Key Getter & LLM Caller ────────────────────────────────────────
def get_vrag_api_key() -> str:
    """Safely obtain the Groq API key without exposing it on the frontend."""
    override = st.session_state.get("vrag_custom_api_key", "").strip()
    if override:
        return override
    env_k = os.getenv("groq_api_key", "").strip() or os.getenv("GROQ_API_KEY", "").strip()
    if env_k:
        return env_k
    if hasattr(st, "secrets"):
        if "groq_api_key" in st.secrets:
            return str(st.secrets["groq_api_key"]).strip()
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    return ""


def call_llm(messages: list, context: str, api_key: str) -> str:
    """Generate an answer using Groq with automatic model fallback."""
    from groq import Groq

    active_sources = st.session_state.get("vrag_active_sources", [])
    num_sources = len(active_sources)
    source_list = ", ".join(active_sources) if active_sources else "unknown"

    if not context or context.strip() == "No relevant excerpts found.":
        return "I could not find relevant excerpts in the transcripts to answer this question."

    if not api_key or not api_key.strip():
        raise ValueError(
            "Groq API key is missing. Add groq_api_key=... to the .env file "
            "or enter your key in the sidebar."
        )

    system_prompt = f"""You are an AI research assistant analyzing podcast transcripts using a Vector RAG system.
Sources currently indexed ({num_sources}): {source_list}.

RULES:
1. ONLY use information explicitly present in the TRANSCRIPT EXCERPTS below.
2. If the excerpts do not contain sufficient evidence, state: "This topic is not covered in the available transcripts."
3. Cite the specific source transcript and approximate timestamp for every claim (e.g. "In [source] around 12:34...").
4. If multiple transcripts discuss the subject, contrast and synthesize their insights.
5. Do NOT guess, hallucinate, or extrapolate beyond the provided text.
6. Maintain an objective, concise, and grounded tone.

TRANSCRIPT EXCERPTS:
{context}"""

    conversation = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str)
    ]

    client = Groq(api_key=api_key.strip())
    preferred_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    candidate_models = [
        preferred_model,
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
    ]
    seen = set()
    unique_candidates = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_exc = None
    for m_idx, m_name in enumerate(unique_candidates):
        try:
            response = client.chat.completions.create(
                model=m_name,
                messages=[{"role": "system", "content": system_prompt}] + conversation,
                max_tokens=1024,
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            err_msg = str(exc).lower()
            if ("model_not_found" in err_msg or "does not exist" in err_msg or "404" in err_msg) and m_idx < len(unique_candidates) - 1:
                continue
            raise exc

    raise last_exc or RuntimeError("All candidate models failed.")


# ─── Ingestion Helper ─────────────────────────────────────────────────────────
def run_ingest(file_path: Path, chunk_size: int, overlap: int, top_k: int, top_n: int, persist_dir: Optional[str] = None):
    """Ingest a single transcript file into the PodcastRAG vector index."""
    target_dir = persist_dir or st.session_state.persist_dir
    if st.session_state.vrag is None:
        from rag_pipeline import PodcastRAG
        st.session_state.vrag = PodcastRAG(
            persist_dir=target_dir,
            chunk_size=chunk_size,
            overlap=overlap,
            top_k_retrieve=top_k,
            top_n_rerank=top_n,
        )

    stats = st.session_state.vrag.ingest(str(file_path))
    st.session_state.vrag_stats_list.append(stats)
    st.session_state.vrag_ingested = bool(st.session_state.vrag._ingested_sources)
    st.session_state.vrag_active_sources = list(st.session_state.vrag._ingested_sources.keys())
    return stats


# ─── Sidebar Navigation & Controls ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Navigation")
    current_page = st.radio(
        "Select View",
        [
            "Vector RAG Assistant",
            "SRT Transcripts Explorer",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Configuration")
    st.caption("LLM Provider: Groq API")

    env_has_key = bool(
        os.getenv("groq_api_key", "").strip()
        or os.getenv("GROQ_API_KEY", "").strip()
        or (hasattr(st, "secrets") and ("groq_api_key" in st.secrets or "GROQ_API_KEY" in st.secrets))
    )

    if env_has_key:
        st.success("🔒 System Groq API Key Active")
        custom_key = st.text_input(
            "Override API Key (Optional)",
            value="",
            type="password",
            placeholder="Leave blank to use system key",
            help="Your key is secure and never shown on screen.",
        )
        if custom_key.strip():
            st.session_state.vrag_custom_api_key = custom_key.strip()
    else:
        st.info("💡 Enter your Groq API key:")
        custom_key = st.text_input(
            "Groq API Key",
            value="",
            type="password",
            placeholder="gsk_...",
            help="Held in session memory only; never saved to disk.",
        )
        if custom_key.strip():
            st.session_state.vrag_custom_api_key = custom_key.strip()

    st.divider()
    st.markdown("### Data Source")
    source_choice = st.radio(
        "Select Pipeline Source",
        ["🎙️ Pre-loaded Tech Podcast Library", "📂 Custom SRT File (Private Session)"],
        index=0 if st.session_state.get("vrag_data_mode") != "custom" else 1,
        help="Query the 4 pre-stored podcast episodes or analyze your own custom transcript privately."
    )
    is_custom = "Custom" in source_choice
    prev_mode = st.session_state.get("vrag_data_mode", "library")
    new_mode = "custom" if is_custom else "library"
    if new_mode != prev_mode:
        st.session_state.vrag_data_mode = new_mode
        st.session_state.vrag_messages = []
        st.session_state.vrag = None
        st.session_state.vrag_ingested = False
        st.session_state.vrag_stats_list = []
        st.session_state.vrag_active_sources = []
        if is_custom:
            st.session_state.persist_dir = str(Path(tempfile.gettempdir()) / f"custom_vrag_{uuid.uuid4().hex}")
        else:
            st.session_state.persist_dir = str(BASE_DIR / "rag_store")
        st.rerun()

    st.divider()
    with st.expander("Vector RAG Hyperparameters", expanded=False):
        chunk_size = st.slider("Chunk Size (tokens)", 200, 800, 500, 50)
        overlap = st.slider("Chunk Overlap (tokens)", 20, 150, 80, 10)
        top_k = st.slider("Vector Retrieve Top-K", 5, 30, 20, 5)
        top_n = st.slider("Cross-Encoder Rerank Top-N", 2, 10, 5, 1)

    if is_custom:
        st.divider()
        st.markdown("### Upload Custom Transcript")
        st.caption("🔒 **Privacy Guarantee**: Your custom transcript is analyzed purely in temporary session memory and is **never** saved to disk or git.")
        uploaded_file = st.file_uploader(
            "Upload .srt / .txt file",
            type=["srt", "txt"],
            accept_multiple_files=False,
            key="custom_vrag_uploader"
        )
        if uploaded_file:
            if st.button(f"Analyze {uploaded_file.name} (Private)", use_container_width=True, type="primary"):
                with st.spinner(f"Indexing custom transcript {uploaded_file.name}..."):
                    temp_dir = Path(tempfile.gettempdir()) / f"vrag_{uuid.uuid4().hex}"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file = temp_dir / Path(uploaded_file.name).name
                    temp_file.write_bytes(uploaded_file.getvalue())
                    try:
                        from rag_pipeline import PodcastRAG
                        st.session_state.vrag = PodcastRAG(
                            persist_dir=str(temp_dir),
                            chunk_size=chunk_size,
                            overlap=overlap,
                            top_k_retrieve=top_k,
                            top_n_rerank=top_n,
                        )
                        run_ingest(temp_file, chunk_size, overlap, top_k, top_n, persist_dir=str(temp_dir))
                        st.success(f"Custom transcript {uploaded_file.name} ready for private analysis!")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error processing {uploaded_file.name}: {exc}")

    if st.session_state.vrag_active_sources:
        st.divider()
        st.markdown("### Active Vector Sources")
        for src in st.session_state.vrag_active_sources:
            st.markdown(f'<span class="source-tag">{src}</span>', unsafe_allow_html=True)

    if st.session_state.vrag_stats_list:
        st.divider()
        st.markdown("### Vector Index Stats")
        total_chunks = sum(s.get("chunks", 0) for s in st.session_state.vrag_stats_list)
        col_s1, col_s2 = st.columns(2)
        col_s1.markdown(
            f'<div class="stat-box"><div class="stat-number">{len(st.session_state.vrag_active_sources)}</div>'
            '<div class="stat-label">Documents</div></div>',
            unsafe_allow_html=True,
        )
        col_s2.markdown(
            f'<div class="stat-box"><div class="stat-number">{total_chunks}</div>'
            '<div class="stat-label">Embed Chunks</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    col_btn_a, col_btn_b = st.columns(2)
    if col_btn_a.button("Clear Chat", use_container_width=True):
        st.session_state.vrag_messages = []
        st.rerun()
    if col_btn_b.button("Reset Index", use_container_width=True):
        st.session_state.vrag_messages = []
        st.session_state.vrag = None
        st.session_state.vrag_ingested = False
        st.session_state.vrag_stats_list = []
        st.session_state.vrag_active_sources = []
        st.session_state.persist_dir = str(Path(tempfile.gettempdir()) / f"rag_store_{uuid.uuid4().hex}")
        st.rerun()


# ─── Top Identity Header ──────────────────────────────────────────────────────
st.markdown("""
<div class="system-badge-container">
    <span class="system-badge">Vector RAG</span>
    <span class="system-subbadge">Dense Embeddings + BM25 + Cross-Encoder</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: VECTOR RAG ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════════
if current_page == "Vector RAG Assistant":
    st.markdown('<div class="main-header">Podcast Vector RAG</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Multi-transcript semantic search with ChromaDB, BM25 sparse scoring, and neural cross-encoder reranking.</div>',
        unsafe_allow_html=True,
    )

    # Architecture Overview Expander with Visual Image
    arch_img_path = ASSETS_DIR / "vector_rag_architecture.jpg"
    with st.expander("Vector RAG Architecture Diagram & Pipeline Overview", expanded=False):
        if arch_img_path.exists():
            st.image(str(arch_img_path), caption="Vector RAG Pipeline: Chunking -> Dense Embeddings + BM25 -> ChromaDB -> Neural Reranker -> Groq LLM", use_container_width=True)
        st.markdown("""
        **How Vector RAG Operates:**
        1. **Chunking**: Transcripts are segmented into token windows with overlapping boundaries.
        2. **Dense & Sparse Indexing**: Each chunk is embedded via `sentence-transformers` and indexed in BM25 for exact keyword hits.
        3. **Hybrid Retrieval**: Queries execute dense similarity search combined with BM25 lexical ranking.
        4. **Cross-Encoder Reranking**: The top candidates are rescored with a cross-encoder before context is fed to the Groq LLM.
        """)

    # When no documents are ingested yet
    if not st.session_state.vrag_ingested:
        st.markdown("---")
        is_custom = st.session_state.get("vrag_data_mode") == "custom"
        if is_custom:
            st.info("📂 **Private Custom Transcript Mode**: Please upload an `.srt` or `.txt` file in the sidebar to index your private transcript.")
            st.caption("🔒 All processing occurs in temporary session memory and is never saved to the server or Git.")
        else:
            st.info("No library transcripts currently loaded in the vector index. You can index all 4 available podcast episodes at once below:")

            available_srts = get_available_srt_files()
            if available_srts:
                st.markdown("#### Pre-loaded Tech Podcast Library")
                st.caption("The 4 tech podcast episodes below are stored locally in the system. Click to index them into ChromaDB:")

                if st.button(f"🚀 Index All {len(available_srts)} Library Transcripts (Elon, Sam, Jeff, Jensen)", type="primary", use_container_width=True):
                    prog = st.progress(0.0, text="Initializing library indexing...")
                    for i, srt_f in enumerate(available_srts):
                        prog.progress((i) / len(available_srts), text=f"Indexing {srt_f.name} ({i+1}/{len(available_srts)})...")
                        run_ingest(srt_f, chunk_size, overlap, top_k, top_n)
                    prog.progress(1.0, text="Vector index built for all library episodes!")
                    st.success("All library episodes indexed into Vector RAG!")
                    st.rerun()

                st.write("")
                cols = st.columns(min(len(available_srts), 4))
                for idx, srt_file in enumerate(available_srts):
                    col = cols[idx % len(cols)]
                    with col:
                        st.markdown(f"**{srt_file.stem}**")
                        st.caption(f"File size: {srt_file.stat().st_size / 1024:.1f} KB")
                        if st.button(f"Index {srt_file.stem}", key=f"quick_index_{idx}", use_container_width=True):
                            with st.spinner(f"Indexing {srt_file.name}..."):
                                run_ingest(srt_file, chunk_size, overlap, top_k, top_n)
                            st.success(f"Indexed {srt_file.name}")
                            st.rerun()

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**1. Upload or Select**\n\nChoose an SRT transcript from the library or upload via the sidebar.")
        c2.markdown("**2. Vector Indexing**\n\nDocuments are tokenized, embedded into dense vectors, and stored in ChromaDB.")
        c3.markdown("**3. Grounded Chat**\n\nAsk questions with precise source citations and timestamped playback intervals.")

    else:
        sources = st.session_state.vrag_active_sources
        src_str = " · ".join(f"`{s}`" for s in sources)
        st.markdown(f"**Indexed Sources ({len(sources)}):** {src_str}")
        st.markdown("---")

        # Chat Message History
        for msg in st.session_state.vrag_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander(f"{len(msg['sources'])} Retrieved Excerpts", expanded=False):
                        for text, meta, score in msg["sources"]:
                            src = meta.get("source", "Transcript")
                            ts = f"{meta.get('start_time', '00:00:00')} → {meta.get('end_time', '00:00:00')}"
                            st.markdown(
                                f'<div class="chunk-box">'
                                f'<span class="source-badge">Source: {src}</span> '
                                f'<span class="timestamp-badge">Time: {ts}</span> '
                                f'<span class="score-badge">Rerank Score: {score:.3f}</span>'
                                f'<br><br>{text}</div>',
                                unsafe_allow_html=True,
                            )

        # Suggested Starters
        if not st.session_state.vrag_messages:
            st.markdown("**Suggested Queries:**")
            suggestions = [
                "What are the central themes discussed across all episodes?",
                "How do the guests compare in their perspectives on artificial intelligence?",
                "What was said regarding autonomous systems and self-driving vehicles?",
                "What major technological hurdles were highlighted for the next decade?",
            ]
            cols = st.columns(2)
            for i, sug in enumerate(suggestions):
                if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                    st.session_state["vrag_prefill"] = sug
                    st.rerun()

        # Chat Input Bar
        user_input = st.chat_input("Ask any question across the indexed transcripts...")
        if not user_input and "vrag_prefill" in st.session_state:
            user_input = st.session_state.pop("vrag_prefill")

        if user_input:
            api_key = get_vrag_api_key()
            if not api_key:
                st.error("Please provide your Groq API key in the sidebar or configure it in .env / Streamlit secrets.")
                st.stop()

            st.session_state.vrag_messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Executing hybrid vector & BM25 retrieval..."):
                        hits = st.session_state.vrag.retrieve(user_input)
                        context = st.session_state.vrag.format_context(hits)

                    with st.spinner("Generating grounded response with Groq LLM..."):
                        answer = call_llm(st.session_state.vrag_messages, context, api_key)

                    st.markdown(answer)

                    if hits:
                        with st.expander(f"{len(hits)} Source Excerpts", expanded=False):
                            for text, meta, score in hits:
                                src = meta.get("source", "Transcript")
                                ts = f"{meta.get('start_time', '00:00:00')} → {meta.get('end_time', '00:00:00')}"
                                st.markdown(
                                    f'<div class="chunk-box">'
                                    f'<span class="source-badge">Source: {src}</span> '
                                    f'<span class="timestamp-badge">Time: {ts}</span> '
                                    f'<span class="score-badge">Score: {score:.3f}</span>'
                                    f'<br><br>{text}</div>',
                                    unsafe_allow_html=True,
                                )

                    st.session_state.vrag_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": hits,
                    })

                except Exception as exc:
                    error_msg = f"Inference Error ({type(exc).__name__}): {exc}"
                    st.error(error_msg)
                    st.session_state.vrag_messages.append({"role": "assistant", "content": error_msg})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: SRT TRANSCRIPTS EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif current_page == "SRT Transcripts Explorer":
    st.markdown('<div class="main-header">SRT Transcripts Library</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Inspect, search, and download podcast SRT subtitle files, or index them directly into Vector RAG.</div>',
        unsafe_allow_html=True,
    )

    available_srts = get_available_srt_files()

    if not available_srts:
        st.warning("No .srt transcript files found in the transcripts directory.")
        st.info("Upload .srt files using the uploader below to populate this library.")
    else:
        # Top-level Library Metrics
        total_files = len(available_srts)
        total_kb = sum(f.stat().st_size for f in available_srts) / 1024

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Available SRT Files", total_files)
        m_col2.metric("Total Library Storage", f"{total_kb:.1f} KB")
        m_col3.metric("Indexed in Vector RAG", len(st.session_state.vrag_active_sources))

        st.markdown("---")

        # Select Transcript File
        file_names = [f.name for f in available_srts]
        selected_file_name = st.selectbox(
            "Select Transcript to Inspect",
            file_names,
            index=0,
            help="Choose an SRT file to view its subtitles, search content, or load into RAG.",
        )

        selected_path = next(f for f in available_srts if f.name == selected_file_name)
        cues = parse_srt(selected_path)
        source_stem = selected_path.stem
        is_already_ingested = source_stem in st.session_state.vrag_active_sources

        # Action Bar for the selected file
        col_act1, col_act2, col_act3 = st.columns([2, 1, 1])

        with col_act1:
            if is_already_ingested:
                st.success(f"Status: Already indexed in Vector RAG ({source_stem})")
            else:
                if st.button(f"Index {source_stem} into Vector RAG", type="primary", use_container_width=True):
                    with st.spinner(f"Ingesting {selected_file_name}..."):
                        run_ingest(selected_path, chunk_size, overlap, top_k, top_n)
                    st.success(f"Successfully indexed {selected_file_name}")
                    st.rerun()

        with col_act2:
            st.download_button(
                label="Download SRT",
                data=selected_path.read_bytes(),
                file_name=selected_path.name,
                mime="text/plain",
                use_container_width=True,
            )

        with col_act3:
            st.caption(f"Total Cues: {len(cues):,}")

        # Search within Transcript
        st.markdown("#### Search Dialogue")
        search_query = st.text_input(
            "Filter dialogue cues by keyword or phrase",
            placeholder="Search keywords (e.g., intelligence, physics, rockets, venture)...",
        )

        filtered_cues = cues
        if search_query.strip():
            query_lower = search_query.strip().lower()
            filtered_cues = [c for c in cues if query_lower in c["text"].lower()]
            st.caption(f"Found {len(filtered_cues)} matching subtitle cues out of {len(cues)} total.")

        # Display Mode Tabs
        tab_timeline, tab_reading, tab_raw = st.tabs([
            "Timeline Cues View",
            "Continuous Reading View",
            "Raw SRT Source",
        ])

        with tab_timeline:
            if not filtered_cues:
                st.info("No subtitle cues match the search query.")
            else:
                cues_per_page = 40
                total_pages = max(1, math.ceil(len(filtered_cues) / cues_per_page))

                p_col1, p_col2 = st.columns([1, 3])
                page_num = p_col1.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1,
                    help="Navigate through pages of subtitle cues.",
                )
                p_col2.caption(f"Showing page {page_num} of {total_pages} ({len(filtered_cues)} cues total)")

                start_idx = (page_num - 1) * cues_per_page
                page_cues = filtered_cues[start_idx : start_idx + cues_per_page]

                for cue in page_cues:
                    st.markdown(
                        f'<div class="srt-cue-card">'
                        f'<div class="srt-cue-header">'
                        f'<span class="srt-cue-id">Cue #{cue["id"]}</span>'
                        f'<span class="timestamp-badge">{cue["start"]} → {cue["end"]}</span>'
                        f'</div>'
                        f'<div class="srt-cue-text">{cue["text"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with tab_reading:
            if not filtered_cues:
                st.info("No matching dialogue.")
            else:
                aggregated_text = "\n\n".join(cue["text"] for cue in filtered_cues)
                st.text_area(
                    "Spoken Dialogue (Plain Text)",
                    value=aggregated_text,
                    height=500,
                    help="Clean concatenated speech dialogue from subtitle cues.",
                )

        with tab_raw:
            raw_content = selected_path.read_text(encoding="utf-8-sig", errors="replace")
            st.code(raw_content[:20000] + ("\n... [truncated]" if len(raw_content) > 20000 else ""), language="text")

    st.markdown("---")
    st.markdown("#### Upload Additional SRT Files")
    new_upload = st.file_uploader(
        "Upload a new .srt file to store in the transcripts library",
        type=["srt"],
        key="srt_page_uploader",
    )
    if new_upload:
        dest = TRANSCRIPTS_DIR / new_upload.name
        dest.write_bytes(new_upload.getvalue())
        st.success(f"Saved {new_upload.name} to transcripts directory.")
        st.rerun()
