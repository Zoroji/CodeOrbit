"""
Unified AI Podcast Intelligence Hub
===================================
Master Portal orchestrating:
1. Vector RAG (ChromaDB + BM25 + Cross-Encoder)
2. GraphRAG (NetworkX + Community Detection + Local/Global/Hybrid)
3. Evaluation Dashboard (RAGAS + Graph Structural Metrics)

Run:
    streamlit run main.py
"""

import sys
import os
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Simple_rag"))
sys.path.insert(0, str(ROOT / "graph_rag"))

# Automatically populate os.environ from st.secrets if deployed on Streamlit Cloud
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

# Configure master page settings
st.set_page_config(
    page_title="AI Podcast Intelligence Hub",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def home_portal():
    # ─── Custom CSS for Premium Glassmorphic Cards ──────────────────────────────
    st.markdown("""
    <style>
        .hub-header {
            font-size: 2.4rem;
            font-weight: 850;
            background: linear-gradient(135deg, #4f46e5 0%, #0d9488 50%, #06b6d4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.3rem;
        }
        .hub-sub {
            color: #64748b;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }
        .portal-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 24px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
            transition: all 0.25s ease-in-out;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .portal-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 28px -4px rgba(0, 0, 0, 0.1);
            border-color: #cbd5e1;
        }
        .card-icon {
            font-size: 2.2rem;
            margin-bottom: 12px;
        }
        .card-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 8px;
        }
        .card-desc {
            color: #475569;
            font-size: 0.92rem;
            line-height: 1.5;
            margin-bottom: 18px;
        }
        .card-tag {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .tag-blue { background: #e0e7ff; color: #3730a3; }
        .tag-teal { background: #ccfbf1; color: #115e59; }
        .tag-purple { background: #f3e8ff; color: #6b21a8; }
        .launch-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            padding: 10px 16px;
            border-radius: 10px;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.95rem;
            transition: background 0.2s;
            text-align: center;
        }
        .btn-vector { background: #4f46e5; color: white !important; }
        .btn-vector:hover { background: #4338ca; }
        .btn-graph { background: #0d9488; color: white !important; }
        .btn-graph:hover { background: #0f766e; }
        .btn-eval { background: #0284c7; color: white !important; }
        .btn-eval:hover { background: #0369a1; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hub-header">🎙️ AI Podcast Intelligence Hub</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hub-sub">Compare, explore, and benchmark <b>Vector RAG</b> against <b>Knowledge Graph RAG</b> across long-form tech podcast transcripts.</div>',
        unsafe_allow_html=True
    )

    # ── Status Metrics ──
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Transcripts Available", "4 Episodes", "Elon, Bezos, Jensen, Sam")
    with col_m2:
        st.metric("Retrieval Engines", "2 Active", "Vector + Graph")
    with col_m3:
        st.metric("Graph Communities", "Modularity", "Leiden / Greedy")
    with col_m4:
        st.metric("Evaluation Suite", "7 Metrics", "RAGAS + Structural")

    st.markdown("---")
    st.markdown("### 🚀 Launch Workspaces")
    st.caption("Select a pipeline below. Click any button to switch directly, or use the link to open in a new browser tab/window.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="portal-card">
            <div>
                <div class="card-icon">⚡</div>
                <span class="card-tag tag-blue">Standard Architecture</span>
                <div class="card-title">Vector RAG</div>
                <div class="card-desc">
                    Fast dense semantic search with <b>ChromaDB</b> (BGE-small), 
                    sparse keyword matching with <b>BM25</b>, 
                    and cross-encoder reranking (<b>MiniLM</b>). 
                    Best for precise, single-passage factual lookups.
                </div>
            </div>
            <div>
                <a href="/vector_rag" target="_blank" class="launch-btn btn-vector">↗ Open in New Window</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Switch to Vector RAG", key="btn_nav_vector", use_container_width=True):
            st.switch_page("Simple_rag/app.py")

    with col2:
        st.markdown("""
        <div class="portal-card">
            <div>
                <div class="card-icon">🕸️</div>
                <span class="card-tag tag-teal">Graph Augmented</span>
                <div class="card-title">GraphRAG</div>
                <div class="card-desc">
                    Entity-relation knowledge graph using <b>spaCy</b>, <b>NetworkX</b>, and 
                    <b>Greedy Modularity</b> community detection. 
                    Supports <b>Local</b> (1-hop traversal), <b>Global</b> (thematic community summaries), 
                    and <b>Hybrid</b> multi-hop queries.
                </div>
            </div>
            <div>
                <a href="/graph_rag" target="_blank" class="launch-btn btn-graph">↗ Open in New Window</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Switch to GraphRAG", key="btn_nav_graph", use_container_width=True):
            st.switch_page("graph_rag/app.py")

    with col3:
        st.markdown("""
        <div class="portal-card">
            <div>
                <div class="card-icon">📊</div>
                <span class="card-tag tag-purple">Benchmark & Audit</span>
                <div class="card-title">Evaluation Hub</div>
                <div class="card-desc">
                    Comprehensive testing across 10 benchmark queries. Evaluates 
                    <b>RAGAS</b> (Faithfulness, Relevancy, Precision, Recall) 
                    and <b>Graph Metrics</b> (Graph Utilization, Entity Coverage, Community Coherence) 
                    with interactive charts.
                </div>
            </div>
            <div>
                <a href="/evaluation" target="_blank" class="launch-btn btn-eval">↗ Open in New Window</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Switch to Evaluation", key="btn_nav_eval", use_container_width=True):
            st.switch_page("eval_app.py")

    # ── Architecture Comparison Section ──
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Technical Comparison at a Glance")

    comp_data = {
        "Feature / Capability": [
            "Core Indexing",
            "Entity Awareness",
            "Cross-Episode Relations",
            "Thematic Summarization",
            "Factual Needle Queries",
            "Global Synthesis Queries",
            "Memory Footprint",
            "Inference Speed"
        ],
        "Vector RAG (Simple RAG)": [
            "ChromaDB (BGE-small) + BM25",
            "None (Raw chunk tokens)",
            "Limited to chunk overlap",
            "Not supported",
            "⭐⭐⭐⭐⭐ (Instant & accurate)",
            "⭐⭐ (Fails across many docs)",
            "~250 MB",
            "~120 ms"
        ],
        "GraphRAG (Knowledge Graph)": [
            "ChromaDB + NetworkX Graph + Community Summaries",
            "Named Entities (spaCy NER + OntoNotes)",
            "High (1-hop & multi-hop entity co-occurrence)",
            "High (Modularity-based community summaries)",
            "⭐⭐⭐⭐ (High precision via local mode)",
            "⭐⭐⭐⭐⭐ (High via community pre-summaries)",
            "~400 MB (Graph + vectors)",
            "~250-400 ms"
        ]
    }
    st.table(comp_data)

# ─── Navigation Setup ──────────────────────────────────────────────────────────
pages = [
    st.Page(home_portal, title="Portal Hub", icon="🏠", default=True),
    st.Page("Simple_rag/app.py", title="Vector RAG", icon="⚡", url_path="vector_rag"),
    st.Page("graph_rag/app.py", title="GraphRAG", icon="🕸️", url_path="graph_rag"),
    st.Page("eval_app.py", title="Evaluation Dashboard", icon="📊", url_path="evaluation"),
]

pg = st.navigation(pages)
pg.run()
