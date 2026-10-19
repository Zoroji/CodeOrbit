"""
GraphRAG Evaluation Dashboard
==============================
Streamlit app to evaluate and visualize GraphRAG performance across
Local / Global / Hybrid modes using RAGAS metrics and Graph Structural metrics.

Run:  streamlit run eval_app.py
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
EVAL_STORE = ROOT / "eval_store"
RESULTS_PATH = EVAL_STORE / "results.json"

for folder in ["graph_rag", "GraphRAG"]:
    p = ROOT / folder
    if p.exists():
        sys.path.insert(0, str(p))
        break

load_dotenv(ROOT / ".env", override=False)

DEFAULT_GROQ_KEY = os.getenv("groq_api_key", "") or os.getenv("GROQ_API_KEY", "")
DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

try:
    st.set_page_config(
        page_title="GraphRAG Evaluation Dashboard",
        page_icon=":material/analytics:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    pass

st.markdown("""
<style>
    .eval-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0d9488 0%, #10b981 50%, #0284c7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        color: #1f2937;
    }
    .mode-local {
        background: #ccfbf1;
        color: #0f766e;
        border-radius: 6px;
        padding: 2px 10px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .mode-global {
        background: #e0e7ff;
        color: #3730a3;
        border-radius: 6px;
        padding: 2px 10px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .mode-hybrid {
        background: #fef3c7;
        color: #b45309;
        border-radius: 6px;
        padding: 2px 10px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .q-factual {
        background: #dbeafe;
        color: #1d4ed8;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .q-relational {
        background: #fef3c7;
        color: #92400e;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .q-thematic {
        background: #f3e8ff;
        color: #6b21a8;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .answer-box {
        padding: 0.85rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-bottom: 0.6rem;
    }
    .winner-chip {
        background: #10b981;
        color: white;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODE_COLORS = {
    "local": "#0d9488",
    "global": "#6366f1",
    "hybrid": "#f59e0b",
}
MODE_LABELS = {"local": "Local Mode", "global": "Global Mode", "hybrid": "Hybrid Mode"}
METRICS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}
METRIC_EXPLAINERS = {
    "Faithfulness": "Is the generated answer grounded in the retrieved context? Penalizes hallucination.",
    "Answer Relevancy": "Does the answer directly address the question that was asked?",
    "Context Precision": "Of all the chunks retrieved, how many were truly relevant?",
    "Context Recall": "Did retrieval surface ALL the information needed to fully answer?",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Evaluation Configuration")

    default_provider_idx = 0 if DEFAULT_GROQ_KEY else 1
    provider = st.selectbox("LLM Provider", ["groq", "openai"], index=default_provider_idx)

    env_key = DEFAULT_GROQ_KEY if provider == "groq" else DEFAULT_OPENAI_KEY
    if not env_key and hasattr(st, "secrets"):
        if provider == "groq":
            env_key = str(st.secrets.get("groq_api_key", "") or st.secrets.get("GROQ_API_KEY", "")).strip()
        else:
            env_key = str(st.secrets.get("OPENAI_API_KEY", "")).strip()

    if env_key:
        st.success(f"🔒 System {provider.upper()} API Key Active")
        api_key = st.text_input(
            f"Override {provider.upper()} Key (Optional)",
            value="",
            type="password",
            placeholder="Leave blank to use system key",
            help="Your key is protected and never shown on screen.",
        )
    else:
        st.info(f"Enter your {provider.upper()} API key:")
        api_key = st.text_input(
            f"{provider.upper()} API Key",
            value="",
            type="password",
            placeholder=f"Enter {provider} key...",
        )

    st.divider()
    force_rerun = st.checkbox("Force re-run (clears previous cache)", value=False)
    run_btn = st.button("Run Evaluation", type="primary", use_container_width=True)

    st.divider()
    st.markdown("#### RAGAS Metrics Reference")
    for label, desc in METRIC_EXPLAINERS.items():
        with st.expander(label):
            st.caption(desc)

    st.divider()
    st.markdown("#### Question Categories")
    st.markdown("""
- **Factual**: Single-source specific facts (Local handles well)
- **Relational**: Multi-entity links across documents (Graph expansion wins)
- **Thematic**: Global themes synthesized across all episodes (Community summaries win)
""")


# ── Run Evaluation Trigger ────────────────────────────────────────────────────
if run_btn:
    final_key = api_key or env_key
    if not final_key:
        st.sidebar.error("API key required.")
    else:
        if force_rerun:
            for f in ["results.json", "ground_truths.json"]:
                p = EVAL_STORE / f
                if p.exists():
                    p.unlink()

        from evaluate import run_evaluation

        st.markdown("### Running Evaluation...")
        progress = st.progress(0.0, text="Initializing...")
        log_holder = st.empty()
        log_lines = []

        def on_progress(msg):
            log_lines.append(msg)
            pct = min(len(log_lines) / 55, 0.95)
            progress.progress(pct, text=msg[:90])
            log_holder.caption("\n".join(log_lines[-4:]))

        try:
            run_evaluation(provider, final_key, progress_fn=on_progress)
            progress.progress(1.0, text="Evaluation completed successfully.")
            log_holder.empty()
            st.success("Evaluation complete! Results saved.")
            st.rerun()
        except Exception as exc:
            progress.empty()
            st.error(f"Evaluation failed: {exc}")
            st.exception(exc)
            st.stop()


# ── Load Results ──────────────────────────────────────────────────────────────
def load_results():
    if RESULTS_PATH.exists():
        try:
            return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


results = load_results()

# ── Main Header ───────────────────────────────────────────────────────────────
st.markdown('<div class="eval-header">GraphRAG Evaluation Dashboard</div>', unsafe_allow_html=True)
st.caption("Comprehensive benchmarking of Local / Global / Hybrid query modes using RAGAS and Graph Structural metrics.")

if results is None:
    st.info("No evaluation results found yet. Configure your API key in the sidebar and click 'Run Evaluation'.")
    st.markdown("""
    **What happens during evaluation:**
    1. **GraphRAG Setup**: Builds or loads the entity knowledge graph & community clusters from local transcripts.
    2. **Ground Truth Synthesis**: Generates reference answers using Hybrid mode as an oracle (cached for consistency).
    3. **Multi-Mode Execution**: Queries all 10 questions across Local, Global, and Hybrid modes.
    4. **RAGAS & Graph Scoring**: Evaluates Faithfulness, Answer Relevancy, Context Precision, Context Recall, Entity Coverage, and Graph Utilization.
    """)
    st.stop()


# ── Parse Results Data ────────────────────────────────────────────────────────
meta = results.get("metadata", {})
questions = results.get("questions", [])
res = results.get("results", {})
modes = list(MODE_LABELS.keys())
metric_labels = list(METRICS.values())

# Aggregate dataframe
score_rows = []
for mode in modes:
    s = res.get(mode, {}).get("scores", {})
    row = {"Mode": MODE_LABELS[mode]}
    for key, label in METRICS.items():
        row[label] = s.get(key)
    score_rows.append(row)
scores_df = pd.DataFrame(score_rows).set_index("Mode")


# ── Section 1: Overview Metadata ──────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Transcripts Ingested", len(meta.get("transcripts", [])))
c2.metric("Evaluated Questions", meta.get("n_questions", len(questions)))
c3.metric("Evaluated Modes", 3)
c4.metric("Provider / Model", f"{meta.get('provider', 'groq').upper()}")

st.divider()


# ── Section 2: Overall Metric Comparison ──────────────────────────────────────
st.markdown('<div class="section-title">Overall RAGAS Scores Across Modes</div>', unsafe_allow_html=True)
st.caption("Average across all test questions. Score range 0.0 to 1.0 (Higher is better).")

fig_bar = go.Figure()
for mode in modes:
    s = res.get(mode, {}).get("scores", {})
    fig_bar.add_trace(go.Bar(
        name=MODE_LABELS[mode],
        x=metric_labels,
        y=[s.get(k) for k in METRICS],
        marker_color=MODE_COLORS[mode],
        text=[f"{s.get(k):.3f}" if s.get(k) is not None else "N/A" for k in METRICS],
        textposition="outside",
        width=0.22,
    ))

fig_bar.update_layout(
    barmode="group",
    height=400,
    yaxis=dict(range=[0, 1.15], title="Score (0–1)", gridcolor="#f3f4f6"),
    xaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=10, b=30, l=40, r=10),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_bar, use_container_width=True)

st.dataframe(
    scores_df.style
        .format("{:.3f}", na_rep="N/A")
        .highlight_max(axis=0, color="#dcfce7")
        .highlight_min(axis=0, color="#fee2e2"),
    use_container_width=True,
)

st.divider()


# ── Section 3: Radar Capability Profiles ──────────────────────────────────────
st.markdown('<div class="section-title">Mode Capability Profiles</div>', unsafe_allow_html=True)
st.caption("Radar visualization of retrieval & synthesis balance.")

fig_radar = go.Figure()
for mode in modes:
    s = res.get(mode, {}).get("scores", {})
    vals = [s.get(k) or 0 for k in METRICS]
    vals_closed = vals + [vals[0]]
    cats_closed = metric_labels + [metric_labels[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=cats_closed,
        fill="toself",
        name=MODE_LABELS[mode],
        line_color=MODE_COLORS[mode],
        fillcolor=MODE_COLORS[mode],
        opacity=0.22,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
    margin=dict(t=20, b=60),
    paper_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()


# ── Section 4: Context Recall by Question Type ────────────────────────────────
st.markdown('<div class="section-title">Context Recall by Question Type</div>', unsafe_allow_html=True)
st.caption("Validates architectural strengths: Local on relational connections, Global on broad themes.")

q_by_type = {}
for q in questions:
    q_by_type.setdefault(q["type"], []).append(q["question"])

type_rows = []
for q_type in ["factual", "relational", "thematic"]:
    type_qs = q_by_type.get(q_type, [])
    for mode in modes:
        per_q = res.get(mode, {}).get("scores", {}).get("per_question", [])
        vals = [r.get("context_recall") for r in per_q
                if r.get("user_input") in type_qs and r.get("context_recall") is not None]
        avg = round(sum(vals) / len(vals), 3) if vals else None
        type_rows.append({"Question Type": q_type.capitalize(), "Mode": MODE_LABELS[mode], "Context Recall": avg})

type_df = pd.DataFrame(type_rows)
if type_df["Context Recall"].notna().any():
    fig_type = px.bar(
        type_df,
        x="Question Type",
        y="Context Recall",
        color="Mode",
        barmode="group",
        color_discrete_map={MODE_LABELS[m]: MODE_COLORS[m] for m in modes},
        height=380,
    )
    fig_type.update_layout(
        yaxis=dict(range=[0, 1.15], title="Context Recall"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_type, use_container_width=True)

st.divider()


# ── Section 5: Graph-Specific Structural Metrics ──────────────────────────────
st.markdown('<div class="section-title">Graph Structural Metrics</div>', unsafe_allow_html=True)
st.caption("Directly inspects knowledge graph utilization, entity coverage, and community coherence.")

# Calculate average graph utilization for Local & Hybrid
local_samples = res.get("local", {}).get("samples", [])
hybrid_samples = res.get("hybrid", {}).get("samples", [])
global_samples = res.get("global", {}).get("samples", [])

def get_avg_metric(samples, key):
    vals = [s.get("graph_metrics", {}).get(key) for s in samples
            if s.get("graph_metrics", {}).get(key) is not None]
    return round(sum(vals) / len(vals), 3) if vals else None

col_g1, col_g2, col_g3 = st.columns(3)
col_g1.metric("Local Graph Utilization", f"{get_avg_metric(local_samples, 'graph_utilization') or 0:.1%}", help="Fraction of chunks sourced exclusively through entity graph expansion.")
col_g2.metric("Hybrid Graph Utilization", f"{get_avg_metric(hybrid_samples, 'graph_utilization') or 0:.1%}", help="Fraction of chunks added by graph traversal in Hybrid mode.")
col_g3.metric("Global Community Coherence", f"{get_avg_metric(global_samples, 'community_coherence') or 0:.1%}", help="% of top entities mentioned in LLM-generated community summaries.")

st.divider()


# ── Section 6: Question-by-Question Deep Dive ─────────────────────────────────
st.markdown('<div class="section-title">Question-by-Question Analysis</div>', unsafe_allow_html=True)

for q in questions:
    q_id = q["id"]
    with st.expander(f"{q_id} [{q['type'].upper()}]: {q['question']}"):
        st.markdown(f"**Question Category:** <span class='q-{q['type']}'>{q['type'].upper()}</span>", unsafe_allow_html=True)

        gt_text = ""
        for m in modes:
            for s in res.get(m, {}).get("samples", []):
                if s["id"] == q_id and s.get("ground_truth"):
                    gt_text = s["ground_truth"]
                    break
            if gt_text:
                break

        if gt_text:
            st.markdown(f"**Reference Ground Truth:**\n> {gt_text}")

        cols = st.columns(3)
        for i, mode in enumerate(modes):
            col = cols[i]
            sample = next((s for s in res.get(mode, {}).get("samples", []) if s["id"] == q_id), None)
            with col:
                st.markdown(f'<span class="mode-{mode}">{MODE_LABELS[mode]}</span>', unsafe_allow_html=True)
                if sample:
                    ans = sample.get("answer", "No answer recorded.")
                    st.markdown(f'<div class="answer-box" style="border-left: 3px solid {MODE_COLORS[mode]}; background: #fbfbfd;">{ans}</div>', unsafe_allow_html=True)
                    gm = sample.get("graph_metrics", {})
                    if "graph_utilization" in gm and gm["graph_utilization"] is not None:
                        st.caption(f"Graph Utilization: {gm['graph_utilization']:.1%}")
                    if "entity_coverage_rate" in gm and gm["entity_coverage_rate"] is not None:
                        st.caption(f"Entity Coverage: {gm['entity_coverage_rate']:.1%}")
