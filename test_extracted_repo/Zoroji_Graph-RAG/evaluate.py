"""
GraphRAG Evaluation — Core Engine
===================================
Evaluates GraphRAG across 3 query modes using RAGAS metrics & Graph structural metrics.

Modes:
  Local  -> entity-graph-expanded retrieval  (factual / relational questions)
  Global -> community-summary-based retrieval (thematic / broad questions)
  Hybrid -> combines both                     (best overall)

RAGAS Metrics:
  Faithfulness      — does the answer stick to the retrieved context?
  Answer Relevancy  — does the answer actually address the question?
  Context Precision — of chunks retrieved, how many were truly relevant?
  Context Recall    — did retrieval surface ALL info needed to answer?

Graph Structural Metrics:
  Entity Coverage Rate  — % of ground truth entities present in retrieved context
  Graph Utilization     — fraction of chunks coming from graph expansion vs flat search
  Community Coherence   — alignment of community summaries with top entities

Run:
  python evaluate.py --provider groq   --api_key gsk-...
  python evaluate.py --provider openai --api_key sk-...
"""

import sys
import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Optional, Callable

ROOT = Path(__file__).resolve().parent

# Support both graph_rag and GraphRAG folder conventions
for folder in ["graph_rag", "GraphRAG"]:
    p = ROOT / folder
    if p.exists():
        sys.path.insert(0, str(p))
        break

from dotenv import load_dotenv
load_dotenv(ROOT / ".env", override=False)

DEFAULT_GROQ_KEY = os.getenv("groq_api_key", "") or os.getenv("GROQ_API_KEY", "")
DEFAULT_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

EVAL_STORE = ROOT / "eval_store"


# ── Transcripts Discovery ─────────────────────────────────────────────────────

def get_transcript_paths():
    """Discover all available .srt transcripts across standard directories."""
    search_dirs = [
        ROOT / "graph_rag" / "transcripts",
        ROOT / "Simple_rag" / "transcripts",
        ROOT / "transcripts",
        ROOT,
    ]
    found = []
    seen = set()
    for d in search_dirs:
        if d.exists() and d.is_dir():
            for f in sorted(d.glob("*.srt")):
                if f.name not in seen:
                    seen.add(f.name)
                    found.append(str(f))
    return found


TRANSCRIPT_PATHS = get_transcript_paths()


# ── Dynamic Test Questions ────────────────────────────────────────────────────
# Questions are categorized into:
#   factual    -> specific single-source fact   -> Local should handle well
#   relational -> entity connection across docs -> Local graph expansion wins
#   thematic   -> synthesis across all docs     -> Global community summaries win

def build_test_questions(transcript_names: list) -> list:
    names_lower = " ".join(transcript_names).lower()
    has_bezos = "bezos" in names_lower
    has_jensen = "jensen" in names_lower or "nvidia" in names_lower

    questions = []

    # 1. Factual Questions (3)
    if has_jensen:
        questions.append({
            "id": "Q01",
            "question": "How does Jensen Huang describe NVIDIA's shift from chip-scale to rack-scale design and why does he say it became necessary?",
            "type": "factual",
            "expected_best_mode": "local",
        })
    elif has_bezos:
        questions.append({
            "id": "Q01",
            "question": "How does Jeff Bezos describe the engineering and manufacturing philosophy behind Blue Origin's reusable rockets?",
            "type": "factual",
            "expected_best_mode": "local",
        })
    else:
        questions.append({
            "id": "Q01",
            "question": "What does Elon Musk explain regarding the physics and manufacturing challenges of Starship?",
            "type": "factual",
            "expected_best_mode": "local",
        })

    questions.extend([
        {
            "id": "Q02",
            "question": "What does Sam Altman say about the OpenAI board saga and how it affected him personally and professionally?",
            "type": "factual",
            "expected_best_mode": "local",
        },
        {
            "id": "Q03",
            "question": "What does Elon Musk say about the role of AI in modern warfare and its implications for geopolitics?",
            "type": "factual",
            "expected_best_mode": "local",
        },
    ])

    # 2. Relational Questions (4)
    if has_bezos:
        questions.append({
            "id": "Q04",
            "question": "How do Elon Musk and Jeff Bezos compare in their visions for space exploration, rocket reusability, and manufacturing off Earth?",
            "type": "relational",
            "expected_best_mode": "local",
        })
    else:
        questions.append({
            "id": "Q04",
            "question": "How are NVIDIA's GPU architecture, large-scale distributed training, and the AI infrastructure buildout connected across the podcast discussions?",
            "type": "relational",
            "expected_best_mode": "local",
        })

    questions.append({
        "id": "Q05",
        "question": "How are Sam Altman, Elon Musk, and Ilya Sutskever connected and how did their relationships shape OpenAI's direction?",
        "type": "relational",
        "expected_best_mode": "local",
    })

    if has_bezos:
        questions.append({
            "id": "Q06",
            "question": "How do Sam Altman and Jeff Bezos discuss the scaling of compute infrastructure, cloud architecture, and capital requirements for frontier AI?",
            "type": "relational",
            "expected_best_mode": "local",
        })
    else:
        questions.append({
            "id": "Q06",
            "question": "How do Jensen Huang and Sam Altman compare in their views on what compute scaling means for reaching AGI?",
            "type": "relational",
            "expected_best_mode": "local",
        })

    questions.append({
        "id": "Q07",
        "question": "What connections exist between robotics, physical AI, neural networks, and autonomous hardware across the conversations?",
        "type": "relational",
        "expected_best_mode": "hybrid",
    })

    # 3. Thematic Questions (3)
    questions.extend([
        {
            "id": "Q08",
            "question": "What are the main recurring themes about the risks and benefits of AGI across all podcast conversations?",
            "type": "thematic",
            "expected_best_mode": "global",
        },
        {
            "id": "Q09",
            "question": "How do the speakers collectively discuss the concentration of AI power and who should govern frontier artificial intelligence?",
            "type": "thematic",
            "expected_best_mode": "global",
        },
        {
            "id": "Q10",
            "question": "What common perspective on the relationship between compute, data, energy constraints, and the pace of AI progress emerges across the interviews?",
            "type": "thematic",
            "expected_best_mode": "global",
        },
    ])

    return questions


# ── LLM Helpers with Retry Logic ──────────────────────────────────────────────

def call_llm(prompt: str, provider: str, api_key: str, max_tokens: int = 400) -> str:
    """Call LLM with exponential backoff for Groq/OpenAI rate limits."""
    for attempt in range(5):
        try:
            if provider == "groq":
                from groq import Groq
                client = Groq(api_key=api_key.strip())
                candidate_models = [GROQ_MODEL, "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
                resp = None
                for m_idx, m_name in enumerate(candidate_models):
                    try:
                        resp = client.chat.completions.create(
                            model=m_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=0.1,
                        )
                        break
                    except Exception as m_exc:
                        m_err = str(m_exc).lower()
                        if ("model_not_found" in m_err or "does not exist" in m_err or "404" in m_err) and m_idx < len(candidate_models) - 1:
                            continue
                        raise m_exc
            else:
                from openai import OpenAI
                client = OpenAI(api_key=api_key.strip())
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.1,
                )
            return resp.choices[0].message.content.strip()

        except Exception as exc:
            err_str = str(exc).lower()
            if "429" in err_str or "rate_limit" in err_str or "too many requests" in err_str:
                wait_time = (attempt + 1) * 3
                time.sleep(wait_time)
            else:
                raise exc

    raise RuntimeError("Exceeded maximum retries for LLM invocation.")


def generate_answer(question: str, context: str, mode: str,
                    provider: str, api_key: str) -> str:
    mode_note = {
        "local":  "You have entity-graph-expanded excerpts relevant to the question.",
        "global": "You have thematic community summaries and excerpts. Synthesize the themes.",
        "hybrid": "You have both entity-specific excerpts and thematic community summaries.",
    }.get(mode, "")

    prompt = f"""You are analyzing podcast transcripts using a knowledge graph RAG system. {mode_note}

RULES:
1. Answer ONLY using the CONTEXT below.
2. If context is insufficient say: "This topic is not sufficiently covered in the transcripts."
3. Cite source and approximate timestamp where possible.
4. Be thorough, factual, and concise.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:"""
    return call_llm(prompt, provider, api_key, max_tokens=600)


# ── Pipeline Setup ────────────────────────────────────────────────────────────

def setup_graphrag(llm_fn: Callable, progress_fn: Optional[Callable] = None, transcript_paths: Optional[list] = None):
    from graph_pipeline import GraphRAG

    paths = transcript_paths or TRANSCRIPT_PATHS
    if not paths:
        raise FileNotFoundError(
            "No .srt transcripts found. Please place .srt files in graph_rag/transcripts or Simple_rag/transcripts."
        )

    grag = GraphRAG(persist_dir=str(EVAL_STORE / "graph_rag_store"))

    if grag.load_state():
        if progress_fn:
            progress_fn("GraphRAG: restored from cache")
        if grag.has_summaries():
            grag._cache_summary_embeddings()
        return grag

    for path in paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Transcript not found: {path}")
        if progress_fn:
            progress_fn(f"Ingesting: {Path(path).stem[:50]}...")
        grag.ingest(path, llm_fn=llm_fn)

    return grag


# ── Graph-Specific Structural Metrics ─────────────────────────────────────────

def compute_graph_metrics(grag, result: dict, question: str, mode: str,
                          ground_truth: str = "") -> dict:
    """
    Computes three graph-specific structural metrics:
      1. entity_coverage_rate: % of ground-truth entities present in retrieved context.
      2. graph_utilization: % of retrieved chunks sourced from graph expansion (not flat search).
      3. community_coherence: % of community top entities explicitly mentioned in summary.
    """
    metrics = {}

    # 1. Entity Coverage Rate
    if ground_truth:
        gt_entities = set(grag.extract_entities(ground_truth))
        ctx_entities = set()
        for text, _, _ in result.get("hits", []):
            ctx_entities.update(grag.extract_entities(text))
        for summary in result.get("summaries", []):
            ctx_entities.update(grag.extract_entities(summary))

        if gt_entities:
            covered = gt_entities & ctx_entities
            metrics["entity_coverage_rate"] = round(len(covered) / len(gt_entities), 3)
            metrics["gt_entities_total"] = len(gt_entities)
            metrics["gt_entities_covered"] = len(covered)
        else:
            metrics["entity_coverage_rate"] = None

    # 2. Graph Utilization (local & hybrid only)
    if mode in ("local", "hybrid"):
        try:
            from graph_pipeline import reciprocal_rank_fusion
            dense = grag.vector_store.dense_search(question, top_k=grag.top_k_retrieve)
            sparse = grag.bm25_index.search(question, top_k=grag.top_k_retrieve)
            fused = reciprocal_rank_fusion(dense, sparse)
            base_keys = {
                f"{meta.get('source', '')}_{meta.get('chunk_id', '')}"
                for _, meta, _ in fused
            }
            final_hits = result.get("hits", [])
            if final_hits:
                graph_only = [
                    1 for _, meta, _ in final_hits
                    if f"{meta.get('source', '')}_{meta.get('chunk_id', '')}" not in base_keys
                ]
                metrics["graph_utilization"] = round(len(graph_only) / len(final_hits), 3)
                metrics["graph_expanded_chunks"] = len(graph_only)
                metrics["base_retrieved_chunks"] = len(final_hits) - len(graph_only)
            else:
                metrics["graph_utilization"] = 0.0
        except Exception as e:
            metrics["graph_utilization"] = None
            metrics["graph_utilization_error"] = str(e)

    # 3. Community Coherence (global & hybrid only)
    if mode in ("global", "hybrid"):
        summaries_used = result.get("summaries", [])
        metrics["communities_used"] = len(summaries_used)
        if summaries_used:
            coherence_scores = []
            for summary_text in summaries_used:
                for cd in grag._communities.values():
                    if cd.get("summary") == summary_text:
                        top_ents = cd["entities"][:10]
                        if top_ents:
                            summary_lower = summary_text.lower()
                            mentioned = sum(1 for e in top_ents if e.lower() in summary_lower)
                            coherence_scores.append(mentioned / len(top_ents))
                        break
            if coherence_scores:
                metrics["community_coherence"] = round(sum(coherence_scores) / len(coherence_scores), 3)
                metrics["communities_evaluated"] = len(coherence_scores)

    return metrics


# ── Query Mode Execution ──────────────────────────────────────────────────────

def query_mode(grag, question: str, mode: str, provider: str, api_key: str,
               ground_truth: str = "") -> dict:
    result = grag.query(question, mode=mode)
    chunks = [text for text, _, _ in result.get("hits", [])]
    summaries = result.get("summaries", [])
    contexts = (summaries + chunks) if summaries else chunks
    answer = generate_answer(question, grag.format_context(result), mode, provider, api_key)
    graph_metrics = compute_graph_metrics(grag, result, question, mode, ground_truth)
    return {"answer": answer, "contexts": contexts, "graph_metrics": graph_metrics}


# ── Ground Truth Generation ───────────────────────────────────────────────────

def get_ground_truths(grag, test_questions: list, provider: str, api_key: str,
                      progress_fn: Optional[Callable] = None) -> dict:
    """Use Hybrid mode as oracle to generate reference answers (cached on disk)."""
    path = EVAL_STORE / "ground_truths.json"
    if path.exists():
        if progress_fn:
            progress_fn("Ground truths: loaded from cache")
        return json.loads(path.read_text())

    if progress_fn:
        progress_fn("Generating ground truths via Hybrid oracle...")
    gt = {}
    for i, q in enumerate(test_questions):
        if progress_fn:
            progress_fn(f"Ground truth {i+1}/{len(test_questions)}: {q['id']}")
        result = grag.query(q["question"], mode="hybrid")
        context = grag.format_context(result)
        prompt = f"""Write a comprehensive REFERENCE ANSWER for the question below based strictly on the context.
Include all essential facts and cite approximate sources/timestamps.

QUESTION: {q["question"]}
CONTEXT:
{context}
REFERENCE ANSWER:"""
        gt[q["id"]] = call_llm(prompt, provider, api_key, max_tokens=500)
        time.sleep(0.8)

    path.write_text(json.dumps(gt, indent=2, ensure_ascii=False))
    return gt


# ── RAGAS Scoring with Groq & HuggingFace Support ────────────────────────────

def score_with_ragas(samples: list, provider: str, api_key: str) -> dict:
    try:
        import warnings
        import pandas as pd
        from ragas import evaluate, EvaluationDataset
        from ragas.dataset_schema import SingleTurnSample
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError:
        raise ImportError("Run: pip install -r eval_requirements.txt")

    # LLM Wrapper for RAGAS Judge
    if provider == "groq":
        from langchain_groq import ChatGroq
        ragas_llm = LangchainLLMWrapper(
            ChatGroq(model=GROQ_MODEL, api_key=api_key)
        )
        # Groq has no embeddings API; use local HuggingFace embeddings
        from langchain_community.embeddings import HuggingFaceEmbeddings
        ragas_emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )
    else:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        ragas_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0)
        )
        ragas_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key))

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["contexts"],
            reference=s["ground_truth"],
        ) for s in samples
    ])

    result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    df = result.to_pandas()
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if key in df.columns:
            val = df[key].mean()
            scores[key] = round(float(val), 4) if not pd.isna(val) else None
        else:
            scores[key] = None
    scores["per_question"] = df.to_dict(orient="records")
    return scores


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_evaluation(provider: str, api_key: str,
                   progress_fn: Optional[Callable] = None,
                   custom_transcript_paths: Optional[list] = None) -> dict:
    EVAL_STORE.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"  {msg}")
        if progress_fn:
            progress_fn(msg)

    paths = custom_transcript_paths or get_transcript_paths()
    if not paths:
        raise FileNotFoundError("No transcript .srt files found to evaluate.")

    test_questions = build_test_questions([Path(p).stem for p in paths])

    # 1. Setup GraphRAG
    log("Setting up GraphRAG pipeline...")
    grag = setup_graphrag(lambda p: call_llm(p, provider, api_key, 300), log, paths)

    # 2. Ground truths
    log("Preparing ground truth answers via Hybrid oracle...")
    gt = get_ground_truths(grag, test_questions, provider, api_key, log)

    # 3. Query all modes
    raw = {"local": [], "global": [], "hybrid": []}
    for i, q in enumerate(test_questions):
        for mode in ["local", "global", "hybrid"]:
            log(f"[{i+1}/{len(test_questions)}] {mode:6s} -> {q['id']}")
            try:
                out = query_mode(grag, q["question"], mode, provider, api_key,
                                 ground_truth=gt.get(q["id"], ""))
                raw[mode].append({
                    "id": q["id"], "question": q["question"], "type": q["type"],
                    "answer": out["answer"], "contexts": out["contexts"],
                    "ground_truth": gt.get(q["id"], ""),
                    "graph_metrics": out.get("graph_metrics", {}),
                })
            except Exception as e:
                log(f"  ERROR {mode}/{q['id']}: {e}")
                raw[mode].append({
                    "id": q["id"], "question": q["question"], "type": q["type"],
                    "answer": f"[Error: {e}]", "contexts": [],
                    "ground_truth": gt.get(q["id"], ""),
                    "graph_metrics": {},
                })
            time.sleep(0.5)

    # 4. RAGAS Scoring
    log("Running RAGAS scoring...")
    scored = {}
    for mode, samples in raw.items():
        log(f"  Scoring mode: {mode}...")
        try:
            scored[mode] = {"scores": score_with_ragas(samples, provider, api_key), "samples": samples}
        except Exception as e:
            log(f"  RAGAS failed for {mode}: {e}")
            scored[mode] = {"scores": {"error": str(e)}, "samples": samples}

    # 5. Save results
    output = {
        "metadata": {
            "provider": provider,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "transcripts": [Path(p).stem for p in paths],
            "n_questions": len(test_questions),
        },
        "questions": test_questions,
        "results": scored,
    }
    out_path = EVAL_STORE / "results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    log(f"Results saved to {out_path}")
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GraphRAG using RAGAS and graph structural metrics.")
    parser.add_argument("--provider", choices=["openai", "groq"], default="groq" if DEFAULT_GROQ_KEY else "openai")
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    key = args.api_key or (
        (os.getenv("groq_api_key") or os.getenv("GROQ_API_KEY")) if args.provider == "groq"
        else os.getenv("OPENAI_API_KEY")
    )

    if not key:
        print(f"Error: API key for {args.provider.upper()} not provided. Set in .env or pass --api_key."); sys.exit(1)

    print(f"\nEvaluating GraphRAG | provider={args.provider} | Model={GROQ_MODEL if args.provider == 'groq' else 'gpt-4o-mini'}\n")
    results = run_evaluation(args.provider, key)

    print("\n" + "=" * 62)
    print(f"{'Mode':<10} {'Faithful':>10} {'AnsRel':>8} {'CtxPrec':>9} {'CtxRec':>8}")
    print("=" * 62)
    for mode, data in results["results"].items():
        s = data.get("scores", {})
        if "error" in s:
            print(f"{mode:<10}  Error: {s['error']}")
        else:
            vals = [f"{s.get(m):.3f}" if s.get(m) is not None else "  N/A"
                    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
            print(f"{mode:<10} {vals[0]:>10} {vals[1]:>8} {vals[2]:>9} {vals[3]:>8}")
    print("=" * 62)
    print("\nView dashboard:  streamlit run eval_app.py\n")
