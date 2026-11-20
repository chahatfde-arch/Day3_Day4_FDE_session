"""Measure retrieval: which of the four retrievers actually finds the right rule?

The one idea this script teaches: **"the RAG works" is not a sentence anyone can act on.**
Pick questions you know the answer to, decide in advance what counts as a correct answer,
and then the four retrievers argue with numbers instead of with opinions.

Two metrics, and they answer different questions:

- **recall@k**: of the golden questions, what share had the right chunk somewhere in the top
  `k`? This is the number that matters when the chunks are fed to an LLM, because the LLM
  reads all `k` of them. "Did the evidence reach the model at all?"
- **MRR** (mean reciprocal rank): 1 for a hit at position 1, 1/2 at position 2, 1/3 at
  position 3, 0 for a miss, averaged. This is the number that matters when a human reads the
  list, or when `k` is small. "Did the evidence reach the model *first*?"

A question counts as a hit when a returned chunk's heading path contains the expected heading,
or the chunk text contains the expected rule id ("3.1"). Both are checked because the two
corpora are different: the Good Bank manual has numbered rules, the SBA SOP has headings only.

Run it:

    LLM_OFFLINE=1 python -m evals.retrieval_eval --k 5
    LLM_OFFLINE=1 python -m evals.retrieval_eval --k 5 --source good_bank_policy

Output goes to the terminal and to `evals/results/retrieval_eval.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from creditlens.config import Settings, settings as default_settings
from creditlens.retrieval.chunking import compare_strategies, format_comparison
from creditlens.retrieval.index import PolicyIndex
from creditlens.retrieval.policy import corpus_paths, load_chunks
from creditlens.schemas import PolicyHit

RETRIEVERS = ("vector", "bm25", "hybrid", "reranked")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_questions(s: Settings = default_settings, source: Optional[str] = None) -> list[dict]:
    """Read `ground_truth/retrieval_questions.json`, optionally filtered to one corpus."""
    path = s.paths.ground_truth / "retrieval_questions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]
    if source:
        questions = [q for q in questions if q["expected_source"] == source]
    return questions


def is_hit(hit: PolicyHit, question: dict) -> bool:
    """Did this chunk answer the question?

    Two ways to be right, and the source must match either way: a chunk from the SBA SOP that
    happens to contain the string "3.1" is not an answer to a Good Bank question.
    """
    if hit.source != question["expected_source"]:
        return False
    expected_section = (question.get("expected_section") or "").strip()
    if expected_section and expected_section.lower() in hit.section.lower():
        return True
    rule_id = (question.get("expected_rule_id") or "").strip()
    return bool(rule_id) and rule_id in hit.text


def rank_of_first_hit(hits: list[PolicyHit], question: dict) -> Optional[int]:
    """1-based position of the first correct chunk, or None when the retriever missed."""
    for position, hit in enumerate(hits, start=1):
        if is_hit(hit, question):
            return position
    return None


def score_questions(index: PolicyIndex, questions: list[dict], k: int,
                    source: Optional[str] = None) -> dict:
    """Run every retriever over every question and collect recall@k, MRR and timings."""
    per_retriever = {name: {"hits": 0, "reciprocal_rank": 0.0, "seconds": 0.0}
                     for name in RETRIEVERS}
    per_question: list[dict] = []

    for question in questions:
        # The eval searches the whole index unless --source was given, because at memo time
        # nobody tells the retriever which corpus the answer is in. That is the honest test.
        row = {"id": question["id"], "question": question["question"],
               "expected_section": question.get("expected_section", ""),
               "expected_rule_id": question.get("expected_rule_id", ""),
               "expected_source": question["expected_source"], "ranks": {}}
        for name in RETRIEVERS:
            started = time.perf_counter()
            hits = _run(index, name, question["question"], k, source)
            per_retriever[name]["seconds"] += time.perf_counter() - started
            rank = rank_of_first_hit(hits, question)
            row["ranks"][name] = rank
            row.setdefault("top_section", {})[name] = hits[0].section if hits else ""
            if rank is not None:
                per_retriever[name]["hits"] += 1
                per_retriever[name]["reciprocal_rank"] += 1.0 / rank
        per_question.append(row)

    total = len(questions) or 1
    summary = {}
    for name, acc in per_retriever.items():
        summary[name] = {
            f"recall@{k}": round(acc["hits"] / total, 3),
            "mrr": round(acc["reciprocal_rank"] / total, 3),
            "hits": acc["hits"],
            "questions": len(questions),
            "avg_latency_ms": round(acc["seconds"] / total * 1000, 1),
        }
    return {"summary": summary, "per_question": per_question}


def _run(index: PolicyIndex, retriever: str, query: str, k: int,
         source: Optional[str]) -> list[PolicyHit]:
    if retriever == "vector":
        return index.search_vector(query, k, source)
    if retriever == "bm25":
        return index.search_bm25(query, k, source)
    if retriever == "hybrid":
        return index.search_hybrid(query, k, source)
    return index.search(query, k, source, rerank=True)


def format_table(summary: dict, k: int) -> str:
    """The table that goes on the slide."""
    header = f"{'retriever':<12} {'recall@' + str(k):>10} {'MRR':>8} {'hits':>7} {'avg ms':>9}"
    lines = [header, "-" * len(header)]
    for name, row in summary.items():
        lines.append(f"{name:<12} {row[f'recall@{k}']:>10.3f} {row['mrr']:>8.3f} "
                     f"{str(row['hits']) + '/' + str(row['questions']):>7} {row['avg_latency_ms']:>9.1f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAGAS: optional, and only when a real LLM is available
# ---------------------------------------------------------------------------

RAGAS_HELP = """
RAGAS was not run. It grades the *answer*, not just the retrieval:
  faithfulness      - is every claim in the answer supported by the retrieved chunks?
  answer_relevancy  - does the answer actually address the question that was asked?
  context_precision - are the retrieved chunks relevant, or is the model reading padding?
All three need a real LLM to judge, so they cannot run offline.

To enable it:
  pip install -e .[evals]                 # installs ragas
  export GEMINI_API_KEY=...               # or any provider the gateway routes to
  unset LLM_OFFLINE                       # RAGAS cannot grade the offline stub
  python -m evals.retrieval_eval --k 5
""".strip()


def run_ragas(index: PolicyIndex, questions: list[dict], k: int,
              source: Optional[str] = None) -> dict:
    """Faithfulness, answer relevancy and context precision over the retrieved contexts.

    Returns `{"skipped": reason}` when ragas is not installed or the run is offline, so the
    eval never fails because an optional grader is missing.
    """
    if default_settings.llm.offline or os.environ.get("LLM_OFFLINE") == "1":
        return {"skipped": "LLM_OFFLINE=1: there is no model to judge with"}
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        return {"skipped": f"ragas not installed ({exc})"}

    from creditlens.gateway.llm import get_gateway

    gateway = get_gateway()
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for question in questions:
        hits = index.search(question["question"], k, source, rerank=True)
        contexts = [h.text for h in hits]
        prompt = ("Answer the credit analyst's question using only the policy extracts below. "
                  "If the extracts do not answer it, say so.\n\n"
                  + "\n\n".join(f"[{h.section}]\n{h.text}" for h in hits)
                  + f"\n\nQuestion: {question['question']}")
        rows["question"].append(question["question"])
        rows["answer"].append(gateway.chat(prompt))
        rows["contexts"].append(contexts)
        rows["ground_truth"].append(question.get("expected_section", ""))
    result = evaluate(Dataset.from_dict(rows),
                      metrics=[faithfulness, answer_relevancy, context_precision])
    return {k2: round(float(v), 3) for k2, v in dict(result).items()}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(k: int = 5, source: Optional[str] = None, s: Settings = default_settings,
        with_ragas: bool = True, with_bakeoff: bool = True) -> dict:
    """Score every retriever on the golden questions and write the report.

    Args:
        k: how many chunks each retriever returns (recall@k is measured at this k).
        source: restrict both the index filter and the question list to one corpus.
        s: settings, so a test can point at a temporary data root.
        with_ragas: attempt the RAGAS block (it skips itself when offline).
        with_bakeoff: include the chunking strategy comparison in the report.

    Returns:
        The same dict that is written to `evals/results/retrieval_eval.json`.
    """
    questions = load_questions(s, source)
    chunks = load_chunks(s)
    index = PolicyIndex().build(chunks)

    report: dict = {
        "k": k,
        "source": source or "all",
        "chunks_indexed": len(chunks),
        "sources_indexed": sorted(index.sources),
        "vector_search": index.vector_search_available,
        "embedding_model": index.embedding_model,
        "questions": len(questions),
    }
    report.update(score_questions(index, questions, k, source))

    if with_bakeoff:
        report["chunking_bakeoff"] = {
            name: compare_strategies(path.read_text(encoding="utf-8", errors="replace"), name)
            for name, path in corpus_paths(s).items() if path.exists()
        }
    report["ragas"] = run_ragas(index, questions, k, source) if with_ragas \
        else {"skipped": "disabled"}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5, help="how many chunks to retrieve (default 5)")
    parser.add_argument("--source", choices=["good_bank_policy", "sba_sop"],
                        help="score one corpus only")
    parser.add_argument("--no-ragas", action="store_true", help="skip the RAGAS block entirely")
    parser.add_argument("--show-misses", action="store_true",
                        help="list the questions the hybrid retriever got wrong")
    args = parser.parse_args()

    report = run(k=args.k, source=args.source, with_ragas=not args.no_ragas)

    print(f"\nCorpus: {report['chunks_indexed']} chunks from {', '.join(report['sources_indexed'])}"
          f"  |  questions: {report['questions']}  |  vector search: "
          f"{'on' if report['vector_search'] else 'OFF (BM25 only)'}")
    print()
    print(format_table(report["summary"], args.k))

    if report.get("chunking_bakeoff"):
        for name, table in report["chunking_bakeoff"].items():
            print(f"\nChunking bake-off: {name}")
            print(format_comparison(table))

    if args.show_misses:
        print("\nMissed by hybrid:")
        for row in report["per_question"]:
            if row["ranks"]["hybrid"] is None:
                print(f"  {row['id']}: {row['question']}")
                print(f"      wanted '{row['expected_section']}', top hit was "
                      f"'{row['top_section']['hybrid']}'")

    ragas = report.get("ragas", {})
    if "skipped" in ragas:
        print("\n" + RAGAS_HELP)
    else:
        print("\nRAGAS: " + ", ".join(f"{k2}={v}" for k2, v in ragas.items()))

    print(f"\nWritten to {RESULTS_DIR / 'retrieval_eval.json'}")


if __name__ == "__main__":
    main()
