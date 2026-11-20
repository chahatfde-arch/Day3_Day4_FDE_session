"""Optional: put the same chunks in Pinecone and see whether a hosted index retrieves better.

The one idea this module teaches: **a vector database is a swappable part.** Qdrant in memory,
Qdrant in Docker, Qdrant Cloud and Pinecone all answer the same question ("which vectors are
nearest this one?"). What differs is where the vectors live, who runs the server, how filters
are expressed and what it costs. So the honest way to choose is to run both over the same
chunks, the same embedding model and the same golden questions, and compare recall@5.

Spoiler, and worth saying out loud in the workshop: on a few hundred chunks of policy text the
recall is usually identical, because the retrieval quality comes from the embedding model and
the chunking, not from the database. You pick a hosted index for operational reasons (someone
else runs it, it survives a restart, it scales past memory), not for better answers.

**Free plan limits**: Pinecone's free "starter" tier exists but its limits (index count,
dimensions, namespaces, storage, idle timeouts) change often. Check the current limits on the
Pinecone pricing and documentation pages before a session rather than trusting any number
written in a repository. Nothing in this file will tell you what the quota is, on purpose.

This file must import cleanly with no `pinecone` package installed, because it is in the same
package as `search_policy`, which every agent imports. Everything heavy happens inside the
functions.

Run it:

    export PINECONE_API_KEY=...
    LLM_OFFLINE=1 python -m creditlens.retrieval.pinecone_compare --k 5
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

from creditlens.config import Settings, settings as default_settings
from creditlens.retrieval.chunking import Chunk
from creditlens.retrieval.index import PolicyIndex
from creditlens.retrieval.policy import load_chunks

INDEX_NAME = os.environ.get("PINECONE_INDEX", "creditlens-policy")
CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
REGION = os.environ.get("PINECONE_REGION", "us-east-1")


class PineconeUnavailable(RuntimeError):
    """Raised when the pinecone client is missing or PINECONE_API_KEY is not set."""


def available() -> bool:
    """True when this comparison can actually run. Callers check first and skip politely."""
    if not os.environ.get("PINECONE_API_KEY"):
        return False
    try:
        import pinecone  # noqa: F401
    except ImportError:
        return False
    return True


def _require() -> None:
    if not os.environ.get("PINECONE_API_KEY"):
        raise PineconeUnavailable("PINECONE_API_KEY is not set. Create a free Pinecone project, "
                                  "copy the API key and export it before running this script.")
    try:
        import pinecone  # noqa: F401
    except ImportError as exc:
        raise PineconeUnavailable("pip install pinecone to run this comparison") from exc


def _client():
    from pinecone import Pinecone

    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def ensure_index(dimension: int, name: str = INDEX_NAME):
    """Create the serverless index if it is not there yet, then wait for it to be ready."""
    _require()
    from pinecone import ServerlessSpec

    pc = _client()
    existing = {spec["name"] for spec in pc.list_indexes()}
    if name not in existing:
        pc.create_index(name=name, dimension=dimension, metric="cosine",
                        spec=ServerlessSpec(cloud=CLOUD, region=REGION))
        # Creation is asynchronous. Poll rather than guess at a sleep duration.
        for _ in range(60):
            if pc.describe_index(name).status.get("ready"):
                break
            time.sleep(2)
    return pc.Index(name)


def upsert_chunks(chunks: list[Chunk], vectors, name: str = INDEX_NAME, batch: int = 100) -> int:
    """Send the same chunks and the same vectors that went into Qdrant.

    Pinecone stores the chunk text in the metadata, exactly as Qdrant stores it in the payload,
    so the `source` filter works the same way on both sides and the comparison stays fair.
    """
    index = ensure_index(len(vectors[0]), name)
    sent = 0
    for start in range(0, len(chunks), batch):
        window = list(zip(chunks[start:start + batch], vectors[start:start + batch]))
        index.upsert(vectors=[{
            "id": chunk.chunk_id,
            "values": list(map(float, vector)),
            "metadata": {"source": chunk.source, "section": chunk.section, "text": chunk.text},
        } for chunk, vector in window])
        sent += len(window)
    return sent


def search_pinecone(query: str, k: int = 5, source: Optional[str] = None,
                    name: str = INDEX_NAME) -> list[dict]:
    """Top `k` chunks from Pinecone, with the same metadata filter Qdrant uses."""
    _require()
    from creditlens.gateway.llm import embed

    index = _client().Index(name)
    vector = embed([query])[0]
    response = index.query(vector=vector, top_k=k, include_metadata=True,
                           filter={"source": {"$eq": source}} if source else None)
    return [{"chunk_id": match["id"], "score": float(match["score"]),
             "section": match["metadata"].get("section", ""),
             "text": match["metadata"].get("text", ""),
             "source": match["metadata"].get("source", "")} for match in response["matches"]]


def compare(k: int = 5, source: Optional[str] = None, s: Settings = default_settings) -> dict:
    """Recall@k for Qdrant vector search and for Pinecone, on the same golden questions."""
    from evals.retrieval_eval import is_hit, load_questions

    _require()
    chunks = load_chunks(s)
    qdrant = PolicyIndex().build(chunks)
    if not qdrant.vector_search_available or qdrant.vectors is None:
        raise PineconeUnavailable("no embeddings available, so there is nothing to compare")
    upsert_chunks(chunks, qdrant.vectors.tolist())

    questions = load_questions(s, source)
    hits = {"qdrant": 0, "pinecone": 0}
    disagreements = []
    for question in questions:
        q_hits = qdrant.search_vector(question["question"], k, source)
        qdrant_ok = any(is_hit(h, question) for h in q_hits)
        p_hits = search_pinecone(question["question"], k, source)
        pinecone_ok = any(_matches(h, question) for h in p_hits)
        hits["qdrant"] += int(qdrant_ok)
        hits["pinecone"] += int(pinecone_ok)
        if qdrant_ok != pinecone_ok:
            disagreements.append({"id": question["id"], "qdrant": qdrant_ok, "pinecone": pinecone_ok})
    total = len(questions) or 1
    return {
        "k": k, "questions": len(questions), "chunks": len(chunks),
        f"qdrant_recall@{k}": round(hits["qdrant"] / total, 3),
        f"pinecone_recall@{k}": round(hits["pinecone"] / total, 3),
        "disagreements": disagreements,
    }


def _matches(hit: dict, question: dict) -> bool:
    """The same hit rule as the eval, applied to a raw Pinecone match dict."""
    if hit.get("source") != question["expected_source"]:
        return False
    section = (question.get("expected_section") or "").strip()
    if section and section.lower() in hit.get("section", "").lower():
        return True
    rule_id = (question.get("expected_rule_id") or "").strip()
    return bool(rule_id) and rule_id in hit.get("text", "")


def main() -> None:  # pragma: no cover - needs a Pinecone account
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--source", choices=["good_bank_policy", "sba_sop"])
    args = parser.parse_args()
    try:
        result = compare(k=args.k, source=args.source)
    except PineconeUnavailable as exc:
        print(f"Skipped: {exc}")
        print("Check the current free-plan limits on the Pinecone website before you start.")
        return
    print(f"chunks: {result['chunks']}   questions: {result['questions']}   k: {result['k']}")
    print(f"  Qdrant   recall@{args.k}: {result[f'qdrant_recall@{args.k}']}")
    print(f"  Pinecone recall@{args.k}: {result[f'pinecone_recall@{args.k}']}")
    if result["disagreements"]:
        print("\nThe two databases disagreed on:")
        for row in result["disagreements"]:
            print(f"  {row['id']}: qdrant={row['qdrant']} pinecone={row['pinecone']}")
    else:
        print("\nIdentical results. Same vectors, same neighbours: choose on operations, not recall.")


if __name__ == "__main__":  # pragma: no cover
    main()
