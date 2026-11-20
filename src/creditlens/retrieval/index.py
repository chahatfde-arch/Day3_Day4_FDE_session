"""The policy index: four ways to find the same paragraph, so you can measure which is best.

The one idea this module teaches: **"search" is not one thing.** The same question can be
answered by matching words, by matching meaning, by combining both, or by asking a second,
slower model to re-read the shortlist. Each has a cost and each fails differently:

| Retriever  | How it decides                        | Fails when                                   |
| ---------- | ------------------------------------- | -------------------------------------------- |
| `bm25`     | rare words shared by query and chunk  | the analyst says "cash flow cover", not DSCR |
| `vector`   | the embedding of the meaning          | an exact id like "3.1" or a rare acronym     |
| `hybrid`   | reciprocal rank fusion of both lists  | rarely; it is the sensible default            |
| `reranked` | a cross-encoder re-reads query+chunk  | never on quality, often on latency and CPU    |

Reciprocal rank fusion (RRF) deserves a sentence, because beginners expect score averaging.
Scores from BM25 and from cosine similarity are on different scales and cannot be averaged.
RRF throws the scores away and keeps only the *ranks*: a chunk that is 1st in one list and
7th in the other scores `1/(60+1) + 1/(60+7)`. The constant 60 is the value from the original
paper; it stops the top result of one list from dominating the fusion.

Everything is free and local: Qdrant runs in memory, embeddings come from
`creditlens.gateway.llm.embed` (a small sentence-transformers model), and the reranker is
optional. Embeddings are cached on disk under `data/cache/retrieval/` keyed by a hash of the
corpus and the model, so a second run of the eval does not re-embed hundreds of chunks.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

from creditlens.config import settings
from creditlens.observability import observe
from creditlens.retrieval.chunking import Chunk
from creditlens.schemas import PolicyHit

log = logging.getLogger(__name__)

RRF_K = 60  # the constant from the reciprocal rank fusion paper
_TOKEN = re.compile(r"[a-z0-9.]+")


def _normalise(token: str) -> str:
    """Fold the endings that stop a keyword search from matching the obvious thing.

    BM25 matches words, not meanings, so "cheque returns" finds nothing in a manual that
    says "cheques returned". A full stemmer (Porter, Snowball) is another dependency and
    another thing to explain; these four suffix rules fix the cases a credit policy actually
    contains. The same rules run over the documents and over the query, so they stay in step.
    """
    if any(ch.isdigit() for ch in token) or len(token) <= 3:
        return token  # never touch "3.1", "1.25x" or short words like "is"
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: len(token) - len(suffix)] + replacement
            break
    # "cheques" -> "chequ" above, so "cheque" must become "chequ" too, or the two never meet.
    if token.endswith("e") and len(token) >= 5:
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, lightly stemmed word tokens for BM25.

    Dots are kept inside tokens on purpose: "3.1" and "1.25" are the most discriminating
    words in a rulebook, and a tokenizer that splits them into "3" and "1" throws that away.
    """
    return [_normalise(t.strip(".") or t) for t in _TOKEN.findall(text.lower())]


def corpus_hash(chunks: list[Chunk], model: str) -> str:
    """A fingerprint of exactly what would be embedded, so the cache can never go stale."""
    h = hashlib.sha256(model.encode("utf-8"))
    for chunk in chunks:
        h.update(chunk.chunk_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(chunk.text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


class PolicyIndex:
    """Holds the chunks once and answers questions about them four different ways.

    Typical use:

        index = PolicyIndex()
        index.build(chunks)
        hits = index.search("what is the minimum DSCR?", k=5, source="good_bank_policy")

    Args:
        location: Qdrant location. Defaults to `QDRANT_URL` if set, else `":memory:"`.
            Point it at `http://localhost:6333` for Docker or at a Qdrant Cloud URL.
        collection: the Qdrant collection name.
        embedding_model: overrides `settings.llm.embedding_model`.
        use_reranker: load the cross-encoder when it is available. Set False to stay fast.
        cache_dir: where the embedding `.npz` files live.
    """

    def __init__(self, location: Optional[str] = None, collection: str = "creditlens_policy",
                 embedding_model: Optional[str] = None, use_reranker: bool = True,
                 cache_dir: Optional[Path] = None) -> None:
        self.location = location or os.environ.get("QDRANT_URL") or ":memory:"
        self.collection = collection
        self.embedding_model = embedding_model or settings.llm.embedding_model
        self.use_reranker = use_reranker
        self.cache_dir = (settings.paths.cache / "retrieval") if cache_dir is None else cache_dir
        self.chunks: list[Chunk] = []
        self.vectors: Optional[np.ndarray] = None
        self.vector_search_available = False
        self._client = None
        self._bm25 = None
        self._bm25_tokens: list[list[str]] = []
        self._reranker = None
        self._reranker_tried = False
        self._by_id: dict[str, Chunk] = {}

    # -- building ----------------------------------------------------------

    @observe(name="retrieval.build_index")
    def build(self, chunks: list[Chunk]) -> "PolicyIndex":
        """Index the chunks: BM25 always, vectors when sentence-transformers is importable."""
        self.chunks = list(chunks)
        self._by_id = {c.chunk_id: c for c in self.chunks}
        self._build_bm25()
        self._build_vectors()
        return self

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi  # declared in the "retrieval" extra

        self._bm25_tokens = [tokenize(c.text + " " + c.section) for c in self.chunks]
        # BM25Okapi cannot be built from an empty corpus, and an empty index is not an error.
        self._bm25 = BM25Okapi(self._bm25_tokens) if self._bm25_tokens else None

    def _build_vectors(self) -> None:
        vectors = self._load_or_compute_vectors()
        if vectors is None:
            self.vector_search_available = False
            return
        self.vectors = vectors
        self._client = self._make_client(vectors.shape[1])
        self._upsert(vectors)
        self.vector_search_available = True

    def _load_or_compute_vectors(self) -> Optional[np.ndarray]:
        """Embed the corpus, reusing `data/cache/retrieval/<hash>.npz` when it matches."""
        if not self.chunks:
            return None
        key = corpus_hash(self.chunks, self.embedding_model)
        cache_path = self.cache_dir / f"{key}.npz"
        if cache_path.exists():
            try:
                with np.load(cache_path) as data:
                    return data["vectors"].astype(np.float32)
            except Exception as exc:  # noqa: BLE001  a corrupt cache must not break search
                log.warning("ignoring unreadable embedding cache %s: %s", cache_path, exc)
        try:
            from creditlens.gateway.llm import embed
        except ImportError:  # pragma: no cover - the gateway is always importable in-repo
            return None
        try:
            raw = embed([c.text for c in self.chunks], model=self.embedding_model)
        except Exception as exc:  # noqa: BLE001  no model on disk and no network: BM25 only
            _warn_no_embeddings(exc)
            return None
        vectors = np.asarray(raw, dtype=np.float32)
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, vectors=vectors)
        except OSError as exc:  # noqa: BLE001  a read-only data folder is not fatal
            log.warning("could not write the embedding cache: %s", exc)
        return vectors

    def _make_client(self, dim: int):
        from qdrant_client import QdrantClient, models

        client = QdrantClient(location=self.location) if self.location != ":memory:" \
            else QdrantClient(":memory:")
        if client.collection_exists(self.collection):
            client.delete_collection(self.collection)
        client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        return client

    def _upsert(self, vectors: np.ndarray) -> None:
        from qdrant_client import models

        points = [models.PointStruct(id=i, vector=vectors[i].tolist(), payload=c.to_payload())
                  for i, c in enumerate(self.chunks)]
        # Batches keep the request small enough for a hosted Qdrant with a body size limit.
        for start in range(0, len(points), 256):
            self._client.upsert(self.collection, points=points[start:start + 256])

    # -- the four retrievers ----------------------------------------------

    def search_vector(self, q: str, k: int = 5, source: Optional[str] = None) -> list[PolicyHit]:
        """Nearest neighbours by meaning, with the `source` metadata filter pushed into Qdrant."""
        if not self.vector_search_available or self._client is None:
            return []
        from creditlens.gateway.llm import embed
        from qdrant_client import models

        query_vector = embed([q], model=self.embedding_model)[0]
        query_filter = None
        if source:
            query_filter = models.Filter(must=[models.FieldCondition(
                key="source", match=models.MatchValue(value=source))])
        response = self._client.query_points(self.collection, query=query_vector, limit=k,
                                             with_payload=True, query_filter=query_filter)
        return [self._hit(point.payload["chunk_id"], float(point.score), "vector")
                for point in response.points]

    def search_bm25(self, q: str, k: int = 5, source: Optional[str] = None) -> list[PolicyHit]:
        """Classic keyword scoring. The metadata filter is applied in Python, after scoring."""
        if self._bm25 is None:
            return []
        query_terms = set(tokenize(q))
        scores = self._bm25.get_scores(list(query_terms))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        hits: list[PolicyHit] = []
        for i in ranked:
            if source and self.chunks[i].source != source:
                continue
            # BM25 scores can be zero or negative in tiny corpora (a term in every document has a
            # negative idf), so "shares a query term" is the honest test for a match, not the sign.
            if not query_terms.intersection(self._bm25_tokens[i]):
                continue
            hits.append(self._hit(self.chunks[i].chunk_id, float(scores[i]), "bm25"))
            if len(hits) >= k:
                break
        return hits

    def search_hybrid(self, q: str, k: int = 5, source: Optional[str] = None) -> list[PolicyHit]:
        """Reciprocal rank fusion of the vector list and the BM25 list."""
        pool = max(k * 4, 20)  # fuse deeper lists than you return, or fusion has nothing to do
        lists = [self.search_vector(q, pool, source), self.search_bm25(q, pool, source)]
        fused: dict[str, float] = {}
        for hits in lists:
            for rank, hit in enumerate(hits, start=1):
                fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [self._hit(chunk_id, score, "hybrid") for chunk_id, score in ordered]

    @observe(name="retrieval.search")
    def search(self, q: str, k: int = 5, source: Optional[str] = None,
               rerank: bool = True) -> list[PolicyHit]:
        """The default retriever: hybrid, then a cross-encoder rerank when one is available."""
        candidates = self.search_hybrid(q, max(k * 4, 20), source)
        if not rerank or not candidates:
            return candidates[:k]
        reranker = self._load_reranker()
        if reranker is None:
            return candidates[:k]
        pairs = [(q, hit.text) for hit in candidates]
        try:
            scores = reranker.predict(pairs)
        except Exception as exc:  # noqa: BLE001  a broken reranker must not lose the hits
            log.warning("reranker failed, returning the hybrid order: %s", exc)
            return candidates[:k]
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        return [self._hit(candidates[i].chunk_id, float(scores[i]), "reranked") for i in order[:k]]

    def search_all(self, q: str, k: int = 5, source: Optional[str] = None) -> dict[str, list[PolicyHit]]:
        """Run every retriever on one question. The eval needs all four lists side by side."""
        return {
            "vector": self.search_vector(q, k, source),
            "bm25": self.search_bm25(q, k, source),
            "hybrid": self.search_hybrid(q, k, source),
            "reranked": self.search(q, k, source, rerank=True),
        }

    # -- helpers -----------------------------------------------------------

    def _hit(self, chunk_id: str, score: float, retriever: str) -> PolicyHit:
        chunk = self._by_id[chunk_id]
        # PolicyHit.source is a Literal, so anything unexpected falls back to the policy manual.
        source = chunk.source if chunk.source in ("good_bank_policy", "sba_sop") else "good_bank_policy"
        return PolicyHit(chunk_id=chunk.chunk_id, source=source, section=chunk.section,
                         text=chunk.text, score=round(score, 6), retriever=retriever)

    def _load_reranker(self):
        """Load the cross-encoder once. Missing model, no network, no torch: return None."""
        if self._reranker is not None or self._reranker_tried:
            return self._reranker
        self._reranker_tried = True
        if not self.use_reranker:
            return None
        try:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(settings.llm.reranker_model)
        except Exception as exc:  # noqa: BLE001  reranking is a bonus, never a requirement
            log.warning("reranker %s unavailable (%s). Hybrid order will be used instead.",
                        settings.llm.reranker_model, exc)
            self._reranker = None
        return self._reranker

    @property
    def sources(self) -> set[str]:
        return {c.source for c in self.chunks}

    def __len__(self) -> int:
        return len(self.chunks)

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (f"PolicyIndex(chunks={len(self.chunks)}, sources={sorted(self.sources)}, "
                f"vectors={'on' if self.vector_search_available else 'off'}, location={self.location!r})")


_warned_no_embeddings = False


def _warn_no_embeddings(exc: Exception) -> None:
    """Say it once, not once per query: there are no vectors, so this is BM25 only."""
    global _warned_no_embeddings
    if not _warned_no_embeddings:
        _warned_no_embeddings = True
        log.warning("embeddings unavailable (%s). Falling back to BM25-only retrieval. "
                    "Install the retrieval extra and let the model download once: "
                    "pip install -e .[retrieval]", exc)
