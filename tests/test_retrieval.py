"""Retrieval tests: chunking, BM25, hybrid fusion, metadata filters and (optionally) vectors.

Everything here passes offline, with no API key, no Docker and no GPU. The one test that needs
the embedding model is marked `slow` and skips itself when the model is neither cached nor
downloadable, so a laptop on conference wifi still gets a green suite.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from creditlens.retrieval.chunking import Chunk, chunk_markdown, compare_strategies
from creditlens.retrieval.index import PolicyIndex, tokenize
from creditlens.retrieval.policy import load_chunks
from creditlens.schemas import PolicyHit

DSCR_QUERY = "minimum DSCR"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def policy_text(s) -> str:
    return (s.paths.raw / "policy" / "credit_policy_manual.md").read_text(encoding="utf-8")


def _hf_cache_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def embedding_model_reachable(model: str, timeout: int = 20) -> bool:
    """True when the model is already on disk, or Hugging Face answers within `timeout`.

    Downloading about 130 MB is fine the first time; waiting forever for a dead network is not.
    """
    if (_hf_cache_root() / f"models--{model.replace('/', '--')}").exists():
        return True
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_section_chunking_keeps_the_heading_path(s):
    chunks = chunk_markdown(policy_text(s), source="good_bank_policy", strategy="section")
    sections = [c.section for c in chunks]
    assert any("3. Financial thresholds" in sec for sec in sections)
    assert any("5. Related parties and group exposure" in sec for sec in sections)
    # Every section chunk carries a breadcrumb; that is the whole point of the strategy.
    assert all(c.section for c in chunks)
    assert all(c.source == "good_bank_policy" for c in chunks)


def test_chunk_ids_are_deterministic_and_unique(s):
    text = policy_text(s)
    first = chunk_markdown(text, source="good_bank_policy")
    second = chunk_markdown(text, source="good_bank_policy")
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert len({c.chunk_id for c in first}) == len(first)
    # The id encodes where it came from, so a citation in a memo is traceable.
    assert all(c.chunk_id.startswith("good_bank_policy:") for c in first)


def test_fixed_strategy_overlaps_and_respects_size():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_markdown(text, source="test", strategy="fixed", size=100, overlap=20)
    assert len(chunks) > 1
    assert all(c.word_count <= 100 for c in chunks)
    # The overlap means the tail of one chunk reappears at the head of the next.
    tail = chunks[0].text.split()[-20:]
    assert chunks[1].text.split()[:20] == tail


def test_sentence_strategy_does_not_cut_mid_sentence():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = chunk_markdown(text, source="test", strategy="sentence", size=4)
    assert len(chunks) == 3
    assert all(c.text.endswith(".") for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_markdown("some text", source="test", strategy="fixed", size=10, overlap=10)


def test_compare_strategies_reports_all_three(s):
    report = compare_strategies(policy_text(s))
    assert set(report) == {"section", "fixed", "sentence"}
    assert all(row["chunks"] > 0 for row in report.values())
    assert report["section"]["chunks_with_section"] == report["section"]["chunks"]
    assert report["fixed"]["chunks_with_section"] == 0


def test_long_pseudo_headings_are_treated_as_text():
    """The SBA SOP was converted from Word, so whole paragraphs carry a Heading style."""
    sentence = " ".join(["word"] * 40)
    text = f"# Real Heading\n\n##### {sentence}\n\nbody text here\n"
    chunks = chunk_markdown(text, source="test", strategy="section")
    assert all("Real Heading" in c.section for c in chunks)
    assert any(sentence in c.text for c in chunks)


# ---------------------------------------------------------------------------
# BM25: works with no model, no network and no vectors
# ---------------------------------------------------------------------------


def test_tokenizer_keeps_rule_ids_and_folds_plurals():
    tokens = tokenize("Cheques returned under rule 3.1 need a DSCR of 1.25x")
    assert "3.1" in tokens          # a rule id must survive tokenisation intact
    assert "1.25x" in tokens
    assert "chequ" in tokens        # "cheques" and "cheque" must reach the same token
    assert tokenize("cheque")[0] == "chequ"
    assert tokenize("returns")[0] == tokenize("returned")[0]


@pytest.fixture(scope="module")
def bm25_index(s) -> PolicyIndex:
    """A BM25-only index: no embeddings are computed, so this is fast and always offline."""
    index = PolicyIndex(use_reranker=False)
    index.chunks = chunk_markdown(policy_text(s), source="good_bank_policy")
    index._by_id = {c.chunk_id: c for c in index.chunks}
    index._build_bm25()
    return index


def test_bm25_finds_the_dscr_rule(bm25_index):
    hits = bm25_index.search_bm25(DSCR_QUERY, k=3)
    assert hits, "BM25 returned nothing for a query about DSCR"
    assert all(isinstance(h, PolicyHit) and h.retriever == "bm25" for h in hits)
    # Rule 3.1 is the answer. BM25 scores words, not meaning, and "minimum" also appears in
    # section 1, so the right chunk is reliably in the top 3 rather than always first. That
    # gap between "found it" and "ranked it first" is what hybrid search and reranking close.
    assert any("3.1" in h.text and "DSCR" in h.text for h in hits)
    assert any("3. Financial thresholds" in h.section for h in hits)


def test_bm25_handles_wording_the_manual_does_not_use(bm25_index):
    """"cheque returns" must find "Cheques returned", or keyword search is useless to an analyst."""
    hits = bm25_index.search_bm25("cheque returns", k=3)
    assert hits
    assert "4. Banking conduct" in hits[0].section


def test_bm25_source_filter(bm25_index):
    assert bm25_index.search_bm25(DSCR_QUERY, k=3, source="good_bank_policy")
    assert bm25_index.search_bm25(DSCR_QUERY, k=3, source="sba_sop") == []


def test_hybrid_returns_policy_hits_with_the_right_source(bm25_index):
    hits = bm25_index.search_hybrid("group exposure limit", k=3, source="good_bank_policy")
    assert hits
    assert all(isinstance(h, PolicyHit) for h in hits)
    assert all(h.source == "good_bank_policy" for h in hits)
    assert all(h.retriever == "hybrid" for h in hits)
    assert any("5. Related parties" in h.section for h in hits)


def test_search_all_runs_every_retriever(bm25_index):
    results = bm25_index.search_all("group exposure limit", k=3)
    assert set(results) == {"vector", "bm25", "hybrid", "reranked"}
    assert results["bm25"] and results["hybrid"]
    # No embeddings in this fixture, so the vector list is empty and hybrid degrades to BM25.
    assert results["vector"] == []


def test_search_falls_back_to_hybrid_order_without_a_reranker(bm25_index):
    hits = bm25_index.search("what is the maximum tenure for working capital?", k=2)
    assert hits
    assert any("6. Product rules" in h.section for h in hits)


def test_metadata_filter_separates_two_corpora():
    """The filter is the thing that stops an SBA rule being cited as Good Bank policy."""
    chunks = [
        Chunk("a:0", "good_bank_policy", "3. Financial thresholds", "DSCR must be at least 1.25x"),
        Chunk("b:0", "sba_sop", "Credit Standards", "The lender must document the DSCR"),
    ]
    index = PolicyIndex(use_reranker=False)
    index.chunks = chunks
    index._by_id = {c.chunk_id: c for c in chunks}
    index._build_bm25()
    good = index.search_bm25("DSCR", k=5, source="good_bank_policy")
    sba = index.search_bm25("DSCR", k=5, source="sba_sop")
    assert [h.chunk_id for h in good] == ["a:0"]
    assert [h.chunk_id for h in sba] == ["b:0"]
    assert len(index.search_bm25("DSCR", k=5)) == 2


def test_empty_index_is_not_an_error():
    index = PolicyIndex(use_reranker=False).build([])
    assert len(index) == 0
    assert index.search_bm25("anything", k=3) == []
    assert index.search("anything", k=3) == []


def test_load_chunks_skips_a_missing_sba_sop(s):
    """The real rulebook is an optional download; the manual alone must still index."""
    chunks = load_chunks(s)
    assert chunks
    assert "good_bank_policy" in {c.source for c in chunks}


# ---------------------------------------------------------------------------
# Vector search: needs the embedding model, so it is optional and marked slow
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_vector_search_finds_the_dscr_chunk(s):
    model = s.llm.embedding_model
    if not embedding_model_reachable(model):
        pytest.skip(f"{model} is not cached and Hugging Face is unreachable")
    pytest.importorskip("sentence_transformers")

    index = PolicyIndex(use_reranker=False).build(
        chunk_markdown(policy_text(s), source="good_bank_policy"))
    if not index.vector_search_available:
        pytest.skip("the embedding model could not be loaded")

    hits = index.search_vector("what cash flow cover do we need on a loan?", k=3)
    assert hits
    assert all(h.retriever == "vector" for h in hits)
    # The question never says "DSCR", which is exactly the query BM25 struggles with.
    assert any("3.1" in h.text for h in hits)


@pytest.mark.slow
def test_embeddings_are_cached_on_disk(s, tmp_path):
    model = s.llm.embedding_model
    if not embedding_model_reachable(model):
        pytest.skip(f"{model} is not cached and Hugging Face is unreachable")
    pytest.importorskip("sentence_transformers")

    chunks = chunk_markdown(policy_text(s), source="good_bank_policy")
    first = PolicyIndex(use_reranker=False, cache_dir=tmp_path).build(chunks)
    if not first.vector_search_available:
        pytest.skip("the embedding model could not be loaded")
    cached = list(tmp_path.glob("*.npz"))
    assert len(cached) == 1, "building the index should write exactly one embedding cache file"

    second = PolicyIndex(use_reranker=False, cache_dir=tmp_path).build(chunks)
    assert second.vectors is not None
    assert second.vectors.shape == first.vectors.shape
    assert list(tmp_path.glob("*.npz")) == cached, "a rebuild must reuse the cache, not add to it"
