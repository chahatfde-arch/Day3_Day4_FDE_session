"""One function the rest of CreditLens calls: `search_policy(query)`.

The one idea this module teaches: **build the index once, not once per question.** Loading a
corpus, chunking it, embedding several hundred chunks and starting Qdrant takes seconds. An
agent that answers twenty policy questions while drafting a memo must not pay that twenty
times, so the index lives in a module-level variable for the lifetime of the process, and the
embeddings live on disk so even the first call of the *next* process is fast.

Two corpora are loaded:

- `good_bank_policy` from `data/raw/policy/credit_policy_manual.md`: the fictional manual the
  memo is actually written against. Rules like 3.1 (DSCR) come from here.
- `sba_sop` from `data/real/sba_sop_50_10.md`: the real US Small Business Administration
  rulebook. It is optional, large and messy, which is exactly why it is here: retrieval that
  works on a two-page manual tells you nothing.

Fallback: when sentence-transformers cannot be imported or the embedding model cannot be
reached, the index silently becomes BM25 only and logs one warning. Search still works, the
agent still cites a rule, and the eval will show the recall you gave up.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from creditlens.config import Settings, settings as default_settings
from creditlens.observability import observe
from creditlens.retrieval.chunking import Chunk, chunk_markdown
from creditlens.retrieval.index import PolicyIndex
from creditlens.schemas import PolicyHit

log = logging.getLogger(__name__)

GOOD_BANK = "good_bank_policy"
SBA_SOP = "sba_sop"

_index: Optional[PolicyIndex] = None
_index_lock = threading.Lock()
_warned_missing_sop = False


def corpus_paths(s: Settings = default_settings) -> dict[str, Path]:
    """Where each corpus lives on disk. The SBA SOP is optional."""
    return {
        GOOD_BANK: s.paths.raw / "policy" / "credit_policy_manual.md",
        SBA_SOP: s.paths.real / "sba_sop_50_10.md",
    }


def load_chunks(s: Settings = default_settings, strategy: str = "section",
                size: int = 400, overlap: int = 50) -> list[Chunk]:
    """Read both corpora from disk and chunk them. Missing SBA SOP is a warning, not an error."""
    global _warned_missing_sop
    chunks: list[Chunk] = []
    for source, path in corpus_paths(s).items():
        if not path.exists():
            if source == SBA_SOP and not _warned_missing_sop:
                _warned_missing_sop = True
                log.warning("%s not found. Run `python -m creditlens.data_real --only sop` to add "
                            "the real SBA rulebook to the index.", path)
            elif source != SBA_SOP:
                log.warning("policy corpus %s is missing", path)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.extend(chunk_markdown(text, source=source, strategy=strategy,  # type: ignore[arg-type]
                                     size=size, overlap=overlap))
    return chunks


def get_index(s: Settings = default_settings, rebuild: bool = False, **kwargs) -> PolicyIndex:
    """The process-wide index, built on first use.

    Args:
        s: settings, so tests can point at a temporary data root.
        rebuild: throw the cached index away and build it again (used by tests and the eval).
        **kwargs: passed to `PolicyIndex`, for example `use_reranker=False`.
    """
    global _index
    with _index_lock:
        if _index is None or rebuild:
            index = PolicyIndex(**kwargs)
            index.build(load_chunks(s))
            log.info("policy index ready: %d chunks from %s", len(index), sorted(index.sources))
            _index = index
        return _index


def reset_index() -> None:
    """Forget the cached index. Tests call this between different data roots."""
    global _index
    with _index_lock:
        _index = None


@observe(name="retrieval.search_policy")
def search_policy(query: str, k: int = 5, source: Optional[str] = None) -> list[PolicyHit]:
    """Find the policy paragraphs that answer `query`.

    This is the function the LangChain tool, the policy checker and the memo writer call.

    Args:
        query: a plain-language question, for example "what is the minimum DSCR?".
        k: how many chunks to return.
        source: restrict to one corpus, "good_bank_policy" or "sba_sop". None searches both.

    Returns:
        Up to `k` `PolicyHit` objects, best first. Each carries the heading path it came from,
        so the memo can cite "3. Financial thresholds" rather than a bare chunk id.
    """
    index = get_index()
    return index.search(query, k=k, source=source, rerank=True)
