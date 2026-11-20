"""Cut a long document into pieces a retriever can rank.

The one idea this module teaches: **a chunk is the unit of truth you hand to the model.**
Too big and the answer drowns in unrelated text (and you pay for tokens you did not need).
Too small and the sentence that matters loses the context that explains it.

Three strategies are implemented so the Day 3 bake-off can compare them on real text:

| Strategy   | How it cuts                                   | Good for                               |
| ---------- | --------------------------------------------- | -------------------------------------- |
| `section`  | on Markdown headings, keeping the heading path | rulebooks: a rule lives under a heading |
| `fixed`    | every `size` words, with `overlap` words back  | messy text with no structure            |
| `sentence` | whole sentences packed up to `size` words      | prose where cutting mid-sentence hurts  |

`section` is the default for CreditLens because both corpora (the Good Bank manual and the
SBA SOP) are rulebooks: "3.1 DSCR must be at least 1.25x" only means something under the
heading "3. Financial thresholds". Keeping that heading path on every chunk is what lets the
credit memo cite a rule, and what lets the eval say "did we retrieve the right section?".

Chunk ids are deterministic (source + position + a short hash of the text) so that a memo
written today cites the same id after the index is rebuilt tomorrow.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

Strategy = Literal["section", "fixed", "sentence"]

# A Markdown heading: one to six hashes, a space, then the heading text.
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# A sentence ends at . ! or ? followed by whitespace. Good enough for policy prose.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w+")


@dataclass
class Chunk:
    """One retrievable piece of a document.

    `section` is the heading path, for example
    "Good Bank Credit Policy Manual > 3. Financial thresholds". The eval checks retrieval
    quality by asking whether the expected heading appears in this string.
    """

    chunk_id: str
    source: str
    section: str
    text: str
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(_WORD.findall(self.text))

    def to_payload(self) -> dict:
        """The dict stored next to the vector in Qdrant (and used for metadata filters)."""
        return {"chunk_id": self.chunk_id, "source": self.source, "section": self.section,
                "text": self.text, **self.metadata}


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def make_chunk_id(source: str, index: int, text: str) -> str:
    """Deterministic id: same source, same position, same text means the same id, always."""
    return f"{source}:{index:04d}:{_short_hash(text)}"


def _words(text: str) -> list[str]:
    return text.split()


def _split_with_overlap(text: str, size: int, overlap: int) -> list[str]:
    """Cut `text` into word windows of `size` words that each repeat `overlap` words of the last.

    The overlap is the cheap insurance against cutting a sentence in half: whichever side of
    the cut the answer lands on, one of the two chunks still contains the whole sentence.
    """
    words = _words(text)
    if len(words) <= size:
        return [text] if text.strip() else []
    step = max(1, size - max(0, overlap))
    pieces: list[str] = []
    start = 0
    while start < len(words):
        piece = " ".join(words[start:start + size])
        if piece.strip():
            pieces.append(piece)
        if start + size >= len(words):
            break
        start += step
    return pieces


def _sentence_pack(text: str, size: int) -> list[str]:
    """Pack whole sentences into groups of at most `size` words."""
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    pieces, current, current_len = [], [], 0
    for sentence in sentences:
        n = len(_words(sentence))
        if current and current_len + n > size:
            pieces.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n
    if current:
        pieces.append(" ".join(current))
    return pieces


MAX_HEADING_WORDS = 14
"""Longer than this and a "heading" is really a paragraph. See `_iter_sections`."""


def _iter_sections(text: str, max_heading_words: int = MAX_HEADING_WORDS) -> Iterable[tuple[str, str]]:
    """Walk the Markdown and yield (heading path, body text) for every heading.

    The heading path is kept on a stack, so a level-4 heading carries the level-1, 2 and 3
    headings above it. That is the "breadcrumb" an analyst needs to trust a citation.

    One defence against messy real documents: the SBA SOP was converted from a Word file in
    which whole paragraphs carry a Heading style, so `#####` sometimes introduces a 60-word
    sentence. Treating those as headings would push the real heading
    ("Citizenship And Residency Requirements") out of the breadcrumb and leave citations
    pointing at nothing. Anything longer than `max_heading_words` words is therefore demoted
    back to body text, which is where it belongs.
    """
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    path = ""

    def current_path() -> str:
        return " > ".join(title for _, title in stack)

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match is not None and len(_words(match.group(2))) > max_heading_words:
            body.append(match.group(2))  # a mis-converted paragraph: keep the words, drop the hashes
            continue
        if match is None:
            body.append(line)
            continue
        # A new heading closes the previous section.
        joined = "\n".join(body).strip()
        if joined:
            yield path or current_path(), joined
        body = []
        level, title = len(match.group(1)), match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = current_path()
    joined = "\n".join(body).strip()
    if joined:
        yield path or current_path(), joined


def chunk_markdown(text: str, source: str, strategy: Strategy = "section",
                   size: int = 400, overlap: int = 50) -> list[Chunk]:
    """Cut a Markdown document into `Chunk` objects.

    Args:
        text: the whole document.
        source: the corpus name that ends up on every chunk and drives metadata filters,
            for example "good_bank_policy" or "sba_sop".
        strategy: "section", "fixed" or "sentence" (see the module docstring).
        size: target chunk length in words.
        overlap: words repeated between neighbouring chunks (used by "section" and "fixed").

    Returns:
        A list of chunks in document order. Chunk ids are stable across runs.
    """
    if strategy != "sentence" and overlap >= size:
        # Sentence packing never overlaps, so a small `size` with the default overlap is fine there.
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    pieces: list[tuple[str, str]] = []  # (section, text)

    if strategy == "section":
        for section, body in _iter_sections(text):
            for part in _split_with_overlap(body, size, overlap):
                pieces.append((section, part))
    elif strategy == "fixed":
        for part in _split_with_overlap(text, size, overlap):
            pieces.append(("", part))
    elif strategy == "sentence":
        for part in _sentence_pack(text, size):
            pieces.append(("", part))
    else:  # pragma: no cover - guarded by the type hint, kept for a clear error
        raise ValueError(f"unknown chunking strategy: {strategy}")

    chunks: list[Chunk] = []
    for index, (section, body) in enumerate(pieces):
        body = body.strip()
        if not body:
            continue
        chunks.append(Chunk(
            chunk_id=make_chunk_id(source, index, body),
            source=source,
            section=section,
            text=body,
            metadata={"strategy": strategy, "position": index, "words": len(_words(body))},
        ))
    return chunks


def compare_strategies(text: str, source: str = "corpus", size: int = 400,
                       overlap: int = 50) -> dict[str, dict]:
    """Run all three strategies on the same text and report the shape of what came out.

    This is the Day 3 bake-off table. There is no single winner: look at the numbers next to
    the eval's recall for the same corpus and decide which trade-off your documents deserve.

    Returns a dict like
    `{"section": {"chunks": 42, "avg_words": 310.5, "min_words": 9, "max_words": 400, ...}}`.
    """
    report: dict[str, dict] = {}
    for strategy in ("section", "fixed", "sentence"):
        chunks = chunk_markdown(text, source, strategy=strategy, size=size, overlap=overlap)  # type: ignore[arg-type]
        counts = [c.word_count for c in chunks] or [0]
        with_section = sum(1 for c in chunks if c.section)
        report[strategy] = {
            "chunks": len(chunks),
            "avg_words": round(sum(counts) / len(counts), 1),
            "min_words": min(counts),
            "max_words": max(counts),
            "total_words": sum(counts),
            "chunks_with_section": with_section,
        }
    return report


def format_comparison(report: dict[str, dict]) -> str:
    """Pretty-print `compare_strategies` as a table for the terminal or the docs."""
    header = f"{'strategy':<10} {'chunks':>7} {'avg words':>10} {'min':>6} {'max':>6} {'with section':>13}"
    lines = [header, "-" * len(header)]
    for strategy, row in report.items():
        lines.append(f"{strategy:<10} {row['chunks']:>7} {row['avg_words']:>10} "
                     f"{row['min_words']:>6} {row['max_words']:>6} {row['chunks_with_section']:>13}")
    return "\n".join(lines)
