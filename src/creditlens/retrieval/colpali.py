"""Visual retrieval: search a page by looking at it, not by reading text off it.

The one idea this module teaches: **the OCR step can be the thing that loses the answer.**
The normal pipeline for a chart or a scanned page is image to text to chunks to embeddings.
Every arrow throws something away: OCR cannot see that a bar is taller than its neighbour, and
a table flattened into a paragraph loses which number was in which column.

ColPali skips the text. It cuts the page image into patches, embeds the patches with a
vision-language model, and scores a question against those patch embeddings directly (the
"late interaction" idea from ColBERT, applied to pixels). The chart is never turned into text,
so nothing is lost on the way in.

The price is a GPU. The model is several gigabytes and scoring is a matrix product per page, so
this is a **Colab demo**, not a laptop step:

    !pip install byaldi
    from creditlens.retrieval.colpali import precompute_embeddings, recall_at_k
    precompute_embeddings()          # once, on a T4
    recall_at_k()                    # does the right chart come back for its own question?

Then download `data/cache/colpali/` and `search()` works on the laptop *if* byaldi and torch
are installed. Without them, every function here raises `ColPaliUnavailable` with instructions,
and the module still imports cleanly, because `creditlens.retrieval` is imported by the agents.

`answer_with_vision()` is the cheap alternative, and worth comparing against: send the chart
image straight to a multimodal model through the gateway and ask the question. No index, no
GPU, but you pay a call per question and you cannot search thirty charts for the right one.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from creditlens.config import Settings, settings as default_settings
from creditlens.observability import observe

log = logging.getLogger(__name__)

DEFAULT_MODEL = "vidore/colpali-v1.2"
INDEX_NAME = "creditlens_charts"


class ColPaliUnavailable(RuntimeError):
    """Raised when byaldi, torch or a GPU is missing. Carries the instructions, not a stack trace."""


def cache_dir(s: Settings = default_settings) -> Path:
    return s.paths.cache / "colpali"


def charts_dir(s: Settings = default_settings) -> Path:
    return s.paths.real / "charts"


def _friendly(reason: str) -> ColPaliUnavailable:
    return ColPaliUnavailable(
        f"{reason}\n\n"
        "ColPali needs a GPU and the byaldi package. Open a Colab notebook with a T4 runtime and:\n"
        "    !pip install byaldi\n"
        "    from creditlens.retrieval.colpali import precompute_embeddings\n"
        "    precompute_embeddings()\n"
        "Then copy data/cache/colpali/ back to your laptop. Everything else in CreditLens runs "
        "on CPU; this one module does not."
    )


def available() -> bool:
    """True when byaldi and torch can be imported. Tests call this before anything else."""
    try:
        import byaldi  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def has_gpu() -> bool:
    """True when a CUDA device is visible. ColPali on CPU is slow enough to look broken."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _load_model(model_name: str = DEFAULT_MODEL, index_root: Optional[Path] = None):
    if not available():
        raise _friendly("byaldi or torch is not installed.")
    from byaldi import RAGMultiModalModel

    return RAGMultiModalModel.from_pretrained(model_name, index_root=str(index_root)) \
        if index_root else RAGMultiModalModel.from_pretrained(model_name)


@observe(name="colpali.precompute")
def precompute_embeddings(s: Settings = default_settings, model_name: str = DEFAULT_MODEL,
                          overwrite: bool = True) -> Path:
    """Index every PNG in `data/real/charts` and save the embeddings under `data/cache/colpali/`.

    Run this once on a GPU. The index is a folder of tensors plus a mapping from document id
    back to the file name, which is what `search()` needs to name the image it found.
    """
    charts = charts_dir(s)
    images = sorted(charts.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"no chart images in {charts}. Run "
                                "`python -m creditlens.data_real --only charts` first.")
    out = cache_dir(s)
    out.mkdir(parents=True, exist_ok=True)
    model = _load_model(model_name, index_root=out)
    model.index(input_path=str(charts), index_name=INDEX_NAME,
                store_collection_with_index=False, overwrite=overwrite)
    # byaldi returns integer doc ids; keep our own map so a hit can name the file.
    mapping = {str(i): image.name for i, image in enumerate(images)}
    (out / "doc_ids.json").write_text(json.dumps(mapping, indent=1), encoding="utf-8")
    log.info("indexed %d charts into %s", len(images), out)
    return out


_model = None


def _index_model(s: Settings = default_settings, model_name: str = DEFAULT_MODEL):
    """Load the precomputed index once per process."""
    global _model
    if _model is not None:
        return _model
    if not available():
        raise _friendly("byaldi or torch is not installed.")
    index_path = cache_dir(s) / INDEX_NAME
    if not index_path.exists():
        raise _friendly(f"no precomputed index at {index_path}.")
    from byaldi import RAGMultiModalModel

    _model = RAGMultiModalModel.from_index(str(index_path))
    return _model


@observe(name="colpali.search")
def search(question: str, k: int = 3, s: Settings = default_settings) -> list[dict]:
    """Find the chart images that answer `question`.

    Returns a list of `{"image": "chart_007.png", "path": ..., "score": 12.3, "rank": 1}`,
    best first. The image, not text, is the retrieval unit.
    """
    model = _index_model(s)
    mapping = _doc_id_map(s)
    results = []
    for rank, result in enumerate(model.search(question, k=k), start=1):
        doc_id = str(getattr(result, "doc_id", getattr(result, "doc_ids", "")))
        name = mapping.get(doc_id, f"doc_{doc_id}.png")
        results.append({"image": name, "path": str(charts_dir(s) / name),
                        "score": float(getattr(result, "score", 0.0)), "rank": rank})
    return results


def _doc_id_map(s: Settings = default_settings) -> dict[str, str]:
    path = cache_dir(s) / "doc_ids.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@observe(name="colpali.recall_at_k")
def recall_at_k(questions_json: Optional[Path] = None, k: int = 3,
                s: Settings = default_settings) -> dict:
    """Score the visual retriever the only way that needs no new labels.

    Every ChartQA row is already a (chart, question) pair written by a human about that chart.
    So the correct answer for "What's the value of the lowest bar?" asked of the whole
    collection is: the chart that question was written about. Anything else is a miss.

    Careful with the result: several charts can plausibly answer "how many bars are shown?",
    so this is a floor on quality, not a ceiling. Read the misses before believing the number.
    """
    questions_json = Path(questions_json) if questions_json else charts_dir(s) / "questions.json"
    rows = json.loads(Path(questions_json).read_text(encoding="utf-8"))
    hits, reciprocal, misses = 0, 0.0, []
    for row in rows:
        results = search(row["question"], k=k, s=s)
        names = [r["image"] for r in results]
        if row["image"] in names:
            rank = names.index(row["image"]) + 1
            hits += 1
            reciprocal += 1.0 / rank
        else:
            misses.append({"question": row["question"], "expected": row["image"], "got": names})
    total = len(rows) or 1
    return {f"recall@{k}": round(hits / total, 3), "mrr": round(reciprocal / total, 3),
            "questions": len(rows), "hits": hits, "misses": misses[:10]}


# ---------------------------------------------------------------------------
# The CPU alternative: ask a multimodal model about one image
# ---------------------------------------------------------------------------


def image_data_url(image_path: Path) -> str:
    """Base64 data URL, the format LiteLLM expects for an image content part."""
    image_path = Path(image_path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@observe(name="colpali.answer_with_vision")
def answer_with_vision(question: str, image_path: Path,
                       s: Settings = default_settings) -> str:
    """Send one chart image and a question to the vision model through the gateway.

    This is the baseline ColPali is compared against. It needs no GPU and no index, but it
    only works once you already know which image to look at, which is exactly the problem
    retrieval exists to solve.
    """
    from creditlens.gateway.llm import get_gateway

    messages = [{"role": "user", "content": [
        {"type": "text", "text": question},
        # The LiteLLM multimodal message format, identical across providers.
        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
    ]}]
    return get_gateway().complete(messages, model=s.llm.vision).text


def main() -> None:  # pragma: no cover - GPU only
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--precompute", action="store_true", help="index data/real/charts (GPU)")
    parser.add_argument("--search", help="a question to look up")
    parser.add_argument("--recall", action="store_true", help="score recall@k over questions.json")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    try:
        if args.precompute:
            print(f"index written to {precompute_embeddings()}")
        elif args.search:
            for row in search(args.search, k=args.k):
                print(f"{row['rank']}. {row['image']}  score={row['score']:.3f}")
        elif args.recall:
            print(json.dumps(recall_at_k(k=args.k), indent=2))
        else:
            parser.print_help()
    except ColPaliUnavailable as exc:
        print(exc)


if __name__ == "__main__":  # pragma: no cover
    main()
