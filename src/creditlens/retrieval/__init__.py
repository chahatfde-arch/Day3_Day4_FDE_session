"""Retrieval for CreditLens AI (milestone 5).

Three kinds of "find the right evidence" live here, and they are deliberately different:

- **Text retrieval** (`chunking`, `index`, `policy`): find the policy paragraph that decides a
  rule. Keyword search, vector search, hybrid fusion and reranking, over the Good Bank manual
  and the real SBA SOP 50 10 rulebook.
- **Graph retrieval** (`graph`): find every company linked to the applicant through a shared
  director or a shared registered address. A chain of links is a graph question, not a text one.
- **Visual retrieval** (`colpali`): find the right chart image for a question, without ever
  turning the chart into text. GPU only, so it is a Colab demo rather than a laptop step.

Nothing here calls a paid service. Embeddings are local (sentence-transformers), the vector
store is in-memory Qdrant, and the graph is NetworkX unless NEO4J_URI points at a real database.
"""
from creditlens.retrieval.chunking import Chunk, chunk_markdown, compare_strategies
from creditlens.retrieval.policy import search_policy

__all__ = ["Chunk", "chunk_markdown", "compare_strategies", "search_policy"]
