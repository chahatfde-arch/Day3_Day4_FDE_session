# Retrieval, graph and visual documents (Day 3)

*Beginner notes for milestone 5. Code: `src/creditlens/retrieval/`, `evals/retrieval_eval.py`,
`ground_truth/retrieval_questions.json`.*

## Three questions, three retrievers

| Priya asks | What has to happen | Retriever |
| --- | --- | --- |
| "What does the policy say about his debt level?" | find a passage by meaning or by rule number | text: vector, keyword, hybrid |
| "Does Rohan own other companies that owe us money?" | follow links between companies, directors and addresses | graph |
| "What does the chart on page 2 show?" | find a page that was never text | visual (ColPali) |

## Chunking

A chunk is the unit we embed and retrieve. `chunk_markdown(text, source, strategy, size, overlap)`:

- `section`: split at Markdown headings and keep the heading path as `section`. Long sections are
  sub-split with overlap. Best for rulebooks: the answer to "rule 3.1" is a whole section.
- `fixed`: every N words with overlap. Works on anything, ignores structure.
- `sentence`: pack whole sentences up to N words. Never cuts mid-sentence.

Run `compare_strategies(text)` on both corpora. The Good Bank manual chunks neatly whatever you do.
The SBA rulebook, with five heading levels and legal cross-references, is where the strategies
diverge, which is why it is in the project.

## Embeddings, vector search, keyword search

- **Embedding**: a small open model (BGE-small, about 130 MB, CPU) turns a chunk into a list of
  numbers. Chunks with similar meaning end up close together. Vectors are cached in
  `data/cache/retrieval/` keyed by a hash of the chunks, so nothing is re-embedded unless a document changes.
- **Vector search** (Qdrant): find the chunks whose vectors are closest to the question's vector.
  In memory by default; `QDRANT_URL` points it at Docker or the Qdrant Cloud free tier.
- **BM25**: classic keyword scoring. It is the only thing that finds "rule 4.2" reliably, because
  "4.2" has no meaning for an embedding. Our tokenizer keeps dots inside tokens for that reason and
  folds plurals so "cheques" matches "cheque".
- **Hybrid with reciprocal rank fusion**: each retriever ranks the chunks; RRF scores a chunk by the
  sum of 1/(60 + rank) across retrievers. Ranks, not scores, so the two scales need not agree.
- **Reranker**: a cross-encoder reads the question and a candidate chunk together and re-scores it.
  Slower, more accurate, so it runs only on the top twenty hybrid candidates.
- **Metadata filter**: `source="good_bank_policy"` restricts a search before ranking. This is the same
  mechanism RBAC uses on Day 6 to keep an analyst inside their region.

`search_policy(query, k, source)` wraps all of this in one call and builds the index once per process.

## Which retriever? (the Day 3 vote)

1. "What is the minimum DSCR?" (vector or hybrid)
2. "Show me rule 4.2." (keyword)
3. "Which companies share a director with Rohan's?" (graph)
4. "Is a cash-heavy business allowed if the owner explains it?" (hybrid, then reranker)
5. "What does the revenue chart show for FY2025?" (visual)
6. "Does the SBA allow loans to businesses under two years old?" (hybrid with `source="sba_sop"`)

## The related-party graph

Policy 5.1: a group is any set of companies linked by a common director or registered address,
directly or through a chain. Chains are what SQL does badly (one self-join per hop) and what graphs do
in one line: a variable-length path.

Nodes: `Company`, `Director`, `Address`. Edges: `DIRECTOR_OF`, `REGISTERED_AT`. `GraphStore` runs on
Neo4j when `NEO4J_URI` is set (AuraDB Free, or `docker compose --profile retrieval`) and on NetworkX in
memory otherwise. Either way `GroupExposure.cypher` carries the query text, so the audit trail reads the
same.

Cypher cheatsheet (`cypher_cheatsheet()` prints the full versions):

```cypher
// companies in the same group as C012, any number of hops
MATCH (c:Company {company_id: $id})-[:DIRECTOR_OF|REGISTERED_AT*1..6]-(other:Company)
RETURN DISTINCT other.company_id

// profile it, then add the index and profile again
PROFILE MATCH (c:Company {company_id: $id}) RETURN c
CREATE INDEX company_id IF NOT EXISTS FOR (c:Company) ON (c.company_id)
```

The registry hides four groups. One is linked only by a shared address: pods that query directors
alone miss it, and the answer key (`data/answer_key/related_party_groups.json`) shows them why.

## Visual retrieval with ColPali

ColPali embeds each page *as an image* using a vision language model, so a question can find the
right chart without OCR. It is heavy for a free GPU: we use thirty real charts (ChartQA) and small
batches, and the trainer pre-computes the embeddings into `data/cache/colpali/` for rooms without a
GPU. `recall_at_k` scores how often a chart is retrieved for its own question; `answer_with_vision`
sends the retrieved image to a vision model through the gateway.

## Evaluation

```bash
python -m evals.retrieval_eval --k 5
python -m evals.retrieval_eval --k 5 --source sba_sop --show-misses
```

Golden questions carry the section heading (and rule id) that must come back. Recall@5 says whether
it was in the top five; MRR rewards putting it first. Naive (vector only) versus hybrid versus
reranked, on both corpora. RAGAS adds model-judged faithfulness and answer relevancy when a key is
available; judge calls count against quota, so it is optional and off in CI.

## Hosted services, free tiers

- **Qdrant Cloud**: create a free cluster, copy the URL and API key into `QDRANT_URL` and `QDRANT_API_KEY`.
- **Neo4j AuraDB Free**: create an instance, download the credentials file, set `NEO4J_URI`,
  `NEO4J_USERNAME`, `NEO4J_PASSWORD`. Free instances pause when idle; open the console to resume.
- **Pinecone**: `python -m creditlens.retrieval.pinecone_compare` with `PINECONE_API_KEY` upserts the
  same chunks and prints recall side by side with Qdrant. Check the free plan's index limits first.
