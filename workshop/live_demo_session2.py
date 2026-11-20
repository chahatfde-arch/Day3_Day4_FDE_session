# %% [markdown]
# # Session 2 live demo, in slide order
# Run one cell at a time in VS Code. Each cell names the slide it goes with.
# The first index build embeds about 1,400 chunks (two or three minutes on CPU); after that it is cached
# under data/cache/retrieval. Build it once before the session.

# %% [markdown]
# ## 0. Before the room fills up
# ```bash
# python workshop/get_ready.py 2
# python -m creditlens.data_real sba_sop    # the SBA rulebook, if data/real/sba_sop_50_10.md is missing
# ```

# %%
import pandas as pd
from creditlens.config import settings
from creditlens.retrieval.policy import corpus_paths, get_index

pd.set_option("display.width", 140)
for name, path in corpus_paths().items():
    print(f"{name:18} {path.exists()!s:5} {path}")
idx = get_index()          # builds or loads the cached index
print("chunks in the index:", len(idx.chunks) if hasattr(idx, "chunks") else "?")

# %% [markdown]
# ## Slide "A tidy manual teaches the mechanics. A real rulebook teaches the rest": chunking bake-off

# %%
from creditlens.retrieval.chunking import compare_strategies
paths = corpus_paths()
for name in ("good_bank_policy", "sba_sop"):
    if paths[name].exists():
        rep = compare_strategies(paths[name].read_text(encoding="utf-8"), source=name, size=400, overlap=50)
        print(name); print(pd.DataFrame(rep).T.to_string()); print()

# %% [markdown]
# ## Slide "Keep the dots, fold the plurals": what BM25 sees

# %%
from creditlens.retrieval.index import tokenize
print(tokenize("What is the minimum DSCR? Rule 3.1 says 1.25x; cheques returned"))

# %% [markdown]
# ## Slide "Four ways to find something": the retriever vote
# Run each question the room voted on. Point at which retriever found rule 3.1 or 4.2 first.

# %%
def show(results, n=3):
    for name, hits in results.items():
        print(f"-- {name}")
        for h in hits[:n]:
            print(f"   {h.source:16} {h.section[:40]:40} {h.score:8.3f}  {h.text[:70]!r}")

for q in ["What is the minimum DSCR and how is it calculated?", "What does rule 4.2 say about returned cheques?"]:
    print("\n###", q); show(idx.search_all(q, k=3, source="good_bank_policy"))

# %% [markdown]
# ## Slide "Merge by rank, not by score": reciprocal rank fusion, done on paper first
# Two lists on the whiteboard, k = 60, then check with the code.

# %%
def rrf(lists, k=60):
    score = {}
    for hits in lists:
        for rank, h in enumerate(hits, 1):
            score[h.chunk_id] = score.get(h.chunk_id, 0) + 1 / (k + rank)
    return sorted(score.items(), key=lambda kv: -kv[1])[:5]

r = idx.search_all("What is the minimum DSCR and how is it calculated?", k=5, source="good_bank_policy")
for cid, s in rrf([r["vector"], r["bm25"]]): print(f"{s:.5f}  {cid}")
print("library hybrid top:", [h.chunk_id for h in r["hybrid"][:5]])

# %% [markdown]
# ## Slide "Compliance says the SBA rulebook must be searchable too": the source filter
# Without the filter, an American rule can come back as bank policy.

# %%
q = "Can we lend to a business under two years old?"
for source in (None, "good_bank_policy", "sba_sop"):
    print(f"\n-- source={source}")
    for h in idx.search(q, k=3, source=source):
        print(f"   {h.source:16} {h.section[:40]:40} {h.text[:70]!r}")

# %% [markdown]
# ## Slide "Who else is linked to this company?": the graph and the hidden-group hunt
# ```bash
# python -m creditlens.retrieval.graph                 # every group
# python -m creditlens.retrieval.graph C019            # one company's exposure
# python -m creditlens.retrieval.graph --cheatsheet    # the Cypher
# ```

# %%
from creditlens.retrieval.graph import GraphStore
gs = GraphStore(backend="networkx").load_from_registry(settings.paths.raw)
for g in gs.all_groups(): print(sorted(g))
for cid in ("C019", "C031"):
    e = gs.group_exposure(cid)
    print(f"\n{cid}: members {[m.company_id for m in e.members]}, exposure {e.total_exposure_inr_000:,} vs {e.limit_inr_000:,}, breach={e.breaches_limit}")
    for m in e.members: print("   ", m.company_id, m.link, "hops", m.hops)

# %% [markdown]
# ## Slide "The graph connects the dots": Rohan and Rahul, part two

# %%
from creditlens.scenarios import case_file
SCN = settings.paths.data / "scenarios"
g2 = GraphStore(backend="networkx").load_from_registry(SCN)
for cid in ("SC01", "SC10"):
    e = g2.group_exposure(cid)
    print(f"{cid}: {[m.company_id for m in e.members]} exposure {e.total_exposure_inr_000:,} vs {e.limit_inr_000:,} -> breach {e.breaches_limit}")
for app in ("A901", "A902"):
    cf = case_file(SCN, app); print(f"\n{app}:", [f["code"] for f in cf["flags"]]); print("  ", cf["note"])

# %% [markdown]
# ## Slide "Charts hold numbers that never appear as text": ColPali (talk through, needs a GPU)
# Show data/real/charts/chart_000.png and the code slide. Do not run it live on a laptop.

# %% [markdown]
# ## Slide "Naive versus hybrid versus reranked": the retrieval evaluation
# ```bash
# python -m evals.retrieval_eval --k 5 --no-ragas --show-misses
# ```
# Then the last quiz, the case verdict, and (deep dive only) the four bonus cases:

# %%
for app in ("A903", "A904", "A905", "A906"):
    cf = case_file(SCN, app); print(app, [f["code"] for f in cf["flags"]])
